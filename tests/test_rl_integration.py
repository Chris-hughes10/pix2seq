"""End-to-end tests for the REINFORCE training step on a tiny model.

Includes an overfit test: on a fixed toy batch, the mean sampled reward must
increase over training steps. This is the strongest synthetic check that the
whole pipeline (rollout -> reward -> advantage -> re-scoring -> gradient)
learns in the right direction.
"""

import pytest
import torch

from conftest import IMAGE_SIZE

from rl import (
    IoUSupervisionLoss,
    RecallReward,
    REINFORCETrainer,
    RLTrainingConfig,
)
from rl.rescoring import extract_object_confidences, rescore_sequences


def _make_batch(num_images=2, seed=3):
    torch.manual_seed(seed)
    images = torch.randn(num_images, 3, IMAGE_SIZE, IMAGE_SIZE)
    gt_boxes = [
        torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
        torch.tensor([[0.1, 0.3, 0.5, 0.9], [0.5, 0.1, 0.9, 0.4]]),
    ][:num_images]
    gt_labels = [torch.tensor([1]), torch.tensor([0, 2])][:num_images]
    return {"image": images, "gt_boxes": gt_boxes, "gt_labels": gt_labels}


def _make_trainer(model, token_processor, advantage_type="loo", num_samples=3,
                  iou_loss=True):
    config = RLTrainingConfig(
        advantage_type=advantage_type,
        num_samples_per_image=num_samples,
        temperature=1.0,
        iou_loss_weight=1.0 if iou_loss else 0.0,
    )
    return REINFORCETrainer(
        model=model,
        token_processor=token_processor,
        reward_fn=RecallReward(token_processor),
        iou_loss_fn=IoUSupervisionLoss() if iou_loss else None,
        config=config,
    )


class TestComputeLosses:
    @pytest.mark.parametrize("advantage_type,num_samples", [
        ("loo", 3),
        ("grpo", 3),
        ("greedy", 2),
        ("greedy", 1),
    ])
    def test_step_runs_and_updates_params(
        self, tiny_model, token_processor, advantage_type, num_samples
    ):
        torch.manual_seed(0)
        trainer = _make_trainer(
            tiny_model, token_processor, advantage_type, num_samples
        )
        batch = _make_batch()

        loss, stats = trainer.compute_losses(batch)
        assert torch.isfinite(loss)

        tiny_model.zero_grad()
        loss.backward()
        grad_norm = sum(
            p.grad.abs().sum().item()
            for p in tiny_model.parameters()
            if p.grad is not None
        )
        assert grad_norm > 0

        for key in (
            "loss/total",
            "loss/rl",
            "loss/iou",
            "reward/sample_mean",
            "reward/baseline_mean",
            "reward/advantage_mean",
            "policy/tokens_per_seq_mean",
        ):
            assert key in stats

        tiny_model.zero_grad()
        tiny_model.eval()

    def test_rollout_grouping(self, tiny_model, token_processor):
        """K samples per image come grouped in repeat_interleave order."""
        torch.manual_seed(0)
        trainer = _make_trainer(tiny_model, token_processor, num_samples=3)
        batch = _make_batch()
        rollout = trainer.sample_rollouts(batch["image"])
        assert rollout.sequences.size(0) == 2 * 3
        assert rollout.gen_log_probs.size(0) == 2 * 3
        assert rollout.greedy_sequences is None

    def test_confidences_differentiable_for_iou_loss(
        self, tiny_model, token_processor, images
    ):
        torch.manual_seed(0)
        with torch.no_grad():
            sequences, _, _, _ = tiny_model.infer(
                images=images,
                max_seq_len=tiny_model.max_seq_len,
                return_log_probs=True,
            )
        rescored = rescore_sequences(
            model=tiny_model,
            images=images,
            sequences=sequences,
            token_processor=token_processor,
            return_class_logits=True,
        )
        boxes, labels, confidences = extract_object_confidences(
            sequences.clone(), rescored.class_logits, token_processor
        )
        for b, l, c in zip(boxes, labels, confidences):
            assert len(b) == len(l) == len(c)
        non_empty = [c for c in confidences if c.numel() > 0]
        if non_empty:
            loss = IoUSupervisionLoss()(
                boxes, confidences, [b.detach() for b in boxes]
            )
            tiny_model.zero_grad()
            loss.backward()
            grads = [
                p.grad for p in tiny_model.parameters() if p.grad is not None
            ]
            assert any(g.abs().sum() > 0 for g in grads)
            tiny_model.zero_grad()


class TestOverfitToyBatch:
    @pytest.mark.slow
    def test_reward_increases(self, token_processor):
        """~80 REINFORCE steps on a fixed toy batch must increase the reward."""
        # Fresh tiny model so the session-scoped fixture stays untouched
        import model.model as model_module

        torch.manual_seed(7)
        model = model_module.Pix2SeqModel(
            max_seq_len=token_processor.max_seq_len,
            image_size=IMAGE_SIZE,
            patch_size=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
            embedding_dim=32,
            num_heads=2,
            dim_feedforward=64,
            dropout=0.0,
            drop_path=0.0,
            bos_token_id=token_processor.BOS_TOKEN,
            eos_token_id=token_processor.EOS_TOKEN,
            coord_vocab_shift=token_processor.coord_vocab_shift,
            base_vocab_shift=token_processor.BASE_VOCAB_SHIFT,
            num_quantization_bins=token_processor.quantization_bins,
            token_processor=token_processor,
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=False,
        )
        model.eval()

        trainer = _make_trainer(
            model, token_processor, advantage_type="loo", num_samples=4,
            iou_loss=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        batch = {
            "image": torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE),
            "gt_boxes": [
                torch.tensor([[0.2, 0.2, 0.7, 0.7]]),
                torch.tensor([[0.3, 0.1, 0.8, 0.5]]),
            ],
            "gt_labels": [torch.tensor([1]), torch.tensor([3])],
        }

        rewards = []
        for _step in range(80):
            loss, stats = trainer.compute_losses(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            rewards.append(stats["reward/sample_mean"])

        first = sum(rewards[:10]) / 10
        last = sum(rewards[-10:]) / 10
        assert last > first, (
            f"reward did not improve: first10={first:.4f}, last10={last:.4f}"
        )
