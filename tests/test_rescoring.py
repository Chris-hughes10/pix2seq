"""Core correctness tests for teacher-forced re-scoring.

The central invariant: re-scored log-probs must equal the log-probs recorded
during generation (same constraints, same temperature), because REINFORCE
requires gradients of the log-probability of the distribution that was
actually sampled from.
"""

import pytest
import torch

from rl.rescoring import rescore_sequences


def _generate(model, images, temperature=1.0, num_samples=1, greedy=False, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        sequences, _, _, log_probs = model.infer(
            images=images,
            max_seq_len=model.max_seq_len,
            temperature=temperature,
            top_k=0,
            top_p=0.0,
            greedy=greedy,
            return_log_probs=True,
            num_samples=num_samples,
        )
    return sequences, log_probs


class TestRescoringMatchesGeneration:
    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_sampled_log_probs_match(
        self, tiny_model, images, token_processor, temperature, num_samples
    ):
        sequences, gen_log_probs = _generate(
            tiny_model, images, temperature=temperature, num_samples=num_samples
        )
        assert sequences.size(0) == images.size(0) * num_samples
        assert gen_log_probs.shape == (sequences.size(0), sequences.size(1) - 1)

        rescored = rescore_sequences(
            model=tiny_model,
            images=images,
            sequences=sequences,
            token_processor=token_processor,
            temperature=temperature,
            num_samples_per_image=num_samples,
        )

        gen_masked = torch.where(
            rescored.mask, gen_log_probs.clone(), torch.zeros_like(gen_log_probs)
        )
        torch.testing.assert_close(
            rescored.log_probs, gen_masked, atol=1e-4, rtol=1e-4
        )

    def test_greedy_log_probs_match(self, tiny_model, images, token_processor):
        sequences, gen_log_probs = _generate(tiny_model, images, greedy=True)
        rescored = rescore_sequences(
            model=tiny_model,
            images=images,
            sequences=sequences,
            token_processor=token_processor,
            temperature=1.0,
        )
        gen_masked = torch.where(
            rescored.mask, gen_log_probs.clone(), torch.zeros_like(gen_log_probs)
        )
        torch.testing.assert_close(
            rescored.log_probs, gen_masked, atol=1e-4, rtol=1e-4
        )

    def test_log_probs_finite_and_masked_zero(
        self, tiny_model, images, token_processor
    ):
        sequences, _ = _generate(tiny_model, images, num_samples=2)
        rescored = rescore_sequences(
            model=tiny_model,
            images=images,
            sequences=sequences,
            token_processor=token_processor,
            num_samples_per_image=2,
        )
        assert torch.isfinite(rescored.log_probs).all()
        assert (rescored.log_probs[~rescored.mask] == 0).all()
        # Valid log-probs are actual log probabilities
        assert (rescored.log_probs[rescored.mask] <= 0).all()


class TestGradients:
    def test_gradients_flow_to_encoder_and_decoder(
        self, tiny_model, images, token_processor
    ):
        sequences, _ = _generate(tiny_model, images, num_samples=2)
        rescored = rescore_sequences(
            model=tiny_model,
            images=images,
            sequences=sequences,
            token_processor=token_processor,
            num_samples_per_image=2,
        )
        loss = -rescored.log_probs.sum()
        tiny_model.zero_grad()
        loss.backward()

        def grad_norm(module):
            return sum(
                p.grad.abs().sum().item()
                for p in module.parameters()
                if p.grad is not None
            )

        assert grad_norm(tiny_model.vit) > 0, "no gradient reached the ViT encoder"
        assert grad_norm(tiny_model.transformer_decoder) > 0
        for p in tiny_model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        tiny_model.zero_grad()

    def test_rescoring_accepts_inference_mode_sequences(
        self, tiny_model, images, token_processor
    ):
        """Sequences straight from infer() (inference tensors) must not raise."""
        sequences, _ = _generate(tiny_model, images)
        rescored = rescore_sequences(
            model=tiny_model,
            images=images,
            sequences=sequences,
            token_processor=token_processor,
        )
        (-rescored.log_probs.sum()).backward()
        tiny_model.zero_grad()


class TestBatchedRescoring:
    def test_num_tgt_per_image_equivalent_to_repeated_images(
        self, tiny_model, images, token_processor
    ):
        sequences, _ = _generate(tiny_model, images, num_samples=3)
        with torch.no_grad():
            grouped = tiny_model(
                images, sequences[:, :-1].clone(), num_tgt_per_image=3
            )
            repeated = tiny_model(
                images.repeat_interleave(3, dim=0), sequences[:, :-1].clone()
            )
        torch.testing.assert_close(grouped, repeated, atol=1e-5, rtol=1e-5)

    def test_class_logits_shape(self, tiny_model, images, token_processor):
        sequences, _ = _generate(tiny_model, images, num_samples=2)
        rescored = rescore_sequences(
            model=tiny_model,
            images=images,
            sequences=sequences,
            token_processor=token_processor,
            num_samples_per_image=2,
            return_class_logits=True,
        )
        num_objects = (sequences.size(1) - 1) // 5
        assert rescored.class_logits.shape == (
            sequences.size(0),
            num_objects,
            token_processor.vocab_size,
        )
