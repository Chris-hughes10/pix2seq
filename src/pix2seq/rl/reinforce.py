"""REINFORCE trainer for Pix2Seq with multi-sample (GRPO-style) baselines.

Implements policy-gradient fine-tuning as described in:
- "Tuning Computer Vision Models with Task Rewards" (Pinto et al., 2023)
- "Self-critical Sequence Training for Image Captioning" (Rennie et al., 2017)

Per training step:
1. Sample K sequences per image without gradients (rollout).
2. Compute per-sequence rewards and group-based advantages.
3. Re-score the sampled sequences with a single teacher-forced forward pass
   through the (possibly DDP-wrapped) model to obtain differentiable log-probs
   (see ``rl.rescoring``).
4. Loss = -(advantage * sum log-prob) + optional IoU-supervision loss.

The trainer owns loss computation only; the training loop owns the optimizer,
backward pass and gradient accumulation (so ``accelerator.accumulate`` works).
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from data.tokenizer import TokenProcessor

from .iou_loss import IoUSupervisionLoss
from .rescoring import extract_object_confidences, rescore_sequences
from .rewards import RecallReward

if TYPE_CHECKING:
    from accelerate import Accelerator


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""

    # "reinforce" is implemented; "ppo" (ratio clipping + KL to reference) is
    # reserved for a future extension - see compute_policy_gradient_loss
    algorithm: str = "reinforce"

    # Advantage baseline:
    #   "greedy": r - r(greedy decode)          (SCST; works with K = 1)
    #   "loo":    r_i - mean(r_others in group) (leave-one-out; requires K >= 2)
    #   "mean":   r_i - mean(group)             (requires K >= 2)
    #   "grpo":   (r_i - mean(group)) / (std(group) + eps)  (requires K >= 2)
    advantage_type: str = "loo"

    # Number of sequences sampled per image (K)
    num_samples_per_image: int = 4

    # Sampling parameters. top_k/top_p must stay disabled: REINFORCE requires
    # the log-probs used in the loss to describe the distribution that was
    # actually sampled from, and re-scoring reproduces the full temperature
    # distribution only
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0

    # Loss weights
    iou_loss_weight: float = 1.0

    # Optional extra whitening of advantages across the whole batch
    normalize_advantages: bool = False

    # Gradient clipping (applied by the training loop)
    max_grad_norm: float = 1.0

    # Optional cap on generation length during rollouts (defaults to the token
    # processor's max_seq_len)
    max_gen_len: Optional[int] = None

    def __post_init__(self):
        if self.algorithm != "reinforce":
            raise NotImplementedError(
                f"algorithm='{self.algorithm}' is not implemented (only 'reinforce')"
            )
        if self.advantage_type not in ("greedy", "loo", "mean", "grpo"):
            raise ValueError(f"Unknown advantage_type: {self.advantage_type}")
        if self.top_k != 0 or self.top_p != 0.0:
            raise ValueError(
                "top_k/top_p filtering must be disabled for RL sampling: the "
                "teacher-forced re-scoring computes log-probs of the full "
                "temperature distribution, so sampling from a truncated one "
                "would bias the policy gradient"
            )
        if self.advantage_type != "greedy" and self.num_samples_per_image < 2:
            raise ValueError(
                f"advantage_type='{self.advantage_type}' requires "
                f"num_samples_per_image >= 2, got {self.num_samples_per_image}"
            )


@dataclass
class Rollout:
    """Sequences sampled for one batch of images.

    All tensors are ordinary (non-inference-mode) tensors without gradients.

    Attributes:
        sequences: [B*K, S] sampled sequences (incl. BOS), grouped per image in
            repeat_interleave order
        gen_log_probs: [B*K, S-1] generation-time log-probs of the sampled
            tokens (useful for tests and as "old" log-probs for a future
            PPO-style clipped objective)
        greedy_sequences: [B, S'] greedy-decoded sequences, only present when
            the greedy (SCST) baseline is used
    """

    sequences: torch.Tensor
    gen_log_probs: torch.Tensor
    greedy_sequences: Optional[torch.Tensor] = None


def compute_advantages(
    sample_rewards: torch.Tensor,
    num_samples: int,
    advantage_type: str,
    greedy_rewards: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-sequence advantages from grouped rewards.

    Args:
        sample_rewards: [B*K] rewards, grouped per image in repeat_interleave
            order (all K samples of image 0 first, then image 1, ...)
        num_samples: K, number of samples per image
        advantage_type: "greedy" | "loo" | "mean" | "grpo"
        greedy_rewards: [B] rewards of the greedy baseline (required for
            advantage_type="greedy")
        eps: Numerical stabiliser for the grpo std division

    Returns:
        advantages: [B*K]
    """
    if advantage_type == "greedy":
        if greedy_rewards is None:
            raise ValueError("greedy_rewards required for advantage_type='greedy'")
        return sample_rewards - greedy_rewards.repeat_interleave(num_samples)

    if num_samples < 2:
        raise ValueError(
            f"advantage_type='{advantage_type}' requires num_samples >= 2"
        )

    grouped = sample_rewards.reshape(-1, num_samples)  # [B, K]
    group_mean = grouped.mean(dim=1, keepdim=True)

    if advantage_type == "loo":
        # Leave-one-out baseline: mean of the *other* samples in the group,
        # equivalent to (r - mean) * K / (K - 1)
        advantages = (grouped - group_mean) * (num_samples / (num_samples - 1))
    elif advantage_type == "mean":
        advantages = grouped - group_mean
    elif advantage_type == "grpo":
        group_std = grouped.std(dim=1, keepdim=True)
        advantages = (grouped - group_mean) / (group_std + eps)
    else:
        raise ValueError(f"Unknown advantage_type: {advantage_type}")

    return advantages.reshape(-1)


def compute_policy_gradient_loss(
    per_token_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    old_per_token_log_probs: Optional[torch.Tensor] = None,
    clip_range: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """REINFORCE policy-gradient loss.

    Args:
        per_token_log_probs: [N, S-1] differentiable log-probs from re-scoring
            (zero at masked positions)
        advantages: [N] per-sequence advantages (detached inside)
        mask: [N, S-1] valid-token mask (up to and including first EOS)
        old_per_token_log_probs: Generation-time log-probs; only used by the
            future PPO-style clipped objective
        clip_range: PPO clip epsilon. Setting this (together with
            old_per_token_log_probs) is the extension point for a clipped
            surrogate objective (GRPO/PPO); not implemented yet.

    Returns:
        (loss, stats) where stats contains sequence-level log-prob diagnostics
    """
    if clip_range is not None:
        raise NotImplementedError(
            "Clipped surrogate objective (PPO/GRPO) is not implemented yet; "
            "old_per_token_log_probs/clip_range are the extension hook"
        )

    log_probs = torch.where(
        mask, per_token_log_probs, torch.zeros_like(per_token_log_probs)
    )
    sequence_log_probs = log_probs.sum(dim=1)  # [N]

    loss = -(advantages.detach() * sequence_log_probs).mean()

    stats = {
        "policy/seq_log_prob_mean": sequence_log_probs.mean().item(),
        "policy/tokens_per_seq_mean": mask.float().sum(dim=1).mean().item(),
    }
    return loss, stats


class REINFORCETrainer:
    """Computes REINFORCE losses for Pix2Seq RL fine-tuning.

    The trainer does not own the optimizer: call ``compute_losses`` inside the
    training loop, then run backward/clip/step there (e.g. within
    ``accelerator.accumulate(model)``).
    """

    def __init__(
        self,
        model: nn.Module,
        token_processor: TokenProcessor,
        reward_fn: RecallReward,
        iou_loss_fn: Optional[IoUSupervisionLoss] = None,
        config: Optional[RLTrainingConfig] = None,
        accelerator: Optional["Accelerator"] = None,
    ):
        """Initialize REINFORCE trainer.

        Args:
            model: Pix2Seq model. Pass the accelerator-prepared (wrapped) model
                so the re-scoring forward pass synchronises gradients under DDP.
            token_processor: TokenProcessor for decoding sequences
            reward_fn: Reward function (RecallReward)
            iou_loss_fn: Optional IoU supervision loss for confidence training
            config: Training configuration
            accelerator: Optional Accelerator for multi-GPU training
        """
        self.model = model
        self.token_processor = token_processor
        self.reward_fn = reward_fn
        self.iou_loss_fn = iou_loss_fn
        self.config = config or RLTrainingConfig()
        self.accelerator = accelerator

        self.max_gen_len = self.config.max_gen_len or token_processor.max_seq_len

        unwrapped = self._get_unwrapped_model()
        has_active_dropout = any(
            isinstance(m, nn.Dropout) and m.p > 0 for m in unwrapped.modules()
        )
        if has_active_dropout:
            warnings.warn(
                "Model has active dropout. Rollouts run in eval mode while "
                "re-scoring runs in train mode, so with dropout enabled the "
                "re-scored distribution will not match the sampling "
                "distribution. Set dropout/drop_path to 0.0 for RL fine-tuning."
            )

    def _get_unwrapped_model(self):
        """Get the unwrapped model (handles DDP wrapping)."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model

    @torch.no_grad()
    def sample_rollouts(self, images: torch.Tensor) -> Rollout:
        """Sample K sequences per image (plus greedy baseline if configured).

        Generation runs on the unwrapped model in eval mode (the KV cache
        requires eval mode, and generation carries no gradients).

        Args:
            images: [B, C, H, W] input images

        Returns:
            Rollout with sequences grouped per image in repeat_interleave order
        """
        unwrapped = self._get_unwrapped_model()
        was_training = unwrapped.training
        unwrapped.eval()

        try:
            sequences, _, _, gen_log_probs = unwrapped.infer(
                images=images,
                max_seq_len=self.max_gen_len,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                greedy=False,
                return_log_probs=True,
                num_samples=self.config.num_samples_per_image,
            )

            greedy_sequences = None
            if self.config.advantage_type == "greedy":
                greedy_sequences, _, _ = unwrapped.infer(
                    images=images,
                    max_seq_len=self.max_gen_len,
                    temperature=self.config.temperature,
                    greedy=True,
                )
                greedy_sequences = greedy_sequences.clone()
        finally:
            if was_training:
                unwrapped.train()

        # Clone to escape inference-mode tensor status (autograd needs to save
        # the sequences during re-scoring)
        return Rollout(
            sequences=sequences.clone(),
            gen_log_probs=gen_log_probs.clone(),
            greedy_sequences=greedy_sequences,
        )

    def compute_losses(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total RL loss for one batch.

        Args:
            batch: Dictionary containing:
                - image: [B, C, H, W] input images
                - gt_boxes: List of [M_i, 4] ground truth boxes (normalized XYXY)
                - gt_labels: List of [M_i] ground truth labels

        Returns:
            (total_loss, stats). The caller runs backward/step.
        """
        images = batch["image"]
        gt_boxes_list = batch["gt_boxes"]
        gt_labels_list = batch["gt_labels"]
        num_samples = self.config.num_samples_per_image

        # 1. Rollout (no gradients)
        rollout = self.sample_rollouts(images)

        # 2. Rewards and advantages. GT lists are repeated per sample to match
        # the repeat_interleave grouping of the sampled sequences
        gt_boxes_rep = [b for b in gt_boxes_list for _ in range(num_samples)]
        gt_labels_rep = [
            labels for labels in gt_labels_list for _ in range(num_samples)
        ]

        sample_rewards = self.reward_fn(
            sequences=rollout.sequences,
            gt_boxes_list=gt_boxes_rep,
            gt_labels_list=gt_labels_rep,
        )

        greedy_rewards = None
        if rollout.greedy_sequences is not None:
            greedy_rewards = self.reward_fn(
                sequences=rollout.greedy_sequences,
                gt_boxes_list=gt_boxes_list,
                gt_labels_list=gt_labels_list,
            )

        advantages = compute_advantages(
            sample_rewards=sample_rewards,
            num_samples=num_samples,
            advantage_type=self.config.advantage_type,
            greedy_rewards=greedy_rewards,
        )

        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        # 3. Teacher-forced re-scoring through the wrapped model (grads + DDP
        # sync). Train mode to match MLE training behaviour; dropout should be
        # 0 (checked at init) so this equals the eval-mode sampling distribution
        self.model.train()
        use_iou_loss = self.iou_loss_fn is not None
        rescored = rescore_sequences(
            model=self.model,
            images=images,
            sequences=rollout.sequences,
            token_processor=self.token_processor,
            temperature=self.config.temperature,
            num_samples_per_image=num_samples,
            return_class_logits=use_iou_loss,
        )

        # 4. Losses
        pg_loss, pg_stats = compute_policy_gradient_loss(
            per_token_log_probs=rescored.log_probs,
            advantages=advantages,
            mask=rescored.mask,
        )

        iou_loss = torch.zeros((), device=pg_loss.device)
        if use_iou_loss:
            pred_boxes, _, pred_confidences = extract_object_confidences(
                sequences=rollout.sequences,
                class_logits=rescored.class_logits,
                token_processor=self.token_processor,
            )
            iou_loss = self.iou_loss_fn(
                pred_boxes_list=pred_boxes,
                pred_confidences_list=pred_confidences,
                gt_boxes_list=gt_boxes_rep,
                device=pg_loss.device,
            )

        total_loss = pg_loss + self.config.iou_loss_weight * iou_loss

        stats = {
            "loss/total": total_loss.item(),
            "loss/rl": pg_loss.item(),
            "loss/iou": iou_loss.item(),
            "reward/sample_mean": sample_rewards.mean().item(),
            "reward/baseline_mean": (
                greedy_rewards.mean().item()
                if greedy_rewards is not None
                else sample_rewards.reshape(-1, num_samples)
                .mean(dim=1)
                .mean()
                .item()
            ),
            "reward/advantage_mean": advantages.mean().item(),
            "reward/advantage_std": (
                advantages.std().item() if advantages.numel() > 1 else 0.0
            ),
            **pg_stats,
        }

        return total_loss, stats
