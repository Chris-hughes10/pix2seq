"""REINFORCE trainer with self-critical baseline (SCST) for Pix2Seq.

Implements Self-Critical Sequence Training as described in:
- "Self-critical Sequence Training for Image Captioning" (Rennie et al., 2017)
- "Tuning Computer Vision Models with Task Rewards" (Pinto et al., 2023)

The training loop structure follows the pattern from simple-ppo.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from data.tokenizer import TokenProcessor

from .rewards import RecallReward
from .iou_loss import IoUSupervisionLoss


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""

    # Baseline type
    baseline: str = "greedy"  # "greedy" or "sample_mean"

    # Loss weights
    iou_loss_weight: float = 1.0

    # Advantage normalization
    normalize_advantages: bool = True

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Generation parameters for sampling
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.4

    # Logging
    log_interval: int = 10


class REINFORCETrainer:
    """REINFORCE trainer with self-critical baseline for Pix2Seq.

    This trainer implements the SCST algorithm:
    1. Sample sequences from the model
    2. Compute greedy baseline sequences
    3. Compute rewards for both
    4. Update policy using REINFORCE gradient with advantage = sample_reward - baseline_reward
    """

    def __init__(
        self,
        model: nn.Module,
        token_processor: TokenProcessor,
        optimizer: torch.optim.Optimizer,
        reward_fn: RecallReward,
        iou_loss_fn: Optional[IoUSupervisionLoss] = None,
        config: Optional[RLTrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize REINFORCE trainer.

        Args:
            model: Pix2Seq model (must have infer method with new RL parameters)
            token_processor: TokenProcessor for decoding sequences
            optimizer: Optimizer for policy updates
            reward_fn: Reward function (RecallReward)
            iou_loss_fn: Optional IoU supervision loss
            config: Training configuration
            device: Device to run training on
        """
        self.model = model
        self.token_processor = token_processor
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.iou_loss_fn = iou_loss_fn
        self.config = config or RLTrainingConfig()
        self.device = device or next(model.parameters()).device

        # Training stats
        self.global_step = 0
        self.stats_history: List[Dict[str, float]] = []

    def collect_samples(
        self,
        images: torch.Tensor,
        gt_boxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Collect sampled and baseline sequences with rewards.

        Args:
            images: [B, C, H, W] input images
            gt_boxes_list: List of [M_i, 4] ground truth boxes per image
            gt_labels_list: List of [M_i] ground truth labels per image

        Returns:
            Dictionary containing:
                - sampled_seqs: [B, S] sampled sequences
                - sampled_log_probs: [B, S] log probs of sampled tokens
                - sampled_logits: [B, N, V] class logits for sampled sequences
                - sample_rewards: [B] rewards for sampled sequences
                - baseline_rewards: [B] rewards for baseline sequences
        """
        # Ensure model is in eval mode for generation but track gradients
        was_training = self.model.training
        self.model.eval()

        # Sample sequences with log probabilities
        sampled_seqs, sampled_logits, _, sampled_log_probs = self.model.infer(
            images=images,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            greedy=False,
            return_log_probs=True,
            training_mode=True,  # Allow gradient flow through log probs
        )

        # Compute baseline sequences (greedy, no gradients needed)
        with torch.no_grad():
            baseline_seqs, baseline_logits, _ = self.model.infer(
                images=images,
                greedy=True,
                return_log_probs=False,
                training_mode=False,
            )

        # Restore training mode if needed
        if was_training:
            self.model.train()

        # Compute rewards
        sample_rewards = self.reward_fn(
            sequences=sampled_seqs,
            class_logits=sampled_logits,
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
        )

        baseline_rewards = self.reward_fn(
            sequences=baseline_seqs,
            class_logits=baseline_logits,
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
        )

        return {
            "sampled_seqs": sampled_seqs,
            "sampled_log_probs": sampled_log_probs,
            "sampled_logits": sampled_logits,
            "sample_rewards": sample_rewards,
            "baseline_rewards": baseline_rewards,
        }

    def compute_advantages(
        self,
        sample_rewards: torch.Tensor,
        baseline_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages using self-critical baseline.

        Args:
            sample_rewards: [B] rewards for sampled sequences
            baseline_rewards: [B] rewards for baseline (greedy) sequences

        Returns:
            advantages: [B] advantage values
        """
        advantages = sample_rewards - baseline_rewards

        if self.config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def get_valid_token_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """Get mask for valid (non-padding) tokens.

        Args:
            sequences: [B, S] token sequences

        Returns:
            mask: [B, S] boolean mask (True for valid tokens)
        """
        # Valid tokens are everything except padding (after EOS)
        padding_token = self.token_processor.PADDING_TOKEN
        eos_token = self.token_processor.EOS_TOKEN

        # Create mask: True for non-padding tokens
        mask = sequences != padding_token

        # Also mask out tokens after EOS
        batch_size, seq_len = sequences.shape
        for b in range(batch_size):
            eos_positions = (sequences[b] == eos_token).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                # Keep tokens up to and including first EOS
                mask[b, first_eos + 1:] = False

        return mask

    def compute_reinforce_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute REINFORCE policy gradient loss.

        Args:
            log_probs: [B, S] log probabilities of each token
            advantages: [B] advantage values (sample_reward - baseline_reward)
            mask: [B, S] valid token mask

        Returns:
            loss: Scalar REINFORCE loss
        """
        # Sum log probs over valid tokens in sequence
        # log_probs is [B, S], we need to account for the fact that
        # log_probs may be shorter than mask if early EOS
        min_len = min(log_probs.size(1), mask.size(1))
        log_probs = log_probs[:, :min_len]
        mask = mask[:, :min_len]

        masked_log_probs = log_probs * mask.float()
        sequence_log_probs = masked_log_probs.sum(dim=1)  # [B]

        # REINFORCE: maximize expected reward = minimize -advantage * log_prob
        # Detach advantages to prevent gradient flow through reward computation
        loss = -(advantages.detach() * sequence_log_probs).mean()

        return loss

    def compute_iou_supervision_loss(
        self,
        sampled_seqs: torch.Tensor,
        sampled_logits: torch.Tensor,
        gt_boxes_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute IoU supervision loss for confidence prediction.

        Args:
            sampled_seqs: [B, S] sampled token sequences
            sampled_logits: [B, N, V] class logits
            gt_boxes_list: List of [M_i, 4] ground truth boxes per image

        Returns:
            loss: Scalar IoU supervision loss
        """
        if self.iou_loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        # Decode sequences to get boxes and confidences
        pred_boxes_list, _, pred_scores_list = (
            self.token_processor.post_process_sequences(
                sequences=sampled_seqs,
                class_logits=sampled_logits,
                confidence_threshold=0.0,  # Include all predictions
            )
        )

        # Filter out None scores and convert to proper format
        valid_boxes = []
        valid_scores = []
        valid_gt = []

        for pred_boxes, pred_scores, gt_boxes in zip(
            pred_boxes_list, pred_scores_list, gt_boxes_list
        ):
            if pred_scores is not None and len(pred_boxes) > 0:
                valid_boxes.append(pred_boxes)
                valid_scores.append(pred_scores)
                valid_gt.append(gt_boxes)

        if len(valid_boxes) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return self.iou_loss_fn(valid_boxes, valid_scores, valid_gt)

    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing:
                - image: [B, C, H, W] input images
                - gt_boxes: List of [M_i, 4] ground truth boxes
                - gt_labels: List of [M_i] ground truth labels

        Returns:
            Dictionary of training statistics
        """
        images = batch["image"].to(self.device)
        gt_boxes_list = batch["gt_boxes"]
        gt_labels_list = batch["gt_labels"]

        # Collect samples and compute rewards
        samples = self.collect_samples(images, gt_boxes_list, gt_labels_list)

        # Compute advantages
        advantages = self.compute_advantages(
            samples["sample_rewards"],
            samples["baseline_rewards"],
        )

        # Get valid token mask
        mask = self.get_valid_token_mask(samples["sampled_seqs"])

        # Compute REINFORCE loss
        rl_loss = self.compute_reinforce_loss(
            samples["sampled_log_probs"],
            advantages,
            mask,
        )

        # Compute IoU supervision loss
        iou_loss = self.compute_iou_supervision_loss(
            samples["sampled_seqs"],
            samples["sampled_logits"],
            gt_boxes_list,
        )

        # Total loss
        total_loss = rl_loss + self.config.iou_loss_weight * iou_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

        # Collect statistics
        stats = {
            "loss/total": total_loss.item(),
            "loss/rl": rl_loss.item(),
            "loss/iou": iou_loss.item() if isinstance(iou_loss, torch.Tensor) else iou_loss,
            "reward/sample_mean": samples["sample_rewards"].mean().item(),
            "reward/baseline_mean": samples["baseline_rewards"].mean().item(),
            "reward/advantage_mean": advantages.mean().item(),
            "reward/advantage_std": advantages.std().item() if len(advantages) > 1 else 0.0,
        }

        self.global_step += 1
        self.stats_history.append(stats)

        return stats

    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """Get averaged statistics over recent steps.

        Args:
            window: Number of recent steps to average

        Returns:
            Dictionary of averaged statistics
        """
        if not self.stats_history:
            return {}

        recent = self.stats_history[-window:]
        keys = recent[0].keys()

        return {
            key: sum(s[key] for s in recent) / len(recent)
            for key in keys
        }
