"""Reinforcement learning module for Pix2Seq.

Implements REINFORCE fine-tuning on task rewards (recall + IoU-supervised
confidences) with multi-sample, GRPO-style advantage baselines, following
"Tuning Computer Vision Models with Task Rewards" (arXiv:2302.08242).
"""

from .iou_loss import IoUSupervisionLoss
from .reinforce import (
    REINFORCETrainer,
    RLTrainingConfig,
    Rollout,
    compute_advantages,
    compute_policy_gradient_loss,
)
from .rescoring import (
    RescoreOutput,
    build_constraint_masks,
    compute_valid_token_mask,
    extract_object_confidences,
    rescore_sequences,
)
from .rewards import RecallReward, compute_iou, compute_recall_at_iou

__all__ = [
    "RecallReward",
    "compute_iou",
    "compute_recall_at_iou",
    "IoUSupervisionLoss",
    "REINFORCETrainer",
    "RLTrainingConfig",
    "Rollout",
    "compute_advantages",
    "compute_policy_gradient_loss",
    "RescoreOutput",
    "build_constraint_masks",
    "compute_valid_token_mask",
    "extract_object_confidences",
    "rescore_sequences",
]
