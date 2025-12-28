"""Reinforcement learning module for Pix2Seq.

Implements SCST (Self-Critical Sequence Training) for fine-tuning
the model directly on task rewards like recall/mAP.
"""

from .rewards import RecallReward, compute_iou, compute_recall_at_iou
from .iou_loss import IoUSupervisionLoss
from .reinforce import REINFORCETrainer, RLTrainingConfig

__all__ = [
    "RecallReward",
    "compute_iou",
    "compute_recall_at_iou",
    "IoUSupervisionLoss",
    "REINFORCETrainer",
    "RLTrainingConfig",
]
