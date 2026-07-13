"""Reward computation for SCST training.

Implements per-image recall-based rewards as described in
"Tuning Computer Vision Models with Task Rewards" (arXiv:2302.08242).
"""

from typing import List, Optional, Tuple

import torch

from data.tokenizer import TokenProcessor


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in XYXY format
        boxes2: [M, 4] boxes in XYXY format

    Returns:
        iou: [N, M] pairwise IoU values
    """
    # Compute intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute union
    union = area1[:, None] + area2[None, :] - intersection

    # Compute IoU
    iou = intersection / (union + 1e-8)

    return iou


def compute_recall_at_iou(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """Compute recall at a given IoU threshold.

    Uses greedy matching: each GT box is matched to the prediction with
    highest IoU (if above threshold).

    Args:
        pred_boxes: [N, 4] predicted boxes in XYXY format
        gt_boxes: [M, 4] ground truth boxes in XYXY format
        iou_threshold: IoU threshold for considering a match

    Returns:
        recall: Fraction of GT boxes that are matched
    """
    if len(gt_boxes) == 0:
        # No ground truth boxes - return 1.0 (nothing to recall)
        return 1.0

    if len(pred_boxes) == 0:
        # No predictions - recall is 0
        return 0.0

    # Compute pairwise IoU
    iou_matrix = compute_iou(pred_boxes, gt_boxes)  # [N, M]

    # For each GT box, find the best matching prediction
    max_iou_per_gt, _ = iou_matrix.max(dim=0)  # [M]

    # Count GT boxes with IoU >= threshold
    matched_gt = (max_iou_per_gt >= iou_threshold).sum().item()

    recall = matched_gt / len(gt_boxes)

    return recall


class RecallReward:
    """Computes per-image recall reward for SCST training.

    The reward is the recall (fraction of GT boxes matched) averaged
    across multiple IoU thresholds.
    """

    def __init__(
        self,
        token_processor: TokenProcessor,
        iou_thresholds: Optional[List[float]] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize the reward function.

        Args:
            token_processor: TokenProcessor for decoding sequences to boxes
            iou_thresholds: List of IoU thresholds to average over.
                           Default: [0.5, 0.55, ..., 0.95] (COCO style)
            class_weights: Optional per-class weights based on frequency.
                          Shape: [num_classes]
        """
        self.token_processor = token_processor
        self.iou_thresholds = iou_thresholds or [
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
        ]
        self.class_weights = class_weights

    def compute_single_image_reward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        pred_labels: Optional[torch.Tensor] = None,
        gt_labels: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute reward for a single image.

        Args:
            pred_boxes: [N, 4] predicted boxes in XYXY format
            gt_boxes: [M, 4] ground truth boxes in XYXY format
            pred_labels: [N] predicted class labels (optional, for class-aware matching)
            gt_labels: [M] ground truth class labels (optional)

        Returns:
            reward: Recall averaged over IoU thresholds
        """
        if len(gt_boxes) == 0:
            return 1.0

        if len(pred_boxes) == 0:
            return 0.0

        # If class labels provided, compute recall per class and average
        if pred_labels is not None and gt_labels is not None:
            return self._compute_class_aware_reward(
                pred_boxes, gt_boxes, pred_labels, gt_labels
            )

        # Class-agnostic recall
        recalls = []
        for threshold in self.iou_thresholds:
            recall = compute_recall_at_iou(pred_boxes, gt_boxes, threshold)
            recalls.append(recall)

        return sum(recalls) / len(recalls)

    def _compute_class_aware_reward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> float:
        """Compute class-aware recall reward.

        Matches predictions to GT only within the same class.
        """
        unique_classes = gt_labels.unique()
        class_recalls = []
        class_counts = []

        for cls in unique_classes:
            # Filter boxes by class
            pred_mask = pred_labels == cls
            gt_mask = gt_labels == cls

            pred_cls_boxes = pred_boxes[pred_mask]
            gt_cls_boxes = gt_boxes[gt_mask]

            # Compute recall for this class
            recalls = []
            for threshold in self.iou_thresholds:
                recall = compute_recall_at_iou(
                    pred_cls_boxes, gt_cls_boxes, threshold
                )
                recalls.append(recall)

            avg_recall = sum(recalls) / len(recalls)

            # Weight by class frequency if weights provided
            if self.class_weights is not None:
                weight = self.class_weights[cls].item()
            else:
                weight = 1.0

            class_recalls.append(avg_recall * weight)
            class_counts.append(gt_mask.sum().item() * weight)

        # Weighted average across classes
        total_weight = sum(class_counts)
        if total_weight == 0:
            return 1.0

        weighted_recall = sum(
            r * c for r, c in zip(class_recalls, class_counts)
        ) / total_weight

        return weighted_recall

    def __call__(
        self,
        sequences: torch.Tensor,
        class_logits: torch.Tensor,
        gt_boxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
        confidence_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Compute rewards for a batch of sequences.

        Args:
            sequences: [B, S] generated token sequences
            class_logits: [B, N, V] logits for class tokens
            gt_boxes_list: List of [M_i, 4] ground truth boxes per image
            gt_labels_list: List of [M_i] ground truth labels per image
            confidence_threshold: Minimum confidence to include a prediction

        Returns:
            rewards: [B] per-image reward values
        """
        batch_size = sequences.size(0)
        device = sequences.device

        # Decode sequences to boxes
        pred_boxes_list, pred_labels_list, pred_scores_list = (
            self.token_processor.post_process_sequences(
                sequences=sequences,
                class_logits=class_logits,
                confidence_threshold=confidence_threshold,
            )
        )

        rewards = []
        for i in range(batch_size):
            pred_boxes = pred_boxes_list[i]
            pred_labels = pred_labels_list[i]
            gt_boxes = gt_boxes_list[i]
            gt_labels = gt_labels_list[i]

            # Ensure tensors are on same device
            if gt_boxes.device != device:
                gt_boxes = gt_boxes.to(device)
            if gt_labels.device != device:
                gt_labels = gt_labels.to(device)

            reward = self.compute_single_image_reward(
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                pred_labels=pred_labels,
                gt_labels=gt_labels,
            )
            rewards.append(reward)

        return torch.tensor(rewards, device=device, dtype=torch.float32)

    def compute_best_iou_per_prediction(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute best IoU for each prediction (for IoU supervision loss).

        Args:
            pred_boxes: [N, 4] predicted boxes
            gt_boxes: [M, 4] ground truth boxes

        Returns:
            best_ious: [N] best IoU value for each prediction
        """
        if len(pred_boxes) == 0:
            return torch.tensor([], device=pred_boxes.device)

        if len(gt_boxes) == 0:
            return torch.zeros(len(pred_boxes), device=pred_boxes.device)

        iou_matrix = compute_iou(pred_boxes, gt_boxes)  # [N, M]
        best_ious, _ = iou_matrix.max(dim=1)  # [N]

        return best_ious
