"""IoU supervision loss for learning box confidence scores.

This loss trains the model's confidence predictions to match the actual
IoU with ground truth boxes, which helps with box ranking at test time.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .rewards import compute_iou


class IoUSupervisionLoss(nn.Module):
    """Supervised loss for box confidence prediction.

    Trains confidence scores to predict the IoU of each box with its
    best-matching ground truth box.
    """

    def __init__(self, min_iou_threshold: float = 0.0):
        """Initialize IoU supervision loss.

        Args:
            min_iou_threshold: Minimum IoU to consider a match.
                              Predictions below this get target IoU = 0.
        """
        super().__init__()
        self.min_iou_threshold = min_iou_threshold
        self.mse_loss = nn.MSELoss(reduction="none")

    def compute_target_ious(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target IoU for each prediction.

        Args:
            pred_boxes: [N, 4] predicted boxes in XYXY format
            gt_boxes: [M, 4] ground truth boxes in XYXY format

        Returns:
            target_ious: [N] IoU of each prediction with best-matching GT
        """
        if len(pred_boxes) == 0:
            return torch.tensor([], device=pred_boxes.device)

        if len(gt_boxes) == 0:
            return torch.zeros(len(pred_boxes), device=pred_boxes.device)

        iou_matrix = compute_iou(pred_boxes, gt_boxes)  # [N, M]
        best_ious, _ = iou_matrix.max(dim=1)  # [N]

        # Apply threshold
        best_ious = torch.where(
            best_ious >= self.min_iou_threshold,
            best_ious,
            torch.zeros_like(best_ious),
        )

        return best_ious

    def forward(
        self,
        pred_boxes_list: List[torch.Tensor],
        pred_confidences_list: List[torch.Tensor],
        gt_boxes_list: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute IoU supervision loss for a batch.

        Args:
            pred_boxes_list: List of [N_i, 4] predicted boxes per image
            pred_confidences_list: List of [N_i] confidence scores per image
            gt_boxes_list: List of [M_i, 4] ground truth boxes per image
            device: Device for the returned loss when there are no predictions
                (otherwise inferred from the predictions)

        Returns:
            loss: Scalar MSE loss between confidences and target IoUs
        """
        total_loss = 0.0
        total_predictions = 0

        for pred_boxes, pred_confs, gt_boxes in zip(
            pred_boxes_list, pred_confidences_list, gt_boxes_list
        ):
            if len(pred_boxes) == 0:
                continue

            # Ensure tensors are on same device
            device = pred_boxes.device
            if gt_boxes.device != device:
                gt_boxes = gt_boxes.to(device)
            if pred_confs.device != device:
                pred_confs = pred_confs.to(device)

            # Compute target IoUs
            target_ious = self.compute_target_ious(pred_boxes, gt_boxes)

            # MSE loss
            loss = self.mse_loss(pred_confs, target_ious)
            total_loss += loss.sum()
            total_predictions += len(pred_boxes)

        if total_predictions == 0:
            if device is None and len(gt_boxes_list) > 0:
                device = gt_boxes_list[0].device
            return torch.zeros((), device=device)

        return total_loss / total_predictions
