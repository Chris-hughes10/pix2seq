"""Tests for the RL module."""

import pytest
import torch

# conftest.py adds src/pix2seq to sys.path
from rl.iou_loss import IoUSupervisionLoss
from rl.rewards import RecallReward, compute_iou, compute_recall_at_iou


class TestComputeIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU = 1."""
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        iou = compute_iou(boxes, boxes)
        assert torch.allclose(iou, torch.tensor([[1.0]]))

    def test_non_overlapping_boxes(self):
        """Non-overlapping boxes should have IoU = 0."""
        boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        boxes2 = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
        iou = compute_iou(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([[0.0]]))

    def test_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])  # Area = 4
        boxes2 = torch.tensor([[1.0, 1.0, 3.0, 3.0]])  # Area = 4
        # Intersection: [1,1] to [2,2] = 1x1 = 1
        # Union: 4 + 4 - 1 = 7
        # IoU = 1/7
        iou = compute_iou(boxes1, boxes2)
        expected = torch.tensor([[1.0 / 7.0]])
        assert torch.allclose(iou, expected, atol=1e-6)

    def test_multiple_boxes(self):
        """Test IoU computation for multiple boxes."""
        boxes1 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
        ])
        boxes2 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5, 1.5],
        ])
        iou = compute_iou(boxes1, boxes2)
        assert iou.shape == (2, 2)
        # First box matches first GT exactly
        assert torch.isclose(iou[0, 0], torch.tensor(1.0))
        # Second box doesn't overlap with either GT
        assert torch.isclose(iou[1, 0], torch.tensor(0.0))


class TestRecallAtIoU:
    """Tests for recall computation."""

    def test_perfect_recall(self):
        """All GT boxes matched should give recall = 1."""
        pred_boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
        ])
        gt_boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
        ])
        recall = compute_recall_at_iou(pred_boxes, gt_boxes, iou_threshold=0.5)
        assert recall == 1.0

    def test_zero_recall(self):
        """No GT boxes matched should give recall = 0."""
        pred_boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
        ])
        gt_boxes = torch.tensor([
            [5.0, 5.0, 6.0, 6.0],
        ])
        recall = compute_recall_at_iou(pred_boxes, gt_boxes, iou_threshold=0.5)
        assert recall == 0.0

    def test_partial_recall(self):
        """Some GT boxes matched should give partial recall."""
        pred_boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
        ])
        gt_boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [5.0, 5.0, 6.0, 6.0],
        ])
        recall = compute_recall_at_iou(pred_boxes, gt_boxes, iou_threshold=0.5)
        assert recall == 0.5

    def test_no_predictions(self):
        """No predictions should give recall = 0."""
        pred_boxes = torch.tensor([]).reshape(0, 4)
        gt_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        recall = compute_recall_at_iou(pred_boxes, gt_boxes, iou_threshold=0.5)
        assert recall == 0.0

    def test_no_ground_truth(self):
        """No GT should give recall = 1."""
        pred_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        gt_boxes = torch.tensor([]).reshape(0, 4)
        recall = compute_recall_at_iou(pred_boxes, gt_boxes, iou_threshold=0.5)
        assert recall == 1.0


class TestRecallReward:
    """Tests for the batch reward computation."""

    def test_call_decodes_sequences(self, token_processor):
        from conftest import make_sequence

        box = [0.2, 0.2, 0.6, 0.6]
        sequences = torch.cat(
            [
                make_sequence(token_processor, boxes=[box], labels=[1]),
                make_sequence(token_processor, boxes=[], labels=[]),  # empty
            ]
        )
        reward_fn = RecallReward(token_processor)
        rewards = reward_fn(
            sequences=sequences,
            gt_boxes_list=[torch.tensor([box]), torch.tensor([box])],
            gt_labels_list=[torch.tensor([1]), torch.tensor([1])],
        )
        assert rewards.shape == (2,)
        assert rewards.dtype == torch.float32
        assert rewards[0].item() == 1.0  # perfect prediction
        assert rewards[1].item() == 0.0  # no predictions

    def test_class_mismatch_gives_zero_recall(self, token_processor):
        from conftest import make_sequence

        box = [0.2, 0.2, 0.6, 0.6]
        sequences = make_sequence(token_processor, boxes=[box], labels=[2])
        reward_fn = RecallReward(token_processor)
        rewards = reward_fn(
            sequences=sequences,
            gt_boxes_list=[torch.tensor([box])],
            gt_labels_list=[torch.tensor([1])],  # different class
        )
        assert rewards[0].item() == 0.0

    def test_class_weights_applied_once(self, token_processor):
        """Weighted recall is a weighted average of per-class recalls."""
        weights = torch.tensor([1.0, 3.0, 1.0, 1.0, 1.0])
        reward_fn = RecallReward(token_processor, class_weights=weights)

        pred_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        pred_labels = torch.tensor([0])
        gt_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
        gt_labels = torch.tensor([0, 1])

        reward = reward_fn._compute_class_aware_reward(
            pred_boxes, gt_boxes, pred_labels, gt_labels
        )
        # class 0: recall 1 (weight 1, count 1); class 1: recall 0 (weight 3,
        # count 1) -> (1*1 + 0*3) / (1 + 3) = 0.25. Double-applied weights
        # would give (1*1 + 0*9) / (1 + 3) instead for other weightings;
        # the key check is the weighted average stays in [0, 1]
        assert abs(reward - 0.25) < 1e-6


class TestIoUSupervisionLoss:
    """Tests for IoU supervision loss."""

    def test_perfect_prediction(self):
        """Perfect predictions should have low loss."""
        loss_fn = IoUSupervisionLoss()

        pred_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]])]
        pred_confs = [torch.tensor([1.0])]  # Predicting IoU = 1
        gt_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]])]

        loss = loss_fn(pred_boxes, pred_confs, gt_boxes)
        assert loss.item() < 0.01  # Should be very small

    def test_wrong_confidence(self):
        """Wrong confidence should have high loss."""
        loss_fn = IoUSupervisionLoss()

        pred_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]])]
        pred_confs = [torch.tensor([0.0])]  # Predicting IoU = 0, but actual = 1
        gt_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]])]

        loss = loss_fn(pred_boxes, pred_confs, gt_boxes)
        assert loss.item() > 0.9  # Should be close to 1.0 (MSE of 1)

    def test_empty_predictions(self):
        """Empty predictions should return zero loss."""
        loss_fn = IoUSupervisionLoss()

        pred_boxes = [torch.tensor([]).reshape(0, 4)]
        pred_confs = [torch.tensor([])]
        gt_boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]])]

        loss = loss_fn(pred_boxes, pred_confs, gt_boxes)
        assert loss.item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
