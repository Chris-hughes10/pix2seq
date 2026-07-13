"""Tests for prediction formatting/scaling and the mAP evaluation path.

The scaling test guards against the double-denormalization class of bug: boxes
from post_process_sequences are normalized [0,1] and must be denormalized
exactly once on the way to COCO evaluation.
"""

import torch
import torch.nn as nn

from conftest import IMAGE_SIZE, make_sequence

from data.base_dataset import coco80_to_coco91_lookup
from rl.evaluation import evaluate_map
from train_rl import RLCollator
from training.trainer import (
    format_predictions_for_evaluation,
    scale_bboxes_to_original_image_size,
)


class TestBoxScaling:
    def test_square_image_round_trip(self):
        """Normalized boxes on an unpadded square image scale by size only."""
        boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
        scaled = scale_bboxes_to_original_image_size(
            boxes,
            resized_hw=torch.tensor([32, 32]),
            original_hw=torch.tensor([32, 32]),
            is_padded=True,
            normalized=True,
        )
        torch.testing.assert_close(scaled, torch.tensor([[8.0, 8.0, 24.0, 24.0]]))

    def test_padded_rectangular_image(self):
        """Padding offset is removed and the pad scale undone."""
        # Original 16x32 (h x w) letterboxed into 32x32: pad_scale = 1,
        # vertical padding (32 - 16)/2 = 8
        boxes = torch.tensor([[0.25, 0.5, 0.75, 0.75]])  # x1,y1,x2,y2 normalized
        scaled = scale_bboxes_to_original_image_size(
            boxes,
            resized_hw=torch.tensor([32, 32]),
            original_hw=torch.tensor([16, 32]),
            is_padded=True,
            normalized=True,
        )
        # Denormalized: (8, 16, 24, 24); remove y padding of 8 -> (8, 8, 24, 16)
        torch.testing.assert_close(scaled, torch.tensor([[8.0, 8.0, 24.0, 16.0]]))

    def test_format_predictions_scales_once(self):
        """format_predictions_for_evaluation expects normalized boxes."""
        predictions_list = []
        format_predictions_for_evaluation(
            boxes_list=[torch.tensor([[0.25, 0.25, 0.75, 0.75]])],
            labels_list=[torch.tensor([2])],
            scores_list=[torch.tensor([0.9])],
            image_ids=[torch.tensor(7)],
            original_sizes=[torch.tensor([32, 32])],
            resized_sizes=[torch.tensor([32, 32])],
            predictions_list=predictions_list,
        )
        assert len(predictions_list) == 1
        row = predictions_list[0][0]
        torch.testing.assert_close(row[:4], torch.tensor([8.0, 8.0, 24.0, 24.0]))
        assert abs(row[4].item() - 0.9) < 1e-6  # score
        assert row[5].item() == 2  # class id
        assert row[6].item() == 7  # image id


class TestRLCollator:
    def test_batch_format(self, token_processor):
        items = []
        for image_id, num_boxes in [(1, 2), (2, 1)]:
            items.append(
                {
                    "image": torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
                    "image_id": torch.tensor(image_id),
                    "orig_image_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
                    "unpadded_image_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
                    "boxes": torch.rand(5, 4),  # padded to 5
                    "labels": torch.zeros(5, dtype=torch.long),
                    "num_boxes": torch.tensor(num_boxes),
                }
            )
        batch = RLCollator(token_processor)(items)
        assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert len(batch["gt_boxes"]) == 2
        assert batch["gt_boxes"][0].shape == (2, 4)  # sliced to num_boxes
        assert batch["gt_boxes"][1].shape == (1, 4)
        assert batch["gt_labels"][0].shape == (2,)


class _StubDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _StubModel(nn.Module):
    """Returns fixed sequences that decode exactly to the ground truth."""

    def __init__(self, sequences, vocab_size, token_processor):
        super().__init__()
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.token_processor = token_processor

    def infer(self, images=None, max_seq_len=None, greedy=False, **kwargs):
        batch_size = images.size(0)
        sequences = self.sequences[:batch_size]
        num_objects = (sequences.size(1) - 1) // 5
        # High logit at each sampled class token -> confidence ~1
        class_logits = torch.zeros(batch_size, num_objects, self.vocab_size)
        for b in range(batch_size):
            for k in range(num_objects):
                class_token = sequences[b, 1 + k * 5 + 4]
                class_logits[b, k, class_token] = 12.0
        return sequences, class_logits, None


class TestEvaluateMap:
    def test_perfect_predictions_give_high_map(self, token_processor):
        box = [0.25, 0.25, 0.75, 0.75]
        label = 0
        lookup = coco80_to_coco91_lookup()

        items, sequences, annotations, images_json = [], [], [], []
        for image_id in (1, 2):
            items.append(
                {
                    "image": torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
                    "image_id": torch.tensor(image_id),
                    "orig_image_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
                    "unpadded_image_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE]),
                    "boxes": torch.tensor([box]),
                    "labels": torch.tensor([label]),
                    "num_boxes": torch.tensor(1),
                }
            )
            sequences.append(
                make_sequence(token_processor, boxes=[box], labels=[label], seq_len=8)
            )
            # GT bbox in original pixel xywh; quantization shifts the decoded
            # box slightly, so use the dequantized coordinates as GT
            coords = token_processor.dequantize(
                token_processor.quantize(torch.tensor(box))
            )
            x1, y1, x2, y2 = (coords * IMAGE_SIZE).tolist()
            annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": lookup[label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            images_json.append(
                {"id": image_id, "width": IMAGE_SIZE, "height": IMAGE_SIZE}
            )

        val_json = {
            "images": images_json,
            "annotations": annotations,
            "categories": [{"id": lookup[label]}],
        }

        model = _StubModel(
            torch.cat(sequences), token_processor.vocab_size, token_processor
        )
        mAP = evaluate_map(
            model=model,
            eval_dataset=_StubDataset(items),
            collate_fn=RLCollator(token_processor),
            token_processor=token_processor,
            val_json=val_json,
            device=torch.device("cpu"),
            max_seq_len=8,
            batch_size=2,
        )
        assert mAP > 0.9, f"expected near-perfect mAP, got {mAP}"
