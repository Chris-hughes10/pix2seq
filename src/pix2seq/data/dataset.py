"""PyTorch dataset implementation for Pix2Seq object detection.

This module provides a modular implementation of the Pix2Seq data processing pipeline,
with support for different base datasets and customizable augmentation strategies.

Key components:
- Base dataset interface for different detection datasets
- Image preprocessing and normalization
- Image and box augmentation pipelines
- Token sequence generation for training
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentations import BBoxAugmentation, ImageAugmentor
from data.base_dataset import COCOBaseDataset
from data.tokenizer import TokenProcessor


class Pix2SeqDataset(Dataset):
    """PyTorch dataset for Pix2Seq object detection."""

    def __init__(
        self,
        base_dataset: COCOBaseDataset,
        num_classes: int,
        training: bool = True,
        max_num_objects: int = 100,
        image_size: int = 640,
        jitter_scale: Tuple[float, float] = (0.3, 2.0),
        color_jitter_strength: float = 0.4,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.categories = base_dataset.get_categories()
        self.training = training
        self.max_instances = max_num_objects

        # Initialize processing components
        self.augmentor = ImageAugmentor(
            image_size=image_size,
            jitter_scale=jitter_scale,
            color_jitter_strength=color_jitter_strength,
            training=training,
            enable_replay=False,
        )

        self.bbox_augmentor = BBoxAugmentation(num_classes=num_classes)

    def __getitem__(self, idx: int) -> dict:
        # Get raw data from base dataset
        image, boxes, labels, image_id, orig_size = self.base_dataset[idx]

        # Apply image augmentations (returns uint8 image and normalized boxes)
        image, boxes, labels, unpadded_size = self.augmentor(
            image, boxes, labels, normalize_boxes=True
        )

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        if len(boxes) > self.max_instances:
            # Track occurrence of truncation for monitoring
            print(
                f"Image {image_id} has {len(boxes)} boxes, exceeding max_instances={self.max_instances}. "
                f"Will {'randomly sample' if self.training else 'truncate to first'} {self.max_instances} boxes."
            )

            if self.training:
                # During training, randomly sample max_instances boxes
                indices = torch.randperm(len(boxes))[: self.max_instances]
                boxes = boxes[indices]
                labels = labels[indices]
            else:
                # During evaluation, take first max_instances boxes to ensure deterministic behavior
                boxes = boxes[: self.max_instances]
                labels = labels[: self.max_instances]

        if self.training:
            # Apply box augmentations to get correct number of instances
            n_noise = max(0, self.max_instances - len(boxes))  # Ensure non-negative
            boxes, labels = self.bbox_augmentor.augment_bbox(
                boxes,
                labels,
                max_jitter=0.05,
                n_noise_bbox=n_noise,
                mix_rate=0.5,
            )

        num_boxes = len(boxes)

        # Normalize image to [0,1] and convert to CHW format
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        image = image.permute(2, 0, 1)  # HWC -> CHW

        return {
            "image": image,  # [3,H,W] float32 normalized
            "boxes": boxes,  # [max_instances,4] float32 normalized XYXY
            "labels": labels,  # [max_instances] long
            "image_id": image_id,  # int
            "orig_image_size": torch.tensor(orig_size),  # [2] long
            "unpadded_image_size": torch.tensor(unpadded_size),  # [2] long
            "num_boxes": torch.tensor(num_boxes),  # int - actual number before padding
        }

    def __len__(self) -> int:
        return len(self.base_dataset)


class Pix2SeqCollator:
    def __init__(
        self, token_processor: TokenProcessor, corrupt_and_randomise: bool = False
    ):
        self.token_processor = token_processor
        self._is_training = False
        self.corrupt_and_randomise = corrupt_and_randomise

    @property  # Add property to ensure consistent state
    def is_training(self):
        return self._is_training

    def set_mode(self, is_training=True):
        if self.corrupt_and_randomise:
            self._is_training = is_training
            self.token_processor._corrupt_class_labels = is_training
            self.token_processor.random_order = is_training

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # First handle fixed size tensors
        batch_data = {
            "image": torch.stack([x["image"] for x in batch]),  # [B,3,H,W]
            "image_id": torch.tensor([x["image_id"] for x in batch]),
            "orig_image_sizes": torch.stack([x["orig_image_size"] for x in batch]),
            "unpadded_image_size": torch.stack(
                [x["unpadded_image_size"] for x in batch]
            ),
            "num_boxes": torch.tensor([x["num_boxes"] for x in batch]),
        }

        # Find max boxes in current batch
        max_boxes = max(x["num_boxes"].item() for x in batch) + 1  # Add 1 for EOS token

        # Pad boxes and labels
        padded_boxes, padded_labels = [], []

        for item in batch:
            num_boxes = item["num_boxes"].item()
            boxes = item["boxes"]  # [N,4]
            labels = item["labels"]  # [N]

            # Random ordering if training
            if self._is_training:
                idx = torch.randperm(num_boxes, device=boxes.device)
                boxes = boxes[idx]
                labels = labels[idx]

            # Always need to pad since we have variable boxes
            if num_boxes < max_boxes:
                pad_boxes = torch.full(
                    (max_boxes - num_boxes, 4),
                    -1,  # Padding value matching TF impl
                    dtype=boxes.dtype,
                    device=boxes.device,
                )
                pad_labels = torch.full(
                    (max_boxes - num_boxes,),
                    -1,  # Padding value matching TF impl
                    dtype=labels.dtype,
                    device=labels.device,
                )

                boxes = torch.cat([boxes, pad_boxes], dim=0)  # [max_boxes,4]
                labels = torch.cat([labels, pad_labels], dim=0)  # [max_boxes]

            padded_boxes.append(boxes)
            padded_labels.append(labels)

        # Stack padded boxes and labels
        batch_data.update(
            {
                "boxes": torch.stack(padded_boxes),  # [B,max_boxes,4]
                "labels": torch.stack(padded_labels),  # [B,max_boxes]
            }
        )

        # Generate sequences using token processor
        input_seq, target_seq, token_weights = self.token_processor.build_sequences(
            boxes=batch_data["boxes"],  # [B,max_boxes,4]
            labels=batch_data["labels"],  # [B,max_boxes]
        )

        # shift target sequence by 1
        target_seq = torch.cat(
            [
                target_seq[:, 1:],
                torch.full_like(target_seq[:, :1], self.token_processor.PADDING_TOKEN),
            ],
            dim=1,
        )

        input_padding_mask = input_seq == self.token_processor.PADDING_TOKEN
        target_padding_mask = target_seq == self.token_processor.PADDING_TOKEN

        batch_data.update(
            {
                "input_seq": input_seq,
                "target_seq": target_seq,
                "token_weights": token_weights,
                "input_padding_mask": input_padding_mask,
                "target_padding_mask": target_padding_mask,
            }
        )

        return batch_data
