from typing import Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch


class ImageAugmentor:
    """Implements Pix2Seq image augmentation pipeline using albumentations."""

    def __init__(
        self,
        image_size: int = 640,
        jitter_scale: Tuple[float, float] = (0.3, 2.0),
        color_jitter_strength: float = 0.4,
        training: bool = True,
        enable_replay: bool = False,  # Add replay flag
    ):
        self.image_size = image_size
        self.training = training
        self.BACKGROUND_VALUE = int(0.3 * 255)
        self.enable_replay = enable_replay
        self.jitter_scale = jitter_scale
        self.color_jitter_strength = color_jitter_strength
        compose_class = A.ReplayCompose if enable_replay else A.Compose

        # Common bbox params
        bbox_params = A.BboxParams(
            format="pascal_voc",
            min_visibility=0.1,
            label_fields=["labels"],
        )

        # Calculate min scale to ensure crop will fit
        scale_limit = (jitter_scale[0] - 1.0, jitter_scale[1] - 1.0)
        if training:
            # Training transforms
            self.transform = compose_class(
                [
                    # Color augmentation
                    A.OneOf(
                        [
                            A.ColorJitter(
                                brightness=color_jitter_strength,
                                contrast=color_jitter_strength,
                                saturation=color_jitter_strength,
                                hue=0.2 * color_jitter_strength,
                                p=0.8,
                            ),
                            A.ToGray(p=0.2),
                        ],
                        p=0.8,
                    ),
                    # Geometric augmentations
                    A.LongestMaxSize(max_size=image_size),
                    A.PadIfNeeded(  # Pad to square
                        min_height=image_size,
                        min_width=image_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=[self.BACKGROUND_VALUE],
                    ),
                    A.RandomScale(scale_limit=scale_limit, p=1.0),
                    A.RandomSizedCrop(
                        min_max_height=(int(image_size * 0.8), image_size),
                        height=image_size,
                        width=image_size,
                        w2h_ratio=1.0,
                        p=1.0,
                    ),
                    A.HorizontalFlip(p=0.5),
                ],
                bbox_params=bbox_params,
            )
        else:
            # Eval transforms
            self.transform = compose_class(
                [
                    A.LongestMaxSize(max_size=image_size),
                    A.PadIfNeeded(
                        min_height=image_size,
                        min_width=image_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=[self.BACKGROUND_VALUE],
                    ),
                ],
                bbox_params=bbox_params,
            )

    def __call__(
        self,
        image: np.ndarray,  # [H,W,3] in uint8 format
        boxes: np.ndarray,  # [N,4] in absolute pixel coordinates
        labels: np.ndarray,  # [N]
        normalize_boxes: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Apply transforms to image and boxes."""
        h, w = image.shape[:2]

        transformed = self.transform(image=image, bboxes=boxes, labels=labels)

        image = transformed["image"]
        boxes = np.array(transformed["bboxes"])
        labels = np.array(transformed["labels"], dtype=labels.dtype)

        # Calculate scale factor for unpadded size
        scale = self.image_size / max(h, w)
        unpadded_size = (int(h * scale), int(w * scale))

        if normalize_boxes and len(boxes) > 0:
            boxes = boxes.astype(np.float32)
            boxes[:, [0, 2]] /= image.shape[1]  # normalize x
            boxes[:, [1, 3]] /= image.shape[0]  # normalize y

        return image, boxes, labels, unpadded_size


class BBoxAugmentation:
    """Implements bounding box augmentation strategies from Pix2Seq paper.

    Creates three types of boxes during training:
    1. Positive examples: Original boxes with small jitter
    2. Hard negatives: Original boxes shifted to wrong locations
    3. Random negatives: Randomly generated boxes

    All negative examples (2,3) are assigned the fake class token.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

        print(
            f"BBoxAugmentation initialized with:"
            f"\n  - num_classes: {num_classes}"
            f"\n  - Will generate fake labels as: {num_classes}"
        )  # Should be 80 for COCO

    def jitter_bbox(
        self,
        bbox: torch.Tensor,
        max_range: float = 0.05,
        truncation: bool = True,
    ) -> torch.Tensor:
        """Applies small jitter to create positive examples."""
        n = len(bbox)
        heights = bbox[:, 2] - bbox[:, 0]
        widths = bbox[:, 3] - bbox[:, 1]
        sizes = torch.stack([heights, widths, heights, widths], -1)

        # Simple truncated normal distribution for jitter
        noise_rate = torch.randn(n, 4) * (max_range / 2.0)
        noise_rate = torch.clamp(noise_rate, -max_range, max_range)

        bbox = bbox + sizes * noise_rate
        if truncation:
            bbox = torch.clamp(bbox, 0.0, 1.0)
        return bbox

    def shift_bbox(self, bbox: torch.Tensor, truncation: bool = True) -> torch.Tensor:
        """Creates hard negative examples by shifting boxes to wrong locations.

        Maintains original box dimensions to create plausible but incorrect boxes.
        """
        n = len(bbox)
        heights = bbox[:, 2] - bbox[:, 0]
        widths = bbox[:, 3] - bbox[:, 1]

        # Random new centers
        cy = torch.rand(n, 1)
        cx = torch.rand(n, 1)

        shifted = torch.cat(
            [
                cx - widths.unsqueeze(1) / 2,  # xmin
                cy - heights.unsqueeze(1) / 2,  # ymin
                cx + widths.unsqueeze(1) / 2,  # xmax
                cy + heights.unsqueeze(1) / 2,  # ymax
            ],
            -1,
        )

        if truncation:
            shifted = torch.clamp(shifted, 0.0, 1.0)
        return shifted

    def random_bbox(
        self,
        n: int,
        max_size: float = 1.0,
        truncation: bool = True,
        return_labels: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Creates random negative examples."""
        # Random centers
        cy = torch.rand(n, 1)
        cx = torch.rand(n, 1)

        # Random dimensions
        h = torch.randn(n, 1) * max_size / 2
        w = torch.randn(n, 1) * max_size / 2

        bbox = torch.cat(
            [
                cx - torch.abs(w) / 2,
                cy - torch.abs(h) / 2,
                cx + torch.abs(w) / 2,
                cy + torch.abs(h) / 2,
            ],
            -1,
        )

        if truncation:
            bbox = torch.clamp(bbox, 0.0, 1.0)

        if return_labels:
            fake_labels = torch.full((n,), self.num_classes, dtype=torch.long)
            return bbox, fake_labels

        return bbox

    def augment_bbox(
        self,
        bbox: torch.Tensor,  # [N,4] normalized XYXY
        bbox_label: torch.Tensor,  # [N] class indices
        max_jitter: float = 0.05,
        n_noise_bbox: int = 0,
        mix_rate: float = 0.5,  # Added mix_rate parameter
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment bbox with noise following TF implementation.

        Args:
            bbox: Original boxes in XYXY format
            bbox_label: Original class labels
            max_jitter: Maximum jitter for positive examples
            n_noise_bbox: Number of noise boxes to add
            mix_rate: Probability of mixing noise boxes with real boxes
        """
        n = len(bbox)

        # Ensure valid n_noise_bbox
        n_noise_bbox = max(0, n_noise_bbox)  # Prevent negative

        if n > 0:
            # Small jitter for real boxes but keep original labels
            bbox = self.jitter_bbox(bbox, max_range=max_jitter)

            if n_noise_bbox > 0:
                dup_bbox_size = torch.randint(0, n_noise_bbox + 1, (1,)).item()
            else:
                dup_bbox_size = 0
        else:
            dup_bbox_size = n_noise_bbox

        bad_bbox_size = n_noise_bbox - dup_bbox_size

        # Create bad boxes
        if bad_bbox_size > 0:
            if n > 0:
                # Generate both shifted and random boxes
                indices = torch.randint(0, n, (bad_bbox_size,))
                bad_bbox_shift = self.shift_bbox(bbox[indices])
                bad_bbox_random = self.random_bbox(bad_bbox_size)

                # Randomly mix shifted and random boxes
                mix_mask = (torch.rand(bad_bbox_size, 1) < 0.5).float()
                bad_bbox = mix_mask * bad_bbox_shift + (1 - mix_mask) * bad_bbox_random
            else:
                # If no real boxes, just use random
                bad_bbox = self.random_bbox(bad_bbox_size)

            # Assign fake class labels
            bad_label = torch.full(
                (bad_bbox_size,),
                self.num_classes,  # This is FAKE_CLASS_TOKEN - BASE_VOCAB_SHIFT
                dtype=bbox_label.dtype,
                device=bbox.device,
            )
        else:
            bad_bbox = bbox.new_zeros((0, 4))
            bad_label = bbox_label.new_zeros(0)

        # Create duplicate boxes if we have real boxes
        if dup_bbox_size > 0 and n > 0:
            dup_indices = torch.randperm(n)[:dup_bbox_size]
            dup_bbox = self.shift_bbox(bbox[dup_indices])
            dup_label = torch.full_like(bbox_label[dup_indices], self.num_classes)
        else:
            dup_bbox = bbox.new_zeros((0, 4))
            dup_label = bbox_label.new_zeros(0)

        # Combine positive and negative examples
        if torch.rand(1) < mix_rate and n > 0:
            # Mix and shuffle everything together
            noise_bbox = torch.cat([bad_bbox, dup_bbox])
            noise_label = torch.cat([bad_label, dup_label])

            bbox_new = torch.cat([bbox, noise_bbox])
            label_new = torch.cat([bbox_label, noise_label])

            # Random shuffle to mix real and noise boxes
            perm = torch.randperm(len(bbox_new))
            bbox_new = bbox_new[perm]
            label_new = label_new[perm]
        else:
            # Simply append noise boxes
            noise_bbox = torch.cat([bad_bbox, dup_bbox])
            noise_label = torch.cat([bad_label, dup_label])

            bbox_new = torch.cat([bbox, noise_bbox]) if n > 0 else noise_bbox
            label_new = torch.cat([bbox_label, noise_label]) if n > 0 else noise_label

        return bbox_new, label_new

    def _validate_outputs(self, bbox_new, label_new, n_orig, n_noise_bbox):
        """Add validation to check box augmentation is working correctly."""
        n_total = len(bbox_new)
        n_fake = (label_new == self.num_classes).sum().item()
        n_real = n_total - n_fake

        print("\nBox Augmentation Stats:")
        print(f"Original boxes: {n_orig}")
        print(f"Total boxes: {n_total}")
        print(f"Real boxes: {n_real}")
        print(f"Fake boxes: {n_fake}")
        print(f"Target noise boxes: {n_noise_bbox}")

        # Validate box coordinates
        if len(bbox_new) > 0:
            print("\nBox Coordinates Stats:")
            print(f"Min coords: {bbox_new.min().item():.3f}")
            print(f"Max coords: {bbox_new.max().item():.3f}")
            print(f"Mean coords: {bbox_new.mean().item():.3f}")
