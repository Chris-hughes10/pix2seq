"""PyTorch dataset implementation for Pix2Seq object detection.

This module provides a modular implementation of the Pix2Seq data processing pipeline,
with support for different base datasets and customizable augmentation strategies.

Key components:
- Base dataset interface for different detection datasets
- Image preprocessing and normalization
- Image and box augmentation pipelines
- Token sequence generation for training
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torchvision.datasets.coco import CocoDetection

# fmt: off
COCO80_TO_COCO91_MAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                        85, 86, 87, 88, 89, 90]


# fmt: on


def coco80_to_coco91_lookup():
    return {i: v for i, v in enumerate(COCO80_TO_COCO91_MAP)}


def coco91_to_coco80_lookup():
    return {v: i for i, v in enumerate(COCO80_TO_COCO91_MAP)}


class COCOBaseDataset(CocoDetection):
    def __init__(
        self,
        img_dir,
        annotation_path,
        filter_empty: bool = False,
        filter_crowd: bool = True,
        min_area: Optional[float] = None,
        tfms=None,
    ):
        """Initialize COCO dataset with filtering options.

        Args:
            img_dir: Root directory containing images
            annotation_path: Path to COCO annotation file
            filter_empty: Whether to filter images without valid annotations
            filter_crowd: Whether to filter crowd annotations
            min_area: Minimum allowed box area in pixels
            tfms: Optional transform pipeline
        """
        super().__init__(root=str(img_dir), annFile=str(annotation_path))
        self.coco91_to_coco80 = coco91_to_coco80_lookup()
        self.valid_category_ids = set(self.coco91_to_coco80.keys())

        self.targets_json = self.coco.dataset
        self.filter_crowd = filter_crowd
        self.min_area = min_area
        self.tfms = tfms

        # Filter image IDs based on criteria
        self.ids = self._filter_image_ids(filter_empty)

    def _filter_image_ids(self, filter_empty: bool) -> List[int]:
        """Filter image IDs based on annotation criteria."""
        ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # Filter annotations
            valid_anns = []
            for ann in anns:
                # Skip crowd annotations if specified
                if self.filter_crowd and ann.get("iscrowd", 0):
                    continue

                # Validate box
                if not self._is_valid_box(ann.get("bbox", []), ann.get("area", 0)):
                    continue

                valid_anns.append(ann)

            # Skip image if no valid annotations and filtering is enabled
            if filter_empty and not valid_anns:
                continue

            ids.append(img_id)

        return ids

    def _is_valid_box(self, bbox: List[float], area: float) -> bool:
        """Check if a bounding box is valid."""
        if not bbox or len(bbox) != 4:
            return False

        # Check if box has valid coordinates
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False

        # Check minimum area if specified
        if self.min_area and area < self.min_area:
            return False

        return True

    def _filter_annotations(self, targets: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Filter and convert annotations to box and class arrays."""
        raw_boxes, class_ids = [], []

        for target in targets:
            # Skip crowd annotations if specified
            if self.filter_crowd and target.get("iscrowd", 0):
                continue

            # Validate box
            if not self._is_valid_box(target.get("bbox", []), target.get("area", 0)):
                continue

            raw_boxes.append(target["bbox"])
            class_ids.append(self.coco91_to_coco80[target["category_id"]])

        if len(raw_boxes) == 0:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])

        else:
            xyxy_bboxes = torchvision.ops.box_convert(
                torch.as_tensor(raw_boxes), "xywh", "xyxy"
            ).numpy()
            class_ids = np.array(class_ids, dtype=np.int64)

        return xyxy_bboxes, class_ids

    def __getitem__(self, index):
        """Get an item with filtered and validated boxes.

        Returns:
            Tuple containing:
            - image: np.ndarray [H,W,3] uint8 RGB image
            - boxes: np.ndarray [N,4] float32 boxes in xyxy format
            - class_ids: np.ndarray [N] int64 class indices (COCO-80)
            - image_id: int COCO image ID
            - image_hw: Tuple[int,int] original image (height, width)
        """
        image_id = self.ids[index]

        image, targets = super().__getitem__(index)
        image = np.array(image)
        image_hw = image.shape[:2]

        xyxy_bboxes, class_ids = self._filter_annotations(targets)

        if self.tfms is not None:
            transformed = self.tfms(image=image, bboxes=xyxy_bboxes, labels=class_ids)
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id, image_hw

    def get_categories(self) -> dict:
        """Get category names from COCO dataset with COCO-80 mapping."""
        categories = {}

        # Get COCO-91 categories
        cats = self.coco.cats  # Use the cats dict directly instead of loadCats()

        for cat_id, cat in cats.items():
            # Only include categories that map to COCO-80
            if cat_id in self.coco91_to_coco80:
                mapped_id = self.coco91_to_coco80[cat_id]  # Convert to COCO-80 index
                cat["id"] = mapped_id  # Update the ID in the category info
                categories[mapped_id] = cat  # Store with COCO-80 index as key

        return categories
