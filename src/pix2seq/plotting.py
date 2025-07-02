from typing import Dict, Optional, Union

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_VOCAB_SHIFT = 10


def show_image_with_boxes(
    image: Union[np.ndarray, torch.Tensor],
    boxes: Union[np.ndarray, torch.Tensor] = None,
    labels: Union[np.ndarray, torch.Tensor] = None,
    title: Optional[str] = None,
    category_names: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    normalized_boxes: bool = False,
    box_color: Optional[str] = None,
    label_prefix: Optional[str] = None,
    real_noise_coloring: bool = False,
) -> plt.Axes:
    """Display an image with its bounding boxes."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 12))

    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    image = to_numpy(image)
    boxes = to_numpy(boxes)
    labels = to_numpy(labels)

    if image.shape[0] == 3:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    ax.imshow(image)

    # Define color mapping - use RGBA arrays directly
    num_classes = 80  # COCO has 80 classes
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))  # Returns RGBA arrays

    if boxes is not None and len(boxes) > 0:
        height, width = image.shape[:2]
        boxes = boxes.copy()

        if normalized_boxes:
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height

        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            if xmax <= xmin or ymax <= ymin:
                continue

            if box_color is not None:
                color = box_color
            elif real_noise_coloring and labels is not None:
                label_id = int(labels[i])
                is_real = label_id != num_classes
                # is_real = (label_id - BASE_VOCAB_SHIFT) < num_classes  # Adjust comparison
                color = "green" if is_real else "red"
            elif labels is not None:
                # Use RGBA array directly
                color = colors[int(labels[i]) % num_classes]
            else:
                color = "red"

            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                linewidth=2,
                edgecolor=color,
            )
            ax.add_patch(rect)

            if labels is not None:
                label_id = int(labels[i])
                if real_noise_coloring:
                    label_text = "Real" if label_id < num_classes else "Fake"
                elif category_names and label_id in category_names:
                    label_text = category_names[label_id]["name"]
                else:
                    label_text = f"Class {label_id}"

                if label_prefix:
                    label_text = f"{label_prefix} {label_text}"
            else:
                label_text = f"{label_prefix or ''} Box {i}"

            ax.text(
                xmin,
                ymin - 2,
                label_text.strip(),
                color=color,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")
    return ax


def visualize_augmentations(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    augmentor,
    category_names: dict,
    num_examples: int = 4,
):
    """Visualize multiple augmentations of the same image."""

    # Create subplot grid - make it taller to accommodate text
    n_cols = min(4, num_examples + 1)
    n_rows = (num_examples + 1 + n_cols - 1) // n_cols  # +1 for original image
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows * 2))
    gs = fig.add_gridspec(n_rows * 2, n_cols)

    def flatten_applied_transforms(applied_transforms):
        """Flatten the list of applied transforms, including nested transforms."""
        flat_list = []
        for transform in applied_transforms:
            flat_list.append(transform)
            if "transforms" in transform:
                flat_list.extend(flatten_applied_transforms(transform["transforms"]))
        return flat_list

    def get_transform_info(transform, applied_transforms=None):
        """Recursively get transform info, handling OneOf and other composite transforms."""
        if isinstance(transform, (A.OneOf, A.Compose, A.SomeOf, A.OneOrOther)):
            children = []
            for t in transform.transforms:
                children.append(get_transform_info(t, applied_transforms))

            tfm_string = "".join(children)
            return tfm_string
        else:
            try:
                params = transform.get_transform_init_args_names()
                params_dict = {
                    k: getattr(transform, k) for k in params if hasattr(transform, k)
                }
                param_str = "\n  ".join([f"{k}: {v}" for k, v in params_dict.items()])
                was_applied = applied_transforms and any(
                    at["__class_fullname__"].endswith(transform.__class__.__name__)
                    for at in applied_transforms
                    if at.get("applied")
                )
                tfm_string = (
                    f"\n  {transform.__class__.__name__}(p={transform.p}):\n [{param_str}]"
                    if was_applied
                    else ""
                )
                return tfm_string
            except:
                return f"{transform.__class__.__name__}"

    # Show original image
    ax_img = fig.add_subplot(gs[0, 0])
    show_image_with_boxes(
        image, boxes, labels, "Original Image", category_names=category_names, ax=ax_img
    )
    ax_txt = fig.add_subplot(gs[1, 0])
    ax_txt.axis("off")

    # Show augmented versions
    for idx in range(num_examples):
        col = (idx + 1) % n_cols
        row = ((idx + 1) // n_cols) * 2

        # Apply augmentation with transform tracking
        transformed = augmentor.transform(
            image=image.copy(), bboxes=boxes.tolist(), labels=labels.tolist()
        )

        aug_image = transformed["image"]
        aug_boxes = np.array(transformed["bboxes"])
        aug_labels = np.array(transformed["labels"], dtype=labels.dtype)

        # Get transform details including which ones were applied
        transforms_info = []
        if isinstance(augmentor.transform, A.ReplayCompose):
            applied_transforms = transformed.get("replay", {}).get("transforms", [])
            applied_transforms = flatten_applied_transforms(applied_transforms)
            for transform in augmentor.transform.transforms:
                transforms_info.append(
                    get_transform_info(transform, applied_transforms)
                )

        # Plot augmented image
        ax_img = fig.add_subplot(gs[row, col])
        show_image_with_boxes(
            aug_image,
            aug_boxes,
            aug_labels,
            f"Augmented {idx + 1}",
            category_names=category_names,
            ax=ax_img,
            normalized_boxes=False,
        )

        # Add transform info in text box below
        ax_txt = fig.add_subplot(gs[row + 1, col])
        ax_txt.axis("off")
        # n_orig = len(boxes)
        # n_aug = len(aug_boxes)
        # info_text = f"Boxes: {n_orig}->{n_aug}\n\n"
        info_text = ""
        if transforms_info:
            info_text += "\n".join(transforms_info)
        else:
            info_text += "No transforms applied"
        ax_txt.text(0.5, 0.5, info_text, va="center", ha="center", wrap=True)

    plt.suptitle("Pix2Seq Augmentation Pipeline", size=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
