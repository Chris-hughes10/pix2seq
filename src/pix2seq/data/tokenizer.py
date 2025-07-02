"""PyTorch dataset implementation for Pix2Seq object detection.

This module provides a modular implementation of the Pix2Seq data processing pipeline,
with support for different base datasets and customizable augmentation strategies.

Key components:
- Base dataset interface for different detection datasets
- Image preprocessing and normalization
- Image and box augmentation pipelines
- Token sequence generation for training
"""

from enum import Enum
from typing import Iterable, List, Optional, Tuple

import torch


class LabelCorruptionStrategy(Enum):
    NONE = "none"  # Keep all original labels
    RANDOM = "rand_cls"  # 50% original, 50% random valid classes
    RANDOM_AND_FAKE = "rand_n_fake_cls"  # 50% original, 25% random, 25% fake


class TokenProcessor:
    """Converts bounding boxes and class labels into token sequences for Pix2Seq.

    Sequence format:
    [y1 x1 y2 x2 c1] [y1 x1 y2 x2 c2] ... [EOS] [PAD] [PAD]
    Note: Detection task token is handled separately by the model.

    Args:
        coord_vocab_shift: Starting index for coordinate tokens (e.g. 1000)
        quantization_bins: Number of bins for coordinate quantization
        noise_bbox_weight: Weight for noise box tokens in training
        eos_token_weight: Weight for end-of-sequence token
        max_seq_len: Maximum allowed sequence length
        num_classes: Number of classes in dataset
        random_order: Whether to randomly order objects in sequence
    """

    def __init__(
        self,
        quantization_bins: int,
        noise_bbox_weight: float,
        eos_token_weight: float,
        max_seq_len: int,
        num_classes: int,
        corrupt_class_labels: bool = False,
        num_special_tokens=10,
        corruption_strategy: LabelCorruptionStrategy = LabelCorruptionStrategy.NONE,
        verbose=True
    ):
        self.quantization_bins = quantization_bins
        self.noise_bbox_weight = noise_bbox_weight
        self.bos_eos_token_weight = eos_token_weight
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self._corrupt_class_labels = corrupt_class_labels
        self._corruption_strategy = corruption_strategy

        # Special tokens
        self.PADDING_TOKEN = 0
        self.BOS_TOKEN = 1
        self.EOS_TOKEN = 2
        self.BASE_VOCAB_SHIFT = num_special_tokens  # Start of class tokens

        # FAKE_CLASS_TOKEN comes immediately after real classes
        self.FAKE_CLASS_TOKEN = self.BASE_VOCAB_SHIFT + self.num_classes
        self.coord_vocab_shift = self.FAKE_CLASS_TOKEN + 1

        # Calculate vocabulary ranges
        self.max_coord_token = self.coord_vocab_shift + self.quantization_bins - 1

        self._validate_init_params()
        if verbose:
            self.log_token_ranges()

    def _validate_init_params(self):
        """Validate initialization parameters and vocab space."""
        total_vocab_size = self.vocab_size
        min_vocab_size = (
            self.FAKE_CLASS_TOKEN + 1
        )  # Must have room for at least 1 coordinate

        if total_vocab_size < min_vocab_size:
            raise ValueError(
                f"Total vocab size ({total_vocab_size}) must be > "
                f"minimum required size ({min_vocab_size})"
            )

        if self.num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        if self.quantization_bins < 100:
            raise ValueError(
                "quantization_bins should be at least 100 for reasonable precision"
            )

        # Validate sequence length (5 tokens per box + EOS)
        min_seq_len = 6  # One box (5) + EOS (1)
        if self.max_seq_len < min_seq_len:
            raise ValueError(
                f"max_seq_len must be at least {min_seq_len}, got {self.max_seq_len}"
            )

    def log_token_ranges(self):
        """Log token range information for debugging."""
        print("\nToken Processor initialized with:")
        print(f"  Special tokens: 0-{self.BASE_VOCAB_SHIFT - 1}")
        print(f"    Padding token: {self.PADDING_TOKEN}")
        print(f"    BOS token: {self.BOS_TOKEN}")
        print(f"    EOS token: {self.EOS_TOKEN}")
        print(f"  Base vocab shift: {self.BASE_VOCAB_SHIFT}")
        print(
            f"  Class tokens: {self.BASE_VOCAB_SHIFT}-{self.BASE_VOCAB_SHIFT + self.num_classes - 1}"
        )
        print(f"  FAKE_CLASS_TOKEN: {self.FAKE_CLASS_TOKEN}")
        print(f"  Coordinate tokens: {self.coord_vocab_shift}-{self.max_coord_token}")
        print(f"  Total vocab size: {self.vocab_size}\n")

    def quantize(self, boxes: torch.Tensor) -> torch.Tensor:
        """Quantize normalized box coordinates to integer tokens."""

        # Scale coordinates to quantization range [0, bins-1]
        # and round to nearest integer (the actual quantization step)
        boxes = torch.round(boxes * (self.quantization_bins - 1))

        # Clamp to valid range (handle any edge cases)
        boxes = torch.clamp(boxes, 0, self.quantization_bins - 1)

        # Shift to coordinate vocabulary range
        boxes = boxes + self.coord_vocab_shift
        return boxes.long()

    def dequantize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert coordinate tokens back to normalized coordinates."""

        # Remove coordinate vocabulary shift
        tokens = tokens - self.coord_vocab_shift
        # Scale back to [0, 1] range
        tokens = torch.clamp(tokens, 0, self.quantization_bins - 1)
        # Convert back to [0,1] normalized coordinates
        return tokens.float() / (self.quantization_bins - 1)

    def corrupt_class_labels(
        self, labels: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Corrupt class labels according to specified strategy.

        For all strategies, we first decide whether to keep original labels (50% probability).
        Then for labels we'll corrupt, we apply the strategy's noise type:
        - NONE: Keep all labels unchanged
        - RANDOM: Replace with random valid classes
        - RANDOM_AND_FAKE: Equal split between random classes and fake token

        Args:
            labels: Class labels [B,N]
            padding_mask: Boolean mask where True indicates padding [B,N]
        """
        # For 'none' strategy or if corruption is disabled, return original labels
        if (
            self._corruption_strategy == LabelCorruptionStrategy.NONE
            or not self._corrupt_class_labels
        ):
            return labels

        batch_size, num_labels = labels.shape
        valid_tokens = ~padding_mask

        # First decide which valid tokens to keep (50% probability)
        keep_mask = (
            torch.rand(batch_size, num_labels, device=labels.device) < 0.5
        ) & valid_tokens

        # Start with original labels
        corrupted = labels.clone()

        # Create random class labels for corruption
        rand_cls = torch.randint(
            self.BASE_VOCAB_SHIFT, self.BASE_VOCAB_SHIFT + self.num_classes, (batch_size, num_labels), device=labels.device
        )

        if self._corruption_strategy == LabelCorruptionStrategy.RANDOM:
            # For tokens we're not keeping, replace with random classes
            corrupted = torch.where(valid_tokens & ~keep_mask, rand_cls, corrupted)

        elif self._corruption_strategy == LabelCorruptionStrategy.RANDOM_AND_FAKE:
            # For tokens we're not keeping, decide between random and fake
            noise_mask = torch.rand(batch_size, num_labels, device=labels.device) < 0.5
            tokens_to_corrupt = valid_tokens & ~keep_mask

            # Apply random classes where noise_mask is True
            corrupted = torch.where(tokens_to_corrupt & noise_mask, rand_cls, corrupted)

            # Apply fake token where noise_mask is False
            fake_cls = torch.full_like(
                labels, self.FAKE_CLASS_TOKEN, device=labels.device
            )
            corrupted = torch.where(
                tokens_to_corrupt & ~noise_mask, fake_cls, corrupted
            )

        return corrupted

    def build_sequences(
        self, boxes: Iterable[torch.Tensor], labels: Iterable[torch.Tensor]
    ):
        """Build token sequences for training."""

        batch_size, num_boxes = boxes.shape[:2]

        # Convert XYXY to YXYX format
        boxes = boxes[..., [1, 0, 3, 2]]

        # Validate coordinates are properly ordered
        ymin, xmin, ymax, xmax = (
            boxes[..., 0],
            boxes[..., 1],
            boxes[..., 2],
            boxes[..., 3],
        )
        invalid_ymax = ymax <= ymin
        invalid_xmax = xmax <= xmin

        if invalid_ymax.any() or invalid_xmax.any():
            # Correction factor
            correction_factor = (
                0.01  # Small percentage to increase by, adjust as needed
            )

            # Apply correction where necessary
            ymax += invalid_ymax * (ymin - ymax + correction_factor)
            xmax += invalid_xmax * (xmin - xmax + correction_factor)

            # Update boxes with corrected values
            boxes = torch.stack([ymin, xmin, ymax, xmax], dim=-1)

        # Convert -1 padding to PADDING_TOKEN (0)
        is_padding = labels == -1

        # Quantize coordinates and handle padding
        boxes = self.quantize(boxes)  # [B,N,4]
        boxes = torch.where(
            is_padding.unsqueeze(-1), torch.full_like(boxes, self.PADDING_TOKEN), boxes
        )

        # Process labels
        target_labels = labels + self.BASE_VOCAB_SHIFT
        target_labels = torch.where(
            is_padding,
            torch.full_like(target_labels, self.PADDING_TOKEN),
            target_labels,
        )

        if self._corrupt_class_labels:
            input_labels = self.corrupt_class_labels(
                labels + self.BASE_VOCAB_SHIFT, is_padding
            )
            input_labels = torch.where(
                is_padding,
                torch.full_like(input_labels, self.PADDING_TOKEN),
                input_labels,
            )
        else:
            input_labels = target_labels

        # Create object sequences first - no BOS/EOS yet
        target_seq = torch.cat([boxes, target_labels.unsqueeze(-1)], dim=-1)  # [B,N,5]
        input_seq = torch.cat([boxes, input_labels.unsqueeze(-1)], dim=-1)  # [B,N,5]

        # Calculate token weights for objects
        # Check which objects are fake/noise objects by comparing their labels to the fake class token
        is_fake = target_labels == self.FAKE_CLASS_TOKEN

        # Calculate weights for bounding box coordinate tokens (4 tokens per object: y_min, x_min, y_max, x_max)
        # fake objects should learn to predict "fake" class but not learn coordinates
        bbox_weights = torch.where(
            is_padding.unsqueeze(-1), # Expand padding mask to match box dimensions [B,N,1] -> [B,N,4]
            torch.zeros_like(boxes, dtype=torch.float32), # Padding tokens get weight 0 (ignored in loss)
            torch.where(
                is_fake.unsqueeze(-1), # Expand fake mask to match box dimensions [B,N,1] -> [B,N,4]
                torch.full_like(boxes, self.noise_bbox_weight, dtype=torch.float32), # Fake objects get specified weights for coordinates
                torch.ones_like(boxes, dtype=torch.float32), torch.ones_like(boxes, dtype=torch.float32),  # Real objects get weight 1.0 (fully learn coordinates)
            ),
        )

        # Calculate weights for class tokens (1 token per object)
        # fake objects still get weight 1.0 for class tokens so model learns to predict "fake"
        label_weights = torch.where(
            is_padding,  # Check if this is a padding token
            torch.zeros_like(labels, dtype=torch.float32), # Padding tokens get weight 0 (ignored)
            torch.ones_like(
                labels, dtype=torch.float32
            ),  # Both real AND fake objects get weight 1.0 for class prediction
        )

        # Combine coordinate and class weights into a single weight tensor
        # Each object has 5 tokens: [y_min, x_min, y_max, x_max, class]
        token_weights = torch.cat(
            [bbox_weights, label_weights.unsqueeze(-1)], dim=-1
        )  # [B, N, 5] where 5 = 4 coordinate tokens + 1 class token

        # Create BOS token
        bos_token = torch.full(
            (batch_size, 1), self.BOS_TOKEN, device=boxes.device
        ).long()
        bos_weight = torch.full(
            (batch_size, 1),
            self.bos_eos_token_weight,
            device=boxes.device,
            dtype=torch.float32,
        )

        # Flatten sequences before adding BOS/EOS
        target_seq = target_seq.reshape(batch_size, -1)  # [B,N*5]
        input_seq = input_seq.reshape(batch_size, -1)  # [B,N*5]
        token_weights = token_weights.reshape(batch_size, -1)  # [B,N*5]

        # Add EOS token
        first_non_padding_idx = (target_seq == self.PADDING_TOKEN).float().argmax(dim=1)
        target_seq[torch.arange(target_seq.size(0)), first_non_padding_idx] = (
            self.EOS_TOKEN
        )
        input_seq[torch.arange(target_seq.size(0)), first_non_padding_idx] = (
            self.EOS_TOKEN
        )
        token_weights[torch.arange(target_seq.size(0)), first_non_padding_idx] = (
            self.bos_eos_token_weight
        )

        # Add BOS tokens
        target_seq = torch.cat([bos_token, target_seq], dim=1)
        input_seq = torch.cat([bos_token, input_seq], dim=1)
        token_weights = torch.cat([bos_weight, token_weights], dim=1)

        return input_seq, target_seq, token_weights

    def decode_tokens(
        self,
        tokens: torch.Tensor,  # [B,S]
        token_scores: Optional[torch.Tensor] = None,  # [B,S]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Decode token sequences back to boxes and class labels.

        Args:
            tokens: Token sequences [B,S]
            token_scores: Optional logits for class tokens [B,num_objects,V]

        Returns:
            Tuple containing:
            - List of box tensors [N,4] for each batch item
            - List of label tensors [N] for each batch item
            - List of score tensors [N] for each batch item (None if no scores provided)
        """
        batch_size = len(tokens)
        # Remove BOS token if present
        if (tokens[:, 0] == self.BOS_TOKEN).all():
            tokens = tokens[:, 1:]
        else:
            raise ValueError("BOS token not found in sequence")

        # Find first padding or EOS token for each sequence
        is_end = (tokens == self.PADDING_TOKEN) | (tokens == self.EOS_TOKEN)
        end_indices = []

        for b in range(batch_size):
            # Find first padding/EOS token
            try:
                end_idx = is_end[b].nonzero()[0].item()
                # Ensure we have complete 5-token groups
                end_idx = (end_idx // 5) * 5
            except IndexError:
                # No padding/EOS found, use all tokens
                end_idx = tokens.size(1)
                # Ensure multiple of 5
                end_idx = (end_idx // 5) * 5
            end_indices.append(end_idx)

        max_len = max(end_indices)
        # Apply mask and reshape maintaining batch dimension
        tokens = tokens[:, :max_len]  # [B, max_len]
        tokens = tokens.reshape(batch_size, -1, 5)  # [B, N, 5]

        # Split coordinates and class labels
        coord_tokens = tokens[..., :4]  # [B,N,4]
        class_tokens = tokens[..., 4]  # [B,N]

        # Convert back to normalized coordinates
        boxes = self.dequantize(coord_tokens)  # [B,N,4]
        # Convert YXYX back to XYXY format
        boxes = boxes[..., [1, 0, 3, 2]]  # [B,N,4]

        # Convert back to class indices
        labels = class_tokens - self.BASE_VOCAB_SHIFT  # [B,N]

        # Process scores if provided
        scores = None
        if token_scores is not None:
            # token_scores shape should be [B,num_objects,V] - only for class tokens
            num_objects = token_scores.size(1)
            num_valid_objects = min(num_objects, class_tokens.size(1))

            # Only use scores for valid objects
            valid_class_tokens = class_tokens[:, :num_valid_objects]  # [B,num_valid]
            valid_scores = token_scores[:, :num_valid_objects]  # [B,num_valid,V]

            # Convert logits to probabilities and get score for predicted class
            class_probs = torch.softmax(valid_scores, dim=-1)  # [B,num_valid,V]
            scores = torch.gather(
                class_probs,
                dim=-1,
                index=valid_class_tokens.unsqueeze(-1),  # [B,num_valid,1]
            ).squeeze(-1)  # [B,num_valid]

            # Pad scores if needed to match number of objects
            if scores.size(1) < boxes.size(1):
                padding = torch.zeros(
                    (batch_size, boxes.size(1) - scores.size(1)),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                scores = torch.cat([scores, padding], dim=1)  # [B,N]

        # Convert to list format
        boxes_list = [boxes[i] for i in range(batch_size)]
        labels_list = [labels[i] for i in range(batch_size)]
        scores_list = [
            scores[i] if scores is not None else None for i in range(batch_size)
        ]

        return boxes_list, labels_list, scores_list

    @property
    def vocab_size(self) -> int:
        return self.coord_vocab_shift + self.quantization_bins

    def max_coord_token(self) -> int:
        return self.coord_vocab_shift + self.quantization_bins - 1

    def post_process_sequences(
        self,
        sequences: torch.Tensor,
        class_logits: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.05,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Post-process and filter invalid boxes from decoded sequences.

        This method takes the output from decode_tokens and filters out invalid boxes
        while maintaining batch efficiency. It works with the existing decode_tokens
        implementation rather than modifying it.

        Args:
            sequences: Generated token sequences [B,S]
            class_logits: Optional logits for class tokens [B,N,V]
            confidence_threshold: Minimum confidence score to keep predictions

        Returns:
            boxes_list: List of valid boxes for each sequence
            labels_list: List of labels for valid boxes
            scores_list: List of confidence scores for valid boxes
        """
        # Use existing decode_tokens to get initial results
        boxes_list, labels_list, scores_list = self.decode_tokens(
            sequences, class_logits
        )

        # Process each sequence's results efficiently
        filtered_boxes, filtered_labels, filtered_scores = [], [], []

        for boxes, labels, scores in zip(boxes_list, labels_list, scores_list):
            # Skip empty results
            if len(boxes) == 0:
                filtered_boxes.append(boxes)
                filtered_labels.append(labels)
                filtered_scores.append(scores if scores is not None else None)
                continue

            # Create validity mask for coordinate constraints
            valid_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

            # Check ymax > ymin and xmax > xmin
            valid_mask &= boxes[:, 2] > boxes[:, 0]  # ymax > ymin
            valid_mask &= boxes[:, 3] > boxes[:, 1]  # xmax > xmin

            # Check coordinates are in [0,1]
            valid_mask &= torch.all((boxes >= 0) & (boxes <= 1), dim=1)

            # Apply confidence threshold if scores available
            if scores is not None:
                valid_mask &= scores > confidence_threshold
                filtered_scores.append(scores[valid_mask])
            else:
                filtered_scores.append(None)

            # Keep valid boxes and labels
            filtered_boxes.append(boxes[valid_mask])
            filtered_labels.append(labels[valid_mask])

        return filtered_boxes, filtered_labels, filtered_scores
