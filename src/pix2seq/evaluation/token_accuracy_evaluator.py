from typing import Dict, Optional, Tuple

import torch


class TokenAccuracyEvaluator:
    """Core logic for computing token accuracy metrics."""

    def __init__(self, pad_token_id: int = 0, eos_token_id: int = 2):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all accumulated metrics tracking state.

        Clears the accumulated_metrics dictionary that stores running metrics for epoch-level calculations.
        Should be called at the start of each epoch.
        """
        self.accumulated_metrics = {}

    def _get_correct_and_total(
        self, pred_tokens: torch.Tensor, target_seq: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate number of correct predictions and total predictions for masked tokens.

        Args:
            pred_tokens: Predicted token indices [N]
            target_seq: Target token indices [N]
            mask: Boolean mask indicating which positions to evaluate [N]

        Returns:
            Tuple containing:
            - Sum of correct predictions for masked positions
            - Total number of masked positions
        """
        correct = (pred_tokens == target_seq) & mask
        total = mask
        return correct.sum(), total.sum()

    def compute_position_accuracies(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        non_pad_mask: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute accuracy metrics for each position in the 5-token object description pattern.

        Calculates separate accuracies for ymin, xmin, ymax, xmax, and class positions.

        Args:
            pred_tokens: Predicted token indices [N]
            target_seq: Target token indices [N]
            non_pad_mask: Boolean mask for non-padding tokens [N]

        Returns:
            Dictionary mapping position names to tuples of (correct_count, total_count)
        """
        metrics = {}
        for pos, name in enumerate(["ymin", "xmin", "ymax", "xmax", "class"]):
            pos_mask = (
                torch.arange(len(target_seq), device=target_seq.device) % 5 == pos
            ) & non_pad_mask
            correct, total = self._get_correct_and_total(
                pred_tokens, target_seq, pos_mask
            )
            metrics[f"pos_{name}"] = (correct, total)
        return metrics

    def compute_first_token_accuracy(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        non_pad_mask: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute accuracy for the first token of each sequence.

        Args:
            pred_tokens: Predicted token indices [N]
            target_seq: Target token indices [N]
            non_pad_mask: Boolean mask for non-padding tokens [N]

        Returns:
            Dictionary with 'first_token' key mapping to tuple of (correct_count, total_count)
        """
        first_token_mask = non_pad_mask.clone()
        first_token_mask[1:] = False  # Only keep first position
        correct, total = self._get_correct_and_total(
            pred_tokens, target_seq, first_token_mask
        )
        return {"first_token": (correct, total)}

    def compute_type_accuracies(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        non_pad_mask: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute accuracy metrics grouped by token type (coordinate vs class tokens).

        Args:
            pred_tokens: Predicted token indices [N]
            target_seq: Target token indices [N]
            non_pad_mask: Boolean mask for non-padding tokens [N]

        Returns:
            Dictionary with keys 'coordinate' and 'class', mapping to tuples of (correct_count, total_count)
        """
        metrics = {}

        # Coordinate tokens (first 4 positions)
        coord_mask = (
            torch.arange(len(target_seq), device=target_seq.device) % 5 < 4
        ) & non_pad_mask
        correct, total = self._get_correct_and_total(
            pred_tokens, target_seq, coord_mask
        )
        metrics["coordinate"] = (correct, total)

        # Class tokens (5th position)
        class_mask = (
            torch.arange(len(target_seq), device=target_seq.device) % 5 == 4
        ) & non_pad_mask
        correct, total = self._get_correct_and_total(
            pred_tokens, target_seq, class_mask
        )
        metrics["class"] = (correct, total)

        return metrics

    def compute_object_accuracies(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        non_pad_mask: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute accuracy metrics for complete objects (all 5 tokens correct).

        An object is considered correct only if all 5 tokens (ymin, xmin, ymax, xmax, class) are predicted correctly.

        Args:
            pred_tokens: Predicted token indices [N]
            target_seq: Target token indices [N]
            non_pad_mask: Boolean mask for non-padding tokens [N]

        Returns:
            Dictionary with key 'object' mapping to tuple of (correct_objects, total_objects)
        """
        # Reshape into object groups of 5 tokens
        seq_len = len(target_seq)
        group_size = 5
        num_complete_groups = seq_len // group_size

        # Only consider complete groups
        pred_groups = pred_tokens[: group_size * num_complete_groups].view(
            -1, group_size
        )
        target_groups = target_seq[: group_size * num_complete_groups].view(
            -1, group_size
        )
        mask_groups = non_pad_mask[: group_size * num_complete_groups].view(
            -1, group_size
        )

        # Object is correct if all tokens match and none are padding
        object_correct = (pred_groups == target_groups).all(dim=1) & mask_groups.all(
            dim=1
        )
        object_valid = mask_groups.all(dim=1)

        return {"object": (object_correct.sum(), object_valid.sum())}

    def compute_sequence_accuracies(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute sequence-level accuracy metrics.

        Calculates three metrics:
        - sequence_length: Whether the predicted number of objects matches the target
        - eos_timing: Whether the EOS token was predicted at the correct position
        - complete_sequence: Whether the entire sequence matches exactly (up to EOS)

        Args:
            pred_tokens: Predicted token indices [N]
            target_seq: Target token indices [N]

        Returns:
            Dictionary mapping metric names to tuples of (correct_count, total_count)
        """
        metrics = {}

        # Find EOS positions
        pred_eos_pos = (pred_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
        target_eos_pos = (target_seq == self.eos_token_id).nonzero(as_tuple=True)[0]

        # Sequence length accuracy (correct number of objects)
        pred_seq_len = pred_eos_pos[0] if len(pred_eos_pos) > 0 else len(pred_tokens)
        target_seq_len = (
            target_eos_pos[0] if len(target_eos_pos) > 0 else len(target_seq)
        )
        seq_len_correct = torch.as_tensor(
            ((pred_seq_len // 5) == (target_seq_len // 5)).clone().detach(),
            device=pred_tokens.device,
        )
        metrics["sequence_length"] = (seq_len_correct, torch.ones_like(seq_len_correct))

        # EOS timing accuracy
        eos_timing_correct = (
            torch.tensor(
                [
                    len(pred_eos_pos) > 0
                    and len(target_eos_pos) > 0
                    and pred_eos_pos[0] == target_eos_pos[0]
                ],
                device=pred_tokens.device,
            )
            .clone()
            .detach()
        )

        metrics["eos_timing"] = (
            eos_timing_correct,
            torch.ones_like(eos_timing_correct),
        )

        # Complete sequence accuracy
        if len(target_eos_pos) > 0:
            target_end = target_eos_pos[0] + 1
        else:
            target_end = len(target_seq)

        if len(pred_eos_pos) > 0:
            pred_end = pred_eos_pos[0] + 1
        else:
            pred_end = len(pred_tokens)

        sequence_correct = (
            torch.tensor(
                [
                    pred_end == target_end
                    and (pred_tokens[:pred_end] == target_seq[:target_end]).all()
                ],
                device=pred_tokens.device,
            )
            .clone()
            .detach()
        )
        metrics["complete_sequence"] = (
            sequence_correct,
            torch.ones_like(sequence_correct),
        )

        return metrics

    def gather_and_normalize_metrics(
        self,
        raw_metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        gather_fn=None,
    ) -> Dict[str, float]:
        """Gather metrics across distributed processes and normalize to accuracies.

        Args:
            gather_fn: Function to gather tensors across distributed processes
            raw_metrics: Dictionary mapping metric names to tuples of (correct_count, total_count)

        Returns:
            Dictionary mapping metric names to normalized accuracy values (0-1 range)

        Also accumulates the normalized metrics for epoch-level calculations.
        """
        normalized_metrics = {}

        for name, (correct, total) in raw_metrics.items():
            # Stack for gathering
            local_counts = torch.stack([correct, total])

            if gather_fn is None:
                gathered_counts = local_counts
            else:
                # Gather from all processes
                gathered_counts = gather_fn(local_counts)

            # Sum across processes
            global_correct = gathered_counts[0::2].sum()
            global_total = gathered_counts[1::2].sum()

            # Normalize
            accuracy = (global_correct / global_total.clamp(min=1)).item()
            normalized_metrics[name] = accuracy

            # Accumulate for epoch metrics
            if name not in self.accumulated_metrics:
                self.accumulated_metrics[name] = []
            self.accumulated_metrics[name].append(accuracy)

        return normalized_metrics

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,
        target_seq: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
        gather_fn=None,
    ) -> Dict[str, float]:
        """Calculate epoch-level metrics from all accumulated batch metrics.

        Returns:
            Dictionary mapping metric names to average accuracy values across all
            batches processed since the last reset_metrics() call.
        """
        # Get predictions
        pred_tokens = torch.argmax(logits, dim=-1)

        # Create mask for non-padding tokens
        non_pad_mask = target_seq != self.pad_token_id
        if token_weights is not None:
            non_pad_mask = non_pad_mask & (token_weights > 0)

        # Compute raw metrics (correct and total counts)
        raw_metrics = {}

        # Overall token accuracy
        correct, total = self._get_correct_and_total(
            pred_tokens, target_seq, non_pad_mask
        )
        raw_metrics["token"] = (correct, total)

        # Add metrics from each category
        raw_metrics.update(
            self.compute_position_accuracies(pred_tokens, target_seq, non_pad_mask)
        )
        raw_metrics.update(
            self.compute_type_accuracies(pred_tokens, target_seq, non_pad_mask)
        )
        raw_metrics.update(
            self.compute_object_accuracies(pred_tokens, target_seq, non_pad_mask)
        )
        raw_metrics.update(self.compute_sequence_accuracies(pred_tokens, target_seq))
        raw_metrics.update(
            self.compute_first_token_accuracy(pred_tokens, target_seq, non_pad_mask)
        )

        # Gather across processes and normalize
        if gather_fn is not None:
            return self.gather_and_normalize_metrics(raw_metrics, gather_fn)
        return raw_metrics

    def get_epoch_metrics(self) -> Dict[str, float]:
        """Calculate epoch-level metrics from accumulated values."""
        epoch_metrics = {}
        for name, values in self.accumulated_metrics.items():
            epoch_metrics[name] = sum(values) / len(values)
        return epoch_metrics
