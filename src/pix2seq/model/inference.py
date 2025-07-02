from typing import Optional, Tuple

import torch

from data.tokenizer import TokenProcessor


class SequenceGenerator:
    """Generates object detection sequences using the TensorFlow Pix2Seq approach.

    This implementation follows the original paper's methodology of using soft constraints
    during generation and handling invalid boxes during post-processing, rather than
    enforcing strict validation during generation.
    """

    def __init__(
        self,
        token_processor: TokenProcessor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.4,
        max_seq_len: Optional[int] = None,
    ):
        """Initialize sequence generator with sampling parameters.

        Args:
            token_processor: TokenProcessor instance for handling token conversions
            temperature: Softmax temperature for sampling (higher = more diverse)
            top_k: Number of highest probability tokens to keep (0 = disabled)
            top_p: Cumulative probability threshold for nucleus sampling
            max_seq_len: Maximum sequence length (uses TokenProcessor's if None)
        """
        self.token_processor = token_processor
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_seq_len = max_seq_len or token_processor.max_seq_len

        # Initialize token mask cache
        self.token_masks = TokenMaskCache(token_processor)

    def generate(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Generate sequences with debug output.

        Args:
            model: The Pix2Seq model
            images: Input images [B,C,H,W]

        Returns:
            pred_seq: Generated sequences [B,L]
            class_logits: Logits for class tokens [B,N,V]
            encoded: Encoded image features
        """
        with torch.inference_mode():
            # Encode all images in batch
            encoded, features = model.encode(images)
            batch_size = encoded.size(0)
            device = encoded.device
            self.token_masks.to(device)

            # Initialize sequences with BOS token for all batch items
            cur_seq = torch.full(
                (batch_size, 1),
                self.token_processor.BOS_TOKEN,
                dtype=torch.long,
                device=device,
            )

            # Track class logits and completed sequences
            # class_logits = []
            completed = torch.zeros(batch_size, dtype=torch.bool, device=device)
            max_objects = (self.max_seq_len - 1) // 5
            stacked_logits = torch.full(
                (batch_size, max_objects, model.vocab_size),
                float("-inf"),
                device=device,
            )
            class_logit_indices = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            idx = 0
            # Process batches until all sequences complete or max length reached
            while not completed.all() and cur_seq.size(1) < self.max_seq_len:
                # Get next token logits for all active sequences
                active_batch = ~completed
                if not active_batch.any():
                    break

                # Get logits for next token
                logits = model.decode(cur_seq, encoded, use_cache=True)
                next_token_logits = logits[:, -1, :] / self.temperature

                # Get position in pattern for constraint checking
                pattern_pos = self._get_pattern_position(cur_seq.size(1))

                # Apply constraints to all sequences in batch
                allowed = self.token_masks.get_allowed_tokens(pattern_pos, cur_seq)
                constrained_next_token_logits = next_token_logits.masked_fill(
                    ~allowed.to(device), float("-inf")
                )

                # For completed sequences, only allow padding token
                padding_logits = torch.full_like(next_token_logits, float("-inf"))
                padding_logits[:, self.token_processor.PADDING_TOKEN] = 0
                constrained_next_token_logits = torch.where(
                    completed.unsqueeze(1),
                    padding_logits,
                    constrained_next_token_logits,
                )

                # Track class logits for active sequences
                if pattern_pos == 4:
                    active_indices = torch.arange(batch_size, device=device)[~completed]
                    stacked_logits[
                        active_indices, class_logit_indices[active_indices]
                    ] = next_token_logits[active_indices]
                    class_logit_indices[active_indices] += 1

                next_tokens = self._sample_next_tokens(constrained_next_token_logits)

                # Update sequences
                cur_seq = torch.cat([cur_seq, next_tokens], dim=1)

                # Update completion status for active sequences

                if pattern_pos == 0:  # Only check completion at start of new sequence
                    newly_completed = (
                        next_tokens.squeeze(-1) == self.token_processor.EOS_TOKEN
                    )

                    if newly_completed.any():
                        # Only validate boxes for sequences that generated EOS
                        sequences_with_eos = cur_seq[newly_completed]
                        valid_boxes = self._validate_box_coordinates(sequences_with_eos)

                        # Update completion status only for sequences that generated EOS
                        newly_completed = (
                            newly_completed.clone()
                        )  # Create a copy to modify
                        newly_completed[newly_completed.clone()] = valid_boxes

                    completed = completed | newly_completed
                idx += 1

            # Stack class logits if we have any
            # stacked_logits = torch.stack(class_logits, dim=1) if class_logits else None

            max_used = class_logit_indices.max().item()
            stacked_logits = stacked_logits[:, :max_used]

            return cur_seq, stacked_logits, features

    def _get_pattern_position(self, seq_len: int) -> int:
        """Get position in the 5-token pattern (ymin,xmin,ymax,xmax,class).

        Args:
            seq_len: Current sequence length including BOS token

        Returns:
            Integer 0-4 indicating position in pattern (0=ymin, 4=class)
        """
        if seq_len <= 1:  # Only BOS token
            return 0
        return (seq_len - 1) % 5

    def _sample_next_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next tokens using temperature and top-k/top-p, handling FAKE_CLASS_TOKEN.

        Args:
            logits: Unnormalized token probabilities [B,V]

        Returns:
            Sampled token indices [B,1]
        """

        if self.top_k > 0:
            # Apply top-k filtering
            indices_to_remove = (
                logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            )
            logits[indices_to_remove] = float("-inf")

        if self.top_p > 0:
            # Apply nucleus sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Apply the mask to the logits
            for batch_idx in range(logits.size(0)):
                indices_to_remove = sorted_indices_to_remove[batch_idx].scatter(
                    0, sorted_indices[batch_idx], sorted_indices_to_remove[batch_idx]
                )
                logits[batch_idx][indices_to_remove] = float("-inf")

        # Sample from filtered distribution
        probs = torch.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1)

        # Check for FAKE_CLASS_TOKEN and replace with next best real class
        fake_token_mask = sampled_tokens == self.token_processor.FAKE_CLASS_TOKEN
        if fake_token_mask.any():
            # Create class token mask
            class_range = torch.arange(
                self.token_processor.BASE_VOCAB_SHIFT,
                self.token_processor.FAKE_CLASS_TOKEN,
                device=logits.device,
            )
            class_mask = torch.zeros_like(logits, dtype=torch.bool)
            class_mask[:, class_range] = True

            # Mask out FAKE_CLASS_TOKEN and invalid tokens
            valid_logits = logits.clone()
            valid_logits[:, self.token_processor.FAKE_CLASS_TOKEN] = float("-inf")
            valid_logits = torch.where(
                class_mask, valid_logits, torch.tensor(float("-inf")).to(logits.device)
            )

            # Get best valid class token
            replacement_probs = torch.softmax(valid_logits, dim=-1)
            replacement_tokens = torch.multinomial(replacement_probs, num_samples=1)

            # Replace FAKE_CLASS_TOKEN with best valid class
            sampled_tokens = torch.where(
                fake_token_mask, replacement_tokens, sampled_tokens
            )

        return sampled_tokens

    def _validate_box_coordinates(self, seq: torch.Tensor) -> bool:
        """
        Args:
            seq: Token sequence including BOS token

        Returns:
            True if coordinates form a valid box
        """

        # Get all complete boxes
        seq_len = seq.size(1) - 1  # Exclude BOS
        num_complete_boxes = seq_len // 5
        if num_complete_boxes == 0:
            return torch.ones(seq.size(0), dtype=torch.bool, device=seq.device)

        # Reshape into boxes
        box_tokens = seq[:, 1 : 1 + num_complete_boxes * 5].view(
            -1, num_complete_boxes, 5
        )

        # Get coordinates
        coord_tokens = box_tokens[..., :4]
        coords = self.token_processor.dequantize(coord_tokens)

        # Validate coordinates
        ymin, xmin, ymax, xmax = coords.unbind(-1)
        valid_y = (ymax > ymin) & (ymin >= 0) & (ymax <= 1)
        valid_x = (xmax > xmin) & (xmin >= 0) & (xmax <= 1)

        # A box is valid if both x and y coordinates are valid
        valid_boxes = valid_y & valid_x
        return valid_boxes.all(dim=1)


class TokenMaskCache:
    """Cache of pre-computed token masks for faster constraint checking."""

    def __init__(self, token_processor):
        self.token_processor = token_processor
        self.device = "cpu"

        # Pre-compute static masks for each position
        self.position_masks = {}

        # ymin position (0) - coordinates or EOS
        ymin_mask = torch.zeros(token_processor.vocab_size, dtype=torch.bool)
        ymin_mask[
            token_processor.coord_vocab_shift : token_processor.coord_vocab_shift
            + token_processor.quantization_bins
        ] = True
        ymin_mask[token_processor.EOS_TOKEN] = True
        self.position_masks[0] = ymin_mask

        # xmin position (1) - just coordinates
        xmin_mask = torch.zeros(token_processor.vocab_size, dtype=torch.bool)
        xmin_mask[
            token_processor.coord_vocab_shift : token_processor.coord_vocab_shift
            + token_processor.quantization_bins
        ] = True
        self.position_masks[1] = xmin_mask

        # Class position (4) - valid class tokens
        class_mask = torch.zeros(token_processor.vocab_size, dtype=torch.bool)
        class_mask[
            token_processor.BASE_VOCAB_SHIFT : token_processor.FAKE_CLASS_TOKEN
        ] = True
        self.position_masks[4] = class_mask

        # Create coordinate range tensor for faster comparisons
        self.coord_range = torch.arange(token_processor.quantization_bins)

    def to(self, device):
        """Move masks to specified device."""
        if device == self.device:
            return self

        self.device = device
        self.coord_range = self.coord_range.to(device)
        for pos in self.position_masks:
            self.position_masks[pos] = self.position_masks[pos].to(device)
        return self

    def get_allowed_tokens(
        self, pattern_pos: int, cur_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get allowed tokens using cached masks where possible."""
        # Use pre-computed masks for static positions
        if pattern_pos in (0, 1, 4):
            return self.position_masks[pattern_pos]

        # Dynamic constraints for ymax/xmax need sequence context
        device = cur_seq.device
        allowed = torch.zeros(
            self.token_processor.vocab_size, dtype=torch.bool, device=device
        )

        if pattern_pos == 2:  # ymax
            seq_len = cur_seq.size(1)
            ymin_values = (
                cur_seq[:, seq_len - 2] - self.token_processor.coord_vocab_shift
            )

            # Vectorized comparison for allowed coordinates
            coord_mask = self.coord_range[None, :] > ymin_values[:, None]
            start_idx = self.token_processor.coord_vocab_shift
            for i, mask in enumerate(coord_mask):
                allowed[start_idx : start_idx + len(mask)][mask] = True

        elif pattern_pos == 3:  # xmax
            seq_len = cur_seq.size(1)
            xmin_values = (
                cur_seq[:, seq_len - 2] - self.token_processor.coord_vocab_shift
            )

            # Vectorized comparison for allowed coordinates
            coord_mask = self.coord_range[None, :] > xmin_values[:, None]
            start_idx = self.token_processor.coord_vocab_shift
            for i, mask in enumerate(coord_mask):
                allowed[start_idx : start_idx + len(mask)][mask] = True

        return allowed


class SingleLayerKVCache(torch.nn.Module):
    """KV Cache for a single transformer layer during inference."""

    def __init__(
        self,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Cache will be created on first use
        self.cache_initialised = False

        # Track current position in sequence
        self.register_buffer(
            "cache_pos", torch.zeros(1, dtype=torch.long), persistent=False
        )

    def _create_buffers(self, batch_size: int, new_keys):
        """Lazily create cache buffers with correct batch size."""
        cache_shape = (batch_size, self.n_heads, self.max_seq_len, self.head_dim)
        self.register_buffer(
            "k_cache",
            torch.zeros(cache_shape, dtype=self.dtype, device=new_keys.device).clone(),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(cache_shape, dtype=self.dtype, device=new_keys.device).clone(),
            persistent=False,
        )
        self.cache_initialised = True

    def update(
        self, new_keys: torch.Tensor, new_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new keys and values in cache and return full state."""
        batch_size = new_keys.size(0)

        # Create cache buffers if they don't exist
        if not self.cache_initialised:
            self._create_buffers(batch_size, new_keys)

        # Update cache at current position
        self.k_cache[:, :, self.cache_pos : self.cache_pos + 1] = new_keys
        self.v_cache[:, :, self.cache_pos : self.cache_pos + 1] = new_values
        self.cache_pos += 1

        # Return all stored states
        return (
            self.k_cache[:, :, : self.cache_pos],
            self.v_cache[:, :, : self.cache_pos],
        )

    @torch.no_grad()
    def reset(self):
        """Reset cache to initial state."""
        self.cache_pos.zero_()
        if self.cache_initialised:
            self.k_cache.zero_()
            self.v_cache.zero_()
