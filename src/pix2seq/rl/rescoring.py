"""Teacher-forced re-scoring of sampled sequences for policy-gradient training.

RL fine-tuning needs differentiable log-probabilities of sampled sequences.
Backpropagating through autoregressive generation is not possible here (the KV
cache updates its buffers in place) and would be memory-prohibitive anyway, so
instead sequences are sampled without gradients and then re-scored with a single
teacher-forced forward pass - the same code path used for MLE training.

For the re-scored distribution to match the sampling distribution exactly, the
structural constraint masks applied during generation (see
``model.inference.TokenMaskCache``) are rebuilt here for every position of the
sampled sequences, and the same softmax temperature is applied.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from data.tokenizer import TokenProcessor


@dataclass
class RescoreOutput:
    """Result of re-scoring sequences with a teacher-forced forward pass.

    Attributes:
        log_probs: [N, S-1] log probability of each generated token
            (targets = sequences[:, 1:]) under the constrained, temperature-scaled
            distribution. Differentiable. Zero at positions excluded by ``mask``.
        mask: [N, S-1] boolean mask - True for tokens up to and including the
            first EOS, False for PAD and anything after the first EOS.
        class_logits: [N, num_objects, V] raw (unscaled, unconstrained) logits at
            the class-token slots, or None. Rows beyond an image's actual object
            count are meaningless - slice per image before use (see
            ``extract_object_confidences``).
    """

    log_probs: torch.Tensor
    mask: torch.Tensor
    class_logits: Optional[torch.Tensor] = None


def build_constraint_masks(
    sequences: torch.Tensor,
    token_processor: TokenProcessor,
) -> torch.Tensor:
    """Rebuild the generation-time allowed-token masks for whole sequences.

    Position ``t`` of the result masks the prediction of ``sequences[:, t + 1]``.
    The token pattern repeats every 5 tokens (ymin, xmin, ymax, xmax, class), so
    the pattern position of the token predicted at input position ``t`` is
    ``t % 5`` (this matches ``SequenceGenerator._get_pattern_position`` with a
    prefix of length ``t + 1``). The dynamic ymax/xmax constraints reference the
    corresponding ymin/xmin token at full-sequence index ``t - 1``.

    Args:
        sequences: [N, S] token sequences including BOS
        token_processor: TokenProcessor defining the vocabulary layout

    Returns:
        [N, S-1, V] boolean masks (True = token allowed)
    """
    device = sequences.device
    num_seqs, seq_len = sequences.shape
    num_positions = seq_len - 1
    vocab_size = token_processor.vocab_size
    coord_shift = token_processor.coord_vocab_shift
    bins = token_processor.quantization_bins

    # Static per-pattern-position masks, mirroring TokenMaskCache
    coords = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    coords[coord_shift : coord_shift + bins] = True

    static = torch.zeros(5, vocab_size, dtype=torch.bool, device=device)
    static[0] = coords
    static[0, token_processor.EOS_TOKEN] = True  # ymin slot may emit EOS
    static[1] = coords  # xmin
    static[2] = coords  # ymax (refined per sample below)
    static[3] = coords  # xmax (refined per sample below)
    static[4, token_processor.BASE_VOCAB_SHIFT : token_processor.FAKE_CLASS_TOKEN] = (
        True  # class
    )

    pattern = torch.arange(num_positions, device=device) % 5  # [S-1]
    masks = static[pattern].unsqueeze(0).expand(num_seqs, -1, -1).clone()

    # Dynamic ymax/xmax constraints: coordinate must be strictly greater than the
    # ymin/xmin token two steps earlier in the target pattern, i.e. at
    # full-sequence index t - 1 for mask position t
    dynamic = (pattern == 2) | (pattern == 3)  # [S-1]
    if dynamic.any():
        ref_tokens = sequences[:, :-2][:, dynamic[1:]]  # [N, D] token at t-1
        min_bins = ref_tokens - coord_shift
        # Same guard as generation: keep at least the top bin allowed when the
        # reference was sampled at the top bin (degenerate box, filtered later)
        min_bins = min_bins.clamp(max=bins - 2)

        coord_allowed = (
            torch.arange(bins, device=device)[None, None, :] > min_bins[:, :, None]
        )  # [N, D, bins]

        dynamic_rows = masks[:, dynamic]
        dynamic_rows[:, :, coord_shift : coord_shift + bins] = coord_allowed
        masks[:, dynamic] = dynamic_rows

    return masks


def compute_valid_token_mask(
    sequences: torch.Tensor,
    token_processor: TokenProcessor,
) -> torch.Tensor:
    """Mask of generated tokens that should contribute to the policy gradient.

    True for every target token (``sequences[:, 1:]``) up to and including the
    first EOS; False for PAD and all tokens after the first EOS. Tokens after the
    first EOS never influence the decoded boxes (decoding truncates at the first
    EOS), so they carry no reward signal.

    Args:
        sequences: [N, S] token sequences including BOS

    Returns:
        [N, S-1] boolean mask
    """
    targets = sequences[:, 1:]
    is_eos = targets == token_processor.EOS_TOKEN
    eos_before = is_eos.long().cumsum(dim=1) - is_eos.long()  # EOS strictly before t
    return (eos_before == 0) & (targets != token_processor.PADDING_TOKEN)


def rescore_sequences(
    model: torch.nn.Module,
    images: torch.Tensor,
    sequences: torch.Tensor,
    token_processor: TokenProcessor,
    temperature: float = 1.0,
    num_samples_per_image: int = 1,
    return_class_logits: bool = False,
) -> RescoreOutput:
    """Compute differentiable log-probs of sequences with one teacher-forced pass.

    Args:
        model: Pix2Seq model (pass the accelerator-wrapped model during
            distributed training so gradient synchronisation hooks fire)
        images: [B, C, H, W] input images
        sequences: [B * num_samples_per_image, S] sampled sequences including BOS,
            grouped per image in repeat_interleave order
        token_processor: TokenProcessor defining the vocabulary layout
        temperature: Softmax temperature used when the sequences were sampled
        num_samples_per_image: Number of sequences per image (K)
        return_class_logits: If True, also return raw logits at class-token slots

    Returns:
        RescoreOutput with log_probs [N, S-1], mask [N, S-1] and optionally
        class_logits [N, num_objects, V]
    """
    # Sequences typically come from generation under torch.inference_mode();
    # clone to get ordinary tensors that autograd may save for backward
    sequences = sequences.clone()

    input_seq = sequences[:, :-1]
    targets = sequences[:, 1:]
    padding_mask = input_seq == token_processor.PADDING_TOKEN

    logits = model(
        images,
        input_seq,
        tgt_padding_mask=padding_mask,
        num_tgt_per_image=num_samples_per_image,
    )  # [N, S-1, V]

    # Match the sampling distribution: temperature scaling first, then the
    # structural constraint masks (identical to generation, where masking with
    # -inf commutes with the scalar division)
    allowed = build_constraint_masks(sequences, token_processor)
    scored = (logits / temperature).masked_fill(~allowed, float("-inf"))
    log_probs_all = torch.log_softmax(scored, dim=-1)

    log_probs = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [N,S-1]

    mask = compute_valid_token_mask(sequences, token_processor)
    # Use where() rather than multiplication: disallowed targets (e.g. PAD after
    # EOS) have log-prob -inf and 0 * -inf would produce NaN
    log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))

    class_logits = None
    if return_class_logits:
        num_objects = targets.size(1) // 5
        class_positions = (
            torch.arange(num_objects, device=sequences.device) * 5 + 4
        )  # class slots at t % 5 == 4
        # Raw logits (no temperature, no constraints): confidences at evaluation
        # time are computed from raw class logits, so IoU supervision must train
        # the same quantity
        class_logits = logits[:, class_positions, :]

    return RescoreOutput(log_probs=log_probs, mask=mask, class_logits=class_logits)


def extract_object_confidences(
    sequences: torch.Tensor,
    class_logits: torch.Tensor,
    token_processor: TokenProcessor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Decode boxes/labels and differentiable confidences for IoU supervision.

    For each sequence, decodes the boxes it actually produced (up to the first
    EOS/PAD), filters invalid boxes the same way ``post_process_sequences`` does,
    and computes the confidence of each kept box as the softmax probability of
    its sampled class token under the re-scored class logits. Confidences remain
    differentiable so an IoU-supervision loss can train them.

    Args:
        sequences: [N, S] token sequences including BOS
        class_logits: [N, num_objects, V] raw class-slot logits from
            ``rescore_sequences(..., return_class_logits=True)``
        token_processor: TokenProcessor defining the vocabulary layout

    Returns:
        Per-sequence lists of (boxes [n_i, 4] normalized XYXY, labels [n_i],
        confidences [n_i])
    """
    boxes_list, labels_list, _ = token_processor.decode_tokens(sequences)

    targets = sequences[:, 1:]
    is_end = (targets == token_processor.PADDING_TOKEN) | (
        targets == token_processor.EOS_TOKEN
    )

    out_boxes: List[torch.Tensor] = []
    out_labels: List[torch.Tensor] = []
    out_confidences: List[torch.Tensor] = []

    for i in range(sequences.size(0)):
        end_positions = is_end[i].nonzero(as_tuple=True)[0]
        end_idx = (
            end_positions[0].item() if len(end_positions) > 0 else targets.size(1)
        )
        num_objects = min(
            end_idx // 5, boxes_list[i].size(0), class_logits.size(1)
        )

        boxes = boxes_list[i][:num_objects]
        labels = labels_list[i][:num_objects]

        if num_objects == 0:
            out_boxes.append(boxes)
            out_labels.append(labels)
            out_confidences.append(
                torch.zeros(0, device=class_logits.device, dtype=class_logits.dtype)
            )
            continue

        class_probs = torch.softmax(class_logits[i, :num_objects], dim=-1)
        class_tokens = labels + token_processor.BASE_VOCAB_SHIFT
        confidences = class_probs.gather(1, class_tokens.unsqueeze(1)).squeeze(1)

        # Same validity filtering as post_process_sequences
        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
            & torch.all((boxes >= 0) & (boxes <= 1), dim=1)
        )

        out_boxes.append(boxes[valid])
        out_labels.append(labels[valid])
        out_confidences.append(confidences[valid])

    return out_boxes, out_labels, out_confidences
