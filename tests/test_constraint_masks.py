"""Tests that the vectorized re-scoring constraint masks match generation.

The re-scored distribution must equal the sampling distribution, so
build_constraint_masks (whole-sequence, vectorized) must reproduce
TokenMaskCache.get_allowed_tokens (per-step) exactly.
"""

import torch

from conftest import make_sequence

from model.inference import SequenceGenerator, TokenMaskCache
from rl.rescoring import build_constraint_masks, compute_valid_token_mask


def _random_valid_sequences(token_processor, num_seqs=8, num_objects=3, seed=0):
    """Random structurally valid sequences (coords obey ymax>ymin, xmax>xmin)."""
    g = torch.Generator().manual_seed(seed)
    bins = token_processor.quantization_bins
    shift = token_processor.coord_vocab_shift
    seqs = []
    for _ in range(num_seqs):
        tokens = [token_processor.BOS_TOKEN]
        for _ in range(num_objects):
            ymin = int(torch.randint(0, bins - 1, (1,), generator=g))
            xmin = int(torch.randint(0, bins - 1, (1,), generator=g))
            ymax = int(torch.randint(ymin + 1, bins, (1,), generator=g))
            xmax = int(torch.randint(xmin + 1, bins, (1,), generator=g))
            cls = int(torch.randint(0, token_processor.num_classes, (1,), generator=g))
            tokens.extend(
                [
                    ymin + shift,
                    xmin + shift,
                    ymax + shift,
                    xmax + shift,
                    cls + token_processor.BASE_VOCAB_SHIFT,
                ]
            )
        tokens.append(token_processor.EOS_TOKEN)
        seqs.append(tokens)
    return torch.tensor(seqs, dtype=torch.long)


class TestBuildConstraintMasks:
    def test_matches_generation_masks_per_step(self, token_processor):
        """Vectorized rebuild == per-step TokenMaskCache on every position."""
        sequences = _random_valid_sequences(token_processor)
        masks = build_constraint_masks(sequences, token_processor)

        cache = TokenMaskCache(token_processor)
        generator = SequenceGenerator(token_processor, max_seq_len=sequences.size(1))

        for t in range(sequences.size(1) - 1):
            prefix = sequences[:, : t + 1]
            pattern_pos = generator._get_pattern_position(prefix.size(1))
            allowed = cache.get_allowed_tokens(pattern_pos, prefix)
            expected = allowed.expand_as(masks[:, t]) if allowed.dim() == 1 else allowed
            assert torch.equal(masks[:, t], expected), f"mismatch at position {t}"

    def test_static_positions(self, token_processor):
        sequences = _random_valid_sequences(token_processor, num_seqs=2)
        masks = build_constraint_masks(sequences, token_processor)
        tp = token_processor

        # Position 0 (ymin): coords + EOS only
        assert masks[0, 0, tp.EOS_TOKEN]
        assert masks[0, 0, tp.coord_vocab_shift]
        assert not masks[0, 0, tp.BOS_TOKEN]
        assert not masks[0, 0, tp.BASE_VOCAB_SHIFT]

        # Position 1 (xmin): coords only, no EOS
        assert not masks[0, 1, tp.EOS_TOKEN]
        assert masks[0, 1, tp.coord_vocab_shift]

        # Position 4 (class): real classes only, no FAKE_CLASS_TOKEN
        assert masks[0, 4, tp.BASE_VOCAB_SHIFT]
        assert masks[0, 4, tp.BASE_VOCAB_SHIFT + tp.num_classes - 1]
        assert not masks[0, 4, tp.FAKE_CLASS_TOKEN]
        assert not masks[0, 4, tp.coord_vocab_shift]

    def test_dynamic_positions_strictly_greater(self, token_processor):
        tp = token_processor
        sequences = make_sequence(
            tp, boxes=[[0.2, 0.4, 0.6, 0.8]], labels=[1], seq_len=8
        )
        masks = build_constraint_masks(sequences, tp)

        # ymax (position 2) must be > ymin bin; ymin = 0.4 -> bin 40 (99 bins)
        ymin_bin = int(round(0.4 * (tp.quantization_bins - 1)))
        ymin_token = tp.coord_vocab_shift + ymin_bin
        assert not masks[0, 2, ymin_token]
        assert masks[0, 2, ymin_token + 1]

        # xmax (position 3) must be > xmin bin; xmin = 0.2
        xmin_bin = int(round(0.2 * (tp.quantization_bins - 1)))
        xmin_token = tp.coord_vocab_shift + xmin_bin
        assert not masks[0, 3, xmin_token]
        assert masks[0, 3, xmin_token + 1]

    def test_top_bin_guard(self, token_processor):
        """ymin at the top bin keeps the top bin allowed (no empty mask)."""
        tp = token_processor
        top = tp.quantization_bins - 1
        seq = torch.tensor(
            [
                [
                    tp.BOS_TOKEN,
                    tp.coord_vocab_shift + top,  # ymin at top bin
                    tp.coord_vocab_shift + 10,  # xmin
                    tp.coord_vocab_shift + top,  # ymax (only allowed choice)
                    tp.coord_vocab_shift + 11,  # xmax
                    tp.BASE_VOCAB_SHIFT,  # class
                    tp.EOS_TOKEN,
                ]
            ],
            dtype=torch.long,
        )
        masks = build_constraint_masks(seq, tp)
        ymax_row = masks[0, 2]
        assert ymax_row.any(), "allowed set must never be empty"
        assert ymax_row[tp.coord_vocab_shift + top]
        assert not ymax_row[tp.coord_vocab_shift + top - 1]

        # Generation-side mask behaves identically
        cache = TokenMaskCache(tp)
        gen_allowed = cache.get_allowed_tokens(2, seq[:, :3])
        assert torch.equal(gen_allowed[0], ymax_row)


class TestComputeValidTokenMask:
    def test_early_eos(self, token_processor):
        tp = token_processor
        seq = make_sequence(tp, boxes=[[0.1, 0.1, 0.5, 0.5]], labels=[0], seq_len=12)
        mask = compute_valid_token_mask(seq, tp)
        # Targets: 5 box tokens + EOS valid, remaining PAD invalid
        assert mask[0, :6].all()
        assert not mask[0, 6:].any()

    def test_immediate_eos(self, token_processor):
        tp = token_processor
        seq = torch.tensor(
            [[tp.BOS_TOKEN, tp.EOS_TOKEN, tp.PADDING_TOKEN, tp.PADDING_TOKEN]]
        )
        mask = compute_valid_token_mask(seq, tp)
        assert mask[0, 0]
        assert not mask[0, 1:].any()

    def test_no_eos(self, token_processor):
        tp = token_processor
        seq = make_sequence(
            tp,
            boxes=[[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]],
            labels=[0, 1],
            seq_len=11,  # BOS + 2 boxes, no room for EOS
        )
        mask = compute_valid_token_mask(seq, tp)
        assert mask.all()

    def test_tokens_after_first_eos_excluded(self, token_processor):
        tp = token_processor
        # EOS mid-sequence followed by more tokens (invalid-box continuation)
        seq = torch.tensor(
            [
                [
                    tp.BOS_TOKEN,
                    tp.EOS_TOKEN,
                    tp.coord_vocab_shift + 5,
                    tp.coord_vocab_shift + 6,
                    tp.EOS_TOKEN,
                    tp.PADDING_TOKEN,
                ]
            ]
        )
        mask = compute_valid_token_mask(seq, tp)
        assert mask[0, 0]  # first EOS included
        assert not mask[0, 1:].any()  # everything after excluded
