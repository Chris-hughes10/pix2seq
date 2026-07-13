"""Shared fixtures for pix2seq tests.

Provides a small TokenProcessor and a tiny, randomly initialised Pix2SeqModel
that build quickly on CPU without network access.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pix2seq"))

from data.tokenizer import TokenProcessor  # noqa: E402
from model.model import Pix2SeqModel  # noqa: E402

MAX_OBJECTS = 3
MAX_SEQ_LEN = 2 + MAX_OBJECTS * 5  # BOS + 3 objects + EOS = 17
NUM_CLASSES = 5
QUANTIZATION_BINS = 100
IMAGE_SIZE = 32


@pytest.fixture(scope="session")
def token_processor() -> TokenProcessor:
    return TokenProcessor(
        quantization_bins=QUANTIZATION_BINS,
        noise_bbox_weight=1.0,
        eos_token_weight=0.1,
        max_seq_len=MAX_SEQ_LEN,
        num_classes=NUM_CLASSES,
        num_special_tokens=10,
        verbose=False,
    )


@pytest.fixture(scope="session")
def tiny_model(token_processor) -> Pix2SeqModel:
    torch.manual_seed(0)
    model = Pix2SeqModel(
        max_seq_len=MAX_SEQ_LEN,
        image_size=IMAGE_SIZE,
        patch_size=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_dim=32,
        num_heads=2,
        dim_feedforward=64,
        dropout=0.0,
        drop_path=0.0,
        bos_token_id=token_processor.BOS_TOKEN,
        eos_token_id=token_processor.EOS_TOKEN,
        coord_vocab_shift=token_processor.coord_vocab_shift,
        base_vocab_shift=token_processor.BASE_VOCAB_SHIFT,
        num_quantization_bins=token_processor.quantization_bins,
        token_processor=token_processor,
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=False,
    )
    model.eval()
    return model


@pytest.fixture()
def images() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)


def make_sequence(token_processor, boxes, labels, seq_len=MAX_SEQ_LEN):
    """Build a token sequence [1, seq_len] from normalized XYXY boxes + labels."""
    tokens = [token_processor.BOS_TOKEN]
    for box, label in zip(boxes, labels):
        xyxy = torch.tensor(box, dtype=torch.float32)
        yxyx = xyxy[[1, 0, 3, 2]]
        coord_tokens = token_processor.quantize(yxyx).tolist()
        tokens.extend(coord_tokens)
        tokens.append(label + token_processor.BASE_VOCAB_SHIFT)
    tokens.append(token_processor.EOS_TOKEN)
    tokens.extend(
        [token_processor.PADDING_TOKEN] * (seq_len - len(tokens))
    )
    return torch.tensor(tokens[:seq_len], dtype=torch.long).unsqueeze(0)
