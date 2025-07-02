import timm
import torch

from model.components.llama_components import (
    LlamaDecoderBlock,
    LlamaEncoderBlock,
    RMSNorm,
    estimate_rope_theta,
)
from model.inference import SequenceGenerator
from model.model import SharedEmbeddingProjection
from torch import nn

from data.tokenizer import TokenProcessor


class LlamaPix2SeqDecoder(nn.Module):
    def __init__(
        self,
        max_seq_len,
        image_size: int = 640,
        patch_size: int = 14,
        num_decoder_layers: int = 6,
        embedding_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        shared_decoder_embedding: bool = True,
        decoder_output_bias: bool = True,
        bos_token_id: int = 1,  # Add BOS token ID parameter
        eos_token_id: int = 2,  # Add EOS token ID parameter
        coord_vocab_shift: int = 1000,  # Coordinate vocab shift
        base_vocab_shift: int = 10,  # Base vocab shift
        num_quantization_bins: int = 1000,  # Quantization bins
        token_processor: TokenProcessor = None,
    ):
        super().__init__()

        num_image_patches = (image_size // patch_size) ** 2
        self.num_patches = num_image_patches + 1  # +1 for CLS token
        self.eos_token_id = eos_token_id  # Store EOS token ID
        self.bos_token_id = bos_token_id
        self.coord_vocab_shift = coord_vocab_shift
        self.BASE_VOCAB_SHIFT = base_vocab_shift
        self.quantization_bins = num_quantization_bins
        self.max_seq_len = max_seq_len
        self.token_processor = token_processor
        self.vocab_size = token_processor.vocab_size

        # Store dimensions needed for KV cache
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        self.num_layers = num_decoder_layers

        self.vit = timm.create_model(
            "eva02_large_patch14_448",  # Significant upgrade from ViT-Base
            pretrained=True,
            img_size=image_size,
            patch_size=patch_size,  # EVA-02 uses 14×14 patches
            num_classes=0,
            drop_path_rate=drop_path,
        )
        num_image_patches = (image_size // patch_size) ** 2  # e.g., (640//16)**2 = 1600
        encoder_seq_len = num_image_patches + 1  # +1 for the CLS token (1601)

        # Project patches to model dimension
        self.encoder_proj = nn.Linear(self.vit.embed_dim, embedding_dim)

        # Decoder
        self.token_embedding = nn.Embedding(token_processor.vocab_size, embedding_dim)

        self.transformer_decoder = nn.ModuleList(
            [
                LlamaDecoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=self.num_heads,
                    hidden_dim=dim_feedforward,
                    dropout=dropout,
                    q_max_len=self.max_seq_len,
                    q_rope_base=estimate_rope_theta(self.max_seq_len, embedding_dim),
                    kv_max_len=encoder_seq_len,
                    k_rope_base=estimate_rope_theta(encoder_seq_len, embedding_dim),
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = RMSNorm(embedding_dim)

        # Output projection with shared embeddings as in TF
        if shared_decoder_embedding:
            self.output_proj = SharedEmbeddingProjection(
                self.token_embedding, bias=decoder_output_bias
            )
        else:
            self.output_proj = nn.Linear(
                embedding_dim, self.vocab_size, bias=decoder_output_bias
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, images, tgt, tgt_padding_mask=None):
        encoded, _ = self.encode(images)
        return self.decode(tgt, encoded, tgt_padding_mask=tgt_padding_mask)

    def encode(self, images, return_features=False):
        # Get patches
        features = self.vit.forward_features(images)  # [B,L,D]

        # Validate patch count
        if features.size(1) != self.num_patches:
            raise ValueError(
                f"Got {features.size(1)} patches but expected {self.num_patches}. "
                f"Image size: {images.size(-2)}x{images.size(-1)}, "
                f"Patch size: {self.vit.patch_embed.patch_size[0]}, "
                f"CLS token is always added by TIMM."
            )

        # Project to model dimension
        encoded = self.encoder_proj(features)  # [B,L,D]

        if encoded.size(1) != self.num_patches:
            raise ValueError(
                f"Got {encoded.size(1)} patches but expected {self.num_patches}. "
                "Check image size and patch size configuration."
            )

        if return_features:
            return encoded, features
        return encoded, None

    def decode(
        self,
        tgt,
        encoder_input,
        tgt_padding_mask=None,
        encoder_padding_mask=None,
        use_cache=False,
    ):
        if (tgt[:, 0] != self.bos_token_id).any():
            raise ValueError("First token must be BOS token")

        decoded = self.token_embedding(tgt)

        for decoder_block in self.transformer_decoder:
            decoded = decoder_block(
                decoded,
                encoder_input,
                padding_mask=tgt_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
                use_cache=use_cache,
            )

        decoded = self.decoder_norm(decoded)

        return self.output_proj(decoded)

    def infer(
        self,
        images=None,
        max_seq_len: int = 751,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.4,
    ):
        """Run inference using the improved sequence generator."""

        max_seq_len = min(max_seq_len, self.max_seq_len)

        # Create and reset caches before inference
        self.create_decoder_caches(max_seq_len)
        self.reset_decoder_caches()

        generator = SequenceGenerator(
            token_processor=self.token_processor,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_seq_len=max_seq_len,
        )

        result = generator.generate(self, images)

        self.clear_decoder_caches()

        return result

    def create_decoder_caches(self, max_seq_len: int = 1024):
        """Create caches for all decoder blocks."""
        if self.training:
            raise RuntimeError("Cache should not be created during training")
        for decoder_block in self.transformer_decoder:
            decoder_block.create_cache(max_seq_len)

    def reset_decoder_caches(self):
        """Reset caches in all decoder blocks."""
        for decoder_block in self.transformer_decoder:
            decoder_block.reset_cache()

    def clear_decoder_caches(self):
        """Remove all decoder caches."""
        for decoder_block in self.transformer_decoder:
            decoder_block.self_attn.kv_cache = None
