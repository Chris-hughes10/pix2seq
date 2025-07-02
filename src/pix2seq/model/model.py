import timm
import torch
import torch.nn as nn
from model.components.positional_encoding import SinusoidalPositionalEncoding
from model.components.transformer import (
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)
from model.inference import SequenceGenerator

from data.tokenizer import TokenProcessor


class SharedEmbeddingProjection(nn.Module):
    """Output projection using shared embedding weights."""

    def __init__(self, embedding: nn.Embedding, bias: bool = True):
        super().__init__()
        self.embedding = embedding
        if bias:
            self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project using transposed embedding weights."""
        out = torch.matmul(x, self.embedding.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class Pix2SeqModel(nn.Module):
    def __init__(
        self,
        max_seq_len,
        image_size: int = 640,
        patch_size: int = 16,
        num_encoder_layers: int = 6,
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

        # Vision Transformer encoder - matching TF configuration
        self.vit = timm.create_model(
            "vit_base_patch16_384",
            pretrained=True,
            img_size=image_size,
            patch_size=patch_size,
            num_classes=0,
            drop_path_rate=drop_path,
        )

        # Project patches to model dimension
        self.encoder_proj = nn.Linear(self.vit.embed_dim, embedding_dim)

        # Position encodings
        self.pos_embed = SinusoidalPositionalEncoding(
            d_model=embedding_dim, max_len=self.num_patches
        )

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, embedding_dim)

        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Learned decoder position embeddings as in TF
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len, embedding_dim)
        )
        nn.init.normal_(self.dec_pos_embed, std=0.02)

        # Decoder layers with cached attention
        self.transformer_decoder = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder_norm = nn.LayerNorm(embedding_dim)

        # Output projection with shared embeddings as in TF
        if shared_decoder_embedding:
            self.output_proj = SharedEmbeddingProjection(
                self.token_embedding, bias=decoder_output_bias
            )
        else:
            self.output_proj = nn.Linear(
                embedding_dim, self.vocab_size, bias=decoder_output_bias
            )

        self._trunc_normal_(self.dec_pos_embed, std=0.02)
        self.reset_parameters()

    def _trunc_normal_(self, t, mean=0.0, std=1.0):
        with torch.no_grad():
            size = t.shape
            tmp = t.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            t.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            t.data.mul_(std).add_(mean)

    def reset_parameters(self):
        """Initialize weights to match TF."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.encoder_proj.weight, std=0.02)
        nn.init.zeros_(self.encoder_proj.bias)

        if not isinstance(self.output_proj, SharedEmbeddingProjection):
            nn.init.normal_(self.output_proj.weight, std=0.02)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

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
        encoded = self.encoder_norm(encoded)

        if encoded.size(1) != self.num_patches:
            raise ValueError(
                f"Got {encoded.size(1)} patches but expected {self.num_patches}. "
                "Check image size and patch size configuration."
            )

        # Add positional embeddings
        pos_emb = self.pos_embed(encoded)  # [1,L,D]
        encoded = encoded + pos_emb  # [B,L,D]

        for encoder_block in self.transformer_encoder:
            encoded = encoder_block(encoded)

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

        seq_len = tgt.size(1)

        tgt = self.token_embedding(tgt)

        # Add positional embeddings

        # Add position embeddings for all tokens
        tgt = tgt + self.dec_pos_embed[:, :seq_len]

        decoded = tgt
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
