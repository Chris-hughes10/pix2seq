import torch.nn as nn
from model.components.attention import MultiHeadAttention
from model.components.mlp import MLP
from model.components.normalization import LayerNorm


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dim_feedforward,
        bias=False,
        dropout=0.0,
        is_causal=False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(embedding_dim, bias=bias)
        self.self_attn = MultiHeadAttention(
            embedding_dim, num_heads, is_causal=is_causal, bias=bias, dropout=dropout
        )

        self.ln_2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = MLP(embedding_dim, dim_feedforward, bias, dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.self_attn(self.ln_1(x), padding_mask=padding_mask, use_cache=False)

        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerDecoderBlock(TransformerEncoderBlock):
    def __init__(
        self, embedding_dim, num_heads, dim_feedforward, bias=False, dropout=0.0
    ):
        super().__init__(
            embedding_dim, num_heads, dim_feedforward, bias, dropout, is_causal=True
        )
        self.ln_cross = LayerNorm(embedding_dim, bias=bias)
        self.cross_attn = MultiHeadAttention(
            embedding_dim, num_heads, is_causal=False, bias=bias, dropout=dropout
        )

    def create_cache(self, max_seq_len: int = 1024):
        """Create a cache for self attention if it doesn't exist."""
        if self.training:
            raise RuntimeError("Cache should not be created during training")
        self.self_attn.create_cache(max_seq_len)

    def reset_cache(self):
        """Reset the cache if it exists."""
        if hasattr(self.self_attn, "kv_cache") and self.self_attn.kv_cache is not None:
            self.self_attn.kv_cache.reset()

    def forward(
        self,
        x,
        encoder_input,
        padding_mask=None,
        encoder_padding_mask=None,
        use_cache=False,
    ):
        x = x + self.self_attn(
            self.ln_1(x), padding_mask=padding_mask, use_cache=use_cache
        )

        # Cross attention with encoder output
        x = x + self.cross_attn(
            self.ln_cross(x),
            encoder_input=encoder_input,
            padding_mask=encoder_padding_mask,
        )

        x = x + self.mlp(self.ln_2(x))
        return x
