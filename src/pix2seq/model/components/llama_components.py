import torch
import torch.nn as nn
from torch.nn import functional as F

from model.components.attention import MultiHeadAttention


class RoPEMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with Rotary Position Embeddings."""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        is_causal=False,
        bias=False,
        dropout=0.0,
        q_max_seq_len=8192,
        kv_max_seq_len=None,
        q_rope_base=500000.0,  # Default Llama 3 RoPE theta base
        k_rope_base=None,  # Default Llama 3 RoPE theta base
    ):
        if (kv_max_seq_len is None) ^ (k_rope_base is None):
            raise ValueError(
                "Both kv_max_seq_len and k_rope_base must be set together or both be None."
            )

        super().__init__(embedding_dim, num_heads, is_causal, bias, dropout)

        # Precompute RoPE position embeddings
        self.q_max_seq_len = q_max_seq_len
        self.kv_max_seq_len = kv_max_seq_len or q_max_seq_len

        self.q_rope_base = q_rope_base
        self.k_rope_base = k_rope_base or q_rope_base

        q_cos, q_sin = precompute_rope_params(
            head_dim=self.head_dim, context_length=q_max_seq_len, theta_base=q_rope_base
        )

        if q_max_seq_len != self.kv_max_seq_len:
            k_cos, k_sin = precompute_rope_params(
                head_dim=self.head_dim,
                context_length=self.kv_max_seq_len,
                theta_base=self.k_rope_base,
            )
        else:
            k_cos, k_sin = q_cos, q_sin

        self.register_buffer("q_cos", q_cos.clone())
        self.register_buffer("q_sin", q_sin.clone())
        self.register_buffer("k_cos", k_cos.clone())
        self.register_buffer("k_sin", k_sin.clone())

    def _compute_qkv(
        self,
        x_for_q,
        x_for_kv,
        batch_size,
        kv_batch_size,
        embed_dim,
        kv_embed_dim,
        use_cache=False,
    ):
        """Override to add RoPE to Q/K."""
        # Get base Q/K/V tensors from parent
        q, k, v = super()._compute_qkv(
            x_for_q, x_for_kv, batch_size, kv_batch_size, embed_dim, kv_embed_dim
        )

        # Apply RoPE to Q and K
        if use_cache and self.kv_cache is not None:
            # During cached inference, use cache position
            start_pos = int(self.kv_cache.cache_pos)
        else:
            # During training or non-cached inference, use full sequence positions
            start_pos = 0

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        # Improved version
        positions_q = torch.arange(start_pos, start_pos + seq_len_q, device=q.device)
        positions_k = torch.arange(start_pos, start_pos + seq_len_k, device=k.device)

        q = compute_rope(q, self.q_cos[positions_q], self.q_sin[positions_q])
        k = compute_rope(k, self.k_cos[positions_k], self.k_sin[positions_k])

        return q, k, v


class SwiGLUFFN(nn.Module):
    """Llama's SwiGLU feed-forward network"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.dropout(x)
        return self.w3(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)


def precompute_rope_params(
    head_dim, theta_base=500_000, context_length=8192, freq_config=None
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )

    # Frequency adjustments
    if freq_config is not None:
        # calculate wavelengths for frequency bands
        low_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["low_freq_factor"]
        )
        high_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["high_freq_factor"]
        )

        # convert frequencies to wavelengths
        wavelen = 2 * torch.pi / inv_freq

        # handle low frequency components
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # smooth transition between frequency bands
        smooth_factor = (
            freq_config["original_context_length"] / wavelen
            - freq_config["low_freq_factor"]
        ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])

        smoothed_inv_freq = (1 - smooth_factor) * (
            inv_freq / freq_config["factor"]
        ) + smooth_factor * inv_freq

        # Apply smoothing to medium frequencies
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


def estimate_rope_theta(max_context_length, embedding_dim):
    return 10000 * (max_context_length / 512) ** (64 / embedding_dim)


class LlamaEncoderBlock(nn.Module):
    """Base Llama block with self-attention"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        context_length: int = 8192,
        rope_base: float = 500000.0,
        dropout: float = 0.0,
        bias: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()

        self.self_attn = RoPEMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            q_max_seq_len=context_length,
            q_rope_base=rope_base,
            dropout=dropout,
            bias=bias,
            is_causal=is_causal,
        )

        self.ff = SwiGLUFFN(dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout)

        self.norm1 = RMSNorm(embedding_dim)
        self.norm2 = RMSNorm(embedding_dim)

    def forward(self, x, padding_mask=None):
        # Self-attention block with residual
        shortcut = x
        x = self.norm1(x)
        x = self.self_attn(x, padding_mask=padding_mask)
        x = x + shortcut

        # FFN block with residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


class LlamaDecoderBlock(LlamaEncoderBlock):
    """Llama block with self-attention and cross-attention"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        q_max_len: int = 8192,
        q_rope_base: float = 500000.0,
        kv_max_len: int = None,
        k_rope_base: float = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        # Initialize parent with causal masking
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            context_length=q_max_len,
            rope_base=q_rope_base,
            dropout=dropout,
            bias=bias,
            is_causal=True,  # Decoder always uses causal mask
        )

        # Add cross attention components
        self.cross_attn = RoPEMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            q_max_seq_len=q_max_len,
            kv_max_seq_len=kv_max_len,
            q_rope_base=q_rope_base,
            k_rope_base=k_rope_base,
            dropout=dropout,
            bias=bias,
            is_causal=False,  # Cross attention is never causal
        )
        self.norm_cross = RMSNorm(embedding_dim)

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
        # Self attention (with optional caching)
        shortcut = x
        x = self.norm1(x)
        x = self.self_attn(x, padding_mask=padding_mask, use_cache=use_cache)
        x = x + shortcut

        # Cross attention with encoder output
        shortcut = x
        x = self.norm_cross(x)
        x = self.cross_attn(
            x, encoder_input=encoder_input, padding_mask=encoder_padding_mask
        )
        x = x + shortcut

        # FFN
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x
