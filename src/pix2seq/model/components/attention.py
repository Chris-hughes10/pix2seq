import torch
import torch.nn as nn
from model.inference import SingleLayerKVCache
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, is_causal=False, bias=False, dropout=0.0
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        # separate query, key, value projections for all heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        # output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.dropout = dropout
        self.is_causal = is_causal

        # Initialize cache as None - will be created during first inference
        self.kv_cache = None

    def create_cache(self, max_seq_len: int = 1024):
        """Create a new KV cache for this layer.

        Should only be called during inference mode."""
        if self.training:
            raise RuntimeError("KV cache should not be created during training")
        self.kv_cache = SingleLayerKVCache(
            max_seq_len=max_seq_len,
            n_heads=self.num_heads,
            head_dim=self.head_dim,  # self.n_embd // self.n_head
        )
        # Move cache to same device as model
        self.kv_cache = self.kv_cache.to(next(self.parameters()).device)
        return self.kv_cache

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
        # calculate query, key, values for all heads in batch using separate projections
        q = self.q_proj(x_for_q)
        k = self.k_proj(x_for_kv)
        v = self.v_proj(x_for_kv)

        # reshape and transpose for attention
        q = q.view(
            batch_size, x_for_q.size(1), self.num_heads, embed_dim // self.num_heads
        ).transpose(1, 2)  # [batch_dim, n_head, target_seq_len, embed_dim/n_head]
        k = k.view(
            kv_batch_size,
            x_for_kv.size(1),
            self.num_heads,
            kv_embed_dim // self.num_heads,
        ).transpose(1, 2)  # [batch_dim, n_head, seq_len, embed_dim/n_head]
        v = v.view(
            kv_batch_size,
            x_for_kv.size(1),
            self.num_heads,
            kv_embed_dim // self.num_heads,
        ).transpose(1, 2)  # [batch_dim, n_head, seq_len, embed_dim/n_head]

        return q, k, v

    def _calculate_attn_mask(
        self,
        target_seq_len,
        seq_len,
        padding_mask,
        use_cache,
        encoder_input,
        batch_size,
        device,
    ):
        attn_mask = None
        should_use_causal = self.is_causal and encoder_input is None

        if padding_mask is not None:
            # Convert padding_mask to correct shape: [batch_dim, seq_len] -> [batch_dim, ..., target_seq_len, seq_len]

            if encoder_input is not None:
                # Cross attention:
                # padding_mask is [batch_dim, seq_len] for memory sequence
                # Need [batch_dim, n_head, target_seq_len, seq_len] since query and key lengths are different
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                    1
                )  # [batch_dim, 1, 1, seq_len]
                padding_mask = padding_mask.expand(
                    batch_size, self.num_heads, target_seq_len, seq_len
                )  # [batch_dim, n_head, target_seq_len, seq_len]
                attn_mask = ~padding_mask

            else:
                # Self attention:
                if should_use_causal:
                    # Combine causal and padding masks
                    causal_mask = torch.tril(
                        torch.ones(
                            target_seq_len,
                            target_seq_len,
                            dtype=torch.bool,
                            device=padding_mask.device,
                        )
                    )
                    # padding_mask is [batch_dim, target_seq_len] for target sequence
                    # Need [B, H, target_seq_len, target_seq_len] since query and key lengths are the same
                    padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                        1
                    )  # [batch_dim, 1, 1, target_seq_len]
                    padding_mask = padding_mask.expand(
                        batch_size, self.num_heads, target_seq_len, target_seq_len
                    )  # [batch_dim, n_head, target_seq_len, target_seq_len]
                    combined_mask = (
                        causal_mask.view(1, 1, target_seq_len, target_seq_len)
                        & ~padding_mask
                    )
                    attn_mask = combined_mask
                    should_use_causal = (
                        False  # We've incorporated causality into the mask
                    )
                else:
                    # just padding mask
                    # padding_mask is [batch_dim, target_seq_len] for target sequence
                    # Need [B, H, target_seq_len, target_seq_len] since query and key lengths are the same
                    padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                        1
                    )  # [batch_dim, 1, 1, target_seq_len]
                    padding_mask = padding_mask.expand(
                        batch_size, self.num_heads, target_seq_len, target_seq_len
                    )  # [batch_dim, n_head, target_seq_len, target_seq_len]
                    attn_mask = ~padding_mask

        elif use_cache and should_use_causal:
            # self attention has issues with causal mask when length of q less than k/v (e.g. when using cache)
            # https://github.com/pytorch/pytorch/issues/144858
            # overcome this by generating our own causal mask
            attn_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            )[-target_seq_len:, :]
            should_use_causal = False

        return attn_mask, should_use_causal

    def forward(self, x, encoder_input=None, padding_mask=None, use_cache=False):
        batch_size, seq_len, embed_dim = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # Never use cache during training
        use_cache = use_cache and not self.training and self.is_causal

        # Handle key/value inputs
        key_value_input = encoder_input if encoder_input is not None else x
        kv_batch_size, seq_len, kv_embed_dim = (
            key_value_input.size()
        )  # seq_len is source sequence length

        if use_cache:
            # When using cache, we only need to process the last token
            # but allow full sequence input for API consistency
            last_token = x[:, -1:, :]
            x_for_kv = (
                last_token if encoder_input is None else key_value_input
            )  # Only compute K,V for last token
            x_for_q = last_token  # Only need Q for last token

        else:
            x_for_kv = key_value_input
            x_for_q = x

        q, k, v = self._compute_qkv(
            x_for_q,
            x_for_kv,
            batch_size,
            kv_batch_size,
            embed_dim,
            kv_embed_dim,
            use_cache=use_cache,
        )

        if use_cache and self.kv_cache is not None:
            if self.kv_cache.cache_pos == 0:
                # First step, just cache the k/v
                (
                    k,
                    v,
                ) = self.kv_cache.update(k, v)

            else:
                # Get cached states
                k, v = self.kv_cache.update(k[:, :, -1:], v[:, :, -1:])

        attn_mask = None
        should_use_causal = self.is_causal and encoder_input is None

        target_seq_len = q.size(2)  # Target sequence length
        seq_len = k.size(2)  # Source sequence length

        attn_mask, should_use_causal = self._calculate_attn_mask(
            target_seq_len,
            seq_len,
            padding_mask,
            use_cache,
            encoder_input,
            batch_size,
            k.device,
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=should_use_causal,  # Will only be True if we have no mask at all
        )

        y = (
            y.transpose(1, 2).contiguous().view(batch_size, x_for_q.size(1), embed_dim)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y
