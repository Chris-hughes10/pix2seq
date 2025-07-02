import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(0).unsqueeze(2)  # [1,L,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional encodings for the input shape."""
        # x: [B,L,D]
        return self.pe[:, : x.size(1)]  # Return only needed length
