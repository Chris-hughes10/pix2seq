import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_embd, dim_feedforward, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, dim_feedforward, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(dim_feedforward, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
