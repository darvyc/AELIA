from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        y = self.down(F.silu(self.gate(z)) * self.up(z))
        return x + torch.sigmoid(self.residual_scale) * y
