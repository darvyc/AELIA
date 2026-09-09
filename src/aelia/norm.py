from __future__ import annotations

import torch
from torch import nn

from .numerics import probability_dtype


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.to(probability_dtype(x))
        normalized = value * value.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight.to(x.dtype) * normalized.to(x.dtype)
