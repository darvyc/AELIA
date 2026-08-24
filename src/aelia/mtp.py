from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


class MultiTokenPrediction(nn.Module):
    def __init__(self, d_model: int, horizons: int = 4, rank: int = 128) -> None:
        super().__init__()
        self.horizons = horizons
        self.norm = RMSNorm(d_model)
        self.down = nn.ModuleList(nn.Linear(d_model, rank, bias=False) for _ in range(horizons))
        self.up = nn.ModuleList(nn.Linear(rank, d_model, bias=False) for _ in range(horizons))
        self.out_norm = nn.ModuleList(RMSNorm(d_model) for _ in range(horizons))

    def transformed_states(self, h: torch.Tensor) -> list[torch.Tensor]:
        z = self.norm(h)
        return [self.out_norm[i](h + self.up[i](F.silu(self.down[i](z)))) for i in range(self.horizons)]

    def logits(self, h: torch.Tensor, output_weight: torch.Tensor) -> list[torch.Tensor]:
        return [F.linear(state, output_weight) for state in self.transformed_states(h)]
