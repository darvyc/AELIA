from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .norm import RMSNorm


class ComputeUtilityPredictor(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128, feature_dim: int = 1) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model + feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, cheap_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.norm(h), cheap_features], dim=-1)).squeeze(-1)


@dataclass
class FLOPDualController:
    budget: float
    learning_rate: float = 1e-4
    shadow_price: float = 0.0

    def update(self, observed_cost: float) -> float:
        self.shadow_price = max(0.0, self.shadow_price + self.learning_rate * (observed_cost - self.budget))
        return self.shadow_price

    def decision(self, utility: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
        return utility > (self.shadow_price * cost)
