from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


class SameSupervisionDeterministicControl(nn.Module):
    """Directly predicts the same teacher distribution embedding used by AELIA."""

    def __init__(self, d_model: int, summary_dim: int, hidden_dim: int | None = None, gamma_max: float = 0.05) -> None:
        super().__init__()
        hidden = hidden_dim or max(d_model, summary_dim)
        self.h_norm = RMSNorm(d_model)
        self.s_norm = RMSNorm(summary_dim)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, summary_dim),
        )
        self.feedback = nn.Linear(summary_dim, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1)
        self.logit_strength = nn.Parameter(torch.tensor(-4.0))
        self.gamma_max = gamma_max

    def predict_summary(self, h: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.h_norm(h))

    def regression_loss(self, h: torch.Tensor, teacher_summary: torch.Tensor) -> torch.Tensor:
        pred = self.predict_summary(h)
        return F.mse_loss(pred, teacher_summary.detach())

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        summary = self.predict_summary(h)
        gate = torch.sigmoid(self.gate(self.h_norm(h)))
        gamma = self.gamma_max * torch.sigmoid(self.logit_strength)
        return h + gamma * gate * self.feedback(self.s_norm(summary)), summary
