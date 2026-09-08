from __future__ import annotations

import torch
from torch import nn

from .mixture import GaussianMixturePredictor, MixtureParams
from .norm import RMSNorm


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def gradient_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientScale.apply(x, scale)


class DistributionSummary(nn.Module):
    """Permutation-invariant finite characteristic sketch plus projected moments."""

    def __init__(self, predictor: GaussianMixturePredictor, projected_moments: int = 32) -> None:
        super().__init__()
        dz = predictor.config.target_dim
        self.predictor = predictor
        projected_moments = min(projected_moments, dz)
        raw = torch.randn(projected_moments, dz)
        q, _ = torch.linalg.qr(raw.transpose(0, 1), mode="reduced")
        self.register_buffer("moment_projection", q.transpose(0, 1))
        self.output_dim = 2 * predictor.config.characteristic_features + 2 * projected_moments + 2

    def forward(self, params: MixtureParams) -> torch.Tensor:
        phi = self.predictor.characteristic(params)
        mean, var_diag = self.predictor.first_two_moments(params)
        mean_proj, var_proj = self.predictor.projected_moments(params, self.moment_projection)
        u_total = var_diag.mean(dim=-1, keepdim=True)
        d_eff = var_diag.sum(dim=-1, keepdim=True).square() / (var_diag.square().sum(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([phi.real.to(mean.dtype), phi.imag.to(mean.dtype), mean_proj, var_proj, u_total, d_eff], dim=-1)


class PredictiveFeedback(nn.Module):
    def __init__(
        self,
        d_model: int,
        summary_dim: int,
        predictive_layers: int,
        c_gamma: float = 0.25,
        gradient_scale_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.summary_norm = RMSNorm(summary_dim)
        self.hidden_norm = RMSNorm(d_model)
        self.proj = nn.Linear(summary_dim, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1)
        self.logit_strength = nn.Parameter(torch.tensor(-4.0))
        self.gamma_max = c_gamma / max(1, predictive_layers)
        self.gradient_scale_value = gradient_scale_value

    def forward(self, h: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        summary = gradient_scale(summary, self.gradient_scale_value)
        d = self.proj(self.summary_norm(summary))
        gate = torch.sigmoid(self.gate(self.hidden_norm(h)))
        gamma = self.gamma_max * torch.sigmoid(self.logit_strength)
        return h + gamma * gate * d

