from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import PredictiveConfig
from .norm import RMSNorm
from .numerics import probability_dtype


@dataclass
class MixtureParams:
    weights: torch.Tensor
    means: torch.Tensor
    diag_var: torch.Tensor
    lowrank: torch.Tensor | None = None
    log_weights: torch.Tensor | None = None


def mixture_log_weights(params: MixtureParams) -> torch.Tensor:
    """Use router log probabilities; exact zero weights have log mass -inf."""
    if params.log_weights is not None:
        return params.log_weights.to(probability_dtype(params.log_weights))
    weights = params.weights.to(probability_dtype(params.weights))
    positive = weights > 0
    # Mask before log so an absent component also has a finite zero gradient.
    return torch.where(positive, weights, torch.ones_like(weights)).log().masked_fill(~positive, -torch.inf)


class GaussianMixturePredictor(nn.Module):
    """Conditional finite Gaussian mixture over whitened future features."""

    def __init__(self, config: PredictiveConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d, cdim, fdim, mdim, k, dz = (
            config.d_model,
            config.context_dim,
            config.feature_dim,
            config.mode_dim,
            config.modes,
            config.target_dim,
        )
        self.norm = RMSNorm(d)
        self.context = nn.Linear(d, cdim, bias=False)
        self.g_proj = nn.Linear(cdim, fdim)
        self.u_proj = nn.Linear(cdim, fdim)
        self.mode_codes = nn.Parameter(torch.randn(k, mdim) / math.sqrt(mdim))
        self.mode_g = nn.Linear(mdim, fdim, bias=False)
        self.mode_u = nn.Linear(mdim, fdim, bias=False)
        self.base_mean = nn.Linear(cdim, dz)
        self.mean = nn.Linear(fdim, dz)
        self.logit_sigma = nn.Linear(fdim, dz)
        self.router = nn.Linear(cdim, k)

        if config.covariance == "diag_lowrank":
            self.orientation = nn.Linear(fdim, config.basis_rank * config.cov_rank)
            self.strength = nn.Linear(fdim, config.cov_rank)
            raw = torch.randn(dz, config.basis_rank)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            self.register_buffer("basis", q)
        else:
            self.orientation = None
            self.strength = None
            self.register_buffer("basis", torch.empty(0))

        omega = torch.randn(config.characteristic_features, dz) / config.characteristic_scale
        norm = omega.norm(dim=-1, keepdim=True).clamp_min(config.eps)
        omega = omega * torch.clamp(4.0 / norm, max=1.0)
        self.register_buffer("omega", omega)

    def _features(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.context(self.norm(h))
        g = self.g_proj(c).unsqueeze(-2) + self.mode_g(self.mode_codes)
        u = self.u_proj(c).unsqueeze(-2) + self.mode_u(self.mode_codes)
        f = F.silu(g) * u
        return c, f

    def forward(self, h: torch.Tensor) -> MixtureParams:
        cfg = self.config
        c, f = self._features(h)
        dtype = probability_dtype(h)
        # The shared feature network may use autocast. Probability parameter
        # projections and all covariance arithmetic retain FP32 or FP64.
        with torch.autocast(device_type=h.device.type, enabled=False):

            def project(layer, value):
                bias = None if layer.bias is None else layer.bias.to(dtype)
                return F.linear(value.to(dtype), layer.weight.to(dtype), bias)

            logits = project(self.router, c) / cfg.mixture_temperature
            log_weights = F.log_softmax(logits, dim=-1)
            means = project(self.base_mean, c).unsqueeze(-2) + project(self.mean, f)
            if cfg.mean_rms_max is not None:
                means = means * (1.0 + means.square().mean(dim=-1, keepdim=True) / cfg.mean_rms_max**2).rsqrt()
            sigma2 = cfg.sigma_min**2 + (cfg.sigma_max**2 - cfg.sigma_min**2) * torch.sigmoid(
                project(self.logit_sigma, f)
            )

            lowrank = None
            if cfg.covariance == "diag_lowrank":
                assert self.orientation is not None and self.strength is not None
                raw_c = project(self.orientation, f).view(*f.shape[:-1], cfg.basis_rank, cfg.cov_rank)
                gram = raw_c.transpose(-1, -2) @ raw_c
                eye = torch.eye(cfg.cov_rank, device=h.device, dtype=dtype)
                jitter = cfg.eps * (1.0 + gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True))
                chol = torch.linalg.cholesky(gram + jitter.unsqueeze(-1) * eye)
                q = torch.linalg.solve_triangular(chol, raw_c.transpose(-1, -2), upper=False).transpose(-1, -2)
                directions = self.basis.to(dtype) @ q
                # exp(logsigmoid / 2) is finite even when a strength saturates.
                root_lam = math.sqrt(cfg.lambda_max) * torch.exp(0.5 * F.logsigmoid(project(self.strength, f)))
                lowrank = directions * root_lam.unsqueeze(-2)
        return MixtureParams(log_weights.exp(), means, sigma2, lowrank, log_weights)

    def component_log_prob(
        self, z: torch.Tensor, params: MixtureParams, observed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Component log densities [..., K], marginalizing unobserved coordinates.

        ``observed_mask`` is boolean and broadcasts to z. Missing values may be
        NaN: they are removed before arithmetic and receive zero gradient.
        """
        cfg = self.config
        if z.shape[-1] != cfg.target_dim or params.means.shape[-1] != cfg.target_dim:
            raise ValueError("observation and means must end in target_dim")
        dtype = probability_dtype(z, params.means, params.diag_var, params.lowrank)
        with torch.autocast(device_type=z.device.type, enabled=False):
            observed = z.to(dtype).unsqueeze(-2)
            means, diag = params.means.to(dtype), params.diag_var.to(dtype)
            u = None if params.lowrank is None else params.lowrank.to(dtype)
            dimensions = cfg.target_dim
            if observed_mask is not None:
                if observed_mask.dtype != torch.bool:
                    raise ValueError("observed_mask must be boolean")
                mask = torch.broadcast_to(observed_mask, z.shape).unsqueeze(-2)
                dimensions = mask.sum(dim=-1).to(dtype)
                observed = observed.masked_fill(~mask, 0.0)
                means = means.masked_fill(~mask, 0.0)
                diag = diag.masked_fill(~mask, 1.0)
                if u is not None:
                    u = u.masked_fill(~mask.unsqueeze(-1), 0.0)
            delta = observed - means
            inv_std = diag.rsqrt()
            b = delta * inv_std
            logdet = diag.log().sum(dim=-1)
            if u is None:
                quad = b.square().sum(dim=-1)
            else:
                v = inv_std.unsqueeze(-1) * u
                m = v.transpose(-1, -2) @ v
                m = m + torch.eye(u.shape[-1], device=z.device, dtype=dtype)
                chol = torch.linalg.cholesky(m)
                a = v.transpose(-1, -2) @ b.unsqueeze(-1)
                solved = torch.cholesky_solve(a, chol)
                # Equivalent to b^T b - a^T M^-1 a, without subtracting two
                # potentially large, nearly equal positive numbers.
                residual = b - (v @ solved).squeeze(-1)
                quad = residual.square().sum(dim=-1) + solved.squeeze(-1).square().sum(dim=-1)
                logdet = logdet + 2.0 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            return -0.5 * (dimensions * math.log(2.0 * math.pi) + logdet + quad)

    def log_prob(
        self, z: torch.Tensor, params: MixtureParams | None = None, observed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if params is None:
            raise ValueError("params are required; call predictor(h) first")
        comp = self.component_log_prob(z, params, observed_mask)
        return torch.logsumexp(mixture_log_weights(params) + comp, dim=-1)

    def nll(
        self,
        z: torch.Tensor,
        params: MixtureParams,
        normalize_dim: bool = True,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = -self.log_prob(z, params, observed_mask)
        if observed_mask is not None:
            counts = torch.broadcast_to(observed_mask, z.shape).sum(dim=-1)
            counts = torch.broadcast_to(counts, loss.shape)
            if normalize_dim:
                loss = loss / counts.clamp_min(1)
            valid = counts > 0
            return loss.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)
        if normalize_dim:
            loss = loss / self.config.target_dim
        return loss.mean()

    def responsibilities(
        self, z: torch.Tensor, params: MixtureParams, observed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        comp = self.component_log_prob(z, params, observed_mask)
        return torch.softmax(mixture_log_weights(params) + comp, dim=-1)

    def characteristic(self, params: MixtureParams) -> torch.Tensor:
        """Complex-valued characteristic function evaluated at fixed frequencies.

        Output shape: [..., J], complex64/complex128.
        """
        dtype = probability_dtype(params.means, params.diag_var, params.lowrank, params.weights)
        with torch.autocast(device_type=params.means.device.type, enabled=False):
            omega = self.omega.to(dtype)
            phase = params.means.to(dtype) @ omega.T
            variance = params.diag_var.to(dtype) @ omega.square().T
            if params.lowrank is not None:
                proj = omega @ params.lowrank.to(dtype)
                variance = variance + proj.square().sum(dim=-1)
            amplitude = params.weights.to(dtype).unsqueeze(-1) * torch.exp(-0.5 * variance)
            real = (amplitude * torch.cos(phase)).sum(dim=-2)
            imag = (amplitude * torch.sin(phase)).sum(dim=-2)
            return torch.complex(real, imag)

    def first_two_moments(self, params: MixtureParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixture mean and diagonal of total covariance."""
        dtype = probability_dtype(params.means, params.diag_var, params.lowrank, params.weights)
        with torch.autocast(device_type=params.means.device.type, enabled=False):
            w, means = params.weights.to(dtype).unsqueeze(-1), params.means.to(dtype)
            mean = (w * means).sum(dim=-2)
            within_diag = params.diag_var.to(dtype)
            if params.lowrank is not None:
                within_diag = within_diag + params.lowrank.to(dtype).square().sum(dim=-1)
            total_var_diag = (w * (within_diag + (means - mean.unsqueeze(-2)).square())).sum(dim=-2)
            return mean, total_var_diag

    def projected_moments(self, params: MixtureParams, projection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and exact marginal variances after a linear projection [J, D]."""
        dtype = probability_dtype(params.means, params.diag_var, params.lowrank, params.weights, projection)
        with torch.autocast(device_type=params.means.device.type, enabled=False):
            p = projection.to(dtype)
            w = params.weights.to(dtype).unsqueeze(-1)
            component_mean = params.means.to(dtype) @ p.T
            mean = (w * component_mean).sum(dim=-2)
            within = params.diag_var.to(dtype) @ p.square().T
            if params.lowrank is not None:
                within = within + (p @ params.lowrank.to(dtype)).square().sum(dim=-1)
            variance = (w * (within + (component_mean - mean.unsqueeze(-2)).square())).sum(dim=-2)
            return mean, variance
