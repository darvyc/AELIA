from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import PredictiveConfig
from .norm import RMSNorm


@dataclass
class MixtureParams:
    weights: torch.Tensor
    means: torch.Tensor
    diag_var: torch.Tensor
    lowrank: torch.Tensor | None = None


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
        logits = self.router(c) / cfg.mixture_temperature
        weights = torch.softmax(logits.float(), dim=-1).to(h.dtype)
        means = self.base_mean(c).unsqueeze(-2) + self.mean(f)
        if cfg.mean_rms_max is not None:
            rms = means.pow(2).mean(dim=-1, keepdim=True).sqrt()
            means = means / torch.sqrt(1.0 + (rms / cfg.mean_rms_max).pow(2))
        sigma2 = cfg.sigma_min**2 + (cfg.sigma_max**2 - cfg.sigma_min**2) * torch.sigmoid(self.logit_sigma(f))

        lowrank = None
        if cfg.covariance == "diag_lowrank":
            assert self.orientation is not None and self.strength is not None
            raw_c = self.orientation(f).view(*f.shape[:-1], cfg.basis_rank, cfg.cov_rank)
            gram = raw_c.transpose(-1, -2) @ raw_c
            eye = torch.eye(cfg.cov_rank, device=h.device, dtype=h.dtype)
            evals, evecs = torch.linalg.eigh(gram + cfg.eps * eye)
            inv_sqrt = evecs @ torch.diag_embed(evals.clamp_min(cfg.eps).rsqrt()) @ evecs.transpose(-1, -2)
            q = raw_c @ inv_sqrt
            directions = torch.einsum("dr,...rk->...dk", self.basis.to(h.dtype), q)
            lam = cfg.lambda_max * torch.sigmoid(self.strength(f))
            lowrank = directions * lam.sqrt().unsqueeze(-2)
        return MixtureParams(weights=weights, means=means, diag_var=sigma2, lowrank=lowrank)

    def component_log_prob(self, z: torch.Tensor, params: MixtureParams) -> torch.Tensor:
        """Return component log densities with shape [..., K]."""
        cfg = self.config
        delta = z.unsqueeze(-2) - params.means
        d_inv = params.diag_var.reciprocal()
        quad_diag = (delta.square() * d_inv).sum(dim=-1)
        logdet_diag = params.diag_var.log().sum(dim=-1)
        if params.lowrank is None:
            quad = quad_diag
            logdet = logdet_diag
        else:
            u = params.lowrank
            weighted_u = d_inv.unsqueeze(-1) * u
            m = u.transpose(-1, -2) @ weighted_u
            eye = torch.eye(cfg.cov_rank, device=z.device, dtype=z.dtype)
            m = m + eye
            a = torch.einsum("...dr,...d->...r", weighted_u, delta)
            chol = torch.linalg.cholesky(m.float()).to(z.dtype)
            solved = torch.cholesky_solve(a.unsqueeze(-1).float(), chol.float()).squeeze(-1).to(z.dtype)
            quad = quad_diag - (a * solved).sum(dim=-1)
            logdet = logdet_diag + 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(dim=-1).to(z.dtype)
        constant = cfg.target_dim * math.log(2.0 * math.pi)
        return -0.5 * (constant + logdet + quad)

    def log_prob(self, z: torch.Tensor, params: MixtureParams | None = None) -> torch.Tensor:
        if params is None:
            raise ValueError("params are required; call predictor(h) first")
        comp = self.component_log_prob(z, params)
        weighted = torch.log(params.weights.clamp_min(self.config.eps)) + comp
        return torch.logsumexp(weighted.float(), dim=-1).to(z.dtype)

    def nll(self, z: torch.Tensor, params: MixtureParams, normalize_dim: bool = True) -> torch.Tensor:
        loss = -self.log_prob(z, params)
        if normalize_dim:
            loss = loss / self.config.target_dim
        return loss.mean()

    def responsibilities(self, z: torch.Tensor, params: MixtureParams) -> torch.Tensor:
        comp = self.component_log_prob(z, params)
        ell = torch.log(params.weights.clamp_min(self.config.eps)) + comp
        return torch.softmax(ell.float(), dim=-1).to(z.dtype)

    def characteristic(self, params: MixtureParams) -> torch.Tensor:
        """Complex-valued characteristic function evaluated at fixed frequencies.

        Output shape: [..., J], complex64/complex128.
        """
        omega = self.omega.to(params.means.dtype)
        phase = torch.einsum("jd,...kd->...kj", omega, params.means)
        diag_term = torch.einsum("jd,...kd->...kj", omega.square(), params.diag_var)
        variance = diag_term
        if params.lowrank is not None:
            proj = torch.einsum("jd,...kdr->...kjr", omega, params.lowrank)
            variance = variance + proj.square().sum(dim=-1)
        amplitude = torch.exp(-0.5 * variance)
        real = (params.weights.unsqueeze(-1) * amplitude * torch.cos(phase)).sum(dim=-2)
        imag = (params.weights.unsqueeze(-1) * amplitude * torch.sin(phase)).sum(dim=-2)
        return torch.complex(real.float(), imag.float())

    def first_two_moments(self, params: MixtureParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixture mean and diagonal of total covariance."""
        w = params.weights.unsqueeze(-1)
        mean = (w * params.means).sum(dim=-2)
        within_diag = params.diag_var
        if params.lowrank is not None:
            within_diag = within_diag + params.lowrank.square().sum(dim=-1)
        second_diag = (w * (within_diag + params.means.square())).sum(dim=-2)
        total_var_diag = second_diag - mean.square()
        return mean, total_var_diag.clamp_min(0.0)
