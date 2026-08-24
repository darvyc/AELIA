from __future__ import annotations

import torch

from .mixture import MixtureParams


def sample_mixture(params: MixtureParams, samples: int) -> torch.Tensor:
    """Draw samples with shape [..., samples, D]."""
    if samples < 1:
        raise ValueError("samples must be positive")
    leading = params.weights.shape[:-1]
    k = params.weights.shape[-1]
    d = params.means.shape[-1]
    flat = int(torch.tensor(leading).prod().item()) if leading else 1
    w = params.weights.reshape(flat, k).float()
    means = params.means.reshape(flat, k, d)
    diag = params.diag_var.reshape(flat, k, d)
    idx = torch.multinomial(w, samples, replacement=True)
    batch_idx = torch.arange(flat, device=w.device).unsqueeze(-1)
    mean = means[batch_idx, idx]
    var = diag[batch_idx, idx]
    x = mean + torch.randn_like(mean) * var.sqrt()
    if params.lowrank is not None:
        rank = params.lowrank.shape[-1]
        low = params.lowrank.reshape(flat, k, d, rank)[batch_idx, idx]
        eps_r = torch.randn(flat, samples, rank, device=x.device, dtype=x.dtype)
        x = x + torch.einsum("...sdr,...sr->...sd", low, eps_r)
    return x.reshape(*leading, samples, d)


def energy_score(samples: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    """Monte Carlo energy score using a cyclic independent-pair approximation."""
    first = torch.linalg.vector_norm(samples - observation.unsqueeze(-2), dim=-1).mean(dim=-1)
    paired = torch.roll(samples, shifts=1, dims=-2)
    second = 0.5 * torch.linalg.vector_norm(samples - paired, dim=-1).mean(dim=-1)
    return first - second


def variogram_score(samples: torch.Tensor, observation: torch.Tensor, power: float = 0.5) -> torch.Tensor:
    if not 0.0 < power <= 2.0:
        raise ValueError("power must be in (0, 2]")
    d = observation.shape[-1]
    i, j = torch.triu_indices(d, d, offset=1, device=observation.device)
    observed = (observation[..., i] - observation[..., j]).abs().pow(power)
    forecast = (samples[..., i] - samples[..., j]).abs().pow(power).mean(dim=-2)
    return (observed - forecast).square().mean(dim=-1)


def rbf_kernel_score(samples: torch.Tensor, observation: torch.Tensor, bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    paired = torch.roll(samples, shifts=1, dims=-2)
    d_xx = (samples - paired).square().sum(dim=-1)
    d_xz = (samples - observation.unsqueeze(-2)).square().sum(dim=-1)
    score = torch.zeros_like(d_xx.mean(dim=-1))
    for ell in bandwidths:
        if ell <= 0:
            raise ValueError("bandwidths must be positive")
        k_xx = torch.exp(-d_xx / (2.0 * ell * ell)).mean(dim=-1)
        k_xz = torch.exp(-d_xz / (2.0 * ell * ell)).mean(dim=-1)
        score = score + k_xx - 2.0 * k_xz
    return score / len(bandwidths)


def projected_pit(params: MixtureParams, observation: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Projected probability integral transforms for the complete mixture.

    directions: [J, D], preferably unit normalized.
    output: [..., J]
    """
    directions = directions.to(observation.dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    projected_obs = torch.einsum("jd,...d->...j", directions, observation)
    projected_mean = torch.einsum("jd,...kd->...kj", directions, params.means)
    projected_var = torch.einsum("jd,...kd->...kj", directions.square(), params.diag_var)
    if params.lowrank is not None:
        low_proj = torch.einsum("jd,...kdr->...kjr", directions, params.lowrank)
        projected_var = projected_var + low_proj.square().sum(dim=-1)
    z = (projected_obs.unsqueeze(-2) - projected_mean) / projected_var.clamp_min(1e-8).sqrt()
    cdf = 0.5 * (1.0 + torch.erf(z / 2.0**0.5))
    return (params.weights.unsqueeze(-1) * cdf).sum(dim=-2)


def branch_coverage_mi(responsibilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Continuation-to-mode mutual information from [..., M, K] responsibilities."""
    if responsibilities.ndim < 2:
        raise ValueError("responsibilities must end in [continuations, modes]")
    r = responsibilities.clamp_min(eps)
    r = r / r.sum(dim=-1, keepdim=True)
    mean_r = r.mean(dim=-2)
    h_mean = -(mean_r * mean_r.log()).sum(dim=-1)
    h_each = -(r * r.log()).sum(dim=-1).mean(dim=-1)
    return h_mean - h_each
