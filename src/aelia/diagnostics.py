from __future__ import annotations

import torch

from .mixture import GaussianMixturePredictor, MixtureParams


def posterior_prior_kl(responsibilities: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r = responsibilities.clamp_min(eps)
    p = weights.clamp_min(eps)
    return (r * (r.log() - p.log())).sum(dim=-1)


def effective_diagonal_dimension(var_diag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return var_diag.sum(dim=-1).square() / (var_diag.square().sum(dim=-1) + eps)


def mode_mutual_information_mc(
    predictor: GaussianMixturePredictor,
    params: MixtureParams,
    samples: int = 64,
) -> torch.Tensor:
    """Monte Carlo estimate of I(M; Z | h) for a single leading batch shape."""
    # Flatten leading dimensions so sampling is simple.
    weights = params.weights.reshape(-1, params.weights.shape[-1])
    means = params.means.reshape(-1, params.means.shape[-2], params.means.shape[-1])
    diag = params.diag_var.reshape_as(means)
    low = None
    if params.lowrank is not None:
        low = params.lowrank.reshape(-1, *params.lowrank.shape[-3:])
    values = []
    for i in range(weights.shape[0]):
        idx = torch.multinomial(weights[i].float(), samples, replacement=True)
        eps = torch.randn(samples, means.shape[-1], device=means.device, dtype=means.dtype)
        z = means[i, idx] + eps * diag[i, idx].sqrt()
        if low is not None:
            eps_r = torch.randn(samples, low.shape[-1], device=means.device, dtype=means.dtype)
            z = z + torch.einsum("sdr,sr->sd", low[i, idx], eps_r)
        one = MixtureParams(weights[i], means[i], diag[i], None if low is None else low[i])
        comp = predictor.component_log_prob(z, one)
        log_joint_component = comp[torch.arange(samples, device=z.device), idx]
        log_mix = torch.logsumexp(torch.log(weights[i].clamp_min(1e-8)) + comp, dim=-1)
        values.append((log_joint_component - log_mix).mean())
    return torch.stack(values).reshape(params.weights.shape[:-1])


def covariance_bounds(params: MixtureParams) -> tuple[torch.Tensor, torch.Tensor]:
    lower = params.diag_var.amin(dim=-1)
    upper = params.diag_var.amax(dim=-1)
    if params.lowrank is not None:
        upper = upper + params.lowrank.square().sum(dim=(-2, -1))
    return lower, upper
