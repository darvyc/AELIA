from __future__ import annotations

import torch
import torch.nn.functional as F

from .mixture import GaussianMixturePredictor, MixtureParams


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ignore_index)


def predictive_nll(
    predictor: GaussianMixturePredictor,
    params: MixtureParams,
    target: torch.Tensor,
    observed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return predictor.nll(target, params, normalize_dim=True, observed_mask=observed_mask)


def characteristic_matching_loss(
    predicted: torch.Tensor, target_samples: torch.Tensor, omega: torch.Tensor
) -> torch.Tensor:
    """Match predicted characteristic values to an empirical continuation law.

    predicted: complex tensor [..., J]
    target_samples: [..., M, D]
    omega: [J, D]
    """
    phase = torch.einsum("jd,...md->...mj", omega.to(target_samples.dtype), target_samples)
    empirical = torch.complex(torch.cos(phase).mean(dim=-2).float(), torch.sin(phase).mean(dim=-2).float())
    return (predicted - empirical).abs().square().mean()


def dead_component_penalty(responsibilities: torch.Tensor, minimum_use: float = 0.02) -> torch.Tensor:
    use = responsibilities.reshape(-1, responsibilities.shape[-1]).mean(dim=0)
    return torch.relu(minimum_use - use).square().sum()
