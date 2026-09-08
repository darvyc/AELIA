from __future__ import annotations

import torch
from torch import nn


class HellingerCountSketch(nn.Module):
    """Fixed CountSketch applied to sqrt probabilities.

    In expectation, squared Euclidean distance between sketches equals the squared
    Euclidean distance between square-root probability vectors, which is twice
    squared Hellinger distance.
    """

    def __init__(self, vocab_size: int, sketch_dim: int, repeats: int = 2, seed: int = 17) -> None:
        super().__init__()
        if vocab_size < 2 or sketch_dim < 1 or repeats < 1:
            raise ValueError("invalid sketch dimensions")
        gen = torch.Generator().manual_seed(seed)
        buckets = torch.randint(sketch_dim, (repeats, vocab_size), generator=gen)
        signs = torch.randint(0, 2, (repeats, vocab_size), generator=gen).mul(2).sub(1).float()
        self.vocab_size = vocab_size
        self.sketch_dim = sketch_dim
        self.repeats = repeats
        self.register_buffer("buckets", buckets)
        self.register_buffer("signs", signs)

    @property
    def output_dim(self) -> int:
        return self.repeats * self.sketch_dim

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.shape[-1] != self.vocab_size:
            raise ValueError("last dimension must equal vocab_size")
        x = probabilities.clamp_min(0.0).sqrt()
        outputs = []
        for r in range(self.repeats):
            out = torch.zeros(*x.shape[:-1], self.sketch_dim, device=x.device, dtype=x.dtype)
            index = self.buckets[r].expand(*x.shape[:-1], -1)
            out.scatter_add_(-1, index, x * self.signs[r].to(x.dtype))
            outputs.append(out)
        return torch.cat(outputs, dim=-1) / self.repeats**0.5


class FrozenWhitening(nn.Module):
    """Frozen shrinkage-whitening transform fitted from a calibration matrix."""

    def __init__(self, mean: torch.Tensor, transform: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("transform", transform)

    @classmethod
    def fit(cls, x: torch.Tensor, shrinkage: float = 0.05, eps: float = 1e-5) -> FrozenWhitening:
        if x.ndim != 2:
            raise ValueError("calibration x must have shape [samples, dim]")
        mean = x.mean(dim=0)
        xc = x - mean
        cov = xc.transpose(0, 1) @ xc / max(1, x.shape[0] - 1)
        d = cov.shape[0]
        scale = torch.trace(cov) / d
        cov = (1.0 - shrinkage) * cov + shrinkage * scale * torch.eye(d, device=x.device, dtype=x.dtype)
        evals, evecs = torch.linalg.eigh(cov + eps * torch.eye(d, device=x.device, dtype=x.dtype))
        transform = evecs @ torch.diag(evals.clamp_min(eps).rsqrt()) @ evecs.transpose(0, 1)
        return cls(mean.detach(), transform.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...d,ed->...e", x - self.mean, self.transform)

