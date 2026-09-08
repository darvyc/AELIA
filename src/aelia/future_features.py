from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FutureObservationAssembler(nn.Module):
    """Concatenate calibrated per-horizon feature blocks.

    Expected inputs use the same leading dimensions and represent:
      semantic coordinate, probability sketch, entropy, top-two margin,
      and realized surprisal.
    """

    def __init__(
        self,
        lambda_semantic: float = 1.0,
        lambda_sketch: float = 1.0,
        lambda_entropy: float = 1.0,
        lambda_margin: float = 1.0,
        lambda_surprisal: float = 1.0,
    ) -> None:
        super().__init__()
        self.scales = (
            lambda_semantic,
            lambda_sketch,
            lambda_entropy,
            lambda_margin,
            lambda_surprisal,
        )

    def forward(
        self,
        semantic: torch.Tensor,
        probability_sketch: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor,
        surprisal: torch.Tensor,
    ) -> torch.Tensor:
        scalar_blocks = []
        for x in (entropy, margin, surprisal):
            scalar_blocks.append(x.unsqueeze(-1) if x.ndim == semantic.ndim - 1 else x)
        return torch.cat(
            [
                self.scales[0] * semantic,
                self.scales[1] * probability_sketch,
                self.scales[2] * scalar_blocks[0],
                self.scales[3] * scalar_blocks[1],
                self.scales[4] * scalar_blocks[2],
            ],
            dim=-1,
        )


class MultiscaleFutureProjector(nn.Module):
    """Frozen multiscale concatenation and random orthogonal projection.

    Input shape: [..., H, observation_dim]. Scale entries use zero-based horizon
    indices into the H axis.
    """

    def __init__(
        self,
        observation_dim: int,
        scales: Sequence[Sequence[int]],
        output_dims: Sequence[int],
        temporal_taus: Sequence[float] | None = None,
        seed: int = 41,
    ) -> None:
        super().__init__()
        if len(scales) != len(output_dims):
            raise ValueError("scales and output_dims must have equal length")
        if temporal_taus is None:
            temporal_taus = [max(1.0, float(len(s))) for s in scales]
        if len(temporal_taus) != len(scales):
            raise ValueError("temporal_taus and scales must have equal length")
        self.scales = [tuple(int(i) for i in scale) for scale in scales]
        self.output_dims = list(output_dims)
        gen = torch.Generator().manual_seed(seed)
        for i, (scale, out_dim, tau) in enumerate(zip(self.scales, output_dims, temporal_taus, strict=True)):
            if not scale:
                raise ValueError("scales must be non-empty")
            if out_dim < 1:
                raise ValueError("output dimensions must be positive")
            h = torch.tensor(scale, dtype=torch.float32)
            weights = torch.softmax(-(h + 1.0) / float(tau), dim=0).sqrt()
            self.register_buffer(f"weights_{i}", weights)
            in_dim = len(scale) * observation_dim
            raw = torch.randn(in_dim, min(out_dim, in_dim), generator=gen)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            projection = q.transpose(0, 1)
            if projection.shape[0] < out_dim:
                pad = torch.zeros(out_dim - projection.shape[0], in_dim)
                projection = torch.cat([projection, pad], dim=0)
            self.register_buffer(f"projection_{i}", projection)

    @property
    def output_dim(self) -> int:
        return sum(self.output_dims)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, scale in enumerate(self.scales):
            weights = getattr(self, f"weights_{i}").to(observations.dtype)
            projection = getattr(self, f"projection_{i}").to(observations.dtype)
            selected = observations[..., list(scale), :]
            selected = selected * weights.view(*([1] * (selected.ndim - 2)), -1, 1)
            flattened = selected.flatten(-2)
            outputs.append(torch.einsum("od,...d->...o", projection, flattened))
        return torch.cat(outputs, dim=-1)

