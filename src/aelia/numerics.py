from __future__ import annotations

import torch


def probability_dtype(*values: torch.Tensor | None) -> torch.dtype:
    """Accumulate in FP32, preserving FP64 numerical reference inputs."""
    return torch.float64 if any(x is not None and x.dtype == torch.float64 for x in values) else torch.float32
