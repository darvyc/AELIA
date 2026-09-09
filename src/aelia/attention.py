from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import AttentionConfig
from .norm import RMSNorm


@dataclass
class AttentionState:
    """Unexpanded grouped-query K/V and document-local position metadata.

    Keys and values: [batch, kv_heads, cached_time, head_dim]. The cache is
    append-only; document IDs isolate packed documents without evicting tokens.
    """

    keys: torch.Tensor
    values: torch.Tensor
    next_position: torch.Tensor
    document_ids: torch.Tensor | None = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, base: float = 10000.0, positions: torch.Tensor | None = None) -> torch.Tensor:
    # x: [B, H, T, D]
    d = x.shape[-1]
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    if positions is None:
        positions = torch.arange(x.shape[-2], device=x.device)
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=dtype) / d))
    phase = positions.to(dtype).unsqueeze(-1) * inv
    if phase.ndim == 2:
        phase = phase.unsqueeze(0)
    cos = torch.repeat_interleave(phase.cos(), 2, dim=-1).to(x.dtype).unsqueeze(1)
    sin = torch.repeat_interleave(phase.sin(), 2, dim=-1).to(x.dtype).unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class CausalGroupedQueryAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, config.query_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(d, config.kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(d, config.kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.query_heads * config.head_dim, d, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        x: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        state: AttentionState | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionState]:
        """Causal attention over this chunk and an optional complete prefix cache."""
        c = self.config
        if x.ndim != 3 or x.shape[-1] != c.d_model or x.shape[1] == 0:
            raise ValueError("x must have shape [batch, positive time, d_model]")
        if state is not None and not use_cache:
            raise ValueError("state requires use_cache=True")
        b, t, _ = x.shape
        if reset_mask is not None and reset_mask.shape != (b, t):
            raise ValueError("reset_mask must have shape [batch, time]")
        z = self.norm(x)
        q = self.q_proj(z).view(b, t, c.query_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(z).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        offset = 0
        start = torch.zeros(b, device=x.device, dtype=torch.long)
        if state is not None:
            expected = (b, c.kv_heads, state.keys.shape[-2], c.head_dim)
            if state.keys.shape != expected or state.values.shape != expected or expected[2] == 0:
                raise ValueError("invalid attention K/V cache shape")
            if state.keys.device != x.device or state.values.device != x.device:
                raise ValueError("attention cache and input must be on the same device")
            if state.keys.dtype != k.dtype or state.values.dtype != v.dtype:
                raise ValueError("attention cache and projected input must have the same dtype")
            if state.next_position.shape != (b,) or state.next_position.device != x.device:
                raise ValueError("next_position must have shape [batch] on the input device")
            if state.document_ids is not None and state.document_ids.shape != (b, expected[2]):
                raise ValueError("document_ids must have shape [batch, cached_time]")
            offset = expected[2]
            start = state.next_position

        local = torch.arange(t, device=x.device)
        positions = start[:, None] + local
        document_ids = None
        if reset_mask is not None or (state is not None and state.document_ids is not None):
            reset = torch.zeros(b, t, device=x.device, dtype=torch.bool) if reset_mask is None else reset_mask.bool()
            # Before a reset, the virtual document start is -start. At each reset
            # the current local index becomes the document start.
            last_reset = torch.where(reset, local, -start[:, None]).cummax(dim=-1).values
            positions = local - last_reset
            previous_ids = torch.zeros(b, offset, device=x.device, dtype=torch.long)
            previous_doc = torch.zeros(b, 1, device=x.device, dtype=torch.long)
            if state is not None and state.document_ids is not None:
                previous_ids = state.document_ids
                previous_doc = previous_ids[:, -1:]
            current_ids = previous_doc + reset.long().cumsum(dim=-1)
            document_ids = torch.cat((previous_ids, current_ids), dim=-1)

        q = apply_rope(q, c.rope_base, positions)
        k = apply_rope(k, c.rope_base, positions)
        if state is not None:
            k = torch.cat((state.keys, k), dim=-2)
            v = torch.cat((state.values, v), dim=-2)
        next_state = AttentionState(k, v, positions[:, -1] + 1, document_ids) if use_cache else None

        mask = None
        is_causal = offset == 0
        if document_ids is not None or (offset > 0 and t > 1):
            # Cached chunk queries occupy rows offset ... offset+t-1. SDPA's
            # non-square is_causal mask is upper-left aligned, so use this offset.
            causal = torch.arange(offset + t, device=x.device)[None, :] <= (offset + local[:, None])
            mask = causal[None, None]
            if document_ids is not None:
                same_document = document_ids[:, -t:, None] == document_ids[:, None, :]
                mask = mask & same_document[:, None]
            is_causal = False
        repeat = c.query_heads // c.kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=c.dropout if self.training else 0.0, is_causal=is_causal
        )
        y = y.transpose(1, 2).reshape(b, t, c.query_heads * c.head_dim)
        output = x + torch.sigmoid(self.residual_scale) * self.o_proj(y)
        return (output, next_state) if use_cache else output
