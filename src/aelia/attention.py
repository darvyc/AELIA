from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import AttentionConfig
from .norm import RMSNorm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    # x: [B, H, T, D]
    d = x.shape[-1]
    pos = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    phase = torch.outer(pos, inv)
    cos = torch.repeat_interleave(phase.cos(), 2, dim=-1).to(x.dtype)[None, None]
    sin = torch.repeat_interleave(phase.sin(), 2, dim=-1).to(x.dtype)[None, None]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.config
        b, t, _ = x.shape
        z = self.norm(x)
        q = self.q_proj(z).view(b, t, c.query_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(z).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        q = apply_rope(q, c.rope_base)
        k = apply_rope(k, c.rope_base)
        repeat = c.query_heads // c.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=c.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, c.query_heads * c.head_dim)
        return x + torch.sigmoid(self.residual_scale) * self.o_proj(y)
