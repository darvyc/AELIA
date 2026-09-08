from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import MemoryConfig
from .norm import RMSNorm


@dataclass
class RecurrentState:
    memory: torch.Tensor


class ContractiveDeltaMemory(nn.Module):
    """Decay-plus-delta associative memory with an explicit contraction ceiling.

    State layout: [batch, heads, d_key, d_value].
    """

    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h, dk, dv, d = config.heads, config.d_key, config.d_value, config.d_model
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, h * dk, bias=False)
        self.k_proj = nn.Linear(d, h * dk, bias=False)
        self.v_proj = nn.Linear(d, h * dv, bias=False)
        self.alpha_proj = nn.Linear(d, h * dk)
        self.beta_proj = nn.Linear(d, h)
        self.out_proj = nn.Linear(h * dv, d, bias=False)
        self.gate_proj = nn.Linear(d, 1)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def initial_state(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> RecurrentState:
        c = self.config
        return RecurrentState(torch.zeros(batch, c.heads, c.d_key, c.d_value, device=device, dtype=dtype))

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        c = self.config
        z = self.norm(x)
        q = self.q_proj(z).view(*x.shape[:-1], c.heads, c.d_key)
        k = self.k_proj(z).view(*x.shape[:-1], c.heads, c.d_key)
        v = self.v_proj(z).view(*x.shape[:-1], c.heads, c.d_value)
        q = F.normalize(q, dim=-1, eps=c.eps)
        k = F.normalize(k, dim=-1, eps=c.eps)
        alpha = c.alpha_max * torch.sigmoid(self.alpha_proj(z)).view(*x.shape[:-1], c.heads, c.d_key)
        beta = torch.sigmoid(self.beta_proj(z)).view(*x.shape[:-1], c.heads, 1)
        return z, q, k, v, alpha, beta

    def step(
        self,
        x: torch.Tensor,
        state: RecurrentState,
        reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Process one token.

        reset is a boolean tensor [batch]. True resets recurrent memory before the token.
        """
        s = state.memory
        if reset is not None:
            keep = (~reset.bool()).to(s.dtype).view(-1, 1, 1, 1)
            s = s * keep

        z, q, k, v_target, alpha, beta = self._project(x)
        s_bar = alpha.unsqueeze(-1) * s
        v_old = torch.einsum("bhkv,bhk->bhv", s_bar, k)
        innovation = v_target - v_old
        s_new = s_bar + beta.unsqueeze(-1) * k.unsqueeze(-1) * innovation.unsqueeze(-2)
        read = torch.einsum("bhkv,bhk->bhv", s_new, q).reshape(x.shape[0], -1)
        delta = self.out_proj(read)
        gate = torch.sigmoid(self.gate_proj(z))
        scale = torch.sigmoid(self.residual_scale)
        return x + scale * gate * delta, RecurrentState(s_new)

    def forward(
        self,
        x: torch.Tensor,
        state: RecurrentState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Process [batch, time, d_model] in causal order."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, d_model]")
        batch, time, _ = x.shape
        if state is None:
            state = self.initial_state(batch, device=x.device, dtype=x.dtype)
        if reset_mask is not None and reset_mask.shape != (batch, time):
            raise ValueError("reset_mask must have shape [batch, time]")
        if time == 0:
            return x, state
        # All input-dependent projections are independent across time.
        z, q, k, v, alpha, beta = self._project(x)
        memory = state.memory
        reads = []
        for t in range(time):
            if reset_mask is not None:
                keep = (~reset_mask[:, t].bool()).to(memory.dtype).view(batch, 1, 1, 1)
                memory = memory * keep
            decayed = alpha[:, t].unsqueeze(-1) * memory
            old = torch.einsum("bhkv,bhk->bhv", decayed, k[:, t])
            innovation = v[:, t] - old
            memory = decayed + (beta[:, t] * k[:, t]).unsqueeze(-1) * innovation.unsqueeze(-2)
            reads.append(torch.einsum("bhkv,bhk->bhv", memory, q[:, t]))
        read = torch.stack(reads, dim=1).flatten(-2)
        gate = torch.sigmoid(self.gate_proj(z))
        y = x + torch.sigmoid(self.residual_scale) * gate * self.out_proj(read)
        return y, RecurrentState(memory)

    @property
    def contraction_ceiling(self) -> float:
        return self.config.alpha_max

