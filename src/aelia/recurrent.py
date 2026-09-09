from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import MemoryConfig
from .norm import RMSNorm
from .numerics import probability_dtype


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
        dtype = torch.float64 if dtype == torch.float64 else torch.float32
        return RecurrentState(torch.zeros(batch, c.heads, c.d_key, c.d_value, device=device, dtype=dtype))

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        c = self.config
        z = self.norm(x)
        leading = x.shape[:-1]
        dtype = probability_dtype(x)
        q = self.q_proj(z).view(*leading, c.heads, c.d_key).to(dtype)
        k = self.k_proj(z).view(*leading, c.heads, c.d_key).to(dtype)
        v = self.v_proj(z).view(*leading, c.heads, c.d_value).to(dtype)
        alpha_logits = self.alpha_proj(z).to(dtype)
        beta_logits = self.beta_proj(z).to(dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q, dim=-1, eps=c.eps)
            k = F.normalize(k, dim=-1, eps=c.eps)
            alpha = c.alpha_max * torch.sigmoid(alpha_logits).view(*leading, c.heads, c.d_key)
            beta = torch.sigmoid(beta_logits).view(*leading, c.heads, 1)
        return z, q, k, v, alpha, beta

    def _validate_state(self, state: RecurrentState, batch: int, device: torch.device) -> None:
        c = self.config
        if state.memory.shape != (batch, c.heads, c.d_key, c.d_value):
            raise ValueError("recurrent state must have shape [batch, heads, d_key, d_value]")
        if state.memory.device != device:
            raise ValueError("recurrent state and input must be on the same device")

    @staticmethod
    def _advance(s, q, k, v, alpha, beta, reset):
        if reset is not None:
            s = s.masked_fill(reset.bool().view(-1, 1, 1, 1), 0.0)
        s_bar = alpha.unsqueeze(-1) * s
        v_old = (s_bar * k.unsqueeze(-1)).sum(dim=-2)
        s_new = s_bar + (beta * k).unsqueeze(-1) * (v - v_old).unsqueeze(-2)
        read = (s_new * q.unsqueeze(-1)).sum(dim=-2)
        return read, s_new

    def step(
        self,
        x: torch.Tensor,
        state: RecurrentState,
        reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Process one token.

        reset is a boolean tensor [batch]. True resets recurrent memory before the token.
        """
        if x.ndim != 2 or x.shape[-1] != self.config.d_model:
            raise ValueError("x must have shape [batch, d_model]")
        self._validate_state(state, x.shape[0], x.device)
        if reset is not None and reset.shape != x.shape[:1]:
            raise ValueError("reset must have shape [batch]")
        z, q, k, v_target, alpha, beta = self._project(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            read, s_new = self._advance(state.memory.to(q.dtype), q, k, v_target, alpha, beta, reset)
        delta = self.out_proj(read.flatten(-2).to(z.dtype))
        gate = torch.sigmoid(self.gate_proj(z))
        scale = torch.sigmoid(self.residual_scale)
        return x + scale * gate * delta, RecurrentState(s_new)

    def forward(
        self,
        x: torch.Tensor,
        state: RecurrentState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Batch token-local projections around an exact causal state scan."""
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError("x must have shape [batch, time, d_model]")
        batch, time, _ = x.shape
        if reset_mask is not None and reset_mask.shape != (batch, time):
            raise ValueError("reset_mask must have shape [batch, time]")
        if state is None:
            state = self.initial_state(batch, device=x.device, dtype=x.dtype)
        self._validate_state(state, batch, x.device)
        if time == 0:
            return x, state
        z, q, k, v, alpha, beta = self._project(x)
        reads = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            memory = state.memory.to(q.dtype)
            for t in range(time):
                reset = None if reset_mask is None else reset_mask[:, t]
                read, memory = self._advance(memory, q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t], reset)
                reads.append(read)
        read = torch.stack(reads, dim=1).flatten(-2).to(z.dtype)
        delta = self.out_proj(read)
        gate = torch.sigmoid(self.gate_proj(z))
        y = x + torch.sigmoid(self.residual_scale) * gate * delta
        return y, RecurrentState(memory)

    @property
    def contraction_ceiling(self) -> float:
        return self.config.alpha_max
