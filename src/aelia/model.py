from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .attention import AttentionState, CausalGroupedQueryAttention
from .config import AttentionConfig, MemoryConfig, ModelConfig, PredictiveConfig
from .feedback import DistributionSummary, PredictiveFeedback
from .ffn import SwiGLU
from .mixture import GaussianMixturePredictor, MixtureParams
from .norm import RMSNorm
from .recurrent import ContractiveDeltaMemory, RecurrentState


@dataclass
class PredictiveOutput:
    layer_index: int
    params: MixtureParams
    summary: torch.Tensor


@dataclass
class AELIAOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    predictive: list[PredictiveOutput]
    recurrent_states: list[RecurrentState | None]
    attention_states: list[AttentionState | None] = field(default_factory=list)


class _RecurrentBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.memory = ContractiveDeltaMemory(
            MemoryConfig(cfg.d_model, cfg.memory_heads, cfg.memory_d_key, cfg.memory_d_value)
        )
        self.ffn = SwiGLU(cfg.d_model, int(cfg.ffn_multiplier * cfg.d_model))

    def forward(self, x: torch.Tensor, state: RecurrentState | None, reset_mask: torch.Tensor | None):
        x, state = self.memory(x, state, reset_mask)
        return self.ffn(x), state


class _AttentionBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn = CausalGroupedQueryAttention(
            AttentionConfig(
                d_model=cfg.d_model,
                query_heads=cfg.attention_query_heads,
                kv_heads=cfg.attention_kv_heads,
                head_dim=cfg.attention_head_dim,
                dropout=cfg.dropout,
            )
        )
        self.ffn = SwiGLU(cfg.d_model, int(cfg.ffn_multiplier * cfg.d_model))

    def forward(self, x: torch.Tensor, reset_mask, state, use_cache):
        result = self.attn(x, reset_mask=reset_mask, state=state, use_cache=use_cache)
        if use_cache:
            x, state = result
        else:
            x, state = result, None
        return self.ffn(x), state


class _PredictiveBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.memory = ContractiveDeltaMemory(
            MemoryConfig(cfg.d_model, cfg.memory_heads, cfg.memory_d_key, cfg.memory_d_value)
        )
        self.predictor = GaussianMixturePredictor(
            PredictiveConfig(
                d_model=cfg.d_model,
                target_dim=cfg.target_dim,
                modes=cfg.modes,
                covariance=cfg.covariance,
                cov_rank=cfg.cov_rank,
                basis_rank=cfg.basis_rank,
                characteristic_features=cfg.characteristic_features,
            )
        )
        self.summary = DistributionSummary(self.predictor)
        self.feedback = PredictiveFeedback(
            cfg.d_model,
            self.summary.output_dim,
            predictive_layers=cfg.predictive_layers,
            c_gamma=cfg.c_gamma,
            gradient_scale_value=cfg.feedback_gradient_scale,
        )
        self.ffn = SwiGLU(cfg.d_model, int(cfg.ffn_multiplier * cfg.d_model))

    def forward(self, x: torch.Tensor, state: RecurrentState | None, reset_mask: torch.Tensor | None):
        x, state = self.memory(x, state, reset_mask)
        params = self.predictor(x)
        summary = self.summary(params)
        x = self.feedback(x, summary)
        return self.ffn(x), state, params, summary


class AELIALM(nn.Module):
    """Reference AELIA language-model core.

    Batched token projections surround an exact recurrent scan. Attention caches
    preserve grouped-query storage and document-local rotary positions.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        blocks = []
        for kind in config.layers:
            if kind == "R":
                blocks.append(_RecurrentBlock(config))
            elif kind == "P":
                blocks.append(_PredictiveBlock(config))
            else:
                blocks.append(_AttentionBlock(config))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        tokens: torch.Tensor,
        recurrent_states: list[RecurrentState | None] | None = None,
        reset_mask: torch.Tensor | None = None,
        *,
        attention_states: list[AttentionState | None] | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        return_predictive: bool = True,
    ) -> AELIAOutput:
        """Process tokens, optionally retaining every layer's state for decoding.

        Carry both state lists into the next cached call. ``logits_to_keep=0``
        projects every hidden state; a positive value projects only that suffix.
        ``reset_mask[b, t]`` begins a document before token t at every layer.
        """
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, positive time]")
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if reset_mask is not None and reset_mask.shape != tokens.shape:
            raise ValueError("reset_mask must match tokens")
        for name, states in (("recurrent_states", recurrent_states), ("attention_states", attention_states)):
            if states is not None and len(states) != len(self.blocks):
                raise ValueError(f"{name} must have one entry per layer")
        if attention_states is not None and not use_cache:
            raise ValueError("attention_states requires use_cache=True")
        if recurrent_states is not None and any(s is not None for s in recurrent_states) and "A" in self.config.layers:
            if attention_states is None or any(
                s is None for kind, s in zip(self.config.layers, attention_states, strict=True) if kind == "A"
            ):
                raise ValueError("hybrid continuation requires attention_states and use_cache=True")
        if attention_states is not None and any(s is not None for s in attention_states):
            for i, kind in enumerate(self.config.layers):
                if kind == "A" and attention_states[i] is None:
                    raise ValueError("continuation requires every attention layer's cache")
                if kind != "A" and (recurrent_states is None or recurrent_states[i] is None):
                    raise ValueError("hybrid continuation requires every recurrent layer's state")
            lengths = {s.keys.shape[-2] for s in attention_states if s is not None}
            if len(lengths) != 1:
                raise ValueError("attention caches must have the same prefix length")
        x = self.embedding(tokens)
        if recurrent_states is None:
            recurrent_states = [None] * len(self.blocks)
        if attention_states is None:
            attention_states = [None] * len(self.blocks)
        new_states: list[RecurrentState | None] = []
        new_attention: list[AttentionState | None] = []
        predictive: list[PredictiveOutput] = []
        for i, (kind, block, state) in enumerate(zip(self.config.layers, self.blocks, recurrent_states, strict=True)):
            if kind == "A" and state is not None:
                raise ValueError("attention layers require a None recurrent state")
            if kind != "A" and attention_states[i] is not None:
                raise ValueError("recurrent layers require a None attention state")
            attention_state = None
            if kind == "R":
                x, state = block(x, state, reset_mask)
                new_states.append(state)
            elif kind == "P":
                x, state, params, summary = block(x, state, reset_mask)
                if return_predictive:
                    predictive.append(PredictiveOutput(i, params, summary))
                new_states.append(state)
            else:
                x, attention_state = block(x, reset_mask, attention_states[i], use_cache)
                new_states.append(None)
            new_attention.append(attention_state)
        h = self.final_norm(x)
        logit_hidden = h[:, -logits_to_keep:] if logits_to_keep else h
        return AELIAOutput(self.lm_head(logit_hidden), h, predictive, new_states, new_attention)
