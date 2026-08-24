from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import CausalGroupedQueryAttention
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

    def forward(self, x: torch.Tensor):
        return self.ffn(self.attn(x))


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

    This implementation favors mathematical transparency and testability. It is not
    fused or kernel-optimized for large-scale training.
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
    ) -> AELIAOutput:
        x = self.embedding(tokens)
        if recurrent_states is None:
            recurrent_states = [None] * len(self.blocks)
        new_states: list[RecurrentState | None] = []
        predictive: list[PredictiveOutput] = []
        for i, (kind, block, state) in enumerate(zip(self.config.layers, self.blocks, recurrent_states)):
            if kind == "R":
                x, state = block(x, state, reset_mask)
                new_states.append(state)
            elif kind == "P":
                x, state, params, summary = block(x, state, reset_mask)
                predictive.append(PredictiveOutput(i, params, summary))
                new_states.append(state)
            else:
                x = block(x)
                new_states.append(None)
        h = self.final_norm(x)
        return AELIAOutput(self.lm_head(h), h, predictive, new_states)
