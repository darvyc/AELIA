from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryConfig:
    d_model: int
    heads: int = 8
    d_key: int = 64
    d_value: int = 64
    alpha_max: float = 0.995
    eps: float = 1e-6

    def validate(self) -> None:
        if self.d_model <= 0 or self.heads <= 0:
            raise ValueError("d_model and heads must be positive")
        if self.d_key <= 0 or self.d_value <= 0:
            raise ValueError("d_key and d_value must be positive")
        if not 0.0 < self.alpha_max < 1.0:
            raise ValueError("alpha_max must be strictly between 0 and 1")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")


@dataclass(frozen=True)
class PredictiveConfig:
    d_model: int
    target_dim: int = 192
    modes: int = 4
    context_dim: int = 256
    feature_dim: int = 256
    mode_dim: int = 64
    covariance: str = "diagonal"
    cov_rank: int = 4
    basis_rank: int = 16
    sigma_min: float = 0.10
    sigma_max: float = 3.00
    lambda_max: float = 2.00
    mixture_temperature: float = 1.0
    characteristic_features: int = 48
    characteristic_scale: float = 1.0
    mean_rms_max: float | None = 4.0
    eps: float = 1e-6

    def validate(self) -> None:
        if min(self.d_model, self.context_dim, self.feature_dim, self.mode_dim) < 1:
            raise ValueError("model and feature dimensions must be positive")
        if self.modes < 1:
            raise ValueError("modes must be at least 1")
        if self.target_dim < 1:
            raise ValueError("target_dim must be positive")
        if self.covariance not in {"diagonal", "diag_lowrank"}:
            raise ValueError("covariance must be 'diagonal' or 'diag_lowrank'")
        if not 0.0 < self.sigma_min < self.sigma_max or not math.isfinite(self.sigma_max):
            raise ValueError("Require 0 < sigma_min < sigma_max")
        if self.covariance == "diag_lowrank":
            if not 1 <= self.cov_rank <= self.basis_rank <= self.target_dim:
                raise ValueError("Require target_dim >= basis_rank >= cov_rank >= 1")
        if not math.isfinite(self.lambda_max) or self.lambda_max < 0:
            raise ValueError("lambda_max must be finite and nonnegative")
        for name in ("mixture_temperature", "characteristic_scale", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.mean_rms_max is not None and (not math.isfinite(self.mean_rms_max) or self.mean_rms_max <= 0):
            raise ValueError("mean_rms_max must be finite and positive, or None")
        if self.characteristic_features < 1:
            raise ValueError("characteristic_features must be positive")


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    rope_base: float = 10000.0
    dropout: float = 0.0

    def validate(self) -> None:
        if min(self.d_model, self.query_heads, self.kv_heads, self.head_dim) < 1:
            raise ValueError("attention dimensions must be positive")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    layers: Sequence[str] = field(default_factory=lambda: ("R", "R", "P", "A") * 3)
    ffn_multiplier: float = 4.0
    memory_heads: int = 8
    memory_d_key: int = 64
    memory_d_value: int = 64
    attention_query_heads: int = 8
    attention_kv_heads: int = 2
    attention_head_dim: int = 64
    target_dim: int = 192
    modes: int = 4
    characteristic_features: int = 48
    covariance: str = "diagonal"
    cov_rank: int = 4
    basis_rank: int = 16
    dropout: float = 0.0
    tie_embeddings: bool = True
    c_gamma: float = 0.25
    feedback_gradient_scale: float = 0.0

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if not self.layers:
            raise ValueError("layers must not be empty")
        invalid = [x for x in self.layers if x not in {"R", "P", "A"}]
        if invalid:
            raise ValueError(f"invalid layer classes: {invalid}")
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if (
            not math.isfinite(self.ffn_multiplier)
            or self.ffn_multiplier <= 0
            or int(self.ffn_multiplier * self.d_model) < 1
        ):
            raise ValueError("ffn_multiplier must be positive")

    @property
    def predictive_layers(self) -> int:
        return sum(x == "P" for x in self.layers)
