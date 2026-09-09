# Implementation status

The repository provides a transparent reference implementation of the architecture core. It is suitable for correctness tests, research integration, small-scale experiments, and kernel development.

| Component | Status | Notes |
|---|---|---|
| RMSNorm | Implemented | Pre-normalization primitive |
| Contractive delta memory | Implemented | Batched projections and exact sequential state scan with document resets |
| Recurrent contraction ceiling | Implemented | Structural `alpha_max < 1` parameterization |
| Grouped-query causal attention | Implemented | PyTorch SDPA with document-local RoPE |
| Hybrid decoding cache | Implemented | Complete recurrent and grouped K/V state; token and chunk continuation |
| Packed-document isolation | Implemented | Recurrent resets and exact attention document masks |
| Suffix-only vocabulary projection | Implemented | `logits_to_keep` selects returned token positions |
| SwiGLU FFN | Implemented | Residual-scaled reference path |
| Gaussian mixture head | Implemented | Shared mode network and bounded diagonal variance |
| Diagonal covariance | Implemented | Default first-stage model |
| Diagonal + low-rank covariance | Implemented | Fixed orthonormal basis and regularized Cholesky orientation |
| Woodbury likelihood | Implemented | Sum-of-squares form; dense FP64 value and gradient references |
| Partial target observations | Implemented | Exact Gaussian marginals and zero-weight unobserved prefixes |
| Probability precision | Implemented | FP32 accumulation, FP64 preservation, autocast isolation |
| Posterior responsibilities | Implemented | Exact untempered posterior |
| Characteristic function | Implemented | Fixed frequency bank |
| Distribution summary | Implemented | Characteristic features and exact projected marginal variances |
| Gradient-scaled feedback | Implemented | Exact forward, scaled backward |
| Hellinger CountSketch | Implemented | Fixed hash/sign maps |
| Shrinkage whitening | Implemented | Calibration fit then frozen transform |
| Same-supervision deterministic control | Implemented | Direct summary regression and matched feedback form |
| MTP transform | Implemented | Low-rank horizon transforms with shared output weight |
| Utility predictor | Implemented | Cheap causal utility regressor |
| FLOP dual controller | Implemented | Budget shadow-price primitive |
| Full teacher pipeline | Specification complete | Requires tokenizer/model training integration |
| Multiscale future feature builder | Implemented core | Observation assembler and frozen projector; teacher integration is dataset-specific |
| Natural/teacher branch sampler | Specification complete | Requires generation pipeline |
| Exact suffix counterfactual routing trainer | Specification complete | Requires end-to-end training harness |
| Entmax sparse routing | Planned experiment | Activated only after dense modes succeed |
| Fused recurrent scan | Systems work | Required for scale |
| Distributed target generation | Systems work | Required for scale |
| TP/SP/CP distributed model | Systems work | Required for scale |
| FP8 validation | Systems work | Hardware/kernel dependent |
| Large-scale checkpointing | Systems work | Training-stack dependent |

## Reference implementation philosophy

The Python implementation intentionally favors an auditable mapping from equations to code. High-throughput deployments should preserve the tested invariants while replacing sequential Python scans and unfused density operations with validated kernels.

[Execution mathematics](EXECUTION_MATH.md) specifies the implemented operators.
[Performance measurements](PERFORMANCE.md) records CPU operator and cached
decoding benchmarks with numerical agreement checks. End-to-end language-model
quality and large-scale GPU throughput remain experimental outcomes.

## What should remain invariant under optimization

Any optimized implementation should preserve:

1. document-boundary recurrent resets;
2. the recurrent contraction ceiling;
3. covariance lower bounds;
4. correct Woodbury subtraction and determinant terms;
5. untempered posterior responsibilities for diagnostics;
6. distribution-level feedback invariance;
7. frozen target coordinates during each density stage;
8. same-supervision control parity;
9. all-in compute accounting.
