# Experimental protocol

AELIA is an empirical architecture hypothesis. Each additional mechanism is accepted only after a matched experiment establishes a statistically meaningful gain.

## 1. Primary model ladder

Use the same tokenizer, corpus, data order, optimizer family, training tokens, context length, precision policy, and target hardware class.

| ID | System | Purpose |
|---|---|---|
| B0 | Competitive attention or hybrid reference | Establish external baseline |
| B1 | Hybrid recurrent-attention backbone | Isolate backbone quality |
| B2 | B1 + MTP | Strong direct future-supervision baseline |
| B3 | B2 + deterministic future-distribution embedding control | Same-supervision deterministic control |
| A1 | B2 + AELIA `K=1`, no feedback | Test conditional-density supervision |
| A2 | B2 + AELIA `K=4`, no feedback | Test explicit multimodality |
| A3 | A2 + characteristic feedback | Test distribution reuse |
| A4 | A3 + low-rank covariance | Test correlated uncertainty |
| A5 | A4 + sparse mode routing | Test mode sparsity |
| A6 | A5 + adaptive predictive compute | Test quality per inference FLOP |

## 2. Decisive inequalities

### Future supervision beyond MTP

```text
Risk(B2 + stable future target) < Risk(B2)
```

### Probability structure

```text
Risk(AELIA) < Risk(B3)
```

B3 receives the same teacher distribution representation and training information.

### Explicit multimodality

```text
Risk(A2) < Risk(A1)
```

and held-out branch evaluation must also show:

```text
Delta_branch_multi > 0
I_branch > 0
I_mode > 0
```

### Feedback

```text
Risk(A3) < Risk(A2)
```

### Low-rank covariance

Retain only if compute-normalized predictive or LM quality improves.

### Adaptive compute

Retain only if quality per inference FLOP improves at a fixed quality or fixed compute operating point.

## 3. All-in compute matching

Report:

```text
C_total
  = C_backbone
  + C_LM_head
  + C_MTP
  + C_predictive
  + C_target
  + C_branch
  + C_counterfactual
  + C_preprocessing_amortized
```

Primary comparisons use all-in compute. Offline branch generation is amortized and still reported.

Recommended matching tolerance:

```text
abs(C_A - C_B) / C_B <= 0.02
```

where practical.

## 4. Predictive distribution evaluation

Report at every checkpoint:

- normalized predictive NLL;
- characteristic-kernel score;
- energy score;
- variogram score;
- projected PIT uniformity;
- low-dimensional Rosenblatt calibration;
- total variance;
- within-mode variance;
- between-mode variance;
- effective covariance dimension;
- posterior-prior KL;
- per-mode posterior use;
- distinguishable mode information `I_mode`;
- active mode count `K_active`;
- branch continuation-to-mode mutual information `I_branch`.

## 5. Target geometry evaluation

For continuation pairs `i,j`, measure:

```text
d_Y(i,j) = semantic or task-relevant continuation distance
d_Z(i,j) = ||Z_i - Z_j||_2
```

Report:

- global rank correlation between `d_Y` and `d_Z`;
- bad collision rate where distant continuations map close together;
- local semantic distortion among nearest neighbors in target space;
- random-projection and feature-removal controls.

A sophisticated density model cannot rescue a target geometry that erases distinctions relevant to language modeling.

## 6. Multimodality diagnostics

Use three independent tests.

### Held-out likelihood gain

```text
Delta_multi
  = E[L_pred,K=1 - L_pred,K>1]
```

### Continuation-to-mode information

With multiple continuations for a prefix, calculate posterior responsibilities for each continuation. Define:

```text
I_branch
  = H(mean_j r_j)
    - mean_j H(r_j)
```

### Distinguishable mode information

Estimate:

```text
I_mode = I(M ; Z | h)
```

by Monte Carlo samples from the learned mixture.

A high router entropy alone is not evidence of meaningful multimodality.

## 7. Routing evaluation

Local probes are insufficient as ground truth. On an exploration subset, rerun the downstream suffix of the model with and without the predictive action at layer `l`:

```text
Delta_true,l,t
  = L_suffix(a_l=0)
    - L_suffix(a_l=1)
```

Train the utility predictor against `Delta_true`.

Report:

- Pearson and Spearman correlation with true utility;
- counterfactual regret;
- predictive FLOPs per token;
- quality gain per predictive FLOP;
- activity by layer and token category;
- realized FLOP budget error.

## 8. Long-context evaluation

Evaluate separately:

- associative recall;
- multi-query recall;
- key collision;
- interference-heavy retrieval;
- repeated-entity tracking;
- long-document QA;
- long-form code completion;
- context extrapolation.

Recurrent-memory gains and predictive-density gains are reported separately.

## 9. Statistical protocol

Use paired seeds when shapes permit. For configuration `i` and seed `s`:

```text
Delta_s
  = Loss_baseline,s
    - Loss_AELIA,s
```

Report:

- mean paired improvement;
- standard error;
- bootstrap confidence interval;
- prespecified practical threshold `delta_min`.

Architecture superiority requires the lower confidence bound to exceed `delta_min`.

## 10. Scaling protocol

Use at least five distinct compute levels, preferably six to eight, with multiple seeds at each level.

Fit the same functional form over the same compute interval for every architecture:

```text
L(C) = L_inf + A C^(-alpha) + error
```

Primary evidence remains direct iso-compute performance. Scaling fits are secondary summaries, not substitutes for matched comparisons.

## 11. Failure flags

Automatically flag sustained windows of:

- posterior mode collapse;
- duplicate active modes;
- covariance floor saturation;
- covariance ceiling saturation;
- low-rank strength saturation;
- predictive residual domination;
- persistent auxiliary-gradient conflict;
- PIT or Rosenblatt calibration failure;
- low routing-utility correlation;
- `K_dist` near 1 on branch-diverse contexts;
- high router entropy with low distinguishable mode information;
- compute routing concentrated on trivial token classes without measured utility.

## 12. Decision gate

No mechanism advances because it is mathematically interesting. It advances only when the corresponding matched experiment clears its quality, compute, calibration, and statistical thresholds.


## Recurrent execution measurement

`scripts/benchmark_recurrent.py` measures the batched projection sequence
operator against repeated calls to the same module's `step` operator.
Both paths include normalization, projections, recurrence and residual output.
Output and final-state agreement are checked before timing.

Observed CPU measurement (PyTorch 2.14.0+cpu, float32, one thread,
batch 4, sequence 128, model width 128, four heads, key/value width 16):

| Execution | Median latency | Tokens/s |
| --- | ---: | ---: |
| Batched projection sequence | 8.439 ms | 60,668 |
| Repeated single-token step | 22.742 ms | 22,513 |

The latency ratio is 2.695. Measurements use `torch.utils.benchmark.Timer`
with at least one second of autorange per path and inference without gradients.
This is a local operator measurement, not a GPU, training, full-model or
language-quality result. Hardware and sequence shape affect the ratio.
