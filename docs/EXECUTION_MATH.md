# AELIA execution mathematics

This document specifies the operators implemented by the PyTorch core. All
equations describe real arithmetic; numerical tests specify floating-point
tolerances. `B`, `T`, `H`, `d_k`, `d_v`, `D`, `K`, and `r` denote batch size,
sequence length, recurrent heads, key width, value width, target dimension,
mixture components, and covariance rank.

## 1. Exact recurrent execution

Normalize the input independently at each token and project all tokens together:

```text
z_t     = RMSNorm(x_t)
q_t     = W_q z_t / max(||W_q z_t||_2, eps)
k_t     = W_k z_t / max(||W_k z_t||_2, eps)
v_t     = W_v z_t
alpha_t = alpha_max * sigmoid(W_alpha z_t + b_alpha)
beta_t  = sigmoid(W_beta z_t + b_beta)
D_t     = diag(alpha_t)
rho_t   = 0 at a document start, 1 otherwise

Sbar_t = D_t (rho_t S_(t-1))
e_t    = v_t - Sbar_t^T k_t
S_t    = Sbar_t + beta_t k_t e_t^T
r_t    = S_t^T q_t
y_t    = x_t + sigmoid(s) * sigmoid(W_gate z_t + b_gate) * W_out r_t
```

The time axis remains sequential only for the memory transition and read.
Normalization, input projections, output projection, and gating are batched.
This factors independent work out of the scan without approximating its
transition or truncating its gradient.

The standalone recurrent operator treats an empty chunk as a no-op and
preserves the supplied state object.

For `A_t = (I - beta_t k_t k_t^T) D_t`, the rank-one factor has eigenvalues
`1 - beta_t ||k_t||_2^2` along the key and `1` on its orthogonal complement.
Because `||k_t||_2 <= 1` and `0 <= beta_t <= 1`:

```text
||A_t||_2 <= alpha_max < 1
||S_t - S'_t||_F <= rho_t * alpha_max * ||S_(t-1) - S'_(t-1)||_F
||S_t||_F <= alpha_max^t ||S_0||_F
             + B_v * (1 - alpha_max^t) / (1 - alpha_max)
```

The last bound assumes `||v_t||_2 <= B_v`; linear write projections alone do
not supply a global input-independent `B_v`. Resets are explicit zeroing
operations, so a discarded nonfinite state cannot contaminate the next document.

Memory transition work is `O(B T H d_k d_v)`. Persistent state contains
`B H d_k d_v` values, independent of context length. Batched projections require
`O(B T H (d_k + d_v))` temporary activations in addition to ordinary residual
activations. Autograd retains the scan computation for training; constant
persistent inference state does not imply constant training activation memory.

## 2. Complete hybrid cache and document isolation

Each recurrent or predictive layer carries its memory matrix. Each attention
layer carries rotated keys, values, document IDs when packing is active, and
the next rotary position for each sequence. K/V storage uses `kv_heads`.

For an incoming chunk of length `T` after a cache of length `P`, local query
index `i` and stored key index `j` may interact exactly when:

```text
allowed(b, i, j) = (j <= P + i) and (doc_query(b, i) == doc_key(b, j))
score(i, j)      = q_i^T k_j / sqrt(d_h) if allowed, -infinity otherwise
attention(i)    = sum_j softmax_j(score(i, j)) v_j
```

A true `reset_mask[b, i]` begins a document before token `i`, increments its
document ID, resets its rotary position to zero, and clears every recurrent
layer. For chunk-local index `i`, incoming next position `p_b`, and reset set `R`:

```text
last_start(b, i) = max({-p_b} union {j in R_b : j <= i})
position(b, i)   = i - last_start(b, i)
next_position_b = position(b, T-1) + 1
```

The implementation evaluates `last_start` with a cumulative maximum. Position
and document metadata persist across chunks, including chunks without resets.

An uncached single-document forward uses SDPA's square causal path. A single
cached query with one document can attend all stored keys. A cached multi-token
chunk uses the explicit offset mask above: SDPA's non-square `is_causal` mask
aligns the query rows at the upper left. See the
[PyTorch SDPA contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

For fixed weights and evaluation mode, full-sequence execution, token decoding,
and chunk decoding implement the same causal function. The tests compare their
logits, and compare each packed-document suffix with its independently processed
document. Attention gradients to future tokens and preceding documents are zero.

Pass both `recurrent_states` and `attention_states` when continuing a hybrid
model, with `use_cache=True`. Each list has one entry per layer and `None` for
the other layer type. State is carried without implicit detachment; use
`torch.inference_mode()` for generation. Discard caches when model weights,
sequence ordering, or the prefix changes.

For `L_A` attention layers, K/V tensors contain
`2 B L_A H_kv T d_h` values. This implementation appends with `torch.cat` and
retains earlier documents behind the mask. Allocation and copying are linear
in cached length; preallocated buffers and document eviction are separate
systems optimizations. Compatibility with PyTorch 2.3 uses temporary K/V head
repetition at the attention call, while the persistent cache remains grouped.

`logits_to_keep=N` projects only the final `N` hidden states; zero projects all
positions. For prefill with one next-token prediction, `N=1` reduces the output
projection from `O(B T d_model vocab_size)` to `O(B d_model vocab_size)`.

## 3. Bounded low-rank covariance

The implementation uses a fixed orthonormal basis `B0` of shape `[D, s]`, with
`r <= s <= D`, and a learned orientation matrix `C` of shape `[s, r]`:

```text
G     = C^T C
eta   = eps * (1 + trace(G) / r)
L L^T = G + eta I
Q     = C L^(-T)
U     = B0 Q diag(sqrt(lambda))
lambda_j = lambda_max * sigmoid(a_lambda,j)
Sigma = diag(d) + U U^T
d_j   = sigma_min^2 + (sigma_max^2 - sigma_min^2) * sigmoid(a_sigma,j)
```

Triangular solution computes `Q` without constructing an inverse. The identity

```text
Q^T Q = I - eta L^(-1) L^(-T) <= I
||U||_2^2 <= lambda_max
sigma_min^2 I <= Sigma <= (sigma_max^2 + lambda_max) I
condition_number(Sigma) <= (sigma_max^2 + lambda_max) / sigma_min^2
```

establishes the covariance bounds. The relative regularizer remains positive
when the orientation is zero or rank deficient. Cholesky differentiation avoids
the repeated-eigenvalue restriction on eigenvector gradients documented by
[PyTorch eigh](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html);
orientation is computed with
[triangular solves](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html).

The square-root strength is evaluated as
`sqrt(lambda_max) * exp(0.5 * logsigmoid(a_lambda))`, which preserves finite
gradients at numerical sigmoid saturation. All correlated directions lie in the
span of `B0`; `basis_rank=D` permits the full target space.

## 4. Stable Gaussian likelihood

For a component with diagonal `D0 = diag(d)`, define:

```text
delta = z - mu
b     = D0^(-1/2) delta
V     = D0^(-1/2) U
M     = I + V^T V
a     = V^T b
w     = solve(M, a)

quadratic = ||b - V w||_2^2 + ||w||_2^2
          = b^T b - a^T solve(M, a)
logdet    = sum_j log(d_j) + 2 sum_j log(cholesky(M)_jj)
log p(z)  = -0.5 * (D log(2 pi) + logdet + quadratic)
```

The sum-of-squares form follows by expanding the first expression and applying
`M w = a`. It avoids cancellation between the two Woodbury terms. The identity
matrix already makes `M` positive definite, so the likelihood factors this
matrix directly. Adding arbitrary jitter to `M` while retaining the original
determinant and quadratic formulas would evaluate a different expression.

Per-component likelihood cost is `O(D r^2 + r^3)`, with `O(D r + r^2)`
working storage. The diagonal path costs `O(D)`. Both paths preserve FP64 inputs;
FP16 and BF16 inputs accumulate in FP32 with autocast disabled for this algebra.

## 5. Exact mixture weights and router gradients

The predictor retains both log probabilities and probabilities:

```text
log_pi = log_softmax(a_router / tau)
pi     = exp(log_pi)
ell_m  = log_pi_m + log p_m(z)
log q  = logsumexp_m(ell_m)
r_m    = softmax_m(ell_m)
L      = -log q / D
dL / d a_router,m = (pi_m - r_m) / (tau D)
```

Retaining `log_pi` preserves very small priors even when their exponentials
underflow. For manually constructed `MixtureParams`, weights must define a
normalized, nonnegative probability vector. A zero weight contributes
`-infinity` to log mass and exactly zero posterior mass. Probability flooring
would assign artificial mass and invalidate the exact router identity.

When supplying `log_weights` manually, they must represent the same law as
`weights`. Construct fresh parameters when permuting or splitting components
so both representations remain consistent.

## 6. Partially observed future targets

Let `I` identify observed coordinates in the frozen density space. Gaussian
marginalization is exact:

```text
q(z_I | h) = sum_m pi_m Normal(z_I ; mu_m,I,
                              diag(d_m,I) + U_m,I U_m,I^T)
```

The implementation masks missing rows of `delta` and `U` to zero and substitutes
one for their diagonal variance. These rows contribute neither quadratic nor
log-determinant terms. The normalizing constant uses `|I|`, not `D`.
Masking occurs before arithmetic, so unobserved target coordinates may be NaN.

For a minibatch, normalized NLL averages per-coordinate losses over prefixes
that contain at least one observation:

```text
valid_t = 1 if |I_t| > 0, otherwise 0
L_batch = sum_t valid_t * [-log q(z_I_t | h_t) / max(1, |I_t|)]
          / max(1, sum_t valid_t)
```

The empty marginal integrates to one. A completely unobserved batch contributes
zero loss and zero gradient. This is per-prefix weighting; a per-coordinate
global average is a distinct objective and must be selected explicitly.

`observed_mask` is boolean and broadcasts to the target tensor. It refers to
coordinates after the fixed target construction. Dense whitening may mix raw
features, so missing raw horizons require valid calibrated blocks or an explicit
observation operator before applying a coordinate mask.

## 7. Covariance-aware distribution feedback

Compute covariance with centered means:

```text
mu_bar = sum_m pi_m mu_m
C      = sum_m pi_m [diag(d_m) + U_m U_m^T
                    + (mu_m - mu_bar)(mu_m - mu_bar)^T]
```

For fixed projection rows `p_j`, the implementation computes exact marginal
variances in the projected coordinates without forming `C`:

```text
mean_j = sum_m pi_m p_j^T mu_m
var_j  = sum_m pi_m [sum_d p_j,d^2 d_m,d + ||U_m^T p_j||_2^2
                    + (p_j^T mu_m - mean_j)^2]
```

This includes both low-rank within-mode correlation and between-mode covariance.
Using only `sum_d p_j,d^2 C_dd` omits cross-coordinate correlation. Centering
also avoids subtracting large, nearly equal raw second moments.

The summary contains `2 J_cf + 2 J_projection + 2` real values: characteristic
real and imaginary parts, projected means, projected marginal variances, average
total variance, and diagonal participation ratio:

```text
average_variance = sum_d C_dd / D
diagonal_participation = (sum_d C_dd)^2 / (sum_d C_dd^2 + eps)
```

The participation ratio uses diagonal coordinates and is not a spectral effective
rank. A full spectral diagnostic uses `trace(C)^2 / trace(C^2)` and includes
off-diagonal entries. Every implemented summary entry is invariant to permutation
and exact splitting of mixture components.

## 8. Validation and measurement

The tests compare Gaussian values and derivatives with dense covariance
distributions in FP64, test missing-coordinate marginals, verify the temperature
factor in router derivatives, exercise degenerate orientations, and check FP16
and BF16 stability. Recurrent output, final state, input gradients, initial-state
gradients, and parameter gradients agree with token-by-token evaluation.

Run `python scripts/benchmark.py` to verify and time the recurrent and cached
decoding paths. [PERFORMANCE.md](PERFORMANCE.md) records measured conditions,
timings, cache storage, and the scope of those measurements.
