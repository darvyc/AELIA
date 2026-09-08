# Mathematical invariants

This document lists properties that an AELIA implementation should preserve independently of optimization quality.

## 1. Autoregressive causality

At position `t`, every inference-time predictive quantity is measurable from `y_<=t`.

```text
h_t = H(y_<=t)
q_t = Q(h_t)
b_t = B(q_t)
```

Therefore future tokens are never inputs to the inference-time predictive branch.

## 2. Recurrent contraction ceiling

With normalized key `k_t`, scalar `beta_t` in `[0, 1]`, and channel decay `D_t` whose largest diagonal value is at most `alpha_max < 1`:

```text
A_t = I - beta_t k_t k_t^T
||A_t||_2 <= 1
||A_t D_t||_2 <= alpha_max < 1
```

The recurrence:

```text
S_t = A_t D_t S_(t-1) + beta_t k_t v_target,t^T
```

therefore satisfies:

```text
||S_t||_F
  <= alpha_max ||S_(t-1)||_F
     + beta_t ||v_target,t||_2
```

If `||v_target,t||_2 <= B_v`:

```text
||S_t||_F
  <= alpha_max^t ||S_0||_F
     + B_v * (1 - alpha_max^t) / (1 - alpha_max)
```

## 3. Strict covariance positivity

For:

```text
Sigma = D + U U^T
```

with:

```text
D >= sigma_min^2 I
```

then:

```text
Sigma >= sigma_min^2 I
lambda_min(Sigma) >= sigma_min^2
```

If additionally:

```text
D <= sigma_max^2 I
||U U^T||_2 <= lambda_max
```

then:

```text
lambda_max(Sigma) <= sigma_max^2 + lambda_max
kappa(Sigma)
  <= (sigma_max^2 + lambda_max) / sigma_min^2
```

## 4. Correct Woodbury quadratic form

For:

```text
Sigma = D + U U^T
M = I + U^T D^(-1) U
a = U^T D^(-1) delta
```

then:

```text
delta^T Sigma^(-1) delta
  = delta^T D^(-1) delta
    - a^T M^(-1) a
```

A stable equivalent evaluates a sum of nonnegative terms. With `b = M^(-1) a`:

```text
r = delta - U b
quad = sum_j r_j^2 / D_jj + sum_l b_l^2
```

Expanding gives `delta^T D^(-1) delta - 2 b^T a + b^T M b`.
Since `M b = a`, this is exactly the Woodbury quadratic. Cholesky
factorization solves the rank-sized system without a matrix inverse.
Density arithmetic uses at least float32, with float64 preserved for
high-precision evaluation and gradient checks.

The determinant is:

```text
log det(Sigma)
  = sum_j log D_jj
    + log det(M)
```

## 5. Proper mixture posterior

For component log density `log p_m(z)` and mixture weight `pi_m`:

```text
ell_m = log(pi_m) + log p_m(z)
r_m   = softmax_m(ell)
```

The negative log-likelihood is:

```text
L = -logsumexp_m(ell_m)
```

and, for ordinary softmax router logits `a_m`:

```text
dL / da_m = pi_m - r_m
```

up to any explicit scalar normalization applied to the loss.

## 6. Characteristic representation invariance

For a Gaussian component:

```text
phi_m(omega)
  = exp(
      i * omega^T mu_m
      - 0.5 * omega^T Sigma_m omega
    )
```

and mixture:

```text
phi_q(omega) = sum_m pi_m phi_m(omega)
```

This quantity is invariant to:

- component permutation;
- splitting a component into identical weighted copies;
- merging identical copies;
- any exact reparameterization preserving the same probability law.

The finite set of stored frequencies is a compressed representation. Equality at finitely many frequencies does not imply equality of arbitrary distributions.

## 7. Hellinger CountSketch expectation

Define:

```text
x(p)_v = sqrt(p_v)
```

and a signed CountSketch `S` with standard independent bucket/sign assumptions. Then:

```text
E[ <Sx(p), Sx(q)> ] = <x(p), x(q)>
```

and:

```text
E[ ||Sx(p) - Sx(q)||_2^2 ]
  = ||sqrt(p) - sqrt(q)||_2^2
  = 2 * Hellinger^2(p, q)
```

## 8. Frozen whitening coordinates

During a density-training stage:

```text
z_hat = W (z - m)
```

uses fixed `W` and `m`. Updating whitening coordinates without correspondingly transforming all mixture means and covariances changes the modeled density space.

## 9. Gradient-scaled feedback

The operator:

```text
SG_eta(x)
  = stopgrad(x)
    + eta * (x - stopgrad(x))
```

has:

```text
forward:  SG_eta(x) = x
backward: d SG_eta / dx = eta I
```

The default decisive experiment uses `eta = 0` so the density parameters are trained by density objectives rather than by the LM feedback path.

## 10. Same-supervision control

A probabilistic gain is interpretable only when the deterministic control receives the same future information. The matched control predicts the same teacher distribution embedding, uses the same feedback width and residual insertion points, and is matched on parameters and measured all-in training FLOPs.


## 11. Exact projected mixture uncertainty

For normalized weights `w_k`, means `mu_k`, and covariances
`Sigma_k = diag(d_k) + U_k U_k^T`, the law of total covariance gives:

```text
mu = sum_k w_k mu_k
Cov(Z) = sum_k w_k [Sigma_k + (mu_k - mu)(mu_k - mu)^T]
```

For each projection row `p`, compute:

```text
m_k = p mu_k
m = sum_k w_k m_k
v_k = sum_j p_j^2 d_kj + ||p U_k||_2^2
Var(p Z) = sum_k w_k [v_k + (m_k - m)^2]
```

This includes both within-component correlation and between-component
correlation without constructing a target_dim by target_dim covariance.
For J projection rows, K modes, target dimension D and rank R, work is
O(J K D (1 + R)); the projected low-rank temporary has J K R elements
per input position. Diagonal mixture variance also uses centered means
to avoid cancellation from subtracting squared large means.

## 12. Causal batched recurrent execution

All q, k, v, alpha and beta projections depend only on the input at their
own position, so their linear maps operate across batch and time together.
The memory recurrence remains sequential:

```text
S_bar,t = diag(alpha_t) S_(t-1)
e_t = v_t - S_bar,t^T k_t
S_t = S_bar,t + beta_t k_t e_t^T
o_t = S_t^T q_t
```

Output projection and residual gates operate on stacked reads. A document
reset zeros memory before its token's recurrence. Nonzero initial states,
input gradients, parameter gradients and final states follow the same
single-token equations. Empty sequences preserve the initial state.

The arithmetic order of batched matrix multiplication can affect rounding.
Projected activations require O(B T H (3 D_key + D_value + 1)) storage;
the recurrent state occupies O(B H D_key D_value). This implementation
trades projected activation storage for fewer small projection launches.
It does not implement a parallel scan or a fused GPU recurrence.
