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
||A_t||_2 = 1
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

The subtraction is essential.

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
