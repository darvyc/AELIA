**AELIA**

**Adaptive Ellipsoidal Latent Inference Architecture**

Full Mathematical and Systems Specification for Autoregressive Language Model Training

*Research architecture specification*

# 0. Abstract

AELIA is an autoregressive language-model architecture in which selected causal hidden states parameterize a conditional probability law over stable, observable, finite-horizon future features. The predictive law is trained by proper conditional-density objectives, represented through distribution-invariant characteristic features and low-order moments, and integrated into the residual stream through a bounded, gated feedback operator.

The architecture combines a hybrid recurrent-attention backbone, contractive delta-rule associative memory, periodic exact causal attention, SwiGLU channel transformations, direct multi-token prediction, a frozen future-feature teacher, shrinkage-whitening multiscale targets, finite Gaussian-mixture prediction, diagonal or structured low-rank covariance, characteristic-kernel distribution matching, and causal compute routing. Every predictive quantity used at inference is computed from the current causal state.

The central training object is the pushforward conditional law P_t^F of future continuations under a fixed future-feature map. AELIA learns q_phi(Z \| h_t) as an amortized approximation to P_t^F. Explicit multimodality is supported by multiple-continuation conditional-density distillation on selected prefixes and evaluated by held-out likelihood gain, continuation-to-mode mutual information, distinguishishable mode information, calibration, and characteristic-kernel scores.

The architecture is evaluated against controls that receive the same teacher, continuation samples, future-feature targets, feedback dimensionality, residual insertion points, parameter budget, and all-in training compute. This isolates the contribution of probabilistic density structure from the contribution of future supervision itself. The resulting system defines a falsifiable, compute-normalized architecture hypothesis.

# Contents

> 1 Scientific objective and operational hypothesis
>
> 2 Notation and causal factorization
>
> 3 Layer topology
>
> 4 Residual normalization
>
> 5 Local causal convolution
>
> 6 Contractive recurrent associative memory
>
> 7 Exact causal attention
>
> 8 Channel transformation
>
> 9 Target-network construction
>
> 10 Conditional future-feature law
>
> 11 Semantic token coordinates
>
> 12 Probability geometry and Hellinger CountSketch
>
> 13 Per-horizon future observation
>
> 14 Multiscale future target
>
> 15 Shrinkage whitening and frozen coordinates
>
> 16 Predictive context and shared mode network
>
> 17 Mixture means and covariance
>
> 18 Low-rank covariance geometry
>
> 19 Gaussian-mixture likelihood
>
> 20 Posterior responsibilities and router training
>
> 21 Multimodality information
>
> 22 Multiple-continuation density distillation
>
> 23 Characteristic-function distribution matching
>
> 24 Distribution-invariant feedback embedding
>
> 25 Finite characteristic-feature approximation
>
> 26 Controlled predictive residual
>
> 27 Density-integrity gradient partition
>
> 28 Multi-token prediction
>
> 29 Same-supervision deterministic control
>
> 30 Calibration and proper scoring
>
> 31 Target-geometry sufficiency
>
> 32 Causal adaptive predictive computation
>
> 33 Sequential marginal utility
>
> 34 FLOP-budget dual control
>
> 35 Numerical precision and stability
>
> 36 Complete training objective
>
> 37 Training curriculum
>
> 38 Mandatory ablations
>
> 39 Compute accounting
>
> 40 Memory complexity
>
> 41 Statistical evaluation
>
> 42 Scaling-law protocol
>
> 43 Failure criteria
>
> 44 First-stage configuration
>
> 45 Second-stage configuration
>
> 46 Forward-pass algorithm
>
> 47 Training-step algorithm
>
> 48 Inference algorithm
>
> 49 Formal architecture definition
>
> 50 Falsifiable claim
>
> Appendix A Parameterization summary
>
> Appendix B Diagnostic definitions
>
> Appendix C Experimental decision gates

# 1. Scientific objective and operational hypothesis

AELIA assigns a structured future-prediction task to selected intermediate states. The training hypothesis concerns finite-budget representation learning: the hidden state becomes more useful for autoregressive prediction when part of its capacity is organized around a proper conditional density over stable future features and when a compact representation of that density is reused by downstream computation.

The hypothesis is operational rather than defined by an unattainable model-class optimum. Let A denote an architecture, Train(A,D,S,P,s) the complete training algorithm under dataset D, all-in compute budget C, parameter budget P, and random seed s, and theta_hat\_(A,s) its terminal parameters. Define validation risk R(A;C,P) as the expectation over seeds and validation examples of next-token negative log-likelihood. Architecture A dominates control B when the lower confidence bound of the paired risk improvement exceeds a prespecified practical threshold delta_min.

theta_hat\_(A,s) = Train(A, D_train, C, P, s)

R(A;C,P) = E_s E\_((x,y)~D_val)\[ -log P\_(theta_hat\_(A,s))(y||x) \]

Delta\_(A,B) = R(B;C,P) - R(A;C,P)

LCB_95(Delta\_(AELIA,B)) \> delta_min

- All-in compute includes backbone training, target-network forward passes, MTP heads, predictive density evaluation, branch generation or amortized offline branch cost, whitening calibration, and counterfactual routing supervision.

- Primary comparisons are made at matched tokenizer, data order, optimizer family, context length, model width, parameter count, training tokens, and measured FLOPs.

- Paired seeds use matched initialization seeds and data-order seeds whenever architecture shapes permit.

# 2. Notation and causal factorization

Let y_1,...,y_T be tokens from vocabulary {1,...,V}. The model retains the standard autoregressive factorization. Hidden state h_t at causal position t is measurable with respect to y\_\<=t. Every inference-time predictive quantity is a deterministic function of h_t and therefore remains causal.

All objectives in this specification are minimized. Positive quantities named loss are negative log-likelihoods or penalties. Entropy and surprisal use their conventional nonnegative sign.

P_theta(y_1,...,y_T) = product\_(t=1)^T P_theta(y_t \| y\_\<t)

L_LM = -sum\_(t=1)^(T-1) log P_theta(y\_(t+1) \| y\_\<=t)

P_theta(y\_(t+1)||y\_\<=t) = softmax(E_out h_t^final)

I(Y_future ; q_t \| h_t) = 0

I(Y_future ; b_t \| h_t) = 0

# 3. Layer topology

Each residual layer belongs to one of three classes: R is a recurrent associative-memory layer, P is a recurrent associative-memory layer followed by the AELIA predictive operator, and A is an exact causal-attention layer. Every layer then applies a channel transformation. Predictive layers are sparse in depth so that exact attention, recurrent memory, and future-density modeling remain separately measurable mechanisms.

L = L_R + L_P + L_A

L_recurrent = L_R + L_P

|**Symbol** | **Function** | **Persistent state** |

| R | Contractive recurrent associative memory + FFN | Fixed-size recurrent state |
| P | Recurrent memory + predictive density + feedback + FFN | Recurrent state; predictive parameters are token-local |
| A | Exact grouped-query causal attention + FFN | KV cache linear in context |

# 4. Residual normalization

All primary sublayers are pre-normalized using RMSNorm. Normalization parameters are learned per channel.

RMS(x) = sqrt( (1/d) sum\_(j=1)^d x_j^2 + epsilon_norm )

RMSNorm_gamma(x) = gamma \* x / RMS(x)

# 5. Local causal convolution

A recurrent layer may apply a narrow depthwise causal convolution before associative memory. Convolutional state resets at logical document boundaries.

c_t = sum\_(j=0)^(w-1) a_j \* z\_(t-j)

u_t = z_t + c_t

- Typical width w is 3 or 4.

- The operator is depthwise, so multiplication by a_j is channelwise.

- Packed-sequence masks prevent state transfer across documents.

# 6. Contractive recurrent associative memory

For each recurrent head, AELIA uses a decay-plus-delta memory rule with an explicit operator-norm bound. The memory matrix S_t is in R^(d_k x d_v). Keys and queries are normalized. A channelwise decay contracts the previous memory, after which a rank-one delta update corrects the value stored at the current key.

The write target v_target,t can be a learned projection of u_t. A scalar beta_t controls the strength of the local correction. The recurrence admits a simple global bound when the decay ceiling alpha_max is strictly below one.

q_t = W_q u_t ; k_t = W_k u_t ; v_target,t = W_v u_t

q_t \<- q_t / (\|||q_t\||_2 + epsilon_q) ; k_t \<- k_t / (\|\|k_t\|\|\_2 + epsilon_k)

alpha\_(t,j) = alpha_max sigmoid(a\_(t,j)), 0 \< alpha_max \< 1

D_t = Diag(alpha_t)

beta_t = sigmoid(a_beta,t)

Sbar_t = D_t S\_(t-1)

v_old,t = Sbar_t^T k_t

e_t = v_target,t - v_old,t

S_t = Sbar_t + beta_t k_t e_t^T

S_t = (I - beta_t k_t k_t^T) D_t S\_(t-1) + beta_t k_t v_target,t^T

\|\| (I - beta_t k_t k_t^T) D_t \|\|\_2 \<= alpha_max \< 1

\|\|S_t\|\|\_F \<= alpha_max \|\|S\_(t-1)\|\|\_F + beta_t \|\|v_target,t\|\|\_2

\|\|S_t\|\|\_F \<= alpha_max^t \|\|S_0\|\|\_F + B_v (1-alpha_max^t)/(1-alpha_max)

- The contraction bound is exact under normalized k_t, beta_t in \[0,1\], and alpha_max \< 1.

- Memory reads use r_t = S_t^T q_t, concatenated across heads and projected to residual width.

- Per-layer memory residual strength is initialized near zero or at a depth-scaled value and is monitored by residual-ratio diagnostics.

# 7. Exact causal attention

Exact-access layers use grouped-query causal attention with rotary position encoding. These layers preserve high-fidelity token-to-token retrieval for events that are inefficient to compress into recurrent state.

Q = RoPE(Z W_Q) ; K = RoPE(Z W_K) ; V = Z W_V

A_h = softmax(Q_h K_h^T / sqrt(d_h) + M_causal)

O = Concat_h(A_h V_h) W_O

h = x + alpha_attn,l O

# 8. Channel transformation

After sequence mixing and predictive feedback, each layer applies a pre-normalized SwiGLU channel transformation.

n_t = RMSNorm(h_t)

FFN(n) = W_down \[ SiLU(W_gate n) \* (W_up n) \]

x\_(t,l+1) = h_t + alpha_FF,l FFN(n_t)

# 9. Target-network construction

AELIA uses a training-only target network theta_bar to define stable future features. The target network is either frozen for a full density-training stage or updated between stages. The preferred density experiment uses a frozen snapshot so that both the teacher distribution and the future coordinate system remain stationary while q_phi is optimized.

When an EMA teacher is used during an earlier representation-learning stage, the EMA update is detached from all predictive targets. Target logits are generated in FP32 before probability compression.

theta_bar\_(s+1) = beta_target theta_bar_s + (1-beta_target) theta_s

p_j^\* = softmax(ell\_(j-1)^(theta_bar) / tau_target)

- Recommended beta_target for EMA preparation: 0.999 to 0.9999.

- Density stages use a frozen target checkpoint and frozen whitening transform.

- Teacher temperature tau_target is recorded as part of the target-law definition.

# 10. Conditional future-feature law

For each prefix y\_\<=t, define a continuation random variable Y_t^+ = (Y\_(t+1),...,Y\_(t+H)). A frozen measurable map F_bar converts the prefix, continuation, and teacher outputs along that continuation into a continuous future feature Z_t. The teacher therefore induces a pushforward conditional probability law P_t^F over Z_t.

This pushforward law is the statistical object learned by the AELIA density. It gives an exact meaning to multiple plausible futures: they are distinct continuation trajectories whose images under F_bar occupy distinguishable regions of future-feature space.

Y_t^+ ~ P_bar(. \| y\_\<=t)

Z_t = F_bar(y\_\<=t, Y_t^+) in R^D_z

P_t^F(A) = P_bar( F_bar(y\_\<=t,Y_t^+) in A \| y\_\<=t )

q_phi(Z \| h_t) approx P_t^F

- Corpus continuations provide one unbiased sample from the observed data process at each eligible prefix.

- Selected prefixes receive M teacher continuations to estimate the conditional feature law directly.

- Natural multi-continuation data and teacher-generated branch data are reported as separate evaluation strata.

# 11. Semantic token coordinates

A fixed semantic token coordinate system supplies the realized-token component of future features. After backbone warmup, freeze an embedding matrix E_0. Frequency weighting controls the contribution of rare and common vocabulary items. Shrinkage covariance normalization produces stable coordinates while limiting amplification of weak embedding directions.

nu_v = f_v^a / sum_u f_u^a, a in \[0.25,0.75\]

m_E = sum_v nu_v E_0\[v\]

C_E = sum_v nu_v (E_0\[v\]-m_E)(E_0\[v\]-m_E)^T + epsilon_E I

C_E,shrink = (1-lambda_E) C_E + lambda_E (tr(C_E)/d_e) I

e_bar(v) = C_E,shrink^(-1/2) (E_0\[v\]-m_E)

c\_\*(v) = R_id e_bar(v)

c(v) = c\_\*(v) / (\|\|c\_\*(v)\|\|\_2 + epsilon_id)

# 12. Probability geometry and Hellinger CountSketch

Teacher probability distributions are compressed through the square-root probability map before sketching. This places every categorical distribution on a unit sphere and makes Euclidean distance proportional to Hellinger distance. Independent signed CountSketch maps then reduce vocabulary dimension while preserving inner products in expectation.

x(p)\_v = sqrt(p_v)

\|\|x(p)\|\|\_2^2 = 1

s(p) = (1/sqrt(R)) Concat\_(r=1)^R \[ S_r x(p) \]

E\[ \|\|s(p)-s(q)\|\|\_2^2 \] = \|\|sqrt(p)-sqrt(q)\|\|\_2^2

Hellinger^2(p,q) = 0.5 \|\|sqrt(p)-sqrt(q)\|\|\_2^2

- Each S_r uses an independent hash h_r(v) and Rademacher sign s_r(v).

- Common high-probability tokens may optionally occupy exact coordinates while only the probability tail is sketched.

- Sketch dimension is selected from measured held-out distortion rather than fixed by convention.

# 13. Per-horizon future observation

For horizon h, construct a future observation from the realized token coordinate, the teacher probability sketch, and low-dimensional uncertainty statistics. All signs follow standard information-theoretic conventions.

H\_(t,h) = -sum_v p\_(t,h)^\*(v) log(p\_(t,h)^\*(v)+epsilon_p)

M\_(t,h) = p\_(t,h)^\*(v_top1) - p\_(t,h)^\*(v_top2)

S\_(t,h) = -log(p\_(t,h)^\*(y\_(t+h)) + epsilon_p)

o\_(t,h) = Concat(lambda_id c(y\_(t+h)), lambda_sk s(p\_(t,h)^\*), lambda_H H\_(t,h), lambda_M M\_(t,h), lambda_S S\_(t,h))

lambda_g = 1 / sqrt(E\[\|\|feature_g\|\|\_2^2\] + epsilon)

# 14. Multiscale future target

Future observations are grouped into dyadic temporal windows. Concatenation preserves within-window temporal identity before a frozen dimensionality-reduction map is applied. The target therefore retains short-, medium-, and longer-horizon structure without requiring a separate density for every token offset.

W_1={1}; W_2={2,3}; W_3={4,5,6,7}; W_s={2^(s-1),...,2^s-1}

omega\_(s,h) = exp(-h/tau_s) / sum\_(r in W_s) exp(-r/tau_s)

O\_(t,s) = Concas\_(h in W_s) \[ sqrt(omega\_(s,h)) o\_(t,h) \]

z\_(t,s) = R_s O\_(t,s)

Z_t = Concat_s z\_(t,s)

- Positions with the complete required horizon are preferred for density training.

- When partial horizons are retained, the likelihood uses the exact marginal density over valid target dimensions rather than zero-filling missing coordinates.

- Projection dimension is chosen from retained variance and empirical pairwise-distortion criteria.

# 15. Shrinkage whitening and frozen coordinates

Each scale is whitened in a frozen coordinate system calibrated on a representative sample. Shrinkage limits condition number and suppresses unstable amplification of low-variance directions. Once density training begins, the whitening mean and transform remain fixed.

For a covariance estimate C_s, shrinkage interpolates between the empirical covariance and its isotropic trace average. The smallest lambda_shrink satisfying a target condition number is selected on calibration data.

m_s^\* = E_cal\[z_s\]

C_s = Cov_cal(z_s) + epsilon_z I

C_s,shrink = (1-lambda_shrink) C_s + lambda_shrink (tr(C_s)/d_s) I

W_s = C_s,shrink^(-1/2)

zhat\_(t,s) = W_s (z\_(t,s) - m_s^\*)

Zhat_t = Concat_s zhat\_(t,s)

- The calibration set is excluded from terminal validation metrics.

- Whitening matrices and means are checkpointed as immutable target metadata.

- A refresh requires an affine pushforward of every predicted mean and covariance; the default training schedule therefore freezes the coordinates.

# 16. Predictive context and shared mode network

At predictive layer l, the causal hidden state is compressed to a predictive context c_t. All mixture modes share the same parameterized prediction network and differ through learned mode codes e_m. This controls parameter growth and encourages comparable mode semantics across tokens.

c_t = W_c,l RMSNorm(h_t), c_t in R^r_p

g_t = W_g c_t ; u_t = W_u c_t

f\_(t,m) = SiLU(g_t + A_g e_m) \* (u_t + A_u e_m)

# 17. Mixture means and covariance

AELIA predicts a finite mixture of positive-definite Gaussian components in the frozen whitened target coordinates. Means share a common context-dependent base and receive a mode-specific correction. Diagonal covariance is bounded between strictly positive floors and finite ceilings.

The default first-stage model uses diagonal covariance. Structured low-rank covariance is introduced only after diagonal density prediction demonstrates a compute-normalized gain.

mu_base,t = W_base c_t

mu_raw\_(t,m) = mu_base,t + W_mu f\_(t,m)

mu\_(t,m) = mu_raw\_(t,m) / sqrt(1 + RMS(mu_raw\_(t,m))^2 / mu_max^2)

sigma\_(t,m,j)^2 = sigma_min^2 + (sigma_max^2-sigma_min^2) sigmoid(a_sigma\_(t,m,j))

D\_(t,m) = Diag(sigma\_(t,m)^2)

sigma_min^2 I \<= D\_(t,m) \<= sigma_max^2 I

# 18. Low-rank covariance geometry

When enabled, structured covariance adds a context-dependent low-rank positive-semidefinite term. A bank of global basis matrices spans a broad uncertainty subspace, while context-dependent mixing and orthonormalization generate token- and mode-specific directions. This avoids confining all correlated uncertainty to one fixed low-dimensional subspace.

V\_(t,m) = sum\_(g=1)^G a\_(t,m,g) B_g C\_(t,m,g)

a\_(t,m,:) = softmax(W_B f\_(t,m))

Q\_(t,m) = V\_(t,m) \[ V\_(t,m)^T V\_(t,m) + epsilon_Q I \]^(-1/2)

lambda\_(t,m,j) = lambda_max sigmoid(a_lambda\_(t,m,j))

U\_(t,m) = Q\_(t,m) Diag(sqrt(lambda\_(t,m)))

Sigma\_(t,m) = D\_(t,m) + U\_(t,m) U\_(t,m)^T

lambda_min(Sigma\_(t,m)) \>= sigma_min^2

lambda_max(Sigma\_(t,m)) \<= sigma_max^2 + lambda_max

kappa(Sigma\_(t,m)) \<= (sigma_max^2 + lambda_max) / sigma_min^2

- Basis-bank rank is chosen so that the concatenated basis span covers the target space as broadly as the compute budget permits.

- Low-rank strengths begin near zero.

- A temporary variance anchor discourages early saturation at covariance bounds.

# 19. Gaussian-mixture likelihood

For each component, inverse and log determinant are evaluated through the Woodbury identity and matrix determinant lemma. Only the r_cov by r_cov inner matrix requires dense factorization. All numerically sensitive operations use FP32.

Sigma = D + U U^T

M = I + U^T D^(-1) U

Sigma^(-1) = D^(-1) - D^(-1) U M^(-1) U^T D^(-1)

delta = Zhat_t - mu_m

a = U^T D^(-1) delta

Q_m = delta^T D^(-1) delta - a^T M^(-1) a

logdet(Sigma_m) = sum_j log(sigma\_(m,j)^2) + logdet(M)

log p_m = -0.5 \[ D_z log(2\*pi) + logdet(Sigma_m) + Q_m \]

ell_m = log(pi_m + epsilon_pi) + log p_m

L_pred,t = -(1/D_z) logsumexp_m(ell_m)

- For diagonal covariance, set U=0 and M=I.

- For partially valid target dimensions I_t, evaluate the Gaussian marginal using mu\_\[I_t\] and Sigma\_\[I_t,I_t\].

- Cholesky factors receive epsilon_chol I before factorization.

# 20. Posterior responsibilities and router training

The mixture router produces dense softmax priors during the primary experiment. True posterior responsibility is computed from the untempered component log posterior. The router gradient is the exact gradient of the normalized negative log-likelihood. A detached posterior preconditioner may rescale this gradient without introducing an additional statistical objective.

pi\_(t,m) = softmax_m(a_pi,t / tau_pi)

r\_(t,m) = exp(ell_m) / sum_j exp(ell_j)

d L_pred,t / d a_pi,m = (pi_m - r_m) / D_z

r_bar_m = (1/N) sum_t r\_(t,m)

L_dead = sum_m max(0, u_min-r_bar_m)^2

- L_dead is active only during early optimization and its coefficient decays to zero.

- Tempered responsibilities may initialize router optimization, while every reported posterior diagnostic uses tau_r=1.

- Sparse entmax routing is a later-stage systems optimization.

# 21. Multimodality information

Mixture entropy measures prior mode uncertainty but remains sensitive to duplication of identical components. AELIA therefore measures distinguishable multimodality by the conditional mutual information between component identity M and sampled future feature Z under the model mixture.

The primary reported statistic is I_mode itself. Effective mode count exp(I_mode) is reported as a monotone descriptive transform with Monte Carlo uncertainty intervals.

H(pi) = -sum_m pi_m log(pi_m + epsilon)

I_mode = sum_m pi_m E\_(z~p_m)\[ log p_m(z) - log q(z) \]

0 \<= I_mode \<= H(pi) \<= log K

K_dist = exp(I_mode)

I_hat_mode = (1/J_mc) sum_j \[ log p\_(m_j)(z_j) - log q(z_j) \]

- Identical components yield I_mode=0.

- Well-separated components approach I_mode=H(pi).

- Confidence intervals are obtained from independent Monte Carlo batches; exp(I_hat_mode) is treated as a transformed estimate rather than an unbiased estimator.

# 22. Multiple-continuation density distillation

On selected prefixes, draw M continuations from the frozen teacher distribution under a recorded sampling policy. Each continuation is mapped through the same frozen future-feature map. The resulting Monte Carlo cross-entropy is an unbiased estimator of the teacher future-feature cross-entropy.

When branch sampling uses a proposal distribution distinct from the target teacher law, self-normalized or exact importance weights are used when their variance is acceptable. Otherwise the branch distribution is defined and reported as its own target law.

Y_t^(j) ~ P_bar(. \| y\_\<=t), j=1,...,M

Z_t^(j) = F_bar(y\_\<=t, Y_t^(j))

L_MC,t = -(1/M) sum\_(j=1)^M log q_phi(Z_t^(j)\|h_t)

E\[L_MC,t\] = H(P_t^F, q_phi) = H(P_t^F) + KL(P_t^F \|\| q_phi)

- Branch prefixes are sampled independently of validation selection.

- Teacher temperature, top-p, truncation, and random seed are recorded.

- Natural branches and synthetic branches have separate metrics.

# 23. Characteristic-function distribution matching

Multiple-continuation samples also define an empirical characteristic function. Matching the model characteristic function to the empirical teacher characteristic function provides a decomposition-invariant distribution-level objective. When frequencies are sampled from the spectral measure of a characteristic kernel, the population objective is the squared maximum mean discrepancy associated with that kernel.

phi_hat_P(omega_j) = (1/M) sum\_(r=1)^M exp(i omega_j^T Z_t^(r))

phi_q(omega_j) = sum_m pi_m exp(i omega_j^T mu_m - 0.5 omega_j^T Sigma_m omega_j)

L_CF,t = (1/J) sum_j w_j \|phi_q(omega_j) - phi_hat_P(omega_j)\|^2

L_CF,t = (1/J) sum_j w_j \[ (Re_q-Re_P)^2 + (Im_q-Im_P)^2 \]

# 24. Distribution-invariant feedback embedding

The predictive residual receives a representation of the probability distribution itself. Characteristic features and distributional moments are invariant to mixture-component permutation, duplication into identical weighted subcomponents, merging of identical subcomponents, and any exact reparameterization that preserves q.

For fixed frequencies omega_j, the Gaussian characteristic function is analytic. The diagonal-plus-low-rank covariance permits efficient computation of omega^T Sigma omega.

theta\_(j,m) = omega_j^T mu_m

v\_(j,m) = sum_k sigma\_(m,k)^2 omega\_(j,k)^2 + \|\|U_m^T omega_j\|\|\_2^2

Re phi_q(omega_j) = sum_m pi_m exp(-0.5 v\_(j,m)) cos(theta\_(j,m))

Im phi_q(omega_j) = sum_m pi_m exp(-0.5 v\_(j,m)) sin(theta\_(j,m))

mu_bar = sum_m pi_m mu_m

C_total = sum_m pi_m \[Sigma_m + mu_m mu_m^T\] - mu_bar mu_bar^T

G_total = P_C C_total P_C^T

b_t = Concat(Re phi_q(omega_1), Im phi_q(omega_1), ..., P_mu mu_bar, vech(G_total), tr(C_total)/D_z, D_eff)

# 25. Finite characteristic-feature approximation

A finite characteristic sketch is a random-feature approximation to an infinite distribution embedding. Frequency count J is selected from a measured approximation budget. For a fixed pair of distributions, the squared characteristic discrepancy at each frequency lies in \[0,4\], yielding a simple concentration bound for an empirical average over independent frequencies.

X_j = \|phi_P(omega_j) - phi_Q(omega_j)\|^2, 0 \<= X_j \<= 4

MMD_hat_J^2 = (1/J) sum_j X_j

Pr(\|MMD_hat_J^2 - E\[X\]\| \>= epsilon) \<= 2 exp(-J epsilon^2 / 8)

J \>= 8 epsilon^(-2) log(2/delta)

- Several frequency bandwidths are used to cover local and global structure.

- Frequencies are fixed after initialization.

- Held-out convergence of the characteristic score with increasing J is reported before selecting the deployed dimension.

# 26. Controlled predictive residual

The feedback vector is normalized, projected to residual width, and gated by the current causal state. Per-layer feedback strength is depth-scaled so that the aggregate maximum additive contribution remains bounded as the number of predictive layers changes.

SG_eta(x) = stopgrad(x) + eta \[x-stopgrad(x)\]

b_feedback = SG\_(eta_fb)(b_t)

d_t = W_pred RMSNorm(b_feedback)

g_pred,t = sigmoid(w_pred^T RMSNorm(h_t) + b_pred)

gamma_l = (c_gamma/L_P) sigmoid(s_l)

Delta_pred,t = gamma_l g_pred,t d_t

h'\_t = h_t + Delta_pred,t

sum_l \|\|Delta_pred,l\|\|\_2 \<= c_gamma s_W sqrt(d_b) \|\|gamma_b\|\|\_infinity

- s_l is initialized negative so feedback begins near zero.

- The primary density experiment uses eta_fb=0.

- Residual-ratio monitoring constrains predictive feedback from dominating sequence mixing or channel computation.

# 27. Density-integrity gradient partition

Density parameters phi are primarily optimized by proper density objectives. When a weak language-model gradient is later allowed through predictive feedback, its component that locally opposes improvement of the density objective is projected away. This preserves the statistical role of mixture means, covariance, and priors while permitting limited task-aligned shaping.

g_pred_phi = grad_phi L_pred

g_LM_phi = grad_phi L_LM

g_LM_phi^safe = g_LM_phi - min(0, g_pred_phi^T g_LM_phi)/(\|\|g_pred_phi\|\|\_2^2+epsilon) g_pred_phi

g_pred_phi^T g_LM_phi^safe \>= 0

g_phi = g_pred_phi + eta_fb g_LM_phi^safe

- The same principle is applied to aggregate auxiliary gradients at selected shared hidden interfaces.

- Auxiliary norms are capped relative to the primary LM gradient.

- Gradient cosine statistics are logged by predictive layer.

# 28. Multi-token prediction

Direct MTP remains a mandatory auxiliary objective and control. Each horizon uses a low-rank transformation of h_t and the shared output embedding. MTP vocabulary projections are included in measured training FLOPs. MTP heads are removed for standard one-token decode.

r\_(t,h) = B_h RMSNorm(h_t)

u\_(t,h) = RMSNorm(h_t + A_h SiLU(r\_(t,h)))

logits\_(t,h) = E_out u\_(t,h)

L_MTP = -sum_t sum\_(h=1)^H_MTP omega_h^MTP log P_h(y\_(t+h)\|h_t)

# 29. Same-supervision deterministic control

The principal deterministic control receives exactly the same frozen teacher, branch continuations, target feature map, characteristic frequencies, feedback dimension, residual gate, insertion points, and all-in training budget. It directly regresses the teacher distribution embedding rather than parameterizing a probability density.

For each selected prefix, use branch samples to estimate the target characteristic embedding and target moments. A deterministic network D_xi predicts that vector from h_t and feeds it through the same residual operator. Parameter width is tuned until measured parameters and FLOPs match the probabilistic arm within the declared tolerance.

b_t^\* = B_hat({Z_t^(1),...,Z_t^(M)})

b_hat_t = D_xi(RMSNorm(h_t))

L_embed,t = (1/d_b) \|\|b_hat_t - stopgrad(b_t^\*)\|\|\_2^2

h'\_t = h_t + gamma_l g_t W_pred RMSNorm(b_hat_t)

\|C_AELIA-C_control\|/C_control \<= epsilon_compute

- This control isolates probability structure from future supervision.

- A second deterministic control predicts per-horizon future features directly before pooling, matching AELIA mode-network depth and width.

- The stronger of the deterministic controls is used for the principal architecture comparison.

# 30. Calibration and proper scoring

AELIA evaluates the complete predictive distribution through complementary calibration and proper-scoring diagnostics. One-dimensional projected PIT tests marginal projections; low-dimensional projected Rosenblatt transforms test conditional dependence; negative log-likelihood measures sharp probabilistic fit; characteristic-kernel score and energy score evaluate the full distribution; a variogram score emphasizes dependence errors.

F_a(s) = sum_m pi_m Phi((s-a^T mu_m)/sqrt(a^T Sigma_m a))

u_a = F_a(a^T z)

ES(Q,z) = E_X\[\|\|X-z\|\|\_2\] - 0.5 E\_(X,X')\[\|\|X-X'\|\|\_2\]

S_k(Q,z) = E\_(X,X'~Q)\[k(X,X')\] - 2 E\_(X~Q)\[k(X,z)\]

VS_p(Q,z) = sum\_(i\<j) w\_(i,j) \[ \|z_i-z_j\|^p - E\_(X~Q)\|X_i-X_j\|^p \]^2

- Projected PIT reports KS, Cramer-von Mises, histogram error, tail frequencies, mean, and variance.

- Projected Rosenblatt tests use random orthonormal projections of dimension 4 to 8 and test both marginal uniformity and inter-coordinate independence.

- Kernel score uses a mixture of RBF bandwidths selected from calibration-set pairwise distances.

# 31. Target-geometry sufficiency

The future feature map is validated independently of the density family. It must preserve language-relevant distinctions between continuations while keeping nearby future trajectories close when their linguistic content is similar. Global rank correlation is supplemented by explicit collision and local-neighborhood statistics.

d_Z(i,j) = \|\|Z_i-Z_j\|\|\_2

C_bad(epsilon_Z,delta_Y) = Pr\[d_Z(i,j)\<=epsilon_Z and d_Y(i,j)\>=delta_Y\]

D_local = (1/N) sum_i (1/k) sum\_(j in NN_k^Z(i)) d_Y(i,j)

rho_target = Spearman(d_Y, d_Z)

- d_Y combines continuation edit distance, semantic embedding distance, and task-specific future-event labels where available.

- Ablations remove semantic token coordinates, probability sketches, surprisal, and individual temporal scales.

- Projection dimensionality is increased until target-collision metrics and downstream density scores plateau.

# 32. Causal adaptive predictive computation

Adaptive execution is trained only after dense predictive computation has demonstrated value. A cheap utility model uses causal features available at the selected layer: normalized hidden state, recurrent novelty, write magnitude, read norm, local entropy estimates, and previous-layer change. It predicts the marginal reduction in final next-token loss produced by executing the predictive operator.

s_mem,t = \|\|v_target,t - S\_(t-1)^T k_t\|\|\_2^2 / d_v

u_hat\_(l,t) = R_eta(RMSNorm(h\_(l,t)), s_mem,t, causal_features\_(l,t))

# 33. Sequential marginal utility

Utility is defined with respect to the real downstream model rather than a local probe. On an exploration subset, the suffix network is evaluated with the predictive action disabled and enabled while later routing actions follow the current routing policy. This trains a policy-aware marginal value estimator.

The local decision threshold is valid for this conditional marginal utility. Interactions between predictive layers are thereby absorbed into the expectation over the current suffix policy.

z_cheap = F\_\>l(h_l ; a_l=0, a\_\>l~pi_eta)

z_full = F\_\>l(h_l+Delta_pred,l ; a_l=1, a\_\>l~pi_eta)

Delta_true,l,t = CE(LMHead(z_cheap),y\_(t+1)) - CE(LMHead(z_full),y\_(t+1))

L_utility = Huber(u_hat\_(l,t), stopgrad(Delta_true,l,t))

U_l(h_l) = E\[Delta_true,l,t \| h_l\]

a_l = 1\[ U_l(h_l) \> lambda_c c_l \]

# 34. FLOP-budget dual control

Routing is constrained by measured predictive FLOPs rather than token activity alone. The cost of an execution includes fixed context projection, active mixture modes, covariance evaluation, and characteristic feedback. A nonnegative dual variable tracks the shadow price of predictive computation.

c\_(t,l) = c_fixed,l + K_active,(t,l) c_mode,l + J K_active,(t,l) c_CF,l

C_pred = sum\_(t,l) a\_(t,l) c\_(t,l)

min E\[L_LM\] subject to E\[C_pred\] \<= B_pred

J_dual = E\[L_LM + lambda_c(C_pred-B_pred)\]

lambda_c \<- max(0, lambda_c + eta_lambda(mean_batch(C_pred)-B_pred))

- Straight-through hard routing is treated as a biased low-variance estimator and benchmarked against a stochastic policy-gradient reference on smaller experiments.

- Sparse mode routing reports K_active separately from K_dist.

- Quality is reported as loss at equal inference FLOPs and as marginal gain per predictive FLOP.

# 35. Numerical precision and stability

AELIA separates throughput-oriented matrix precision from probability-critical arithmetic. Covariance bounds, Cholesky jitter, shrinkage whitening, normalized keys, and residual scaling jointly define the numerical stability envelope.

- BF16 or validated FP8: large backbone matrix multiplications.

- FP32: mixture logits, variance logits, low-rank strengths, Woodbury inner matrices, Cholesky factors, log determinants, Mahalanobis forms, log-sum-exp, responsibilities, entmax thresholds, characteristic phases, whitening calibration, and calibration diagnostics.

- Cholesky uses M \<- M + epsilon_chol I with epsilon_chol typically 1e-5 to 1e-4.

- Global gradient norm is clipped to g_max, typically 1.

- Variance floor sigma_min and ceiling sigma_max are selected in whitened coordinates and logged with saturation fractions.

# 36. Complete training objective

The complete minimized objective uses only terms whose interpretation is explicit. Distribution matching and branch terms are activated on their eligible prefix subsets. Compute budgeting is handled by a dual constraint.

L_total = L_LM + lambda_MTP L_MTP + lambda_pred L_pred + lambda_MC L_MC + lambda_CF L_CF + lambda_embed_control L_control + lambda_dead L_dead + lambda_utility L_utility + lambda_anchor L_anchor

lambda_x(s) = r_x(s) lambda_x,max

r_x(s) = min(1, max(0,(s-s_x,start)/(s_x,ramp)))

- Control-specific losses are optimized only in their control runs.

- L_dead and L_anchor decay to zero.

- Predictive feedback begins after held-out density NLL has materially improved over initialization.

# 37. Training curriculum

AELIA is trained through gated stages so that each mechanism is admitted only after its prerequisite has demonstrated value.

- Phase 1 - Train the hybrid recurrent-attention backbone with next-token loss.

- Phase 2 - Add direct MTP and establish the strongest future-supervision baseline.

- Phase 3 - Freeze the target network; construct semantic coordinates, probability sketches, multiscale projections, and shrinkage-whitening calibration.

- Phase 4 - Train K=1 diagonal density with eta_fb=0.

- Phase 5 - Train K=4 diagonal density and multiple-continuation subsets.

- Phase 6 - Train the same-supervision deterministic future-distribution embedding control.

- Phase 7 - Enable characteristic feedback from near-zero gamma after density competence is established.

- Phase 8 - Compare diagonal and structured low-rank covariance.

- Phase 9 - Introduce sparse mode routing if dense multimodality is useful.

- Phase 10 - Train sequential utility and FLOP-budgeted adaptive execution.

- Phase 11 - Scale only configurations that pass all compute-normalized decision gates.

# 38. Mandatory ablations

| **ID** | **Configuration**                                          | **Question**                                 |
|--------|------------------------------------------------------------|----------------------------------------------|
| A0     | Hybrid backbone                                            | Backbone reference                           |
| A1     | Hybrid + MTP                                               | Direct future supervision                    |
| A2     | A1 + same-supervision deterministic distribution embedding | Future representation without density        |
| A3     | A1 + AELIA K=1, no feedback                                | Conditional density bottleneck               |
| A4     | A1 + AELIA K=4, no feedback                                | Explicit multimodality                       |
| A5     | A4 + characteristic feedback                               | Distribution reuse                           |
| A6     | A5 + branch MC density distillation                        | Direct conditional-law supervision           |
| A7     | A6 + characteristic distribution matching                  | Distribution-level teacher matching          |
| A8     | A7 + low-rank covariance                                   | Correlated uncertainty                       |
| A9     | A7 + sparse modes                                          | Mode compute sparsity                        |
| A10    | A9 + adaptive compute                                      | Token/layer compute routing                  |
| A11    | A7 without semantic token coordinates                      | Categorical target ablation                  |
| A12    | A7 without probability sketches                            | Teacher-distribution target ablation         |
| A13    | A7 with raw probability sketch                             | Hellinger geometry ablation                  |
| A14    | A7 with marginal standardization                           | Whitening ablation                           |
| A15    | A7 with weak eta_fb                                        | Density-integrity feedback-gradient ablation |

# 39. Compute accounting

Every architecture comparison uses measured all-in training FLOPs. The cost ledger includes target construction and branch generation, even when some steps are performed offline. Online throughput and all-in research cost are reported separately.

For each predictive mode, head cost includes mean outputs, variance outputs, optional low-rank orientation coefficients, covariance strengths, likelihood evaluation, and characteristic features.

C_total = C_backbone + C_LM_head + C_MTP + C_predictive + C_target + C_branch + C_counterfactual + C_preprocessing_amortized

n_mode_out = 2 D_z + r_0 r_cov + r_cov

C_mode_head approx 2 d_f n_mode_out

C_likelihood = O(D_z r_cov + r_cov^3)

C_characteristic = O(J(D_z+r_cov))

- Measured hardware counters take precedence over analytic FLOP estimates when both are available.

- Report training tokens per second, model FLOP utilization, peak memory, prefill throughput, and decode throughput.

- Iso-compute comparisons enforce a declared tolerance epsilon_compute, preferably 2 percent or tighter.

# 40. Memory complexity

Recurrent state is constant in context length, while exact-attention KV state grows linearly with context. Both are reported explicitly. The crossover point identifies the context length at which the KV cache of exact layers equals recurrent-state storage.

M_rec,values = (L_R+L_P) H d_k d_v

M_rec,bytes = M_rec,values b

M_KV,token = 2 L_A H_KV d_h b

M_total(T) approx M_rec,bytes + T M_KV,token + M_other

T_cross = M_rec,bytes / M_KV,token

# 41. Statistical evaluation

Architecture conclusions are based on repeated runs. Primary comparisons use paired seeds and matched data order when feasible. Report mean improvement, standard error, bootstrap confidence intervals, and a practical-effect threshold. Density diagnostics receive separate uncertainty intervals over examples and branch samples.

For unpaired architecture shapes, use a hierarchical model or unpaired bootstrap rather than manufacturing a paired difference.

Delta_s = L\_(baseline,s) - L\_(AELIA,s)

Delta_bar = (1/n) sum_s Delta_s

SE(Delta_bar) = sd(Delta_s)/sqrt(n)

Superior iff LCB_95(Delta_bar) \> delta_min

# 42. Scaling-law protocol

Compute scaling is evaluated over at least five distinct all-in compute levels with repeated seeds. Direct iso-compute differences are the primary evidence; parametric scaling-law fits are secondary summaries. All architectures are fit over the same compute interval.

For a power-law residual L(C)=L_inf+A C^(-alpha), the local loss slope can translate an AELIA loss gain into an approximate equivalent baseline-compute multiplier.

L(C) = L_inf + A C^(-alpha) + epsilon

dL/d log C = -alpha A C^(-alpha)

log(C_equiv/C) approx Delta_L / \[alpha A C^(-alpha)\]

- Use 5 to 8 compute levels where budget permits.

- Report confidence intervals for L_inf, A, and alpha.

- Report interpolation performance separately from extrapolation.

# 43. Failure criteria

Automated diagnostics flag structural failures before scale-up. Thresholds are calibrated by model size and target dimension, then frozen for each experimental family.

| **Failure**                   | **Primary signal**                            | **Interpretation**                                             |
|-------------------------------|-----------------------------------------------|----------------------------------------------------------------|
| Mode collapse                 | max_m r_bar_m \> 0.95 over sustained windows  | Single component absorbs posterior mass                        |
| Duplicate modes               | small pairwise symmetric KL with K_active\>1  | Computational duplication                                      |
| Covariance floor saturation   | large fraction sigma^2 near sigma_min^2       | Over-sharp density                                             |
| Covariance ceiling saturation | large fraction sigma^2 near sigma_max^2       | Underfit or unstable density                                   |
| Low-rank saturation           | lambda near lambda_max                        | Insufficient covariance parameterization or optimization issue |
| Predictive domination         | R_pred approaches mixer or FFN residual early | Feedback strength too large                                    |
| Auxiliary conflict            | persistent negative LM/auxiliary cosine       | Representation objective interference                          |
| Calibration failure           | PIT/Rosenblatt nonuniformity or dependence    | Miscalibrated predictive law                                   |
| Artificial multimodality      | high H(pi), low I_mode                        | Duplicated or overlapping modes                                |
| Branch mismatch               | I_branch near zero with diverse branches      | Modes do not track continuation alternatives                   |
| Router failure                | low correlation u_hat vs Delta_true           | Utility model uninformative                                    |
| Compute collapse              | routing dominated by trivial categories       | Budget policy learns shortcut                                  |

# 44. First-stage configuration

The first serious experiment is deliberately compact and uses only the mechanisms required to test the central probability-structure hypothesis.

| **Parameter**                 | **Recommended value**                |
|-------------------------------|--------------------------------------|
| Layers                        | 24                                   |
| Residual width d              | 2048                                 |
| Recurrent + predictive mixers | 20                                   |
| Exact-attention layers        | 4                                    |
| Predictive layers             | 4                                    |
| Recurrent heads               | 16                                   |
| d_k, d_v                      | 64 to 128                            |
| Mixture K                     | 1 and 4 matched experiments          |
| Future windows                | {1}, {2,3}, {4,5,6,7}                |
| D_z                           | 192                                  |
| Covariance                    | Diagonal                             |
| Characteristic frequencies J  | 64                                   |
| MTP horizon                   | 4                                    |
| Mode routing                  | Dense softmax                        |
| Predictive token activity     | 1.0                                  |
| eta_fb                        | 0                                    |
| Adaptive compute              | Disabled                             |
| Branch subset                 | 2 to 10 percent of eligible prefixes |

# 45. Second-stage configuration

A larger configuration is admitted after the first-stage model clears every deterministic-control and K=1 decision gate.

| **Parameter**                 | **Recommended value**                            |
|-------------------------------|--------------------------------------------------|
| Layers                        | 32                                               |
| Residual width d              | 4096                                             |
| Recurrent + predictive mixers | 26                                               |
| Exact-attention layers        | 6                                                |
| Predictive layers             | 6 to 8                                           |
| K_max                         | 6                                                |
| D_z                           | 256 to 384                                       |
| r_cov                         | 4                                                |
| Basis-bank total span         | 64 to 128 before orthonormalization              |
| Characteristic frequencies    | 96 to 128                                        |
| MTP horizon                   | 4                                                |
| Adaptive activity             | Budgeted 0.25 to 0.5 equivalent dense FLOPs      |
| Sparse routing                | Entmax after dense warmup                        |
| eta_fb                        | 0 initially; weak safe gradient only as ablation |

# 46. Forward-pass algorithm

At each predictive layer and token, the forward computation is a causal map from the residual state to a finite conditional density, a distribution-invariant embedding, and a gated residual correction.

c_t = W_c RMSNorm(h_t)

f\_(t,m) = SiLU(W_g c_t + A_g e_m) \* (W_u c_t + A_u e_m)

pi_t = softmax(W_pi c_t)

mu\_(t,m) = bounded(W_base c_t + W_mu f\_(t,m))

sigma\_(t,m)^2 = sigma_min^2 + (sigma_max^2-sigma_min^2) sigmoid(W_sigma f\_(t,m))

Sigma\_(t,m) = Diag(sigma\_(t,m)^2) + U\_(t,m)U\_(t,m)^T

q(Z\|h_t) = sum_m pi\_(t,m) N(Z;mu\_(t,m),Sigma\_(t,m))

b_t = CharacteristicMomentSummary(q(.\|h_t))

h'\_t = h_t + gamma_l g_pred,t W_pred RMSNorm(SG\_(eta_fb)(b_t))

- For the diagonal first-stage model, U\_(t,m)=0.

- If compute routing is enabled, the entire predictive branch is skipped when a\_(t,l)=0.

- All token-local density outputs can be discarded after the residual update during ordinary inference.

# 47. Training-step algorithm

A training step separates target construction, proper density optimization, residual feedback, and primary language modeling.

- 1\. Run the online backbone causally on the packed token batch with document-reset masks.

- 2\. Compute next-token LM loss and MTP losses.

- 3\. For eligible density positions, retrieve or compute frozen target features Zhat_t.

- 4\. Evaluate q_phi(Zhat_t\|h_t), posterior responsibilities, and density NLL.

- 5\. On the branch subset, retrieve M frozen teacher continuations, construct Zhat_t^(j), and evaluate L_MC plus optional L_CF.

- 6\. Build the analytic characteristic-moment embedding b_t from q_phi.

- 7\. Apply the bounded predictive residual using eta_fb=0 in the primary experiment.

- 8\. Aggregate auxiliary gradients at selected hidden interfaces and apply conflict projection and norm caps.

- 9\. Update online parameters with AdamW or the validated large-scale optimizer.

- 10\. Update routing utility only from exploration examples with downstream counterfactual labels.

- 11\. Log all residual, covariance, calibration, multimodality, routing, memory, and compute diagnostics.

# 48. Inference algorithm

During autoregressive decode, target construction, branch generation, whitening updates, MTP heads, and density losses are absent. Each recurrent layer updates fixed-size memory; exact layers append KV state; selected predictive layers compute q from the current hidden state only when their routing decision executes.

- 1\. Embed the new token and advance recurrent convolution state.

- 2\. Update recurrent associative memory with the contractive delta rule.

- 3\. Execute exact attention at A layers using cached grouped-query K/V.

- 4\. At an active predictive layer, compute mixture parameters from h_t.

- 5\. Compute only the characteristic and moment statistics required by b_t.

- 6\. Apply the bounded gated predictive residual.

- 7\. Apply the SwiGLU channel transformation.

- 8\. Produce next-token logits from the final residual state.

- 9\. Discard token-local mixture parameters after their feedback computation.

# 49. Formal architecture definition

An AELIA predictive layer is a causal operator that maps hidden state to a conditional probability law over a frozen future-feature space, maps that law to a decomposition-invariant distribution embedding, and applies a bounded residual transformation. The distribution embedding is a function of q itself, up to finite random-feature approximation.

The full model is a composition of recurrent memory, exact attention, predictive density, and channel operators under the ordinary autoregressive token factorization.

h_t -\> Theta_t -\> q_phi(Z\|h_t) -\> B(q_phi) -\> h'\_t

Theta_t = {pi\_(t,m), mu\_(t,m), Sigma\_(t,m)}\_(m=1)^K

q_Theta1 = q_Theta2 =\> B(q_Theta1) = B(q_Theta2) up to finite-feature numerical error

# 50. Falsifiable architecture claim

AELIA claims that, under fixed all-in training compute and parameter budgets, selected intermediate states trained to approximate a stable conditional future-feature law can produce lower autoregressive validation risk than direct MTP alone and than deterministic future-representation controls receiving the same supervision and feedback bandwidth.

Explicit multimodality earns architectural status only when K\>1 improves held-out conditional density prediction and autoregressive validation risk, while branch data show positive continuation-to-mode information and the fitted mixture shows positive distinguishable mode information. Distributional feedback earns architectural status only when it improves the model relative to the same density trained without feedback. Low-rank covariance and adaptive execution earn status only through compute-normalized improvements.

The architecture therefore scales by evidence: every additional mechanism is admitted by a measured decision gate tied to the scientific quantity it is intended to improve.

R(AELIA;C,P) \< R(Hybrid+MTP;C,P)

R(AELIA;C,P) \< R(SameSupervisionDeterministicControl;C,P)

R(AELIA_K\>1;C,P) \< R(AELIA_K=1;C,P)

Delta_branch_multi \> 0 ; I_branch \> 0 ; I_mode \> 0

R(AELIA_feedback;C,P) \< R(AELIA_no_feedback;C,P)

# A. Parameterization summary

| **Group**           | **Core parameters**                                 | **Primary constraint**                            |
|---------------------|-----------------------------------------------------|---------------------------------------------------|
| Backbone            | E_in, recurrent projections, attention Q/K/V/O, FFN | Matched across controls                           |
| Predictive context  | W_c, W_g, W_u, A_g, A_u, e_m                        | Shared mode network                               |
| Means               | W_base, W_mu                                        | Optional RMS mean bound                           |
| Diagonal covariance | W_sigma, b_sigma                                    | sigma_min^2 \<= sigma^2 \<= sigma_max^2           |
| Low-rank covariance | basis bank B_g, W_B, C heads, W_lambda              | PSD addition; bounded strengths                   |
| Router              | W_pi, b_pi, tau_pi                                  | Dense softmax first; sparse later                 |
| Feedback            | fixed omega_j, P_mu, P_C, W_pred, gate              | Distribution-invariant inputs; depth-scaled gamma |
| MTP                 | A_h, B_h, shared E_out                              | Training-only auxiliary heads                     |
| Compute policy      | R_eta, lambda_c                                     | Causal features; FLOP budget                      |

# B. Diagnostic definitions

The following diagnostics are reported by predictive layer, model scale, context-length bucket, token-entropy bucket, document type, and training checkpoint.

U_total = tr(C_total)/D_z

C_within = sum_m pi_m Sigma_m

C_between = sum_m pi_m (mu_m-mu_bar)(mu_m-mu_bar)^T

C_total = C_within + C_between

U_within = tr(C_within)/D_z

U_between = tr(C_between)/D_z

D_eff = tr(C_total)^2 / (tr(C_total^2)+epsilon)

KL(r_t\|\|pi_t) = sum_m r\_(t,m) log((r\_(t,m)+epsilon)/(pi\_(t,m)+epsilon))

R_pred,l = E\[\|\|Delta_pred,l\|\|\_2\] / E\[\|\|h_l\|\|\_2\]

Gain_per_FLOP = sum\_(t,l) a\_(t,l) Delta_true,(t,l) / sum\_(t,l) a\_(t,l) c\_(t,l)

- Predictive NLL and branch NLL.

- Characteristic-kernel score, energy score, variogram score.

- Projected PIT and projected Rosenblatt calibration.

- H(pi), I_mode, K_dist, K_active, I_branch, Coverage_MI.

- Posterior-prior KL and per-mode posterior use.

- Mean diagonal variance, floor/ceiling saturation, low-rank strength quantiles.

- Characteristic-summary norm and predictive residual norm.

- Utility correlation, policy regret, activity by layer and token category.

- All-in train FLOPs/token, online FLOPs/token, tokens/s, MFU, peak memory, prefill and decode throughput.

# C. Experimental decision gates

| **Gate**                 | **Required evidence**                                                           | **Action**                            |
|--------------------------|------------------------------------------------------------------------------|---------------------------------------|
| G1 Future supervision    | Hybrid+MTP beats hybrid at iso-compute                                          | Retain MTP                            |
| G2 Density bottleneck    | K=1 AELIA beats Hybrid+MTP or improves representation at matched all-in compute | Retain conditional density            |
| G3 Probability structure | AELIA beats same-supervision deterministic embedding control                    | Retain probabilistic parameterization |
| G4 Multimodality         | K\>1 improves held-out NLL and LM risk; I_branch\>0; I_mode\>0                  | Retain mixture K\>1                   |
| G5 Distribution matching | L_CF improves held-out density or LM metrics at matched compute                 | Retain characteristic distillation    |
| G6 Feedback              | Feedback beats no-feedback density                                              | Retain residual reuse                 |
| G7 Covariance            | Low-rank beats diagonal on quality per FLOP                                     | Retain low-rank covariance            |
| G8 Sparsity              | Sparse routing preserves quality while reducing realized predictive FLOPs       | Retain sparse modes                   |
| G9 Adaptive compute      | Budgeted routing improves loss at equal inference FLOPs                         | Retain adaptive execution             |
| G10 Scale                | G1-G6 reproduce at two or more model scales                                     | Proceed to aggressive scaling         |

# Compact mathematical core

h_t = HybridCausalBackbone(y\_\<=t)

Y_t^+ ~ P_bar(.\|y\_\<=t)

Z_t = F_bar(y\_\<=t,Y_t^+)

q_phi(Z\|h_t) = sum_m pi\_(t,m) N(Z;mu\_(t,m),Sigma\_(t,m))

L_density = E\_(Z~P_t^F)\[-log q_phi(Z\|h_t)\]

b_t = CharacteristicMomentSummary(q_phi(.\|h_t))

h'\_t = h_t + gamma_l g_t W_pred RMSNorm(SG\_(eta_fb)(b_t))

x\_(t,l+1) = h'\_t + FFN(RMSNorm(h'\_t))

AELIA is the constrained causal loop from hidden state to stable conditional future law, from future law to an invariant distribution representation, and from that representation to a bounded residual correction, evaluated under same-supervision and all-in compute controls.
