import math

import pytest
import torch

from aelia.config import PredictiveConfig
from aelia.feedback import DistributionSummary
from aelia.mixture import GaussianMixturePredictor, MixtureParams


def _predictor(**kwargs):
    return GaussianMixturePredictor(
        PredictiveConfig(
            d_model=8,
            target_dim=4,
            modes=2,
            context_dim=8,
            feature_dim=8,
            mode_dim=4,
            cov_rank=2,
            basis_rank=4,
            characteristic_features=3,
            **kwargs,
        )
    )


@pytest.mark.parametrize("rank", [0, 2])
def test_float64_likelihood_and_gradients_match_dense_gaussian(rank):
    torch.manual_seed(21)
    predictor = _predictor().double()
    z = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    means = torch.randn(2, 2, 4, dtype=torch.float64, requires_grad=True)
    diag = torch.rand_like(means).add(0.2).requires_grad_()
    low = torch.randn(2, 2, 4, rank, dtype=torch.float64, requires_grad=True) if rank else None
    params = MixtureParams(torch.full((2, 2), 0.5, dtype=torch.float64), means, diag, low)
    actual = predictor.component_log_prob(z, params)
    covariance = torch.diag_embed(diag)
    if low is not None:
        covariance = covariance + low @ low.transpose(-1, -2)
    expected = torch.distributions.MultivariateNormal(means, covariance_matrix=covariance).log_prob(z[:, None])
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (z, means, diag) if low is None else (z, means, diag, low)
    actual_grad = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), inputs)
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("rank", [0, 2])
def test_partial_observation_matches_dense_marginal_and_masks_nan(rank):
    torch.manual_seed(22)
    predictor = _predictor().double()
    mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
    z = torch.randn(2, 4, dtype=torch.float64).masked_fill(~mask, torch.nan).requires_grad_()
    means = torch.randn(2, 2, 4, dtype=torch.float64, requires_grad=True)
    diag = torch.rand_like(means).add(0.1).requires_grad_()
    low = torch.randn(2, 2, 4, rank, dtype=torch.float64, requires_grad=True) if rank else None
    params = MixtureParams(torch.full((2, 2), 0.5, dtype=torch.float64), means, diag, low)
    covariance = torch.diag_embed(diag)
    if low is not None:
        covariance = covariance + low @ low.transpose(-1, -2)
    expected = []
    for b in range(2):
        selected = mask[b]
        dense = covariance[b][:, selected][:, :, selected]
        dist = torch.distributions.MultivariateNormal(means[b][:, selected], covariance_matrix=dense)
        expected.append(dist.log_prob(z[b, selected]))
    expected = torch.stack(expected)
    actual = predictor.component_log_prob(z, params, observed_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    nll = predictor.nll(z, params, observed_mask=mask)
    target = (-torch.logsumexp(expected - math.log(2), -1) / mask.sum(-1)).mean()
    torch.testing.assert_close(nll, target, rtol=1e-12, atol=1e-12)
    nll.backward()
    assert torch.count_nonzero(z.grad[~mask]) == 0
    assert torch.count_nonzero(means.grad.masked_select(~mask[:, None])) == 0
    assert torch.isfinite(z.grad).all() and torch.isfinite(means.grad).all()


def test_empty_observation_has_unit_density_and_zero_training_loss():
    predictor = _predictor(covariance="diag_lowrank").double()
    params = predictor(torch.randn(2, 8, dtype=torch.float64))
    z = torch.full((2, 4), torch.nan, dtype=torch.float64)
    mask = torch.zeros_like(z, dtype=torch.bool)
    torch.testing.assert_close(
        predictor.log_prob(z, params, mask), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0
    )
    loss = predictor.nll(z, params, observed_mask=mask)
    assert loss.item() == 0.0
    loss.backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in predictor.parameters() if p.grad is not None)


def test_exact_zero_component_has_no_density_or_posterior_mass():
    predictor = _predictor().double()
    weights = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    means = torch.tensor([[20.0] * 4, [0.0] * 4], dtype=torch.float64)
    params = MixtureParams(weights, means, torch.ones_like(means))
    z = torch.zeros(4, dtype=torch.float64)
    expected = predictor.component_log_prob(z, params)[0]
    actual = predictor.log_prob(z, params)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(predictor.responsibilities(z, params), torch.tensor([1.0, 0.0], dtype=torch.float64))
    actual.backward()
    assert torch.isfinite(weights.grad).all() and weights.grad[1] == 0


def test_router_gradient_includes_temperature_and_dimension_normalization():
    torch.manual_seed(23)
    predictor = _predictor(mixture_temperature=0.7).double()
    with torch.no_grad():
        predictor.router.weight.zero_()
        predictor.router.bias.copy_(torch.tensor([0.8, -1.3]))
    params = predictor(torch.randn(3, 8, dtype=torch.float64))
    z = torch.randn(3, 4, dtype=torch.float64)
    responsibilities = predictor.responsibilities(z, params).detach()
    expected = ((params.weights.detach() - responsibilities) / (4 * 0.7)).mean(dim=0)
    predictor.nll(z, params).backward()
    torch.testing.assert_close(predictor.router.bias.grad, expected, rtol=1e-12, atol=1e-12)


def test_extreme_router_retains_finite_log_probability():
    predictor = _predictor()
    with torch.no_grad():
        predictor.router.weight.zero_()
        predictor.router.bias.copy_(torch.tensor([0.0, -1000.0]))
    params = predictor(torch.zeros(8))
    assert params.weights[1] == 0.0
    assert params.log_weights[1] == -1000.0
    assert torch.isfinite(params.log_weights).all()


@pytest.mark.parametrize("strength", [0.0, -1000.0])
def test_degenerate_lowrank_orientation_has_finite_gradients(strength):
    predictor = _predictor(covariance="diag_lowrank")
    with torch.no_grad():
        predictor.orientation.weight.zero_()
        predictor.orientation.bias.zero_()
        predictor.strength.weight.zero_()
        predictor.strength.bias.fill_(strength)
        predictor.base_mean.weight.zero_()
        predictor.base_mean.bias.zero_()
        predictor.mean.weight.zero_()
        predictor.mean.bias.zero_()
    params = predictor(torch.zeros(2, 8))
    predictor.nll(torch.randn(2, 4), params).backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


def test_lowrank_covariance_bounds_and_autocast_precision():
    torch.manual_seed(24)
    predictor = _predictor(covariance="diag_lowrank")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        params = predictor(torch.randn(2, 8))
        loss = predictor.nll(torch.randn(2, 4), params)
        phi = predictor.characteristic(params)
    assert params.means.dtype == params.diag_var.dtype == params.lowrank.dtype == loss.dtype == torch.float32
    assert phi.dtype == torch.complex64
    spectral = torch.linalg.matrix_norm(params.lowrank, ord=2).square()
    assert torch.all(spectral <= predictor.config.lambda_max + 1e-6)
    covariance = torch.diag_embed(params.diag_var) + params.lowrank @ params.lowrank.transpose(-1, -2)
    eig = torch.linalg.eigvalsh(covariance)
    assert eig.min() >= predictor.config.sigma_min**2 - 1e-6
    assert eig.max() <= predictor.config.sigma_max**2 + predictor.config.lambda_max + 1e-6
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


def test_mahalanobis_form_retains_small_positive_value():
    predictor = _predictor()
    low = torch.zeros(1, 4, 1)
    low[0, 0, 0] = 100.0
    diag = torch.full((1, 4), 1e-4)
    params = MixtureParams(torch.ones(1), torch.zeros(1, 4), diag, low)
    z = torch.tensor([100.0, 0.0, 0.0, 0.0])
    expected_diag = torch.tensor([10000.0001, 0.0001, 0.0001, 0.0001], dtype=torch.float64)
    expected = torch.distributions.MultivariateNormal(
        torch.zeros(4, dtype=torch.float64), torch.diag(expected_diag)
    ).log_prob(z.double())
    torch.testing.assert_close(predictor.log_prob(z, params).double(), expected, rtol=2e-6, atol=2e-6)


def test_projected_moments_include_correlations_and_between_mode_variance():
    torch.manual_seed(25)
    predictor = _predictor().double()
    means = torch.randn(2, 2, 4, dtype=torch.float64) + 100000.0
    diag = torch.rand_like(means).add(0.1)
    low = torch.randn(2, 2, 4, 2, dtype=torch.float64)
    weights = torch.tensor([[0.3, 0.7], [0.8, 0.2]], dtype=torch.float64)
    params = MixtureParams(weights, means, diag, low)
    projection = torch.randn(3, 4, dtype=torch.float64)
    mean = (weights[..., None] * means).sum(-2)
    centered = means - mean[:, None]
    covariance = (
        weights[..., None, None]
        * (torch.diag_embed(diag) + low @ low.transpose(-1, -2) + centered[..., None] * centered[..., None, :])
    ).sum(-3)
    actual_mean, actual_var = predictor.projected_moments(params, projection)
    expected_var = (projection @ covariance @ projection.T).diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(actual_mean, mean @ projection.T, rtol=1e-12, atol=1e-10)
    torch.testing.assert_close(actual_var, expected_var, rtol=1e-10, atol=1e-9)
    _, diag_var = predictor.first_two_moments(params)
    torch.testing.assert_close(diag_var, covariance.diagonal(dim1=-2, dim2=-1), rtol=1e-12, atol=1e-12)
    assert predictor.characteristic(params).dtype == torch.complex128


def test_complete_summary_is_invariant_under_component_split():
    torch.manual_seed(26)
    predictor = _predictor().double()
    summary = DistributionSummary(predictor, projected_moments=3).double()
    means = torch.randn(1, 4, dtype=torch.float64)
    diag = torch.ones_like(means)
    low = torch.randn(1, 4, 2, dtype=torch.float64)
    one = MixtureParams(torch.ones(1, dtype=torch.float64), means, diag, low)
    split = MixtureParams(
        torch.tensor([0.3, 0.7], dtype=torch.float64), means.expand(2, -1), diag.expand(2, -1), low.expand(2, -1, -1)
    )
    torch.testing.assert_close(summary(one), summary(split), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"covariance": "diag_lowrank", "target_dim": 2, "basis_rank": 3},
        {"mixture_temperature": 0.0},
        {"eps": 0.0},
        {"characteristic_scale": 0.0},
        {"lambda_max": -1.0},
        {"mean_rms_max": 0.0},
        {"sigma_max": math.inf},
    ],
)
def test_predictive_configuration_enforces_mathematical_domain(kwargs):
    with pytest.raises(ValueError):
        PredictiveConfig(d_model=8, **kwargs).validate()
