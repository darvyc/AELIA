import copy

import pytest
import torch

from aelia.config import MemoryConfig, PredictiveConfig
from aelia.mixture import GaussianMixturePredictor, MixtureParams
from aelia.recurrent import ContractiveDeltaMemory, RecurrentState


@pytest.mark.parametrize('reset', [False, True])
def test_batched_recurrence_matches_steps_and_gradients(reset):
    torch.manual_seed(7)
    module = ContractiveDeltaMemory(MemoryConfig(d_model=16, heads=2, d_key=4, d_value=3)).double()
    reference = copy.deepcopy(module)
    x = torch.randn(2, 7, 16, dtype=torch.double, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    initial = torch.randn(2, 2, 4, 3, dtype=torch.double, requires_grad=True)
    ir = initial.detach().clone().requires_grad_()
    mask = torch.rand(2, 7) < 0.3 if reset else None
    y, state = module(x, RecurrentState(initial), mask)
    sr = RecurrentState(ir)
    outputs = []
    for t in range(7):
        out, sr = reference.step(xr[:, t], sr, None if mask is None else mask[:, t])
        outputs.append(out)
    yr = torch.stack(outputs, 1)
    torch.testing.assert_close(y, yr)
    torch.testing.assert_close(state.memory, sr.memory)
    (y.square().sum() + state.memory.square().sum()).backward()
    (yr.square().sum() + sr.memory.square().sum()).backward()
    for actual, expected in [(x.grad, xr.grad), (initial.grad, ir.grad)]:
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(module.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_projected_moments_match_dense_covariance():
    torch.manual_seed(8)
    predictor = GaussianMixturePredictor(PredictiveConfig(d_model=8, target_dim=4))
    weights = torch.softmax(torch.randn(2, 3, dtype=torch.double), -1)
    means = torch.randn(2, 3, 4, dtype=torch.double)
    diag = torch.rand_like(means) + 0.1
    u = torch.randn(2, 3, 4, 2, dtype=torch.double)
    p = torch.randn(5, 4, dtype=torch.double)
    params = MixtureParams(weights, means, diag, u)
    mean, variance = predictor.projected_moments(params, p)
    center = (weights[..., None] * means).sum(-2)
    offsets = means - center[:, None]
    cov = torch.diag_embed(diag) + u @ u.transpose(-1, -2)
    cov = cov + offsets[..., None] * offsets[..., None, :]
    cov = (weights[..., None, None] * cov).sum(-3)
    torch.testing.assert_close(mean, center @ p.T)
    torch.testing.assert_close(variance, torch.diagonal(p @ cov @ p.T, dim1=-2, dim2=-1))


def test_centered_variance_preserves_small_uncertainty():
    predictor = GaussianMixturePredictor(PredictiveConfig(d_model=8, target_dim=4))
    params = MixtureParams(torch.tensor([[0.5, 0.5]]), torch.full((1, 2, 4), 1e6), torch.ones(1, 2, 4))
    _, variance = predictor.first_two_moments(params)
    torch.testing.assert_close(variance, torch.ones_like(variance))


def test_lowrank_density_double_precision_and_gradients():
    torch.manual_seed(9)
    predictor = GaussianMixturePredictor(PredictiveConfig(d_model=8, target_dim=4))
    z = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    means = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    diag = (torch.rand(2, 3, 4, dtype=torch.double) + 0.2).requires_grad_()
    u = torch.randn(2, 3, 4, 2, dtype=torch.double, requires_grad=True)
    weights = torch.full((2, 3), 1 / 3, dtype=torch.double)
    actual = predictor.component_log_prob(z, MixtureParams(weights, means, diag, u))
    dense = torch.distributions.MultivariateNormal(means, covariance_matrix=torch.diag_embed(diag) + u @ u.transpose(-1, -2))
    expected = dense.log_prob(z[:, None])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    ga = torch.autograd.grad(actual.sum(), (z, means, diag, u), retain_graph=True)
    ge = torch.autograd.grad(expected.sum(), (z, means, diag, u))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-9, rtol=1e-9)


def test_density_amp_accumulates_in_float32():
    predictor = GaussianMixturePredictor(PredictiveConfig(d_model=8, target_dim=4))
    params = MixtureParams(torch.ones(1, 1), torch.zeros(1, 1, 4), torch.ones(1, 1, 4), torch.ones(1, 1, 4, 2))
    z = torch.randn(1, 4)
    expected = predictor.component_log_prob(z, params)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = predictor.component_log_prob(z, params)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_empty_sequence_preserves_state():
    module = ContractiveDeltaMemory(MemoryConfig(d_model=8, heads=1, d_key=2, d_value=2))
    state = RecurrentState(torch.randn(2, 1, 2, 2))
    x = torch.empty(2, 0, 8)
    y, final = module(x, state)
    assert y.shape == x.shape
    assert final is state


def test_chunked_sequence_matches_full_sequence():
    torch.manual_seed(10)
    module = ContractiveDeltaMemory(MemoryConfig(d_model=8, heads=1, d_key=2, d_value=2))
    x = torch.randn(2, 9, 8)
    y, final = module(x)
    prefix, state = module(x[:, :4])
    suffix, chunk_final = module(x[:, 4:], state)
    torch.testing.assert_close(torch.cat((prefix, suffix), 1), y)
    torch.testing.assert_close(chunk_final.memory, final.memory)
