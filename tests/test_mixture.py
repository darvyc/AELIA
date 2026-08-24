import torch

from aelia.config import PredictiveConfig
from aelia.mixture import GaussianMixturePredictor, MixtureParams


def _predictor(covariance="diagonal"):
    return GaussianMixturePredictor(
        PredictiveConfig(
            d_model=24,
            target_dim=8,
            modes=3,
            context_dim=16,
            feature_dim=16,
            mode_dim=8,
            covariance=covariance,
            cov_rank=2,
            basis_rank=4,
            characteristic_features=7,
        )
    )


def test_weights_responsibilities_and_nll():
    torch.manual_seed(0)
    predictor = _predictor()
    h = torch.randn(2, 4, 24)
    z = torch.randn(2, 4, 8)
    p = predictor(h)
    r = predictor.responsibilities(z, p)
    assert torch.allclose(p.weights.sum(-1), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(r.sum(-1), torch.ones(2, 4), atol=1e-5)
    assert torch.isfinite(predictor.nll(z, p))


def test_lowrank_woodbury_matches_dense_gaussian():
    torch.manual_seed(1)
    predictor = _predictor("diag_lowrank")
    h = torch.randn(2, 24)
    z = torch.randn(2, 8)
    p = predictor(h)
    comp = predictor.component_log_prob(z, p)
    dense_values = []
    constant = 8 * torch.log(torch.tensor(2.0 * torch.pi))
    for b in range(2):
        row = []
        for k in range(3):
            d = torch.diag(p.diag_var[b, k])
            u = p.lowrank[b, k]
            sigma = d + u @ u.T
            delta = z[b] - p.means[b, k]
            sign, logdet = torch.linalg.slogdet(sigma)
            assert sign > 0
            quad = delta @ torch.linalg.solve(sigma, delta)
            row.append(-0.5 * (constant + logdet + quad))
        dense_values.append(torch.stack(row))
    dense = torch.stack(dense_values)
    assert torch.allclose(comp, dense, atol=2e-4, rtol=2e-4)


def test_characteristic_is_permutation_invariant():
    torch.manual_seed(2)
    predictor = _predictor()
    p = predictor(torch.randn(2, 24))
    phi1 = predictor.characteristic(p)
    perm = torch.tensor([2, 0, 1])
    p2 = MixtureParams(p.weights[:, perm], p.means[:, perm], p.diag_var[:, perm], None)
    phi2 = predictor.characteristic(p2)
    assert torch.allclose(phi1, phi2, atol=1e-6, rtol=1e-6)


def test_identical_component_split_keeps_characteristic():
    torch.manual_seed(3)
    predictor = GaussianMixturePredictor(
        PredictiveConfig(
            d_model=12,
            target_dim=4,
            modes=2,
            context_dim=8,
            feature_dim=8,
            mode_dim=4,
            characteristic_features=5,
        )
    )
    mean = torch.randn(1, 1, 4)
    var = torch.ones(1, 1, 4)
    p1 = MixtureParams(torch.tensor([[1.0, 0.0]]), torch.cat([mean, mean], 1), torch.cat([var, var], 1), None)
    p2 = MixtureParams(torch.tensor([[0.25, 0.75]]), torch.cat([mean, mean], 1), torch.cat([var, var], 1), None)
    assert torch.allclose(predictor.characteristic(p1), predictor.characteristic(p2), atol=1e-6)
