import torch

from aelia.mixture import MixtureParams
from aelia.scoring import branch_coverage_mi, energy_score, projected_pit, sample_mixture, variogram_score


def test_scoring_shapes_and_pit_bounds():
    torch.manual_seed(0)
    weights = torch.tensor([[0.4, 0.6]])
    means = torch.tensor([[[0.0, 0.0], [1.0, -1.0]]])
    var = torch.ones_like(means) * 0.5
    params = MixtureParams(weights, means, var)
    samples = sample_mixture(params, 64)
    z = torch.tensor([[0.2, -0.1]])
    assert samples.shape == (1, 64, 2)
    assert energy_score(samples, z).shape == (1,)
    assert variogram_score(samples, z).shape == (1,)
    directions = torch.eye(2)
    pit = projected_pit(params, z, directions)
    assert pit.shape == (1, 2)
    assert ((pit >= 0) & (pit <= 1)).all()


def test_branch_coverage_zero_when_posteriors_identical():
    r = torch.tensor([[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]])
    mi = branch_coverage_mi(r)
    assert torch.allclose(mi, torch.zeros_like(mi), atol=1e-6)
