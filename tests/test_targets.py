import torch

from aelia.targets import FrozenWhitening, HellingerCountSketch


def test_countsketch_shape_and_repeatability():
    p = torch.softmax(torch.randn(3, 11), dim=-1)
    a = HellingerCountSketch(11, 7, repeats=3, seed=10)
    b = HellingerCountSketch(11, 7, repeats=3, seed=10)
    assert a(p).shape == (3, 21)
    assert torch.equal(a.buckets, b.buckets)
    assert torch.allclose(a(p), b(p))


def test_frozen_whitening_centers_calibration_data():
    torch.manual_seed(0)
    x = torch.randn(1024, 6) @ torch.randn(6, 6) + 3.0
    whitening = FrozenWhitening.fit(x, shrinkage=0.05)
    z = whitening(x)
    assert z.mean(0).abs().max() < 2e-4
    assert torch.isfinite(z).all()
