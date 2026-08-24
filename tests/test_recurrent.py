import torch

from aelia.config import MemoryConfig
from aelia.recurrent import ContractiveDeltaMemory


def test_recurrent_shapes_and_finite_state():
    torch.manual_seed(0)
    module = ContractiveDeltaMemory(MemoryConfig(d_model=32, heads=2, d_key=8, d_value=8, alpha_max=0.95))
    x = torch.randn(3, 5, 32)
    y, state = module(x)
    assert y.shape == x.shape
    assert state.memory.shape == (3, 2, 8, 8)
    assert torch.isfinite(y).all()
    assert torch.isfinite(state.memory).all()


def test_document_reset_clears_history():
    torch.manual_seed(1)
    module = ContractiveDeltaMemory(MemoryConfig(d_model=16, heads=1, d_key=4, d_value=4, alpha_max=0.9))
    x = torch.randn(1, 4, 16)
    reset = torch.tensor([[False, False, True, False]])
    y_packed, _ = module(x, reset_mask=reset)
    y_suffix, _ = module(x[:, 2:])
    assert torch.allclose(y_packed[:, 2:], y_suffix, atol=1e-5, rtol=1e-5)


def test_contraction_ceiling_is_strict():
    module = ContractiveDeltaMemory(MemoryConfig(d_model=16, heads=1, d_key=4, d_value=4, alpha_max=0.97))
    assert 0.0 < module.contraction_ceiling < 1.0
