import copy

import torch

from aelia.config import MemoryConfig
from aelia.recurrent import ContractiveDeltaMemory, RecurrentState


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


def test_batched_scan_matches_steps_and_all_gradients():
    torch.manual_seed(14)
    module = ContractiveDeltaMemory(MemoryConfig(12, 2, 3, 4)).double()
    reference = copy.deepcopy(module)
    x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    initial = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    initial_ref = initial.detach().clone().requires_grad_()
    reset = torch.tensor([[False, False, True, False, False], [False, True, False, False, True]])
    output, state = module(x, RecurrentState(initial), reset)
    ref_state = RecurrentState(initial_ref)
    outputs = []
    for t in range(x.shape[1]):
        y, ref_state = reference.step(x_ref[:, t], ref_state, reset[:, t])
        outputs.append(y)
    expected = torch.stack(outputs, dim=1)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.memory, ref_state.memory, rtol=1e-12, atol=1e-12)
    (output.square().sum() + state.memory.square().sum()).backward()
    (expected.square().sum() + ref_state.memory.square().sum()).backward()
    for actual, wanted in [(x.grad, x_ref.grad), (initial.grad, initial_ref.grad)]:
        torch.testing.assert_close(actual, wanted, rtol=1e-11, atol=1e-11)
    for actual, wanted in zip(module.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-10, atol=1e-11)


def test_dense_transition_and_state_perturbation_bound():
    torch.manual_seed(15)
    module = ContractiveDeltaMemory(MemoryConfig(12, 2, 3, 4, alpha_max=0.93)).double()
    x = torch.randn(2, 12, dtype=torch.float64)
    s0 = torch.randn(2, 2, 3, 4, dtype=torch.float64)
    s1 = torch.randn_like(s0)
    _, q, k, v, alpha, beta = module._project(x)
    transition = (torch.eye(3) - beta.unsqueeze(-1) * k.unsqueeze(-1) * k.unsqueeze(-2)) @ torch.diag_embed(alpha)
    expected = transition @ s0 + (beta * k).unsqueeze(-1) * v.unsqueeze(-2)
    _, actual = module.step(x, RecurrentState(s0))
    _, other = module.step(x, RecurrentState(s1))
    torch.testing.assert_close(actual.memory, expected, rtol=1e-12, atol=1e-12)
    assert torch.all(torch.linalg.matrix_norm(actual.memory - other.memory) <= 0.93 * torch.linalg.matrix_norm(s0 - s1))
    # A boundary discards even nonfinite state from the preceding document.
    _, cleared = module.step(x, RecurrentState(torch.full_like(s0, torch.nan)), torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(cleared.memory).all()


def test_half_precision_memory_and_norm_remain_finite():
    module = ContractiveDeltaMemory(MemoryConfig(8, 1, 4, 4)).half()
    x = torch.full((1, 3, 8), 1000.0, dtype=torch.float16, requires_grad=True)
    y, state = module(x)
    assert y.dtype == torch.float16
    assert state.memory.dtype == torch.float32
    assert torch.isfinite(y).all() and torch.isfinite(state.memory).all()
    y.float().mean().backward()
    assert torch.isfinite(x.grad).all()
