import pytest
import torch

from aelia.attention import CausalGroupedQueryAttention
from aelia.config import AttentionConfig


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1, 1), (2, 3, 2)])
def test_cached_attention_matches_full_prefix(packed, chunks):
    torch.manual_seed(11)
    module = CausalGroupedQueryAttention(AttentionConfig(16, 4, 2, 4)).double().eval()
    x = torch.randn(2, 7, 16, dtype=torch.float64)
    reset = (
        torch.tensor([[True, False, False, True, False, False, False], [False, False, True, False, False, True, False]])
        if packed
        else None
    )
    full = module(x, reset_mask=reset)
    state, offset, outputs = None, 0, []
    for count in chunks:
        boundary = None if reset is None else reset[:, offset : offset + count]
        previous = None if state is None else state.keys.clone()
        y, next_state = module(x[:, offset : offset + count], reset_mask=boundary, state=state, use_cache=True)
        if previous is not None:
            torch.testing.assert_close(state.keys, previous, rtol=0, atol=0)
        outputs.append(y)
        offset += count
        assert next_state.keys.shape == (2, 2, offset, 4)
        state = next_state
    torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=1e-12, atol=1e-12)


def test_attention_has_zero_future_and_other_document_gradients():
    torch.manual_seed(12)
    module = CausalGroupedQueryAttention(AttentionConfig(16, 4, 1, 4)).double()
    x = torch.randn(1, 7, 16, dtype=torch.float64, requires_grad=True)
    reset = torch.tensor([[False, False, False, True, False, False, False]])
    output = module(x, reset_mask=reset)
    output[:, 4].square().sum().backward()
    assert torch.count_nonzero(x.grad[:, :3]) == 0
    assert torch.count_nonzero(x.grad[:, 5:]) == 0
    assert torch.count_nonzero(x.grad[:, 3:5]) > 0
    torch.testing.assert_close(output[:, 3:], module(x[:, 3:]), rtol=1e-12, atol=1e-12)


def test_reset_after_unpacked_cache_and_continuation_without_reset_mask():
    torch.manual_seed(13)
    module = CausalGroupedQueryAttention(AttentionConfig(16, 4, 2, 4)).double().eval()
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    _, state = module(x[:, :3], use_cache=True)
    first, state = module(x[:, 3:4], reset_mask=torch.ones(2, 1, dtype=torch.bool), state=state, use_cache=True)
    rest, state = module(x[:, 4:], state=state, use_cache=True)
    torch.testing.assert_close(torch.cat((first, rest), 1), module(x[:, 3:]), rtol=1e-12, atol=1e-12)
    assert torch.equal(state.next_position, torch.tensor([3, 3]))
