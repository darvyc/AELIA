import pytest
import torch

from aelia import AELIALM, ModelConfig


def test_reference_model_forward():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=97,
        d_model=32,
        layers=("R", "P", "A"),
        ffn_multiplier=2.0,
        memory_heads=2,
        memory_d_key=8,
        memory_d_value=8,
        attention_query_heads=4,
        attention_kv_heads=2,
        attention_head_dim=8,
        target_dim=8,
        modes=2,
        characteristic_features=4,
    )
    model = AELIALM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(tokens)
    assert out.logits.shape == (2, 6, cfg.vocab_size)
    assert len(out.predictive) == 1
    assert out.predictive[0].params.means.shape == (2, 6, 2, 8)


def _small_model():
    return AELIALM(
        ModelConfig(
            vocab_size=31,
            d_model=16,
            layers=("R", "P", "A", "R", "A"),
            ffn_multiplier=2,
            memory_heads=2,
            memory_d_key=4,
            memory_d_value=4,
            attention_query_heads=4,
            attention_kv_heads=2,
            attention_head_dim=4,
            target_dim=6,
            modes=2,
            characteristic_features=3,
        )
    ).eval()


@pytest.mark.parametrize("packed", [False, True])
def test_hybrid_cache_and_last_token_logits(packed):
    torch.manual_seed(16)
    model = _small_model()
    tokens = torch.randint(0, 31, (2, 8))
    reset = (
        torch.tensor(
            [
                [False, False, True, False, False, False, True, False],
                [True, False, False, False, True, False, False, False],
            ]
        )
        if packed
        else None
    )
    with torch.no_grad():
        full = model(tokens, reset_mask=reset)
        pieces = []
        out, start = None, 0
        for end in (3, 4, 7, 8):
            out = model(
                tokens[:, start:end],
                recurrent_states=None if out is None else out.recurrent_states,
                attention_states=None if out is None else out.attention_states,
                reset_mask=None if reset is None else reset[:, start:end],
                use_cache=True,
                return_predictive=False,
            )
            assert out.predictive == []
            pieces.append(out.logits)
            start = end
        torch.testing.assert_close(torch.cat(pieces, dim=1), full.logits, rtol=2e-5, atol=2e-5)
        suffix = model(tokens, reset_mask=reset, logits_to_keep=1)
        torch.testing.assert_close(suffix.logits, full.logits[:, -1:], rtol=2e-5, atol=2e-5)


def test_hybrid_document_isolation():
    torch.manual_seed(17)
    model = _small_model()
    tokens = torch.randint(0, 31, (2, 7))
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    reset[:, 3] = True
    with torch.no_grad():
        full = model(tokens, reset_mask=reset)
        independent = model(tokens[:, 3:])
    torch.testing.assert_close(full.logits[:, 3:], independent.logits, rtol=2e-5, atol=2e-5)


def test_incomplete_state_lists_are_rejected():
    model = _small_model()
    tokens = torch.ones(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="one entry per layer"):
        model(tokens, recurrent_states=[None])
    out = model(tokens, use_cache=True)
    with pytest.raises(ValueError, match="attention_states"):
        model(tokens, recurrent_states=out.recurrent_states)
    with pytest.raises(ValueError, match="recurrent layer"):
        model(tokens, attention_states=out.attention_states, use_cache=True)


def test_bfloat16_autocast_training_step():
    torch.manual_seed(18)
    model = _small_model().train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.randint(0, 31, (2, 5)))
        predictor = model.blocks[1].predictor
        loss = out.logits.float().square().mean() + predictor.nll(torch.randn(2, 5, 6), out.predictive[0].params)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
