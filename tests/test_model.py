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
