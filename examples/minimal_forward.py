import torch

from aelia import AELIALM, ModelConfig


def main() -> None:
    config = ModelConfig(
        vocab_size=32000,
        d_model=256,
        layers=("R", "R", "P", "A", "R", "P"),
        ffn_multiplier=3.0,
        memory_heads=4,
        memory_d_key=32,
        memory_d_value=32,
        attention_query_heads=4,
        attention_kv_heads=2,
        attention_head_dim=64,
        target_dim=64,
        modes=4,
        characteristic_features=24,
    )
    model = AELIALM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    out = model(tokens)
    print("logits:", tuple(out.logits.shape))
    print("predictive layers:", len(out.predictive))
    print("first mixture mean:", tuple(out.predictive[0].params.means.shape))


if __name__ == "__main__":
    main()
