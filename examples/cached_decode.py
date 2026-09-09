"""Greedy continuation with complete hybrid state and suffix-only logits."""

import torch

from aelia import AELIALM, ModelConfig


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=97,
        d_model=32,
        layers=("R", "P", "A"),
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
    model = AELIALM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 12))
    out = model(prompt, use_cache=True, logits_to_keep=1, return_predictive=False)
    generated = []
    for _ in range(8):
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token)
        out = model(
            token,
            recurrent_states=out.recurrent_states,
            attention_states=out.attention_states,
            use_cache=True,
            logits_to_keep=1,
            return_predictive=False,
        )
    print("Generated token IDs from an untrained model:", torch.cat(generated, dim=1).tolist())


if __name__ == "__main__":
    main()
