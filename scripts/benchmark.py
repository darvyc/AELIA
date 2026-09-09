"""Reproducible operator and forced-token decoding benchmarks.

Run with an installed checkout: python scripts/benchmark.py --output results.json
Both comparisons verify numerical agreement before timing. Measurements include
prefill for decoding and autograd for the forward/backward recurrent case.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from aelia import AELIALM, ContractiveDeltaMemory, MemoryConfig, ModelConfig


def measure(fn, threads: int, minimum: float) -> dict:
    result = Timer("fn()", globals={"fn": fn}, num_threads=threads).blocked_autorange(min_run_time=minimum)
    return {"median_ms": result.median * 1000, "iqr_ms": result.iqr * 1000, "measurements": len(result.raw_times)}


def recurrent(args) -> dict:
    module = ContractiveDeltaMemory(MemoryConfig(128, 4, 16, 16)).to(args.device)
    x = torch.randn(2, args.sequence, 128, device=args.device, requires_grad=True)

    def token_steps():
        state = module.initial_state(x.shape[0], device=x.device, dtype=x.dtype)
        output = []
        for t in range(x.shape[1]):
            y, state = module.step(x[:, t], state)
            output.append(y)
        return torch.stack(output, 1), state

    def batched():
        return module(x)

    with torch.no_grad():
        expected, expected_state = token_steps()
        actual, actual_state = batched()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual_state.memory, expected_state.memory, rtol=2e-5, atol=2e-5)

    def evaluate(fn, backward):
        if backward:
            module.zero_grad(set_to_none=True)
            x.grad = None
            y, state = fn()
            (y.square().mean() + state.memory.square().mean()).backward()
        else:
            with torch.inference_mode():
                fn()

    cases = {}
    for name, backward in (("forward", False), ("forward_backward", True)):
        reference = measure(lambda backward=backward: evaluate(token_steps, backward), args.threads, args.minimum)
        batched_result = measure(lambda backward=backward: evaluate(batched, backward), args.threads, args.minimum)
        cases[name] = {
            "token_steps": reference,
            "batched_projections": batched_result,
            "speedup": reference["median_ms"] / batched_result["median_ms"],
        }
    return {
        "shape": list(x.shape),
        "heads": 4,
        "d_key": 16,
        "d_value": 16,
        "max_output_error": (actual - expected).abs().max().item(),
        "state_bytes": actual_state.memory.numel() * actual_state.memory.element_size(),
        "cases": cases,
    }


def decoding(args) -> dict:
    config = ModelConfig(
        vocab_size=1024,
        d_model=128,
        layers=("R", "P", "A", "R"),
        ffn_multiplier=2,
        memory_heads=4,
        memory_d_key=16,
        memory_d_value=16,
        attention_query_heads=4,
        attention_kv_heads=2,
        attention_head_dim=32,
        target_dim=32,
        modes=4,
        characteristic_features=8,
    )
    model = AELIALM(config).to(args.device).eval()
    tokens = torch.randint(0, config.vocab_size, (1, args.sequence + args.decode_tokens - 1), device=args.device)

    @torch.inference_mode()
    def full_prefix():
        outputs = []
        for end in range(args.sequence, tokens.shape[1] + 1):
            outputs.append(model(tokens[:, :end], logits_to_keep=1, return_predictive=False).logits)
        return torch.cat(outputs, 1)

    @torch.inference_mode()
    def cached():
        out = model(tokens[:, : args.sequence], use_cache=True, logits_to_keep=1, return_predictive=False)
        outputs = [out.logits]
        for position in range(args.sequence, tokens.shape[1]):
            out = model(
                tokens[:, position : position + 1],
                out.recurrent_states,
                attention_states=out.attention_states,
                use_cache=True,
                logits_to_keep=1,
                return_predictive=False,
            )
            outputs.append(out.logits)
        return torch.cat(outputs, 1), out

    expected = full_prefix()
    actual, out = cached()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    reference = measure(full_prefix, args.threads, args.minimum)
    cached_result = measure(cached, args.threads, args.minimum)
    kv_bytes = sum(
        s.keys.numel() * s.keys.element_size() + s.values.numel() * s.values.element_size()
        for s in out.attention_states
        if s is not None
    )
    return {
        "batch": 1,
        "prompt_tokens": args.sequence,
        "predicted_positions": args.decode_tokens,
        "model": {"d_model": config.d_model, "vocab_size": config.vocab_size, "layers": list(config.layers)},
        "full_prefix": reference,
        "cached": cached_result,
        "speedup": reference["median_ms"] / cached_result["median_ms"],
        "max_logit_error": (actual - expected).abs().max().item(),
        "kv_tensor_bytes": kv_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--minimum", type=float, default=1.0, help="Minimum seconds per timing case")
    parser.add_argument("--suite", choices=("all", "recurrent", "decode"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.threads, args.sequence, args.decode_tokens) < 1 or args.minimum <= 0:
        parser.error("threads, sequence, decode-tokens and minimum must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1729)
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        processor = next(
            (line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")),
            processor,
        )
    result = {
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": processor,
            "device": args.device,
            "threads": args.threads,
            "dtype": "float32",
            "seed": 1729,
        }
    }
    if args.suite in ("all", "recurrent"):
        result["recurrent"] = recurrent(args)
    if args.suite in ("all", "decode"):
        result["decoding"] = decoding(args)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
