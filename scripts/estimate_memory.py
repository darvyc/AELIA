from __future__ import annotations

import argparse


def mib(x: float) -> float:
    return x / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate recurrent and exact-attention persistent state for AELIA.")
    parser.add_argument("--recurrent-layers", type=int, default=26)
    parser.add_argument("--recurrent-heads", type=int, default=16)
    parser.add_argument("--d-key", type=int, default=128)
    parser.add_argument("--d-value", type=int, default=128)
    parser.add_argument("--attention-layers", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bytes", type=int, default=2, dest="bytes_per_value", help="Bytes per K/V value")
    parser.add_argument("--recurrent-bytes", type=int, default=4, help="Bytes per recurrent accumulator")
    args = parser.parse_args()

    rec_values = args.recurrent_layers * args.recurrent_heads * args.d_key * args.d_value
    rec_bytes = rec_values * args.recurrent_bytes
    kv_per_token = 2 * args.attention_layers * args.kv_heads * args.head_dim * args.bytes_per_value
    cross = rec_bytes / kv_per_token if kv_per_token else float("inf")

    print(f"recurrent state: {rec_values:,} values = {mib(rec_bytes):.2f} MiB / sequence")
    print(f"exact KV state:   {kv_per_token:,} bytes / context token")
    print(f"crossover:        {cross:,.0f} context tokens")


if __name__ == "__main__":
    main()
