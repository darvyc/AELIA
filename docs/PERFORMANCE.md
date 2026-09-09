# AELIA execution measurements

Measurements use FP32 on an AMD EPYC 9V74 CPU with one PyTorch thread,
PyTorch 2.14.0+cpu, Python 3.12.14, and random seed 1729. Inputs and weights are
synthetic. The benchmark checks numerical agreement before measuring each
implementation and uses `torch.utils.benchmark.Timer.blocked_autorange` with a
minimum of one second per case.

| Workload | Reference median | AELIA execution median | Ratio |
|---|---:|---:|---:|
| Recurrent forward, batched projections | 19.928 ms | 3.337 ms | 5.97x |
| Recurrent forward and backward, batched projections | 87.356 ms | 22.104 ms | 3.95x |
| Hybrid cached decoding, including prefill | 230.184 ms | 40.422 ms | 5.69x |

The recurrent reference evaluates the same module through its public `step`
operator for every token. The batched path evaluates token-local projections
together and scans the identical memory equation. Shape is `[2, 128, 128]`, with
four heads, key width 16, and value width 16. Both paths include normalization,
all projections, residuals, and state output. The backward case includes gradient
reset, forward evaluation, a scalar output/state objective, and backward.
Maximum forward difference is `2.38e-7`; persistent recurrent state is 8,192 bytes.

The decoding reference recomputes the full prefix for every predicted position.
Both paths project only the last token to the vocabulary and omit returned
predictive diagnostics. The cached path includes its initial prefill and carries
both recurrent and attention state. The model has layers `R, P, A, R`, hidden
width 128, vocabulary size 1,024, and a batch size of one. The prompt has 128
tokens and the benchmark evaluates 16 next-token distributions with fixed input
tokens. Maximum logit difference is `3.05e-5`. Final grouped-query K/V tensors
occupy 73,216 bytes, excluding metadata, recurrent states, and temporary buffers.

Raw medians, interquartile ranges, sample counts, and environment details are
stored in [benchmarks/cpu.json](benchmarks/cpu.json). These are operator and
small-model execution measurements. GPU performance, trained-model quality,
large-model training throughput, and total teacher-generation cost require
their own experiments. CPU ratios do not establish those outcomes.

## Reproduce

```bash
pip install -e ".[dev]"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark.py --output results.json
python scripts/benchmark.py --suite recurrent --sequence 512 --threads 4
python scripts/benchmark.py --suite decode --device cuda --sequence 512
```

`--minimum` sets the minimum measurement duration for each case. Report device,
thread count, dtype, shapes, medians, dispersion, and correctness tolerances
together. The CUDA command is provided for measurement on suitable hardware;
the recorded results use CPU execution.
