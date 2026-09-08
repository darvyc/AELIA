"""Run with PYTHONPATH=src python scripts/benchmark_recurrent.py."""
import torch
from torch.utils.benchmark import Timer

from aelia.config import MemoryConfig
from aelia.recurrent import ContractiveDeltaMemory


def main():
    torch.manual_seed(42)
    module = ContractiveDeltaMemory(MemoryConfig(d_model=128, heads=4, d_key=16, d_value=16)).eval()
    x = torch.randn(4, 128, 128)

    @torch.no_grad()
    def sequence():
        return module(x)

    @torch.no_grad()
    def steps():
        state = module.initial_state(4, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            y, state = module.step(x[:, t], state)
            outputs.append(y)
        return torch.stack(outputs, 1), state

    y, state = sequence()
    reference, ref_state = steps()
    torch.testing.assert_close(y, reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state.memory, ref_state.memory, atol=1e-5, rtol=1e-5)
    results = {}
    for name, run in [('sequence', sequence), ('steps', steps)]:
        timing = Timer('run()', globals={'run': run}, num_threads=1).blocked_autorange(min_run_time=1.0)
        results[name] = timing.median
        print(f'{name}: {1000 * timing.median:.3f} ms; {512 / timing.median:.0f} tokens/s')
    print(f'Ratio (steps / sequence): {results["steps"] / results["sequence"]:.3f}')
    print(f'PyTorch {torch.__version__}; CPU; float32; batch=4; time=128; threads=1')


if __name__ == '__main__':
    main()
