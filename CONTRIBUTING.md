# Contributing

AELIA contributions should preserve the distinction between architectural hypotheses, mathematical invariants, and engineering optimizations.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

## Pull-request requirements

A contribution that changes a core mathematical operator should include:

1. the equation or invariant being implemented;
2. a focused test for the invariant;
3. a numerical comparison against a dense or direct reference where possible;
4. an explanation of any approximation introduced;
5. measured compute or memory impact for performance-sensitive changes.

Large architectural additions should also identify the experiment that would falsify their value.

## Code style

- Python 3.10 or newer;
- type annotations for public functions;
- Ruff for linting and formatting;
- PyTorch operations should preserve batch-leading dimensions where practical;
- avoid silent changes in precision around likelihood, covariance, and calibration code.

## Research contributions

New density families, recurrent mixers, calibration tests, or routing algorithms should be introduced as controlled alternatives. Keep the simplest successful path available so the contribution can be evaluated at matched compute.
