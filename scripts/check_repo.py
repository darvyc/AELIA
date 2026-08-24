from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = [
        "README.md",
        "pyproject.toml",
        "LICENSE",
        "docs/SPECIFICATION.md",
        "src/aelia/model.py",
        "tests/test_model.py",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        print("Missing required files:", *missing, sep="\n  - ")
        return 1
    return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
