.PHONY: install test lint format check example

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check src tests examples scripts

format:
	python -m ruff format src tests examples scripts

check: lint test

example:
	python examples/minimal_forward.py
