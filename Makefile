.PHONY: setup sync test format build smoke

setup: sync

sync:
	uv sync

test:
	uv run pytest

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

build:
	uv build

smoke: build
	uv run --isolated --no-project --with dist/*.whl -- python -c "import uni_rl; print(uni_rl.__version__)"
