.PHONY: setup sync test format build smoke publish-test

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

publish-test: build
	UV_PUBLISH_URL=https://test.pypi.org/legacy/ UV_PUBLISH_TOKEN=$$(python3 -c "import configparser, os; c = configparser.ConfigParser(); c.read(os.path.expanduser('~/.pypirc')); print(c['testpypi']['password'])") uv publish dist/*
