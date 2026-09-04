.PHONY: setup sync sync-ci test format build smoke

setup: sync

sync:
	uv sync

# CI-only: skip the CUDA wheels (~2 GB of nvidia-*/triton) and install CPU
# torch instead; used by the mypy / pyright / test jobs.
CI_NO_INSTALL := $(addprefix --no-install-package ,torch triton \
	nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
	nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
	nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
	nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
	nvidia-nvjitlink-cu12 nvidia-nvtx-cu12)

sync-ci:
	uv sync $(CI_NO_INSTALL)
	uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

test:
	uv run pytest

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

build:
	uv build

smoke: build
	uv run --isolated --no-project --with dist/*.whl -- python -c "import uni_rl; print(uni_rl.__version__)"
