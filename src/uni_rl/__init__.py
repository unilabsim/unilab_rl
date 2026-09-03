"""uni_rl: RL algorithms and async runtimes extracted from UniLab."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uni-rl")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"
