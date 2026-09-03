"""Shared fixtures for uni_rl tests."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch


@pytest.fixture
def mp_ctx():
    return torch.multiprocessing.get_context("spawn")


@pytest.fixture
def tiny_weight_shapes():
    """Small MLP param shapes dict — linear(8,16) + bias, linear(16,3) + bias."""
    return {
        "layer1.weight": torch.Size([16, 8]),
        "layer1.bias": torch.Size([16]),
        "layer2.weight": torch.Size([3, 16]),
        "layer2.bias": torch.Size([3]),
    }


class _FakeSpace:
    """Gym-style space stub; the env contract only reads ``.shape``."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


@dataclass
class _FakeEnvState:
    """Minimal ``EnvStateProtocol`` implementation (random obs, zero reward)."""

    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    final_observation: dict[str, np.ndarray] | None = None

    def replace(self, **updates: Any) -> "_FakeEnvState":
        return dataclasses.replace(self, **updates)


class _FakePlayCapabilities:
    supports_physics_state_playback = False


class FakeVecEnv:
    """Minimal ``EnvProtocol`` implementation for runner/collector unit tests."""

    def __init__(
        self,
        num_envs: int,
        obs_groups_spec: dict[str, int],
        action_dim: int,
    ) -> None:
        self._num_envs = num_envs
        self._obs_groups_spec = dict(obs_groups_spec)
        self._action_dim = action_dim
        self._state: _FakeEnvState | None = None
        self.closed = False
        self.cfg = type("Cfg", (), {"max_episode_seconds": 1.0, "ctrl_dt": 0.02})()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return dict(self._obs_groups_spec)

    @property
    def observation_space(self) -> _FakeSpace:
        return _FakeSpace((sum(self._obs_groups_spec.values()),))

    @property
    def action_space(self) -> _FakeSpace:
        return _FakeSpace((self._action_dim,))

    @property
    def state(self) -> _FakeEnvState | None:
        return self._state

    @property
    def play_capabilities(self) -> _FakePlayCapabilities:
        return _FakePlayCapabilities()

    def init_state(self) -> _FakeEnvState:
        self._state = _FakeEnvState(
            obs={
                k: np.zeros((self._num_envs, d), dtype=np.float32)
                for k, d in self._obs_groups_spec.items()
            },
            reward=np.zeros((self._num_envs,), dtype=np.float32),
            terminated=np.zeros((self._num_envs,), dtype=bool),
            truncated=np.zeros((self._num_envs,), dtype=bool),
            info={"steps": np.zeros((self._num_envs,), dtype=np.uint32)},
        )
        return self._state

    def step(self, actions: np.ndarray) -> _FakeEnvState:
        if self._state is None:
            self.init_state()
        assert self._state is not None
        return self._state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        if self._state is None:
            self.init_state()
        n = len(env_indices)
        return (
            {k: np.zeros((n, d), dtype=np.float32) for k, d in self._obs_groups_spec.items()},
            {},
        )

    def set_nan_guard(self, guard: Any) -> None:
        del guard

    def close(self) -> None:
        self.closed = True


class FakeEnvFactory:
    """Picklable ``EnvFactory`` producing :class:`FakeVecEnv` probes.

    Top-level class so it survives spawn-process pickling like a real
    registry-backed factory would.
    """

    def __init__(
        self,
        obs_groups_spec: dict[str, int] | None = None,
        action_dim: int = 2,
    ) -> None:
        self.obs_groups_spec = dict(obs_groups_spec or {"obs": 4, "critic": 7})
        self.action_dim = action_dim
        self.calls: list[tuple[int, Any]] = []

    def __call__(self, num_envs: int, env_cfg_override: Any = None) -> FakeVecEnv:
        self.calls.append((num_envs, env_cfg_override))
        return FakeVecEnv(num_envs, self.obs_groups_spec, self.action_dim)


@pytest.fixture
def fake_env_factory() -> FakeEnvFactory:
    return FakeEnvFactory()
