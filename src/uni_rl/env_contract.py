"""THE env contract consumed by uni_rl's algo and runtime layer.

uni_rl never constructs environments itself and never imports an env
registry. Environments are injected by the caller (e.g. UniLab's training
scripts) as an :data:`EnvFactory`. Any vectorized env satisfying the
structural protocols below can drive uni_rl collectors and runners — the
protocols capture exactly the attributes the algo code reads, no more.

Contract summary (numpy-based, autoresetting vectorized env):

- ``step(actions)`` returns a state with ``obs`` (dict of ``(num_envs, dim)``
  arrays keyed by observation group), ``reward``, ``terminated``,
  ``truncated``, ``info`` and an optional ``final_observation`` holding the
  pre-reset observations of envs that terminated this step.
- ``obs_groups_spec`` maps observation group name -> dim, e.g.
  ``{"obs": 48}`` or ``{"obs": 48, "critic": 101}``. The ``"obs"`` group is
  the actor input; ``"critic"`` is the optional privileged critic input.
- ``reset(env_indices)`` returns ``(obs_dict, info_dict)`` for the selected
  envs; ``init_state()`` performs cold-path initialization and returns the
  first state; ``state`` is ``None`` before ``init_state()``.

Because collectors run in ``multiprocessing`` spawn subprocesses, an
``EnvFactory`` must be picklable by reference — use a top-level function or
``functools.partial`` of one, never a closure or lambda.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EnvStateProtocol(Protocol):
    """One vectorized env step/reset result (structural contract)."""

    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None


@runtime_checkable
class EnvPlayCapabilitiesProtocol(Protocol):
    """Playback capabilities read by NaN-guard wiring in collectors."""

    supports_physics_state_playback: bool


@runtime_checkable
class EnvProtocol(Protocol):
    """Vectorized numpy env consumed by uni_rl runners and collectors."""

    @property
    def num_envs(self) -> int: ...

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """Observation group dims, e.g. ``{"obs": 48, "critic": 101}``."""
        ...

    @property
    def observation_space(self) -> Any:
        """Gym-style flat observation space (only ``.shape`` is read)."""
        ...

    @property
    def action_space(self) -> Any:
        """Gym-style action space (only ``.shape`` is read)."""
        ...

    @property
    def state(self) -> EnvStateProtocol | None:
        """Current state; ``None`` before :meth:`init_state` is called."""
        ...

    @property
    def cfg(self) -> Any:
        """Env config; runners read ``max_episode_seconds`` and ``ctrl_dt``."""
        ...

    @property
    def play_capabilities(self) -> EnvPlayCapabilitiesProtocol: ...

    def init_state(self) -> EnvStateProtocol: ...

    def step(self, actions: np.ndarray) -> EnvStateProtocol: ...

    def reset(
        self, env_indices: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def set_nan_guard(self, guard: Any) -> None: ...

    def close(self) -> None: ...


# Injected env constructor. Arguments are ``(num_envs, env_cfg_override)``;
# the override mapping is opaque to uni_rl and interpreted by the factory's
# owner (e.g. UniLab's registry-backed factory). One factory serves both the
# learner-side dim probe (``num_envs=1``) and collector construction
# (``num_envs=N``). Must be picklable — see module docstring.
EnvFactory = Callable[[int, Mapping[str, Any] | None], EnvProtocol]

__all__ = [
    "EnvFactory",
    "EnvPlayCapabilitiesProtocol",
    "EnvProtocol",
    "EnvStateProtocol",
]
