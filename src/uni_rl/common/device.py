"""Probe env dims through an injected env factory (issue #1479)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from uni_rl.env_contract import EnvFactory
from uni_rl.utils.observations import get_obs_dims as _get_obs_dims


def get_env_dims(
    env_factory: EnvFactory,
    env_cfg_override: Mapping[str, Any] | None = None,
) -> tuple[int, int, int]:
    """Get (actor_obs_dim, action_dim, critic_obs_dim) from a probe env.

    Builds a one-env probe through the injected factory, reads the dims from
    the env contract, and closes the probe again. ``env_cfg_override`` is an
    opaque mapping forwarded to the factory.
    """
    env = env_factory(1, env_cfg_override)
    try:
        obs_dim, critic_dim = _get_obs_dims(dict(env.obs_groups_spec))
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = action_shape[0]
    finally:
        env.close()
    return obs_dim, action_dim, critic_dim


__all__ = ["get_env_dims"]
