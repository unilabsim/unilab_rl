"""Tests for the optional algo-capabilities extension point in env_contract."""

from __future__ import annotations

import numpy as np
from conftest import FakeVecEnv

from uni_rl.env_contract import (
    EnvAlgoCapabilities,
    EnvAlgoCapabilitiesProtocol,
    SupportsAlgoCapabilitiesProtocol,
    get_algo_capabilities,
)


def _make_env(algo_capabilities=None) -> FakeVecEnv:
    return FakeVecEnv(
        num_envs=2, obs_groups_spec={"obs": 4}, action_dim=3, algo_capabilities=algo_capabilities
    )


def test_get_algo_capabilities_default_when_env_does_not_provide():
    env = _make_env()
    assert not isinstance(env, SupportsAlgoCapabilitiesProtocol)

    caps = get_algo_capabilities(env)
    assert isinstance(caps, EnvAlgoCapabilitiesProtocol)
    assert caps.action_low is None
    assert caps.action_high is None
    assert caps.joint_names is None


def test_get_algo_capabilities_returns_provided_values():
    provided = EnvAlgoCapabilities(
        action_low=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        joint_names=("hip", "knee", "ankle"),
    )
    env = _make_env(algo_capabilities=provided)
    assert isinstance(env, SupportsAlgoCapabilitiesProtocol)

    caps = get_algo_capabilities(env)
    assert caps is provided
    np.testing.assert_array_equal(caps.action_low, provided.action_low)
    assert caps.action_high is None
    assert caps.joint_names == ("hip", "knee", "ankle")


def test_runtime_checkable_isinstance_semantics():
    assert isinstance(EnvAlgoCapabilities(), EnvAlgoCapabilitiesProtocol)

    class _Full:
        action_low = np.zeros((2,), dtype=np.float32)
        action_high = np.ones((2,), dtype=np.float32)
        joint_names = ("a", "b")

    class _MissingField:
        action_low = np.zeros((2,), dtype=np.float32)

    # runtime_checkable protocols check attribute presence only.
    assert isinstance(_Full(), EnvAlgoCapabilitiesProtocol)
    assert not isinstance(_MissingField(), EnvAlgoCapabilitiesProtocol)
    assert not isinstance(object(), EnvAlgoCapabilitiesProtocol)
    assert not isinstance(object(), SupportsAlgoCapabilitiesProtocol)


def test_default_carrier_is_frozen():
    caps = EnvAlgoCapabilities()
    try:
        caps.action_low = np.zeros((1,), dtype=np.float32)  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("EnvAlgoCapabilities must be frozen")
