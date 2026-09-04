"""Tests for the torch actor factory helpers.

NOTE: UniLab's tests/utils/test_algo_utils.py additionally covers
``unilab.base.registry.ensure_registries``; that registry stays in UniLab
(issue #1479), so those tests remain there.
"""

from __future__ import annotations

import pytest

from uni_rl.algos.common.actor_factory import build_actor


class TestBuildActor:
    """Tests for build_actor."""

    def test_builds_sac_actor(self) -> None:
        actor = build_actor(
            algo_type="sac",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=256,
            use_layer_norm=True,
            device="cpu",
        )
        assert hasattr(actor, "forward")

    def test_builds_sac_actor_without_layer_norm(self) -> None:
        actor = build_actor(
            algo_type="sac",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=128,
            use_layer_norm=False,
            device="cpu",
        )
        assert hasattr(actor, "forward")

    def test_builds_td3_actor(self) -> None:
        actor = build_actor(
            algo_type="td3",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=256,
            use_layer_norm=True,
            device="cpu",
            num_envs=1,
        )
        assert hasattr(actor, "forward")

    def test_builds_td3_actor_for_multiple_envs(self) -> None:
        actor = build_actor(
            algo_type="td3",
            obs_dim=8,
            action_dim=2,
            actor_hidden_dim=128,
            use_layer_norm=False,
            device="cpu",
            num_envs=4,
        )
        assert hasattr(actor, "forward")

    def test_builds_flashsac_actor(self) -> None:
        actor = build_actor(
            algo_type="flashsac",
            obs_dim=98,
            action_dim=29,
            actor_hidden_dim=128,
            use_layer_norm=False,
            device="cpu",
            actor_num_blocks=2,
            actor_noise_zeta_mu=2.0,
            actor_noise_zeta_max=16,
        )
        assert hasattr(actor, "forward")
        assert hasattr(actor, "explore")

    def test_raises_for_unknown_algo_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown algo_type"):
            build_actor(
                algo_type="unknown",
                obs_dim=10,
                action_dim=4,
                actor_hidden_dim=256,
                use_layer_norm=True,
                device="cpu",
            )

    def test_builds_sac_actor_with_different_dims(self) -> None:
        actor = build_actor(
            algo_type="sac",
            obs_dim=49,
            action_dim=12,
            actor_hidden_dim=512,
            use_layer_norm=True,
            device="cpu",
        )
        assert hasattr(actor, "forward")
