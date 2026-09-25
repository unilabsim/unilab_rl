"""WarpSAC migration tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from uni_rl.algos.common.actor_factory import build_actor
from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.warp_sac.double_buffer import build_warpsac_double_buffer_runner
from uni_rl.algos.warp_sac.learner import WarpSACLearner
from uni_rl.algos.warp_sac.replay import WarpSACReplayPipeline, _biased_replay_indices
from uni_rl.offpolicy.double_buffer_runner import algo_display_name
from uni_rl.offpolicy.worker import sample_offpolicy_actions


def _sample(**kwargs: Any) -> torch.Tensor:
    defaults = {
        "visible_size": 100,
        "capacity": 100,
        "current_ptr": 100,
        "sample_count": 20_000,
        "min_weight": 0.1,
        "num_buckets": 0,
        "device": torch.device("cpu"),
    }
    defaults.update(kwargs)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    return _biased_replay_indices(generator=generator, **defaults)


def test_replay_bias_prefers_recent_data_for_positive_decay() -> None:
    indices = _sample(decay_step=20)

    recent_mean = float(indices.float().mean())
    old_indices = _sample(decay_step=-20)
    old_mean = float(old_indices.float().mean())

    assert recent_mean > old_mean + 10.0


def test_negative_decay_keeps_sampling_close_to_uniform() -> None:
    indices = _sample(decay_step=-20)

    assert 40.0 < float(indices.float().mean()) < 60.0


def test_zero_min_weight_falls_back_when_all_weights_vanish() -> None:
    indices = _sample(decay_step=1, min_weight=0.0)

    assert indices.numel() == 20_000


def test_replay_bias_maps_logical_rows_onto_wrapped_ring() -> None:
    indices = _sample(
        visible_size=4,
        capacity=4,
        current_ptr=6,
        decay_step=2,
    )

    assert set(indices.tolist()) <= {0, 1, 2, 3}


def test_warpsac_learner_inherits_flashsac_training_interface() -> None:
    learner = WarpSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cpu",
    )
    batch = {
        "obs": torch.randn(8, 4),
        "critic": torch.randn(8, 6),
        "actions": torch.tanh(torch.randn(8, 2)),
        "rewards": torch.randn(8),
        "next_obs": torch.randn(8, 4),
        "next_critic": torch.randn(8, 6),
        "dones": torch.zeros(8),
        "truncated": torch.zeros(8),
    }

    assert isinstance(learner, FlashSACLearner)
    assert learner.update_critic(batch)
    assert learner.update_actor(batch)


def test_warpsac_can_disable_inherited_parameter_normalization() -> None:
    learner = WarpSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cpu",
        actor_normalize_parameters=False,
        critic_normalize_parameters=False,
    )
    actor_weight = learner.actor.embedder.norm.weight.detach().clone()
    critic_weight = learner.critic.embedder.norm.weight.detach().clone()
    target_weight = learner.target_critic.embedder.norm.weight.detach().clone()
    learner.actor.normalize_parameters()
    learner.critic.normalize_parameters()

    assert torch.equal(learner.actor.embedder.norm.weight, actor_weight)
    assert torch.equal(learner.critic.embedder.norm.weight, critic_weight)
    assert torch.equal(learner.target_critic.embedder.norm.weight, target_weight)
    assert torch.equal(target_weight, critic_weight)


def test_runtime_routes_warp_sac_exploration_and_display_name() -> None:
    actor = WarpSACLearner(
        obs_dim=3,
        action_dim=1,
        critic_obs_dim=3,
        actor_hidden_dim=4,
        critic_hidden_dim=4,
        actor_num_blocks=0,
        critic_num_blocks=0,
        num_atoms=3,
        device="cpu",
    ).actor

    action = sample_offpolicy_actions(actor, "warpsac", torch.randn(4, 3), torch.zeros(4))

    assert action.shape == (4, 1)
    assert algo_display_name("warpsac") == "WarpSAC"


def test_actor_factory_builds_warp_sac_actor() -> None:
    actor = build_actor(
        "warpsac",
        obs_dim=3,
        action_dim=1,
        actor_hidden_dim=4,
        use_layer_norm=False,
        device="cpu",
        actor_num_blocks=0,
    )

    action = actor.explore(torch.randn(4, 3))

    assert action.shape == (4, 1)


class _FakeRunner:
    last: "_FakeRunner | None" = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeRunner.last = self


class _FakeEnv:
    obs_groups_spec = {"obs": 4, "critic": 6}

    class action_space:
        shape = (2,)

    def close(self) -> None:
        pass


def _fake_env_factory(num_envs: int, cfg: dict[str, Any] | None = None) -> _FakeEnv:
    return _FakeEnv()


@pytest.mark.usefixtures("monkeypatch")
def test_warpsac_builder_uses_regime_aware_replay_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.warp_sac.double_buffer as module

    monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    cfg = OmegaConf.create(
        {
            "training": {
                "task_name": "fake",
                "sim_backend": "mujoco",
                "env_steps_per_sync": 1,
                "use_amp": False,
                "trace_enabled": False,
                "trace_output_dir": "logs",
                "trace_thread_time": False,
                "trace_cuda_events": False,
                "inference_request_timeout_sec": 17.0,
            },
            "algo": {
                "num_envs": 4,
                "replay_buffer_n": 8,
                "batch_size": 4,
                "learning_starts": 4,
                "updates_per_step": 1,
                "policy_frequency": 2,
                "seed": 1,
                "gamma": 0.99,
                "tau": 0.01,
                "actor_lr": 1e-3,
                "critic_lr": 1e-3,
                "actor_hidden_dim": 8,
                "critic_hidden_dim": 8,
                "num_atoms": 5,
                "obs_normalization": False,
                "decay_step": 123,
                "replay_min_weight": 0.2,
                "replay_num_buckets": 64,
                "algo_params": {
                    "actor_num_blocks": 1,
                    "critic_num_blocks": 1,
                    "critic_min_v": -5.0,
                    "critic_max_v": 5.0,
                    "temp_initial_value": 0.1,
                    "temp_target_sigma": 0.1,
                    "temp_target_entropy": 1.0,
                    "actor_bc_alpha": 0.0,
                    "actor_noise_zeta_mu": 2.0,
                    "actor_noise_zeta_max": 16,
                    "learning_rate_init": 1e-3,
                    "learning_rate_peak": 1e-3,
                    "learning_rate_end": 5e-4,
                    "learning_rate_warmup_steps": 0,
                    "learning_rate_decay_steps": 10,
                    "normalize_reward": False,
                    "normalized_g_max": 5.0,
                    "n_step": 1,
                    "amp_dtype": "bf16",
                    "use_compile": False,
                },
            },
        }
    )
    runner = build_warpsac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )

    replay_factory = runner.kwargs["replay_pipeline_factory"]
    assert replay_factory.func is WarpSACReplayPipeline
    assert replay_factory.keywords == {
        "decay_step": 123,
        "min_weight": 0.2,
        "num_buckets": 64,
    }
    assert runner.kwargs["algo_type"] == "warpsac"
    assert runner.kwargs["policy_before_critic"] is True
    assert runner.kwargs["target_frequency"] == 1
