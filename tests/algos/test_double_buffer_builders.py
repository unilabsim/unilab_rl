"""Forwarding tests for the double-buffer runner builder helpers.

The builders are the only public construction path for
``DoubleBufferOffPolicyRunner`` from Hydra owner configs; every runner kwarg
the builders accept must reach the runner unchanged (issue #1481 added
``backend_device_binder`` after UniLab had to set it post-construction).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf


class _FakeEnv:
    obs_groups_spec = {"obs": 4, "critic": 6}
    action_space = SimpleNamespace(shape=(2,))

    def close(self):
        return None


def _fake_env_factory(num_envs, env_cfg_override):
    del num_envs, env_cfg_override
    return _FakeEnv()


class _FakeLearner:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class _FakeRunner:
    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs


def _binder(backend: str) -> str | None:
    del backend
    return None


def _training_cfg() -> dict[str, Any]:
    return {
        "task_name": "FakeTask",
        "sim_backend": "fake",
        "env_steps_per_sync": 1,
        "use_amp": False,
        "trace_enabled": False,
        "trace_output_dir": "logs",
        "trace_thread_time": False,
        "trace_cuda_events": False,
    }


def _algo_cfg(extra: dict[str, Any]) -> dict[str, Any]:
    base = {
        "num_envs": 4,
        "replay_buffer_n": 8,
        "batch_size": 4,
        "learning_starts": 4,
        "updates_per_step": 1,
        "policy_frequency": 1,
        "seed": 1,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "num_atoms": 1,
        "obs_normalization": False,
    }
    base.update(extra)
    return base


def _sac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "use_layer_norm": False,
                    "algo_params": {
                        "alpha_lr": 1e-3,
                        "alpha_init": 1.0,
                        "target_entropy_ratio": 1.0,
                        "max_grad_norm": 1.0,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                    },
                }
            ),
        }
    )


def _td3_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "algo_params": {
                        "v_min": -10.0,
                        "v_max": 10.0,
                        "init_scale": 1.0,
                        "log_std_min": -5.0,
                        "log_std_max": 2.0,
                        "weight_decay": 0.0,
                        "use_cdq": True,
                        "policy_noise": 0.2,
                        "noise_clip": 0.5,
                    },
                }
            ),
        }
    )


def _flashsac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "algo_params": {
                        "actor_num_blocks": 1,
                        "critic_num_blocks": 1,
                        "critic_min_v": -10.0,
                        "critic_max_v": 10.0,
                        "temp_initial_value": 1.0,
                        "temp_target_sigma": 1.0,
                        "temp_target_entropy": 1.0,
                        "actor_bc_alpha": 1.0,
                        "actor_noise_zeta_mu": 0.0,
                        "actor_noise_zeta_max": 0.0,
                        "learning_rate_init": 1e-3,
                        "learning_rate_peak": 1e-3,
                        "learning_rate_end": 1e-3,
                        "learning_rate_warmup_steps": 1,
                        "learning_rate_decay_steps": 1,
                        "normalize_reward": False,
                        "normalized_g_max": 1.0,
                        "n_step": 1,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                        "use_cuda_graph_critic": False,
                        "use_cuda_graph_actor": False,
                        "use_cuda_graph_critic_packed_staging": False,
                        "use_cuda_graph_actor_packed_staging": False,
                    },
                }
            ),
        }
    )


@pytest.mark.parametrize("with_binder", [False, True])
def test_sac_builder_forwards_backend_device_binder(
    monkeypatch: pytest.MonkeyPatch, with_binder: bool
) -> None:
    import uni_rl.fast_sac.double_buffer as module

    monkeypatch.setattr(module, "FastSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    kwargs: dict[str, Any] = {}
    if with_binder:
        kwargs["backend_device_binder"] = _binder
    runner = module.build_sac_double_buffer_runner(
        _sac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_binder"] is (_binder if with_binder else None)


@pytest.mark.parametrize("with_binder", [False, True])
def test_td3_builder_forwards_backend_device_binder(
    monkeypatch: pytest.MonkeyPatch, with_binder: bool
) -> None:
    import uni_rl.fast_td3.double_buffer as module

    monkeypatch.setattr(module, "FastTD3Learner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    kwargs: dict[str, Any] = {}
    if with_binder:
        kwargs["backend_device_binder"] = _binder
    runner = module.build_td3_double_buffer_runner(
        _td3_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_binder"] is (_binder if with_binder else None)


@pytest.mark.parametrize("with_binder", [False, True])
def test_flashsac_builder_forwards_backend_device_binder(
    monkeypatch: pytest.MonkeyPatch, with_binder: bool
) -> None:
    import uni_rl.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    # CPU-only test host: bypass the CUDA/MPS replay-device gate and seeding.
    monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)

    kwargs: dict[str, Any] = {}
    if with_binder:
        kwargs["backend_device_binder"] = _binder
    runner = module.build_flashsac_double_buffer_runner(
        _flashsac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_binder"] is (_binder if with_binder else None)
