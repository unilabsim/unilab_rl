from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

import uni_rl.offpolicy.worker as worker_module
from uni_rl.algos.common.collector_timing import extract_env_step_breakdown_timing_ms
from uni_rl.offpolicy.worker import (
    _publish_collector_ready,
    _publish_inference_tick,
    _wait_for_inference_tick,
    resolve_offpolicy_actor_priv_info,
    sample_offpolicy_actions,
)


def test_collector_publishes_ready_after_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})
            events.append("env_init")

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False

        def attach_stop_event(self, stop) -> None:
            del stop

    def publish_ready(coordination_queue, event) -> bool:
        assert not metrics_queue.empty()
        events.append("ready")
        event.set()
        assert _publish_collector_ready(coordination_queue, None)

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_publish_collector_ready", publish_ready)
    monkeypatch.setattr(
        worker_module,
        "_publish_inference_tick",
        lambda *args, **kwargs: events.append("inference_tick"),
    )

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=1,
        replay_buffer=_ReplayBuffer(),
        inference_slot=None,
        inference_request_queue=inference_request_queue,
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=None,
        trace_enabled=False,
        trace_thread_time=False,
    )

    assert events == ["env_init", "ready"]
    assert metrics_queue.get_nowait()["runtime_manifest"]["inference_owner"] == "learner"
    assert inference_request_queue.get_nowait() == worker_module.COLLECTOR_READY_TICK


def test_collector_binds_backend_device_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, ...]] = []

    class _EnvMaterializedError(RuntimeError):
        pass

    def bind_device(device: str) -> str:
        events.append(("bind", device))
        return device

    def env_factory(num_envs, env_cfg_override=None):
        del env_cfg_override
        events.append(("make", str(num_envs)))
        raise _EnvMaterializedError

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *args, **kwargs: None)

    with pytest.raises(_EnvMaterializedError):
        worker_module._run_collector(
            stop_event=None,
            env_factory=env_factory,
            num_envs=2,
            replay_buffer=None,
            inference_slot=None,
            inference_request_queue=None,
            inference_response_queue=None,
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=None,
            sim_backend="mjwarp",
            backend_device="cuda:3",
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
            backend_device_binder=bind_device,
        )

    # The real configure_backend_process_device resolves mjwarp -> cuda:3 and
    # must bind it before the env factory runs.
    assert events == [
        ("bind", "cuda:3"),
        ("make", "2"),
    ]


def test_inference_request_publish_timeout_is_explicit() -> None:
    requests: queue.Queue[int] = queue.Queue(maxsize=1)
    requests.put_nowait(0)

    with pytest.raises(TimeoutError, match="publishing off-policy inference tick 1"):
        _publish_inference_tick(requests, 1, threading.Event(), timeout=0.01)


def test_inference_response_wait_timeout_is_explicit() -> None:
    with pytest.raises(TimeoutError, match="waiting for off-policy inference tick 0"):
        _wait_for_inference_tick(queue.Queue(), 0, threading.Event(), timeout=0.01)


class _DummyActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        assert dones is not None
        self.calls.append((obs.clone(), dones.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def test_extract_env_step_breakdown_timing_ms_maps_env_owned_keys_only() -> None:
    timing = extract_env_step_breakdown_timing_ms(
        {
            "timing": {
                "step_core_ms": 1.5,
                "update_state_ms": 2.5,
                "reset_done_ms": 0.25,
                "apply_action_ms": 9.0,
            }
        }
    )

    assert timing == {
        "env_step_backend_ms": 1.5,
        "env_step_update_state_ms": 2.5,
        "env_step_reset_done_ms": 0.25,
    }


@pytest.mark.parametrize("algo_type", ["sac", "flashsac"])
def test_sample_offpolicy_actions_uses_actor_explore(algo_type: str) -> None:
    actor = _DummyActor()
    obs = torch.zeros(4, 5)
    dones = torch.zeros(4)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type=algo_type,
        obs_torch=obs,
        prev_dones_torch=dones,
    )

    assert len(actor.calls) == 1
    assert actor.calls[0][2] is False
    assert actions.shape == (4, 3)


def test_sample_offpolicy_actions_rejects_unknown_algo() -> None:
    actor = _DummyActor()

    with pytest.raises(ValueError, match="learner action sampling"):
        sample_offpolicy_actions(
            actor=actor,
            algo_type="unknown",
            obs_torch=torch.zeros(2, 4),
            prev_dones_torch=torch.zeros(2),
        )


class _DummyPrivInfoActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.calls.append((obs.clone(), priv_info.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def _register_dummy_priv_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from uni_rl.offpolicy import actor_adapter as actor_adapter_module

    def resolve_priv_info(obs_np, critic_np, info):
        if info is not None and info.get("critic_info") is not None:
            return np.asarray(info["critic_info"], dtype=np.float32)
        return np.asarray(critic_np[:, obs_np.shape[1] :], dtype=np.float32)

    monkeypatch.setitem(
        actor_adapter_module._ADAPTERS,
        "dummy_priv_sac",
        actor_adapter_module.OffPolicyActorAdapter(
            algo_type="dummy_priv_sac",
            sample_actions=lambda actor, obs, dones, priv: actor.explore(
                obs, priv, deterministic=False
            ),
            resolve_priv_info=resolve_priv_info,
        ),
    )


def test_sample_offpolicy_actions_uses_adapter_sample_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_dummy_priv_adapter(monkeypatch)
    actor = _DummyPrivInfoActor()
    obs = torch.zeros(4, 5)
    priv_info = torch.randn(4, 2)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type="dummy_priv_sac",
        obs_torch=obs,
        prev_dones_torch=torch.zeros(4),
        priv_info_torch=priv_info,
    )

    assert actions.shape == (4, 3)
    assert len(actor.calls) == 1
    torch.testing.assert_close(actor.calls[0][1], priv_info)


def test_resolve_offpolicy_actor_priv_info_prefers_explicit_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    _register_dummy_priv_adapter(monkeypatch)
    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.ones((2, 2), dtype=np.float32)
    critic = np.concatenate([obs, critic_tail], axis=1)
    explicit = np.full((2, 2), 7.0, dtype=np.float32)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="dummy_priv_sac",
        obs_np=obs,
        critic_np=critic,
        info={"critic_info": explicit},
    )

    np.testing.assert_allclose(resolved, explicit)


def test_resolve_offpolicy_actor_priv_info_uses_critic_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    _register_dummy_priv_adapter(monkeypatch)
    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.arange(4, dtype=np.float32).reshape(2, 2)
    critic = np.concatenate([obs, critic_tail], axis=1)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="dummy_priv_sac",
        obs_np=obs,
        critic_np=critic,
        info={},
    )

    np.testing.assert_allclose(resolved, critic_tail)


def test_resolve_offpolicy_actor_priv_info_returns_none_without_adapter() -> None:
    import numpy as np

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="sac",
        obs_np=np.zeros((2, 3), dtype=np.float32),
        critic_np=np.zeros((2, 5), dtype=np.float32),
        info=None,
    )

    assert resolved is None
