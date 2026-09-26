"""Off-policy learner-owned cold-path preparation contracts."""

from __future__ import annotations

import copy
import random
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from uni_rl.algos.fast_sac.learner import FastSACLearner
from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.warp_sac.learner import WarpSACLearner
from uni_rl.offpolicy.warmup import OffPolicyWarmupContext


def _context(
    *,
    obs_dim: int = 4,
    action_dim: int = 2,
    batch_size: int = 3,
) -> OffPolicyWarmupContext:
    return OffPolicyWarmupContext(
        inference_observations=torch.zeros(batch_size, obs_dim + action_dim),
        inference_dones=torch.zeros(batch_size),
        batch_size=batch_size,
        updates_per_step=2,
        policy_frequency=1,
        target_frequency=1,
        policy_before_critic=False,
    )


def _assert_equivalent(left: Any, right: Any, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_equivalent(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list | tuple):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, values in enumerate(zip(left, right, strict=True)):
            _assert_equivalent(values[0], values[1], f"{path}[{index}]")
        return
    assert left == right, path


@pytest.mark.parametrize(
    "learner",
    [
        FastSACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            device="cpu",
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            num_atoms=3,
            use_layer_norm=False,
        ),
        FlashSACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=6,
            device="cpu",
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            actor_num_blocks=1,
            critic_num_blocks=1,
            num_atoms=5,
            use_compile=False,
        ),
        WarpSACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=6,
            device="cpu",
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            actor_num_blocks=1,
            critic_num_blocks=1,
            num_atoms=5,
            use_compile=False,
        ),
    ],
    ids=["fastsac", "flashsac", "warpsac"],
)
def test_compatibility_warmup_restores_complete_learner_state(learner: Any) -> None:
    torch.manual_seed(321)
    random.seed(321)
    learner._optimizer_found_inf.zero_()
    if hasattr(learner, "_q_update_finite"):
        learner._q_update_finite.zero_()
    if hasattr(learner, "_critic_update_finite"):
        learner._critic_update_finite.zero_()
    learner._pending_actor_metric_values = torch.zeros(1)
    learner._pending_cycle_critic_metric_values = torch.zeros(1)
    learner._pending_cycle_metric_values = torch.zeros(1)
    saved_state = copy.deepcopy(learner.get_state_dict())
    saved_cpu_rng = torch.random.get_rng_state()
    saved_python_rng = random.getstate()

    learner.prepare_for_collection(_context())

    _assert_equivalent(saved_state, learner.get_state_dict())
    torch.testing.assert_close(torch.random.get_rng_state(), saved_cpu_rng)
    assert random.getstate() == saved_python_rng
    assert learner.update_count == 0
    assert bool(learner._optimizer_found_inf.item()) is False
    assert learner._pending_actor_metric_values is None
    assert learner._pending_cycle_critic_metric_values is None
    assert learner._pending_cycle_metric_values is None
    if hasattr(learner, "_q_update_finite"):
        assert bool(learner._q_update_finite.item()) is False
    if hasattr(learner, "_critic_update_finite"):
        assert bool(learner._critic_update_finite.item()) is False


def test_flashsac_whole_cycle_warmup_captures_representative_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        use_compile=False,
    )
    calls: list[dict[str, Any]] = []
    original_state = copy.deepcopy(learner.get_state_dict())

    def capture_graph(large_batch: dict[str, torch.Tensor], **kwargs: Any) -> None:
        calls.append({"batch": large_batch, "kwargs": kwargs})

    monkeypatch.setattr(learner, "_compile_full_update_cycle", True)
    monkeypatch.setattr(learner, "_ensure_update_cycle_graph", capture_graph)
    learner.prepare_for_collection(_context(batch_size=4))

    assert len(calls) == 1
    assert calls[0]["kwargs"] == {
        "updates_per_step": 2,
        "policy_frequency": 1,
        "target_frequency": 1,
        "policy_before_critic": False,
    }
    assert {key: tuple(value.shape) for key, value in calls[0]["batch"].items()} == {
        "obs": (8, 4),
        "actions": (8, 2),
        "rewards": (8,),
        "next_obs": (8, 4),
        "dones": (8,),
        "truncated": (8,),
        "critic": (8, 6),
        "next_critic": (8, 6),
    }
    _assert_equivalent(original_state, learner.get_state_dict())


def test_fastsac_whole_cycle_warmup_captures_before_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        use_layer_norm=False,
    )
    called = False

    def capture_graph(large_batch: dict[str, torch.Tensor], **kwargs: Any) -> None:
        nonlocal called
        called = True
        assert large_batch["obs"].shape == (8, 4)
        assert kwargs["updates_per_step"] == 2

    monkeypatch.setattr(learner, "_compile_full_update_cycle", True)
    monkeypatch.setattr(learner, "_ensure_update_cycle_graph", capture_graph)
    original_state = copy.deepcopy(learner.get_state_dict())
    learner.prepare_for_collection(_context(batch_size=4))

    assert called
    _assert_equivalent(original_state, learner.get_state_dict())


@pytest.mark.parametrize(
    ("module_name", "learner_cls"),
    [
        ("uni_rl.algos.fast_sac.learner", FastSACLearner),
        ("uni_rl.algos.flash_sac.learner", FlashSACLearner),
    ],
    ids=["fastsac", "flashsac"],
)
def test_compatibility_warmup_exercises_compiled_update_paths(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    learner_cls: type[Any],
) -> None:
    module = pytest.importorskip(module_name)
    invoked: set[str] = set()

    def compile_fn(function: Any, **kwargs: Any) -> Any:
        name = getattr(function, "__name__", "<unnamed>")
        assert kwargs["dynamic"] is False

        def compiled(*args: Any, **call_kwargs: Any) -> Any:
            invoked.add(name)
            return function(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(module, "get_torch_compile_for_cuda", lambda *args, **kwargs: compile_fn)
    learner_kwargs = (
        {"critic_obs_dim": 5, "num_atoms": 3, "use_layer_norm": False}
        if learner_cls is FastSACLearner
        else {
            "critic_obs_dim": 6,
            "actor_num_blocks": 1,
            "critic_num_blocks": 1,
            "num_atoms": 5,
        }
    )
    learner = learner_cls(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        use_compile=True,
        **learner_kwargs,
    )

    assert learner.use_compile is True
    learner.prepare_for_collection(_context())

    assert {"_critic_loss_tensors", "_actor_loss_tensors"}.issubset(invoked)


def test_runner_prepare_uses_custom_learner_and_runtime_hooks() -> None:
    from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    calls: list[str] = []
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.algo_type = "custom_runtime_sac"
    runner.obs_normalization = False
    runner.obs_dim = 4
    runner.batch_size = 2
    runner.updates_per_step = 1
    runner.policy_frequency = 1
    runner.target_frequency = 1
    runner.policy_before_critic = False
    runner.learner = type(
        "Learner",
        (),
        {
            "actor": object(),
            "prepare_for_collection": staticmethod(lambda context: calls.append("learner")),
        },
    )()
    runner.learner_prepare_hook = lambda learner, context: calls.append("runtime")

    class Pipeline:
        def warmup(self) -> None:
            calls.append("replay_pipeline")

    runner._prepare_learner(
        inference_observations=torch.zeros(2, 6),
        inference_dones=torch.zeros(2),
        replay_pipeline=Pipeline(),
    )

    assert calls == ["replay_pipeline", "learner", "runtime"]


def test_actor_adapter_warmup_hook_is_preferred_and_rng_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.offpolicy.actor_adapter as actor_adapter_module
    from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

    calls: list[int] = []

    def warmup(actor, observations, dones, context):
        del actor, dones, context
        calls.append(1)
        return torch.rand(observations.shape[0], 3)

    monkeypatch.setitem(
        actor_adapter_module._ADAPTERS,
        "warm_custom_sac",
        actor_adapter_module.OffPolicyActorAdapter(
            algo_type="warm_custom_sac",
            sample_actions=lambda *_: (_ for _ in ()).throw(AssertionError()),
            warmup_actions=warmup,
        ),
    )

    class Actor:
        pass

    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.device = "cpu"
    runner.obs_dim = 4
    runner.obs_normalization = False
    runner.algo_type = "warm_custom_sac"
    runner.learner = SimpleNamespace(actor=Actor())
    before_cpu = torch.random.get_rng_state()
    before_python = random.getstate()

    runner._warm_representative_actor(_context())

    assert calls == [1]
    torch.testing.assert_close(torch.random.get_rng_state(), before_cpu)
    assert random.getstate() == before_python


def test_gpu_resident_replay_pipeline_warmup_uses_scratch_gather_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uni_rl.ipc.replay_pipelines.gpu_resident import GPUResidentReplayPipeline

    pipeline = object.__new__(GPUResidentReplayPipeline)
    pipeline._closed = False
    pipeline._main_thread_submission = True
    pipeline._learner_thread_id = threading.get_ident()
    pipeline._device = torch.device("cpu")
    pipeline._capacity = 32
    pipeline._sample_count = 8
    pipeline._base_seed = 99
    pipeline._cold = 1
    pipeline._gpu_packed = {
        0: torch.zeros(8, 12),
        1: torch.empty(8, 12),
    }
    gathered: list[tuple[int, int]] = []

    def gather_rows(*, visible_size: int, slot: int, gen: torch.Generator) -> None:
        gathered.append((visible_size, slot))
        assert isinstance(gen, torch.Generator)

    monkeypatch.setattr(pipeline, "_gather_rows", gather_rows)
    monkeypatch.setattr(
        pipeline, "_large_batch_view", lambda slot: {"obs": pipeline._gpu_packed[slot]}
    )
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)

    pipeline.warmup_result = pipeline.warmup()

    assert gathered == [(8, 1)]
    warmup_batch = pipeline.warmup_result
    assert warmup_batch is not None
    assert warmup_batch["obs"].shape == (8, 12)
    assert torch.count_nonzero(warmup_batch["obs"]) == 0
