from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from conftest import FakeVecEnv
from rsl_rl.runners import OnPolicyRunner

from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from uni_rl.algos.rsl_rl_runtime import RslRlPPORuntime, resolve_rsl_rl_ppo_runtime
from uni_rl.algos.rsl_rl_training_state import TrainingStateOnPolicyRunner


class _Progress:
    def __init__(self, steps: int = 0):
        self.state = {"schema": "example-v1", "steps": steps, "adaptive": [0.3, 0.8]}
        self.import_count = 0

    def export_training_state(self) -> Mapping[str, Any]:
        return self.state

    def import_training_state(self, state: Mapping[str, Any]) -> None:
        if state["schema"] != "example-v1":
            raise ValueError("Incompatible owner schema")
        self.state = deepcopy(dict(state))
        self.import_count += 1


class _StateWrapper(RslRlVecEnvWrapper):
    def __init__(self):
        self.progress = _Progress()
        super().__init__(FakeVecEnv(4, {"obs": 3, "critic": 4}, 2))

    def export_training_state(self) -> Mapping[str, Any]:
        return self.progress.export_training_state()

    def import_training_state(self, state: Mapping[str, Any]) -> None:
        self.progress.import_training_state(state)


def _config() -> dict[str, Any]:
    model = {"class_name": "MLPModel", "hidden_dims": [4], "obs_normalization": True}
    return {
        "actor": {
            **model,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
        },
        "critic": dict(model),
        "algorithm": {
            "class_name": "uni_rl.algos.rsl_rl_ppo:FinalObservationAwarePPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 2,
        "save_interval": 10,
    }


def _runner() -> TrainingStateOnPolicyRunner:
    runner = TrainingStateOnPolicyRunner(_StateWrapper(), _config())
    # RSL-RL 5.0 initializes its writer in learn(), unlike newer releases.
    # These checkpoint-only tests establish the same public logger lifecycle.
    runner.logger.init_logging_writer()
    return runner


def test_training_state_roundtrip_keeps_algorithm_optimizer_and_infos(tmp_path):
    runner = _runner()
    runner.learn(1)
    runner.current_learning_iteration = 7
    provider = runner.training_state_provider
    assert isinstance(provider, _StateWrapper)
    provider.progress.state["steps"] = 123
    infos = {"user": "retained"}
    path = str(tmp_path / "model.pt")
    runner.save(path, infos)
    assert infos == {"user": "retained"}
    provider.progress.state["steps"] = 999

    resumed = _runner()
    loaded_infos = resumed.load(path, map_location="cpu")
    resumed_provider = resumed.training_state_provider
    assert isinstance(resumed_provider, _StateWrapper)
    assert resumed_provider.progress.state == {
        "schema": "example-v1",
        "steps": 123,
        "adaptive": [0.3, 0.8],
    }
    assert resumed_provider.progress.import_count == 1
    assert loaded_infos["user"] == "retained"
    assert loaded_infos["uni_rl_training_state"]["version"] == 1
    assert resumed.current_learning_iteration == 7
    for key, value in runner.alg.actor.state_dict().items():
        assert torch.equal(value, resumed.alg.actor.state_dict()[key])
    original_optimizer = runner.alg.optimizer.state_dict()
    restored_optimizer = resumed.alg.optimizer.state_dict()
    assert original_optimizer["state"]
    assert original_optimizer["param_groups"] == restored_optimizer["param_groups"]
    for parameter, values in original_optimizer["state"].items():
        for key, value in values.items():
            restored = restored_optimizer["state"][parameter][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored)
            else:
                assert value == restored


def test_explicit_provider_is_used_instead_of_wrapper(tmp_path):
    provider = _Progress(42)
    wrapper = _StateWrapper()
    runner = TrainingStateOnPolicyRunner(wrapper, _config(), training_state_provider=provider)
    runner.logger.init_logging_writer()
    path = str(tmp_path / "model.pt")
    runner.save(path)
    provider.state["steps"] = 0
    runner.load(path)
    assert provider.state["steps"] == 42
    assert provider.import_count == 1
    assert wrapper.progress.import_count == 0


def test_state_runner_requires_declared_provider_without_env_discovery():
    # A provider behind .env is intentionally not discovered by this runner.
    wrapper = SimpleNamespace(env=_Progress())
    with pytest.raises(TypeError, match="TrainingStateProvider"):
        TrainingStateOnPolicyRunner(wrapper, _config())


@pytest.mark.parametrize(
    "envelope,match",
    [
        (None, "missing required training state"),
        ({"version": 2, "state": {}}, "envelope version"),
        ({"version": True, "state": {}}, "envelope version"),
        ({"version": 1, "state": []}, "must be a mapping"),
        ({"version": 1, "state": {"steps": float("nan")}}, "finite"),
    ],
)
def test_invalid_envelope_fails_before_loading_algorithm(tmp_path, envelope, match):
    runner = _runner()
    runner.current_learning_iteration = 99
    path = str(tmp_path / "model.pt")
    runner.save(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["infos"]["uni_rl_training_state"] = envelope
    torch.save(checkpoint, path)
    resumed = _runner()
    with pytest.raises(ValueError, match=match):
        resumed.load(path)
    assert resumed.current_learning_iteration == 0
    assert resumed.training_state_provider.progress.import_count == 0


def test_owner_schema_errors_are_not_swallowed(tmp_path):
    runner = _runner()
    runner.training_state_provider.progress.state["schema"] = "future-schema"
    path = str(tmp_path / "model.pt")
    runner.save(path)
    resumed = _runner()
    with pytest.raises(ValueError, match="Incompatible owner schema"):
        resumed.load(path)
    assert resumed.training_state_provider.progress.import_count == 0


@pytest.mark.parametrize("value", [torch.zeros(2), {1: "not a string key"}, object()])
def test_save_rejects_nonportable_owner_state_before_writing(tmp_path, value):
    runner = _runner()
    runner.training_state_provider.progress.state["invalid"] = value
    path = tmp_path / "model.pt"
    with pytest.raises(TypeError, match="Training state"):
        runner.save(str(path))
    assert not path.exists()


def test_save_rejects_reference_cycles(tmp_path):
    runner = _runner()
    state = runner.training_state_provider.progress.state
    state["self"] = state
    with pytest.raises(ValueError, match="reference cycles"):
        runner.save(str(tmp_path / "model.pt"))


def test_plain_runner_stays_unchanged_and_legacy_policy_load_is_explicit(tmp_path):
    plain = OnPolicyRunner(_StateWrapper(), _config())
    plain.logger.init_logging_writer()
    plain.current_learning_iteration = 5
    path = str(tmp_path / "plain.pt")
    plain.save(path)
    assert torch.load(path, weights_only=False)["infos"] is None
    another_plain = OnPolicyRunner(_StateWrapper(), _config())
    another_plain.load(path)
    assert another_plain.current_learning_iteration == 5

    resumed = _runner()
    with pytest.raises(ValueError, match="missing required training state"):
        resumed.load(path)
    resumed.load(path, load_cfg={"actor": True}, restore_training_state=False)
    assert resumed.training_state_provider.progress.import_count == 0
    for key, value in plain.alg.actor.state_dict().items():
        assert torch.equal(value, resumed.alg.actor.state_dict()[key])


@pytest.mark.parametrize("load_cfg", [None, {}, {"actor": True, "optimizer": True}])
def test_training_resume_cannot_opt_out_of_curriculum_state(load_cfg):
    with pytest.raises(ValueError, match="explicit actor-only"):
        _runner().load("unused.pt", load_cfg=load_cfg, restore_training_state=False)


def test_runtime_default_keeps_standard_runner_selection():
    runtime = resolve_rsl_rl_ppo_runtime({}, default_wrapper_cls=RslRlVecEnvWrapper)
    assert runtime.runner_cls is None
    assert runtime.wrapper_cls is RslRlVecEnvWrapper


@pytest.mark.parametrize("legacy", [False, True])
def test_runtime_custom_runner_and_legacy_wrapper_only_resolvers(monkeypatch, legacy):
    runtime = (
        SimpleNamespace(wrapper_cls=_StateWrapper)
        if legacy
        else RslRlPPORuntime(_StateWrapper, TrainingStateOnPolicyRunner)
    )
    monkeypatch.setattr("rsl_rl.utils.resolve_callable", lambda _path: lambda _cfg: runtime)
    resolved = resolve_rsl_rl_ppo_runtime(
        {"runtime_resolver": "example:runtime"}, default_wrapper_cls=RslRlVecEnvWrapper
    )
    assert resolved.wrapper_cls is _StateWrapper
    assert resolved.runner_cls is (None if legacy else TrainingStateOnPolicyRunner)


def test_runtime_rejects_nonclass_runner_selection(monkeypatch):
    runtime = SimpleNamespace(wrapper_cls=_StateWrapper, runner_cls="not-resolved")
    monkeypatch.setattr("rsl_rl.utils.resolve_callable", lambda _path: lambda _cfg: runtime)
    with pytest.raises(TypeError, match="runner_cls.*class or None"):
        resolve_rsl_rl_ppo_runtime(
            {"runtime_resolver": "example:runtime"}, default_wrapper_cls=RslRlVecEnvWrapper
        )
