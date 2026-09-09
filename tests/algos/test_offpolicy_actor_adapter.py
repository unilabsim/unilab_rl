"""Unit tests for the off-policy actor adapter extension registry."""

from __future__ import annotations

import sys

import pytest
import torch

import uni_rl.offpolicy.actor_adapter as actor_adapter_module
from uni_rl.offpolicy.actor_adapter import (
    OffPolicyActorAdapter,
    get_offpolicy_actor_adapter,
    import_actor_adapter_modules,
    register_offpolicy_actor_adapter,
)


@pytest.fixture()
def registered_adapter(monkeypatch: pytest.MonkeyPatch):
    def register(adapter: OffPolicyActorAdapter) -> OffPolicyActorAdapter:
        monkeypatch.setitem(actor_adapter_module._ADAPTERS, adapter.algo_type, adapter)
        return adapter

    return register


def test_register_and_get_adapter(registered_adapter) -> None:
    adapter = registered_adapter(OffPolicyActorAdapter(algo_type="custom_sac"))

    assert get_offpolicy_actor_adapter("custom_sac") is adapter


def test_register_rejects_duplicate_algo_type() -> None:
    adapter = OffPolicyActorAdapter(algo_type="dup_sac")
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setitem(actor_adapter_module._ADAPTERS, "dup_sac", adapter)
        with pytest.raises(ValueError, match="already registered.*dup_sac"):
            register_offpolicy_actor_adapter(OffPolicyActorAdapter(algo_type="dup_sac"))


def test_register_rejects_non_adapter() -> None:
    with pytest.raises(TypeError, match="OffPolicyActorAdapter"):
        register_offpolicy_actor_adapter(object())  # type: ignore[arg-type]


def test_register_rejects_empty_algo_type() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        register_offpolicy_actor_adapter(OffPolicyActorAdapter(algo_type=""))


def test_get_adapter_returns_none_for_unknown_algo_type() -> None:
    assert get_offpolicy_actor_adapter("definitely_not_registered") is None


def test_import_actor_adapter_modules_triggers_registration(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "tmp_spawn_adapter_mod"
    (tmp_path / f"{module_name}.py").write_text(
        "from uni_rl.offpolicy.actor_adapter import (\n"
        "    OffPolicyActorAdapter,\n"
        "    register_offpolicy_actor_adapter,\n"
        ")\n"
        "register_offpolicy_actor_adapter(OffPolicyActorAdapter(algo_type='spawn_test_sac'))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        import_actor_adapter_modules([module_name])
        assert get_offpolicy_actor_adapter("spawn_test_sac") is not None
    finally:
        actor_adapter_module._ADAPTERS.pop("spawn_test_sac", None)
        sys.modules.pop(module_name, None)


def test_import_actor_adapter_modules_accepts_none_and_empty() -> None:
    import_actor_adapter_modules(None)
    import_actor_adapter_modules(())


def test_import_actor_adapter_modules_error_is_actionable() -> None:
    with pytest.raises(ImportError, match="actor_adapter_modules entry 'no_such_pkg_xyz'"):
        import_actor_adapter_modules(["no_such_pkg_xyz"])


def test_build_actor_uses_registered_adapter(registered_adapter) -> None:
    from uni_rl.algos.common.actor_factory import build_actor

    captured: dict = {}

    def fake_build_actor(**kwargs):
        captured.update(kwargs)
        return "custom-actor"

    registered_adapter(OffPolicyActorAdapter(algo_type="custom_sac", build_actor=fake_build_actor))

    actor = build_actor(
        algo_type="custom_sac",
        obs_dim=10,
        action_dim=4,
        actor_hidden_dim=256,
        use_layer_norm=True,
        device="cpu",
        priv_info_dim=3,
        priv_info_embed_dim=9,
        priv_mlp_hidden_dims=(16, 9),
    )

    assert actor == "custom-actor"
    assert captured == {
        "obs_dim": 10,
        "action_dim": 4,
        "actor_hidden_dim": 256,
        "use_layer_norm": True,
        "device": "cpu",
        "priv_info_dim": 3,
        "priv_info_embed_dim": 9,
        "priv_mlp_hidden_dims": (16, 9),
    }


def test_build_actor_rejects_adapter_without_build_hook(registered_adapter) -> None:
    from uni_rl.algos.common.actor_factory import build_actor

    registered_adapter(OffPolicyActorAdapter(algo_type="custom_sac"))

    with pytest.raises(ValueError, match="does not provide build_actor"):
        build_actor(
            algo_type="custom_sac",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=256,
            use_layer_norm=True,
            device="cpu",
        )


def test_build_actor_unknown_algo_error_mentions_adapter_registration() -> None:
    from uni_rl.algos.common.actor_factory import build_actor

    with pytest.raises(ValueError, match="register_offpolicy_actor_adapter"):
        build_actor(
            algo_type="unknown",
            obs_dim=10,
            action_dim=4,
            actor_hidden_dim=256,
            use_layer_norm=True,
            device="cpu",
        )


def test_sample_actions_uses_adapter_hook(registered_adapter) -> None:
    from uni_rl.offpolicy.worker import sample_offpolicy_actions

    calls: list[tuple] = []

    def sample_actions(actor, obs, prev_dones, priv_info):
        calls.append((actor, obs, prev_dones, priv_info))
        return obs[:, :2]

    registered_adapter(OffPolicyActorAdapter(algo_type="custom_sac", sample_actions=sample_actions))

    obs = torch.zeros(4, 5)
    dones = torch.zeros(4)
    priv = torch.ones(4, 2)
    actions = sample_offpolicy_actions(
        actor="the-actor",
        algo_type="custom_sac",
        obs_torch=obs,
        prev_dones_torch=dones,
        priv_info_torch=priv,
    )

    assert actions.shape == (4, 2)
    assert calls == [("the-actor", obs, dones, priv)]


def test_sample_actions_rejects_adapter_without_sample_hook(registered_adapter) -> None:
    from uni_rl.offpolicy.worker import sample_offpolicy_actions

    registered_adapter(OffPolicyActorAdapter(algo_type="custom_sac"))

    with pytest.raises(ValueError, match="does not provide sample_actions"):
        sample_offpolicy_actions(
            actor=object(),
            algo_type="custom_sac",
            obs_torch=torch.zeros(2, 4),
            prev_dones_torch=torch.zeros(2),
        )


def test_sample_actions_unknown_algo_error_mentions_adapter_registration() -> None:
    from uni_rl.offpolicy.worker import sample_offpolicy_actions

    with pytest.raises(ValueError, match="register_offpolicy_actor_adapter"):
        sample_offpolicy_actions(
            actor=object(),
            algo_type="unknown",
            obs_torch=torch.zeros(2, 4),
            prev_dones_torch=torch.zeros(2),
        )


def test_actor_context_from_obs_hook(registered_adapter) -> None:
    obs_device = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    adapter = registered_adapter(
        OffPolicyActorAdapter(
            algo_type="custom_sac",
            actor_context_from_obs=lambda obs, obs_dim: obs[:, obs_dim:],
        )
    )

    context = adapter.actor_context_from_obs(obs_device, 3)
    torch.testing.assert_close(context, obs_device[:, 3:])

    # An adapter without the hook yields no actor context.
    plain = registered_adapter(OffPolicyActorAdapter(algo_type="plain_sac"))
    assert plain.actor_context_from_obs is None
