from __future__ import annotations

import pytest


def test_offpolicy_runtime_defaults_to_standard_sac_overrides() -> None:
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    runtime = OffPolicyRuntime()

    assert runtime.learner_cls is None
    assert runtime.algo_type is None
    assert runtime.actor_adapter_modules == ()
    assert runtime.build_model_kwargs(obs_dim=4, critic_obs_dim=6) == {}


def test_offpolicy_runtime_rejects_marker_without_resolver() -> None:
    from uni_rl.offpolicy.runtime import resolve_custom_offpolicy_runtime

    with pytest.raises(ValueError, match="runtime_impl='custom_sac'.*runtime_resolver"):
        resolve_custom_offpolicy_runtime({"runtime_impl": "custom_sac"})


def test_resolve_actor_adapter_modules_combines_config_and_runtime() -> None:
    from uni_rl.offpolicy.runtime import OffPolicyRuntime, resolve_actor_adapter_modules

    runtime = OffPolicyRuntime(actor_adapter_modules=("pkg.adapters", "pkg.more"))

    modules = resolve_actor_adapter_modules(
        {"actor_adapter_modules": ["pkg.adapters", "pkg.config"]},
        runtime,
    )

    assert modules == ("pkg.adapters", "pkg.config", "pkg.more")


def test_resolve_actor_adapter_modules_defaults_to_empty() -> None:
    from uni_rl.offpolicy.runtime import resolve_actor_adapter_modules

    assert resolve_actor_adapter_modules({}, None) == ()


def test_resolve_actor_adapter_modules_rejects_non_string_entries() -> None:
    from uni_rl.offpolicy.runtime import resolve_actor_adapter_modules

    with pytest.raises(ValueError, match="non-empty dotted module strings"):
        resolve_actor_adapter_modules({"actor_adapter_modules": [42]}, None)
