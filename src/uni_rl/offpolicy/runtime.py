"""Runtime resolution helpers for off-policy script assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class OffPolicyRuntime:
    """Optional runtime overrides for the generic off-policy SAC path.

    All fields are optional so custom runtimes only declare the behaviour they
    need to change from standard SAC.
    """

    learner_cls: type[Any] | None = None
    algo_type: str | None = None
    actor_kwargs: dict[str, Any] = field(default_factory=dict)
    actor_adapter_modules: tuple[str, ...] = ()

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        """Build learner model kwargs from the environment observation contract."""
        del obs_dim, critic_obs_dim
        return dict(self.actor_kwargs)


def resolve_actor_adapter_modules(
    rl_cfg: dict[str, Any],
    runtime: OffPolicyRuntime | None,
) -> tuple[str, ...]:
    """Resolve dotted actor-adapter registration modules from owner config.

    Combines the plain ``algo.actor_adapter_modules`` config key (usable
    without a custom runtime resolver) with the runtime-provided modules,
    deduplicated in declaration order.
    """
    raw = rl_cfg.get("actor_adapter_modules") or ()
    if isinstance(raw, str):
        raw = (raw,)
    modules: list[str] = []
    for entry in list(raw) + list(runtime.actor_adapter_modules if runtime else ()):
        if not isinstance(entry, str) or not entry:
            raise ValueError(
                "actor_adapter_modules entries must be non-empty dotted module "
                f"strings, got {entry!r}."
            )
        if entry not in modules:
            modules.append(entry)
    return tuple(modules)


def _resolve_callable(path: str) -> Any:
    module_path: str
    attr_name: str
    if ":" in path:
        module_path, attr_name = path.split(":", 1)
    else:
        module_path, _, attr_name = path.rpartition(".")
    if not module_path or not attr_name:
        raise ValueError(f"Invalid runtime resolver path: {path!r}")
    resolved = getattr(import_module(module_path), attr_name)
    if not callable(resolved):
        raise TypeError(f"Runtime resolver {path!r} is not callable.")
    return resolved


def resolve_custom_offpolicy_runtime(rl_cfg: dict[str, Any]) -> OffPolicyRuntime | None:
    """Resolve an optional custom off-policy runtime from owner config."""
    runtime_resolver = rl_cfg.get("runtime_resolver")
    if runtime_resolver in (None, ""):
        runtime_impl = rl_cfg.get("runtime_impl")
        if runtime_impl not in (None, ""):
            raise ValueError(
                "Off-policy owner config selected "
                f"runtime_impl={runtime_impl!r} but did not define algo.runtime_resolver."
            )
        return None

    resolver = _resolve_callable(str(runtime_resolver))
    runtime = resolver(rl_cfg)
    if runtime is None:
        raise ValueError(
            f"Off-policy runtime resolver {runtime_resolver!r} returned None "
            "for rl_cfg runtime selection."
        )

    if not isinstance(runtime, OffPolicyRuntime):
        raise TypeError(
            f"Off-policy runtime resolver {runtime_resolver!r} must return "
            "an OffPolicyRuntime instance."
        )
    return runtime
