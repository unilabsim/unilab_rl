"""Extension registry for custom off-policy actor behavior.

External algorithm packages teach the generic off-policy runtime
(``uni_rl.offpolicy``) about a custom actor type by registering an
:class:`OffPolicyActorAdapter` at module import time, then listing that module
in the ``actor_adapter_modules`` off-policy runtime config so spawn collector
subprocesses import it (and re-run the registration) at startup.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass(frozen=True)
class OffPolicyActorAdapter:
    """Hooks that adapt a custom off-policy actor to the generic runtime.

    All hooks are optional; the runtime only calls the hooks an adapter
    provides and treats missing hooks as "no custom behavior".

    Args:
        algo_type: Off-policy ``algo_type`` this adapter serves.
        build_actor: Actor builder with the same keyword parameters as the
            custom branch of ``uni_rl.algos.common.actor_factory.build_actor``:
            ``obs_dim``, ``action_dim``, ``actor_hidden_dim``,
            ``use_layer_norm``, ``device``, ``priv_info_dim``,
            ``priv_info_embed_dim``, ``priv_mlp_hidden_dims``.
        sample_actions: ``(actor, obs_torch, prev_dones_torch,
            priv_info_torch) -> actions`` exploration sampling hook used by
            learner-side inference.
        resolve_priv_info: ``(obs_np, critic_np, info) -> priv_info_np | None``
            hook that extracts the actor's privileged context from env
            observations on the collector side.
        actor_context_from_obs: ``(obs_device, obs_dim) -> context | None``
            hook that slices the packed learner-side inference observation into
            the actor's extra context (e.g. the privileged tail).
    """

    algo_type: str
    build_actor: Callable[..., Any] | None = None
    sample_actions: (
        Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor] | None
    ) = None
    resolve_priv_info: Callable[[np.ndarray, np.ndarray, dict | None], np.ndarray | None] | None = (
        None
    )
    actor_context_from_obs: Callable[[torch.Tensor, int], torch.Tensor | None] | None = None


_ADAPTERS: dict[str, OffPolicyActorAdapter] = {}


def register_offpolicy_actor_adapter(adapter: OffPolicyActorAdapter) -> None:
    """Register an off-policy actor adapter for ``adapter.algo_type``."""
    if not isinstance(adapter, OffPolicyActorAdapter):
        raise TypeError(
            "register_offpolicy_actor_adapter expects an OffPolicyActorAdapter, "
            f"got {type(adapter).__name__}."
        )
    if not adapter.algo_type:
        raise ValueError("OffPolicyActorAdapter.algo_type must be a non-empty string.")
    existing = _ADAPTERS.get(adapter.algo_type)
    if existing is not None:
        raise ValueError(
            f"An off-policy actor adapter is already registered for "
            f"algo_type={adapter.algo_type!r}."
        )
    _ADAPTERS[adapter.algo_type] = adapter


def get_offpolicy_actor_adapter(algo_type: str) -> OffPolicyActorAdapter | None:
    """Return the adapter registered for ``algo_type``, or ``None``."""
    return _ADAPTERS.get(str(algo_type))


def import_actor_adapter_modules(modules: Iterable[str] | None) -> None:
    """Import dotted modules so their import-time adapter registrations run.

    Spawn collector subprocesses do not inherit registrations made in the
    parent process; the off-policy runtime forwards ``actor_adapter_modules``
    from the owner config and calls this helper in every process that resolves
    adapters (learner and collector).
    """
    for module in modules or ():
        try:
            import_module(str(module))
        except ImportError as exc:
            raise ImportError(
                f"Failed to import actor_adapter_modules entry {module!r}: {exc}. "
                "Ensure the package that registers the off-policy actor adapter "
                "is installed or importable on PYTHONPATH in this process."
            ) from exc
