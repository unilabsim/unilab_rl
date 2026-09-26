"""Learner-owned cold-path preparation contract for off-policy training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class OffPolicyWarmupContext:
    """Representative shapes and lifecycle options for no-side-effect warmup.

    The inference tensors are learner-device scratch allocations. When the
    runtime provides ``replay_batch``, those views alias a replay pipeline's
    scratch cold slot to avoid allocating a third full training batch during
    warmup. They are deliberately
    synthetic: constructing an environment or consuming replay data during
    preparation would change collection and training semantics.
    """

    inference_observations: torch.Tensor
    inference_dones: torch.Tensor
    batch_size: int
    updates_per_step: int
    policy_frequency: int
    target_frequency: int
    policy_before_critic: bool
    replay_batch: dict[str, torch.Tensor] | None = None


class OffPolicyLearnerPreparation(Protocol):
    """Optional owner-layer hook invoked before the collector subprocess starts."""

    def prepare_for_collection(self, warmup_context: OffPolicyWarmupContext) -> None:
        """Prepare learner-owned cold paths without changing training state."""
        ...


@dataclass
class _TorchRngState:
    cpu: torch.Tensor
    cuda: torch.Tensor | None
    python: object
    device: torch.device


def capture_rng_state(device: str | torch.device) -> _TorchRngState:
    torch_device = torch.device(device)
    cuda_state = (
        torch.cuda.get_rng_state(torch_device)
        if torch_device.type == "cuda" and torch.cuda.is_initialized()
        else None
    )
    return _TorchRngState(
        cpu=torch.random.get_rng_state(),
        cuda=cuda_state,
        python=random.getstate(),
        device=torch_device,
    )


def restore_rng_state(state: _TorchRngState) -> None:
    torch.random.set_rng_state(state.cpu)
    if state.cuda is not None:
        torch.cuda.set_rng_state(state.cuda, state.device)
    random.setstate(state.python)  # type: ignore[arg-type]


def make_offpolicy_warmup_batch(
    context: OffPolicyWarmupContext,
    *,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Build a representative replay batch without touching replay state."""
    rows = max(1, int(context.batch_size) * max(1, int(context.updates_per_step)))
    torch_device = torch.device(device)

    def tensor(*shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=torch_device)

    return {
        "obs": tensor(rows, obs_dim),
        "actions": tensor(rows, action_dim),
        "rewards": tensor(rows),
        "next_obs": tensor(rows, obs_dim),
        "dones": tensor(rows),
        "truncated": tensor(rows),
        "critic": tensor(rows, critic_obs_dim),
        "next_critic": tensor(rows, critic_obs_dim),
    }


__all__ = [
    "OffPolicyLearnerPreparation",
    "OffPolicyWarmupContext",
    "capture_rng_state",
    "make_offpolicy_warmup_batch",
    "restore_rng_state",
]
