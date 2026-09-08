"""RSL-RL runner extension for explicit downstream training progress.

The checkpoint's ``infos['uni_rl_training_state']`` envelope has version 1 and
an opaque ``state`` payload. The provider owns that payload's schema. Plain
OnPolicyRunner remains the default; opt into this runner through the runtime
bundle when curriculum/progress restoration is required.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner

from uni_rl.training_state import TrainingStateProvider, _copy_training_state

_STATE_KEY = "uni_rl_training_state"
_STATE_VERSION = 1


class TrainingStateOnPolicyRunner(OnPolicyRunner):
    """Round-trip owner progress alongside the standard algorithm checkpoint.

    Supply a provider explicitly, or pass a wrapper implementing the declared
    TrainingStateProvider protocol. No underlying environment is discovered.
    Inference-only/legacy policy loads must explicitly disable state restoration.
    Load errors must abort resume: loading algorithm and owner state is not a
    transaction that rolls back a successfully loaded algorithm on owner error.
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
        *,
        training_state_provider: TrainingStateProvider | None = None,
    ) -> None:
        provider = env if training_state_provider is None else training_state_provider
        if not isinstance(provider, TrainingStateProvider):
            raise TypeError(
                "TrainingStateOnPolicyRunner requires an explicit TrainingStateProvider "
                "or a wrapper implementing export_training_state/import_training_state"
            )
        self.training_state_provider: TrainingStateProvider = provider
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos: dict | None = None) -> None:
        state = _copy_training_state(self.training_state_provider.export_training_state())
        checkpoint_infos = dict(infos or {})
        checkpoint_infos[_STATE_KEY] = {"version": _STATE_VERSION, "state": state}
        # Preserve the parent runner's algorithm/optimizer/iteration, logging,
        # and artifact behavior (including consumer-installed logger adapters).
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        *,
        restore_training_state: bool = True,
    ) -> dict:
        state: dict[str, Any] | None = None
        if not restore_training_state and (
            load_cfg is None
            or not load_cfg.get("actor")
            or any(enabled for key, enabled in load_cfg.items() if key != "actor")
        ):
            raise ValueError(
                "restore_training_state=False requires an explicit actor-only load_cfg; "
                "optimizer/iteration resume cannot omit curriculum state"
            )
        if restore_training_state:
            # Preflight the envelope before the parent mutates algorithm state.
            # The parent owns checkpoint loading; its public load path is kept
            # intact rather than duplicating algorithm or logging internals.
            checkpoint = torch.load(path, weights_only=False, map_location=map_location)
            infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
            envelope = infos.get(_STATE_KEY) if isinstance(infos, Mapping) else None
            if not isinstance(envelope, Mapping):
                raise ValueError(
                    "Checkpoint is missing required training state; curriculum resume "
                    "cannot proceed. Use restore_training_state=False only for an "
                    "explicit policy-only load."
                )
            version = envelope.get("version")
            if type(version) is not int or version != _STATE_VERSION:
                raise ValueError(f"Unsupported training state envelope version: {version!r}")
            if not isinstance(envelope.get("state"), Mapping):
                raise ValueError("Invalid training state envelope: 'state' must be a mapping")
            state = _copy_training_state(envelope["state"])

        result: dict = super().load(path, load_cfg, strict, map_location)
        if state is not None:
            self.training_state_provider.import_training_state(state)
        return result
