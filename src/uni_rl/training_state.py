"""Explicit, simulator-independent checkpoint state supplied by a training owner.

Providers own the payload schema and its migrations. Values must be plain JSON
data (string-keyed dictionaries, lists, finite numbers, strings, booleans, null),
so exporting progress cannot accidentally pickle an environment or manager.
Convert arrays to lists explicitly when the task's schema requires them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainingStateProvider(Protocol):
    """Opt-in state boundary, called only when saving or restoring a checkpoint.

    Include an owner-specific schema/version in the payload. Import must validate
    the entire payload before changing owner state, and reject incompatible or
    incomplete progress. The runner does not inspect the environment or invent
    missing state. This is training progress, not a physics snapshot or RNG state.
    """

    def export_training_state(self) -> Mapping[str, Any]:
        """Return all progress needed to resume this owner's training curriculum."""
        ...

    def import_training_state(self, state: Mapping[str, Any]) -> None:
        """Validate the owner's schema, then restore its progress atomically."""
        ...


def _copy_training_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate plain data and detach the snapshot from mutable provider state."""
    ancestors: set[int] = set()

    def copy_value(value: Any) -> Any:
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Training state numbers must be finite")
            return value
        if type(value) not in (dict, list):
            raise TypeError(
                "Training state must contain only plain JSON dictionaries/lists/scalars"
            )
        identity = id(value)
        if identity in ancestors:
            raise ValueError("Training state must not contain reference cycles")
        ancestors.add(identity)
        try:
            if isinstance(value, dict):
                if any(type(key) is not str for key in value):
                    raise TypeError("Training state dictionary keys must be strings")
                return {key: copy_value(item) for key, item in value.items()}
            return [copy_value(item) for item in value]
        finally:
            ancestors.remove(identity)

    if not isinstance(state, Mapping):
        raise TypeError("Training state export must return a mapping")
    result: dict[str, Any] = copy_value(dict(state))
    return result
