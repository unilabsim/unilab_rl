"""HORA-owned APPO entry helpers.

Play-mode orchestration (``play_hora_appo``) is UniLab-side business and was
removed from uni_rl in issue #1479; it is re-homed on the UniLab side under
issue #1480 (see UniLab git history for the removed implementation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uni_rl.hora.appo_runner import HoraAPPORunner

from .runtime import is_hora_appo_runtime


@dataclass(frozen=True)
class HoraAPPORuntime:
    """Resolved HORA APPO entrypoints used by the generic APPO script.

    Args:
        runner_cls: Runner class used for HORA APPO training mode.

    Returns:
        Immutable entrypoint bundle consumed by generic APPO script assembly.
    """

    runner_cls: type[HoraAPPORunner]


def resolve_hora_appo_runtime(rl_cfg: dict[str, Any]) -> HoraAPPORuntime | None:
    """Resolve HORA APPO entrypoints from an explicit runtime marker.

    Args:
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.

    Returns:
        ``HoraAPPORuntime`` when the owner config selects HORA APPO, otherwise
        ``None``.
    """
    if not is_hora_appo_runtime(rl_cfg):
        return None
    return HoraAPPORuntime(runner_cls=HoraAPPORunner)


__all__ = ["HoraAPPORunner", "HoraAPPORuntime", "resolve_hora_appo_runtime"]
