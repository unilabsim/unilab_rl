"""Terminal-observation contract resolution for autoresetting envs.

Ported from UniLab's ``unilab.base.final_observation`` (issue #1479) with
identical names and semantics. Only the pieces the algo layer consumes are
ported: ``TerminalObservationContract`` and
``resolve_terminal_observation_contract``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TerminalObservationContract:
    terminal_obs: np.ndarray | None
    terminal_mask: np.ndarray
    timeout_terminal_mask: np.ndarray
    terminal_critic: np.ndarray | None = None


def resolve_terminal_observation_contract(
    next_obs_batch_size: int,
    final_observation: dict[str, Any] | None = None,
    done: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
    truncated: np.ndarray | None = None,
) -> TerminalObservationContract:
    """Resolve terminal observation facts without constructing patched next obs."""
    terminal_mask = _resolve_terminal_mask(next_obs_batch_size, done, info)
    resolved_final_observation = _resolve_final_observation(final_observation, info)

    terminal_obs: np.ndarray | None = None
    terminal_critic: np.ndarray | None = None
    if np.any(terminal_mask) and isinstance(resolved_final_observation, dict):
        terminal_obs = resolved_final_observation.get("obs")
        terminal_critic = resolved_final_observation.get("critic")

    timeout_terminal_mask = terminal_mask
    if truncated is not None:
        timeout_terminal_mask = np.logical_and(
            terminal_mask, np.asarray(truncated, dtype=bool).ravel()
        )

    return TerminalObservationContract(
        terminal_obs=terminal_obs,
        terminal_mask=terminal_mask,
        timeout_terminal_mask=timeout_terminal_mask,
        terminal_critic=terminal_critic,
    )


def _resolve_final_observation(
    final_observation: dict[str, Any] | None,
    info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(final_observation, dict):
        return final_observation
    if isinstance(info, dict):
        final_obs = info.get("final_observation")
        if isinstance(final_obs, dict):
            return final_obs
    return None


def _resolve_terminal_mask(
    next_obs_batch_size: int,
    done: np.ndarray | None,
    info: dict[str, Any] | None,
) -> np.ndarray:
    if done is not None:
        done_mask = np.asarray(done, dtype=bool).ravel()
        if done_mask.shape == (next_obs_batch_size,):
            return done_mask
        return np.zeros((next_obs_batch_size,), dtype=bool)
    if isinstance(info, dict):
        terminal_mask = np.asarray(info.get("_final_observation"), dtype=bool)
        if terminal_mask.shape == (next_obs_batch_size,):
            return terminal_mask
    return np.zeros((next_obs_batch_size,), dtype=bool)
