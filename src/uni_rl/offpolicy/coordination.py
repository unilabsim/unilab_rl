"""Phase-aware coordination state shared by an off-policy learner and collector."""

from __future__ import annotations

import multiprocessing as mp
import os
from enum import IntEnum


class LearnerPhase(IntEnum):
    """Lifecycle phases observable by the lock-step collector.

    ``WAITING_FOR_COLLECTOR`` means the learner is actively polling for work and
    publishes progress. ``BUSY`` deliberately covers inference, replay work, and
    learner updates; those phases may contain unbounded machine-dependent cold
    work and therefore must not be converted into a latency deadline.
    """

    STOPPED = 0
    WAITING_FOR_COLLECTOR = 1
    BUSY = 2


class LearnerCoordinationState:
    """Small cross-process learner phase/progress record.

    The progress counter is not a heartbeat for ``BUSY`` phases: compilation and
    CUDA graph capture are allowed to stop learner-thread progress indefinitely.
    It only lets a collector distinguish a live waiting loop from a corrupted
    request queue while the learner process remains alive.
    """

    def __init__(self, *, phase: LearnerPhase = LearnerPhase.STOPPED) -> None:
        context = mp.get_context("spawn")
        self._phase = context.Value("i", int(phase))
        self._progress = context.Value("I", 0)

    def set_phase(self, phase: LearnerPhase) -> None:
        self._phase.value = int(phase)

    def mark_waiting(self) -> None:
        self.set_phase(LearnerPhase.WAITING_FOR_COLLECTOR)
        self.mark_progress()

    def mark_busy(self) -> None:
        self.set_phase(LearnerPhase.BUSY)

    def mark_stopped(self) -> None:
        self.set_phase(LearnerPhase.STOPPED)

    def mark_progress(self) -> None:
        with self._progress.get_lock():
            self._progress.value = (self._progress.value + 1) & 0xFFFFFFFF

    def snapshot(self) -> tuple[LearnerPhase, int]:
        with self._phase.get_lock(), self._progress.get_lock():
            return LearnerPhase(self._phase.value), int(self._progress.value)


def learner_pid_is_alive(pid: int | None) -> bool:
    """Return whether the learner PID is still observable.

    ``None`` means liveness is not available (used by focused unit tests and the
    compatibility helper API), so callers must not declare death.
    """
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists but belongs to another uid. In the normal spawn setup
        # this is still evidence that the learner has not been reaped yet.
        return True
    except OSError:
        return True
    return True


__all__ = [
    "LearnerCoordinationState",
    "LearnerPhase",
    "learner_pid_is_alive",
]
