from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .distill import HoraDistillationTrainer
from .models import HoraActorModel, HoraCriticModel, HoraSharedActorCritic
from .ppo import HoraPPO

if TYPE_CHECKING:
    from .appo import HoraAPPORunner
    from .sac_learner import HoraSACLearner
    from .sac_models import HoraSACActor

__all__ = [
    "HoraActorModel",
    "HoraAPPORunner",
    "HoraCriticModel",
    "HoraDistillationTrainer",
    "HoraPPO",
    "HoraSACActor",
    "HoraSACLearner",
    "HoraSharedActorCritic",
]


def __getattr__(name: str) -> Any:
    if name == "HoraAPPORunner":
        from .appo import HoraAPPORunner

        return HoraAPPORunner
    if name in {"HoraSACActor", "HoraSACLearner"}:
        from .sac_learner import HoraSACLearner
        from .sac_models import HoraSACActor

        exports = {
            "HoraSACActor": HoraSACActor,
            "HoraSACLearner": HoraSACLearner,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
