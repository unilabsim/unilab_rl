"""WarpSAC learner built on the UniLab FlashSAC runtime contract."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from uni_rl.algos.flash_sac.layers import UnitBatchNorm, UnitLinear, UnitRMSNorm
from uni_rl.algos.flash_sac.learner import FlashSACLearner


def _disable_parameter_normalization(module: nn.Module) -> None:
    """Restore FlashSAC's pre-normalization parameter layout."""
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, UnitLinear):
                nn.init.orthogonal_(child.w.weight)
            elif isinstance(child, UnitBatchNorm):
                child.weight.fill_(1.0)
                child.bias.zero_()
            elif isinstance(child, UnitRMSNorm):
                child.weight.fill_(1.0)


def _noop_normalize_parameters() -> None:
    """Instance-level override used when WarpSAC disables normalization."""


class WarpSACLearner(FlashSACLearner):
    """WarpSAC learner.

    WarpSAC reuses FlashSAC's distributional critic, actor, temperature, AMP,
    compile, and CUDA-graph training implementation. Its regime-aware replay
    behavior is owned by :class:`WarpSACReplayPipeline` in this package.
    """

    def __init__(
        self,
        *args,
        actor_normalize_parameters: bool = True,
        critic_normalize_parameters: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.actor_normalize_parameters = bool(actor_normalize_parameters)
        self.critic_normalize_parameters = bool(critic_normalize_parameters)
        if not self.actor_normalize_parameters:
            _disable_parameter_normalization(self.actor)
            cast(Any, self.actor).normalize_parameters = _noop_normalize_parameters
        if not self.critic_normalize_parameters:
            _disable_parameter_normalization(self.critic)
            self.target_critic.load_state_dict(self.critic.state_dict())
            cast(Any, self.critic).normalize_parameters = _noop_normalize_parameters
