"""Algorithm-agnostic learner boilerplate shared by the off-policy SAC learners.

``LearnerBoilerplateMixin`` centralizes the pieces of ``FastSACLearner`` and
``FlashSACLearner`` that are identical between the two: AMP dtype/scaler
resolution, autocast context, observation-normalizer updates, gradient-sync
plumbing, CUDA graph release, and loss-method compilation. The mixin only
declares the attribute contract; consuming learners own the attributes and
the algorithm-specific CUDA graph lifecycle (capture/reset/materialize).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
import torch.nn as nn

from uni_rl.algos.common.compile import get_torch_compile_for_cuda
from uni_rl.algos.common.normalization import EmpiricalNormalization


def polyak_update_target(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-average ``target`` parameters toward ``source`` in place."""
    with torch.no_grad():
        target_params = cast(list[torch.Tensor], list(target.parameters()))
        source_params = cast(list[torch.Tensor], list(source.parameters()))
        try:
            torch._foreach_lerp_(target_params, source_params, tau)
        except RuntimeError:
            for tgt, src in zip(target_params, source_params):
                tgt.lerp_(src, tau)


def fused_adam_supported(device_type: str) -> bool:
    """Whether this torch build provides a fused Adam/AdamW kernel for the device.

    Fused kernels collapse the per-parameter host loop into one launch; on MPS
    the single-tensor fallback additionally performs a blocking `.item()` per
    parameter per step, which dominates learner time there.  CUDA has always
    been supported; MPS fused kernels ship with torch >= 2.6.  Other devices
    keep the default path.
    """
    if device_type == "cuda":
        return True
    if device_type != "mps":
        return False
    try:
        from torch.utils._foreach_utils import (  # pyright: ignore[reportPrivateImportUsage]
            _get_fused_kernels_supported_devices,
        )
    except ImportError:
        return False
    return device_type in _get_fused_kernels_supported_devices()


def resolve_finite_check_flags(device_type: str, device_gated: bool) -> tuple[bool, bool]:
    """Return ``(host_finite_checks, metrics_finite_checks)`` for a learner.

    ``device_gated`` marks the CUDA compile/graph paths whose fused kernels
    skip non-finite steps on-device.  MPS has no equivalent (its fused AdamW
    kernel ignores ``found_inf``), so host checks there are deferred to the
    metrics-reading update of each iteration — every check is a blocking MPS
    sync, and a non-finite loss is detected at that same sync boundary.
    """
    host_finite_checks = not device_gated
    return host_finite_checks, host_finite_checks and device_type == "mps"


class LearnerBoilerplateMixin:
    """Shared AMP / gradient-sync / obs-normalizer / compile boilerplate.

    Attribute contract provided by the consuming learner:
    """

    device: Any
    use_amp: bool
    _amp_dtype: torch.dtype
    scaler: Any | None
    obs_normalizer: EmpiricalNormalization | nn.Identity
    use_cuda_graph_critic: bool
    use_cuda_graph_actor: bool
    dp_cuda_graph_gradient_sync: bool
    _gradient_sync: Callable[[Iterable[torch.Tensor]], None] | None
    _gradient_sync_graph_replay_recorder: Callable[[int], None] | None
    _active_cuda_graph_gradient_sync_calls: list[int] | None
    _critic_loss_tensors: Callable[..., Any]
    _actor_loss_tensors: Callable[..., Any]
    _compile_loss_cudagraphs: bool = False
    _host_finite_checks: bool
    _metrics_finite_checks: bool

    def _finite_check_ok(self, loss: torch.Tensor, read_metrics: bool) -> bool:
        """Host-side finite guard for one loss.

        On MPS the check is deferred to the metrics-reading update of each
        iteration (``_metrics_finite_checks``); every other device keeps the
        original per-update check semantics.
        """
        if not self._host_finite_checks:
            return True
        if self._metrics_finite_checks and not read_metrics:
            return True
        return bool(torch.isfinite(loss))

    def _reset_critic_cuda_graph(self) -> None:
        raise NotImplementedError

    def _reset_actor_cuda_graph(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str, device_type: str) -> torch.dtype:
        normalized = amp_dtype.lower()
        if normalized == "auto":
            return torch.bfloat16
        if normalized == "fp16":
            return torch.float16
        if normalized == "bf16":
            return torch.bfloat16
        raise ValueError("amp_dtype must be one of: auto, fp16, bf16")

    @staticmethod
    def _should_use_grad_scaler(
        use_amp: bool,
        device_type: str,
        amp_dtype: torch.dtype,
    ) -> bool:
        return bool(use_amp) and device_type == "cuda" and amp_dtype == torch.float16

    def _autocast(self):
        return torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=self._amp_dtype,
            enabled=self.use_amp,
        )

    @torch.no_grad()
    def _update_obs_normalizer(self, obs: torch.Tensor) -> None:
        if isinstance(self.obs_normalizer, nn.Identity):
            return
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        normalizer.update(obs)

    def _compile_training_methods(self) -> None:
        compile_fn = get_torch_compile_for_cuda(self.device, warn=True)
        if compile_fn is None:
            return

        compile_kwargs = {"options": {"triton.cudagraphs": bool(self._compile_loss_cudagraphs)}}
        if not self.use_cuda_graph_critic:
            self._critic_loss_tensors = compile_fn(self._critic_loss_tensors, **compile_kwargs)
        if not self.use_cuda_graph_actor:
            self._actor_loss_tensors = compile_fn(self._actor_loss_tensors, **compile_kwargs)

    def set_gradient_sync(
        self,
        sync: Callable[[Iterable[torch.Tensor]], None] | None,
        *,
        graph_replay_recorder: Callable[[int], None] | None = None,
    ) -> None:
        """Attach the per-optimizer gradient collective used by multi-GPU DP."""
        if sync is None and graph_replay_recorder is not None:
            raise ValueError("graph_replay_recorder requires a gradient sync callback")
        if sync != self._gradient_sync:
            self._reset_critic_cuda_graph()
            self._reset_actor_cuda_graph()
        self._gradient_sync = sync
        self._gradient_sync_graph_replay_recorder = graph_replay_recorder
        self.dp_cuda_graph_gradient_sync = self._dp_cuda_graph_gradient_sync_enabled()

    def _dp_cuda_graph_gradient_sync_enabled(self) -> bool:
        return bool(
            self._gradient_sync is not None
            and self.scaler is None
            and (self.use_cuda_graph_critic or self.use_cuda_graph_actor)
        )

    def _sync_gradients(self, parameters: Iterable[torch.Tensor]) -> None:
        if self._gradient_sync is not None:
            self._gradient_sync(parameters)
            if self._active_cuda_graph_gradient_sync_calls is not None:
                self._active_cuda_graph_gradient_sync_calls[0] += 1

    def _record_cuda_graph_gradient_replay(self, collective_calls: int) -> None:
        if self._gradient_sync_graph_replay_recorder is not None and collective_calls > 0:
            self._gradient_sync_graph_replay_recorder(collective_calls)

    def release_cuda_graphs(self) -> None:
        """Release captured NCCL nodes before the process group is destroyed."""
        self._reset_critic_cuda_graph()
        self._reset_actor_cuda_graph()
