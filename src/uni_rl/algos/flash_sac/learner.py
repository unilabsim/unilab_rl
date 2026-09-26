"""FlashSAC learner adapted to UniLab's off-policy contract."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim

from uni_rl.algos.common.compile import get_torch_compile_for_cuda, is_hip_runtime
from uni_rl.algos.common.learner_boilerplate import (
    LearnerBoilerplateMixin,
    fused_adam_supported,
    polyak_update_target,
    resolve_finite_check_flags,
)
from uni_rl.algos.common.normalization import EmpiricalNormalization
from uni_rl.algos.flash_sac.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from uni_rl.algos.flash_sac.update import (
    build_lr_lambda,
    compute_categorical_td_target,
    resolve_target_entropy,
    select_min_q_log_probs,
)


@dataclass
class RunningMeanStd:
    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    @classmethod
    def create(cls, device: torch.device) -> "RunningMeanStd":
        return cls(
            mean=torch.zeros(1, device=device, dtype=torch.float32),
            var=torch.ones(1, device=device, dtype=torch.float32),
            count=torch.tensor(1e-4, device=device, dtype=torch.float32),
        )

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1).to(dtype=torch.float32)
        if x.numel() == 0:
            return
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = torch.tensor(float(x.numel()), device=x.device, dtype=torch.float32)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.pow(2) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count

        # Keep the original storage alive.  A whole-cycle CUDA Graph captures
        # these addresses, while the runner continues to update reward
        # statistics between learner cycles.
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state_dict["mean"])
        self.var.copy_(state_dict["var"])
        self.count.copy_(state_dict["count"])


class RewardNormalizer:
    """Adaptive reward scaling with running discounted-return statistics."""

    def __init__(
        self,
        gamma: float,
        g_max: float,
        device: torch.device,
        eps: float = 1e-8,
    ):
        self.gamma = gamma
        self.g_max = g_max
        self.eps = eps
        self.device = device
        self.rms = RunningMeanStd.create(device)
        self.g_r = torch.zeros(0, device=device, dtype=torch.float32)
        self.g_r_max = torch.tensor(0.0, device=device, dtype=torch.float32)

    def _ensure_g_r_shape(self, num_envs: int) -> None:
        if self.g_r.shape == (num_envs,):
            return
        self.g_r = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

    def update_from_transitions(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        rewards = rewards.to(device=self.device, dtype=torch.float32)
        dones = dones.to(device=self.device, dtype=torch.float32)

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)
        if rewards.numel() == 0:
            return

        num_envs = int(rewards.shape[-1])
        self._ensure_g_r_shape(num_envs)
        done = torch.clamp(dones, min=0.0, max=1.0)

        for step in range(rewards.shape[0]):
            self.g_r.mul_(self.gamma * (1.0 - done[step])).add_(rewards[step])
            self.g_r_max.copy_(torch.maximum(self.g_r_max, self.g_r.abs().max()))
            self.rms.update(self.g_r)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        denominator = torch.maximum(
            torch.sqrt(self.rms.var + self.eps),
            self.g_r_max / max(self.g_max, self.eps),
        )
        return rewards / denominator

    def state_dict(self) -> dict[str, Any]:
        return {
            "rms": self.rms.state_dict(),
            "g_r": self.g_r,
            "g_r_max": self.g_r_max,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rms.load_state_dict(state_dict["rms"])
        if self.g_r.shape != state_dict["g_r"].shape:
            self.g_r = torch.empty_like(state_dict["g_r"], device=self.device)
        self.g_r.copy_(state_dict["g_r"])
        self.g_r_max.copy_(state_dict["g_r_max"])


class FlashSACLearner(LearnerBoilerplateMixin):
    # FlashSAC's loss/actor kernels are compatible with Inductor CUDA Graph
    # replay.  Keeping this enabled removes repeated host launches; metric
    # tensors are staged and read once per learner cycle below.
    _compile_loss_cudagraphs = True
    supports_deferred_update_metrics = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        actor_num_blocks: int = 2,
        critic_num_blocks: int = 2,
        num_atoms: int = 101,
        critic_min_v: float = -5.0,
        critic_max_v: float = 5.0,
        temp_initial_value: float = 0.01,
        temp_target_sigma: float = 0.15,
        temp_target_entropy: float | None = None,
        actor_bc_alpha: float = 0.0,
        actor_noise_zeta_mu: float = 2.0,
        actor_noise_zeta_max: int = 16,
        learning_rate_init: float = 3e-4,
        learning_rate_peak: float = 3e-4,
        learning_rate_end: float = 1.5e-4,
        learning_rate_warmup_steps: int = 0,
        learning_rate_decay_steps: int = 500000,
        normalize_reward: bool = True,
        normalized_g_max: float = 5.0,
        n_step: int = 1,
        obs_normalization: bool = False,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        use_compile: bool = False,
        compile_full_objectives: bool = False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.actor_bc_alpha = actor_bc_alpha
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.update_count = 0
        self._device_type = self.device.type
        self._nvidia_cuda = self._device_type == "cuda" and not is_hip_runtime()
        self.use_amp = bool(use_amp and self.device.type in ("cuda", "xpu"))
        self.amp_dtype = amp_dtype
        self._amp_dtype = self._resolve_amp_dtype(amp_dtype, self.device.type)
        compile_fn = get_torch_compile_for_cuda(self.device, warn=not self._nvidia_cuda)
        if self._nvidia_cuda and compile_fn is None:
            raise RuntimeError("FlashSAC requires CUDA Inductor/Triton on NVIDIA CUDA")
        # NVIDIA CUDA always uses the performance path; the legacy opt-out is
        # retained only for ROCm/HIP, MPS, CPU, and other compatibility devices.
        self.use_compile = self._nvidia_cuda or bool(use_compile and compile_fn is not None)
        self.compile_full_objectives = bool(compile_full_objectives and self.use_compile)
        # Host-side ``Tensor.item``/truth checks synchronize the device.  The
        # compiled CUDA path uses the fused optimizer's device gate; MPS has no
        # device-side skip (its fused Adam kernel ignores ``found_inf``), so
        # host checks there run only on the metrics-reading update of each
        # iteration — see `resolve_finite_check_flags`.  CPU/eager CUDA paths
        # retain the explicit per-update safety behavior.
        self._host_finite_checks, self._metrics_finite_checks = resolve_finite_check_flags(
            self._device_type,
            device_gated=self._device_type == "cuda" and self.use_compile,
        )
        self._gradient_sync: Callable[[Iterable[torch.Tensor]], None] | None = None
        self.actor = FlashSACActor(
            num_blocks=actor_num_blocks,
            input_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
            noise_zeta_mu=actor_noise_zeta_mu,
            noise_zeta_max=actor_noise_zeta_max,
            device=self.device,
        )
        self.critic = FlashSACDoubleCritic(
            num_blocks=critic_num_blocks,
            input_dim=self.critic_obs_dim + action_dim,
            hidden_dim=critic_hidden_dim,
            num_bins=num_atoms,
            min_v=critic_min_v,
            max_v=critic_max_v,
            device=self.device,
        )
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.eval()
        self.temperature = FlashSACTemperature(temp_initial_value).to(self.device)
        self._optimizer_grad_scale = torch.ones((), device=self.device)
        self._optimizer_found_inf = torch.zeros((), device=self.device)

        self.target_entropy = resolve_target_entropy(
            action_dim=action_dim,
            target_sigma=temp_target_sigma,
            target_entropy=temp_target_entropy,
        )

        self.obs_normalizer: EmpiricalNormalization | nn.Identity
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=self.device)
        else:
            self.obs_normalizer = nn.Identity()

        self.reward_normalizer = (
            RewardNormalizer(gamma=self.gamma, g_max=normalized_g_max, device=self.device)
            if normalize_reward
            else None
        )

        # GradScaler is only needed for fp16 (cuda); bf16 on xpu doesn't need it.
        self.scaler: Any | None = (
            getattr(torch.amp, "GradScaler")("cuda")
            if self._should_use_grad_scaler(self.use_amp, self.device.type, self._amp_dtype)
            else None
        )
        lr_peak = learning_rate_peak if learning_rate_peak > 0 else actor_lr
        # Fused Adam collapses the per-parameter host loop (and its per-param
        # blocking `.item()` syncs on MPS) into one kernel; `capturable`
        # stays CUDA-only.
        _fused = fused_adam_supported(self.device.type)
        optimizer_kwargs: dict[str, Any] = {"fused": _fused}
        if _fused and self.device.type == "cuda":
            optimizer_kwargs["capturable"] = True
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_peak, **optimizer_kwargs)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_peak, **optimizer_kwargs)
        self.temperature_optimizer = optim.Adam(
            self.temperature.parameters(), lr=lr_peak, **optimizer_kwargs
        )
        self._pending_actor_metric_values: torch.Tensor | None = None

        scheduler_fn = build_lr_lambda(
            init_lr=learning_rate_init,
            peak_lr=lr_peak,
            end_lr=learning_rate_end,
            warmup_steps=learning_rate_warmup_steps,
            decay_steps=learning_rate_decay_steps,
        )
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, scheduler_fn)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, scheduler_fn)
        self.temperature_scheduler = optim.lr_scheduler.LambdaLR(
            self.temperature_optimizer, scheduler_fn
        )

        self._lr_schedule_fn = scheduler_fn
        self._lr_peak = lr_peak
        self._pending_cycle_critic_metric_values: torch.Tensor | None = None
        self._pending_cycle_metric_values: torch.Tensor | None = None
        self._critic_update_finite = torch.ones((), device=self.device)
        self._target_tau = torch.full((), tau, device=self.device)
        self._update_cycle_graph: torch.cuda.CUDAGraph | None = None
        self._update_cycle_graph_cache_key: tuple[object, ...] | None = None
        self._update_cycle_static_batch: dict[str, torch.Tensor] | None = None
        self._update_cycle_graph_metric_values: torch.Tensor | None = None
        self._update_cycle_lr_specs: list[
            tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR, torch.Tensor, int]
        ] = []
        self._update_cycle_lr_cursors: dict[int, int] = {}
        self._capture_update_cycle = False
        self._compile_full_update_cycle = bool(self.use_compile and self._nvidia_cuda)
        if self._compile_full_update_cycle and self.scaler is not None:
            raise ValueError(
                "FlashSAC CUDA compile mode requires bf16 (or fp32); "
                "fp16 GradScaler is incompatible with the whole-cycle graph"
            )
        if self._compile_full_update_cycle:
            if obs_normalization:
                raise ValueError(
                    "FlashSAC whole-cycle CUDA graphs do not yet support obs normalization"
                )
            self.compile_full_objectives = True
        if self.use_compile:
            if self._compile_full_update_cycle:
                self._materialize_capturable_optimizer_state()
            self._compile_training_methods()

    @property
    def use_update_cycle(self) -> bool:
        """Whether this learner owns the whole-cycle update orchestration."""
        return self._compile_full_update_cycle

    def set_gradient_sync(self, sync: Callable[[Iterable[torch.Tensor]], None] | None) -> None:
        """Attach the compatibility-device DP reduction."""
        if sync is not None and self._compile_full_update_cycle:
            raise RuntimeError("FlashSAC NVIDIA CUDA whole-cycle mode does not support DP fallback")
        self._gradient_sync = sync

    def _compile_training_methods(self) -> None:
        compile_fn = get_torch_compile_for_cuda(self.device, warn=True)
        if compile_fn is None:
            return

        if self._compile_full_update_cycle:
            # The raw whole-cycle graph must not nest Inductor CUDA Graph Trees.
            # Architecture-portable max-autotune lets Triton select kernels on
            # the installed GPU instead of restoring Ada/Blackwell learner paths.
            compile_kwargs = {
                "dynamic": False,
                "mode": "max-autotune-no-cudagraphs",
            }
        else:
            compile_kwargs = {
                "dynamic": False,
                "options": {
                    "triton.cudagraphs": bool(self._compile_loss_cudagraphs),
                },
            }
        if self.compile_full_objectives:
            self._critic_objective_tensors = compile_fn(  # type: ignore[method-assign]
                self._critic_objective_tensors, **compile_kwargs
            )
            self._actor_objective_tensors = compile_fn(  # type: ignore[method-assign]
                self._actor_objective_tensors, **compile_kwargs
            )
            return
        self.actor.get_mean_and_std = compile_fn(  # type: ignore[method-assign]
            self.actor.get_mean_and_std, **compile_kwargs
        )
        self.__dict__["_critic_loss_tensors"] = compile_fn(
            self._critic_loss_tensors,
            **compile_kwargs,
        )
        self.__dict__["_actor_loss_tensors"] = compile_fn(
            self._actor_loss_tensors,
            **compile_kwargs,
        )

    def _materialize_capturable_optimizer_state(self) -> None:
        """Create fused Adam state before the whole-cycle graph is captured."""
        optimizers = (self.critic_optimizer, self.actor_optimizer, self.temperature_optimizer)
        saved_groups = [
            [group["lr"] for group in optimizer.param_groups] for optimizer in optimizers
        ]
        try:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0
                    for parameter in group["params"]:
                        parameter.grad = torch.zeros_like(parameter)
                optimizer.step()
        finally:
            for optimizer, saved_group in zip(optimizers, saved_groups, strict=True):
                for group, lr in zip(optimizer.param_groups, saved_group, strict=True):
                    group["lr"] = lr
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.zero_()
            for optimizer in optimizers:
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.zero_()

    def _zero_optimizer_gradients(self, optimizer: optim.Optimizer) -> None:
        """Zero gradients without the graph-skipped optimizer Python entrypoint."""
        if not self._compile_full_update_cycle:
            optimizer.zero_grad(set_to_none=True)
            return
        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if gradients:
            torch._foreach_zero_(gradients)

    def _arm_optimizer_finite_gate(
        self,
        optimizer: optim.Optimizer,
        loss: torch.Tensor,
    ) -> None:
        """Arm a persistent graph-safe gate for fused CUDA Adam."""
        if self._host_finite_checks or self._device_type != "cuda":
            return
        found_inf = self._optimizer_found_inf
        found_inf.copy_(torch.logical_not(torch.isfinite(loss.detach()).all()))
        if self._gradient_sync is not None:
            gradients = [
                parameter.grad
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if gradients:
                torch._amp_foreach_non_finite_check_and_unscale_(
                    gradients,
                    found_inf,
                    self._optimizer_grad_scale,
                )
        if optimizer is self.critic_optimizer:
            self._critic_update_finite.copy_(torch.logical_not(found_inf.detach()))
        setattr(optimizer, "grad_scale", self._optimizer_grad_scale)
        setattr(optimizer, "found_inf", found_inf)

    def _stage_update_cycle_lr(self, optimizer: optim.Optimizer) -> None:
        """Bind the next preallocated LR tensor while capturing a cycle."""
        if not self._capture_update_cycle:
            return
        cursor = self._update_cycle_lr_cursors.get(id(optimizer), 0)
        specs = [spec for spec in self._update_cycle_lr_specs if spec[0] is optimizer]
        if cursor >= len(specs):
            raise RuntimeError("FlashSAC update-cycle LR staging exhausted")
        optimizer.param_groups[0]["lr"] = specs[cursor][2]
        self._update_cycle_lr_cursors[id(optimizer)] = cursor + 1

    def _maybe_normalize_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if isinstance(self.obs_normalizer, nn.Identity):
            return obs
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        if update:
            self._update_obs_normalizer(obs)
            return cast(torch.Tensor, normalizer(obs, update=False))
        return cast(torch.Tensor, normalizer(obs, update=False))

    def update_reward_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if self.reward_normalizer is None:
            return
        self.reward_normalizer.update_from_transitions(rewards, dones)

    def _critic_loss_tensors(
        self,
        next_q_values: torch.Tensor,
        next_q_log_probs_full: torch.Tensor,
        support: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        actor_entropy: torch.Tensor,
        pred_log_probs: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        next_q_log_probs = select_min_q_log_probs(next_q_values, next_q_log_probs_full)
        target_probs = compute_categorical_td_target(
            support=support,
            target_log_probs=next_q_log_probs,
            reward=rewards,
            dones=dones,
            truncated=truncated,
            actor_entropy=actor_entropy,
            gamma=gamma,
        )
        return cast(torch.Tensor, -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean())

    def _actor_loss_tensors(
        self,
        log_probs: torch.Tensor,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        expert_actions: torch.Tensor,
        temp_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        min_q = torch.min(q_values[0], q_values[1])
        actor_loss = (temp_value.detach() * log_probs - min_q).mean()
        if self.actor_bc_alpha > 0:
            bc_loss = torch.mean((actions - expert_actions) ** 2)
            actor_loss = actor_loss + self.actor_bc_alpha * min_q.abs().mean().detach() * bc_loss
        entropy = -log_probs.detach().mean()
        return actor_loss, entropy

    def _critic_objective_tensors(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        critic_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full critic forward/target/projection/loss objective."""
        gamma = self.gamma**self.n_step
        obs_all = torch.cat([critic_obs, critic_next_obs], dim=0)

        with torch.no_grad():
            with self._autocast():
                next_actions, actor_info = self.actor(next_obs, training=False)
                actor_entropy = self.temperature().detach() * actor_info["log_prob"]
                act_all = torch.cat([actions, next_actions], dim=0)
                qs_all, q_info_all = self.target_critic(obs_all, act_all, training=True)
                next_q_values = qs_all.chunk(2, dim=1)[1]
                next_q_log_probs_full = q_info_all["log_prob"].chunk(2, dim=1)[1]
                support = cast(torch.Tensor, self.target_critic.predictor.support)

        with self._autocast():
            _, pred_info_all = self.critic(obs_all, act_all, training=True)
            pred_log_probs = pred_info_all["log_prob"].chunk(2, dim=1)[0]
            critic_loss = self._critic_loss_tensors(
                next_q_values,
                next_q_log_probs_full,
                support,
                rewards,
                dones,
                truncated,
                actor_entropy,
                pred_log_probs,
                gamma,
            )
        reward_scale_std = (
            torch.sqrt(self.reward_normalizer.rms.var)
            if self.reward_normalizer is not None
            else torch.ones((), device=self.device)
        )
        return critic_loss, reward_scale_std

    def _actor_objective_tensors(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        expert_actions: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full actor/critic forward and actor objective."""
        obs_all = torch.cat([obs, next_obs], dim=0)
        with self._autocast():
            actions_all, actor_info_all = self.actor(obs_all, training=True)
            actions = actions_all.chunk(2, dim=0)[0]
            log_probs = actor_info_all["log_prob"].chunk(2, dim=0)[0]
            q_values, _ = self.critic(critic_obs, actions, training=False)
            actor_loss, entropy = self._actor_loss_tensors(
                log_probs, q_values, actions, expert_actions, self.temperature()
            )
        return actor_loss, entropy

    def update_critic(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)
        truncated = batch["truncated"].to(self.device)
        critic_obs = batch["critic"].to(self.device)
        critic_next_obs = batch["next_critic"].to(self.device)

        obs = self._maybe_normalize_obs(obs, update=True)
        next_obs = self._maybe_normalize_obs(next_obs, update=False)

        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.normalize(rewards)

        critic_loss, reward_scale_std = self._critic_objective_tensors(
            actions, rewards, next_obs, dones, truncated, critic_obs, critic_next_obs
        )

        self._zero_optimizer_gradients(self.critic_optimizer)
        if self._finite_check_ok(critic_loss, read_metrics):
            if self.scaler is not None:
                self.scaler.scale(critic_loss).backward()
                self._sync_gradients(self.critic.parameters())
                self.scaler.unscale_(self.critic_optimizer)
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
            else:
                critic_loss.backward()
                self._sync_gradients(self.critic.parameters())
                self._arm_optimizer_finite_gate(self.critic_optimizer, critic_loss)
                self._stage_update_cycle_lr(self.critic_optimizer)
                self.critic_optimizer.step()
        if not self._capture_update_cycle:
            self.critic_scheduler.step()
        self.critic.normalize_parameters()

        if not read_metrics:
            self._pending_cycle_critic_metric_values = torch.stack(
                [tensor.detach().reshape(()) for tensor in (critic_loss, reward_scale_std)]
            )
            return {}
        return self._read_metric_tensors(
            ("critic_loss", "reward_scale_std"),
            (critic_loss, reward_scale_std),
        )

    def update_actor(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        expert_actions = batch["actions"].to(self.device)
        critic_obs = batch["critic"].to(self.device)

        obs = self._maybe_normalize_obs(obs, update=False)
        next_obs = self._maybe_normalize_obs(next_obs, update=False)

        self._pending_actor_metric_values = None
        actor_loss, entropy = self._actor_objective_tensors(
            obs,
            next_obs,
            expert_actions,
            critic_obs,
        )

        self._zero_optimizer_gradients(self.actor_optimizer)
        if self._finite_check_ok(actor_loss, read_metrics):
            if self.scaler is not None:
                self.scaler.scale(actor_loss).backward()
                self._sync_gradients(self.actor.parameters())
                self.scaler.unscale_(self.actor_optimizer)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                actor_loss.backward(inputs=list(self.actor.parameters()))
                self._sync_gradients(self.actor.parameters())
                self._arm_optimizer_finite_gate(self.actor_optimizer, actor_loss)
                self._stage_update_cycle_lr(self.actor_optimizer)
                self.actor_optimizer.step()
        if not self._capture_update_cycle:
            self.actor_scheduler.step()
        self.actor.normalize_parameters()

        temp_value = self.temperature()
        temp_loss = temp_value * (entropy - self.target_entropy)
        self._zero_optimizer_gradients(self.temperature_optimizer)
        if self._finite_check_ok(temp_loss, read_metrics):
            temp_loss.backward(inputs=list(self.temperature.parameters()))
            self._sync_gradients(self.temperature.parameters())
            self._arm_optimizer_finite_gate(self.temperature_optimizer, temp_loss)
            self._stage_update_cycle_lr(self.temperature_optimizer)
            self.temperature_optimizer.step()
        if not self._capture_update_cycle:
            self.temperature_scheduler.step()

        actor_metric_tensors = (actor_loss, entropy, temp_value, temp_loss)
        if not read_metrics:
            # Keep a private device-side snapshot.  The cycle-end drain below
            # performs the only D2H read, after all compiled replays finish.
            self._pending_actor_metric_values = torch.stack(
                [tensor.detach().reshape(()) for tensor in actor_metric_tensors]
            )
            return {}
        return self._read_metric_tensors(
            ("actor_loss", "actor_entropy", "temperature", "temperature_loss"),
            actor_metric_tensors,
        )

    @staticmethod
    def _read_metric_tensors(
        names: tuple[str, ...],
        tensors: tuple[torch.Tensor, ...],
    ) -> dict[str, float]:
        values = torch.stack([tensor.detach().reshape(()) for tensor in tensors]).cpu().tolist()
        return {name: float(value) for name, value in zip(names, values, strict=True)}

    def read_deferred_actor_metrics(self) -> dict[str, float]:
        values = self._pending_actor_metric_values
        self._pending_actor_metric_values = None
        if values is None:
            return {}
        return {
            name: float(value)
            for name, value in zip(
                ("actor_loss", "actor_entropy", "temperature", "temperature_loss"),
                values.cpu().tolist(),
                strict=True,
            )
        }

    def read_deferred_cycle_metrics(self) -> dict[str, float]:
        values = self._pending_cycle_metric_values
        has_actor = values is not None and values.numel() == 6
        self._pending_cycle_critic_metric_values = None
        self._pending_actor_metric_values = None
        self._pending_cycle_metric_values = None
        if values is None:
            return {}
        metric_names: tuple[str, ...] = ("critic_loss", "reward_scale_std")
        if has_actor:
            metric_names = metric_names + (
                "actor_loss",
                "actor_entropy",
                "temperature",
                "temperature_loss",
            )
        return {
            name: float(value)
            for name, value in zip(metric_names, values.cpu().tolist(), strict=True)
        }

    def _run_update_cycle_core(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        self._pending_actor_metric_values = None
        self._pending_cycle_critic_metric_values = None
        self._pending_cycle_metric_values = None
        self._update_cycle_lr_cursors = {}
        batch_size = int(next(iter(large_batch.values())).shape[0]) // updates_per_step
        for update_idx in range(updates_per_step):
            start = update_idx * batch_size
            end = start + batch_size
            batch = {key: value[start:end] for key, value in large_batch.items()}
            do_actor_update = update_idx % policy_frequency == 0
            if policy_before_critic and do_actor_update:
                self.update_actor(batch, read_metrics=False)
            self.update_critic(batch, read_metrics=False)
            if not policy_before_critic and do_actor_update:
                self.update_actor(batch, read_metrics=False)
            if update_idx % target_frequency == 0:
                self.soft_update_target()

        critic_values = self._pending_cycle_critic_metric_values
        actor_values = self._pending_actor_metric_values
        if critic_values is None:
            return
        self._pending_cycle_metric_values = (
            torch.cat((critic_values, actor_values)) if actor_values is not None else critic_values
        )

    def _update_cycle_graph_key(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> tuple[object, ...]:
        shapes = tuple((key, tuple(value.shape), value.dtype) for key, value in large_batch.items())
        reward_state_shape = (
            tuple(self.reward_normalizer.g_r.shape) if self.reward_normalizer is not None else None
        )
        return (
            updates_per_step,
            policy_frequency,
            target_frequency,
            policy_before_critic,
            shapes,
            reward_state_shape,
        )

    def _prepare_update_cycle_lr_specs(
        self,
        *,
        updates_per_step: int,
        policy_frequency: int,
    ) -> None:
        actor_updates = len(range(0, updates_per_step, policy_frequency))
        self._update_cycle_lr_specs = []
        for scheduler, optimizer, count in (
            (self.critic_scheduler, self.critic_optimizer, updates_per_step),
            (self.actor_scheduler, self.actor_optimizer, actor_updates),
            (self.temperature_scheduler, self.temperature_optimizer, actor_updates),
        ):
            for offset in range(count):
                tensor = torch.empty((), device=self.device, dtype=torch.float32)
                self._update_cycle_lr_specs.append((optimizer, scheduler, tensor, offset))

    def _fill_update_cycle_lr_specs(self, *, zero: bool) -> None:
        for _optimizer, scheduler, tensor, offset in self._update_cycle_lr_specs:
            if zero:
                tensor.zero_()
                continue
            lr = self._lr_peak * self._lr_schedule_fn(scheduler.last_epoch + offset)
            tensor.fill_(float(lr))

    def _advance_update_cycle_schedulers(self) -> None:
        counts: dict[int, tuple[optim.lr_scheduler.LambdaLR, int]] = {}
        for _optimizer, scheduler, _tensor, offset in self._update_cycle_lr_specs:
            key = id(scheduler)
            _scheduler, current = counts.get(key, (scheduler, 0))
            counts[key] = (scheduler, max(current, offset + 1))
        for scheduler, count in counts.values():
            for _ in range(count):
                scheduler.step()

    def _restore_update_cycle_group_lrs(self, saved_groups: list[list[object]]) -> None:
        optimizers = (self.critic_optimizer, self.actor_optimizer, self.temperature_optimizer)
        for optimizer, saved_group in zip(optimizers, saved_groups, strict=True):
            for group, lr in zip(optimizer.param_groups, saved_group, strict=True):
                group["lr"] = lr

    def soft_update_target(self) -> None:
        if self._device_type == "cuda":
            with torch.no_grad():
                tau = self._target_tau * self._critic_update_finite.to(self._target_tau.dtype)
                for target, source in zip(
                    self.target_critic.parameters(),
                    self.critic.parameters(),
                    strict=True,
                ):
                    target.lerp_(source, tau)
        else:
            polyak_update_target(self.target_critic, self.critic, self.tau)

    def _warm_update_cycle_graph(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        """Compile/dry-run captured kernels without changing training state."""
        modules: tuple[nn.Module, ...] = (
            self.actor,
            self.critic,
            self.target_critic,
            self.temperature,
        )
        optimizers = (self.critic_optimizer, self.actor_optimizer, self.temperature_optimizer)
        schedulers = (self.critic_scheduler, self.actor_scheduler, self.temperature_scheduler)
        saved_models = [copy.deepcopy(module.state_dict()) for module in modules]
        saved_optimizers = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        saved_schedulers = [copy.deepcopy(scheduler.state_dict()) for scheduler in schedulers]
        saved_groups = [
            [group["lr"] for group in optimizer.param_groups] for optimizer in optimizers
        ]
        saved_obs = (
            copy.deepcopy(self.obs_normalizer.state_dict())
            if not isinstance(self.obs_normalizer, nn.Identity)
            else None
        )
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)
        stream = torch.cuda.Stream(device=self.device)
        self._prepare_update_cycle_lr_specs(
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
        )
        self._fill_update_cycle_lr_specs(zero=True)
        try:
            self._capture_update_cycle = True
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                self._run_update_cycle_core(
                    large_batch,
                    updates_per_step=updates_per_step,
                    policy_frequency=policy_frequency,
                    target_frequency=target_frequency,
                    policy_before_critic=policy_before_critic,
                )
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
        finally:
            self._capture_update_cycle = False
            self._restore_update_cycle_group_lrs(saved_groups)
            for module, state_dict in zip(modules, saved_models, strict=True):
                module.load_state_dict(state_dict)
            for optimizer, state_dict in zip(optimizers, saved_optimizers, strict=True):
                optimizer.load_state_dict(state_dict)
            for scheduler, state_dict in zip(schedulers, saved_schedulers, strict=True):
                scheduler.load_state_dict(state_dict)
            if saved_obs is not None:
                self.obs_normalizer.load_state_dict(saved_obs)
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
            self._zero_optimizer_gradients(self.critic_optimizer)
            self._zero_optimizer_gradients(self.actor_optimizer)
            self._zero_optimizer_gradients(self.temperature_optimizer)
            torch.cuda.synchronize(self.device)

    def _ensure_update_cycle_graph(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        key = self._update_cycle_graph_key(
            large_batch,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            target_frequency=target_frequency,
            policy_before_critic=policy_before_critic,
        )
        if self._update_cycle_graph_cache_key == key and self._update_cycle_graph is not None:
            assert self._update_cycle_static_batch is not None
            for input_key, tensor in self._update_cycle_static_batch.items():
                tensor.copy_(large_batch[input_key])
            return

        self._update_cycle_graph_cache_key = key
        self._update_cycle_graph = None
        self._update_cycle_static_batch = {
            key: value.detach().clone() for key, value in large_batch.items()
        }
        static_batch = self._update_cycle_static_batch
        self._warm_update_cycle_graph(
            static_batch,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            target_frequency=target_frequency,
            policy_before_critic=policy_before_critic,
        )
        self._prepare_update_cycle_lr_specs(
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
        )
        self._fill_update_cycle_lr_specs(zero=False)
        saved_groups = [
            [group["lr"] for group in optimizer.param_groups]
            for optimizer in (
                self.critic_optimizer,
                self.actor_optimizer,
                self.temperature_optimizer,
            )
        ]
        graph = torch.cuda.CUDAGraph()
        try:
            self._capture_update_cycle = True
            with torch.cuda.device(self.device), torch.cuda.graph(graph):
                self._run_update_cycle_core(
                    static_batch,
                    updates_per_step=updates_per_step,
                    policy_frequency=policy_frequency,
                    target_frequency=target_frequency,
                    policy_before_critic=policy_before_critic,
                )
        finally:
            self._capture_update_cycle = False
            self._restore_update_cycle_group_lrs(saved_groups)
        self._update_cycle_graph = graph
        self._update_cycle_graph_metric_values = self._pending_cycle_metric_values

    def update_cycle(
        self,
        large_batch: dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
        read_metrics: bool = False,
    ) -> None:
        """Run the complete learner block used by the off-policy runner."""
        del read_metrics
        if not self.use_update_cycle:
            self._run_update_cycle_core(
                large_batch,
                updates_per_step=updates_per_step,
                policy_frequency=policy_frequency,
                target_frequency=target_frequency,
                policy_before_critic=policy_before_critic,
            )
            return
        self._ensure_update_cycle_graph(
            large_batch,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            target_frequency=target_frequency,
            policy_before_critic=policy_before_critic,
        )
        assert self._update_cycle_graph is not None
        self._fill_update_cycle_lr_specs(zero=False)
        self._update_cycle_graph.replay()
        self._advance_update_cycle_schedulers()
        self._pending_cycle_metric_values = self._update_cycle_graph_metric_values

    def _reset_update_cycle_graph(self) -> None:
        """Invalidate captured storage after checkpoint or DP state changes."""
        self._update_cycle_graph = None
        self._update_cycle_graph_cache_key = None
        self._update_cycle_static_batch = None
        self._update_cycle_graph_metric_values = None
        self._update_cycle_lr_specs = []
        self._update_cycle_lr_cursors = {}
        self._pending_actor_metric_values = None
        self._pending_cycle_critic_metric_values = None
        self._pending_cycle_metric_values = None

    def dp_initial_sync_tensors(self) -> dict[str, torch.Tensor]:
        """Model state broadcast once from rank 0 before collection starts.

        The values alias the parameter/buffer storage of ``actor``, ``critic``,
        ``target_critic`` and ``temperature`` rather than copies. Optimizer
        state starts empty and remains aligned because every actual optimizer
        update uses the same cross-rank mean gradient. Observation/reward
        normalizer statistics remain rank-local.
        """
        tensors: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("actor", self.actor),
            ("critic", self.critic),
            ("target_critic", self.target_critic),
            ("temperature", self.temperature),
        ):
            for key, value in module.state_dict().items():
                tensors[f"{prefix}.{key}"] = value
        return tensors

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "temperature": self.temperature.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "temperature_scheduler": self.temperature_scheduler.state_dict(),
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict")
                else None
            ),
            "reward_normalizer": (
                self.reward_normalizer.state_dict() if self.reward_normalizer is not None else None
            ),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.temperature.load_state_dict(state_dict["temperature"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.temperature_optimizer.load_state_dict(state_dict["temperature_optimizer"])
        self.actor_scheduler.load_state_dict(state_dict["actor_scheduler"])
        self.critic_scheduler.load_state_dict(state_dict["critic_scheduler"])
        self.temperature_scheduler.load_state_dict(state_dict["temperature_scheduler"])
        if state_dict.get("obs_normalizer") and hasattr(self.obs_normalizer, "load_state_dict"):
            self.obs_normalizer.load_state_dict(state_dict["obs_normalizer"])
        if self.reward_normalizer is not None and state_dict.get("reward_normalizer"):
            self.reward_normalizer.load_state_dict(state_dict["reward_normalizer"])
        self.update_count = int(state_dict.get("update_count", 0))
        self._reset_update_cycle_graph()
