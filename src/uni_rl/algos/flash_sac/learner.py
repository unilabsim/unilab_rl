"""FlashSAC learner adapted to UniLab's off-policy contract."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim

from uni_rl.algos.common.compile import get_torch_compile_for_cuda
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

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


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
            self.g_r = self.gamma * (1.0 - done[step]) * self.g_r + rewards[step]
            self.g_r_max = torch.maximum(self.g_r_max, self.g_r.abs().max())
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
        self.g_r = state_dict["g_r"]
        self.g_r_max = state_dict["g_r_max"]


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
        self.use_amp = bool(use_amp and self.device.type in ("cuda", "xpu"))
        self.amp_dtype = amp_dtype
        self._amp_dtype = self._resolve_amp_dtype(amp_dtype, self.device.type)
        self.use_compile = bool(
            use_compile and get_torch_compile_for_cuda(self.device, warn=True) is not None
        )
        self.compile_full_objectives = bool(compile_full_objectives and self.use_compile)
        self._device_type = self.device.type
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
        self._zero_metric = torch.zeros((), device=self.device)
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

        if self.use_compile:
            self._compile_training_methods()

    def _compile_training_methods(self) -> None:
        compile_fn = get_torch_compile_for_cuda(self.device, warn=True)
        if compile_fn is None:
            return

        compile_kwargs = {"options": {"triton.cudagraphs": bool(self._compile_loss_cudagraphs)}}
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
        super()._compile_training_methods()

    @contextmanager
    def _optimizer_finite_gate(
        self,
        optimizer: optim.Optimizer,
        loss: torch.Tensor,
    ) -> Iterator[None]:
        """Skip fused CUDA optimizer steps on non-finite values on-device."""
        if self._host_finite_checks or self._device_type != "cuda":
            yield
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
        setattr(optimizer, "grad_scale", self._optimizer_grad_scale)
        setattr(optimizer, "found_inf", found_inf)
        try:
            yield
        finally:
            delattr(optimizer, "grad_scale")
            delattr(optimizer, "found_inf")

    @contextmanager
    def _critic_parameters_frozen(self) -> Iterator[None]:
        """Exclude critic parameters from actor autograd while keeping dQ/da."""
        states = [(parameter, parameter.requires_grad) for parameter in self.critic.parameters()]
        for parameter, _ in states:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, requires_grad in states:
                parameter.requires_grad_(requires_grad)

    @staticmethod
    def _snapshot_module_buffers(*modules: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (buffer, buffer.detach().clone()) for module in modules for buffer in module.buffers()
        ]

    @staticmethod
    def _restore_module_buffers(snapshot: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        with torch.no_grad():
            for buffer, saved in snapshot:
                buffer.copy_(saved)

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

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(requires_grad)

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

        self.critic_optimizer.zero_grad(set_to_none=True)
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
                with self._optimizer_finite_gate(self.critic_optimizer, critic_loss):
                    self.critic_optimizer.step()
        self.critic_scheduler.step()
        self.critic.normalize_parameters()

        if not read_metrics:
            return {}
        metric_names: tuple[str, ...] = ("Loss/critic",)
        metric_tensors: tuple[torch.Tensor, ...] = (critic_loss,)
        if self.reward_normalizer is not None:
            metric_names = (*metric_names, "Train/reward_scale_std")
            metric_tensors = (*metric_tensors, reward_scale_std)
        return self._read_metric_tensors(
            metric_names,
            metric_tensors,
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
        with self._critic_parameters_frozen():
            actor_loss, entropy = self._actor_objective_tensors(
                obs,
                next_obs,
                expert_actions,
                critic_obs,
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self._finite_check_ok(actor_loss, read_metrics):
            if self.scaler is not None:
                self.scaler.scale(actor_loss).backward()
                self._sync_gradients(self.actor.parameters())
                self.scaler.unscale_(self.actor_optimizer)
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                actor_loss.backward()
                self._sync_gradients(self.actor.parameters())
                with self._optimizer_finite_gate(self.actor_optimizer, actor_loss):
                    self.actor_optimizer.step()
        self.actor_scheduler.step()
        self.actor.normalize_parameters()

        temp_value = self.temperature()
        temp_loss = temp_value * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        if self._finite_check_ok(temp_loss, read_metrics):
            temp_loss.backward()
            self._sync_gradients(self.temperature.parameters())
        with self._optimizer_finite_gate(self.temperature_optimizer, temp_loss):
            self.temperature_optimizer.step()
        self.temperature_scheduler.step()

        post_update_temperature = self.temperature()
        actor_metric_tensors = (actor_loss, entropy, post_update_temperature, temp_loss)
        if not read_metrics:
            # Keep a private device-side snapshot.  The cycle-end drain below
            # performs the only D2H read, after all compiled replays finish.
            self._pending_actor_metric_values = torch.stack(
                [tensor.detach().reshape(()) for tensor in actor_metric_tensors]
            )
            return {}
        return self._read_metric_tensors(
            (
                "Loss/actor",
                "Loss/entropy",
                "Policy/temperature",
                "Loss/temperature",
            ),
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
                (
                    "Loss/actor",
                    "Loss/entropy",
                    "Policy/temperature",
                    "Loss/temperature",
                ),
                values.cpu().tolist(),
                strict=True,
            )
        }

    def soft_update_target(self) -> None:
        polyak_update_target(self.target_critic, self.critic, self.tau)

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
