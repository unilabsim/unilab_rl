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
    supports_cuda_graph_packed_staging = True
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
        use_cuda_graph_critic: bool = False,
        use_cuda_graph_actor: bool = False,
        use_cuda_graph_critic_packed_staging: bool = False,
        use_cuda_graph_actor_packed_staging: bool = False,
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
        # Host-side ``Tensor.item``/truth checks synchronize the device.
        # Compiled and manually captured CUDA paths use the fused optimizer's
        # device gate; MPS has no device-side skip (its fused Adam kernel
        # ignores ``found_inf``), so host checks there run only on the
        # metrics-reading update of each iteration — see
        # `resolve_finite_check_flags`.  CPU/eager CUDA paths retain the
        # explicit per-update safety behavior.
        self._host_finite_checks, self._metrics_finite_checks = resolve_finite_check_flags(
            self._device_type,
            device_gated=(
                self._device_type == "cuda"
                and (self.use_compile or use_cuda_graph_critic or use_cuda_graph_actor)
            ),
        )
        self.use_cuda_graph_critic = bool(use_cuda_graph_critic)
        self.use_cuda_graph_actor = bool(use_cuda_graph_actor)
        # When a manual graph owns the full update, Inductor should only fuse
        # kernels.  Its internal CUDA Graph Trees cannot be nested safely in
        # the outer capture.  Pure torch.compile keeps Trees enabled.
        self._compile_loss_cudagraphs = not (
            self.use_cuda_graph_critic or self.use_cuda_graph_actor
        )
        self._gradient_sync: Callable[[Iterable[torch.Tensor]], None] | None = None
        self._gradient_sync_graph_replay_recorder: Callable[[int], None] | None = None
        self.dp_cuda_graph_gradient_sync = False
        self._active_cuda_graph_gradient_sync_calls: list[int] | None = None
        self.use_cuda_graph_critic_packed_staging = bool(
            use_cuda_graph_critic_packed_staging and self.use_cuda_graph_critic
        )
        self.use_cuda_graph_actor_packed_staging = bool(
            use_cuda_graph_actor_packed_staging and self.use_cuda_graph_actor
        )
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
        self._cuda_graph_critic: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_critic_static_inputs: dict[str, torch.Tensor] | None = None
        self._cuda_graph_sac_static_packed_input: torch.Tensor | None = None
        self._cuda_graph_sac_static_source_ptr: int | None = None
        self._cuda_graph_critic_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cuda_graph_critic_metric_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cuda_graph_critic_shapes: dict[str, torch.Size] | None = None
        self._cuda_graph_critic_gradient_sync_calls = 0
        self._cuda_graph_actor: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_actor_static_inputs: dict[str, torch.Tensor] | None = None
        self._cuda_graph_actor_static_packed_input: torch.Tensor | None = None
        self._cuda_graph_actor_outputs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._cuda_graph_actor_metric_buffers: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._cuda_graph_actor_shapes: dict[str, torch.Size] | None = None
        self._cuda_graph_actor_gradient_sync_calls = 0
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

    @staticmethod
    def _critic_graph_input_keys() -> tuple[str, ...]:
        return (
            "obs",
            "actions",
            "rewards",
            "next_obs",
            "dones",
            "truncated",
            "critic",
            "next_critic",
        )

    def _critic_graph_input_shapes(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Size]:
        return {key: inputs[key].shape for key in self._critic_graph_input_keys()}

    @staticmethod
    def _graph_width(shape: torch.Size) -> int:
        value = 1
        for dim in shape[1:]:
            value *= int(dim)
        return value

    @classmethod
    def _sac_graph_offsets(
        cls,
        actor_shapes: dict[str, torch.Size],
        critic_shapes: dict[str, torch.Size],
    ) -> dict[str, tuple[int, int]]:
        widths = {
            "obs": cls._graph_width(actor_shapes["obs"]),
            "critic": cls._graph_width(critic_shapes["critic"]),
            "actions": cls._graph_width(critic_shapes["actions"]),
            "rewards": cls._graph_width(critic_shapes["rewards"]),
            "next_obs": cls._graph_width(critic_shapes["next_obs"]),
            "next_critic": cls._graph_width(critic_shapes["next_critic"]),
            "dones": cls._graph_width(critic_shapes["dones"]),
            "truncated": cls._graph_width(critic_shapes["truncated"]),
        }
        offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for key in (
            "obs",
            "critic",
            "actions",
            "rewards",
            "next_obs",
            "next_critic",
            "dones",
            "truncated",
        ):
            key_width = widths[key]
            offsets[key] = (offset, key_width)
            offset += key_width
        return offsets

    @classmethod
    def _critic_graph_static_views_from_sac_packed(
        cls,
        packed: torch.Tensor,
        critic_shapes: dict[str, torch.Size],
        actor_shapes: dict[str, torch.Size],
    ) -> dict[str, torch.Tensor]:
        offsets = cls._sac_graph_offsets(actor_shapes, critic_shapes)
        views: dict[str, torch.Tensor] = {}
        for key in cls._critic_graph_input_keys():
            offset, width = offsets[key]
            views[key] = packed.narrow(1, offset, width).view(critic_shapes[key])
        return views

    @classmethod
    def _actor_graph_static_views_from_sac_packed(
        cls,
        packed: torch.Tensor,
        actor_shapes: dict[str, torch.Size],
    ) -> dict[str, torch.Tensor]:
        batch_size = int(actor_shapes["obs"][0])
        critic_shapes = {
            "critic": actor_shapes["critic"],
            "actions": actor_shapes["actions"],
            "rewards": torch.Size((batch_size,)),
            "next_obs": actor_shapes["next_obs"],
            "next_critic": actor_shapes["critic"],
            "dones": torch.Size((batch_size,)),
            "truncated": torch.Size((batch_size,)),
        }
        offsets = cls._sac_graph_offsets(actor_shapes, critic_shapes)
        views: dict[str, torch.Tensor] = {}
        for key in cls._actor_graph_input_keys():
            source_key = "actions" if key == "actions" else key
            offset, width = offsets[source_key]
            views[key] = packed.narrow(1, offset, width).view(actor_shapes[key])
        return views

    def _prepare_critic_graph_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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

        prepared = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "truncated": truncated,
            "critic": critic_obs,
            "next_critic": critic_next_obs,
        }
        if "sac_graph_packed_source" in batch:
            prepared["sac_graph_packed_source"] = batch["sac_graph_packed_source"].to(self.device)
        return prepared

    def _copy_critic_graph_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        assert self._cuda_graph_critic_static_inputs is not None
        packed_source = inputs.get("sac_graph_packed_source")
        if packed_source is not None and self._cuda_graph_sac_static_packed_input is not None:
            self._cuda_graph_sac_static_packed_input.copy_(packed_source)
            self._cuda_graph_sac_static_source_ptr = int(packed_source.data_ptr())
            return
        for key, tensor in self._cuda_graph_critic_static_inputs.items():
            tensor.copy_(inputs[key])

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

    def _update_critic_capture_candidate(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        update_target: bool = False,
        normalize_parameters: bool = False,
        metric_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = inputs["actions"]
        rewards = inputs["rewards"]
        next_obs = inputs["next_obs"]
        dones = inputs["dones"]
        truncated = inputs["truncated"]
        critic_obs = inputs["critic"]
        critic_next_obs = inputs["next_critic"]

        critic_loss, reward_scale_std = self._critic_objective_tensors(
            actions,
            rewards,
            next_obs,
            dones,
            truncated,
            critic_obs,
            critic_next_obs,
        )
        # AOTAutograd/Inductor plan temporary storage across the compiled
        # forward/backward boundary.  A clone made during outer graph capture
        # still lives in that graph's private pool and may be reused by later
        # optimizer kernels.  Pre-capture buffers give replay metrics stable
        # addresses and explicit lifetimes outside the graph pool.
        if metric_buffers is None:
            metric_critic_loss = critic_loss.detach().clone()
            metric_reward_scale_std = reward_scale_std.detach().clone()
        else:
            metric_critic_loss, metric_reward_scale_std = metric_buffers
            metric_critic_loss.copy_(critic_loss.detach().reshape(()))
            metric_reward_scale_std.copy_(reward_scale_std.detach().reshape(()))

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._sync_gradients(self.critic.parameters())
        with self._optimizer_finite_gate(self.critic_optimizer, critic_loss):
            self.critic_optimizer.step()
        if normalize_parameters:
            self.critic.normalize_parameters()
        if update_target:
            polyak_update_target(self.target_critic, self.critic, self.tau)
        return metric_critic_loss, metric_reward_scale_std

    def _reset_critic_cuda_graph(self) -> None:
        graph = self._cuda_graph_critic
        self._cuda_graph_critic = None
        if isinstance(graph, torch.cuda.CUDAGraph):
            graph.reset()
        self._cuda_graph_critic_static_inputs = None
        self._cuda_graph_sac_static_packed_input = None
        self._cuda_graph_sac_static_source_ptr = None
        self._cuda_graph_critic_outputs = None
        self._cuda_graph_critic_metric_buffers = None
        self._cuda_graph_critic_shapes = None
        self._cuda_graph_critic_gradient_sync_calls = 0

    def _materialize_capturable_critic_optimizer_state(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> None:
        optimizer_lrs = [group["lr"] for group in self.critic_optimizer.param_groups]
        optimizer_weight_decays = [
            group["weight_decay"] for group in self.critic_optimizer.param_groups
        ]
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if self.device.type == "cuda" else None
        module_buffers = self._snapshot_module_buffers(
            self.actor,
            self.critic,
            self.target_critic,
        )
        try:
            for group in self.critic_optimizer.param_groups:
                group["lr"] = 0.0
                group["weight_decay"] = 0.0
            self._update_critic_capture_candidate(
                inputs,
                update_target=False,
                normalize_parameters=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            self._restore_module_buffers(module_buffers)
            for group, lr, weight_decay in zip(
                self.critic_optimizer.param_groups,
                optimizer_lrs,
                optimizer_weight_decays,
                strict=True,
            ):
                group["lr"] = lr
                group["weight_decay"] = weight_decay

        self.critic_optimizer.zero_grad(set_to_none=True)
        for state in self.critic_optimizer.state.values():
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                step.zero_()
            elif step is not None:
                state["step"] = 0
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                tensor = state.get(name)
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()

    def _capture_critic_cuda_graph(self, inputs: dict[str, torch.Tensor]) -> None:
        self._cuda_graph_critic_shapes = self._critic_graph_input_shapes(inputs)
        packed_source = inputs.get("sac_graph_packed_source")
        if self.use_cuda_graph_critic_packed_staging and packed_source is not None:
            self._cuda_graph_sac_static_packed_input = packed_source.detach().clone()
            actor_shapes = self._actor_graph_input_shapes(inputs)
            self._cuda_graph_critic_static_inputs = self._critic_graph_static_views_from_sac_packed(
                self._cuda_graph_sac_static_packed_input,
                self._cuda_graph_critic_shapes,
                actor_shapes,
            )
        else:
            self._cuda_graph_sac_static_packed_input = None
            self._cuda_graph_critic_static_inputs = {
                key: inputs[key].detach().clone() for key in self._critic_graph_input_keys()
            }
        self._copy_critic_graph_inputs(inputs)

        graph = torch.cuda.CUDAGraph()
        self._cuda_graph_critic_metric_buffers = (
            torch.empty((), device=self.device),
            torch.empty((), device=self.device),
        )
        capture_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
        capture_stream.wait_stream(torch.cuda.current_stream())
        sync_calls = [0]
        self._active_cuda_graph_gradient_sync_calls = sync_calls
        try:
            with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
                self._cuda_graph_critic_outputs = self._update_critic_capture_candidate(
                    self._cuda_graph_critic_static_inputs,
                    update_target=True,
                    normalize_parameters=True,
                    metric_buffers=self._cuda_graph_critic_metric_buffers,
                )
        finally:
            self._active_cuda_graph_gradient_sync_calls = None
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._cuda_graph_critic = graph
        self._cuda_graph_critic_gradient_sync_calls = sync_calls[0]

    def _critic_graph_output_metrics(self, *, read_items: bool = True) -> dict[str, float]:
        if not read_items:
            return {}
        assert self._cuda_graph_critic_outputs is not None
        critic_loss, reward_scale_std = self._cuda_graph_critic_outputs
        return self._read_metric_tensors(
            ("critic_loss", "reward_scale_std"),
            (critic_loss, reward_scale_std),
        )

    def update_critic_cuda_graph(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
        if not self.use_cuda_graph_critic:
            return self.update_critic(batch, read_metrics=read_metrics)
        if self.device.type != "cuda":
            return self.update_critic(batch, read_metrics=read_metrics)
        if self.scaler is not None:
            return self.update_critic(batch, read_metrics=read_metrics)
        if not isinstance(self.obs_normalizer, nn.Identity):
            return self.update_critic(batch, read_metrics=read_metrics)

        inputs = self._prepare_critic_graph_inputs(batch)
        if self._cuda_graph_critic_shapes != self._critic_graph_input_shapes(inputs):
            self._reset_critic_cuda_graph()
            self._materialize_capturable_critic_optimizer_state(inputs)
            self._capture_critic_cuda_graph(inputs)
            # Capturing records kernels but does not execute the training
            # update.  Replay once so the first call has the same semantics as
            # every subsequent call instead of silently dropping one update.
            assert self._cuda_graph_critic is not None
            self._cuda_graph_critic.replay()
            self._record_cuda_graph_gradient_replay(self._cuda_graph_critic_gradient_sync_calls)
            self.critic_scheduler.step()
            return self._critic_graph_output_metrics(read_items=read_metrics)

        assert self._cuda_graph_critic is not None
        self._copy_critic_graph_inputs(inputs)
        self._cuda_graph_critic.replay()
        self._record_cuda_graph_gradient_replay(self._cuda_graph_critic_gradient_sync_calls)
        self.critic_scheduler.step()
        return self._critic_graph_output_metrics(read_items=read_metrics)

    @staticmethod
    def _actor_graph_input_keys() -> tuple[str, ...]:
        return ("obs", "next_obs", "actions", "critic")

    def _actor_graph_input_shapes(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Size]:
        return {key: inputs[key].shape for key in self._actor_graph_input_keys()}

    def _prepare_actor_graph_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        expert_actions = batch["actions"].to(self.device)
        critic_obs = batch["critic"].to(self.device)
        prepared = {
            "obs": self._maybe_normalize_obs(obs, update=False),
            "next_obs": self._maybe_normalize_obs(next_obs, update=False),
            "actions": expert_actions,
            "critic": critic_obs,
        }
        if "sac_graph_packed_source" in batch:
            prepared["sac_graph_packed_source"] = batch["sac_graph_packed_source"].to(self.device)
        return prepared

    def _copy_actor_graph_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        assert self._cuda_graph_actor_static_inputs is not None
        packed_source = inputs.get("sac_graph_packed_source")
        if packed_source is not None:
            static_packed = self._cuda_graph_actor_static_packed_input
            if static_packed is None:
                static_packed = self._cuda_graph_sac_static_packed_input
            if static_packed is not None:
                source_ptr = int(packed_source.data_ptr())
                if (
                    static_packed is not self._cuda_graph_sac_static_packed_input
                    or self._cuda_graph_sac_static_source_ptr != source_ptr
                ):
                    static_packed.copy_(packed_source)
                    if static_packed is self._cuda_graph_sac_static_packed_input:
                        self._cuda_graph_sac_static_source_ptr = source_ptr
                return
        for key, tensor in self._cuda_graph_actor_static_inputs.items():
            tensor.copy_(inputs[key])

    def _update_actor_capture_candidate(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        normalize_parameters: bool = False,
        metric_buffers: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = inputs["obs"]
        next_obs = inputs["next_obs"]
        expert_actions = inputs["actions"]
        critic_obs = inputs["critic"]
        with self._critic_parameters_frozen():
            actor_loss, entropy = self._actor_objective_tensors(
                obs,
                next_obs,
                expert_actions,
                critic_obs,
            )
        if metric_buffers is None:
            metric_actor_loss = actor_loss.detach().clone()
            metric_entropy = entropy.detach().clone()
        else:
            metric_actor_loss, metric_entropy, _, _ = metric_buffers
            metric_actor_loss.copy_(actor_loss.detach().reshape(()))
            metric_entropy.copy_(entropy.detach().reshape(()))

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._sync_gradients(self.actor.parameters())
        with self._optimizer_finite_gate(self.actor_optimizer, actor_loss):
            self.actor_optimizer.step()
        if normalize_parameters:
            self.actor.normalize_parameters()

        temp_value = self.temperature()
        temp_loss = temp_value * (entropy - self.target_entropy)
        if metric_buffers is None:
            metric_temp_value = temp_value.detach().clone()
            metric_temp_loss = temp_loss.detach().clone()
        else:
            _, _, metric_temp_value, metric_temp_loss = metric_buffers
            metric_temp_value.copy_(temp_value.detach().reshape(()))
            metric_temp_loss.copy_(temp_loss.detach().reshape(()))
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
        self._sync_gradients(self.temperature.parameters())
        with self._optimizer_finite_gate(self.temperature_optimizer, temp_loss):
            self.temperature_optimizer.step()
        return metric_actor_loss, metric_entropy, metric_temp_value, metric_temp_loss

    def _reset_actor_cuda_graph(self) -> None:
        graph = self._cuda_graph_actor
        self._cuda_graph_actor = None
        if isinstance(graph, torch.cuda.CUDAGraph):
            graph.reset()
        self._cuda_graph_actor_static_inputs = None
        self._cuda_graph_actor_static_packed_input = None
        self._cuda_graph_actor_outputs = None
        self._cuda_graph_actor_metric_buffers = None
        self._cuda_graph_actor_shapes = None
        self._cuda_graph_actor_gradient_sync_calls = 0

    def _materialize_capturable_actor_optimizer_state(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> None:
        optimizers = (self.actor_optimizer, self.temperature_optimizer)
        optimizer_lrs = [
            [group["lr"] for group in optimizer.param_groups] for optimizer in optimizers
        ]
        optimizer_weight_decays = [
            [group["weight_decay"] for group in optimizer.param_groups] for optimizer in optimizers
        ]
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if self.device.type == "cuda" else None
        module_buffers = self._snapshot_module_buffers(self.actor, self.critic)
        try:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0
                    group["weight_decay"] = 0.0
            self._update_actor_capture_candidate(inputs, normalize_parameters=False)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            self._restore_module_buffers(module_buffers)
            for optimizer, lrs, weight_decays in zip(
                optimizers,
                optimizer_lrs,
                optimizer_weight_decays,
                strict=True,
            ):
                for group, lr, weight_decay in zip(
                    optimizer.param_groups,
                    lrs,
                    weight_decays,
                    strict=True,
                ):
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
            for state in optimizer.state.values():
                step = state.get("step")
                if isinstance(step, torch.Tensor):
                    step.zero_()
                elif step is not None:
                    state["step"] = 0
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    tensor = state.get(name)
                    if isinstance(tensor, torch.Tensor):
                        tensor.zero_()

    def _capture_actor_cuda_graph(self, inputs: dict[str, torch.Tensor]) -> None:
        self._cuda_graph_actor_shapes = self._actor_graph_input_shapes(inputs)
        packed_source = inputs.get("sac_graph_packed_source")
        if self.use_cuda_graph_actor_packed_staging and packed_source is not None:
            if (
                self._cuda_graph_sac_static_packed_input is not None
                and self._cuda_graph_sac_static_packed_input.shape == packed_source.shape
            ):
                self._cuda_graph_actor_static_packed_input = (
                    self._cuda_graph_sac_static_packed_input
                )
            else:
                self._cuda_graph_actor_static_packed_input = packed_source.detach().clone()
            self._cuda_graph_actor_static_inputs = self._actor_graph_static_views_from_sac_packed(
                self._cuda_graph_actor_static_packed_input,
                self._cuda_graph_actor_shapes,
            )
        else:
            self._cuda_graph_actor_static_packed_input = None
            self._cuda_graph_actor_static_inputs = {
                key: inputs[key].detach().clone() for key in self._actor_graph_input_keys()
            }
        self._copy_actor_graph_inputs(inputs)

        graph = torch.cuda.CUDAGraph()
        self._cuda_graph_actor_metric_buffers = (
            torch.empty((), device=self.device),
            torch.empty((), device=self.device),
            torch.empty((), device=self.device),
            torch.empty((), device=self.device),
        )
        capture_stream = cast(torch.cuda.Stream, torch.cuda.Stream())
        capture_stream.wait_stream(torch.cuda.current_stream())
        sync_calls = [0]
        self._active_cuda_graph_gradient_sync_calls = sync_calls
        try:
            with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
                self._cuda_graph_actor_outputs = self._update_actor_capture_candidate(
                    self._cuda_graph_actor_static_inputs,
                    normalize_parameters=True,
                    metric_buffers=self._cuda_graph_actor_metric_buffers,
                )
        finally:
            self._active_cuda_graph_gradient_sync_calls = None
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self._cuda_graph_actor = graph
        self._cuda_graph_actor_gradient_sync_calls = sync_calls[0]

    def _actor_graph_output_metrics(self, *, read_items: bool = True) -> dict[str, float]:
        if not read_items:
            return {}
        assert self._cuda_graph_actor_outputs is not None
        actor_loss, entropy, temp_value, temp_loss = self._cuda_graph_actor_outputs
        return self._read_metric_tensors(
            ("actor_loss", "actor_entropy", "temperature", "temperature_loss"),
            (actor_loss, entropy, temp_value, temp_loss),
        )

    def update_actor_cuda_graph(
        self,
        batch: dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> dict[str, float]:
        if not self.use_cuda_graph_actor:
            return self.update_actor(batch, read_metrics=read_metrics)
        if self.device.type != "cuda":
            return self.update_actor(batch, read_metrics=read_metrics)
        if self.scaler is not None:
            return self.update_actor(batch, read_metrics=read_metrics)
        if not isinstance(self.obs_normalizer, nn.Identity):
            return self.update_actor(batch, read_metrics=read_metrics)

        inputs = self._prepare_actor_graph_inputs(batch)
        if self._cuda_graph_actor_shapes != self._actor_graph_input_shapes(inputs):
            self._reset_actor_cuda_graph()
            self._materialize_capturable_actor_optimizer_state(inputs)
            self._capture_actor_cuda_graph(inputs)
            assert self._cuda_graph_actor is not None
            self._cuda_graph_actor.replay()
            self._record_cuda_graph_gradient_replay(self._cuda_graph_actor_gradient_sync_calls)
            self.actor_scheduler.step()
            self.temperature_scheduler.step()
            return self._actor_graph_output_metrics(read_items=read_metrics)

        assert self._cuda_graph_actor is not None
        self._copy_actor_graph_inputs(inputs)
        self._cuda_graph_actor.replay()
        self._record_cuda_graph_gradient_replay(self._cuda_graph_actor_gradient_sync_calls)
        self.actor_scheduler.step()
        self.temperature_scheduler.step()
        return self._actor_graph_output_metrics(read_items=read_metrics)

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
            if self._cuda_graph_actor_outputs is None:
                return {}
            return self._actor_graph_output_metrics(read_items=True)
        return {
            name: float(value)
            for name, value in zip(
                ("actor_loss", "actor_entropy", "temperature", "temperature_loss"),
                values.cpu().tolist(),
                strict=True,
            )
        }

    @property
    def cuda_graph_critic_captures_target_update(self) -> bool:
        """Whether critic graph replay already performs the Polyak update."""
        return bool(
            self.use_cuda_graph_critic
            and self._device_type == "cuda"
            and self.scaler is None
            and isinstance(self.obs_normalizer, nn.Identity)
        )

    def soft_update_target(self) -> None:
        polyak_update_target(self.target_critic, self.critic, self.tau)

    def _dp_cuda_graph_gradient_sync_enabled(self) -> bool:
        return bool(
            self._gradient_sync is not None
            and self.scaler is None
            and isinstance(self.obs_normalizer, nn.Identity)
            and (self.use_cuda_graph_critic or self.use_cuda_graph_actor)
        )

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
