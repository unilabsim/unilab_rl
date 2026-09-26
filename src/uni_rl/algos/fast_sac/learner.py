"""FastSAC Learner — replicated from holosoma's FastSAC implementation.

Network architecture:
- Actor: MLP with SiLU + LayerNorm, tanh-squashed Gaussian
- Critic: Distributional Q-Networks (C51 variant, num_atoms=101)
- Automatic entropy coefficient (alpha) learning

Hyperparameters aligned with holosoma FastSACConfig defaults.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from uni_rl.algos.common.compile import get_torch_compile_for_cuda, is_hip_runtime
from uni_rl.algos.common.learner_boilerplate import (
    LearnerBoilerplateMixin,
    fused_adam_supported,
    polyak_update_target,
    resolve_finite_check_flags,
)
from uni_rl.algos.common.normalization import EmpiricalNormalization
from uni_rl.offpolicy.warmup import (
    OffPolicyWarmupContext,
    capture_rng_state,
    make_offpolicy_warmup_batch,
    restore_rng_state,
)


@contextmanager
def _cuda_nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# Actor Network (holosoma-style: SiLU + LayerNorm + Tanh squashing)
# ---------------------------------------------------------------------------


class SACActor(nn.Module):
    """Stochastic actor for SAC with tanh-squashed Gaussian policy.

    Architecture: Linear→LN→SiLU → Linear→LN→SiLU → Linear→LN→SiLU → fc_mu + fc_logstd
    Hidden dims: [hidden_dim, hidden_dim//2, hidden_dim//4]
    """

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: str | torch.device = "cpu",
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.use_tanh = use_tanh

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 4, action_dim, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, action_dim, device=device)

        # Zero-init output heads (holosoma style)
        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

        # Action scaling
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(action_dim, device=device))
        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(action_dim, device=device))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, mean, log_std)."""
        x = self.net(obs)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)

        # Squash log_std to [log_std_min, log_std_max] (SpinUp / Denis Yarats style)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        # NaN protection: clamp mean to prevent exploding values
        mean = torch.clamp(mean, -10.0, 10.0)
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=self.log_std_min)

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def as_export_module(self) -> "nn.Module":
        """Return a single-input/single-output wrapper suitable for torch.onnx.export."""
        actor = self

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = actor

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action, _, _ = self.base(obs)
                return cast(torch.Tensor, action)

        return _Wrapper()

    def get_actions_and_log_probs(
        self,
        obs: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities. Returns (action, log_prob, log_std)."""
        _, mean, log_std = self(obs)
        action, log_prob = self._sample_action_and_log_prob(mean, log_std, eps=eps)
        return action, log_prob, log_std

    def _sample_action_and_log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        if eps is None:
            eps = torch.randn_like(mean)
        raw_action = mean + std * eps
        log_prob = -0.5 * (
            ((raw_action - mean) / std).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi)
        )

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            action = raw_action

        return action, log_prob.sum(1)

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get exploration actions.

        Args:
            obs: Batched observations.
            dones: Unused for SAC; kept for API alignment with other off-policy actors.
            deterministic: Whether to return deterministic policy actions.
        """
        # Backward compatibility: previous signature was explore(obs, deterministic=False).
        if isinstance(dones, bool):
            deterministic = dones
            dones = None
        _ = dones

        _, mean, log_std = self.forward(obs)
        if deterministic:
            if self.use_tanh:
                return torch.tanh(mean) * self.action_scale + self.action_bias
            return mean

        raw_action = mean + log_std.exp() * torch.randn_like(mean)

        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action


# ---------------------------------------------------------------------------
# Distributional Q-Network (C51 variant, from holosoma)
# ---------------------------------------------------------------------------


class DistributionalQNetwork(nn.Module):
    """Single distributional Q-network (C51).

    Architecture: Linear→LN→SiLU → Linear→LN→SiLU → Linear→LN→SiLU → Linear(num_atoms)
    Input: concat(obs, action)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        input_dim = obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)  # type: ignore[no-any-return]

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Categorical projection for distributional RL."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        # target_z is clamped to [v_min, v_max], so b stays within
        # [0, num_atoms - 1].  Splitting the mass between floor(b) and
        # floor(b)+1 (clamped to the support) reproduces the integer-b edge
        # cases without the eq/logical_and/where mask chain.
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = (lower + 1).clamp(max=self.num_atoms - 1)
        upper_weight = b - lower.float()
        lower_weight = 1.0 - upper_weight

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        # Build the row offsets inside the traced expression.  A Python-side
        # cache would retain a CUDA Graph Trees output tensor; its storage can
        # be overwritten by the next replay and then fail when Dynamo reads it.
        # Flattened rows are separated by ``num_atoms``, not by one atom.
        offset = torch.arange(batch_size, device=device).unsqueeze(1) * self.num_atoms

        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1
        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * lower_weight).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * upper_weight).view(-1))
        return proj_dist


class SACCritic(nn.Module):
    """Ensemble of distributional Q-networks for SAC.

    Uses ``num_q_networks`` independent DistributionalQNetwork instances.
    """

    q_support: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.num_q_networks = num_q_networks

        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    num_atoms=num_atoms,
                    v_min=v_min,
                    v_max=v_max,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    device=device,
                )
                for _ in range(num_q_networks)
            ]
        )

        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns stacked logits: (num_q_nets, batch, num_atoms)."""
        outputs = [qnet(obs, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Project for all Q-networks: (num_q_nets, batch, num_atoms)."""
        projections = [
            qnet.projection(  # type: ignore[operator]
                obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from probabilities using support."""
        return torch.sum(probs * self.q_support, dim=-1)


# ---------------------------------------------------------------------------
# FastSACLearner — the training algorithm
# ---------------------------------------------------------------------------


class FastSACLearner(LearnerBoilerplateMixin):
    """FastSAC learner with holosoma-aligned hyperparameters.

    Key hyperparameters (aligned with holosoma FastSACConfig):
    - gamma=0.97, tau=0.125
    - batch_size=8192, num_updates=8, policy_frequency=4
    - alpha_init=0.001, target_entropy_ratio=0.0
    - AdamW with betas=(0.9, 0.95), weight_decay=0.001
    - Distributional critic (C51, num_atoms=101)
    """

    supports_deferred_update_metrics = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int,
        device: str = "cpu",
        # Hyperparameters aligned with holosoma
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.001,
        target_entropy_ratio: float = 0.0,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        num_q_networks: int = 2,
        use_layer_norm: bool = True,
        use_tanh: bool = True,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        weight_decay: float = 0.001,
        max_grad_norm: float = 0.0,
        use_autotune: bool = True,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        use_compile: bool = False,
        obs_normalization: bool = False,
        nvtx_profile_ranges: bool = False,
    ):
        self.device = device
        self._device_type = torch.device(device).type
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.use_autotune = use_autotune
        self.use_amp = bool(use_amp) and self._device_type in ("cuda", "xpu")
        self._nvidia_cuda = self._device_type == "cuda" and not is_hip_runtime()
        compile_fn = get_torch_compile_for_cuda(self.device, warn=not self._nvidia_cuda)
        if self._nvidia_cuda and compile_fn is None:
            raise RuntimeError("FastSAC requires CUDA Inductor/Triton on NVIDIA CUDA")
        # NVIDIA CUDA always uses the performance path; the legacy opt-out is
        # retained only for ROCm/HIP, MPS, CPU, and other compatibility devices.
        self.use_compile = self._nvidia_cuda or (bool(use_compile) and compile_fn is not None)
        # The compiled CUDA Graph hot path cannot branch on host-visible finite
        # checks and relies on the existing NaN guard/metrics boundary.  MPS
        # has no device-side skip mechanism (its fused AdamW kernel ignores
        # `found_inf`), so host checks there run only on the metrics-reading
        # update of each iteration — see `resolve_finite_check_flags`.
        self._host_finite_checks, self._metrics_finite_checks = resolve_finite_check_flags(
            self._device_type,
            device_gated=self._device_type == "cuda" and self.use_compile,
        )
        self._gradient_sync: Callable[[Iterable[torch.Tensor]], None] | None = None
        self.nvtx_profile_ranges = bool(nvtx_profile_ranges) and self._device_type == "cuda"
        self.amp_dtype = amp_dtype
        self._amp_dtype = self._resolve_amp_dtype(amp_dtype, self._device_type)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_obs_dim = critic_obs_dim

        # Build actor (uses obs only)
        self.actor = SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )

        self.qnet = SACCritic(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            use_layer_norm=use_layer_norm,
            num_q_networks=num_q_networks,
            device=device,
        )

        # Target critic
        self.qnet_target = SACCritic(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=critic_hidden_dim,
            use_layer_norm=use_layer_norm,
            num_q_networks=num_q_networks,
            device=device,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Entropy coefficient
        self.log_alpha = torch.tensor([math.log(alpha_init)], requires_grad=True, device=device)
        self.target_entropy = -action_dim * target_entropy_ratio
        self._zero_metric = torch.zeros((), device=device)
        self._optimizer_grad_scale = torch.ones((), device=device)
        self._optimizer_found_inf = torch.zeros((), device=device)

        self.obs_normalizer: EmpiricalNormalization | nn.Identity
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()

        # Fused AdamW collapses the per-parameter host loop into one kernel.
        # Besides CUDA, torch >= 2.6 ships an MPS fused kernel; on MPS the
        # single-tensor fallback additionally performs a blocking `.item()`
        # per parameter per step, which dominates learner time there.
        # `capturable` stays CUDA-only (rejected by torch on other devices).
        _fused = fused_adam_supported(self._device_type)
        _optimizer_cuda_kwargs = (
            {"capturable": True} if _fused and self._device_type == "cuda" else {}
        )

        # Optimizers (AdamW with holosoma betas)
        self.q_optimizer = optim.AdamW(
            self.qnet.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
            fused=_fused,
            betas=(0.9, 0.95),
            **_optimizer_cuda_kwargs,
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            fused=_fused,
            betas=(0.9, 0.95),
            **_optimizer_cuda_kwargs,
        )
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha],
            lr=alpha_lr,
            fused=_fused,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            **_optimizer_cuda_kwargs,
        )

        # Step counter
        self.update_count = 0

        # AMP scaler for mixed precision (fp16 only; bf16 has fp32 range and skips scaler)
        self.scaler = (
            torch.amp.GradScaler("cuda")  # pyright: ignore[reportPrivateImportUsage]
            if self._should_use_grad_scaler(self.use_amp, self._device_type, self._amp_dtype)
            else None
        )
        self._pending_actor_metric_values: torch.Tensor | None = None
        self._pending_cycle_critic_metric_values: torch.Tensor | None = None
        self._pending_cycle_metric_values: torch.Tensor | None = None
        self._q_update_finite = torch.ones((), device=device)
        self._target_tau = torch.full((), tau, device=device)
        self._update_cycle_graph: torch.cuda.CUDAGraph | None = None
        self._update_cycle_graph_cache_key: tuple[object, ...] | None = None
        self._update_cycle_static_batch: Dict[str, torch.Tensor] | None = None
        self._update_cycle_graph_metric_values: torch.Tensor | None = None
        self._compile_full_update_cycle = bool(self.use_compile and self._nvidia_cuda)
        if self._compile_full_update_cycle and self.scaler is not None:
            raise ValueError(
                "FastSAC CUDA compile mode requires bf16 (or fp32); "
                "fp16 GradScaler is incompatible with the whole-cycle graph"
            )
        if self._compile_full_update_cycle and not isinstance(self.obs_normalizer, nn.Identity):
            raise ValueError("FastSAC whole-cycle CUDA graphs do not yet support obs normalization")
        if self._compile_full_update_cycle and self.nvtx_profile_ranges:
            raise ValueError("FastSAC whole-cycle CUDA graphs do not support NVTX ranges")
        if self.use_compile:
            if self._compile_full_update_cycle:
                self._materialize_capturable_optimizer_state()
            self._compile_training_methods()

    @property
    def use_update_cycle(self) -> bool:
        """Whether this learner owns the whole-cycle update orchestration."""
        return self._compile_full_update_cycle

    def prepare_for_collection(self, warmup_context: OffPolicyWarmupContext) -> None:
        """Compile/capture all learner-owned update cold paths before collection.

        The whole-cycle path delegates to the owner's zero-learning-rate warmup
        and graph capture helper, which restores model, optimizer, RNG, and
        gradient state. Compatibility paths use synthetic replay data and then
        restore the complete checkpointable learner state and all RNG streams.
        The resulting compiled-code cache is intentionally retained.
        """
        large_batch = warmup_context.replay_batch or make_offpolicy_warmup_batch(
            warmup_context,
            obs_dim=self.obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        if self.use_update_cycle:
            self._ensure_update_cycle_graph(
                large_batch,
                updates_per_step=warmup_context.updates_per_step,
                policy_frequency=warmup_context.policy_frequency,
                target_frequency=warmup_context.target_frequency,
                policy_before_critic=warmup_context.policy_before_critic,
            )
            return

        saved_state = copy.deepcopy(self.get_state_dict())
        saved_scaler = copy.deepcopy(self.scaler.state_dict()) if self.scaler is not None else None
        saved_found_inf = self._optimizer_found_inf.detach().clone()
        saved_q_finite = self._q_update_finite.detach().clone()
        rng_state = capture_rng_state(self.device)
        try:
            batch_size = max(1, int(warmup_context.batch_size))
            for update_idx in range(max(1, int(warmup_context.updates_per_step))):
                start = update_idx * batch_size
                batch = {
                    key: value[start : start + batch_size] for key, value in large_batch.items()
                }
                do_actor_update = update_idx % max(1, warmup_context.policy_frequency) == 0
                if warmup_context.policy_before_critic and do_actor_update:
                    self.update_actor(batch, read_metrics=False)
                self.update_critic(batch, read_metrics=False)
                if not warmup_context.policy_before_critic and do_actor_update:
                    self.update_actor(batch, read_metrics=False)
                if update_idx % max(1, warmup_context.target_frequency) == 0:
                    self.soft_update_target()
        finally:
            self.load_state_dict(saved_state)
            if self.scaler is not None and saved_scaler is not None:
                self.scaler.load_state_dict(saved_scaler)
            self._optimizer_found_inf.copy_(saved_found_inf)
            self._q_update_finite.copy_(saved_q_finite)
            self._pending_actor_metric_values = None
            self._pending_cycle_critic_metric_values = None
            self._pending_cycle_metric_values = None
            restore_rng_state(rng_state)
            self._zero_optimizer_gradients(self.q_optimizer)
            self._zero_optimizer_gradients(self.actor_optimizer)
            self._zero_optimizer_gradients(self.alpha_optimizer)
            if self._device_type == "cuda":
                torch.cuda.synchronize(self.device)

    def set_gradient_sync(self, sync: Callable[[Iterable[torch.Tensor]], None] | None) -> None:
        """Attach the compatibility-device DP reduction."""
        if sync is not None and self._compile_full_update_cycle:
            raise RuntimeError("FastSAC NVIDIA CUDA whole-cycle mode does not support DP fallback")
        self._gradient_sync = sync

    def normalize_obs(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        """Normalize actor observations using running statistics."""
        if isinstance(self.obs_normalizer, nn.Identity):
            return obs
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        if update:
            self._update_obs_normalizer(obs)
            return cast(torch.Tensor, normalizer(obs, update=False))
        return cast(torch.Tensor, normalizer(obs, update=False))

    def _arm_optimizer_finite_gate(
        self,
        optimizer: optim.Optimizer,
        loss: torch.Tensor,
    ) -> None:
        """Arm a persistent, graph-safe fused-optimizer finite gate.

        The gate tensors stay attached to the optimizer. Fused CUDA AdamW only
        reads those attributes while stepping, so sharing one device scalar
        across optimizers is safe as long as it is re-armed immediately before
        each step.
        """
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
            if optimizer is self.q_optimizer:
                self._q_update_finite.copy_(torch.logical_not(found_inf.detach()))
        elif optimizer is self.q_optimizer:
            self._q_update_finite.copy_(torch.isfinite(loss.detach().all()))
        setattr(optimizer, "grad_scale", self._optimizer_grad_scale)
        setattr(optimizer, "found_inf", found_inf)

    def _compile_training_methods(self) -> None:
        """Compile loss kernels for the selected FastSAC orchestration scope."""
        compile_fn = get_torch_compile_for_cuda(self.device, warn=True)
        if compile_fn is None:
            return
        if self._compile_full_update_cycle:
            # The owner-managed CUDA Graph must not nest Inductor Graph Trees.
            # Portable max-autotune lets Triton select kernels for the installed
            # GPU instead of restoring architecture-specific learner paths.
            compile_kwargs = {
                "dynamic": False,
                "mode": "max-autotune-no-cudagraphs",
            }
        else:
            compile_kwargs = {
                "dynamic": False,
                "options": {
                    "triton.cudagraphs": True,
                },
            }
        self.__dict__["_critic_loss_tensors"] = compile_fn(
            self._critic_loss_tensors,
            **compile_kwargs,
        )
        self.__dict__["_actor_loss_tensors"] = compile_fn(
            self._actor_loss_tensors,
            **compile_kwargs,
        )

    def _materialize_capturable_optimizer_state(self) -> None:
        """Create fused AdamW state before the whole-cycle graph is captured."""
        optimizers = (self.q_optimizer, self.actor_optimizer, self.alpha_optimizer)
        saved_groups = [
            [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
            for optimizer in optimizers
        ]
        try:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0
                    group["weight_decay"] = 0.0
                    for parameter in group["params"]:
                        parameter.grad = torch.zeros_like(parameter)
                optimizer.step()
        finally:
            for optimizer, saved_group in zip(optimizers, saved_groups, strict=True):
                for group, (lr, weight_decay) in zip(
                    optimizer.param_groups,
                    saved_group,
                    strict=True,
                ):
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.zero_()
            for optimizer in optimizers:
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.zero_()

    def _zero_optimizer_gradients(self, optimizer: optim.Optimizer) -> None:
        """Zero gradients without the optimizer's graph-skipped Python entrypoint."""
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

    def _get_actions_and_log_probs_for_critic(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actor actions for critic targets.

        Subclasses can use ``critic_obs`` to supply auxiliary policy context while
        preserving the standard SAC update path.
        """
        del critic_obs
        return self.actor.get_actions_and_log_probs(actor_obs, eps=eps)

    def _get_actions_and_log_probs_for_actor(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actor actions for the actor loss update."""
        del critic_obs
        return self.actor.get_actions_and_log_probs(actor_obs, eps=eps)

    def _critic_loss_tensors(
        self,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        next_action_eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bootstrap = torch.clamp(1.0 - dones.float() + truncated.float(), 0.0, 1.0)
        discount = torch.full_like(dones, self.gamma)

        with torch.no_grad():
            with self._autocast():
                next_actions, next_log_probs, _ = self._get_actions_and_log_probs_for_critic(
                    next_obs,
                    critic_next_obs,
                    eps=next_action_eps,
                )
            adjusted_rewards = (
                rewards - discount * bootstrap * self.log_alpha.exp() * next_log_probs
            )

            with self._autocast():
                target_distributions = self.qnet_target.projection(
                    critic_next_obs, next_actions, adjusted_rewards, bootstrap, discount
                )
                target_values = self.qnet_target.get_value(target_distributions)
                target_q_max = target_values.max()
                target_q_min = target_values.min()

        with self._autocast():
            q_outputs = self.qnet(critic_obs, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1).clamp(min=-30.0)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        return qf_loss, target_q_max, target_q_min, next_log_probs.detach()

    def _alpha_loss_tensor(self, next_log_probs: torch.Tensor) -> torch.Tensor:
        entropy_error_mean = (next_log_probs + self.target_entropy).detach().mean()
        return -(self.log_alpha.exp() * entropy_error_mean)

    def _actor_loss_tensors(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action_eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._autocast():
            actions, log_probs, log_std = self._get_actions_and_log_probs_for_actor(
                obs,
                critic_obs,
                eps=action_eps,
            )

        with torch.no_grad():
            policy_entropy = -log_probs.mean()

        with self._autocast():
            q_outputs = self.qnet(critic_obs, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = self.qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        return actor_loss, policy_entropy

    @staticmethod
    def _read_metric_tensors(
        names: tuple[str, ...],
        tensors: tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            return {name: float(tensor.item()) for name, tensor in zip(names, tensors, strict=True)}
        values = torch.stack([tensor.detach().reshape(()) for tensor in tensors]).cpu().tolist()
        return {name: float(value) for name, value in zip(names, values, strict=True)}

    def read_deferred_actor_metrics(self) -> Dict[str, float]:
        values = self._pending_actor_metric_values
        self._pending_actor_metric_values = None
        if values is not None:
            names = (
                "Loss/actor",
                "Train/actor_gradient_norm",
                "Loss/entropy",
            )
            return {
                name: float(value) for name, value in zip(names, values.cpu().tolist(), strict=True)
            }
        return {}

    def read_deferred_cycle_metrics(self) -> Dict[str, float]:
        values = self._pending_cycle_metric_values
        has_actor = values is not None and values.numel() == 9
        self._pending_cycle_critic_metric_values = None
        self._pending_actor_metric_values = None
        self._pending_cycle_metric_values = None
        if values is None:
            return {}
        metric_names: tuple[str, ...] = (
            "Loss/critic",
            "Train/critic_gradient_norm",
            "Train/target_q_max",
            "Train/target_q_min",
            "Loss/temperature",
            "Policy/temperature",
        )
        if has_actor:
            metric_names = metric_names + (
                "Loss/actor",
                "Train/actor_gradient_norm",
                "Loss/entropy",
            )
        return {
            name: float(value)
            for name, value in zip(metric_names, values.cpu().tolist(), strict=True)
        }

    def _run_update_cycle_core(
        self,
        large_batch: Dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        self._pending_actor_metric_values = None
        self._pending_cycle_critic_metric_values = None
        self._pending_cycle_metric_values = None
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

    @staticmethod
    def _update_cycle_graph_key(
        large_batch: Dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> tuple[object, ...]:
        shapes = tuple((key, tuple(value.shape), value.dtype) for key, value in large_batch.items())
        return (
            updates_per_step,
            policy_frequency,
            target_frequency,
            policy_before_critic,
            shapes,
        )

    def _warm_update_cycle_graph(
        self,
        large_batch: Dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
    ) -> None:
        """Compile and dry-run all captured kernels without changing training state."""
        modules: tuple[nn.Module, ...] = (self.actor, self.qnet, self.qnet_target)
        optimizers = (self.q_optimizer, self.actor_optimizer, self.alpha_optimizer)
        saved_models = [copy.deepcopy(module.state_dict()) for module in modules]
        saved_optimizers = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        saved_groups = [
            [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
            for optimizer in optimizers
        ]
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)
        stream = torch.cuda.Stream(device=self.device)
        try:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0
                    group["weight_decay"] = 0.0
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
            for optimizer, saved_group in zip(optimizers, saved_groups, strict=True):
                for group, (lr, weight_decay) in zip(
                    optimizer.param_groups,
                    saved_group,
                    strict=True,
                ):
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay
            for module, state_dict in zip(modules, saved_models, strict=True):
                module.load_state_dict(state_dict)
            for optimizer, state_dict in zip(optimizers, saved_optimizers, strict=True):
                optimizer.load_state_dict(state_dict)
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
            self._zero_optimizer_gradients(self.q_optimizer)
            self._zero_optimizer_gradients(self.actor_optimizer)
            self._zero_optimizer_gradients(self.alpha_optimizer)
            torch.cuda.synchronize(self.device)

    def _ensure_update_cycle_graph(
        self,
        large_batch: Dict[str, torch.Tensor],
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
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(self.device), torch.cuda.graph(graph):
            self._run_update_cycle_core(
                static_batch,
                updates_per_step=updates_per_step,
                policy_frequency=policy_frequency,
                target_frequency=target_frequency,
                policy_before_critic=policy_before_critic,
            )
        self._update_cycle_graph = graph
        self._update_cycle_graph_metric_values = self._pending_cycle_metric_values

    def update_cycle(
        self,
        large_batch: Dict[str, torch.Tensor],
        *,
        updates_per_step: int,
        policy_frequency: int,
        target_frequency: int,
        policy_before_critic: bool,
        read_metrics: bool = False,
    ) -> None:
        """Run the complete learner update block used by the off-policy runner."""
        del read_metrics  # Metrics are staged on device and read once by the runner.
        if not self.use_update_cycle:
            raise RuntimeError(
                "FastSAC update_cycle() requires the NVIDIA CUDA whole-cycle path; "
                "use the per-update methods for compatibility devices"
            )
        self._ensure_update_cycle_graph(
            large_batch,
            updates_per_step=updates_per_step,
            policy_frequency=policy_frequency,
            target_frequency=target_frequency,
            policy_before_critic=policy_before_critic,
        )
        assert self._update_cycle_graph is not None
        self._update_cycle_graph.replay()
        self._pending_cycle_metric_values = self._update_cycle_graph_metric_values

    def update_critic(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> Dict[str, float]:
        """One critic update step."""
        obs = batch["obs"]
        critic_obs = batch["critic"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        critic_next_obs = batch["next_critic"]
        dones = batch["dones"]
        truncated = batch["truncated"]

        self.normalize_obs(obs, update=True)
        next_obs = self.normalize_obs(next_obs, update=False)

        with _cuda_nvtx_range("critic/loss_compiled", self.nvtx_profile_ranges):
            qf_loss, target_q_max, target_q_min, next_log_probs = self._critic_loss_tensors(
                critic_obs,
                actions,
                rewards,
                next_obs,
                critic_next_obs,
                dones,
                truncated,
            )

        # Skip if NaN
        if self._finite_check_ok(qf_loss, read_metrics):
            self._zero_optimizer_gradients(self.q_optimizer)
            if self.scaler:
                with _cuda_nvtx_range("critic/backward", self.nvtx_profile_ranges):
                    self.scaler.scale(qf_loss).backward()
                self._sync_gradients(self.qnet.parameters())
                self.scaler.unscale_(self.q_optimizer)
                if self.max_grad_norm > 0:
                    with _cuda_nvtx_range("critic/grad_clip", self.nvtx_profile_ranges):
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.qnet.parameters(), max_norm=self.max_grad_norm
                        )
                else:
                    critic_grad_norm = self._zero_metric
                with _cuda_nvtx_range("critic/q_optimizer_step", self.nvtx_profile_ranges):
                    self.scaler.step(self.q_optimizer)
                self.scaler.update()
            else:
                with _cuda_nvtx_range("critic/backward", self.nvtx_profile_ranges):
                    qf_loss.backward()
                self._sync_gradients(self.qnet.parameters())
                if self.max_grad_norm > 0:
                    with _cuda_nvtx_range("critic/grad_clip", self.nvtx_profile_ranges):
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.qnet.parameters(), max_norm=self.max_grad_norm
                        )
                else:
                    critic_grad_norm = self._zero_metric
                with _cuda_nvtx_range("critic/q_optimizer_step", self.nvtx_profile_ranges):
                    self._arm_optimizer_finite_gate(self.q_optimizer, qf_loss)
                    self.q_optimizer.step()
        else:
            critic_grad_norm = self._zero_metric

        # Alpha loss (temperature update) - matching holosoma
        alpha_loss = self._zero_metric
        if self.use_autotune:
            with _cuda_nvtx_range("critic/alpha_update", self.nvtx_profile_ranges):
                self._zero_optimizer_gradients(self.alpha_optimizer)
            with _cuda_nvtx_range("critic/alpha_loss", self.nvtx_profile_ranges):
                alpha_loss = self._alpha_loss_tensor(next_log_probs)
            if self._finite_check_ok(alpha_loss, read_metrics):
                with _cuda_nvtx_range("critic/alpha_backward", self.nvtx_profile_ranges):
                    alpha_loss.backward()
                self._sync_gradients((self.log_alpha,))
                with _cuda_nvtx_range("critic/alpha_optimizer_step", self.nvtx_profile_ranges):
                    self._arm_optimizer_finite_gate(self.alpha_optimizer, alpha_loss)
                    self.alpha_optimizer.step()

        if not read_metrics:
            self._pending_cycle_critic_metric_values = torch.stack(
                [
                    tensor.detach().reshape(())
                    for tensor in (
                        qf_loss,
                        critic_grad_norm,
                        target_q_max,
                        target_q_min,
                        alpha_loss,
                        self.log_alpha.exp(),
                    )
                ]
            )
            return {}
        return self._read_metric_tensors(
            (
                "Loss/critic",
                "Train/critic_gradient_norm",
                "Train/target_q_max",
                "Train/target_q_min",
                "Loss/temperature",
                "Policy/temperature",
            ),
            (
                qf_loss,
                critic_grad_norm,
                target_q_max,
                target_q_min,
                alpha_loss,
                self.log_alpha.exp(),
            ),
        )

    def update_actor(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        read_metrics: bool = True,
    ) -> Dict[str, float]:
        """One actor update step."""
        obs = batch["obs"]
        critic_obs = batch["critic"]

        obs = self.normalize_obs(obs, update=False)
        self._pending_actor_metric_values = None
        with _cuda_nvtx_range("actor/loss_compiled", self.nvtx_profile_ranges):
            actor_loss, policy_entropy = self._actor_loss_tensors(obs, critic_obs)

        # Skip if NaN
        if self._finite_check_ok(actor_loss, read_metrics):
            self._zero_optimizer_gradients(self.actor_optimizer)
            if self.scaler:
                with _cuda_nvtx_range("actor/backward", self.nvtx_profile_ranges):
                    self.scaler.scale(actor_loss).backward()
                self._sync_gradients(self.actor.parameters())
                self.scaler.unscale_(self.actor_optimizer)
                if self.max_grad_norm > 0:
                    with _cuda_nvtx_range("actor/grad_clip", self.nvtx_profile_ranges):
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), max_norm=self.max_grad_norm
                        )
                else:
                    actor_grad_norm = self._zero_metric
                with _cuda_nvtx_range("actor/optimizer_step", self.nvtx_profile_ranges):
                    self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                with _cuda_nvtx_range("actor/backward", self.nvtx_profile_ranges):
                    actor_loss.backward(inputs=list(self.actor.parameters()))
                self._sync_gradients(self.actor.parameters())
                if self.max_grad_norm > 0:
                    with _cuda_nvtx_range("actor/grad_clip", self.nvtx_profile_ranges):
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), max_norm=self.max_grad_norm
                        )
                else:
                    actor_grad_norm = self._zero_metric
                with _cuda_nvtx_range("actor/optimizer_step", self.nvtx_profile_ranges):
                    self._arm_optimizer_finite_gate(self.actor_optimizer, actor_loss)
                    self.actor_optimizer.step()
        else:
            actor_grad_norm = self._zero_metric

        actor_metric_tensors = (
            actor_loss,
            actor_grad_norm,
            policy_entropy,
        )
        if not read_metrics:
            # Inductor CUDA Graph Trees overwrite their output storage on a
            # later compiled call.  Stage the three scalars now so they remain
            # valid until the single cycle-end D2H read.
            self._pending_actor_metric_values = torch.stack(
                [tensor.detach().reshape(()) for tensor in actor_metric_tensors]
            )
            return {}
        return self._read_metric_tensors(
            (
                "Loss/actor",
                "Train/actor_gradient_norm",
                "Loss/entropy",
            ),
            actor_metric_tensors,
        )

    def soft_update_target(self) -> None:
        """Polyak-average update of the target Q-network."""
        with _cuda_nvtx_range("target/soft_update_loop", self.nvtx_profile_ranges):
            if self._device_type == "cuda":
                with torch.no_grad():
                    tau = self._target_tau * self._q_update_finite.to(self._target_tau.dtype)
                    for target, source in zip(
                        self.qnet_target.parameters(),
                        self.qnet.parameters(),
                        strict=True,
                    ):
                        target.lerp_(source, tau)
            else:
                polyak_update_target(self.qnet_target, self.qnet, self.tau)

    def _reset_update_cycle_graph(self) -> None:
        """Invalidate a raw cycle graph after checkpoint state storage changes."""
        self._update_cycle_graph = None
        self._update_cycle_graph_cache_key = None
        self._update_cycle_static_batch = None
        self._update_cycle_graph_metric_values = None
        self._pending_actor_metric_values = None
        self._pending_cycle_critic_metric_values = None
        self._pending_cycle_metric_values = None

    def dp_initial_sync_tensors(self) -> Dict[str, torch.Tensor]:
        """Model state broadcast once from rank 0 before collection starts.

        The values alias the parameter/buffer storage of ``actor``, ``qnet``
        and ``qnet_target`` (plus the ``log_alpha`` leaf) rather than copies,
        so startup broadcast updates the model in place. Optimizer state starts
        empty and remains aligned because every actual optimizer update uses
        the same cross-rank mean gradient.
        """
        tensors: Dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("actor", self.actor),
            ("qnet", self.qnet),
            ("qnet_target", self.qnet_target),
        ):
            for key, value in module.state_dict().items():
                tensors[f"{prefix}.{key}"] = value
        tensors["log_alpha"] = self.log_alpha
        return tensors

    def get_state_dict(self) -> Dict[str, Any]:
        """Save all components."""
        return {
            "actor": self.actor.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict")
                else None
            ),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load all components."""
        self.actor.load_state_dict(state_dict["actor"])
        self.qnet.load_state_dict(state_dict["qnet"])
        self.qnet_target.load_state_dict(state_dict["qnet_target"])
        self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        if state_dict.get("obs_normalizer") and hasattr(self.obs_normalizer, "load_state_dict"):
            self.obs_normalizer.load_state_dict(state_dict["obs_normalizer"])
        self.update_count = state_dict.get("update_count", 0)
        self._reset_update_cycle_graph()


# ---------------------------------------------------------------------------
