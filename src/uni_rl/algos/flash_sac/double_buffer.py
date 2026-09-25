"""FlashSAC builder for the device-authoritative replay path."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf

from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.env_contract import EnvFactory
from uni_rl.ipc.replay_pipelines.gpu_resident import require_offpolicy_replay_device
from uni_rl.offpolicy.actor_adapter import import_actor_adapter_modules
from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from uni_rl.offpolicy.runtime import resolve_actor_adapter_modules, resolve_custom_offpolicy_runtime
from uni_rl.utils.device import get_default_device
from uni_rl.utils.nan_guard import NanGuardCfg
from uni_rl.utils.observations import get_obs_dims
from uni_rl.utils.seed import apply_training_seed

if TYPE_CHECKING:
    from uni_rl.ipc.dp_sync import DpParameterSync


def _validate_flashsac_double_buffer_runtime(
    cfg: DictConfig,
    *,
    replay_prefetch_mode: str,
) -> None:
    if replay_prefetch_mode != "one_tick":
        raise ValueError("FlashSAC device replay requires replay_prefetch_mode='one_tick'")
    if cfg.algo.algo_params.n_step != 1:
        raise ValueError("FlashSAC-B initially supports n_step=1 only")


def build_flashsac_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_factory: EnvFactory,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str | None = None,
    nan_guard_cfg: NanGuardCfg | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
    backend_device_binder: Callable[[str], str | None] | None = None,
) -> Any:
    """Build FlashSAC with the bounded-ingress device replay pipeline."""
    device = require_offpolicy_replay_device(device or get_default_device())
    apply_training_seed(cfg.algo.seed, torch_runtime=True, cuda=True)
    _validate_flashsac_double_buffer_runtime(
        cfg,
        replay_prefetch_mode=replay_prefetch_mode,
    )
    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    custom_runtime = resolve_custom_offpolicy_runtime(rl_cfg)
    # Import registration modules before the learner builds its actor so
    # custom adapter registrations are live in this process; the runner
    # forwards the list to the spawn collector.
    actor_adapter_modules = resolve_actor_adapter_modules(rl_cfg, custom_runtime)
    import_actor_adapter_modules(actor_adapter_modules)

    env = env_factory(1, env_cfg_override)
    try:
        obs_dim, critic_obs_dim = get_obs_dims(dict(env.obs_groups_spec))
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = int(action_shape[0])
    finally:
        env.close()

    learner_cls: type[Any] = FlashSACLearner
    algo_type = "flashsac"
    learner_extra_kwargs: dict[str, Any] = {}
    if custom_runtime is not None:
        learner_extra_kwargs = cast(
            dict[str, Any],
            custom_runtime.build_model_kwargs(
                obs_dim=int(obs_dim),
                critic_obs_dim=int(critic_obs_dim),
            ),
        )
        if custom_runtime.learner_cls is not None:
            learner_cls = custom_runtime.learner_cls
        if custom_runtime.algo_type is not None:
            algo_type = str(custom_runtime.algo_type)

    learner_kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "critic_obs_dim": critic_obs_dim,
        "gamma": cfg.algo.gamma,
        "tau": cfg.algo.tau,
        "actor_lr": cfg.algo.actor_lr,
        "critic_lr": cfg.algo.critic_lr,
        "actor_hidden_dim": cfg.algo.actor_hidden_dim,
        "critic_hidden_dim": cfg.algo.critic_hidden_dim,
        "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
        "critic_num_blocks": cfg.algo.algo_params.critic_num_blocks,
        "num_atoms": cfg.algo.num_atoms,
        "critic_min_v": cfg.algo.algo_params.critic_min_v,
        "critic_max_v": cfg.algo.algo_params.critic_max_v,
        "temp_initial_value": cfg.algo.algo_params.temp_initial_value,
        "temp_target_sigma": cfg.algo.algo_params.temp_target_sigma,
        "temp_target_entropy": cfg.algo.algo_params.temp_target_entropy,
        "actor_bc_alpha": cfg.algo.algo_params.actor_bc_alpha,
        "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
        "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
        "learning_rate_init": cfg.algo.algo_params.learning_rate_init,
        "learning_rate_peak": cfg.algo.algo_params.learning_rate_peak,
        "learning_rate_end": cfg.algo.algo_params.learning_rate_end,
        "learning_rate_warmup_steps": cfg.algo.algo_params.learning_rate_warmup_steps,
        "learning_rate_decay_steps": cfg.algo.algo_params.learning_rate_decay_steps,
        "normalize_reward": cfg.algo.algo_params.normalize_reward,
        "normalized_g_max": cfg.algo.algo_params.normalized_g_max,
        "n_step": cfg.algo.algo_params.n_step,
        "obs_normalization": cfg.algo.obs_normalization,
        "use_amp": cfg.training.use_amp,
        "amp_dtype": cfg.algo.algo_params.amp_dtype,
        "use_compile": cfg.algo.algo_params.use_compile,
        "compile_full_objectives": bool(
            getattr(cfg.algo.algo_params, "compile_full_objectives", False)
        ),
        "use_cuda_graph_critic": cfg.algo.algo_params.use_cuda_graph_critic,
        "use_cuda_graph_actor": cfg.algo.algo_params.use_cuda_graph_actor,
        "use_cuda_graph_critic_packed_staging": (
            cfg.algo.algo_params.use_cuda_graph_critic_packed_staging
        ),
        "use_cuda_graph_actor_packed_staging": (
            cfg.algo.algo_params.use_cuda_graph_actor_packed_staging
        ),
    }
    learner_kwargs.update(learner_extra_kwargs)
    learner = learner_cls(device=device, **learner_kwargs)

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type=algo_type,
        env_factory=env_factory,
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=cfg.algo.batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=cfg.algo.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        nan_guard_cfg=nan_guard_cfg,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
        backend_device_binder=backend_device_binder,
        inference_request_timeout_sec=cfg.training.inference_request_timeout_sec,
        actor_adapter_modules=actor_adapter_modules,
    )
