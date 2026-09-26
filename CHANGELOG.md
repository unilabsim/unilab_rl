# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `log_interval` backend-logging throttle on `BaseTrainingLogger`,
  `OffPolicyLogger`, `OnPolicyLogger`, `OffPolicyRunner`, and `APPORunner`
  (all default 1). The off-policy builders (`fast_sac`, `flash_sac`,
  `warp_sac`) read it from `cfg.training.log_interval`. Terminal rendering is
  unaffected; only TensorBoard/wandb writes are gated, and the final iteration
  is always logged.

### Fixed

- TensorBoard scalar writes are now batched into a single event record per
  training step. Previously each `add_scalar` call produced one record, and
  the writer thread's per-record open/write/close saturated the async queue
  (depth 10), blocking the learner main thread for ~160 ms per iteration when
  the log directory lives on a network filesystem (FUSE). Falls back to
  per-scalar writes when the batched path is unavailable.

## [1.4.0] - 2026-09-25

### Added

- New WarpSAC algorithm package. `WarpSACLearner` inherits UniLab's FlashSAC
  learner without modifying it, while `WarpSACReplayPipeline` adds the official
  implementation's bucketed linear age-bias replay sampling to the asynchronous
  device-authoritative runtime.
- WarpSAC double-buffer builder with `decay_step`, `replay_min_weight`,
  `replay_num_buckets`, `target_frequency`, and actor/critic
  parameter-normalization switches.
- Generic off-policy replay-pipeline injection so algorithm owners can provide
  specialized device-resident samplers without changing FlashSAC.

### Removed

- Removed the unused manual whole-update CUDA Graph learner path and its four
  public options: `use_cuda_graph_critic`, `use_cuda_graph_actor`,
  `use_cuda_graph_critic_packed_staging`, and
  `use_cuda_graph_actor_packed_staging`. CUDA learners retain the faster
  default `torch.compile` path with Inductor CUDA Graph Trees.
- Removed manual graph-only replay packing and NCCL gradient-capture plumbing.
  GPU-resident packed replay and ordinary flat-gradient DP averaging remain
  the single runtime paths.

### Fixed

- FastSAC's compiled C51 projection no longer caches an Inductor CUDA Graph
  Trees output tensor in Python. Recreating the row-offset tensor inside the
  traced expression avoids stale output storage across compiled replays; the
  recreated offsets retain the original `num_atoms` row stride and therefore
  preserve one normalized distribution per replay row.

## [1.3.4] - 2026-09-24

### Added

- New FlashSAC `compile_full_objectives` option (default `false`): extends
  `torch.compile` from loss-only helpers to the complete critic and actor
  objectives, forwarded through the FlashSAC double-buffer builder.

### Changed

- FlashSAC categorical TD projection is now CUDA Graph capture-safe: support
  bounds and bin-width arithmetic stay on device instead of syncing through
  host scalars.
- FlashSAC learner cycles reduce host synchronization by deferring metric
  D2H reads to the end of the cycle, gating finite-value checks on the
  device-side optimizer path, and freezing critic parameters during actor
  updates while preserving the required `dQ/da` gradient.

### Fixed

- FlashSAC manual CUDA Graph lifecycle: the first captured update is now
  replayed instead of dropped, critic target-network updates are captured
  inside the critic graph, and persistent metric buffers prevent output
  overwrite across replays.

## [1.3.3] - 2026-09-24

### Changed

- FastSAC's `torch.compile` path now enables Inductor CUDA Graph replay for
  fused critic/actor loss kernels and defers scalar metric reads to the final
  update in each learner cycle.
- FastSAC actor updates no longer accumulate unused critic-parameter gradients.
  The policy still receives the same `dQ/da` gradient.
- Compiled FastSAC updates replace per-loss host finite-check synchronization
  with fused-optimizer device gating, preserving non-finite step suppression
  without fragmenting the learner window.
- FastSAC critic CUDA Graph replay now captures the Polyak target-network
  update, removing the graph-external foreach launches between critic replays.
- The off-policy runtime manifest now reports the effective CUDA Graph replay,
  packed-staging, target-update capture, and eager-fallback state.

### Fixed

- Removed a redundant CUDA stream synchronization between learner-owned actor
  inference and its blocking D2H action copy. CUDA event timing preserves the
  forward-duration metric without adding another graph-boundary sync.
- FastSAC CUDA Graph calls now fail closed to eager updates when observation
  normalization is active, matching the existing FlashSAC safety behavior.

## [1.3.2] - 2026-09-22

### Removed

- The TD3 (FastTD3) algorithm: the whole `uni_rl.algos.fast_td3` package
  (`TD3Actor`, `FastTD3Learner`, `build_td3_double_buffer_runner`), the
  built-in `td3` actor branch in `uni_rl.algos.common.actor_factory`, the
  `"td3"` entries in the off-policy worker exploration routing and the
  double-buffer runner display names, and the TD3-only
  `Critic` / `DistributionalQNetwork` networks in
  `uni_rl.algos.common` (FastSAC and FlashSAC each define their own critic
  networks). UniLab has dropped its TD3 task configs and dispatch branches
  accordingly.

## [1.3.1] - 2026-09-22

### Removed

- The RSL-RL wrapper layer (`uni_rl.algos.rsl_rl`,
  `uni_rl.algos.rsl_rl_ppo`, `uni_rl.algos.rsl_rl_runtime`,
  `uni_rl.algos.rsl_rl_training_state`): `FinalObservationAwarePPO`,
  `resolve_rsl_rl_ppo_runtime` / `RslRlPPORuntime`,
  `TrainingStateOnPolicyRunner`, `RslRlVecEnvWrapper`,
  `get_policy_obs_dims`, and the PPO script-assembly helpers
  (`apply_rsl_rl_rank_seed`, `resolve_rsl_rl_device`,
  `ppo_samples_per_iteration`, `finish_rsl_rl_distributed`,
  `rsl_rl_single_process_topology`, `normalize_ppo_train_cfg`). UniLab's PPO
  path now drives upstream rsl_rl directly and owns the VecEnv adapter
  (`unilab.rl`), so nothing here has a consumer left. APPO keeps using
  rsl_rl's model classes (`MLPModel`, `GaussianDistribution`); only the
  wrapper/runtime layer is gone. `uni_rl.training_state.TrainingStateProvider`
  remains as the owner progress-checkpoint protocol.

## [1.3.0] - 2026-09-17

### Changed

- The FastSAC, FlashSAC, and FastTD3 double-buffer builders now require and
  directly read `training.inference_request_timeout_sec`.

### Fixed

- Added an off-policy collector-ready handshake after environment
  initialization and runtime-manifest publication. The learner now starts its
  inference-tick timeout only after collector readiness, so backend-owned cold
  starts (such as Genesis JIT and first reset) cannot consume the steady-state
  tick budget. FlashSAC and FastTD3 builders also forward
  `training.inference_request_timeout_sec`, matching FastSAC.

## [1.2.1] - 2026-09-15

### Removed

- `FastSACRunner` and `FlashSACRunner` kwargs-style runner classes. They were
  stale duplicates of the `build_*_double_buffer_runner` builder functions
  (lacking `dp_sync`, `nan_guard_cfg`, `collector_cpu_ids`,
  `actor_adapter_modules`, and `inference_request_timeout_sec` support) with no
  consumers in uni_rl or UniLab. Use the builder functions instead; the
  `FlashSACRunner` re-export in `uni_rl.algos.flash_sac` is gone with them.
- The unused submit/ready half of the replay transfer backend contract:
  `ReplayTransferBackend.submit_h2d` / `ready_query` /
  `wait_current_stream_for_ready` / `synchronize_ready` / `clear_ready` /
  `supports_async_submit`, the corresponding `CudaLikeReplayTransferBackend`
  and `TorchCopyReplayTransferBackend` implementations, and
  `native_h2d.submit_h2d`. `GPUResidentReplayPipeline` performs the H2D copy
  inline; existing custom backends with extra methods remain compatible.
  `native_h2d.is_available` / `get_diagnostic` are kept.
- Dead public helpers with zero consumers in uni_rl and UniLab:
  `uni_rl.algos.common.safe_tensor`, `EmpiricalNormalization.inverse`,
  `TraceRecorder.span`, `OffPolicyLogger.update_replay_queue`,
  `uni_rl.utils.device.get_device_info_line`,
  `uni_rl.utils.seed.apply_configured_training_seed`,
  `TrainingSeedInfo.to_dict`, and
  `uni_rl.utils.observations.get_critic_base_dim` (equivalent to
  `get_obs_dims(spec)[1]`).
- Internal dead code: `APPOLearner.train_mode`, write-only attributes
  (`SharedWeightSync._param_shapes`, `APPOLearner.last_update_metrics`,
  `SACActor.device_`), and the `inference_wait_ms` metric key compatibility
  branch (producers have emitted `learner_action_wait_ms` exclusively).

### Changed

- Shared learner boilerplate (AMP dtype resolution, grad-scaler/autocast,
  gradient sync, obs-normalizer update, Polyak target update, CUDA-graph
  release/compile helpers) is consolidated into
  `uni_rl.algos.common.learner_boilerplate`; `fast_sac` and `flash_sac`
  learners no longer carry 24 byte-identical method copies. Behavior is
  bit-identical (verified by A/B comparison).
- Collector metrics draining is shared between `APPORunner` and
  `OffPolicyRunner` via `uni_rl.logging.metrics_drain.drain_collector_metrics`,
  replacing two acknowledged copies of the dispatch logic.
- `flash_sac`'s inlined categorical TD projection now calls
  `update.compute_categorical_td_target`, removing the duplicated projection
  math (verified bit-identical).

### Fixed

- Removed stale `dist/` build artifacts (1.0.0/1.1.0) that broke the
  `make smoke` wheel glob, and the leftover `uni_rl.algos.hora` `__pycache__`.
- `README_zh.md` now includes the "PPO curriculum checkpoint state" section,
  in sync with the English README.

## [1.2.0] - 2026-09-10

### Changed

- No code changes since 1.1.3. This minor bump re-anchors the public-contract
  changes shipped in 1.1.3 (the new off-policy actor adapter API and the
  removal of the `uni_rl.algos.hora` namespace) under a minor version, per the
  semver discipline that public-contract changes require at least a minor
  bump. Consumers pinning `~=1.1` should review the 1.1.3 changelog entries
  before upgrading.

## [1.1.3] - 2026-09-09

### Added

- Generic off-policy actor adapter registry
  (`uni_rl.offpolicy.actor_adapter.OffPolicyActorAdapter`,
  `register_offpolicy_actor_adapter`, `get_offpolicy_actor_adapter`) so external
  packages can plug custom actor construction, exploration sampling, privileged
  context extraction, and inference-context slicing into the generic off-policy
  runtime. Spawn-safety is provided by the new optional
  `algo.actor_adapter_modules` config key (also on `OffPolicyRuntime`), whose
  dotted modules are imported in both the learner process and the spawn
  collector subprocess.

### Removed

- The HORA implementation (`uni_rl.algos.hora`) and its hardcoded `hora_sac`
  branches in the generic off-policy runtime moved to the standalone
  `sharpa_rl_unilab` repository. The old import namespace is removed without a
  forwarding shim; consumers register an `OffPolicyActorAdapter` instead.

## [1.1.1] - 2026-09-08

### Added

- Optional PPO runtime runner selection and `TrainingStateOnPolicyRunner`, with
  an explicit versioned checkpoint envelope for downstream-owned curriculum
  progress. Existing PPO runners are unchanged; requested state restoration
  rejects missing or incompatible envelopes rather than restarting a curriculum
  ([#16](https://github.com/unilabsim/unilab_rl/issues/16)).

### Removed

- The HIM-PPO implementation and tests moved to
  [legged-manipulation_unilab](https://github.com/unilabsim/legged-manipulation_unilab)
  under [UniLab #1528](https://github.com/unilabsim/UniLab/issues/1528).
  The old import namespace is removed without a forwarding shim.

## [1.1.0] - 2026-09-06

### Fixed

- `uni_rl.utils.device.resolve_backend_process_device` now treats `newton`
  like `mjwarp`: both backends require an explicit CUDA process device shared
  with the learner, so off-policy collectors invoke the injected
  `backend_device_binder` for `newton` runs instead of silently skipping the
  binding (previously the spawned collector built the backend without a bound
  device).
- `DpRankSupervisor` now re-runs the downstream owner's original
  `sys.argv[0]` entry script for spawned off-policy ranks instead of redirecting
  it to the nonexistent `uni_rl/scripts/` directory, restoring multi-GPU
  SAC/TD3 launches from installed consumers such as UniLab (#12).

## [1.0.0] - 2026-09-04

First stable release. The public contract (`uni_rl.env_contract` protocols and
factory signature, runner / `runtime_resolver` conventions, algorithm config
keys) is now covered by semantic versioning.

### Added

- `README_zh.md`（简体中文 README）and a Citation section (UniLab paper,
  `jia2026unilab`) in both READMEs.

### Changed

- Rewrote the README: documents the relationship with UniLab, PyPI
  installation, env-contract usage, and development commands. PyPI is the
  release channel; TestPyPI instructions were removed.

## [0.3.0] - 2026-09-04

### Added

- Optional env algo-capabilities extension point in `uni_rl.env_contract`:
  `EnvAlgoCapabilitiesProtocol` (per-dimension `action_low` / `action_high`
  bounds and `joint_names`, all fields optional), the
  `SupportsAlgoCapabilitiesProtocol` provider protocol, the frozen
  `EnvAlgoCapabilities` default carrier, and the `get_algo_capabilities(env)`
  helper that falls back to an all-`None` default for envs that do not provide
  capabilities. Intended for algorithm-side features such as per-joint action
  scaling and symmetry augmentation; cold-path reads only (runner init, dim
  probe). (UniLab issue #1487)

## [0.2.0] - 2026-09-04

### Changed

- Grouped algorithm packages under `uni_rl.algos` (`appo`, `fast_sac`,
  `fast_td3`, `flash_sac`, `him_ppo`, `hora`, `rsl_rl` wrappers, `common`).
- Added CI (ruff / mypy / pyright / pytest+coverage) and release workflows,
  plus `AGENTS.md` contributor guidance.

## [0.1.0] - 2026-09-04

### Added

- Migrated the RL algorithm and async runtime layer from UniLab into the
  standalone `uni_rl` package: PPO/APPO/SAC/TD3/FlashSAC/HIM-PPO/HORA runners,
  learners, collectors, IPC, and training logging.
- Decoupled `uni_rl` from `unilab` via the injected env contract
  (`uni_rl.env_contract.EnvFactory` / `EnvProtocol`) and dependency injection;
  `uni_rl` never imports `unilab` / `unisim`.
- Forwarded `backend_device_binder` through runner builders.
