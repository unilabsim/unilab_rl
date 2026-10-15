# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.15] - 2026-10-16

### Added

- `env_contract.SupportsCompletionReceiptsProtocol` and
  `get_produces_completion_receipts()` (Motphys/UniLab#2363): envs may declare
  `produces_completion_receipts: bool`. Envs that declare `False` (synchronous
  producers that never emit step receipts, e.g. host-bridge backends) are
  accepted at CUDA replay ingress without a receipt; undeclared envs default to
  `True` (receipt required — fail-closed direction unchanged).

### Changed

- The mujoco + CUDA-learner off-policy topology now resolves to the CUDA
  tensor ring by default (UniLab #2365). Driven by UniLab's
  `training.collector_tensor_device=auto` (collector env tensors follow the
  learner device), the probe reports CUDA-native observations and, with no
  explicit transport config, placement resolution selects the CUDA tensor ring
  (`cuda_no_host_boundary` staging, replay ingress on the learner device, no
  host action staging). On CUDA platforms the CPU transport is no longer
  reachable for tensor-native envs; the CPU ring path remains for CPU-only and
  Apple Silicon `torch.mps` (Metal Performance Shaders) learners.

### Removed

- Shrunk the replay completion surface to the minimal async step-only event
  contract (Motphys/UniLab#2363). `COMPLETION_OPERATIONS` is now step-only;
  `selected_reset`/`mask_reset` (and any other removed operation) receipts are
  rejected fail-closed. The `stream_id` receipt field and the
  `CompletionTokenProtocol.ipc_handle` transport hook are deleted, and the
  generic backend-token IPC reconstruction (`import_shared_completion`,
  `EventCompletion.ipc_handle`) is removed. The shared-memory completion
  metadata now carries only the replay-owned ingress publication event handle
  plus `contract_version`/`sequence`/`producer_id` for validation; the learner
  imports and waits that replay-owned event (`ReplayBuffer.import_ingress_publication`,
  replacing `import_shared_completion`) before its H2D span copies. Read-before-copy
  gating, the replay-owned publication event recorded after ingress writes,
  stale-sequence rejection, producer/operation binding, and fail-closed CUDA
  ingress without a receipt are unchanged. `COMPLETION_CONTRACT_VERSION` is
  bumped to 3 to track the unisim completion-token contract v3 (token fields are
  exactly `contract_version`/`operation`/`sequence`/`producer_id`, step-only);
  v2 receipts fail closed.

- Removed the NumPy observation runtime from the off-policy collector with no
  compatibility layer (#119). Collector initialization now fails closed with a
  `TypeError` unless the actor (`obs`) and critic observation groups are Torch
  tensors; the `tensor_collector` branch, the NumPy
  observation/critic/reward/done/truncation transition carriers, the
  `torch.from_numpy` replay bridge, `_collector_action_numpy`,
  `resolve_offpolicy_actor_priv_info`, and the NumPy terminal-observation
  contract helper (`uni_rl.utils.final_observation`) are removed. The
  collector runtime manifest no longer reports a `tensor_native_env` mode.
  Privileged actor adapters (`resolve_priv_info`) fail closed at collector
  startup.

- Removed the NumPy seam from the off-policy inference hot path (UniLab #2365).
  `SharedInferenceRing` action egress is a cloned Torch tensor on every
  platform (the CPU `actions.numpy().copy()` path is gone), so the
  collector↔learner boundary is Torch-tensor-only and the collector-side
  NumPy→tensor restage before `TorchEnv.step()` is removed.
  `SharedInferenceRing.publish_observation` accepts only Torch tensors and
  fails closed with a `TypeError` on NumPy input. The dead single-slot
  `SharedInferenceSlot` exchange and its tests are deleted.

- Removed the collector graph protocol, configuration, runner state, public
  lifecycle manifest, and whole-collector completion operation with no
  compatibility path (Motphys/UniLab#2358 and Motphys/UniLab#2359). Replay
  ingress now uses the neutral v2 completion contract:
  `CompletionTokenProtocol`, `wait_completion`, `import_shared_completion`,
  top-level env `completion` receipts, `producer_id`, and
  publication-sequence metadata. Backend physics graphs and FlashSAC learner
  CUDA graphs are unchanged. Runtime manifests use schema version 2 so v1
  collector-graph manifests fail closed.

### Fixed

- The CPU-ring inference staging policy no longer labels non-CUDA accelerator
  learners (e.g. Apple Metal Performance Shaders, `torch.mps`) as
  `cpu_no_device_transfer`; they now report
  `cpu_ring_explicit_learner_device_copy` because the ring `copy_` boundary
  still crosses devices (UniLab #2365).
- Replay ingress now validates and waits for the producer completion before
  its first transition-source read or copy, instead of enqueueing the wait
  after the ingress copies. Unexpected CUDA IPC handle export failures also
  fail closed rather than silently marking receipt metadata unavailable.
- Replay publication now accepts any validated structural completion-v2
  producer receipt, rather than requiring a `uni_rl` concrete token class, and
  localizes the waited metadata at the ingress publication boundary.
- Off-policy collectors previously preserved the pre-action observation and
  critic sources in persistent staging carriers so autoreset could not
  overwrite `obs[t]` before replay published transition `t`. After the
  env-side observation-ownership fix (UniLab #2362), public observation
  tensors are caller-owned, so this collector-side staging workaround has
  been removed again; replay now receives the env-provided tensors directly.

## [1.4.14] - 2026-10-14

### Added

- Added neutral collector graph-mode negotiation and public whole-collector-tick
  completion evidence for the integration-only K=1 collector graph
  (Motphys/UniLab#2311): replay ingress validates operation, capture identity,
  monotonic sequence, and collector invalidation manifests; spawned CUDA
  collectors transport fixed-width terminal and ingress receipt state; and
  runtime manifests record the negotiated backend lifecycle and replay/fallback
  evidence fail-closed.
- Restored collector timing component means and fixed CUDA spawn collector
  metric publication (Motphys/UniLab#2311).

### Fixed

- CUDA replay ingress submission is now single-owner. The replay sync daemon is
  the sole consumer of published ingress tokens, learner progress no longer
  races it, and H2D copies wait on the collector receipt using the matching
  `collector_tick` operation without producer-stream synchronization
  (Motphys/UniLab#2312).

## [1.4.13] - 2026-10-13

### Added

- Replay ingress consumes a neutral public graph-completion receipt contract
  without importing UniLab or UniSim (Motphys/UniLab#2309). CUDA ingress
  publication is gated on operation, capture identity, and monotonic sequence;
  stale, malformed, or missing CUDA receipts fail closed.
- Collector-side CUDA ingress publication records an interprocess event after
  slot writes without synchronizing the producer stream. Learner-side device
  copies wait on that receipt on their own stream, and spawned collectors can
  export/import opaque IPC handles for consumer-stream ordering.
- Runtime manifests publish and validate graph lifecycle evidence fail-closed.

## [1.4.12] - 2026-10-12

### Added

- Off-policy collectors now publish backend tensor-runtime diagnostics into
  the runtime manifest as `backend_diagnostics` (unilabsim/UniLab#2278). Envs
  expose them through the new `uni_rl.env_contract.get_backend_runtime_diagnostics`
  extension point (cold path only); the payload records `available`,
  `backend_type`, and per-optimization `requested` / `enabled` /
  `disable_reason` entries (e.g. MJWarp `cuda_graph`). Envs without the
  provider record an explicit unavailable marker, and the runtime manifest
  schema validates the section fail-closed.

## [1.4.11] - 2026-10-09

### Added

- FlashSAC now exposes `use_whole_cycle_cuda_graph` independently from
  `use_compile` and `compile_full_objectives`.
- SAC learners support accelerated MPS update paths using either explicitly
  validated BF16 AMP or device-supported `torch.compile` (#98).

### Changed

- Clarified the `Perf/collector_env_step_ms` and `Perf/collection_time`
  metric descriptions (unilabsim/UniLab#2102): the former is the EMA of one
  vectorized environment step, the latter one complete rollout
  (`steps_per_env` steps plus bookkeeping), so the two must not be compared
  directly.

### Fixed

- Orphaned collector processes after an abnormal learner exit
  (SIGKILL, OOM killer, test-timeout kill). `daemon=True` only cleans up
  when the parent exits through the Python interpreter, so spawn collectors
  were reparented to PID 1 and kept running the env hot loop on a
  `stop_event` nobody would ever set. Every collector entry point now
  installs a parent-death watchdog (`uni_rl.ipc.parent_watchdog`): a daemon
  thread compares `os.getppid()` against the learner PID handed over at
  spawn (handed, not read back — a learner killed during the child's slow
  spawn boot leaves it reparented before its entry point runs). On parent
  death the watchdog sets `stop_event` for a graceful drain and, after a
  bounded grace period, forces `os._exit` so a wedged interpreter
  finalization (native thread join) cannot keep the process alive either.
  The mechanism is POSIX-portable and covers both macOS and Linux without
  relying on `daemon=True` or `prctl(PR_SET_PDEATHSIG)`. Entry points
  running inline in the spawner's own process (test doubles) install no
  watchdog.

### Fixed

- FlashSAC honors an explicit `use_compile=false` on NVIDIA CUDA instead of
  silently forcing Inductor compilation and whole-cycle CUDA Graph replay
  (#94).
- Off-policy terminal logs retain the last terminal reward data and tag the
  `Rewards` header with its source iteration (#99).

## [1.4.10] - 2026-10-09

### Fixed

- APPO rollout shared-memory tensor views are bounded to each field's logical
  payload. Allocation padding exposed by platform page granularity no longer
  breaks tensor reshaping during ring creation or attachment (#91).

## [1.4.9] - 2026-10-08

### Fixed

- Off-policy terminal logging now separates replay-ingress diagnostics from
  losses and policy metrics with stable short labels. Terminal rendering is
  width-aware, and immaterial collector timing diagnostics are hidden until
  they reach 1% of the collector cycle (#86).

## [1.4.8] - 2026-10-08

### Fixed

- APPO timeout bootstrap correction now supports a CPU-authoritative
  environment with a CUDA collector device. The terminal-row selector is moved
  to the final-observation device before indexing, restoring the correction to
  the reward carrier's device (#84 follow-up).

## [1.4.7] - 2026-10-08

### Added

- Explicit `resume_checkpoint` support on `DoubleBufferOffPolicyRunner.learn`,
  restoring the complete learner state and resuming from the checkpointed
  update count. Invalid or missing checkpoint progress metadata fails closed.
- Reviewed tensor-runtime defaults and deterministic bounds for inference-ring
  capacity, collector metric intervals, replay ingress depth, replay ingress
  slot rows, and learner sampling. SAC and FlashSAC resolve one validated
  settings object before environment probing, and runtime manifests
  record configured, default, effective, and maximum values.
- Configurable bounded replay-ingress slots that may contain fewer rows than a
  collector vector. The collector publishes vectors in bounded chunks with
  contiguous commit order, per-chunk terminal observation patching, backpressure,
  and explicit partial-prefix semantics during shutdown.
- Conservative combined CUDA pre-flight accounting for inference IPC, replay
  storage, learner double-buffer batches, replay ingress, and allocator
  workspace reserve. Impossible combinations fail before collector or replay
  resource allocation.
- Bounded inference scheduling diagnostics for tensor-native collectors, including
  queue depth, action backlog, in-flight depth, publication lag, wait time, and
  the legal sequential dependency graph in runtime manifests.
- Conservative pre-flight CUDA inference accounting for the shared inference
  ring, IPC/timing events, learner inference scratch, FlashSAC persistent
  exploration scratch, and allocator/workspace reserve. CUDA learners now
  preallocate timing events before collector startup.
- A deterministic cross-process CUDA regression covering delayed action clones
  with bounded slot reuse, plus deterministic cleanup for partially constructed
  inference rings.

### Changed

- APPO collection is tensor-native (#84). The environment contract now requires
  contiguous float32 Torch actions, Torch observations/rewards/flags, and
  one-dimensional Torch reset indices. APPO rollout IPC and learner staging use
  tensor-owned shared storage with explicit device copies; NumPy is no longer
  a trainer boundary. APPO also accepts the off-policy tensor NaN-guard factory.
- Gaussian APPO target/current standard deviations are derived directly from
  distribution parameters (`std_param` or `log_std_param`), eliminating
  dependence on `MLPModel.output_std` sample-time cache state.
- Made `SAC` the sole public identity for the high-performance SAC
  implementation, with no compatibility layer: the package is
  `uni_rl.algos.sac`, the learner class is `SACLearner`, and display/log names
  use `SAC`. Existing consumers must update imports and runtime references.
- Removed the separate age-biased SAC algorithm identity and package.
  FlashSAC now owns its substantive behavior as options: age-biased device
  replay (`decay_step`, `replay_min_weight`, `replay_num_buckets`), actor/critic
  parameter-normalization switches, target-update frequency, and
  actor-before-critic ordering. Defaults preserve FlashSAC's uniform replay,
  normalized parameters, target frequency 1, and critic-before-actor ordering.
  No old compatibility layer is retained.

### Fixed

- Collector inference metrics distinguish configured ring capacity from dynamic
  observation backlog and action backlog, preventing capacity from being
  interpreted as concurrent in-flight requests.

## [1.4.4] - 2026-09-29

### Fixed

- FlashSAC whole-cycle CUDA graph capture no longer poisons the
  process on failure. The torch.compile capture-error mode is thread-local
  instead of process-global, and a failed capture (e.g.
  `cudaErrorStreamCaptureUnsupported` at larger observation dims) falls back
  to eager execution for the affected region while compiled paths stay
  enabled elsewhere (#53, #56).

## [1.4.3] - 2026-09-27

### Fixed

- Normal off-policy training completion no longer crashes the lock-step
  collector with "Learner stopped before inference tick". The runner now
  requests collector stop before publishing the STOPPED phase, and the
  collector treats a STOPPED phase accompanied by a stop request as a graceful
  release.

## [1.4.2] - 2026-09-27

### Added

- A learner-owned off-policy preparation phase between DP initialization and
  collector startup. SAC and FlashSAC warm representative actor
  inference, replay gather/device paths, and compiled update/CUDA-graph paths
  before tick 0.
- Custom learner/runtime/actor warmup hooks: `prepare_for_collection`,
  `OffPolicyRuntime.learner_prepare_hook`, and
  `OffPolicyActorAdapter.warmup_actions`.
- Cross-process learner phase and progress coordination for lock-step
  collectors.

### Changed

- Off-policy coordination distinguishes busy learners, stopped learners, dead
  learner PIDs, and stalled waiting loops instead of imposing a fixed response
  deadline.
- `training.inference_request_timeout_sec` is deprecated and ignored by the
  builders; direct runner construction retains the argument only for source
  compatibility.

### Fixed

- Learner-owned cold work completes before collector startup, removing
  first-update compilation and graph-capture pressure from the action wait.
- Learner exceptions and normal completion stop/join collectors, preventing
  orphaned lock-step collectors.
- Compatibility-device warmup restores model, optimizer, scheduler, normalizer,
  counter, finite-gate, deferred-metric, gradient, and RNG state. Compiler and
  graph caches remain intentionally warm.

## [1.4.1] - 2026-09-27

### Added

- `log_interval` backend-logging throttle on `BaseTrainingLogger`,
  `OffPolicyLogger`, `OnPolicyLogger`, `OffPolicyRunner`, and `APPORunner`
  (all default 1). The off-policy builders (`sac`, `flash_sac`) read it
  from `cfg.training.log_interval`. Terminal rendering is
  unaffected; only TensorBoard/wandb writes are gated, and the final iteration
  is always logged.

### Changed

- SAC and FlashSAC now use one owner-managed
  whole-update-cycle CUDA Graph on NVIDIA CUDA. Both compile that graph
  with architecture-portable Inductor max autotuning. The legacy NVIDIA
  opt-out and loss-graph fallback were removed; incompatible options fail
  closed instead of silently selecting the slower path. ROCm/HIP, MPS, CPU,
  and other compatibility devices retain their existing eager or Inductor
  paths.
- Added a canonical TensorBoard/wandb metric schema with explicit owners, units,
  aggregation windows, DP reductions, and step axes. Fields already available in
  upstream RSL-RL use its tags verbatim; extensions stay in the terse `Train`,
  `Loss`, `Policy`, `Episode`, and `Perf` groups. Learners emit canonical source
  keys directly, while retired keys fail closed. APPO now also emits the
  upstream-aligned `Policy/mean_std` actor diagnostic. Redundant derived
  percentages, cycle totals, residual timing, and the double-smoothed runner
  return chart are no longer persisted. Historical event files are not rewritten;
  the migration reference is in `docs/metrics.md`.
- Collector episode returns now enter logger state only at the logged step;
  collector counter updates no longer duplicate reward-history entries. DP
  metadata states the exact per-rank mean reduction for episode and reward-term
  fields.

### Removed

- Removed ambiguous logger source aliases and the `reward`/`reward_metrics`
  interfaces. Runner 10-report reward smoothing is checkpoint state only; the
  persisted episode-return field is the collector's 100-episode mean.
- Removed constant configuration charts: staging-pool capacity and the disabled
  reward-normalization scale are no longer persisted as scalars. The APPO
  update count is derived from its configured epoch/minibatch schedule instead
  of being charted.
- Removed derived or semantically mismatched charts: APPO staging-pool occupancy,
  active collector throughput, learner replay throughput, and SAC's
  pre-update action standard deviation are no longer persisted. Async reward
  terms are now averaged across the collector reports in each learner iteration.

### Fixed

- The final on-policy iteration is forced to log on upstream RSL-RL's zero-based
  iteration axis when `log_interval` does not divide the final iteration.
- TensorBoard scalar writes are now batched into a single event record per
  training step. Previously each `add_scalar` call produced one record, and
  the writer thread's per-record open/write/close saturated the async queue
  (depth 10), blocking the learner main thread for ~160 ms per iteration when
  the log directory lives on a network filesystem (FUSE). Falls back to
  per-scalar writes when the batched path is unavailable.

## [1.4.0] - 2026-09-25

### Added

- New age-biased SAC algorithm package built on FlashSAC, with bucketed linear
  age-bias replay sampling in the asynchronous device-authoritative runtime.
- Its double-buffer builder exposed `decay_step`, `replay_min_weight`,
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

- SAC's compiled C51 projection no longer caches an Inductor CUDA Graph
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

- SAC's `torch.compile` path now enables Inductor CUDA Graph replay for
  fused critic/actor loss kernels and defers scalar metric reads to the final
  update in each learner cycle.
- SAC actor updates no longer accumulate unused critic-parameter gradients.
  The policy still receives the same `dQ/da` gradient.
- Compiled SAC updates replace per-loss host finite-check synchronization
  with fused-optimizer device gating, preserving non-finite step suppression
  without fragmenting the learner window.
- SAC critic CUDA Graph replay now captures the Polyak target-network
  update, removing the graph-external foreach launches between critic replays.
- The off-policy runtime manifest now reports the effective CUDA Graph replay,
  packed-staging, target-update capture, and eager-fallback state.

### Fixed

- Removed a redundant CUDA stream synchronization between learner-owned actor
  inference and its blocking D2H action copy. CUDA event timing preserves the
  forward-duration metric without adding another graph-boundary sync.
- SAC CUDA Graph calls now fail closed to eager updates when observation
  normalization is active, matching the existing FlashSAC safety behavior.

## [1.3.2] - 2026-09-22

### Removed

- The TD3 (FastTD3) algorithm: the whole `uni_rl.algos.fast_td3` package
  (`TD3Actor`, `FastTD3Learner`, `build_td3_double_buffer_runner`), the
  built-in `td3` actor branch in `uni_rl.algos.common.actor_factory`, the
  `"td3"` entries in the off-policy worker exploration routing and the
  double-buffer runner display names, and the TD3-only
  `Critic` / `DistributionalQNetwork` networks in
  `uni_rl.algos.common` (SAC and FlashSAC each define their own critic
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

- The SAC, FlashSAC, and FastTD3 double-buffer builders now require and
  directly read `training.inference_request_timeout_sec`.

### Fixed

- Added an off-policy collector-ready handshake after environment
  initialization and runtime-manifest publication. The learner now starts its
  inference-tick timeout only after collector readiness, so backend-owned cold
  starts (such as Genesis JIT and first reset) cannot consume the steady-state
  tick budget. FlashSAC and FastTD3 builders also forward
  `training.inference_request_timeout_sec`, matching SAC.

## [1.2.1] - 2026-09-15

### Removed

- `SACRunner` and `FlashSACRunner` kwargs-style runner classes. They were
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
  `uni_rl.algos.common.learner_boilerplate`; `sac` and `flash_sac`
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

- Grouped algorithm packages under `uni_rl.algos` (`appo`, `sac`,
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
