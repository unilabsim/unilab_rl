# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
