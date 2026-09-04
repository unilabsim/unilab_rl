# unilab-rl Agent Principles

**Always use `uv run`, not python**.

uni_rl（distribution 名 `unilab-rl`）是从 UniLab 拆出的 **RL 算法与异步 runtime** 独立包：PPO/APPO/SAC/TD3/FlashSAC/HIM-PPO/HORA 的 runner、learner、collector、IPC 与训练日志。

## Core Principles

1. **Dependency boundary（不可破坏）**: uni_rl 永远不 import `unilab` / `unisim`；`tests/test_smoke.py::test_no_unilab_dependency` 强制该契约。
2. **Env 注入**: uni_rl 不构造 env。算法通过 `uni_rl.env_contract.EnvFactory = Callable[[int, Mapping | None], EnvProtocol]` 接收 env 工厂；factory 必须可被 pickle 引用（collector 跑在 spawn 子进程），禁止闭包 / lambda。
3. ** layering**：`uni_rl/algos/` 是算法层（runner / learner / collector）；`ipc/`、`logging/`、`offpolicy/`、`utils/`、`env_contract.py` 是 runtime 基础设施，留在顶层。algos 可以依赖基础设施层，基础设施层不依赖 algos。
4. **Import 纪律**: 只用绝对 import；注意不要与第三方 `rsl_rl` 包混淆（`uni_rl.algos.rsl_rl*` 是我们的封装层）。
5. **Fix at owner layer**: 算法行为归属 algo owner 模块，不在消费方（UniLab）打补丁。

## Layout

- `src/uni_rl/algos/` — `appo`（异步 PPO）、`fast_sac` / `fast_td3` / `flash_sac`（off-policy learner + double-buffer builder）、`him_ppo`、`hora`（teacher / distillation 套件）、`rsl_rl.py` / `rsl_rl_ppo.py` / `rsl_rl_runtime.py`（rsl_rl 封装）、`common`（共享网络 / normalization / compile 辅助）
- `src/uni_rl/ipc/` — async runner、shm rollout/replay buffer、replay pipeline、DP gradient sync、memory budget
- `src/uni_rl/offpolicy/` — 通用 off-policy double-buffer runner 脚手架
- `src/uni_rl/logging/` — tensorboard / wandb logger、trace recorder
- `src/uni_rl/utils/` — device / seed / nan_guard / observations / final_observation
- `src/uni_rl/env_contract.py` — 注入式 env contract

## 开发命令

- `make sync` / `uv sync` — 安装依赖（dev group 含 pytest / ruff / mypy / pyright）
- `make test` / `uv run pytest` — 全量测试（默认排除 `slow` marker）
- `make format` — ruff check --fix + ruff format
- `uv run mypy src/uni_rl` / `uv run pyright` — 类型 gate（pyright 固定 1.1.408，见 pyproject 注释）

## PR Gate

PR 合入 `main` 前必须 CI 全绿（ruff lint / ruff format / mypy / pyright / pytest+coverage）。本地等价于：`make format && uv run mypy src/uni_rl && uv run pyright && uv run pytest --cov=src/uni_rl`。

## Release 流程（TestPyPI）

1. 在 PR 中 bump `pyproject.toml` 的 `version` 并通过 CI 合入 main。
2. 在 main 上打 tag：`git tag v<X.Y.Z> && git push origin v<X.Y.Z>`。
3. `.github/workflows/release.yml` 自动：`uv build` → wheel 与 sdist 隔离 smoke → 发布到 **TestPyPI**（`TESTPYPI_TOKEN` secret）。
4. 验证：`uv run --isolated --no-project --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --with unilab-rl==<X.Y.Z> -- python -c "import uni_rl; print(uni_rl.__version__)"`（CDN 滞后时加 `--refresh-package unilab-rl` 重试）。

正式发布到 PyPI 需要 maintainer 决策；当前只发布 TestPyPI。

## Context

- 母仓库（消费方 / env contract 参考集成）: [unilabsim/UniLab](https://github.com/unilabsim/UniLab)
- 拆分背景: UniLab roadmap #1476（issues #1477–#1481）
- 命名说明: distribution 名 `unilab-rl`（`uni-rl` 与既有 `unirl` 项目 ultranormalized 冲突，不可注册）；import namespace 保持 `uni_rl`。
