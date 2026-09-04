# unilab-rl

[![PyPI](https://img.shields.io/pypi/v/unilab-rl)](https://pypi.org/project/unilab-rl/)
[![CI](https://github.com/unilabsim/unilab_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/unilabsim/unilab_rl/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

English | [简体中文](README_zh.md)

Reinforcement learning algorithms and asynchronous runtimes extracted from
[UniLab](https://github.com/unilabsim/UniLab), packaged as a standalone,
simulator-agnostic library.

- Distribution name: `unilab-rl`
- Import namespace: `uni_rl`
- Repository: [unilabsim/unilab_rl](https://github.com/unilabsim/unilab_rl)

## Relationship with UniLab

`uni_rl` is the RL algorithm and async-runtime layer of the UniLab project,
split out into its own package. [UniLab](https://github.com/unilabsim/UniLab)
remains the consumer side: it owns the physics backends, task suites, and
training entrypoints, and injects environments into `uni_rl` through
`uni_rl.env_contract.EnvFactory`. `uni_rl` never imports `unilab` / `unisim`
and never constructs environments itself, so any vectorized environment
satisfying the contract — including simulators outside UniLab — can drive the
algorithms in this package.

If you train with UniLab you already get `uni_rl` transitively. Install
`unilab-rl` directly when you want to reuse its algorithms and async runtime
with your own environment stack.

> Naming note: the originally intended distribution name `uni-rl` is
> unregistrable on PyPI because it ultranormalizes to the existing `unirl`
> project. The distribution is therefore published as `unilab-rl`; the import
> namespace remains `uni_rl` as designed.

## Contents

- **On-policy**: PPO via [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
  (`FinalObservationAwarePPO`, `RslRlVecEnvWrapper`), HIM-PPO, and the HORA
  teacher-policy suite (incl. distillation trainer)
- **Async PPO (APPO)**: native collector/learner multiprocess implementation
- **Off-policy**: FastSAC, FastTD3, and FlashSAC with double-buffer async runners
- **Runtime infrastructure**: shared-memory rollout/replay buffers, replay
  pipelines, data-parallel gradient sync, memory budgeting, tensorboard/wandb
  training loggers, and a trace recorder

## Layout

- `uni_rl.algos.*` — the algorithm layer: on-policy (`rsl_rl` PPO wrappers,
  `him_ppo`, `hora` teacher/distillation suite), async on-policy (`appo`),
  off-policy learners (`fast_sac`, `fast_td3`, `flash_sac`), and shared
  algorithm helpers (`common`)
- `uni_rl.ipc` — runtime infrastructure: async runner, shared-memory
  rollout/replay buffers, replay pipelines, DP gradient sync, memory budget
- `uni_rl.offpolicy` — the generic off-policy double-buffer runner scaffolding
- `uni_rl.logging` — tensorboard/wandb training loggers, trace recorder
- `uni_rl.utils` — device, seed, nan-guard, observation helpers
- `uni_rl.env_contract` — the injected env factory/protocol contract

## Installation

```bash
pip install unilab-rl
# or, with uv:
uv add unilab-rl
```

Requires Python 3.10–3.13 and PyTorch ≥ 2.7.

## Usage

`uni_rl` does not construct environments. Inject a picklable env factory
(`EnvFactory = Callable[[int, Mapping | None], EnvProtocol]`) into the runner
of your chosen algorithm:

```python
from collections.abc import Mapping

from uni_rl.env_contract import EnvProtocol


def make_env(num_envs: int, cfg: Mapping | None) -> EnvProtocol:
    """Top-level factory (picklable by reference; no closures/lambdas)."""
    ...
```

The env contract is a minimal numpy-based, autoresetting vectorized-env
protocol: dict observations keyed by observation group (`obs_groups_spec`),
`step()` with final-observation semantics, and `reset()` returning
`(obs, info)`. See the module docstring in
[`src/uni_rl/env_contract.py`](src/uni_rl/env_contract.py) for the full
contract, and the *new algorithm recipe* section in
[`AGENTS.md`](AGENTS.md) for how to plug in a custom algorithm via
`runtime_resolver` without forking.

## Design contract

`uni_rl` does **not** depend on any simulator or environment library.
Algorithm behavior is owned by the algo modules under `uni_rl.algos.*`;
runtime infrastructure (`ipc`, `logging`, `offpolicy`, `utils`,
`env_contract`) lives at the top level and never depends on the algorithm
layer. See UniLab's training entrypoints for reference env integrations.

## Development

```bash
make sync      # install dependencies (uv)
make test      # pytest
make format    # ruff check --fix + ruff format
uv run mypy src/uni_rl && uv run pyright   # type gates
```

## Citation

If you use `unilab-rl` in your research, please cite the UniLab paper:

```bibtex
@article{jia2026unilab,
  title   = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author  = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal = {arXiv preprint arXiv:2605.30313},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.30313}
}
```

## License

Apache-2.0, same as UniLab.
