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

UniLab consumes `uni_rl` as an optional extra (`unilab[uni_rl]`) for APPO,
off-policy algorithms, and multi-GPU data-parallel PPO launches; its
single-process PPO path drives upstream rsl_rl directly. Install `unilab-rl`
directly when you want to reuse its algorithms and async runtime with your own
environment stack.

> Naming note: the originally intended distribution name `uni-rl` is
> unregistrable on PyPI because it ultranormalizes to the existing `unirl`
> project. The distribution is therefore published as `unilab-rl`; the import
> namespace remains `uni_rl` as designed.

## Contents

- **Async PPO (APPO)**: native collector/learner multiprocess implementation
  (actor/critic networks built on
  [rsl_rl](https://github.com/leggedrobotics/rsl_rl) model classes)
- **Off-policy**: FastSAC and FlashSAC with double-buffer async runners
- **Runtime infrastructure**: shared-memory rollout/replay buffers, replay
  pipelines, data-parallel gradient sync, memory budgeting, tensorboard/wandb
  training loggers, and a trace recorder

## Layout

- `uni_rl.algos.*` — the algorithm layer: async on-policy (`appo`),
  off-policy learners (`fast_sac`, `flash_sac`), and shared
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

## External multi-node data parallelism

`uni_rl.ipc.dp_launcher` also supports ranks that live on different hosts.
An external orchestrator starts each rank itself and injects the topology
through environment variables:

```bash
UNILAB_DP_EXTERNAL=1        # enable external topology parsing
UNILAB_DP_WORLD_SIZE=2      # global rank count
UNILAB_DP_RANK=1            # this process's global rank
UNILAB_DP_RENDEZVOUS_URL=tcp://192.168.100.1:29501  # TCPStore rendezvous
UNILAB_DP_LOG_DIR=logs/run  # canonical run directory (rank 0 writes)
```

`DpParameterSync` then joins a TCP rendezvous instead of the default local
FileStore, every rank keeps its full host CPU budget (no affinity
partitioning), and device indices stay host-local. Each rank runs a complete
collector + replay + learner pipeline; gradients are all-reduced after every
backward pass, keeping all ranks bitwise-identical from the rank-0
initialization broadcast onward.

The contract is validated end to end by two-node training in UniLab; a
single-host loopback variant (gloo backend) is covered by
`tests/ipc/test_dp_multinode.py`.

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
