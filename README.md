# uni-rl

Reinforcement learning algorithms and async runtimes extracted from
[UniLab](https://github.com/unilabsim/UniLab), usable as a standalone package.

Distribution name: `uni-rl` · import namespace: `uni_rl`

## Contents

- **On-policy**: PPO via [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
  (`FinalObservationAwarePPO`, `RslRlVecEnvWrapper`), HIM-PPO, HORA teacher-policy suite
  (incl. distillation trainer)
- **Async PPO (APPO)**: native collector/learner multiprocess implementation
- **Off-policy**: FastSAC, FastTD3, FlashSAC with double-buffer async runners
- **Runtime infrastructure**: `uni_rl.ipc` (async runner, shared-memory rollout/replay
  buffers, replay pipelines, DP gradient sync, memory budget), `uni_rl.logging`
  (tensorboard/wandb training loggers, trace recorder)

## Installation

Currently published on TestPyPI only:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ uni-rl
```

## Design contract

`uni_rl` does **not** depend on any simulator or environment library. Algorithms
consume a minimal vectorized-env contract (dict observations, `reset() -> (obs, info)`,
`step()` with final-observation semantics); environment construction is injected by the
caller (see UniLab's training entrypoints for reference integrations).

## License

Apache-2.0, same as UniLab.
