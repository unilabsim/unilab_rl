from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from uni_rl.utils.seed import (
    apply_training_seed,
    derive_worker_seed,
    resolve_training_seed,
)


def test_resolve_training_seed_prefers_algo_seed_over_legacy_training_seed():
    cfg = OmegaConf.create({"algo": {"seed": 7}, "training": {"seed": 99}})

    seed_info = resolve_training_seed(cfg)

    assert seed_info.configured_seed == 7
    assert seed_info.configured_seed_source == "algo.seed"
    assert seed_info.effective_seed == 7


def test_apply_training_seed_controls_python_numpy_and_torch_rng():
    apply_training_seed(123, torch_runtime=True, cuda=True)
    first = (random.random(), np.random.rand(), torch.rand(3))

    apply_training_seed(123, torch_runtime=True, cuda=True)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert second[0] == first[0]
    assert second[1] == first[1]
    assert torch.equal(second[2], first[2])


def test_apply_training_seed_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        apply_training_seed(-1)


def test_derive_worker_seed_is_deterministic_and_distinct_from_base_seed():
    assert derive_worker_seed(10, worker_index=0) == 11
    assert derive_worker_seed(10, worker_index=3) == 14
    assert derive_worker_seed(None, worker_index=3) is None


# NOTE: UniLab's tests/training/test_seed_contract.py additionally covers
# owner-config Hydra composition against UniLab's conf tree; that part stays
# in UniLab as an integration test (issue #1478).
