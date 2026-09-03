"""Shared fixtures for uni_rl tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def mp_ctx():
    return torch.multiprocessing.get_context("spawn")


@pytest.fixture
def tiny_weight_shapes():
    """Small MLP param shapes dict — linear(8,16) + bias, linear(16,3) + bias."""
    return {
        "layer1.weight": torch.Size([16, 8]),
        "layer1.bias": torch.Size([16]),
        "layer2.weight": torch.Size([3, 16]),
        "layer2.bias": torch.Size([3]),
    }
