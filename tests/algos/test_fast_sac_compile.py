from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from uni_rl.algos.fast_sac.learner import (
    DistributionalQNetwork,
    FastSACLearner,
    SACActor,
)


def _small_fast_sac_learner(*, use_autotune: bool = True) -> FastSACLearner:
    return FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=use_autotune,
        max_grad_norm=0.0,
    )


def _small_offpolicy_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.linspace(-0.4, 0.7, steps=batch_size * 4).view(batch_size, 4),
        "critic": torch.linspace(-0.2, 0.9, steps=batch_size * 5).view(batch_size, 5),
        "actions": torch.linspace(-0.5, 0.5, steps=batch_size * 2).view(batch_size, 2),
        "rewards": torch.linspace(-0.3, 0.6, steps=batch_size),
        "next_obs": torch.linspace(0.1, 1.2, steps=batch_size * 4).view(batch_size, 4),
        "next_critic": torch.linspace(-0.7, 0.4, steps=batch_size * 5).view(batch_size, 5),
        "dones": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "truncated": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }


def test_fast_sac_compile_targets_training_hot_paths(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
    )
    learner.device = "cuda"
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FastSACLearner._critic_loss_tensors",
            {"options": {"triton.cudagraphs": True}},
        ),
        (
            "FastSACLearner._actor_loss_tensors",
            {"options": {"triton.cudagraphs": True}},
        ),
    ]


def test_fast_sac_categorical_projection_preserves_rows() -> None:
    num_atoms = 5
    qnet = DistributionalQNetwork(
        obs_dim=4,
        action_dim=2,
        num_atoms=num_atoms,
        v_min=-2.0,
        v_max=2.0,
        hidden_dim=8,
        use_layer_norm=False,
    )
    # A uniform next-state distribution makes the expected projection independent
    # of the initialized hidden layers.
    with torch.no_grad():
        qnet.net[-1].weight.zero_()
        qnet.net[-1].bias.zero_()
    support = torch.linspace(qnet.v_min, qnet.v_max, num_atoms)
    rewards = torch.tensor([-0.7, 0.0, 0.3, 1.1])
    bootstrap = torch.tensor([0.0, 1.0, 1.0, 1.0])
    discount = torch.full_like(rewards, 0.9)

    projected = qnet.projection(
        torch.zeros(4, 4),
        torch.zeros(4, 2),
        rewards,
        bootstrap,
        discount,
        support,
        support.device,
    )

    expected = torch.zeros_like(projected)
    next_probability = 1.0 / num_atoms
    for row, reward in enumerate(rewards):
        target_z = (reward + bootstrap[row] * discount[row] * support).clamp(qnet.v_min, qnet.v_max)
        position = (target_z - qnet.v_min) / ((qnet.v_max - qnet.v_min) / (num_atoms - 1))
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=num_atoms - 1)
        upper_weight = position - lower
        expected[row].scatter_add_(0, lower, next_probability * (1.0 - upper_weight))
        expected[row].scatter_add_(0, upper, next_probability * upper_weight)

    assert torch.allclose(projected, expected, atol=1e-6)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_fast_sac_cuda_adamw_optimizers_are_capturable(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only optimizer kwargs require a CUDA-enabled torch build")

    calls: list[dict[str, Any]] = []

    class _FakeAdamW:
        def __init__(self, _params, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _FakeAdamW)

    for device in ("cuda", torch.device("cuda")):
        FastSACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=5,
            device=device,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            num_atoms=3,
            num_q_networks=2,
            use_layer_norm=False,
            use_autotune=False,
            use_compile=False,
        )

    assert len(calls) == 6
    assert all(call["fused"] for call in calls)
    assert all(call["capturable"] for call in calls)


def test_fast_sac_cpu_adamw_optimizers_keep_default_capturability(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeAdamW:
        def __init__(self, _params, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _FakeAdamW)

    FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
        use_compile=False,
    )

    assert len(calls) == 3
    assert not any(call["fused"] for call in calls)
    assert all("capturable" not in call for call in calls)


def test_fast_sac_amp_dtype_resolution_and_scaler_rules() -> None:
    assert FastSACLearner._resolve_amp_dtype("auto", "cuda") is torch.bfloat16
    assert FastSACLearner._resolve_amp_dtype("auto", "xpu") is torch.bfloat16
    assert FastSACLearner._resolve_amp_dtype("fp16", "cuda") is torch.float16
    assert FastSACLearner._resolve_amp_dtype("bf16", "cuda") is torch.bfloat16

    assert FastSACLearner._should_use_grad_scaler(True, "cuda", torch.float16)
    assert not FastSACLearner._should_use_grad_scaler(True, "cuda", torch.bfloat16)
    assert not FastSACLearner._should_use_grad_scaler(True, "xpu", torch.bfloat16)
    assert not FastSACLearner._should_use_grad_scaler(False, "cuda", torch.float16)

    with pytest.raises(ValueError, match="amp_dtype"):
        FastSACLearner._resolve_amp_dtype("tf32", "cuda")


def test_fast_sac_alpha_loss_helper_matches_reference_value_and_grad() -> None:
    learner = FastSACLearner(
        obs_dim=4,
        action_dim=3,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=True,
    )
    next_log_probs = torch.tensor([-1.25, -0.5, 0.25, 1.5], dtype=torch.float32)
    learner.target_entropy = -1.75
    learner.log_alpha.data.fill_(-2.0)

    reference_log_alpha = learner.log_alpha.detach().clone().requires_grad_(True)
    reference_loss = (-reference_log_alpha.exp() * (next_log_probs + learner.target_entropy)).mean()
    reference_loss.backward()

    learner.log_alpha.grad = None
    alpha_loss = learner._alpha_loss_tensor(next_log_probs)
    alpha_loss.backward()

    assert torch.allclose(alpha_loss.detach(), reference_loss.detach())
    assert learner.log_alpha.grad is not None
    assert reference_log_alpha.grad is not None
    assert torch.allclose(learner.log_alpha.grad, reference_log_alpha.grad)
    assert not next_log_probs.requires_grad


def test_sac_actor_tensor_gaussian_sampling_matches_normal_reference() -> None:
    actor = SACActor(
        obs_dim=4,
        action_dim=3,
        hidden_dim=12,
        use_layer_norm=False,
        action_scale=torch.tensor([0.5, 1.5, 2.0]),
        action_bias=torch.tensor([-0.25, 0.0, 0.75]),
    )
    obs = torch.tensor(
        [
            [-1.0, -0.25, 0.5, 1.25],
            [0.25, 0.5, -0.75, 1.0],
        ],
        dtype=torch.float32,
    )

    _, mean, log_std = actor(obs)
    std = log_std.exp()
    eps = torch.tensor(
        [
            [-0.5, 0.25, 1.0],
            [1.5, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    raw_action = mean + std * eps
    dist = torch.distributions.Normal(mean, std)
    tanh_action = torch.tanh(raw_action)
    expected_action = tanh_action * actor.action_scale + actor.action_bias
    expected_log_prob = dist.log_prob(raw_action)
    expected_log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
    expected_log_prob -= torch.log(actor.action_scale + 1e-6)
    expected_log_prob = expected_log_prob.sum(1)

    action, log_prob = actor._sample_action_and_log_prob(mean, log_std, eps=eps)

    torch.testing.assert_close(action, expected_action)
    torch.testing.assert_close(log_prob, expected_log_prob)


def test_sac_actor_tensor_gaussian_sampling_matches_normal_without_tanh() -> None:
    actor = SACActor(obs_dim=2, action_dim=3, hidden_dim=12, use_layer_norm=False, use_tanh=False)
    mean = torch.tensor(
        [[-0.5, 0.25, 1.0], [1.5, -1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    log_std = torch.tensor(
        [[-1.0, -0.25, 0.5], [0.0, -0.75, 0.25]],
        dtype=torch.float32,
        requires_grad=True,
    )
    eps = torch.tensor(
        [[0.25, -1.5, 0.75], [-0.5, 1.0, 1.5]],
        dtype=torch.float32,
    )

    action, log_prob = actor._sample_action_and_log_prob(mean, log_std, eps=eps)

    reference_mean = mean.detach().clone().requires_grad_(True)
    reference_log_std = log_std.detach().clone().requires_grad_(True)
    reference_std = reference_log_std.exp()
    reference_raw_action = reference_mean + reference_std * eps
    reference_dist = torch.distributions.Normal(reference_mean, reference_std)
    expected_log_prob = reference_dist.log_prob(reference_raw_action).sum(1)

    torch.testing.assert_close(action, reference_raw_action)
    torch.testing.assert_close(log_prob, expected_log_prob)

    loss = (action + log_prob.unsqueeze(1)).sum()
    reference_loss = (reference_raw_action + expected_log_prob.unsqueeze(1)).sum()
    loss.backward()
    reference_loss.backward()

    assert mean.grad is not None
    assert log_std.grad is not None
    assert reference_mean.grad is not None
    assert reference_log_std.grad is not None
    torch.testing.assert_close(mean.grad, reference_mean.grad)
    torch.testing.assert_close(log_std.grad, reference_log_std.grad)


def test_fast_sac_actor_update_does_not_accumulate_critic_gradients() -> None:
    learner = _small_fast_sac_learner()

    learner.update_actor(_small_offpolicy_batch())

    assert all(parameter.requires_grad for parameter in learner.qnet.parameters())
    assert all(parameter.grad is None for parameter in learner.qnet.parameters())


def test_fast_sac_public_updates_can_defer_metric_reads() -> None:
    learner = _small_fast_sac_learner()
    batch = _small_offpolicy_batch()

    assert learner.update_critic(batch, read_metrics=False) == {}
    assert learner.update_actor(batch, read_metrics=False) == {}


@pytest.mark.parametrize(("loss_value", "grad_value"), [(float("nan"), 1.0), (1.0, float("nan"))])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only fused optimizer gate")
def test_fast_sac_device_finite_gate_skips_nonfinite_optimizer_step(
    loss_value: float,
    grad_value: float,
) -> None:
    learner = FastSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cuda:0",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_compile=True,
    )
    parameter = next(learner.qnet.parameters())
    parameter.grad = torch.full_like(parameter, grad_value)
    if not math.isfinite(grad_value):
        learner._gradient_sync = lambda _parameters: None
    before = parameter.detach().clone()

    with learner._optimizer_finite_gate(
        learner.q_optimizer,
        torch.full((), loss_value, device="cuda:0"),
    ):
        learner.q_optimizer.step()
    torch.cuda.synchronize()

    torch.testing.assert_close(parameter, before)
