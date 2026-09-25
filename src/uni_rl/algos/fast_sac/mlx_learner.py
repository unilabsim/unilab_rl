"""Experimental MLX learner for FastSAC (Apple Silicon spike).

Trains the FastSAC actor/critic with MLX instead of torch MPS: each update
(forward, backward, AdamW, finite gating) is fused with ``mx.compile`` — the
graph-wide fusion that torch's MPS backend lacks.

The torch ``SACActor`` is kept as the inference/checkpoint/export copy and is
re-synced from the MLX weights after every actor update, so the double-buffer
runner, sim2sim validation, and ONNX export paths are unchanged.  Batches
arrive as torch tensors (MPS) and are converted to MLX arrays per update.

``mlx`` is an optional dependency: it is imported lazily at construction, so
the module stays importable (and CI-green) without it.  Select this learner
from an owner config with::

    algo.runtime_resolver: uni_rl.algos.fast_sac.mlx_learner:resolve_mlx_fastsac_runtime

Spike limitations: single-GPU only (no DP), no resume via load_state_dict,
use_tanh=True and num_q_networks=2 only, no grad clipping.
"""

from __future__ import annotations

import importlib
import math
from functools import partial
from typing import Any, Dict, cast

import numpy as np
import torch
import torch.nn as nn

from uni_rl.algos.common.normalization import EmpiricalNormalization
from uni_rl.algos.fast_sac.learner import SACActor
from uni_rl.offpolicy.runtime import OffPolicyRuntime

_mx: Any = None
_nn_mlx: Any = None
_mlx_utils: Any = None


def _mlx():
    global _mx, _nn_mlx, _mlx_utils
    if _mx is None:
        try:
            _mx = importlib.import_module("mlx.core")
            _nn_mlx = importlib.import_module("mlx.nn")
            _mlx_utils = importlib.import_module("mlx.utils")
        except ImportError as exc:
            raise ImportError(
                "MLXFastSACLearner requires the 'mlx' package (pip install mlx; Apple Silicon only)"
            ) from exc
    return _mx


def _tree_map(fn, *trees):
    return _mlx_utils.tree_map(fn, *trees)


def _tree_zeros(tree):
    return _tree_map(_mx.zeros_like, tree)


def _tree_assign(dst, src) -> None:
    """Replace leaves of ``dst`` with ``src`` in place, preserving container
    identity (required for mx.compile state capture)."""
    if isinstance(dst, dict):
        for k in dst:
            if isinstance(dst[k], (dict, list)):
                _tree_assign(dst[k], src[k])
            else:
                dst[k] = src[k]
    elif isinstance(dst, list):
        for i in range(len(dst)):
            if isinstance(dst[i], (dict, list)):
                _tree_assign(dst[i], src[i])
            else:
                dst[i] = src[i]
    else:
        raise TypeError(f"Cannot tree-assign into leaf of type {type(dst)}")


# ---------------------------------------------------------------------------
# MLX functional network definitions (mirror SACActor / DistributionalQNetwork)
# ---------------------------------------------------------------------------


def _linear_init(out_dim: int, in_dim: int, zero: bool = False):
    mx = _mx
    if zero:
        w = mx.zeros((out_dim, in_dim))
    else:
        bound = 1.0 / math.sqrt(in_dim)
        w = mx.random.uniform(-bound, bound, (out_dim, in_dim))
    return {"w": w, "b": mx.zeros((out_dim,))}


def _ln_init(dim: int):
    mx = _mx
    return {"w": mx.ones((dim,)), "b": mx.zeros((dim,))}


def _linear(p, x):
    return x @ p["w"].T + p["b"]


def _layer_norm(p, x, eps: float = 1e-5):
    mx = _mx
    return mx.fast.layer_norm(x, p["w"], p["b"], eps)


def _silu(x):
    mx = _mx
    return x * mx.sigmoid(x)


def _actor_params(obs_dim: int, action_dim: int, hidden_dim: int, use_layer_norm: bool):
    dims = [obs_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4]
    return {
        "layers": [_linear_init(dims[i + 1], dims[i]) for i in range(3)],
        "ln": [_ln_init(dims[i + 1]) for i in range(3)] if use_layer_norm else [],
        "fc_mu": _linear_init(action_dim, dims[3], zero=True),
        "fc_logstd": _linear_init(action_dim, dims[3], zero=True),
    }


def _critic_params(obs_dim: int, action_dim: int, hidden_dim: int, num_atoms: int, use_ln: bool):
    """Twin critic as ensemble-stacked arrays: w (2, out, in), b (2, out)."""
    mx = _mx
    dims = [obs_dim + action_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4, num_atoms]
    layers = []
    for i in range(4):
        base = _linear_init(dims[i + 1], dims[i])
        layers.append(
            {
                "w": mx.stack(
                    [base["w"], mx.random.uniform(-1, 1, base["w"].shape) * 0 + base["w"]]
                ),
                "b": mx.stack([base["b"], base["b"] + 0.0]),
            }
        )
    lns = []
    for i in range(3):
        base = _ln_init(dims[i + 1])
        lns.append(
            {
                "w": mx.stack([base["w"], base["w"] + 0.0]),
                "b": mx.stack([base["b"], base["b"] + 0.0]),
            }
        )
    return {"layers": layers, "ln": lns if use_ln else []}


def _trunk(params, x, use_layer_norm: bool, silu_after_last: bool):
    layers = params["layers"]
    for i, layer in enumerate(layers):
        x = _linear(layer, x)
        if use_layer_norm and i < 3:
            x = _layer_norm(params["ln"][i], x)
        if i < len(layers) - 1 or silu_after_last:
            x = _silu(x)
    return x


def _actor_dist(params, obs, use_layer_norm: bool, log_std_min: float, log_std_max: float):
    """Mirror SACActor.forward: returns (mean, log_std); scale=1, bias=0."""
    mx = _mx
    x = _trunk(params, obs, use_layer_norm, silu_after_last=True)
    mean = _linear(params["fc_mu"], x)
    log_std = _linear(params["fc_logstd"], x)
    log_std = mx.tanh(log_std)
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    mean = mx.clip(mean, -10.0, 10.0)
    return mean, log_std


def _sample_action(params, obs, use_layer_norm, log_std_min, log_std_max, eps=None):
    """Mirror SACActor._sample_action_and_log_prob (action_scale=1, bias=0)."""
    mx = _mx
    mean, log_std = _actor_dist(params, obs, use_layer_norm, log_std_min, log_std_max)
    std = mx.exp(log_std)
    if eps is None:
        eps = mx.random.normal(mean.shape)
    raw = mean + std * eps
    log_prob = -0.5 * (eps**2 + 2.0 * log_std + math.log(2.0 * math.pi))
    tanh_action = mx.tanh(raw)
    log_prob = log_prob - mx.log(1 - tanh_action**2 + 1e-6)
    return tanh_action, log_prob.sum(1), log_std


def _critic_params(obs_dim: int, action_dim: int, hidden_dim: int, num_atoms: int, use_ln: bool):
    dims = [obs_dim + action_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4, num_atoms]
    return {
        "layers": [_linear_init(dims[i + 1], dims[i]) for i in range(4)],
        "ln": [_ln_init(dims[i + 1]) for i in range(3)] if use_ln else [],
    }


def _critic_logits(params, obs, actions, use_layer_norm: bool):
    """Single-critic forward. Returns (B, num_atoms) logits.

    Note: ensemble-batching the twin critic as (2,B,·) bmm was measured slower
    than two sequential passes with MLX on Apple Silicon (same as torch MPS).
    """
    mx = _mx
    x = mx.concatenate([obs, actions], axis=-1)
    return _trunk(params, x, use_layer_norm, silu_after_last=False)


# ---------------------------------------------------------------------------
# Functional AdamW (torch semantics: decoupled weight decay + bias correction)
# ---------------------------------------------------------------------------


def _adamw_update(p, g, m, v, t, lr, weight_decay, gate, betas=(0.9, 0.95), eps=1e-8):
    """AdamW step with a scalar ``gate`` (0/1) selecting skip vs. apply.

    Sanitizes non-finite gradients to zero before any arithmetic: a plain
    ``gate * x`` would still yield NaN when x is NaN (0 * NaN = NaN).  When
    gated off, parameters and moments are left untouched, mirroring torch's
    skipped optimizer step.
    """
    mx = _mx
    b1, b2 = betas

    def _sel(new, old):
        return mx.where(gate > 0, new, old)

    t_new = t + gate
    g_safe = _tree_map(lambda g_: mx.where(mx.isfinite(g_), g_, mx.zeros_like(g_)), g)
    m = _tree_map(lambda m_, g_: _sel(b1 * m_ + (1 - b1) * g_, m_), m, g_safe)
    v = _tree_map(lambda v_, g_: _sel(b2 * v_ + (1 - b2) * g_ * g_, v_), v, g_safe)
    bc1 = 1 - b1**t_new
    bc2 = 1 - b2**t_new
    new_p = _tree_map(
        lambda p_, m_, v_: _sel(
            p_ - lr * ((m_ / bc1) / (mx.sqrt(v_ / bc2) + eps) + weight_decay * p_), p_
        ),
        p,
        m,
        v,
    )
    return new_p, m, v, t_new


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class MLXFastSACLearner:
    """FastSAC learner that trains with MLX and serves inference via torch.

    Public interface mirrors ``FastSACLearner`` as consumed by
    ``DoubleBufferOffPolicyRunner``.
    """

    supports_deferred_update_metrics = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int,
        device: str = "cpu",
        gamma: float = 0.97,
        tau: float = 0.125,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.001,
        target_entropy_ratio: float = 0.0,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        num_q_networks: int = 2,
        use_layer_norm: bool = True,
        use_tanh: bool = True,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        weight_decay: float = 0.001,
        max_grad_norm: float = 0.0,
        use_autotune: bool = True,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        use_compile: bool = True,
        obs_normalization: bool = False,
        nvtx_profile_ranges: bool = False,
        use_cuda_graph_critic: bool = False,
        use_cuda_graph_critic_packed_staging: bool = False,
        use_cuda_graph_actor: bool = False,
        use_cuda_graph_actor_packed_staging: bool = False,
    ):
        mx = _mlx()
        if any(
            (
                use_cuda_graph_critic,
                use_cuda_graph_critic_packed_staging,
                use_cuda_graph_actor,
                use_cuda_graph_actor_packed_staging,
            )
        ):
            raise ValueError("CUDA graphs are CUDA-only; MLXFastSACLearner does not use them")
        device_type = torch.device(device).type
        if device_type not in ("mps", "cpu"):
            raise ValueError(f"MLXFastSACLearner only supports mps/cpu devices, got {device!r}")
        if not use_tanh:
            raise ValueError("MLXFastSACLearner spike only implements use_tanh=True")
        if num_q_networks != 2:
            raise ValueError("MLXFastSACLearner spike assumes num_q_networks=2")
        if max_grad_norm > 0:
            raise ValueError("MLXFastSACLearner spike does not implement grad clipping")

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.use_autotune = use_autotune
        self.use_compile = bool(use_compile)
        self.use_amp = False
        # Finite gating happens in-graph (the compiled step zeroes the update
        # on non-finite loss/grads), so no host checks are needed.
        self._host_finite_checks = False
        self.critic_obs_dim = critic_obs_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_layer_norm = use_layer_norm
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.weight_decay = weight_decay
        self.target_entropy = -action_dim * target_entropy_ratio

        # Torch inference/export actor (kept in sync from the MLX weights).
        self.actor = SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )

        self.obs_normalizer: EmpiricalNormalization | nn.Identity
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()

        # MLX parameter trees, initialised from the torch actor so both copies
        # start from identical weights.
        self._actor_p = _actor_params(obs_dim, action_dim, actor_hidden_dim, use_layer_norm)
        self._critic_p = [
            _critic_params(critic_obs_dim, action_dim, critic_hidden_dim, num_atoms, use_layer_norm)
            for _ in range(num_q_networks)
        ]
        self._load_torch_actor_into_mlx()
        mx.eval(self._actor_p, self._critic_p)

        self._critic_target = _tree_map(lambda x: x + 0.0, self._critic_p)
        self._log_alpha = {"a": mx.array([math.log(alpha_init)])}

        self._critic_opt = {
            "m": _tree_zeros(self._critic_p),
            "v": _tree_zeros(self._critic_p),
            "t": mx.array(0.0),
        }
        self._actor_opt = {
            "m": _tree_zeros(self._actor_p),
            "v": _tree_zeros(self._actor_p),
            "t": mx.array(0.0),
        }
        self._alpha_opt = {
            "m": _tree_zeros(self._log_alpha),
            "v": _tree_zeros(self._log_alpha),
            "t": mx.array(0.0),
        }

        self._q_support = mx.linspace(v_min, v_max, num_atoms)

        self.update_count = 0
        self._pending_actor_metric_values: tuple | None = None

        self._state = [
            self._critic_p,
            self._actor_p,
            self._log_alpha,
            self._critic_target,
            self._critic_opt,
            self._actor_opt,
            self._alpha_opt,
            mx.random.state,
        ]
        if self.use_compile:
            compile_step = partial(mx.compile, inputs=self._state, outputs=self._state)
            self._critic_step = compile_step(self._critic_step_impl)
            self._actor_step = compile_step(self._actor_step_impl)
            self._target_step = compile_step(self._target_step_impl)
        else:
            self._critic_step = self._critic_step_impl
            self._actor_step = self._actor_step_impl
            self._target_step = self._target_step_impl

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    @staticmethod
    def _t2m(t: torch.Tensor):
        return _mx.array(np.asarray(t.detach().cpu()))

    def _load_torch_actor_into_mlx(self) -> None:
        """Initialise MLX actor params from the torch SACActor weights."""
        mx = _mx
        sd = self.actor.state_dict()
        p = self._actor_p
        for i in range(3):
            p["layers"][i]["w"] = mx.array(np.asarray(sd[f"net.{3 * i}.weight"].cpu()))
            p["layers"][i]["b"] = mx.array(np.asarray(sd[f"net.{3 * i}.bias"].cpu()))
            if self.use_layer_norm:
                p["ln"][i]["w"] = mx.array(np.asarray(sd[f"net.{3 * i + 1}.weight"].cpu()))
                p["ln"][i]["b"] = mx.array(np.asarray(sd[f"net.{3 * i + 1}.bias"].cpu()))
        p["fc_mu"]["w"] = mx.array(np.asarray(sd["fc_mu.weight"].cpu()))
        p["fc_mu"]["b"] = mx.array(np.asarray(sd["fc_mu.bias"].cpu()))
        p["fc_logstd"]["w"] = mx.array(np.asarray(sd["fc_logstd.weight"].cpu()))
        p["fc_logstd"]["b"] = mx.array(np.asarray(sd["fc_logstd.bias"].cpu()))

    def _sync_torch_actor_from_mlx(self) -> None:
        """Copy MLX actor weights into the torch inference/export actor."""
        p = self._actor_p
        sd: dict[str, torch.Tensor] = {}
        for i in range(3):
            sd[f"net.{3 * i}.weight"] = torch.from_numpy(np.asarray(p["layers"][i]["w"]))
            sd[f"net.{3 * i}.bias"] = torch.from_numpy(np.asarray(p["layers"][i]["b"]))
            if self.use_layer_norm:
                sd[f"net.{3 * i + 1}.weight"] = torch.from_numpy(np.asarray(p["ln"][i]["w"]))
                sd[f"net.{3 * i + 1}.bias"] = torch.from_numpy(np.asarray(p["ln"][i]["b"]))
        sd["fc_mu.weight"] = torch.from_numpy(np.asarray(p["fc_mu"]["w"]))
        sd["fc_mu.bias"] = torch.from_numpy(np.asarray(p["fc_mu"]["b"]))
        sd["fc_logstd.weight"] = torch.from_numpy(np.asarray(p["fc_logstd"]["w"]))
        sd["fc_logstd.bias"] = torch.from_numpy(np.asarray(p["fc_logstd"]["b"]))
        self.actor.load_state_dict(sd, strict=False)  # action_scale/bias buffers are static
        self.actor.to(self.device)

    def _critic_state_dict_torch(self, params, prefix: str) -> dict[str, torch.Tensor]:
        """Key an MLX critic tree list like SACCritic's state_dict (qnets.<i>.net.*)."""
        sd: dict[str, torch.Tensor] = {}
        for qi, qnet in enumerate(params):
            for layer_i, layer in enumerate(qnet["layers"]):
                sd[f"{prefix}.{qi}.net.{3 * layer_i}.weight"] = torch.from_numpy(
                    np.asarray(layer["w"])
                )
                sd[f"{prefix}.{qi}.net.{3 * layer_i}.bias"] = torch.from_numpy(
                    np.asarray(layer["b"])
                )
                if self.use_layer_norm and layer_i < 3:
                    sd[f"{prefix}.{qi}.net.{3 * layer_i + 1}.weight"] = torch.from_numpy(
                        np.asarray(qnet["ln"][layer_i]["w"])
                    )
                    sd[f"{prefix}.{qi}.net.{3 * layer_i + 1}.bias"] = torch.from_numpy(
                        np.asarray(qnet["ln"][layer_i]["b"])
                    )
        return sd

    # ------------------------------------------------------------------
    # Losses (functional; eps is injectable for the parity test)
    # ------------------------------------------------------------------

    def _critic_loss_mx(self, critic_p, actor_p, log_alpha, target_p, batch, eps=None):
        """Mirror FastSACLearner._critic_loss_tensors + the alpha loss."""
        mx = _mx
        obs, critic_obs, actions, rewards, next_obs, next_critic, dones, truncated = batch
        bootstrap = mx.clip(1.0 - dones + truncated, 0.0, 1.0)
        discount = mx.full(dones.shape, self.gamma)

        sg_actor = _tree_map(mx.stop_gradient, actor_p)
        next_actions, next_log_probs, _ = _sample_action(
            sg_actor, next_obs, self.use_layer_norm, self.log_std_min, self.log_std_max, eps=eps
        )
        alpha = mx.exp(mx.stop_gradient(log_alpha["a"]))
        adjusted_rewards = rewards - discount * bootstrap * alpha * next_log_probs

        sg_target = _tree_map(mx.stop_gradient, target_p)
        per_q_proj = []
        for q in range(2):
            logits = _critic_logits(sg_target[q], next_critic, next_actions, self.use_layer_norm)
            next_dist = mx.softmax(logits, axis=-1)
            per_q_proj.append(self._projection(adjusted_rewards, bootstrap, discount, next_dist))
        proj = mx.stack(per_q_proj, axis=0)
        target_values = (proj * self._q_support).sum(-1)

        losses = []
        for q in range(2):
            logits = _critic_logits(critic_p[q], critic_obs, actions, self.use_layer_norm)
            logp = mx.maximum(_nn_mlx.log_softmax(logits, axis=-1), -30.0)
            losses.append(-(proj[q] * logp).sum(-1))
        qf_loss = mx.stack(losses, axis=0).mean(axis=1).sum(axis=0)

        entropy_error = (mx.stop_gradient(next_log_probs) + self.target_entropy).mean()
        alpha_loss = -(mx.exp(log_alpha["a"]) * entropy_error)
        return (
            qf_loss,
            alpha_loss,
            target_values.max(),
            target_values.min(),
            mx.stop_gradient(next_log_probs),
        )

    def _projection(self, rewards, bootstrap, discount, next_dist):
        """C51 projection for one (B, atoms) dist; mirrors DistributionalQNetwork.projection."""
        mx = _mx
        batch_size, atoms = next_dist.shape
        delta_z = (self.v_max - self.v_min) / (atoms - 1)
        target_z = rewards[:, None] + bootstrap[:, None] * discount[:, None] * self._q_support
        target_z = mx.clip(target_z, self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = mx.floor(b).astype(mx.int32)
        upper = mx.minimum(lower + 1, atoms - 1)
        upper_w = b - lower.astype(mx.float32)
        lower_w = 1.0 - upper_w

        row_offsets = mx.arange(batch_size)[:, None] * atoms
        flat_lower = (lower + row_offsets).reshape(-1)
        flat_upper = (upper + row_offsets).reshape(-1)
        proj = mx.zeros((batch_size * atoms,))
        proj = proj.at[flat_lower].add((next_dist * lower_w).reshape(-1))
        proj = proj.at[flat_upper].add((next_dist * upper_w).reshape(-1))
        return proj.reshape(batch_size, atoms)

    def _actor_loss_mx(self, actor_p, critic_p, log_alpha, batch, eps=None):
        """Mirror FastSACLearner._actor_loss_tensors (critic stop-graded)."""
        mx = _mx
        obs, critic_obs = batch
        actions, log_probs, log_std = _sample_action(
            actor_p, obs, self.use_layer_norm, self.log_std_min, self.log_std_max, eps=eps
        )
        sg_critic = _tree_map(mx.stop_gradient, critic_p)
        q_values = []
        for q in range(2):
            logits = _critic_logits(sg_critic[q], critic_obs, actions, self.use_layer_norm)
            probs = mx.softmax(logits, axis=-1)
            q_values.append((probs * self._q_support).sum(-1))
        qf_value = mx.stack(q_values, axis=0).mean(axis=0)
        alpha = mx.exp(mx.stop_gradient(log_alpha["a"]))
        actor_loss = (alpha * log_probs - qf_value).mean()
        policy_entropy = -log_probs.mean()
        action_std = mx.exp(log_std).mean()
        return actor_loss, policy_entropy, action_std

    # ------------------------------------------------------------------
    # Update steps (optionally compiled with state capture)
    # ------------------------------------------------------------------

    def _finite_gate(self, loss, grads):
        """On-device zero-out of an update on non-finite loss or gradients."""
        mx = _mx
        finite = mx.isfinite(loss).all()
        for _, arr in _mlx_utils.tree_flatten(grads):
            finite = finite & mx.isfinite(arr).all()
        return mx.where(finite, 1.0, 0.0)

    def _critic_step_impl(self, batch):
        mx = _mx

        def joint_loss(critic_p, log_alpha):
            qf_loss, alpha_loss, tq_max, tq_min, nlp = self._critic_loss_mx(
                critic_p, self._actor_p, log_alpha, self._critic_target, batch
            )
            return qf_loss + alpha_loss.reshape(()), (qf_loss, alpha_loss, tq_max, tq_min, nlp)

        (loss, aux), grads = mx.value_and_grad(joint_loss, argnums=(0, 1))(
            self._critic_p, self._log_alpha
        )
        qf_loss, alpha_loss, tq_max, tq_min, _nlp = aux
        critic_grads, alpha_grads = grads

        gate = self._finite_gate(qf_loss, critic_grads)
        new_p, m, v, t = _adamw_update(
            self._critic_p,
            critic_grads,
            self._critic_opt["m"],
            self._critic_opt["v"],
            self._critic_opt["t"],
            self.critic_lr,
            self.weight_decay,
            gate,
        )
        _tree_assign(self._critic_p, new_p)
        _tree_assign(self._critic_opt["m"], m)
        _tree_assign(self._critic_opt["v"], v)
        self._critic_opt["t"] = t

        if self.use_autotune:
            gate_a = self._finite_gate(alpha_loss, alpha_grads)
            new_a, m, v, t = _adamw_update(
                self._log_alpha,
                alpha_grads,
                self._alpha_opt["m"],
                self._alpha_opt["v"],
                self._alpha_opt["t"],
                self.alpha_lr,
                0.0,
                gate_a,
            )
            _tree_assign(self._log_alpha, new_a)
            _tree_assign(self._alpha_opt["m"], m)
            _tree_assign(self._alpha_opt["v"], v)
            self._alpha_opt["t"] = t

        return (
            qf_loss.reshape(()),
            alpha_loss.reshape(()),
            tq_max.reshape(()),
            tq_min.reshape(()),
            mx.exp(self._log_alpha["a"]).reshape(()),
        )

    def _actor_step_impl(self, batch):
        mx = _mx

        def loss_fn(actor_p):
            actor_loss, policy_entropy, action_std = self._actor_loss_mx(
                actor_p, self._critic_p, self._log_alpha, batch
            )
            return actor_loss, (policy_entropy, action_std)

        (actor_loss, aux), grads = mx.value_and_grad(loss_fn)(self._actor_p)
        policy_entropy, action_std = aux

        gate = self._finite_gate(actor_loss, grads)
        new_p, m, v, t = _adamw_update(
            self._actor_p,
            grads,
            self._actor_opt["m"],
            self._actor_opt["v"],
            self._actor_opt["t"],
            self.actor_lr,
            self.weight_decay,
            gate,
        )
        _tree_assign(self._actor_p, new_p)
        _tree_assign(self._actor_opt["m"], m)
        _tree_assign(self._actor_opt["v"], v)
        self._actor_opt["t"] = t

        return (
            actor_loss.reshape(()),
            policy_entropy.reshape(()),
            action_std.reshape(()),
        )

    def _target_step_impl(self):
        new_target = _tree_map(
            lambda tp, p: (1 - self.tau) * tp + self.tau * p, self._critic_target, self._critic_p
        )
        _tree_assign(self._critic_target, new_target)
        return self._critic_target[0]["layers"][0]["w"].sum().reshape(())

    # ------------------------------------------------------------------
    # Public learner interface
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        if isinstance(self.obs_normalizer, nn.Identity):
            return obs
        normalizer = cast(EmpiricalNormalization, self.obs_normalizer)
        if update:
            with torch.no_grad():
                normalizer.update(obs)
        return cast(torch.Tensor, normalizer(obs, update=False))

    def update_critic(self, batch: Dict[str, torch.Tensor], *, read_metrics: bool = True):
        obs = self.normalize_obs(batch["obs"], update=True)
        next_obs = self.normalize_obs(batch["next_obs"], update=False)
        mx_batch = (
            self._t2m(obs),
            self._t2m(batch["critic"]),
            self._t2m(batch["actions"]),
            self._t2m(batch["rewards"]),
            self._t2m(next_obs),
            self._t2m(batch["next_critic"]),
            self._t2m(batch["dones"]),
            self._t2m(batch["truncated"]),
        )
        out = self._critic_step(mx_batch)
        if not read_metrics:
            return {}
        mx = _mx
        mx.eval(out)
        qf_loss, alpha_loss, tq_max, tq_min, alpha = (v.item() for v in out)
        return {
            "qf_loss": qf_loss,
            "critic_grad_norm": 0.0,
            "target_q_max": tq_max,
            "target_q_min": tq_min,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
        }

    def update_actor(self, batch: Dict[str, torch.Tensor], *, read_metrics: bool = True):
        obs = self.normalize_obs(batch["obs"], update=False)
        mx_batch = (
            self._t2m(obs),
            self._t2m(batch["critic"]),
        )
        out = self._actor_step(mx_batch)
        self._sync_torch_actor_from_mlx()
        if not read_metrics:
            self._pending_actor_metric_values = out
            return {}
        mx = _mx
        mx.eval(out)
        actor_loss, policy_entropy, action_std = (v.item() for v in out)
        return {
            "actor_loss": actor_loss,
            "actor_grad_norm": 0.0,
            "policy_entropy": policy_entropy,
            "action_std": action_std,
        }

    def read_deferred_actor_metrics(self) -> Dict[str, float]:
        values = self._pending_actor_metric_values
        self._pending_actor_metric_values = None
        if values is None:
            return {}
        mx = _mx
        mx.eval(values)
        actor_loss, policy_entropy, action_std = (v.item() for v in values)
        return {
            "actor_loss": actor_loss,
            "actor_grad_norm": 0.0,
            "policy_entropy": policy_entropy,
            "action_std": action_std,
        }

    def soft_update_target(self) -> None:
        self._target_step()

    def set_gradient_sync(self, sync) -> None:
        raise NotImplementedError("MLXFastSACLearner does not support multi-GPU DP")

    def dp_initial_sync_tensors(self):
        raise NotImplementedError("MLXFastSACLearner does not support multi-GPU DP")

    def get_state_dict(self) -> Dict[str, Any]:
        mx = _mx
        mx.eval(self._critic_p, self._critic_target, self._actor_p, self._log_alpha)
        return {
            "actor": self.actor.state_dict(),
            "qnet": self._critic_state_dict_torch(self._critic_p, "qnets"),
            "qnet_target": self._critic_state_dict_torch(self._critic_target, "qnets"),
            "log_alpha": torch.tensor([self._log_alpha["a"].item()]),
            "mlx_optimizer_state": {
                "critic": _tree_map(lambda x: np.asarray(x), self._critic_opt),
                "actor": _tree_map(lambda x: np.asarray(x), self._actor_opt),
                "alpha": _tree_map(lambda x: np.asarray(x), self._alpha_opt),
            },
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict")
                else None
            ),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        raise NotImplementedError("MLXFastSACLearner spike does not support resume")


def resolve_mlx_fastsac_runtime(rl_cfg: dict[str, Any]) -> OffPolicyRuntime:
    """Owner-config resolver: select the MLX FastSAC learner."""
    return OffPolicyRuntime(learner_cls=MLXFastSACLearner, algo_type="sac")
