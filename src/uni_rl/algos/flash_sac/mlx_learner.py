"""Experimental MLX learner for FlashSAC (Apple Silicon spike).

Trains the FlashSAC actor/critic with MLX instead of torch MPS: each update
(forward, backward, Adam, finite gating, weight normalization, BN running-stat
update, cosine LR schedule) is fused with ``mx.compile`` — the graph-wide
fusion that torch's MPS backend lacks.  Mirrors
``uni_rl.algos.fast_sac.mlx_learner`` in structure.

The torch ``FlashSACActor`` is kept as the inference/checkpoint/export copy
(weights *and* BN running stats — exploration runs with ``training=False``)
and is re-synced from the MLX weights after every actor update, so the
double-buffer runner, sim2sim validation, and ONNX export paths are
unchanged.  Batches arrive as torch tensors (MPS) and are converted to MLX
arrays per update.  ``RewardNormalizer`` stays on the torch side: the runner
feeds it torch tensors and ``update_critic`` normalizes rewards before the
MLX conversion, keeping it out of the compiled graph.

``mlx`` is an optional dependency: it is imported lazily at construction, so
the module stays importable (and CI-green) without it.  Select this learner
from an owner config with::

    algo.runtime_resolver: uni_rl.algos.flash_sac.mlx_learner:resolve_mlx_flashsac_runtime

Spike limitations: single-GPU only (no DP), no resume via load_state_dict,
num_qs=2, n_step=1, actor_bc_alpha=0 only, no grad clipping.  NaN gating is
done in-graph (loss/grads sanitized, update selected with ``mx.where``);
unlike the torch path, BN running-stat updates are also gated, so a poisoned
batch cannot corrupt the inference BN statistics.
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
from uni_rl.algos.flash_sac.learner import RewardNormalizer
from uni_rl.algos.flash_sac.network import FlashSACActor
from uni_rl.algos.flash_sac.update import resolve_target_entropy
from uni_rl.offpolicy.runtime import OffPolicyRuntime

_mx: Any = None
_nn_mlx: Any = None
_mlx_utils: Any = None

_BN_EPS = 1e-5
_BN_MOMENTUM = 0.01
_RMS_EPS = 1e-6
_BLOCK_EXPANSION = 4


def _mlx():
    global _mx, _nn_mlx, _mlx_utils
    if _mx is None:
        try:
            _mx = importlib.import_module("mlx.core")
            _nn_mlx = importlib.import_module("mlx.nn")
            _mlx_utils = importlib.import_module("mlx.utils")
        except ImportError as exc:
            raise ImportError(
                "MLXFlashSACLearner requires the 'mlx' package (pip install mlx; Apple Silicon only)"
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
# Parameter-tree initialisation (orthogonal UnitLinear; QR is CPU-only in MLX)
# ---------------------------------------------------------------------------


def _ortho_init(out_dim: int, in_dim: int):
    """Row-orthogonal (out_dim, in_dim) init; QR is CPU-only in MLX."""
    mx = _mx
    if out_dim <= in_dim:
        q, r = mx.linalg.qr(mx.random.normal((in_dim, out_dim), stream=mx.cpu), stream=mx.cpu)
        w = q.T * mx.sign(mx.diag(r))[:, None]
    else:
        q, r = mx.linalg.qr(mx.random.normal((out_dim, in_dim), stream=mx.cpu), stream=mx.cpu)
        w = q * mx.sign(mx.diag(r))[None, :]
    return mx.array(np.asarray(w))  # back to the default (GPU) stream


def _actor_params(obs_dim: int, action_dim: int, hidden_dim: int, num_blocks: int):
    mx = _mx
    inner = hidden_dim * _BLOCK_EXPANSION
    return {
        "embedder": {
            "w": _ortho_init(hidden_dim, obs_dim),
            "bn_w": mx.ones((obs_dim,)),
            "bn_b": mx.zeros((obs_dim,)),
        },
        "blocks": [
            {
                "w1": _ortho_init(inner, hidden_dim),
                "bn1_w": mx.ones((inner,)),
                "bn1_b": mx.zeros((inner,)),
                "w2": _ortho_init(hidden_dim, inner),
                "bn2_w": mx.ones((hidden_dim,)),
                "bn2_b": mx.zeros((hidden_dim,)),
            }
            for _ in range(num_blocks)
        ],
        "post_norm_w": mx.ones((hidden_dim,)),
        "mean_w": _ortho_init(action_dim, hidden_dim),
        "mean_b": mx.zeros((action_dim,)),
        "std_w": _ortho_init(action_dim, hidden_dim),
        "std_b": mx.zeros((action_dim,)),
    }


def _actor_bn_buffers(obs_dim: int, hidden_dim: int, num_blocks: int):
    mx = _mx
    inner = hidden_dim * _BLOCK_EXPANSION
    return {
        "embedder": {"mean": mx.zeros((obs_dim,)), "var": mx.ones((obs_dim,))},
        "blocks": [
            {
                "bn1": {"mean": mx.zeros((inner,)), "var": mx.ones((inner,))},
                "bn2": {"mean": mx.zeros((hidden_dim,)), "var": mx.ones((hidden_dim,))},
            }
            for _ in range(num_blocks)
        ],
    }


def _stack2(tree_a, tree_b):
    """Stack two identical single-q trees along a new leading axis."""
    return _tree_map(lambda a, b: _mx.stack([a, b]), tree_a, tree_b)


def _critic_params(input_dim: int, hidden_dim: int, num_blocks: int, num_bins: int):
    """Twin critic as ensemble-stacked arrays: w (2, out, in), bn (2, f)."""
    mx = _mx
    inner = hidden_dim * _BLOCK_EXPANSION
    return {
        "embedder": {
            "w": mx.stack([_ortho_init(hidden_dim, input_dim) for _ in range(2)]),
            "bn_w": mx.ones((2, input_dim)),
            "bn_b": mx.zeros((2, input_dim)),
        },
        "blocks": [
            {
                "w1": mx.stack([_ortho_init(inner, hidden_dim) for _ in range(2)]),
                "bn1_w": mx.ones((2, inner)),
                "bn1_b": mx.zeros((2, inner)),
                "w2": mx.stack([_ortho_init(hidden_dim, inner) for _ in range(2)]),
                "bn2_w": mx.ones((2, hidden_dim)),
                "bn2_b": mx.zeros((2, hidden_dim)),
            }
            for _ in range(num_blocks)
        ],
        "post_norm_w": mx.ones((2, hidden_dim)),
        "logit_w": mx.stack([_ortho_init(num_bins, hidden_dim) for _ in range(2)]),
        "logit_b": mx.zeros((2, num_bins)),
    }


def _critic_bn_buffers(input_dim: int, hidden_dim: int, num_blocks: int):
    mx = _mx
    inner = hidden_dim * _BLOCK_EXPANSION
    return {
        "embedder": {"mean": mx.zeros((2, input_dim)), "var": mx.ones((2, input_dim))},
        "blocks": [
            {
                "bn1": {"mean": mx.zeros((2, inner)), "var": mx.ones((2, inner))},
                "bn2": {"mean": mx.zeros((2, hidden_dim)), "var": mx.ones((2, hidden_dim))},
            }
            for _ in range(num_blocks)
        ],
    }


# ---------------------------------------------------------------------------
# Functional network definitions (mirror layers.py / network.py)
# ---------------------------------------------------------------------------


def _unit_bn(x, w, b, buf, training: bool):
    """UnitBatchNorm on (B, f). Returns (y, stats|None); stats are biased."""
    mx = _mx
    if training:
        mean = mx.mean(x, axis=0, keepdims=True)
        var = mx.var(x, axis=0, keepdims=True)
        y = (x - mean) * mx.rsqrt(var + _BN_EPS)
        return y * w + b, {"mean": mean.reshape(-1), "var": var.reshape(-1)}
    y = (x - buf["mean"]) * mx.rsqrt(buf["var"] + _BN_EPS)
    return y * w + b, None


def _ens_bn(x, w, b, buf, training: bool):
    """EnsembleUnitBatchNorm on (2, B, f). Batch stats over axis 1."""
    mx = _mx
    if training:
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        y = (x - mean) * mx.rsqrt(var + _BN_EPS)
        stats = {"mean": mean[:, 0, :], "var": var[:, 0, :]}
        return y * w[:, None, :] + b[:, None, :], stats
    y = (x - buf["mean"][:, None, :]) * mx.rsqrt(buf["var"][:, None, :] + _BN_EPS)
    return y * w[:, None, :] + b[:, None, :], None


def _rms_norm(x, w):
    mx = _mx
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + _RMS_EPS)
    return (x / rms) * w


def _ens_rms_norm(x, w):
    mx = _mx
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + _RMS_EPS)
    return (x / rms) * w[:, None, :]


def _softplus(x):
    mx = _mx
    return mx.logaddexp(x, mx.zeros_like(x))


def _actor_encode(p, bn, obs, training: bool, stats: list):
    """Embedder -> blocks -> post RMSNorm. Appends BN stats when training."""
    mx = _mx
    x, s = _unit_bn(obs, p["embedder"]["bn_w"], p["embedder"]["bn_b"], bn["embedder"], training)
    if training:
        stats.append(("embedder", s))
    x = x @ p["embedder"]["w"].T
    for i, blk in enumerate(p["blocks"]):
        residual = x
        x = x @ blk["w1"].T
        x, s = _unit_bn(x, blk["bn1_w"], blk["bn1_b"], bn["blocks"][i]["bn1"], training)
        if training:
            stats.append((f"blocks.{i}.bn1", s))
        x = mx.maximum(x, 0.0)
        x = x @ blk["w2"].T
        x, s = _unit_bn(x, blk["bn2_w"], blk["bn2_b"], bn["blocks"][i]["bn2"], training)
        if training:
            stats.append((f"blocks.{i}.bn2", s))
        x = mx.maximum(x, 0.0)
        x = x + residual
    return _rms_norm(x, p["post_norm_w"])


def _actor_forward(p, bn, obs, training: bool, log_std_min: float, log_std_max: float, eps=None):
    """Mirror FlashSACActor.forward -> NormalTanhPolicy. Returns
    (tanh_action, log_prob, mean, std, stats_list)."""
    mx = _mx
    stats: list = []
    x = _actor_encode(p, bn, obs, training, stats)
    mean = x @ p["mean_w"].T + p["mean_b"]
    raw_log_std = x @ p["std_w"].T + p["std_b"]
    log_std = log_std_min + (log_std_max - log_std_min) * 0.5 * (1.0 + mx.tanh(raw_log_std))
    std = mx.exp(log_std)
    if eps is None:
        eps = mx.random.normal(mean.shape)
    raw = mean + std * eps
    action = mx.tanh(raw)
    z = (raw - mean) / std
    log_prob = -0.5 * z * z - log_std - 0.5 * math.log(2.0 * math.pi)
    jacobian = 2.0 * (math.log(2.0) - raw - _softplus(-2.0 * raw))
    log_prob = (log_prob - jacobian).sum(-1)
    return action, log_prob, mean, std, stats


def _critic_forward_ens(p, bn, obs, actions, training: bool, support):
    """Mirror FlashSACDoubleCritic.forward (ensemble einsum path).

    Returns (values (2, B), log_probs (2, B, bins), stats_tree|None).
    """
    mx = _mx
    x = mx.concatenate([obs, actions], axis=-1)
    x = mx.broadcast_to(x[None], (2, x.shape[0], x.shape[1]))
    stats: dict[str, Any] = {"embedder": None, "blocks": []}
    x, s = _ens_bn(x, p["embedder"]["bn_w"], p["embedder"]["bn_b"], bn["embedder"], training)
    if training:
        stats["embedder"] = s
    x = mx.matmul(x, p["embedder"]["w"].transpose(0, 2, 1))
    for i, blk in enumerate(p["blocks"]):
        residual = x
        x = mx.matmul(x, blk["w1"].transpose(0, 2, 1))
        x, s1 = _ens_bn(x, blk["bn1_w"], blk["bn1_b"], bn["blocks"][i]["bn1"], training)
        x = mx.maximum(x, 0.0)
        x = mx.matmul(x, blk["w2"].transpose(0, 2, 1))
        x, s2 = _ens_bn(x, blk["bn2_w"], blk["bn2_b"], bn["blocks"][i]["bn2"], training)
        x = mx.maximum(x, 0.0)
        x = x + residual
        if training:
            stats["blocks"].append({"bn1": s1, "bn2": s2})
    x = _ens_rms_norm(x, p["post_norm_w"])
    logits = mx.matmul(x, p["logit_w"].transpose(0, 2, 1)) + p["logit_b"][:, None, :]
    log_probs = _nn_mlx.log_softmax(logits, axis=-1)
    values = (mx.exp(log_probs) * support).sum(-1)
    return values, log_probs, stats if training else None


def _critic_forward_q(p, bn, q: int, obs, actions, training: bool, support):
    """Per-q critic forward on plain (B, ·) tensors (bench variant)."""
    mx = _mx
    pq = _tree_map(lambda a: a[q], p)
    bq = _tree_map(lambda a: a[q], bn)
    x = mx.concatenate([obs, actions], axis=-1)
    stats: dict[str, Any] = {"embedder": None, "blocks": []}
    x, s = _unit_bn(x, pq["embedder"]["bn_w"], pq["embedder"]["bn_b"], bq["embedder"], training)
    if training:
        stats["embedder"] = s
    x = x @ pq["embedder"]["w"].T
    for i, blk in enumerate(pq["blocks"]):
        residual = x
        x = x @ blk["w1"].T
        x, s1 = _unit_bn(x, blk["bn1_w"], blk["bn1_b"], bq["blocks"][i]["bn1"], training)
        x = mx.maximum(x, 0.0)
        x = x @ blk["w2"].T
        x, s2 = _unit_bn(x, blk["bn2_w"], blk["bn2_b"], bq["blocks"][i]["bn2"], training)
        x = mx.maximum(x, 0.0)
        x = x + residual
        if training:
            stats["blocks"].append({"bn1": s1, "bn2": s2})
    x = _rms_norm(x, pq["post_norm_w"])
    logits = x @ pq["logit_w"].T + pq["logit_b"]
    log_probs = _nn_mlx.log_softmax(logits, axis=-1)
    values = (mx.exp(log_probs) * support).sum(-1)
    return values, log_probs, stats if training else None


def _critic_forward_dispatch(ensemble: bool, p, bn, obs, actions, training: bool, support):
    if ensemble:
        return _critic_forward_ens(p, bn, obs, actions, training, support)
    v0, l0, s0 = _critic_forward_q(p, bn, 0, obs, actions, training, support)
    v1, l1, s1 = _critic_forward_q(p, bn, 1, obs, actions, training, support)
    mx = _mx
    values = mx.stack([v0, v1])
    log_probs = mx.stack([l0, l1])
    stats = _stack2(s0, s1) if training else None
    return values, log_probs, stats


# ---------------------------------------------------------------------------
# normalize_parameters (post-optimizer-step weight normalization)
# ---------------------------------------------------------------------------


def _norm_rows(w):
    """F.normalize(w, dim=-1, eps=1e-8) for (..., out, in)."""
    mx = _mx
    n = mx.sqrt(mx.sum(w * w, axis=-1, keepdims=True))
    return w / mx.maximum(n, 1e-8)


def _norm_bn_affine(w, b):
    mx = _mx
    d = w.shape[-1]
    f = math.sqrt(d) * mx.rsqrt(mx.sum(w * w + b * b, axis=-1, keepdims=True) + 1e-8)
    return w * f, b * f


def _norm_rms_w(w):
    mx = _mx
    d = w.shape[-1]
    return w * (math.sqrt(d) * mx.rsqrt(mx.sum(w * w, axis=-1, keepdims=True) + 1e-8))


def _normalize_actor_tree(p):
    out = {
        "embedder": {"w": _norm_rows(p["embedder"]["w"])},
        "blocks": [],
        "post_norm_w": _norm_rms_w(p["post_norm_w"]),
        "mean_w": _norm_rows(p["mean_w"]),
        "mean_b": p["mean_b"],
        "std_w": _norm_rows(p["std_w"]),
        "std_b": p["std_b"],
    }
    bw, bb = _norm_bn_affine(p["embedder"]["bn_w"], p["embedder"]["bn_b"])
    out["embedder"]["bn_w"], out["embedder"]["bn_b"] = bw, bb
    for blk in p["blocks"]:
        nb = {"w1": _norm_rows(blk["w1"]), "w2": _norm_rows(blk["w2"])}
        nb["bn1_w"], nb["bn1_b"] = _norm_bn_affine(blk["bn1_w"], blk["bn1_b"])
        nb["bn2_w"], nb["bn2_b"] = _norm_bn_affine(blk["bn2_w"], blk["bn2_b"])
        out["blocks"].append(nb)
    return out


def _normalize_critic_tree(p):
    out = {
        "embedder": {"w": _norm_rows(p["embedder"]["w"])},
        "blocks": [],
        "post_norm_w": _norm_rms_w(p["post_norm_w"]),
        "logit_w": _norm_rows(p["logit_w"]),
        "logit_b": p["logit_b"],
    }
    bw, bb = _norm_bn_affine(p["embedder"]["bn_w"], p["embedder"]["bn_b"])
    out["embedder"]["bn_w"], out["embedder"]["bn_b"] = bw, bb
    for blk in p["blocks"]:
        nb = {"w1": _norm_rows(blk["w1"]), "w2": _norm_rows(blk["w2"])}
        nb["bn1_w"], nb["bn1_b"] = _norm_bn_affine(blk["bn1_w"], blk["bn1_b"])
        nb["bn2_w"], nb["bn2_b"] = _norm_bn_affine(blk["bn2_w"], blk["bn2_b"])
        out["blocks"].append(nb)
    return out


# ---------------------------------------------------------------------------
# Functional Adam (torch semantics: bias correction, betas (0.9, 0.999), wd=0)
# ---------------------------------------------------------------------------


def _adam_update(p, g, m, v, t, lr, gate, betas=(0.9, 0.999), eps=1e-8):
    """Adam step with a scalar ``gate`` (0/1) selecting skip vs. apply.

    Sanitizes non-finite gradients to zero before any arithmetic: a plain
    ``gate * x`` would still yield NaN when x is NaN (0 * NaN = NaN).  When
    gated off, parameters, moments and the step counter are left untouched.
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
        lambda p_, m_, v_: _sel(p_ - lr * (m_ / bc1) / (mx.sqrt(v_ / bc2) + eps), p_),
        p,
        m,
        v,
    )
    return new_p, m, v, t_new


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class MLXFlashSACLearner:
    """FlashSAC learner that trains with MLX and serves inference via torch.

    Public interface mirrors ``FlashSACLearner`` as consumed by
    ``DoubleBufferOffPolicyRunner``.
    """

    supports_deferred_update_metrics = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        actor_num_blocks: int = 2,
        critic_num_blocks: int = 2,
        num_atoms: int = 101,
        critic_min_v: float = -5.0,
        critic_max_v: float = 5.0,
        temp_initial_value: float = 0.01,
        temp_target_sigma: float = 0.15,
        temp_target_entropy: float | None = None,
        actor_bc_alpha: float = 0.0,
        actor_noise_zeta_mu: float = 2.0,
        actor_noise_zeta_max: int = 16,
        learning_rate_init: float = 3e-4,
        learning_rate_peak: float = 3e-4,
        learning_rate_end: float = 1.5e-4,
        learning_rate_warmup_steps: int = 0,
        learning_rate_decay_steps: int = 500000,
        normalize_reward: bool = True,
        normalized_g_max: float = 5.0,
        n_step: int = 1,
        obs_normalization: bool = False,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        use_compile: bool = True,
        compile_full_objectives: bool = False,
        use_cuda_graph_critic: bool = False,
        use_cuda_graph_actor: bool = False,
        use_cuda_graph_critic_packed_staging: bool = False,
        use_cuda_graph_actor_packed_staging: bool = False,
    ):
        mx = _mlx()
        if any(
            (
                use_cuda_graph_critic,
                use_cuda_graph_actor,
                use_cuda_graph_critic_packed_staging,
                use_cuda_graph_actor_packed_staging,
            )
        ):
            raise ValueError("CUDA graphs are CUDA-only; MLXFlashSACLearner does not use them")
        device_type = torch.device(device).type
        if device_type not in ("mps", "cpu"):
            raise ValueError(f"MLXFlashSACLearner only supports mps/cpu devices, got {device!r}")
        if n_step != 1:
            raise ValueError("MLXFlashSACLearner spike only implements n_step=1")
        if actor_bc_alpha > 0:
            raise ValueError("MLXFlashSACLearner spike only implements actor_bc_alpha=0")

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.use_compile = bool(use_compile)
        self.use_amp = False
        # Finite gating happens in-graph (the compiled step zeroes the update
        # on non-finite loss/grads), so no host checks are needed.
        self._host_finite_checks = False
        self.use_cuda_graph_critic = False
        self.use_cuda_graph_actor = False
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.critic_min_v = critic_min_v
        self.critic_max_v = critic_max_v
        self.actor_num_blocks = actor_num_blocks
        self.critic_num_blocks = critic_num_blocks
        self.log_std_min = -10.0
        self.log_std_max = 2.0
        # Ensemble-batched twin critic (torch einsum layout) vs per-q loop;
        # bench selects the faster variant.  Baked into compiled traces.
        self._ensemble_critic = True

        lr_peak = learning_rate_peak if learning_rate_peak > 0 else actor_lr
        self._lr_peak = float(lr_peak)
        self._lr_init_ratio = learning_rate_init / lr_peak if lr_peak > 0 else 1.0
        self._lr_end_ratio = learning_rate_end / lr_peak if lr_peak > 0 else 1.0
        self._lr_warmup = max(int(learning_rate_warmup_steps), 0)
        self._lr_decay = max(int(learning_rate_decay_steps), self._lr_warmup + 1)

        self.target_entropy = resolve_target_entropy(
            action_dim=action_dim,
            target_sigma=temp_target_sigma,
            target_entropy=temp_target_entropy,
        )

        # Torch inference/export actor (kept in sync from the MLX weights).
        self.actor = FlashSACActor(
            num_blocks=actor_num_blocks,
            input_dim=obs_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
            noise_zeta_mu=actor_noise_zeta_mu,
            noise_zeta_max=actor_noise_zeta_max,
            device=self.device,
        )

        self.obs_normalizer: EmpiricalNormalization | nn.Identity
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=self.device)
        else:
            self.obs_normalizer = nn.Identity()

        self.reward_normalizer = (
            RewardNormalizer(
                gamma=self.gamma,
                g_max=normalized_g_max,
                device=torch.device(self.device),
            )
            if normalize_reward
            else None
        )

        # MLX parameter/buffer trees.  The actor tree is seeded from the torch
        # actor so both copies start from identical weights.
        self._actor_p = _actor_params(obs_dim, action_dim, actor_hidden_dim, actor_num_blocks)
        self._actor_bn = _actor_bn_buffers(obs_dim, actor_hidden_dim, actor_num_blocks)
        self._critic_p = _critic_params(
            critic_obs_dim + action_dim, critic_hidden_dim, critic_num_blocks, num_atoms
        )
        self._critic_bn = _critic_bn_buffers(
            critic_obs_dim + action_dim, critic_hidden_dim, critic_num_blocks
        )
        self._load_torch_actor_into_mlx()
        self._critic_target_p = _tree_map(lambda x: x + 0.0, self._critic_p)
        self._critic_target_bn = _tree_map(lambda x: x + 0.0, self._critic_bn)
        self._log_temp = {"a": mx.array([math.log(temp_initial_value)])}
        mx.eval(
            self._actor_p,
            self._actor_bn,
            self._critic_p,
            self._critic_bn,
            self._critic_target_p,
            self._critic_target_bn,
        )

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
        self._temp_opt = {
            "m": _tree_zeros(self._log_temp),
            "v": _tree_zeros(self._log_temp),
            "t": mx.array(0.0),
        }

        self._q_support = mx.linspace(critic_min_v, critic_max_v, num_atoms)
        self._bin_width = max((critic_max_v - critic_min_v) / (num_atoms - 1), 1e-8)
        self._gamma_n = float(gamma**n_step)

        self.update_count = 0
        self._pending_actor_metric_values: tuple | None = None

        self._state = [
            self._critic_p,
            self._critic_bn,
            self._actor_p,
            self._actor_bn,
            self._critic_target_p,
            self._critic_target_bn,
            self._log_temp,
            self._critic_opt,
            self._actor_opt,
            self._temp_opt,
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
        """Initialise MLX actor params/BN buffers from the torch FlashSACActor."""
        mx = _mx
        sd = self.actor.state_dict()
        p, bn = self._actor_p, self._actor_bn

        def arr(key: str):
            return mx.array(np.asarray(sd[key].cpu()))

        p["embedder"]["w"] = arr("embedder.w.w.weight")
        p["embedder"]["bn_w"] = arr("embedder.norm.weight")
        p["embedder"]["bn_b"] = arr("embedder.norm.bias")
        bn["embedder"]["mean"] = arr("embedder.norm.running_mean")
        bn["embedder"]["var"] = arr("embedder.norm.running_var")
        for i, blk in enumerate(p["blocks"]):
            blk["w1"] = arr(f"encoder.{i}.w1.w.weight")
            blk["bn1_w"] = arr(f"encoder.{i}.norm1.weight")
            blk["bn1_b"] = arr(f"encoder.{i}.norm1.bias")
            bn["blocks"][i]["bn1"]["mean"] = arr(f"encoder.{i}.norm1.running_mean")
            bn["blocks"][i]["bn1"]["var"] = arr(f"encoder.{i}.norm1.running_var")
            blk["w2"] = arr(f"encoder.{i}.w2.w.weight")
            blk["bn2_w"] = arr(f"encoder.{i}.norm2.weight")
            blk["bn2_b"] = arr(f"encoder.{i}.norm2.bias")
            bn["blocks"][i]["bn2"]["mean"] = arr(f"encoder.{i}.norm2.running_mean")
            bn["blocks"][i]["bn2"]["var"] = arr(f"encoder.{i}.norm2.running_var")
        p["post_norm_w"] = arr("post_norm.weight")
        p["mean_w"] = arr("predictor.mean_w.w.weight")
        p["mean_b"] = arr("predictor.mean_bias")
        p["std_w"] = arr("predictor.std_w.w.weight")
        p["std_b"] = arr("predictor.std_bias")

    def _sync_torch_actor_from_mlx(self) -> None:
        """Copy MLX actor weights and BN running stats into the torch actor."""
        p, bn = self._actor_p, self._actor_bn

        def t(x):
            return torch.from_numpy(np.asarray(x))

        sd: dict[str, torch.Tensor] = {
            "embedder.w.w.weight": t(p["embedder"]["w"]),
            "embedder.norm.weight": t(p["embedder"]["bn_w"]),
            "embedder.norm.bias": t(p["embedder"]["bn_b"]),
            "embedder.norm.running_mean": t(bn["embedder"]["mean"]),
            "embedder.norm.running_var": t(bn["embedder"]["var"]),
            "post_norm.weight": t(p["post_norm_w"]),
            "predictor.mean_w.w.weight": t(p["mean_w"]),
            "predictor.mean_bias": t(p["mean_b"]),
            "predictor.std_w.w.weight": t(p["std_w"]),
            "predictor.std_bias": t(p["std_b"]),
        }
        for i, blk in enumerate(p["blocks"]):
            sd[f"encoder.{i}.w1.w.weight"] = t(blk["w1"])
            sd[f"encoder.{i}.norm1.weight"] = t(blk["bn1_w"])
            sd[f"encoder.{i}.norm1.bias"] = t(blk["bn1_b"])
            sd[f"encoder.{i}.norm1.running_mean"] = t(bn["blocks"][i]["bn1"]["mean"])
            sd[f"encoder.{i}.norm1.running_var"] = t(bn["blocks"][i]["bn1"]["var"])
            sd[f"encoder.{i}.w2.w.weight"] = t(blk["w2"])
            sd[f"encoder.{i}.norm2.weight"] = t(blk["bn2_w"])
            sd[f"encoder.{i}.norm2.bias"] = t(blk["bn2_b"])
            sd[f"encoder.{i}.norm2.running_mean"] = t(bn["blocks"][i]["bn2"]["mean"])
            sd[f"encoder.{i}.norm2.running_var"] = t(bn["blocks"][i]["bn2"]["var"])
        # zeta_cdf and the non-persistent exploration buffers are static.
        self.actor.load_state_dict(sd, strict=False)
        self.actor.to(self.device)

    def _critic_state_dict_torch(self, params, bn, num_blocks: int) -> dict[str, torch.Tensor]:
        """Key MLX critic trees like FlashSACDoubleCritic's state_dict."""
        sd: dict[str, torch.Tensor] = {}

        def t(x):
            return torch.from_numpy(np.asarray(x))

        sd["embedder.w.weight"] = t(params["embedder"]["w"])
        sd["embedder.norm.weight"] = t(params["embedder"]["bn_w"])
        sd["embedder.norm.bias"] = t(params["embedder"]["bn_b"])
        sd["embedder.norm.running_mean"] = t(bn["embedder"]["mean"])
        sd["embedder.norm.running_var"] = t(bn["embedder"]["var"])
        for i, blk in enumerate(params["blocks"]):
            sd[f"encoder.{i}.w1.weight"] = t(blk["w1"])
            sd[f"encoder.{i}.norm1.weight"] = t(blk["bn1_w"])
            sd[f"encoder.{i}.norm1.bias"] = t(blk["bn1_b"])
            sd[f"encoder.{i}.norm1.running_mean"] = t(bn["blocks"][i]["bn1"]["mean"])
            sd[f"encoder.{i}.norm1.running_var"] = t(bn["blocks"][i]["bn1"]["var"])
            sd[f"encoder.{i}.w2.weight"] = t(blk["w2"])
            sd[f"encoder.{i}.norm2.weight"] = t(blk["bn2_w"])
            sd[f"encoder.{i}.norm2.bias"] = t(blk["bn2_b"])
            sd[f"encoder.{i}.norm2.running_mean"] = t(bn["blocks"][i]["bn2"]["mean"])
            sd[f"encoder.{i}.norm2.running_var"] = t(bn["blocks"][i]["bn2"]["var"])
        sd["post_norm.weight"] = t(params["post_norm_w"])
        sd["predictor.logit_w.weight"] = t(params["logit_w"])
        sd["predictor.logit_bias"] = t(params["logit_b"])
        sd["predictor.support"] = torch.linspace(
            self.critic_min_v, self.critic_max_v, self.num_atoms
        )
        return sd

    # ------------------------------------------------------------------
    # LR schedule (in-graph, driven by each optimizer's own step counter)
    # ------------------------------------------------------------------

    def _lr_factor(self, t):
        """build_lr_lambda semantics; t is the pre-update optimizer step."""
        mx = _mx
        progress = mx.clip(
            (t - self._lr_warmup) / float(self._lr_decay - self._lr_warmup), 0.0, 1.0
        )
        cosine = self._lr_end_ratio + (1.0 - self._lr_end_ratio) * 0.5 * (
            1.0 + mx.cos(math.pi * progress)
        )
        if self._lr_warmup > 0:
            warm = self._lr_init_ratio + (1.0 - self._lr_init_ratio) * (t / float(self._lr_warmup))
            return mx.where(t < self._lr_warmup, warm, cosine)
        return cosine

    # ------------------------------------------------------------------
    # Losses (functional; eps is injectable for the parity test)
    # ------------------------------------------------------------------

    def _td_target(self, next_q_log_probs, rewards, dones, truncated, actor_entropy):
        """Mirror update.compute_categorical_td_target (floor/ceil scatter_add)."""
        mx = _mx
        batch_size, num_bins = next_q_log_probs.shape
        bootstrap = mx.clip(1.0 - dones + truncated, 0.0, 1.0)[:, None]
        target = rewards[:, None] + bootstrap * self._gamma_n * (
            self._q_support[None, :] - actor_entropy[:, None]
        )
        target = mx.clip(target, self.critic_min_v, self.critic_max_v)
        offsets = (target - self.critic_min_v) / self._bin_width
        lower = mx.clip(mx.floor(offsets), 0, num_bins - 1).astype(mx.int32)
        upper = mx.clip(mx.ceil(offsets), 0, num_bins - 1).astype(mx.int32)
        frac = offsets - lower.astype(mx.float32)
        probs = mx.exp(next_q_log_probs)

        row_offsets = mx.arange(batch_size)[:, None] * num_bins
        flat = mx.zeros((batch_size * num_bins,))
        flat = flat.at[(lower + row_offsets).reshape(-1)].add((probs * (1.0 - frac)).reshape(-1))
        flat = flat.at[(upper + row_offsets).reshape(-1)].add((probs * frac).reshape(-1))
        return flat.reshape(batch_size, num_bins)

    def _critic_loss_mx(self, critic_p, batch, eps=None):
        """Mirror FlashSACLearner._critic_objective_tensors.

        Returns (critic_loss, (target_bn_stats, online_bn_stats)).
        """
        mx = _mx
        obs, critic_obs, actions, rewards, next_obs, next_critic, dones, truncated = batch
        del obs  # actor-side obs are unused by the critic objective
        batch_size = rewards.shape[0]

        next_actions, next_log_prob, _, _, _ = _actor_forward(
            self._actor_p,
            self._actor_bn,
            next_obs,
            False,
            self.log_std_min,
            self.log_std_max,
            eps=eps,
        )
        actor_entropy = mx.exp(mx.stop_gradient(self._log_temp["a"])) * next_log_prob
        obs_all = mx.concatenate([critic_obs, next_critic], axis=0)
        act_all = mx.concatenate([actions, next_actions], axis=0)

        tq_values, tq_log_probs, stats_target = _critic_forward_dispatch(
            self._ensemble_critic,
            self._critic_target_p,
            self._critic_target_bn,
            obs_all,
            act_all,
            True,
            self._q_support,
        )
        next_q_values = tq_values[:, batch_size:]
        next_q_log_probs_full = tq_log_probs[:, batch_size:, :]

        _, pred_log_probs_all, stats_online = _critic_forward_dispatch(
            self._ensemble_critic,
            critic_p,
            self._critic_bn,
            obs_all,
            act_all,
            True,
            self._q_support,
        )
        pred_log_probs = pred_log_probs_all[:, :batch_size, :]

        min_indices = mx.argmin(next_q_values, axis=0)
        next_q_log_probs = next_q_log_probs_full[min_indices, mx.arange(batch_size)]
        target_probs = self._td_target(next_q_log_probs, rewards, dones, truncated, actor_entropy)
        critic_loss = -(target_probs[None, :, :] * pred_log_probs).sum(-1).mean()
        return critic_loss, (stats_target, stats_online)

    def _actor_loss_mx(self, actor_p, batch, eps=None):
        """Mirror FlashSACLearner._actor_objective_tensors (critic frozen).

        Returns (actor_loss, (entropy, actor_bn_stats)).
        """
        mx = _mx
        obs, next_obs, critic_obs = batch
        batch_size = critic_obs.shape[0]
        obs_all = mx.concatenate([obs, next_obs], axis=0)
        actions_all, log_probs_all, _, _, stats_actor = _actor_forward(
            actor_p, self._actor_bn, obs_all, True, self.log_std_min, self.log_std_max, eps=eps
        )
        actions = actions_all[:batch_size]
        log_probs = log_probs_all[:batch_size]

        q_values, _, _ = _critic_forward_dispatch(
            self._ensemble_critic,
            self._critic_p,
            self._critic_bn,
            critic_obs,
            actions,
            False,
            self._q_support,
        )
        min_q = mx.minimum(q_values[0], q_values[1])
        temp = mx.exp(mx.stop_gradient(self._log_temp["a"]))
        actor_loss = (temp * log_probs - min_q).mean()
        entropy = -mx.stop_gradient(log_probs).mean()
        return actor_loss, (entropy, stats_actor)

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

    @staticmethod
    def _apply_bn_stats(bn_tree, stats_list, batch_count: int, gate) -> None:
        """lerp BN running stats toward batch stats (gated, sanitized)."""
        mx = _mx
        correction = batch_count / max(batch_count - 1, 1)

        def leaf(buf, stats):
            mean = mx.where(mx.isfinite(stats["mean"]), stats["mean"], mx.zeros_like(stats["mean"]))
            var = mx.where(mx.isfinite(stats["var"]), stats["var"], mx.zeros_like(stats["var"]))
            new_mean = buf["mean"] * (1.0 - _BN_MOMENTUM) + mean * _BN_MOMENTUM
            new_var = buf["var"] * (1.0 - _BN_MOMENTUM) + var * correction * _BN_MOMENTUM
            return {
                "mean": mx.where(gate > 0, new_mean, buf["mean"]),
                "var": mx.where(gate > 0, new_var, buf["var"]),
            }

        if isinstance(stats_list, dict):
            # Critic stats tree mirrors the buffer tree.
            def rec(b, s):
                if "mean" in b:
                    return leaf(b, s)
                if isinstance(b, dict):
                    return {k: rec(b[k], s[k]) for k in b}
                return [rec(bb, ss) for bb, ss in zip(b, s)]

            _tree_assign(bn_tree, rec(bn_tree, stats_list))
            return
        # Actor stats: flat list of (path, stats) pairs.
        lookup = {"embedder": bn_tree["embedder"]}
        for i, blk in enumerate(bn_tree["blocks"]):
            lookup[f"blocks.{i}.bn1"] = blk["bn1"]
            lookup[f"blocks.{i}.bn2"] = blk["bn2"]
        for path, stats in stats_list:
            buf = lookup[path]
            _tree_assign(buf, leaf(buf, stats))

    def _critic_step_impl(self, batch):
        mx = _mx

        def loss_fn(critic_p):
            loss, aux = self._critic_loss_mx(critic_p, batch)
            return loss, aux

        (critic_loss, aux), grads = mx.value_and_grad(loss_fn)(self._critic_p)
        stats_target, stats_online = aux

        gate = self._finite_gate(critic_loss, grads)
        lr = self._lr_peak * self._lr_factor(self._critic_opt["t"])
        new_p, m, v, t = _adam_update(
            self._critic_p,
            grads,
            self._critic_opt["m"],
            self._critic_opt["v"],
            self._critic_opt["t"],
            lr,
            gate,
        )
        # normalize_parameters runs unconditionally, as in the torch learner
        # (idempotent on already-normalized rows when the update is gated off).
        new_p = _normalize_critic_tree(new_p)
        _tree_assign(self._critic_p, new_p)
        _tree_assign(self._critic_opt["m"], m)
        _tree_assign(self._critic_opt["v"], v)
        self._critic_opt["t"] = t

        batch_count = int(batch[1].shape[0]) * 2  # obs_all = cat(critic, next_critic)
        self._apply_bn_stats(self._critic_bn, stats_online, batch_count, gate)
        self._apply_bn_stats(self._critic_target_bn, stats_target, batch_count, gate)

        return (critic_loss.reshape(()), gate.reshape(()))

    def _actor_step_impl(self, batch):
        mx = _mx

        def loss_fn(actor_p):
            loss, aux = self._actor_loss_mx(actor_p, batch)
            return loss, aux

        (actor_loss, aux), grads = mx.value_and_grad(loss_fn)(self._actor_p)
        entropy, stats_actor = aux

        gate = self._finite_gate(actor_loss, grads)
        lr = self._lr_peak * self._lr_factor(self._actor_opt["t"])
        new_p, m, v, t = _adam_update(
            self._actor_p,
            grads,
            self._actor_opt["m"],
            self._actor_opt["v"],
            self._actor_opt["t"],
            lr,
            gate,
        )
        new_p = _normalize_actor_tree(new_p)
        _tree_assign(self._actor_p, new_p)
        _tree_assign(self._actor_opt["m"], m)
        _tree_assign(self._actor_opt["v"], v)
        self._actor_opt["t"] = t

        batch_count = int(batch[0].shape[0]) * 2  # obs_all = cat(obs, next_obs)
        self._apply_bn_stats(self._actor_bn, stats_actor, batch_count, gate)

        # Temperature update: temp_loss = temp * (entropy - target_entropy),
        # entropy detached; metric temperature is the pre-update value.
        temp_value = mx.exp(self._log_temp["a"])

        def temp_loss_fn(log_temp):
            return (mx.exp(log_temp["a"]) * (entropy - self.target_entropy)).reshape(())

        temp_loss, temp_grads = mx.value_and_grad(temp_loss_fn)(self._log_temp)
        gate_t = self._finite_gate(temp_loss, temp_grads)
        lr_t = self._lr_peak * self._lr_factor(self._temp_opt["t"])
        new_a, m, v, t = _adam_update(
            self._log_temp,
            temp_grads,
            self._temp_opt["m"],
            self._temp_opt["v"],
            self._temp_opt["t"],
            lr_t,
            gate_t,
        )
        _tree_assign(self._log_temp, new_a)
        _tree_assign(self._temp_opt["m"], m)
        _tree_assign(self._temp_opt["v"], v)
        self._temp_opt["t"] = t

        return (
            actor_loss.reshape(()),
            entropy.reshape(()),
            temp_value.reshape(()),
            temp_loss.reshape(()),
        )

    def _target_step_impl(self):
        """Polyak-average target parameters (torch polyak skips BN buffers)."""
        new_target = _tree_map(
            lambda tp, p: (1 - self.tau) * tp + self.tau * p,
            self._critic_target_p,
            self._critic_p,
        )
        _tree_assign(self._critic_target_p, new_target)
        return self._critic_target_p["logit_w"].sum().reshape(())

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

    def update_reward_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        if self.reward_normalizer is None:
            return
        self.reward_normalizer.update_from_transitions(rewards, dones)

    def update_critic(self, batch: Dict[str, torch.Tensor], *, read_metrics: bool = True):
        obs = self.normalize_obs(batch["obs"].to(self.device), update=True)
        next_obs = self.normalize_obs(batch["next_obs"].to(self.device), update=False)
        rewards = batch["rewards"].to(self.device)
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.normalize(rewards)
        mx_batch = (
            self._t2m(obs),
            self._t2m(batch["critic"].to(self.device)),
            self._t2m(batch["actions"].to(self.device)),
            self._t2m(rewards),
            self._t2m(next_obs),
            self._t2m(batch["next_critic"].to(self.device)),
            self._t2m(batch["dones"].to(self.device)),
            self._t2m(batch["truncated"].to(self.device)),
        )
        out = self._critic_step(mx_batch)
        if not read_metrics:
            return {}
        mx = _mx
        mx.eval(out)
        critic_loss = out[0].item()
        reward_scale_std = (
            float(torch.sqrt(self.reward_normalizer.rms.var).item())
            if self.reward_normalizer is not None
            else 1.0
        )
        return {
            "critic_loss": critic_loss,
            "reward_scale_std": reward_scale_std,
        }

    def update_actor(self, batch: Dict[str, torch.Tensor], *, read_metrics: bool = True):
        obs = self.normalize_obs(batch["obs"].to(self.device), update=False)
        next_obs = self.normalize_obs(batch["next_obs"].to(self.device), update=False)
        mx_batch = (
            self._t2m(obs),
            self._t2m(next_obs),
            self._t2m(batch["critic"].to(self.device)),
        )
        out = self._actor_step(mx_batch)
        self._sync_torch_actor_from_mlx()
        if not read_metrics:
            self._pending_actor_metric_values = out
            return {}
        mx = _mx
        mx.eval(out)
        actor_loss, entropy, temp_value, temp_loss = (v.item() for v in out)
        return {
            "actor_loss": actor_loss,
            "actor_entropy": entropy,
            "temperature": temp_value,
            "temperature_loss": temp_loss,
        }

    def read_deferred_actor_metrics(self) -> Dict[str, float]:
        values = self._pending_actor_metric_values
        self._pending_actor_metric_values = None
        if values is None:
            return {}
        mx = _mx
        mx.eval(values)
        actor_loss, entropy, temp_value, temp_loss = (v.item() for v in values)
        return {
            "actor_loss": actor_loss,
            "actor_entropy": entropy,
            "temperature": temp_value,
            "temperature_loss": temp_loss,
        }

    def soft_update_target(self) -> None:
        self._target_step()

    def set_gradient_sync(self, sync) -> None:
        raise NotImplementedError("MLXFlashSACLearner does not support multi-GPU DP")

    def dp_initial_sync_tensors(self):
        raise NotImplementedError("MLXFlashSACLearner does not support multi-GPU DP")

    def get_state_dict(self) -> Dict[str, Any]:
        mx = _mx
        mx.eval(
            self._critic_p,
            self._critic_bn,
            self._critic_target_p,
            self._critic_target_bn,
            self._actor_p,
            self._actor_bn,
            self._log_temp,
        )
        return {
            "actor": self.actor.state_dict(),
            "critic": self._critic_state_dict_torch(
                self._critic_p, self._critic_bn, self.critic_num_blocks
            ),
            "target_critic": self._critic_state_dict_torch(
                self._critic_target_p, self._critic_target_bn, self.critic_num_blocks
            ),
            "temperature": {"log_temp": torch.tensor([self._log_temp["a"].item()])},
            "mlx_optimizer_state": {
                "critic": _tree_map(lambda x: np.asarray(x), self._critic_opt),
                "actor": _tree_map(lambda x: np.asarray(x), self._actor_opt),
                "temperature": _tree_map(lambda x: np.asarray(x), self._temp_opt),
            },
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict")
                else None
            ),
            "reward_normalizer": (
                self.reward_normalizer.state_dict() if self.reward_normalizer is not None else None
            ),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        raise NotImplementedError("MLXFlashSACLearner spike does not support resume")


def resolve_mlx_flashsac_runtime(rl_cfg: dict[str, Any]) -> OffPolicyRuntime:
    """Owner-config resolver: select the MLX FlashSAC learner."""
    return OffPolicyRuntime(learner_cls=MLXFlashSACLearner, algo_type="flashsac")
