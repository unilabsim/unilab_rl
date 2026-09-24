"""Multi-node data-parallel topology tests (external launch + TCP rendezvous)."""

from __future__ import annotations

import multiprocessing as mp
import socket
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from uni_rl.ipc.dp_launcher import (
    UNILAB_DP_EXTERNAL,
    UNILAB_DP_RANK,
    UNILAB_DP_RENDEZVOUS_URL,
    UNILAB_DP_WORLD_SIZE,
    ExternalDpTopology,
    apply_dp_rank_config,
    current_external_dp_topology,
    resolve_collector_cpu_ids,
)
from uni_rl.ipc.dp_sync import DpParameterSync

_SPAWN_CTX = mp.get_context("spawn")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_external_topology_reads_valid_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_EXTERNAL, "1")
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, "2")
    monkeypatch.setenv(UNILAB_DP_RANK, "1")
    monkeypatch.setenv(UNILAB_DP_RENDEZVOUS_URL, "tcp://192.168.100.1:29501")

    assert current_external_dp_topology() == ExternalDpTopology(
        world_size=2,
        rank=1,
        rendezvous_url="tcp://192.168.100.1:29501",
    )


def test_external_topology_defaults_off(monkeypatch: pytest.MonkeyPatch):
    for name in (
        UNILAB_DP_EXTERNAL,
        UNILAB_DP_WORLD_SIZE,
        UNILAB_DP_RANK,
        UNILAB_DP_RENDEZVOUS_URL,
    ):
        monkeypatch.delenv(name, raising=False)

    assert current_external_dp_topology() is None


@pytest.mark.parametrize(
    ("world_size", "rank", "url"),
    [
        ("1", "0", "tcp://host:1"),
        ("2", "2", "tcp://host:1"),
        ("2", "0", ""),
        ("2", "0", "file:///tmp/x"),
    ],
)
def test_external_topology_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, world_size: str, rank: str, url: str
):
    monkeypatch.setenv(UNILAB_DP_EXTERNAL, "1")
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, world_size)
    monkeypatch.setenv(UNILAB_DP_RANK, rank)
    monkeypatch.setenv(UNILAB_DP_RENDEZVOUS_URL, url)

    with pytest.raises(ValueError):
        current_external_dp_topology()


def test_non_colocated_ranks_keep_the_full_host_cpu_budget():
    assert resolve_collector_cpu_ids(2, 1, cpu_count=20, colocated=False) is None


def test_apply_dp_rank_config_maps_global_seed_to_local_device():
    cfg = OmegaConf.create({"algo": {"seed": 7}})

    device = apply_dp_rank_config(cfg, (0,), 1, device_rank=0)

    assert device == "cuda:0"
    assert cfg.algo.seed == 8


def test_dp_sync_rejects_ambiguous_rendezvous(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        DpParameterSync(world_size=2, rank=0)
    with pytest.raises(ValueError, match="exactly one"):
        DpParameterSync(
            world_size=2,
            rank=0,
            rendezvous_path=str(tmp_path / "rendezvous"),
            rendezvous_url="tcp://127.0.0.1:29501",
        )
    with pytest.raises(ValueError, match="tcp://"):
        DpParameterSync(world_size=2, rank=0, rendezvous_url="file:///tmp/x")


def _tcp_broadcast_gradient_worker(rank: int, rendezvous_url: str, result_queue) -> None:
    sync = DpParameterSync(
        world_size=2,
        rank=rank,
        rendezvous_url=rendezvous_url,
        backend="gloo",
        timeout_s=60,
    )
    sync.start()
    tensors = {
        "actor.weight": torch.full((4, 3), float(1 + rank), requires_grad=True),
        "critic.bias": torch.full((2,), float(10 + rank), requires_grad=True),
    }
    sync.broadcast_from_rank0(tensors)
    for tensor in tensors.values():
        tensor.grad = torch.full_like(tensor, float(rank + 1))
    sync.allreduce_gradients(tensors.values())
    # Mean of per-rank grads (1 and 2) must land on both ranks bitwise.
    expected = {key: torch.full_like(value, 1.5) for key, value in tensors.items()}
    result = all(torch.equal(value.grad, expected[key]) for key, value in tensors.items())
    sync.close()
    # Plain lists avoid torch's shared-memory queue reduction, which can
    # wedge when a spawned worker exits while its storage handles are in flight.
    result_queue.put((rank, result, {key: value.tolist() for key, value in tensors.items()}))


def test_broadcast_and_gradient_mean_over_tcp_rendezvous():
    rendezvous_url = f"tcp://127.0.0.1:{_free_port()}"
    result_queue = _SPAWN_CTX.Queue()
    procs = [
        _SPAWN_CTX.Process(
            target=_tcp_broadcast_gradient_worker,
            args=(rank, rendezvous_url, result_queue),
        )
        for rank in range(2)
    ]
    try:
        for proc in procs:
            proc.start()
        results = sorted(result_queue.get(timeout=120) for _ in range(2))
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
    assert [rank for rank, _ok, _tensors in results] == [0, 1]
    assert all(ok for _rank, ok, _tensors in results)
    # Broadcast rank-0 values, so post-broadcast parameters agree bitwise too.
    tensors0, tensors1 = results[0][2], results[1][2]
    assert tensors0.keys() == tensors1.keys()
    for key, value in tensors0.items():
        assert value == tensors1[key]
