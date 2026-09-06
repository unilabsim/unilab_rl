from __future__ import annotations

from typing import Any

from uni_rl.ipc import dp_launcher


class _FakePopen:
    calls: list["_FakePopen"] = []

    def __init__(self, command: list[str], *, env: dict[str, str], start_new_session: bool) -> None:
        self.command = command
        self.env = env
        self.start_new_session = start_new_session
        self.pid = 123456
        self.returncode = 0
        type(self).calls.append(self)

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode


def test_dp_rank_supervisor_reuses_downstream_entry_script(monkeypatch: Any) -> None:
    """Spawned ranks must re-run the owner application's script.

    ``uni_rl`` deliberately does not ship a ``scripts/`` package; the entry
    point is supplied by the downstream consumer such as UniLab.
    """
    _FakePopen.calls.clear()
    monkeypatch.setattr(dp_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_install_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher.DpRankSupervisor, "_restore_signal_handlers", lambda self: None)
    monkeypatch.setattr(dp_launcher, "_process_group_exists", lambda child: False)
    monkeypatch.setattr(
        dp_launcher.sys,
        "argv",
        ["/workspace/UniLab/src/unilab/scripts/train_sac.py", "task=g1_walk_flat", "--debug"],
    )

    with dp_launcher.DpRankSupervisor((0, 1), "/tmp/run"):
        pass

    assert len(_FakePopen.calls) == 1
    assert _FakePopen.calls[0].command == [
        dp_launcher.sys.executable,
        "/workspace/UniLab/src/unilab/scripts/train_sac.py",
        "task=g1_walk_flat",
        "--debug",
    ]
