"""Contract tests for HORA distill teacher-owner Hydra composition."""

from __future__ import annotations

from pathlib import Path

from uni_rl.hora import distill_config

# NOTE: UniLab's tests/algos/test_hora_distill_config.py additionally covers
# _student_model_defaults against UniLab's packaged conf tree
# (hora_distill/student_model/*.yaml); those tests stay in UniLab as
# integration tests (issue #1478).


# ---------------------------------------------------------------------------
# Hydra composition capabilities used by teacher-owner configs
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_teacher_owner_config_supports_nested_defaults_packages_and_interpolation(
    tmp_path: Path,
) -> None:
    conf_dir = tmp_path / "conf" / "ppo"
    _write(
        conf_dir / "config.yaml",
        "defaults:\n  - _self_\n  - task: group/owner\n  - extra: packed\n\nroot_scalar: 3\n",
    )
    # Package directive: file content lands under `extra`, not at the root.
    _write(conf_dir / "extra" / "packed.yaml", "# @package extra\ninner: ${root_scalar}\n")
    _write(
        conf_dir / "task" / "group" / "owner.yaml",
        "# @package _global_\ndefaults:\n  - group/mid\n  - _self_\n\nleaf: owner\n",
    )
    # Nested defaults: an included group file with its own defaults list.
    _write(
        conf_dir / "task" / "group" / "mid.yaml",
        "# @package _global_\ndefaults:\n  - nested_leaf\n  - _self_\n\nmid_value: 5\n",
    )
    _write(
        conf_dir / "task" / "group" / "nested_leaf.yaml",
        "# @package _global_\nnested_value: 42\n",
    )

    cfg = distill_config.load_teacher_owner_config("ppo", "group/owner", root_dir=tmp_path)

    assert cfg.nested_value == 42
    assert cfg.mid_value == 5
    assert cfg.leaf == "owner"
    assert cfg.extra.inner == 3
