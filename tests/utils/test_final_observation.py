"""Tests for uni_rl.utils.final_observation.

Ported from UniLab's ``tests/utils/test_final_observation.py`` (pre-split),
adapted to the public surface uni_rl kept: ``TerminalObservationContract`` and
``resolve_terminal_observation_contract``.
"""

from __future__ import annotations

import numpy as np

from uni_rl.utils.final_observation import resolve_terminal_observation_contract


def test_resolve_terminal_observation_contract_returns_terminal_rows_without_copying_next_obs():
    terminal_contract = resolve_terminal_observation_contract(
        next_obs_batch_size=2,
        final_observation={
            "obs": np.array([[5.0, 5.0], [8.0, 9.0]], dtype=np.float32),
            "critic": np.array([[1.0], [7.0]], dtype=np.float32),
        },
        done=np.array([False, True]),
        truncated=np.array([False, True]),
    )

    np.testing.assert_array_equal(terminal_contract.terminal_mask, np.array([False, True]))
    np.testing.assert_array_equal(terminal_contract.timeout_terminal_mask, np.array([False, True]))
    np.testing.assert_array_equal(
        terminal_contract.terminal_obs,
        np.array([[5.0, 5.0], [8.0, 9.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_contract.terminal_critic,
        np.array([[1.0], [7.0]], dtype=np.float32),
    )


def test_no_done_and_no_info_yields_zero_terminal_mask_and_no_terminal_obs():
    terminal_contract = resolve_terminal_observation_contract(next_obs_batch_size=3)

    np.testing.assert_array_equal(terminal_contract.terminal_mask, np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(terminal_contract.timeout_terminal_mask, np.zeros(3, dtype=bool))
    assert terminal_contract.terminal_obs is None
    assert terminal_contract.terminal_critic is None


def test_final_observation_falls_back_to_info_dict():
    final_obs = {"obs": np.array([[4.0], [5.0]], dtype=np.float32)}
    terminal_contract = resolve_terminal_observation_contract(
        next_obs_batch_size=2,
        done=np.array([True, False]),
        info={"final_observation": final_obs},
    )

    np.testing.assert_array_equal(terminal_contract.terminal_mask, np.array([True, False]))
    np.testing.assert_array_equal(terminal_contract.terminal_obs, final_obs["obs"])
    assert terminal_contract.terminal_critic is None


def test_terminal_mask_falls_back_to_info_flag_and_done_shape_mismatch_is_zeroed():
    terminal_contract = resolve_terminal_observation_contract(
        next_obs_batch_size=2,
        info={"_final_observation": np.array([True, False])},
    )
    np.testing.assert_array_equal(terminal_contract.terminal_mask, np.array([True, False]))

    mismatched = resolve_terminal_observation_contract(
        next_obs_batch_size=2,
        done=np.array([True, True, True]),
    )
    np.testing.assert_array_equal(mismatched.terminal_mask, np.zeros(2, dtype=bool))


def test_truncated_narrows_timeout_terminal_mask():
    terminal_contract = resolve_terminal_observation_contract(
        next_obs_batch_size=3,
        done=np.array([True, True, True]),
        truncated=np.array([True, False, True]),
    )

    np.testing.assert_array_equal(terminal_contract.terminal_mask, np.array([True, True, True]))
    np.testing.assert_array_equal(
        terminal_contract.timeout_terminal_mask, np.array([True, False, True])
    )
