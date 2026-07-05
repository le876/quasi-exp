from __future__ import annotations

import numpy as np

from quasi_exp.opt.pso_inverse import (
    _joint_preference_score,
    _select_joint_preferred_candidate,
)


def _ranges() -> dict[str, tuple[float, float]]:
    return {
        "beta1": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta2": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta3": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta4": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta5": (np.deg2rad(-10.0), np.deg2rad(10.0)),
        "beta6": (np.deg2rad(-10.0), np.deg2rad(10.0)),
    }


def _cfg() -> dict:
    return {
        "joint_preference": {
            "enabled": True,
            "mode": "prefer_distal",
            "xyz_tol_m": 0.004,
            "beta_indices_by_joint": {
                "joint1": [0, 1],
                "joint2": [2, 3],
                "joint3": [4, 5],
            },
            "joint_limit_deg": {
                "joint1": [5.0, 5.0],
                "joint2": [5.0, 5.0],
                "joint3": [10.0, 10.0],
            },
            "proximal_weight": 1.0,
            "middle_weight": 1.0,
            "distal_reward": 1.25,
        }
    }


def test_joint_preference_score_rewards_distal_and_penalizes_proximal() -> None:
    proximal = np.deg2rad(np.array([4.0, 4.0, 3.0, 3.0, 1.0, 1.0], dtype=float))
    distal = np.deg2rad(np.array([1.0, 1.0, 1.0, 1.0, 8.0, 8.0], dtype=float))

    assert _joint_preference_score(distal, _cfg(), _ranges()) < _joint_preference_score(
        proximal,
        _cfg(),
        _ranges(),
    )


def test_select_joint_preferred_candidate_applies_only_within_xyz_tolerance() -> None:
    best_xyz = np.deg2rad(np.array([4.0, 4.0, 3.0, 3.0, 1.0, 1.0], dtype=float))
    distal = np.deg2rad(np.array([1.0, 1.0, 1.0, 1.0, 8.0, 8.0], dtype=float))
    candidates = np.vstack([best_xyz, distal])

    chosen = _select_joint_preferred_candidate(
        candidates=candidates,
        xyz_errs=np.array([0.001, 0.0045], dtype=float),
        inverse_pso_cfg=_cfg(),
        beta_ranges_rad=_ranges(),
        default_index=0,
    )
    assert chosen == 1

    chosen = _select_joint_preferred_candidate(
        candidates=candidates,
        xyz_errs=np.array([0.001, 0.0061], dtype=float),
        inverse_pso_cfg=_cfg(),
        beta_ranges_rad=_ranges(),
        default_index=0,
    )
    assert chosen == 0


def test_select_joint_preferred_candidate_disabled_returns_default() -> None:
    cfg = _cfg()
    cfg["joint_preference"]["enabled"] = False
    candidates = np.zeros((2, 6), dtype=float)

    assert (
        _select_joint_preferred_candidate(
            candidates=candidates,
            xyz_errs=np.array([0.002, 0.002], dtype=float),
            inverse_pso_cfg=cfg,
            beta_ranges_rad=_ranges(),
            default_index=1,
        )
        == 1
    )
