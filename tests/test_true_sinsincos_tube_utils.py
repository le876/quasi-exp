from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_utils():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "true_sinsincos_tube_utils.py"
    spec = importlib.util.spec_from_file_location("true_sinsincos_tube_utils", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_true_sinsincos_ellipse_is_rank2_and_closed() -> None:
    mod = _load_utils()

    target = mod.make_true_sinsincos_ellipse(
        ellipse_id="E75",
        center=(1.08, 0.20, 0.17),
        amp_xy_mm=75.0,
        n_points=360,
        include_endpoint=True,
    )
    report = mod.trajectory_geometry_metrics(target[["x_target_m", "y_target_m", "z_target_m"]].to_numpy())

    assert report["rank2_gate_pass"] is True
    assert report["closed_loop_gap_m"] < 1.0e-9
    assert report["svd2_over_svd1"] > 0.3
    assert report["svd3_over_svd1"] < 1.0e-9
    assert math.isclose(report["amp_z_over_xy_mean"], 1.5, rel_tol=0.05)
    assert report["polygon_area_m2"] > 0.0


def test_same_phase_trajectory_is_rejected_as_line() -> None:
    mod = _load_utils()
    t = np.linspace(0.0, 2.0 * math.pi, 361)
    xyz = np.column_stack(
        [
            1.08 + 0.075 * np.sin(t),
            0.20 + 0.075 * np.sin(t),
            0.17 + 0.1125 * np.sin(t),
        ]
    )

    report = mod.trajectory_geometry_metrics(xyz)

    assert report["rank2_gate_pass"] is False
    assert report["svd2_over_svd1"] < 1.0e-6
    assert report["polygon_area_m2"] < 1.0e-8


def test_recompute_support_uses_current_targets_not_inherited_candidate_fields() -> None:
    mod = _load_utils()
    targets = pd.DataFrame(
        {
            "ellipse_id": ["E"] * 3,
            "angle_idx": [0, 1, 2],
            "x_target_m": [1.0, 1.01, 1.02],
            "y_target_m": [0.0, 0.0, 0.0],
            "z_target_m": [0.0, 0.0, 0.0],
            "nn_p95_mm": [999.0, 999.0, 999.0],
            "tube_count_p10": [0.0, 0.0, 0.0],
        }
    )
    pool = pd.DataFrame(
        {
            "x_m": [1.0, 1.01, 1.02],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "beta1_rad": [0.0, 0.0, 0.0],
            "beta2_rad": [0.0, 0.0, 0.0],
            "beta3_rad": [0.0, 0.0, 0.0],
            "beta4_rad": [0.0, 0.0, 0.0],
            "beta5_rad": [0.0, 0.0, 0.0],
            "beta6_rad": [0.0, 0.0, 0.0],
        }
    )

    summary, per_angle = mod.recompute_support_for_targets(
        targets,
        pool,
        pool_label="unit",
        tube_radius_mm=1.0,
        beta_pair_radius_mm=1.0,
    )

    assert summary.loc[0, "true_nn_p95_mm"] == 0.0
    assert summary.loc[0, "true_tube_count_p10"] == 1.0
    assert "nn_p95_mm" not in summary.columns
    assert per_angle["true_nn_mm"].max() == 0.0

