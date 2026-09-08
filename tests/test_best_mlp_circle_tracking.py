from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "plot_best_mlp_circle_tracking.py"
    spec = importlib.util.spec_from_file_location("plot_best_mlp_circle_tracking", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_circle_points_lie_in_yz_plane() -> None:
    mod = _load_module()

    points = mod.make_circle_points(x0=1.10, radius=0.20, n_points=8, phase_rad=0.0)

    assert points.shape == (8, 3)
    assert np.allclose(points[:, 0], 1.10)
    assert np.isclose(np.sqrt(points[0, 1] ** 2 + points[0, 2] ** 2), 0.20)
    assert np.isclose(points[0, 1], 0.20)
    assert np.isclose(points[0, 2], 0.0)


def test_tracking_metrics_report_mm_error_and_tension_quality() -> None:
    mod = _load_module()
    target = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.0, 0.1],
        ],
        dtype=float,
    )
    achieved = target + np.array(
        [
            [0.001, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.003],
        ],
        dtype=float,
    )
    tension = np.array(
        [
            [10.0, 20.0],
            [-1.0, 30.0],
            [2500.0, 40.0],
        ],
        dtype=float,
    )

    metrics = mod.compute_tracking_metrics(target, achieved, tension, tension_upper_n=2000.0)

    assert np.isclose(metrics["ee_mean_mm"], 2.0)
    assert np.isclose(metrics["ee_max_mm"], 3.0)
    assert metrics["negative_tension_ratio"] == 1.0 / 6.0
    assert metrics["over_upper_tension_ratio"] == 1.0 / 6.0
    assert metrics["tension_max_n"] == 2500.0


def test_select_best_candidate_prefers_valid_large_radius_then_low_p95() -> None:
    mod = _load_module()
    rows = [
        {
            "candidate_id": "tiny",
            "x0_m": 1.2,
            "radius_m": 0.04,
            "ee_p95_mm": 0.5,
            "ee_rmse_mm": 0.5,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
        },
        {
            "candidate_id": "invalid_tension",
            "x0_m": 1.1,
            "radius_m": 0.18,
            "ee_p95_mm": 1.0,
            "ee_rmse_mm": 1.0,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.1,
        },
        {
            "candidate_id": "best",
            "x0_m": 1.05,
            "radius_m": 0.16,
            "ee_p95_mm": 2.0,
            "ee_rmse_mm": 1.4,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
        },
        {
            "candidate_id": "worse",
            "x0_m": 1.0,
            "radius_m": 0.30,
            "ee_p95_mm": 3.0,
            "ee_rmse_mm": 1.0,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
        },
    ]

    best = mod.select_best_candidate(rows, min_showcase_radius_m=0.10)

    assert best["candidate_id"] == "best"


def test_build_candidate_grid_keeps_radius_inside_local_safe_workspace() -> None:
    mod = _load_module()

    candidates = mod.build_candidate_grid(
        x_values=[1.10],
        local_safe_radius_by_x={1.10: 0.32},
        fractions=[0.5, 1.0],
        min_radius_m=0.10,
        max_global_radius_m=0.38,
        n_points=64,
    )

    assert [c.radius_m for c in candidates] == [0.16, 0.32]
    assert all(c.x0_m == 1.10 for c in candidates)
    assert all(c.n_points == 64 for c in candidates)


def test_default_artifact_paths_use_model_dir_and_shared_trial_scaler() -> None:
    mod = _load_module()

    assert mod.DEFAULT_MODEL.name == "model.joblib"
    assert mod.DEFAULT_MODEL.parent.name == "mlp_large"
    assert mod.DEFAULT_SCALER.name == "scaler.joblib"
    assert mod.DEFAULT_SCALER.parent.name == "grid_relabel_iid_L2_alpha1em05_es"


def test_circle_axis_limits_keep_3d_plot_on_radius_scale() -> None:
    mod = _load_module()

    limits = mod.circle_axis_limits(x0_m=1.20, radius_m=0.15, margin=1.1)

    assert np.allclose(limits["x"], (1.035, 1.365))
    assert np.allclose(limits["y"], (-0.165, 0.165))
    assert np.allclose(limits["z"], (-0.165, 0.165))
