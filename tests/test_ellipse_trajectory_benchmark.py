from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_ellipse_trajectory_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_ellipse_trajectory_benchmark", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_diagonal_ellipse_points_preserves_200_200_300_ratio() -> None:
    mod = _load_module()

    points = mod.make_diagonal_ellipse_points(
        center_x=1.08,
        center_y=0.02,
        center_z=-0.03,
        amp_xy=0.10,
        n_points=12,
    )

    assert points.shape == (12, 3)
    assert np.isclose(points[0, 0], 1.08)
    assert np.isclose(points[0, 1], 0.02)
    assert np.isclose(points[0, 2], 0.12)
    assert np.allclose(points[:, 0] - 1.08, points[:, 1] - 0.02)
    assert np.isclose(np.max(points[:, 0] - 1.08), 0.10)
    assert np.isclose(np.max(points[:, 2] + 0.03), 0.15)


def test_build_ellipse_candidate_grid_keeps_millimeter_center_x_unique() -> None:
    mod = _load_module()

    candidates = mod.build_ellipse_candidate_grid(
        center_x_values=[1.080, 1.081, 1.082],
        centers_yz=[(0.0, 0.0)],
        amp_xy_values=[0.10],
        n_points=360,
        x_id_decimals=3,
    )

    assert len(candidates) == 3
    assert [c.candidate_id for c in candidates] == [
        "ell_cx1p080_cy0p000_cz0p000_a0p100_z0p150",
        "ell_cx1p081_cy0p000_cz0p000_a0p100_z0p150",
        "ell_cx1p082_cy0p000_cz0p000_a0p100_z0p150",
    ]
    assert all(np.isclose(c.amp_z_m, 1.5 * c.amp_xy_m) for c in candidates)


def test_compute_axis_error_metrics_reports_each_axis_and_fixed_bias() -> None:
    mod = _load_module()
    target = np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.1]])
    achieved = target + np.array([[0.003, -0.001, 0.004], [0.004, -0.002, 0.005], [0.005, -0.003, 0.006]])

    metrics = mod.compute_axis_error_metrics(target, achieved, fixed_bias_threshold_mm=2.0)

    assert np.isclose(metrics["xerr_mean_mm"], 4.0)
    assert np.isclose(metrics["yerr_mean_mm"], -2.0)
    assert np.isclose(metrics["zerr_mean_mm"], 5.0)
    assert np.isclose(metrics["axiserr_max_p95_abs_mm"], metrics["zerr_p95_abs_mm"])
    assert metrics["fixed_x_bias_gt2mm"] is True
    assert metrics["fixed_y_bias_gt2mm"] is False
    assert metrics["fixed_z_bias_gt2mm"] is True
    assert metrics["fixed_any_axis_bias_gt2mm"] is True


def test_select_geometry_candidates_by_amp_band_keeps_best_supported_per_amplitude() -> None:
    mod = _load_module()
    candidates = [
        mod.EllipseCandidate("a10_bad", 1.08, 0.0, 0.0, 0.10, 0.15, 360),
        mod.EllipseCandidate("a10_good", 1.081, 0.0, 0.0, 0.10, 0.15, 360),
        mod.EllipseCandidate("a12_good", 1.08, 0.0, 0.0, 0.12, 0.18, 360),
        mod.EllipseCandidate("a12_unsupported", 1.081, 0.0, 0.0, 0.12, 0.18, 360),
    ]
    meta = {
        "a10_bad": {"supported": True, "nearest_x_diff_mean_mm": 0.5, "nn_dist_p95_mm": 4.0},
        "a10_good": {"supported": True, "nearest_x_diff_mean_mm": 0.2, "nn_dist_p95_mm": 2.0},
        "a12_good": {"supported": True, "nearest_x_diff_mean_mm": 0.1, "nn_dist_p95_mm": 3.0},
        "a12_unsupported": {"supported": False, "nearest_x_diff_mean_mm": 0.0, "nn_dist_p95_mm": 1.0},
    }

    selected = mod.select_geometry_candidates_by_amp_band(
        candidates,
        meta,
        max_nearest_x_mean_abs_mm=2.0,
        max_candidates_per_amp=1,
    )

    assert [c.candidate_id for c in selected] == ["a10_good", "a12_good"]


def test_select_showcase_row_prefers_largest_quality_ellipse_before_lowest_error() -> None:
    mod = _load_module()
    rows = [
        {
            "candidate_id": "small_precise",
            "amp_xy_m": 0.08,
            "supported": True,
            "ee_p95_mm": 1.0,
            "ee_rmse_mm": 0.8,
            "axiserr_max_p95_abs_mm": 0.5,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
            "fixed_any_axis_bias_gt2mm": False,
        },
        {
            "candidate_id": "large_good",
            "amp_xy_m": 0.12,
            "supported": True,
            "ee_p95_mm": 4.0,
            "ee_rmse_mm": 2.0,
            "axiserr_max_p95_abs_mm": 2.0,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
            "fixed_any_axis_bias_gt2mm": False,
        },
        {
            "candidate_id": "largest_bad_axis",
            "amp_xy_m": 0.14,
            "supported": True,
            "ee_p95_mm": 2.0,
            "ee_rmse_mm": 1.0,
            "axiserr_max_p95_abs_mm": 4.0,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
            "fixed_any_axis_bias_gt2mm": False,
        },
    ]

    best = mod.select_showcase_row(rows, quality_ee_p95_mm=5.0, quality_axis_p95_mm=3.0)

    assert best["candidate_id"] == "large_good"
    assert best["selection_note"] == "display_largest_under_quality"
