from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_circle_trajectory_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_circle_trajectory_benchmark", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_offset_circle_points_keep_fixed_x_and_centered_radius() -> None:
    mod = _load_module()

    points = mod.make_offset_circle_points(x0=1.2, center_y=0.08, center_z=-0.04, radius=0.15, n_points=12)

    assert points.shape == (12, 3)
    assert np.allclose(points[:, 0], 1.2)
    assert np.allclose(np.sqrt((points[:, 1] - 0.08) ** 2 + (points[:, 2] + 0.04) ** 2), 0.15)
    assert np.isclose(points[0, 1], 0.23)
    assert np.isclose(points[0, 2], -0.04)


def test_build_candidate_grid_includes_paper_style_showcase_circle() -> None:
    mod = _load_module()

    candidates = mod.build_candidate_grid(
        x_values=[1.2],
        centers=[(0.0, 0.0), (0.08, 0.0)],
        radii=[0.1512, 0.2],
        n_points=240,
    )

    assert any(c.candidate_id == "x1p20_cy0p000_cz0p000_r0p151" for c in candidates)
    assert len({c.candidate_id for c in candidates}) == 4
    assert all(c.n_points == 240 for c in candidates)


def test_build_candidate_grid_can_keep_dense_millimeter_x_values_unique() -> None:
    mod = _load_module()

    candidates = mod.build_candidate_grid(
        x_values=[1.195, 1.196, 1.197],
        centers=[(0.0, 0.0)],
        radii=[0.1512],
        n_points=240,
        x_id_decimals=3,
    )

    assert len(candidates) == 3
    assert [c.candidate_id for c in candidates] == [
        "x1p195_cy0p000_cz0p000_r0p151",
        "x1p196_cy0p000_cz0p000_r0p151",
        "x1p197_cy0p000_cz0p000_r0p151",
    ]


def test_builtin_model_registry_contains_best_mlp_and_direct_baseline_models() -> None:
    mod = _load_module()

    registry = mod.build_builtin_model_registry(repo_root=REPO_ROOT)
    ids = {spec.model_id for spec in registry}

    assert "best_capacity_relabel_iid_L2_mlp_large" in ids
    assert "relabel_iid_mlp_large" in ids
    assert "relabel_radius_mlp" in ids
    assert "relabel_angular_sector_lgbm" in ids
    assert all(spec.feature_set == "poly_heavy" for spec in registry)
    assert all(spec.model_path.name == "model.joblib" for spec in registry)
    assert all(spec.scaler_path.name == "scaler.joblib" for spec in registry)


def test_classify_split_region_reports_majority_and_ratios() -> None:
    mod = _load_module()
    split_membership = np.array(["train", "train", "test", "val", "test"], dtype=object)

    train_region = mod.classify_split_region(np.array([0, 1, 0]), split_membership, majority_threshold=0.60)
    mixed_region = mod.classify_split_region(np.array([0, 2, 3, 4]), split_membership, majority_threshold=0.60)

    assert train_region["region"] == "train_majority"
    assert train_region["train_ratio"] == 1.0
    assert mixed_region["region"] == "mixed"
    assert mixed_region["test_ratio"] == 0.5


def test_workspace_support_gate_uses_p95_nearest_neighbor_distance() -> None:
    mod = _load_module()

    supported = mod.compute_support_metrics(np.array([0.001, 0.010, 0.020]), max_nn_p95_m=0.03)
    unsupported = mod.compute_support_metrics(np.array([0.001, 0.040, 0.050]), max_nn_p95_m=0.03)

    assert supported["supported"] is True
    assert supported["nn_dist_p95_mm"] <= 30.0
    assert unsupported["supported"] is False
    assert unsupported["nn_dist_p95_mm"] > 30.0


def test_fixed_x_bias_requires_same_sign_and_threshold() -> None:
    mod = _load_module()

    assert mod.has_fixed_axis_bias(np.array([2.1, 2.5, 3.0]), threshold_mm=2.0) is True
    assert mod.has_fixed_axis_bias(np.array([-2.1, -2.5, -3.0]), threshold_mm=2.0) is True
    assert mod.has_fixed_axis_bias(np.array([0.5, 2.5, 3.0]), threshold_mm=2.0) is False
    assert mod.has_fixed_axis_bias(np.array([-2.5, 2.5, 3.0]), threshold_mm=2.0) is False


def test_compute_tracking_metrics_reports_x_axis_error() -> None:
    mod = _load_module()
    target = np.array([[1.20, 0.0, 0.0], [1.20, 0.1, 0.0], [1.20, 0.0, 0.1]])
    achieved = np.array([[1.203, 0.0, 0.0], [1.204, 0.1, 0.0], [1.205, 0.0, 0.1]])
    tension = np.ones((3, 12)) * 500.0

    metrics = mod.compute_tracking_metrics(target, achieved, tension, tension_upper_n=2000.0, fixed_bias_threshold_mm=2.0)

    assert np.isclose(metrics["xerr_mean_mm"], 4.0)
    assert np.isclose(metrics["xerr_min_mm"], 3.0)
    assert np.isclose(metrics["xerr_max_mm"], 5.0)
    assert metrics["fixed_x_bias_gt2mm"] is True
    assert metrics["xerr_same_sign_ratio"] == 1.0


def test_candidate_workspace_metadata_reports_nearest_x_bias_and_strict_gate() -> None:
    mod = _load_module()
    xyz = np.array(
        [
            [1.200, 0.0, 0.0],
            [1.201, 0.1, 0.0],
            [1.199, 0.0, 0.1],
        ]
    )
    context = mod.WorkspaceContext(xyz_m=xyz, nn=mod.NearestNeighbors(n_neighbors=1).fit(xyz), split_membership={})
    target = np.array(
        [
            [1.200, 0.0, 0.0],
            [1.200, 0.1, 0.0],
            [1.200, 0.0, 0.1],
        ]
    )

    meta, _ = mod.candidate_workspace_metadata(
        target,
        context,
        max_nn_p95_m=0.008,
        majority_threshold=0.60,
        max_nearest_x_mean_abs_mm=2.0,
        fixed_bias_threshold_mm=2.0,
    )

    assert meta["supported"] is True
    assert meta["strict_geometry_supported"] is True
    assert np.isclose(meta["nearest_x_diff_mean_mm"], 0.0)
    assert meta["fixed_nearest_x_bias_gt2mm"] is False


def test_filter_strict_geometry_candidates_applies_radius_support_and_x_bias() -> None:
    mod = _load_module()
    candidates = [
        mod.CircleCandidate("ok", 1.2, 0.0, 0.0, 0.1512, 240),
        mod.CircleCandidate("tiny", 1.2, 0.0, 0.0, 0.08, 240),
        mod.CircleCandidate("far", 1.2, 0.0, 0.0, 0.1512, 240),
        mod.CircleCandidate("xbias", 1.2, 0.0, 0.0, 0.1512, 240),
    ]
    meta = {
        "ok": {"supported": True, "nearest_x_diff_mean_mm": 0.5},
        "tiny": {"supported": True, "nearest_x_diff_mean_mm": 0.1},
        "far": {"supported": False, "nearest_x_diff_mean_mm": 0.1},
        "xbias": {"supported": True, "nearest_x_diff_mean_mm": 2.5},
    }

    selected = mod.filter_strict_geometry_candidates(
        candidates,
        meta,
        min_radius_m=0.10,
        max_nearest_x_mean_abs_mm=2.0,
    )

    assert [c.candidate_id for c in selected] == ["ok"]


def test_select_best_row_filters_support_tension_and_radius() -> None:
    mod = _load_module()
    rows = [
        {
            "candidate_id": "tiny",
            "model_id": "m",
            "radius_m": 0.05,
            "supported": True,
            "ee_p95_mm": 0.1,
            "ee_rmse_mm": 0.1,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
        },
        {
            "candidate_id": "bad_tension",
            "model_id": "m",
            "radius_m": 0.20,
            "supported": True,
            "ee_p95_mm": 0.2,
            "ee_rmse_mm": 0.2,
            "negative_tension_ratio": 0.01,
            "over_upper_tension_ratio": 0.0,
        },
        {
            "candidate_id": "unsupported",
            "model_id": "m",
            "radius_m": 0.20,
            "supported": False,
            "ee_p95_mm": 0.3,
            "ee_rmse_mm": 0.3,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
        },
        {
            "candidate_id": "best",
            "model_id": "m",
            "radius_m": 0.18,
            "supported": True,
            "ee_p95_mm": 2.0,
            "ee_rmse_mm": 1.5,
            "negative_tension_ratio": 0.0,
            "over_upper_tension_ratio": 0.0,
        },
    ]

    best = mod.select_best_row(rows, min_radius_m=0.10)

    assert best["candidate_id"] == "best"
