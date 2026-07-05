from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConstantThetaModel:
    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.theta.reshape(1, -1), (len(X), 1))


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_theta_only_ellipse_trajectory_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_theta_only_ellipse_trajectory_benchmark", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_make_diagonal_ellipse_points_preserves_scaled_200_200_300_ratio() -> None:
    mod = _load_module()

    points = mod.make_diagonal_ellipse_points(
        center_x=1.08,
        center_y=0.02,
        center_z=-0.03,
        amp_xy=0.04,
        n_points=16,
    )

    assert points.shape == (16, 3)
    assert np.allclose(points[:, 0] - 1.08, points[:, 1] - 0.02)
    assert np.isclose(np.max(points[:, 0] - 1.08), 0.04)
    assert np.isclose(np.max(points[:, 2] + 0.03), 0.06)


def test_filter_workspace_supported_candidates_uses_nn_and_x_bias_gates() -> None:
    mod = _load_module()
    candidates = [
        mod.EllipseCandidate("ok", 1.08, 0.0, 0.0, 0.02, 0.03, 32),
        mod.EllipseCandidate("far", 1.08, 0.0, 0.0, 0.02, 0.03, 32),
        mod.EllipseCandidate("xbias", 1.08, 0.0, 0.0, 0.02, 0.03, 32),
    ]
    meta = {
        "ok": {"supported": True, "nn_dist_p95_mm": 4.0, "nearest_x_diff_mean_mm": 0.5},
        "far": {"supported": False, "nn_dist_p95_mm": 12.0, "nearest_x_diff_mean_mm": 0.2},
        "xbias": {"supported": True, "nn_dist_p95_mm": 4.0, "nearest_x_diff_mean_mm": 3.0},
    }

    selected = mod.filter_workspace_supported_candidates(
        candidates,
        meta,
        max_nearest_x_mean_abs_mm=2.0,
        max_candidates_per_amp=4,
    )

    assert [c.candidate_id for c in selected] == ["ok"]


def test_select_showcase_rows_returns_largest_qualified_and_lowest_error() -> None:
    mod = _load_module()
    rows = [
        {
            "model_id": "m",
            "candidate_id": "small_best",
            "amp_xy_m": 0.02,
            "supported": True,
            "ee_p95_mm": 1.0,
            "ee_rmse_mm": 0.8,
            "axiserr_max_p95_abs_mm": 0.8,
            "fixed_any_axis_bias_gt2mm": False,
        },
        {
            "model_id": "m",
            "candidate_id": "large_good",
            "amp_xy_m": 0.06,
            "supported": True,
            "ee_p95_mm": 8.0,
            "ee_rmse_mm": 5.0,
            "axiserr_max_p95_abs_mm": 4.0,
            "fixed_any_axis_bias_gt2mm": False,
        },
        {
            "model_id": "m",
            "candidate_id": "largest_bad_bias",
            "amp_xy_m": 0.08,
            "supported": True,
            "ee_p95_mm": 2.0,
            "ee_rmse_mm": 1.5,
            "axiserr_max_p95_abs_mm": 1.2,
            "fixed_any_axis_bias_gt2mm": True,
        },
    ]

    showcase = mod.select_showcase_rows(rows, quality_ee_p95_mm=20.0, quality_axis_p95_mm=10.0)

    assert showcase["largest_qualified"]["candidate_id"] == "large_good"
    assert showcase["lowest_error"]["candidate_id"] == "small_best"


def test_markdown_table_does_not_require_optional_tabulate_dependency() -> None:
    mod = _load_module()
    df = pd.DataFrame(
        [
            {
                "selection_note": "lowest_error",
                "model_id": "iid_lgbm",
                "candidate_id": "c1",
                "amp_xy_m": 0.02,
                "ee_p95_mm": 1.2,
            }
        ]
    )

    table = mod._markdown_table(df, "Rows")

    assert "| selection_note |" in table
    assert "iid_lgbm" in table


def test_evaluate_model_candidate_accepts_theta_only_model_package(tmp_path: Path) -> None:
    mod = _load_module()

    theta = np.zeros(30, dtype=float)
    package = {
        "kind": "theta_only",
        "model": ConstantThetaModel(theta),
        "y_scaler": None,
        "input_cols": ["x_m", "y_m", "z_m"],
        "theta_cols": [f"theta_{i}_rad" for i in range(1, 31)],
        "model_name": "constant",
        "split": "iid",
    }
    model_path = tmp_path / "model.joblib"
    dump(package, model_path)

    loaded = mod.load_theta_model_package(model_path)
    target = np.array([[1.2, 0.0, 0.0], [1.2, 0.01, 0.0]], dtype=float)
    lengths = np.full(31, 0.04, dtype=float)
    p_end = np.array([0.04, 0.0, 0.0, 1.0], dtype=float)
    candidate = mod.EllipseCandidate("c", 1.2, 0.0, 0.0, 0.01, 0.015, 2)

    row, points = mod.evaluate_model_candidate(
        model_id="constant_iid",
        model_name="constant",
        split="iid",
        candidate=candidate,
        target_xyz_m=target,
        workspace_meta={"supported": True, "nn_dist_p95_mm": 1.0, "nearest_x_diff_mean_mm": 0.0},
        package=loaded,
        lengths_m=lengths,
        p_end_local_m=p_end,
        fixed_bias_threshold_mm=2.0,
    )

    assert row["model_id"] == "constant_iid"
    assert row["ee_p95_mm"] >= 0.0
    assert set(["target_x_m", "achieved_x_m", "x_error_mm"]).issubset(points.columns)
    assert len(points) == 2
