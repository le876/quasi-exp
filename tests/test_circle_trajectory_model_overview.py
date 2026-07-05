from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "plot_circle_trajectory_model_overview.py"
    spec = importlib.util.spec_from_file_location("plot_circle_trajectory_model_overview", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_model_3d_view_matches_single_trajectory_view() -> None:
    mod = _load_module()

    assert mod.TRAJECTORY_3D_VIEW == {"elev": 22, "azim": -62}


def test_3d_common_limits_use_reference_style_cubic_box_aspect() -> None:
    mod = _load_module()

    class FakeAxes:
        def __init__(self) -> None:
            self.xlim = None
            self.ylim = None
            self.zlim = None
            self.box_aspect = None

        def set_xlim(self, *args):
            self.xlim = args

        def set_ylim(self, *args):
            self.ylim = args

        def set_zlim(self, *args):
            self.zlim = args

        def set_box_aspect(self, aspect):
            self.box_aspect = aspect

    ax = FakeAxes()

    mod._set_3d_common_limits(
        ax,
        {
            "x": (1.17, 1.21),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
        },
    )

    assert ax.xlim == (1.17, 1.21)
    assert ax.ylim == (-0.2, 0.2)
    assert ax.zlim == (-0.2, 0.2)
    assert ax.box_aspect == mod.TRAJECTORY_3D_BOX_ASPECT


def test_best_by_model_scoreboard_sorts_by_p95_and_keeps_numeric_columns() -> None:
    mod = _load_module()
    payload = {
        "best_by_model": {
            "slow": {
                "model_id": "slow",
                "candidate_id": "c1",
                "ee_mean_mm": 2.0,
                "ee_rmse_mm": 2.2,
                "ee_p95_mm": 4.0,
                "ee_max_mm": 5.0,
                "pred_time_ms_per_point": 0.2,
                "tension_min_n": 100.0,
                "tension_max_n": 800.0,
                "negative_tension_ratio": 0.0,
                "over_upper_tension_ratio": 0.0,
                "supported": True,
                "selection_note": "supported_radius_tension_filtered",
            },
            "fast": {
                "model_id": "fast",
                "candidate_id": "c2",
                "ee_mean_mm": 1.0,
                "ee_rmse_mm": 1.2,
                "ee_p95_mm": 3.0,
                "ee_max_mm": 4.0,
                "pred_time_ms_per_point": 0.1,
                "tension_min_n": 90.0,
                "tension_max_n": 700.0,
                "negative_tension_ratio": 0.0,
                "over_upper_tension_ratio": 0.0,
                "supported": True,
                "selection_note": "supported_radius_tension_filtered",
            },
        }
    }

    scoreboard = mod.best_by_model_scoreboard(payload)

    assert scoreboard["model_id"].tolist() == ["fast", "slow"]
    for col in mod.SCOREBOARD_COLUMNS:
        assert col in scoreboard.columns


def test_select_unique_points_for_rows_drops_duplicate_representative_labels() -> None:
    mod = _load_module()
    points = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m1", "m2"],
            "candidate_id": ["c1", "c1", "c1", "c1", "c2"],
            "angle_deg": [0.0, 90.0, 0.0, 90.0, 0.0],
            "representative_label": ["a", "a", "b", "b", "a"],
            "target_y_m": [0.1, 0.0, 0.1, 0.0, 0.2],
            "target_z_m": [0.0, 0.1, 0.0, 0.1, 0.0],
            "achieved_y_m": [0.11, 0.01, 0.11, 0.01, 0.21],
            "achieved_z_m": [0.01, 0.11, 0.01, 0.11, 0.01],
            "ee_error_mm": [1.0, 2.0, 1.0, 2.0, 3.0],
        }
    )
    rows = pd.DataFrame({"model_id": ["m1", "m2"], "candidate_id": ["c1", "c2"]})

    selected = mod.select_unique_points_for_rows(points, rows)

    assert len(selected) == 3
    assert selected.groupby(["model_id", "candidate_id"]).size().to_dict() == {("m1", "c1"): 2, ("m2", "c2"): 1}
    assert "representative_label" not in selected.columns


def test_write_scoreboard_table_outputs_markdown_rankings() -> None:
    mod = _load_module()
    scoreboard = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "candidate_id": ["c1", "c2"],
            "ee_mean_mm": [1.1, 2.2],
            "ee_rmse_mm": [1.2, 2.3],
            "ee_p95_mm": [1.5, 3.0],
            "ee_max_mm": [2.0, 4.0],
            "pred_time_ms_per_point": [0.01, 0.02],
            "tension_min_n": [100.0, 90.0],
            "tension_max_n": [700.0, 800.0],
            "negative_tension_ratio": [0.0, 0.01],
            "over_upper_tension_ratio": [0.0, 0.0],
            "supported": [True, False],
            "selection_note": ["ok", "fallback"],
        }
    )

    markdown = mod.scoreboard_to_markdown(scoreboard, title="Best by model", top_n=2)

    assert "Best by model" in markdown
    assert "ee_p95_mm" in markdown
    assert "m1" in markdown
    assert "m2" in markdown


def test_plot_all_model_3d_grid_writes_png(tmp_path: Path) -> None:
    mod = _load_module()
    points = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "candidate_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
            "target_x_m": [1.20, 1.20, 1.20, 1.18, 1.18, 1.18],
            "target_y_m": [0.00, 0.10, 0.00, 0.00, 0.08, 0.00],
            "target_z_m": [0.10, 0.00, -0.10, 0.08, 0.00, -0.08],
            "achieved_x_m": [1.20, 1.21, 1.20, 1.16, 1.17, 1.16],
            "achieved_y_m": [0.00, 0.09, 0.00, 0.00, 0.06, 0.00],
            "achieved_z_m": [0.10, 0.01, -0.09, 0.06, 0.00, -0.06],
        }
    )
    scoreboard = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "candidate_id": ["c1", "c2"],
            "ee_p95_mm": [3.5, 12.0],
            "ee_max_mm": [4.2, 15.0],
            "radius_m": [0.10, 0.08],
            "supported": [True, True],
        }
    )

    out_path = tmp_path / "all_models_3d_grid.png"
    mod.plot_all_model_3d_grid(points, scoreboard, "3D overview", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_all_model_x_error_trajectory_3d_grid_writes_png(tmp_path: Path) -> None:
    mod = _load_module()
    points = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "candidate_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
            "target_x_m": [1.20, 1.20, 1.20, 1.18, 1.18, 1.18],
            "target_y_m": [0.00, 0.10, 0.00, 0.00, 0.08, 0.00],
            "target_z_m": [0.10, 0.00, -0.10, 0.08, 0.00, -0.08],
            "achieved_x_m": [1.20, 1.21, 1.19, 1.16, 1.17, 1.20],
            "achieved_y_m": [0.00, 0.09, 0.00, 0.00, 0.06, 0.00],
            "achieved_z_m": [0.10, 0.01, -0.09, 0.06, 0.00, -0.06],
        }
    )
    scoreboard = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "candidate_id": ["c1", "c2"],
            "ee_p95_mm": [3.5, 12.0],
            "ee_max_mm": [4.2, 15.0],
            "radius_m": [0.10, 0.08],
            "supported": [True, True],
        }
    )

    out_path = tmp_path / "all_models_x_error_trajectory_3d_grid.png"
    mod.plot_all_model_x_error_trajectory_3d_grid(points, scoreboard, "x error trajectory", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_all_model_x_error_angle_grid_writes_png(tmp_path: Path) -> None:
    mod = _load_module()
    points = pd.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "candidate_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
            "angle_deg": [0.0, 120.0, 240.0, 0.0, 120.0, 240.0],
            "x_error_mm": [0.1, -0.2, 0.3, 2.5, 2.7, 2.9],
        }
    )
    scoreboard = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "candidate_id": ["c1", "c2"],
            "ee_p95_mm": [1.5, 4.0],
            "xerr_p95_abs_mm": [0.3, 2.9],
            "fixed_x_bias_gt2mm": [False, True],
        }
    )

    out_path = tmp_path / "all_models_xerr_vs_angle_grid.png"
    mod.plot_all_model_x_error_angle_grid(points, scoreboard, "x error by angle", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_update_readme_indexes_3d_overview_files(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path
    (out_dir / "README.md").write_text("# Existing\n", encoding="utf-8")
    scoreboard = pd.DataFrame(
        {
            "model_id": ["m1"],
            "candidate_id": ["c1"],
            "ee_mean_mm": [1.0],
            "ee_rmse_mm": [1.1],
            "ee_p95_mm": [1.5],
            "ee_max_mm": [2.0],
            "pred_time_ms_per_point": [0.01],
            "tension_min_n": [100.0],
            "tension_max_n": [700.0],
            "negative_tension_ratio": [0.0],
            "over_upper_tension_ratio": [0.0],
            "supported": [True],
            "selection_note": ["ok"],
        }
    )

    mod.update_readme(
        out_dir,
        [
            "all_models_best_3d_grid.png",
            "all_models_same_global_candidate_3d_grid.png",
            "all_models_best_x_error_trajectory_3d_grid.png",
            "all_models_same_global_candidate_x_error_trajectory_3d_grid.png",
            "all_models_best_xerr_vs_angle_grid.png",
        ],
        scoreboard,
        scoreboard,
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "all_models_best_3d_grid.png" in text
    assert "all_models_same_global_candidate_3d_grid.png" in text
    assert "all_models_best_x_error_trajectory_3d_grid.png" in text
    assert "all_models_same_global_candidate_x_error_trajectory_3d_grid.png" in text
    assert "all_models_best_xerr_vs_angle_grid.png" in text
    assert "x-axis error" in text
    assert "3D" in text
