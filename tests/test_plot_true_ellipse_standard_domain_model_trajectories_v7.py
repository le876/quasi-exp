from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "plot_true_ellipse_standard_domain_model_trajectories_v7.py"
    spec = importlib.util.spec_from_file_location(
        "plot_true_ellipse_standard_domain_model_trajectories_v7", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prediction(radius_mm: float, *, rows: int = 64) -> pd.DataFrame:
    angle = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)
    radius_m = radius_mm / 1000.0
    target = np.column_stack(
        [
            1.1 + radius_m * np.sin(angle),
            0.1 + radius_m * np.sin(angle + 0.55),
            -0.2 + 1.5 * radius_m * np.sin(angle + 1.05),
        ]
    )
    achieved = target + np.column_stack(
        [
            0.0003 * np.cos(angle),
            0.0002 * np.sin(angle),
            0.0004 * np.cos(angle + 0.2),
        ]
    )
    return pd.DataFrame(
        {
            "dense_angle_idx": np.arange(rows),
            "angle_rad": angle,
            "radius_mm": radius_mm,
            "x_target_m": target[:, 0],
            "y_target_m": target[:, 1],
            "z_target_m": target[:, 2],
            "achieved_x_m": achieved[:, 0],
            "achieved_y_m": achieved[:, 1],
            "achieved_z_m": achieved[:, 2],
            "x_error_mm": (achieved[:, 0] - target[:, 0]) * 1000.0,
            "y_error_mm": (achieved[:, 1] - target[:, 1]) * 1000.0,
            "z_error_mm": (achieved[:, 2] - target[:, 2]) * 1000.0,
            "ee_err_mm": np.linalg.norm(achieved - target, axis=1) * 1000.0,
            "evidence_class": "accepted_continuous_truth",
            "beta_truth_available": True,
            "output_link_id": "identity",
        }
    )


def test_dynamic_camera_is_near_plane_normal_and_not_edge_on() -> None:
    mod = _load_module()
    xyz = _prediction(104.0)[["x_target_m", "y_target_m", "z_target_m"]].to_numpy()

    view = mod.trajectory_view_from_xyz(xyz)
    camera = mod.camera_direction_from_view(view)
    _u, _s, vh = np.linalg.svd(xyz - xyz.mean(axis=0), full_matrices=False)
    normal = vh[-1]

    assert abs(float(np.dot(camera, normal))) > 0.90
    assert view["azimuth_offset_deg"] == 12.0


def test_model_composite_uses_existing_style_and_writes_audited_png(tmp_path: Path) -> None:
    mod = _load_module()
    curves = {radius: _prediction(radius) for radius in (100.0, 102.5, 104.0)}
    output = tmp_path / "model.png"

    record = mod.render_model_composite(
        config_id="mlp_fixture__identity",
        curves=curves,
        output_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 10_000
    assert record["diagnostic_only"] is True
    assert record["changes_v7_formal_gate"] is False
    assert record["radii_mm"] == [100.0, 102.5, 104.0]
    assert record["view_policy"] == "svd_plane_normal_plus_12deg"
    assert len(record["source_curve_hashes"]) == 3


def test_formal_training_suite_renders_every_seed_and_centerline_holdout(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    run_dir = tmp_path / "formal_training"
    final_dir = run_dir / "03_final_models"
    prediction_root = final_dir / "holdout_predictions"
    labels = {
        "validation_integer_centerline": 95.0,
        "validation_half_phase": 95.0,
        "test_integer_centerline": 102.5,
        "test_half_phase": 102.5,
    }
    for seed in (11, 12):
        seed_dir = prediction_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for label, radius in labels.items():
            curve = _prediction(radius)
            if "half_phase" in label:
                curve["angle_rad"] += np.pi / len(curve)
            curve.to_parquet(seed_dir / f"{label}.parquet", index=False)
    final_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "preset": "formal",
        "selected_config_id": "mlp_fixture__tanh_bounds",
        "seed_count": 2,
        "seeds": [11, 12],
        "holdout": {
            "validation_radius_mm": 95.0,
            "test_radius_mm": 102.5,
        },
        "formal_claims_allowed": True,
        "evidence_only": False,
        "model_evidence_gate_pass": True,
        "formal_model_gate_pass": True,
    }
    (final_dir / "final_training_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    rendered = mod.render_formal_training_trajectory_suite(run_dir=run_dir)

    assert rendered["visualization_integrity_gate_pass"] is True
    assert rendered["formal_claims_allowed"] is True
    assert rendered["evidence_only"] is False
    assert rendered["view_policy"] == "svd_plane_normal_plus_12deg"
    assert rendered["seeds"] == [11, 12]
    assert len(rendered["seed_figures"]) == 2
    assert len(rendered["prediction_sources"]) == 8
    assert {entry["evaluation_label"] for entry in rendered["prediction_sources"]} == set(
        labels
    )
    for figure in rendered["seed_figures"]:
        path = Path(figure["png_path"])
        assert path.is_file()
        assert path.stat().st_size > 10_000
        assert figure["evaluation_labels"] == list(labels)
