from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_family_training_v5.py"
    spec = importlib.util.spec_from_file_location("run_true_ellipse_family_training_v5", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trajectory_dataset() -> pd.DataFrame:
    rows = []
    for radius_mm in (75.0, 80.0, 85.0, 87.5, 100.0):
        trajectory_id = f"family@{radius_mm:g}"
        for angle_idx in range(3):
            for is_centerline in (False, True):
                rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "family_id": "family",
                        "candidate_id": "family",
                        "radius_mm": radius_mm,
                        "angle_idx": angle_idx,
                        "angle_rad": float(angle_idx),
                        "is_centerline": is_centerline,
                        "x_target_m": radius_mm / 1000.0,
                        "y_target_m": angle_idx / 1000.0,
                        "z_target_m": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_whole_trajectory_split_has_no_radius_or_trajectory_leakage() -> None:
    mod = _load_module()
    dataset = _trajectory_dataset()

    split = mod.make_whole_trajectory_split(
        dataset,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        validation_radius_mm=85.0,
    )

    train_ids = set(dataset.iloc[split.train_idx]["trajectory_id"])
    validation_ids = set(dataset.iloc[split.validation_idx]["trajectory_id"])
    primary_ids = set(dataset.iloc[split.primary_idx]["trajectory_id"])
    stretch_ids = set(dataset.iloc[split.stretch_idx]["trajectory_id"])
    assert train_ids == {"family@75", "family@80"}
    assert validation_ids == {"family@85"}
    assert primary_ids == {"family@87.5"}
    assert stretch_ids == {"family@100"}
    assert train_ids.isdisjoint(validation_ids | primary_ids | stretch_ids)
    assert split.trajectory_leakage_count == 0


def test_training_indices_exclude_all_centerlines_and_heldout_radii() -> None:
    mod = _load_module()
    dataset = _trajectory_dataset()
    split = mod.make_whole_trajectory_split(
        dataset,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        validation_radius_mm=85.0,
    )

    indices = mod.strict_training_indices(dataset, split)
    training = dataset.iloc[indices]

    assert not training["is_centerline"].any()
    assert set(training["radius_mm"]) == {75.0, 80.0}


def test_v5_model_grid_contains_v4_baseline_and_controlled_parameter_variants() -> None:
    mod = _load_module()
    configs = mod.v5_model_configs()
    ids = [config.config_id for config in configs]

    assert len(ids) == len(set(ids))
    assert "mlp_beta6_large_poly_heavy_relu_a1em06" in ids
    assert {config.activation for config in configs} == {"relu", "tanh"}
    assert {config.alpha for config in configs} == {1.0e-6, 1.0e-4}
    assert {config.feature_set for config in configs} == {"raw", "poly_medium", "poly_heavy"}
    assert len({config.hidden_layers for config in configs}) >= 2


def test_goal_report_requires_true_holdout_generalization_and_strict_training_support() -> None:
    mod = _load_module()
    status = pd.DataFrame(
        {
            "radius_mm": [75.0, 80.0, 82.5, 85.0, 87.5, 100.0],
            "strict_support_gate_pass": [True, True, True, True, True, False],
            "stable_model_gate_pass": [True, True, True, True, True, True],
            "true_holdout_gate_pass": [False, False, False, False, True, False],
            "trajectory_materialized": [True, True, True, True, True, False],
        }
    )

    report = mod.build_v5_goal_report(
        status,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        anchor_mm=75.0,
    )

    assert report["primary_goal_pass"] is True
    assert report["stretch_goal_pass"] is False
    assert report["strict_supported_rmax_mm"] == 87.5
    assert report["primary_evidence_kind"] == "whole_radius_holdout"


def test_training_cli_defaults_keep_87p5_primary_and_100_stretch_separate() -> None:
    mod = _load_module()
    args = mod.parse_args([])

    assert args.primary_radius_mm == 87.5
    assert args.stretch_radius_mm == 100.0
    assert args.validation_radius_mm == 85.0
    assert args.preset == "formal"
    assert mod.parse_phases("all") == ["audit", "split", "screen", "train", "sweep", "summary"]


def test_training_audit_gates_out_dataset_without_primary_radius(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _trajectory_dataset()
    dataset = dataset[dataset["radius_mm"].le(85.0)].reset_index(drop=True)
    dataset["sample_id"] = [f"sample-{idx}" for idx in range(len(dataset))]
    dataset["tube_offset_id"] = "n1_0_n2_0"
    for column, value in {
        "center_x_m": 1.0,
        "center_y_m": 0.0,
        "center_z_m": 0.0,
        "phase_y_rad": 1.0,
        "phase_z_rad": 2.0,
    }.items():
        dataset[column] = value
    for column in mod.v4.BETA_COLS:
        dataset[column] = 0.0

    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    mod.write_json(expansion_dir / "05_dataset" / "dataset_report.json", {"dataset_gate_pass": True})
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.out_dir = tmp_path / "training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)

    report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert report["checks"]["primary_radius_materialized"] is False
    assert report["checks"]["validation_radius_materialized"] is True
    assert report["audit_gate_pass"] is False


def test_support_scan_uses_only_explicit_training_indices() -> None:
    mod = _load_module()
    metadata = {
        "center_x_m": 0.0,
        "center_y_m": 0.0,
        "center_z_m": 0.0,
        "phase_y_rad": 0.0,
        "phase_z_rad": 0.0,
    }
    target, _angle = mod.v4.generate_exact_ellipse(metadata, amp_xy_mm=75.0, n_points=4)
    dataset = pd.DataFrame(np.vstack([target + np.asarray([0.020, 0.0, 0.0]), target]), columns=mod.v4.TARGET_XYZ_COLS)

    report = mod.compute_support_scan(
        dataset,
        training_indices=np.arange(4),
        metadata=metadata,
        radii_mm=[75.0],
        n_points=4,
    )

    assert np.isclose(report.iloc[0]["nn_p95_mm"], 20.0)
    assert bool(report.iloc[0]["strict_support_gate_pass"]) is False


def test_training_worker_honors_materialized_split_assignment(tmp_path: Path) -> None:
    mod = _load_module()
    args = mod.parse_args([])
    robot_config = mod.v4.load_config(str(args.robot_config))
    robot_inputs = mod.v4.load_robot_inputs(robot_config)
    theta_sign = float(robot_config.get("kinematics", {}).get("theta_sign", -1.0))
    rows = []
    split_by_radius = {
        75.0: "train",
        80.0: "train",
        85.0: "validation_radius",
        87.5: "primary_holdout",
        100.0: "stretch_holdout",
    }
    for radius_rank, (radius_mm, split_name) in enumerate(split_by_radius.items()):
        for angle_idx in range(8):
            for is_centerline in (False, True):
                phase = 2.0 * np.pi * angle_idx / 8.0
                beta = np.deg2rad(
                    np.asarray([0.2, -0.2, 0.4, -0.4, 2.0, -2.0])
                    + 0.05 * radius_rank
                    + 0.03 * np.sin(phase)
                    + (0.01 if not is_centerline else 0.0)
                )
                xyz = mod.v4.fk_dh_batch(
                    mod.v4.theta_from_beta_batch(beta.reshape(1, 6), theta_sign=theta_sign),
                    lengths_m=robot_inputs.lengths_m,
                    p_end_local_m=robot_inputs.p_end_local_m,
                )[0]
                row = {
                    "sample_id": f"{radius_mm}:{angle_idx}:{is_centerline}",
                    "trajectory_id": f"family@{radius_mm:g}",
                    "family_id": "family",
                    "candidate_id": "family",
                    "radius_mm": radius_mm,
                    "angle_idx": angle_idx,
                    "angle_rad": phase,
                    "tube_offset_id": "center" if is_centerline else "offset",
                    "is_centerline": is_centerline,
                    "x_target_m": xyz[0],
                    "y_target_m": xyz[1],
                    "z_target_m": xyz[2],
                    "split": split_name,
                    "used_for_training": split_name == "train" and not is_centerline,
                }
                for idx, column in enumerate(mod.v4.BETA_COLS):
                    row[column] = beta[idx]
                rows.append(row)
    dataset = pd.DataFrame(rows)
    assignment = dataset[
        ["sample_id", "trajectory_id", "family_id", "candidate_id", "radius_mm", "angle_idx", "tube_offset_id", "is_centerline", "split", "used_for_training"]
    ].copy()
    dataset_path = tmp_path / "dataset.parquet"
    assignment_path = tmp_path / "assignment.parquet"
    result_path = tmp_path / "result.json"
    package_path = tmp_path / "model.joblib"
    dataset.drop(columns=["split", "used_for_training"]).to_parquet(dataset_path, index=False)
    assignment.to_parquet(assignment_path, index=False)
    task = {
        "task_id": "tiny",
        "mode": "final",
        "config": {
            "architecture": "tiny",
            "hidden_layers": [16],
            "feature_set": "raw",
            "activation": "relu",
            "alpha": 1.0e-6,
        },
        "seed": 17,
        "dataset": str(dataset_path),
        "assignment": str(assignment_path),
        "robot_config": str(args.robot_config),
        "theta_sign": theta_sign,
        "evaluation_family_id": "family",
        "result_path": str(result_path),
        "package_path": str(package_path),
        "prediction_dir": str(tmp_path / "predictions"),
        "angle_stride": 1,
        "max_iter": 10,
        "batch_size": 16,
        "evaluation_specs": [
            {"label": "primary", "split": "primary_holdout", "radius_mm": 87.5},
            {"label": "stretch", "split": "stretch_holdout", "radius_mm": 100.0},
        ],
    }

    result = mod.run_training_worker(task)

    assert result["train_rows"] == 16
    assert result["evaluations"]["primary"]["rows"] == 8
    assert result["evaluations"]["stretch"]["rows"] == 8
    assert package_path.exists()
