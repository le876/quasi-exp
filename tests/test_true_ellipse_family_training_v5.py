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


def _audit_ready_dataset(mod) -> pd.DataFrame:
    dataset = _trajectory_dataset()
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
    return dataset


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


def test_training_markdown_rendering_does_not_require_pandas_tabulate(monkeypatch) -> None:
    mod = _load_module()

    def unexpected_optional_dependency(*_args, **_kwargs):
        raise AssertionError("pandas.to_markdown must not be called")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", unexpected_optional_dependency)
    rendered = mod.dataframe_to_markdown(pd.DataFrame({"radius_mm": [87.5], "gate_pass": [False]}))

    assert "| radius_mm | gate_pass |" in rendered
    assert "| 87.5 | False |" in rendered


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


def test_formal_training_protocol_rejects_validation_and_screen_grid_downgrades() -> None:
    mod = _load_module()
    protocol_report = getattr(mod, "formal_training_protocol_report", lambda _args: {})

    formal = protocol_report(mod.parse_args([]))
    wrong_validation = protocol_report(mod.parse_args(["--validation-radius-mm", "75"]))
    truncated_screen = protocol_report(mod.parse_args(["--screen-config-limit", "1"]))

    assert formal.get("formal_training_protocol_gate_pass") is True
    assert formal["checks"]["validation_radius_85"] is True
    assert formal["checks"]["screen_config_limit_zero"] is True
    assert formal["checks"]["exact_24_model_configs"] is True
    assert formal["checks"]["v4_baseline_present"] is True
    assert wrong_validation.get("formal_training_protocol_gate_pass") is False
    assert truncated_screen.get("formal_training_protocol_gate_pass") is False


def test_formal_screen_gate_requires_all_24_configs_and_v4_baseline() -> None:
    mod = _load_module()
    args = mod.parse_args([])
    protocol = mod.formal_training_protocol_report(args)
    config_ids = [config.config_id for config in mod.v5_model_configs()]
    report = {
        "preset": "formal",
        "formal_training_protocol_gate_pass": True,
        "formal_training_protocol_fingerprint": protocol["formal_training_protocol_fingerprint"],
        "validation_radius_mm": 85.0,
        "screen_config_limit": 0,
        "screen_seed_count": 1,
        "screen_config_count": 24,
        "screen_run_count": 24,
        "screen_config_ids": config_ids,
    }
    screen_gate = getattr(mod, "formal_screen_protocol_gate_pass", lambda *_args, **_kwargs: False)

    assert screen_gate(report, args=args) is True
    assert screen_gate({**report, "screen_config_count": 1, "screen_run_count": 1}, args=args) is False
    assert screen_gate(
        {**report, "screen_config_ids": [config_id for config_id in config_ids if config_id != mod.V4_BASELINE_CONFIG_ID]},
        args=args,
    ) is False


def test_formal_model_claims_require_a_valid_audit_split_and_screen_chain() -> None:
    mod = _load_module()
    claims_gate = getattr(mod, "formal_model_claims_allowed", lambda *_args, **_kwargs: False)
    args = mod.parse_args([])
    split = {"formal_claims_allowed": True}
    screen = {"formal_screen_protocol_gate_pass": True, "formal_claims_allowed": True}

    assert claims_gate(args, split_report=split, screen_report=screen) is True
    assert claims_gate(
        args,
        split_report=split,
        screen_report={**screen, "formal_screen_protocol_gate_pass": False},
    ) is False
    args.preset = "pilot"
    assert claims_gate(args, split_report=split, screen_report=screen) is False


def test_training_audit_gates_out_dataset_without_primary_radius(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset = dataset[dataset["radius_mm"].le(85.0)].reset_index(drop=True)

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


def test_training_audit_rejects_cross_family_supervision(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    other = dataset.copy()
    other["sample_id"] = "other:" + other["sample_id"].astype(str)
    other["trajectory_id"] = "other:" + other["trajectory_id"].astype(str)
    other["family_id"] = "other"
    other["candidate_id"] = "other"
    dataset = pd.concat([dataset, other], ignore_index=True)
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
    assert report["checks"]["single_fixed_family"] is False
    assert report["audit_gate_pass"] is False


def test_formal_training_audit_requires_a_formal_expansion_protocol(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    mod.write_json(
        expansion_dir / "05_dataset" / "dataset_report.json",
        {
            "dataset_gate_pass": True,
            "formal_expansion_protocol_gate_pass": False,
        },
    )
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.out_dir = tmp_path / "formal-training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)
    formal_report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert formal_report["checks"]["source_formal_expansion_protocol"] is False
    assert formal_report["audit_gate_pass"] is False

    args.preset = "smoke"
    args.out_dir = tmp_path / "smoke-training"
    smoke_report = mod.phase_audit(args)
    assert smoke_report["audit_gate_pass"] is True
    assert smoke_report["formal_claims_allowed"] is False


def test_formal_training_audit_rejects_dataset_not_bound_to_source_report(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    manifest_path = expansion_dir / "05_dataset" / "trajectory_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"trajectory_id": sorted(dataset["trajectory_id"].unique())}).to_csv(manifest_path, index=False)
    mod.write_json(
        expansion_dir / "05_dataset" / "dataset_report.json",
        {
            "dataset_gate_pass": True,
            "formal_expansion_protocol_gate_pass": True,
            "formal_dataset_gate_pass": True,
            "dataset_path": str(tmp_path / "different-dataset.parquet"),
            "dataset_sha256": "0" * 64,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": mod.file_sha256(manifest_path),
            "artifact_binding_complete": True,
            "protocol": {
                "primary_radius_mm": 87.5,
                "stretch_radius_mm": 100.0,
                "radius_anchors_mm": [75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0],
                "final_points": 360,
                "tube_offsets_mm": [-5.0, -2.5, 0.0, 2.5, 5.0],
            },
        },
    )
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.out_dir = tmp_path / "training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)

    report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert report["checks"]["source_dataset_path_match"] is False
    assert report["checks"]["source_dataset_sha256_match"] is False
    assert report["audit_gate_pass"] is False


def test_formal_training_audit_rejects_robot_config_mismatch(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    manifest_path = expansion_dir / "05_dataset" / "trajectory_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("trajectory_id\nfixture\n", encoding="utf-8")
    source_robot_config = tmp_path / "source_robot.yaml"
    source_robot_config.write_text("kinematics:\n  theta_sign: -1\n", encoding="utf-8")
    training_robot_config = tmp_path / "training_robot.yaml"
    training_robot_config.write_text("kinematics:\n  theta_sign: 1\n", encoding="utf-8")
    source_protocol = {
        "primary_radius_mm": 87.5,
        "stretch_radius_mm": 100.0,
        "radius_anchors_mm": [75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0],
        "final_points": 360,
        "tube_offsets_mm": [-5.0, -2.5, 0.0, 2.5, 5.0],
    }
    mod.write_json(
        expansion_dir / "05_dataset" / "dataset_report.json",
        {
            "dataset_gate_pass": True,
            "formal_expansion_protocol_gate_pass": True,
            "formal_dataset_gate_pass": True,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": mod.file_sha256(dataset_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": mod.file_sha256(manifest_path),
            "robot_config_path": str(source_robot_config.resolve()),
            "robot_config_sha256": mod.file_sha256(source_robot_config),
            "artifact_binding_complete": True,
            "protocol": source_protocol,
            "protocol_fingerprint": mod.stable_fingerprint(source_protocol),
        },
    )
    monkeypatch.setattr(
        mod,
        "materialized_trajectory_checks",
        lambda *_args, **_kwargs: {
            "dataset_trajectories_complete": True,
            "source_manifest_complete": True,
        },
    )
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.robot_config = training_robot_config
    args.out_dir = tmp_path / "training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)

    report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert report["checks"]["source_robot_config_path_match"] is False
    assert report["checks"]["source_robot_config_sha256_match"] is False


def test_formal_training_audit_rejects_incomplete_bound_family_coverage(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    dataset_dir = expansion_dir / "05_dataset"
    pointwise_dir = expansion_dir / "02_pointwise"
    branch_dir = expansion_dir / "03_branch"
    dataset_dir.mkdir(parents=True)
    pointwise_dir.mkdir(parents=True)
    branch_dir.mkdir(parents=True)

    manifest_path = dataset_dir / "trajectory_manifest.csv"
    manifest_path.write_text("trajectory_id\nfixture\n", encoding="utf-8")
    robot_config = tmp_path / "robot.yaml"
    robot_config.write_text("kinematics:\n  theta_sign: -1\n", encoding="utf-8")
    family_ids = [f"family-{index}" for index in range(4)]
    pointwise_report_path = pointwise_dir / "pointwise_report.json"
    pointwise_selection_path = pointwise_dir / "selected_families.csv"
    branch_report_path = branch_dir / "branch_report.json"
    mod.write_json(pointwise_report_path, {"selected_family_count": 4})
    pd.DataFrame({"candidate_id": family_ids}).to_csv(pointwise_selection_path, index=False)
    mod.write_json(
        branch_report_path,
        {
            "selected_family_count": 4,
            "selected_families": family_ids,
            "executed_families": family_ids,
            "all_selected_families_executed": True,
        },
    )
    source_protocol = {
        "primary_radius_mm": 87.5,
        "stretch_radius_mm": 100.0,
        "radius_anchors_mm": [75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0],
        "final_points": 360,
        "tube_offsets_mm": [-5.0, -2.5, 0.0, 2.5, 5.0],
    }
    reduced_coverage = mod.formal_family_coverage_report(
        {"selected_family_count": 4},
        {
            "selected_family_count": 4,
            "selected_families": family_ids,
            "executed_families": family_ids,
            "all_selected_families_executed": True,
        },
        pointwise_selected_families=family_ids,
    )
    mod.write_json(
        dataset_dir / "dataset_report.json",
        {
            "dataset_gate_pass": True,
            "formal_expansion_protocol_gate_pass": True,
            "formal_dataset_gate_pass": True,
            "formal_family_coverage_gate_pass": True,
            "formal_family_coverage_fingerprint": reduced_coverage["formal_family_coverage_fingerprint"],
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": mod.file_sha256(dataset_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": mod.file_sha256(manifest_path),
            "robot_config_path": str(robot_config.resolve()),
            "robot_config_sha256": mod.file_sha256(robot_config),
            "pointwise_report_path": str(pointwise_report_path.resolve()),
            "pointwise_report_sha256": mod.file_sha256(pointwise_report_path),
            "pointwise_selection_path": str(pointwise_selection_path.resolve()),
            "pointwise_selection_sha256": mod.file_sha256(pointwise_selection_path),
            "branch_report_path": str(branch_report_path.resolve()),
            "branch_report_sha256": mod.file_sha256(branch_report_path),
            "artifact_binding_complete": True,
            "protocol": source_protocol,
            "protocol_fingerprint": mod.stable_fingerprint(source_protocol),
        },
    )
    monkeypatch.setattr(
        mod,
        "materialized_trajectory_checks",
        lambda *_args, **_kwargs: {
            "dataset_trajectories_complete": True,
            "source_manifest_complete": True,
        },
    )
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.robot_config = robot_config
    args.out_dir = tmp_path / "training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)

    report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert report["checks"]["source_family_coverage_gate"] is False
    assert report["checks"]["source_family_coverage_fingerprint_match"] is True
    assert report["audit_gate_pass"] is False


def test_formal_training_audit_rejects_source_protocol_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    manifest_path = expansion_dir / "05_dataset" / "trajectory_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"trajectory_id": sorted(dataset["trajectory_id"].unique())}).to_csv(manifest_path, index=False)
    mod.write_json(
        expansion_dir / "05_dataset" / "dataset_report.json",
        {
            "dataset_gate_pass": True,
            "formal_expansion_protocol_gate_pass": True,
            "formal_dataset_gate_pass": True,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": mod.file_sha256(dataset_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": mod.file_sha256(manifest_path),
            "artifact_binding_complete": True,
            "protocol": {
                "primary_radius_mm": 80.0,
                "stretch_radius_mm": 100.0,
                "radius_anchors_mm": [75.0, 80.0, 85.0, 87.5, 100.0],
                "final_points": 360,
                "tube_offsets_mm": [-5.0, -2.5, 0.0, 2.5, 5.0],
            },
        },
    )
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.out_dir = tmp_path / "training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)

    report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert report["checks"]["source_primary_radius_match"] is False
    assert report["checks"]["source_radius_anchors_match"] is False
    assert report["checks"]["source_protocol_matches_training"] is False


def test_formal_audit_gate_depends_on_formal_training_protocol(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    manifest_path = expansion_dir / "05_dataset" / "trajectory_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("trajectory_id\nfixture\n", encoding="utf-8")
    source_protocol = {
        "primary_radius_mm": 87.5,
        "stretch_radius_mm": 100.0,
        "radius_anchors_mm": [75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0],
        "final_points": 360,
        "tube_offsets_mm": [-5.0, -2.5, 0.0, 2.5, 5.0],
    }
    mod.write_json(
        expansion_dir / "05_dataset" / "dataset_report.json",
        {
            "dataset_gate_pass": True,
            "formal_expansion_protocol_gate_pass": True,
            "formal_dataset_gate_pass": True,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": mod.file_sha256(dataset_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": mod.file_sha256(manifest_path),
            "artifact_binding_complete": True,
            "protocol": source_protocol,
            "protocol_fingerprint": mod.stable_fingerprint(source_protocol),
        },
    )
    monkeypatch.setattr(
        mod,
        "materialized_trajectory_checks",
        lambda *_args, **_kwargs: {
            "dataset_trajectories_complete": True,
            "source_manifest_complete": True,
        },
    )
    args = mod.parse_args(["--validation-radius-mm", "75"])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.out_dir = tmp_path / "training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)

    report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert report["formal_training_protocol_gate_pass"] is False
    assert report["formal_claims_allowed"] is False


def test_formal_training_audit_rejects_incomplete_360x25_trajectories(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = _audit_ready_dataset(mod)
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    manifest_path = expansion_dir / "05_dataset" / "trajectory_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest = pd.DataFrame(
        {
            "trajectory_id": sorted(dataset["trajectory_id"].unique()),
            "family_id": "family",
            "radius_mm": sorted(dataset["radius_mm"].unique()),
            "rows": 6,
            "expected_rows": 9000,
            "angle_count": 3,
            "offset_count": 1,
            "duplicate_angle_offset_rows": 6,
            "tube_success_ratio": 1.0,
            "trajectory_complete": False,
            "selected_for_dataset": True,
        }
    )
    manifest.to_csv(manifest_path, index=False)
    protocol = {
        "primary_radius_mm": 87.5,
        "stretch_radius_mm": 100.0,
        "radius_anchors_mm": [75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0],
        "final_points": 360,
        "tube_offsets_mm": [-5.0, -2.5, 0.0, 2.5, 5.0],
    }
    mod.write_json(
        expansion_dir / "05_dataset" / "dataset_report.json",
        {
            "dataset_gate_pass": True,
            "formal_expansion_protocol_gate_pass": True,
            "formal_dataset_gate_pass": True,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": mod.file_sha256(dataset_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": mod.file_sha256(manifest_path),
            "artifact_binding_complete": True,
            "protocol": protocol,
            "protocol_fingerprint": mod.stable_fingerprint(protocol),
        },
    )
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir
    args.out_dir = tmp_path / "training"

    with pytest.raises(RuntimeError, match="audit failed"):
        mod.phase_audit(args)

    report = mod.read_json(args.out_dir / "00_audit" / "audit_report.json")
    assert report["checks"]["source_manifest_complete"] is False
    assert report["checks"]["dataset_trajectories_complete"] is False
    assert report["audit_gate_pass"] is False


def test_materialized_trajectory_checks_require_manifest_family_and_radius_match() -> None:
    mod = _load_module()
    rows = []
    for angle_idx in range(2):
        for dn1 in (-1.0, 1.0):
            for dn2 in (-1.0, 1.0):
                rows.append(
                    {
                        "trajectory_id": "family@75",
                        "family_id": "family",
                        "radius_mm": 75.0,
                        "angle_idx": angle_idx,
                        "tube_offset_id": f"n1_{dn1:g}_n2_{dn2:g}",
                        "delta_n1_mm": dn1,
                        "delta_n2_mm": dn2,
                        "tube_success": True,
                    }
                )
    dataset = pd.DataFrame(rows)
    manifest = pd.DataFrame(
        [
            {
                "trajectory_id": "family@75",
                "family_id": "different-family",
                "radius_mm": 80.0,
                "rows": 8,
                "expected_rows": 8,
                "angle_count": 2,
                "offset_count": 4,
                "duplicate_angle_offset_rows": 0,
                "tube_success_ratio": 1.0,
                "trajectory_complete": True,
                "selected_for_dataset": True,
            }
        ]
    )

    checks = mod.materialized_trajectory_checks(
        dataset,
        manifest,
        final_points=2,
        tube_offsets_mm=[-1.0, 1.0],
    )

    assert checks["dataset_trajectories_complete"] is True
    assert checks["source_manifest_complete"] is False


def test_formal_seed_gate_requires_four_of_exactly_five_and_nonformal_cannot_claim_strict() -> None:
    mod = _load_module()
    five = pd.DataFrame({"model_gate_pass": [True, True, True, True, False]})
    one = pd.DataFrame({"model_gate_pass": [True]})
    three = pd.DataFrame({"model_gate_pass": [True, True, True]})

    assert mod.aggregate_v5_seed_gate(five, preset="formal")["stable_gate_pass"] is True
    assert mod.aggregate_v5_seed_gate(one, preset="formal")["stable_gate_pass"] is False
    assert mod.aggregate_v5_seed_gate(three, preset="pilot")["stable_gate_pass"] is False
    with pytest.raises(ValueError, match="exactly 5 unique seeds"):
        mod.resolve_final_seeds("1", preset="formal")
    assert mod.resolve_final_seeds(",".join(map(str, mod.DEFAULT_SEEDS)), preset="formal") == list(mod.DEFAULT_SEEDS)


def test_training_task_fingerprint_covers_protocol_inputs_and_cache_reuse() -> None:
    mod = _load_module()
    task = {
        "task_id": "formal_model_s1",
        "mode": "final",
        "preset": "formal",
        "config": {"architecture": "tiny"},
        "seed": 1,
        "dataset": "/data/a.parquet",
        "dataset_sha256": "data-a",
        "assignment": "/data/split.parquet",
        "assignment_sha256": "split-a",
        "robot_config": "/data/robot.yaml",
        "robot_config_sha256": "robot-a",
        "evaluation_family_id": "family",
        "angle_stride": 1,
        "max_iter": 800,
        "batch_size": 256,
        "evaluation_specs": [{"label": "primary", "split": "primary_holdout", "radius_mm": 87.5}],
        "training_task_strategy_version": mod.TRAINING_TASK_STRATEGY_VERSION,
        "result_path": "/tmp/result.json",
        "package_path": "/tmp/model.joblib",
        "prediction_dir": "/tmp/predictions",
    }
    fingerprint = mod.training_task_fingerprint(task)
    task["task_fingerprint"] = fingerprint

    assert mod.cached_training_result_is_compatible(task, {"task_fingerprint": fingerprint}) is True
    assert mod.cached_training_result_is_compatible(task, {"task_fingerprint": "old"}) is False
    changed = dict(task)
    changed["preset"] = "pilot"
    assert mod.training_task_fingerprint(changed) != fingerprint
    changed = dict(task)
    changed["evaluation_specs"] = [{"label": "primary", "split": "primary_holdout", "radius_mm": 100.0}]
    assert mod.training_task_fingerprint(changed) != fingerprint


def test_split_task_fingerprint_covers_holdout_protocol(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(dataset_path, index=False)
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    audit = {"task_fingerprint": "audit-current"}

    current = mod.split_task_fingerprint(args, audit)
    args.validation_radius_mm = 82.5
    changed_radius = mod.split_task_fingerprint(args, audit)
    changed_audit = mod.split_task_fingerprint(args, {"task_fingerprint": "audit-new"})

    assert current != changed_radius
    assert changed_radius != changed_audit
    assert mod.phase_report_is_compatible(
        {"strategy_version": mod.SPLIT_STRATEGY_VERSION, "task_fingerprint": current},
        strategy_version=mod.SPLIT_STRATEGY_VERSION,
        task_fingerprint=current,
    ) is True


def test_training_audit_fingerprint_covers_bound_manifest_content(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(dataset_path, index=False)
    expansion_dir = tmp_path / "expansion"
    manifest_path = expansion_dir / "05_dataset" / "trajectory_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("trajectory_id\na\n", encoding="utf-8")
    mod.write_json(
        expansion_dir / "05_dataset" / "dataset_report.json",
        {"manifest_path": str(manifest_path.resolve())},
    )
    args = mod.parse_args([])
    args.tube_dataset = dataset_path
    args.expansion_dir = expansion_dir

    current = mod.training_audit_task_fingerprint(args)
    manifest_path.write_text("trajectory_id\nb\n", encoding="utf-8")
    changed = mod.training_audit_task_fingerprint(args)

    assert changed != current


def test_absent_stretch_partition_is_reported_as_not_held_out() -> None:
    mod = _load_module()
    dataset = _trajectory_dataset()
    dataset = dataset[~np.isclose(dataset["radius_mm"], 100.0)].reset_index(drop=True)
    split = mod.make_whole_trajectory_split(
        dataset,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        validation_radius_mm=85.0,
    )

    status = mod.whole_radius_holdout_status(split)

    assert status["primary_whole_radius_held_out"] is True
    assert status["stretch_whole_radius_held_out"] is False


def test_strict_sweep_gate_requires_beta_truth_and_uses_materialized_centerline() -> None:
    mod = _load_module()
    metrics = {
        "ee_p95_mm": 1.0,
        "ee_max_mm": 2.0,
        "axiserr_max_p95_abs_mm": 1.0,
        "fixed_any_axis_bias_gt2mm": False,
        "delta_beta_p95_deg": 0.1,
        "delta_beta_max_deg": 0.2,
        "delta2_beta_p95_deg": 0.01,
        "seam_beta_rms_deg": 0.1,
        "beta_bound_violation_count": 0,
    }
    assert mod.strict_sweep_model_gate(metrics) is False
    assert mod.strict_sweep_model_gate({**metrics, "beta_p95_deg": 0.5}) is True

    dataset = _trajectory_dataset()
    for idx, column in enumerate(mod.v4.BETA_COLS):
        dataset[column] = 0.01 * idx
    metadata = {
        "center_x_m": 0.0,
        "center_y_m": 0.0,
        "center_z_m": 0.0,
        "phase_y_rad": 0.0,
        "phase_z_rad": 0.0,
    }
    target, beta_true = mod.radius_sweep_reference(
        dataset,
        family_id="family",
        metadata=metadata,
        radius_mm=75.0,
        n_points=4,
    )
    generated, missing_beta = mod.radius_sweep_reference(
        dataset,
        family_id="family",
        metadata=metadata,
        radius_mm=82.5,
        n_points=4,
    )
    assert len(target) == 3
    assert beta_true is not None and beta_true.shape == (3, 6)
    assert len(generated) == 4
    assert missing_beta is None


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
    contamination = dataset[dataset["radius_mm"].isin([75.0, 80.0])].copy()
    contamination["sample_id"] = "other:" + contamination["sample_id"].astype(str)
    contamination["trajectory_id"] = "other:" + contamination["trajectory_id"].astype(str)
    contamination["family_id"] = "other"
    contamination["candidate_id"] = "other"
    dataset = pd.concat([dataset, contamination], ignore_index=True)
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
    task["task_fingerprint"] = mod.training_task_fingerprint(task)

    result = mod.run_training_worker(task)

    assert result["train_rows"] == 16
    assert result["evaluations"]["primary"]["rows"] == 8
    assert result["evaluations"]["stretch"]["rows"] == 8
    assert package_path.exists()

    leaked_assignment = assignment.copy()
    leaked_assignment.loc[
        np.isclose(leaked_assignment["radius_mm"].to_numpy(dtype=float), 87.5),
        "used_for_training",
    ] = True
    leaked_assignment.to_parquet(assignment_path, index=False)
    leaked_task = dict(task)
    leaked_task["task_id"] = "tiny_leaked"
    leaked_task["task_fingerprint"] = mod.training_task_fingerprint(leaked_task)
    with pytest.raises(ValueError, match="leaks an evaluation radius"):
        mod.run_training_worker(leaked_task)
