from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load


REPO_ROOT = Path(__file__).resolve().parents[1]


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float)


class _ZeroModel:
    n_iter_ = 1
    loss_ = 0.0

    def predict(self, features):
        return np.zeros((len(features), 6), dtype=float)


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_standard_domain_diagnostic_training_v7.py"
    spec = importlib.util.spec_from_file_location(
        "run_true_ellipse_standard_domain_diagnostic_training_v7", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _centerline(radius_mm: float, *, points: int = 4) -> pd.DataFrame:
    angle = np.arange(points, dtype=float) * 2.0 * np.pi / points
    frame = pd.DataFrame(
        {
            "candidate_id": ["family"] * points,
            "family_id": ["family"] * points,
            "angle_idx": np.arange(points, dtype=int),
            "angle_rad": angle,
            "center_x_m": [1.0] * points,
            "center_y_m": [0.1] * points,
            "center_z_m": [-0.2] * points,
            "amp_xy_mm": [radius_mm] * points,
            "amp_z_mm": [1.5 * radius_mm] * points,
            "phase_y_rad": [0.3] * points,
            "phase_z_rad": [0.8] * points,
            "radius_mm": [radius_mm] * points,
            "x_target_m": 1.0 + radius_mm / 1000.0 * np.sin(angle),
            "y_target_m": 0.1 + radius_mm / 1000.0 * np.sin(angle + 0.3),
            "z_target_m": -0.2 + 1.5 * radius_mm / 1000.0 * np.sin(angle + 0.8),
        }
    )
    for index in range(1, 7):
        frame[f"beta{index}_rad"] = 0.01 * index * np.sin(angle + index / 10.0)
    return frame


def test_diagnostic_grid_is_24_base_configs_crossed_with_two_links() -> None:
    mod = _load_module()

    configs = mod.diagnostic_model_configs()

    assert len(configs) == 48
    assert len({config.config_id for config in configs}) == 48
    assert {config.output_link_id for config in configs} == {"identity", "tanh_bounds"}
    assert all(
        sum(config.base_config == other.base_config for other in configs) == 2
        for config in configs
    )


def test_screen_dataset_physically_excludes_every_post_selection_radius(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    radii = (75.0, 100.0, 101.0, 102.0, 102.5, 103.5, 104.0)
    frames: dict[Path, pd.DataFrame] = {}
    manifest = {}
    for radius in radii:
        path = tmp_path / f"r{radius:g}.parquet"
        path.write_bytes(f"source:{radius:g}".encode())
        frames[path] = _centerline(radius)
        manifest[f"{radius:g}"] = {
            "path": str(path),
            "sha256": mod.file_sha256(path),
        }
    reads: list[Path] = []
    real_read_parquet = pd.read_parquet

    def fake_read(path, *args, **kwargs):
        source = Path(path)
        reads.append(source)
        return frames[source].copy()

    monkeypatch.setattr(mod.pd, "read_parquet", fake_read)
    output = tmp_path / "screen.parquet"
    report = mod.materialize_screen_dataset(
        manifest,
        output_path=output,
        expected_points_per_radius=4,
    )
    dataset = real_read_parquet(output)

    assert sorted(dataset["radius_mm"].unique().tolist()) == [75.0, 100.0, 102.5]
    assert sorted(report["training_radii_mm"]) == [75.0, 100.0]
    assert report["validation_radii_mm"] == [102.5]
    assert report["post_selection_registered_radii_mm"] == [101.0, 102.0, 103.5, 104.0]
    assert not set(reads) & {Path(manifest[f"{r:g}"]["path"]) for r in (101, 102, 103.5, 104)}
    assert dataset.loc[dataset["split"].eq("train"), "used_for_training"].all()
    assert not dataset.loc[dataset["split"].eq("validation"), "used_for_training"].any()
    assert dataset["diagnostic_only"].eq(True).all()
    assert dataset["formal_claims_allowed"].eq(False).all()
    assert dataset["changes_v7_formal_gate"].eq(False).all()
    assert dataset["static_inverse_claim_radius_mm"].isna().all()


def test_screen_ranking_requires_both_integer_and_half_phase_before_metrics() -> None:
    mod = _load_module()
    rows = []
    for config, both_pass, error, margin, fit_s in (
        ("a", False, 0.1, 1.0, 1.0),
        ("b", True, 1.5, 0.5, 2.0),
        ("c", True, 1.5, 0.7, 3.0),
    ):
        for label in ("validation_integer", "validation_half_phase"):
            rows.append(
                {
                    "config_id": config,
                    "architecture": "mlp",
                    "hidden_layers": (8,),
                    "feature_set": "raw",
                    "activation": "relu",
                    "alpha": 1.0e-4,
                    "output_link_id": "identity",
                    "evaluation_label": label,
                    "model_gate_pass": bool(both_pass or label == "validation_integer"),
                    "ee_p95_mm": error,
                    "beta_p95_deg": error / 2.0,
                    "axiserr_max_p95_abs_mm": error / 3.0,
                    "prediction_min_joint_margin_deg": margin,
                    "fit_s": fit_s,
                }
            )

    ranking = mod.rank_diagnostic_screen(pd.DataFrame(rows))

    assert ranking["config_id"].tolist() == ["c", "b", "a"]
    assert ranking.loc[0, "both_validation_gates_pass"]
    assert ranking.loc[2, "selection_eligible"] is False or not bool(
        ranking.loc[2, "selection_eligible"]
    )
    assert ranking["diagnostic_only"].eq(True).all()
    assert ranking["formal_claims_allowed"].eq(False).all()
    assert ranking["changes_v7_formal_gate"].eq(False).all()
    assert ranking["static_inverse_claim_radius_mm"].isna().all()


def test_no_gate_fallback_is_explicit_and_deterministic() -> None:
    mod = _load_module()
    ranking = pd.DataFrame(
        {
            "config_id": ["best-numeric", "runner-up"],
            "both_validation_gates_pass": [False, False],
            "worst_ee_p95_mm": [1.0, 2.0],
        }
    )

    selected = mod.select_ranked_config(ranking)

    assert selected["selected_config_id"] == "best-numeric"
    assert selected["selection_note"] == "fallback_best_no_gate"
    assert selected["selected_validation_gate_pass"] is False


def test_diagnostic_package_and_final_claims_are_fail_closed() -> None:
    mod = _load_module()
    metadata = mod.diagnostic_package_metadata(
        source_manifest_sha256="manifest",
        screen_dataset_sha256="screen",
        config_id="config",
        seed=20260711,
    )
    claims = mod.diagnostic_claims()

    assert metadata["kind"] == "beta6_pose_standard_domain_v7_centerline_diagnostic"
    assert metadata["diagnostic_only"] is True
    assert metadata["formal_claims_allowed"] is False
    assert metadata["changes_v7_formal_gate"] is False
    assert claims == {
        "diagnostic_only": True,
        "formal_claims_allowed": False,
        "changes_v7_formal_gate": False,
        "static_inverse_claim_radius_mm": None,
    }


def test_cache_lineage_matching_is_fail_closed() -> None:
    mod = _load_module()
    report = {
        "source_dataset_task_fingerprint": "dataset-new",
        "source_selection_task_fingerprint": "selection-new",
    }

    assert mod.cache_lineage_matches(
        report,
        {
            "source_dataset_task_fingerprint": "dataset-new",
            "source_selection_task_fingerprint": "selection-new",
        },
    )
    assert not mod.cache_lineage_matches(
        report, {"source_dataset_task_fingerprint": "dataset-old"}
    )
    assert not mod.cache_lineage_matches(
        report, {"source_stability_task_fingerprint": "stability-new"}
    )


def test_phase_artifact_fingerprint_changes_with_model_or_csv_content() -> None:
    mod = _load_module()
    records = {"model": {"package_sha256": "model-a", "task_fingerprint": "task"}}
    artifacts = {"metrics": {"path": "/metrics.csv", "sha256": "csv-a"}}
    baseline = mod.phase_artifact_fingerprint(
        model_records=records, artifacts=artifacts
    )

    assert baseline != mod.phase_artifact_fingerprint(
        model_records={
            "model": {"package_sha256": "model-b", "task_fingerprint": "task"}
        },
        artifacts=artifacts,
    )
    assert baseline != mod.phase_artifact_fingerprint(
        model_records=records,
        artifacts={"metrics": {"path": "/metrics.csv", "sha256": "csv-b"}},
    )


def test_training_result_record_hashes_and_validates_worker_json(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    result_path = tmp_path / "worker-result.json"
    package_path = tmp_path / "model.joblib"
    result_path.write_text('{"metric": 1}\n', encoding="utf-8")
    package_path.write_bytes(b"model")

    record = mod.training_result_artifact_record(
        result_path=result_path,
        package_path=package_path,
        package_sha256=mod.file_sha256(package_path),
        task_fingerprint="task",
    )

    assert record["result_sha256"] == mod.file_sha256(result_path)
    assert mod.training_result_artifact_matches(record)
    result_path.write_text('{"metric": 2}\n', encoding="utf-8")
    assert not mod.training_result_artifact_matches(record)


def test_summary_requires_both_result_artifact_fingerprints() -> None:
    mod = _load_module()

    assert mod.summary_result_artifact_fingerprints(
        {"result_artifact_fingerprint": "screen-bytes"},
        {"result_artifact_fingerprint": "stability-bytes"},
    ) == {"screen": "screen-bytes", "stability": "stability-bytes"}
    with pytest.raises(ValueError, match="result artifact fingerprint"):
        mod.summary_result_artifact_fingerprints(
            {"result_artifact_fingerprint": "screen-bytes"}, {}
        )


def test_screen_cache_rejects_same_size_worker_result_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    screen_dir = tmp_path / "02_screen"
    screen_dir.mkdir(parents=True)
    model_results = {}
    for index in range(2):
        result_path = screen_dir / f"result-{index}.json"
        package_path = screen_dir / f"model-{index}.joblib"
        result_path.write_text('{"metric": 1}\n', encoding="utf-8")
        package_path.write_bytes(f"model-{index}".encode())
        model_results[f"config-{index}"] = mod.training_result_artifact_record(
            result_path=result_path,
            package_path=package_path,
            package_sha256=mod.file_sha256(package_path),
            task_fingerprint=f"task-{index}",
        )
    metrics_path = screen_dir / "metrics.csv"
    ranking_path = screen_dir / "ranking.csv"
    metrics_path.write_text("value\n1\n", encoding="utf-8")
    ranking_path.write_text("value\n1\n", encoding="utf-8")
    artifacts = {
        "metrics": mod._artifact_record(metrics_path),
        "ranking": mod._artifact_record(ranking_path),
    }
    report = {
        "preset": "smoke",
        "source_dataset_task_fingerprint": "dataset",
        "screen_config_count": 2,
        "model_results": model_results,
        "artifacts": artifacts,
        "result_artifact_fingerprint": mod.phase_artifact_fingerprint(
            model_records=model_results, artifacts=artifacts
        ),
    }
    mod.write_json(screen_dir / "selection_report.json", report)
    Path(model_results["config-0"]["result_path"]).write_text(
        '{"metric": 2}\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        mod,
        "ensure_dataset_report",
        lambda _args: {"task_fingerprint": "dataset"},
    )
    rebuilt = {"rebuilt": True}
    monkeypatch.setattr(mod, "phase_screen", lambda _args: rebuilt)
    args = Namespace(out_dir=str(tmp_path), preset="smoke", screen_config_limit=0)

    assert mod.ensure_screen_report(args) is rebuilt


def test_stability_cache_rejects_a_different_frozen_selection(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    report_path = tmp_path / "03_stability" / "stability_report.json"
    report_path.parent.mkdir(parents=True)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    mod.write_json(
        report_path,
        {
            "preset": "smoke",
            "seeds": [20260711],
            "selected_config_id": "old-config",
            "source_selection_task_fingerprint": "old-selection",
            "model_paths": {"20260711": str(model_path)},
            "model_hashes": {"20260711": mod.file_sha256(model_path)},
        },
    )
    args = Namespace(
        out_dir=str(tmp_path),
        preset="smoke",
        seeds="20260711",
    )
    monkeypatch.setattr(
        mod,
        "ensure_screen_report",
        lambda _args: {
            "selected_config_id": "new-config",
            "task_fingerprint": "new-selection",
        },
    )
    rebuilt = {"rebuilt": True}
    monkeypatch.setattr(mod, "phase_stability", lambda _args: rebuilt)

    assert mod.ensure_stability_report(args) is rebuilt


def test_audit_defers_post_selection_content_hashing(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    source = tmp_path / "r104.parquet"
    source.write_bytes(b"deferred-centerline")
    calls: list[Path] = []

    def forbidden_hash(path):
        calls.append(Path(path))
        raise AssertionError("post-selection centerline was hashed before selection")

    monkeypatch.setattr(mod, "file_sha256", forbidden_hash)
    record = mod.audit_centerline_entry(
        {
            "path": str(source),
            "sha256": "registered-manifest-hash",
            "branch_hash": "registered-branch-hash",
        },
        radius_mm=104.0,
    )

    assert calls == []
    assert record["exists"] is True
    assert record["hash_matches"] is None
    assert record["hash_verification_phase"] == "post_selection"


def test_post_selection_materialization_rejects_mixed_families(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    manifest = {}
    frames = {}
    for radius, family in ((75.0, "family-a"), (104.0, "family-b")):
        source = tmp_path / f"r{radius:g}.parquet"
        source.write_bytes(f"source-{radius:g}".encode())
        frame = _centerline(radius)
        frame["family_id"] = family
        frames[source] = frame
        manifest[f"{radius:g}"] = {
            "path": str(source),
            "sha256": mod.file_sha256(source),
        }
    selection = tmp_path / "selection.json"
    mod.write_json(selection, {"selected_config_id": "selected"})
    monkeypatch.setattr(
        mod.pd,
        "read_parquet",
        lambda path, *args, **kwargs: frames[Path(path)].copy(),
    )

    with np.testing.assert_raises_regex(ValueError, "single family"):
        mod.materialize_registered_centerlines_after_selection(
            strict_manifest=manifest,
            selection_report_path=selection,
            output_path=tmp_path / "all.parquet",
            expected_points_per_radius=4,
        )


def test_full_protocol_rejects_a_truncated_model_screen() -> None:
    mod = _load_module()
    args = Namespace(preset="full", screen_config_limit=47)

    with np.testing.assert_raises_regex(ValueError, "all 48"):
        mod.validate_protocol_args(args)


def test_accepted_radius_summary_carries_the_diagnostic_claim_boundary() -> None:
    mod = _load_module()
    metrics = pd.DataFrame(
        {
            "radius_mm": [102.5],
            "seed": [20260711],
            "diagnostic_tracking_gate_pass": [True],
            "ee_p95_mm": [1.0],
            "beta_p95_deg": [0.1],
            "prediction_min_joint_margin_deg": [1.0],
            "beta_bound_violation_count": [0],
        }
    )

    summary = mod._stable_radius_summary(metrics, preset="smoke")

    assert summary["diagnostic_only"].eq(True).all()
    assert summary["formal_claims_allowed"].eq(False).all()
    assert summary["changes_v7_formal_gate"].eq(False).all()
    assert summary["static_inverse_claim_radius_mm"].isna().all()


def test_evaluation_cache_rejects_a_changed_metrics_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    evaluation_dir = tmp_path / "04_evaluation"
    evaluation_dir.mkdir(parents=True)
    artifacts = {}
    for name in (
        "screen_model_metrics",
        "stability_metrics",
        "unsupported_metrics",
        "prediction_manifest",
        "all_registered_centerlines",
    ):
        path = evaluation_dir / f"{name}.dat"
        path.write_bytes(name.encode())
        artifacts[name] = {"path": str(path), "sha256": mod.file_sha256(path)}
    report = {
        "preset": "smoke",
        "source_selection_task_fingerprint": "selection",
        "source_stability_task_fingerprint": "stability",
        "screen_model_metrics_path": artifacts["screen_model_metrics"]["path"],
        "stability_metrics_path": artifacts["stability_metrics"]["path"],
        "unsupported_metrics_path": artifacts["unsupported_metrics"]["path"],
        "prediction_manifest_path": artifacts["prediction_manifest"]["path"],
        "all_registered_centerlines": {
            "dataset_path": artifacts["all_registered_centerlines"]["path"],
            "dataset_sha256": artifacts["all_registered_centerlines"]["sha256"],
        },
        "artifacts": artifacts,
    }
    mod.write_json(evaluation_dir / "evaluation_report.json", report)
    args = Namespace(out_dir=str(tmp_path), preset="smoke")
    monkeypatch.setattr(
        mod,
        "ensure_screen_report",
        lambda _args: {"task_fingerprint": "selection"},
    )
    monkeypatch.setattr(
        mod,
        "ensure_stability_report",
        lambda _args: {"task_fingerprint": "stability"},
    )
    rebuilt = {"rebuilt": True}
    monkeypatch.setattr(mod, "phase_evaluate", lambda _args: rebuilt)

    assert mod.ensure_evaluation_report(args) == report
    Path(artifacts["stability_metrics"]["path"]).write_bytes(b"tampered")
    assert mod.ensure_evaluation_report(args) is rebuilt


def test_dense_curve_has_integer_and_half_phase_target_samples() -> None:
    mod = _load_module()
    source = _centerline(102.5, points=360)

    dense = mod.dense_centerline_frame(source, n_points=720, include_beta_truth=True)

    assert len(dense) == 720
    assert np.isclose(dense.iloc[0]["angle_rad"], 0.0)
    assert np.isclose(dense.iloc[1]["angle_rad"], np.pi / 360.0)
    assert dense["phase_kind"].iloc[::2].eq("integer").all()
    assert dense["phase_kind"].iloc[1::2].eq("half_phase").all()
    assert dense[[f"beta{i}_rad" for i in range(1, 7)]].notna().all().all()


def test_training_worker_saves_diagnostic_package_and_two_validation_evaluations(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    train = _centerline(75.0, points=8)
    validation = _centerline(102.5, points=8)
    frames = []
    for frame, split in ((train, "train"), (validation, "validation")):
        materialized = frame.copy()
        materialized["trajectory_id"] = f"family@{materialized['radius_mm'].iloc[0]:g}"
        materialized["sample_id"] = [f"{split}:{index}" for index in range(len(frame))]
        materialized["split"] = split
        materialized["used_for_training"] = split == "train"
        materialized["is_centerline"] = True
        frames.append(materialized)
    dataset = pd.concat(frames, ignore_index=True)
    dataset_path = tmp_path / "screen.parquet"
    dataset.to_parquet(dataset_path, index=False)
    robot_config = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"
    package_path = tmp_path / "model.joblib"
    result_path = tmp_path / "result.json"
    config = mod.diagnostic_model_configs()[0]

    def fake_fit(**kwargs):
        return _ZeroModel(), _IdentityScaler(), _IdentityScaler(), ["x", "y", "z"], 0.01

    monkeypatch.setattr(mod.formal_training.v5.v4, "_fit_scaled_model", fake_fit)
    task = mod.make_training_task(
        dataset_path=dataset_path,
        source_manifest_sha256="manifest-hash",
        robot_config_path=robot_config,
        config=config,
        seed=20260711,
        max_iter=1,
        angle_stride=1,
        mode="unit",
        result_path=result_path,
        package_path=package_path,
        dense_points=16,
    )

    result = mod.run_training_worker(task)
    package = load(package_path)

    assert result["train_rows"] == 8
    assert set(result["evaluations"]) == {
        "validation_integer",
        "validation_half_phase",
    }
    assert result_path.is_file()
    assert package["kind"] == mod.DIAGNOSTIC_KIND
    assert package["diagnostic_only"] is True
    assert package["formal_claims_allowed"] is False
    assert package["changes_v7_formal_gate"] is False
    assert package["static_inverse_claim_radius_mm"] is None
    assert package["config_id"] == config.config_id
    assert package["dataset_sha256"] == mod.file_sha256(dataset_path)


def test_screen_task_builder_assigns_a_package_to_every_model(tmp_path: Path) -> None:
    mod = _load_module()
    dataset = tmp_path / "screen.parquet"
    dataset.write_bytes(b"screen")
    robot_config = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"
    configs = mod.diagnostic_model_configs()[:3]

    tasks = mod.build_screen_tasks(
        configs=configs,
        dataset_path=dataset,
        source_manifest_sha256="manifest",
        robot_config_path=robot_config,
        out_dir=tmp_path / "screen",
        seed=20260711,
        max_iter=30,
        angle_stride=30,
        dense_points=720,
    )

    assert len(tasks) == 3
    assert len({task["package_path"] for task in tasks}) == 3
    assert all(task["package_path"] for task in tasks)
    assert all(task["mode"] == "v7_diagnostic_screen" for task in tasks)
    assert all(task["task_fingerprint"] == mod.training_task_fingerprint(task) for task in tasks)


def test_unsupported_target_has_no_beta_truth_and_evaluation_never_invents_it() -> None:
    mod = _load_module()
    source = _centerline(104.0, points=16)
    unsupported = mod.target_only_centerline_frame(source, radius_mm=120.0, n_points=32)
    robot_config = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"
    cfg = mod.formal_training.v5.v4.load_config(str(robot_config))
    robot_inputs = mod.formal_training.v5.v4.load_robot_inputs(cfg)
    package = {
        "model": _ZeroModel(),
        "x_scaler": _IdentityScaler(),
        "y_scaler": _IdentityScaler(),
        "feature_set": "raw",
        "joint_domain_id": mod.JOINT_DOMAIN_ID,
        "output_link_id": "identity",
    }

    metrics, prediction = mod.evaluate_package_trajectory(
        package=package,
        frame=unsupported,
        robot_inputs=robot_inputs,
        theta_sign=float(cfg.get("kinematics", {}).get("theta_sign", -1.0)),
        evidence_class="unsupported_model_extrapolation",
    )

    assert len(unsupported) == 32
    assert not set(mod.BETA_COLUMNS) & set(unsupported.columns)
    assert metrics["beta_truth_available"] is False
    assert metrics["evidence_class"] == "unsupported_model_extrapolation"
    assert "beta_p95_deg" not in metrics
    assert prediction["beta_truth_available"].eq(False).all()
    assert prediction["diagnostic_only"].eq(True).all()
    assert prediction["formal_claims_allowed"].eq(False).all()
    assert prediction["changes_v7_formal_gate"].eq(False).all()
    assert prediction["static_inverse_claim_radius_mm"].isna().all()
    assert not any(column.startswith("true_beta") for column in prediction)


def test_accepted_dense_evaluation_keeps_beta_truth_and_diagnostic_label() -> None:
    mod = _load_module()
    dense = mod.dense_centerline_frame(
        _centerline(104.0, points=16), n_points=32, include_beta_truth=True
    )
    robot_config = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"
    cfg = mod.formal_training.v5.v4.load_config(str(robot_config))
    robot_inputs = mod.formal_training.v5.v4.load_robot_inputs(cfg)
    package = {
        "model": _ZeroModel(),
        "x_scaler": _IdentityScaler(),
        "y_scaler": _IdentityScaler(),
        "feature_set": "raw",
        "joint_domain_id": mod.JOINT_DOMAIN_ID,
        "output_link_id": "identity",
    }

    metrics, prediction = mod.evaluate_package_trajectory(
        package=package,
        frame=dense,
        robot_inputs=robot_inputs,
        theta_sign=float(cfg.get("kinematics", {}).get("theta_sign", -1.0)),
        evidence_class="accepted_continuous_truth",
    )

    assert metrics["beta_truth_available"] is True
    assert metrics["evidence_class"] == "accepted_continuous_truth"
    assert "beta_p95_deg" in metrics
    assert prediction["beta_truth_available"].eq(True).all()
    assert prediction["diagnostic_only"].eq(True).all()
    assert prediction["formal_claims_allowed"].eq(False).all()
    assert prediction["changes_v7_formal_gate"].eq(False).all()
    assert prediction["static_inverse_claim_radius_mm"].isna().all()
    assert all(f"true_beta{i}_rad" in prediction for i in range(1, 7))
