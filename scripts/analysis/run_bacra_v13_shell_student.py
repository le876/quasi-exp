#!/usr/bin/env python3
"""Train and seal the known-chart V13 shell Student after shell Gates pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from quasi_exp.teacher.bacra_student import StudentStrategy, train_one_seed
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from run_bacra_v13_ellipsoidal_shell_atlas import load_config, project_root_from
from run_trajectory_canonical_teacher_v10 import load_environment


PROTOCOL_ID = "bacra-v13-formal-shell-known-chart-student"
CLAIM_SCOPE = "simulation_formal_shell_known_chart_student_learnability"


def _gate(path: Path, checks: Mapping[str, bool], *, semantics: str, **evidence: Any) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "claim_scope": CLAIM_SCOPE,
        "deployment_claim_gate_pass": False,
        "gate_semantics": semantics,
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, payload)
    return payload


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _features(frame: pd.DataFrame, charts: tuple[str, ...]) -> np.ndarray:
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    lookup = {chart: index for index, chart in enumerate(charts)}
    one_hot = np.zeros((len(frame), len(charts)), dtype=np.float32)
    for row_id, chart_id in enumerate(frame["chart_id"].astype(str)):
        if chart_id not in lookup:
            raise ValueError(f"sealed set contains unseen chart_id {chart_id}")
        one_hot[row_id, lookup[chart_id]] = 1.0
    return np.column_stack((xyz, one_hot))


def _sealed_report(
    model_path: Path,
    charts: tuple[str, ...],
    sealed: pd.DataFrame,
    environment: Any,
    *,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    prediction = np.asarray(
        model.predict(
            _features(sealed, charts),
            batch_size=int(batch_size),
            verbose=0,
        ),
        dtype=float,
    )
    achieved = np.asarray(environment.fk(prediction), dtype=float).reshape(-1, 3)
    target = sealed.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    truth = sealed.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    fk_error_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
    beta_gap_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(prediction - truth), axis=1))
    )
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    actual_bounds = np.all(prediction >= bounds[:, 0] - 1.0e-12, axis=1) & np.all(
        prediction <= bounds[:, 1] + 1.0e-12, axis=1
    )
    details = sealed.loc[:, ["sample_id", "chart_id", "cell_id", *XYZ_COLUMNS]].copy()
    details.insert(0, "seed", int(seed))
    details["student_fk_error_mm"] = fk_error_mm
    details["student_beta_rms_deg"] = beta_gap_deg
    details["student_actual_bounds"] = actual_bounds
    report = {
        "seed": int(seed),
        "sealed_row_count": int(len(sealed)),
        "chart_ids": list(charts),
        "sealed_fk_p50_mm": float(np.percentile(fk_error_mm, 50)),
        "sealed_fk_p95_mm": float(np.percentile(fk_error_mm, 95)),
        "sealed_fk_max_mm": float(np.max(fk_error_mm)),
        "sealed_beta_rms_p95_deg": float(np.percentile(beta_gap_deg, 95)),
        "sealed_beta_rms_max_deg": float(np.max(beta_gap_deg)),
        "sealed_actual_bounds_pass": bool(actual_bounds.all()),
    }
    return report, details


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output).resolve()
    if output_root.exists():
        raise FileExistsError(f"Student output already exists: {output_root}")
    output_root.mkdir(parents=True)
    config = load_config(args.config, "formal")
    summary_path = source_root / "05_summary/gate.json"
    dataset_path = source_root / "04_dense_dataset/A3_shell_dataset.parquet"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_dataset_hash = str(summary.get("dataset_sha256", ""))
    actual_dataset_hash = sha256_file(dataset_path)
    git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protocol = output_root / "00_protocol"
    protocol.mkdir()
    protocol_gate = _gate(
        protocol / "gate.json",
        {
            "source_summary_gate_pass": bool(summary.get("gate_pass")),
            "source_dataset_exists": dataset_path.is_file(),
            "source_dataset_hash_matches_summary": actual_dataset_hash == expected_dataset_hash,
            "worktree_clean": git_status == "",
            "formal_dataset_rows": int(summary.get("dataset_rows", -1)) == 200000,
            "automatic_chart_classifier_disabled": config["student"]["automatic_chart_classifier"] is False,
        },
        semantics="locked_formal_shell_dataset_before_student",
        source_root=str(source_root),
        source_summary_sha256=sha256_file(summary_path),
        source_dataset_sha256=actual_dataset_hash,
        git_sha=git_sha,
    )
    if not protocol_gate["gate_pass"]:
        return {"gate_pass": False, "stopped_after": "protocol"}
    # Sealed labels are not opened here.  The first read is restricted to the
    # train/validation rows used for model selection.
    development = pd.read_parquet(
        dataset_path,
        filters=[("split", "in", ["train", "validation"])],
    )
    development["split_role"] = development["split"]
    environment = load_environment(
        project_root,
        project_root / str(config["robot_config"]),
    )
    geometry = StudentGeometry(
        lengths_m=environment.lengths_m,
        p_end_local_m=environment.p_end_local_m,
        theta_sign=environment.theta_sign,
        beta_bounds_rad=environment.bounds,
    )
    student_config = config["student"]
    reports: list[dict[str, Any]] = []
    models_root = output_root / "01_students"
    models_root.mkdir()
    for seed in student_config["seeds"]:
        model_root = models_root / f"seed_{int(seed)}"
        report = dict(
            train_one_seed(
                development,
                strategy=StudentStrategy.CHART_CONDITIONED,
                geometry=geometry,
                seed=int(seed),
                hidden_units=student_config["hidden_units"],
                lambda_fk=float(student_config["lambda_fk"]),
                learning_rate=float(student_config["learning_rate"]),
                batch_size=int(student_config["batch_size"]),
                max_epochs=int(student_config["max_epochs"]),
                patience=int(student_config["patience"]),
                output_dir=model_root,
            )
        )
        reports.append(report)
        atomic_write_json(model_root / "validation_report.json", report)
    model_lock = {
        "source_dataset_sha256": actual_dataset_hash,
        "git_sha": git_sha,
        "models": {
            str(report["seed"]): {
                "path": str(models_root / f"seed_{report['seed']}/model.keras"),
                "sha256": sha256_file(models_root / f"seed_{report['seed']}/model.keras"),
                "validation_report_sha256": sha256_file(
                    models_root / f"seed_{report['seed']}/validation_report.json"
                ),
            }
            for report in reports
        },
    }
    lock_stage = output_root / "02_model_lock"
    lock_stage.mkdir()
    atomic_write_json(lock_stage / "model_lock.json", model_lock)
    # Only after all model bytes are locked do we open the sealed cell split.
    sealed = pd.read_parquet(
        dataset_path,
        filters=[("split", "==", "sealed")],
    )
    sealed_reports: list[dict[str, Any]] = []
    sealed_details: list[pd.DataFrame] = []
    for report in reports:
        seed = int(report["seed"])
        charts = tuple(str(value) for value in report["chart_ids"])
        sealed_report, details = _sealed_report(
            models_root / f"seed_{seed}/model.keras",
            charts,
            sealed,
            environment,
            batch_size=int(student_config["batch_size"]),
            seed=seed,
        )
        sealed_reports.append(sealed_report)
        sealed_details.append(details)
    sealed_stage = output_root / "03_sealed_shell"
    sealed_stage.mkdir()
    _atomic_parquet(pd.concat(sealed_details, ignore_index=True), sealed_stage / "sealed_predictions.parquet")
    atomic_write_json(sealed_stage / "sealed_reports.json", {"reports": sealed_reports})
    validation_p95 = [float(value["validation_fk_p95_mm"]) for value in reports]
    validation_max = [float(value["validation_fk_max_mm"]) for value in reports]
    sealed_p95 = [float(value["sealed_fk_p95_mm"]) for value in sealed_reports]
    sealed_max = [float(value["sealed_fk_max_mm"]) for value in sealed_reports]
    final = output_root / "04_summary"
    final.mkdir()
    gate = _gate(
        final / "gate.json",
        {
            "three_registered_seeds_complete": len(reports) == 3,
            "model_lock_written_before_sealed_evaluation": (lock_stage / "model_lock.json").is_file(),
            "known_chart_architecture": all(value["strategy"] == StudentStrategy.CHART_CONDITIONED.value for value in reports),
            "validation_fk_p95": max(validation_p95) <= float(student_config["validation_fk_p95_max_mm"]),
            "validation_fk_max": max(validation_max) <= float(student_config["validation_fk_max_mm"]),
            "sealed_fk_p95": max(sealed_p95) <= float(student_config["sealed_fk_p95_max_mm"]),
            "sealed_fk_max": max(sealed_max) <= float(student_config["sealed_fk_max_mm"]),
            "sealed_predictions_in_bounds": all(bool(value["sealed_actual_bounds_pass"]) for value in sealed_reports),
            "automatic_chart_classifier_disabled": student_config["automatic_chart_classifier"] is False,
        },
        semantics="formal_shell_known_chart_student_learnability",
        source_dataset_sha256=actual_dataset_hash,
        model_lock_sha256=sha256_file(lock_stage / "model_lock.json"),
        validation_reports=reports,
        sealed_reports=sealed_reports,
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "gate_pass": bool(gate["gate_pass"]),
        "stopped_after": None if gate["gate_pass"] else "student_summary",
    }
    atomic_write_json(output_root / "run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SOURCE_ROOT / "configs/bacra_v13_ellipsoidal_shell_atlas.yaml"))
    parser.add_argument("--project-root", default=str(project_root_from(SOURCE_ROOT)))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report.get("gate_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
