#!/usr/bin/env python3
"""Train a Gold-set-aware Student against certified cyclic Teacher graphs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import run_bacra_v12 as v12
import run_bacra_v12_7_canonical_student as canonical_runner
import run_bacra_v12_8_geometry_holdout as v128
from quasi_exp.teacher.bacra_canonical_student import (
    evaluate_canonical_predictions,
)
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.gold_set_student import (
    assign_cyclic_gold_section,
    fine_tune_gold_set_student,
    save_uncompiled_model,
)


PROTOCOL_ID = (
    "branch-aware-canonical-region-atlas-v12.11-"
    "gold-set-aware-student"
)
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)
PREDICTED_BETA_COLUMNS = tuple(
    f"predicted_{name}" for name in BETA_COLUMNS
)


def _source_path(project_root: Path, value: str) -> Path:
    return v12._source_path(project_root, str(value))


def _roots(
    config: Mapping[str, Any], project_root: Path
) -> tuple[Path, Path]:
    policy = config["gold_set_student"]
    return (
        _source_path(project_root, str(policy["source_v12_9_root"])),
        _source_path(project_root, str(policy["source_v12_10_root"])),
    )


def _gate(
    path: Path,
    checks: Mapping[str, bool],
    *,
    semantics: str,
    **evidence: Any,
) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "gate_semantics": str(semantics),
        "claim_scope": "simulation_gold_set_aware_student_refit",
        "deployment_claim_gate_pass": False,
        "fresh_geometry_generalization_claim": False,
        "historical_sealed_artifacts_read": False,
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, payload)
    return payload


def _family_ids(source_v12_9: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.parent.name
            for path in (
                source_v12_9 / "01_teacher_reference/families"
            ).glob("core_interpolation_*/report.json")
        )
    )


def stage_protocol(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / "00_protocol"
    stage.mkdir(parents=True, exist_ok=True)
    source_v12_9, source_v12_10 = _roots(config, project_root)
    policy = config["gold_set_student"]
    v129_gate_path = source_v12_9 / "02_summary/gate.json"
    v1210_gate_path = source_v12_10 / "03_summary/gate.json"
    v1210_marker_path = (
        source_v12_10
        / "V12_10_STUDENT_GOLD_SET_BRANCH_DIAGNOSIS_COMPLETED.json"
    )
    source_paths = {
        "v12_9_summary_gate": v129_gate_path,
        "v12_10_summary_gate": v1210_gate_path,
        "v12_10_completion_marker": v1210_marker_path,
    }
    hashes = {
        name: sha256_file(path) if path.is_file() else None
        for name, path in source_paths.items()
    }
    expected_hashes = {
        str(key): str(value)
        for key, value in policy["expected_source_sha256"].items()
    }
    family_ids = _family_ids(source_v12_9)
    family_evidence: dict[str, Any] = {}
    family_sources_valid = True
    for family_id in family_ids:
        root = (
            source_v12_9
            / "01_teacher_reference/families"
            / family_id
        )
        report_path = root / "report.json"
        reference_path = root / "reference.parquet"
        gold_path = root / "gold_candidates.parquet"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        valid = bool(
            report.get("gate_pass", False)
            and (
                report.get("independent_hard_cyclic_certificate")
                or {}
            ).get("gate_pass", False)
            and reference_path.is_file()
            and gold_path.is_file()
        )
        family_sources_valid = family_sources_valid and valid
        family_evidence[family_id] = {
            "valid": valid,
            "reference_sha256": (
                sha256_file(reference_path)
                if reference_path.is_file()
                else None
            ),
            "gold_candidates_sha256": (
                sha256_file(gold_path) if gold_path.is_file() else None
            ),
            "report_sha256": (
                sha256_file(report_path)
                if report_path.is_file()
                else None
            ),
        }

    seeds = tuple(map(int, config["canonical_student"]["formal_seeds"]))
    model_paths = {
        str(seed): v128._model_paths(config, project_root, seed)[0]
        for seed in seeds
    }
    prediction_paths = {
        str(seed): (
            source_v12_10
            / f"01_predictions/seed_{seed}/predictions.parquet"
        )
        for seed in seeds
    }
    model_prediction_sources_exist = all(
        path.is_file()
        for path in (*model_paths.values(), *prediction_paths.values())
    )
    checks = {
        "source_artifacts_exist": all(
            path.is_file() for path in source_paths.values()
        ),
        "source_hashes_match": hashes == expected_hashes,
        "v12_9_teacher_gate_pass": bool(
            json.loads(v129_gate_path.read_text(encoding="utf-8")).get(
                "gate_pass", False
            )
            if v129_gate_path.is_file()
            else False
        ),
        "v12_10_diagnosis_gate_pass": bool(
            json.loads(v1210_gate_path.read_text(encoding="utf-8")).get(
                "gate_pass", False
            )
            if v1210_gate_path.is_file()
            else False
        ),
        "four_certified_core_gold_graphs_present": bool(
            len(family_ids) == int(policy["core_family_count"])
            and family_sources_valid
        ),
        "five_initial_student_sources_present": (
            model_prediction_sources_exist
        ),
        "training_python_exists": python.is_file(),
    }
    atomic_write_json(stage / "frozen_config.json", dict(config))
    atomic_write_json(
        stage / "runtime.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "requested_training_python": str(python),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                )
            },
        },
    )
    atomic_write_json(
        stage / "source_manifest.json",
        {
            "source_paths": {
                name: str(path) for name, path in source_paths.items()
            },
            "source_sha256": hashes,
            "family_evidence": family_evidence,
            "initial_models": {
                seed: {
                    "path": str(path),
                    "sha256": sha256_file(path)
                    if path.is_file()
                    else None,
                }
                for seed, path in model_paths.items()
            },
            "initial_predictions": {
                seed: {
                    "path": str(path),
                    "sha256": sha256_file(path)
                    if path.is_file()
                    else None,
                }
                for seed, path in prediction_paths.items()
            },
        },
    )
    return _gate(
        stage / "gate.json",
        checks,
        semantics="v12_9_gold_graph_and_v12_10_student_admission",
        family_ids=list(family_ids),
        seeds=list(seeds),
        source_sha256=hashes,
    )


def _gpu_probe(python: Path) -> dict[str, Any]:
    command = [
        str(python),
        "-c",
        (
            "import json,tensorflow as tf;"
            "g=tf.config.list_physical_devices('GPU');"
            "print(json.dumps({'tensorflow':tf.__version__,"
            "'gpus':[x.name for x in g]}))"
        ),
    ]
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    process = subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
        check=False,
    )
    payload: dict[str, Any] = {
        "return_code": int(process.returncode),
        "gpus": [],
        "stderr_tail": process.stderr[-2000:],
    }
    if process.returncode == 0 and process.stdout.strip():
        payload.update(json.loads(process.stdout.strip().splitlines()[-1]))
    payload["gpu_available"] = bool(payload.get("gpus"))
    return payload


def _trajectory_metrics(prediction: np.ndarray) -> dict[str, float]:
    beta = np.asarray(prediction, dtype=float)
    edge = np.rad2deg(
        np.sqrt(
            np.mean(
                np.square(np.roll(beta, -1, axis=0) - beta),
                axis=1,
            )
        )
    )
    velocity = np.roll(beta, -1, axis=0) - beta
    acceleration = np.rad2deg(
        np.sqrt(
            np.mean(
                np.square(np.roll(velocity, -1, axis=0) - velocity),
                axis=1,
            )
        )
    )
    return {
        "phase_beta_rms_p95_deg": float(np.percentile(edge, 95)),
        "phase_beta_rms_max_deg": float(np.max(edge)),
        "acceleration_beta_rms_p95_deg": float(
            np.percentile(acceleration, 95)
        ),
        "seam_beta_rms_deg": float(edge[-1]),
    }


def _prediction_frame(
    reference: pd.DataFrame,
    prediction: np.ndarray,
    *,
    seed: int,
) -> pd.DataFrame:
    frame = reference.loc[
        :, ["family_id", "group_id", "phase_idx", *XYZ_COLUMNS]
    ].copy()
    for index, name in enumerate(PREDICTED_BETA_COLUMNS):
        frame[name] = prediction[:, index]
    frame["seed"] = int(seed)
    return frame


def _make_assignment_training_frame(
    reference: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    ordered_reference = reference.sort_values(
        "phase_idx", kind="stable"
    ).reset_index(drop=True)
    ordered_assignment = assignment.sort_values(
        "phase_idx", kind="stable"
    ).reset_index(drop=True)
    frame = ordered_reference.loc[
        :, ["family_id", "group_id", "phase_idx", *XYZ_COLUMNS]
    ].copy()
    frame.loc[:, BETA_COLUMNS] = ordered_assignment.loc[
        :, BETA_COLUMNS
    ].to_numpy(dtype=float)
    frame["gold_candidate_idx"] = ordered_assignment[
        "candidate_idx"
    ].to_numpy(dtype=int)
    frame["gold_residual_mm"] = ordered_assignment[
        "residual_mm"
    ].to_numpy(dtype=float)
    frame["gold_minimum_joint_margin_deg"] = ordered_assignment[
        "minimum_joint_margin_deg"
    ].to_numpy(dtype=float)
    return frame


def train_worker(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    policy = config["gold_set_student"]
    seed = int(args.seed)
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(
        int(config["parallel"]["student_intraop_threads"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(config["parallel"]["student_interop_threads"])
    )
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    project_root = Path(args.project_root).resolve()
    source_v12_9, source_v12_10 = _roots(config, project_root)
    family_ids = _family_ids(source_v12_9)
    references: dict[str, pd.DataFrame] = {}
    gold_sets: dict[str, pd.DataFrame] = {}
    for family_id in family_ids:
        root = (
            source_v12_9
            / "01_teacher_reference/families"
            / family_id
        )
        references[family_id] = pd.read_parquet(
            root / "reference.parquet"
        ).sort_values("phase_idx", kind="stable")
        gold_sets[family_id] = pd.read_parquet(
            root / "gold_candidates.parquet"
        )

    initial_prediction = pd.read_parquet(
        source_v12_10
        / f"01_predictions/seed_{seed}/predictions.parquet"
    ).sort_values(["family_id", "phase_idx"], kind="stable")
    current_prediction = {
        family_id: initial_prediction.loc[
            initial_prediction["family_id"].eq(family_id),
            PREDICTED_BETA_COLUMNS,
        ].to_numpy(dtype=float)
        for family_id in family_ids
    }
    initial_model_path = v128._model_paths(
        config, project_root, seed
    )[0]
    model = tf.keras.models.load_model(
        initial_model_path, compile=False
    )
    geometry = canonical_runner._geometry(config, project_root)
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    round_reports: list[dict[str, Any]] = []
    for round_index in range(int(policy["em_rounds"])):
        assignment_frames: list[pd.DataFrame] = []
        assignment_reports: list[dict[str, Any]] = []
        for family_id in family_ids:
            assignment, report = assign_cyclic_gold_section(
                current_prediction[family_id],
                gold_sets[family_id],
                max_transition_deg=float(
                    policy["hard_transition_deg"]
                ),
            )
            if not bool(report.get("success", False)):
                raise RuntimeError(
                    f"no hard cyclic Gold assignment for {family_id}"
                )
            training = _make_assignment_training_frame(
                references[family_id], assignment
            )
            assignment_frames.append(training)
            assignment_reports.append(
                {"family_id": family_id, **report}
            )
        training_frame = pd.concat(
            assignment_frames, ignore_index=True
        )
        round_root = destination / f"round_{round_index:02d}"
        round_root.mkdir(parents=True, exist_ok=True)
        v12._atomic_parquet(
            training_frame, round_root / "assigned_gold_section.parquet"
        )
        model, history, training_report = fine_tune_gold_set_student(
            model,
            training_frame,
            geometry=geometry,
            seed=seed + round_index * 1000,
            learning_rate=float(policy["learning_rate"]),
            batch_size=int(policy["batch_size"]),
            max_epochs=int(policy["max_epochs_per_round"]),
            patience=int(policy["patience"]),
            beta_loss_scale_deg=tuple(
                map(float, policy["beta_loss_scale_deg"])
            ),
            lambda_fk=float(policy["lambda_fk"]),
            lambda_margin=float(policy["lambda_margin"]),
            lambda_continuity=float(policy["lambda_continuity"]),
            margin_target_deg=float(policy["margin_target_deg"]),
            transition_target_deg=float(
                policy["hard_transition_deg"]
            ),
        )
        history.to_csv(round_root / "history.csv", index=False)
        save_uncompiled_model(model, round_root / "model.keras")
        ordered_reference = pd.concat(
            [references[name] for name in family_ids],
            ignore_index=True,
        ).sort_values(["family_id", "phase_idx"], kind="stable")
        prediction = np.asarray(
            model.predict(
                ordered_reference.loc[:, XYZ_COLUMNS].to_numpy(
                    dtype=np.float32
                ),
                batch_size=int(policy["batch_size"]),
                verbose=0,
            ),
            dtype=float,
        )
        offset = 0
        for family_id in family_ids:
            count = len(references[family_id])
            current_prediction[family_id] = prediction[
                offset : offset + count
            ]
            offset += count
        round_report = {
            "round_index": round_index,
            "assignments": assignment_reports,
            "training": training_report,
        }
        atomic_write_json(round_root / "report.json", round_report)
        round_reports.append(round_report)

    save_uncompiled_model(model, destination / "model.keras")
    final_assignments: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    family_reports: list[dict[str, Any]] = []
    admission = config["exploratory_admission"]
    trajectory_gate = config["student_trajectory_gate"]
    for family_id in family_ids:
        prediction = current_prediction[family_id]
        assignment, assignment_report = assign_cyclic_gold_section(
            prediction,
            gold_sets[family_id],
            max_transition_deg=float(policy["hard_transition_deg"]),
        )
        training = _make_assignment_training_frame(
            references[family_id], assignment
        )
        metrics = evaluate_canonical_predictions(
            training, prediction, geometry=geometry
        )
        trajectory = _trajectory_metrics(prediction)
        gold_joint_p95 = list(
            map(
                float,
                metrics["validation_beta_abs_p95_by_joint_deg"],
            )
        )
        checks = {
            "hard_cyclic_gold_assignment": bool(
                assignment_report.get("success", False)
            ),
            "all_joint_abs_p95_below_limit": bool(
                max(gold_joint_p95)
                < float(admission["joint_abs_p95_max_deg"])
            ),
            "fk_p95_below_limit": bool(
                float(metrics["validation_fk_p95_mm"])
                < float(admission["fk_p95_max_mm"])
            ),
            "fk_max_below_limit": bool(
                float(metrics["validation_fk_max_mm"])
                < float(admission["fk_max_mm"])
            ),
            "minimum_joint_margin_above_limit": bool(
                float(metrics["predicted_minimum_joint_margin_deg"])
                > float(admission["minimum_joint_margin_min_deg"])
            ),
        }
        trajectory_checks = {
            "phase_p95": bool(
                trajectory["phase_beta_rms_p95_deg"]
                <= float(trajectory_gate["phase_beta_rms_p95_deg"])
            ),
            "phase_max": bool(
                trajectory["phase_beta_rms_max_deg"]
                <= float(trajectory_gate["phase_beta_rms_max_deg"])
            ),
            "acceleration_p95": bool(
                trajectory["acceleration_beta_rms_p95_deg"]
                <= float(
                    trajectory_gate[
                        "acceleration_beta_rms_p95_deg"
                    ]
                )
            ),
            "seam": bool(
                trajectory["seam_beta_rms_deg"]
                <= float(trajectory_gate["seam_beta_rms_deg"])
            ),
        }
        family_report = {
            "family_id": family_id,
            "seed": seed,
            "checks": checks,
            "student_family_admission_gate_pass": bool(
                all(checks.values())
            ),
            "trajectory_checks": trajectory_checks,
            "student_cyclic_trajectory_gate_pass": bool(
                all(trajectory_checks.values())
            ),
            "gold_assignment": assignment_report,
            **metrics,
            **trajectory,
        }
        family_reports.append(family_report)
        training["seed"] = seed
        final_assignments.append(training)
        prediction_frames.append(
            _prediction_frame(
                references[family_id], prediction, seed=seed
            )
        )

    assignments = pd.concat(final_assignments, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    v12._atomic_parquet(
        assignments, destination / "final_assigned_gold_sections.parquet"
    )
    v12._atomic_parquet(
        predictions, destination / "predictions.parquet"
    )
    seed_admission_pass = bool(
        all(
            row["student_family_admission_gate_pass"]
            and row["student_cyclic_trajectory_gate_pass"]
            for row in family_reports
        )
    )
    report = {
        "seed": seed,
        "em_round_count": int(policy["em_rounds"]),
        "initial_model_path": str(initial_model_path),
        "initial_model_sha256": sha256_file(initial_model_path),
        "rounds": round_reports,
        "families": family_reports,
        "seed_admission_gate_pass": seed_admission_pass,
        "model_sha256": sha256_file(destination / "model.keras"),
        "predictions_sha256": sha256_file(
            destination / "predictions.parquet"
        ),
        "assigned_gold_sections_sha256": sha256_file(
            destination / "final_assigned_gold_sections.parquet"
        ),
        "device": (
            "gpu:0"
            if tf.config.list_physical_devices("GPU")
            and not args.force_cpu
            else "cpu"
        ),
    }
    atomic_write_json(destination / "report.json", report)
    print(
        json.dumps(
            {
                "seed": seed,
                "seed_admission_gate_pass": seed_admission_pass,
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


def _flatten_family(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "checks",
            "trajectory_checks",
            "gold_assignment",
            "validation_beta_abs_p50_by_joint_deg",
            "validation_beta_abs_p95_by_joint_deg",
            "validation_beta_abs_max_by_joint_deg",
            "validation_beta_signed_bias_by_joint_deg",
        }
    } | {
        f"gold_abs_p95_{joint}_deg": float(value)
        for joint, value in zip(
            BETA_COLUMNS,
            row["validation_beta_abs_p95_by_joint_deg"],
        )
    } | {
        f"assignment_{key}": value
        for key, value in row["gold_assignment"].items()
    }


def stage_train(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    protocol_gate = json.loads(
        (output_root / "00_protocol/gate.json").read_text(
            encoding="utf-8"
        )
    )
    stage = output_root / "01_training"
    stage.mkdir(parents=True, exist_ok=True)
    if not bool(protocol_gate["gate_pass"]):
        return _gate(
            stage / "gate.json",
            {"protocol_gate_pass": False},
            semantics="gold_set_student_training_completion",
            stopped_before_compute=True,
        )
    gpu_probe = _gpu_probe(python)
    seeds = tuple(map(int, config["canonical_student"]["formal_seeds"]))
    gpu = bool(gpu_probe["gpu_available"])
    requested_workers = (
        1 if gpu else int(config["parallel"]["student_cpu_workers"])
    )
    commands: list[tuple[str, Sequence[str], Path]] = []
    for seed in seeds:
        task_id = f"seed_{seed}"
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "--worker",
            "train",
            "--config",
            str(config["config_path"]),
            "--preset",
            str(config["preset"]),
            "--project-root",
            str(project_root),
            "--seed",
            str(seed),
            "--worker-output",
            str(stage / task_id),
        ]
        if not gpu:
            command.append("--force-cpu")
        commands.append(
            (task_id, command, stage / "logs" / f"{task_id}.log")
        )
    started = time.time()
    parallel = v12._run_subprocess_tasks(
        commands,
        requested_workers=requested_workers,
        manifest_path=stage / "parallel_manifest.json",
    )
    reports = [
        json.loads(
            (stage / f"seed_{seed}/report.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    ]
    family_rows = [
        _flatten_family(family)
        for report in reports
        for family in report["families"]
    ]
    metrics = pd.DataFrame(family_rows).sort_values(
        ["family_id", "seed"], kind="stable"
    )
    v12._atomic_parquet(
        metrics, stage / "metrics_per_family_seed.parquet"
    )
    seed_pass_count = int(
        sum(bool(row["seed_admission_gate_pass"]) for row in reports)
    )
    required = int(config["gold_set_student"]["required_seed_passes"])
    training_checks = {
        "protocol_gate_pass": bool(protocol_gate["gate_pass"]),
        "all_five_seed_tasks_complete": len(reports) == 5,
        "all_hard_cyclic_assignments_succeeded": bool(
            metrics["assignment_success"].all()
        ),
        "all_training_models_saved": all(
            (stage / f"seed_{seed}/model.keras").is_file()
            for seed in seeds
        ),
    }
    student_admission_gate_pass = seed_pass_count >= required
    return _gate(
        stage / "gate.json",
        training_checks,
        semantics="gold_set_student_training_completion",
        training_completed=True,
        student_admission_gate_pass=student_admission_gate_pass,
        seed_admission_pass_count=seed_pass_count,
        required_seed_passes=required,
        gpu_probe=gpu_probe,
        device_plan={
            "device": "gpu:0" if gpu else "cpu",
            "requested_workers": requested_workers,
            "effective_workers": min(requested_workers, len(seeds)),
            "seed_order": list(seeds),
            "gpu_seed_execution_serial": gpu,
            "cpu_seed_execution_parallel": not gpu,
        },
        parallel_evidence=parallel,
        wall_time_s=float(time.time() - started),
        metrics_sha256=sha256_file(
            stage / "metrics_per_family_seed.parquet"
        ),
    )


def stage_summary(
    config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    stage = output_root / "02_summary"
    stage.mkdir(parents=True, exist_ok=True)
    protocol_gate = json.loads(
        (output_root / "00_protocol/gate.json").read_text(
            encoding="utf-8"
        )
    )
    training_gate = json.loads(
        (output_root / "01_training/gate.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = pd.read_parquet(
        output_root / "01_training/metrics_per_family_seed.parquet"
    )
    summary = (
        metrics.groupby("family_id", sort=True)
        .agg(
            seed_count=("seed", "size"),
            seed_admission_pass_count=(
                "student_family_admission_gate_pass",
                "sum",
            ),
            worst_gold_assignment_rms_p95_deg=(
                "assignment_assignment_rms_p95_deg",
                "max",
            ),
            worst_fk_p95_mm=("validation_fk_p95_mm", "max"),
            worst_fk_max_mm=("validation_fk_max_mm", "max"),
            minimum_predicted_margin_deg=(
                "predicted_minimum_joint_margin_deg",
                "min",
            ),
            worst_phase_transition_max_deg=(
                "phase_beta_rms_max_deg",
                "max",
            ),
            worst_seam_deg=("seam_beta_rms_deg", "max"),
        )
        .reset_index()
    )
    summary.to_csv(stage / "family_summary.csv", index=False)
    gate = _gate(
        stage / "gate.json",
        {
            "protocol_gate_pass": bool(protocol_gate["gate_pass"]),
            "five_seed_training_complete": bool(
                training_gate["gate_pass"]
            ),
        },
        semantics="v12_11_training_completion",
        training_completed=True,
        student_admission_gate_pass=bool(
            training_gate["student_admission_gate_pass"]
        ),
        seed_admission_pass_count=int(
            training_gate["seed_admission_pass_count"]
        ),
        required_seed_passes=int(
            training_gate["required_seed_passes"]
        ),
        family_summary_sha256=sha256_file(
            stage / "family_summary.csv"
        ),
        next_required_checkpoint=(
            "freeze_selected_models_then_evaluate_on_newly_generated_"
            "decision_independent_full_loop_task_geometries"
        ),
    )
    atomic_write_json(
        output_root
        / str(config["gold_set_student"]["completion_marker"]),
        {
            "protocol_id": PROTOCOL_ID,
            "training_completed": True,
            "student_admission_gate_pass": bool(
                training_gate["student_admission_gate_pass"]
            ),
            "seed_admission_pass_count": int(
                training_gate["seed_admission_pass_count"]
            ),
            "deployment_claim_gate_pass": False,
            "fresh_geometry_generalization_claim": False,
            "historical_sealed_artifacts_read": False,
        },
    )
    return gate


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, args.preset)
    if str(config["protocol_id"]) != PROTOCOL_ID:
        raise ValueError("unexpected V12.11 protocol_id")
    project_root = Path(args.project_root).resolve()
    output_root = (
        Path(args.output).resolve()
        if args.output
        else _source_path(project_root, str(config["output_root"]))
    )
    python = Path(args.python).resolve()
    stages = {
        "protocol": lambda: stage_protocol(
            config, project_root, output_root, python=python
        ),
        "train": lambda: stage_train(
            config, project_root, output_root, python=python
        ),
        "summary": lambda: stage_summary(config, output_root),
    }
    requested = tuple(stages) if args.stage == "all" else (args.stage,)
    reports: dict[str, Any] = {}
    gate_paths = {
        "protocol": "00_protocol/gate.json",
        "train": "01_training/gate.json",
        "summary": "02_summary/gate.json",
    }
    for name in requested:
        path = output_root / gate_paths[name]
        report = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else stages[name]()
        )
        reports[name] = report
        if not bool(report["gate_pass"]):
            break
    stage_gates = {
        name: bool(
            json.loads(
                (output_root / path).read_text(encoding="utf-8")
            )["gate_pass"]
        )
        for name, path in gate_paths.items()
        if (output_root / path).is_file()
    }
    run_report = {
        "protocol_id": PROTOCOL_ID,
        "output_root": str(output_root),
        "requested_stage": str(args.stage),
        "stage_gate_pass": stage_gates,
        "student_admission_gate_pass": bool(
            reports.get("summary", {}).get(
                "student_admission_gate_pass",
                reports.get("train", {}).get(
                    "student_admission_gate_pass", False
                ),
            )
        ),
        "deployment_claim_gate_pass": False,
        "fresh_geometry_generalization_claim": False,
        "historical_sealed_artifacts_read": False,
    }
    atomic_write_json(output_root / "run_report.json", run_report)
    return run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            SOURCE_ROOT / "configs/bacra_v12_11_gold_set_student.yaml"
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(v12.project_root_from(SOURCE_ROOT)),
    )
    parser.add_argument("--output")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--preset", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument(
        "--stage",
        choices=("all", "protocol", "train", "summary"),
        default="all",
    )
    parser.add_argument("--worker", choices=("train",))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-output")
    parser.add_argument("--force-cpu", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker == "train":
        return train_worker(args)
    report = run(args)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
