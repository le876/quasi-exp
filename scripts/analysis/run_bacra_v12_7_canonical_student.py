#!/usr/bin/env python3
"""Train a canonical-beta V12.7 Student without reopening sealed data."""

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
from quasi_exp.teacher.bacra_canonical_student import (
    evaluate_canonical_predictions,
    select_pilot_candidate,
)
from quasi_exp.teacher.bacra_student import StudentGeometry
from quasi_exp.teacher.dense_chart_sampling import XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file


PROTOCOL_ID = "branch-aware-canonical-region-atlas-v12.7-canonical-student"
SUPPORTED_PROTOCOL_PREFIX = "branch-aware-canonical-region-atlas-v12.7"
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)


def _source_paths(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    policy = config["canonical_student"]
    source_root = v12._source_path(
        project_root, str(policy["source_root"])
    )
    return {
        "public_dataset": source_root / str(policy["public_dataset"]),
        "baseline_metrics": source_root / str(policy["baseline_metrics"]),
    }


def _geometry(
    config: Mapping[str, Any], project_root: Path
) -> StudentGeometry:
    environment = v12.load_environment(
        project_root,
        v12._source_path(project_root, str(config["robot_config"])),
    )
    return StudentGeometry(
        lengths_m=environment.lengths_m,
        p_end_local_m=environment.p_end_local_m,
        theta_sign=environment.theta_sign,
        beta_bounds_rad=environment.bounds,
    )


def _gate(
    path: Path,
    checks: Mapping[str, bool],
    *,
    semantics: str,
    **evidence: Any,
) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    frozen_config = path.parent.parent / "00_protocol/frozen_config.json"
    protocol_id = PROTOCOL_ID
    if frozen_config.is_file():
        protocol_id = str(
            json.loads(frozen_config.read_text(encoding="utf-8"))[
                "protocol_id"
            ]
        )
    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "gate_semantics": semantics,
        "claim_scope": (
            "simulation_public_validation_canonical_student_diagnostic"
        ),
        "deployment_claim_gate_pass": False,
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, payload)
    return payload


def stage_protocol(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / "00_protocol"
    stage.mkdir(parents=True, exist_ok=True)
    paths = _source_paths(config, project_root)
    expected = {
        "public_dataset": str(
            config["canonical_student"][
                "expected_public_dataset_sha256"
            ]
        ),
        "baseline_metrics": str(
            config["canonical_student"][
                "expected_baseline_metrics_sha256"
            ]
        ),
    }
    actual = {
        name: sha256_file(path) if path.is_file() else None
        for name, path in paths.items()
    }
    dataset = (
        pd.read_parquet(paths["public_dataset"])
        if paths["public_dataset"].is_file()
        else pd.DataFrame()
    )
    role_counts = (
        dataset["split_role"].value_counts().sort_index().to_dict()
        if "split_role" in dataset
        else {}
    )
    checks = {
        "source_artifacts_exist": all(path.is_file() for path in paths.values()),
        "source_hashes_match": actual == expected,
        "train_rows_present": int(role_counts.get("train", 0)) > 0,
        "validation_rows_present": int(role_counts.get("validation", 0)) > 0,
        "sealed_rows_absent": int(role_counts.get("sealed_test", 0)) == 0,
        "gpu_python_exists": python.is_file(),
    }
    atomic_write_json(stage / "frozen_config.json", dict(config))
    atomic_write_json(
        stage / "runtime.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "tensorflow",
                )
            },
            "requested_python": str(python),
            "source_paths": {
                name: str(path) for name, path in paths.items()
            },
            "source_sha256": actual,
            "split_role_counts": role_counts,
        },
    )
    return _gate(
        stage / "gate.json",
        checks,
        semantics="canonical_student_source_admission",
        source_paths={name: str(path) for name, path in paths.items()},
        source_sha256=actual,
        split_role_counts=role_counts,
        sealed_evaluation_opened=False,
    )


def _gpu_smoke(python: Path) -> dict[str, Any]:
    code = (
        "import json,tensorflow as tf;"
        "g=tf.config.list_physical_devices('GPU');"
        "x=tf.ones((512,512));"
        "y=tf.linalg.matmul(x,x);"
        "print(json.dumps({'tensorflow':tf.__version__,"
        "'gpus':[d.name for d in g],'result_device':y.device,"
        "'result_sum':float(tf.reduce_sum(y).numpy())}))"
    )
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    probe = subprocess.run(
        [str(python), "-c", code],
        cwd=SOURCE_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"TensorFlow GPU smoke failed:\n{probe.stderr[-4000:]}"
        )
    payload = json.loads(probe.stdout.strip().splitlines()[-1])
    if not payload["gpus"] or "GPU:0" not in payload["result_device"]:
        raise RuntimeError(f"TensorFlow did not execute on GPU: {payload}")
    return payload


def _gpu_sample() -> tuple[float, float] | None:
    sample = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if sample.returncode != 0:
        return None
    values = sample.stdout.strip().splitlines()[0].split(",")
    return float(values[0].strip()), float(values[1].strip())


def _lambda_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _beta_loss_scale_values(config: Mapping[str, Any]) -> tuple[float, ...]:
    configured = config["canonical_student"]["beta_loss_scale_deg"]
    values = np.asarray(configured, dtype=float)
    if values.ndim == 0:
        return (float(values),)
    return tuple(map(float, values.reshape(-1)))


def _beta_loss_scale_label(config: Mapping[str, Any]) -> str:
    values = _beta_loss_scale_values(config)
    if len(values) == 1:
        return f"uniform{_lambda_label(values[0])}deg"
    if len(values) == 6 and values[:4] == (1.0, 1.0, 1.0, 1.0) and values[5] == 1.0:
        return f"beta5scale{_lambda_label(values[4])}deg"
    return "jointweighted"


def _run_training_task(
    *,
    python: Path,
    config: Mapping[str, Any],
    project_root: Path,
    dataset: Path,
    seed: int,
    lambda_fk: float,
    output: Path,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        str(python),
        str(SOURCE_ROOT / "scripts/analysis/run_bacra_v12.py"),
        "_student-worker",
        "--config",
        str(config["config_path"]),
        "--preset",
        "pilot",
        "--project-root",
        str(project_root),
        "--dataset",
        str(dataset),
        "--strategy",
        "static_xyz_to_beta6",
        "--seed",
        str(int(seed)),
        "--lambda-fk",
        str(float(lambda_fk)),
    ]
    command.extend(
        [
            "--beta-loss-scale-deg",
            *(str(value) for value in _beta_loss_scale_values(config)),
            "--output",
            str(output),
        ]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    utilization: list[float] = []
    memory_mib: list[float] = []
    with log_path.open("w", encoding="utf-8") as log:
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        process = subprocess.Popen(
            command,
            cwd=SOURCE_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        while process.poll() is None:
            sample = _gpu_sample()
            if sample is not None:
                utilization.append(sample[0])
                memory_mib.append(sample[1])
            time.sleep(1.0)
    return_code = int(process.returncode)
    if return_code != 0:
        tail = log_path.read_text(
            encoding="utf-8", errors="replace"
        )[-5000:]
        raise RuntimeError(
            f"Student task seed={seed} lambda={lambda_fk} failed:\n{tail}"
        )
    return {
        "seed": int(seed),
        "lambda_fk": float(lambda_fk),
        "return_code": return_code,
        "wall_time_s": float(time.time() - started),
        "gpu_utilization_mean_percent": (
            float(np.mean(utilization)) if utilization else None
        ),
        "gpu_utilization_max_percent": (
            float(np.max(utilization)) if utilization else None
        ),
        "gpu_peak_memory_mib": (
            float(np.max(memory_mib)) if memory_mib else None
        ),
        "gpu_sample_count": int(len(utilization)),
        "log_path": str(log_path),
    }


def _evaluate_models(
    tasks: Sequence[Mapping[str, Any]],
    *,
    dataset: Path,
    geometry: StudentGeometry,
    batch_size: int,
) -> list[dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    frame = pd.read_parquet(dataset)
    validation = frame.loc[
        frame["split_role"].eq("validation")
    ].reset_index(drop=True)
    features = validation.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        output = Path(str(task["output"]))
        model = tf.keras.models.load_model(
            output / "model.keras", compile=False
        )
        prediction = np.asarray(
            model.predict(
                features, batch_size=int(batch_size), verbose=0
            ),
            dtype=float,
        )
        report = {
            **evaluate_canonical_predictions(
                validation, prediction, geometry=geometry
            ),
            "task_id": str(task["task_id"]),
            "seed": int(task["seed"]),
            "lambda_fk": float(task["lambda_fk"]),
            "model_path": str(output / "model.keras"),
        }
        atomic_write_json(output / "canonical_metrics.json", report)
        rows.append(report)
    return rows


def stage_pilot(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    protocol_gate = json.loads(
        (output_root / "00_protocol/gate.json").read_text(encoding="utf-8")
    )
    stage = output_root / "01_pilot"
    stage.mkdir(parents=True, exist_ok=True)
    gpu_smoke = _gpu_smoke(python)
    source = _source_paths(config, project_root)
    policy = config["canonical_student"]
    tasks: list[dict[str, Any]] = []
    hardware: list[dict[str, Any]] = []
    for lambda_fk in map(float, policy["lambda_fk_candidates"]):
        label = _lambda_label(lambda_fk)
        task_id = (
            f"{_beta_loss_scale_label(config)}_lambda{label}_"
            f"seed{int(policy['pilot_seed'])}"
        )
        output = stage / task_id
        task = {
            "task_id": task_id,
            "seed": int(policy["pilot_seed"]),
            "lambda_fk": lambda_fk,
            "output": str(output),
        }
        hardware.append(
            _run_training_task(
                python=python,
                config=config,
                project_root=project_root,
                dataset=source["public_dataset"],
                seed=task["seed"],
                lambda_fk=lambda_fk,
                output=output,
                log_path=stage / "logs" / f"{task_id}.log",
            )
        )
        tasks.append(task)
    rows = _evaluate_models(
        tasks,
        dataset=source["public_dataset"],
        geometry=_geometry(config, project_root),
        batch_size=int(config["student"]["batch_size"]),
    )
    selected, evaluated = select_pilot_candidate(
        rows,
        fk_p95_max_mm=float(policy["fk_p95_max_mm"]),
        fk_max_mm=float(policy["fk_max_mm"]),
        beta_rms_p95_max_deg=float(
            policy["beta_rms_p95_max_deg"]
        ),
        joint_abs_p95_max_deg=float(
            policy["joint_abs_p95_max_deg"]
        ),
    )
    v12._atomic_parquet(pd.DataFrame(evaluated), stage / "pilot_metrics.parquet")
    atomic_write_json(stage / "hardware.json", hardware)
    if selected is not None:
        atomic_write_json(stage / "selected_config.json", selected)
    candidate_count = len(list(policy["lambda_fk_candidates"]))
    checks = {
        "protocol_gate_pass": bool(protocol_gate["gate_pass"]),
        "lambda_candidates_complete": len(evaluated) == candidate_count,
        "gpu_execution_verified": bool(gpu_smoke["gpus"]),
        "fk_eligible_candidate_exists": selected is not None,
        "selected_candidate_meets_canonical_beta_target": bool(
            selected is not None
            and selected["canonical_beta_target_pass"]
        ),
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="canonical_student_pilot_selection",
        gpu_smoke=gpu_smoke,
        selected_candidate=selected,
        lambda_fk_candidates=list(
            map(float, policy["lambda_fk_candidates"])
        ),
        lambda_candidate_count=candidate_count,
        sealed_evaluation_opened=False,
    )


def stage_formal(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    pilot_gate = json.loads(
        (output_root / "01_pilot/gate.json").read_text(encoding="utf-8")
    )
    if not bool(pilot_gate["gate_pass"]):
        raise RuntimeError("formal training requires a passed pilot Gate")
    selected = json.loads(
        (output_root / "01_pilot/selected_config.json").read_text(
            encoding="utf-8"
        )
    )
    stage = output_root / "02_formal"
    stage.mkdir(parents=True, exist_ok=True)
    gpu_smoke = _gpu_smoke(python)
    source = _source_paths(config, project_root)
    policy = config["canonical_student"]
    selected_lambda = float(selected["lambda_fk"])
    tasks: list[dict[str, Any]] = []
    hardware: list[dict[str, Any]] = []
    for seed in map(int, policy["formal_seeds"]):
        task_id = f"lambda{_lambda_label(selected_lambda)}_seed{seed}"
        output = stage / f"seed_{seed}"
        task = {
            "task_id": task_id,
            "seed": seed,
            "lambda_fk": selected_lambda,
            "output": str(output),
        }
        hardware.append(
            _run_training_task(
                python=python,
                config=config,
                project_root=project_root,
                dataset=source["public_dataset"],
                seed=seed,
                lambda_fk=selected_lambda,
                output=output,
                log_path=stage / "logs" / f"{task_id}.log",
            )
        )
        tasks.append(task)
    rows = _evaluate_models(
        tasks,
        dataset=source["public_dataset"],
        geometry=_geometry(config, project_root),
        batch_size=int(config["student"]["batch_size"]),
    )
    evaluated: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        row["worst_joint_abs_p95_deg"] = max(
            map(float, row["validation_beta_abs_p95_by_joint_deg"])
        )
        row["seed_gate_pass"] = bool(
            float(row["validation_beta_rms_p95_deg"])
            <= float(policy["beta_rms_p95_max_deg"])
            and float(row["worst_joint_abs_p95_deg"])
            <= float(policy["joint_abs_p95_max_deg"])
            and float(row["validation_fk_p95_mm"])
            <= float(policy["fk_p95_max_mm"])
            and float(row["validation_fk_max_mm"])
            <= float(policy["fk_max_mm"])
        )
        evaluated.append(row)
    metrics = pd.DataFrame(evaluated)
    v12._atomic_parquet(metrics, stage / "formal_metrics_per_seed.parquet")
    atomic_write_json(stage / "hardware.json", hardware)
    pass_count = int(metrics["seed_gate_pass"].sum())
    median_beta = float(
        metrics["validation_beta_rms_p95_deg"].median()
    )
    baseline = float(policy["baseline_median_beta_rms_p95_deg"])
    improvement = (baseline - median_beta) / baseline
    checks = {
        "pilot_gate_pass": bool(pilot_gate["gate_pass"]),
        "five_seed_tasks_complete": len(metrics) == 5,
        "required_seed_passes": pass_count
        >= int(policy["required_seed_passes"]),
        "median_beta_improvement": improvement
        >= float(policy["median_beta_improvement_min"]),
        "gpu_execution_verified": bool(gpu_smoke["gpus"]),
    }
    gate = _gate(
        stage / "gate.json",
        checks,
        semantics="canonical_student_public_validation",
        selected_lambda_fk=selected_lambda,
        seed_pass_count=pass_count,
        required_seed_passes=int(policy["required_seed_passes"]),
        median_validation_beta_rms_p95_deg=median_beta,
        baseline_median_validation_beta_rms_p95_deg=baseline,
        median_beta_improvement_fraction=float(improvement),
        gpu_smoke=gpu_smoke,
        sealed_evaluation_opened=False,
    )
    if gate["gate_pass"]:
        marker_name = str(
            policy.get(
                "completion_marker",
                "V12_7_CANONICAL_STUDENT_DIAGNOSTIC_COMPLETED.json",
            )
        )
        atomic_write_json(
            output_root / marker_name,
            {
                "protocol_id": str(config["protocol_id"]),
                "selected_lambda_fk": selected_lambda,
                "seed_pass_count": pass_count,
                "median_validation_beta_rms_p95_deg": median_beta,
                "median_beta_improvement_fraction": float(improvement),
                "deployment_claim_gate_pass": False,
                "sealed_evaluation_opened": False,
            },
        )
    return gate


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, "pilot")
    protocol_id = str(config["protocol_id"])
    if not protocol_id.startswith(SUPPORTED_PROTOCOL_PREFIX):
        raise ValueError("unexpected canonical Student protocol_id")
    project_root = Path(args.project_root).resolve()
    output_root = (
        Path(args.output).resolve()
        if args.output
        else v12._source_path(project_root, str(config["output_root"]))
    )
    python = Path(args.python).resolve()
    stages = {
        "protocol": lambda: stage_protocol(
            config, project_root, output_root, python=python
        ),
        "pilot": lambda: stage_pilot(
            config, project_root, output_root, python=python
        ),
        "formal": lambda: stage_formal(
            config, project_root, output_root, python=python
        ),
    }
    requested = tuple(stages) if args.stage == "all" else (args.stage,)
    reports: dict[str, Any] = {}
    for name in requested:
        gate_path = output_root / {
            "protocol": "00_protocol/gate.json",
            "pilot": "01_pilot/gate.json",
            "formal": "02_formal/gate.json",
        }[name]
        if gate_path.is_file():
            report = json.loads(gate_path.read_text(encoding="utf-8"))
        else:
            report = stages[name]()
        reports[name] = report
        if not bool(report["gate_pass"]):
            break
    all_stage_gates: dict[str, bool] = {}
    for name, relative in {
        "protocol": "00_protocol/gate.json",
        "pilot": "01_pilot/gate.json",
        "formal": "02_formal/gate.json",
    }.items():
        gate_path = output_root / relative
        if gate_path.is_file():
            all_stage_gates[name] = bool(
                json.loads(gate_path.read_text(encoding="utf-8"))[
                    "gate_pass"
                ]
            )
    run_report = {
        "protocol_id": protocol_id,
        "output_root": str(output_root),
        "requested_stage": str(args.stage),
        "stage_gate_pass": all_stage_gates,
        "deployment_claim_gate_pass": False,
        "sealed_evaluation_opened": False,
    }
    atomic_write_json(output_root / "run_report.json", run_report)
    return run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            SOURCE_ROOT / "configs/bacra_v12_7_canonical_student.yaml"
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(v12.project_root_from(SOURCE_ROOT)),
    )
    parser.add_argument("--output")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument(
        "--stage",
        choices=("all", "protocol", "pilot", "formal"),
        default="all",
    )
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
