#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from joblib import dump, load


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_beta6_training_v4 as v4  # noqa: E402
from true_ellipse_family_v5_utils import parse_float_csv  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "runs" / "true_ellipse_family_expansion_v5" / "05_dataset" / "true_ellipse_family_tubes_v5.parquet"
DEFAULT_EXPANSION = REPO_ROOT / "runs" / "true_ellipse_family_expansion_v5"
DEFAULT_OUT = REPO_ROOT / "runs" / "true_ellipse_family_training_v5"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml"
DEFAULT_SEEDS = (20260713, 20260714, 20260715, 20260716, 20260717)
ALL_PHASES = ["audit", "split", "screen", "train", "sweep", "summary"]


@dataclass(frozen=True)
class WholeTrajectorySplit:
    train_idx: np.ndarray
    validation_idx: np.ndarray
    primary_idx: np.ndarray
    stretch_idx: np.ndarray
    train_trajectory_ids: tuple[str, ...]
    validation_trajectory_ids: tuple[str, ...]
    primary_trajectory_ids: tuple[str, ...]
    stretch_trajectory_ids: tuple[str, ...]
    trajectory_leakage_count: int


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_csv(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def parse_phases(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        phases = [part.strip() for part in value.split(",") if part.strip()]
    else:
        phases = [str(part).strip() for part in value if str(part).strip()]
    if phases == ["all"]:
        return list(ALL_PHASES)
    unknown = sorted(set(phases) - set(ALL_PHASES))
    if unknown:
        raise ValueError(f"unsupported V5 training phases: {unknown}")
    return phases


def make_whole_trajectory_split(
    dataset: pd.DataFrame,
    *,
    primary_radius_mm: float,
    stretch_radius_mm: float,
    validation_radius_mm: float,
) -> WholeTrajectorySplit:
    required = {"trajectory_id", "radius_mm"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"whole-trajectory split missing columns: {missing}")
    radius = dataset["radius_mm"].to_numpy(dtype=float)
    primary_mask = np.isclose(radius, float(primary_radius_mm), atol=1.0e-8)
    stretch_mask = np.isclose(radius, float(stretch_radius_mm), atol=1.0e-8)
    validation_mask = np.isclose(radius, float(validation_radius_mm), atol=1.0e-8)
    if np.any(primary_mask & stretch_mask) or np.any(primary_mask & validation_mask) or np.any(stretch_mask & validation_mask):
        raise ValueError("primary, stretch, and validation radii must be distinct")
    holdout_mask = primary_mask | stretch_mask | validation_mask
    train_idx = np.flatnonzero(~holdout_mask).astype(np.int64)
    validation_idx = np.flatnonzero(validation_mask).astype(np.int64)
    primary_idx = np.flatnonzero(primary_mask).astype(np.int64)
    stretch_idx = np.flatnonzero(stretch_mask).astype(np.int64)
    if not len(train_idx) or not len(validation_idx) or not len(primary_idx):
        raise ValueError("whole-trajectory split requires non-empty train, validation, and primary partitions")

    def ids(indices: np.ndarray) -> tuple[str, ...]:
        return tuple(sorted(dataset.iloc[indices]["trajectory_id"].astype(str).unique().tolist()))

    train_ids = ids(train_idx)
    validation_ids = ids(validation_idx)
    primary_ids = ids(primary_idx)
    stretch_ids = ids(stretch_idx)
    sets = [set(train_ids), set(validation_ids), set(primary_ids), set(stretch_ids)]
    leakage = sum(len(sets[left] & sets[right]) for left in range(len(sets)) for right in range(left + 1, len(sets)))
    if leakage:
        raise AssertionError(f"trajectory leakage across partitions: {leakage}")
    return WholeTrajectorySplit(
        train_idx=train_idx,
        validation_idx=validation_idx,
        primary_idx=primary_idx,
        stretch_idx=stretch_idx,
        train_trajectory_ids=train_ids,
        validation_trajectory_ids=validation_ids,
        primary_trajectory_ids=primary_ids,
        stretch_trajectory_ids=stretch_ids,
        trajectory_leakage_count=int(leakage),
    )


def strict_training_indices(dataset: pd.DataFrame, split: WholeTrajectorySplit) -> np.ndarray:
    if "is_centerline" not in dataset.columns:
        raise ValueError("strict training pool requires is_centerline")
    noncenter = ~dataset.iloc[split.train_idx]["is_centerline"].astype(bool).to_numpy()
    return np.asarray(split.train_idx, dtype=np.int64)[noncenter]


def strict_support_gate(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics.get("nn_p95_mm", np.inf)) <= 5.0
        and float(metrics.get("nn_max_mm", np.inf)) <= 8.0
        and float(metrics.get("tube_count_p10", -np.inf)) >= 32.0
    )


def compute_support_scan(
    dataset: pd.DataFrame,
    *,
    training_indices: np.ndarray,
    metadata: Mapping[str, Any],
    radii_mm: Iterable[float],
    n_points: int = 360,
) -> pd.DataFrame:
    training_xyz = dataset.iloc[np.asarray(training_indices, dtype=np.int64)][v4.TARGET_XYZ_COLS].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for radius_mm in parse_float_csv(radii_mm):
        target, _angle = v4.generate_exact_ellipse(metadata, amp_xy_mm=float(radius_mm), n_points=int(n_points))
        metrics = v4.compute_support_metrics(training_xyz, target, radius_mm=15.0)
        rows.append(
            {
                "radius_mm": float(radius_mm),
                "amp_z_mm": 1.5 * float(radius_mm),
                **metrics,
                "strict_support_gate_pass": strict_support_gate(metrics),
            }
        )
    return pd.DataFrame(rows)


def select_evaluation_family(dataset: pd.DataFrame, *, primary_radius_mm: float, stretch_radius_mm: float) -> str:
    rows: list[dict[str, Any]] = []
    for family_id, part in dataset.groupby("family_id", sort=False):
        radii = np.asarray(sorted(part["radius_mm"].astype(float).unique()), dtype=float)
        rows.append(
            {
                "family_id": str(family_id),
                "has_primary": bool(np.any(np.isclose(radii, float(primary_radius_mm), atol=1.0e-8))),
                "has_stretch": bool(np.any(np.isclose(radii, float(stretch_radius_mm), atol=1.0e-8))),
                "radius_count": int(len(radii)),
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking[ranking["has_primary"]]
    if eligible.empty:
        raise ValueError(f"no dataset family materializes the primary radius {primary_radius_mm:g}mm")
    eligible = eligible.sort_values(
        ["has_stretch", "radius_count", "family_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    return str(eligible.iloc[0]["family_id"])


def v5_model_configs() -> list[v4.ModelConfig]:
    architectures = {
        "mlp_beta6_large": (512, 256, 128, 64),
        "mlp_beta6_wide": (512, 384, 256, 128, 64),
    }
    configs: list[v4.ModelConfig] = []
    for architecture, hidden_layers in architectures.items():
        for feature_set in ("raw", "poly_medium", "poly_heavy"):
            for activation in ("relu", "tanh"):
                for alpha in (1.0e-6, 1.0e-4):
                    configs.append(
                        v4.ModelConfig(
                            architecture=architecture,
                            hidden_layers=hidden_layers,
                            feature_set=feature_set,
                            activation=activation,
                            alpha=alpha,
                        )
                    )
    baseline_id = "mlp_beta6_large_poly_heavy_relu_a1em06"
    return sorted(
        configs,
        key=lambda config: (
            0 if config.config_id == baseline_id else 1,
            0 if config.feature_set == "poly_heavy" else 1 if config.feature_set == "poly_medium" else 2,
            0 if config.activation == "relu" else 1,
            config.alpha,
            config.architecture,
        ),
    )


def _row_at_radius(status: pd.DataFrame, radius_mm: float) -> pd.Series | None:
    mask = np.isclose(status["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
    if not np.any(mask):
        return None
    return status.loc[mask].iloc[0]


def _contiguous_true_radius(status: pd.DataFrame, *, gate_col: str, anchor_mm: float) -> float | None:
    rows = status.sort_values("radius_mm")
    rows = rows[rows["radius_mm"].to_numpy(dtype=float) >= float(anchor_mm) - 1.0e-9]
    if rows.empty or not np.isclose(float(rows.iloc[0]["radius_mm"]), float(anchor_mm), atol=1.0e-8):
        return None
    maximum: float | None = None
    for _idx, row in rows.iterrows():
        if not bool(row[gate_col]):
            break
        maximum = float(row["radius_mm"])
    return maximum


def build_v5_goal_report(
    status: pd.DataFrame,
    *,
    primary_radius_mm: float,
    stretch_radius_mm: float,
    anchor_mm: float,
) -> dict[str, Any]:
    rows = status.copy()
    rows["strict_gate_pass"] = (
        rows["strict_support_gate_pass"].astype(bool)
        & rows["stable_model_gate_pass"].astype(bool)
        & rows["trajectory_materialized"].astype(bool)
    )
    primary = _row_at_radius(rows, float(primary_radius_mm))
    stretch = _row_at_radius(rows, float(stretch_radius_mm))

    def goal_pass(row: pd.Series | None) -> bool:
        return bool(
            row is not None
            and row["strict_gate_pass"]
            and row["true_holdout_gate_pass"]
        )

    return {
        "primary_radius_mm": float(primary_radius_mm),
        "stretch_radius_mm": float(stretch_radius_mm),
        "primary_goal_pass": goal_pass(primary),
        "stretch_goal_pass": goal_pass(stretch),
        "strict_supported_rmax_mm": _contiguous_true_radius(rows, gate_col="strict_gate_pass", anchor_mm=float(anchor_mm)),
        "primary_evidence_kind": "whole_radius_holdout" if goal_pass(primary) else "insufficient",
        "stretch_evidence_kind": "whole_radius_holdout" if goal_pass(stretch) else "insufficient",
    }


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_audit"
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.tube_dataset)
    expansion_report_path = Path(args.expansion_dir) / "05_dataset" / "dataset_report.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"V5 training dataset is missing: {dataset_path}")
    dataset = pd.read_parquet(dataset_path)
    required = {
        "sample_id",
        "trajectory_id",
        "family_id",
        "candidate_id",
        "radius_mm",
        "angle_idx",
        "angle_rad",
        "tube_offset_id",
        "is_centerline",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "phase_y_rad",
        "phase_z_rad",
        *v4.TARGET_XYZ_COLS,
        *v4.BETA_COLS,
    }
    missing = sorted(required - set(dataset.columns))
    expansion_report = read_json(expansion_report_path) if expansion_report_path.exists() else {}
    radii = dataset["radius_mm"].to_numpy(dtype=float) if "radius_mm" in dataset else np.asarray([], dtype=float)
    checks = {
        "required_columns": not missing,
        "unique_sample_ids": bool("sample_id" in dataset and dataset["sample_id"].is_unique),
        "multiple_trajectories": bool("trajectory_id" in dataset and dataset["trajectory_id"].nunique() >= 3),
        "multiple_radii": bool("radius_mm" in dataset and dataset["radius_mm"].nunique() >= 3),
        "validation_radius_materialized": bool(
            np.any(np.isclose(radii, float(args.validation_radius_mm), atol=1.0e-8))
        ),
        "primary_radius_materialized": bool(
            np.any(np.isclose(radii, float(args.primary_radius_mm), atol=1.0e-8))
        ),
        "source_dataset_gate": bool(expansion_report.get("dataset_gate_pass", False)),
    }
    report = {
        "dataset": str(dataset_path),
        "rows": int(len(dataset)),
        "trajectory_count": int(dataset["trajectory_id"].nunique()) if "trajectory_id" in dataset else 0,
        "family_count": int(dataset["family_id"].nunique()) if "family_id" in dataset else 0,
        "radii_mm": sorted(float(value) for value in dataset["radius_mm"].unique()) if "radius_mm" in dataset else [],
        "missing_columns": missing,
        "checks": checks,
        "audit_gate_pass": bool(all(checks.values())),
        "source_dataset_report": str(expansion_report_path),
    }
    write_json(out / "audit_report.json", report)
    if not report["audit_gate_pass"]:
        raise RuntimeError(f"V5 training audit failed: {checks}, missing={missing}")
    return report


def phase_split(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_split_and_support"
    out.mkdir(parents=True, exist_ok=True)
    audit_path = Path(args.out_dir) / "00_audit" / "audit_report.json"
    if not audit_path.exists():
        phase_audit(args)
    dataset = pd.read_parquet(args.tube_dataset)
    family_id = select_evaluation_family(
        dataset,
        primary_radius_mm=float(args.primary_radius_mm),
        stretch_radius_mm=float(args.stretch_radius_mm),
    )
    split = make_whole_trajectory_split(
        dataset,
        primary_radius_mm=float(args.primary_radius_mm),
        stretch_radius_mm=float(args.stretch_radius_mm),
        validation_radius_mm=float(args.validation_radius_mm),
    )
    train_indices = strict_training_indices(dataset, split)
    assignment = dataset[
        ["sample_id", "trajectory_id", "family_id", "candidate_id", "radius_mm", "angle_idx", "tube_offset_id", "is_centerline"]
    ].copy()
    assignment.insert(0, "row_index", np.arange(len(dataset), dtype=np.int64))
    assignment["split"] = "train"
    assignment.loc[split.validation_idx, "split"] = "validation_radius"
    assignment.loc[split.primary_idx, "split"] = "primary_holdout"
    assignment.loc[split.stretch_idx, "split"] = "stretch_holdout"
    assignment["used_for_training"] = False
    assignment.loc[train_indices, "used_for_training"] = True
    assignment.to_parquet(out / "split_assignment.parquet", index=False, compression="zstd")
    metadata_row = dataset[dataset["family_id"].astype(str).eq(family_id)].iloc[0]
    metadata = {
        key: float(metadata_row[key])
        for key in ("center_x_m", "center_y_m", "center_z_m", "phase_y_rad", "phase_z_rad")
    }
    anchors = parse_float_csv(args.radius_anchors_mm)
    support = compute_support_scan(
        dataset,
        training_indices=train_indices,
        metadata=metadata,
        radii_mm=anchors,
        n_points=360,
    )
    family_radii = dataset[dataset["family_id"].astype(str).eq(family_id)]["radius_mm"].to_numpy(dtype=float)
    support["trajectory_materialized"] = [
        bool(np.any(np.isclose(family_radii, float(radius), atol=1.0e-8))) for radius in support["radius_mm"]
    ]
    support.to_csv(out / "training_only_support_by_radius.csv", index=False)
    train_radii = sorted(float(value) for value in dataset.iloc[train_indices]["radius_mm"].unique())
    report = {
        "evaluation_family_id": family_id,
        "evaluation_family_metadata": metadata,
        "rows": int(len(dataset)),
        "training_rows_noncenter": int(len(train_indices)),
        "training_trajectory_ids": list(split.train_trajectory_ids),
        "validation_trajectory_ids": list(split.validation_trajectory_ids),
        "primary_trajectory_ids": list(split.primary_trajectory_ids),
        "stretch_trajectory_ids": list(split.stretch_trajectory_ids),
        "training_radii_mm": train_radii,
        "validation_radius_mm": float(args.validation_radius_mm),
        "primary_radius_mm": float(args.primary_radius_mm),
        "stretch_radius_mm": float(args.stretch_radius_mm),
        "trajectory_leakage_count": int(split.trajectory_leakage_count),
        "centerline_training_rows": int(dataset.iloc[train_indices]["is_centerline"].astype(bool).sum()),
        "split_gate_pass": bool(
            split.trajectory_leakage_count == 0
            and len(train_indices) > 0
            and len(train_radii) >= 2
            and len(split.validation_idx) > 0
            and len(split.primary_idx) > 0
        ),
        "support_path": str(out / "training_only_support_by_radius.csv"),
        "assignment_path": str(out / "split_assignment.parquet"),
    }
    write_json(out / "split_report.json", report)
    if not report["split_gate_pass"]:
        raise RuntimeError(f"V5 whole-trajectory split gate failed: {report}")
    return report


def preset_settings(preset: str) -> dict[str, int]:
    if preset == "smoke":
        return {"angle_stride": 30, "max_iter": 30, "screen_seed_count": 1, "final_seed_count": 1, "config_limit": 2}
    if preset == "pilot":
        return {"angle_stride": 5, "max_iter": 300, "screen_seed_count": 1, "final_seed_count": 3, "config_limit": 8}
    if preset == "formal":
        return {"angle_stride": 2, "max_iter": 800, "screen_seed_count": 1, "final_seed_count": 5, "config_limit": 0}
    raise ValueError(f"unsupported preset: {preset}")


def _config_from_dict(data: Mapping[str, Any]) -> v4.ModelConfig:
    return v4.ModelConfig(
        architecture=str(data["architecture"]),
        hidden_layers=tuple(int(value) for value in data["hidden_layers"]),
        feature_set=str(data["feature_set"]),
        activation=str(data["activation"]),
        alpha=float(data["alpha"]),
    )


def run_training_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    dataset = pd.read_parquet(task["dataset"])
    assignment = pd.read_parquet(task["assignment"])
    if len(dataset) != len(assignment):
        raise ValueError("training dataset and split assignment row counts differ")
    config = _config_from_dict(task["config"])
    seed = int(task["seed"])
    stride = max(1, int(task["angle_stride"]))
    training_mask = assignment["used_for_training"].astype(bool).to_numpy()
    training_mask &= dataset["angle_idx"].to_numpy(dtype=np.int64) % stride == 0
    train_idx = np.flatnonzero(training_mask).astype(np.int64)
    if not len(train_idx):
        raise ValueError("training worker received an empty strict training partition")
    model, x_scaler, y_scaler, feature_names, fit_s = v4._fit_scaled_model(
        config=config,
        seed=seed,
        xyz_train=dataset.iloc[train_idx][v4.TARGET_XYZ_COLS].to_numpy(dtype=float),
        beta_train=dataset.iloc[train_idx][v4.BETA_COLS].to_numpy(dtype=float),
        max_iter=int(task["max_iter"]),
        batch_size=int(task.get("batch_size", 256)),
    )
    package: dict[str, Any] = {
        "kind": "beta6_pose_v5",
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": v4.TARGET_XYZ_COLS,
        "feature_set": config.feature_set,
        "feature_names": feature_names,
        "target_cols": v4.BETA_COLS,
        "beta_cols": v4.BETA_COLS,
        "theta_cols": v4.THETA_COLS,
        "theta_sign": float(task["theta_sign"]),
        "robot_config": str(task["robot_config"]),
        "dataset": str(task["dataset"]),
        "evaluation_family_id": str(task["evaluation_family_id"]),
        "train_trajectory_ids": sorted(assignment.loc[training_mask, "trajectory_id"].astype(str).unique().tolist()),
        "seed": seed,
        "model_name": config.architecture,
        "activation": config.activation,
        "alpha": config.alpha,
    }
    robot_config = v4.load_config(str(task["robot_config"]))
    robot_inputs = v4.load_robot_inputs(robot_config)
    theta_sign = float(task["theta_sign"])
    evaluations: dict[str, Any] = {}
    prediction_dir = Path(task["prediction_dir"]) if task.get("prediction_dir") else None
    for spec in task["evaluation_specs"]:
        label = str(spec["label"])
        split_name = str(spec["split"])
        mask = assignment["split"].astype(str).eq(split_name).to_numpy()
        mask &= dataset["family_id"].astype(str).eq(str(task["evaluation_family_id"])).to_numpy()
        mask &= dataset["is_centerline"].astype(bool).to_numpy()
        eval_idx = np.flatnonzero(mask).astype(np.int64)
        if not len(eval_idx):
            evaluations[label] = {
                "rows": 0,
                "model_gate_pass": False,
                "reason": "holdout_not_materialized",
                "radius_mm": float(spec["radius_mm"]),
            }
            continue
        eval_xyz = dataset.iloc[eval_idx][v4.TARGET_XYZ_COLS].to_numpy(dtype=float)
        beta_true = dataset.iloc[eval_idx][v4.BETA_COLS].to_numpy(dtype=float)
        beta_pred = v4.predict_beta(package, eval_xyz)
        metrics, achieved, _theta = v4.evaluate_beta_prediction(
            beta_pred=beta_pred,
            beta_true=beta_true,
            target_xyz=eval_xyz,
            lengths_m=robot_inputs.lengths_m,
            p_end_local_m=robot_inputs.p_end_local_m,
            theta_sign=theta_sign,
            periodic=True,
        )
        metrics.update(
            {
                "radius_mm": float(spec["radius_mm"]),
                "fit_s": fit_s,
                "n_iter": int(model.n_iter_),
                "loss": float(model.loss_),
                "train_rows": int(len(train_idx)),
                "eval_rows": int(len(eval_idx)),
                "model_gate_pass": v4.centerline_model_gate(metrics, require_beta_error=True),
            }
        )
        evaluations[label] = metrics
        if prediction_dir is not None:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction = dataset.iloc[eval_idx][
                ["trajectory_id", "family_id", "radius_mm", "angle_idx", "angle_rad", *v4.TARGET_XYZ_COLS]
            ].copy()
            for idx, column in enumerate(v4.BETA_COLS):
                prediction[f"true_{column}"] = beta_true[:, idx]
                prediction[f"pred_{column}"] = beta_pred[:, idx]
            for idx, axis in enumerate("xyz"):
                prediction[f"achieved_{axis}_m"] = achieved[:, idx]
            prediction["ee_err_mm"] = np.linalg.norm(achieved - eval_xyz, axis=1) * 1000.0
            prediction.to_parquet(prediction_dir / f"{label}.parquet", index=False, compression="zstd")
    package_path = Path(task["package_path"]) if task.get("package_path") else None
    if package_path is not None:
        package_path.parent.mkdir(parents=True, exist_ok=True)
        dump(package, package_path)
    result = {
        "task_id": str(task["task_id"]),
        "mode": str(task["mode"]),
        "seed": seed,
        "config_id": config.config_id,
        "config": asdict(config),
        "train_rows": int(len(train_idx)),
        "fit_s": fit_s,
        "evaluations": evaluations,
        "package_path": str(package_path) if package_path is not None else None,
    }
    write_json(Path(task["result_path"]), result)
    return result


def _execute_training_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_dir: Path,
    workers: int,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    task_dir.mkdir(parents=True, exist_ok=True)
    pending: list[Path] = []
    results: list[dict[str, Any]] = []
    for task in tasks:
        task_path = task_dir / f"{task['task_id']}.json"
        result_path = Path(task["result_path"])
        if bool(skip_existing) and result_path.exists():
            results.append(read_json(result_path))
            continue
        write_json(task_path, task)
        pending.append(task_path)

    def execute(task_path: Path) -> dict[str, Any]:
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path)],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"V5 training worker failed for {task_path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        task = read_json(task_path)
        return read_json(Path(task["result_path"]))

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            results.extend(executor.map(execute, pending))
    return sorted(results, key=lambda result: str(result["task_id"]))


def _task_base(
    args: argparse.Namespace,
    *,
    task_id: str,
    mode: str,
    config: v4.ModelConfig,
    seed: int,
    result_path: Path,
    package_path: Path | None,
    prediction_dir: Path | None,
    angle_stride: int,
    max_iter: int,
    evaluation_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    split_report = read_json(Path(args.out_dir) / "01_split_and_support" / "split_report.json")
    robot_config = v4.load_config(str(args.robot_config))
    return {
        "task_id": task_id,
        "mode": mode,
        "config": asdict(config),
        "seed": int(seed),
        "dataset": str(args.tube_dataset),
        "assignment": str(Path(args.out_dir) / "01_split_and_support" / "split_assignment.parquet"),
        "robot_config": str(args.robot_config),
        "theta_sign": float(robot_config.get("kinematics", {}).get("theta_sign", -1.0)),
        "evaluation_family_id": str(split_report["evaluation_family_id"]),
        "result_path": str(result_path),
        "package_path": str(package_path) if package_path is not None else None,
        "prediction_dir": str(prediction_dir) if prediction_dir is not None else None,
        "angle_stride": int(angle_stride),
        "max_iter": int(max_iter),
        "batch_size": 256,
        "evaluation_specs": evaluation_specs,
    }


def _flatten_evaluation_results(results: list[dict[str, Any]], *, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = dict(result["evaluations"].get(label, {}))
        rows.append(
            {
                "task_id": result["task_id"],
                "mode": result["mode"],
                "seed": int(result["seed"]),
                "config_id": result["config_id"],
                **result["config"],
                **metrics,
                "package_path": result.get("package_path"),
            }
        )
    return pd.DataFrame(rows)


def rank_model_screen(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_id, part in metrics.groupby("config_id", sort=False):
        first = part.iloc[0]
        rows.append(
            {
                "config_id": str(config_id),
                "architecture": first["architecture"],
                "hidden_layers": first["hidden_layers"],
                "feature_set": first["feature_set"],
                "activation": first["activation"],
                "alpha": float(first["alpha"]),
                "screen_seed_count": int(len(part)),
                "all_seed_gate_pass": bool(part["model_gate_pass"].fillna(False).astype(bool).all()),
                "passed_seed_count": int(part["model_gate_pass"].fillna(False).astype(bool).sum()),
                "ee_p95_median_mm": float(part["ee_p95_mm"].median()),
                "beta_p95_median_deg": float(part["beta_p95_deg"].median()),
                "axis_p95_median_mm": float(part["axiserr_max_p95_abs_mm"].median()),
                "fit_s_median": float(part["fit_s"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["all_seed_gate_pass", "ee_p95_median_mm", "beta_p95_median_deg", "axis_p95_median_mm", "fit_s_median"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def phase_screen(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_model_screen"
    out.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.out_dir) / "01_split_and_support" / "split_report.json"
    if not split_path.exists():
        phase_split(args)
    settings = preset_settings(str(args.preset))
    seeds = parse_int_csv(args.seeds)[: int(settings["screen_seed_count"])]
    configs = v5_model_configs()
    configured_limit = int(args.screen_config_limit)
    limit = configured_limit if configured_limit > 0 else int(settings["config_limit"])
    if limit > 0:
        configs = configs[:limit]
    evaluation_specs = [
        {"label": "validation", "split": "validation_radius", "radius_mm": float(args.validation_radius_mm)}
    ]
    tasks: list[dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            task_id = f"screen_{config.config_id}_s{seed}"
            tasks.append(
                _task_base(
                    args,
                    task_id=task_id,
                    mode="screen",
                    config=config,
                    seed=seed,
                    result_path=out / "worker_results" / f"{task_id}.json",
                    package_path=None,
                    prediction_dir=None,
                    angle_stride=int(settings["angle_stride"]),
                    max_iter=int(settings["max_iter"]),
                    evaluation_specs=evaluation_specs,
                )
            )
    results = _execute_training_tasks(
        tasks,
        task_dir=out / "worker_tasks",
        workers=int(args.workers),
        skip_existing=bool(args.skip_existing),
    )
    flat = _flatten_evaluation_results(results, label="validation")
    ranking = rank_model_screen(flat)
    flat.to_csv(out / "screen_metrics.csv", index=False)
    ranking.to_csv(out / "screen_config_ranking.csv", index=False)
    selected_id = str(ranking.iloc[0]["config_id"])
    selected_result = next(result for result in results if result["config_id"] == selected_id)
    report = {
        "selected_config_id": selected_id,
        "selected_config": selected_result["config"],
        "selected_validation_gate_pass": bool(ranking.iloc[0]["all_seed_gate_pass"]),
        "screen_seed_count": int(len(seeds)),
        "screen_config_count": int(len(configs)),
        "screen_run_count": int(len(results)),
        "ranking_path": str(out / "screen_config_ranking.csv"),
    }
    write_json(out / "selection_report.json", report)
    return report


def phase_train(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_final_models"
    out.mkdir(parents=True, exist_ok=True)
    selection_path = Path(args.out_dir) / "02_model_screen" / "selection_report.json"
    if not selection_path.exists():
        phase_screen(args)
    selection = read_json(selection_path)
    config = _config_from_dict(selection["selected_config"])
    settings = preset_settings(str(args.preset))
    seeds = parse_int_csv(args.seeds)[: int(settings["final_seed_count"])]
    evaluation_specs = [
        {"label": "primary", "split": "primary_holdout", "radius_mm": float(args.primary_radius_mm)},
        {"label": "stretch", "split": "stretch_holdout", "radius_mm": float(args.stretch_radius_mm)},
    ]
    tasks: list[dict[str, Any]] = []
    for seed in seeds:
        task_id = f"final_{config.config_id}_s{seed}"
        tasks.append(
            _task_base(
                args,
                task_id=task_id,
                mode="final",
                config=config,
                seed=seed,
                result_path=out / "worker_results" / f"{task_id}.json",
                package_path=out / "model_checkpoints" / f"seed_{seed}" / "model.joblib",
                prediction_dir=out / "holdout_predictions" / f"seed_{seed}",
                angle_stride=1,
                max_iter=int(settings["max_iter"]),
                evaluation_specs=evaluation_specs,
            )
        )
    results = _execute_training_tasks(
        tasks,
        task_dir=out / "worker_tasks",
        workers=int(args.workers),
        skip_existing=bool(args.skip_existing),
    )
    primary = _flatten_evaluation_results(results, label="primary")
    stretch = _flatten_evaluation_results(results, label="stretch")
    primary.to_csv(out / "primary_holdout_metrics_all_seeds.csv", index=False)
    stretch.to_csv(out / "stretch_holdout_metrics_all_seeds.csv", index=False)
    primary_gate = v4.aggregate_seed_gate(primary, required_fraction=0.8)
    stretch_gate = v4.aggregate_seed_gate(stretch, required_fraction=0.8)
    report = {
        "selected_config_id": config.config_id,
        "selected_config": asdict(config),
        "seed_count": int(len(seeds)),
        "primary_holdout_gate": primary_gate,
        "stretch_holdout_gate": stretch_gate,
        "primary_whole_radius_held_out": True,
        "stretch_whole_radius_held_out": True,
        "formal_sweep_allowed": bool(primary_gate["stable_gate_pass"]),
        "model_paths": {
            str(result["seed"]): result["package_path"] for result in results if result.get("package_path")
        },
    }
    write_json(out / "final_training_report.json", report)
    return report


def phase_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "04_radius_sweep"
    out.mkdir(parents=True, exist_ok=True)
    training_path = Path(args.out_dir) / "03_final_models" / "final_training_report.json"
    if not training_path.exists():
        phase_train(args)
    training_report = read_json(training_path)
    split_report = read_json(Path(args.out_dir) / "01_split_and_support" / "split_report.json")
    support = pd.read_csv(Path(args.out_dir) / "01_split_and_support" / "training_only_support_by_radius.csv")
    metadata = split_report["evaluation_family_metadata"]
    robot_config = v4.load_config(str(args.robot_config))
    robot_inputs = v4.load_robot_inputs(robot_config)
    theta_sign = float(robot_config.get("kinematics", {}).get("theta_sign", -1.0))
    seed_rows: list[dict[str, Any]] = []
    for seed_text, package_text in training_report["model_paths"].items():
        seed = int(seed_text)
        package = load(package_text)
        for radius_mm in parse_float_csv(args.radius_anchors_mm):
            target, _angle = v4.generate_exact_ellipse(metadata, amp_xy_mm=float(radius_mm), n_points=360)
            beta_pred = v4.predict_beta(package, target)
            metrics, _achieved, _theta = v4.evaluate_beta_prediction(
                beta_pred=beta_pred,
                target_xyz=target,
                lengths_m=robot_inputs.lengths_m,
                p_end_local_m=robot_inputs.p_end_local_m,
                theta_sign=theta_sign,
                periodic=True,
            )
            metrics["model_gate_pass"] = v4.centerline_model_gate(metrics, require_beta_error=False)
            seed_rows.append({"seed": seed, "radius_mm": float(radius_mm), **metrics})
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(out / "radius_sweep_all_seeds.csv", index=False)
    status_rows: list[dict[str, Any]] = []
    for radius_mm, part in seed_metrics.groupby("radius_mm", sort=True):
        support_mask = np.isclose(support["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8)
        if not np.any(support_mask):
            raise ValueError(f"support scan is missing radius {radius_mm:g}")
        support_row = support.loc[support_mask].iloc[0]
        aggregate = v4.aggregate_seed_gate(part, required_fraction=0.8)
        if np.isclose(float(radius_mm), float(args.primary_radius_mm), atol=1.0e-8):
            true_holdout = bool(training_report["primary_holdout_gate"]["stable_gate_pass"])
        elif np.isclose(float(radius_mm), float(args.stretch_radius_mm), atol=1.0e-8):
            true_holdout = bool(training_report["stretch_holdout_gate"]["stable_gate_pass"])
        else:
            true_holdout = False
        status_rows.append(
            {
                "radius_mm": float(radius_mm),
                "amp_z_mm": 1.5 * float(radius_mm),
                **aggregate,
                "stable_model_gate_pass": bool(aggregate["stable_gate_pass"]),
                "strict_support_gate_pass": bool(support_row["strict_support_gate_pass"]),
                "trajectory_materialized": bool(support_row["trajectory_materialized"]),
                "true_holdout_gate_pass": true_holdout,
                "nn_p95_mm": float(support_row["nn_p95_mm"]),
                "nn_max_mm": float(support_row["nn_max_mm"]),
                "tube_count_p10": float(support_row["tube_count_p10"]),
                "ee_p95_median_mm": float(part["ee_p95_mm"].median()),
                "ee_max_worst_mm": float(part["ee_max_mm"].max()),
                "axis_p95_median_mm": float(part["axiserr_max_p95_abs_mm"].median()),
            }
        )
    status = pd.DataFrame(status_rows)
    status["strict_gate_pass"] = (
        status["stable_model_gate_pass"].astype(bool)
        & status["strict_support_gate_pass"].astype(bool)
        & status["trajectory_materialized"].astype(bool)
    )
    status.to_csv(out / "radius_status.csv", index=False)
    goal = build_v5_goal_report(
        status,
        primary_radius_mm=float(args.primary_radius_mm),
        stretch_radius_mm=float(args.stretch_radius_mm),
        anchor_mm=75.0,
    )
    report = {
        "sweep_executed": True,
        "evaluation_family_id": split_report["evaluation_family_id"],
        **goal,
        "status_path": str(out / "radius_status.csv"),
        "seed_metrics_path": str(out / "radius_sweep_all_seeds.csv"),
    }
    write_json(out / "goal_report.json", report)
    return report


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "05_summary"
    out.mkdir(parents=True, exist_ok=True)
    goal_path = Path(args.out_dir) / "04_radius_sweep" / "goal_report.json"
    if not goal_path.exists():
        phase_sweep(args)
    audit = read_json(Path(args.out_dir) / "00_audit" / "audit_report.json")
    split = read_json(Path(args.out_dir) / "01_split_and_support" / "split_report.json")
    selection = read_json(Path(args.out_dir) / "02_model_screen" / "selection_report.json")
    training = read_json(Path(args.out_dir) / "03_final_models" / "final_training_report.json")
    goal = read_json(goal_path)
    status = pd.read_csv(Path(args.out_dir) / "04_radius_sweep" / "radius_status.csv")
    primary = _row_at_radius(status, float(args.primary_radius_mm))
    stretch = _row_at_radius(status, float(args.stretch_radius_mm))

    def metric(row: pd.Series | None, key: str) -> str:
        if row is None:
            return "n/a"
        value = row.get(key, "n/a")
        return f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value)

    lines = [
        "# True Ellipse V5 whole-trajectory generalization summary",
        "",
        f"- Primary goal: `{float(args.primary_radius_mm):g} mm` strict support-backed.",
        f"- Stretch challenge: `{float(args.stretch_radius_mm):g} mm` strict support-backed.",
        f"- Evaluation family: `{split['evaluation_family_id']}`.",
        f"- Selected model: `{selection['selected_config_id']}`.",
        f"- Primary whole-radius holdout gate: `{training['primary_holdout_gate']['stable_gate_pass']}` ({training['primary_holdout_gate']['passed_seed_count']}/{training['primary_holdout_gate']['total_seed_count']}).",
        f"- Stretch whole-radius holdout gate: `{training['stretch_holdout_gate']['stable_gate_pass']}` ({training['stretch_holdout_gate']['passed_seed_count']}/{training['stretch_holdout_gate']['total_seed_count']}).",
        f"- Primary strict goal pass: `{goal['primary_goal_pass']}`.",
        f"- Stretch strict goal pass: `{goal['stretch_goal_pass']}`.",
        f"- Connected strict supported radius: `{goal['strict_supported_rmax_mm']} mm`.",
        "",
        "## Leakage and support boundary",
        "",
        f"- Trajectory leakage count: `{split['trajectory_leakage_count']}`.",
        f"- Centerline training rows: `{split['centerline_training_rows']}`.",
        f"- Strict training rows: `{split['training_rows_noncenter']}`.",
        f"- Primary support nn p95/max/count p10: `{metric(primary, 'nn_p95_mm')}/{metric(primary, 'nn_max_mm')}/{metric(primary, 'tube_count_p10')}`.",
        f"- Stretch support nn p95/max/count p10: `{metric(stretch, 'nn_p95_mm')}/{metric(stretch, 'nn_max_mm')}/{metric(stretch, 'tube_count_p10')}`.",
        "",
        "## Radius status",
        "",
        status.to_markdown(index=False),
        "",
        "The primary and stretch claims require a materialized robust trajectory, strict support computed only from the non-centerline training partition, and a 4/5-seed model gate. The two goals are reported independently.",
        "",
    ]
    summary_path = out / "experiment_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    docs_path = REPO_ROOT / "docs" / "TrueEllipseFamilyGeneralizationV5实验记录.md" if str(args.preset) == "formal" else None
    if docs_path is not None:
        docs_path.write_text("\n".join(lines), encoding="utf-8")
    report = {
        "audit_gate_pass": bool(audit["audit_gate_pass"]),
        "split_gate_pass": bool(split["split_gate_pass"]),
        "selected_config_id": selection["selected_config_id"],
        "primary_goal_pass": bool(goal["primary_goal_pass"]),
        "stretch_goal_pass": bool(goal["stretch_goal_pass"]),
        "strict_supported_rmax_mm": goal["strict_supported_rmax_mm"],
        "summary_path": str(summary_path),
        "docs_path": str(docs_path) if docs_path is not None else None,
    }
    write_json(out / "final_report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a generalized beta6 model on V5 whole-trajectory holdouts.")
    parser.add_argument("--tube-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--expansion-dir", type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preset", choices=["smoke", "pilot", "formal"], default="formal")
    parser.add_argument("--phases", default="all")
    parser.add_argument("--primary-radius-mm", type=float, default=87.5)
    parser.add_argument("--stretch-radius-mm", type=float, default=100.0)
    parser.add_argument("--validation-radius-mm", type=float, default=85.0)
    parser.add_argument("--radius-anchors-mm", default="75,80,82.5,85,87.5,90,92.5,95,97.5,100")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--screen-config-limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--worker-task", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    phases = parse_phases(args.phases)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    handlers = {
        "audit": phase_audit,
        "split": phase_split,
        "screen": phase_screen,
        "train": phase_train,
        "sweep": phase_sweep,
        "summary": phase_summary,
    }
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for phase in phases:
        results[phase] = handlers[phase](args)
    payload = {
        "mode": "true_ellipse_family_training_v5",
        "preset": str(args.preset),
        "phases": phases,
        "elapsed_s": float(time.perf_counter() - started),
        "results": results,
    }
    write_json(Path(args.out_dir) / "run_report.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.worker_task is not None:
        result = run_training_worker(read_json(Path(args.worker_task)))
        print(json.dumps({"task_id": result["task_id"], "ok": True}, ensure_ascii=False))
        return 0
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
