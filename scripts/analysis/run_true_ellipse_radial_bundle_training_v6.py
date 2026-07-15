#!/usr/bin/env python3
"""Train the static beta6 inverse only after the V6 100 mm upstream gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_family_training_v5 as v5  # noqa: E402
from true_ellipse_family_v5_utils import dataframe_to_markdown, file_sha256, stable_fingerprint  # noqa: E402


MODEL_INPUT_COLUMNS = ["x_target_m", "y_target_m", "z_target_m"]
BETA_COLUMNS = [f"beta{i}_rad" for i in range(1, 7)]
FORMAL_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
VALIDATION_RADIUS_MM = 92.5
TEST_RADIUS_MM = 100.0
V4_BASELINE_CONFIG_ID = v5.V4_BASELINE_CONFIG_ID
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "runs" / "true_ellipse_radial_bundle_v6"
DEFAULT_DATASET = DEFAULT_UPSTREAM_DIR / "03_dataset" / "true_ellipse_radial_bundle_tubes_v6.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "true_ellipse_radial_bundle_training_v6"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml"
ALL_PHASES = ["audit", "split", "screen", "train", "summary"]
TRAINING_STRATEGY_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        raise ValueError(f"unsupported V6 training phases: {unknown}")
    return phases


def model_configs() -> list[v5.v4.ModelConfig]:
    return v5.v5_model_configs()


def preset_settings(preset: str) -> dict[str, int]:
    if str(preset) == "smoke":
        return {"angle_stride": 30, "max_iter": 30, "config_limit": 2, "final_seed_count": 1}
    if str(preset) == "pilot":
        return {"angle_stride": 5, "max_iter": 300, "config_limit": 8, "final_seed_count": 3}
    if str(preset) == "formal":
        return {"angle_stride": 2, "max_iter": 800, "config_limit": 0, "final_seed_count": 5}
    raise ValueError(f"unsupported preset: {preset}")


def formal_training_protocol_report(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_int_csv(args.seeds)
    configs = model_configs()
    config_ids = [config.config_id for config in configs]
    checks = {
        "formal_preset": str(args.preset) == "formal",
        "validation_radius_92p5": np.isclose(float(args.validation_radius_mm), VALIDATION_RADIUS_MM),
        "test_radius_100": np.isclose(float(args.test_radius_mm), TEST_RADIUS_MM),
        "exact_five_seeds": seeds == list(FORMAL_SEEDS),
        "screen_config_limit_zero": int(args.screen_config_limit) == 0,
        "exact_24_configs": len(config_ids) == 24 and len(set(config_ids)) == 24,
        "v4_baseline_present": V4_BASELINE_CONFIG_ID in config_ids,
        "formal_angle_stride_two": int(preset_settings(str(args.preset))["angle_stride"]) == 2,
        "xyz_only_model_inputs": MODEL_INPUT_COLUMNS == list(v5.v4.TARGET_XYZ_COLS),
    }
    normalized = {key: bool(value) for key, value in checks.items()}
    protocol = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "preset": str(args.preset),
        "validation_radius_mm": float(args.validation_radius_mm),
        "test_radius_mm": float(args.test_radius_mm),
        "seeds": seeds,
        "screen_config_limit": int(args.screen_config_limit),
        "screen_config_ids": config_ids,
        "model_input_columns": list(MODEL_INPUT_COLUMNS),
        "settings": preset_settings(str(args.preset)),
    }
    return {
        "formal_training_protocol_gate_pass": bool(all(normalized.values())),
        "checks": normalized,
        "protocol": protocol,
        "protocol_fingerprint": stable_fingerprint(protocol),
    }


def make_training_assignment(
    dataset: pd.DataFrame,
    *,
    validation_radius_mm: float = VALIDATION_RADIUS_MM,
    test_radius_mm: float = TEST_RADIUS_MM,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "sample_id",
        "family_id",
        "trajectory_id",
        "radius_mm",
        "angle_idx",
        "is_centerline",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"V6 training assignment missing columns: {missing}")
    assignment = dataset[
        [
            "sample_id",
            "family_id",
            "trajectory_id",
            "radius_mm",
            "angle_idx",
            "is_centerline",
        ]
    ].copy()
    radius = assignment["radius_mm"].to_numpy(dtype=float)
    assignment["split"] = "train"
    assignment.loc[
        np.isclose(radius, float(validation_radius_mm), atol=1.0e-8), "split"
    ] = "validation"
    assignment.loc[np.isclose(radius, float(test_radius_mm), atol=1.0e-8), "split"] = "test"
    if "split" in dataset and not assignment["split"].astype(str).reset_index(drop=True).equals(
        dataset["split"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("dataset split labels disagree with the registered whole-radius protocol")
    assignment["used_for_training"] = bool(False)
    training_mask = assignment["split"].eq("train") & ~assignment["is_centerline"].astype(bool)
    assignment.loc[training_mask, "used_for_training"] = True
    trajectory_split_counts = assignment.groupby("trajectory_id")["split"].nunique()
    radius_split_counts = assignment.groupby("radius_mm")["split"].nunique()
    validation_present = bool(np.any(np.isclose(radius, float(validation_radius_mm), atol=1.0e-8)))
    test_present = bool(np.any(np.isclose(radius, float(test_radius_mm), atol=1.0e-8)))
    report = {
        "rows": int(len(assignment)),
        "training_rows": int(assignment["used_for_training"].sum()),
        "validation_rows": int(assignment["split"].eq("validation").sum()),
        "test_rows": int(assignment["split"].eq("test").sum()),
        "training_radius_count": int(
            assignment.loc[assignment["used_for_training"], "radius_mm"].nunique()
        ),
        "trajectory_leakage_count": int((trajectory_split_counts > 1).sum()),
        "radius_leakage_count": int((radius_split_counts > 1).sum()),
        "validation_radius_present": validation_present,
        "test_radius_present": test_present,
        "training_centerline_count": int(
            assignment.loc[assignment["used_for_training"], "is_centerline"].astype(bool).sum()
        ),
    }
    report["split_gate_pass"] = bool(
        len(assignment)
        and report["training_rows"] > 0
        and report["training_radius_count"] >= 2
        and validation_present
        and test_present
        and report["trajectory_leakage_count"] == 0
        and report["radius_leakage_count"] == 0
        and report["training_centerline_count"] == 0
    )
    return assignment, report


def aggregate_formal_seed_gate(rows: pd.DataFrame) -> dict[str, Any]:
    total = int(len(rows))
    unique_seeds = int(rows["seed"].nunique()) if total and "seed" in rows else 0
    passed = int(rows["model_gate_pass"].fillna(False).astype(bool).sum()) if total else 0
    return {
        "total_seed_count": total,
        "unique_seed_count": unique_seeds,
        "passed_seed_count": passed,
        "required_seed_count": 4,
        "stable_gate_pass": bool(total == 5 and unique_seeds == 5 and passed >= 4),
    }


def formal_model_gate_pass(
    *,
    formal_claims_allowed: bool,
    validation_gate: Mapping[str, Any],
    test_gate: Mapping[str, Any],
) -> bool:
    return bool(
        formal_claims_allowed
        and validation_gate.get("stable_gate_pass", False)
        and test_gate.get("stable_gate_pass", False)
    )


def upstream_training_authorized(dataset_report: Mapping[str, Any]) -> bool:
    return bool(
        dataset_report.get("formal_dataset_gate_pass", False)
        and dataset_report.get("training_only_support_100mm_pass", False)
    )


def _phase_compatible(
    report: Mapping[str, Any], *, strategy_version: int, task_fingerprint: str
) -> bool:
    return bool(
        int(report.get("strategy_version", -1)) == int(strategy_version)
        and str(report.get("task_fingerprint", "")) == str(task_fingerprint)
    )


def _resolved_seeds(args: argparse.Namespace) -> list[int]:
    seeds = parse_int_csv(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("V6 training seeds must be non-empty and unique")
    if str(args.preset) == "formal":
        if seeds != list(FORMAL_SEEDS):
            raise ValueError(f"formal V6 training requires exactly {list(FORMAL_SEEDS)}")
        return seeds
    return seeds[: int(preset_settings(str(args.preset))["final_seed_count"])]


def _upstream_dataset_report_path(args: argparse.Namespace) -> Path:
    return Path(args.upstream_dir) / "03_dataset" / "dataset_report.json"


def _audit_task_fingerprint(args: argparse.Namespace) -> str:
    report_path = _upstream_dataset_report_path(args)
    return stable_fingerprint(
        {
            "phase": "audit",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "protocol": formal_training_protocol_report(args)["protocol"],
            "upstream_dataset_report_sha256": file_sha256(report_path),
            "dataset_sha256": file_sha256(args.tube_dataset),
            "robot_config_sha256": file_sha256(args.robot_config),
        }
    )


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_audit"
    out.mkdir(parents=True, exist_ok=True)
    upstream_path = _upstream_dataset_report_path(args)
    if not upstream_path.is_file():
        raise FileNotFoundError(f"missing V6 upstream dataset report: {upstream_path}")
    if not Path(args.tube_dataset).is_file():
        raise FileNotFoundError(f"missing V6 tube dataset: {args.tube_dataset}")
    if not Path(args.robot_config).is_file():
        raise FileNotFoundError(f"missing robot config: {args.robot_config}")
    upstream = read_json(upstream_path)
    protocol = formal_training_protocol_report(args)
    dataset_hash = file_sha256(args.tube_dataset)
    expected_hash = str(upstream.get("dataset_sha256", ""))
    dataset = pd.read_parquet(args.tube_dataset)
    required = {
        "sample_id",
        "family_id",
        "trajectory_id",
        "radius_mm",
        "angle_idx",
        "is_centerline",
        *MODEL_INPUT_COLUMNS,
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(dataset.columns))
    checks = {
        "upstream_training_authorized": upstream_training_authorized(upstream),
        "dataset_hash_matches_upstream": bool(expected_hash and dataset_hash == expected_hash),
        "dataset_rows_match_upstream": int(len(dataset)) == int(upstream.get("rows", -1)),
        "dataset_has_required_columns": not missing,
        "single_fixed_family": int(dataset["family_id"].nunique()) == 1 if "family_id" in dataset else False,
        "sample_ids_unique": bool(dataset["sample_id"].is_unique) if "sample_id" in dataset else False,
        "robot_config_present": Path(args.robot_config).is_file(),
    }
    if str(args.preset) == "formal":
        checks["formal_training_protocol"] = bool(protocol["formal_training_protocol_gate_pass"])
    task_fingerprint = _audit_task_fingerprint(args)
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        "audit_gate_pass": bool(all(checks.values())),
        "formal_claims_allowed": bool(
            all(checks.values())
            and upstream.get("formal_dataset_gate_pass", False)
            and protocol["formal_training_protocol_gate_pass"]
        ),
        "checks": {key: bool(value) for key, value in checks.items()},
        "missing_required_columns": missing,
        "dataset_rows": int(len(dataset)),
        "dataset_sha256": dataset_hash,
        "upstream_dataset_report_path": str(upstream_path.resolve()),
        "upstream_dataset_report_sha256": file_sha256(upstream_path),
        "upstream_dataset_task_fingerprint": str(upstream.get("task_fingerprint", "")),
        "evaluation_family_id": str(dataset["family_id"].iloc[0]) if len(dataset) and "family_id" in dataset else None,
        "formal_training_protocol": protocol,
    }
    write_json(out / "audit_report.json", report)
    if not report["audit_gate_pass"]:
        raise RuntimeError(f"V6 training audit gate failed: {report['checks']}")
    return report


def ensure_audit_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "00_audit" / "audit_report.json"
    expected = _audit_task_fingerprint(args)
    if path.exists():
        cached = read_json(path)
        if _phase_compatible(
            cached,
            strategy_version=TRAINING_STRATEGY_VERSION,
            task_fingerprint=expected,
        ):
            return cached
    return phase_audit(args)


def _split_task_fingerprint(args: argparse.Namespace, audit: Mapping[str, Any]) -> str:
    return stable_fingerprint(
        {
            "phase": "split",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "source_audit": str(audit.get("task_fingerprint", "")),
            "validation_radius_mm": float(args.validation_radius_mm),
            "test_radius_mm": float(args.test_radius_mm),
            "dataset_sha256": file_sha256(args.tube_dataset),
        }
    )


def phase_split(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_split"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    dataset = pd.read_parquet(args.tube_dataset)
    assignment, split = make_training_assignment(
        dataset,
        validation_radius_mm=float(args.validation_radius_mm),
        test_radius_mm=float(args.test_radius_mm),
    )
    assignment.insert(0, "row_index", np.arange(len(assignment), dtype=np.int64))
    assignment_path = out / "split_assignment.parquet"
    assignment.to_parquet(assignment_path, index=False, compression="zstd")
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": _split_task_fingerprint(args, audit),
        "source_audit_task_fingerprint": str(audit.get("task_fingerprint", "")),
        "formal_claims_allowed": bool(audit.get("formal_claims_allowed", False)),
        "evaluation_family_id": str(audit["evaluation_family_id"]),
        **split,
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": file_sha256(assignment_path),
    }
    write_json(out / "split_report.json", report)
    if not report["split_gate_pass"]:
        raise RuntimeError(f"V6 whole-radius split gate failed: {report}")
    return report


def ensure_split_report(args: argparse.Namespace) -> dict[str, Any]:
    audit = ensure_audit_report(args)
    expected = _split_task_fingerprint(args, audit)
    report_path = Path(args.out_dir) / "01_split" / "split_report.json"
    assignment_path = Path(args.out_dir) / "01_split" / "split_assignment.parquet"
    if report_path.exists() and assignment_path.exists():
        cached = read_json(report_path)
        if (
            _phase_compatible(
                cached,
                strategy_version=TRAINING_STRATEGY_VERSION,
                task_fingerprint=expected,
            )
            and str(cached.get("assignment_sha256", "")) == file_sha256(assignment_path)
        ):
            return cached
    return phase_split(args)


def _training_task(
    args: argparse.Namespace,
    *,
    task_id: str,
    mode: str,
    config: v5.v4.ModelConfig,
    seed: int,
    result_path: Path,
    package_path: Path | None,
    prediction_dir: Path | None,
    angle_stride: int,
    max_iter: int,
    evaluation_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    split = read_json(Path(args.out_dir) / "01_split" / "split_report.json")
    robot = v5.v4.load_config(str(args.robot_config))
    assignment = Path(args.out_dir) / "01_split" / "split_assignment.parquet"
    task: dict[str, Any] = {
        "task_id": task_id,
        "mode": mode,
        "preset": str(args.preset),
        "training_task_strategy_version": TRAINING_STRATEGY_VERSION,
        "config": asdict(config),
        "seed": int(seed),
        "dataset": str(Path(args.tube_dataset).resolve()),
        "dataset_sha256": file_sha256(args.tube_dataset),
        "assignment": str(assignment.resolve()),
        "assignment_sha256": file_sha256(assignment),
        "robot_config": str(Path(args.robot_config).resolve()),
        "robot_config_sha256": file_sha256(args.robot_config),
        "theta_sign": float(robot.get("kinematics", {}).get("theta_sign", -1.0)),
        "evaluation_family_id": str(split["evaluation_family_id"]),
        "result_path": str(result_path.resolve()),
        "package_path": str(package_path.resolve()) if package_path is not None else None,
        "prediction_dir": str(prediction_dir.resolve()) if prediction_dir is not None else None,
        "angle_stride": int(angle_stride),
        "max_iter": int(max_iter),
        "batch_size": 256,
        "evaluation_specs": evaluation_specs,
    }
    task["task_fingerprint"] = v5.training_task_fingerprint(task)
    return task


def _screen_task_fingerprint(args: argparse.Namespace, split: Mapping[str, Any]) -> str:
    settings = preset_settings(str(args.preset))
    configs = model_configs()
    limit = int(args.screen_config_limit) or int(settings["config_limit"])
    if limit > 0:
        configs = configs[:limit]
    return stable_fingerprint(
        {
            "phase": "screen",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "source_split": str(split.get("task_fingerprint", "")),
            "preset": str(args.preset),
            "seed": int(_resolved_seeds(args)[0]),
            "configs": [asdict(config) for config in configs],
            "validation_radius_mm": float(args.validation_radius_mm),
            "angle_stride": int(settings["angle_stride"]),
            "max_iter": int(settings["max_iter"]),
            "assignment_sha256": str(split.get("assignment_sha256", "")),
        }
    )


def phase_screen(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_model_screen"
    out.mkdir(parents=True, exist_ok=True)
    split = ensure_split_report(args)
    task_fingerprint = _screen_task_fingerprint(args, split)
    settings = preset_settings(str(args.preset))
    configs = model_configs()
    limit = int(args.screen_config_limit) or int(settings["config_limit"])
    if limit > 0:
        configs = configs[:limit]
    seed = int(_resolved_seeds(args)[0])
    tasks = [
        _training_task(
            args,
            task_id=f"screen_{config.config_id}_s{seed}",
            mode="v6_screen",
            config=config,
            seed=seed,
            result_path=out / "worker_results" / f"screen_{config.config_id}_s{seed}.json",
            package_path=None,
            prediction_dir=None,
            angle_stride=int(settings["angle_stride"]),
            max_iter=int(settings["max_iter"]),
            evaluation_specs=[
                {
                    "label": "validation_92p5",
                    "split": "validation",
                    "radius_mm": float(args.validation_radius_mm),
                }
            ],
        )
        for config in configs
    ]
    results = v5._execute_training_tasks(
        tasks,
        task_dir=out / "worker_tasks",
        workers=int(args.workers),
        skip_existing=bool(args.skip_existing),
    )
    metrics = v5._flatten_evaluation_results(results, label="validation_92p5")
    ranking = v5.rank_model_screen(metrics)
    metrics.to_csv(out / "screen_metrics.csv", index=False)
    ranking.to_csv(out / "screen_config_ranking.csv", index=False)
    selected_id = str(ranking.iloc[0]["config_id"])
    selected = next(result for result in results if str(result["config_id"]) == selected_id)
    formal_screen_protocol = bool(
        str(args.preset) == "formal"
        and len(configs) == 24
        and len(results) == 24
        and int(args.screen_config_limit) == 0
        and V4_BASELINE_CONFIG_ID in {config.config_id for config in configs}
        and seed == FORMAL_SEEDS[0]
    )
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        "source_split_task_fingerprint": str(split.get("task_fingerprint", "")),
        "preset": str(args.preset),
        "screen_seed": seed,
        "screen_config_count": int(len(configs)),
        "screen_run_count": int(len(results)),
        "screen_config_ids": [config.config_id for config in configs],
        "selected_config_id": selected_id,
        "selected_config": selected["config"],
        "selected_validation_gate_pass": bool(ranking.iloc[0]["all_seed_gate_pass"]),
        "formal_screen_protocol_gate_pass": formal_screen_protocol,
        "formal_claims_allowed": bool(
            split.get("formal_claims_allowed", False) and formal_screen_protocol
        ),
        "ranking_path": str((out / "screen_config_ranking.csv").resolve()),
    }
    write_json(out / "selection_report.json", report)
    if str(args.preset) == "formal" and not formal_screen_protocol:
        raise RuntimeError(f"V6 formal screen protocol failed: {report}")
    return report


def ensure_screen_report(args: argparse.Namespace) -> dict[str, Any]:
    split = ensure_split_report(args)
    expected = _screen_task_fingerprint(args, split)
    path = Path(args.out_dir) / "02_model_screen" / "selection_report.json"
    ranking = Path(args.out_dir) / "02_model_screen" / "screen_config_ranking.csv"
    if path.exists() and ranking.exists():
        cached = read_json(path)
        if _phase_compatible(
            cached,
            strategy_version=TRAINING_STRATEGY_VERSION,
            task_fingerprint=expected,
        ):
            return cached
    return phase_screen(args)


def _final_task_fingerprint(args: argparse.Namespace, selection: Mapping[str, Any]) -> str:
    return stable_fingerprint(
        {
            "phase": "train",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "source_screen": str(selection.get("task_fingerprint", "")),
            "selected_config": selection.get("selected_config", {}),
            "seeds": _resolved_seeds(args),
            "validation_radius_mm": float(args.validation_radius_mm),
            "test_radius_mm": float(args.test_radius_mm),
            "max_iter": int(preset_settings(str(args.preset))["max_iter"]),
            "dataset_sha256": file_sha256(args.tube_dataset),
            "assignment_sha256": file_sha256(
                Path(args.out_dir) / "01_split" / "split_assignment.parquet"
            ),
        }
    )


def phase_train(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_final_models"
    out.mkdir(parents=True, exist_ok=True)
    selection = ensure_screen_report(args)
    split = read_json(Path(args.out_dir) / "01_split" / "split_report.json")
    if str(args.preset) == "formal" and not bool(selection.get("formal_claims_allowed", False)):
        raise RuntimeError("V6 formal final training is blocked by the audit/split/screen chain")
    config = v5._config_from_dict(selection["selected_config"])
    seeds = _resolved_seeds(args)
    max_iter = int(preset_settings(str(args.preset))["max_iter"])
    tasks = [
        _training_task(
            args,
            task_id=f"final_{config.config_id}_s{seed}",
            mode="v6_final",
            config=config,
            seed=int(seed),
            result_path=out / "worker_results" / f"final_{config.config_id}_s{seed}.json",
            package_path=out / "model_checkpoints" / f"seed_{seed}" / "model.joblib",
            prediction_dir=out / "holdout_predictions" / f"seed_{seed}",
            angle_stride=1,
            max_iter=max_iter,
            evaluation_specs=[
                {
                    "label": "validation_92p5",
                    "split": "validation",
                    "radius_mm": float(args.validation_radius_mm),
                },
                {
                    "label": "test_100",
                    "split": "test",
                    "radius_mm": float(args.test_radius_mm),
                },
            ],
        )
        for seed in seeds
    ]
    results = v5._execute_training_tasks(
        tasks,
        task_dir=out / "worker_tasks",
        workers=int(args.workers),
        skip_existing=bool(args.skip_existing),
    )
    validation = v5._flatten_evaluation_results(results, label="validation_92p5")
    test = v5._flatten_evaluation_results(results, label="test_100")
    validation.to_csv(out / "validation_92p5_metrics_all_seeds.csv", index=False)
    test.to_csv(out / "test_100_metrics_all_seeds.csv", index=False)
    validation_gate = aggregate_formal_seed_gate(validation)
    test_gate = aggregate_formal_seed_gate(test)
    formal_claims = bool(
        split.get("formal_claims_allowed", False)
        and selection.get("formal_claims_allowed", False)
        and str(args.preset) == "formal"
        and seeds == list(FORMAL_SEEDS)
    )
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": _final_task_fingerprint(args, selection),
        "source_screen_task_fingerprint": str(selection.get("task_fingerprint", "")),
        "preset": str(args.preset),
        "selected_config_id": config.config_id,
        "selected_config": asdict(config),
        "seed_count": int(len(seeds)),
        "seeds": seeds,
        "validation_92p5_gate": validation_gate,
        "test_100_gate": test_gate,
        "formal_claims_allowed": formal_claims,
        "formal_model_gate_pass": formal_model_gate_pass(
            formal_claims_allowed=formal_claims,
            validation_gate=validation_gate,
            test_gate=test_gate,
        ),
        "model_paths": {
            str(result["seed"]): result["package_path"]
            for result in results
            if result.get("package_path")
        },
    }
    write_json(out / "final_training_report.json", report)
    return report


def ensure_training_report(args: argparse.Namespace) -> dict[str, Any]:
    selection = ensure_screen_report(args)
    expected = _final_task_fingerprint(args, selection)
    path = Path(args.out_dir) / "03_final_models" / "final_training_report.json"
    if path.exists():
        cached = read_json(path)
        model_paths = cached.get("model_paths", {})
        if (
            _phase_compatible(
                cached,
                strategy_version=TRAINING_STRATEGY_VERSION,
                task_fingerprint=expected,
            )
            and len(model_paths) == len(_resolved_seeds(args))
            and all(Path(model_path).is_file() for model_path in model_paths.values())
        ):
            return cached
    return phase_train(args)


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "04_summary"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    split = ensure_split_report(args)
    selection = ensure_screen_report(args)
    training = ensure_training_report(args)
    validation = pd.read_csv(
        Path(args.out_dir) / "03_final_models" / "validation_92p5_metrics_all_seeds.csv"
    )
    test = pd.read_csv(Path(args.out_dir) / "03_final_models" / "test_100_metrics_all_seeds.csv")
    lines = [
        "# True Ellipse V6 whole-radius model summary",
        "",
        f"- Upstream formal dataset gate: `{audit['formal_claims_allowed']}`.",
        f"- Training rows (non-centerline only): `{split['training_rows']}`.",
        f"- Validation radius: `{float(args.validation_radius_mm):g} mm` (never used for fitting or selection outside validation).",
        f"- Test radius: `{float(args.test_radius_mm):g} mm` (used only after model selection).",
        f"- Selected configuration: `{selection['selected_config_id']}`.",
        f"- 100 mm stable seed gate: `{training['test_100_gate']['stable_gate_pass']}` ({training['test_100_gate']['passed_seed_count']}/{training['test_100_gate']['total_seed_count']}).",
        f"- Formal model claim: `{training['formal_model_gate_pass']}`.",
        "",
        "## Validation (92.5 mm)",
        "",
        dataframe_to_markdown(validation),
        "",
        "## Final untouched test (100 mm)",
        "",
        dataframe_to_markdown(test),
        "",
        "Only Cartesian target coordinates (x, y, z) are model inputs. Radius, angle, family and branch metadata are retained for auditing and never exposed to the regressor.",
        "",
    ]
    summary_path = out / "experiment_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    report = {
        "audit_gate_pass": bool(audit["audit_gate_pass"]),
        "split_gate_pass": bool(split["split_gate_pass"]),
        "selected_config_id": selection["selected_config_id"],
        "validation_92p5_gate_pass": bool(training["validation_92p5_gate"]["stable_gate_pass"]),
        "test_100_gate_pass": bool(training["test_100_gate"]["stable_gate_pass"]),
        "formal_model_gate_pass": bool(training["formal_model_gate_pass"]),
        "summary_path": str(summary_path.resolve()),
    }
    write_json(out / "final_report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V6 static beta6 inverse on whole-radius train/validation/test partitions."
    )
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--tube-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phases", default="all")
    parser.add_argument("--preset", choices=("smoke", "pilot", "formal"), default="formal")
    parser.add_argument("--validation-radius-mm", type=float, default=VALIDATION_RADIUS_MM)
    parser.add_argument("--test-radius-mm", type=float, default=TEST_RADIUS_MM)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FORMAL_SEEDS))
    parser.add_argument("--screen-config-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    phases = parse_phases(args.phases)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    handlers = {
        "audit": phase_audit,
        "split": phase_split,
        "screen": phase_screen,
        "train": phase_train,
        "summary": phase_summary,
    }
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for phase in phases:
        results[phase] = handlers[phase](args)
    payload = {
        "mode": "true_ellipse_radial_bundle_training_v6",
        "preset": str(args.preset),
        "phases": phases,
        "elapsed_s": float(time.perf_counter() - started),
        "results": results,
    }
    write_json(Path(args.out_dir) / "run_report.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
