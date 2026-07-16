#!/usr/bin/env python3
"""Train the V7 static beta6 inverse under the registered standard domain."""

from __future__ import annotations

import argparse
import json
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
from joblib import dump


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_family_training_v5 as v5  # noqa: E402
import run_true_ellipse_standard_domain_v7 as upstream_v7  # noqa: E402
import true_ellipse_radial_bundle_engine as engine  # noqa: E402
from true_ellipse_family_v5_utils import (  # noqa: E402
    dataframe_to_markdown,
    file_sha256,
    stable_fingerprint,
)


MODEL_INPUT_COLUMNS = ["x_target_m", "y_target_m", "z_target_m"]
BETA_COLUMNS = [f"beta{index}_rad" for index in range(1, 7)]
FORMAL_SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
V4_BASELINE_CONFIG_ID = v5.V4_BASELINE_CONFIG_ID
DEFAULT_UPSTREAM_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_v7"
DEFAULT_DATASET = (
    DEFAULT_UPSTREAM_DIR / "03_dataset" / "true_ellipse_standard_domain_tubes_v7.parquet"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_training_v7"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"
ALL_PHASES = ("audit", "split", "screen", "train", "summary")
TRAINING_STRATEGY_VERSION = 2

write_json = upstream_v7.write_json
read_json = upstream_v7.read_json
_json_default = upstream_v7._json_default


@dataclass(frozen=True)
class LinkedModelConfig:
    base_config: v5.v4.ModelConfig
    output_link_id: str

    @property
    def config_id(self) -> str:
        return f"{self.base_config.config_id}__{self.output_link_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "base_config": asdict(self.base_config),
            "output_link_id": self.output_link_id,
            "output_link_fingerprint": engine.registered_output_link(
                self.output_link_id
            ).fingerprint,
        }


@dataclass(frozen=True)
class TrainingInputs:
    dataset_path: Path
    dataset_sha256: str
    assignment_path: Path
    assignment_sha256: str
    robot_config_path: Path
    robot_config_sha256: str
    theta_sign: float
    family_id: str
    challenge_paths: dict[str, Path]
    challenge_hashes: dict[str, str]


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
        raise ValueError(f"unsupported V7 training phases: {unknown}")
    return phases


def model_configs() -> list[LinkedModelConfig]:
    return [
        LinkedModelConfig(base_config=base, output_link_id=link_id)
        for base in v5.v5_model_configs()
        for link_id in ("identity", "tanh_bounds")
    ]


def _config_from_dict(payload: Mapping[str, Any]) -> LinkedModelConfig:
    base = payload["base_config"]
    return LinkedModelConfig(
        base_config=v5.v4.ModelConfig(
            architecture=str(base["architecture"]),
            hidden_layers=tuple(int(value) for value in base["hidden_layers"]),
            feature_set=str(base["feature_set"]),
            activation=str(base["activation"]),
            alpha=float(base["alpha"]),
        ),
        output_link_id=str(payload["output_link_id"]),
    )


def encode_training_targets(
    beta_rad: np.ndarray,
    *,
    output_link_id: str,
    domain: engine.JointDomainSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    return engine.registered_output_link(output_link_id).encode(beta_rad, domain=domain)


def deployed_centerline_model_gate(metrics: Mapping[str, Any]) -> bool:
    return bool(
        v5.v4.centerline_model_gate(dict(metrics), require_beta_error=True)
        and int(metrics.get("beta_bound_violation_count", 1)) == 0
        and float(metrics.get("prediction_min_joint_margin_deg", -np.inf)) >= 0.01
    )


def final_evaluation_specs(
    *,
    preset: str,
    holdout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validation = float(holdout["validation_radius_mm"])
    test = float(holdout["test_radius_mm"])
    specs = [
        {
            "label": "validation_integer_centerline",
            "split": "validation",
            "radius_mm": validation,
            "source": "dataset",
            "evaluation_kind": "integer_centerline",
        },
        {
            "label": "validation_half_phase",
            "split": "validation",
            "radius_mm": validation,
            "source": "challenge",
            "evaluation_kind": "half_phase_centerline",
        },
        {
            "label": "validation_tube_diagnostic",
            "split": "validation",
            "radius_mm": validation,
            "source": "dataset",
            "evaluation_kind": "tube_diagnostic",
        },
    ]
    if str(preset) == "formal":
        specs.extend(
            [
                {
                    "label": "test_integer_centerline",
                    "split": "test",
                    "radius_mm": test,
                    "source": "dataset",
                    "evaluation_kind": "integer_centerline",
                },
                {
                    "label": "test_half_phase",
                    "split": "test",
                    "radius_mm": test,
                    "source": "challenge",
                    "evaluation_kind": "half_phase_centerline",
                },
                {
                    "label": "test_tube_diagnostic",
                    "split": "test",
                    "radius_mm": test,
                    "source": "dataset",
                    "evaluation_kind": "tube_diagnostic",
                },
            ]
        )
    return specs


def formal_model_gate_pass(
    *,
    formal_claims_allowed: bool,
    seed_protocol_pass: bool,
    target_link_clip_count: int,
    gates: Mapping[str, Mapping[str, Any]],
) -> bool:
    required = (
        "validation_integer_centerline",
        "validation_half_phase",
        "test_integer_centerline",
        "test_half_phase",
    )
    return bool(
        formal_claims_allowed
        and seed_protocol_pass
        and int(target_link_clip_count) == 0
        and all(bool(gates.get(label, {}).get("stable_gate_pass", False)) for label in required)
    )


def make_training_assignment(
    dataset: pd.DataFrame,
    *,
    validation_radius_mm: float,
    test_radius_mm: float,
    require_test: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "sample_id",
        "family_id",
        "trajectory_id",
        "radius_mm",
        "angle_idx",
        "is_centerline",
        "split",
        "used_for_training",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"V7 training assignment missing columns: {missing}")
    if not pd.api.types.is_bool_dtype(dataset["is_centerline"].dtype):
        raise ValueError("V7 training assignment requires boolean is_centerline")
    assignment = dataset[
        [
            "sample_id",
            "family_id",
            "trajectory_id",
            "radius_mm",
            "angle_idx",
            "is_centerline",
            "split",
            "used_for_training",
        ]
    ].copy()
    radius = assignment["radius_mm"].to_numpy(dtype=float)
    expected_split = np.full(len(assignment), "train", dtype=object)
    expected_split[np.isclose(radius, float(validation_radius_mm), atol=1.0e-8)] = "validation"
    expected_split[np.isclose(radius, float(test_radius_mm), atol=1.0e-8)] = "test"
    split_matches = bool(np.array_equal(assignment["split"].astype(str).to_numpy(), expected_split))
    expected_training = (expected_split == "train") & ~assignment["is_centerline"].to_numpy(
        dtype=bool
    )
    training_matches = bool(
        np.array_equal(assignment["used_for_training"].to_numpy(dtype=bool), expected_training)
    )
    trajectory_leakage = int((assignment.groupby("trajectory_id")["split"].nunique() > 1).sum())
    radius_leakage = int((assignment.groupby("radius_mm")["split"].nunique() > 1).sum())
    validation_present = bool(
        np.any(np.isclose(radius, float(validation_radius_mm), atol=1.0e-8))
    )
    test_present = bool(np.any(np.isclose(radius, float(test_radius_mm), atol=1.0e-8)))
    report = {
        "rows": int(len(assignment)),
        "validation_radius_mm": float(validation_radius_mm),
        "test_radius_mm": float(test_radius_mm),
        "training_rows": int(expected_training.sum()),
        "training_radius_count": int(assignment.loc[expected_training, "radius_mm"].nunique()),
        "training_centerline_count": int(
            assignment.loc[expected_training, "is_centerline"].astype(bool).sum()
        ),
        "validation_present": validation_present,
        "test_present": test_present,
        "upstream_split_matches": split_matches,
        "upstream_training_mask_matches": training_matches,
        "trajectory_leakage_count": trajectory_leakage,
        "radius_leakage_count": radius_leakage,
    }
    report["split_gate_pass"] = bool(
        len(assignment)
        and report["training_rows"] > 0
        and report["training_radius_count"] >= 1
        and report["training_centerline_count"] == 0
        and validation_present
        and (test_present or not bool(require_test))
        and split_matches
        and training_matches
        and trajectory_leakage == 0
        and radius_leakage == 0
    )
    return assignment, report


def preset_settings(preset: str) -> dict[str, int]:
    if str(preset) == "smoke":
        return {"angle_stride": 30, "max_iter": 30, "config_limit": 2, "seed_count": 1}
    if str(preset) == "pilot":
        return {"angle_stride": 5, "max_iter": 300, "config_limit": 8, "seed_count": 3}
    if str(preset) == "formal":
        return {"angle_stride": 2, "max_iter": 800, "config_limit": 0, "seed_count": 5}
    raise ValueError(f"unsupported preset: {preset}")


def _resolved_seeds(args: argparse.Namespace) -> list[int]:
    seeds = parse_int_csv(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("V7 training seeds must be non-empty and unique")
    if str(args.preset) == "formal":
        return seeds
    return seeds[: int(preset_settings(str(args.preset))["seed_count"])]


def _upstream_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.upstream_dir) / "03_dataset" / "dataset_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"V7 upstream dataset report is missing: {path}")
    return read_json(path)


def resolve_holdout(
    args: argparse.Namespace,
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    report = _upstream_report(args) if upstream is None else upstream
    registered = report.get("holdout") or {}
    validation = float(registered.get("validation_radius_mm", np.nan))
    test = float(registered.get("test_radius_mm", np.nan))
    if not np.isfinite(validation) or not np.isfinite(test):
        raise ValueError("V7 upstream report has no registered dynamic holdout")
    if args.validation_radius_mm is not None and not np.isclose(
        float(args.validation_radius_mm), validation, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("validation radius override disagrees with the upstream registration")
    if args.test_radius_mm is not None and not np.isclose(
        float(args.test_radius_mm), test, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("test radius override disagrees with the upstream registration")
    return {"validation_radius_mm": validation, "test_radius_mm": test}


def resolve_training_sources(
    args: argparse.Namespace,
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a test-free source view for smoke/pilot before opening data."""

    formal = str(args.preset) == "formal"
    path_key = "dataset_path" if formal else "nonformal_dataset_path"
    hash_key = "dataset_sha256" if formal else "nonformal_dataset_sha256"
    raw_path = upstream.get(path_key)
    expected_hash = str(upstream.get(hash_key, ""))
    if not raw_path or not expected_hash:
        raise ValueError(f"upstream dataset report is missing {path_key}/{hash_key}")
    dataset_path = Path(str(raw_path)).resolve()
    if not dataset_path.is_file() or file_sha256(dataset_path) != expected_hash:
        raise ValueError(f"registered training dataset changed: {path_key}")
    if formal and Path(args.tube_dataset).resolve() != dataset_path:
        raise ValueError("formal dataset override disagrees with upstream registration")

    registered_paths = upstream.get("challenge_paths") or {}
    registered_reports = upstream.get("challenge_reports") or {}
    active_splits = ("validation", "test") if formal else ("validation",)
    challenge_paths: dict[str, Path] = {}
    challenge_hashes: dict[str, str] = {}
    for split in active_splits:
        raw_challenge = registered_paths.get(split)
        expected_challenge_hash = str(
            registered_reports.get(split, {}).get("challenge_artifact_sha256", "")
        )
        if not raw_challenge or not expected_challenge_hash:
            raise ValueError(f"registered {split} challenge is missing")
        challenge_path = Path(str(raw_challenge)).resolve()
        if (
            not challenge_path.is_file()
            or file_sha256(challenge_path) != expected_challenge_hash
        ):
            raise ValueError(f"registered {split} challenge changed")
        challenge_paths[split] = challenge_path
        challenge_hashes[split] = expected_challenge_hash
    return {
        "dataset_path": dataset_path,
        "dataset_sha256": expected_hash,
        "challenge_paths": challenge_paths,
        "challenge_hashes": challenge_hashes,
        "test_access_allowed": formal,
    }


def formal_training_protocol_report(
    args: argparse.Namespace,
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = _upstream_report(args) if upstream is None else dict(upstream)
    holdout = resolve_holdout(args, source)
    configs = model_configs()
    config_ids = [config.config_id for config in configs]
    seeds = parse_int_csv(args.seeds)
    checks = {
        "formal_preset": str(args.preset) == "formal",
        "upstream_formal_dataset": bool(source.get("formal_dataset_gate_pass", False)),
        "dynamic_test_at_least_105mm": float(holdout["test_radius_mm"]) >= 105.0,
        "registered_validation_gap_7p5mm": np.isclose(
            holdout["test_radius_mm"] - holdout["validation_radius_mm"], 7.5
        ),
        "exact_five_seeds": seeds == list(FORMAL_SEEDS),
        "screen_config_limit_zero": int(args.screen_config_limit) == 0,
        "exact_48_configs": len(config_ids) == 48 and len(set(config_ids)) == 48,
        "two_registered_output_links": {config.output_link_id for config in configs}
        == {"identity", "tanh_bounds"},
        "both_v4_baselines_present": sum(
            config.base_config.config_id == V4_BASELINE_CONFIG_ID for config in configs
        )
        == 2,
        "xyz_only_model_inputs": MODEL_INPUT_COLUMNS == list(v5.v4.TARGET_XYZ_COLS),
        "standard_joint_domain": engine.registered_joint_domain(
            "standard_beta34_10deg_v1"
        ).domain_id
        == "standard_beta34_10deg_v1",
        "declared_runtime": upstream_v7.declared_runtime_gate(),
    }
    normalized = {key: bool(value) for key, value in checks.items()}
    protocol = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "preset": str(args.preset),
        **holdout,
        "seeds": seeds,
        "screen_config_limit": int(args.screen_config_limit),
        "screen_config_ids": config_ids,
        "model_input_columns": list(MODEL_INPUT_COLUMNS),
        "joint_domain_id": "standard_beta34_10deg_v1",
        "joint_domain_fingerprint": engine.registered_joint_domain(
            "standard_beta34_10deg_v1"
        ).fingerprint,
        "output_link_ids": ["identity", "tanh_bounds"],
        "settings": preset_settings(str(args.preset)),
        "runtime": upstream_v7.runtime_environment_report(),
    }
    return {
        "formal_training_protocol_gate_pass": bool(all(normalized.values())),
        "checks": normalized,
        "protocol": protocol,
        "protocol_fingerprint": stable_fingerprint(protocol),
    }


def _audit_task_fingerprint(args: argparse.Namespace) -> str:
    report_path = Path(args.upstream_dir) / "03_dataset" / "dataset_report.json"
    upstream = read_json(report_path) if report_path.is_file() else {}
    protocol = formal_training_protocol_report(args, upstream) if upstream.get("holdout") else {}
    sources = resolve_training_sources(args, upstream) if upstream.get("holdout") else None
    return stable_fingerprint(
        {
            "phase": "v7_training_audit",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "upstream_report": file_sha256(report_path) if report_path.is_file() else "missing",
            "dataset": sources["dataset_sha256"] if sources else "missing",
            "challenges": sources["challenge_hashes"] if sources else {},
            "robot_config": file_sha256(args.robot_config)
            if Path(args.robot_config).is_file()
            else "missing",
            "protocol": protocol.get("protocol_fingerprint", "missing"),
        }
    )


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_audit"
    out.mkdir(parents=True, exist_ok=True)
    upstream_path = Path(args.upstream_dir) / "03_dataset" / "dataset_report.json"
    if not upstream_path.is_file():
        raise FileNotFoundError(f"V7 upstream dataset report is missing: {upstream_path}")
    upstream = read_json(upstream_path)
    holdout = resolve_holdout(args, upstream)
    protocol = formal_training_protocol_report(args, upstream)
    sources = resolve_training_sources(args, upstream)
    dataset_path = Path(sources["dataset_path"])
    dataset = pd.read_parquet(dataset_path)
    required = {
        "sample_id",
        "family_id",
        "trajectory_id",
        "radius_mm",
        "angle_idx",
        "is_centerline",
        "split",
        "used_for_training",
        *MODEL_INPUT_COLUMNS,
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(dataset.columns))
    challenge_paths = dict(sources["challenge_paths"])
    challenge_hashes = dict(sources["challenge_hashes"])
    challenge_reports = upstream.get("challenge_reports") or {}
    domain = engine.registered_joint_domain("standard_beta34_10deg_v1")
    robot_config = v5.v4.load_config(str(args.robot_config))
    configured = robot_config.get("sampling", {}).get("beta_ranges_rad", {})
    configured_bounds = np.asarray(
        [configured.get(f"beta{index}", [np.nan, np.nan]) for index in range(1, 7)],
        dtype=float,
    )
    checks = {
        "upstream_formal_dataset": bool(upstream.get("formal_dataset_gate_pass", False)),
        "dataset_path_bound": bool(
            dataset_path == Path(sources["dataset_path"])
        ),
        "dataset_hash_bound": str(sources["dataset_sha256"]) == file_sha256(dataset_path),
        "required_columns": not missing,
        "unique_sample_ids": bool("sample_id" in dataset and dataset["sample_id"].is_unique),
        "single_family": bool("family_id" in dataset and dataset["family_id"].nunique() == 1),
        "validation_challenge_bound": bool(
            challenge_paths.get("validation", Path("/missing")).is_file()
            and challenge_reports.get("validation", {}).get("challenge_gate_pass", False)
            and challenge_reports.get("validation", {}).get("challenge_artifact_sha256")
            == file_sha256(challenge_paths["validation"])
        ),
        "standard_domain_matches_config": bool(
            configured_bounds.shape == (6, 2)
            and np.isfinite(configured_bounds).all()
            and np.allclose(configured_bounds, domain.bounds_rad, atol=1.0e-12, rtol=0.0)
        ),
        "registered_validation_present": bool(
            np.any(np.isclose(dataset["radius_mm"], holdout["validation_radius_mm"]))
        ),
    }
    if str(args.preset) == "formal":
        checks["test_challenge_bound"] = bool(
            challenge_paths.get("test", Path("/missing")).is_file()
            and challenge_reports.get("test", {}).get("challenge_gate_pass", False)
            and challenge_reports.get("test", {}).get("challenge_artifact_sha256")
            == challenge_hashes.get("test")
        )
        checks["registered_test_present"] = bool(
            np.any(np.isclose(dataset["radius_mm"], holdout["test_radius_mm"]))
        )
    else:
        checks["registered_test_inaccessible"] = bool(
            "test" not in challenge_paths
            and not np.any(np.isclose(dataset["radius_mm"], holdout["test_radius_mm"]))
        )
    audit_gate = bool(all(checks.values()))
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": _audit_task_fingerprint(args),
        **protocol,
        "upstream_dataset_task_fingerprint": str(upstream.get("task_fingerprint", "")),
        "holdout": holdout,
        "family_id": str(dataset["family_id"].iloc[0]) if len(dataset) and "family_id" in dataset else None,
        "checks_recomputed": checks,
        "missing_columns": missing,
        "audit_gate_pass": audit_gate,
        "formal_claims_allowed": bool(audit_gate and protocol["formal_training_protocol_gate_pass"]),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": str(sources["dataset_sha256"]),
        "challenge_paths": {key: str(path.resolve()) for key, path in challenge_paths.items()},
        "challenge_hashes": challenge_hashes,
        "test_access_allowed": bool(sources["test_access_allowed"]),
        "robot_config_path": str(Path(args.robot_config).resolve()),
        "robot_config_sha256": file_sha256(args.robot_config),
    }
    write_json(out / "audit_report.json", report)
    if not audit_gate:
        raise RuntimeError(f"V7 training audit gate failed: {checks}")
    return report


def ensure_audit_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "00_audit" / "audit_report.json"
    expected = _audit_task_fingerprint(args)
    if path.exists():
        cached = read_json(path)
        if (
            int(cached.get("strategy_version", -1)) == TRAINING_STRATEGY_VERSION
            and str(cached.get("task_fingerprint", "")) == expected
        ):
            return cached
    return phase_audit(args)


def _split_task_fingerprint(args: argparse.Namespace, audit: Mapping[str, Any]) -> str:
    return stable_fingerprint(
        {
            "phase": "v7_training_split",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "audit": audit.get("task_fingerprint", ""),
            "dataset": str(audit.get("dataset_sha256", "")),
            "holdout": audit["holdout"],
        }
    )


def phase_split(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_split"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    dataset = pd.read_parquet(audit["dataset_path"])
    assignment, split = make_training_assignment(
        dataset,
        **audit["holdout"],
        require_test=str(args.preset) == "formal",
    )
    assignment.insert(0, "row_index", np.arange(len(assignment), dtype=np.int64))
    assignment_path = out / "split_assignment.parquet"
    assignment.to_parquet(assignment_path, index=False, compression="zstd")
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": _split_task_fingerprint(args, audit),
        "source_audit_task_fingerprint": audit["task_fingerprint"],
        "formal_claims_allowed": bool(audit.get("formal_claims_allowed", False)),
        "family_id": audit["family_id"],
        **split,
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": file_sha256(assignment_path),
    }
    write_json(out / "split_report.json", report)
    if not report["split_gate_pass"]:
        raise RuntimeError(f"V7 training split gate failed: {report}")
    return report


def ensure_split_report(args: argparse.Namespace) -> dict[str, Any]:
    audit = ensure_audit_report(args)
    expected = _split_task_fingerprint(args, audit)
    path = Path(args.out_dir) / "01_split" / "split_report.json"
    assignment_path = Path(args.out_dir) / "01_split" / "split_assignment.parquet"
    if path.exists() and assignment_path.exists():
        cached = read_json(path)
        if (
            str(cached.get("task_fingerprint", "")) == expected
            and str(cached.get("assignment_sha256", "")) == file_sha256(assignment_path)
        ):
            return cached
    return phase_split(args)


def resolve_training_inputs(args: argparse.Namespace) -> TrainingInputs:
    audit = read_json(Path(args.out_dir) / "00_audit" / "audit_report.json")
    split = read_json(Path(args.out_dir) / "01_split" / "split_report.json")
    robot_config = v5.v4.load_config(str(args.robot_config))
    challenge_paths = {key: Path(value).resolve() for key, value in audit["challenge_paths"].items()}
    return TrainingInputs(
        dataset_path=Path(audit["dataset_path"]).resolve(),
        dataset_sha256=str(audit["dataset_sha256"]),
        assignment_path=Path(split["assignment_path"]).resolve(),
        assignment_sha256=file_sha256(split["assignment_path"]),
        robot_config_path=Path(args.robot_config).resolve(),
        robot_config_sha256=file_sha256(args.robot_config),
        theta_sign=float(robot_config.get("kinematics", {}).get("theta_sign", -1.0)),
        family_id=str(split["family_id"]),
        challenge_paths=challenge_paths,
        challenge_hashes={key: file_sha256(path) for key, path in challenge_paths.items()},
    )


def training_task_fingerprint(task: Mapping[str, Any]) -> str:
    ignored = {"task_fingerprint", "result_path", "package_path", "prediction_dir"}
    semantic = {key: value for key, value in task.items() if key not in ignored}
    return stable_fingerprint(
        {"strategy_version": TRAINING_STRATEGY_VERSION, "task": semantic}
    )


def _training_task(
    args: argparse.Namespace,
    *,
    shared: TrainingInputs,
    task_id: str,
    mode: str,
    config: LinkedModelConfig,
    seed: int,
    result_path: Path,
    package_path: Path | None,
    prediction_dir: Path | None,
    angle_stride: int,
    max_iter: int,
    evaluation_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "task_id": str(task_id),
        "mode": str(mode),
        "preset": str(args.preset),
        "training_strategy_version": TRAINING_STRATEGY_VERSION,
        "config": config.as_dict(),
        "seed": int(seed),
        "dataset": str(shared.dataset_path),
        "dataset_sha256": shared.dataset_sha256,
        "assignment": str(shared.assignment_path),
        "assignment_sha256": shared.assignment_sha256,
        "robot_config": str(shared.robot_config_path),
        "robot_config_sha256": shared.robot_config_sha256,
        "theta_sign": shared.theta_sign,
        "family_id": shared.family_id,
        "challenge_paths": {key: str(value) for key, value in shared.challenge_paths.items()},
        "challenge_hashes": shared.challenge_hashes,
        "joint_domain_id": "standard_beta34_10deg_v1",
        "joint_domain_fingerprint": engine.registered_joint_domain(
            "standard_beta34_10deg_v1"
        ).fingerprint,
        "result_path": str(result_path.resolve()),
        "package_path": str(package_path.resolve()) if package_path is not None else None,
        "prediction_dir": str(prediction_dir.resolve()) if prediction_dir is not None else None,
        "angle_stride": int(angle_stride),
        "max_iter": int(max_iter),
        "batch_size": 256,
        "evaluation_specs": evaluation_specs,
    }
    task["task_fingerprint"] = training_task_fingerprint(task)
    return task


def predict_beta(package: Mapping[str, Any], xyz: np.ndarray) -> np.ndarray:
    features, _names = v5.v4.build_features(
        np.asarray(xyz, dtype=float), feature_set=str(package["feature_set"])
    )
    pred_scaled = package["model"].predict(package["x_scaler"].transform(features))
    latent = np.asarray(package["y_scaler"].inverse_transform(pred_scaled), dtype=float).reshape(
        -1, 6
    )
    domain = engine.registered_joint_domain(str(package["joint_domain_id"]))
    link = engine.registered_output_link(str(package["output_link_id"]))
    return link.decode(latent, domain=domain)


def _evaluation_frame(
    *,
    dataset: pd.DataFrame,
    assignment: pd.DataFrame,
    task: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    split = str(spec["split"])
    radius = float(spec["radius_mm"])
    kind = str(spec["evaluation_kind"])
    if str(spec["source"]) == "challenge":
        path = Path(task["challenge_paths"][split])
        if file_sha256(path) != str(task["challenge_hashes"][split]):
            raise ValueError(f"challenge artifact changed after task registration: {split}")
        frame = pd.read_parquet(path)
        if not frame["split"].astype(str).eq(split).all():
            raise ValueError("challenge split metadata mismatch")
        if not np.isclose(frame["radius_mm"].to_numpy(dtype=float), radius, atol=1.0e-8).all():
            raise ValueError("challenge radius metadata mismatch")
        return frame.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    mask = assignment["split"].astype(str).eq(split).to_numpy()
    mask &= np.isclose(dataset["radius_mm"].to_numpy(dtype=float), radius, atol=1.0e-8)
    mask &= dataset["family_id"].astype(str).eq(str(task["family_id"])).to_numpy()
    if kind == "integer_centerline":
        mask &= dataset["is_centerline"].astype(bool).to_numpy()
    elif kind != "tube_diagnostic":
        raise ValueError(f"unsupported dataset evaluation kind: {kind}")
    return dataset.loc[mask].sort_values(
        [column for column in ("tube_offset_id", "angle_idx") if column in dataset],
        kind="stable",
    ).reset_index(drop=True)


def run_training_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    expected = training_task_fingerprint(task)
    if str(task.get("task_fingerprint", "")) != expected:
        raise ValueError("V7 training task fingerprint mismatch")
    for path_key, hash_key in (
        ("dataset", "dataset_sha256"),
        ("assignment", "assignment_sha256"),
        ("robot_config", "robot_config_sha256"),
    ):
        if file_sha256(task[path_key]) != str(task[hash_key]):
            raise ValueError(f"V7 training input changed after fingerprinting: {path_key}")
    dataset = pd.read_parquet(task["dataset"])
    assignment = pd.read_parquet(task["assignment"])
    if len(dataset) != len(assignment) or not dataset["sample_id"].astype(str).reset_index(
        drop=True
    ).equals(assignment["sample_id"].astype(str).reset_index(drop=True)):
        raise ValueError("V7 dataset and assignment are misaligned")
    config = _config_from_dict(task["config"])
    seed = int(task["seed"])
    stride = max(1, int(task["angle_stride"]))
    training_mask = assignment["used_for_training"].astype(bool).to_numpy()
    training_mask &= dataset["family_id"].astype(str).eq(str(task["family_id"])).to_numpy()
    training_mask &= dataset["angle_idx"].to_numpy(dtype=np.int64) % stride == 0
    evaluation_radii = [float(spec["radius_mm"]) for spec in task["evaluation_specs"]]
    for radius in evaluation_radii:
        if np.any(
            training_mask
            & np.isclose(dataset["radius_mm"].to_numpy(dtype=float), radius, atol=1.0e-8)
        ):
            raise ValueError("V7 evaluation radius leaked into the training partition")
    train_idx = np.flatnonzero(training_mask).astype(np.int64)
    if not len(train_idx):
        raise ValueError("V7 training worker received an empty training partition")
    domain = engine.registered_joint_domain(str(task["joint_domain_id"]))
    if domain.fingerprint != str(task["joint_domain_fingerprint"]):
        raise ValueError("V7 training task joint-domain fingerprint mismatch")
    beta_train = dataset.iloc[train_idx][BETA_COLUMNS].to_numpy(dtype=float)
    latent_train, link_report = encode_training_targets(
        beta_train,
        output_link_id=config.output_link_id,
        domain=domain,
    )
    model, x_scaler, y_scaler, feature_names, fit_s = v5.v4._fit_scaled_model(
        config=config.base_config,
        seed=seed,
        xyz_train=dataset.iloc[train_idx][MODEL_INPUT_COLUMNS].to_numpy(dtype=float),
        beta_train=latent_train,
        max_iter=int(task["max_iter"]),
        batch_size=int(task.get("batch_size", 256)),
    )
    package: dict[str, Any] = {
        "kind": "beta6_pose_standard_domain_v7",
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": list(MODEL_INPUT_COLUMNS),
        "feature_set": config.base_config.feature_set,
        "feature_names": feature_names,
        "target_cols": list(BETA_COLUMNS),
        "beta_cols": list(BETA_COLUMNS),
        "theta_cols": list(v5.v4.THETA_COLS),
        "theta_sign": float(task["theta_sign"]),
        "joint_domain_id": domain.domain_id,
        "joint_domain_fingerprint": domain.fingerprint,
        "output_link_id": config.output_link_id,
        "output_link_fingerprint": engine.registered_output_link(
            config.output_link_id
        ).fingerprint,
        "target_link_clip_count": int(link_report["target_link_clip_count"]),
        "robot_config": str(task["robot_config"]),
        "dataset": str(task["dataset"]),
        "family_id": str(task["family_id"]),
        "seed": seed,
        "model_name": config.base_config.architecture,
        "activation": config.base_config.activation,
        "alpha": config.base_config.alpha,
        "task_fingerprint": expected,
    }
    robot_config = v5.v4.load_config(str(task["robot_config"]))
    robot_inputs = v5.v4.load_robot_inputs(robot_config)
    evaluations: dict[str, Any] = {}
    prediction_dir = Path(task["prediction_dir"]) if task.get("prediction_dir") else None
    for spec in task["evaluation_specs"]:
        label = str(spec["label"])
        frame = _evaluation_frame(
            dataset=dataset,
            assignment=assignment,
            task=task,
            spec=spec,
        )
        if frame.empty:
            evaluations[label] = {
                "rows": 0,
                "model_gate_pass": False,
                "reason": "registered_evaluation_not_materialized",
                "radius_mm": float(spec["radius_mm"]),
            }
            continue
        eval_xyz = frame[MODEL_INPUT_COLUMNS].to_numpy(dtype=float)
        beta_true = frame[BETA_COLUMNS].to_numpy(dtype=float)
        beta_pred = predict_beta(package, eval_xyz)
        is_tube = str(spec["evaluation_kind"]) == "tube_diagnostic"
        groups = frame["tube_offset_id"].astype(str).to_numpy() if is_tube else None
        angle = frame["angle_rad"].to_numpy(dtype=float) if is_tube else None
        metrics, achieved, _theta = v5.v4.evaluate_beta_prediction(
            beta_pred=beta_pred,
            beta_true=beta_true,
            target_xyz=eval_xyz,
            lengths_m=robot_inputs.lengths_m,
            p_end_local_m=robot_inputs.p_end_local_m,
            theta_sign=float(task["theta_sign"]),
            groups=groups,
            angle=angle,
            periodic=True,
            joint_domain=domain,
        )
        metrics.update(
            {
                "radius_mm": float(spec["radius_mm"]),
                "split": str(spec["split"]),
                "evaluation_kind": str(spec["evaluation_kind"]),
                "output_link_id": config.output_link_id,
                "fit_s": fit_s,
                "n_iter": int(model.n_iter_),
                "loss": float(model.loss_),
                "train_rows": int(len(train_idx)),
                "eval_rows": int(len(frame)),
                "model_gate_pass": deployed_centerline_model_gate(metrics),
                "formal_gate_role": not is_tube,
            }
        )
        evaluations[label] = metrics
        if prediction_dir is not None:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            keep = [
                column
                for column in (
                    "trajectory_id",
                    "family_id",
                    "radius_mm",
                    "angle_idx",
                    "angle_rad",
                    "tube_offset_id",
                    *MODEL_INPUT_COLUMNS,
                )
                if column in frame
            ]
            prediction = frame[keep].copy()
            for index, column in enumerate(BETA_COLUMNS):
                prediction[f"true_{column}"] = beta_true[:, index]
                prediction[f"pred_{column}"] = beta_pred[:, index]
            for index, axis in enumerate("xyz"):
                prediction[f"achieved_{axis}_m"] = achieved[:, index]
            prediction["ee_err_mm"] = np.linalg.norm(achieved - eval_xyz, axis=1) * 1000.0
            prediction.to_parquet(
                prediction_dir / f"{label}.parquet", index=False, compression="zstd"
            )
    package_path = Path(task["package_path"]) if task.get("package_path") else None
    if package_path is not None:
        package_path.parent.mkdir(parents=True, exist_ok=True)
        dump(package, package_path)
    result = {
        "task_id": str(task["task_id"]),
        "mode": str(task["mode"]),
        "seed": seed,
        "config_id": config.config_id,
        "config": config.as_dict(),
        "output_link_id": config.output_link_id,
        "target_link_clip_count": int(link_report["target_link_clip_count"]),
        "train_rows": int(len(train_idx)),
        "fit_s": fit_s,
        "evaluations": evaluations,
        "package_path": str(package_path) if package_path is not None else None,
        "task_fingerprint": expected,
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
        expected = training_task_fingerprint(task)
        if str(task.get("task_fingerprint", "")) != expected:
            raise ValueError(f"stale V7 task fingerprint: {task.get('task_id')}")
        task_path = task_dir / f"{task['task_id']}.json"
        result_path = Path(task["result_path"])
        if bool(skip_existing) and result_path.exists():
            cached = read_json(result_path)
            package_ready = bool(not task.get("package_path") or Path(task["package_path"]).exists())
            if (
                package_ready
                and str(cached.get("task_fingerprint", "")) == expected
            ):
                results.append(cached)
                continue
        write_json(task_path, task)
        pending.append(task_path)

    def execute(task_path: Path) -> dict[str, Any]:
        env = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path)],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"V7 training worker failed for {task_path}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        task = read_json(task_path)
        return read_json(Path(task["result_path"]))

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            results.extend(executor.map(execute, pending))
    return sorted(results, key=lambda result: str(result["task_id"]))


def _flatten_results(results: list[dict[str, Any]], *, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        base = result["config"]["base_config"]
        rows.append(
            {
                "task_id": result["task_id"],
                "mode": result["mode"],
                "seed": int(result["seed"]),
                "config_id": result["config_id"],
                **base,
                "output_link_id": result["output_link_id"],
                "target_link_clip_count": int(result["target_link_clip_count"]),
                **dict(result["evaluations"].get(label, {})),
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
                "output_link_id": first["output_link_id"],
                "screen_seed_count": int(len(part)),
                "all_seed_gate_pass": bool(
                    part["model_gate_pass"].fillna(False).astype(bool).all()
                ),
                "passed_seed_count": int(
                    part["model_gate_pass"].fillna(False).astype(bool).sum()
                ),
                "ee_p95_median_mm": float(part["ee_p95_mm"].median()),
                "beta_p95_median_deg": float(part["beta_p95_deg"].median()),
                "axis_p95_median_mm": float(part["axiserr_max_p95_abs_mm"].median()),
                "prediction_min_margin_deg": float(
                    part["prediction_min_joint_margin_deg"].min()
                ),
                "fit_s_median": float(part["fit_s"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "all_seed_gate_pass",
            "ee_p95_median_mm",
            "beta_p95_median_deg",
            "axis_p95_median_mm",
            "prediction_min_margin_deg",
            "fit_s_median",
        ],
        ascending=[False, True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _screen_task_fingerprint(args: argparse.Namespace, split: Mapping[str, Any]) -> str:
    settings = preset_settings(str(args.preset))
    configs = model_configs()
    limit = int(args.screen_config_limit) or int(settings["config_limit"])
    if limit > 0:
        configs = configs[:limit]
    return stable_fingerprint(
        {
            "phase": "v7_model_screen",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "split": split.get("task_fingerprint", ""),
            "preset": str(args.preset),
            "seed": int(_resolved_seeds(args)[0]),
            "configs": [config.as_dict() for config in configs],
            "settings": settings,
            "assignment": split.get("assignment_sha256", ""),
        }
    )


def phase_screen(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_model_screen"
    out.mkdir(parents=True, exist_ok=True)
    split = ensure_split_report(args)
    audit = read_json(Path(args.out_dir) / "00_audit" / "audit_report.json")
    holdout = audit["holdout"]
    settings = preset_settings(str(args.preset))
    configs = model_configs()
    limit = int(args.screen_config_limit) or int(settings["config_limit"])
    if limit > 0:
        configs = configs[:limit]
    seed = int(_resolved_seeds(args)[0])
    shared = resolve_training_inputs(args)
    validation_half = next(
        spec
        for spec in final_evaluation_specs(preset=str(args.preset), holdout=holdout)
        if spec["label"] == "validation_half_phase"
    )
    tasks = [
        _training_task(
            args,
            shared=shared,
            task_id=f"screen_{config.config_id}_s{seed}",
            mode="v7_screen",
            config=config,
            seed=seed,
            result_path=out / "worker_results" / f"screen_{config.config_id}_s{seed}.json",
            package_path=None,
            prediction_dir=None,
            angle_stride=int(settings["angle_stride"]),
            max_iter=int(settings["max_iter"]),
            evaluation_specs=[validation_half],
        )
        for config in configs
    ]
    results = _execute_training_tasks(
        tasks,
        task_dir=out / "worker_tasks",
        workers=int(args.workers),
        skip_existing=bool(args.skip_existing),
    )
    metrics = _flatten_results(results, label="validation_half_phase")
    ranking = rank_model_screen(metrics)
    metrics.to_csv(out / "screen_metrics.csv", index=False)
    ranking.to_csv(out / "screen_config_ranking.csv", index=False)
    selected_id = str(ranking.iloc[0]["config_id"])
    selected = next(result for result in results if result["config_id"] == selected_id)
    formal_screen = bool(
        str(args.preset) == "formal"
        and len(configs) == 48
        and len(results) == 48
        and int(args.screen_config_limit) == 0
        and seed == FORMAL_SEEDS[0]
    )
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": _screen_task_fingerprint(args, split),
        "source_split_task_fingerprint": split["task_fingerprint"],
        "preset": str(args.preset),
        "screen_seed": seed,
        "screen_config_count": int(len(configs)),
        "screen_run_count": int(len(results)),
        "screen_config_ids": [config.config_id for config in configs],
        "selected_config_id": selected_id,
        "selected_config": selected["config"],
        "selected_validation_gate_pass": bool(ranking.iloc[0]["all_seed_gate_pass"]),
        "formal_screen_protocol_gate_pass": formal_screen,
        "formal_claims_allowed": bool(
            split.get("formal_claims_allowed", False) and formal_screen
        ),
        "ranking_path": str((out / "screen_config_ranking.csv").resolve()),
    }
    write_json(out / "selection_report.json", report)
    if str(args.preset) == "formal" and not formal_screen:
        raise RuntimeError(f"V7 formal model screen protocol failed: {report}")
    return report


def ensure_screen_report(args: argparse.Namespace) -> dict[str, Any]:
    split = ensure_split_report(args)
    expected = _screen_task_fingerprint(args, split)
    path = Path(args.out_dir) / "02_model_screen" / "selection_report.json"
    ranking = Path(args.out_dir) / "02_model_screen" / "screen_config_ranking.csv"
    if path.exists() and ranking.exists():
        cached = read_json(path)
        if str(cached.get("task_fingerprint", "")) == expected:
            return cached
    return phase_screen(args)


def aggregate_seed_gate(rows: pd.DataFrame, *, formal: bool) -> dict[str, Any]:
    total = int(len(rows))
    unique = int(rows["seed"].nunique()) if total and "seed" in rows else 0
    passed = int(rows["model_gate_pass"].fillna(False).astype(bool).sum()) if total else 0
    if formal:
        stable = bool(total == 5 and unique == 5 and passed >= 4)
        required = 4
    else:
        stable = bool(total > 0 and unique == total and passed == total)
        required = total
    return {
        "total_seed_count": total,
        "unique_seed_count": unique,
        "passed_seed_count": passed,
        "required_seed_count": int(required),
        "stable_gate_pass": stable,
    }


def _final_task_fingerprint(args: argparse.Namespace, selection: Mapping[str, Any]) -> str:
    audit = read_json(Path(args.out_dir) / "00_audit" / "audit_report.json")
    return stable_fingerprint(
        {
            "phase": "v7_final_training",
            "strategy_version": TRAINING_STRATEGY_VERSION,
            "screen": selection.get("task_fingerprint", ""),
            "selected_config": selection.get("selected_config", {}),
            "seeds": _resolved_seeds(args),
            "evaluation_specs": final_evaluation_specs(
                preset=str(args.preset), holdout=audit["holdout"]
            ),
            "max_iter": preset_settings(str(args.preset))["max_iter"],
            "dataset": str(audit.get("dataset_sha256", "")),
            "assignment": file_sha256(
                Path(args.out_dir) / "01_split" / "split_assignment.parquet"
            ),
        }
    )


def phase_train(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_final_models"
    out.mkdir(parents=True, exist_ok=True)
    selection = ensure_screen_report(args)
    split = read_json(Path(args.out_dir) / "01_split" / "split_report.json")
    audit = read_json(Path(args.out_dir) / "00_audit" / "audit_report.json")
    if str(args.preset) == "formal" and not bool(selection.get("formal_claims_allowed", False)):
        raise RuntimeError("V7 formal training is blocked by the audit/split/screen chain")
    config = _config_from_dict(selection["selected_config"])
    seeds = _resolved_seeds(args)
    specs = final_evaluation_specs(preset=str(args.preset), holdout=audit["holdout"])
    shared = resolve_training_inputs(args)
    tasks = [
        _training_task(
            args,
            shared=shared,
            task_id=f"final_{config.config_id}_s{seed}",
            mode="v7_final",
            config=config,
            seed=int(seed),
            result_path=out / "worker_results" / f"final_{config.config_id}_s{seed}.json",
            package_path=out / "model_checkpoints" / f"seed_{seed}" / "model.joblib",
            prediction_dir=out / "holdout_predictions" / f"seed_{seed}",
            angle_stride=1,
            max_iter=int(preset_settings(str(args.preset))["max_iter"]),
            evaluation_specs=specs,
        )
        for seed in seeds
    ]
    results = _execute_training_tasks(
        tasks,
        task_dir=out / "worker_tasks",
        workers=int(args.workers),
        skip_existing=bool(args.skip_existing),
    )
    metrics: dict[str, pd.DataFrame] = {}
    gates: dict[str, dict[str, Any]] = {}
    formal = str(args.preset) == "formal"
    for spec in specs:
        label = str(spec["label"])
        table = _flatten_results(results, label=label)
        metrics[label] = table
        table.to_csv(out / f"{label}_metrics_all_seeds.csv", index=False)
        gates[label] = aggregate_seed_gate(table, formal=formal)
    target_clip_count = int(sum(int(result["target_link_clip_count"]) for result in results))
    seed_protocol = bool(seeds == list(FORMAL_SEEDS) and len(results) == 5)
    formal_claims = bool(
        split.get("formal_claims_allowed", False)
        and selection.get("formal_claims_allowed", False)
        and formal
    )
    report = {
        "strategy_version": TRAINING_STRATEGY_VERSION,
        "task_fingerprint": _final_task_fingerprint(args, selection),
        "source_screen_task_fingerprint": selection["task_fingerprint"],
        "preset": str(args.preset),
        "holdout": audit["holdout"],
        "selected_config_id": config.config_id,
        "selected_config": config.as_dict(),
        "seed_count": int(len(seeds)),
        "seeds": seeds,
        "formal_seed_protocol_pass": seed_protocol,
        "target_link_clip_count": target_clip_count,
        "evaluation_gates": gates,
        "test_evaluated": formal,
        "tube_diagnostic_is_formal_gate": False,
        "formal_claims_allowed": formal_claims,
        "formal_model_gate_pass": formal_model_gate_pass(
            formal_claims_allowed=formal_claims,
            seed_protocol_pass=seed_protocol,
            target_link_clip_count=target_clip_count,
            gates=gates,
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
            str(cached.get("task_fingerprint", "")) == expected
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
    holdout = audit["holdout"]
    lines = [
        "# True Ellipse V7 standard-domain model summary",
        "",
        f"- Upstream formal dataset gate: `{audit['formal_claims_allowed']}`.",
        f"- Training rows (non-centerline only): `{split['training_rows']}`.",
        f"- Dynamic validation radius: `{holdout['validation_radius_mm']:g} mm`.",
        f"- Dynamic test radius: `{holdout['test_radius_mm']:g} mm`.",
        f"- Selected configuration: `{selection['selected_config_id']}`.",
        f"- Test evaluated: `{training['test_evaluated']}`.",
        f"- Training-target link clip count: `{training['target_link_clip_count']}`.",
        f"- Formal model claim: `{training['formal_model_gate_pass']}`.",
        "",
        "The formal gate covers integer-phase and half-phase centerlines at both registered",
        "holdouts. Full tube predictions are reported as diagnostics and do not alter the claim.",
        "Only Cartesian target coordinates (x, y, z) are supplied to the regressor.",
    ]
    for label in training["evaluation_gates"]:
        metrics_path = Path(args.out_dir) / "03_final_models" / f"{label}_metrics_all_seeds.csv"
        if metrics_path.is_file():
            lines.extend(["", f"## {label}", "", dataframe_to_markdown(pd.read_csv(metrics_path))])
    summary_path = out / "experiment_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "audit_gate_pass": bool(audit["audit_gate_pass"]),
        "split_gate_pass": bool(split["split_gate_pass"]),
        "selected_config_id": selection["selected_config_id"],
        "holdout": holdout,
        "evaluation_gates": training["evaluation_gates"],
        "formal_model_gate_pass": bool(training["formal_model_gate_pass"]),
        "static_inverse_claim_radius_mm": float(holdout["test_radius_mm"])
        if training["formal_model_gate_pass"]
        else None,
        "summary_path": str(summary_path.resolve()),
    }
    write_json(out / "final_report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the standard-domain V7 beta6 inverse with identity and bounded links."
    )
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--tube-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phases", default="all")
    parser.add_argument("--preset", choices=("smoke", "pilot", "formal"), default="formal")
    parser.add_argument("--validation-radius-mm", type=float, default=None)
    parser.add_argument("--test-radius-mm", type=float, default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FORMAL_SEEDS))
    parser.add_argument("--screen-config-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
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
        "summary": phase_summary,
    }
    results = {phase: handlers[phase](args) for phase in phases}
    report = {
        "mode": "true_ellipse_standard_domain_training_v7",
        "preset": str(args.preset),
        "phases": phases,
        "out_dir": str(Path(args.out_dir).resolve()),
        "results": results,
    }
    write_json(Path(args.out_dir) / "run_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    if args.worker_task is not None:
        result = run_training_worker(read_json(Path(args.worker_task)))
        print(json.dumps(result, ensure_ascii=False, default=_json_default))
        return 0
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
