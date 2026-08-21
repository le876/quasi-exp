#!/usr/bin/env python3
"""Train non-formal V7 centerline diagnostics and evaluate their trajectories.

This experiment is deliberately separated from the formal V7 pipeline.  It trains
on the accepted continuous centerlines at radii up to 100 mm, selects a model on
the physically isolated 102.5 mm centerline, and only then opens the registered
101/102/103.5/104 mm post-selection curves.  Nothing produced here is allowed to
change the formal V7 model gate or claim radius.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import dump, load


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_standard_domain_training_v7 as formal_training  # noqa: E402
import run_true_ellipse_standard_domain_v7 as upstream_v7  # noqa: E402
import true_ellipse_radial_bundle_engine as engine  # noqa: E402
from true_ellipse_family_v5_utils import (  # noqa: E402
    dataframe_to_markdown,
    stable_fingerprint,
)


MODEL_INPUT_COLUMNS = list(formal_training.MODEL_INPUT_COLUMNS)
BETA_COLUMNS = list(formal_training.BETA_COLUMNS)
FORMAL_SEEDS = tuple(formal_training.FORMAL_SEEDS)
DIAGNOSTIC_KIND = "beta6_pose_standard_domain_v7_centerline_diagnostic"
JOINT_DOMAIN_ID = "standard_beta34_10deg_v1"
TRAIN_MAX_RADIUS_MM = 100.0
VALIDATION_RADIUS_MM = 102.5
POST_SELECTION_RADII_MM = (101.0, 102.0, 103.5, 104.0)
UNSUPPORTED_RADII_MM = (105.0, 107.5, 110.0, 112.5, 115.0, 117.5, 120.0)
DISPLAY_RADII_MM = (100.0, 102.5, 104.0)
BEST_UNSUPPORTED_DISPLAY_RADII_MM = (105.0, 110.0, 115.0, 120.0)
SCREEN_SEED = FORMAL_SEEDS[0]
ALL_PHASES = ("audit", "dataset", "screen", "stability", "evaluate", "visualize", "summary")
STRATEGY_VERSION = 1

DEFAULT_UPSTREAM_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_v7"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "true_ellipse_standard_domain_training_v7_diagnostic_centerline"
)
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"

read_json = upstream_v7.read_json
write_json = upstream_v7.write_json
_json_default = upstream_v7._json_default


def file_sha256(path: str | Path) -> str:
    """Hash the current bytes without trusting size/mtime metadata caches."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrajectoryEvidenceClass(StrEnum):
    ACCEPTED_CONTINUOUS_TRUTH = "accepted_continuous_truth"
    UNSUPPORTED_MODEL_EXTRAPOLATION = "unsupported_model_extrapolation"


def diagnostic_claims() -> dict[str, Any]:
    """Return the immutable fail-closed claim boundary for every artifact."""

    return {
        "diagnostic_only": True,
        "formal_claims_allowed": False,
        "changes_v7_formal_gate": False,
        "static_inverse_claim_radius_mm": None,
    }


def add_diagnostic_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the immutable claim boundary to every materialized table row."""

    for field, value in diagnostic_claims().items():
        frame[field] = value
    return frame


def diagnostic_model_configs() -> list[formal_training.LinkedModelConfig]:
    """The complete V5 24-config grid crossed with identity/bounded links."""

    return formal_training.model_configs()


def parse_int_csv(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def parse_float_csv(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(item) for item in value]


def parse_phases(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        phases = [part.strip() for part in value.split(",") if part.strip()]
    else:
        phases = [str(part).strip() for part in value if str(part).strip()]
    if phases == ["all"]:
        return list(ALL_PHASES)
    unknown = sorted(set(phases) - set(ALL_PHASES))
    if unknown:
        raise ValueError(f"unsupported V7-D1 phases: {unknown}")
    return phases


def cache_lineage_matches(
    report: Mapping[str, Any], expected: Mapping[str, str]
) -> bool:
    """Require every current upstream fingerprint to be present and identical."""

    return bool(expected) and all(
        bool(str(fingerprint))
        and str(report.get(field, "")) == str(fingerprint)
        for field, fingerprint in expected.items()
    )


def _hashed_artifact_matches(record: Mapping[str, Any]) -> bool:
    path_value = record.get("path")
    expected_sha256 = str(record.get("sha256", ""))
    if not path_value or not expected_sha256:
        return False
    path = Path(path_value)
    return path.is_file() and file_sha256(path) == expected_sha256


def _artifact_record(path: str | Path) -> dict[str, str]:
    artifact = Path(path).resolve()
    return {"path": str(artifact), "sha256": file_sha256(artifact)}


def training_result_artifact_record(
    *,
    result_path: str | Path,
    package_path: str | Path,
    package_sha256: str,
    task_fingerprint: str,
) -> dict[str, str]:
    """Content-address a worker result together with its fitted model package."""

    result = Path(result_path).resolve()
    package = Path(package_path).resolve()
    expected_package_sha256 = str(package_sha256)
    if not result.is_file():
        raise FileNotFoundError(f"missing V7-D1 worker result: {result}")
    if not package.is_file() or file_sha256(package) != expected_package_sha256:
        raise ValueError(f"V7-D1 model package hash changed: {package}")
    return {
        "result_path": str(result),
        "result_sha256": file_sha256(result),
        "package_path": str(package),
        "package_sha256": expected_package_sha256,
        "task_fingerprint": str(task_fingerprint),
    }


def training_result_artifact_matches(record: Mapping[str, Any]) -> bool:
    """Fail closed unless both the worker JSON and model bytes still match."""

    return _hashed_artifact_matches(
        {"path": record.get("result_path"), "sha256": record.get("result_sha256")}
    ) and _hashed_artifact_matches(
        {
            "path": record.get("package_path"),
            "sha256": record.get("package_sha256"),
        }
    )


def phase_artifact_fingerprint(
    *, model_records: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> str:
    """Bind a phase lineage to worker, model, and tabular artifact bytes."""

    return stable_fingerprint(
        {
            "strategy_version": STRATEGY_VERSION,
            "model_records": model_records,
            "artifacts": artifacts,
        }
    )


def summary_result_artifact_fingerprints(
    selection: Mapping[str, Any], stability: Mapping[str, Any]
) -> dict[str, str]:
    """Expose both fitted-result byte lineages in the final report."""

    fingerprints = {
        "screen": str(selection.get("result_artifact_fingerprint", "")),
        "stability": str(stability.get("result_artifact_fingerprint", "")),
    }
    if not all(fingerprints.values()):
        raise ValueError("missing V7-D1 result artifact fingerprint")
    return fingerprints


def validate_protocol_args(args: argparse.Namespace) -> None:
    """Reject CLI overrides that would make a run falsely look like full."""

    if str(args.preset) == "full" and int(args.screen_config_limit) != 0:
        raise ValueError("V7-D1 full protocol must screen all 48 configurations")


def preset_settings(preset: str) -> dict[str, int]:
    if str(preset) == "smoke":
        return {"config_limit": 2, "max_iter": 30, "angle_stride": 30, "seed_count": 1}
    if str(preset) == "pilot":
        return {"config_limit": 8, "max_iter": 300, "angle_stride": 5, "seed_count": 3}
    if str(preset) == "full":
        return {"config_limit": 0, "max_iter": 800, "angle_stride": 1, "seed_count": 5}
    raise ValueError(f"unsupported V7-D1 preset: {preset}")


def diagnostic_package_metadata(
    *,
    source_manifest_sha256: str,
    screen_dataset_sha256: str,
    config_id: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "kind": DIAGNOSTIC_KIND,
        **diagnostic_claims(),
        "source_manifest_sha256": str(source_manifest_sha256),
        "screen_dataset_sha256": str(screen_dataset_sha256),
        "config_id": str(config_id),
        "seed": int(seed),
        "split_policy": {
            "training_radius_max_mm": TRAIN_MAX_RADIUS_MM,
            "validation_radius_mm": VALIDATION_RADIUS_MM,
            "post_selection_radii_mm": list(POST_SELECTION_RADII_MM),
            "post_selection_physically_absent_from_screen": True,
        },
        "training_strategy_version": STRATEGY_VERSION,
    }
    payload["provenance_fingerprint"] = stable_fingerprint(payload)
    return payload


def _radius_key(radius_mm: float) -> str:
    return f"{float(radius_mm):g}"


def _source_entry(manifest: Mapping[str, Any], radius_mm: float) -> Mapping[str, Any]:
    key = _radius_key(radius_mm)
    if key not in manifest:
        raise ValueError(f"strict centerline manifest is missing radius {key} mm")
    entry = manifest[key]
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"strict centerline artifact is missing: {path}")
    actual = file_sha256(path)
    if actual != str(entry.get("sha256", "")):
        raise ValueError(f"strict centerline hash changed at {key} mm")
    return entry


def _is_post_selection_radius(radius_mm: float) -> bool:
    return any(
        np.isclose(float(radius_mm), registered, atol=1.0e-8, rtol=0.0)
        for registered in POST_SELECTION_RADII_MM
    )


def audit_centerline_entry(
    entry: Mapping[str, Any], *, radius_mm: float
) -> dict[str, Any]:
    """Audit screen inputs now and defer outer-curve content reads until selection."""

    path = Path(entry["path"])
    exists = path.is_file()
    expected_sha256 = str(entry.get("sha256", ""))
    deferred = _is_post_selection_radius(radius_mm)
    actual_sha256 = None
    hash_matches: bool | None = None
    if exists and expected_sha256 and not deferred:
        actual_sha256 = file_sha256(path)
        hash_matches = actual_sha256 == expected_sha256
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "exists": exists,
        "hash_matches": hash_matches,
        "actual_sha256": actual_sha256,
        "hash_verification_phase": "post_selection" if deferred else "audit",
        "branch_hash": str(entry.get("branch_hash", "")),
    }


def _validate_centerline_frame(
    frame: pd.DataFrame,
    *,
    radius_mm: float,
    expected_points: int,
) -> pd.DataFrame:
    required = {
        "family_id",
        "angle_idx",
        "angle_rad",
        "radius_mm",
        *MODEL_INPUT_COLUMNS,
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"centerline {radius_mm:g} mm is missing columns: {missing}")
    ordered = frame.sort_values("angle_idx", kind="stable").reset_index(drop=True).copy()
    if len(ordered) != int(expected_points):
        raise ValueError(
            f"centerline {radius_mm:g} mm has {len(ordered)} rows, expected {expected_points}"
        )
    if ordered["angle_idx"].astype(int).tolist() != list(range(int(expected_points))):
        raise ValueError(f"centerline {radius_mm:g} mm does not contain a complete phase grid")
    if not np.isclose(
        ordered["radius_mm"].to_numpy(dtype=float), float(radius_mm), atol=1.0e-8
    ).all():
        raise ValueError(f"centerline {radius_mm:g} mm has inconsistent radius metadata")
    numeric = ordered[[*MODEL_INPUT_COLUMNS, *BETA_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"centerline {radius_mm:g} mm contains non-finite model values")
    family_ids = ordered["family_id"].astype(str).unique().tolist()
    if len(family_ids) != 1:
        raise ValueError(f"centerline {radius_mm:g} mm contains multiple families")
    return ordered


def materialize_screen_dataset(
    strict_manifest: Mapping[str, Any],
    *,
    output_path: str | Path,
    expected_family_id: str | None = None,
    train_max_radius_mm: float = TRAIN_MAX_RADIUS_MM,
    validation_radius_mm: float = VALIDATION_RADIUS_MM,
    expected_points_per_radius: int = 360,
) -> dict[str, Any]:
    """Write only train + validation curves; never open post-selection curves."""

    registered = sorted(float(key) for key in strict_manifest)
    training_radii = [radius for radius in registered if radius <= train_max_radius_mm + 1.0e-9]
    validation_radii = [
        radius
        for radius in registered
        if np.isclose(radius, validation_radius_mm, atol=1.0e-8)
    ]
    post_selection = [
        radius
        for radius in POST_SELECTION_RADII_MM
        if _radius_key(radius) in strict_manifest
    ]
    if not training_radii or validation_radii != [float(validation_radius_mm)]:
        raise ValueError("screen dataset requires non-empty training and exactly one validation radius")
    frames: list[pd.DataFrame] = []
    source_records: dict[str, dict[str, Any]] = {}
    for radius in [*training_radii, *validation_radii]:
        entry = _source_entry(strict_manifest, radius)
        source_path = Path(entry["path"])
        frame = _validate_centerline_frame(
            pd.read_parquet(source_path),
            radius_mm=radius,
            expected_points=expected_points_per_radius,
        )
        family_id = str(frame["family_id"].iloc[0])
        trajectory_id = f"{family_id}@r{radius:07.2f}mm"
        frame["trajectory_id"] = trajectory_id
        frame["sample_id"] = [
            f"{trajectory_id}:phase-{index:04d}"
            for index in frame["angle_idx"].to_numpy(dtype=int)
        ]
        is_train = radius <= train_max_radius_mm + 1.0e-9
        frame["split"] = "train" if is_train else "validation"
        frame["used_for_training"] = bool(is_train)
        frame["is_centerline"] = True
        frame["source_centerline_path"] = str(source_path.resolve())
        frame["source_centerline_sha256"] = str(entry["sha256"])
        add_diagnostic_claim_columns(frame)
        frames.append(frame)
        source_records[_radius_key(radius)] = {
            "path": str(source_path.resolve()),
            "sha256": str(entry["sha256"]),
            "rows": int(len(frame)),
        }
    dataset = pd.concat(frames, ignore_index=True)
    family_ids = sorted(dataset["family_id"].astype(str).unique().tolist())
    if not dataset["sample_id"].is_unique:
        raise ValueError("screen centerline sample IDs are not unique")
    if set(np.round(dataset.loc[dataset["used_for_training"], "radius_mm"], 8)) & set(
        np.round(POST_SELECTION_RADII_MM, 8)
    ):
        raise ValueError("post-selection radius leaked into diagnostic model training")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False, compression="zstd")
    report = {
        **diagnostic_claims(),
        "dataset_path": str(output.resolve()),
        "dataset_sha256": file_sha256(output),
        "rows": int(len(dataset)),
        "training_rows": int(dataset["used_for_training"].sum()),
        "validation_rows": int(dataset["split"].eq("validation").sum()),
        "training_radii_mm": training_radii,
        "validation_radii_mm": validation_radii,
        "post_selection_registered_radii_mm": post_selection,
        "post_selection_physically_absent": not bool(
            dataset["radius_mm"].isin(post_selection).any()
        ),
        "family_ids": family_ids,
        "source_centerlines": source_records,
    }
    report["dataset_gate_pass"] = bool(
        report["training_rows"] > 0
        and report["validation_rows"] == int(expected_points_per_radius)
        and report["post_selection_physically_absent"]
        and len(family_ids) == 1
        and (
            expected_family_id is None
            or family_ids == [str(expected_family_id)]
        )
        and dataset["is_centerline"].astype(bool).all()
        and dataset.loc[dataset["split"].eq("train"), "used_for_training"].all()
        and not dataset.loc[dataset["split"].eq("validation"), "used_for_training"].any()
    )
    return report


def dense_centerline_frame(
    source: pd.DataFrame,
    *,
    n_points: int = 720,
    include_beta_truth: bool = True,
) -> pd.DataFrame:
    """Evaluate the analytic ellipse and periodic beta interpolation on a dense grid."""

    required = {
        "angle_idx",
        "angle_rad",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "radius_mm",
        "phase_y_rad",
        "phase_z_rad",
    }
    if include_beta_truth:
        required.update(BETA_COLUMNS)
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"dense centerline source is missing columns: {missing}")
    ordered = source.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    if len(ordered) < 3 or int(n_points) < len(ordered):
        raise ValueError("dense centerline requires at least the source phase resolution")

    def unique_float(column: str) -> float:
        values = ordered[column].to_numpy(dtype=float)
        if not np.allclose(values, values[0], atol=1.0e-12, rtol=0.0):
            raise ValueError(f"dense centerline source has non-constant {column}")
        return float(values[0])

    n = int(n_points)
    angle = np.arange(n, dtype=float) * (2.0 * math.pi / n)
    center_x = unique_float("center_x_m")
    center_y = unique_float("center_y_m")
    center_z = unique_float("center_z_m")
    radius_mm = unique_float("radius_mm")
    phase_y = unique_float("phase_y_rad")
    phase_z = unique_float("phase_z_rad")
    radius_m = radius_mm / 1000.0
    dense = pd.DataFrame(
        {
            "dense_angle_idx": np.arange(n, dtype=int),
            "angle_idx": np.arange(n, dtype=int),
            "angle_rad": angle,
            "radius_mm": radius_mm,
            "center_x_m": center_x,
            "center_y_m": center_y,
            "center_z_m": center_z,
            "amp_xy_mm": radius_mm,
            "amp_z_mm": 1.5 * radius_mm,
            "phase_y_rad": phase_y,
            "phase_z_rad": phase_z,
            "x_target_m": center_x + radius_m * np.sin(angle),
            "y_target_m": center_y + radius_m * np.sin(angle + phase_y),
            "z_target_m": center_z + 1.5 * radius_m * np.sin(angle + phase_z),
        }
    )
    for column in ("family_id", "candidate_id", "ellipse_id"):
        if column in ordered:
            dense[column] = str(ordered[column].iloc[0])
    phase_position = np.arange(n, dtype=np.int64) * len(ordered)
    integer_mask = phase_position % n == 0
    dense["phase_kind"] = np.where(integer_mask, "integer", "half_phase")
    if include_beta_truth:
        source_angle = np.mod(ordered["angle_rad"].to_numpy(dtype=float), 2.0 * math.pi)
        order = np.argsort(source_angle, kind="stable")
        for column in BETA_COLUMNS:
            dense[column] = np.interp(
                angle,
                source_angle[order],
                ordered[column].to_numpy(dtype=float)[order],
                period=2.0 * math.pi,
            )
    return dense


def target_only_centerline_frame(
    template: pd.DataFrame,
    *,
    radius_mm: float,
    n_points: int = 720,
) -> pd.DataFrame:
    """Build an analytic target without manufacturing IK/beta truth."""

    source = template.copy()
    source["radius_mm"] = float(radius_mm)
    if "amp_xy_mm" in source:
        source["amp_xy_mm"] = float(radius_mm)
    if "amp_z_mm" in source:
        source["amp_z_mm"] = 1.5 * float(radius_mm)
    target = dense_centerline_frame(
        source,
        n_points=int(n_points),
        include_beta_truth=False,
    )
    return target.drop(columns=[column for column in BETA_COLUMNS if column in target])


def evaluate_package_trajectory(
    *,
    package: Mapping[str, Any],
    frame: pd.DataFrame,
    robot_inputs: Any,
    theta_sign: float,
    evidence_class: str | TrajectoryEvidenceClass,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one dense path while preserving whether beta truth exists."""

    try:
        evidence = TrajectoryEvidenceClass(evidence_class)
    except ValueError as exc:
        raise ValueError(
            f"unsupported trajectory evidence class: {evidence_class}"
        ) from exc
    beta_truth_available = all(column in frame for column in BETA_COLUMNS)
    if (
        evidence is TrajectoryEvidenceClass.ACCEPTED_CONTINUOUS_TRUTH
        and not beta_truth_available
    ):
        raise ValueError("accepted continuous evaluation requires beta truth")
    if (
        evidence is TrajectoryEvidenceClass.UNSUPPORTED_MODEL_EXTRAPOLATION
        and beta_truth_available
    ):
        raise ValueError("unsupported extrapolation must not carry beta truth")
    xyz = frame[MODEL_INPUT_COLUMNS].to_numpy(dtype=float)
    beta_pred = _predict_beta(package, xyz)
    beta_true = frame[BETA_COLUMNS].to_numpy(dtype=float) if beta_truth_available else None
    domain = engine.registered_joint_domain(
        str(package.get("joint_domain_id", JOINT_DOMAIN_ID))
    )
    metrics, achieved, _theta = formal_training.v5.v4.evaluate_beta_prediction(
        beta_pred=beta_pred,
        beta_true=beta_true,
        target_xyz=xyz,
        lengths_m=robot_inputs.lengths_m,
        p_end_local_m=robot_inputs.p_end_local_m,
        theta_sign=float(theta_sign),
        periodic=True,
        joint_domain=domain,
    )
    if beta_truth_available:
        tracking_gate = formal_training.deployed_centerline_model_gate(metrics)
    else:
        tracking_gate = bool(
            formal_training.v5.v4.centerline_model_gate(
                metrics, require_beta_error=False
            )
            and int(metrics.get("beta_bound_violation_count", 1)) == 0
            and float(metrics.get("prediction_min_joint_margin_deg", -np.inf)) >= 0.01
        )
    metrics.update(
        {
            "radius_mm": float(frame["radius_mm"].iloc[0]),
            "evidence_class": evidence.value,
            "beta_truth_available": bool(beta_truth_available),
            "diagnostic_tracking_gate_pass": tracking_gate,
            **diagnostic_claims(),
        }
    )
    keep = [
        column
        for column in (
            "dense_angle_idx",
            "angle_idx",
            "angle_rad",
            "phase_kind",
            "radius_mm",
            *MODEL_INPUT_COLUMNS,
        )
        if column in frame
    ]
    prediction = frame[keep].copy()
    prediction["evidence_class"] = evidence.value
    prediction["beta_truth_available"] = bool(beta_truth_available)
    add_diagnostic_claim_columns(prediction)
    if beta_truth_available and beta_true is not None:
        for index, column in enumerate(BETA_COLUMNS):
            prediction[f"true_{column}"] = beta_true[:, index]
    for index, column in enumerate(BETA_COLUMNS):
        prediction[f"pred_{column}"] = beta_pred[:, index]
    for index, axis in enumerate("xyz"):
        prediction[f"achieved_{axis}_m"] = achieved[:, index]
        prediction[f"{axis}_m"] = achieved[:, index]
        prediction[f"{axis}_error_mm"] = (achieved[:, index] - xyz[:, index]) * 1000.0
    prediction["ee_err_mm"] = np.linalg.norm(achieved - xyz, axis=1) * 1000.0
    margins = engine.joint_margin_matrix_deg(beta_pred, domain=domain)
    prediction["prediction_min_joint_margin_deg"] = np.min(margins, axis=1)
    prediction["prediction_any_joint_out_of_bounds"] = np.min(margins, axis=1) < -1.0e-9
    return metrics, prediction


def rank_diagnostic_screen(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "config_id",
        "evaluation_label",
        "model_gate_pass",
        "ee_p95_mm",
        "beta_p95_deg",
        "axiserr_max_p95_abs_mm",
        "prediction_min_joint_margin_deg",
        "fit_s",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"diagnostic screen metrics missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for config_id, part in metrics.groupby("config_id", sort=False):
        labels = set(part["evaluation_label"].astype(str))
        expected = {"validation_integer", "validation_half_phase"}
        complete = labels == expected and len(part) == 2
        first = part.iloc[0]
        both_pass = bool(complete and part["model_gate_pass"].fillna(False).astype(bool).all())
        row = {
            **diagnostic_claims(),
            "config_id": str(config_id),
            "architecture": first.get("architecture", ""),
            "hidden_layers": first.get("hidden_layers", ""),
            "feature_set": first.get("feature_set", ""),
            "activation": first.get("activation", ""),
            "alpha": float(first.get("alpha", np.nan)),
            "output_link_id": first.get("output_link_id", ""),
            "validation_evaluation_count": int(len(part)),
            "both_validation_gates_pass": both_pass,
            "selection_eligible": both_pass,
            "worst_ee_p95_mm": float(part["ee_p95_mm"].max()),
            "worst_beta_p95_deg": float(part["beta_p95_deg"].max()),
            "worst_axis_p95_mm": float(part["axiserr_max_p95_abs_mm"].max()),
            "minimum_joint_margin_deg": float(part["prediction_min_joint_margin_deg"].min()),
            "fit_s": float(part["fit_s"].max()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "selection_eligible",
            "worst_ee_p95_mm",
            "worst_beta_p95_deg",
            "worst_axis_p95_mm",
            "minimum_joint_margin_deg",
            "fit_s",
            "config_id",
        ],
        ascending=[False, True, True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_ranked_config(ranking: pd.DataFrame) -> dict[str, Any]:
    if ranking.empty:
        raise ValueError("cannot select a diagnostic model from an empty ranking")
    first = ranking.iloc[0]
    passed = bool(first.get("both_validation_gates_pass", False))
    return {
        "selected_config_id": str(first["config_id"]),
        "selected_validation_gate_pass": passed,
        "selection_note": "validation_gate_best" if passed else "fallback_best_no_gate",
    }


def training_task_fingerprint(task: Mapping[str, Any]) -> str:
    ignored = {"task_fingerprint", "result_path", "package_path", "prediction_dir"}
    semantic = {key: value for key, value in task.items() if key not in ignored}
    return stable_fingerprint(
        {"strategy_version": STRATEGY_VERSION, "diagnostic_training_task": semantic}
    )


def make_training_task(
    *,
    dataset_path: str | Path,
    source_manifest_sha256: str,
    robot_config_path: str | Path,
    config: formal_training.LinkedModelConfig,
    seed: int,
    max_iter: int,
    angle_stride: int,
    mode: str,
    result_path: str | Path,
    package_path: str | Path,
    prediction_dir: str | Path | None = None,
    dense_points: int = 720,
) -> dict[str, Any]:
    dataset = Path(dataset_path)
    robot_config = Path(robot_config_path)
    task: dict[str, Any] = {
        "task_id": f"{mode}_{config.config_id}_s{int(seed)}",
        "mode": str(mode),
        "strategy_version": STRATEGY_VERSION,
        **diagnostic_claims(),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": file_sha256(dataset),
        "source_manifest_sha256": str(source_manifest_sha256),
        "robot_config": str(robot_config.resolve()),
        "robot_config_sha256": file_sha256(robot_config),
        "config": config.as_dict(),
        "seed": int(seed),
        "max_iter": int(max_iter),
        "batch_size": 256,
        "angle_stride": max(1, int(angle_stride)),
        "dense_points": int(dense_points),
        "joint_domain_id": JOINT_DOMAIN_ID,
        "joint_domain_fingerprint": engine.registered_joint_domain(
            JOINT_DOMAIN_ID
        ).fingerprint,
        "result_path": str(Path(result_path).resolve()),
        "package_path": str(Path(package_path).resolve()),
        "prediction_dir": (
            str(Path(prediction_dir).resolve()) if prediction_dir is not None else None
        ),
    }
    task["task_fingerprint"] = training_task_fingerprint(task)
    return task


def build_screen_tasks(
    *,
    configs: Sequence[formal_training.LinkedModelConfig],
    dataset_path: str | Path,
    source_manifest_sha256: str,
    robot_config_path: str | Path,
    out_dir: str | Path,
    seed: int,
    max_iter: int,
    angle_stride: int,
    dense_points: int,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    return [
        make_training_task(
            dataset_path=dataset_path,
            source_manifest_sha256=source_manifest_sha256,
            robot_config_path=robot_config_path,
            config=config,
            seed=int(seed),
            max_iter=int(max_iter),
            angle_stride=int(angle_stride),
            mode="v7_diagnostic_screen",
            result_path=out
            / "worker_results"
            / f"screen_{config.config_id}_s{int(seed)}.json",
            package_path=out
            / "model_packages"
            / config.config_id
            / f"seed_{int(seed)}.joblib",
            prediction_dir=None,
            dense_points=int(dense_points),
        )
        for config in configs
    ]


def _predict_beta(package: Mapping[str, Any], xyz: np.ndarray) -> np.ndarray:
    return formal_training.predict_beta(package, np.asarray(xyz, dtype=float))


def _evaluate_validation_frame(
    *,
    package: Mapping[str, Any],
    frame: pd.DataFrame,
    robot_inputs: Any,
    theta_sign: float,
    label: str,
    fit_s: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    xyz = frame[MODEL_INPUT_COLUMNS].to_numpy(dtype=float)
    beta_true = frame[BETA_COLUMNS].to_numpy(dtype=float)
    beta_pred = _predict_beta(package, xyz)
    domain = engine.registered_joint_domain(JOINT_DOMAIN_ID)
    metrics, achieved, _theta = formal_training.v5.v4.evaluate_beta_prediction(
        beta_pred=beta_pred,
        beta_true=beta_true,
        target_xyz=xyz,
        lengths_m=robot_inputs.lengths_m,
        p_end_local_m=robot_inputs.p_end_local_m,
        theta_sign=float(theta_sign),
        periodic=True,
        joint_domain=domain,
    )
    metrics.update(
        {
            "evaluation_label": str(label),
            "radius_mm": VALIDATION_RADIUS_MM,
            "eval_rows": int(len(frame)),
            "fit_s": float(fit_s),
            "model_gate_pass": formal_training.deployed_centerline_model_gate(metrics),
            **diagnostic_claims(),
        }
    )
    prediction = frame[
        [column for column in ("angle_idx", "angle_rad", "phase_kind", *MODEL_INPUT_COLUMNS) if column in frame]
    ].copy()
    for index, column in enumerate(BETA_COLUMNS):
        prediction[f"true_{column}"] = beta_true[:, index]
        prediction[f"pred_{column}"] = beta_pred[:, index]
    for index, axis in enumerate("xyz"):
        prediction[f"achieved_{axis}_m"] = achieved[:, index]
    prediction["ee_err_mm"] = np.linalg.norm(achieved - xyz, axis=1) * 1000.0
    return metrics, prediction


def run_training_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    expected = training_task_fingerprint(task)
    if str(task.get("task_fingerprint", "")) != expected:
        raise ValueError("V7-D1 training task fingerprint mismatch")
    for path_key, hash_key in (
        ("dataset", "dataset_sha256"),
        ("robot_config", "robot_config_sha256"),
    ):
        if file_sha256(task[path_key]) != str(task[hash_key]):
            raise ValueError(f"V7-D1 training input changed after registration: {path_key}")
    if not bool(task.get("diagnostic_only", False)) or bool(
        task.get("formal_claims_allowed", True)
    ):
        raise ValueError("V7-D1 worker received a claim-bearing task")
    dataset = pd.read_parquet(task["dataset"])
    required = {
        "sample_id",
        "trajectory_id",
        "angle_idx",
        "radius_mm",
        "split",
        "used_for_training",
        *MODEL_INPUT_COLUMNS,
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"V7-D1 screen dataset missing columns: {missing}")
    training_mask = dataset["used_for_training"].astype(bool).to_numpy()
    training_mask &= dataset["split"].astype(str).eq("train").to_numpy()
    training_mask &= dataset["radius_mm"].to_numpy(dtype=float) <= TRAIN_MAX_RADIUS_MM + 1.0e-9
    training_mask &= (
        dataset["angle_idx"].to_numpy(dtype=np.int64) % max(1, int(task["angle_stride"]))
        == 0
    )
    validation_mask = dataset["split"].astype(str).eq("validation").to_numpy()
    validation_mask &= np.isclose(
        dataset["radius_mm"].to_numpy(dtype=float), VALIDATION_RADIUS_MM, atol=1.0e-8
    )
    if np.any(training_mask & validation_mask):
        raise ValueError("V7-D1 validation radius leaked into training")
    if dataset.loc[training_mask, "radius_mm"].isin(POST_SELECTION_RADII_MM).any():
        raise ValueError("V7-D1 post-selection radius leaked into training")
    train_idx = np.flatnonzero(training_mask).astype(np.int64)
    validation = dataset.loc[validation_mask].sort_values("angle_idx", kind="stable").reset_index(
        drop=True
    )
    if not len(train_idx) or validation.empty:
        raise ValueError("V7-D1 worker requires non-empty train and validation partitions")
    config = formal_training._config_from_dict(task["config"])
    domain = engine.registered_joint_domain(str(task["joint_domain_id"]))
    if domain.fingerprint != str(task["joint_domain_fingerprint"]):
        raise ValueError("V7-D1 joint-domain fingerprint mismatch")
    beta_train = dataset.iloc[train_idx][BETA_COLUMNS].to_numpy(dtype=float)
    latent_train, link_report = formal_training.encode_training_targets(
        beta_train,
        output_link_id=config.output_link_id,
        domain=domain,
    )
    model, x_scaler, y_scaler, feature_names, fit_s = formal_training.v5.v4._fit_scaled_model(
        config=config.base_config,
        seed=int(task["seed"]),
        xyz_train=dataset.iloc[train_idx][MODEL_INPUT_COLUMNS].to_numpy(dtype=float),
        beta_train=latent_train,
        max_iter=int(task["max_iter"]),
        batch_size=int(task.get("batch_size", 256)),
    )
    robot_config = formal_training.v5.v4.load_config(str(task["robot_config"]))
    robot_inputs = formal_training.v5.v4.load_robot_inputs(robot_config)
    theta_sign = float(robot_config.get("kinematics", {}).get("theta_sign", -1.0))
    package: dict[str, Any] = {
        **diagnostic_package_metadata(
            source_manifest_sha256=str(task["source_manifest_sha256"]),
            screen_dataset_sha256=str(task["dataset_sha256"]),
            config_id=config.config_id,
            seed=int(task["seed"]),
        ),
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": list(MODEL_INPUT_COLUMNS),
        "feature_set": config.base_config.feature_set,
        "feature_names": feature_names,
        "target_cols": list(BETA_COLUMNS),
        "beta_cols": list(BETA_COLUMNS),
        "theta_cols": list(formal_training.v5.v4.THETA_COLS),
        "theta_sign": theta_sign,
        "joint_domain_id": domain.domain_id,
        "joint_domain_fingerprint": domain.fingerprint,
        "output_link_id": config.output_link_id,
        "output_link_fingerprint": engine.registered_output_link(
            config.output_link_id
        ).fingerprint,
        "target_link_clip_count": int(link_report["target_link_clip_count"]),
        "robot_config": str(task["robot_config"]),
        "robot_config_sha256": str(task["robot_config_sha256"]),
        "dataset": str(task["dataset"]),
        "dataset_sha256": str(task["dataset_sha256"]),
        "config": config.as_dict(),
        "model_name": config.base_config.architecture,
        "activation": config.base_config.activation,
        "alpha": config.base_config.alpha,
        "task_fingerprint": expected,
        "train_rows": int(len(train_idx)),
    }
    dense = dense_centerline_frame(
        validation,
        n_points=int(task.get("dense_points", 720)),
        include_beta_truth=True,
    )
    half = dense.loc[dense["phase_kind"].eq("half_phase")].reset_index(drop=True)
    evaluations: dict[str, Any] = {}
    prediction_dir = Path(task["prediction_dir"]) if task.get("prediction_dir") else None
    for label, frame in (
        ("validation_integer", validation),
        ("validation_half_phase", half),
    ):
        metrics, prediction = _evaluate_validation_frame(
            package=package,
            frame=frame,
            robot_inputs=robot_inputs,
            theta_sign=theta_sign,
            label=label,
            fit_s=fit_s,
        )
        metrics.update(
            {
                "n_iter": int(model.n_iter_),
                "loss": float(model.loss_),
                "train_rows": int(len(train_idx)),
                "output_link_id": config.output_link_id,
            }
        )
        evaluations[label] = metrics
        if prediction_dir is not None:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(
                prediction_dir / f"{label}.parquet", index=False, compression="zstd"
            )
    package_path = Path(task["package_path"])
    package_path.parent.mkdir(parents=True, exist_ok=True)
    dump(package, package_path)
    result = {
        "task_id": str(task["task_id"]),
        "mode": str(task["mode"]),
        **diagnostic_claims(),
        "seed": int(task["seed"]),
        "config_id": config.config_id,
        "config": config.as_dict(),
        "output_link_id": config.output_link_id,
        "target_link_clip_count": int(link_report["target_link_clip_count"]),
        "train_rows": int(len(train_idx)),
        "fit_s": float(fit_s),
        "evaluations": evaluations,
        "package_path": str(package_path.resolve()),
        "package_sha256": file_sha256(package_path),
        "task_fingerprint": expected,
    }
    write_json(Path(task["result_path"]), result)
    return result


def _expected_strict_radii() -> tuple[float, ...]:
    return (
        75.0,
        76.0,
        77.0,
        78.0,
        79.0,
        80.0,
        81.0,
        82.0,
        82.5,
        83.5,
        84.5,
        85.0,
        86.0,
        87.0,
        87.5,
        88.5,
        89.5,
        90.0,
        91.0,
        92.0,
        92.5,
        93.5,
        94.5,
        95.0,
        96.0,
        97.0,
        97.5,
        98.5,
        99.5,
        100.0,
        101.0,
        102.0,
        102.5,
        103.5,
        104.0,
    )


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_audit"
    out.mkdir(parents=True, exist_ok=True)
    upstream = Path(args.upstream_dir)
    paths = {
        "strict_path_manifest": upstream / "01_radial" / "strict_path_manifest.json",
        "radial_report": upstream / "01_radial" / "radial_report.json",
        "tube_report": upstream / "02_tube" / "tube_report.json",
        "robot_config": Path(args.robot_config),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V7-D1 audit sources are missing: {missing}")
    strict_manifest = read_json(paths["strict_path_manifest"])
    radial = read_json(paths["radial_report"])
    tube = read_json(paths["tube_report"])
    registered_radii = sorted(float(key) for key in strict_manifest)
    expected_radii = list(_expected_strict_radii())
    source_entries: dict[str, dict[str, Any]] = {}
    screen_artifact_hashes_match = True
    deferred_artifacts_registered = True
    for key, entry in strict_manifest.items():
        radius = float(key)
        record = audit_centerline_entry(entry, radius_mm=radius)
        source_entries[str(key)] = record
        if _is_post_selection_radius(radius):
            deferred_artifacts_registered &= bool(
                record["exists"] and record["sha256"]
            )
        else:
            screen_artifact_hashes_match &= bool(record["hash_matches"])
    family_id = str(radial.get("family_id", ""))
    protocol_family_id = str(radial.get("protocol", {}).get("family_id", ""))
    family_ids = [family_id] if family_id else []
    checks = {
        "strict_manifest_exact_35_radii": registered_radii == expected_radii,
        "all_screen_centerlines_present_and_hashed": screen_artifact_hashes_match,
        "post_selection_centerline_hashes_deferred": deferred_artifacts_registered
        and all(
            source_entries[_radius_key(radius)]["hash_verification_phase"]
            == "post_selection"
            and source_entries[_radius_key(radius)]["hash_matches"] is None
            for radius in POST_SELECTION_RADII_MM
        ),
        "registered_protocol_fixed_single_family": bool(
            family_id and family_id == protocol_family_id
        ),
        "standard_joint_domain": str(radial.get("joint_domain_id", ""))
        == JOINT_DOMAIN_ID,
        "radial_formal_gate_remains_failed": not bool(
            radial.get("formal_radial_gate_pass", True)
        ),
        "tube_formal_gate_remains_failed": not bool(tube.get("formal_tube_gate_pass", True)),
        "formal_strict_radius_unchanged": np.isclose(
            float(tube.get("radial_strict_rmax_mm", np.nan)),
            VALIDATION_RADIUS_MM,
            atol=1.0e-8,
        ),
        "formal_geometry_claim_absent": tube.get("strict_geometry_rmax_mm") is None,
    }
    source_manifest_sha256 = file_sha256(paths["strict_path_manifest"])
    fingerprint = stable_fingerprint(
        {
            "strategy_version": STRATEGY_VERSION,
            "phase": "audit",
            "sources": {key: file_sha256(path) for key, path in paths.items()},
            "strict_entries": strict_manifest,
            "claims": diagnostic_claims(),
        }
    )
    report = {
        "strategy_version": STRATEGY_VERSION,
        "task_fingerprint": fingerprint,
        **diagnostic_claims(),
        "upstream_dir": str(upstream.resolve()),
        "strict_path_manifest_path": str(paths["strict_path_manifest"].resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "source_reports": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in paths.items()
        },
        "centerline_manifest": source_entries,
        "registered_continuous_centerline_radii_mm": registered_radii,
        "registered_continuous_centerline_count": len(registered_radii),
        "logical_centerline_rows": len(registered_radii) * 360,
        "family_ids": family_ids,
        "upstream_formal_radial_gate_pass": bool(
            radial.get("formal_radial_gate_pass", False)
        ),
        "upstream_formal_tube_gate_pass": bool(tube.get("formal_tube_gate_pass", False)),
        "upstream_radial_strict_rmax_mm": tube.get("radial_strict_rmax_mm"),
        "upstream_strict_geometry_rmax_mm": tube.get("strict_geometry_rmax_mm"),
        "checks": checks,
        "audit_gate_pass": bool(all(checks.values())),
    }
    write_json(out / "audit_report.json", report)
    if not report["audit_gate_pass"]:
        raise RuntimeError(f"V7-D1 source audit failed: {checks}")
    return report


def ensure_audit_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "00_audit" / "audit_report.json"
    if path.is_file():
        cached = read_json(path)
        manifest_path = Path(cached.get("strict_path_manifest_path", ""))
        source_reports = cached.get("source_reports", {})
        centerlines = cached.get("centerline_manifest", {})
        if (
            bool(cached.get("audit_gate_pass", False))
            and manifest_path.is_file()
            and file_sha256(manifest_path) == str(cached.get("source_manifest_sha256", ""))
            and source_reports
            and all(
                _hashed_artifact_matches(record)
                for record in source_reports.values()
            )
            and centerlines
            and all(
                (
                    bool(record.get("path"))
                    and Path(record["path"]).is_file()
                    and bool(record.get("sha256"))
                )
                if _is_post_selection_radius(float(radius))
                else _hashed_artifact_matches(record)
                for radius, record in centerlines.items()
            )
        ):
            return cached
    return phase_audit(args)


def phase_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_dataset"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    manifest = read_json(Path(audit["strict_path_manifest_path"]))
    report = materialize_screen_dataset(
        manifest,
        output_path=out / "screen_train_validation_centerlines.parquet",
        expected_family_id=str(audit["family_ids"][0]),
        expected_points_per_radius=360,
    )
    report.update(
        {
            "strategy_version": STRATEGY_VERSION,
            "source_audit_task_fingerprint": str(audit["task_fingerprint"]),
            "source_manifest_sha256": str(audit["source_manifest_sha256"]),
            "logical_all_centerline_rows": int(audit["logical_centerline_rows"]),
            "logical_all_centerline_count": int(
                audit["registered_continuous_centerline_count"]
            ),
            "post_selection_rows_deferred_until_selection": int(
                len(POST_SELECTION_RADII_MM) * 360
            ),
        }
    )
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": STRATEGY_VERSION,
            "phase": "dataset",
            "audit": audit["task_fingerprint"],
            "dataset_sha256": report["dataset_sha256"],
            "split": report["split_policy"] if "split_policy" in report else {
                "train_max": TRAIN_MAX_RADIUS_MM,
                "validation": VALIDATION_RADIUS_MM,
            },
        }
    )
    write_json(out / "dataset_report.json", report)
    if not bool(report["dataset_gate_pass"]):
        raise RuntimeError(f"V7-D1 screen dataset gate failed: {report}")
    return report


def ensure_dataset_report(args: argparse.Namespace) -> dict[str, Any]:
    audit = ensure_audit_report(args)
    path = Path(args.out_dir) / "01_dataset" / "dataset_report.json"
    if path.is_file():
        cached = read_json(path)
        dataset = Path(cached.get("dataset_path", ""))
        if (
            bool(cached.get("dataset_gate_pass", False))
            and cache_lineage_matches(
                cached,
                {"source_audit_task_fingerprint": audit["task_fingerprint"]},
            )
            and dataset.is_file()
            and file_sha256(dataset) == str(cached.get("dataset_sha256", ""))
        ):
            return cached
    return phase_dataset(args)


def _execute_training_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    task_dir: str | Path,
    workers: int,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    directory = Path(task_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pending: list[Path] = []
    results: list[dict[str, Any]] = []
    for task_value in tasks:
        task = dict(task_value)
        expected = training_task_fingerprint(task)
        if str(task.get("task_fingerprint", "")) != expected:
            raise ValueError(f"stale V7-D1 task fingerprint: {task.get('task_id')}")
        result_path = Path(task["result_path"])
        package_path = Path(task["package_path"])
        if bool(skip_existing) and result_path.is_file() and package_path.is_file():
            cached = read_json(result_path)
            if (
                str(cached.get("task_fingerprint", "")) == expected
                and file_sha256(package_path) == str(cached.get("package_sha256", ""))
            ):
                results.append(cached)
                continue
        task_path = directory / f"{task['task_id']}.json"
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
                f"V7-D1 worker failed for {task_path}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        task = read_json(task_path)
        return read_json(Path(task["result_path"]))

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            results.extend(executor.map(execute, pending))
    return sorted(results, key=lambda row: str(row["task_id"]))


def flatten_training_results(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        base = result["config"]["base_config"]
        for label, metrics in result["evaluations"].items():
            rows.append(
                {
                    "task_id": result["task_id"],
                    "mode": result["mode"],
                    "seed": int(result["seed"]),
                    "config_id": result["config_id"],
                    **base,
                    "output_link_id": result["output_link_id"],
                    "target_link_clip_count": int(result["target_link_clip_count"]),
                    "evaluation_label": str(label),
                    **dict(metrics),
                    "package_path": result["package_path"],
                    "package_sha256": result["package_sha256"],
                }
            )
    return pd.DataFrame(rows)


def _resolved_stability_seeds(args: argparse.Namespace) -> list[int]:
    seeds = parse_int_csv(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("V7-D1 seeds must be non-empty and unique")
    if str(args.preset) == "full" and tuple(seeds) != FORMAL_SEEDS:
        raise ValueError(f"V7-D1 full run requires exact seeds {FORMAL_SEEDS}")
    return seeds[: preset_settings(str(args.preset))["seed_count"]]


def phase_screen(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_screen"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    dataset = ensure_dataset_report(args)
    settings = preset_settings(str(args.preset))
    configs = diagnostic_model_configs()
    limit = int(args.screen_config_limit) or int(settings["config_limit"])
    if limit > 0:
        configs = configs[:limit]
    tasks = build_screen_tasks(
        configs=configs,
        dataset_path=dataset["dataset_path"],
        source_manifest_sha256=audit["source_manifest_sha256"],
        robot_config_path=args.robot_config,
        out_dir=out,
        seed=SCREEN_SEED,
        max_iter=int(settings["max_iter"]),
        angle_stride=int(settings["angle_stride"]),
        dense_points=720,
    )
    results = _execute_training_tasks(
        tasks,
        task_dir=out / "worker_tasks",
        workers=int(args.workers),
        skip_existing=bool(args.skip_existing),
    )
    metrics = flatten_training_results(results)
    ranking = rank_diagnostic_screen(metrics)
    selection = select_ranked_config(ranking)
    metrics.to_csv(out / "screen_validation_metrics.csv", index=False)
    ranking.to_csv(out / "screen_config_ranking.csv", index=False)
    artifacts = {
        "screen_validation_metrics": _artifact_record(
            out / "screen_validation_metrics.csv"
        ),
        "screen_config_ranking": _artifact_record(out / "screen_config_ranking.csv"),
    }
    task_by_config = {str(task["config"]["config_id"]): task for task in tasks}
    result_manifest = {
        result["config_id"]: training_result_artifact_record(
            result_path=task_by_config[result["config_id"]]["result_path"],
            package_path=result["package_path"],
            package_sha256=result["package_sha256"],
            task_fingerprint=result["task_fingerprint"],
        )
        for result in results
    }
    result_artifact_fingerprint = phase_artifact_fingerprint(
        model_records=result_manifest,
        artifacts=artifacts,
    )
    selected_result = next(result for result in results if result["config_id"] == selection["selected_config_id"])
    report = {
        "strategy_version": STRATEGY_VERSION,
        **diagnostic_claims(),
        "preset": str(args.preset),
        "source_dataset_task_fingerprint": dataset["task_fingerprint"],
        "screen_seed": SCREEN_SEED,
        "screen_config_count": len(configs),
        "screen_run_count": len(results),
        "complete_48_config_screen": len(configs) == 48 and len(results) == 48,
        **selection,
        "selected_config": selected_result["config"],
        "selected_screen_result_path": result_manifest[selection["selected_config_id"]][
            "result_path"
        ],
        "selected_screen_package_path": selected_result["package_path"],
        "selected_screen_package_sha256": selected_result["package_sha256"],
        "metrics_path": str((out / "screen_validation_metrics.csv").resolve()),
        "ranking_path": str((out / "screen_config_ranking.csv").resolve()),
        "artifacts": artifacts,
        "model_results": result_manifest,
        "result_artifact_fingerprint": result_artifact_fingerprint,
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": STRATEGY_VERSION,
            "phase": "screen",
            "dataset": dataset["task_fingerprint"],
            "tasks": [task["task_fingerprint"] for task in tasks],
            "result_artifacts": result_artifact_fingerprint,
            "selected": selection,
        }
    )
    write_json(out / "selection_report.json", report)
    return report


def ensure_screen_report(args: argparse.Namespace) -> dict[str, Any]:
    dataset = ensure_dataset_report(args)
    path = Path(args.out_dir) / "02_screen" / "selection_report.json"
    settings = preset_settings(str(args.preset))
    expected_count = int(args.screen_config_limit) or int(settings["config_limit"])
    if expected_count <= 0:
        expected_count = len(diagnostic_model_configs())
    if path.is_file():
        cached = read_json(path)
        model_results = cached.get("model_results", {})
        artifacts = cached.get("artifacts", {})
        result_artifact_fingerprint = phase_artifact_fingerprint(
            model_records=model_results,
            artifacts=artifacts,
        )
        if (
            str(cached.get("preset", "")) == str(args.preset)
            and cache_lineage_matches(
                cached,
                {"source_dataset_task_fingerprint": dataset["task_fingerprint"]},
            )
            and int(cached.get("screen_config_count", 0)) == expected_count
            and len(model_results) == expected_count
            and artifacts
            and str(cached.get("result_artifact_fingerprint", ""))
            == result_artifact_fingerprint
            and all(
                _hashed_artifact_matches(record) for record in artifacts.values()
            )
            and all(training_result_artifact_matches(entry) for entry in model_results.values())
        ):
            return cached
    return phase_screen(args)


def aggregate_stability_gate(rows: pd.DataFrame, *, preset: str) -> dict[str, Any]:
    total = int(len(rows))
    unique = int(rows["seed"].nunique()) if total else 0
    passed = int(rows["model_gate_pass"].fillna(False).astype(bool).sum()) if total else 0
    required = 4 if str(preset) == "full" else total
    stable = bool(
        total > 0
        and unique == total
        and ((str(preset) == "full" and total == 5 and passed >= 4) or (str(preset) != "full" and passed == total))
    )
    return {
        "total_seed_count": total,
        "unique_seed_count": unique,
        "passed_seed_count": passed,
        "required_seed_count": required,
        "stable_gate_pass": stable,
    }


def phase_stability(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_stability"
    out.mkdir(parents=True, exist_ok=True)
    selection = ensure_screen_report(args)
    dataset = ensure_dataset_report(args)
    audit = ensure_audit_report(args)
    config = formal_training._config_from_dict(selection["selected_config"])
    seeds = _resolved_stability_seeds(args)
    settings = preset_settings(str(args.preset))
    results: list[dict[str, Any]] = []
    result_paths: dict[int, str] = {}
    if SCREEN_SEED in seeds:
        screen_result_path = str(selection["selected_screen_result_path"])
        results.append(read_json(Path(screen_result_path)))
        result_paths[SCREEN_SEED] = screen_result_path
    tasks = [
        make_training_task(
            dataset_path=dataset["dataset_path"],
            source_manifest_sha256=audit["source_manifest_sha256"],
            robot_config_path=args.robot_config,
            config=config,
            seed=seed,
            max_iter=int(settings["max_iter"]),
            angle_stride=int(settings["angle_stride"]),
            mode="v7_diagnostic_stability",
            result_path=out / "worker_results" / f"stability_{config.config_id}_s{seed}.json",
            package_path=out / "model_packages" / config.config_id / f"seed_{seed}.joblib",
            prediction_dir=out / "validation_predictions" / f"seed_{seed}",
            dense_points=720,
        )
        for seed in seeds
        if seed != SCREEN_SEED
    ]
    result_paths.update(
        {int(task["seed"]): str(task["result_path"]) for task in tasks}
    )
    results.extend(
        _execute_training_tasks(
            tasks,
            task_dir=out / "worker_tasks",
            workers=int(args.workers),
            skip_existing=bool(args.skip_existing),
        )
    )
    results = sorted(results, key=lambda row: int(row["seed"]))
    metrics = flatten_training_results(results)
    gates: dict[str, Any] = {}
    for label in ("validation_integer", "validation_half_phase"):
        table = metrics.loc[metrics["evaluation_label"].eq(label)].reset_index(drop=True)
        table.to_csv(out / f"{label}_metrics_all_seeds.csv", index=False)
        gates[label] = aggregate_stability_gate(table, preset=str(args.preset))
    artifacts = {
        label: _artifact_record(out / f"{label}_metrics_all_seeds.csv")
        for label in ("validation_integer", "validation_half_phase")
    }
    model_results = {
        str(row["seed"]): training_result_artifact_record(
            result_path=result_paths[int(row["seed"])],
            package_path=row["package_path"],
            package_sha256=row["package_sha256"],
            task_fingerprint=row["task_fingerprint"],
        )
        for row in results
    }
    result_artifact_fingerprint = phase_artifact_fingerprint(
        model_records=model_results,
        artifacts=artifacts,
    )
    report = {
        "strategy_version": STRATEGY_VERSION,
        **diagnostic_claims(),
        "preset": str(args.preset),
        "source_selection_task_fingerprint": selection["task_fingerprint"],
        "selected_config_id": config.config_id,
        "selected_config": config.as_dict(),
        "seeds": seeds,
        "seed_count": len(seeds),
        "reused_screen_seed": SCREEN_SEED in seeds,
        "full_seed_protocol_pass": bool(
            str(args.preset) == "full" and tuple(seeds) == FORMAL_SEEDS and len(results) == 5
        ),
        "evaluation_gates": gates,
        "artifacts": artifacts,
        "model_results": model_results,
        "result_artifact_fingerprint": result_artifact_fingerprint,
        "model_paths": {str(row["seed"]): row["package_path"] for row in results},
        "model_hashes": {str(row["seed"]): row["package_sha256"] for row in results},
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": STRATEGY_VERSION,
            "phase": "stability",
            "screen": selection["task_fingerprint"],
            "results": [row["task_fingerprint"] for row in results],
            "result_artifacts": result_artifact_fingerprint,
        }
    )
    write_json(out / "stability_report.json", report)
    return report


def ensure_stability_report(args: argparse.Namespace) -> dict[str, Any]:
    selection = ensure_screen_report(args)
    path = Path(args.out_dir) / "03_stability" / "stability_report.json"
    expected_seeds = _resolved_stability_seeds(args)
    if path.is_file():
        cached = read_json(path)
        artifacts = cached.get("artifacts", {})
        model_results = cached.get("model_results", {})
        result_artifact_fingerprint = phase_artifact_fingerprint(
            model_records=model_results,
            artifacts=artifacts,
        )
        if (
            str(cached.get("preset", "")) == str(args.preset)
            and str(cached.get("selected_config_id", ""))
            == str(selection["selected_config_id"])
            and cache_lineage_matches(
                cached,
                {
                    "source_selection_task_fingerprint": selection[
                        "task_fingerprint"
                    ]
                },
            )
            and [int(seed) for seed in cached.get("seeds", [])] == expected_seeds
            and artifacts
            and model_results
            and str(cached.get("result_artifact_fingerprint", ""))
            == result_artifact_fingerprint
            and all(
                _hashed_artifact_matches(record) for record in artifacts.values()
            )
            and all(training_result_artifact_matches(entry) for entry in model_results.values())
        ):
            return cached
    return phase_stability(args)


def _radius_slug(radius_mm: float) -> str:
    return f"r{float(radius_mm):06.2f}".replace(".", "p")


def materialize_registered_centerlines_after_selection(
    *,
    strict_manifest: Mapping[str, Any],
    selection_report_path: str | Path,
    output_path: str | Path,
    expected_family_id: str | None = None,
    expected_points_per_radius: int = 360,
) -> dict[str, Any]:
    """Open all 35 sources only after a frozen selection report exists."""

    selection_path = Path(selection_report_path)
    if not selection_path.is_file():
        raise FileNotFoundError("post-selection centerlines require a frozen selection report")
    selection_sha256 = file_sha256(selection_path)
    selection = read_json(selection_path)
    if not selection.get("selected_config_id"):
        raise ValueError("post-selection centerlines require a completed model selection")
    frames: list[pd.DataFrame] = []
    source_records: dict[str, Any] = {}
    for radius in sorted(float(key) for key in strict_manifest):
        entry = _source_entry(strict_manifest, radius)
        path = Path(entry["path"])
        frame = _validate_centerline_frame(
            pd.read_parquet(path),
            radius_mm=radius,
            expected_points=expected_points_per_radius,
        )
        family_id = str(frame["family_id"].iloc[0])
        trajectory_id = f"{family_id}@r{radius:07.2f}mm"
        frame["trajectory_id"] = trajectory_id
        frame["sample_id"] = [
            f"{trajectory_id}:phase-{index:04d}"
            for index in frame["angle_idx"].to_numpy(dtype=int)
        ]
        if radius <= TRAIN_MAX_RADIUS_MM + 1.0e-9:
            role = "training_truth"
        elif np.isclose(radius, VALIDATION_RADIUS_MM, atol=1.0e-8):
            role = "validation_truth"
        elif radius in POST_SELECTION_RADII_MM:
            role = "post_selection_truth"
        else:
            role = "registered_continuous_truth"
        frame["diagnostic_role"] = role
        frame["source_centerline_path"] = str(path.resolve())
        frame["source_centerline_sha256"] = str(entry["sha256"])
        add_diagnostic_claim_columns(frame)
        frames.append(frame)
        source_records[_radius_key(radius)] = {
            "path": str(path.resolve()),
            "sha256": str(entry["sha256"]),
            "rows": len(frame),
            "diagnostic_role": role,
        }
    dataset = pd.concat(frames, ignore_index=True)
    if not dataset["sample_id"].is_unique:
        raise ValueError("all-centerline diagnostic sample IDs are not unique")
    family_ids = sorted(dataset["family_id"].astype(str).unique().tolist())
    if len(family_ids) != 1 or (
        expected_family_id is not None
        and family_ids != [str(expected_family_id)]
    ):
        raise ValueError(
            "post-selection centerlines must resolve to the registered single family"
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False, compression="zstd")
    return {
        **diagnostic_claims(),
        "selection_frozen_before_outer_read": True,
        "selection_report_path": str(selection_path.resolve()),
        "selection_report_sha256_before_outer_read": selection_sha256,
        "dataset_path": str(output.resolve()),
        "dataset_sha256": file_sha256(output),
        "rows": int(len(dataset)),
        "radius_count": int(dataset["radius_mm"].nunique()),
        "trajectory_count": int(dataset["trajectory_id"].nunique()),
        "family_ids": family_ids,
        "fixed_single_family_gate_pass": True,
        "post_selection_rows": int(
            dataset["diagnostic_role"].eq("post_selection_truth").sum()
        ),
        "source_centerlines": source_records,
    }


def _verified_diagnostic_package(
    *,
    path: str | Path,
    expected_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    package_path = Path(path)
    if file_sha256(package_path) != str(expected_sha256):
        raise ValueError(f"diagnostic model package hash changed: {package_path}")
    package = load(package_path)
    checks = {
        "kind": package.get("kind") == DIAGNOSTIC_KIND,
        "diagnostic_only": bool(package.get("diagnostic_only", False)),
        "no_formal_claims": not bool(package.get("formal_claims_allowed", True)),
        "does_not_change_v7": not bool(package.get("changes_v7_formal_gate", True)),
        "claim_radius_absent": package.get("static_inverse_claim_radius_mm") is None,
        "source_manifest": str(package.get("source_manifest_sha256", ""))
        == str(source_manifest_sha256),
    }
    if not all(checks.values()):
        raise ValueError(f"diagnostic model package failed claim/provenance checks: {checks}")
    return package


def _prediction_record(
    *,
    path: Path,
    config_id: str,
    seed: int,
    radius_mm: float,
    evidence_class: str | TrajectoryEvidenceClass,
) -> dict[str, Any]:
    evidence = TrajectoryEvidenceClass(evidence_class)
    return {
        **diagnostic_claims(),
        "config_id": str(config_id),
        "seed": int(seed),
        "radius_mm": float(radius_mm),
        "evidence_class": evidence.value,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def _evaluate_package_over_radii(
    *,
    package: Mapping[str, Any],
    config_id: str,
    seed: int,
    package_path: str | Path,
    package_sha256: str,
    frames: Mapping[float, pd.DataFrame],
    evidence_class: str | TrajectoryEvidenceClass,
    robot_inputs: Any,
    theta_sign: float,
    prediction_dir: Path | None,
    save_radii: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    save_set = {float(radius) for radius in save_radii}
    for radius, frame in sorted(frames.items()):
        metrics, prediction = evaluate_package_trajectory(
            package=package,
            frame=frame,
            robot_inputs=robot_inputs,
            theta_sign=theta_sign,
            evidence_class=evidence_class,
        )
        metrics.update(
            {
                "config_id": str(config_id),
                "seed": int(seed),
                "output_link_id": str(package["output_link_id"]),
                "package_path": str(Path(package_path).resolve()),
                "package_sha256": str(package_sha256),
            }
        )
        metrics_rows.append(metrics)
        if prediction_dir is not None and float(radius) in save_set:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            path = prediction_dir / f"{_radius_slug(radius)}.parquet"
            prediction["config_id"] = str(config_id)
            prediction["seed"] = int(seed)
            prediction["output_link_id"] = str(package["output_link_id"])
            prediction.to_parquet(path, index=False, compression="zstd")
            predictions.append(
                _prediction_record(
                    path=path,
                    config_id=config_id,
                    seed=seed,
                    radius_mm=radius,
                    evidence_class=evidence_class,
                )
            )
    return metrics_rows, predictions


def _best_visual_seed(metrics: pd.DataFrame) -> int:
    rows: list[dict[str, Any]] = []
    for seed, part in metrics.groupby("seed", sort=False):
        rows.append(
            {
                "seed": int(seed),
                "all_tracking_gates_pass": bool(
                    part["diagnostic_tracking_gate_pass"].astype(bool).all()
                ),
                "worst_ee_p95_mm": float(part["ee_p95_mm"].max()),
                "worst_beta_p95_deg": float(part["beta_p95_deg"].max()),
                "minimum_joint_margin_deg": float(
                    part["prediction_min_joint_margin_deg"].min()
                ),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "all_tracking_gates_pass",
            "worst_ee_p95_mm",
            "worst_beta_p95_deg",
            "minimum_joint_margin_deg",
            "seed",
        ],
        ascending=[False, True, True, False, True],
        kind="stable",
    )
    return int(ranking.iloc[0]["seed"])


def phase_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "04_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    selection = ensure_screen_report(args)
    stability = ensure_stability_report(args)
    audit = ensure_audit_report(args)
    selection_path = Path(args.out_dir) / "02_screen" / "selection_report.json"
    selection_sha_before = file_sha256(selection_path)
    manifest = read_json(Path(audit["strict_path_manifest_path"]))
    materialized = materialize_registered_centerlines_after_selection(
        strict_manifest=manifest,
        selection_report_path=selection_path,
        output_path=out / "registered_centerlines_after_selection.parquet",
        expected_family_id=str(audit["family_ids"][0]),
        expected_points_per_radius=360,
    )
    if materialized["selection_report_sha256_before_outer_read"] != selection_sha_before:
        raise RuntimeError("selection report changed while post-selection data were opened")
    centerlines = pd.read_parquet(materialized["dataset_path"])
    accepted_radii = (100.0, 101.0, 102.0, 102.5, 103.5, 104.0)
    accepted_frames: dict[float, pd.DataFrame] = {}
    for radius in accepted_radii:
        source = centerlines.loc[
            np.isclose(centerlines["radius_mm"].to_numpy(dtype=float), radius, atol=1.0e-8)
        ].reset_index(drop=True)
        accepted_frames[radius] = dense_centerline_frame(
            source, n_points=720, include_beta_truth=True
        )
    template = centerlines.loc[
        np.isclose(centerlines["radius_mm"].to_numpy(dtype=float), 104.0, atol=1.0e-8)
    ].reset_index(drop=True)
    unsupported_frames = {
        radius: target_only_centerline_frame(template, radius_mm=radius, n_points=720)
        for radius in UNSUPPORTED_RADII_MM
    }
    robot_config = formal_training.v5.v4.load_config(str(args.robot_config))
    robot_inputs = formal_training.v5.v4.load_robot_inputs(robot_config)
    theta_sign = float(robot_config.get("kinematics", {}).get("theta_sign", -1.0))

    screen_metrics: list[dict[str, Any]] = []
    screen_predictions: list[dict[str, Any]] = []
    for config_id, entry in sorted(selection["model_results"].items()):
        package = _verified_diagnostic_package(
            path=entry["package_path"],
            expected_sha256=entry["package_sha256"],
            source_manifest_sha256=audit["source_manifest_sha256"],
        )
        metrics, predictions = _evaluate_package_over_radii(
            package=package,
            config_id=config_id,
            seed=SCREEN_SEED,
            package_path=entry["package_path"],
            package_sha256=entry["package_sha256"],
            frames=accepted_frames,
            evidence_class=TrajectoryEvidenceClass.ACCEPTED_CONTINUOUS_TRUTH,
            robot_inputs=robot_inputs,
            theta_sign=theta_sign,
            prediction_dir=out / "screen_model_predictions" / config_id,
            save_radii=DISPLAY_RADII_MM,
        )
        screen_metrics.extend(metrics)
        screen_predictions.extend(predictions)
    screen_table = pd.DataFrame(screen_metrics)
    screen_table.to_csv(out / "screen_model_accepted_metrics.csv", index=False)

    stability_metrics: list[dict[str, Any]] = []
    stability_predictions: list[dict[str, Any]] = []
    unsupported_metrics: list[dict[str, Any]] = []
    unsupported_predictions: list[dict[str, Any]] = []
    selected_config_id = str(stability["selected_config_id"])
    for seed_text, model_path in sorted(stability["model_paths"].items(), key=lambda row: int(row[0])):
        seed = int(seed_text)
        package_sha = str(stability["model_hashes"][seed_text])
        package = _verified_diagnostic_package(
            path=model_path,
            expected_sha256=package_sha,
            source_manifest_sha256=audit["source_manifest_sha256"],
        )
        metrics, predictions = _evaluate_package_over_radii(
            package=package,
            config_id=selected_config_id,
            seed=seed,
            package_path=model_path,
            package_sha256=package_sha,
            frames=accepted_frames,
            evidence_class=TrajectoryEvidenceClass.ACCEPTED_CONTINUOUS_TRUTH,
            robot_inputs=robot_inputs,
            theta_sign=theta_sign,
            prediction_dir=out / "stability_predictions" / f"seed_{seed}",
            save_radii=DISPLAY_RADII_MM,
        )
        stability_metrics.extend(metrics)
        stability_predictions.extend(predictions)
        metrics, predictions = _evaluate_package_over_radii(
            package=package,
            config_id=selected_config_id,
            seed=seed,
            package_path=model_path,
            package_sha256=package_sha,
            frames=unsupported_frames,
            evidence_class=TrajectoryEvidenceClass.UNSUPPORTED_MODEL_EXTRAPOLATION,
            robot_inputs=robot_inputs,
            theta_sign=theta_sign,
            prediction_dir=out / "unsupported_predictions" / f"seed_{seed}",
            save_radii=BEST_UNSUPPORTED_DISPLAY_RADII_MM,
        )
        unsupported_metrics.extend(metrics)
        unsupported_predictions.extend(predictions)
    stability_table = pd.DataFrame(stability_metrics)
    stability_table.to_csv(out / "selected_config_accepted_metrics_all_seeds.csv", index=False)
    unsupported_table = pd.DataFrame(unsupported_metrics)
    unsupported_table.to_csv(
        out / "selected_config_unsupported_metrics_all_seeds.csv", index=False
    )
    best_seed = _best_visual_seed(stability_table)
    prediction_manifest = {
        **diagnostic_claims(),
        "screen_models": screen_predictions,
        "selected_config_stability": stability_predictions,
        "selected_config_unsupported": unsupported_predictions,
    }
    prediction_manifest_path = out / "prediction_manifest.json"
    write_json(prediction_manifest_path, prediction_manifest)
    artifacts = {
        "screen_model_metrics": _artifact_record(
            out / "screen_model_accepted_metrics.csv"
        ),
        "stability_metrics": _artifact_record(
            out / "selected_config_accepted_metrics_all_seeds.csv"
        ),
        "unsupported_metrics": _artifact_record(
            out / "selected_config_unsupported_metrics_all_seeds.csv"
        ),
        "prediction_manifest": _artifact_record(prediction_manifest_path),
        "all_registered_centerlines": {
            "path": str(Path(materialized["dataset_path"]).resolve()),
            "sha256": str(materialized["dataset_sha256"]),
        },
    }
    report = {
        "strategy_version": STRATEGY_VERSION,
        **diagnostic_claims(),
        "preset": str(args.preset),
        "source_selection_task_fingerprint": selection["task_fingerprint"],
        "source_stability_task_fingerprint": stability["task_fingerprint"],
        "selection_frozen_before_outer_read": True,
        "selection_report_sha256_before_outer_read": selection_sha_before,
        "all_registered_centerlines": materialized,
        "accepted_evaluation_radii_mm": list(accepted_radii),
        "unsupported_evaluation_radii_mm": list(UNSUPPORTED_RADII_MM),
        "screen_model_count": int(screen_table["config_id"].nunique()),
        "screen_metric_rows": int(len(screen_table)),
        "selected_config_id": selected_config_id,
        "stability_seed_count": int(stability_table["seed"].nunique()),
        "best_visual_seed": best_seed,
        "accepted_beta_truth_available": True,
        "unsupported_beta_truth_available": False,
        "screen_model_metrics_path": str(
            (out / "screen_model_accepted_metrics.csv").resolve()
        ),
        "stability_metrics_path": str(
            (out / "selected_config_accepted_metrics_all_seeds.csv").resolve()
        ),
        "unsupported_metrics_path": str(
            (out / "selected_config_unsupported_metrics_all_seeds.csv").resolve()
        ),
        "prediction_manifest_path": str(prediction_manifest_path.resolve()),
        "artifacts": artifacts,
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": STRATEGY_VERSION,
            "phase": "evaluate",
            "selection": selection["task_fingerprint"],
            "stability": stability["task_fingerprint"],
            "all_centerlines": materialized["dataset_sha256"],
            "prediction_manifest": prediction_manifest,
        }
    )
    write_json(out / "evaluation_report.json", report)
    return report


def ensure_evaluation_report(args: argparse.Namespace) -> dict[str, Any]:
    selection = ensure_screen_report(args)
    stability = ensure_stability_report(args)
    path = Path(args.out_dir) / "04_evaluation" / "evaluation_report.json"
    if path.is_file():
        cached = read_json(path)
        artifacts = cached.get("artifacts", {})
        if (
            str(cached.get("preset", "")) == str(args.preset)
            and cache_lineage_matches(
                cached,
                {
                    "source_selection_task_fingerprint": selection[
                        "task_fingerprint"
                    ],
                    "source_stability_task_fingerprint": stability[
                        "task_fingerprint"
                    ],
                },
            )
            and artifacts
            and all(
                _hashed_artifact_matches(record) for record in artifacts.values()
            )
        ):
            return cached
    return phase_evaluate(args)


def phase_visualize(args: argparse.Namespace) -> dict[str, Any]:
    evaluation = ensure_evaluation_report(args)
    import plot_true_ellipse_standard_domain_model_trajectories_v7 as plotter

    report = plotter.render_model_trajectory_suite(
        run_dir=args.out_dir,
        out_dir=Path(args.out_dir) / "05_visualization",
    )
    report["preset"] = str(args.preset)
    report["source_evaluation_task_fingerprint"] = evaluation["task_fingerprint"]
    write_json(Path(args.out_dir) / "05_visualization" / "visualization_report.json", report)
    if not bool(report.get("visualization_integrity_gate_pass", False)):
        raise RuntimeError(f"V7-D1 visualization integrity gate failed: {report['checks']}")
    return report


def ensure_visualization_report(args: argparse.Namespace) -> dict[str, Any]:
    evaluation = ensure_evaluation_report(args)
    path = Path(args.out_dir) / "05_visualization" / "visualization_report.json"
    if path.is_file():
        cached = read_json(path)
        evaluation_path = Path(
            cached.get(
                "evaluation_report_path",
                Path(args.out_dir) / "04_evaluation" / "evaluation_report.json",
            )
        )
        manifest_path = Path(cached.get("prediction_manifest_path", ""))
        figures = [
            *cached.get("individual_models", []),
            *cached.get("summary_figures", []),
        ]
        if (
            str(cached.get("preset", "")) == str(args.preset)
            and cache_lineage_matches(
                cached,
                {
                    "source_evaluation_task_fingerprint": evaluation[
                        "task_fingerprint"
                    ]
                },
            )
            and evaluation_path.is_file()
            and file_sha256(evaluation_path)
            == str(cached.get("evaluation_report_sha256", ""))
            and manifest_path.is_file()
            and file_sha256(manifest_path)
            == str(cached.get("prediction_manifest_sha256", ""))
            and bool(cached.get("visualization_integrity_gate_pass", False))
            and figures
            and all(
                Path(item["png_path"]).is_file()
                and file_sha256(item["png_path"]) == str(item["png_sha256"])
                for item in figures
            )
        ):
            return cached
    return phase_visualize(args)


def _stable_radius_summary(metrics: pd.DataFrame, *, preset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for radius, part in metrics.groupby("radius_mm", sort=True):
        total = int(part["seed"].nunique())
        passed = int(part["diagnostic_tracking_gate_pass"].astype(bool).sum())
        required = 4 if str(preset) == "full" else total
        stable = bool(
            (str(preset) == "full" and total == 5 and passed >= 4)
            or (str(preset) != "full" and total > 0 and passed == total)
        )
        rows.append(
            {
                **diagnostic_claims(),
                "radius_mm": float(radius),
                "seed_count": total,
                "passed_seed_count": passed,
                "required_seed_count": required,
                "diagnostic_stable_tracking_pass": stable,
                "ee_p95_worst_seed_mm": float(part["ee_p95_mm"].max()),
                "beta_p95_worst_seed_deg": float(part["beta_p95_deg"].max()),
                "minimum_predicted_joint_margin_deg": float(
                    part["prediction_min_joint_margin_deg"].min()
                ),
                "total_bound_violation_count": int(
                    part["beta_bound_violation_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "06_summary"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    dataset = ensure_dataset_report(args)
    selection = ensure_screen_report(args)
    stability = ensure_stability_report(args)
    evaluation = ensure_evaluation_report(args)
    visualization = ensure_visualization_report(args)
    screen_ranking = pd.read_csv(selection["ranking_path"])
    accepted = pd.read_csv(evaluation["stability_metrics_path"])
    unsupported = pd.read_csv(evaluation["unsupported_metrics_path"])
    radius_summary = _stable_radius_summary(accepted, preset=str(args.preset))
    radius_summary_path = out / "selected_config_accepted_radius_summary.csv"
    radius_summary.to_csv(radius_summary_path, index=False)
    stable_radii = radius_summary.loc[
        radius_summary["diagnostic_stable_tracking_pass"].astype(bool), "radius_mm"
    ].astype(float).tolist()
    max_stable = max(stable_radii) if stable_radii else None
    top_columns = [
        column
        for column in (
            "config_id",
            "output_link_id",
            "both_validation_gates_pass",
            "worst_ee_p95_mm",
            "worst_beta_p95_deg",
            "minimum_joint_margin_deg",
            "fit_s",
        )
        if column in screen_ranking
    ]
    top = screen_ranking[top_columns].head(10)
    lines = [
        "# True Ellipse V7-D1 诊断中心线模型实验",
        "",
        "> 本实验只回答“V7 已生成中心线能否被静态 xyz→beta6 MLP 跟踪”。它不属于正式 V7 模型门，不能上调正式可拟合半径。",
        "",
        f"- 运行预设：`{args.preset}`。",
        f"- 固定 family：`{audit['family_ids'][0]}`。",
        f"- 注册连续中心线：`{audit['registered_continuous_centerline_count']}` 条 / `{audit['logical_centerline_rows']}` 行。",
        f"- 训练集：`{dataset['training_rows']}` 行、`{len(dataset['training_radii_mm'])}` 条完整半径，最大 `100 mm`。",
        f"- 模型选择验证：`102.5 mm` 整数相位与半相位。",
        f"- screen 配置：`{selection['screen_config_count']}`；最佳配置：`{selection['selected_config_id']}`。",
        f"- 选择说明：`{selection['selection_note']}`。",
        f"- 稳定性 seeds：`{stability['seeds']}`。",
        f"- Screen 结果产物指纹：`{selection['result_artifact_fingerprint']}`。",
        f"- Stability 结果产物指纹：`{stability['result_artifact_fingerprint']}`。",
        f"- 接受连续路径中诊断稳定通过的最大已评估半径：`{max_stable}`。",
        f"- 正式 V7 模型结论：`False`；正式静态逆模型 claim 半径：`null`。",
        "",
        "## 证据边界",
        "",
        "- 100/101/102/102.5/103.5/104 mm 使用连续中心线及周期插值 beta 诊断真值。",
        "- 105–120 mm 仅使用解析目标轨迹，经模型预测 beta 后做 FK；没有 beta 真值，不能报告 beta 拟合能力。",
        "- 101/102/103.5/104 mm 在配置选择报告冻结后才读取，未进入 screen 数据文件。",
        "- 所有模型包、指标、预测和图片均携带 `diagnostic_only=true`、`formal_claims_allowed=false`。",
        "",
        "## Screen 前十名",
        "",
        dataframe_to_markdown(top),
        "",
        "## 最佳配置各接受半径稳定性",
        "",
        dataframe_to_markdown(radius_summary),
        "",
        "## 不支持半径的解释",
        "",
        f"外推指标共有 `{len(unsupported)}` 行，但 `beta_truth_available` 始终为 `False`；这些结果只用于观察 Cartesian 跟踪、连续性、关节边界与模型失稳趋势。",
        "",
        "## 图像",
        "",
        f"- 单模型轨迹图：`{len(visualization['individual_models'])}` 张。",
        f"- 汇总图：`{len(visualization['summary_figures'])}` 张。",
        "- 相机策略：SVD 轨迹平面法向 + 12° 方位偏移，避免椭圆被压成直线。",
    ]
    summary_path = out / "experiment_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "strategy_version": STRATEGY_VERSION,
        **diagnostic_claims(),
        "preset": str(args.preset),
        "selected_config_id": selection["selected_config_id"],
        "selection_note": selection["selection_note"],
        "screen_config_count": int(selection["screen_config_count"]),
        "complete_48_config_screen": bool(selection["complete_48_config_screen"]),
        "stability_seeds": stability["seeds"],
        "diagnostic_stable_accepted_radii_mm": stable_radii,
        "diagnostic_max_stable_evaluated_accepted_radius_mm": max_stable,
        "unsupported_beta_truth_available": False,
        "formal_model_gate_pass": False,
        "upstream_v7_formal_radial_gate_pass": bool(
            audit["upstream_formal_radial_gate_pass"]
        ),
        "upstream_v7_formal_tube_gate_pass": bool(
            audit["upstream_formal_tube_gate_pass"]
        ),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "radius_summary_path": str(radius_summary_path.resolve()),
        "radius_summary_sha256": file_sha256(radius_summary_path),
        "visualization_integrity_gate_pass": bool(
            visualization["visualization_integrity_gate_pass"]
        ),
        "result_artifact_fingerprints": summary_result_artifact_fingerprints(
            selection, stability
        ),
        "source_task_fingerprints": {
            "audit": audit["task_fingerprint"],
            "dataset": dataset["task_fingerprint"],
            "screen": selection["task_fingerprint"],
            "stability": stability["task_fingerprint"],
            "evaluation": evaluation["task_fingerprint"],
        },
    }
    report["task_fingerprint"] = stable_fingerprint(report)
    write_json(out / "final_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualize the non-formal V7 centerline model diagnostic."
    )
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--phases", default="all")
    parser.add_argument("--preset", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in FORMAL_SEEDS))
    parser.add_argument("--screen-config-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--worker-task", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_protocol_args(args)
    phases = parse_phases(args.phases)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    handlers = {
        "audit": phase_audit,
        "dataset": phase_dataset,
        "screen": phase_screen,
        "stability": phase_stability,
        "evaluate": phase_evaluate,
        "visualize": phase_visualize,
        "summary": phase_summary,
    }
    results: dict[str, Any] = {}
    for phase in phases:
        results[phase] = handlers[phase](args)
    report = {
        "mode": "true_ellipse_standard_domain_training_v7_diagnostic_centerline",
        **diagnostic_claims(),
        "preset": str(args.preset),
        "phases": phases,
        "out_dir": str(Path(args.out_dir).resolve()),
        "results": results,
    }
    write_json(Path(args.out_dir) / "run_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_task is not None:
        result = run_training_worker(read_json(args.worker_task))
        print(json.dumps(result, ensure_ascii=False, default=_json_default))
        return 0
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
