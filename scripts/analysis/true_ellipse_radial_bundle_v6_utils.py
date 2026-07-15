#!/usr/bin/env python3
"""Utilities for the V6 fixed-family radial path-bundle experiment.

V6 treats the inverse solution as a two-dimensional surface ``beta(phase,
radius)``.  The module deliberately keeps target construction, radial
prediction, cyclic correction, robustness checks, and dataset partitioning
separate so each evidence boundary can be audited independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import true_ellipse_atlas_utils as atlas


RADIAL_BUNDLE_STRATEGY_VERSION = 1
SOLVER_STRATEGY_VERSION = "true-ellipse-radial-bundle-v6.1"
FORMAL_RADII_MM = (75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0)
FORMAL_CUT_INDICES = (0, 90, 180, 270)
FORMAL_TUBE_OFFSETS_MM = (-5.0, -2.5, 0.0, 2.5, 5.0)
VALIDATION_RADIUS_MM = 92.5
TEST_RADIUS_MM = 100.0
ANCHOR_SCHEDULES: dict[str, tuple[float, float, float]] = {
    "conservative": (1.0, 0.3, 0.1),
    "balanced": (0.3, 0.1, 0.03),
    "loose": (0.1, 0.03, 0.01),
}


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    center_x_m: float
    center_y_m: float
    center_z_m: float
    phase_y_rad: float
    phase_z_rad: float


def generate_radius_targets(
    family: FamilySpec,
    *,
    radius_mm: float,
    n_points: int,
) -> pd.DataFrame:
    """Generate one true sin/sin/cos ellipse without changing its geometry."""
    targets = atlas.make_axis_phase_ellipse(
        candidate_id=str(family.family_id),
        center=(float(family.center_x_m), float(family.center_y_m), float(family.center_z_m)),
        amp_xy_mm=float(radius_mm),
        phase_y_rad=float(family.phase_y_rad),
        phase_z_rad=float(family.phase_z_rad),
        n_points=int(n_points),
        include_endpoint=False,
    )
    targets["family_id"] = str(family.family_id)
    targets["radius_mm"] = float(radius_mm)
    return targets


def validate_target_geometry(
    targets: pd.DataFrame,
    *,
    family: FamilySpec,
    radius_mm: float,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    """Fail closed when a target no longer matches the registered family."""
    required = {
        "family_id",
        "radius_mm",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "phase_y_rad",
        "phase_z_rad",
        "amp_xy_mm",
        "amp_z_mm",
        *atlas.TARGET_XYZ_COLS,
    }
    missing = sorted(required - set(targets.columns))
    geometry = (
        atlas.trajectory_geometry_metrics(targets[atlas.TARGET_XYZ_COLS].to_numpy(dtype=float))
        if not missing and len(targets)
        else {"rank2_gate_pass": False}
    )

    def all_close(column: str, value: float) -> bool:
        return bool(
            column in targets
            and len(targets)
            and np.allclose(targets[column].to_numpy(dtype=float), float(value), atol=float(atol), rtol=0.0)
        )

    family_id_preserved = bool(
        "family_id" in targets
        and len(targets)
        and targets["family_id"].astype(str).eq(str(family.family_id)).all()
    )
    center_preserved = bool(
        all_close("center_x_m", family.center_x_m)
        and all_close("center_y_m", family.center_y_m)
        and all_close("center_z_m", family.center_z_m)
    )
    phase_preserved = bool(
        all_close("phase_y_rad", family.phase_y_rad)
        and all_close("phase_z_rad", family.phase_z_rad)
    )
    amplitude_preserved = bool(
        all_close("radius_mm", radius_mm)
        and all_close("amp_xy_mm", radius_mm)
        and all_close("amp_z_mm", 1.5 * float(radius_mm))
    )
    report: dict[str, Any] = {
        **geometry,
        "rows": int(len(targets)),
        "missing_columns": missing,
        "family_id_preserved": family_id_preserved,
        "center_fields_preserved": center_preserved,
        "phase_fields_preserved": phase_preserved,
        "amplitude_fields_preserved": amplitude_preserved,
    }
    report["target_geometry_gate_pass"] = bool(
        not missing
        and report.get("rank2_gate_pass", False)
        and family_id_preserved
        and center_preserved
        and phase_preserved
        and amplitude_preserved
    )
    return report


def _aligned_fixed_family_path(
    path: pd.DataFrame,
    *,
    family_id: str,
    expected_radius_mm: float | None = None,
) -> pd.DataFrame:
    required = {"angle_idx", "family_id", *atlas.BETA_COLS}
    missing = sorted(required - set(path.columns))
    if missing:
        raise ValueError(f"radial parent path missing columns: {missing}")
    if path.empty:
        raise ValueError("radial parent path is empty")
    ordered = path.sort_values("angle_idx", kind="stable").reset_index(drop=True).copy()
    angle_idx = ordered["angle_idx"].to_numpy(dtype=np.int64)
    if len(np.unique(angle_idx)) != len(angle_idx) or not np.array_equal(angle_idx, np.arange(len(ordered))):
        raise ValueError("radial parent path must contain each contiguous angle_idx exactly once")
    if not ordered["family_id"].astype(str).eq(str(family_id)).all():
        raise ValueError("fixed-family invariant violated by radial parent path")
    beta = ordered[atlas.BETA_COLS].to_numpy(dtype=float)
    if not np.all(np.isfinite(beta)):
        raise ValueError("radial parent path contains non-finite beta values")
    if expected_radius_mm is not None:
        if "radius_mm" not in ordered:
            raise ValueError("radial parent path is missing radius_mm")
        if not np.allclose(
            ordered["radius_mm"].to_numpy(dtype=float),
            float(expected_radius_mm),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError("radial parent path radius does not match its registered radius")
    return ordered


def build_radial_predictor(
    parent: pd.DataFrame,
    *,
    parent_radius_mm: float,
    target_radius_mm: float,
    family_id: str,
    previous_parent: pd.DataFrame | None = None,
    previous_parent_radius_mm: float | None = None,
    bounds: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a full-path copy or secant predictor aligned at every phase."""
    current = _aligned_fixed_family_path(
        parent,
        family_id=str(family_id),
        expected_radius_mm=float(parent_radius_mm),
    )
    current_beta = current[atlas.BETA_COLS].to_numpy(dtype=float)
    predictor_beta = current_beta.copy()
    predictor_type = "parent_copy"
    previous_radius: float | None = None
    if previous_parent is not None:
        if previous_parent_radius_mm is None:
            raise ValueError("previous_parent_radius_mm is required for radial secant prediction")
        previous_radius = float(previous_parent_radius_mm)
        previous = _aligned_fixed_family_path(
            previous_parent,
            family_id=str(family_id),
            expected_radius_mm=previous_radius,
        )
        if previous["angle_idx"].tolist() != current["angle_idx"].tolist():
            raise ValueError("radial parents must have identical angle_idx values")
        denominator = float(parent_radius_mm) - previous_radius
        if abs(denominator) <= 1.0e-12:
            raise ValueError("radial secant parents must have distinct radii")
        factor = (float(target_radius_mm) - float(parent_radius_mm)) / denominator
        previous_beta = previous[atlas.BETA_COLS].to_numpy(dtype=float)
        predictor_beta = current_beta + factor * (current_beta - previous_beta)
        predictor_type = "radial_secant"
    if bounds is not None:
        limits = np.asarray(bounds, dtype=float).reshape(6, 2)
        predictor_beta = np.clip(predictor_beta, limits[:, 0], limits[:, 1])

    keep_columns = [column for column in ("angle_idx", "angle_rad") if column in current]
    output = current[keep_columns].copy()
    for idx, column in enumerate(atlas.BETA_COLS):
        output[column] = predictor_beta[:, idx]
    output["family_id"] = str(family_id)
    output["radius_mm"] = float(target_radius_mm)
    output["parent_radius_mm"] = float(parent_radius_mm)
    output["previous_parent_radius_mm"] = previous_radius
    output["radial_predictor_type"] = predictor_type
    report = {
        "radial_predictor_type": predictor_type,
        "rows": int(len(output)),
        "angle_count": int(output["angle_idx"].nunique()),
        "full_parent_path_used": bool(
            len(output) == len(current)
            and output["angle_idx"].tolist() == current["angle_idx"].tolist()
        ),
        "parent_radius_mm": float(parent_radius_mm),
        "previous_parent_radius_mm": previous_radius,
        "target_radius_mm": float(target_radius_mm),
        "family_id": str(family_id),
    }
    return output, report


def rotate_for_cut(frame: pd.DataFrame, *, cut_idx: int) -> pd.DataFrame:
    """Rotate a cyclic path while retaining immutable physical angle indices."""
    if "angle_idx" not in frame:
        raise ValueError("cut rotation requires angle_idx")
    ordered = frame.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    matches = np.flatnonzero(ordered["angle_idx"].to_numpy(dtype=np.int64) == int(cut_idx))
    if len(matches) != 1:
        raise ValueError(f"cut_idx {int(cut_idx)} must identify exactly one angle")
    position = int(matches[0])
    rotated = pd.concat([ordered.iloc[position:], ordered.iloc[:position]], ignore_index=True).copy()
    rotated["solver_order_idx"] = np.arange(len(rotated), dtype=np.int64)
    return rotated


def restore_angle_order(frame: pd.DataFrame) -> pd.DataFrame:
    if "angle_idx" not in frame:
        raise ValueError("angle-order restoration requires angle_idx")
    return frame.sort_values("angle_idx", kind="stable").reset_index(drop=True)


def cyclic_beta_delta_rms_deg(beta_rad: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    if len(beta) == 0:
        return np.asarray([], dtype=float)
    delta = np.roll(beta, -1, axis=0) - beta
    return np.sqrt(np.mean(np.square(delta), axis=1)) * 180.0 / math.pi


def anchor_stages(schedule: str) -> tuple[dict[str, float | str], ...]:
    """Return V5-compatible cyclic stages while retaining radial anchoring."""
    name = str(schedule).strip().lower()
    if name not in ANCHOR_SCHEDULES:
        raise ValueError(f"unsupported radial anchor schedule: {schedule}")
    anchors = ANCHOR_SCHEDULES[name]
    velocity = (0.1, 1.0, 2.0)
    acceleration = (0.0, 0.25, 0.5)
    posture = (0.0, 0.01, 0.05)
    return tuple(
        {
            "name": f"{name}_{index + 1}",
            "lambda_velocity": float(velocity[index]),
            "lambda_acceleration": float(acceleration[index]),
            "lambda_anchor": float(anchors[index]),
            "lambda_posture": float(posture[index]),
        }
        for index in range(3)
    )


def _path_beta_by_angle(path: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    required = {"angle_idx", *atlas.BETA_COLS}
    missing = sorted(required - set(path.columns))
    if missing:
        raise ValueError(f"cut-invariance path missing columns: {missing}")
    ordered = restore_angle_order(path)
    indices = ordered["angle_idx"].astype(int).tolist()
    if len(indices) != len(set(indices)):
        raise ValueError("cut-invariance path contains duplicate angle_idx values")
    return indices, ordered[atlas.BETA_COLS].to_numpy(dtype=float)


def cut_invariance_report(
    paths: Mapping[tuple[int, str], pd.DataFrame],
    *,
    required_cuts: Sequence[int] = FORMAL_CUT_INDICES,
    required_predictors: Sequence[str] = ("parent_copy", "radial_secant"),
    threshold_deg: float = 1.0,
) -> dict[str, Any]:
    required = [(int(cut), str(predictor)) for cut in required_cuts for predictor in required_predictors]
    missing = [f"{cut}:{predictor}" for cut, predictor in required if (cut, predictor) not in paths]
    pairwise: list[dict[str, Any]] = []
    ordered_paths: dict[tuple[int, str], tuple[list[int], np.ndarray]] = {
        key: _path_beta_by_angle(paths[key]) for key in required if key in paths
    }
    keys = list(ordered_paths)
    for left_idx, left_key in enumerate(keys):
        left_angles, left_beta = ordered_paths[left_key]
        for right_key in keys[left_idx + 1 :]:
            right_angles, right_beta = ordered_paths[right_key]
            if left_angles != right_angles:
                raise ValueError("cut-invariance comparison requires identical angle_idx values")
            diff = np.sqrt(np.mean(np.square(left_beta - right_beta), axis=1)) * 180.0 / math.pi
            pairwise.append(
                {
                    "left": f"{left_key[0]}:{left_key[1]}",
                    "right": f"{right_key[0]}:{right_key[1]}",
                    "branch_diff_mean_deg": float(np.mean(diff)),
                    "branch_diff_p95_deg": float(np.percentile(diff, 95)),
                    "branch_diff_max_deg": float(np.max(diff)),
                }
            )
    p95_max = max((row["branch_diff_p95_deg"] for row in pairwise), default=float("inf"))
    return {
        "required_run_count": int(len(required)),
        "materialized_run_count": int(len(ordered_paths)),
        "missing_runs": missing,
        "pairwise_comparison_count": int(len(pairwise)),
        "pairwise_branch_diff_p95_max_deg": float(p95_max),
        "pairwise": pairwise,
        "cut_invariance_gate_pass": bool(not missing and pairwise and p95_max <= float(threshold_deg)),
    }


def _validate_solved_path(
    path: pd.DataFrame,
    *,
    family_id: str,
    radius_mm: float,
) -> pd.DataFrame:
    return _aligned_fixed_family_path(
        path,
        family_id=str(family_id),
        expected_radius_mm=float(radius_mm),
    )


def adaptive_radius_walk(
    initial_path: pd.DataFrame,
    *,
    start_radius_mm: float,
    checkpoints_mm: Sequence[float],
    family_id: str,
    solve_step: Callable[[pd.DataFrame, float, float], tuple[pd.DataFrame, Mapping[str, Any]]],
    base_step_mm: float = 1.0,
    retry_steps_mm: Sequence[float] = (0.5, 0.25),
) -> tuple[dict[float, pd.DataFrame], dict[str, Any]]:
    """Walk a fixed branch radially, using bounded half/quarter-step recovery."""
    if float(base_step_mm) <= 0.0:
        raise ValueError("base_step_mm must be positive")
    retries = tuple(float(step) for step in retry_steps_mm)
    if any(step <= 0.0 for step in retries) or any(
        left <= right for left, right in zip(retries, retries[1:])
    ):
        raise ValueError("retry_steps_mm must be strictly decreasing and positive")
    checkpoints = [float(value) for value in checkpoints_mm]
    if any(value <= float(start_radius_mm) for value in checkpoints) or checkpoints != sorted(set(checkpoints)):
        raise ValueError("radius checkpoints must be unique, sorted, and above the start radius")

    current_radius = float(start_radius_mm)
    current_path = _validate_solved_path(
        initial_path,
        family_id=str(family_id),
        radius_mm=current_radius,
    )
    passed: dict[float, pd.DataFrame] = {current_radius: current_path}
    attempts: list[dict[str, Any]] = []

    def execute(target_radius: float, step_kind: str) -> bool:
        nonlocal current_path, current_radius
        solved_path, raw_report = solve_step(current_path.copy(), current_radius, float(target_radius))
        solved = _validate_solved_path(
            solved_path,
            family_id=str(family_id),
            radius_mm=float(target_radius),
        )
        gate = bool(raw_report.get("radial_bundle_gate_pass", False))
        attempts.append(
            {
                "parent_radius_mm": float(current_radius),
                "target_radius_mm": float(target_radius),
                "step_mm": float(target_radius - current_radius),
                "step_kind": str(step_kind),
                "radial_bundle_gate_pass": gate,
                "report": dict(raw_report),
            }
        )
        if gate:
            current_radius = float(target_radius)
            current_path = solved
            passed[current_radius] = current_path
        return gate

    for checkpoint in checkpoints:
        while current_radius < checkpoint - 1.0e-10:
            nominal = min(checkpoint, current_radius + float(base_step_mm))
            if execute(nominal, "base"):
                continue
            failed_delta = nominal - current_radius
            recovered = False
            for retry in retries:
                if retry >= failed_delta - 1.0e-12:
                    continue
                retry_target = min(checkpoint, current_radius + retry)
                if execute(retry_target, f"retry_{retry:g}mm"):
                    recovered = True
                    break
            if not recovered:
                return passed, {
                    "radial_walk_gate_pass": False,
                    "family_id": str(family_id),
                    "last_pass_radius_mm": float(current_radius),
                    "failed_checkpoint_mm": float(checkpoint),
                    "attempts": attempts,
                }
    return passed, {
        "radial_walk_gate_pass": True,
        "family_id": str(family_id),
        "last_pass_radius_mm": float(current_radius),
        "failed_checkpoint_mm": None,
        "attempts": attempts,
    }


def assign_whole_radius_splits(
    dataset: pd.DataFrame,
    *,
    validation_radius_mm: float = VALIDATION_RADIUS_MM,
    test_radius_mm: float = TEST_RADIUS_MM,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"family_id", "trajectory_id", "radius_mm"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"dataset split missing columns: {missing}")
    assigned = dataset.copy()
    radius = assigned["radius_mm"].to_numpy(dtype=float)
    assigned["split"] = "train"
    assigned.loc[np.isclose(radius, float(validation_radius_mm), atol=1.0e-8), "split"] = "validation"
    assigned.loc[np.isclose(radius, float(test_radius_mm), atol=1.0e-8), "split"] = "test"
    trajectory_split_counts = assigned.groupby("trajectory_id")["split"].nunique()
    radius_split_counts = assigned.groupby("radius_mm")["split"].nunique()
    family_count = int(assigned["family_id"].astype(str).nunique())
    validation_present = bool(np.any(np.isclose(radius, float(validation_radius_mm), atol=1.0e-8)))
    test_present = bool(np.any(np.isclose(radius, float(test_radius_mm), atol=1.0e-8)))
    formal_radii_present = bool(
        len(np.unique(radius)) == len(FORMAL_RADII_MM)
        and np.allclose(sorted(np.unique(radius)), FORMAL_RADII_MM, atol=1.0e-8, rtol=0.0)
    )
    report = {
        "rows": int(len(assigned)),
        "unique_family_count": family_count,
        "validation_radius_mm": float(validation_radius_mm),
        "test_radius_mm": float(test_radius_mm),
        "validation_radius_present": validation_present,
        "test_radius_present": test_present,
        "formal_radii_present": formal_radii_present,
        "trajectory_leakage_count": int((trajectory_split_counts > 1).sum()),
        "radius_leakage_count": int((radius_split_counts > 1).sum()),
    }
    report["whole_radius_split_gate_pass"] = bool(
        len(assigned)
        and family_count == 1
        and validation_present
        and test_present
        and formal_radii_present
        and report["trajectory_leakage_count"] == 0
        and report["radius_leakage_count"] == 0
    )
    return assigned, report


def training_only_support_pool(dataset_with_split: pd.DataFrame) -> pd.DataFrame:
    if "split" not in dataset_with_split:
        raise ValueError("training-only support requires a materialized split column")
    mask = dataset_with_split["split"].astype(str).eq("train")
    if "is_centerline" in dataset_with_split:
        mask &= ~dataset_with_split["is_centerline"].astype(bool)
    training = dataset_with_split[mask].copy()
    if training.empty:
        raise ValueError("training-only support pool is empty")
    if np.any(np.isclose(training["radius_mm"].to_numpy(dtype=float), VALIDATION_RADIUS_MM, atol=1.0e-8)):
        raise ValueError("validation radius leaked into the training-only support pool")
    if np.any(np.isclose(training["radius_mm"].to_numpy(dtype=float), TEST_RADIUS_MM, atol=1.0e-8)):
        raise ValueError("test radius leaked into the training-only support pool")
    return training.reset_index(drop=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fingerprint(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def radial_task_fingerprint(
    *,
    family_id: str,
    target_radius_mm: float,
    parent_paths: Sequence[str | Path],
    anchor_schedule: str,
    robot_config_path: str | Path,
    strategy_version: int = RADIAL_BUNDLE_STRATEGY_VERSION,
    extra_protocol: Mapping[str, Any] | None = None,
) -> str:
    if str(anchor_schedule) not in ANCHOR_SCHEDULES:
        raise ValueError(f"unsupported radial anchor schedule: {anchor_schedule}")
    parents = [Path(path) for path in parent_paths]
    if not parents:
        raise ValueError("radial task fingerprint requires at least one parent path")
    config = Path(robot_config_path)
    payload = {
        "strategy_version": int(strategy_version),
        "solver_strategy_version": SOLVER_STRATEGY_VERSION,
        "family_id": str(family_id),
        "target_radius_mm": float(target_radius_mm),
        "parent_paths": [
            {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": int(path.stat().st_size)}
            for path in parents
        ],
        "anchor_schedule": str(anchor_schedule),
        "anchor_values": list(ANCHOR_SCHEDULES[str(anchor_schedule)]),
        "robot_config": {
            "path": str(config.resolve()),
            "sha256": file_sha256(config),
            "bytes": int(config.stat().st_size),
        },
        "extra_protocol": dict(extra_protocol or {}),
    }
    return _stable_fingerprint(payload)


def artifact_hash_manifest(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        manifest[str(path)] = {
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
    return manifest


def artifact_manifests_match(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> bool:
    return _stable_fingerprint(dict(before)) == _stable_fingerprint(dict(after))


def correct_radial_predictor(
    targets: pd.DataFrame,
    predictor: pd.DataFrame,
    *,
    cut_idx: int,
    anchor_schedule: str,
    bounds: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int,
    compute_conditioning: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Jointly correct all phases, with cyclic seam terms and radial anchors."""
    required_targets = {"angle_idx", *atlas.TARGET_XYZ_COLS}
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_targets:
        raise ValueError(f"radial correction targets missing columns: {missing_targets}")
    required_predictor = {"angle_idx", *atlas.BETA_COLS}
    missing_predictor = sorted(required_predictor - set(predictor.columns))
    if missing_predictor:
        raise ValueError(f"radial predictor missing columns: {missing_predictor}")
    target_ordered = restore_angle_order(targets)
    predictor_ordered = restore_angle_order(predictor)
    if target_ordered["angle_idx"].astype(int).tolist() != predictor_ordered["angle_idx"].astype(int).tolist():
        raise ValueError("radial correction requires target and predictor angle alignment")
    for column in ("family_id", "radius_mm"):
        if column in target_ordered and column in predictor_ordered:
            left = target_ordered[column].to_numpy()
            right = predictor_ordered[column].to_numpy()
            if column == "family_id":
                same = np.array_equal(left.astype(str), right.astype(str))
            else:
                same = np.allclose(left.astype(float), right.astype(float), atol=1.0e-8, rtol=0.0)
            if not same:
                raise ValueError(f"radial correction {column} mismatch")

    rotated_targets = rotate_for_cut(target_ordered, cut_idx=int(cut_idx))
    rotated_predictor = rotate_for_cut(predictor_ordered, cut_idx=int(cut_idx))
    stage_specs = anchor_stages(str(anchor_schedule))
    corrected, raw_report = atlas.optimize_cyclic_trajectory(
        targets=rotated_targets,
        initial_beta=rotated_predictor[atlas.BETA_COLS].to_numpy(dtype=float),
        bounds=np.asarray(bounds, dtype=float),
        lengths_m=np.asarray(lengths_m, dtype=float),
        p_end_local_m=np.asarray(p_end_local_m, dtype=float),
        theta_sign=float(theta_sign),
        stages=stage_specs,
        max_nfev=int(max_nfev),
        compute_conditioning=bool(compute_conditioning),
        stop_on_centerline_gate=True,
    )
    restored = restore_angle_order(corrected)
    predictor_type = (
        str(predictor_ordered["radial_predictor_type"].iloc[0])
        if "radial_predictor_type" in predictor_ordered
        else "unknown"
    )
    restored["radial_predictor_type"] = predictor_type
    restored["angle_cut_idx"] = int(cut_idx)
    restored["anchor_schedule"] = str(anchor_schedule)
    restored["solver_strategy_version"] = SOLVER_STRATEGY_VERSION
    report = dict(raw_report)
    report.update(
        {
            "angle_cut_idx": int(cut_idx),
            "anchor_schedule": str(anchor_schedule),
            "radial_predictor_type": predictor_type,
            "rows": int(len(restored)),
            "full_predictor_path_used": bool(
                len(rotated_predictor) == len(predictor_ordered)
                and len(rotated_predictor) == len(rotated_targets)
            ),
            "radial_anchor_retained": bool(
                stage_specs and all(float(stage["lambda_anchor"]) > 0.0 for stage in stage_specs)
            ),
            "input_predictor_branch_hash": hashlib.sha256(
                predictor_ordered[atlas.BETA_COLS].to_numpy(dtype=float).tobytes()
            ).hexdigest(),
            "output_branch_hash": atlas.branch_hash(restored),
        }
    )
    return restored, report


def filter_candidate_layers(
    candidates: pd.DataFrame,
    *,
    expected_angle_indices: Sequence[int],
    residual_limit_mm: float = 2.0,
    cluster_threshold_deg: float = 0.25,
    max_candidates_per_angle: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"angle_idx", "xyz_residual_mm", *atlas.BETA_COLS}
    missing_columns = sorted(required - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"radial candidates missing columns: {missing_columns}")
    expected = [int(value) for value in expected_angle_indices]
    kept: list[pd.DataFrame] = []
    missing_angles: list[int] = []
    layer_counts: dict[int, int] = {}
    for angle_idx in expected:
        layer = candidates[
            candidates["angle_idx"].astype(int).eq(angle_idx)
            & candidates["xyz_residual_mm"].astype(float).le(float(residual_limit_mm))
        ].copy()
        if layer.empty:
            missing_angles.append(angle_idx)
            layer_counts[angle_idx] = 0
            continue
        clustered = atlas.cluster_beta_candidates(
            layer,
            beta_rms_threshold_deg=float(cluster_threshold_deg),
            max_clusters=int(max_candidates_per_angle),
        ).head(int(max_candidates_per_angle))
        clustered["candidate_layer_rank"] = np.arange(len(clustered), dtype=np.int64)
        layer_counts[angle_idx] = int(len(clustered))
        kept.append(clustered)
    output = pd.concat(kept, ignore_index=True, sort=False) if kept else candidates.iloc[0:0].copy()
    report = {
        "input_candidate_rows": int(len(candidates)),
        "candidate_rows": int(len(output)),
        "expected_angle_count": int(len(expected)),
        "materialized_angle_count": int(output["angle_idx"].nunique()) if len(output) else 0,
        "missing_angle_indices": missing_angles,
        "residual_limit_mm": float(residual_limit_mm),
        "cluster_threshold_deg": float(cluster_threshold_deg),
        "max_candidates_per_angle": int(max_candidates_per_angle),
        "layer_counts": layer_counts,
        "candidate_layer_gate_pass": bool(not missing_angles and len(output)),
    }
    return output, report


def exact_repeatability_report(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    left_angles, left_beta = _path_beta_by_angle(left)
    right_angles, right_beta = _path_beta_by_angle(right)
    same_angles = left_angles == right_angles
    exact = bool(same_angles and left_beta.shape == right_beta.shape and np.array_equal(left_beta, right_beta))
    max_abs = (
        float(np.max(np.abs(left_beta - right_beta)))
        if same_angles and left_beta.shape == right_beta.shape and left_beta.size
        else float("inf")
    )
    return {
        "rows": int(len(left_beta)) if same_angles else 0,
        "same_angle_indices": bool(same_angles),
        "max_abs_beta_diff_rad": max_abs,
        "left_branch_hash": atlas.branch_hash(left),
        "right_branch_hash": atlas.branch_hash(right),
        "deterministic_exact_gate_pass": exact,
    }


def repeat_input_for_strategy(
    direct_predictor: pd.DataFrame,
    *,
    selected_strategy: str,
    candidate_graph_predictor: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the exact initializer used by the selected deterministic solver path."""
    strategy = str(selected_strategy)
    if strategy == "direct_joint_corrector":
        return direct_predictor.copy()
    if strategy == "candidate_graph_joint_corrector":
        if candidate_graph_predictor is None:
            raise ValueError("candidate-graph repeat requires its selected graph predictor")
        return candidate_graph_predictor.copy()
    raise ValueError(f"unsupported selected solver strategy: {strategy}")


def generate_radial_candidate_layers(
    targets: pd.DataFrame,
    predictor: pd.DataFrame,
    *,
    bounds: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int,
    pointwise_candidates: pd.DataFrame | None = None,
    max_seed_count: int = 8,
    residual_limit_mm: float = 2.0,
    cluster_threshold_deg: float = 0.25,
    max_candidates_per_angle: int = 8,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate bounded per-angle IK candidates around a radial predictor."""
    target_ordered = restore_angle_order(targets)
    predictor_ordered = restore_angle_order(predictor)
    if target_ordered["angle_idx"].astype(int).tolist() != predictor_ordered["angle_idx"].astype(int).tolist():
        raise ValueError("candidate generation requires target/predictor angle alignment")
    limits = np.asarray(bounds, dtype=float).reshape(6, 2)
    pointwise = pointwise_candidates if pointwise_candidates is not None else pd.DataFrame()
    records: list[dict[str, Any]] = []
    for position, target_row in target_ordered.iterrows():
        angle_idx = int(target_row["angle_idx"])
        prediction = predictor_ordered.iloc[position][atlas.BETA_COLS].to_numpy(dtype=float)
        jac = atlas.numerical_jacobian_beta(
            prediction,
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
        )
        null_seeds = atlas.generate_nullspace_seeds(
            prediction,
            jac,
            bounds=limits,
            scales_deg=(0.25, 0.5, 1.0),
            seeds_per_scale=2,
            seed=int(seed) + angle_idx,
        )
        seeds = [prediction]
        sources = ["radial_predictor"]
        if not pointwise.empty and {"angle_idx", *atlas.BETA_COLS}.issubset(pointwise.columns):
            layer = pointwise[pointwise["angle_idx"].astype(int).eq(angle_idx)]
            if "xyz_residual_mm" in layer:
                layer = layer.sort_values("xyz_residual_mm", kind="stable")
            for row in layer.head(2)[atlas.BETA_COLS].to_numpy(dtype=float):
                seeds.append(row)
                sources.append("pointwise_pool")
        for row in null_seeds:
            seeds.append(row)
            sources.append("nullspace")
        deduplicated: list[np.ndarray] = []
        deduplicated_sources: list[str] = []
        seen: set[bytes] = set()
        for source, row in zip(sources, seeds):
            clipped = np.clip(np.asarray(row, dtype=float).reshape(6), limits[:, 0], limits[:, 1])
            key = np.round(clipped, 12).tobytes()
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(clipped)
            deduplicated_sources.append(source)
            if len(deduplicated) >= int(max_seed_count):
                break
        solutions = atlas.solve_beta_ik_many(
            target_row[atlas.TARGET_XYZ_COLS].to_numpy(dtype=float),
            init_betas=np.asarray(deduplicated, dtype=float),
            bounds=limits,
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
            max_nfev=int(max_nfev),
            lambda_limit=0.0,
            center_beta=prediction,
            lambda_center=1.0e-3,
        )
        for solution in solutions:
            record = target_row.to_dict()
            record.update(
                {
                    "candidate_seed_rank": int(solution.seed_rank),
                    "candidate_seed_source": deduplicated_sources[int(solution.seed_rank)],
                    "xyz_residual_mm": float(solution.residual_mm),
                    "inverse_nfev": int(solution.nfev),
                }
            )
            for beta_idx, column in enumerate(atlas.BETA_COLS):
                record[column] = float(solution.beta_rad[beta_idx])
            for xyz_idx, column in enumerate(atlas.XYZ_COLS):
                record[column] = float(solution.xyz_m[xyz_idx])
            records.append(record)
    raw = pd.DataFrame(records)
    filtered, filter_report = filter_candidate_layers(
        raw,
        expected_angle_indices=target_ordered["angle_idx"].astype(int).tolist(),
        residual_limit_mm=float(residual_limit_mm),
        cluster_threshold_deg=float(cluster_threshold_deg),
        max_candidates_per_angle=int(max_candidates_per_angle),
    )
    report = {
        **filter_report,
        "raw_candidate_rows": int(len(raw)),
        "max_seed_count": int(max_seed_count),
        "candidate_generation_gate_pass": bool(filter_report["candidate_layer_gate_pass"]),
    }
    return filtered, report


def candidate_graph_rescue(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a cyclic candidate path before applying the same joint corrector."""
    selected, report = atlas.link_cyclic_branch_soft(
        candidates,
        lambda_velocity=1.0,
        closure_weight=5.0,
        lambda_posture=0.05,
        lambda_kappa=0.01,
        residual_scale_mm=2.0,
    )
    output = selected.sort_values("angle_idx", kind="stable").reset_index(drop=True) if len(selected) else selected
    report = dict(report)
    report["candidate_graph_gate_pass"] = bool(
        report.get("success", False)
        and len(output)
        and int(output["angle_idx"].nunique()) == len(output)
    )
    return output, report
