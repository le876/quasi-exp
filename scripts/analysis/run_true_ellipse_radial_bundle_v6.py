#!/usr/bin/env python3
"""Run the V6 full-parent radial path-bundle experiment.

The runner is fail-closed: it materializes a fixed-family tube dataset only
after the same branch reaches 100 mm under cut-invariance, exact rerun, and the
unchanged V3/V5 centerline and tube gates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_family_expansion_v5 as v5_expansion  # noqa: E402
import run_true_ellipse_family_training_v5 as v5_training  # noqa: E402
import true_ellipse_radial_bundle_v6_utils as v6  # noqa: E402
from true_ellipse_radial_bundle_engine import (  # noqa: E402
    JointDomainSpec,
    balanced_joint_margin_policy,
    evaluate_joint_margin_gate,
    joint_margin_report,
    registered_joint_domain,
)
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_ellipse_family_v5_utils import dataframe_to_markdown, stable_fingerprint  # noqa: E402


PRIMARY_FAMILY_ID = "c0273_a100_py210_pz330_s0243"
DEFAULT_V5_DIR = REPO_ROOT / "runs" / "true_ellipse_family_expansion_v5"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "true_ellipse_radial_bundle_v6"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml"
ALL_PHASES = ["audit", "radial", "tube", "dataset", "summary"]
RUNNER_STRATEGY_VERSION = 1
TUBE_STRATEGY_VERSION = 1
DATASET_STRATEGY_VERSION = 1
TUBE_OUTER_SHELL_STAGE = {
    "name": "tube_outer_shell_parent_joint",
    "lambda_velocity": 0.1,
    "lambda_acceleration": 0.0,
    "lambda_anchor": 1.0,
    "lambda_posture": 0.0,
}

write_json = v6.write_json
read_json = v6.read_json
_json_default = v6.json_default


def _resolved_joint_domain(args: argparse.Namespace) -> JointDomainSpec:
    return registered_joint_domain(str(getattr(args, "joint_domain_id", "current_v6")))


def _joint_margin_decision(path: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    domain = _resolved_joint_domain(args)
    policy = balanced_joint_margin_policy()
    margin = joint_margin_report(
        path[v6.atlas.BETA_COLS].to_numpy(dtype=float),
        domain=domain,
        at_bound_tolerance_deg=policy.at_bound_tolerance_deg,
    )
    raw = evaluate_joint_margin_gate(margin, policy=policy)
    required = bool(getattr(args, "require_joint_margin_gate", False))
    return {
        **margin,
        "joint_margin_policy_id": policy.policy_id,
        "joint_margin_policy_fingerprint": policy.fingerprint,
        "raw_joint_margin_gate_pass": bool(raw["joint_margin_gate_pass"]),
        "joint_margin_gate_required": required,
        "job_margin_gate_pass": bool(not required or raw["joint_margin_gate_pass"]),
        "joint_margin_checks": raw["checks"],
    }


def parse_float_csv(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(item) for item in value]


def parse_int_csv(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def parse_name_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_phases(value: str | Iterable[str]) -> list[str]:
    phases = parse_name_csv(value)
    if phases == ["all"]:
        return list(ALL_PHASES)
    unknown = sorted(set(phases) - set(ALL_PHASES))
    if unknown:
        raise ValueError(f"unsupported V6 phases: {unknown}")
    return phases


def formal_protocol_report(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = parse_float_csv(args.radius_checkpoints_mm)
    cuts = parse_int_csv(args.cut_indices)
    schedules = parse_name_csv(args.anchor_schedules)
    retry_steps = parse_float_csv(args.retry_steps_mm)
    offsets = parse_float_csv(args.tube_offsets_mm)
    checks = {
        "primary_fixed_family": str(args.family_id) == PRIMARY_FAMILY_ID,
        "previous_radius_75": np.isclose(float(args.previous_radius_mm), 75.0),
        "start_radius_80": np.isclose(float(args.start_radius_mm), 80.0),
        "target_radius_100": np.isclose(float(args.target_radius_mm), 100.0),
        "formal_radius_checkpoints": bool(
            len(checkpoints) == 8
            and np.allclose(checkpoints, (82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0))
        ),
        "formal_cut_indices": cuts == list(v6.FORMAL_CUT_INDICES),
        "all_anchor_schedules": schedules == list(v6.ANCHOR_SCHEDULES),
        "base_step_1mm": np.isclose(float(args.base_step_mm), 1.0),
        "retry_steps_half_quarter": retry_steps == [0.5, 0.25],
        "final_points_360": int(args.final_points) == 360,
        "max_candidates_eight": int(args.max_candidates_per_angle) == 8,
        "candidate_cluster_quarter_degree": np.isclose(float(args.candidate_cluster_deg), 0.25),
        "candidate_residual_two_mm": np.isclose(float(args.candidate_residual_mm), 2.0),
        "formal_tube_offsets": bool(
            len(offsets) == 5
            and len(set(offsets)) == 5
            and np.allclose(sorted(offsets), v6.FORMAL_TUBE_OFFSETS_MM)
        ),
        "max_opt_nfev_40": int(args.max_opt_nfev) == 40,
        "max_ik_nfev_200": int(args.max_ik_nfev) == 200,
        "family_search_8192_64_16_8": bool(
            int(args.family_search_sobol_samples) == 8192
            and int(args.family_search_top_support) == 64
            and int(args.family_search_top_pointwise) == 16
            and int(args.family_search_top_formal) == 8
        ),
        "formal_seed": int(args.seed) == 20260715,
    }
    normalized = {key: bool(value) for key, value in checks.items()}
    protocol = {
        "runner_strategy_version": RUNNER_STRATEGY_VERSION,
        "solver_strategy_version": v6.SOLVER_STRATEGY_VERSION,
        "family_id": str(args.family_id),
        "previous_radius_mm": float(args.previous_radius_mm),
        "start_radius_mm": float(args.start_radius_mm),
        "target_radius_mm": float(args.target_radius_mm),
        "radius_checkpoints_mm": checkpoints,
        "cut_indices": cuts,
        "anchor_schedules": schedules,
        "base_step_mm": float(args.base_step_mm),
        "retry_steps_mm": retry_steps,
        "final_points": int(args.final_points),
        "max_candidates_per_angle": int(args.max_candidates_per_angle),
        "candidate_cluster_deg": float(args.candidate_cluster_deg),
        "candidate_residual_mm": float(args.candidate_residual_mm),
        "tube_offsets_mm": offsets,
        "max_opt_nfev": int(args.max_opt_nfev),
        "max_ik_nfev": int(args.max_ik_nfev),
        "family_search_sobol_samples": int(args.family_search_sobol_samples),
        "family_search_top_support": int(args.family_search_top_support),
        "family_search_top_pointwise": int(args.family_search_top_pointwise),
        "family_search_top_formal": int(args.family_search_top_formal),
        "seed": int(args.seed),
    }
    return {
        "formal_protocol_gate_pass": bool(all(normalized.values())),
        "checks": normalized,
        "protocol": protocol,
        "protocol_fingerprint": stable_fingerprint(protocol),
    }


def aggregate_radius_bundle_gate(
    *,
    geometry_report: Mapping[str, Any],
    job_reports: Sequence[Mapping[str, Any]],
    required_cuts: Sequence[int],
    required_predictors: Sequence[str],
    cut_report: Mapping[str, Any],
    repeatability_report: Mapping[str, Any],
) -> dict[str, Any]:
    required = [(int(cut), str(predictor)) for cut in required_cuts for predictor in required_predictors]
    selected: dict[tuple[int, str], Mapping[str, Any]] = {}
    for report in job_reports:
        key = (int(report.get("cut_idx", -1)), str(report.get("radial_predictor_type", "")))
        if bool(report.get("selected", False)):
            selected[key] = report
    missing = [f"{cut}:{predictor}" for cut, predictor in required if (cut, predictor) not in selected]
    failed = [
        f"{cut}:{predictor}"
        for cut, predictor in required
        if (cut, predictor) in selected
        and not bool(
            selected[(cut, predictor)].get(
                "job_gate_pass",
                selected[(cut, predictor)].get("centerline_gate_pass", False),
            )
        )
    ]
    report = {
        "required_job_count": int(len(required)),
        "selected_job_count": int(len(selected)),
        "missing_jobs": missing,
        "failed_jobs": failed,
        "target_geometry_gate_pass": bool(geometry_report.get("target_geometry_gate_pass", False)),
        "all_job_centerline_gates_pass": bool(not missing and not failed),
        "cut_invariance_gate_pass": bool(cut_report.get("cut_invariance_gate_pass", False)),
        "deterministic_exact_gate_pass": bool(
            repeatability_report.get("deterministic_exact_gate_pass", False)
        ),
    }
    report["radial_bundle_gate_pass"] = bool(
        report["target_geometry_gate_pass"]
        and report["all_job_centerline_gates_pass"]
        and report["cut_invariance_gate_pass"]
        and report["deterministic_exact_gate_pass"]
    )
    return report


def _required_job_failed(report: Mapping[str, Any]) -> bool:
    return bool(
        not report.get("selected", False)
        or not report.get(
            "job_gate_pass",
            report.get("centerline_gate_pass", False),
        )
    )


def annotate_tube_rows(
    tube: pd.DataFrame,
    *,
    family_id: str,
    radius_mm: float,
    parent_radius_mm: float | None,
    radial_predictor_type: str,
    branch_hash: str,
) -> pd.DataFrame:
    required = {
        "angle_idx",
        "tube_offset_id",
        "is_centerline",
        *v6.atlas.TARGET_XYZ_COLS,
        *v6.atlas.BETA_COLS,
    }
    missing = sorted(required - set(tube.columns))
    if missing:
        raise ValueError(f"V6 tube annotation missing columns: {missing}")
    output = tube.copy()
    family = str(family_id)
    trajectory_id = f"{family}@r{float(radius_mm):g}"
    output["family_id"] = family
    output["branch_id"] = f"{family}:radial_bundle_v6"
    output["trajectory_id"] = trajectory_id
    output["ellipse_id"] = trajectory_id
    output["radius_mm"] = float(radius_mm)
    output["parent_radius_mm"] = (
        float(parent_radius_mm) if parent_radius_mm is not None else np.nan
    )
    output["solver_strategy_version"] = v6.SOLVER_STRATEGY_VERSION
    output["radial_predictor_type"] = str(radial_predictor_type)
    output["branch_hash"] = str(branch_hash)
    output["sample_id"] = [
        f"{trajectory_id}:{int(angle_idx):03d}:{offset_id}"
        for angle_idx, offset_id in zip(output["angle_idx"], output["tube_offset_id"])
    ]
    if not output["sample_id"].is_unique:
        raise ValueError("V6 tube annotation produced duplicate sample_id values")
    return output


def radius_slug(radius_mm: float) -> str:
    return f"r{float(radius_mm):06.2f}".replace(".", "p")


def load_family_spec(v5_dir: str | Path, family_id: str) -> v6.FamilySpec:
    selected_path = Path(v5_dir) / "02_pointwise" / "selected_families.csv"
    selected = pd.read_csv(selected_path)
    rows = selected[selected["candidate_id"].astype(str).eq(str(family_id))]
    if len(rows) != 1:
        raise ValueError(f"V5 selected_families must contain exactly one row for {family_id}")
    row = rows.iloc[0]
    return v6.FamilySpec(
        family_id=str(family_id),
        center_x_m=float(row["center_x_m"]),
        center_y_m=float(row["center_y_m"]),
        center_z_m=float(row["center_z_m"]),
        phase_y_rad=float(row["phase_y_rad"]),
        phase_z_rad=float(row["phase_z_rad"]),
    )


def v5_centerline_path(v5_dir: str | Path, family_id: str, radius_mm: float) -> Path:
    return (
        Path(v5_dir)
        / "03_branch"
        / str(family_id)
        / radius_slug(float(radius_mm))
        / "selected_centerline_360.parquet"
    )


def load_v5_parent_path(v5_dir: str | Path, family_id: str, radius_mm: float) -> pd.DataFrame:
    path = v5_centerline_path(v5_dir, family_id, radius_mm)
    frame = pd.read_parquet(path).sort_values("angle_idx", kind="stable").reset_index(drop=True)
    frame["family_id"] = str(family_id)
    frame["radius_mm"] = float(radius_mm)
    frame["parent_radius_mm"] = np.nan
    frame["solver_strategy_version"] = "true-ellipse-family-v5-import"
    return frame


def _load_robot(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    config = load_config(str(args.robot_config))
    inputs = load_robot_inputs(config)
    theta_sign = float(config.get("kinematics", {}).get("theta_sign", -1.0))
    return inputs.lengths_m, inputs.p_end_local_m, theta_sign


def _critical_legacy_artifact_paths(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    v5_dir = Path(args.v5_dir)
    family_id = str(args.family_id)
    paths: list[Path] = [
        v5_dir / "00_audit" / "audit_report.json",
        v5_dir / "02_pointwise" / "selected_families.csv",
        v5_dir / "02_pointwise" / "pointwise_radius_summary.csv",
        v5_dir / "02_pointwise" / family_id / radius_slug(float(args.target_radius_mm)) / "targets.parquet",
        v5_dir
        / "02_pointwise"
        / family_id
        / radius_slug(float(args.target_radius_mm))
        / "pointwise_candidates.parquet",
        v5_dir / "02_pointwise" / family_id / radius_slug(float(args.target_radius_mm)) / "pointwise_report.json",
        v5_dir / "03_branch" / "branch_radius_summary.csv",
        v5_dir / "04_tube" / "tube_radius_summary.csv",
        v5_dir / "05_dataset" / "dataset_report.json",
        v5_dir / "06_summary" / "expansion_summary.json",
        v5_dir / "run_report.json",
        Path(args.robot_config),
    ]
    for radius_mm in (float(args.previous_radius_mm), float(args.start_radius_mm)):
        radius_dir = v5_dir / "03_branch" / family_id / radius_slug(radius_mm)
        paths.extend(
            [
                radius_dir / "selected_centerline_360.parquet",
                radius_dir / "branch_radius_report.json",
            ]
        )
    audit_path = v5_dir / "00_audit" / "audit_report.json"
    if audit_path.exists():
        source_report = read_json(audit_path)
        paths.extend(Path(value) for value in source_report.get("sources", {}).values())
    runs_root = v5_dir.parent
    paths.extend(
        [
            runs_root / "true_ellipse_beta6_training_v4" / "06_summary" / "final_report.json",
            runs_root / "true_ellipse_family_training_v5" / "00_audit" / "audit_report.json",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    missing = [str(path) for path in unique if not path.is_file()]
    return [path for path in unique if path.is_file()], missing


def _audit_parent_path(
    args: argparse.Namespace,
    *,
    family: v6.FamilySpec,
    radius_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = v5_centerline_path(args.v5_dir, family.family_id, radius_mm)
    report_path = path.with_name("branch_radius_report.json")
    centerline = load_v5_parent_path(args.v5_dir, family.family_id, radius_mm)
    target = v6.generate_radius_targets(family, radius_mm=float(radius_mm), n_points=int(args.final_points))
    source_target = centerline[v6.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
    expected_target = target[v6.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
    target_diff_mm = np.linalg.norm(source_target - expected_target, axis=1) * 1000.0
    smoothness = v6.atlas.smoothness_report(centerline)
    smoothness.update(v6.atlas.evaluate_centerline_gates(smoothness))
    stored = read_json(report_path)
    robustness = v5_expansion.evaluate_branch_robustness(
        centerline_report=smoothness,
        forward_reverse=stored.get("forward_reverse", {}),
        repeatability=stored.get("deterministic_repeatability", {}),
    )
    audit = {
        "radius_mm": float(radius_mm),
        "path": str(path.resolve()),
        "sha256": v6.file_sha256(path),
        "rows": int(len(centerline)),
        "angle_count": int(centerline["angle_idx"].nunique()),
        "target_diff_max_mm": float(np.max(target_diff_mm)),
        "source_branch_report_path": str(report_path.resolve()),
        "source_branch_report_sha256": v6.file_sha256(report_path),
        "recomputed_centerline_gate_pass": bool(smoothness["centerline_gate_pass"]),
        "recomputed_branch_robustness_gate_pass": bool(robustness["branch_robustness_gate_pass"]),
    }
    audit["parent_path_gate_pass"] = bool(
        len(centerline) == int(args.final_points)
        and audit["angle_count"] == int(args.final_points)
        and audit["target_diff_max_mm"] <= 1.0e-6
        and audit["recomputed_centerline_gate_pass"]
        and audit["recomputed_branch_robustness_gate_pass"]
    )
    return centerline, audit


def recompute_pointwise_certificate(
    source_targets: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    family: v6.FamilySpec,
    radius_mm: float,
) -> dict[str, Any]:
    """Recheck a V5 coarse certificate at its own registered resolution."""
    ordered = source_targets.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    resolution = int(len(ordered))
    expected = v6.generate_radius_targets(
        family,
        radius_mm=float(radius_mm),
        n_points=resolution,
    )
    if resolution:
        delta_mm = np.linalg.norm(
            ordered[v6.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
            - expected[v6.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float),
            axis=1,
        ) * 1000.0
        target_diff_max_mm = float(np.max(delta_mm))
    else:
        target_diff_max_mm = float("inf")
    pointwise = v5_expansion.summarize_pointwise_candidates(
        candidates,
        target_count=resolution,
    )
    pointwise.update(
        {
            "certificate_resolution_points": resolution,
            "registered_target_diff_max_mm": target_diff_max_mm,
            "registered_target_match_gate_pass": bool(
                resolution > 0 and target_diff_max_mm <= 1.0e-6
            ),
        }
    )
    pointwise["pointwise_certificate_gate_pass"] = bool(
        pointwise["pointwise_gate_pass"] and pointwise["registered_target_match_gate_pass"]
    )
    return pointwise


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_audit"
    out.mkdir(parents=True, exist_ok=True)
    protocol = formal_protocol_report(args)
    family = load_family_spec(args.v5_dir, args.family_id)
    target = v6.generate_radius_targets(
        family,
        radius_mm=float(args.target_radius_mm),
        n_points=int(args.final_points),
    )
    geometry = v6.validate_target_geometry(
        target,
        family=family,
        radius_mm=float(args.target_radius_mm),
    )
    pointwise_dir = (
        Path(args.v5_dir)
        / "02_pointwise"
        / str(args.family_id)
        / radius_slug(float(args.target_radius_mm))
    )
    pointwise_candidates_path = pointwise_dir / "pointwise_candidates.parquet"
    pointwise_candidates = pd.read_parquet(pointwise_candidates_path)
    source_targets = pd.read_parquet(pointwise_dir / "targets.parquet")
    pointwise = recompute_pointwise_certificate(
        source_targets,
        pointwise_candidates,
        family=family,
        radius_mm=float(args.target_radius_mm),
    )
    previous_parent, previous_audit = _audit_parent_path(
        args,
        family=family,
        radius_mm=float(args.previous_radius_mm),
    )
    start_parent, start_audit = _audit_parent_path(
        args,
        family=family,
        radius_mm=float(args.start_radius_mm),
    )
    del previous_parent, start_parent
    artifact_paths, missing_artifacts = _critical_legacy_artifact_paths(args)
    legacy_manifest = v6.artifact_hash_manifest(artifact_paths)
    write_json(out / "legacy_v2_v5_artifact_manifest.json", legacy_manifest)
    checks = {
        "target_geometry": bool(geometry["target_geometry_gate_pass"]),
        "v5_100mm_pointwise": bool(pointwise["pointwise_gate_pass"]),
        "pointwise_target_matches_registered_family": bool(
            pointwise["registered_target_match_gate_pass"]
        ),
        "previous_parent_robust": bool(previous_audit["parent_path_gate_pass"]),
        "start_parent_robust": bool(start_audit["parent_path_gate_pass"]),
        "legacy_artifacts_complete": not missing_artifacts,
        "robot_config_present": Path(args.robot_config).is_file(),
    }
    audit_gate = bool(all(checks.values()))
    report = {
        "strategy_version": RUNNER_STRATEGY_VERSION,
        **protocol,
        "family": family.__dict__,
        "geometry": geometry,
        "pointwise_100mm": pointwise,
        "pointwise_candidates_path": str(pointwise_candidates_path.resolve()),
        "pointwise_candidates_sha256": v6.file_sha256(pointwise_candidates_path),
        "pointwise_target_diff_max_mm": float(pointwise["registered_target_diff_max_mm"]),
        "previous_parent": previous_audit,
        "start_parent": start_audit,
        "checks_recomputed": checks,
        "missing_legacy_artifacts": missing_artifacts,
        "legacy_artifact_count": int(len(legacy_manifest)),
        "legacy_artifact_manifest_path": str((out / "legacy_v2_v5_artifact_manifest.json").resolve()),
        "legacy_artifact_manifest_fingerprint": stable_fingerprint(legacy_manifest),
        "audit_gate_pass": audit_gate,
        "formal_audit_gate_pass": bool(audit_gate and protocol["formal_protocol_gate_pass"]),
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "protocol": protocol["protocol"],
            "legacy_artifacts": legacy_manifest,
            "family": family.__dict__,
        }
    )
    write_json(out / "audit_report.json", report)
    return report


def ensure_audit_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "00_audit" / "audit_report.json"
    if path.exists():
        cached = read_json(path)
        current_protocol = formal_protocol_report(args)
        if str(cached.get("protocol_fingerprint", "")) == str(current_protocol["protocol_fingerprint"]):
            return cached
    return phase_audit(args)


def _radius_bundle_task_fingerprint(
    args: argparse.Namespace,
    *,
    audit_report: Mapping[str, Any],
    family: v6.FamilySpec,
    target_radius_mm: float,
    parent_artifact_paths: Sequence[Path],
) -> str:
    parents = [
        {
            "path": str(path.resolve()),
            "sha256": v6.file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in parent_artifact_paths
    ]
    domain_payload: dict[str, Any] = {}
    if hasattr(args, "joint_domain_id"):
        domain = _resolved_joint_domain(args)
        domain_payload = {
            "joint_domain_id": domain.domain_id,
            "joint_domain_fingerprint": domain.fingerprint,
            "require_joint_margin_gate": bool(getattr(args, "require_joint_margin_gate", False)),
            "lambda_margin": float(getattr(args, "lambda_margin", 0.0)),
            "soft_margin_deg": float(getattr(args, "soft_margin_deg", 0.25)),
            "rescue_candidates_per_angle": int(
                getattr(args, "rescue_candidates_per_angle", args.max_candidates_per_angle)
            ),
            "rescue_kappa_threshold": float(
                getattr(args, "rescue_kappa_threshold", -np.inf)
            ),
            "rescue_cache_strategy": "shared_predictor_rescue_cache_v2",
            "share_rescue_cache_across_cuts": bool(
                getattr(args, "share_rescue_cache_across_cuts", False)
            ),
            "stop_after_first_failed_job": bool(
                getattr(args, "stop_after_first_failed_job", False)
            ),
        }
    return stable_fingerprint(
        {
            "phase": "radial_radius",
            "strategy_version": RUNNER_STRATEGY_VERSION,
            "solver_strategy_version": v6.SOLVER_STRATEGY_VERSION,
            "protocol": formal_protocol_report(args)["protocol"],
            "audit_task_fingerprint": str(audit_report.get("task_fingerprint", "")),
            "family": family.__dict__,
            "target_radius_mm": float(target_radius_mm),
            "parents": parents,
            "robot_config": {
                "path": str(Path(args.robot_config).resolve()),
                "sha256": v6.file_sha256(args.robot_config),
            },
            **domain_payload,
        }
    )


def _pointwise_candidates_for_radius(
    args: argparse.Namespace,
    *,
    family_id: str,
    radius_mm: float,
) -> pd.DataFrame | None:
    path = (
        Path(args.v5_dir)
        / "02_pointwise"
        / str(family_id)
        / radius_slug(float(radius_mm))
        / "pointwise_candidates.parquet"
    )
    return pd.read_parquet(path) if path.is_file() else None


def _rescue_cache_fingerprint(
    *,
    targets: pd.DataFrame,
    predictor: pd.DataFrame,
    domain: JointDomainSpec,
    rescue_budget: int,
    max_ik_nfev: int,
    residual_limit_mm: float,
    cluster_threshold_deg: float,
    seed: int,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    pointwise_candidates: pd.DataFrame | None = None,
    expanded_seed_kappa_threshold: float = -np.inf,
) -> str:
    target_columns = ["angle_idx", *v6.atlas.TARGET_XYZ_COLS]
    predictor_columns = ["angle_idx", *v6.atlas.BETA_COLS]
    pointwise = pointwise_candidates if pointwise_candidates is not None else pd.DataFrame()
    pointwise_columns = [
        column
        for column in ("angle_idx", "xyz_residual_mm", *v6.atlas.BETA_COLS)
        if column in pointwise
    ]
    return stable_fingerprint(
        {
            "strategy": "shared_predictor_rescue_cache_v2",
            "targets": np.round(
                targets[target_columns].to_numpy(dtype=float), 12
            ).tolist(),
            "predictor": np.round(
                predictor[predictor_columns].to_numpy(dtype=float), 12
            ).tolist(),
            "pointwise": (
                np.round(pointwise[pointwise_columns].to_numpy(dtype=float), 12).tolist()
                if len(pointwise) and pointwise_columns
                else []
            ),
            "joint_domain_fingerprint": domain.fingerprint,
            "kinematics": {
                "lengths_m": np.round(np.asarray(lengths_m, dtype=float), 12).tolist(),
                "p_end_local_m": np.round(
                    np.asarray(p_end_local_m, dtype=float), 12
                ).tolist(),
                "theta_sign": float(theta_sign),
            },
            "rescue_budget": int(rescue_budget),
            "max_ik_nfev": int(max_ik_nfev),
            "residual_limit_mm": float(residual_limit_mm),
            "cluster_threshold_deg": float(cluster_threshold_deg),
            "seed": int(seed),
            "expanded_seed_kappa_threshold": float(expanded_seed_kappa_threshold),
        }
    )


def _correct_one_job(
    args: argparse.Namespace,
    *,
    radius_dir: Path,
    targets: pd.DataFrame,
    predictor: pd.DataFrame,
    predictor_report: Mapping[str, Any],
    cut_idx: int,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    family_id: str,
    target_radius_mm: float,
    rescue_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any], pd.DataFrame | None]:
    predictor_type = str(predictor_report["radial_predictor_type"])
    job_dir = radius_dir / "jobs" / predictor_type / f"cut_{int(cut_idx):03d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    schedules = parse_name_csv(args.anchor_schedules)
    attempts: list[dict[str, Any]] = []
    selected_path: pd.DataFrame | None = None
    selected_schedule: str | None = None
    selected_report: dict[str, Any] | None = None
    selected_strategy: str | None = None
    selected_input_predictor: pd.DataFrame | None = None
    domain = _resolved_joint_domain(args)
    lambda_margin = float(getattr(args, "lambda_margin", 0.0))
    soft_margin_deg = float(getattr(args, "soft_margin_deg", 0.25))

    def apply_job_gate(
        corrected_path: pd.DataFrame,
        correction: Mapping[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(correction)
        margin = _joint_margin_decision(corrected_path, args)
        enriched.update(margin)
        enriched["job_gate_pass"] = bool(
            enriched.get("centerline_gate_pass", False)
            and margin["job_margin_gate_pass"]
        )
        return enriched

    for schedule in schedules:
        corrected, correction_report = v6.correct_radial_predictor(
            targets,
            predictor,
            cut_idx=int(cut_idx),
            anchor_schedule=str(schedule),
            bounds=domain.bounds_rad,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            max_nfev=int(args.max_opt_nfev),
            compute_conditioning=True,
            lambda_margin=lambda_margin,
            soft_margin_deg=soft_margin_deg,
        )
        correction_report = apply_job_gate(corrected, correction_report)
        corrected.to_parquet(job_dir / f"{schedule}.parquet", index=False, compression="zstd")
        write_json(job_dir / f"{schedule}.json", correction_report)
        attempts.append(
            {
                "strategy": "direct_joint_corrector",
                "anchor_schedule": str(schedule),
                "centerline_gate_pass": bool(correction_report.get("centerline_gate_pass", False)),
                "job_gate_pass": bool(correction_report.get("job_gate_pass", False)),
                "report_path": str((job_dir / f"{schedule}.json").resolve()),
                "path": str((job_dir / f"{schedule}.parquet").resolve()),
                **{
                    key: correction_report.get(key)
                    for key in (
                        "selected_stage",
                        "residual_p95_mm",
                        "residual_max_mm",
                        "delta_beta_p95_deg",
                        "delta_beta_max_deg",
                        "delta2_beta_p95_deg",
                        "seam_beta_rms_deg",
                        "kappa_p95",
                        "sigma3_p05_m",
                    )
                },
            }
        )
        if bool(correction_report.get("job_gate_pass", False)):
            selected_path = corrected
            selected_schedule = str(schedule)
            selected_report = correction_report
            selected_strategy = "direct_joint_corrector"
            selected_input_predictor = v6.repeat_input_for_strategy(
                predictor,
                selected_strategy=selected_strategy,
            )
            break

    rescue_report: dict[str, Any] | None = None
    if selected_path is None:
        rescue_budget = int(
            getattr(args, "rescue_candidates_per_angle", args.max_candidates_per_angle)
        )
        pointwise = _pointwise_candidates_for_radius(
            args,
            family_id=str(family_id),
            radius_mm=float(target_radius_mm),
        )
        cache_fingerprint = _rescue_cache_fingerprint(
            targets=targets,
            predictor=predictor,
            domain=domain,
            rescue_budget=rescue_budget,
            max_ik_nfev=int(args.max_ik_nfev),
            residual_limit_mm=float(args.candidate_residual_mm),
            cluster_threshold_deg=float(args.candidate_cluster_deg),
            seed=int(args.seed),
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
            pointwise_candidates=pointwise,
            expanded_seed_kappa_threshold=float(
                getattr(args, "rescue_kappa_threshold", -np.inf)
            ),
        )
        shared_cache = rescue_cache if rescue_cache is not None else {}
        cached_payload = shared_cache.get(predictor_type)
        cache_dir = radius_dir / "rescue_cache" / predictor_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_report_path = cache_dir / "cache_report.json"
        cache_candidates_path = cache_dir / "candidates.parquet"
        cache_graph_path = cache_dir / "graph_path.parquet"
        cache_hit = bool(
            cached_payload is not None
            and str(cached_payload.get("fingerprint", "")) == cache_fingerprint
        )
        if not cache_hit and bool(args.skip_existing) and all(
            path.is_file()
            for path in (cache_report_path, cache_candidates_path, cache_graph_path)
        ):
            cached_report = read_json(cache_report_path)
            if (
                str(cached_report.get("fingerprint", "")) == cache_fingerprint
                and str(cached_report.get("candidates_sha256", ""))
                == v6.file_sha256(cache_candidates_path)
                and str(cached_report.get("graph_path_sha256", ""))
                == v6.file_sha256(cache_graph_path)
            ):
                cached_payload = {
                    "fingerprint": cache_fingerprint,
                    "candidates": pd.read_parquet(cache_candidates_path),
                    "candidate_report": cached_report["candidate_report"],
                    "graph_path": pd.read_parquet(cache_graph_path),
                    "graph_report": cached_report["graph_report"],
                }
                shared_cache[predictor_type] = cached_payload
                cache_hit = True
        if cache_hit and cached_payload is not None:
            candidates = cached_payload["candidates"]
            candidate_report = cached_payload["candidate_report"]
            graph_path = cached_payload["graph_path"]
            graph_report = cached_payload["graph_report"]
        else:
            candidates, candidate_report = v6.generate_radial_candidate_layers(
                targets,
                predictor,
                bounds=domain.bounds_rad,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                max_nfev=int(args.max_ik_nfev),
                pointwise_candidates=pointwise,
                max_seed_count=rescue_budget,
                residual_limit_mm=float(args.candidate_residual_mm),
                cluster_threshold_deg=float(args.candidate_cluster_deg),
                max_candidates_per_angle=rescue_budget,
                seed=int(args.seed),
                workers=int(args.workers) if rescue_budget > 8 else 1,
                expanded_seed_kappa_threshold=float(
                    getattr(args, "rescue_kappa_threshold", -np.inf)
                ),
            )
            graph_path, graph_report = v6.candidate_graph_rescue(candidates)
            candidates.to_parquet(
                cache_candidates_path,
                index=False,
                compression="zstd",
            )
            graph_path.to_parquet(
                cache_graph_path,
                index=False,
                compression="zstd",
            )
            cache_report = {
                "fingerprint": cache_fingerprint,
                "predictor_type": predictor_type,
                "candidate_report": candidate_report,
                "graph_report": graph_report,
                "candidates_path": str(cache_candidates_path.resolve()),
                "candidates_sha256": v6.file_sha256(cache_candidates_path),
                "graph_path": str(cache_graph_path.resolve()),
                "graph_path_sha256": v6.file_sha256(cache_graph_path),
            }
            write_json(cache_report_path, cache_report)
            cached_payload = {
                "fingerprint": cache_fingerprint,
                "candidates": candidates,
                "candidate_report": candidate_report,
                "graph_path": graph_path,
                "graph_report": graph_report,
            }
            shared_cache[predictor_type] = cached_payload
        if len(graph_path):
            graph_predictor = graph_path[["angle_idx", *v6.atlas.BETA_COLS]].copy()
            if "angle_rad" in graph_path:
                graph_predictor["angle_rad"] = graph_path["angle_rad"].to_numpy(dtype=float)
            graph_predictor["family_id"] = str(family_id)
            graph_predictor["radius_mm"] = float(target_radius_mm)
            graph_predictor["radial_predictor_type"] = f"{predictor_type}_candidate_graph"
            for schedule in schedules:
                corrected, correction_report = v6.correct_radial_predictor(
                    targets,
                    graph_predictor,
                    cut_idx=int(cut_idx),
                    anchor_schedule=str(schedule),
                    bounds=domain.bounds_rad,
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                    theta_sign=theta_sign,
                    max_nfev=int(args.max_opt_nfev),
                    compute_conditioning=True,
                    lambda_margin=lambda_margin,
                    soft_margin_deg=soft_margin_deg,
                )
                correction_report = apply_job_gate(corrected, correction_report)
                corrected.to_parquet(
                    job_dir / f"rescue_{schedule}.parquet", index=False, compression="zstd"
                )
                write_json(job_dir / f"rescue_{schedule}.json", correction_report)
                attempts.append(
                    {
                        "strategy": "candidate_graph_joint_corrector",
                        "anchor_schedule": str(schedule),
                        "centerline_gate_pass": bool(
                            correction_report.get("centerline_gate_pass", False)
                        ),
                        "job_gate_pass": bool(correction_report.get("job_gate_pass", False)),
                        "report_path": str((job_dir / f"rescue_{schedule}.json").resolve()),
                        "path": str((job_dir / f"rescue_{schedule}.parquet").resolve()),
                    }
                )
                if bool(correction_report.get("job_gate_pass", False)):
                    selected_path = corrected
                    selected_schedule = str(schedule)
                    selected_report = correction_report
                    selected_strategy = "candidate_graph_joint_corrector"
                    selected_input_predictor = v6.repeat_input_for_strategy(
                        predictor,
                        selected_strategy=selected_strategy,
                        candidate_graph_predictor=graph_predictor,
                    )
                    break
        rescue_report = {
            "candidate_report": candidate_report,
            "graph_report": graph_report,
            "cache_hit": cache_hit,
            "cache_fingerprint": cache_fingerprint,
            "cache_report_path": str(cache_report_path.resolve()),
        }
        write_json(job_dir / "rescue_report.json", rescue_report)

    report: dict[str, Any] = {
        "cut_idx": int(cut_idx),
        "radial_predictor_type": predictor_type,
        "selected": selected_path is not None,
        "selected_strategy": selected_strategy,
        "selected_anchor_schedule": selected_schedule,
        "centerline_gate_pass": bool(
            selected_report is not None and selected_report.get("centerline_gate_pass", False)
        ),
        "job_gate_pass": bool(
            selected_report is not None and selected_report.get("job_gate_pass", False)
        ),
        "attempts": attempts,
        "rescue": rescue_report,
    }
    if selected_report is not None:
        report.update(selected_report)
        report["cut_idx"] = int(cut_idx)
        report["radial_predictor_type"] = predictor_type
        report["selected"] = True
        report["selected_anchor_schedule"] = selected_schedule
    write_json(job_dir / "job_report.json", report)
    if selected_path is not None:
        selected_path.to_parquet(job_dir / "selected.parquet", index=False, compression="zstd")
        if selected_input_predictor is None:
            raise AssertionError("selected radial job is missing its exact repeat initializer")
        selected_input_predictor.to_parquet(
            job_dir / "selected_input_predictor.parquet", index=False, compression="zstd"
        )
    return selected_path, report, selected_input_predictor


def _solve_radial_radius(
    args: argparse.Namespace,
    *,
    audit_report: Mapping[str, Any],
    family: v6.FamilySpec,
    parent_path: pd.DataFrame,
    parent_radius_mm: float,
    parent_artifact_path: Path,
    previous_parent: pd.DataFrame | None,
    previous_parent_radius_mm: float | None,
    previous_parent_artifact_path: Path | None,
    target_radius_mm: float,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    radius_dir = Path(args.out_dir) / "01_radial" / radius_slug(float(target_radius_mm))
    radius_dir.mkdir(parents=True, exist_ok=True)
    parent_artifacts = [Path(parent_artifact_path)]
    if previous_parent_artifact_path is not None:
        parent_artifacts.append(Path(previous_parent_artifact_path))
    task_fingerprint = _radius_bundle_task_fingerprint(
        args,
        audit_report=audit_report,
        family=family,
        target_radius_mm=float(target_radius_mm),
        parent_artifact_paths=parent_artifacts,
    )
    report_path = radius_dir / "radius_bundle_report.json"
    selected_centerline_path = radius_dir / "selected_centerline_360.parquet"
    attempt_path = radius_dir / "selected_attempt_360.parquet"
    if bool(args.skip_existing) and report_path.exists():
        cached = read_json(report_path)
        artifact = selected_centerline_path if bool(cached.get("radial_bundle_gate_pass", False)) else attempt_path
        if (
            str(cached.get("task_fingerprint", "")) == task_fingerprint
            and artifact.is_file()
            and str(cached.get("selected_artifact_sha256", "")) == v6.file_sha256(artifact)
        ):
            return pd.read_parquet(artifact), cached, artifact

    targets = v6.generate_radius_targets(
        family,
        radius_mm=float(target_radius_mm),
        n_points=int(args.final_points),
    )
    geometry = v6.validate_target_geometry(
        targets,
        family=family,
        radius_mm=float(target_radius_mm),
    )
    targets.to_parquet(radius_dir / "targets_360.parquet", index=False, compression="zstd")
    predictors: dict[str, pd.DataFrame] = {}
    predictor_reports: dict[str, dict[str, Any]] = {}
    domain = _resolved_joint_domain(args)
    copy_predictor, copy_report = v6.build_radial_predictor(
        parent_path,
        parent_radius_mm=float(parent_radius_mm),
        target_radius_mm=float(target_radius_mm),
        family_id=family.family_id,
        bounds=domain.bounds_rad,
    )
    predictors["parent_copy"] = copy_predictor
    predictor_reports["parent_copy"] = copy_report
    if previous_parent is not None and previous_parent_radius_mm is not None:
        secant_predictor, secant_report = v6.build_radial_predictor(
            parent_path,
            parent_radius_mm=float(parent_radius_mm),
            target_radius_mm=float(target_radius_mm),
            previous_parent=previous_parent,
            previous_parent_radius_mm=float(previous_parent_radius_mm),
            family_id=family.family_id,
            bounds=domain.bounds_rad,
        )
        predictors["radial_secant"] = secant_predictor
        predictor_reports["radial_secant"] = secant_report
    for predictor_type, predictor in predictors.items():
        predictor.to_parquet(
            radius_dir / f"predictor_{predictor_type}.parquet", index=False, compression="zstd"
        )

    cuts = parse_int_csv(args.cut_indices)
    required_predictors = ["parent_copy", "radial_secant"] if previous_parent is not None else ["parent_copy"]
    job_reports: list[dict[str, Any]] = []
    selected_paths: dict[tuple[int, str], pd.DataFrame] = {}
    selected_repeat_inputs: dict[tuple[int, str], pd.DataFrame] = {}
    share_rescue_cache = bool(
        getattr(args, "share_rescue_cache_across_cuts", False)
    )
    rescue_cache: dict[str, dict[str, Any]] | None = (
        {} if share_rescue_cache else None
    )
    stop_after_first_failure = bool(
        getattr(args, "stop_after_first_failed_job", False)
    )
    first_failed_job: str | None = None
    for predictor_type in required_predictors:
        predictor = predictors[predictor_type]
        for cut_idx in cuts:
            selected, report, repeat_input = _correct_one_job(
                args,
                radius_dir=radius_dir,
                targets=targets,
                predictor=predictor,
                predictor_report=predictor_reports[predictor_type],
                cut_idx=int(cut_idx),
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                family_id=family.family_id,
                target_radius_mm=float(target_radius_mm),
                rescue_cache=rescue_cache,
            )
            job_reports.append(report)
            if _required_job_failed(report):
                if first_failed_job is None:
                    first_failed_job = f"{int(cut_idx)}:{predictor_type}"
                if stop_after_first_failure:
                    break
            if selected is not None:
                selected_paths[(int(cut_idx), predictor_type)] = selected
                if repeat_input is None:
                    raise AssertionError("selected radial job returned no deterministic repeat input")
                selected_repeat_inputs[(int(cut_idx), predictor_type)] = repeat_input
        if stop_after_first_failure and first_failed_job is not None:
            break

    cut_report = v6.cut_invariance_report(
        selected_paths,
        required_cuts=cuts,
        required_predictors=required_predictors,
        threshold_deg=1.0,
    )
    canonical_key = (0, "radial_secant") if (0, "radial_secant") in selected_paths else (0, "parent_copy")
    canonical = selected_paths.get(canonical_key)
    canonical_job = next(
        (
            report
            for report in job_reports
            if int(report.get("cut_idx", -1)) == canonical_key[0]
            and str(report.get("radial_predictor_type", "")) == canonical_key[1]
            and bool(report.get("selected", False))
        ),
        None,
    )
    if canonical is not None and canonical_job is not None:
        repeat, repeat_raw = v6.correct_radial_predictor(
            targets,
            selected_repeat_inputs[canonical_key],
            cut_idx=canonical_key[0],
            anchor_schedule=str(canonical_job["selected_anchor_schedule"]),
            bounds=domain.bounds_rad,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            max_nfev=int(args.max_opt_nfev),
            compute_conditioning=True,
            lambda_margin=float(getattr(args, "lambda_margin", 0.0)),
            soft_margin_deg=float(getattr(args, "soft_margin_deg", 0.25)),
        )
        repeat.to_parquet(radius_dir / "deterministic_repeat_360.parquet", index=False, compression="zstd")
        write_json(radius_dir / "deterministic_repeat_optimizer.json", repeat_raw)
        repeatability = v6.exact_repeatability_report(canonical, repeat)
    else:
        repeatability = {
            "deterministic_exact_gate_pass": False,
            "reason": "canonical_job_missing",
        }
    gate = aggregate_radius_bundle_gate(
        geometry_report=geometry,
        job_reports=job_reports,
        required_cuts=cuts,
        required_predictors=required_predictors,
        cut_report=cut_report,
        repeatability_report=repeatability,
    )
    if canonical is None:
        canonical = copy_predictor.copy()
    canonical["family_id"] = family.family_id
    canonical["radius_mm"] = float(target_radius_mm)
    canonical["parent_radius_mm"] = float(parent_radius_mm)
    canonical["previous_parent_radius_mm"] = (
        float(previous_parent_radius_mm) if previous_parent_radius_mm is not None else np.nan
    )
    canonical["branch_id"] = f"{family.family_id}:radial_bundle_v6"
    canonical["solver_strategy_version"] = v6.SOLVER_STRATEGY_VERSION
    canonical["radial_predictor_type"] = canonical_key[1]
    canonical["branch_hash"] = v6.atlas.branch_hash(canonical)
    canonical.to_parquet(attempt_path, index=False, compression="zstd")
    artifact_path = attempt_path
    if gate["radial_bundle_gate_pass"]:
        canonical.to_parquet(selected_centerline_path, index=False, compression="zstd")
        artifact_path = selected_centerline_path
    report = {
        "strategy_version": RUNNER_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        "family_id": family.family_id,
        "radius_mm": float(target_radius_mm),
        "parent_radius_mm": float(parent_radius_mm),
        "previous_parent_radius_mm": previous_parent_radius_mm,
        "geometry": geometry,
        "predictors": predictor_reports,
        "required_predictors": required_predictors,
        "required_cuts": cuts,
        "job_sweep_stopped_early": bool(
            stop_after_first_failure and first_failed_job is not None
        ),
        "first_failed_job": first_failed_job,
        "jobs": job_reports,
        "cut_invariance": cut_report,
        "deterministic_repeatability": repeatability,
        "canonical_job": {
            "cut_idx": canonical_key[0],
            "radial_predictor_type": canonical_key[1],
            "anchor_schedule": None if canonical_job is None else canonical_job.get("selected_anchor_schedule"),
        },
        "canonical_branch_hash": v6.atlas.branch_hash(canonical),
        "selected_artifact_path": str(artifact_path.resolve()),
        "selected_artifact_sha256": v6.file_sha256(artifact_path),
        **gate,
    }
    write_json(report_path, report)
    return canonical, report, artifact_path


def phase_radial(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_radial"
    out.mkdir(parents=True, exist_ok=True)
    audit_report = ensure_audit_report(args)
    if not bool(audit_report.get("audit_gate_pass", False)):
        report = {
            "strategy_version": RUNNER_STRATEGY_VERSION,
            "radial_walk_gate_pass": False,
            "target_radius_materialized": False,
            "reason": "audit_gate_failed",
        }
        write_json(out / "radial_report.json", report)
        return report
    family = load_family_spec(args.v5_dir, args.family_id)
    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    previous_radius = float(args.previous_radius_mm)
    start_radius = float(args.start_radius_mm)
    history: dict[float, pd.DataFrame] = {
        previous_radius: load_v5_parent_path(args.v5_dir, family.family_id, previous_radius),
        start_radius: load_v5_parent_path(args.v5_dir, family.family_id, start_radius),
    }
    artifact_paths: dict[float, Path] = {
        previous_radius: v5_centerline_path(args.v5_dir, family.family_id, previous_radius),
        start_radius: v5_centerline_path(args.v5_dir, family.family_id, start_radius),
    }

    def solve_step(
        parent_path: pd.DataFrame,
        parent_radius_mm: float,
        target_radius_mm: float,
    ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        print(
            f"[V6 radial] {float(parent_radius_mm):g} -> {float(target_radius_mm):g} mm",
            flush=True,
        )
        lower = sorted(radius for radius in history if radius < float(parent_radius_mm) - 1.0e-10)
        previous = lower[-1] if lower else None
        solved, report, artifact = _solve_radial_radius(
            args,
            audit_report=audit_report,
            family=family,
            parent_path=parent_path,
            parent_radius_mm=float(parent_radius_mm),
            parent_artifact_path=artifact_paths[float(parent_radius_mm)],
            previous_parent=None if previous is None else history[previous],
            previous_parent_radius_mm=previous,
            previous_parent_artifact_path=None if previous is None else artifact_paths[previous],
            target_radius_mm=float(target_radius_mm),
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
        )
        if bool(report.get("radial_bundle_gate_pass", False)):
            history[float(target_radius_mm)] = solved
            artifact_paths[float(target_radius_mm)] = artifact
        print(
            f"[V6 radial] radius={float(target_radius_mm):g} pass="
            f"{bool(report.get('radial_bundle_gate_pass', False))}",
            flush=True,
        )
        return solved, report

    passed, walk = v6.adaptive_radius_walk(
        history[start_radius],
        start_radius_mm=start_radius,
        checkpoints_mm=parse_float_csv(args.radius_checkpoints_mm),
        family_id=family.family_id,
        solve_step=solve_step,
        base_step_mm=float(args.base_step_mm),
        retry_steps_mm=parse_float_csv(args.retry_steps_mm),
    )
    del passed
    attempts = walk.get("attempts", [])
    summary_rows = [
        {
            "parent_radius_mm": item["parent_radius_mm"],
            "radius_mm": item["target_radius_mm"],
            "step_mm": item["step_mm"],
            "step_kind": item["step_kind"],
            "radial_bundle_gate_pass": item["radial_bundle_gate_pass"],
            "selected_artifact_path": item["report"].get("selected_artifact_path"),
            "canonical_branch_hash": item["report"].get("canonical_branch_hash"),
            "reason": "passed" if item["radial_bundle_gate_pass"] else "radial_bundle_failed",
        }
        for item in attempts
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "radius_attempt_summary.csv", index=False)
    manifest_path = Path(audit_report["legacy_artifact_manifest_path"])
    before = read_json(manifest_path)
    current_paths = [Path(path) for path in before]
    after = v6.artifact_hash_manifest(current_paths)
    legacy_unchanged = v6.artifact_manifests_match(before, after)
    target_materialized = bool(
        walk.get("radial_walk_gate_pass", False)
        and np.isclose(float(walk.get("last_pass_radius_mm", -np.inf)), float(args.target_radius_mm))
    )
    report = {
        "strategy_version": RUNNER_STRATEGY_VERSION,
        **formal_protocol_report(args),
        "audit_task_fingerprint": str(audit_report.get("task_fingerprint", "")),
        "family_id": family.family_id,
        "previous_radius_mm": previous_radius,
        "start_radius_mm": start_radius,
        "target_radius_mm": float(args.target_radius_mm),
        "radial_walk_gate_pass": bool(walk.get("radial_walk_gate_pass", False)),
        "last_pass_radius_mm": float(walk.get("last_pass_radius_mm", start_radius)),
        "target_radius_materialized": target_materialized,
        "attempt_count": int(len(attempts)),
        "attempts": attempts,
        "radius_summary_path": str((out / "radius_attempt_summary.csv").resolve()),
        "legacy_v2_v5_artifacts_unchanged": legacy_unchanged,
        "single_fixed_family_attempt_complete": True,
        "family_search_required": not target_materialized,
        "next_mode_if_failed": "bounded_8192_sobol_family_search_then_multi_chart",
    }
    report["formal_radial_gate_pass"] = bool(
        target_materialized
        and report["formal_protocol_gate_pass"]
        and legacy_unchanged
    )
    report["task_fingerprint"] = stable_fingerprint(
        {
            "audit": report["audit_task_fingerprint"],
            "protocol": report["protocol"],
            "attempt_artifacts": [
                {
                    "radius_mm": row["radius_mm"],
                    "path": row["selected_artifact_path"],
                    "branch_hash": row["canonical_branch_hash"],
                }
                for row in summary_rows
            ],
        }
    )
    write_json(out / "radial_report.json", report)
    return report


def ensure_radial_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "01_radial" / "radial_report.json"
    if path.exists():
        cached = read_json(path)
        if str(cached.get("protocol_fingerprint", "")) == str(
            formal_protocol_report(args)["protocol_fingerprint"]
        ):
            return cached
    return phase_radial(args)


def _tube_offset_file_id(pair: tuple[float, float]) -> str:
    return f"n1_{pair[0]:g}_n2_{pair[1]:g}".replace("-", "m").replace(".", "p")


def tube_worker_mode(*, n_points: int, workers: int) -> str:
    if int(workers) <= 1:
        return "serial"
    return "subprocess" if int(n_points) >= 360 else "thread"


def tube_outer_shell_refinement_jobs(
    offsets_mm: Iterable[float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pair each second-normal outer curve with its same-n1 inner parent."""
    offsets = sorted(float(value) for value in offsets_mm)
    if len(offsets) < 3 or len(offsets) != len(set(offsets)):
        raise ValueError("outer-shell refinement requires at least three unique offsets")
    jobs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for outer_n2, parent_n2 in ((offsets[0], offsets[1]), (offsets[-1], offsets[-2])):
        jobs.extend(
            [((float(n1), outer_n2), (float(n1), parent_n2)) for n1 in offsets]
        )
    return jobs


def _tube_quality_metrics(
    tube: pd.DataFrame,
    *,
    final_points: int,
    offsets_mm: Sequence[float],
) -> dict[str, Any]:
    expected_rows = int(final_points) * len(offsets_mm) ** 2
    residual = tube["xyz_residual_mm"].to_numpy(dtype=float)
    unique_grid_rows = int(
        tube[["angle_idx", "tube_offset_id"]].drop_duplicates().shape[0]
    )
    local = v6.atlas.local_beta_consistency_report(
        tube,
        radius_mm=10.0,
        max_neighbors=None,
        multi_branch_threshold_deg=3.0,
    )
    metrics: dict[str, Any] = {
        "rows": int(len(tube)),
        "expected_rows": expected_rows,
        "normal_grid_size": int(tube["tube_offset_id"].nunique()),
        "normal_grid_offsets_mm": [float(value) for value in offsets_mm],
        "target_success_ratio": float(tube["tube_success"].astype(bool).mean()),
        "normal_grid_coverage_ratio": float(unique_grid_rows / max(expected_rows, 1)),
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        **local,
    }
    metrics["tube_gate_pass"] = v5_expansion.tube_gate_pass(metrics)
    return metrics


def _centerline_for_radius(
    args: argparse.Namespace,
    *,
    family_id: str,
    radius_mm: float,
) -> tuple[pd.DataFrame, Path]:
    if np.isclose(float(radius_mm), float(args.previous_radius_mm), atol=1.0e-8) or np.isclose(
        float(radius_mm), float(args.start_radius_mm), atol=1.0e-8
    ):
        path = v5_centerline_path(args.v5_dir, family_id, radius_mm)
        return load_v5_parent_path(args.v5_dir, family_id, radius_mm), path
    path = Path(args.out_dir) / "01_radial" / radius_slug(radius_mm) / "selected_centerline_360.parquet"
    centerline = pd.read_parquet(path).sort_values("angle_idx", kind="stable").reset_index(drop=True)
    return centerline, path


def _tube_radius_task_fingerprint(
    args: argparse.Namespace,
    *,
    radial_report: Mapping[str, Any],
    family_id: str,
    radius_mm: float,
    centerline_path: Path,
) -> str:
    return stable_fingerprint(
        {
            "phase": "tube_radius",
            "strategy_version": TUBE_STRATEGY_VERSION,
            "radial_task_fingerprint": str(radial_report.get("task_fingerprint", "")),
            "family_id": str(family_id),
            "radius_mm": float(radius_mm),
            "centerline_path": str(centerline_path.resolve()),
            "centerline_sha256": v6.file_sha256(centerline_path),
            "tube_offsets_mm": parse_float_csv(args.tube_offsets_mm),
            "max_ik_nfev": int(args.max_ik_nfev),
            "robot_config_sha256": v6.file_sha256(args.robot_config),
            "solver_strategy_version": v6.SOLVER_STRATEGY_VERSION,
        }
    )


def _center_curve(
    centerline: pd.DataFrame,
    tube_targets: pd.DataFrame,
    *,
    theta_sign: float,
) -> pd.DataFrame:
    center_targets = tube_targets[
        np.isclose(tube_targets["delta_n1_mm"].to_numpy(dtype=float), 0.0)
        & np.isclose(tube_targets["delta_n2_mm"].to_numpy(dtype=float), 0.0)
    ].sort_values("angle_idx", kind="stable").reset_index(drop=True)
    center = centerline.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    beta = center[v6.atlas.BETA_COLS].to_numpy(dtype=float)
    xyz = center[v6.atlas.XYZ_COLS].to_numpy(dtype=float)
    output = center_targets.copy()
    for idx, column in enumerate(v6.atlas.BETA_COLS):
        output[column] = beta[:, idx]
    theta = v6.atlas.theta_from_beta_batch(beta, theta_sign=float(theta_sign))
    for idx, column in enumerate(v6.atlas.THETA_COLS):
        output[column] = theta[:, idx]
    for idx, column in enumerate(v6.atlas.XYZ_COLS):
        output[column] = xyz[:, idx]
    output["xyz_residual_mm"] = np.linalg.norm(
        xyz - output[v6.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float), axis=1
    ) * 1000.0
    output["tube_success"] = output["xyz_residual_mm"].le(1.5)
    output["parent_offset_id"] = "centerline"
    output["inverse_nfev"] = 0
    return output


def _refine_tube_outer_shell(
    *,
    radius_dir: Path,
    tube_targets: pd.DataFrame,
    solved: Mapping[tuple[float, float], pd.DataFrame],
    offsets_mm: Sequence[float],
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int,
) -> tuple[dict[tuple[float, float], pd.DataFrame], list[dict[str, Any]]]:
    """Jointly re-lift both n2 outer shells from their same-n1 inner curves."""
    refined = {pair: curve.copy() for pair, curve in solved.items()}
    refined_dir = radius_dir / "outer_shell_joint_refinement"
    refined_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for pair, parent_pair in tube_outer_shell_refinement_jobs(offsets_mm):
        if pair not in solved or parent_pair not in solved:
            raise ValueError(f"outer-shell refinement is missing pair={pair} parent={parent_pair}")
        targets = tube_targets[
            np.isclose(tube_targets["delta_n1_mm"].to_numpy(dtype=float), pair[0])
            & np.isclose(tube_targets["delta_n2_mm"].to_numpy(dtype=float), pair[1])
        ].sort_values("angle_idx", kind="stable").reset_index(drop=True)
        parent = solved[parent_pair].sort_values("angle_idx", kind="stable").reset_index(drop=True)
        corrected, optimizer_report = v6.atlas.optimize_cyclic_trajectory(
            targets=targets,
            initial_beta=parent[v6.atlas.BETA_COLS].to_numpy(dtype=float),
            bounds=v6.atlas.beta_bounds_rad("current"),
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
            stages=[TUBE_OUTER_SHELL_STAGE],
            max_nfev=int(max_nfev),
            compute_conditioning=True,
            stop_on_centerline_gate=False,
        )
        corrected["tube_success"] = corrected["xyz_residual_mm"].le(1.5)
        corrected["parent_offset_id"] = (
            f"n1_{parent_pair[0]:g}_n2_{parent_pair[1]:g}"
        )
        corrected["inverse_nfev"] = int(optimizer_report.get("optimizer_nfev", 0))
        corrected["tube_refinement_strategy"] = str(TUBE_OUTER_SHELL_STAGE["name"])
        file_id = _tube_offset_file_id(pair)
        curve_path = refined_dir / f"{file_id}.parquet"
        corrected.to_parquet(curve_path, index=False, compression="zstd")
        curve_report = {
            "pair": [float(pair[0]), float(pair[1])],
            "parent_pair": [float(parent_pair[0]), float(parent_pair[1])],
            "rows": int(len(corrected)),
            "success_ratio": float(corrected["tube_success"].mean()),
            "residual_p95_mm": float(np.percentile(corrected["xyz_residual_mm"], 95)),
            "residual_max_mm": float(corrected["xyz_residual_mm"].max()),
            "delta_beta_p95_deg": optimizer_report.get("delta_beta_p95_deg"),
            "delta_beta_max_deg": optimizer_report.get("delta_beta_max_deg"),
            "delta2_beta_p95_deg": optimizer_report.get("delta2_beta_p95_deg"),
            "seam_beta_rms_deg": optimizer_report.get("seam_beta_rms_deg"),
            "optimizer_nfev": optimizer_report.get("optimizer_nfev"),
            "curve_path": str(curve_path.resolve()),
            "curve_sha256": v6.file_sha256(curve_path),
        }
        write_json(refined_dir / f"{file_id}.json", curve_report)
        reports.append(curve_report)
        refined[pair] = corrected
    return refined, reports


def materialize_tube_radius(
    args: argparse.Namespace,
    *,
    radial_report: Mapping[str, Any],
    family_id: str,
    radius_mm: float,
    centerline: pd.DataFrame,
    centerline_path: Path,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    radius_dir = Path(args.out_dir) / "02_tube" / radius_slug(float(radius_mm))
    radius_dir.mkdir(parents=True, exist_ok=True)
    task_fingerprint = _tube_radius_task_fingerprint(
        args,
        radial_report=radial_report,
        family_id=str(family_id),
        radius_mm=float(radius_mm),
        centerline_path=centerline_path,
    )
    report_path = radius_dir / "tube_quality_report.json"
    tube_path = radius_dir / "tube_small.parquet"
    attempt_path = radius_dir / "tube_attempt.parquet"
    if bool(args.skip_existing) and report_path.exists():
        cached = read_json(report_path)
        artifact = tube_path if bool(cached.get("tube_gate_pass", False)) else attempt_path
        if (
            str(cached.get("task_fingerprint", "")) == task_fingerprint
            and artifact.is_file()
            and str(cached.get("tube_artifact_sha256", "")) == v6.file_sha256(artifact)
            and (
                bool(cached.get("tube_gate_pass", False))
                or bool(cached.get("outer_shell_joint_refinement_attempted", False))
            )
        ):
            return pd.read_parquet(artifact), cached, artifact

    offsets = parse_float_csv(args.tube_offsets_mm)
    if not (
        len(offsets) == 5
        and len(set(offsets)) == 5
        and np.allclose(sorted(offsets), v6.FORMAL_TUBE_OFFSETS_MM)
    ):
        raise ValueError(f"V6 formal tube offsets are fixed at {list(v6.FORMAL_TUBE_OFFSETS_MM)}")
    centerline = centerline.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    tube_targets = v6.atlas.make_normal_tube_targets(centerline, offsets_mm=offsets)
    tube_targets.to_parquet(radius_dir / "tube_targets.parquet", index=False, compression="zstd")
    center_beta = centerline[v6.atlas.BETA_COLS].to_numpy(dtype=float)
    pinv = np.asarray(
        [
            v6.atlas.weighted_damped_pinv(
                v6.atlas.numerical_jacobian_beta(
                    beta,
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                    theta_sign=theta_sign,
                ),
                damping=1.0e-3,
                weights=np.asarray([4, 4, 2, 2, 1, 1], dtype=float),
            )
            for beta in center_beta
        ],
        dtype=float,
    )
    np.save(radius_dir / "weighted_pinv.npy", pinv)
    offset_pairs = [(float(left), float(right)) for left in offsets for right in offsets]
    center_pair = (0.0, 0.0)
    curves_dir = radius_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    center_curve = _center_curve(centerline, tube_targets, theta_sign=theta_sign)
    center_curve_path = curves_dir / f"{_tube_offset_file_id(center_pair)}.parquet"
    center_curve.to_parquet(center_curve_path, index=False, compression="zstd")
    solved: dict[tuple[float, float], pd.DataFrame] = {center_pair: center_curve}
    solved_paths: dict[tuple[float, float], Path] = {center_pair: center_curve_path}
    curve_targets_dir = radius_dir / "curve_targets"
    worker_tasks_dir = radius_dir / "worker_tasks"
    curve_targets_dir.mkdir(parents=True, exist_ok=True)
    worker_tasks_dir.mkdir(parents=True, exist_ok=True)
    worker_mode = tube_worker_mode(n_points=len(centerline), workers=int(args.workers))
    radius_groups: dict[float, list[tuple[float, float]]] = {}
    for pair in offset_pairs:
        if pair != center_pair:
            radius_groups.setdefault(round(math.hypot(pair[0], pair[1]), 8), []).append(pair)

    for offset_radius in sorted(radius_groups):
        print(
            f"[V6 tube] radius={float(radius_mm):g} mm offset_norm={offset_radius:g} mm",
            flush=True,
        )
        jobs: list[tuple[tuple[float, float], tuple[float, float], pd.DataFrame]] = []
        for pair in radius_groups[offset_radius]:
            smaller = [
                old
                for old in solved
                if math.hypot(old[0], old[1]) < offset_radius - 1.0e-12
            ]
            parent_pair = (
                min(smaller, key=lambda old: math.hypot(old[0] - pair[0], old[1] - pair[1]))
                if smaller
                else center_pair
            )
            curve_targets = tube_targets[
                np.isclose(tube_targets["delta_n1_mm"].to_numpy(dtype=float), pair[0])
                & np.isclose(tube_targets["delta_n2_mm"].to_numpy(dtype=float), pair[1])
            ].sort_values("angle_idx", kind="stable").reset_index(drop=True)
            jobs.append((pair, parent_pair, curve_targets))

        def solve_job(
            job: tuple[tuple[float, float], tuple[float, float], pd.DataFrame]
        ) -> tuple[tuple[float, float], pd.DataFrame]:
            pair, parent_pair, curve_targets = job
            curve = v5_expansion.solve_tube_curve(
                curve_targets,
                centerline=centerline,
                parent_curve=solved[parent_pair],
                weighted_pinv=pinv,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                max_nfev=int(args.max_ik_nfev),
                parent_offset_id=f"n1_{parent_pair[0]:g}_n2_{parent_pair[1]:g}",
            )
            return pair, curve

        if worker_mode == "serial":
            results = [solve_job(job) for job in jobs]
        elif worker_mode == "thread":
            with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
                results = list(executor.map(solve_job, jobs))
        else:
            task_paths: list[Path] = []
            pair_outputs: dict[tuple[float, float], Path] = {}
            for pair, parent_pair, curve_targets in jobs:
                file_id = _tube_offset_file_id(pair)
                targets_path = curve_targets_dir / f"{file_id}.parquet"
                output_path = curves_dir / f"{file_id}.parquet"
                worker_report_path = output_path.with_suffix(".json")
                task_path = worker_tasks_dir / f"{file_id}.json"
                curve_targets.to_parquet(targets_path, index=False, compression="zstd")
                worker_fingerprint = stable_fingerprint(
                    {
                        "radius_task_fingerprint": task_fingerprint,
                        "pair": pair,
                        "parent_pair": parent_pair,
                        "parent_curve_sha256": v6.file_sha256(solved_paths[parent_pair]),
                    }
                )
                pair_outputs[pair] = output_path
                if bool(args.skip_existing) and output_path.exists() and worker_report_path.exists():
                    cached_worker = read_json(worker_report_path)
                    if str(cached_worker.get("task_fingerprint", "")) == worker_fingerprint:
                        continue
                write_json(
                    task_path,
                    {
                        "kind": "tube_curve",
                        "task_id": f"{radius_slug(radius_mm)}:{file_id}",
                        "task_fingerprint": worker_fingerprint,
                        "robot_config": str(Path(args.robot_config).resolve()),
                        "centerline_path": str(centerline_path.resolve()),
                        "curve_targets_path": str(targets_path.resolve()),
                        "parent_curve_path": str(solved_paths[parent_pair].resolve()),
                        "pinv_path": str((radius_dir / "weighted_pinv.npy").resolve()),
                        "max_nfev": int(args.max_ik_nfev),
                        "parent_offset_id": f"n1_{parent_pair[0]:g}_n2_{parent_pair[1]:g}",
                        "offset_id": f"n1_{pair[0]:g}_n2_{pair[1]:g}",
                        "output_path": str(output_path.resolve()),
                        "report_path": str(worker_report_path.resolve()),
                    },
                )
                task_paths.append(task_path)
            v5_expansion._execute_worker_tasks(task_paths, workers=int(args.workers))
            results = [(pair, pd.read_parquet(path)) for pair, path in pair_outputs.items()]
        for pair, curve in results:
            solved[pair] = curve
            curve_path = curves_dir / f"{_tube_offset_file_id(pair)}.parquet"
            if worker_mode != "subprocess":
                curve.to_parquet(curve_path, index=False, compression="zstd")
            solved_paths[pair] = curve_path

    initial_tube = pd.concat(
        [solved[pair] for pair in offset_pairs], ignore_index=True, sort=False
    )
    initial_attempt_path = radius_dir / "tube_initial_attempt.parquet"
    initial_tube.to_parquet(initial_attempt_path, index=False, compression="zstd")
    initial_quality = _tube_quality_metrics(
        initial_tube,
        final_points=int(args.final_points),
        offsets_mm=offsets,
    )
    tube = initial_tube
    refinement_reports: list[dict[str, Any]] = []
    refinement_attempted = not bool(initial_quality["tube_gate_pass"])
    if refinement_attempted:
        refined_curves, refinement_reports = _refine_tube_outer_shell(
            radius_dir=radius_dir,
            tube_targets=tube_targets,
            solved=solved,
            offsets_mm=offsets,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            max_nfev=int(args.max_opt_nfev),
        )
        tube = pd.concat(
            [refined_curves[pair] for pair in offset_pairs],
            ignore_index=True,
            sort=False,
        )
    tube.to_parquet(attempt_path, index=False, compression="zstd")
    final_metrics = _tube_quality_metrics(
        tube,
        final_points=int(args.final_points),
        offsets_mm=offsets,
    )
    quality: dict[str, Any] = {
        "strategy_version": TUBE_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        "family_id": str(family_id),
        "radius_mm": float(radius_mm),
        "centerline_path": str(centerline_path.resolve()),
        "centerline_sha256": v6.file_sha256(centerline_path),
        "worker_mode": worker_mode,
        "outer_shell_joint_refinement_attempted": refinement_attempted,
        "outer_shell_joint_refinement_gate_pass": bool(
            refinement_attempted and final_metrics["tube_gate_pass"]
        ),
        "outer_shell_joint_refinement_stage": dict(TUBE_OUTER_SHELL_STAGE),
        "outer_shell_joint_refinement_reports": refinement_reports,
        "initial_quality": initial_quality,
        "initial_attempt_path": str(initial_attempt_path.resolve()),
        "initial_attempt_sha256": v6.file_sha256(initial_attempt_path),
        **final_metrics,
    }
    if quality["tube_gate_pass"]:
        tube.to_parquet(tube_path, index=False, compression="zstd")
        artifact_path = tube_path
    else:
        artifact_path = attempt_path
    quality["tube_artifact_path"] = str(artifact_path.resolve())
    quality["tube_artifact_sha256"] = v6.file_sha256(artifact_path)
    write_json(report_path, quality)
    return tube, quality, artifact_path


def _reuse_v5_tube(
    args: argparse.Namespace,
    *,
    family_id: str,
    radius_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    radius_dir = Path(args.v5_dir) / "04_tube" / family_id / radius_slug(radius_mm)
    tube_path = radius_dir / "tube_small.parquet"
    report_path = radius_dir / "tube_quality_report.json"
    tube = pd.read_parquet(tube_path)
    raw = read_json(report_path)
    recomputed = v5_expansion.tube_gate_pass(raw)
    report = {
        **raw,
        "source_kind": "v5_hash_bound_reuse",
        "source_report_path": str(report_path.resolve()),
        "source_report_sha256": v6.file_sha256(report_path),
        "tube_artifact_path": str(tube_path.resolve()),
        "tube_artifact_sha256": v6.file_sha256(tube_path),
        "tube_gate_pass": bool(recomputed and len(tube) == int(args.final_points) * 25),
    }
    return tube, report, tube_path


def phase_tube(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_tube"
    out.mkdir(parents=True, exist_ok=True)
    radial_report = ensure_radial_report(args)
    if not bool(radial_report.get("target_radius_materialized", False)):
        report = {
            "strategy_version": TUBE_STRATEGY_VERSION,
            "tube_gate_pass": False,
            "formal_tube_gate_pass": False,
            "reason": "100mm_radial_bundle_not_materialized",
        }
        write_json(out / "tube_report.json", report)
        return report
    family_id = str(args.family_id)
    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    summary_rows: list[dict[str, Any]] = []
    tube_paths: dict[str, str] = {}
    connected = True
    for radius_mm in v6.FORMAL_RADII_MM:
        if not connected:
            summary_rows.append(
                {
                    "family_id": family_id,
                    "radius_mm": float(radius_mm),
                    "tube_gate_pass": False,
                    "executed": False,
                    "reason": "previous_formal_radius_failed",
                }
            )
            continue
        if radius_mm <= float(args.start_radius_mm) + 1.0e-8:
            print(f"[V6 tube] reusing audited V5 tube at {float(radius_mm):g} mm", flush=True)
            tube, quality, artifact = _reuse_v5_tube(
                args,
                family_id=family_id,
                radius_mm=float(radius_mm),
            )
        else:
            print(f"[V6 tube] solving formal tube at {float(radius_mm):g} mm", flush=True)
            centerline, centerline_path = _centerline_for_radius(
                args,
                family_id=family_id,
                radius_mm=float(radius_mm),
            )
            tube, quality, artifact = materialize_tube_radius(
                args,
                radial_report=radial_report,
                family_id=family_id,
                radius_mm=float(radius_mm),
                centerline=centerline,
                centerline_path=centerline_path,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
            )
        del tube
        passed = bool(quality.get("tube_gate_pass", False))
        connected = passed
        if passed:
            tube_paths[f"{float(radius_mm):g}"] = str(artifact.resolve())
        summary_rows.append(
            {
                "family_id": family_id,
                "radius_mm": float(radius_mm),
                "tube_gate_pass": passed,
                "executed": True,
                "reason": "passed" if passed else "tube_gate_failed",
                "rows": quality.get("rows"),
                "residual_p95_mm": quality.get("residual_p95_mm"),
                "residual_max_mm": quality.get("residual_max_mm"),
                "tube10_beta_rms_p95_deg": quality.get("tube10_beta_rms_p95_deg"),
                "multi_branch_ratio": quality.get("multi_branch_ratio"),
                "tube_artifact_path": str(artifact.resolve()),
                "tube_artifact_sha256": quality.get("tube_artifact_sha256"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "tube_radius_summary.csv", index=False)
    all_formal = bool(
        len(summary) == len(v6.FORMAL_RADII_MM)
        and summary["executed"].astype(bool).all()
        and summary["tube_gate_pass"].astype(bool).all()
    )
    report = {
        "strategy_version": TUBE_STRATEGY_VERSION,
        **formal_protocol_report(args),
        "radial_task_fingerprint": str(radial_report.get("task_fingerprint", "")),
        "family_id": family_id,
        "formal_radii_mm": list(v6.FORMAL_RADII_MM),
        "tube_gate_pass": all_formal,
        "target_100mm_tube_materialized": bool(
            all_formal and "100" in tube_paths
        ),
        "tube_paths": tube_paths,
        "summary_path": str((out / "tube_radius_summary.csv").resolve()),
    }
    report["formal_tube_gate_pass"] = bool(
        report["tube_gate_pass"]
        and report["formal_protocol_gate_pass"]
        and bool(radial_report.get("formal_radial_gate_pass", False))
    )
    report["task_fingerprint"] = stable_fingerprint(
        {
            "radial": report["radial_task_fingerprint"],
            "protocol": report["protocol"],
            "tube_artifacts": [
                {
                    "radius_mm": row["radius_mm"],
                    "sha256": row.get("tube_artifact_sha256"),
                }
                for row in summary_rows
            ],
        }
    )
    write_json(out / "tube_report.json", report)
    return report


def ensure_tube_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "02_tube" / "tube_report.json"
    if path.exists():
        cached = read_json(path)
        if str(cached.get("protocol_fingerprint", "")) == str(
            formal_protocol_report(args)["protocol_fingerprint"]
        ):
            return cached
    return phase_tube(args)


def _centerline_provenance(
    args: argparse.Namespace,
    *,
    family_id: str,
    radius_mm: float,
) -> dict[str, Any]:
    centerline, centerline_path = _centerline_for_radius(
        args,
        family_id=family_id,
        radius_mm=float(radius_mm),
    )
    if radius_mm <= float(args.start_radius_mm) + 1.0e-8:
        parent_radius = None if np.isclose(radius_mm, args.previous_radius_mm) else float(args.previous_radius_mm)
        predictor_type = "v5_robust_parent"
    else:
        radius_report_path = (
            Path(args.out_dir) / "01_radial" / radius_slug(radius_mm) / "radius_bundle_report.json"
        )
        radius_report = read_json(radius_report_path)
        parent_radius = float(radius_report["parent_radius_mm"])
        predictor_type = str(radius_report["canonical_job"]["radial_predictor_type"])
    return {
        "centerline": centerline,
        "centerline_path": centerline_path,
        "centerline_sha256": v6.file_sha256(centerline_path),
        "parent_radius_mm": parent_radius,
        "radial_predictor_type": predictor_type,
        "branch_hash": v6.atlas.branch_hash(centerline),
    }


def phase_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_dataset"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "dataset_report.json"
    attempt_path = out / "dataset_attempt.parquet"
    final_path = out / "true_ellipse_radial_bundle_tubes_v6.parquet"
    manifest_path = out / "trajectory_manifest.csv"
    tube_report = ensure_tube_report(args)
    if not bool(tube_report.get("tube_gate_pass", False)):
        report = {
            "strategy_version": DATASET_STRATEGY_VERSION,
            "dataset_gate_pass": False,
            "formal_dataset_gate_pass": False,
            "reason": "formal_100mm_tube_chain_not_materialized",
        }
        write_json(report_path, report)
        return report
    family = load_family_spec(args.v5_dir, args.family_id)
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for radius_mm in v6.FORMAL_RADII_MM:
        tube_path = Path(tube_report["tube_paths"][f"{float(radius_mm):g}"])
        tube = pd.read_parquet(tube_path)
        provenance = _centerline_provenance(
            args,
            family_id=family.family_id,
            radius_mm=float(radius_mm),
        )
        annotated = annotate_tube_rows(
            tube,
            family_id=family.family_id,
            radius_mm=float(radius_mm),
            parent_radius_mm=provenance["parent_radius_mm"],
            radial_predictor_type=provenance["radial_predictor_type"],
            branch_hash=provenance["branch_hash"],
        )
        expected_rows = int(args.final_points) * 25
        duplicate_count = int(
            annotated.duplicated(["angle_idx", "tube_offset_id"], keep=False).sum()
        )
        complete = bool(
            len(annotated) == expected_rows
            and int(annotated["angle_idx"].nunique()) == int(args.final_points)
            and int(annotated["tube_offset_id"].nunique()) == 25
            and duplicate_count == 0
            and annotated["sample_id"].is_unique
        )
        manifest_rows.append(
            {
                "trajectory_id": str(annotated["trajectory_id"].iloc[0]),
                "family_id": family.family_id,
                "radius_mm": float(radius_mm),
                "parent_radius_mm": provenance["parent_radius_mm"],
                "rows": int(len(annotated)),
                "expected_rows": expected_rows,
                "angle_count": int(annotated["angle_idx"].nunique()),
                "offset_count": int(annotated["tube_offset_id"].nunique()),
                "duplicate_angle_offset_rows": duplicate_count,
                "tube_success_ratio": float(annotated["tube_success"].mean()),
                "trajectory_complete": complete,
                "radial_predictor_type": provenance["radial_predictor_type"],
                "branch_hash": provenance["branch_hash"],
                "centerline_path": str(Path(provenance["centerline_path"]).resolve()),
                "centerline_sha256": provenance["centerline_sha256"],
                "tube_path": str(tube_path.resolve()),
                "tube_sha256": v6.file_sha256(tube_path),
            }
        )
        frames.append(annotated)
    dataset = pd.concat(frames, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifest_rows)
    assigned, split_report = v6.assign_whole_radius_splits(dataset)
    assigned.to_parquet(attempt_path, index=False, compression="zstd")
    manifest.to_csv(manifest_path, index=False)
    conflict = v5_expansion.branch_conflict_report(
        assigned,
        voxel_mm=2.0,
        threshold_deg=3.0,
    )
    training_pool = v6.training_only_support_pool(assigned)
    # The helper resets its index, so recover the exact source indices by sample_id.
    source_index = pd.Series(assigned.index.to_numpy(), index=assigned["sample_id"].astype(str))
    training_indices = source_index.loc[training_pool["sample_id"].astype(str)].to_numpy(dtype=np.int64)
    support = v5_training.compute_support_scan(
        assigned,
        training_indices=training_indices,
        metadata=family.__dict__,
        radii_mm=v6.FORMAL_RADII_MM,
        n_points=int(args.final_points),
    )
    support.to_csv(out / "training_only_support_by_radius.csv", index=False)
    support_all = bool(len(support) == len(v6.FORMAL_RADII_MM) and support["strict_support_gate_pass"].all())
    target_support = bool(
        support.loc[np.isclose(support["radius_mm"], v6.TEST_RADIUS_MM), "strict_support_gate_pass"].all()
    )
    required_metadata = {
        "family_id",
        "branch_id",
        "trajectory_id",
        "radius_mm",
        "parent_radius_mm",
        "angle_idx",
        "tube_offset_id",
        "is_centerline",
        "solver_strategy_version",
        "radial_predictor_type",
        "branch_hash",
        "sample_id",
        "split",
    }
    missing_metadata = sorted(required_metadata - set(assigned.columns))
    model_input_columns = list(v6.atlas.TARGET_XYZ_COLS)
    forbidden_model_inputs = sorted(
        set(model_input_columns)
        & {"family_id", "branch_id", "radius_mm", "angle_idx", "angle_rad", "trajectory_id"}
    )
    completeness_gate = bool(
        len(manifest) == len(v6.FORMAL_RADII_MM)
        and manifest["trajectory_complete"].all()
        and manifest["tube_success_ratio"].ge(0.99).all()
        and assigned["sample_id"].is_unique
        and int(assigned["family_id"].nunique()) == 1
    )
    dataset_gate = bool(
        not missing_metadata
        and completeness_gate
        and split_report["whole_radius_split_gate_pass"]
        and conflict["branch_conflict_gate_pass"]
        and support_all
        and target_support
        and not forbidden_model_inputs
    )
    if dataset_gate:
        assigned.to_parquet(final_path, index=False, compression="zstd")
    report = {
        "strategy_version": DATASET_STRATEGY_VERSION,
        **formal_protocol_report(args),
        "tube_task_fingerprint": str(tube_report.get("task_fingerprint", "")),
        "family_id": family.family_id,
        "dataset_gate_pass": dataset_gate,
        "rows": int(len(assigned)),
        "expected_rows": int(len(v6.FORMAL_RADII_MM) * int(args.final_points) * 25),
        "trajectory_count": int(assigned["trajectory_id"].nunique()),
        "unique_family_count": int(assigned["family_id"].nunique()),
        "radii_mm": sorted(float(value) for value in assigned["radius_mm"].unique()),
        "missing_metadata_columns": missing_metadata,
        "unique_sample_ids": bool(assigned["sample_id"].is_unique),
        "all_trajectories_complete": completeness_gate,
        "split": split_report,
        "branch_conflict": conflict,
        "training_only_support_all_radii_pass": support_all,
        "training_only_support_100mm_pass": target_support,
        "training_only_support_rows": int(len(training_pool)),
        "training_only_support_path": str((out / "training_only_support_by_radius.csv").resolve()),
        "model_input_columns": model_input_columns,
        "forbidden_model_inputs": forbidden_model_inputs,
        "attempt_path": str(attempt_path.resolve()),
        "attempt_sha256": v6.file_sha256(attempt_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": v6.file_sha256(manifest_path),
        "dataset_path": str(final_path.resolve()) if dataset_gate else None,
        "dataset_sha256": v6.file_sha256(final_path) if dataset_gate else None,
    }
    report["formal_dataset_gate_pass"] = bool(
        dataset_gate
        and report["formal_protocol_gate_pass"]
        and bool(tube_report.get("formal_tube_gate_pass", False))
    )
    report["task_fingerprint"] = stable_fingerprint(
        {
            "tube": report["tube_task_fingerprint"],
            "protocol": report["protocol"],
            "attempt_sha256": report["attempt_sha256"],
            "manifest_sha256": report["manifest_sha256"],
            "dataset_sha256": report["dataset_sha256"],
        }
    )
    write_json(report_path, report)
    return report


def ensure_dataset_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "03_dataset" / "dataset_report.json"
    if path.exists():
        cached = read_json(path)
        if str(cached.get("protocol_fingerprint", "")) == str(
            formal_protocol_report(args)["protocol_fingerprint"]
        ):
            return cached
    return phase_dataset(args)


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "04_summary"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    radial = ensure_radial_report(args)
    tube = ensure_tube_report(args)
    dataset = ensure_dataset_report(args)
    legacy_manifest = read_json(audit["legacy_artifact_manifest_path"])
    current_manifest = v6.artifact_hash_manifest([Path(path) for path in legacy_manifest])
    legacy_unchanged = v6.artifact_manifests_match(legacy_manifest, current_manifest)
    strict_upstream = bool(
        audit.get("formal_audit_gate_pass", False)
        and radial.get("formal_radial_gate_pass", False)
        and tube.get("formal_tube_gate_pass", False)
        and dataset.get("formal_dataset_gate_pass", False)
        and legacy_unchanged
    )
    report = {
        **formal_protocol_report(args),
        "family_id": str(args.family_id),
        "target_radius_mm": float(args.target_radius_mm),
        "audit_gate_pass": bool(audit.get("formal_audit_gate_pass", False)),
        "radial_100mm_gate_pass": bool(radial.get("formal_radial_gate_pass", False)),
        "tube_100mm_gate_pass": bool(tube.get("formal_tube_gate_pass", False)),
        "dataset_gate_pass": bool(dataset.get("formal_dataset_gate_pass", False)),
        "training_only_support_100mm_pass": bool(
            dataset.get("training_only_support_100mm_pass", False)
        ),
        "legacy_v2_v5_artifacts_unchanged": legacy_unchanged,
        "strict_upstream_100mm_gate_pass": strict_upstream,
        "model_training_authorized": strict_upstream,
        "strict_model_claim_made": False,
        "static_inverse_claim_radius_mm": None,
        "single_branch_failure_requires_multi_chart": bool(
            radial.get("single_fixed_family_attempt_complete", False)
            and not radial.get("target_radius_materialized", False)
        ),
        "reports": {
            "audit": str((Path(args.out_dir) / "00_audit" / "audit_report.json").resolve()),
            "radial": str((Path(args.out_dir) / "01_radial" / "radial_report.json").resolve()),
            "tube": str((Path(args.out_dir) / "02_tube" / "tube_report.json").resolve()),
            "dataset": str((Path(args.out_dir) / "03_dataset" / "dataset_report.json").resolve()),
        },
    }
    write_json(out / "summary_report.json", report)
    radius_summary_path = Path(args.out_dir) / "01_radial" / "radius_attempt_summary.csv"
    radius_summary = pd.read_csv(radius_summary_path) if radius_summary_path.exists() else pd.DataFrame()
    lines = [
        "# True Ellipse Radial Bundle V6 summary",
        "",
        f"- Fixed family: `{args.family_id}`.",
        f"- Formal 100 mm radial gate: `{report['radial_100mm_gate_pass']}`.",
        f"- Formal 100 mm tube gate: `{report['tube_100mm_gate_pass']}`.",
        f"- Whole-radius dataset gate: `{report['dataset_gate_pass']}`.",
        f"- Training-only support at 100 mm: `{report['training_only_support_100mm_pass']}`.",
        f"- V2–V5 artifacts unchanged: `{legacy_unchanged}`.",
        f"- Model training authorized: `{report['model_training_authorized']}`.",
        "- No strict model-radius claim is made by this upstream generator.",
        "",
        "## Radial attempts",
        "",
        dataframe_to_markdown(radius_summary),
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lift one fixed true-ellipse family to 100 mm with a full-parent radial path bundle."
    )
    parser.add_argument("--v5-dir", type=Path, default=DEFAULT_V5_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phases", default="all")
    parser.add_argument("--family-id", default=PRIMARY_FAMILY_ID)
    parser.add_argument("--previous-radius-mm", type=float, default=75.0)
    parser.add_argument("--start-radius-mm", type=float, default=80.0)
    parser.add_argument("--target-radius-mm", type=float, default=100.0)
    parser.add_argument(
        "--radius-checkpoints-mm",
        default="82.5,85,87.5,90,92.5,95,97.5,100",
    )
    parser.add_argument("--cut-indices", default="0,90,180,270")
    parser.add_argument("--anchor-schedules", default="conservative,balanced,loose")
    parser.add_argument("--base-step-mm", type=float, default=1.0)
    parser.add_argument("--retry-steps-mm", default="0.5,0.25")
    parser.add_argument("--final-points", type=int, default=360)
    parser.add_argument("--max-candidates-per-angle", type=int, default=8)
    parser.add_argument("--candidate-cluster-deg", type=float, default=0.25)
    parser.add_argument("--candidate-residual-mm", type=float, default=2.0)
    parser.add_argument("--tube-offsets-mm", default="-5,-2.5,0,2.5,5")
    parser.add_argument("--max-opt-nfev", type=int, default=40)
    parser.add_argument("--max-ik-nfev", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--family-search-sobol-samples", type=int, default=8192)
    parser.add_argument("--family-search-top-support", type=int, default=64)
    parser.add_argument("--family-search-top-pointwise", type=int, default=16)
    parser.add_argument("--family-search-top-formal", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-existing", action="store_true")
    parser.set_defaults(
        stop_after_first_failed_job=False,
        share_rescue_cache_across_cuts=False,
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    phases = parse_phases(args.phases)
    handlers = {
        "audit": phase_audit,
        "radial": phase_radial,
        "tube": phase_tube,
        "dataset": phase_dataset,
        "summary": phase_summary,
    }
    results: dict[str, Any] = {}
    for phase in phases:
        results[phase] = handlers[phase](args)
    report = {
        "mode": "true_ellipse_radial_bundle_v6",
        "phases": phases,
        "out_dir": str(Path(args.out_dir).resolve()),
        **formal_protocol_report(args),
        "results": results,
    }
    write_json(Path(args.out_dir) / "run_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
