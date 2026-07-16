#!/usr/bin/env python3
"""Run the registered standard-domain true-ellipse V7 experiment.

V7 is an adapter over the shared radial/tube engine.  It expands beta3/beta4
to the repository's standard ±10 degree sampling domain, requires learnable
joint-margin labels, searches one fixed family through 120 mm, and keeps
strict geometry, exploratory rescue, and fitted-model claims separate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_family_expansion_v5 as v5_expansion  # noqa: E402
import run_true_ellipse_family_training_v5 as v5_training  # noqa: E402
import run_true_ellipse_radial_bundle_v6 as v6_runner  # noqa: E402
import true_ellipse_radial_bundle_engine as engine  # noqa: E402
import true_ellipse_radial_bundle_v6_utils as v6_utils  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_ellipse_family_v5_utils import stable_fingerprint  # noqa: E402


PRIMARY_FAMILY_ID = "c0273_a100_py210_pz330_s0243"
DEFAULT_V5_DIR = PROJECT_ROOT / "runs" / "true_ellipse_family_expansion_v5"
DEFAULT_V6_DIR = PROJECT_ROOT / "runs" / "true_ellipse_radial_bundle_v6"
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_v7"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"
ALL_PHASES = ("audit", "radial", "tube", "dataset", "summary")
RUNNER_STRATEGY_VERSION = 1
TUBE_STRATEGY_VERSION = 3
DATASET_STRATEGY_VERSION = 1

write_json = v6_utils.write_json
read_json = v6_utils.read_json
file_sha256 = v6_utils.file_sha256
_json_default = v6_utils.json_default


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
        raise ValueError(f"unsupported V7 phases: {unknown}")
    return phases


def _domain() -> engine.JointDomainSpec:
    return engine.registered_joint_domain("standard_beta34_10deg_v1")


def _margin_policy() -> engine.JointMarginPolicy:
    return engine.balanced_joint_margin_policy()


def formal_protocol_report(args: argparse.Namespace) -> dict[str, Any]:
    registered = engine.v7_standard_domain_protocol()
    domain = engine.registered_joint_domain(str(args.joint_domain_id))
    checkpoints = tuple(parse_float_csv(args.radius_checkpoints_mm))
    cuts = tuple(parse_int_csv(args.cut_indices))
    anchors = tuple(parse_name_csv(args.anchor_schedules))
    checks = {
        "formal_preset": str(args.preset) == "formal",
        "fixed_family": str(args.family_id) == registered.family_id,
        "standard_joint_domain": str(args.joint_domain_id) == registered.joint_domain_id,
        "target_120mm": np.isclose(float(args.target_radius_mm), registered.target_radius_mm),
        "formal_checkpoints": checkpoints == registered.formal_checkpoints_mm,
        "four_cyclic_cuts": cuts == registered.cut_indices,
        "all_anchor_schedules": anchors == tuple(v6_utils.ANCHOR_SCHEDULES),
        "base_step_1mm": np.isclose(float(args.base_step_mm), registered.base_step_mm),
        "retry_half_quarter": tuple(parse_float_csv(args.retry_steps_mm)) == registered.retry_steps_mm,
        "final_points_360": int(args.final_points) == 360,
        "strict_candidates_eight": int(args.max_candidates_per_angle) == registered.strict_candidate_limit,
        "rescue_candidates_24": int(args.rescue_candidates_per_angle) == registered.rescue_candidate_limit,
        "balanced_margin_required": bool(args.require_joint_margin_gate)
        and np.isclose(float(args.soft_margin_deg), _margin_policy().soft_barrier_margin_deg),
        "registered_margin_weight": np.isclose(float(args.lambda_margin), 1.0e-2),
        "formal_tube_offsets": tuple(parse_float_csv(args.tube_offsets_mm)) == v6_utils.FORMAL_TUBE_OFFSETS_MM,
    }
    normalized = {key: bool(value) for key, value in checks.items()}
    protocol = {
        **registered.as_dict(),
        "family_id": str(args.family_id),
        "joint_domain_id": str(args.joint_domain_id),
        "start_radius_mm": float(args.start_radius_mm),
        "target_radius_mm": float(args.target_radius_mm),
        "formal_checkpoints_mm": list(checkpoints),
        "base_step_mm": float(args.base_step_mm),
        "retry_steps_mm": parse_float_csv(args.retry_steps_mm),
        "strict_candidate_limit": int(args.max_candidates_per_angle),
        "rescue_candidate_limit": int(args.rescue_candidates_per_angle),
        "preset": str(args.preset),
        "cut_indices": list(cuts),
        "anchor_schedules": list(anchors),
        "tube_offsets_mm": parse_float_csv(args.tube_offsets_mm),
        "final_points": int(args.final_points),
        "lambda_margin": float(args.lambda_margin),
        "soft_margin_deg": float(args.soft_margin_deg),
        "joint_domain_fingerprint": domain.fingerprint,
        "joint_margin_policy_fingerprint": _margin_policy().fingerprint,
    }
    return {
        "formal_protocol_gate_pass": bool(all(normalized.values())),
        "checks": normalized,
        "protocol": protocol,
        "protocol_fingerprint": stable_fingerprint(protocol),
        "joint_domain_id": domain.domain_id,
        "joint_domain_fingerprint": domain.fingerprint,
        "joint_margin_policy_id": _margin_policy().policy_id,
        "joint_margin_policy_fingerprint": _margin_policy().fingerprint,
    }


def rescue_admission_gate(metrics: Mapping[str, Any]) -> bool:
    """Admission to expensive rescue only; never a strict acceptance gate."""

    return bool(
        metrics.get("finite", False)
        and metrics.get("in_domain", False)
        and float(metrics.get("success_ratio_le2mm", -np.inf)) >= 0.80
        and float(metrics.get("residual_max_mm", np.inf)) <= 20.0
    )


def materialized_dataset_radii(strict_geometry_rmax_mm: float) -> tuple[float, ...]:
    protocol = engine.v7_standard_domain_protocol()
    checkpoints = [
        float(radius)
        for radius in protocol.formal_checkpoints_mm
        if float(radius) <= float(strict_geometry_rmax_mm) + 1.0e-9
    ]
    radii = sorted(set(checkpoints) | set(engine.training_anchor_radii(strict_geometry_rmax_mm)))
    return tuple(radii)


def resample_parent_curve(parent: pd.DataFrame, *, n_points: int) -> pd.DataFrame:
    """Select an exact cyclic sub-grid while giving it local angle indices."""

    if "angle_idx" not in parent or "angle_rad" not in parent:
        raise ValueError("parent resampling requires angle_idx and angle_rad")
    ordered = parent.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    target_count = int(n_points)
    if target_count <= 0 or target_count > len(ordered):
        raise ValueError("parent resampling requires 0 < n_points <= source rows")
    positions = np.floor(
        np.linspace(0.0, float(len(ordered)), target_count, endpoint=False)
    ).astype(np.int64)
    if len(set(positions.tolist())) != target_count:
        raise ValueError("parent resampling did not produce a unique cyclic grid")
    sampled = ordered.iloc[positions].copy().reset_index(drop=True)
    sampled["source_angle_idx"] = sampled["angle_idx"].astype(int)
    sampled["angle_idx"] = np.arange(target_count, dtype=np.int64)
    return sampled


def generate_half_phase_targets(
    family: v6_utils.FamilySpec,
    *,
    radius_mm: float,
    n_points: int = 360,
) -> pd.DataFrame:
    n = int(n_points)
    if n <= 0:
        raise ValueError("half-phase challenge requires a positive resolution")
    angle = (np.arange(n, dtype=float) + 0.5) * (2.0 * math.pi / n)
    radius_m = float(radius_mm) / 1000.0
    output = pd.DataFrame(
        {
            "family_id": str(family.family_id),
            "candidate_id": str(family.family_id),
            "angle_idx": np.arange(n, dtype=int),
            "angle_rad": angle,
            "center_x_m": float(family.center_x_m),
            "center_y_m": float(family.center_y_m),
            "center_z_m": float(family.center_z_m),
            "amp_xy_mm": float(radius_mm),
            "amp_z_mm": 1.5 * float(radius_mm),
            "phase_y_rad": float(family.phase_y_rad),
            "phase_z_rad": float(family.phase_z_rad),
            "radius_mm": float(radius_mm),
        }
    )
    output["x_target_m"] = float(family.center_x_m) + radius_m * np.sin(angle)
    output["y_target_m"] = float(family.center_y_m) + radius_m * np.sin(
        angle + float(family.phase_y_rad)
    )
    output["z_target_m"] = float(family.center_z_m) + 1.5 * radius_m * np.sin(
        angle + float(family.phase_z_rad)
    )
    return output


def build_half_phase_predictor(
    centerline: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Interpolate a periodic integer-phase label curve onto half phases."""

    required_center = {"angle_idx", *v6_utils.atlas.BETA_COLS}
    required_targets = {"angle_idx", "angle_rad", *v6_utils.atlas.TARGET_XYZ_COLS}
    missing_center = sorted(required_center - set(centerline.columns))
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_center:
        raise ValueError(f"half-phase centerline missing columns: {missing_center}")
    if missing_targets:
        raise ValueError(f"half-phase targets missing columns: {missing_targets}")
    center = centerline.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    target = targets.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    if len(center) != len(target) or len(center) == 0:
        raise ValueError("half-phase interpolation requires equal non-empty phase curves")
    expected = list(range(len(center)))
    if center["angle_idx"].astype(int).tolist() != expected:
        raise ValueError("half-phase centerline must contain one complete cyclic phase grid")
    if target["angle_idx"].astype(int).tolist() != expected:
        raise ValueError("half-phase targets must contain one complete cyclic phase grid")
    beta = center[v6_utils.atlas.BETA_COLS].to_numpy(dtype=float)
    predictor = target.copy()
    interpolated = 0.5 * (beta + np.roll(beta, -1, axis=0))
    for index, column in enumerate(v6_utils.atlas.BETA_COLS):
        predictor[column] = interpolated[:, index]
    predictor["radial_predictor_type"] = "cyclic_half_phase_average"
    return predictor


def assign_dynamic_radius_splits(
    dataset: pd.DataFrame,
    *,
    validation_radius_mm: float,
    test_radius_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"sample_id", "family_id", "trajectory_id", "radius_mm", "is_centerline"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"dynamic split dataset missing columns: {missing}")
    if not pd.api.types.is_bool_dtype(dataset["is_centerline"].dtype):
        raise ValueError("dynamic split requires a boolean is_centerline column")
    assigned = dataset.copy()
    radius = assigned["radius_mm"].to_numpy(dtype=float)
    assigned["split"] = "train"
    assigned.loc[np.isclose(radius, float(validation_radius_mm), atol=1.0e-8), "split"] = "validation"
    assigned.loc[np.isclose(radius, float(test_radius_mm), atol=1.0e-8), "split"] = "test"
    assigned["used_for_training"] = assigned["split"].eq("train") & ~assigned["is_centerline"]
    trajectory_leakage = int((assigned.groupby("trajectory_id")["split"].nunique() > 1).sum())
    radius_leakage = int((assigned.groupby("radius_mm")["split"].nunique() > 1).sum())
    validation_present = bool(np.any(np.isclose(radius, float(validation_radius_mm), atol=1.0e-8)))
    test_present = bool(np.any(np.isclose(radius, float(test_radius_mm), atol=1.0e-8)))
    report = {
        "rows": int(len(assigned)),
        "validation_radius_mm": float(validation_radius_mm),
        "test_radius_mm": float(test_radius_mm),
        "validation_present": validation_present,
        "test_present": test_present,
        "training_rows": int(assigned["used_for_training"].sum()),
        "training_radius_count": int(
            assigned.loc[assigned["used_for_training"], "radius_mm"].nunique()
        ),
        "trajectory_leakage_count": trajectory_leakage,
        "radius_leakage_count": radius_leakage,
        "unique_family_count": int(assigned["family_id"].astype(str).nunique()),
    }
    report["split_gate_pass"] = bool(
        len(assigned)
        and assigned["sample_id"].is_unique
        and validation_present
        and test_present
        and report["training_rows"] > 0
        and report["training_radius_count"] >= 2
        and trajectory_leakage == 0
        and radius_leakage == 0
        and report["unique_family_count"] == 1
    )
    return assigned, report


def formal_tube_label_gate(metrics: Mapping[str, Any]) -> bool:
    return bool(
        int(metrics.get("rows", 0)) == int(metrics.get("expected_rows", -1))
        and float(metrics.get("tube_success_ratio", -np.inf)) == 1.0
        and bool(metrics.get("surface_gate_pass", False))
        and bool(metrics.get("joint_margin_gate_pass", False))
        and bool(metrics.get("tube_gate_pass", False))
    )


def compute_holdout_support_candidates(
    dataset: pd.DataFrame,
    *,
    candidate_test_radii_mm: Iterable[float],
    validation_gap_mm: float = 7.5,
    support_radius_mm: float = 15.0,
) -> pd.DataFrame:
    """Audit each dynamic holdout pair using only its prospective training pool."""

    required = {"radius_mm", "is_centerline", *v6_utils.atlas.TARGET_XYZ_COLS}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"holdout support dataset missing columns: {missing}")
    radii = dataset["radius_mm"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for test_radius in sorted(set(parse_float_csv(candidate_test_radii_mm)), reverse=True):
        validation_radius = float(test_radius) - float(validation_gap_mm)
        holdout_mask = np.isclose(radii, float(test_radius), atol=1.0e-8)
        holdout_mask |= np.isclose(radii, validation_radius, atol=1.0e-8)
        training_mask = ~holdout_mask & ~dataset["is_centerline"].astype(bool).to_numpy()
        training_xyz = dataset.loc[training_mask, v6_utils.atlas.TARGET_XYZ_COLS].to_numpy(
            dtype=float
        )
        for role, radius_mm in (("validation", validation_radius), ("test", float(test_radius))):
            target_mask = np.isclose(radii, radius_mm, atol=1.0e-8)
            center_mask = target_mask & dataset["is_centerline"].astype(bool).to_numpy()
            selected_mask = center_mask if np.any(center_mask) else target_mask
            target_rows = dataset.loc[selected_mask].copy()
            if "angle_idx" in target_rows:
                target_rows = target_rows.sort_values("angle_idx", kind="stable").drop_duplicates(
                    "angle_idx", keep="first"
                )
            target_xyz = target_rows[v6_utils.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
            if len(training_xyz) and len(target_xyz):
                support = v5_training.v4.compute_support_metrics(
                    training_xyz,
                    target_xyz,
                    radius_mm=float(support_radius_mm),
                )
                support_pass = v5_training.strict_support_gate(support)
            else:
                support = {
                    "nn_mean_mm": float("inf"),
                    "nn_p95_mm": float("inf"),
                    "nn_max_mm": float("inf"),
                    "tube_count_p10": 0.0,
                    "tube_count_median": 0.0,
                }
                support_pass = False
            rows.append(
                {
                    "test_radius_mm": float(test_radius),
                    "validation_radius_mm": validation_radius,
                    "role": role,
                    "radius_mm": radius_mm,
                    "training_rows": int(training_mask.sum()),
                    "target_rows": int(len(target_xyz)),
                    "held_out_radius_excluded": bool(not np.any(training_mask & target_mask)),
                    **support,
                    "strict_support_gate_pass": bool(support_pass),
                }
            )
    return pd.DataFrame(rows)


def formal_dataset_gate(checks: Mapping[str, Any]) -> bool:
    required = (
        "formal_protocol_gate_pass",
        "formal_tube_gate_pass",
        "completeness_gate_pass",
        "split_gate_pass",
        "branch_conflict_gate_pass",
        "joint_margin_gate_pass",
        "holdout_selection_gate_pass",
        "validation_support_gate_pass",
        "test_support_gate_pass",
        "challenge_gate_pass",
    )
    return bool(all(bool(checks.get(name, False)) for name in required))


def extract_tube_centerline(tube: pd.DataFrame, *, expected_points: int) -> pd.DataFrame:
    required = {"angle_idx", "is_centerline", *v6_utils.atlas.BETA_COLS}
    missing = sorted(required - set(tube.columns))
    if missing:
        raise ValueError(f"tube centerline extraction missing columns: {missing}")
    centerline = tube.loc[tube["is_centerline"].astype(bool)].sort_values(
        "angle_idx", kind="stable"
    ).reset_index(drop=True)
    expected = list(range(int(expected_points)))
    if centerline["angle_idx"].astype(int).tolist() != expected:
        raise ValueError("final tube does not contain exactly one centerline row per phase")
    if "sample_id" in centerline and not centerline["sample_id"].is_unique:
        raise ValueError("final tube centerline sample IDs are not unique")
    return centerline


def _preset_settings(preset: str) -> dict[str, Any]:
    if str(preset) == "smoke":
        return {"final_points": 36, "cut_indices": (0,), "max_opt_nfev": 8}
    if str(preset) == "pilot":
        return {"final_points": 72, "cut_indices": (0, 18, 36, 54), "max_opt_nfev": 20}
    if str(preset) == "formal":
        return {"final_points": 360, "cut_indices": (0, 90, 180, 270), "max_opt_nfev": 40}
    raise ValueError(f"unsupported preset: {preset}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registered = engine.v7_standard_domain_protocol()
    parser = argparse.ArgumentParser(
        description="Search and materialize the fixed-family V7 standard-domain ellipse bundle."
    )
    parser.add_argument("--preset", choices=("smoke", "pilot", "formal"), default="formal")
    parser.add_argument("--v5-dir", type=Path, default=DEFAULT_V5_DIR)
    parser.add_argument("--v6-dir", type=Path, default=DEFAULT_V6_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phases", default="all")
    parser.add_argument("--family-id", default=registered.family_id)
    parser.add_argument("--joint-domain-id", default=registered.joint_domain_id)
    parser.add_argument("--previous-radius-mm", type=float, default=75.0)
    parser.add_argument("--start-radius-mm", type=float, default=75.0)
    parser.add_argument("--target-radius-mm", type=float, default=registered.target_radius_mm)
    parser.add_argument(
        "--radius-checkpoints-mm",
        default=",".join(f"{value:g}" for value in registered.formal_checkpoints_mm),
    )
    parser.add_argument("--cut-indices", default=",".join(str(value) for value in registered.cut_indices))
    parser.add_argument("--anchor-schedules", default=",".join(v6_utils.ANCHOR_SCHEDULES))
    parser.add_argument("--base-step-mm", type=float, default=registered.base_step_mm)
    parser.add_argument("--retry-steps-mm", default=",".join(f"{value:g}" for value in registered.retry_steps_mm))
    parser.add_argument("--final-points", type=int, default=360)
    parser.add_argument("--max-candidates-per-angle", type=int, default=registered.strict_candidate_limit)
    parser.add_argument("--rescue-candidates-per-angle", type=int, default=registered.rescue_candidate_limit)
    parser.add_argument("--candidate-cluster-deg", type=float, default=0.25)
    parser.add_argument("--candidate-residual-mm", type=float, default=2.0)
    parser.add_argument("--tube-offsets-mm", default="-5,-2.5,0,2.5,5")
    parser.add_argument("--max-opt-nfev", type=int, default=40)
    parser.add_argument("--max-ik-nfev", type=int, default=200)
    parser.add_argument("--lambda-margin", type=float, default=1.0e-2)
    parser.add_argument("--soft-margin-deg", type=float, default=0.25)
    parser.add_argument(
        "--require-joint-margin-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    # V6's shared radial interface fingerprints these legacy budgets.
    parser.add_argument("--family-search-sobol-samples", type=int, default=8192)
    parser.add_argument("--family-search-top-support", type=int, default=64)
    parser.add_argument("--family-search-top-pointwise", type=int, default=16)
    parser.add_argument("--family-search-top-formal", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--skip-existing", action="store_true")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)

    def explicitly_set(option: str) -> bool:
        return any(token == option or token.startswith(f"{option}=") for token in raw_argv)

    settings = _preset_settings(str(args.preset))
    if not explicitly_set("--final-points"):
        args.final_points = int(settings["final_points"])
    if not explicitly_set("--cut-indices"):
        args.cut_indices = ",".join(str(value) for value in settings["cut_indices"])
    if not explicitly_set("--max-opt-nfev"):
        args.max_opt_nfev = int(settings["max_opt_nfev"])
    return args


def _load_robot(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    config = load_config(str(args.robot_config))
    inputs = load_robot_inputs(config)
    theta_sign = float(config.get("kinematics", {}).get("theta_sign", -1.0))
    return inputs.lengths_m, inputs.p_end_local_m, theta_sign


def phase_audit(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "00_audit"
    out.mkdir(parents=True, exist_ok=True)
    protocol = formal_protocol_report(args)
    source_paths = {
        "v6_audit": Path(args.v6_dir) / "00_audit" / "audit_report.json",
        "v6_radial": Path(args.v6_dir) / "01_radial" / "radial_report.json",
        "v6_tube": Path(args.v6_dir) / "02_tube" / "tube_report.json",
        "v6_dataset": Path(args.v6_dir) / "03_dataset" / "dataset_report.json",
        "v6_checkpoint": REPO_ROOT / "docs" / "checkpoints" / "2026-07-15-true-ellipse-radial-bundle-v6.md",
        "robot_config": Path(args.robot_config),
    }
    missing = sorted(name for name, path in source_paths.items() if not path.is_file())
    source_reports = {
        name: read_json(path)
        for name, path in source_paths.items()
        if name.startswith("v6_") and path.is_file() and path.suffix == ".json"
    }
    domain = engine.registered_joint_domain(str(args.joint_domain_id))
    config = load_config(str(args.robot_config)) if Path(args.robot_config).is_file() else {}
    configured = config.get("sampling", {}).get("beta_ranges_rad", {})
    configured_bounds = np.asarray(
        [configured.get(f"beta{index}", [np.nan, np.nan]) for index in range(1, 7)],
        dtype=float,
    )
    domain_matches_config = bool(
        configured_bounds.shape == (6, 2)
        and np.isfinite(configured_bounds).all()
        and np.allclose(configured_bounds, domain.bounds_rad, atol=1.0e-12, rtol=0.0)
    )
    checks = {
        "sources_complete": not missing,
        "v6_audit_pass": bool(source_reports.get("v6_audit", {}).get("formal_audit_gate_pass", False)),
        "v6_radial_pass": bool(source_reports.get("v6_radial", {}).get("formal_radial_gate_pass", False)),
        "v6_tube_pass": bool(source_reports.get("v6_tube", {}).get("formal_tube_gate_pass", False)),
        "v6_dataset_pass": bool(source_reports.get("v6_dataset", {}).get("formal_dataset_gate_pass", False)),
        "standard_domain_matches_config": domain_matches_config,
        "fixed_family": str(args.family_id) == PRIMARY_FAMILY_ID,
    }
    manifest = {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for name, path in source_paths.items()
        if path.is_file()
    }
    report = {
        "strategy_version": RUNNER_STRATEGY_VERSION,
        **protocol,
        "source_manifest": manifest,
        "missing_sources": missing,
        "checks_recomputed": checks,
        "audit_gate_pass": bool(all(checks.values())),
    }
    report["formal_audit_gate_pass"] = bool(
        report["audit_gate_pass"] and report["formal_protocol_gate_pass"]
    )
    report["task_fingerprint"] = stable_fingerprint(
        {"protocol": report["protocol_fingerprint"], "sources": manifest}
    )
    write_json(out / "audit_report.json", report)
    return report


def ensure_audit_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "00_audit" / "audit_report.json"
    expected = formal_protocol_report(args)["protocol_fingerprint"]
    if path.exists():
        cached = read_json(path)
        if str(cached.get("protocol_fingerprint", "")) == str(expected):
            return cached
    return phase_audit(args)


def _pointwise_exploratory_report(
    *,
    family: v6_utils.FamilySpec,
    radius_mm: float,
    seed_path: pd.DataFrame,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int,
    n_points: int = 72,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    domain = _domain()
    targets = v6_utils.generate_radius_targets(
        family,
        radius_mm=float(radius_mm),
        n_points=int(n_points),
    )
    ordered_seed = seed_path.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    seed_positions = np.linspace(0, len(ordered_seed), len(targets), endpoint=False).astype(int)
    seed_beta = ordered_seed.iloc[seed_positions][v6_utils.atlas.BETA_COLS].to_numpy(dtype=float)
    records: list[dict[str, Any]] = []
    for position, target_row in targets.iterrows():
        solution = v6_utils.atlas.solve_beta_ik(
            target_row[v6_utils.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float),
            init_betas=seed_beta[position].reshape(1, 6),
            bounds=domain.bounds_rad,
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
            max_nfev=int(max_nfev),
            lambda_limit=1.0e-2,
        )
        row = target_row.to_dict()
        row["xyz_residual_mm"] = float(solution.residual_mm)
        for index, column in enumerate(v6_utils.atlas.BETA_COLS):
            row[column] = float(solution.beta_rad[index])
        records.append(row)
    candidates = pd.DataFrame(records)
    beta = candidates[v6_utils.atlas.BETA_COLS].to_numpy(dtype=float)
    residual = candidates["xyz_residual_mm"].to_numpy(dtype=float)
    bounds = domain.bounds_rad
    finite = bool(np.isfinite(beta).all() and np.isfinite(residual).all())
    in_domain = bool(
        finite
        and np.all(beta >= bounds[:, 0][None, :] - 1.0e-12)
        and np.all(beta <= bounds[:, 1][None, :] + 1.0e-12)
    )
    report = {
        "radius_mm": float(radius_mm),
        "rows": int(len(candidates)),
        "finite": finite,
        "in_domain": in_domain,
        "success_ratio_le2mm": float(np.mean(residual <= 2.0)),
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
    }
    report["rescue_admission_pass"] = rescue_admission_gate(report)
    return candidates, report


def phase_radial(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_radial"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    if not bool(audit.get("audit_gate_pass", False)):
        report = {
            "strategy_version": RUNNER_STRATEGY_VERSION,
            "strict_geometry_rmax_mm": None,
            "exploratory_rescue_rmax_mm": None,
            "formal_radial_gate_pass": False,
            "reason": "audit_gate_failed",
        }
        write_json(out / "radial_report.json", report)
        return report

    family = v6_runner.load_family_spec(args.v5_dir, args.family_id)
    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    source_radius = float(args.start_radius_mm)
    source_path = v6_runner.v5_centerline_path(args.v5_dir, family.family_id, source_radius)
    source = v6_runner.load_v5_parent_path(args.v5_dir, family.family_id, source_radius)
    source = resample_parent_curve(source, n_points=int(args.final_points))
    relabeled, relabel_report, relabel_artifact = v6_runner._solve_radial_radius(
        args,
        audit_report=audit,
        family=family,
        parent_path=source,
        parent_radius_mm=source_radius,
        parent_artifact_path=source_path,
        previous_parent=None,
        previous_parent_radius_mm=None,
        previous_parent_artifact_path=None,
        target_radius_mm=source_radius,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    history: dict[float, pd.DataFrame] = {}
    artifact_paths: dict[float, Path] = {}
    if bool(relabel_report.get("radial_bundle_gate_pass", False)):
        history[source_radius] = relabeled
        artifact_paths[source_radius] = relabel_artifact

    attempts: list[dict[str, Any]] = [
        {
            "parent_radius_mm": source_radius,
            "target_radius_mm": source_radius,
            "step_mm": 0.0,
            "step_kind": "standard_domain_relabel",
            "radial_bundle_gate_pass": bool(relabel_report.get("radial_bundle_gate_pass", False)),
            "report": relabel_report,
        }
    ]
    walk: dict[str, Any]
    if source_radius not in history:
        walk = {
            "radial_walk_gate_pass": False,
            "last_pass_radius_mm": source_radius,
            "failed_checkpoint_mm": source_radius,
            "attempts": [],
        }
    else:

        def solve_step(
            parent_path: pd.DataFrame,
            parent_radius_mm: float,
            target_radius_mm: float,
        ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
            lower = sorted(radius for radius in history if radius < float(parent_radius_mm) - 1.0e-10)
            previous_radius = lower[-1] if lower else None
            solved, raw_report, artifact = v6_runner._solve_radial_radius(
                args,
                audit_report=audit,
                family=family,
                parent_path=parent_path,
                parent_radius_mm=float(parent_radius_mm),
                parent_artifact_path=artifact_paths[float(parent_radius_mm)],
                previous_parent=None if previous_radius is None else history[previous_radius],
                previous_parent_radius_mm=previous_radius,
                previous_parent_artifact_path=(
                    None if previous_radius is None else artifact_paths[previous_radius]
                ),
                target_radius_mm=float(target_radius_mm),
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
            )
            if bool(raw_report.get("radial_bundle_gate_pass", False)):
                history[float(target_radius_mm)] = solved
                artifact_paths[float(target_radius_mm)] = artifact
            return solved, raw_report

        checkpoints = [
            radius
            for radius in parse_float_csv(args.radius_checkpoints_mm)
            if radius > source_radius + 1.0e-9
        ]
        _passed, walk = v6_utils.adaptive_radius_walk(
            history[source_radius],
            start_radius_mm=source_radius,
            checkpoints_mm=checkpoints,
            family_id=family.family_id,
            solve_step=solve_step,
            base_step_mm=float(args.base_step_mm),
            retry_steps_mm=parse_float_csv(args.retry_steps_mm),
        )
        attempts.extend(walk.get("attempts", []))

    last_pass = max(history) if history else source_radius
    seed_path = history.get(last_pass, source)
    formal_checkpoints = tuple(parse_float_csv(args.radius_checkpoints_mm))
    status_rows: list[dict[str, Any]] = []
    exploratory_dir = out / "exploratory_pointwise"
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    for radius in formal_checkpoints:
        strict = any(np.isclose(radius, passed, atol=1.0e-8) for passed in history)
        if strict:
            status_rows.append(
                {
                    "radius_mm": float(radius),
                    "strict_gate_pass": True,
                    "rescue_admission_pass": True,
                    "status_kind": "strict_path_bundle",
                }
            )
            continue
        candidates, exploratory = _pointwise_exploratory_report(
            family=family,
            radius_mm=float(radius),
            seed_path=seed_path,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=theta_sign,
            max_nfev=int(args.max_ik_nfev),
            n_points=min(72, int(args.final_points)),
        )
        slug = v6_runner.radius_slug(radius)
        candidates.to_parquet(exploratory_dir / f"{slug}.parquet", index=False, compression="zstd")
        write_json(exploratory_dir / f"{slug}.json", exploratory)
        status_rows.append(
            {
                "radius_mm": float(radius),
                "strict_gate_pass": False,
                "status_kind": "exploratory_pointwise",
                **exploratory,
            }
        )
    status = pd.DataFrame(status_rows).sort_values("radius_mm").reset_index(drop=True)
    status.to_csv(out / "radius_status.csv", index=False)
    frontiers = engine.compute_radius_frontiers(status, anchor_mm=source_radius)
    path_manifest = {
        f"{radius:g}": {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "branch_hash": v6_utils.atlas.branch_hash(history[radius]),
        }
        for radius, path in sorted(artifact_paths.items())
    }
    write_json(out / "strict_path_manifest.json", path_manifest)
    strict_rmax = frontiers["strict_geometry_rmax_mm"]
    report = {
        "strategy_version": RUNNER_STRATEGY_VERSION,
        **formal_protocol_report(args),
        "audit_task_fingerprint": str(audit.get("task_fingerprint", "")),
        "family_id": family.family_id,
        "attempt_count": int(len(attempts)),
        "attempts": attempts,
        "walk": walk,
        "path_manifest": path_manifest,
        "path_manifest_path": str((out / "strict_path_manifest.json").resolve()),
        "radius_status_path": str((out / "radius_status.csv").resolve()),
        **frontiers,
    }
    report["formal_radial_gate_pass"] = bool(
        report["formal_protocol_gate_pass"]
        and strict_rmax is not None
        and float(strict_rmax) >= 105.0
    )
    report["task_fingerprint"] = stable_fingerprint(
        {
            "audit": report["audit_task_fingerprint"],
            "protocol": report["protocol_fingerprint"],
            "paths": path_manifest,
            "status": status_rows,
        }
    )
    write_json(out / "radial_report.json", report)
    return report


def ensure_radial_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "01_radial" / "radial_report.json"
    expected = formal_protocol_report(args)["protocol_fingerprint"]
    if path.exists():
        cached = read_json(path)
        if str(cached.get("protocol_fingerprint", "")) == str(expected):
            return cached
    return phase_radial(args)


def _surface_stage_specs(lambda_margin: float, soft_margin_deg: float) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            **stage,
            "lambda_margin": float(lambda_margin),
            "soft_margin_deg": float(soft_margin_deg),
        }
        for stage in v6_utils.anchor_stages("balanced")
    )


def run_surface_stage_schedule(
    *,
    initial_surface: Any,
    stage_specs: Sequence[Mapping[str, Any]],
    solve_stage: Callable[[Any, Mapping[str, Any]], tuple[Any, Mapping[str, Any]]],
    assess_stage: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Advance one surface stage at a time and stop on the first full gate pass."""

    stages = tuple(stage_specs)
    if not stages:
        raise ValueError("surface stage schedule must not be empty")
    current = initial_surface
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for stage in stages:
        current, raw_report = solve_stage(current, stage)
        selected = assess_stage(current, raw_report, stage)
        attempts.append(selected)
        if bool(selected.get("formal_tube_label_gate_pass", False)):
            break
    return current, selected, attempts


def _initial_tube_surface(
    *,
    centerline: pd.DataFrame,
    tube_targets: pd.DataFrame,
    offsets_mm: Sequence[float],
    bounds: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    max_nfev: int,
) -> pd.DataFrame:
    del max_nfev  # The initializer is a predictor; formal correction happens surface-wise.
    center = centerline.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    center_beta = center[v6_utils.atlas.BETA_COLS].to_numpy(dtype=float)
    center_xyz = center[v6_utils.atlas.XYZ_COLS].to_numpy(dtype=float)
    pinv = np.asarray(
        [
            v6_utils.atlas.weighted_damped_pinv(
                v6_utils.atlas.numerical_jacobian_beta(
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
    pairs = engine.tube_surface_sweep_order(offsets_mm, direction="outward")
    center_pair = (0.0, 0.0)
    solved: dict[tuple[float, float], pd.DataFrame] = {}
    adjacency = engine.tube_surface_adjacency(offsets_mm)
    for pair in pairs:
        mask = np.isclose(tube_targets["delta_n1_mm"].to_numpy(dtype=float), pair[0])
        mask &= np.isclose(tube_targets["delta_n2_mm"].to_numpy(dtype=float), pair[1])
        curve_targets = tube_targets.loc[mask].sort_values("angle_idx", kind="stable").reset_index(drop=True)
        target_xyz = curve_targets[v6_utils.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
        if pair == center_pair:
            beta_prediction = center_beta.copy()
            parent_pair = center_pair
        else:
            neighbor_pairs = [
                right if left == pair else left
                for left, right in adjacency
                if (left == pair and right in solved) or (right == pair and left in solved)
            ]
            parent_pair = min(
                neighbor_pairs or [center_pair],
                key=lambda item: math.hypot(item[0] - pair[0], item[1] - pair[1]),
            )
            direct = center_beta + np.einsum(
                "nij,nj->ni",
                pinv,
                target_xyz - center_xyz,
            )
            parent = solved[parent_pair].sort_values("angle_idx", kind="stable").reset_index(
                drop=True
            )
            parent_beta = parent[v6_utils.atlas.BETA_COLS].to_numpy(dtype=float)
            parent_target = parent[v6_utils.atlas.TARGET_XYZ_COLS].to_numpy(dtype=float)
            incremental = parent_beta + np.einsum(
                "nij,nj->ni",
                pinv,
                target_xyz - parent_target,
            )
            beta_prediction = 0.5 * direct + 0.5 * incremental
        beta_prediction = np.clip(
            beta_prediction,
            np.asarray(bounds, dtype=float)[:, 0],
            np.asarray(bounds, dtype=float)[:, 1],
        )
        theta = v6_utils.atlas.theta_from_beta_batch(
            beta_prediction,
            theta_sign=float(theta_sign),
        )
        achieved = v6_utils.atlas.fk_from_beta_batch(
            beta_prediction,
            lengths_m=np.asarray(lengths_m, dtype=float),
            p_end_local_m=np.asarray(p_end_local_m, dtype=float),
            theta_sign=float(theta_sign),
        )
        curve = curve_targets.copy()
        for index, column in enumerate(v6_utils.atlas.BETA_COLS):
            curve[column] = beta_prediction[:, index]
        for index, column in enumerate(v6_utils.atlas.THETA_COLS):
            curve[column] = theta[:, index]
        for index, column in enumerate(v6_utils.atlas.XYZ_COLS):
            curve[column] = achieved[:, index]
        curve["xyz_residual_mm"] = np.linalg.norm(target_xyz - achieved, axis=1) * 1000.0
        curve["tube_success"] = curve["xyz_residual_mm"].le(1.5)
        curve["parent_offset_id"] = (
            "centerline"
            if pair == center_pair
            else f"n1_{parent_pair[0]:g}_n2_{parent_pair[1]:g}"
        )
        curve["inverse_nfev"] = 0
        curve["initializer_strategy"] = "jacobian_neighbor_predictor_v1"
        solved[pair] = curve
    return pd.concat([solved[pair] for pair in pairs], ignore_index=True, sort=False)


def _materialize_tube_radius(
    args: argparse.Namespace,
    *,
    family_id: str,
    radius_mm: float,
    centerline: pd.DataFrame,
    centerline_path: Path,
    parent_radius_mm: float | None,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    radius_dir = Path(args.out_dir) / "02_tube" / v6_runner.radius_slug(radius_mm)
    radius_dir.mkdir(parents=True, exist_ok=True)
    domain = _domain()
    policy = _margin_policy()
    offsets = tuple(parse_float_csv(args.tube_offsets_mm))
    task_fingerprint = stable_fingerprint(
        {
            "phase": "v7_tube_surface",
            "strategy_version": TUBE_STRATEGY_VERSION,
            "protocol": formal_protocol_report(args)["protocol_fingerprint"],
            "family_id": str(family_id),
            "radius_mm": float(radius_mm),
            "centerline_sha256": file_sha256(centerline_path),
            "domain": domain.fingerprint,
            "margin": policy.fingerprint,
            "offsets": offsets,
            "cuts": parse_int_csv(args.cut_indices),
        }
    )
    report_path = radius_dir / "tube_quality_report.json"
    tube_path = radius_dir / "tube_surface.parquet"
    attempt_path = radius_dir / "tube_surface_attempt.parquet"
    if bool(args.skip_existing) and report_path.exists():
        cached = read_json(report_path)
        artifact = tube_path if bool(cached.get("formal_tube_label_gate_pass", False)) else attempt_path
        if (
            str(cached.get("task_fingerprint", "")) == task_fingerprint
            and artifact.is_file()
            and str(cached.get("tube_artifact_sha256", "")) == file_sha256(artifact)
        ):
            return pd.read_parquet(artifact), cached, artifact

    center = centerline.sort_values("angle_idx", kind="stable").reset_index(drop=True)
    targets = v6_utils.atlas.make_normal_tube_targets(center, offsets_mm=offsets)
    targets.to_parquet(radius_dir / "tube_targets.parquet", index=False, compression="zstd")
    initial = _initial_tube_surface(
        centerline=center,
        tube_targets=targets,
        offsets_mm=offsets,
        bounds=domain.bounds_rad,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        max_nfev=int(args.max_ik_nfev),
    )
    initial.to_parquet(radius_dir / "tube_initial_surface.parquet", index=False, compression="zstd")
    selected_surface = initial
    selected_report: dict[str, Any] = {}
    barrier_attempts: list[dict[str, Any]] = []
    for weight in policy.barrier_weights:
        stages = _surface_stage_specs(weight, policy.soft_barrier_margin_deg)

        def solve_stage(
            stage_initial: pd.DataFrame,
            stage: Mapping[str, Any],
        ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
            return engine.optimize_tube_surface(
                tube_targets=targets,
                initial_surface=stage_initial,
                offsets_mm=offsets,
                domain=domain,
                margin_policy=policy,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
                stage_specs=(stage,),
                cut_indices=parse_int_csv(args.cut_indices),
                sweep_directions=("outward", "inward"),
                max_nfev=int(args.max_opt_nfev),
                compute_conditioning=False,
                cut_workers=int(args.workers),
            )

        def assess_stage(
            stage_surface: pd.DataFrame,
            surface_report: Mapping[str, Any],
            stage: Mapping[str, Any],
        ) -> dict[str, Any]:
            stage_surface["tube_success"] = stage_surface["xyz_residual_mm"].le(1.5)
            geometric = v6_runner._tube_quality_metrics(
                stage_surface,
                final_points=int(args.final_points),
                offsets_mm=offsets,
            )
            combined = {
                **surface_report,
                **geometric,
                "rows": int(len(stage_surface)),
                "expected_rows": int(args.final_points) * len(offsets) ** 2,
                "tube_success_ratio": float(stage_surface["tube_success"].mean()),
                "lambda_margin": float(weight),
                "surface_stage": str(stage["name"]),
            }
            combined["formal_tube_label_gate_pass"] = formal_tube_label_gate(combined)
            return combined

        surface, combined, stage_attempts = run_surface_stage_schedule(
            initial_surface=initial,
            stage_specs=stages,
            solve_stage=solve_stage,
            assess_stage=assess_stage,
        )
        barrier_attempts.extend(
            {
                key: attempt.get(key)
                for key in (
                    "surface_stage",
                    "lambda_margin",
                    "residual_p95_mm",
                    "residual_max_mm",
                    "tube_success_ratio",
                    "tube10_beta_rms_p95_deg",
                    "min_joint_margin_deg",
                    "joint_margin_p01_deg",
                    "joint_margin_p05_deg",
                    "surface_gate_pass",
                    "tube_gate_pass",
                    "formal_tube_label_gate_pass",
                )
            }
            for attempt in stage_attempts
        )
        selected_surface = surface
        selected_report = combined
        if combined["formal_tube_label_gate_pass"]:
            break
    annotated = v6_runner.annotate_tube_rows(
        selected_surface,
        family_id=str(family_id),
        radius_mm=float(radius_mm),
        parent_radius_mm=parent_radius_mm,
        radial_predictor_type="standard_domain_surface",
        branch_hash=v6_utils.atlas.branch_hash(center),
    )
    annotated["solver_strategy_version"] = "true-ellipse-standard-domain-v7.1"
    annotated["joint_margin_policy_id"] = policy.policy_id
    annotated["tube_surface_task_fingerprint"] = task_fingerprint
    annotated.to_parquet(attempt_path, index=False, compression="zstd")
    artifact = attempt_path
    if bool(selected_report.get("formal_tube_label_gate_pass", False)):
        annotated.to_parquet(tube_path, index=False, compression="zstd")
        artifact = tube_path
    report = {
        "strategy_version": TUBE_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        "family_id": str(family_id),
        "radius_mm": float(radius_mm),
        "parent_radius_mm": parent_radius_mm,
        "centerline_path": str(centerline_path.resolve()),
        "centerline_sha256": file_sha256(centerline_path),
        "barrier_attempts": barrier_attempts,
        **selected_report,
        "tube_artifact_path": str(artifact.resolve()),
        "tube_artifact_sha256": file_sha256(artifact),
    }
    write_json(report_path, report)
    return annotated, report, artifact


def _load_or_materialize_centerline(
    args: argparse.Namespace,
    *,
    radius_mm: float,
    manifest: dict[str, dict[str, Any]],
    family: v6_utils.FamilySpec,
    radial_report: Mapping[str, Any],
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[pd.DataFrame, Path, float | None]:
    key = f"{float(radius_mm):g}"
    if key in manifest:
        path = Path(manifest[key]["path"])
        return pd.read_parquet(path), path, None
    available = sorted(float(value) for value in manifest if float(value) < float(radius_mm) - 1.0e-9)
    if not available:
        raise ValueError(f"no radial parent available for {radius_mm:g} mm")
    parent_radius = available[-1]
    previous_radius = available[-2] if len(available) >= 2 else None
    parent_path = Path(manifest[f"{parent_radius:g}"]["path"])
    previous_path = None if previous_radius is None else Path(manifest[f"{previous_radius:g}"]["path"])
    solved, report, artifact = v6_runner._solve_radial_radius(
        args,
        audit_report=ensure_audit_report(args),
        family=family,
        parent_path=pd.read_parquet(parent_path),
        parent_radius_mm=parent_radius,
        parent_artifact_path=parent_path,
        previous_parent=None if previous_path is None else pd.read_parquet(previous_path),
        previous_parent_radius_mm=previous_radius,
        previous_parent_artifact_path=previous_path,
        target_radius_mm=float(radius_mm),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    if not bool(report.get("radial_bundle_gate_pass", False)):
        raise RuntimeError(f"training-anchor centerline failed at {radius_mm:g} mm")
    manifest[key] = {
        "path": str(artifact.resolve()),
        "sha256": file_sha256(artifact),
        "branch_hash": v6_utils.atlas.branch_hash(solved),
        "training_anchor": True,
    }
    return solved, artifact, parent_radius


def phase_tube(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "02_tube"
    out.mkdir(parents=True, exist_ok=True)
    radial = ensure_radial_report(args)
    strict_radial = radial.get("strict_geometry_rmax_mm")
    if strict_radial is None:
        report = {
            "strategy_version": TUBE_STRATEGY_VERSION,
            "strict_geometry_rmax_mm": None,
            "formal_tube_gate_pass": False,
            "reason": "no_strict_radial_frontier",
        }
        write_json(out / "tube_report.json", report)
        return report
    family = v6_runner.load_family_spec(args.v5_dir, args.family_id)
    lengths_m, p_end_local_m, theta_sign = _load_robot(args)
    manifest = {str(key): dict(value) for key, value in radial.get("path_manifest", {}).items()}
    radii = materialized_dataset_radii(float(strict_radial))
    tube_paths: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []
    checkpoint_pass: dict[float, bool] = {}
    previous_radius: float | None = None
    for radius in radii:
        try:
            centerline, centerline_path, parent_radius = _load_or_materialize_centerline(
                args,
                radius_mm=float(radius),
                manifest=manifest,
                family=family,
                radial_report=radial,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
            )
            tube, quality, artifact = _materialize_tube_radius(
                args,
                family_id=family.family_id,
                radius_mm=float(radius),
                centerline=centerline,
                centerline_path=centerline_path,
                parent_radius_mm=parent_radius if parent_radius is not None else previous_radius,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
            )
            del tube
            passed = bool(quality.get("formal_tube_label_gate_pass", False))
            if passed:
                tube_paths[f"{float(radius):g}"] = str(artifact.resolve())
            reason = "passed" if passed else "tube_or_label_gate_failed"
        except Exception as exc:  # noqa: BLE001 - preserve later-radius diagnostics
            quality = {}
            artifact = Path(args.out_dir) / "02_tube" / v6_runner.radius_slug(radius) / "tube_surface_attempt.parquet"
            passed = False
            reason = f"{type(exc).__name__}: {exc}"
        previous_radius = float(radius)
        if any(np.isclose(radius, check, atol=1.0e-8) for check in engine.v7_standard_domain_protocol().formal_checkpoints_mm):
            checkpoint_pass[float(radius)] = passed
        summary_rows.append(
            {
                "family_id": family.family_id,
                "radius_mm": float(radius),
                "is_training_anchor": bool(radius in engine.training_anchor_radii(float(strict_radial))),
                "formal_tube_label_gate_pass": passed,
                "reason": reason,
                "rows": quality.get("rows"),
                "residual_p95_mm": quality.get("residual_p95_mm"),
                "residual_max_mm": quality.get("residual_max_mm"),
                "tube_success_ratio": quality.get("tube_success_ratio"),
                "min_joint_margin_deg": quality.get("min_joint_margin_deg"),
                "tube_artifact_path": str(artifact.resolve()),
                "tube_artifact_sha256": quality.get("tube_artifact_sha256"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "tube_radius_summary.csv", index=False)
    strict_tube: float | None = None
    for radius in engine.v7_standard_domain_protocol().formal_checkpoints_mm:
        if radius > float(strict_radial) + 1.0e-9:
            break
        if not checkpoint_pass.get(float(radius), False):
            break
        strict_tube = float(radius)
    all_dataset_radii_pass = bool(
        len(summary) == len(radii)
        and summary["formal_tube_label_gate_pass"].fillna(False).astype(bool).all()
    )
    report = {
        "strategy_version": TUBE_STRATEGY_VERSION,
        **formal_protocol_report(args),
        "radial_task_fingerprint": str(radial.get("task_fingerprint", "")),
        "family_id": family.family_id,
        "radial_strict_rmax_mm": float(strict_radial),
        "strict_geometry_rmax_mm": strict_tube,
        "materialized_radii_mm": list(radii),
        "all_dataset_radii_pass": all_dataset_radii_pass,
        "tube_paths": tube_paths,
        "centerline_manifest": manifest,
        "summary_path": str((out / "tube_radius_summary.csv").resolve()),
    }
    report["formal_tube_gate_pass"] = bool(
        report["formal_protocol_gate_pass"]
        and strict_tube is not None
        and strict_tube >= 105.0
        and all_dataset_radii_pass
    )
    report["task_fingerprint"] = stable_fingerprint(
        {
            "radial": report["radial_task_fingerprint"],
            "protocol": report["protocol_fingerprint"],
            "tubes": [
                {
                    "radius_mm": row["radius_mm"],
                    "sha256": row.get("tube_artifact_sha256"),
                    "pass": row["formal_tube_label_gate_pass"],
                }
                for row in summary_rows
            ],
        }
    )
    write_json(out / "tube_report.json", report)
    return report


def ensure_tube_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "02_tube" / "tube_report.json"
    expected = formal_protocol_report(args)["protocol_fingerprint"]
    if path.exists():
        cached = read_json(path)
        if str(cached.get("protocol_fingerprint", "")) == str(expected):
            return cached
    return phase_tube(args)


def _challenge_task_fingerprint(
    args: argparse.Namespace,
    *,
    radius_mm: float,
    split: str,
    centerline_path: Path,
) -> str:
    return stable_fingerprint(
        {
            "phase": "v7_half_phase_challenge",
            "strategy_version": DATASET_STRATEGY_VERSION,
            "protocol": formal_protocol_report(args)["protocol_fingerprint"],
            "radius_mm": float(radius_mm),
            "split": str(split),
            "centerline_sha256": file_sha256(centerline_path),
            "joint_domain": _domain().fingerprint,
            "joint_margin_policy": _margin_policy().fingerprint,
            "cuts": parse_int_csv(args.cut_indices),
            "anchor_schedules": parse_name_csv(args.anchor_schedules),
        }
    )


def materialize_half_phase_challenge(
    args: argparse.Namespace,
    *,
    family: v6_utils.FamilySpec,
    radius_mm: float,
    split: str,
    centerline_path: Path,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Solve and audit the registered unseen half-phase centerline."""

    challenge_dir = Path(args.out_dir) / "03_dataset" / "challenges" / str(split)
    challenge_dir.mkdir(parents=True, exist_ok=True)
    report_path = challenge_dir / "challenge_report.json"
    final_path = challenge_dir / "half_phase_centerline.parquet"
    attempt_path = challenge_dir / "half_phase_centerline_attempt.parquet"
    task_fingerprint = _challenge_task_fingerprint(
        args,
        radius_mm=float(radius_mm),
        split=str(split),
        centerline_path=centerline_path,
    )
    if bool(args.skip_existing) and report_path.exists():
        cached = read_json(report_path)
        artifact = final_path if bool(cached.get("challenge_gate_pass", False)) else attempt_path
        if (
            str(cached.get("task_fingerprint", "")) == task_fingerprint
            and artifact.is_file()
            and str(cached.get("challenge_artifact_sha256", "")) == file_sha256(artifact)
        ):
            return pd.read_parquet(artifact), cached, artifact

    centerline = pd.read_parquet(centerline_path)
    targets = generate_half_phase_targets(
        family,
        radius_mm=float(radius_mm),
        n_points=int(args.final_points),
    )
    predictor = build_half_phase_predictor(centerline, targets)
    targets.to_parquet(challenge_dir / "half_phase_targets.parquet", index=False, compression="zstd")
    predictor.to_parquet(
        challenge_dir / "half_phase_predictor.parquet", index=False, compression="zstd"
    )
    domain = _domain()
    policy = _margin_policy()
    cuts = parse_int_csv(args.cut_indices)
    schedules = parse_name_csv(args.anchor_schedules)
    paths: dict[tuple[int, str], pd.DataFrame] = {}
    selected_schedules: dict[int, str] = {}
    job_reports: list[dict[str, Any]] = []
    for cut_idx in cuts:
        selected: pd.DataFrame | None = None
        for schedule in schedules:
            corrected, raw = v6_utils.correct_radial_predictor(
                targets,
                predictor,
                cut_idx=int(cut_idx),
                anchor_schedule=str(schedule),
                bounds=domain.bounds_rad,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=float(theta_sign),
                max_nfev=int(args.max_opt_nfev),
                compute_conditioning=True,
                lambda_margin=float(args.lambda_margin),
                soft_margin_deg=float(args.soft_margin_deg),
            )
            margin = engine.joint_margin_report(
                corrected[v6_utils.atlas.BETA_COLS].to_numpy(dtype=float),
                domain=domain,
                at_bound_tolerance_deg=policy.at_bound_tolerance_deg,
            )
            margin_gate = engine.evaluate_joint_margin_gate(margin, policy=policy)
            job_gate = bool(raw.get("centerline_gate_pass", False) and margin_gate["joint_margin_gate_pass"])
            artifact_path = challenge_dir / f"cut_{int(cut_idx):03d}_{schedule}.parquet"
            corrected.to_parquet(artifact_path, index=False, compression="zstd")
            job_reports.append(
                {
                    "cut_idx": int(cut_idx),
                    "anchor_schedule": str(schedule),
                    "selected": job_gate,
                    "centerline_gate_pass": bool(raw.get("centerline_gate_pass", False)),
                    "job_gate_pass": job_gate,
                    "path": str(artifact_path.resolve()),
                    **{
                        key: raw.get(key)
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
                    **margin,
                    **margin_gate,
                }
            )
            if job_gate:
                selected = corrected
                selected_schedules[int(cut_idx)] = str(schedule)
                paths[(int(cut_idx), "half_phase")] = corrected
                break
        if selected is None:
            continue
    cut_report = v6_utils.cut_invariance_report(
        paths,
        required_cuts=cuts,
        required_predictors=("half_phase",),
        threshold_deg=1.0,
    )
    canonical = paths.get((0, "half_phase"))
    if canonical is not None and 0 in selected_schedules:
        repeat, _repeat_raw = v6_utils.correct_radial_predictor(
            targets,
            predictor,
            cut_idx=0,
            anchor_schedule=selected_schedules[0],
            bounds=domain.bounds_rad,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=float(theta_sign),
            max_nfev=int(args.max_opt_nfev),
            compute_conditioning=True,
            lambda_margin=float(args.lambda_margin),
            soft_margin_deg=float(args.soft_margin_deg),
        )
        repeatability = v6_utils.exact_repeatability_report(canonical, repeat)
    else:
        repeatability = {
            "deterministic_exact_gate_pass": False,
            "reason": "cut_zero_challenge_missing",
        }
    if canonical is None:
        canonical = predictor.copy()
    annotated = engine.annotate_joint_margins(canonical, domain=domain)
    annotated["challenge_kind"] = "half_phase_centerline"
    annotated["split"] = str(split)
    annotated["used_for_training"] = False
    annotated["is_centerline"] = True
    annotated["trajectory_id"] = f"{family.family_id}@r{float(radius_mm):g}:half_phase"
    annotated["ellipse_id"] = annotated["trajectory_id"]
    annotated["branch_id"] = f"{family.family_id}:standard_domain_v7"
    annotated["sample_id"] = [
        f"{family.family_id}@r{float(radius_mm):g}:half:{int(angle_idx):03d}"
        for angle_idx in annotated["angle_idx"]
    ]
    annotated["solver_strategy_version"] = "true-ellipse-standard-domain-v7.1"
    annotated.to_parquet(attempt_path, index=False, compression="zstd")
    all_jobs_pass = bool(len(paths) == len(cuts))
    challenge_gate = bool(
        all_jobs_pass
        and cut_report.get("cut_invariance_gate_pass", False)
        and repeatability.get("deterministic_exact_gate_pass", False)
        and len(annotated) == int(args.final_points)
        and annotated["sample_id"].is_unique
    )
    artifact = attempt_path
    if challenge_gate:
        annotated.to_parquet(final_path, index=False, compression="zstd")
        artifact = final_path
    report = {
        "strategy_version": DATASET_STRATEGY_VERSION,
        "task_fingerprint": task_fingerprint,
        "family_id": family.family_id,
        "radius_mm": float(radius_mm),
        "split": str(split),
        "challenge_kind": "half_phase_centerline",
        "rows": int(len(annotated)),
        "expected_rows": int(args.final_points),
        "all_cut_jobs_pass": all_jobs_pass,
        "selected_schedules": {str(key): value for key, value in selected_schedules.items()},
        "jobs": job_reports,
        "cut_invariance": cut_report,
        "deterministic_repeatability": repeatability,
        "challenge_gate_pass": challenge_gate,
        "challenge_artifact_path": str(artifact.resolve()),
        "challenge_artifact_sha256": file_sha256(artifact),
        "centerline_path": str(centerline_path.resolve()),
        "centerline_sha256": file_sha256(centerline_path),
    }
    write_json(report_path, report)
    return annotated, report, artifact


def phase_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "03_dataset"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "dataset_report.json"
    attempt_path = out / "dataset_attempt.parquet"
    final_path = out / "true_ellipse_standard_domain_tubes_v7.parquet"
    manifest_path = out / "trajectory_manifest.csv"
    support_path = out / "holdout_support_candidates.csv"
    centerline_dir = out / "integer_centerlines"
    centerline_dir.mkdir(parents=True, exist_ok=True)
    tube_report = ensure_tube_report(args)
    if not bool(tube_report.get("formal_tube_gate_pass", False)):
        report = {
            "strategy_version": DATASET_STRATEGY_VERSION,
            **formal_protocol_report(args),
            "formal_dataset_gate_pass": False,
            "dataset_gate_pass": False,
            "reason": "strict_tube_chain_below_registered_105mm_minimum_or_incomplete",
            "strict_geometry_rmax_mm": tube_report.get("strict_geometry_rmax_mm"),
        }
        write_json(report_path, report)
        return report

    family = v6_runner.load_family_spec(args.v5_dir, args.family_id)
    radii = tuple(float(value) for value in tube_report["materialized_radii_mm"])
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    dataset_centerlines: dict[str, dict[str, Any]] = {}
    for radius_mm in radii:
        key = f"{radius_mm:g}"
        tube_path = Path(tube_report["tube_paths"][key])
        tube = pd.read_parquet(tube_path)
        expected_rows = int(args.final_points) * len(parse_float_csv(args.tube_offsets_mm)) ** 2
        duplicate_count = int(tube.duplicated(["angle_idx", "tube_offset_id"], keep=False).sum())
        complete = bool(
            len(tube) == expected_rows
            and int(tube["angle_idx"].nunique()) == int(args.final_points)
            and int(tube["tube_offset_id"].nunique())
            == len(parse_float_csv(args.tube_offsets_mm)) ** 2
            and duplicate_count == 0
            and tube["sample_id"].is_unique
            and tube["tube_success"].fillna(False).astype(bool).all()
        )
        manifest_rows.append(
            {
                "trajectory_id": str(tube["trajectory_id"].iloc[0]),
                "family_id": family.family_id,
                "radius_mm": float(radius_mm),
                "rows": int(len(tube)),
                "expected_rows": expected_rows,
                "angle_count": int(tube["angle_idx"].nunique()),
                "offset_count": int(tube["tube_offset_id"].nunique()),
                "duplicate_angle_offset_rows": duplicate_count,
                "tube_success_ratio": float(tube["tube_success"].astype(bool).mean()),
                "trajectory_complete": complete,
                "tube_path": str(tube_path.resolve()),
                "tube_sha256": file_sha256(tube_path),
            }
        )
        final_centerline = extract_tube_centerline(tube, expected_points=int(args.final_points))
        final_centerline_path = centerline_dir / f"{v6_runner.radius_slug(radius_mm)}.parquet"
        final_centerline.to_parquet(
            final_centerline_path,
            index=False,
            compression="zstd",
        )
        dataset_centerlines[key] = {
            "path": str(final_centerline_path.resolve()),
            "sha256": file_sha256(final_centerline_path),
            "source_tube_sha256": file_sha256(tube_path),
            "rows": int(len(final_centerline)),
        }
        frames.append(tube)
    dataset = pd.concat(frames, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    completeness_gate = bool(
        len(manifest) == len(radii)
        and manifest["trajectory_complete"].fillna(False).astype(bool).all()
        and dataset["sample_id"].is_unique
        and dataset["family_id"].astype(str).nunique() == 1
    )
    conflict = v5_expansion.branch_conflict_report(dataset, voxel_mm=2.0, threshold_deg=3.0)
    margin = engine.joint_margin_report(
        dataset[v6_utils.atlas.BETA_COLS].to_numpy(dtype=float),
        domain=_domain(),
        at_bound_tolerance_deg=_margin_policy().at_bound_tolerance_deg,
    )
    margin_gate = engine.evaluate_joint_margin_gate(margin, policy=_margin_policy())
    dataset = engine.annotate_joint_margins(dataset, domain=_domain())

    strict_rmax = float(tube_report["strict_geometry_rmax_mm"])
    candidates = [
        float(value)
        for value in engine.v7_standard_domain_protocol().formal_checkpoints_mm
        if 105.0 - 1.0e-9 <= float(value) <= strict_rmax + 1.0e-9
    ]
    support = compute_holdout_support_candidates(
        dataset,
        candidate_test_radii_mm=candidates,
        validation_gap_mm=7.5,
    )
    support.to_csv(support_path, index=False)
    holdout: dict[str, Any] | None = None
    for candidate in sorted(candidates, reverse=True):
        pair = support[np.isclose(support["test_radius_mm"], candidate, atol=1.0e-8)]
        try:
            selected = engine.select_registered_holdouts(
                strict_geometry_rmax_mm=float(candidate),
                support_by_radius=pair[["radius_mm", "strict_support_gate_pass"]],
                minimum_test_radius_mm=float(candidate),
                validation_gap_mm=7.5,
            )
        except ValueError:
            continue
        holdout = selected
        break
    if holdout is None:
        assigned = dataset.copy()
        assigned["split"] = "unassigned"
        assigned["used_for_training"] = False
        split_report = {"split_gate_pass": False, "reason": "no_support_backed_holdout_pair"}
        validation_support = False
        test_support = False
    else:
        assigned, split_report = assign_dynamic_radius_splits(
            dataset,
            validation_radius_mm=float(holdout["validation_radius_mm"]),
            test_radius_mm=float(holdout["test_radius_mm"]),
        )
        selected_support = support[
            np.isclose(support["test_radius_mm"], float(holdout["test_radius_mm"]), atol=1.0e-8)
        ]
        validation_support = bool(
            selected_support.loc[
                selected_support["role"].eq("validation"), "strict_support_gate_pass"
            ].all()
        )
        test_support = bool(
            selected_support.loc[selected_support["role"].eq("test"), "strict_support_gate_pass"].all()
        )
    assigned.to_parquet(attempt_path, index=False, compression="zstd")

    challenge_reports: dict[str, Any] = {}
    challenge_paths: dict[str, str] = {}
    challenge_gate = False
    if holdout is not None:
        lengths_m, p_end_local_m, theta_sign = _load_robot(args)
        for split_name, radius_key in (
            ("validation", "validation_radius_mm"),
            ("test", "test_radius_mm"),
        ):
            radius_mm = float(holdout[radius_key])
            centerline_path = Path(dataset_centerlines[f"{radius_mm:g}"]["path"])
            _challenge, challenge_report, challenge_path = materialize_half_phase_challenge(
                args,
                family=family,
                radius_mm=radius_mm,
                split=split_name,
                centerline_path=centerline_path,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                theta_sign=theta_sign,
            )
            challenge_reports[split_name] = challenge_report
            challenge_paths[split_name] = str(challenge_path.resolve())
        challenge_gate = bool(
            set(challenge_reports) == {"validation", "test"}
            and all(report.get("challenge_gate_pass", False) for report in challenge_reports.values())
        )

    required_metadata = {
        "family_id",
        "branch_id",
        "trajectory_id",
        "radius_mm",
        "angle_idx",
        "tube_offset_id",
        "is_centerline",
        "solver_strategy_version",
        "radial_predictor_type",
        "branch_hash",
        "sample_id",
        "split",
        "used_for_training",
        "joint_domain_id",
        "min_joint_margin_deg",
    }
    missing_metadata = sorted(required_metadata - set(assigned.columns))
    checks = {
        "formal_protocol_gate_pass": bool(formal_protocol_report(args)["formal_protocol_gate_pass"]),
        "formal_tube_gate_pass": bool(tube_report.get("formal_tube_gate_pass", False)),
        "completeness_gate_pass": bool(completeness_gate and not missing_metadata),
        "split_gate_pass": bool(split_report.get("split_gate_pass", False)),
        "branch_conflict_gate_pass": bool(conflict.get("branch_conflict_gate_pass", False)),
        "joint_margin_gate_pass": bool(margin_gate.get("joint_margin_gate_pass", False)),
        "holdout_selection_gate_pass": bool(holdout and holdout.get("selection_gate_pass", False)),
        "validation_support_gate_pass": validation_support,
        "test_support_gate_pass": test_support,
        "challenge_gate_pass": challenge_gate,
    }
    dataset_gate = formal_dataset_gate(checks)
    if dataset_gate:
        assigned.to_parquet(final_path, index=False, compression="zstd")
    report = {
        "strategy_version": DATASET_STRATEGY_VERSION,
        **formal_protocol_report(args),
        "tube_task_fingerprint": str(tube_report.get("task_fingerprint", "")),
        "family_id": family.family_id,
        "strict_geometry_rmax_mm": strict_rmax,
        "dataset_gate_pass": dataset_gate,
        "formal_dataset_gate_pass": dataset_gate,
        "checks_recomputed": checks,
        "rows": int(len(assigned)),
        "expected_rows": int(len(radii) * int(args.final_points) * 25),
        "trajectory_count": int(assigned["trajectory_id"].nunique()),
        "radii_mm": list(radii),
        "unique_sample_ids": bool(assigned["sample_id"].is_unique),
        "missing_metadata_columns": missing_metadata,
        "split": split_report,
        "holdout": holdout,
        "branch_conflict": conflict,
        "joint_margin": margin,
        "joint_margin_gate": margin_gate,
        "validation_support_gate_pass": validation_support,
        "test_support_gate_pass": test_support,
        "support_candidates_path": str(support_path.resolve()),
        "support_candidates_sha256": file_sha256(support_path),
        "challenge_gate_pass": challenge_gate,
        "challenge_reports": challenge_reports,
        "challenge_paths": challenge_paths,
        "integer_centerline_manifest": dataset_centerlines,
        "attempt_path": str(attempt_path.resolve()),
        "attempt_sha256": file_sha256(attempt_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_path": str(final_path.resolve()) if dataset_gate else None,
        "dataset_sha256": file_sha256(final_path) if dataset_gate else None,
        "model_input_columns": list(v6_utils.atlas.TARGET_XYZ_COLS),
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "tube": report["tube_task_fingerprint"],
            "protocol": report["protocol_fingerprint"],
            "attempt": report["attempt_sha256"],
            "manifest": report["manifest_sha256"],
            "support": report["support_candidates_sha256"],
            "challenges": {
                key: value.get("challenge_artifact_sha256")
                for key, value in challenge_reports.items()
            },
        }
    )
    write_json(report_path, report)
    return report


def ensure_dataset_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "03_dataset" / "dataset_report.json"
    expected = formal_protocol_report(args)["protocol_fingerprint"]
    if path.exists():
        cached = read_json(path)
        if str(cached.get("protocol_fingerprint", "")) == str(expected):
            return cached
    return phase_dataset(args)


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "04_summary"
    out.mkdir(parents=True, exist_ok=True)
    audit = ensure_audit_report(args)
    radial = ensure_radial_report(args)
    tube = ensure_tube_report(args)
    dataset = ensure_dataset_report(args)
    strict_upstream = bool(
        audit.get("formal_audit_gate_pass", False)
        and radial.get("formal_radial_gate_pass", False)
        and tube.get("formal_tube_gate_pass", False)
        and dataset.get("formal_dataset_gate_pass", False)
    )
    report = {
        **formal_protocol_report(args),
        "family_id": str(args.family_id),
        "exploration_target_radius_mm": float(args.target_radius_mm),
        "strict_radial_rmax_mm": radial.get("strict_geometry_rmax_mm"),
        "exploratory_rescue_rmax_mm": radial.get("exploratory_rescue_rmax_mm"),
        "strict_tube_rmax_mm": tube.get("strict_geometry_rmax_mm"),
        "validation_radius_mm": (dataset.get("holdout") or {}).get("validation_radius_mm"),
        "test_radius_mm": (dataset.get("holdout") or {}).get("test_radius_mm"),
        "audit_gate_pass": bool(audit.get("formal_audit_gate_pass", False)),
        "radial_gate_pass": bool(radial.get("formal_radial_gate_pass", False)),
        "tube_gate_pass": bool(tube.get("formal_tube_gate_pass", False)),
        "dataset_gate_pass": bool(dataset.get("formal_dataset_gate_pass", False)),
        "challenge_gate_pass": bool(dataset.get("challenge_gate_pass", False)),
        "strict_upstream_gate_pass": strict_upstream,
        "model_training_authorized": strict_upstream,
        "strict_model_claim_made": False,
        "static_inverse_claim_radius_mm": None,
        "reports": {
            "audit": str((Path(args.out_dir) / "00_audit" / "audit_report.json").resolve()),
            "radial": str((Path(args.out_dir) / "01_radial" / "radial_report.json").resolve()),
            "tube": str((Path(args.out_dir) / "02_tube" / "tube_report.json").resolve()),
            "dataset": str((Path(args.out_dir) / "03_dataset" / "dataset_report.json").resolve()),
        },
    }
    write_json(out / "summary_report.json", report)
    lines = [
        "# True Ellipse Standard-Domain V7 summary",
        "",
        f"- Fixed family: `{args.family_id}`.",
        f"- Standard-domain formal protocol: `{report['formal_protocol_gate_pass']}`.",
        f"- Strict radial frontier: `{report['strict_radial_rmax_mm']} mm`.",
        f"- Exploratory rescue frontier: `{report['exploratory_rescue_rmax_mm']} mm`.",
        f"- Strict tube/label frontier: `{report['strict_tube_rmax_mm']} mm`.",
        f"- Dynamic validation radius: `{report['validation_radius_mm']} mm`.",
        f"- Dynamic test radius: `{report['test_radius_mm']} mm`.",
        f"- Integer + half-phase dataset gate: `{report['dataset_gate_pass']}`.",
        f"- Model training authorized: `{report['model_training_authorized']}`.",
        "",
        "This upstream report makes no fitted-model radius claim. The static inverse radius is only",
        "registered after the separate V7 training protocol passes its five-seed formal gate.",
    ]
    summary_path = out / "experiment_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["summary_path"] = str(summary_path.resolve())
    write_json(out / "summary_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    handlers = {
        "audit": phase_audit,
        "radial": phase_radial,
        "tube": phase_tube,
        "dataset": phase_dataset,
        "summary": phase_summary,
    }
    phases = parse_phases(args.phases)
    results = {phase: handlers[phase](args) for phase in phases}
    report = {
        "mode": "true_ellipse_standard_domain_v7",
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
