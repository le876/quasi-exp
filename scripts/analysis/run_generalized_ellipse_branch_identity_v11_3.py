#!/usr/bin/env python3
"""Run the V11.3 canonical branch-identity repair experiment.

This protocol consumes the frozen V11.2 anchor variants, preserves their
formal Phase-1 failure, and evaluates BI-0..BI-4 without authorizing tube or
student stages.  Traversal variants are audits of one consensus branch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

from quasi_exp.teacher.branch_identity import (
    BETA_COLUMNS,
    BranchIdentityGate,
    BranchRepairPolicy,
    ReferenceBranchTeacher,
    analyze_branch_variants,
    audit_consensus_variants,
    link_reference_cyclic_candidates,
    optimize_consensus_branch,
    select_canonical_root_phase,
    select_canonical_root_solution,
)
from quasi_exp.teacher.canonical import (
    TeacherPolicy,
    TeacherTrajectory,
    TeacherVariant,
    _correct_target,
    _trajectory_metrics,
)
from quasi_exp.teacher.dataset import trajectory_frame
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.large_scale import ReachabilityAtlas

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


STAGE_DIRS = {
    "protocol": "00_protocol",
    "forensics": "00_branch_forensics",
    "root": "01_canonical_root",
    "pilot": "02_pilot",
    "formal": "03_formal",
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def load_config(path: str | Path, *, preset: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V11.3 config must contain a mapping")
    normalized = str(preset).lower()
    if normalized not in {"smoke", "formal"}:
        raise ValueError("preset must be smoke or formal")
    if normalized == "smoke":
        payload = _deep_merge(payload, payload.get("smoke", {}))
    payload.pop("smoke", None)
    payload["preset"] = normalized
    payload["config_path"] = str(config_path)
    return payload


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if ".worktrees" in resolved.parts:
        index = resolved.parts.index(".worktrees")
        return Path(*resolved.parts[:index])
    return resolved


def variant_specs(cuts: Sequence[int]) -> list[tuple[str, str, int]]:
    normalized = [int(value) for value in cuts]
    # Primary and repeat deliberately use identical traversal, seed and policy.
    output = [("primary", "forward", 0), ("repeat", "forward", 0)]
    output.extend(
        (f"forward_cut{cut:04d}", "forward", cut)
        for cut in normalized
        if cut != 0
    )
    output.extend(
        (f"reverse_cut{cut:04d}", "reverse", cut) for cut in normalized
    )
    return output


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_gate(path: Path, *, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    artifact_sha256 = {
        item.relative_to(path.parent).as_posix(): sha256_file(item)
        for item in sorted(path.parent.rglob("*"))
        if item.is_file() and item.resolve() != path.resolve()
    }
    payload = {
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
        "artifact_sha256": artifact_sha256,
    }
    atomic_write_json(path, payload)
    return payload


def _candidate_ids(config: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *map(str, config["candidates"]["pilot"]),
                *map(str, config["candidates"]["negative_control"]),
            ]
        )
    )


def _source_anchor_root(config: Mapping[str, Any], project_root: Path) -> Path:
    return project_root / str(config["source_artifact_root"])


def _source_candidate_root(
    config: Mapping[str, Any], project_root: Path, candidate_id: str
) -> Path:
    return _source_anchor_root(config, project_root) / "verify" / str(candidate_id)


def _load_source_frames(
    config: Mapping[str, Any], project_root: Path, candidate_id: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    root = _source_candidate_root(config, project_root, candidate_id)
    primary_path = root / "primary/centerline.parquet"
    if not primary_path.is_file():
        raise FileNotFoundError(primary_path)
    primary = pd.read_parquet(primary_path).sort_values(
        "phase_idx", kind="stable"
    ).reset_index(drop=True)
    variants: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob("*/centerline.parquet")):
        name = path.parent.name
        if name == "primary":
            continue
        variants[name] = pd.read_parquet(path).sort_values(
            "phase_idx", kind="stable"
        ).reset_index(drop=True)
    return primary, variants


def _subsample_frame(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    requested = int(count)
    if requested < 3 or requested > len(frame):
        raise ValueError("phase count must be between three and source inventory")
    if requested == len(frame):
        output = frame.copy()
    else:
        positions = np.floor(
            np.arange(requested, dtype=float) * len(frame) / requested
        ).astype(np.int64)
        output = frame.iloc[positions].copy()
    output = output.reset_index(drop=True)
    output["phase_idx"] = np.arange(requested, dtype=np.int64)
    output["phase_rad"] = np.arange(requested) * (2.0 * math.pi / requested)
    return output


def _aligned_source_variants(
    primary: pd.DataFrame,
    variants: Mapping[str, pd.DataFrame],
    *,
    count: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    reference = _subsample_frame(primary, int(count))
    output = {
        name: _subsample_frame(frame, int(count)) for name, frame in variants.items()
    }
    target_columns = ["target_x_m", "target_y_m", "target_z_m"]
    target = reference[target_columns].to_numpy(float)
    for name, frame in output.items():
        if not np.allclose(frame[target_columns].to_numpy(float), target, atol=1.0e-12):
            raise ValueError(f"variant {name} target inventory is not phase aligned")
    return reference, output


def _plot_forensics(report: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    gaps = report.per_phase_variant_gap
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for variant, group in gaps.groupby("variant", sort=True):
        axis.plot(group["phase_idx"], group["gap_rms_deg"], label=str(variant), lw=0.9)
    axis.axhline(1.0, color="black", ls="--", lw=0.8)
    axis.set(xlabel="phase_idx", ylabel="beta RMS gap (deg)")
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "variant_gap.png", dpi=160)
    plt.close(fig)

    joint = report.per_joint_gap
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for variant, group in joint.groupby("variant", sort=True):
        axis.plot(group["joint"], group["abs_gap_p95_deg"], marker="o", label=str(variant))
    axis.set(ylabel="per-joint |gap| p95 (deg)")
    axis.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "per_joint_gap.png", dpi=160)
    plt.close(fig)


def _forensics_for_candidate(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    candidate_id: str,
    count: int,
    directory: Path,
) -> tuple[Any, pd.DataFrame, dict[str, pd.DataFrame]]:
    primary_source, variant_source = _load_source_frames(
        config, project_root, candidate_id
    )
    primary, variants = _aligned_source_variants(
        primary_source, variant_source, count=int(count)
    )
    report = analyze_branch_variants(
        primary,
        variants,
        jacobian=lambda beta: environment.jacobian(beta),
        cluster_threshold_deg=float(config["canonical_root"]["cluster_threshold_deg"]),
    )
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(report.per_phase_variant_gap, directory / "per_phase_variant_gap.parquet")
    report.per_joint_gap.to_csv(directory / "per_joint_gap.csv", index=False)
    _atomic_parquet(report.branch_clusters, directory / "branch_clusters.parquet")
    report.nullspace_gap_report.to_csv(directory / "nullspace_gap_report.csv", index=False)
    report.transition_intervals.to_csv(directory / "branch_transition_intervals.csv", index=False)
    report.variant_summary.to_csv(directory / "variant_summary.csv", index=False)
    _plot_forensics(report, directory / "branch_transition_plots")
    return report, primary, variants


def _teacher_policy(config: Mapping[str, Any], *, seed: int) -> TeacherPolicy:
    values = dict(config["teacher_policy"])
    values["variant"] = TeacherVariant(str(values.get("variant", "T3")))
    values["solver_seed"] = int(seed)
    return TeacherPolicy(**values)


def _reachability_atlas(project_root: Path, config: Mapping[str, Any]) -> ReachabilityAtlas:
    path = project_root / str(config["v10_evidence_root"]) / "reachability_atlas.parquet"
    frame = pd.read_parquet(path)
    return ReachabilityAtlas(
        xyz_m=frame[["x_m", "y_m", "z_m"]].to_numpy(float),
        beta_rad=frame[[f"beta{index}_rad" for index in range(1, 7)]].to_numpy(float),
    )


def _frame_beta(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(BETA_COLUMNS)].to_numpy(float)


def _frame_target(frame: pd.DataFrame) -> np.ndarray:
    return frame[["target_x_m", "target_y_m", "target_z_m"]].to_numpy(float)


def _family_metadata(frame: pd.DataFrame) -> tuple[str, float]:
    return str(frame.iloc[0]["family_id"]), float(frame.iloc[0]["radius_mm"])


def _metrics_for_beta(environment: Any, target: np.ndarray, beta: np.ndarray) -> dict[str, float]:
    achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    metrics = _trajectory_metrics(beta, target, achieved)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    margin = np.minimum(
        beta - bounds[:, 0][None, :], bounds[:, 1][None, :] - beta
    )
    metrics["joint_margin_min_deg"] = float(np.rad2deg(np.min(margin)))
    return metrics


def _trajectory_from_beta(
    environment: Any,
    target: np.ndarray,
    beta: np.ndarray,
    *,
    policy: TeacherPolicy,
    trajectory_id: str,
    family_id: str,
    radius_mm: float,
    direction: str,
    cut: int,
) -> TeacherTrajectory:
    achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    metrics = _metrics_for_beta(environment, target, beta)
    return TeacherTrajectory(
        beta_rad=np.asarray(beta, dtype=float),
        theta_rad=np.asarray(environment.theta(beta), dtype=float),
        achieved_xyz_m=achieved,
        target_xyz_m=np.asarray(target, dtype=float),
        chart_id=np.zeros(len(beta), dtype=np.int64),
        branch_id=np.zeros(len(beta), dtype=np.int64),
        metrics=metrics,
        provenance={
            "trajectory_id": trajectory_id,
            "family_id": family_id,
            "radius_mm": radius_mm,
            "teacher_policy_id": policy.fingerprint,
            "solver_seed": int(policy.solver_seed),
            "traversal_direction": direction,
            "cyclic_cut": int(cut),
        },
        success=bool(metrics["residual_max_mm"] <= policy.tracking_tolerance_mm),
    )


def _gate_from_config(config: Mapping[str, Any]) -> BranchIdentityGate:
    return BranchIdentityGate(**dict(config["gates"]))


def _repair_policy(
    config: Mapping[str, Any],
    *,
    trust_radius_deg: float,
    reference_seed_mode: str,
    enforce_reference_trust: bool,
    lambda_reference: float | None = None,
) -> BranchRepairPolicy:
    repair = config["repair"]
    return BranchRepairPolicy(
        beta_weights=tuple(float(value) for value in repair["beta_weights"]),
        trust_radius_deg=float(trust_radius_deg),
        candidate_budget=int(repair["candidate_budget"]),
        lambda_reference=float(
            repair["lambda_reference"]
            if lambda_reference is None
            else lambda_reference
        ),
        lambda_previous=float(repair["lambda_previous"]),
        local_seed_std_deg=float(repair["local_seed_std_deg"]),
        root_cluster_threshold_deg=float(
            config["canonical_root"]["cluster_threshold_deg"]
        ),
        reference_seed_mode=str(reference_seed_mode),
        enforce_reference_trust=bool(enforce_reference_trust),
    )


def _solve_audit_variants(
    *,
    environment: Any,
    target: np.ndarray,
    reference: np.ndarray,
    root_phase_idx: int,
    root_beta: np.ndarray,
    family_id: str,
    radius_mm: float,
    cuts: Sequence[int],
    corrector_policy: TeacherPolicy,
    repair_policy: BranchRepairPolicy,
    seed_base: int,
) -> tuple[dict[str, np.ndarray], dict[str, TeacherTrajectory]]:
    teacher = ReferenceBranchTeacher(environment)
    paths: dict[str, np.ndarray] = {}
    trajectories: dict[str, TeacherTrajectory] = {}
    for name, direction, cut in variant_specs(cuts):
        trajectory = teacher.solve(
            target,
            reference_beta=reference,
            root_phase_idx=int(root_phase_idx),
            canonical_root_beta=root_beta,
            direction=direction,
            cut=int(cut) % len(target),
            corrector_policy=corrector_policy,
            repair_policy=repair_policy,
            solver_seed=int(seed_base),
            trajectory_id=f"{family_id}:{name}",
            family_id=family_id,
            radius_mm=radius_mm,
        )
        paths[name] = trajectory.beta_rad
        trajectories[name] = trajectory
    return paths, trajectories


def _write_strategy(
    *,
    directory: Path,
    environment: Any,
    target: np.ndarray,
    consensus: np.ndarray,
    variants: Mapping[str, np.ndarray],
    variant_trajectories: Mapping[str, TeacherTrajectory] | None,
    policy: TeacherPolicy,
    family_id: str,
    radius_mm: float,
    root_phase_idx: int,
    gate: BranchIdentityGate,
    extra: Mapping[str, Any] | None = None,
    additional_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    consensus_trajectory = _trajectory_from_beta(
        environment,
        target,
        consensus,
        policy=policy,
        trajectory_id=f"{family_id}:canonical-consensus",
        family_id=family_id,
        radius_mm=radius_mm,
        direction="consensus",
        cut=int(root_phase_idx),
    )
    _atomic_parquet(
        trajectory_frame(consensus_trajectory, environment),
        directory / "canonical_consensus_branch.parquet",
    )
    variant_root = directory / "audit_variants"
    for name, beta in variants.items():
        trajectory = (
            variant_trajectories[name]
            if variant_trajectories is not None and name in variant_trajectories
            else _trajectory_from_beta(
                environment,
                target,
                beta,
                policy=policy,
                trajectory_id=f"{family_id}:{name}",
                family_id=family_id,
                radius_mm=radius_mm,
                direction="source",
                cut=0,
            )
        )
        _atomic_parquet(
            trajectory_frame(trajectory, environment),
            variant_root / name / "centerline.parquet",
        )
    audit = audit_consensus_variants(
        consensus,
        variants,
        cluster_threshold_deg=float(gate.variant_beta_rms_p95_deg / 2.0),
    )
    _atomic_parquet(audit.per_phase_gap, directory / "per_phase_audit_gap.parquet")
    audit.summary.to_csv(directory / "variant_audit_summary.csv", index=False)
    _atomic_parquet(audit.branch_clusters, directory / "audit_branch_clusters.parquet")
    numerical = _metrics_for_beta(environment, target, consensus)
    solver_success = (
        {name: bool(value.success) for name, value in variant_trajectories.items()}
        if variant_trajectories is not None
        else None
    )
    trust_violations = (
        {
            name: float(value.metrics.get("trust_violation_count", 0.0))
            for name, value in variant_trajectories.items()
        }
        if variant_trajectories is not None
        else None
    )
    gate_report = gate.evaluate(
        numerical_metrics=numerical,
        variant_audit=audit,
        variant_solver_success=solver_success,
        variant_trust_violation_count=trust_violations,
    )
    gate_report["checks"]["candidate_layer_budget"] = True
    gate_report["checks"].update(
        {key: bool(value) for key, value in dict(additional_checks or {}).items()}
    )
    gate_report["gate_pass"] = bool(all(gate_report["checks"].values()))
    summary = {
        "root_phase_idx": int(root_phase_idx),
        "numerical_metrics": numerical,
        "variant_metrics": audit.summary.to_dict("records"),
        "max_branch_cluster_count": int(audit.branch_clusters["cluster_count"].max()),
        **gate_report,
        **dict(extra or {}),
    }
    atomic_write_json(directory / "gate.json", summary)
    return summary


def _candidate_layers(
    environment: Any,
    target: np.ndarray,
    paths: Sequence[np.ndarray],
    *,
    reference: np.ndarray,
    cluster_threshold_deg: float,
    min_candidates: int,
    max_candidates: int,
    corrector_policy: TeacherPolicy,
    solver_seed: int,
    seed_step_deg: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build 8--16 distinct, corrected nodes from paths, predictors and null seeds."""

    minimum = int(min_candidates)
    maximum = int(max_candidates)
    if minimum < 1 or maximum < minimum:
        raise ValueError("candidate limits must satisfy 1 <= min <= max")
    if float(seed_step_deg) <= 0.0:
        raise ValueError("seed_step_deg must be positive")
    all_paths = [np.asarray(reference, dtype=float), *[np.asarray(path) for path in paths]]
    reference_path = np.asarray(reference, dtype=float).reshape(len(target), 6)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    layers: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for phase in range(len(target)):
        kept: list[np.ndarray] = []

        def keep(candidate: np.ndarray) -> bool:
            value = np.asarray(candidate, dtype=float).reshape(6)
            if not np.isfinite(value).all():
                return False
            if np.any(value < bounds[:, 0] - 1.0e-12) or np.any(
                value > bounds[:, 1] + 1.0e-12
            ):
                return False
            if any(
                np.rad2deg(np.sqrt(np.mean(np.square(value - previous))))
                < float(cluster_threshold_deg)
                for previous in kept
            ):
                return False
            kept.append(value.copy())
            return True

        for path in all_paths:
            keep(path[phase])
            if len(kept) >= maximum:
                break
        anchor = reference_path[phase]
        predictor = (
            reference_path[(phase - 1) % len(target)]
            + reference_path[(phase - 1) % len(target)]
            - reference_path[(phase - 2) % len(target)]
        )
        seed_bank = [predictor]
        jacobian = np.asarray(environment.jacobian(anchor), dtype=float).reshape(3, 6)
        _u, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
        tolerance = max(jacobian.shape) * np.finfo(float).eps * max(
            float(singular[0]) if len(singular) else 0.0, 1.0
        )
        null_basis = vh[int(np.sum(singular > tolerance)) :]
        step = math.radians(float(seed_step_deg))
        for direction in null_basis:
            seed_bank.extend([anchor + step * direction, anchor - step * direction])
        rng = np.random.default_rng(int(solver_seed) + phase)
        attempts = 0
        while len(kept) < minimum and attempts < 96:
            if attempts < len(seed_bank):
                seed = seed_bank[attempts]
            else:
                direction = rng.normal(size=6)
                direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
                scale = step * (1.0 + (attempts - len(seed_bank)) // 12)
                seed = anchor + scale * direction
            beta, _residual, _iterations, _success = _correct_target(
                environment,
                target[phase],
                np.clip(seed, bounds[:, 0], bounds[:, 1]),
                corrector_policy,
            )
            keep(beta)
            attempts += 1
        layer = np.vstack(kept)
        achieved = np.asarray(environment.fk(layer), dtype=float).reshape(-1, 3)
        error = np.linalg.norm(achieved - target[phase][None, :], axis=1) * 1000.0
        layers.append(layer)
        residuals.append(error)
    return layers, residuals


def _root_for_candidate(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    atlas: ReachabilityAtlas,
    candidate_id: str,
    primary: pd.DataFrame,
    variants: Mapping[str, pd.DataFrame],
    directory: Path,
) -> dict[str, Any]:
    report = analyze_branch_variants(
        primary,
        variants,
        jacobian=lambda beta: environment.jacobian(beta),
        cluster_threshold_deg=float(config["canonical_root"]["cluster_threshold_deg"]),
    )
    root = select_canonical_root_phase(primary, branch_clusters=report.branch_clusters)
    phase = int(root["phase_idx"])
    target = _frame_target(primary)[phase]
    primary_beta = _frame_beta(primary)[phase]
    count = int(config["canonical_root"]["seed_count"])
    tree = cKDTree(atlas.xyz_m)
    _distance, indices = tree.query(target, k=min(count - 1, len(atlas.xyz_m)))
    seeds = [primary_beta, *atlas.beta_rad[np.atleast_1d(indices)].tolist()]
    policy = _teacher_policy(config, seed=int(config["teacher_policy"]["solver_seed"]))
    solved = []
    residuals = []
    for seed in seeds[:count]:
        beta, residual, _iterations, _success = _correct_target(
            environment, target, np.asarray(seed, dtype=float), policy
        )
        solved.append(beta)
        residuals.append(residual)
    selection = select_canonical_root_solution(
        np.vstack(solved),
        residual_mm=np.asarray(residuals),
        environment=environment,
        safe_margin_deg=float(config["canonical_root"]["safe_margin_deg"]),
        cluster_threshold_deg=float(config["canonical_root"]["cluster_threshold_deg"]),
    )
    directory.mkdir(parents=True, exist_ok=True)
    selection.candidate_table.to_csv(directory / "root_candidates.csv", index=False)
    selection.cluster_table.to_csv(directory / "root_clusters.csv", index=False)
    selected_cluster = selection.cluster_table[
        selection.cluster_table["cluster_id"] == selection.selected_cluster_id
    ].iloc[0]
    payload = {
        **root,
        "candidate_id": candidate_id,
        "canonical_root_branch_id": f"{candidate_id}:cluster{selection.selected_cluster_id}",
        "canonical_root_beta": selection.selected_beta.tolist(),
        "candidate_count": int(len(selection.candidate_table)),
        "root_cluster_count": int(len(selection.cluster_table)),
        "selected_cluster_candidate_count": int(selected_cluster["candidate_count"]),
        "selected_cluster_id": int(selection.selected_cluster_id),
    }
    atomic_write_json(directory / "canonical_root.json", payload)
    return payload


def _run_candidate_experiment(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    candidate_id: str,
    count: int,
    cuts: Sequence[int],
    directory: Path,
    root_stage: Path,
    formal_methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    primary_source, variants_source = _load_source_frames(config, project_root, candidate_id)
    primary, frozen_variants = _aligned_source_variants(
        primary_source, variants_source, count=int(count)
    )
    target = _frame_target(primary)
    primary_beta = _frame_beta(primary)
    family_id, radius_mm = _family_metadata(primary)
    corrector = _teacher_policy(
        config, seed=int(config["teacher_policy"]["solver_seed"])
    )
    root_artifact = json.loads(
        (root_stage / candidate_id / "canonical_root.json").read_text(encoding="utf-8")
    )
    pilot_count = int(config["phase_counts"]["pilot"])
    root_phase = int(
        round(int(root_artifact["phase_idx"]) * len(target) / pilot_count)
    ) % len(target)
    root_beta, _root_residual, _root_iterations, _root_success = _correct_target(
        environment,
        target[root_phase],
        np.asarray(root_artifact["canonical_root_beta"], dtype=float),
        corrector,
    )
    strict_gate = _gate_from_config(config)
    rows: list[dict[str, Any]] = []

    bi0_variants = {
        name: _frame_beta(frame) for name, frame in frozen_variants.items()
    }
    if formal_methods is None or "BI-0" in formal_methods:
        summary = _write_strategy(
            directory=directory / "BI-0",
            environment=environment,
            target=target,
            consensus=primary_beta,
            variants=bi0_variants,
            variant_trajectories=None,
            policy=corrector,
            family_id=family_id,
            radius_mm=radius_mm,
            root_phase_idx=root_phase,
            gate=strict_gate,
            extra={"strategy": "BI-0", "method": "independent_source_variants"},
        )
        rows.append(_ranking_row(candidate_id, "BI-0", summary))

    provisional_policy = _repair_policy(
        config,
        trust_radius_deg=max(
            float(value) for value in config["repair"]["trust_radius_pilot_deg"]
        ),
        reference_seed_mode="none",
        enforce_reference_trust=False,
        lambda_reference=0.0,
    )
    teacher = ReferenceBranchTeacher(environment)
    provisional_forward = teacher.solve(
        target,
        reference_beta=primary_beta,
        root_phase_idx=root_phase,
        canonical_root_beta=root_beta,
        direction="forward",
        cut=root_phase,
        corrector_policy=corrector,
        repair_policy=provisional_policy,
        solver_seed=int(corrector.solver_seed + 100),
        trajectory_id=f"{family_id}:provisional-forward",
        family_id=family_id,
        radius_mm=radius_mm,
    )
    provisional_reverse = teacher.solve(
        target,
        reference_beta=primary_beta,
        root_phase_idx=root_phase,
        canonical_root_beta=root_beta,
        direction="reverse",
        cut=root_phase,
        corrector_policy=corrector,
        repair_policy=provisional_policy,
        solver_seed=int(corrector.solver_seed + 101),
        trajectory_id=f"{family_id}:provisional-reverse",
        family_id=family_id,
        radius_mm=radius_mm,
    )
    provisional = (
        provisional_forward.beta_rad
        if provisional_forward.metrics["residual_max_mm"]
        <= provisional_reverse.metrics["residual_max_mm"]
        else provisional_reverse.beta_rad
    )
    provisional_dir = directory / "provisional_reference"
    _atomic_parquet(
        trajectory_frame(provisional_forward, environment),
        provisional_dir / "forward.parquet",
    )
    _atomic_parquet(
        trajectory_frame(provisional_reverse, environment),
        provisional_dir / "reverse.parquet",
    )

    bi1_policy = _repair_policy(
        config,
        trust_radius_deg=max(
            float(value) for value in config["repair"]["trust_radius_pilot_deg"]
        ),
        reference_seed_mode="first",
        enforce_reference_trust=False,
        lambda_reference=0.0,
    )
    bi1_paths, bi1_trajectories = _solve_audit_variants(
        environment=environment,
        target=target,
        reference=provisional,
        root_phase_idx=root_phase,
        root_beta=root_beta,
        family_id=family_id,
        radius_mm=radius_mm,
        cuts=cuts,
        corrector_policy=corrector,
        repair_policy=bi1_policy,
        seed_base=int(corrector.solver_seed + 200),
    )
    if formal_methods is None or "BI-1" in formal_methods:
        summary = _write_strategy(
            directory=directory / "BI-1",
            environment=environment,
            target=target,
            consensus=provisional,
            variants=bi1_paths,
            variant_trajectories=bi1_trajectories,
            policy=corrector,
            family_id=family_id,
            radius_mm=radius_mm,
            root_phase_idx=root_phase,
            gate=strict_gate,
            extra={"strategy": "BI-1", "method": "shared_canonical_root"},
        )
        rows.append(_ranking_row(candidate_id, "BI-1", summary))

    bi2_results: list[tuple[float, dict[str, np.ndarray], dict[str, TeacherTrajectory], dict[str, Any]]] = []
    trust_values = (
        [float(config["repair"]["selected_trust_radius_deg"])]
        if formal_methods is not None
        else [float(value) for value in config["repair"]["trust_radius_pilot_deg"]]
    )
    for trust in trust_values:
        policy = _repair_policy(
            config,
            trust_radius_deg=trust,
            reference_seed_mode="all",
            enforce_reference_trust=True,
        )
        paths, trajectories = _solve_audit_variants(
            environment=environment,
            target=target,
            reference=provisional,
            root_phase_idx=root_phase,
            root_beta=root_beta,
            family_id=family_id,
            radius_mm=radius_mm,
            cuts=cuts,
            corrector_policy=corrector,
            repair_policy=policy,
            seed_base=int(corrector.solver_seed + 300 + round(100 * trust)),
        )
        trust_name = str(trust).replace(".", "p")
        summary = _write_strategy(
            directory=directory / "BI-2" / f"trust_{trust_name}deg",
            environment=environment,
            target=target,
            consensus=provisional,
            variants=paths,
            variant_trajectories=trajectories,
            policy=corrector,
            family_id=family_id,
            radius_mm=radius_mm,
            root_phase_idx=root_phase,
            gate=strict_gate,
            extra={
                "strategy": "BI-2",
                "method": "root_plus_reference_continuation",
                "trust_radius_deg": trust,
            },
        )
        bi2_results.append((trust, paths, trajectories, summary))
    bi2_results.sort(
        key=lambda item: (
            -sum(bool(value) for value in item[3]["checks"].values()),
            max(
                row["gap_p95_deg"]
                for row in item[3]["variant_metrics"]
                if row["variant"] != "repeat"
            ),
            abs(item[0] - float(config["repair"]["selected_trust_radius_deg"])),
        )
    )
    best_trust, bi2_paths, bi2_trajectories, bi2_summary = bi2_results[0]
    if formal_methods is None or "BI-2" in formal_methods:
        rows.append(_ranking_row(candidate_id, "BI-2", bi2_summary))

    pool_paths = [
        *bi0_variants.values(),
        *bi1_paths.values(),
        *bi2_paths.values(),
        provisional_forward.beta_rad,
        provisional_reverse.beta_rad,
    ]
    layers, residuals = _candidate_layers(
        environment,
        target,
        pool_paths,
        reference=provisional,
        cluster_threshold_deg=float(config["canonical_root"]["cluster_threshold_deg"]),
        min_candidates=int(config["repair"]["graph_candidates_min_per_phase"]),
        max_candidates=int(config["repair"]["graph_candidates_per_phase"]),
        corrector_policy=corrector,
        solver_seed=int(corrector.solver_seed + 350),
        seed_step_deg=float(config["repair"]["graph_seed_step_deg"]),
    )
    graph_consensus, graph_report = link_reference_cyclic_candidates(
        layers,
        residuals,
        reference_beta=provisional,
        root_phase_idx=root_phase,
        canonical_root_beta=root_beta,
        beta_weights=np.asarray(config["repair"]["beta_weights"], dtype=float),
        lambda_velocity=float(config["repair"]["graph_lambda_velocity"]),
        lambda_reference=float(config["repair"]["graph_lambda_reference"]),
        closure_weight=float(config["repair"]["graph_closure_weight"]),
        root_cluster_threshold_deg=float(config["canonical_root"]["cluster_threshold_deg"]),
        lambda_acceleration=float(config["repair"]["graph_lambda_acceleration"]),
    )
    if not graph_report.get("success", False):
        graph_consensus = provisional.copy()
    graph_policy = _repair_policy(
        config,
        trust_radius_deg=float(best_trust),
        reference_seed_mode="all",
        enforce_reference_trust=True,
    )
    bi3_paths, bi3_trajectories = _solve_audit_variants(
        environment=environment,
        target=target,
        reference=graph_consensus,
        root_phase_idx=root_phase,
        root_beta=root_beta,
        family_id=family_id,
        radius_mm=radius_mm,
        cuts=cuts,
        corrector_policy=corrector,
        repair_policy=graph_policy,
        seed_base=int(corrector.solver_seed + 400),
    )
    bi3_summary = _write_strategy(
        directory=directory / "BI-3",
        environment=environment,
        target=target,
        consensus=graph_consensus,
        variants=bi3_paths,
        variant_trajectories=bi3_trajectories,
        policy=corrector,
        family_id=family_id,
        radius_mm=radius_mm,
        root_phase_idx=root_phase,
        gate=strict_gate,
        extra={
            "strategy": "BI-3",
            "method": "cyclic_candidate_graph",
            "selected_trust_radius_deg": float(best_trust),
            "graph_report": graph_report,
            "candidate_layer_count_min": int(min(map(len, layers))),
            "candidate_layer_count_max": int(max(map(len, layers))),
        },
        additional_checks={
            "candidate_layer_budget": bool(
                min(map(len, layers))
                >= int(config["repair"]["graph_candidates_min_per_phase"])
                and max(map(len, layers))
                <= int(config["repair"]["graph_candidates_per_phase"])
            )
        },
    )
    if formal_methods is None or "BI-3" in formal_methods:
        rows.append(_ranking_row(candidate_id, "BI-3", bi3_summary))

    optimized = optimize_consensus_branch(
        environment,
        target,
        graph_consensus,
        reference_beta=graph_consensus,
        root_phase_idx=root_phase,
        canonical_root_beta=root_beta,
        policy=corrector,
        reference_weight=float(config["repair"]["consensus_reference_weight"]),
        reference_scale_deg=float(config["repair"]["consensus_reference_scale_deg"]),
        root_cluster_threshold_deg=float(config["canonical_root"]["cluster_threshold_deg"]),
        trajectory_id=f"{family_id}:BI-4-consensus",
        family_id=family_id,
        radius_mm=radius_mm,
    )
    bi4_paths, bi4_trajectories = _solve_audit_variants(
        environment=environment,
        target=target,
        reference=optimized.beta_rad,
        root_phase_idx=root_phase,
        root_beta=root_beta,
        family_id=family_id,
        radius_mm=radius_mm,
        cuts=cuts,
        corrector_policy=corrector,
        repair_policy=graph_policy,
        seed_base=int(corrector.solver_seed + 500),
    )
    bi4_summary = _write_strategy(
        directory=directory / "BI-4",
        environment=environment,
        target=target,
        consensus=optimized.beta_rad,
        variants=bi4_paths,
        variant_trajectories=bi4_trajectories,
        policy=corrector,
        family_id=family_id,
        radius_mm=radius_mm,
        root_phase_idx=root_phase,
        gate=strict_gate,
        extra={
            "strategy": "BI-4",
            "method": "graph_plus_consensus_whole_curve_optimization",
            "selected_trust_radius_deg": float(best_trust),
            "optimizer_success": bool(optimized.success),
            "optimizer_metrics": dict(optimized.metrics),
        },
        additional_checks={
            "candidate_layer_budget": bool(
                min(map(len, layers))
                >= int(config["repair"]["graph_candidates_min_per_phase"])
                and max(map(len, layers))
                <= int(config["repair"]["graph_candidates_per_phase"])
            )
        },
    )
    if formal_methods is None or "BI-4" in formal_methods:
        rows.append(_ranking_row(candidate_id, "BI-4", bi4_summary))
    return rows


def _ranking_row(candidate_id: str, strategy: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    variants = pd.DataFrame(summary["variant_metrics"])
    traversal = variants[variants["variant"] != "repeat"]
    repeat = variants[variants["variant"] == "repeat"]
    return {
        "candidate_id": str(candidate_id),
        "strategy": str(strategy),
        "gate_pass": bool(summary["gate_pass"]),
        "passed_check_count": int(sum(bool(value) for value in summary["checks"].values())),
        "check_count": int(len(summary["checks"])),
        "residual_p95_mm": float(summary["numerical_metrics"]["residual_p95_mm"]),
        "residual_max_mm": float(summary["numerical_metrics"]["residual_max_mm"]),
        "joint_margin_min_deg": float(summary["numerical_metrics"]["joint_margin_min_deg"]),
        "repeat_gap_p95_deg": float(repeat["gap_p95_deg"].max()) if not repeat.empty else math.inf,
        "variant_gap_p95_deg": float(traversal["gap_p95_deg"].max()) if not traversal.empty else math.inf,
        "variant_gap_max_deg": float(traversal["gap_max_deg"].max()) if not traversal.empty else math.inf,
        "variant_ratio_gt_1deg": float(traversal["ratio_gap_gt_1deg"].max()) if not traversal.empty else math.inf,
        "max_branch_cluster_count": int(summary["max_branch_cluster_count"]),
    }


def _trajectory_inventories_complete(stage: Path, expected_count: int) -> bool:
    paths = [
        *stage.rglob("canonical_consensus_branch.parquet"),
        *stage.rglob("centerline.parquet"),
        *stage.rglob("provisional_reference/*.parquet"),
    ]
    if not paths:
        return False
    expected = np.arange(int(expected_count), dtype=np.int64)
    for path in paths:
        phase = pd.read_parquet(path, columns=["phase_idx"])["phase_idx"].to_numpy(
            dtype=np.int64
        )
        if len(phase) != len(expected) or not np.array_equal(np.sort(phase), expected):
            return False
    return True


def run_protocol(
    *, config: Mapping[str, Any], source_root: Path, project_root: Path, output: Path
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["protocol"]
    stage.mkdir(parents=True, exist_ok=True)
    source_anchor = _source_anchor_root(config, project_root)
    source_gate_path = source_anchor / "gate.json"
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    resolved = copy.deepcopy(dict(config))
    resolved.pop("config_path", None)
    (stage / "protocol_v11_3.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    inputs: dict[str, str] = {
        "config": sha256_file(Path(str(config["config_path"]))),
        "source_phase1_gate": sha256_file(source_gate_path),
        "runner": sha256_file(Path(__file__).resolve()),
        "branch_identity_module": sha256_file(
            source_root / "src/quasi_exp/teacher/branch_identity.py"
        ),
    }
    for candidate_id in _candidate_ids(config):
        root = _source_candidate_root(config, project_root, candidate_id)
        for path in sorted(root.glob("*/centerline.parquet")):
            inputs[f"{candidate_id}/{path.parent.name}"] = sha256_file(path)
    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "source_protocol_id": str(config["source_protocol_id"]),
        "input_sha256": inputs,
        "runtime": runtime_fingerprint(),
        "manifest_sha256": _canonical_sha(inputs),
    }
    atomic_write_json(stage / "artifact_manifest.json", manifest)
    return _write_gate(
        stage / "gate.json",
        checks={
            "source_phase1_failure_preserved": bool(source_gate["gate_pass"] is False),
            "three_frozen_candidates_present": bool(len(_candidate_ids(config)) == 3),
            "tube_not_authorized": bool(
                config["protocol_constraints"]["tube_authorized"] is False
            ),
            "source_inputs_hashed": bool(all(len(value) == 64 for value in inputs.values())),
        },
        protocol_id=str(config["protocol_id"]),
        source_phase1_gate_pass=bool(source_gate["gate_pass"]),
        manifest_sha256=manifest["manifest_sha256"],
    )


def run_forensics(
    *, config: Mapping[str, Any], project_root: Path, environment: Any, output: Path
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["forensics"]
    summaries = []
    count = int(config["phase_counts"]["forensics"])
    for candidate_id in _candidate_ids(config):
        report, _primary, variants = _forensics_for_candidate(
            config=config,
            project_root=project_root,
            environment=environment,
            candidate_id=candidate_id,
            count=count,
            directory=stage / candidate_id,
        )
        summaries.extend(
            {"candidate_id": candidate_id, **row}
            for row in report.variant_summary.to_dict("records")
        )
        if len(variants) < 8:
            raise RuntimeError(f"{candidate_id} has fewer than eight audit variants")
    pd.DataFrame(summaries).to_csv(stage / "forensics_summary.csv", index=False)
    return _write_gate(
        stage / "gate.json",
        checks={
            "three_candidates_audited": len(set(row["candidate_id"] for row in summaries)) == 3,
            "all_repeat_plus_seven_variants_present": len(summaries) >= 24,
            "branch_difference_reproduced": max(row["gap_p95_deg"] for row in summaries) > 5.0,
            "phase_inventory_complete": count == int(config["phase_counts"]["forensics"]),
        },
        phase_count=count,
        summary_row_count=len(summaries),
    )


def run_roots(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    atlas: ReachabilityAtlas,
    output: Path,
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["root"]
    descriptions = []
    count = int(config["phase_counts"]["pilot"])
    for candidate_id in map(str, config["candidates"]["pilot"]):
        primary_source, variants_source = _load_source_frames(config, project_root, candidate_id)
        primary, variants = _aligned_source_variants(
            primary_source, variants_source, count=count
        )
        descriptions.append(
            _root_for_candidate(
                config=config,
                project_root=project_root,
                environment=environment,
                atlas=atlas,
                candidate_id=candidate_id,
                primary=primary,
                variants=variants,
                directory=stage / candidate_id,
            )
        )
    return _write_gate(
        stage / "gate.json",
        checks={
            "two_pilot_roots_frozen": len(descriptions) == 2,
            "root_multiseed_budget_complete": all(
                row["candidate_count"] == int(config["canonical_root"]["seed_count"])
                for row in descriptions
            ),
            "unique_root_branch_id": len(
                {row["canonical_root_branch_id"] for row in descriptions}
            )
            == len(descriptions),
        },
        roots=descriptions,
    )


def run_pilot(
    *, config: Mapping[str, Any], project_root: Path, environment: Any, output: Path
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["pilot"]
    rows: list[dict[str, Any]] = []
    count = int(config["phase_counts"]["pilot"])
    cuts = config["variants"]["pilot_cuts"]
    for candidate_id in map(str, config["candidates"]["pilot"]):
        rows.extend(
            _run_candidate_experiment(
                config=config,
                project_root=project_root,
                environment=environment,
                candidate_id=candidate_id,
                count=count,
                cuts=cuts,
                directory=stage / candidate_id,
                root_stage=output / STAGE_DIRS["root"],
            )
        )
    # A2_134 remains a BI-0 negative control and cannot win method selection.
    control = str(config["candidates"]["negative_control"][0])
    primary_source, variants_source = _load_source_frames(config, project_root, control)
    primary, variants = _aligned_source_variants(primary_source, variants_source, count=count)
    family_id, radius_mm = _family_metadata(primary)
    policy = _teacher_policy(config, seed=int(config["teacher_policy"]["solver_seed"]))
    forensics = analyze_branch_variants(
        primary,
        variants,
        jacobian=lambda beta: environment.jacobian(beta),
        cluster_threshold_deg=float(config["canonical_root"]["cluster_threshold_deg"]),
    )
    root = select_canonical_root_phase(primary, branch_clusters=forensics.branch_clusters)
    control_summary = _write_strategy(
        directory=stage / control / "BI-0",
        environment=environment,
        target=_frame_target(primary),
        consensus=_frame_beta(primary),
        variants={name: _frame_beta(frame) for name, frame in variants.items()},
        variant_trajectories=None,
        policy=policy,
        family_id=family_id,
        radius_mm=radius_mm,
        root_phase_idx=int(root["phase_idx"]),
        gate=_gate_from_config(config),
        extra={"strategy": "BI-0", "negative_control": True},
    )
    rows.append(_ranking_row(control, "BI-0", control_summary))
    ranking = pd.DataFrame(rows).sort_values(
        ["gate_pass", "passed_check_count", "variant_gap_p95_deg", "joint_margin_min_deg"],
        ascending=[False, False, True, False],
        kind="stable",
    )
    ranking.to_csv(stage / "strategy_ranking.csv", index=False)
    eligible = ranking[
        ranking["candidate_id"].isin(config["candidates"]["pilot"])
        & ranking["strategy"].isin(["BI-1", "BI-2", "BI-3", "BI-4"])
    ]
    method_score = (
        eligible.groupby("strategy", as_index=False)
        .agg(
            gate_pass_count=("gate_pass", "sum"),
            passed_check_count=("passed_check_count", "sum"),
            variant_gap_p95_deg=("variant_gap_p95_deg", "max"),
            joint_margin_min_deg=("joint_margin_min_deg", "min"),
        )
        .sort_values(
            ["gate_pass_count", "passed_check_count", "variant_gap_p95_deg", "joint_margin_min_deg"],
            ascending=[False, False, True, False],
            kind="stable",
        )
    )
    method_score.to_csv(stage / "method_ranking.csv", index=False)
    top_methods = method_score.head(2)["strategy"].astype(str).tolist()
    atomic_write_json(stage / "selected_formal_methods.json", {"methods": top_methods})
    return _write_gate(
        stage / "gate.json",
        checks={
            "two_candidates_complete": set(config["candidates"]["pilot"]).issubset(
                set(ranking["candidate_id"])
            ),
            "five_ablation_groups_complete": all(
                int((ranking["strategy"] == strategy).sum()) >= 2
                for strategy in ("BI-0", "BI-1", "BI-2", "BI-3", "BI-4")
            ),
            "negative_control_is_bi0_only": int(
                (ranking["candidate_id"] == control).sum()
            )
            == 1,
            "two_methods_selected_for_formal": len(top_methods) == 2,
        },
        phase_count=count,
        selected_formal_methods=top_methods,
        ranking_rows=int(len(ranking)),
    )


def run_formal(
    *, config: Mapping[str, Any], project_root: Path, environment: Any, output: Path
) -> dict[str, Any]:
    pilot_selection = json.loads(
        (output / STAGE_DIRS["pilot"] / "selected_formal_methods.json").read_text(
            encoding="utf-8"
        )
    )
    methods = set(map(str, pilot_selection["methods"]))
    stage = output / STAGE_DIRS["formal"]
    rows: list[dict[str, Any]] = []
    count = int(config["phase_counts"]["formal"])
    for candidate_id in map(str, config["candidates"]["pilot"]):
        rows.extend(
            _run_candidate_experiment(
                config=config,
                project_root=project_root,
                environment=environment,
                candidate_id=candidate_id,
                count=count,
                cuts=config["variants"]["formal_cuts"],
                directory=stage / candidate_id,
                root_stage=output / STAGE_DIRS["root"],
                formal_methods=methods,
            )
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["gate_pass", "passed_check_count", "variant_gap_p95_deg", "joint_margin_min_deg"],
        ascending=[False, False, True, False],
        kind="stable",
    )
    ranking.to_csv(stage / "formal_ranking.csv", index=False)
    restart_authorized = bool(ranking["gate_pass"].any())
    gate = _write_gate(
        stage / "gate.json",
        checks={
            "two_methods_rerun_at_720_phase": set(ranking["strategy"]) == methods,
            "two_candidates_rerun_at_720_phase": set(ranking["candidate_id"])
            == set(map(str, config["candidates"]["pilot"])),
            "all_formal_rows_have_complete_phase_inventory": (
                count == int(config["phase_counts"]["formal"])
                and _trajectory_inventories_complete(stage, count)
            ),
            "at_least_one_candidate_method_passes": restart_authorized,
        },
        phase_count=count,
        selected_methods=sorted(methods),
        phase1_restart_authorized=restart_authorized,
        tube_authorized=False,
        passing_candidate_methods=ranking[ranking["gate_pass"]][
            ["candidate_id", "strategy"]
        ].to_dict("records"),
    )
    atomic_write_json(
        output / "BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json",
        {
            "protocol_id": str(config["protocol_id"]),
            "branch_identity_experiment_complete": True,
            "phase1_restart_authorized": restart_authorized,
            "tube_authorized": False,
            "formal_stage_gate_pass": bool(gate["gate_pass"]),
        },
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generalized_ellipse_region_v11_branch_identity.yaml",
    )
    parser.add_argument("--preset", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--stage",
        choices=("all", "protocol", "forensics", "root", "pilot", "formal"),
        default="all",
    )
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    project_root = project_root_from(source_root)
    config = load_config(source_root / args.config, preset=args.preset)
    output = project_root / str(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    environment = load_environment(
        project_root, project_root / str(config["robot_config"])
    )
    atlas = _reachability_atlas(project_root, config)
    selected = (
        list(STAGE_DIRS)
        if args.stage == "all"
        else [str(args.stage)]
    )
    for stage in selected:
        if stage == "protocol":
            run_protocol(
                config=config,
                source_root=source_root,
                project_root=project_root,
                output=output,
            )
        elif stage == "forensics":
            run_forensics(
                config=config,
                project_root=project_root,
                environment=environment,
                output=output,
            )
        elif stage == "root":
            run_roots(
                config=config,
                project_root=project_root,
                environment=environment,
                atlas=atlas,
                output=output,
            )
        elif stage == "pilot":
            run_pilot(
                config=config,
                project_root=project_root,
                environment=environment,
                output=output,
            )
        elif stage == "formal":
            run_formal(
                config=config,
                project_root=project_root,
                environment=environment,
                output=output,
            )
    print(output)


if __name__ == "__main__":
    main()
