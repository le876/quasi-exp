#!/usr/bin/env python3
"""Challenge the teacher at 0.5, 0.75 and 1.0 m actual major semiaxes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import yaml

from quasi_exp.teacher.atlas import build_local_atlas
from quasi_exp.teacher.canonical import (
    CanonicalTeacher,
    TeacherPolicy,
    TeacherVariant,
    TrajectorySpec,
)
from quasi_exp.teacher.dataset import evaluate_centerline_gate, trajectory_frame
from quasi_exp.teacher.experiment import ExperimentManifest, atomic_write_json
from quasi_exp.teacher.large_scale import (
    EllipseChallenge,
    ReachabilityAtlas,
    assess_chain_length_necessity,
    fit_ellipse_pose_to_atlas,
)

from run_trajectory_canonical_teacher_v10 import (
    load_environment,
    project_root_from,
    runtime_fingerprint,
)


def _sample_reachability_atlas(environment: Any, *, count: int, seed: int, margin_deg: float) -> ReachabilityAtlas:
    sample_count = int(count)
    exponent = int(np.log2(sample_count))
    if 2**exponent != sample_count:
        raise ValueError("reachability atlas sample_count must be a power of two")
    margin = np.deg2rad(float(margin_deg))
    low = np.asarray(environment.bounds[:, 0], dtype=float) + margin
    high = np.asarray(environment.bounds[:, 1], dtype=float) - margin
    if np.any(low > high):
        raise ValueError("sampling margin removes the registered joint domain")
    unit = qmc.Sobol(d=6, scramble=True, seed=int(seed)).random_base2(exponent)
    beta = qmc.scale(unit, low, high)
    beta = np.vstack([np.zeros((1, 6), dtype=float), beta])
    return ReachabilityAtlas(xyz_m=environment.fk(beta), beta_rad=beta)


def _workspace_diameter_witness(environment: Any, *, seed: int) -> dict[str, Any]:
    bounds = [tuple(float(value) for value in row) for row in environment.bounds] * 2

    def objective(flat: np.ndarray) -> float:
        xyz = environment.fk(np.asarray(flat, dtype=float).reshape(2, 6))
        return -float(np.linalg.norm(xyz[0] - xyz[1]))

    result = differential_evolution(
        objective,
        bounds,
        seed=int(seed),
        maxiter=180,
        popsize=12,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    beta = np.asarray(result.x, dtype=float).reshape(2, 6)
    xyz = environment.fk(beta)
    return {
        "semantics": "numerical_lower_bound_witness_not_hard_upper_bound",
        "distance_m": float(-result.fun),
        "beta_deg": np.rad2deg(beta).tolist(),
        "xyz_m": xyz.tolist(),
        "midpoint_m": np.mean(xyz, axis=0).tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "evaluations": int(result.nfev),
    }


def _teacher_policy(config: dict[str, Any], variant: TeacherVariant, seed: int) -> TeacherPolicy:
    screen = config["teacher_screen"]
    if variant == TeacherVariant.T1:
        weights = dict(
            lambda_velocity=0.1,
            lambda_acceleration=0.0,
            lambda_posture=0.01,
            lambda_conditioning=0.0,
        )
    elif variant == TeacherVariant.T4:
        weights = dict(
            lambda_velocity=10.0,
            lambda_acceleration=1.0,
            lambda_posture=0.2,
            lambda_conditioning=0.1,
        )
    else:
        raise ValueError(f"unsupported large-scale screen variant: {variant.value}")
    return TeacherPolicy(
        variant=variant,
        candidate_budget=int(screen["candidate_budget"]),
        solver_seed=int(seed),
        max_corrector_iterations=int(screen["max_corrector_iterations"]),
        tracking_tolerance_mm=float(screen["tracking_tolerance_mm"]),
        safe_joint_margin_deg=1.5,
        **weights,
    )


def _challenge_slug(major_semiaxis_m: float) -> str:
    return f"a{major_semiaxis_m:0.3f}m".replace(".", "p")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.protocol_config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    requested = tuple(float(value) for value in config["major_semiaxes_m"])
    if requested != (0.5, 0.75, 1.0):
        raise ValueError("the registered challenge set must remain exactly 0.5/0.75/1.0 m")
    challenges = tuple(
        EllipseChallenge.from_major_semiaxis_m(
            value,
            minor_to_major_ratio=float(config["minor_to_major_ratio"]),
        )
        for value in requested
    )
    runtime = runtime_fingerprint()
    worker = Path(__file__).resolve()
    source_root = worker.parents[2]
    manifest = ExperimentManifest.create(
        protocol_id=str(config["protocol_id"]),
        teacher_variants=tuple(str(value) for value in config["teacher_screen"]["variants"]),
        family_id="optimized_large_scale_pose_v10",
        radii_mm=tuple(challenge.radius_parameter_m * 1000.0 for challenge in challenges),
        phase_count=int(config["pose_search"]["phase_count"]),
        tube_offsets_mm=(),
        solver_seed=int(config["solver_seed"]),
        input_files=(config_path, Path(args.robot_config).resolve()),
        worker_code_files=(
            worker,
            source_root / "src/quasi_exp/teacher/large_scale.py",
            source_root / "src/quasi_exp/teacher/canonical.py",
            source_root / "src/quasi_exp/teacher/forward.py",
            source_root / "src/quasi_exp/teacher/dataset.py",
            source_root / "src/quasi_exp/teacher/atlas.py",
        ),
        runtime_fingerprint=runtime,
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest.as_dict():
            raise RuntimeError("large-scale output has a different frozen manifest")
    atomic_write_json(manifest_path, manifest.as_dict())
    atomic_write_json(output / "runtime.json", runtime)

    environment = load_environment(project_root, Path(args.robot_config))
    atlas_started = time.perf_counter()
    atlas_config = config["reachability_atlas"]
    atlas = _sample_reachability_atlas(
        environment,
        count=int(atlas_config["sample_count"]),
        seed=int(config["solver_seed"]),
        margin_deg=float(atlas_config["sampling_margin_deg"]),
    )
    atlas_wall = time.perf_counter() - atlas_started
    atlas_frame = pd.DataFrame(atlas.xyz_m, columns=["x_m", "y_m", "z_m"])
    for index in range(6):
        atlas_frame[f"beta{index + 1}_rad"] = atlas.beta_rad[:, index]
    atlas_frame.to_parquet(output / "reachability_atlas.parquet", index=False)
    workspace = {
        "sample_count": int(len(atlas.xyz_m)),
        "wall_time_s": float(atlas_wall),
        "xyz_min_m": np.min(atlas.xyz_m, axis=0).tolist(),
        "xyz_max_m": np.max(atlas.xyz_m, axis=0).tolist(),
        "axis_span_m": np.ptp(atlas.xyz_m, axis=0).tolist(),
        "radial_norm_min_m": float(np.min(np.linalg.norm(atlas.xyz_m, axis=1))),
        "radial_norm_max_m": float(np.max(np.linalg.norm(atlas.xyz_m, axis=1))),
        "diameter_witness": _workspace_diameter_witness(
            environment, seed=int(config["solver_seed"])
        ),
    }
    atomic_write_json(output / "workspace_report.json", workspace)
    chain_rows = assess_chain_length_necessity(
        challenges,
        robot_max_reach_m=float(config["robot_max_reach_m"]),
        tube_radius_m=float(config["tube_radius_m"]),
    )
    atomic_write_json(output / "chain_length_gate.json", chain_rows)

    teacher = CanonicalTeacher(environment)
    pose_config = config["pose_search"]
    reports: list[dict[str, Any]] = []
    for challenge_index, (challenge, chain_gate) in enumerate(zip(challenges, chain_rows)):
        challenge_dir = output / _challenge_slug(challenge.major_semiaxis_m)
        challenge_dir.mkdir(parents=True, exist_ok=True)
        fits = [
            fit_ellipse_pose_to_atlas(
                challenge,
                atlas,
                phase_count=int(pose_config["phase_count"]),
                seed=int(config["solver_seed"]) + challenge_index * 100 + restart,
                max_iterations=int(pose_config["max_iterations"]),
                population_size=int(pose_config["population_size"]),
            )
            for restart in range(int(pose_config["restarts"]))
        ]
        fit = min(fits, key=lambda value: value.objective_mm)
        pose_report = {
            "major_semiaxis_m": challenge.major_semiaxis_m,
            "minor_semiaxis_m": challenge.minor_semiaxis_m,
            "major_diameter_m": challenge.major_diameter_m,
            "legacy_radius_parameter_m": challenge.radius_parameter_m,
            "scale_over_legacy_100mm": challenge.major_semiaxis_m
            / (float(config["baseline_legacy_R_m"]) * float(config["legacy_major_axis_gain"])),
            "center_m": fit.center_m.tolist(),
            "major_direction": fit.major_direction.tolist(),
            "minor_direction": fit.minor_direction.tolist(),
            "atlas_match": dict(fit.match.metrics),
            "pose_objective_mm": fit.objective_mm,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
            "evaluations": fit.evaluations,
            "restart_objectives_mm": [value.objective_mm for value in fits],
            "max_target_norm_m": float(np.max(np.linalg.norm(fit.target_xyz_m, axis=1))),
            "min_chain_reach_margin_m": float(
                float(config["robot_max_reach_m"])
                - np.max(np.linalg.norm(fit.target_xyz_m, axis=1))
            ),
            "chain_length_necessity": chain_gate,
        }
        atomic_write_json(challenge_dir / "pose_report.json", pose_report)
        target_frame = pd.DataFrame(
            fit.target_xyz_m, columns=["target_x_m", "target_y_m", "target_z_m"]
        )
        target_frame["nearest_distance_mm"] = fit.match.nearest_distance_mm
        for index in range(6):
            target_frame[f"initial_beta{index + 1}_rad"] = fit.match.initial_beta_path_rad[:, index]
        target_frame.to_parquet(challenge_dir / "pose_targets.parquet", index=False)

        teacher_reports: dict[str, Any] = {}
        for variant_name in config["teacher_screen"]["variants"]:
            variant = TeacherVariant(str(variant_name))
            policy = _teacher_policy(config, variant, int(config["solver_seed"]))
            spec = TrajectorySpec(
                trajectory_id=f"large-scale@a{challenge.major_semiaxis_m:g}m",
                family_id=f"optimized_a{challenge.major_semiaxis_m:g}m",
                radius_mm=challenge.radius_parameter_m * 1000.0,
                target_xyz_m=fit.target_xyz_m,
            )
            solve_started = time.perf_counter()
            trajectory = teacher.solve(
                spec,
                policy,
                root_beta=fit.match.initial_beta_path_rad[0],
                initial_beta_path=fit.match.initial_beta_path_rad,
            )
            solve_wall = time.perf_counter() - solve_started
            atlas_report: dict[str, Any] | None = None
            if variant == TeacherVariant.T4:
                local_atlas = build_local_atlas(trajectory, environment, policy)
                trajectory = replace(trajectory, chart_id=local_atlas.phase_chart_ids)
                trajectory.metrics["chart_overlap_gap_p95_deg"] = local_atlas.overlap_gap_p95_deg
                trajectory.metrics["chart_overlap_gate_pass"] = float(local_atlas.overlap_gate_pass)
                atlas_report = {
                    "chart_count": len(local_atlas.charts),
                    "overlap_gap_p95_deg": local_atlas.overlap_gap_p95_deg,
                    "overlap_gate_pass": local_atlas.overlap_gate_pass,
                }
            gate = evaluate_centerline_gate(
                trajectory, thresholds=config["gates"]["centerline"]
            )
            if atlas_report is not None:
                gate["checks"]["chart_overlap"] = bool(atlas_report["overlap_gate_pass"])
                gate["centerline_gate_pass"] = bool(all(gate["checks"].values()))
            variant_dir = challenge_dir / variant.value
            variant_dir.mkdir(parents=True, exist_ok=True)
            trajectory_frame(trajectory, environment).to_parquet(
                variant_dir / "centerline.parquet", index=False
            )
            variant_report = {
                "variant": variant.value,
                "metrics": dict(trajectory.metrics),
                "provenance": dict(trajectory.provenance),
                "wall_time_s": float(solve_wall),
                "atlas": atlas_report,
                **gate,
            }
            atomic_write_json(variant_dir / "report.json", variant_report)
            teacher_reports[variant.value] = variant_report

        t1_pass = bool(teacher_reports["T1"]["centerline_gate_pass"])
        t4_pass = bool(teacher_reports["T4"]["centerline_gate_pass"])
        if t4_pass and not t1_pass:
            attribution = "method_enabled_centerline_scale_gain"
        elif t4_pass and t1_pass:
            attribution = "geometry_enabled_not_T4_specific"
        elif not t4_pass and not t1_pass:
            attribution = "centerline_challenge_failed"
        else:
            attribution = "baseline_only_unexpected"
        report = {
            "challenge": pose_report,
            "teachers": teacher_reports,
            "qualitative_threshold_reached": bool(
                challenge.major_semiaxis_m >= float(config["qualitative_scale_threshold_m"])
            ),
            "centerline_attribution": attribution,
            "centerline_challenge_pass": t4_pass,
            "tube_challenge_status": "not_run_until_centerline_pass",
        }
        atomic_write_json(challenge_dir / "report.json", report)
        reports.append(report)

    summary = {
        "protocol_id": str(config["protocol_id"]),
        "challenge_definition": str(config["challenge_definition"]),
        "major_semiaxes_m": list(requested),
        "workspace": workspace,
        "challenges": reports,
        "completed": True,
        "wall_time_s": float(time.time() - started),
    }
    atomic_write_json(output / "report.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_project = project_root_from(Path(__file__).resolve().parents[2])
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=default_project / "configs/robot_rods_only_standard_100k.yaml",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "configs/large_scale_ellipse_challenge_v10.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_project
        / "runs/trajectory_canonical_teacher_v10/07_large_scale_challenge",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))
