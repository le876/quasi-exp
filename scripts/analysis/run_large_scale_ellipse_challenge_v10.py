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
from quasi_exp.teacher.dataset import (
    evaluate_centerline_gate,
    evaluate_tube_gate,
    measured_multi_branch_ratio,
    trajectory_frame,
)
from quasi_exp.teacher.experiment import ExperimentManifest, atomic_write_json
from quasi_exp.teacher.large_scale import (
    REGISTERED_MAJOR_AXIS_GAIN,
    REGISTERED_MINOR_TO_MAJOR_RATIO,
    EllipseChallenge,
    ReachabilityAtlas,
    assess_chain_length_necessity,
    fit_ellipse_pose_to_atlas,
    is_promising_centerline_screen,
)

from run_trajectory_canonical_teacher_v10 import (
    load_environment,
    normal_frame,
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
    elif variant == TeacherVariant.T3:
        weights = dict(
            lambda_velocity=5.0,
            lambda_acceleration=0.5,
            lambda_posture=0.1,
            lambda_conditioning=0.05,
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


def _solve_centerline(
    *,
    teacher: CanonicalTeacher,
    environment: Any,
    config: dict[str, Any],
    variant: TeacherVariant,
    challenge: EllipseChallenge,
    targets: np.ndarray,
    initial_beta_path: np.ndarray,
    output: Path,
) -> tuple[Any, Any | None, dict[str, Any]]:
    policy = _teacher_policy(config, variant, int(config["solver_seed"]))
    spec = TrajectorySpec(
        trajectory_id=f"large-scale@a{challenge.major_semiaxis_m:g}m",
        family_id=f"optimized_a{challenge.major_semiaxis_m:g}m",
        radius_mm=challenge.radius_parameter_m * 1000.0,
        target_xyz_m=targets,
    )
    solve_started = time.perf_counter()
    trajectory = teacher.solve(
        spec,
        policy,
        root_beta=initial_beta_path[0],
        initial_beta_path=initial_beta_path,
    )
    solve_wall = time.perf_counter() - solve_started
    local_atlas = None
    atlas_report: dict[str, Any] | None = None
    if variant == TeacherVariant.T4:
        local_atlas = build_local_atlas(
            trajectory,
            environment,
            policy,
            stride=max(1, len(targets) // 15),
        )
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
    output.mkdir(parents=True, exist_ok=True)
    trajectory_frame(trajectory, environment).to_parquet(
        output / "centerline.parquet", index=False
    )
    report = {
        "variant": variant.value,
        "metrics": dict(trajectory.metrics),
        "provenance": dict(trajectory.provenance),
        "wall_time_s": float(solve_wall),
        "atlas": atlas_report,
        **gate,
    }
    atomic_write_json(output / "centerline_report.json", report)
    return trajectory, local_atlas, report


def _run_tube(
    *,
    teacher: CanonicalTeacher,
    environment: Any,
    config: dict[str, Any],
    variant: TeacherVariant,
    challenge: EllipseChallenge,
    centerline: Any,
    local_atlas: Any | None,
    output: Path,
) -> dict[str, Any]:
    offsets = tuple(float(value) for value in config["formal_upgrade"]["tube_offsets_mm"])
    policy = _teacher_policy(config, variant, int(config["solver_seed"]))
    targets = np.asarray(centerline.target_xyz_m, dtype=float)
    n1, n2 = normal_frame(targets)
    frames: list[pd.DataFrame] = []
    residual_groups: list[np.ndarray] = []
    local_groups: list[np.ndarray] = []
    successes = 0
    total = 0
    for offset1 in offsets:
        for offset2 in offsets:
            shifted = targets + (offset1 * n1 + offset2 * n2) / 1000.0
            if offset1 == 0.0 and offset2 == 0.0:
                tube_trajectory = centerline
            else:
                spec = TrajectorySpec(
                    trajectory_id=(
                        f"large-scale@a{challenge.major_semiaxis_m:g}m"
                        f":n1{offset1:+g}:n2{offset2:+g}"
                    ),
                    family_id=f"optimized_a{challenge.major_semiaxis_m:g}m",
                    radius_mm=challenge.radius_parameter_m * 1000.0,
                    target_xyz_m=shifted,
                    tube_offsets_mm=offsets,
                )
                predicted = (
                    local_atlas.predict_path(shifted)
                    if local_atlas is not None
                    else centerline.beta_rad
                )
                tube_trajectory = teacher.solve(
                    spec,
                    policy,
                    root_beta=centerline.beta_rad[0],
                    initial_beta_path=predicted,
                )
            frames.append(
                trajectory_frame(
                    tube_trajectory,
                    environment,
                    tube_n1_mm=offset1,
                    tube_n2_mm=offset2,
                )
            )
            residual = np.linalg.norm(
                tube_trajectory.achieved_xyz_m - shifted, axis=1
            ) * 1000.0
            local = np.rad2deg(
                np.sqrt(
                    np.mean(
                        np.square(tube_trajectory.beta_rad - centerline.beta_rad),
                        axis=1,
                    )
                )
            )
            residual_groups.append(residual)
            local_groups.append(local)
            successes += int(np.count_nonzero(residual <= 3.0))
            total += len(residual)
    tube_frame = pd.concat(frames, ignore_index=True)
    tube_frame.to_parquet(output / "tube.parquet", index=False)
    all_residual = np.concatenate(residual_groups)
    all_local = np.concatenate(local_groups)
    metrics = {
        "success_rate": float(successes / total),
        "residual_p95_mm": float(np.percentile(all_residual, 95)),
        "residual_max_mm": float(np.max(all_residual)),
        "local_beta_rms_p95_deg": float(np.percentile(all_local, 95)),
        "multi_branch_ratio": measured_multi_branch_ratio(tube_frame),
    }
    report = {
        "variant": variant.value,
        "metrics": metrics,
        **evaluate_tube_gate(metrics, thresholds=config["gates"]["tube"]),
    }
    atomic_write_json(output / "tube_report.json", report)
    return report


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
    if not np.isclose(
        float(config["legacy_major_axis_gain"]),
        REGISTERED_MAJOR_AXIS_GAIN,
        atol=1.0e-12,
    ):
        raise ValueError("protocol legacy_major_axis_gain differs from frozen geometry lineage")
    if not np.isclose(
        float(config["minor_to_major_ratio"]),
        REGISTERED_MINOR_TO_MAJOR_RATIO,
        atol=1.0e-12,
    ):
        raise ValueError("protocol minor_to_major_ratio differs from frozen geometry lineage")
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
    environment = load_environment(project_root, Path(args.robot_config))
    derived_robot_max_reach_m = float(
        np.sum(environment.lengths_m[:30])
        + np.linalg.norm(environment.p_end_local_m[:3])
    )
    if not np.isclose(
        derived_robot_max_reach_m,
        float(config["robot_max_reach_m"]),
        atol=1.0e-9,
    ):
        raise ValueError("protocol robot_max_reach_m differs from the loaded robot geometry")
    formal = config["formal_upgrade"]
    variants = tuple(
        dict.fromkeys(
            [str(value) for value in config["teacher_screen"]["variants"]]
            + [str(value) for value in formal["variants"]]
        )
    )
    manifest = ExperimentManifest.create(
        protocol_id=str(config["protocol_id"]),
        teacher_variants=variants,
        family_id="optimized_large_scale_pose_v10",
        radii_mm=tuple(challenge.radius_parameter_m * 1000.0 for challenge in challenges),
        phase_count=int(formal["phase_count"]),
        tube_offsets_mm=tuple(float(value) for value in formal["tube_offsets_mm"]),
        solver_seed=int(config["solver_seed"]),
        input_files=(config_path, Path(args.robot_config).resolve()),
        worker_code_files=(
            worker,
            source_root / "src/quasi_exp/teacher/large_scale.py",
            source_root / "src/quasi_exp/teacher/canonical.py",
            source_root / "src/quasi_exp/teacher/forward.py",
            source_root / "src/quasi_exp/teacher/dataset.py",
            source_root / "src/quasi_exp/teacher/atlas.py",
            source_root / "scripts/analysis/run_trajectory_canonical_teacher_v10.py",
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
        robot_max_reach_m=derived_robot_max_reach_m,
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
                beta_bounds_rad=environment.bounds,
                safe_joint_margin_deg=float(pose_config["safe_joint_margin_deg"]),
                joint_margin_penalty_mm_per_deg=float(
                    pose_config["joint_margin_penalty_mm_per_deg"]
                ),
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
            / (float(config["baseline_legacy_R_m"]) * REGISTERED_MAJOR_AXIS_GAIN),
            "normalized_legacy_radius_parameter": challenge.radius_parameter_m
            / derived_robot_max_reach_m,
            "normalized_major_semiaxis": challenge.major_semiaxis_m
            / derived_robot_max_reach_m,
            "normalized_major_diameter": challenge.major_diameter_m
            / derived_robot_max_reach_m,
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
                derived_robot_max_reach_m
                - np.max(np.linalg.norm(fit.target_xyz_m, axis=1))
            ),
            "robot_scale_design_lower_bounds": {
                "current_robot_max_reach_m": derived_robot_max_reach_m,
                "minimum_reach_from_translation_independent_diameter_m": (
                    challenge.major_semiaxis_m + float(config["tube_radius_m"])
                ),
                "minimum_reach_for_fitted_center_and_tube_m": float(
                    np.max(np.linalg.norm(fit.target_xyz_m, axis=1))
                    + float(config["tube_radius_m"])
                ),
                "uniform_length_scale_lower_bound_for_fitted_center": float(
                    (
                        np.max(np.linalg.norm(fit.target_xyz_m, axis=1))
                        + float(config["tube_radius_m"])
                    )
                    / derived_robot_max_reach_m
                ),
                "semantics": (
                    "necessary geometry lower bounds; segment allocation and expanded joint-domain "
                    "design remain conditional on exact teacher failure points"
                ),
            },
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

        screen_reports: dict[str, Any] = {}
        for variant_name in config["teacher_screen"]["variants"]:
            variant = TeacherVariant(str(variant_name))
            _trajectory, _local_atlas, variant_report = _solve_centerline(
                teacher=teacher,
                environment=environment,
                config=config,
                variant=variant,
                challenge=challenge,
                targets=fit.target_xyz_m,
                initial_beta_path=fit.match.initial_beta_path_rad,
                output=challenge_dir / "screen" / variant.value,
            )
            screen_reports[variant.value] = variant_report

        t4_screen_hard_pass = is_promising_centerline_screen(
            screen_reports["T4"]["metrics"],
            solver_success=bool(screen_reports["T4"]["checks"]["solver_success"]),
            residual_p95_limit_mm=float(
                config["gates"]["centerline"]["residual_p95_mm"]
            ),
            residual_max_limit_mm=float(
                config["gates"]["centerline"]["residual_max_mm"]
            ),
        )
        formal_reports: dict[str, Any] = {}
        if t4_screen_hard_pass:
            formal_targets = challenge.generate_targets(
                center_m=fit.center_m,
                major_direction=fit.major_direction,
                minor_direction=fit.minor_direction,
                phase_count=int(formal["phase_count"]),
            )
            formal_match = atlas.match_targets(
                formal_targets, beta_bounds_rad=environment.bounds
            )
            for variant_name in formal["variants"]:
                variant = TeacherVariant(str(variant_name))
                variant_dir = challenge_dir / "formal" / variant.value
                centerline, local_atlas, centerline_report = _solve_centerline(
                    teacher=teacher,
                    environment=environment,
                    config=config,
                    variant=variant,
                    challenge=challenge,
                    targets=formal_targets,
                    initial_beta_path=formal_match.initial_beta_path_rad,
                    output=variant_dir,
                )
                if bool(centerline_report["centerline_gate_pass"]):
                    tube_report = _run_tube(
                        teacher=teacher,
                        environment=environment,
                        config=config,
                        variant=variant,
                        challenge=challenge,
                        centerline=centerline,
                        local_atlas=local_atlas,
                        output=variant_dir,
                    )
                    status = "completed"
                else:
                    tube_report = None
                    status = "skipped_centerline_gate_failed"
                formal_reports[variant.value] = {
                    "centerline": centerline_report,
                    "tube": tube_report,
                    "tube_status": status,
                    "full_gate_pass": bool(
                        centerline_report["centerline_gate_pass"]
                        and tube_report is not None
                        and tube_report["tube_gate_pass"]
                    ),
                }

        baseline_pass = bool(
            formal_reports.get("T3", {}).get("full_gate_pass", False)
        )
        t4_pass = bool(formal_reports.get("T4", {}).get("full_gate_pass", False))
        if t4_pass and not baseline_pass:
            attribution = "T4_atlas_enabled_full_gate_scale_gain"
        elif t4_pass and baseline_pass:
            attribution = "geometry_enabled_not_T4_atlas_specific"
        elif not t4_pass and not baseline_pass:
            attribution = "large_scale_full_gate_failed"
        else:
            attribution = "generalized_V7_structure_baseline_only_unexpected"
        report = {
            "challenge": pose_report,
            "baseline_semantics": {
                "exact_T0_V7_replay": "not_applicable_to_new_optimized_geometry",
                "attribution_comparator": "T3_graph_plus_whole_trajectory_correction",
                "T1_role": "Jacobian_continuation_mechanism_ablation_only",
            },
            "screen_teachers": screen_reports,
            "screen_hard_gate_pass": t4_screen_hard_pass,
            "formal_teachers": formal_reports,
            "requested_scale_meets_qualitative_threshold": bool(
                challenge.major_semiaxis_m >= float(config["qualitative_scale_threshold_m"])
            ),
            "qualitative_method_gate_pass": bool(t4_pass and not baseline_pass),
            "large_scale_full_gate_pass": t4_pass,
            "method_attribution": attribution,
            "formal_upgrade_status": (
                "completed" if t4_screen_hard_pass else "skipped_screen_hard_gate_failed"
            ),
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
