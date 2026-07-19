#!/usr/bin/env python3
"""Run the frozen V10 canonical-teacher Pilot A protocol.

The runner is deliberately resumable at one (variant, tube offset) trajectory
per directory.  A completed parquet/report pair is never recomputed unless
``--force`` is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
from dataclasses import replace

import numpy as np
import pandas as pd

from quasi_exp.io.config import load_config
from quasi_exp.io.robot_inputs import load_robot_inputs
from quasi_exp.teacher.canonical import (
    CanonicalTeacher,
    TeacherPolicy,
    TeacherTrajectory,
    TeacherVariant,
    TrajectorySpec,
)
from quasi_exp.teacher.atlas import build_local_atlas
from quasi_exp.teacher.dataset import (
    evaluate_centerline_gate,
    evaluate_tube_gate,
    trajectory_frame,
    measured_multi_branch_ratio,
)
from quasi_exp.teacher.experiment import ExperimentManifest, atomic_write_json
from quasi_exp.teacher.forward import ForwardEnvironment


FAMILY_ID = "c0273_a100_py210_pz330_s0243"
BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
TARGET_COLUMNS = ("x_target_m", "y_target_m", "z_target_m")
STANDARD_BOUNDS_RAD = np.deg2rad(
    np.asarray([[-5.0, 5.0], [-5.0, 5.0], [-10.0, 10.0], [-10.0, 10.0], [-15.0, 15.0], [-15.0, 15.0]])
)


def project_root_from(path: Path) -> Path:
    resolved = path.resolve()
    if ".worktrees" in resolved.parts:
        index = resolved.parts.index(".worktrees")
        return Path(*resolved.parts[:index])
    return resolved


def radius_slug(radius_mm: float) -> str:
    return f"r{float(radius_mm):06.2f}".replace(".", "p")


def runtime_fingerprint() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "sklearn", "tensorflow"):
        try:
            module = __import__(name)
            packages[name] = str(module.__version__)
        except Exception as exc:  # optional TF during teacher-only CPU runs
            packages[name] = f"unavailable:{type(exc).__name__}"
    try:
        source_root = Path(__file__).resolve().parents[2]
        git_sha = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_sha": git_sha,
        "source_root": str(Path(__file__).resolve().parents[2]),
    }


def load_environment(project_root: Path, robot_config: Path) -> ForwardEnvironment:
    config = load_config(robot_config)
    # Robot input paths are historically project-relative.
    original = Path.cwd()
    try:
        import os

        os.chdir(project_root)
        inputs = load_robot_inputs(config)
    finally:
        os.chdir(original)
    return ForwardEnvironment(
        lengths_m=inputs.lengths_m,
        p_end_local_m=inputs.p_end_local_m,
        theta_sign=float(config.get("kinematics", {}).get("theta_sign", -1.0)),
        beta_bounds_rad=STANDARD_BOUNDS_RAD,
    )


def load_family(project_root: Path, family_id: str) -> pd.Series:
    path = project_root / "runs/true_ellipse_family_expansion_v5/02_pointwise/selected_families.csv"
    table = pd.read_csv(path)
    rows = table[table["candidate_id"].astype(str).eq(str(family_id))]
    if len(rows) != 1:
        raise ValueError(f"expected one registered family row for {family_id}")
    return rows.iloc[0]


def generate_targets(family: pd.Series, radius_mm: float, phase_count: int) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * math.pi, int(phase_count), endpoint=False)
    radius_m = float(radius_mm) / 1000.0
    return np.column_stack(
        [
            float(family["center_x_m"]) + radius_m * np.sin(phase),
            float(family["center_y_m"]) + radius_m * np.sin(phase + float(family["phase_y_rad"])),
            float(family["center_z_m"]) + 1.5 * radius_m * np.sin(phase + float(family["phase_z_rad"])),
        ]
    )


def legacy_path(project_root: Path, radius_mm: float, phase_count: int) -> tuple[np.ndarray, Path]:
    requested = (
        project_root
        / "runs/true_ellipse_standard_domain_v7/01_radial"
        / radius_slug(radius_mm)
        / "selected_centerline_360.parquet"
    )
    path = requested
    if not path.exists():
        candidates = list(
            (project_root / "runs/true_ellipse_standard_domain_v7/01_radial").glob(
                "r*/selected_centerline_360.parquet"
            )
        )
        if not candidates:
            raise FileNotFoundError(str(requested))
        def parsed_radius(candidate: Path) -> float:
            return float(candidate.parent.name[1:].replace("p", "."))
        path = min(candidates, key=lambda candidate: abs(parsed_radius(candidate) - float(radius_mm)))
    frame = pd.read_parquet(path).sort_values("angle_idx", kind="stable")
    if 360 % int(phase_count) != 0:
        raise ValueError("phase_count must evenly subsample the 360-point V7 baseline")
    frame = frame.iloc[:: 360 // int(phase_count)].reset_index(drop=True)
    return frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float), path


def normal_frame(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.roll(targets, -1, axis=0) - np.roll(targets, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    reference = np.tile(np.asarray([0.0, 0.0, 1.0]), (len(targets), 1))
    parallel = np.abs(np.sum(reference * tangent, axis=1)) > 0.9
    reference[parallel] = np.asarray([0.0, 1.0, 0.0])
    n1 = np.cross(tangent, reference)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = np.cross(tangent, n1)
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
    return n1, n2


def policy_for(
    variant: TeacherVariant,
    seed: int,
    preset: str,
    registered: dict[str, Any] | None = None,
) -> TeacherPolicy:
    budget = 4 if preset == "smoke" else 16
    common = dict(
        variant=variant,
        candidate_budget=budget,
        solver_seed=int(seed),
        tracking_tolerance_mm=1.0,
        max_corrector_iterations=30 if preset == "smoke" else 100,
        safe_joint_margin_deg=1.5,
    )
    if registered is not None:
        values = registered[variant.value]
        return TeacherPolicy(**common, **{key: float(value) for key, value in values.items()})
    if variant == TeacherVariant.T1:
        return TeacherPolicy(**common, lambda_velocity=0.1, lambda_acceleration=0.0, lambda_posture=0.01, lambda_conditioning=0.0)
    if variant == TeacherVariant.T2:
        return TeacherPolicy(**common, lambda_velocity=1.0, lambda_acceleration=0.1, lambda_posture=0.05, lambda_conditioning=0.01)
    if variant == TeacherVariant.T3:
        return TeacherPolicy(**common, lambda_velocity=5.0, lambda_acceleration=0.5, lambda_posture=0.1, lambda_conditioning=0.05)
    if variant == TeacherVariant.T4:
        return TeacherPolicy(**common, lambda_velocity=10.0, lambda_acceleration=1.0, lambda_posture=0.2, lambda_conditioning=0.1)
    return TeacherPolicy(**common)


def legacy_trajectory(
    environment: ForwardEnvironment,
    targets: np.ndarray,
    beta: np.ndarray,
    *,
    radius_mm: float,
    seed: int,
    traversal_direction: str = "forward",
    cyclic_cut: int = 0,
) -> TeacherTrajectory:
    achieved = environment.fk(beta)
    residual = np.linalg.norm(achieved - targets, axis=1) * 1000.0
    velocity = np.roll(beta, -1, axis=0) - beta
    acceleration = np.roll(beta, -1, axis=0) - 2 * beta + np.roll(beta, 1, axis=0)
    velocity_deg = np.rad2deg(np.sqrt(np.mean(np.square(velocity), axis=1)))
    acceleration_deg = np.rad2deg(np.sqrt(np.mean(np.square(acceleration), axis=1)))
    margin = np.rad2deg(np.minimum(beta - STANDARD_BOUNDS_RAD[:, 0], STANDARD_BOUNDS_RAD[:, 1] - beta))
    metrics = {
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        "delta_beta_rms_p95_deg": float(np.percentile(velocity_deg, 95)),
        "delta_beta_rms_max_deg": float(np.max(velocity_deg)),
        "acceleration_beta_rms_p95_deg": float(np.percentile(acceleration_deg, 95)),
        "seam_beta_rms_deg": float(velocity_deg[-1]),
        "joint_margin_min_deg": float(np.min(margin)),
    }
    policy = policy_for(TeacherVariant.T0, seed, "pilot")
    return TeacherTrajectory(
        beta_rad=beta,
        theta_rad=environment.theta(beta),
        achieved_xyz_m=achieved,
        target_xyz_m=targets,
        chart_id=np.zeros(len(beta), dtype=np.int64),
        branch_id=np.zeros(len(beta), dtype=np.int64),
        metrics=metrics,
        provenance={
            "teacher_policy_id": policy.fingerprint,
            "teacher_variant": "T0",
            "trajectory_id": f"{FAMILY_ID}@r{radius_mm:g}:center",
            "family_id": FAMILY_ID,
            "radius_mm": radius_mm,
            "solver_seed": seed,
            "traversal_direction": traversal_direction,
            "cyclic_cut": int(cyclic_cut),
            "root_configuration_id": "v7-selected-centerline",
        },
        success=bool(np.max(residual) <= 3.0 and np.min(margin) >= -1e-8),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_config = load_config(args.protocol_config)
    if str(protocol_config["family_id"]) != str(args.family_id):
        raise ValueError("family-id differs from the frozen protocol config")
    family_path = project_root / "runs/true_ellipse_family_expansion_v5/02_pointwise/selected_families.csv"
    legacy_beta, legacy_source = legacy_path(project_root, args.radius_mm, args.phase_count)
    manifest = ExperimentManifest.create(
        protocol_id=str(protocol_config["protocol_id"]),
        teacher_variants=args.variants,
        family_id=args.family_id,
        radii_mm=(args.radius_mm,),
        phase_count=args.phase_count,
        tube_offsets_mm=args.tube_offsets_mm,
        solver_seed=args.solver_seed,
        traversal_direction=args.traversal_direction,
        cyclic_cut=args.cyclic_cut,
        input_files=(family_path, args.robot_config, args.protocol_config, legacy_source),
        worker_code_files=(
            Path(__file__),
            Path(__file__).resolve().parents[2] / "src/quasi_exp/teacher/canonical.py",
            Path(__file__).resolve().parents[2] / "src/quasi_exp/teacher/forward.py",
            Path(__file__).resolve().parents[2] / "src/quasi_exp/teacher/dataset.py",
        ),
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest.as_dict():
            raise RuntimeError("output contains a different protocol manifest; choose a new output or use --force")
    atomic_write_json(manifest_path, manifest.as_dict())
    atomic_write_json(output / "runtime.json", runtime_fingerprint())
    environment = load_environment(project_root, Path(args.robot_config))
    family = load_family(project_root, args.family_id)
    targets = generate_targets(family, args.radius_mm, args.phase_count)
    n1, n2 = normal_frame(targets)
    teacher = CanonicalTeacher(environment)
    rows: list[dict[str, Any]] = []

    for variant_name in args.variants:
        variant = TeacherVariant(variant_name)
        policy = policy_for(variant, args.solver_seed, args.preset, protocol_config["teacher_policies"])
        variant_dir = output / variant.value
        variant_dir.mkdir(parents=True, exist_ok=True)
        center_path = variant_dir / "centerline.parquet"
        report_path = variant_dir / "centerline_report.json"
        reusable = (
            center_path.exists()
            and report_path.exists()
            and (args.preset == "smoke" or (variant_dir / "tube_report.json").exists())
        )
        if reusable and not args.force:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(report)
            continue
        if not args.force and (center_path.exists() or report_path.exists()):
            raise RuntimeError(
                f"incomplete cached variant {variant.value}; preserve it and resume into a new output, or explicitly use --force"
            )
        if variant == TeacherVariant.T0:
            trajectory = legacy_trajectory(
                environment,
                targets,
                legacy_beta,
                radius_mm=args.radius_mm,
                seed=args.solver_seed,
                traversal_direction=args.traversal_direction,
                cyclic_cut=args.cyclic_cut,
            )
        else:
            spec = TrajectorySpec(
                trajectory_id=f"{args.family_id}@r{args.radius_mm:g}:center",
                family_id=args.family_id,
                radius_mm=args.radius_mm,
                target_xyz_m=targets,
                traversal_direction=args.traversal_direction,
                cyclic_cut=args.cyclic_cut,
            )
            solve_started = time.perf_counter()
            trajectory = teacher.solve(
                spec,
                policy,
                root_beta=legacy_beta[0],
                initial_beta_path=legacy_beta,
            )
            trajectory.provenance["wall_time_s"] = time.perf_counter() - solve_started
        atlas = None
        if variant == TeacherVariant.T4:
            atlas = build_local_atlas(trajectory, environment, policy)
            trajectory = replace(trajectory, chart_id=atlas.phase_chart_ids)
            trajectory.metrics["chart_overlap_gap_p95_deg"] = atlas.overlap_gap_p95_deg
            trajectory.metrics["chart_overlap_gate_pass"] = float(atlas.overlap_gate_pass)
        trajectory_frame(trajectory, environment).to_parquet(center_path, index=False)
        gate = evaluate_centerline_gate(trajectory)
        report = {
            "variant": variant.value,
            "metrics": dict(trajectory.metrics),
            "provenance": dict(trajectory.provenance),
            **gate,
        }
        atomic_write_json(report_path, report)
        rows.append(report)

        # Smoke validates the executable centerline chain. Pilot expands to
        # the registered 3x3 normal tube and writes one resumable curve/job.
        if args.preset == "pilot":
            tube_frames: list[pd.DataFrame] = []
            tube_residuals: list[np.ndarray] = []
            tube_local_beta: list[np.ndarray] = []
            successes = 0
            total = 0
            for offset1 in args.tube_offsets_mm:
                for offset2 in args.tube_offsets_mm:
                    if offset1 == 0.0 and offset2 == 0.0:
                        tube_frames.append(trajectory_frame(trajectory, environment))
                        tube_residuals.append(
                            np.linalg.norm(trajectory.achieved_xyz_m - trajectory.target_xyz_m, axis=1) * 1000.0
                        )
                        tube_local_beta.append(np.zeros(len(targets)))
                        successes += len(targets)
                        total += len(targets)
                        continue
                    shifted = targets + (offset1 * n1 + offset2 * n2) / 1000.0
                    tube_spec = TrajectorySpec(
                        trajectory_id=f"{args.family_id}@r{args.radius_mm:g}:n1{offset1:+g}:n2{offset2:+g}",
                        family_id=args.family_id,
                        radius_mm=args.radius_mm,
                        target_xyz_m=shifted,
                        tube_offsets_mm=tuple(args.tube_offsets_mm),
                        traversal_direction=args.traversal_direction,
                        cyclic_cut=args.cyclic_cut,
                    )
                    tube_traj = teacher.solve(
                        tube_spec,
                        policy,
                        root_beta=trajectory.beta_rad[0],
                        initial_beta_path=(
                            atlas.predict_path(shifted)
                            if atlas is not None
                            else trajectory.beta_rad
                        ),
                    )
                    tube_frames.append(
                        trajectory_frame(tube_traj, environment, tube_n1_mm=offset1, tube_n2_mm=offset2)
                    )
                    residuals = np.linalg.norm(tube_traj.achieved_xyz_m - shifted, axis=1) * 1000.0
                    local = np.rad2deg(
                        np.sqrt(np.mean(np.square(tube_traj.beta_rad - trajectory.beta_rad), axis=1))
                    )
                    tube_residuals.append(residuals)
                    tube_local_beta.append(local)
                    successes += int(np.count_nonzero(residuals <= 3.0))
                    total += len(residuals)
            tube_frame = pd.concat(tube_frames, ignore_index=True)
            tube_frame.to_parquet(variant_dir / "tube.parquet", index=False)
            all_residual = np.concatenate(tube_residuals)
            all_local = np.concatenate(tube_local_beta)
            tube_metrics = {
                "success_rate": successes / total,
                "residual_p95_mm": float(np.percentile(all_residual, 95)),
                "residual_max_mm": float(np.max(all_residual)),
                "local_beta_rms_p95_deg": float(np.percentile(all_local, 95)),
                "multi_branch_ratio": measured_multi_branch_ratio(tube_frame),
            }
            atomic_write_json(
                variant_dir / "tube_report.json",
                {"variant": variant.value, "metrics": tube_metrics, **evaluate_tube_gate(tube_metrics)},
            )

    summary = {
        "protocol_sha256": manifest.protocol_sha256,
        "preset": args.preset,
        "wall_time_s": time.time() - started,
        "variants": rows,
        "completed": True,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_project = project_root_from(Path(__file__).resolve().parents[2])
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--robot-config", type=Path, default=default_project / "configs/robot_rods_only_standard_100k.yaml")
    parser.add_argument("--protocol-config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/trajectory_canonical_teacher_v10.yaml")
    parser.add_argument("--output", type=Path, default=default_project / "runs/trajectory_canonical_teacher_v10/pilot_a")
    parser.add_argument("--family-id", default=FAMILY_ID)
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument("--preset", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--phase-count", type=int)
    parser.add_argument("--tube-offsets-mm", type=float, nargs="+")
    parser.add_argument("--variants", nargs="+", choices=tuple(item.value for item in TeacherVariant))
    parser.add_argument("--solver-seed", type=int, default=20260720)
    parser.add_argument("--traversal-direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--cyclic-cut", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.phase_count is None:
        args.phase_count = 24 if args.preset == "smoke" else 180
    if args.tube_offsets_mm is None:
        args.tube_offsets_mm = (0.0,) if args.preset == "smoke" else (-1.0, 0.0, 1.0)
    else:
        args.tube_offsets_mm = tuple(args.tube_offsets_mm)
    if args.variants is None:
        args.variants = ("T0", "T1", "T2", "T3", "T4")
    else:
        args.variants = tuple(args.variants)
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
