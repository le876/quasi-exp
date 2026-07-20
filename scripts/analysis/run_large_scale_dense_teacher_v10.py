#!/usr/bin/env python3
"""Materialise the frozen 720-phase T3 labels for large-scale students.

The registered 0.5 m and 0.75 m ellipse poses are inputs: this runner never
re-searches a pose or changes the registered joint domain.  Existing sparse
T3 solutions are used only as periodic initialisation for the exact teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quasi_exp.teacher.canonical import CanonicalTeacher, TeacherVariant, TrajectorySpec
from quasi_exp.teacher.dataset import trajectory_frame
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.large_scale import EllipseChallenge
from quasi_exp.teacher.student import (
    assign_dense_phase_split,
    materialize_phase_splits,
    periodic_beta_interpolation,
)

from run_large_scale_ellipse_challenge_v10 import _teacher_policy
from run_trajectory_canonical_teacher_v10 import load_environment, project_root_from, runtime_fingerprint


PHASE_COUNT = 720


def _source_path(challenge_dir: Path, major_semiaxis_m: float) -> Path:
    candidates = (
        challenge_dir / "formal/T3/centerline.parquet",
        challenge_dir / "screen/T3/centerline.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no registered T3 initialisation for a={major_semiaxis_m:g} m")


def _load_pose(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        pose = json.load(handle)
    required = {"center_m", "major_direction", "minor_direction", "major_semiaxis_m"}
    missing = required - set(pose)
    if missing:
        raise ValueError(f"pose report is missing keys: {sorted(missing)}")
    return pose


def _teacher_report_gate_pass(report: dict[str, Any], major_semiaxis_m: float) -> bool:
    expected_eligible = PHASE_COUNT if np.isclose(major_semiaxis_m, 0.5) else 180
    dense_path = Path(str(report.get("dense_teacher_path", "")))
    expected_hash = str(report.get("dense_teacher_sha256", ""))
    return bool(
        int(report.get("phase_count", -1)) == PHASE_COUNT
        and int(report.get("eligible_count", -1)) >= expected_eligible
        and dense_path.is_file()
        and bool(expected_hash)
        and sha256_file(dense_path) == expected_hash
    )


def run_one(
    *,
    major_semiaxis_m: float,
    project_root: Path,
    output_root: Path,
    environment: Any,
    config: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    slug = f"a{major_semiaxis_m:0.3f}m".replace(".", "p")
    challenge_dir = (
        project_root
        / "runs/trajectory_canonical_teacher_v10/07_large_scale_challenge"
        / slug
    )
    output = output_root / slug / "teacher_dense"
    completed = output / "teacher_report.json"
    if completed.exists() and not force:
        with completed.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        gate_pass = _teacher_report_gate_pass(cached, major_semiaxis_m)
        if not gate_pass:
            raise RuntimeError(f"cached dense teacher report failed its gate: {completed}")
        if cached.get("dense_teacher_gate_pass") is not True:
            cached["dense_teacher_gate_pass"] = True
            atomic_write_json(completed, cached)
        return cached

    pose_path = challenge_dir / "pose_report.json"
    source_path = _source_path(challenge_dir, major_semiaxis_m)
    pose = _load_pose(pose_path)
    registered_major = float(pose["major_semiaxis_m"])
    if not np.isclose(registered_major, major_semiaxis_m, rtol=0.0, atol=1.0e-12):
        raise ValueError("registered pose major semiaxis does not match requested challenge")
    challenge = EllipseChallenge.from_major_semiaxis_m(
        major_semiaxis_m,
        minor_to_major_ratio=float(config["minor_to_major_ratio"]),
    )
    targets = challenge.generate_targets(
        center_m=np.asarray(pose["center_m"], dtype=float),
        major_direction=np.asarray(pose["major_direction"], dtype=float),
        minor_direction=np.asarray(pose["minor_direction"], dtype=float),
        phase_count=PHASE_COUNT,
    )
    source = pd.read_parquet(source_path)
    initial = periodic_beta_interpolation(source, phase_count=PHASE_COUNT)
    policy = _teacher_policy(config, TeacherVariant.T3, int(config["solver_seed"]))
    spec = TrajectorySpec(
        trajectory_id=f"large-scale-dense@a{major_semiaxis_m:g}m",
        family_id=f"fixed-optimized-a{major_semiaxis_m:g}m",
        radius_mm=challenge.radius_parameter_m * 1000.0,
        target_xyz_m=targets,
    )
    started = time.perf_counter()
    trajectory = CanonicalTeacher(environment).solve(
        spec,
        policy,
        root_beta=initial[0],
        initial_beta_path=initial,
    )
    wall_time = time.perf_counter() - started
    frame = assign_dense_phase_split(
        trajectory_frame(trajectory, environment),
        expected_phase_count=PHASE_COUNT,
        residual_limit_mm=3.0,
    )
    output.mkdir(parents=True, exist_ok=True)
    full_path = output / "dense_teacher.parquet"
    frame.to_parquet(full_path, index=False, compression="zstd")
    split_report = materialize_phase_splits(frame, output_dir=output / "splits")
    residual = frame["teacher_fk_residual_mm"].to_numpy(dtype=float)
    report: dict[str, Any] = {
        "protocol_id": "large-scale-student-tracking-v10.1",
        "teacher_variant": "T3",
        "major_semiaxis_m": float(major_semiaxis_m),
        "minor_semiaxis_m": float(challenge.minor_semiaxis_m),
        "phase_count": PHASE_COUNT,
        "pose_researched": False,
        "registered_joint_domain_changed": False,
        "teacher_success": bool(trajectory.success),
        "eligible_count": int(frame["label_eligible"].sum()),
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        "joint_margin_min_deg": float(frame["joint_margin_min_deg"].min()),
        "wall_time_s": float(wall_time),
        "trajectory_metrics": dict(trajectory.metrics),
        "pose_path": str(pose_path.resolve()),
        "pose_sha256": sha256_file(pose_path),
        "initialisation_path": str(source_path.resolve()),
        "initialisation_sha256": sha256_file(source_path),
        "dense_teacher_path": str(full_path.resolve()),
        "dense_teacher_sha256": sha256_file(full_path),
        "split_report": split_report,
    }
    report["dense_teacher_gate_pass"] = _teacher_report_gate_pass(
        report, major_semiaxis_m
    )
    # The primary selection benchmark must have an exact eligible ring.  The
    # 0.75 m stress test deliberately retains invalid rows if any are found.
    if np.isclose(major_semiaxis_m, 0.5) and not report["dense_teacher_gate_pass"]:
        atomic_write_json(completed, report)
        raise RuntimeError("0.5 m dense T3 teacher did not produce 720 eligible labels")
    if not report["dense_teacher_gate_pass"]:
        atomic_write_json(completed, report)
        raise RuntimeError("dense T3 teacher produced fewer than 180 eligible labels")
    atomic_write_json(completed, report)
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    project_root = project_root_from(repo_root)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=repo_root / "configs/large_scale_ellipse_challenge_v10.yaml",
        type=Path,
    )
    parser.add_argument(
        "--robot-config",
        default=project_root / "configs/robot_rods_only_standard_100k.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        default=project_root
        / "runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking",
        type=Path,
    )
    parser.add_argument("--major", nargs="+", type=float, default=(0.5, 0.75))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    environment = load_environment(project_root, args.robot_config.resolve())
    reports = [
        run_one(
            major_semiaxis_m=float(major),
            project_root=project_root,
            output_root=args.output_root.resolve(),
            environment=environment,
            config=config,
            force=bool(args.force),
        )
        for major in args.major
    ]
    summary = {
        "protocol_id": "large-scale-student-tracking-v10.1",
        "runtime": runtime_fingerprint(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "robot_config_path": str(args.robot_config.resolve()),
        "robot_config_sha256": sha256_file(args.robot_config.resolve()),
        "reports": reports,
    }
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root.resolve() / "teacher_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
