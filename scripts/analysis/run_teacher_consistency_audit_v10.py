#!/usr/bin/env python3
"""Pilot B: repeatability, direction, cyclic-cut and label consistency audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from quasi_exp.teacher.audit import (
    audit_label_consistency,
    compare_aligned_trajectories,
    recommend_student_representation,
)
from quasi_exp.teacher.experiment import atomic_write_json


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    reports = {}
    combined = []
    for variant in args.variants:
        jobs = [
            ("repeat_a", "forward", 0),
            ("repeat_b", "forward", 0),
            ("reverse", "reverse", 0),
            ("cut90", "forward", 45),
            ("cut180", "forward", 90),
            ("cut270", "forward", 135),
        ]
        frames = {}
        for name, direction, cut in jobs:
            out = root / variant / name
            if not (out / "summary.json").exists():
                subprocess.run(
                    [
                        args.python, str(args.teacher_runner), "--preset", "smoke",
                        "--phase-count", "180", "--variants", variant,
                        "--radius-mm", str(args.radius_mm), "--project-root", str(args.project_root),
                        "--protocol-config", str(args.protocol_config), "--output", str(out),
                        "--traversal-direction", direction, "--cyclic-cut", str(cut),
                    ],
                    check=True,
                )
            frame = pd.read_parquet(out / variant / "centerline.parquet")
            frames[name] = frame
            labelled = frame.copy()
            labelled["trajectory_id"] = f"{variant}:{name}"
            combined.append(labelled)
        reports[variant] = {
            "repeatability": compare_aligned_trajectories(frames["repeat_a"], frames["repeat_b"]),
            "direction": compare_aligned_trajectories(frames["repeat_a"], frames["reverse"]),
            "cuts": {
                name: compare_aligned_trajectories(frames["repeat_a"], frames[name])
                for name in ("cut90", "cut180", "cut270")
            },
        }
    consistency = audit_label_consistency(pd.concat(combined, ignore_index=True))
    report = {
        "protocol_id": "trajectory-teacher-consistency-v10.2",
        "teacher_audits": reports,
        "cross_trajectory_consistency": consistency,
        "recommended_student_representation": recommend_student_representation(consistency),
    }
    atomic_write_json(root / "audit_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = Path(__file__).resolve().parents[2]
    parser.add_argument("--python", default=__import__("sys").executable)
    parser.add_argument("--teacher-runner", type=Path, default=source / "scripts/analysis/run_trajectory_canonical_teacher_v10.py")
    parser.add_argument("--protocol-config", type=Path, default=source / "configs/trajectory_canonical_teacher_v10.yaml")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs=2, required=True)
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parsed = parser.parse_args()
    print(json.dumps(run(parsed), indent=2, allow_nan=False))
