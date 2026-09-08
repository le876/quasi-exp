#!/usr/bin/env python3
"""Locate centerline and tube radius frontiers on the frozen 0.5 mm grid."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import yaml

from quasi_exp.teacher.experiment import atomic_write_json
from quasi_exp.teacher.protocol import locate_radius_frontier


def run(args: argparse.Namespace) -> dict[str, object]:
    config = yaml.safe_load(Path(args.protocol_config).read_text(encoding="utf-8"))
    anchors = tuple(float(value) for value in config["frontier"]["coarse_anchors_mm"])
    resolution = float(config["frontier"]["resolution_mm"])
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    def ensure(radius_mm: float) -> dict[str, object]:
        radius_dir = root / f"r{radius_mm:06.2f}".replace(".", "p")
        summary = radius_dir / "summary.json"
        if not summary.exists():
            subprocess.run(
                [
                    args.python, str(args.teacher_runner), "--preset", "pilot",
                    "--variants", args.variant, "--radius-mm", str(radius_mm),
                    "--phase-count", "180", "--project-root", str(args.project_root),
                    "--protocol-config", str(args.protocol_config), "--output", str(radius_dir),
                ],
                check=True,
            )
        center = json.loads((radius_dir / args.variant / "centerline_report.json").read_text())
        tube = json.loads((radius_dir / args.variant / "tube_report.json").read_text())
        return {"center": center, "tube": tube}

    center = locate_radius_frontier(
        lambda radius: bool(ensure(radius)["center"]["centerline_gate_pass"]),
        coarse_anchors_mm=anchors, resolution_mm=resolution,
    )
    tube = locate_radius_frontier(
        lambda radius: bool(ensure(radius)["tube"]["tube_gate_pass"]),
        coarse_anchors_mm=anchors, resolution_mm=resolution,
    )
    report = {
        "protocol_id": "trajectory-canonical-teacher-dual-frontier-v10.2",
        "variant": args.variant,
        "centerline_frontier": asdict(center),
        "tube_frontier": asdict(tube),
    }
    atomic_write_json(root / "frontier_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--python", default=str(Path(__import__('sys').executable)))
    parser.add_argument("--teacher-runner", type=Path, default=source_root / "scripts/analysis/run_trajectory_canonical_teacher_v10.py")
    parser.add_argument("--protocol-config", type=Path, default=source_root / "configs/trajectory_canonical_teacher_v10.yaml")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("T0", "T1", "T2", "T3", "T4"), required=True)
    parsed = parser.parse_args()
    print(json.dumps(run(parsed), indent=2, allow_nan=False))
