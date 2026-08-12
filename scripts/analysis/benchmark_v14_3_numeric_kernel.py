#!/usr/bin/env python3
"""Benchmark the opt-in V14.3 endpoint FK and analytic Jacobian kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quasi_exp.teacher.optimized_forward import optimized_forward
from run_trajectory_canonical_teacher_v10 import load_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--jacobian-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rows < 1 or args.jacobian_rows < 1 or args.jacobian_rows > args.rows:
        raise SystemExit("require rows >= jacobian-rows >= 1")
    project_root = Path(str(SOURCE_ROOT).split("/.worktrees/", 1)[0])
    reference = load_environment(
        project_root, SOURCE_ROOT / "configs/robot_rods_only_standard_100k.yaml"
    )
    optimized = optimized_forward(reference)
    rng = np.random.default_rng(args.seed)
    beta = rng.uniform(reference.bounds[:, 0], reference.bounds[:, 1], (args.rows, 6))

    started = time.perf_counter()
    reference_xyz = reference.fk(beta)
    reference_fk_s = time.perf_counter() - started
    started = time.perf_counter()
    optimized_xyz = optimized.fk(beta)
    optimized_fk_s = time.perf_counter() - started

    subset = beta[: args.jacobian_rows]
    started = time.perf_counter()
    numerical = np.stack([reference.numerical_jacobian(row) for row in subset])
    numerical_jacobian_s = time.perf_counter() - started
    started = time.perf_counter()
    _xyz, analytic = optimized.fk_and_jacobian(subset)
    analytic_jacobian_s = time.perf_counter() - started

    report = {
        "schema_version": 1,
        "rows": args.rows,
        "jacobian_rows": args.jacobian_rows,
        "seed": args.seed,
        "reference_fk_s": reference_fk_s,
        "optimized_fk_s": optimized_fk_s,
        "fk_speedup": reference_fk_s / optimized_fk_s,
        "fk_max_abs_m": float(np.max(np.abs(reference_xyz - optimized_xyz))),
        "numerical_jacobian_s": numerical_jacobian_s,
        "analytic_fk_jacobian_s": analytic_jacobian_s,
        "jacobian_speedup": numerical_jacobian_s / analytic_jacobian_s,
        "jacobian_max_abs_m_per_rad": float(np.max(np.abs(numerical - analytic))),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
