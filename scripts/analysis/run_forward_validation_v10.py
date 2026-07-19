#!/usr/bin/env python3
"""Validate the authoritative FK/Jacobian and target-only IK recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from quasi_exp.teacher.canonical import TeacherPolicy, TeacherVariant, _correct_target
from quasi_exp.teacher.experiment import atomic_write_json

from run_trajectory_canonical_teacher_v10 import (
    STANDARD_BOUNDS_RAD,
    load_environment,
    project_root_from,
    runtime_fingerprint,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = load_environment(Path(args.project_root), Path(args.robot_config))
    rng = np.random.default_rng(int(args.seed))
    margin = np.deg2rad(float(args.sampling_margin_deg))
    low = STANDARD_BOUNDS_RAD[:, 0] + margin
    high = STANDARD_BOUNDS_RAD[:, 1] - margin
    source = rng.uniform(low, high, size=(int(args.sample_count), 6))

    jacobian_started = time.perf_counter()
    jacobian = environment.validate_jacobian(
        source,
        direction_seed=int(args.seed),
        relative_error_p95_limit=0.01,
        absolute_error_p95_m_limit=5.0e-5,
    )
    jacobian_wall = time.perf_counter() - jacobian_started

    policy = TeacherPolicy(
        variant=TeacherVariant.T1,
        candidate_budget=1,
        damping=1.0e-3,
        max_corrector_iterations=int(args.max_iterations),
        tracking_tolerance_mm=0.5,
        solver_seed=int(args.seed),
    )
    targets = environment.fk(source)
    recovered = np.empty_like(source)
    solver_success = np.zeros(len(source), dtype=bool)
    iterations = np.zeros(len(source), dtype=np.int64)
    recovery_started = time.perf_counter()
    # The root is fixed and does not expose the source beta to the inverse
    # solver; the source beta is used only to generate a known-reachable x.
    root_bank = [np.zeros(6, dtype=float)]
    root_bank.extend(
        rng.uniform(low, high, size=(max(int(args.root_candidate_count) - 1, 0), 6))
    )
    for index, target in enumerate(targets):
        best: tuple[float, np.ndarray, int, bool] | None = None
        for root in root_bank:
            beta, residual_mm, count, success = _correct_target(environment, target, root, policy)
            candidate = (residual_mm, beta, count, success)
            if best is None or residual_mm < best[0]:
                best = candidate
            if success:
                break
        assert best is not None
        recovered[index] = best[1]
        solver_success[index] = best[3]
        iterations[index] = best[2]
    recovery_wall = time.perf_counter() - recovery_started
    recovery = environment.evaluate_synthetic_recovery(
        source,
        recovered,
        solver_success=solver_success,
        residual_p95_limit_m=5.0e-4,
        residual_max_limit_m=3.0e-3,
        success_rate_limit=0.99,
    )
    raw = pd.DataFrame()
    for axis in range(6):
        raw[f"source_beta{axis + 1}_rad"] = source[:, axis]
        raw[f"recovered_beta{axis + 1}_rad"] = recovered[:, axis]
    raw["solver_success"] = solver_success
    raw["iterations"] = iterations
    raw["residual_mm"] = np.linalg.norm(environment.fk(recovered) - targets, axis=1) * 1000.0
    raw.to_parquet(output / "synthetic_recovery.parquet", index=False)
    report = {
        "protocol_id": "forward-environment-validation-v10.1",
        "sample_count": int(args.sample_count),
        "seed": int(args.seed),
        "source_beta_hidden_from_inverse_solver": True,
        "root_candidate_count": int(args.root_candidate_count),
        "jacobian": {
            "passed": jacobian.passed,
            "metrics": dict(jacobian.metrics),
            "per_scale": [dict(row) for row in jacobian.per_scale],
            "wall_time_s": jacobian_wall,
        },
        "synthetic_recovery": {
            "passed": recovery.passed,
            "metrics": dict(recovery.metrics),
            "wall_time_s": recovery_wall,
        },
        "forward_gate_pass": bool(jacobian.passed and recovery.passed),
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_project = project_root_from(Path(__file__).resolve().parents[2])
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--robot-config", type=Path, default=default_project / "configs/robot_rods_only_standard_100k.yaml")
    parser.add_argument("--output", type=Path, default=default_project / "runs/trajectory_canonical_teacher_v10/01_forward_validation")
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--sampling-margin-deg", type=float, default=1.5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--root-candidate-count", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))
