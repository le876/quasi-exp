#!/usr/bin/env python3
"""Process-isolated retry15 candidate shard worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import yaml
from scipy.spatial import cKDTree

from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.retry15_candidate_solver import (
    Retry15CandidatePolicy,
    solve_target_candidates,
)


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    return resolved.parent.parent if resolved.parent.name == ".worktrees" else resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--seed-bank", required=True)
    parser.add_argument("--y-seed-bank", required=True)
    parser.add_argument("--z-seed-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-budget", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    environment = optimized_forward(
        load_environment(
            project_root_from(SOURCE_ROOT),
            SOURCE_ROOT / str(config["sources"]["robot_config"]),
        )
    )
    targets = pd.read_parquet(args.targets)
    seed_banks = {
        "full": pd.read_parquet(args.seed_bank),
        "y_seam": pd.read_parquet(args.y_seed_bank),
        "z_seam": pd.read_parquet(args.z_seed_bank),
    }
    seed_trees = {
        key: cKDTree(frame.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(float))
        for key, frame in seed_banks.items()
    }
    solver = config["candidate_solver"]
    policy = Retry15CandidatePolicy(
        seed_budget=int(args.seed_budget),
        nearest_seed_count=int(solver["nearest_seed_count"]),
        beta_weights=tuple(map(float, solver["beta_weights"])),
        damping=float(solver["damping"]),
        maximum_dls_iterations=int(solver["maximum_dls_iterations"]),
        maximum_least_squares_evaluations=int(
            solver["maximum_least_squares_evaluations"]
        ),
        maximum_step_rms_deg=float(solver["maximum_step_rms_deg"]),
        residual_maximum_mm=float(solver["residual_maximum_mm"]),
        difficult_slsqp_seed_count=int(solver["difficult_slsqp_seed_count"]),
    )
    frames = []
    for row in targets.to_dict("records"):
        y_zero = abs(float(row["y_m"])) <= 1.0e-12
        z_zero = abs(float(row["z_m"])) <= 1.0e-12
        bank_key = "y_seam" if y_zero and not z_zero else "z_seam" if z_zero and not y_zero else "full"
        frames.append(
            solve_target_candidates(
                environment,
                row,
                seed_banks[bank_key],
                policy=policy,
                seed_tree=seed_trees[bank_key],
            )
        )
    candidates = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    candidates.to_parquet(temporary, index=False)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "target_count": len(targets),
                "candidate_attempt_count": len(candidates),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
