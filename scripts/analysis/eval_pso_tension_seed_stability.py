from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs
from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.opt.pso import solve_tensions_pso
from quasi_exp.opt.tension_canonical import canonicalize_tension


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if len(seeds) < 2:
        raise ValueError("--seeds must contain at least two comma-separated integers")
    return seeds


def _select_row_positions(n_rows: int, num_theta: int) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("dataset is empty")
    count = min(max(1, int(num_theta)), n_rows)
    return np.unique(np.linspace(0, n_rows - 1, num=count, dtype=int))


def _pairwise_mae(tensions: np.ndarray) -> np.ndarray:
    tensions = np.asarray(tensions, dtype=float)
    values: list[float] = []
    for i in range(tensions.shape[0]):
        for j in range(i + 1, tensions.shape[0]):
            values.append(float(np.mean(np.abs(tensions[i] - tensions[j]))))
    return np.asarray(values, dtype=float)


def evaluate_gate(
    *,
    median_pairwise_mae_n: float,
    p95_pairwise_mae_n: float,
    feasible_rate: float,
    median_threshold_n: float,
    p95_threshold_n: float,
) -> dict[str, Any]:
    gate_pass = bool(
        np.isfinite(median_pairwise_mae_n)
        and np.isfinite(p95_pairwise_mae_n)
        and float(feasible_rate) >= 1.0
        and float(median_pairwise_mae_n) < float(median_threshold_n)
        and float(p95_pairwise_mae_n) < float(p95_threshold_n)
    )
    return {
        "gate_pass": gate_pass,
        "median_pairwise_mae_n": float(median_pairwise_mae_n),
        "p95_pairwise_mae_n": float(p95_pairwise_mae_n),
        "feasible_rate": float(feasible_rate),
        "median_threshold_n": float(median_threshold_n),
        "p95_threshold_n": float(p95_threshold_n),
    }


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)

    df = pd.read_parquet(args.dataset)
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    if len(theta_cols) != 30:
        raise SystemExit(f"dataset must contain 30 theta_*_rad columns, got {len(theta_cols)}")

    seeds = _parse_seeds(args.seeds)
    row_positions = _select_row_positions(len(df), int(args.num_theta))
    feasible_rms = float(args.feasible_rms_rnorm)
    canonical_cfg = dict(cfg.get("canonical_tension", {}))
    canonical_enabled = bool(canonical_cfg.get("enabled", False)) and (not bool(args.disable_canonical))

    per_run_rows: list[dict[str, Any]] = []
    per_theta_rows: list[dict[str, Any]] = []
    all_pairwise: list[float] = []
    all_pso_pairwise: list[float] = []

    for pos in row_positions.tolist():
        sample_id = int(df.iloc[pos]["sample_id"]) if "sample_id" in df.columns else int(pos)
        theta = df.iloc[pos][theta_cols].to_numpy(dtype=float)
        cache = model.build_cache(theta)

        tensions = []
        pso_tensions = []
        rms_values = []
        pso_rms_values = []
        for seed in seeds:
            res = solve_tensions_pso(model, cache, cfg["pso"], rng_seed=int(seed))
            pso_rms_rnorm = float(np.sqrt(res.mean_rnorm2))
            T_final = np.asarray(res.T_base_12, dtype=float)
            c_success = False
            c_elapsed_s = 0.0
            c_nfev = 0
            c_method = "disabled"
            if canonical_enabled:
                c_res = canonicalize_tension(model, cache, [res.T_base_12], canonical_cfg)
                T_final = np.asarray(c_res.T_base_12, dtype=float)
                c_success = bool(c_res.success)
                c_elapsed_s = float(c_res.elapsed_s)
                c_nfev = int(c_res.nfev)
                c_method = str(c_res.method)
                rms_rnorm = float(c_res.rms_rnorm)
            else:
                rms_rnorm = pso_rms_rnorm
            pso_rms_values.append(pso_rms_rnorm)
            rms_values.append(rms_rnorm)
            pso_tensions.append(np.asarray(res.T_base_12, dtype=float))
            tensions.append(T_final)
            per_run_rows.append(
                {
                    "sample_id": sample_id,
                    "row_position": int(pos),
                    "seed": int(seed),
                    "pso_rms_rnorm": pso_rms_rnorm,
                    "rms_rnorm": rms_rnorm,
                    "feasible": bool(rms_rnorm <= feasible_rms),
                    "canonical_enabled": canonical_enabled,
                    "canonical_success": c_success,
                    "canonical_method": c_method,
                    "canonical_elapsed_s": c_elapsed_s,
                    "canonical_nfev": c_nfev,
                    "max_tension_n": float(res.max_tension),
                    "best_cost": float(res.best_cost),
                    "iters_used": int(res.iters_used),
                    "evals": int(res.evals),
                }
            )

        tension_arr = np.vstack(tensions)
        pso_tension_arr = np.vstack(pso_tensions)
        pairwise = _pairwise_mae(tension_arr)
        pso_pairwise = _pairwise_mae(pso_tension_arr)
        all_pairwise.extend(pairwise.tolist())
        all_pso_pairwise.extend(pso_pairwise.tolist())
        feasible_count = int(np.sum(np.asarray(rms_values) <= feasible_rms))
        per_theta_rows.append(
            {
                "sample_id": sample_id,
                "row_position": int(pos),
                "seed_count": len(seeds),
                "feasible_count": feasible_count,
                "feasible_rate": float(feasible_count / len(seeds)),
                "pso_rms_rnorm_min": float(np.min(pso_rms_values)),
                "pso_rms_rnorm_median": float(np.median(pso_rms_values)),
                "pso_rms_rnorm_max": float(np.max(pso_rms_values)),
                "rms_rnorm_min": float(np.min(rms_values)),
                "rms_rnorm_median": float(np.median(rms_values)),
                "rms_rnorm_max": float(np.max(rms_values)),
                "pso_pairwise_mae_median_n": float(np.median(pso_pairwise)),
                "pso_pairwise_mae_p95_n": float(np.quantile(pso_pairwise, 0.95)),
                "pairwise_mae_median_n": float(np.median(pairwise)),
                "pairwise_mae_p95_n": float(np.quantile(pairwise, 0.95)),
                "pairwise_mae_max_n": float(np.max(pairwise)),
                "per_cable_std_mean_n": float(np.std(tension_arr, axis=0).mean()),
                "per_cable_std_max_n": float(np.std(tension_arr, axis=0).max()),
            }
        )

    all_pairwise_arr = np.asarray(all_pairwise, dtype=float)
    all_pso_pairwise_arr = np.asarray(all_pso_pairwise, dtype=float)
    feasible_rate = float(np.mean([row["feasible"] for row in per_run_rows])) if per_run_rows else 0.0
    summary = evaluate_gate(
        median_pairwise_mae_n=float(np.median(all_pairwise_arr)),
        p95_pairwise_mae_n=float(np.quantile(all_pairwise_arr, 0.95)),
        feasible_rate=feasible_rate,
        median_threshold_n=float(args.median_threshold_n),
        p95_threshold_n=float(args.p95_threshold_n),
    )
    summary.update(
        {
            "config": str(args.config),
            "dataset": str(args.dataset),
            "num_theta_requested": int(args.num_theta),
            "num_theta_evaluated": int(len(row_positions)),
            "seeds": seeds,
            "feasible_rms_rnorm": feasible_rms,
            "objective": str(cfg.get("pso", {}).get("objective", "weighted_sum")),
            "canonical_enabled": canonical_enabled,
            "canonical_success_rate": float(np.mean([row["canonical_success"] for row in per_run_rows]))
            if canonical_enabled and per_run_rows
            else 0.0,
            "canonical_elapsed_s_mean": float(np.mean([row["canonical_elapsed_s"] for row in per_run_rows]))
            if canonical_enabled and per_run_rows
            else 0.0,
            "canonical_nfev_mean": float(np.mean([row["canonical_nfev"] for row in per_run_rows]))
            if canonical_enabled and per_run_rows
            else 0.0,
            "pso_median_pairwise_mae_n": float(np.median(all_pso_pairwise_arr)),
            "pso_p95_pairwise_mae_n": float(np.quantile(all_pso_pairwise_arr, 0.95)),
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_run_rows).to_csv(out_dir / "per_run.csv", index=False)
    pd.DataFrame(per_theta_rows).to_csv(out_dir / "per_theta.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_unstable and not bool(summary["gate_pass"]):
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--num-theta", type=int, default=10)
    ap.add_argument("--seeds", type=str, default="1001,1002,1003,1004,1005,1006,1007,1008")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--feasible-rms-rnorm", type=float, default=6.0e-2)
    ap.add_argument("--median-threshold-n", type=float, default=100.0)
    ap.add_argument("--p95-threshold-n", type=float, default=200.0)
    ap.add_argument("--disable-canonical", action="store_true")
    ap.add_argument("--fail-on-unstable", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
