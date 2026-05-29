from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs
from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.opt.pso import solve_tensions_pso
from quasi_exp.opt.segmented_tension import solve_tensions_segmented


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
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _select_row_positions(n_rows: int, num_theta: int) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("dataset is empty")
    count = min(max(1, int(num_theta)), n_rows)
    return np.unique(np.linspace(0, n_rows - 1, num=count, dtype=int))


def _pairwise_mae(tensions: np.ndarray) -> np.ndarray:
    arr = np.asarray(tensions, dtype=float)
    values: list[float] = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            values.append(float(np.mean(np.abs(arr[i] - arr[j]))))
    return np.asarray(values, dtype=float)


def _q(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _solver_cfg_from_args(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(cfg.get("segmented_tension", {}))
    if args.segmented_max_nfev is not None:
        out["max_nfev"] = int(args.segmented_max_nfev)
    if args.segmented_t_ref_n is not None:
        out["t_ref_n"] = float(args.segmented_t_ref_n)
    if args.feasible_rms_rnorm is not None:
        out["feasible_rms_rnorm"] = float(args.feasible_rms_rnorm)
    return out


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)
    solver_cfg = _solver_cfg_from_args(cfg, args)

    df = pd.read_parquet(args.dataset)
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    if len(theta_cols) != 30:
        raise SystemExit(f"dataset must contain 30 theta_*_rad columns, got {len(theta_cols)}")

    row_positions = _select_row_positions(len(df), int(args.num_theta))
    seeds = _parse_seeds(args.pso_seeds)
    feasible_rms = float(args.feasible_rms_rnorm)
    segmented_repeats = max(1, int(args.segmented_repeats))

    per_run_rows: list[dict[str, Any]] = []
    per_theta_rows: list[dict[str, Any]] = []
    all_segmented_pairwise: list[float] = []
    all_pso_pairwise: list[float] = []
    section_rms_by_name: dict[str, list[float]] = {"third": [], "second": [], "first": []}
    section_nfev_by_name: dict[str, list[int]] = {"third": [], "second": [], "first": []}

    for pos in row_positions.tolist():
        sample_id = int(df.iloc[pos]["sample_id"]) if "sample_id" in df.columns else int(pos)
        theta = df.iloc[pos][theta_cols].to_numpy(dtype=float)
        cache = model.build_cache(theta)

        segmented_tensions = []
        segmented_rms = []
        segmented_elapsed = []
        segmented_success = []
        for repeat in range(segmented_repeats):
            res = solve_tensions_segmented(model, cache, solver_cfg)
            segmented_tensions.append(np.asarray(res.T_base_12, dtype=float))
            segmented_rms.append(float(res.rms_rnorm))
            segmented_elapsed.append(float(res.elapsed_s))
            segmented_success.append(bool(res.success))
            for name, value in res.section_rms_rnorm.items():
                section_rms_by_name.setdefault(name, []).append(float(value))
            for name, value in res.section_nfev.items():
                section_nfev_by_name.setdefault(name, []).append(int(value))
            per_run_rows.append(
                {
                    "solver": "segmented",
                    "sample_id": sample_id,
                    "row_position": int(pos),
                    "repeat_or_seed": int(repeat),
                    "rms_rnorm": float(res.rms_rnorm),
                    "feasible": bool(res.rms_rnorm <= feasible_rms),
                    "success": bool(res.success),
                    "elapsed_s": float(res.elapsed_s),
                    "max_tension_n": float(res.max_tension),
                    "nfev_total": int(sum(res.section_nfev.values())),
                    "section_third_rms_rnorm": float(res.section_rms_rnorm.get("third", np.nan)),
                    "section_second_rms_rnorm": float(res.section_rms_rnorm.get("second", np.nan)),
                    "section_first_rms_rnorm": float(res.section_rms_rnorm.get("first", np.nan)),
                }
            )

        seg_arr = np.vstack(segmented_tensions)
        seg_pairwise = _pairwise_mae(seg_arr) if segmented_repeats >= 2 else np.asarray([0.0], dtype=float)
        all_segmented_pairwise.extend(seg_pairwise.tolist())

        pso_tensions = []
        pso_rms = []
        pso_elapsed = []
        if not args.skip_pso:
            for seed in seeds:
                t0 = time.perf_counter()
                res = solve_tensions_pso(model, cache, cfg["pso"], rng_seed=int(seed))
                elapsed = float(time.perf_counter() - t0)
                rms = float(np.sqrt(res.mean_rnorm2))
                pso_tensions.append(np.asarray(res.T_base_12, dtype=float))
                pso_rms.append(rms)
                pso_elapsed.append(elapsed)
                per_run_rows.append(
                    {
                        "solver": "pso",
                        "sample_id": sample_id,
                        "row_position": int(pos),
                        "repeat_or_seed": int(seed),
                        "rms_rnorm": rms,
                        "feasible": bool(rms <= feasible_rms),
                        "success": bool(rms <= feasible_rms),
                        "elapsed_s": elapsed,
                        "max_tension_n": float(res.max_tension),
                        "nfev_total": int(res.evals),
                        "section_third_rms_rnorm": float("nan"),
                        "section_second_rms_rnorm": float("nan"),
                        "section_first_rms_rnorm": float("nan"),
                    }
                )

        if pso_tensions:
            pso_arr = np.vstack(pso_tensions)
            pso_pairwise = _pairwise_mae(pso_arr)
            all_pso_pairwise.extend(pso_pairwise.tolist())
        else:
            pso_pairwise = np.asarray([], dtype=float)

        per_theta_rows.append(
            {
                "sample_id": sample_id,
                "row_position": int(pos),
                "segmented_repeat_count": int(segmented_repeats),
                "segmented_rms_rnorm_median": float(np.median(segmented_rms)),
                "segmented_rms_rnorm_max": float(np.max(segmented_rms)),
                "segmented_success_rate": float(np.mean(segmented_success)),
                "segmented_elapsed_s_mean": float(np.mean(segmented_elapsed)),
                "segmented_pairwise_mae_median_n": float(np.median(seg_pairwise)),
                "segmented_pairwise_mae_max_n": float(np.max(seg_pairwise)),
                "segmented_repeat_max_abs_delta_n": float(np.max(np.ptp(seg_arr, axis=0))),
                "pso_seed_count": int(len(seeds) if not args.skip_pso else 0),
                "pso_rms_rnorm_median": float(np.median(pso_rms)) if pso_rms else float("nan"),
                "pso_rms_rnorm_max": float(np.max(pso_rms)) if pso_rms else float("nan"),
                "pso_elapsed_s_mean": float(np.mean(pso_elapsed)) if pso_elapsed else float("nan"),
                "pso_pairwise_mae_median_n": float(np.median(pso_pairwise)) if pso_pairwise.size else float("nan"),
                "pso_pairwise_mae_max_n": float(np.max(pso_pairwise)) if pso_pairwise.size else float("nan"),
            }
        )

    per_run_df = pd.DataFrame(per_run_rows)
    seg_rows = per_run_df[per_run_df["solver"] == "segmented"]
    pso_rows = per_run_df[per_run_df["solver"] == "pso"]
    seg_elapsed = seg_rows["elapsed_s"].to_numpy(dtype=float)
    pso_elapsed = pso_rows["elapsed_s"].to_numpy(dtype=float) if len(pso_rows) else np.asarray([], dtype=float)

    summary: dict[str, Any] = {
        "config": str(args.config),
        "dataset": str(args.dataset),
        "num_theta_requested": int(args.num_theta),
        "num_theta_evaluated": int(len(row_positions)),
        "feasible_rms_rnorm": feasible_rms,
        "segmented_solver_cfg": solver_cfg,
        "segmented": {
            "repeats": int(segmented_repeats),
            "success_rate": float(seg_rows["success"].mean()) if len(seg_rows) else 0.0,
            "feasible_rate": float(seg_rows["feasible"].mean()) if len(seg_rows) else 0.0,
            "rms_rnorm": _q(seg_rows["rms_rnorm"].to_numpy(dtype=float)),
            "elapsed_s": _q(seg_elapsed),
            "pairwise_mae_n": _q(all_segmented_pairwise),
            "repeat_max_abs_delta_n": float(
                np.max([row["segmented_repeat_max_abs_delta_n"] for row in per_theta_rows])
            )
            if per_theta_rows
            else float("nan"),
            "estimated_10k_s": float(np.mean(seg_elapsed) * 10000.0) if seg_elapsed.size else float("nan"),
            "estimated_100k_s": float(np.mean(seg_elapsed) * 100000.0) if seg_elapsed.size else float("nan"),
            "section_rms_rnorm": {name: _q(values) for name, values in section_rms_by_name.items()},
            "section_nfev": {name: _q(values) for name, values in section_nfev_by_name.items()},
        },
        "pso": {
            "skipped": bool(args.skip_pso),
            "seeds": seeds if not args.skip_pso else [],
            "feasible_rate": float(pso_rows["feasible"].mean()) if len(pso_rows) else float("nan"),
            "rms_rnorm": _q(pso_rows["rms_rnorm"].to_numpy(dtype=float)) if len(pso_rows) else _q([]),
            "elapsed_s": _q(pso_elapsed),
            "pairwise_mae_n": _q(all_pso_pairwise),
        },
    }
    if pso_elapsed.size and seg_elapsed.size:
        summary["speedup_vs_pso_mean"] = float(np.mean(pso_elapsed) / np.mean(seg_elapsed))
    else:
        summary["speedup_vs_pso_mean"] = float("nan")
    summary = _json_safe(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_df.to_csv(out_dir / "per_run.csv", index=False)
    pd.DataFrame(per_theta_rows).to_csv(out_dir / "per_theta.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--num-theta", type=int, default=20)
    ap.add_argument("--pso-seeds", type=str, default="1001,1002,1003")
    ap.add_argument("--segmented-repeats", type=int, default=2)
    ap.add_argument("--segmented-max-nfev", type=int, default=80)
    ap.add_argument("--segmented-t-ref-n", type=float, default=800.0)
    ap.add_argument("--feasible-rms-rnorm", type=float, default=6.0e-2)
    ap.add_argument("--skip-pso", action="store_true")
    ap.add_argument("--out-dir", required=True, type=Path)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
