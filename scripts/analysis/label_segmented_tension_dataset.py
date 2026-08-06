#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.quasi_static import QuasiStaticModel  # noqa: E402
from quasi_exp.opt.segmented_tension import solve_tensions_segmented  # noqa: E402
from subprocess_pool import poll_results, start_workers, stop_workers, submit_task  # noqa: E402


TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
CASE_COLS = [f"case_{i}" for i in range(1, 13)]


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _solver_meta(res, *, sample_id: int, elapsed_s: float, cache: Any) -> dict[str, Any]:
    total_nfev = int(sum(int(v) for v in res.section_nfev.values()))
    meta: dict[str, Any] = {
        "sample_id": int(sample_id),
        "tension_solver_method": "segmented_canonical",
        "rms_rnorm": float(res.rms_rnorm),
        "mean_rnorm2": float(res.mean_rnorm2),
        "max_tension": float(res.max_tension),
        "best_cost": float(res.mean_rnorm2),
        "evals": total_nfev,
        "elapsed_s": float(elapsed_s),
        "canonical_enabled": True,
        "canonical_adopted": bool(res.success),
        "canonical_success": bool(res.success),
        "canonical_method": "segmented_canonical",
        "segmented_success": bool(res.success),
        "segmented_elapsed_s": float(res.elapsed_s),
        "segmented_rms_rnorm": float(res.rms_rnorm),
        "segmented_mean_rnorm2": float(res.mean_rnorm2),
        "segmented_max_tension": float(res.max_tension),
    }
    case_flag = np.asarray(getattr(cache, "case_flag_12", np.zeros(12, dtype=int)), dtype=int).reshape(12)
    for i, value in enumerate(case_flag.tolist(), start=1):
        meta[f"case_{i}"] = int(value)
    for name in ["third", "second", "first"]:
        meta[f"segmented_section_{name}_rms_rnorm"] = float(res.section_rms_rnorm.get(name, np.nan))
        meta[f"segmented_section_{name}_elapsed_s"] = float(res.section_elapsed_s.get(name, np.nan))
        meta[f"segmented_section_{name}_nfev"] = int(res.section_nfev.get(name, 0))
    return meta


def worker_main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)
    solver_cfg = dict(cfg.get("segmented_tension", {}))
    if args.segmented_max_nfev is not None:
        solver_cfg["max_nfev"] = int(args.segmented_max_nfev)
    if args.segmented_t_ref_n is not None:
        solver_cfg["t_ref_n"] = float(args.segmented_t_ref_n)
    if args.feasible_rms_rnorm is not None:
        solver_cfg["feasible_rms_rnorm"] = float(args.feasible_rms_rnorm)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            task = json.loads(line)
            sample_id = int(task["sample_id"])
            theta = np.asarray(task["theta_rad"], dtype=float).reshape(30)
            cache = model.build_cache(theta)
            t0 = time.perf_counter()
            res = solve_tensions_segmented(model, cache, solver_cfg)
            elapsed_s = float(time.perf_counter() - t0)
            out = {
                "ok": bool(res.success),
                "sample_id": sample_id,
                "tension_n": np.asarray(res.T_base_12, dtype=float).reshape(12).tolist(),
                "meta": _solver_meta(res, sample_id=sample_id, elapsed_s=elapsed_s, cache=cache),
            }
        except Exception as exc:  # noqa: BLE001
            sid = int(task.get("sample_id", -1)) if "task" in locals() else -1
            out = {"ok": False, "sample_id": sid, "error": repr(exc)}
        print(json.dumps(out, allow_nan=False), flush=True)
    return 0


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def run_label(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.time()
    cfg = load_config(args.config)
    solver_cfg = dict(cfg.get("segmented_tension", {}))
    if args.segmented_max_nfev is not None:
        solver_cfg["max_nfev"] = int(args.segmented_max_nfev)
    if args.segmented_t_ref_n is not None:
        solver_cfg["t_ref_n"] = float(args.segmented_t_ref_n)
    if args.feasible_rms_rnorm is not None:
        solver_cfg["feasible_rms_rnorm"] = float(args.feasible_rms_rnorm)

    dataset = pd.read_parquet(args.dataset)
    theta_cols = _numbered_cols(list(dataset.columns), "theta_", "_rad")
    if len(theta_cols) != 30:
        raise SystemExit(f"dataset must contain 30 theta columns, got {len(theta_cols)}")
    if "sample_id" not in dataset.columns:
        dataset = dataset.copy()
        dataset.insert(0, "sample_id", np.arange(len(dataset), dtype=int))
    dataset = dataset.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if int(args.num_samples) > 0 and int(args.num_samples) < len(dataset):
        dataset = dataset.sample(n=int(args.num_samples), random_state=int(args.seed)).sort_values("sample_id", kind="mergesort").reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(args.config),
    ]
    if args.segmented_max_nfev is not None:
        cmd.extend(["--segmented-max-nfev", str(args.segmented_max_nfev)])
    if args.segmented_t_ref_n is not None:
        cmd.extend(["--segmented-t-ref-n", str(args.segmented_t_ref_n)])
    if args.feasible_rms_rnorm is not None:
        cmd.extend(["--feasible-rms-rnorm", str(args.feasible_rms_rnorm)])

    workers = start_workers(cmd, int(args.workers))
    results: dict[int, dict[str, Any]] = {}
    in_flight = 0
    next_idx = 0
    max_in_flight = max(1, int(args.workers) * int(args.inflight_per_worker))
    pbar = tqdm(total=len(dataset), desc="segmented_tension_label")
    try:
        while len(results) < len(dataset):
            dead = [w for w in workers if w.proc.poll() is not None]
            if dead:
                raise SystemExit("worker exited during tension labeling")
            while next_idx < len(dataset) and in_flight < max_in_flight:
                row = dataset.iloc[next_idx]
                task = {
                    "sample_id": int(row["sample_id"]),
                    "theta_rad": [float(row[c]) for c in theta_cols],
                }
                submit_task(workers[next_idx % len(workers)], task)
                next_idx += 1
                in_flight += 1
            res = poll_results(workers, timeout_s=0.5)
            if res is None:
                continue
            in_flight -= 1
            sample_id = int(res.get("sample_id", -1))
            if not bool(res.get("ok", False)) and not bool(args.accept_infeasible):
                raise SystemExit(f"tension solve failed sample_id={sample_id}: {res}")
            results[sample_id] = res
            pbar.update(1)
    finally:
        pbar.close()
        stop_workers(workers)

    out_dataset = dataset.copy()
    meta_rows: list[dict[str, Any]] = []
    for row_pos, row in out_dataset.iterrows():
        sample_id = int(row["sample_id"])
        res = results[sample_id]
        if "tension_n" in res:
            out_dataset.loc[row_pos, TENSION_COLS] = np.asarray(res["tension_n"], dtype=float)
        else:
            out_dataset.loc[row_pos, TENSION_COLS] = np.full(12, np.nan, dtype=float)
        meta = dict(res.get("meta", {}))
        meta["sample_id"] = sample_id
        if not bool(res.get("ok", False)):
            meta["error"] = str(res.get("error", "unknown"))
        meta_rows.append(meta)
    out_meta = pd.DataFrame(meta_rows).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    for col in CASE_COLS:
        if col not in out_meta.columns:
            out_meta[col] = 0

    pq.write_table(pa.Table.from_pandas(out_dataset, preserve_index=False), out_dir / "dataset.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(out_meta, preserve_index=False), out_dir / "dataset_meta.parquet", compression="zstd")

    elapsed_s = float(time.time() - t_start)
    success = out_meta["canonical_success"].fillna(False).to_numpy(dtype=bool) if "canonical_success" in out_meta.columns else np.zeros(len(out_meta), dtype=bool)
    rms = out_meta["rms_rnorm"].to_numpy(dtype=float) if "rms_rnorm" in out_meta.columns else np.asarray([], dtype=float)
    max_t = out_meta["max_tension"].to_numpy(dtype=float) if "max_tension" in out_meta.columns else np.asarray([], dtype=float)
    per_row = out_meta["elapsed_s"].to_numpy(dtype=float) if "elapsed_s" in out_meta.columns else np.asarray([], dtype=float)
    summary = {
        "mode": "segmented_tension_label",
        "source_dataset": str(args.dataset),
        "config": str(args.config),
        "out_dir": str(out_dir),
        "rows": int(len(out_dataset)),
        "workers": int(args.workers),
        "solver_cfg": solver_cfg,
        "accepted_infeasible": int(np.sum(~success)),
        "success_rate": float(np.mean(success)) if success.size else 0.0,
        "elapsed_s": elapsed_s,
        "sec_per_row_wall": elapsed_s / max(int(len(out_dataset)), 1),
        "solver_elapsed_s": _quantiles(per_row),
        "rms_rnorm": _quantiles(rms),
        "max_tension_n": _quantiles(max_t),
        "estimated_20k_wall_s": elapsed_s / max(int(len(out_dataset)), 1) * 20000.0,
        "estimated_100k_wall_s": elapsed_s / max(int(len(out_dataset)), 1) * 100000.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_standard_2k_segmented_canonical.yaml")
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--num-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inflight-per-worker", type=int, default=4)
    ap.add_argument("--segmented-max-nfev", type=int, default=None)
    ap.add_argument("--segmented-t-ref-n", type=float, default=None)
    ap.add_argument("--feasible-rms-rnorm", type=float, default=None)
    ap.add_argument("--accept-infeasible", action="store_true")
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return worker_main(args)
    if args.dataset is None or args.out_dir is None:
        raise SystemExit("--dataset and --out-dir are required")
    run_label(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
