#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.quasi_static import QuasiStaticModel  # noqa: E402
from quasi_exp.model.sampling import beta_to_theta  # noqa: E402
from quasi_exp.opt.segmented_tension import solve_tensions_segmented  # noqa: E402
from subprocess_pool import poll_results, start_workers, stop_workers, submit_task  # noqa: E402


TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
XYZ_COLS = ["x_m", "y_m", "z_m"]


class AnchorInfo(NamedTuple):
    anchor_tension_n: np.ndarray
    neighbor_sample_ids: list[int]
    neighbor_distances: list[float]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def beta_scales_from_config(cfg: dict[str, Any]) -> np.ndarray:
    ranges = cfg["sampling"]["beta_ranges_rad"]
    return np.asarray([max(abs(float(ranges[f"beta{i}"][0])), abs(float(ranges[f"beta{i}"][1])), 1.0e-12) for i in range(1, 7)], dtype=float)


def _source_ids(meta: pd.DataFrame) -> np.ndarray:
    col = "source_sample_id" if "source_sample_id" in meta.columns else "sample_id"
    return meta[col].to_numpy(dtype=int)


def robust_anchor(tensions: np.ndarray, *, stat: str = "median") -> np.ndarray:
    arr = np.asarray(tensions, dtype=float).reshape(-1, 12)
    if arr.shape[0] == 0:
        raise ValueError("cannot anchor from zero tensions")
    if arr.shape[0] == 1:
        return arr[0].copy()
    if stat == "median":
        return np.median(arr, axis=0)
    if stat != "huber_mean":
        raise ValueError(f"unsupported anchor_stat: {stat}")
    center = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - center.reshape(1, 12)), axis=0)
    delta = np.maximum(1.5 * 1.4826 * mad, 1.0)
    out = center.copy()
    for _ in range(8):
        resid = arr - out.reshape(1, 12)
        weights = np.minimum(1.0, delta.reshape(1, 12) / np.maximum(np.abs(resid), 1.0e-12))
        out = np.sum(arr * weights, axis=0) / np.maximum(np.sum(weights, axis=0), 1.0e-12)
    return out


def _beta_cols(meta: pd.DataFrame, distance_space: str) -> list[str]:
    if distance_space in {"effective_beta", "effective_beta_xyz"}:
        if set(EFFECTIVE_BETA_COLS).issubset(meta.columns):
            return EFFECTIVE_BETA_COLS
        raise ValueError("effective_beta distance requested but meta is missing effective_beta_*_rad")
    if distance_space in {"beta", "beta_xyz"}:
        if set(BETA_COLS).issubset(meta.columns):
            return BETA_COLS
        raise ValueError("beta distance requested but meta is missing beta*_rad")
    raise ValueError(f"unsupported distance_space: {distance_space}")


def _features(meta: pd.DataFrame, *, distance_space: str, beta_scales: np.ndarray | None) -> np.ndarray:
    cols = _beta_cols(meta, distance_space)
    beta = meta[cols].to_numpy(dtype=float)
    if beta_scales is None:
        scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    else:
        scale = np.asarray(beta_scales, dtype=float).reshape(6)
    feat = [beta / scale.reshape(1, 6)]
    if distance_space.endswith("_xyz"):
        missing = [c for c in XYZ_COLS if c not in meta.columns]
        if missing:
            raise ValueError(f"{distance_space} requires xyz columns in meta: {missing}")
        xyz = meta[XYZ_COLS].to_numpy(dtype=float)
        xyz_scale = max(float(np.nanmax(np.linalg.norm(xyz - np.nanmean(xyz, axis=0), axis=1))), 1.0e-6)
        feat.append(xyz / xyz_scale)
    return np.concatenate(feat, axis=1)


def _layer_groups(meta: pd.DataFrame) -> np.ndarray:
    if "layer_label" in meta.columns:
        return meta["layer_label"].astype(str).to_numpy()
    if "source_component" in meta.columns:
        return meta["source_component"].astype(str).to_numpy()
    return np.full(len(meta), "single_layer", dtype=object)


def build_layer_graph_anchors(
    *,
    pool_meta: pd.DataFrame,
    pool_tension: np.ndarray,
    selected_meta: pd.DataFrame,
    k_neighbors: int,
    distance_space: str = "effective_beta_xyz",
    anchor_stat: str = "median",
    beta_scales: np.ndarray | None = None,
) -> list[AnchorInfo]:
    if int(k_neighbors) < 1:
        raise ValueError("k_neighbors must be >= 1")
    if len(pool_meta) != int(np.asarray(pool_tension).shape[0]):
        raise ValueError("pool_meta and pool_tension row counts differ")
    pool_feat = _features(pool_meta, distance_space=distance_space, beta_scales=beta_scales)
    sel_feat = _features(selected_meta, distance_space=distance_space, beta_scales=beta_scales)
    pool_groups = _layer_groups(pool_meta)
    selected_groups = _layer_groups(selected_meta)
    pool_source_ids = _source_ids(pool_meta)
    selected_source_ids = _source_ids(selected_meta)

    anchors_by_pos: dict[int, AnchorInfo] = {}
    for group in sorted(set(str(v) for v in selected_groups.tolist())):
        pool_idx = np.where(pool_groups == group)[0]
        sel_idx = np.where(selected_groups == group)[0]
        if pool_idx.size == 0:
            raise ValueError(f"no pool rows for layer={group}")
        n_query = min(int(k_neighbors) + 1, int(pool_idx.size))
        nn = NearestNeighbors(n_neighbors=n_query, algorithm="auto").fit(pool_feat[pool_idx])
        dist, local_idx = nn.kneighbors(sel_feat[sel_idx])
        for row_pos, drow, lrow in zip(sel_idx.tolist(), dist, local_idx):
            source_id = int(selected_source_ids[row_pos])
            candidate_pool_idx = pool_idx[np.asarray(lrow, dtype=int)]
            candidate_dist = np.asarray(drow, dtype=float)
            keep = pool_source_ids[candidate_pool_idx] != source_id
            candidate_pool_idx = candidate_pool_idx[keep][: int(k_neighbors)]
            candidate_dist = candidate_dist[keep][: int(k_neighbors)]
            if candidate_pool_idx.size == 0:
                fallback = int(pool_idx[0])
                anchors_by_pos[row_pos] = AnchorInfo(
                    anchor_tension_n=np.asarray(pool_tension[fallback], dtype=float).reshape(12),
                    neighbor_sample_ids=[int(pool_source_ids[fallback])],
                    neighbor_distances=[0.0],
                )
                continue
            anchors_by_pos[row_pos] = AnchorInfo(
                anchor_tension_n=robust_anchor(pool_tension[candidate_pool_idx], stat=anchor_stat),
                neighbor_sample_ids=[int(v) for v in pool_source_ids[candidate_pool_idx].tolist()],
                neighbor_distances=[float(v) for v in candidate_dist.tolist()],
            )
    return [anchors_by_pos[i] for i in range(len(selected_meta))]


def _solver_meta(res, *, task: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    total_nfev = int(sum(int(v) for v in res.section_nfev.values()))
    meta: dict[str, Any] = {
        "rms_rnorm": float(res.rms_rnorm),
        "mean_rnorm2": float(res.mean_rnorm2),
        "max_tension": float(res.max_tension),
        "best_cost": float(res.mean_rnorm2),
        "iters_used": 0,
        "evals": total_nfev,
        "pso_seed": int(task.get("pso_seed", 0)),
        "elapsed_s": float(elapsed_s),
        "canonical_enabled": True,
        "canonical_adopted": bool(res.success),
        "canonical_success": bool(res.success),
        "canonical_method": "graph_anchor_segmented_canonical",
        "segmented_success": bool(res.success),
        "anchor_enabled": True,
        "anchor_k": int(task["anchor_k"]),
        "anchor_w": float(task["w_anchor"]),
        "anchor_stat": str(task["anchor_stat"]),
        "anchor_scope": "same_layer_graph",
        "anchor_distance_space": str(task["anchor_distance_space"]),
        "anchor_neighbor_count": int(len(task.get("anchor_neighbor_sample_ids", []))),
        "anchor_neighbor_mean_dist": float(np.mean(task.get("anchor_neighbor_distances", [0.0]))),
        "anchor_neighbor_max_dist": float(np.max(task.get("anchor_neighbor_distances", [0.0]))),
    }
    for name in ["third", "second", "first"]:
        meta[f"segmented_section_{name}_rms_rnorm"] = float(res.section_rms_rnorm.get(name, np.nan))
        meta[f"segmented_section_{name}_elapsed_s"] = float(res.section_elapsed_s.get(name, np.nan))
        meta[f"segmented_section_{name}_nfev"] = int(res.section_nfev.get(name, 0))
    return meta


def worker_main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)
    solver_cfg_base = dict(cfg.get("segmented_tension", {}))
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            task = json.loads(line)
            beta = np.asarray(task["beta6_rad"], dtype=float).reshape(6)
            theta_raw = beta_to_theta(beta)
            cache = model.build_cache(theta_raw)
            solver_cfg = dict(solver_cfg_base)
            solver_cfg["anchor_tension_n"] = task["anchor_tension_n"]
            solver_cfg["w_anchor"] = float(task["w_anchor"])
            t0 = time.perf_counter()
            res = solve_tensions_segmented(model, cache, solver_cfg)
            elapsed_s = float(time.perf_counter() - t0)
            out = {
                "ok": bool(res.success),
                "sample_id": int(task["sample_id"]),
                "source_sample_id": int(task.get("source_sample_id", task["sample_id"])),
                "tension_n": np.asarray(res.T_base_12, dtype=float).reshape(12).tolist(),
                "meta": _solver_meta(res, task=task, elapsed_s=elapsed_s),
            }
        except Exception as exc:
            out = {"ok": False, "sample_id": int(task.get("sample_id", -1)) if "task" in locals() else -1, "error": repr(exc)}
        print(json.dumps(out), flush=True)


def _merge_xyz_into_meta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if set(XYZ_COLS).issubset(meta.columns):
        return meta.copy()
    return meta.merge(dataset[["sample_id"] + XYZ_COLS], on="sample_id", how="left", validate="one_to_one")


def run_relabel(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.time()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    meta = pd.read_parquet(args.meta).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if dataset["sample_id"].tolist() != meta["sample_id"].tolist():
        raise SystemExit("dataset/meta sample_id mismatch")

    pool_meta = _merge_xyz_into_meta(dataset, meta)
    selected_meta = pool_meta.copy().reset_index(drop=True)
    selected_meta["source_sample_id"] = selected_meta["sample_id"].astype(int)
    if int(args.num_samples) > 0 and int(args.num_samples) < len(selected_meta):
        selected_meta = selected_meta.sample(n=int(args.num_samples), random_state=int(args.seed)).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    source_ids = selected_meta["source_sample_id"].to_numpy(dtype=int)
    selected_dataset = dataset.set_index("sample_id").loc[source_ids].reset_index(drop=True)
    selected_dataset["sample_id"] = np.arange(len(selected_dataset), dtype=int)
    selected_meta["sample_id"] = np.arange(len(selected_meta), dtype=int)

    beta_scales = beta_scales_from_config(cfg)
    anchors = build_layer_graph_anchors(
        pool_meta=pool_meta,
        pool_tension=dataset[TENSION_COLS].to_numpy(dtype=float),
        selected_meta=selected_meta,
        k_neighbors=int(args.anchor_k),
        distance_space=str(args.distance_space),
        anchor_stat=str(args.anchor_stat),
        beta_scales=beta_scales,
    )

    beta_cols = EFFECTIVE_BETA_COLS if str(args.distance_space).startswith("effective_beta") else BETA_COLS
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", 1.0))
    task_beta_scale = 1.0 / theta_sign if str(args.distance_space).startswith("effective_beta") else 1.0

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(args.config)]
    workers = start_workers(cmd, int(args.workers))
    results: dict[int, dict[str, Any]] = {}
    in_flight = 0
    next_idx = 0
    max_in_flight = max(1, int(args.workers) * 4)
    pbar = tqdm(total=len(selected_meta), desc="graph_relabel")
    try:
        while len(results) < len(selected_meta):
            dead = [w for w in workers if w.proc.poll() is not None]
            if dead:
                raise SystemExit("worker exited during graph relabel")
            while next_idx < len(selected_meta) and in_flight < max_in_flight:
                mrow = selected_meta.iloc[next_idx]
                anchor = anchors[next_idx]
                task = {
                    "sample_id": int(mrow["sample_id"]),
                    "source_sample_id": int(mrow["source_sample_id"]),
                    "beta6_rad": [float(mrow[c]) * task_beta_scale for c in beta_cols],
                    "anchor_tension_n": anchor.anchor_tension_n.tolist(),
                    "anchor_neighbor_sample_ids": anchor.neighbor_sample_ids,
                    "anchor_neighbor_distances": anchor.neighbor_distances,
                    "anchor_k": int(args.anchor_k),
                    "w_anchor": float(args.w_anchor),
                    "anchor_stat": str(args.anchor_stat),
                    "anchor_distance_space": str(args.distance_space),
                    "pso_seed": int(cfg.get("pso", {}).get("rng_seed", 0)) + int(mrow["sample_id"]),
                }
                submit_task(workers[next_idx % len(workers)], task)
                next_idx += 1
                in_flight += 1
            res = poll_results(workers, timeout_s=0.5)
            if res is None:
                continue
            in_flight -= 1
            if not bool(res.get("ok", False)) and not bool(args.accept_infeasible):
                raise SystemExit(f"relabel failed sample_id={res.get('sample_id')}: {res}")
            results[int(res["sample_id"])] = res
            pbar.update(1)
    finally:
        pbar.close()
        stop_workers(workers)

    out_dataset = selected_dataset.copy()
    out_meta = selected_meta.copy()
    for sample_id in range(len(out_dataset)):
        res = results[sample_id]
        out_dataset.loc[sample_id, TENSION_COLS] = np.asarray(res["tension_n"], dtype=float)
        for key, value in dict(res["meta"]).items():
            out_meta.loc[sample_id, key] = value
        out_meta.loc[sample_id, "anchor_neighbor_sample_ids"] = ",".join(str(v) for v in anchors[sample_id].neighbor_sample_ids)
    pq.write_table(pa.Table.from_pandas(out_dataset, preserve_index=False), out_dir / "dataset.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(out_meta, preserve_index=False), out_dir / "dataset_meta.parquet", compression="zstd")
    elapsed_s = float(time.time() - t_start)
    report = {
        "mode": "graph_anchor_tension_relabel",
        "source_dataset": str(args.dataset),
        "source_meta": str(args.meta),
        "accepted": int(len(out_dataset)),
        "accepted_infeasible": int(sum(1 for r in results.values() if not bool(r.get("ok", False)))),
        "elapsed_s": elapsed_s,
        "sec_per_accepted": elapsed_s / max(int(len(out_dataset)), 1),
        "anchor_k": int(args.anchor_k),
        "w_anchor": float(args.w_anchor),
        "anchor_stat": str(args.anchor_stat),
        "distance_space": str(args.distance_space),
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--num-samples", type=int, default=0, help="0 means relabel all rows")
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--distance-space", choices=["effective_beta_xyz", "beta_xyz", "effective_beta", "beta"], default="effective_beta_xyz")
    ap.add_argument("--anchor-k", type=int, default=16)
    ap.add_argument("--w-anchor", type=float, default=20.0)
    ap.add_argument("--anchor-stat", choices=["median", "huber_mean"], default="median")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accept-infeasible", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        worker_main(args)
        return 0
    if args.dataset is None or args.meta is None or args.out_dir is None:
        raise SystemExit("--dataset, --meta, and --out-dir are required outside --worker")
    run_relabel(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
