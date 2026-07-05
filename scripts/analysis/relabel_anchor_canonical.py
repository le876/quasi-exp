#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quasi_exp.io import load_config, load_robot_inputs
from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.opt.segmented_tension import solve_tensions_segmented
from subprocess_pool import poll_results, start_workers, stop_workers, submit_task


TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]


class AnchorInfo(NamedTuple):
    anchor_tension_n: np.ndarray
    neighbor_sample_ids: list[int]
    neighbor_distances: list[float]


def beta_scales_from_config(cfg: dict) -> np.ndarray:
    ranges = cfg["sampling"]["beta_ranges_rad"]
    scales = []
    for i in range(1, 7):
        lo, hi = ranges[f"beta{i}"]
        scales.append(max(abs(float(lo)), abs(float(hi)), 1.0e-12))
    return np.asarray(scales, dtype=float)


def _normalized_beta(meta: pd.DataFrame, beta_scales: np.ndarray | None) -> np.ndarray:
    beta = meta[BETA_COLS].to_numpy(dtype=float)
    if beta_scales is None:
        return beta
    return beta / np.asarray(beta_scales, dtype=float).reshape(1, 6)


def _beta_cols_for_distance_space(meta: pd.DataFrame, distance_space: str) -> list[str]:
    distance_space = str(distance_space)
    if distance_space == "effective_beta":
        missing = [c for c in EFFECTIVE_BETA_COLS if c not in meta.columns]
        if missing:
            raise ValueError(f"effective_beta distance requested but meta missing columns: {missing}")
        return EFFECTIVE_BETA_COLS
    if distance_space == "beta":
        missing = [c for c in BETA_COLS if c not in meta.columns]
        if missing:
            raise ValueError(f"beta distance requested but meta missing columns: {missing}")
        return BETA_COLS
    raise ValueError(f"unsupported distance_space: {distance_space}")


def _normalized_beta_from_cols(meta: pd.DataFrame, cols: list[str], beta_scales: np.ndarray | None) -> np.ndarray:
    beta = meta[cols].to_numpy(dtype=float)
    if beta_scales is None:
        return beta
    return beta / np.asarray(beta_scales, dtype=float).reshape(1, 6)


def _source_ids(meta: pd.DataFrame) -> np.ndarray:
    col = "source_sample_id" if "source_sample_id" in meta.columns else "sample_id"
    return meta[col].to_numpy(dtype=int)


def _weighted_anchor(tensions: np.ndarray, distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=float).reshape(-1)
    if d.size == 0:
        raise ValueError("cannot compute anchor from zero neighbors")
    sigma = max(float(np.max(d)), 1.0e-9)
    weights = np.exp(-0.5 * np.square(d / sigma))
    weights = weights / max(float(np.sum(weights)), 1.0e-12)
    return np.sum(np.asarray(tensions, dtype=float) * weights[:, None], axis=0)


def build_same_component_beta_knn_anchors(
    *,
    pool_meta: pd.DataFrame,
    pool_tension: np.ndarray,
    selected_meta: pd.DataFrame,
    k_neighbors: int,
    beta_scales: np.ndarray | None = None,
) -> list[AnchorInfo]:
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1")
    if len(pool_meta) != int(np.asarray(pool_tension).shape[0]):
        raise ValueError("pool_meta and pool_tension row counts differ")

    pool_source_ids = _source_ids(pool_meta)
    selected_source_ids = _source_ids(selected_meta)
    pool_beta = _normalized_beta(pool_meta, beta_scales)
    selected_beta = _normalized_beta(selected_meta, beta_scales)

    anchors_by_selected_pos: dict[int, AnchorInfo] = {}
    for component in sorted(selected_meta["source_component"].astype(str).unique().tolist()):
        pool_mask = pool_meta["source_component"].astype(str).to_numpy() == component
        sel_mask = selected_meta["source_component"].astype(str).to_numpy() == component
        pool_idx = np.where(pool_mask)[0]
        sel_idx = np.where(sel_mask)[0]
        if pool_idx.size == 0:
            raise ValueError(f"no pool rows for component={component}")

        n_query = min(int(k_neighbors) + 1, int(pool_idx.size))
        nn = NearestNeighbors(n_neighbors=n_query, algorithm="auto").fit(pool_beta[pool_idx])
        dist, local_idx = nn.kneighbors(selected_beta[sel_idx])
        for row_pos, drow, lrow in zip(sel_idx.tolist(), dist, local_idx):
            source_id = int(selected_source_ids[row_pos])
            candidate_pool_idx = pool_idx[np.asarray(lrow, dtype=int)]
            candidate_dist = np.asarray(drow, dtype=float)
            keep = pool_source_ids[candidate_pool_idx] != source_id
            candidate_pool_idx = candidate_pool_idx[keep][:k_neighbors]
            candidate_dist = candidate_dist[keep][:k_neighbors]
            if candidate_pool_idx.size == 0:
                self_matches = np.where(pool_source_ids == source_id)[0]
                fallback_idx = int(self_matches[0]) if self_matches.size else int(pool_idx[0])
                anchors_by_selected_pos[row_pos] = AnchorInfo(
                    anchor_tension_n=np.asarray(pool_tension[fallback_idx], dtype=float).reshape(12),
                    neighbor_sample_ids=[int(pool_source_ids[fallback_idx])],
                    neighbor_distances=[0.0],
                )
                continue
            anchors_by_selected_pos[row_pos] = AnchorInfo(
                anchor_tension_n=_weighted_anchor(pool_tension[candidate_pool_idx], candidate_dist),
                neighbor_sample_ids=[int(v) for v in pool_source_ids[candidate_pool_idx].tolist()],
                neighbor_distances=[float(v) for v in candidate_dist.tolist()],
            )

    return [anchors_by_selected_pos[i] for i in range(len(selected_meta))]


def build_selected_branch_graph_anchors(
    *,
    pool_meta: pd.DataFrame,
    pool_tension: np.ndarray,
    selected_meta: pd.DataFrame,
    k_neighbors: int,
    distance_space: str = "effective_beta",
    no_cross_branch_anchor: bool = True,
    beta_scales: np.ndarray | None = None,
) -> list[AnchorInfo]:
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1")
    if len(pool_meta) != int(np.asarray(pool_tension).shape[0]):
        raise ValueError("pool_meta and pool_tension row counts differ")

    cols = _beta_cols_for_distance_space(pool_meta, distance_space)
    _ = _beta_cols_for_distance_space(selected_meta, distance_space)
    pool_source_ids = _source_ids(pool_meta)
    selected_source_ids = _source_ids(selected_meta)
    pool_beta = _normalized_beta_from_cols(pool_meta, cols, beta_scales)
    selected_beta = _normalized_beta_from_cols(selected_meta, cols, beta_scales)

    if no_cross_branch_anchor and "graph_component_id" in pool_meta.columns and "graph_component_id" in selected_meta.columns:
        pool_groups = pool_meta["graph_component_id"].astype(str).to_numpy()
        selected_groups = selected_meta["graph_component_id"].astype(str).to_numpy()
    elif no_cross_branch_anchor and "graph_branch_label" in pool_meta.columns and "graph_branch_label" in selected_meta.columns:
        pool_groups = pool_meta["graph_branch_label"].astype(str).to_numpy()
        selected_groups = selected_meta["graph_branch_label"].astype(str).to_numpy()
    else:
        pool_groups = np.full(len(pool_meta), "selected_branch_pool", dtype=object)
        selected_groups = np.full(len(selected_meta), "selected_branch_pool", dtype=object)

    anchors_by_selected_pos: dict[int, AnchorInfo] = {}
    for group in sorted(set(str(v) for v in selected_groups.tolist())):
        pool_idx = np.where(pool_groups == group)[0]
        sel_idx = np.where(selected_groups == group)[0]
        if pool_idx.size == 0:
            raise ValueError(f"no pool rows for selected graph group={group}")
        n_query = min(int(k_neighbors) + 1, int(pool_idx.size))
        nn = NearestNeighbors(n_neighbors=n_query, algorithm="auto").fit(pool_beta[pool_idx])
        dist, local_idx = nn.kneighbors(selected_beta[sel_idx])
        for row_pos, drow, lrow in zip(sel_idx.tolist(), dist, local_idx):
            source_id = int(selected_source_ids[row_pos])
            candidate_pool_idx = pool_idx[np.asarray(lrow, dtype=int)]
            candidate_dist = np.asarray(drow, dtype=float)
            keep = pool_source_ids[candidate_pool_idx] != source_id
            candidate_pool_idx = candidate_pool_idx[keep][:k_neighbors]
            candidate_dist = candidate_dist[keep][:k_neighbors]
            if candidate_pool_idx.size == 0:
                self_matches = np.where(pool_source_ids == source_id)[0]
                fallback_idx = int(self_matches[0]) if self_matches.size else int(pool_idx[0])
                anchors_by_selected_pos[row_pos] = AnchorInfo(
                    anchor_tension_n=np.asarray(pool_tension[fallback_idx], dtype=float).reshape(12),
                    neighbor_sample_ids=[int(pool_source_ids[fallback_idx])],
                    neighbor_distances=[0.0],
                )
                continue
            anchors_by_selected_pos[row_pos] = AnchorInfo(
                anchor_tension_n=_weighted_anchor(pool_tension[candidate_pool_idx], candidate_dist),
                neighbor_sample_ids=[int(v) for v in pool_source_ids[candidate_pool_idx].tolist()],
                neighbor_distances=[float(v) for v in candidate_dist.tolist()],
            )
    return [anchors_by_selected_pos[i] for i in range(len(selected_meta))]


def _allocate_component_counts(meta: pd.DataFrame, num_samples: int) -> dict[str, int]:
    counts = meta["source_component"].astype(str).value_counts().sort_index()
    raw = counts / counts.sum() * int(num_samples)
    base = np.floor(raw).astype(int)
    remainder = int(num_samples) - int(base.sum())
    if remainder > 0:
        order = sorted(counts.index.tolist(), key=lambda c: (-float(raw[c] - base[c]), str(c)))
        for component in order[:remainder]:
            base[component] += 1
    return {str(k): int(v) for k, v in base.to_dict().items()}


def select_stratified(meta: pd.DataFrame, num_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    counts = _allocate_component_counts(meta, num_samples)
    selected = []
    for component, n in counts.items():
        block = meta.loc[meta["source_component"].astype(str) == component]
        if n > len(block):
            raise ValueError(f"requested {n} rows for {component}, only {len(block)} available")
        idx = rng.choice(block.index.to_numpy(), size=n, replace=False)
        selected.append(meta.loc[np.sort(idx)])
    out = pd.concat(selected, axis=0).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    out["source_sample_id"] = out["sample_id"].astype(int)
    out["sample_id"] = np.arange(len(out), dtype=int)
    return out


def _solver_meta(res, *, task: dict, elapsed_s: float) -> dict:
    section_nfev = {name: int(res.section_nfev.get(name, 0)) for name in ["third", "second", "first"]}
    total_nfev = int(sum(section_nfev.values()))
    meta = {
        "rms_rnorm": float(res.rms_rnorm),
        "mean_rnorm2": float(res.mean_rnorm2),
        "max_tension": float(res.max_tension),
        "best_cost": float(res.mean_rnorm2),
        "iters_used": 0,
        "evals": total_nfev,
        "pso_seed": int(task.get("pso_seed", 0)),
        "elapsed_s": float(elapsed_s),
        "pso_elapsed_s": 0.0,
        "canonical_enabled": True,
        "canonical_adopted": bool(res.success),
        "canonical_success": bool(res.success),
        "canonical_method": "segmented_anchor_canonical",
        "canonical_elapsed_s": float(res.elapsed_s),
        "canonical_nit": 0,
        "canonical_nfev": total_nfev,
        "canonical_objective": float(res.mean_rnorm2),
        "canonical_rms_rnorm": float(res.rms_rnorm),
        "canonical_mean_rnorm2": float(res.mean_rnorm2),
        "canonical_max_tension": float(res.max_tension),
        "canonical_source_index": 0,
        "segmented_success": bool(res.success),
        "segmented_elapsed_s": float(res.elapsed_s),
        "segmented_rms_rnorm": float(res.rms_rnorm),
        "segmented_mean_rnorm2": float(res.mean_rnorm2),
        "segmented_max_tension": float(res.max_tension),
        "anchor_enabled": True,
        "anchor_k": int(task["anchor_k"]),
        "anchor_w": float(task["w_anchor"]),
        "anchor_neighbor_count": int(len(task.get("anchor_neighbor_sample_ids", []))),
        "anchor_neighbor_mean_dist": float(np.mean(task.get("anchor_neighbor_distances", [0.0]))),
        "anchor_neighbor_max_dist": float(np.max(task.get("anchor_neighbor_distances", [0.0]))),
        "anchor_scope": str(task["anchor_scope"]),
        "anchor_distance_space": str(task["anchor_distance_space"]),
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
    base_solver_cfg = dict(cfg.get("segmented_tension", {}))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = None
        try:
            task = json.loads(line)
            t0 = time.perf_counter()
            beta = np.asarray(task["beta6_rad"], dtype=float).reshape(6)
            theta_raw = beta_to_theta(beta)
            cache = model.build_cache(theta_raw)
            solver_cfg = dict(base_solver_cfg)
            solver_cfg["anchor_tension_n"] = task["anchor_tension_n"]
            solver_cfg["w_anchor"] = float(task["w_anchor"])
            res = solve_tensions_segmented(model, cache, solver_cfg)
            elapsed_s = float(time.perf_counter() - t0)
            out = {
                "ok": bool(res.success and np.isfinite(res.T_base_12).all()),
                "sample_id": int(task["sample_id"]),
                "source_sample_id": int(task["source_sample_id"]),
                "tension_n": np.asarray(res.T_base_12, dtype=float).reshape(12).tolist(),
                "meta": _solver_meta(res, task=task, elapsed_s=elapsed_s),
            }
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001
            out = {
                "ok": False,
                "sample_id": int(task.get("sample_id", -1)) if isinstance(task, dict) else -1,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def run_relabel(args: argparse.Namespace) -> None:
    t_start = time.time()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(args.dataset)
    meta = pd.read_parquet(args.meta)
    if dataset["sample_id"].tolist() != meta["sample_id"].tolist():
        dataset = dataset.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        meta = meta.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if dataset["sample_id"].tolist() != meta["sample_id"].tolist():
        raise SystemExit("dataset/meta sample_id mismatch")

    selected_meta = select_stratified(meta, num_samples=int(args.num_samples), seed=int(args.seed))
    source_ids = selected_meta["source_sample_id"].to_numpy(dtype=int)
    selected_dataset = dataset.set_index("sample_id").loc[source_ids].reset_index(drop=True)
    selected_dataset["sample_id"] = np.arange(len(selected_dataset), dtype=int)

    pool_tension = dataset[TENSION_COLS].to_numpy(dtype=float)
    beta_scales = beta_scales_from_config(cfg)
    if str(args.component_scope) == "selected_branch_graph":
        anchors = build_selected_branch_graph_anchors(
            pool_meta=meta,
            pool_tension=pool_tension,
            selected_meta=selected_meta,
            k_neighbors=int(args.anchor_k),
            distance_space=str(args.distance_space),
            no_cross_branch_anchor=bool(args.no_cross_branch_anchor),
            beta_scales=beta_scales,
        )
    else:
        anchors = build_same_component_beta_knn_anchors(
            pool_meta=meta,
            pool_tension=pool_tension,
            selected_meta=selected_meta,
            k_neighbors=int(args.anchor_k),
            beta_scales=beta_scales,
        )
    task_beta_cols = _beta_cols_for_distance_space(selected_meta, str(args.distance_space))
    task_beta_scale = 1.0
    if str(args.distance_space) == "effective_beta":
        theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", 1.0))
        if abs(theta_sign) < 1.0e-12:
            raise ValueError("kinematics.theta_sign must be nonzero when using effective_beta relabel")
        task_beta_scale = 1.0 / theta_sign

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(args.config)]
    workers = start_workers(cmd, int(args.workers))
    results: dict[int, dict] = {}
    in_flight = 0
    next_idx = 0
    max_in_flight = max(1, int(args.workers) * 4)
    pbar = tqdm(total=len(selected_meta), desc="relabel")
    try:
        while len(results) < len(selected_meta):
            dead = [w for w in workers if w.proc.poll() is not None]
            if dead:
                err_lines = []
                for w in dead:
                    while not w.err_q.empty():
                        err_lines.append(w.err_q.get())
                raise SystemExit(f"worker exited during relabel: {err_lines[:8] if err_lines else ['no stderr']}")

            while next_idx < len(selected_meta) and in_flight < max_in_flight:
                mrow = selected_meta.iloc[next_idx]
                anchor = anchors[next_idx]
                task = {
                    "sample_id": int(mrow["sample_id"]),
                    "source_sample_id": int(mrow["source_sample_id"]),
                    "beta6_rad": [float(mrow[c]) * task_beta_scale for c in task_beta_cols],
                    "anchor_tension_n": anchor.anchor_tension_n.tolist(),
                    "anchor_neighbor_sample_ids": anchor.neighbor_sample_ids,
                    "anchor_neighbor_distances": anchor.neighbor_distances,
                    "anchor_k": int(args.anchor_k),
                    "w_anchor": float(args.w_anchor),
                    "anchor_scope": str(args.component_scope),
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
            if not bool(res.get("ok", False)):
                can_keep_infeasible = bool(args.accept_infeasible) and "tension_n" in res and "meta" in res
                if not can_keep_infeasible:
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
        "mode": "anchor_relabel",
        "source_dataset": str(args.dataset),
        "source_meta": str(args.meta),
        "accepted": int(len(out_dataset)),
        "accepted_infeasible": int(sum(1 for r in results.values() if not bool(r.get("ok", False)))),
        "elapsed_s": elapsed_s,
        "sec_per_accepted": elapsed_s / max(int(len(out_dataset)), 1),
        "component_counts": {str(k): int(v) for k, v in out_meta["source_component"].value_counts().sort_index().to_dict().items()},
        "anchor_k": int(args.anchor_k),
        "w_anchor": float(args.w_anchor),
        "component_scope": str(args.component_scope),
        "distance_space": str(args.distance_space),
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--component-scope", choices=["same_component", "selected_branch_graph"], default="same_component")
    ap.add_argument("--distance-space", choices=["beta", "effective_beta"], default="beta")
    ap.add_argument("--anchor-k", type=int, default=16)
    ap.add_argument("--w-anchor", type=float, default=10.0)
    ap.add_argument("--w-anchor-sweep", default="", help="Comma-separated values; handled by pipeline wrapper.")
    ap.add_argument("--no-cross-branch-anchor", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--accept-infeasible", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args)
        return
    if args.dataset is None or args.meta is None or args.out_dir is None:
        raise SystemExit("--dataset, --meta, and --out-dir are required outside --worker")
    run_relabel(args)


if __name__ == "__main__":
    main()
