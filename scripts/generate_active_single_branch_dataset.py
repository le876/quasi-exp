#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.kinematics import forward_kinematics  # noqa: E402
from quasi_exp.model.quasi_static import QuasiStaticModel  # noqa: E402
from quasi_exp.model.sampling import (  # noqa: E402
    beta_to_theta,
    build_mixed_beta_tasks,
    effective_beta_from_theta,
)
from quasi_exp.opt.pso_inverse import solve_inverse_joint_pso  # noqa: E402
from quasi_exp.opt.tension_labeler import solve_tension_label  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
SOURCE_BETA_COLS = [f"source_beta_{i}_rad" for i in range(1, 7)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
TARGET_COLS = ["target_x_m", "target_y_m", "target_z_m"]
ACHIEVED_COLS = ["achieved_x_m", "achieved_y_m", "achieved_z_m"]
POLICY_VERSION = "active_xyz_distal_v3_graph"


class _ActiveCandidate(NamedTuple):
    row_index: int
    target_id: int
    candidate_id: int
    unary_score: float
    beta: np.ndarray
    theta: np.ndarray
    tension: np.ndarray


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _beta_bounds(beta_ranges_rad: dict[str, Any]) -> np.ndarray:
    bounds = np.zeros((6, 2), dtype=float)
    for i in range(6):
        key = f"beta{i + 1}"
        if key not in beta_ranges_rad:
            raise ValueError(f"sampling.beta_ranges_rad missing {key}")
        lo, hi = beta_ranges_rad[key]
        bounds[i, 0] = float(lo)
        bounds[i, 1] = float(hi)
    return bounds


def _beta_limit_scales() -> np.ndarray:
    return np.asarray([np.deg2rad(5.0)] * 4 + [np.deg2rad(10.0)] * 2, dtype=float)


def _workspace_bins(
    xyz: np.ndarray,
    *,
    radius_bins: int,
    z_bins: int,
    angle_bins: int,
) -> list[tuple[int, int, int]]:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    radius = np.linalg.norm(xyz, axis=1)
    z = xyz[:, 2]
    angle = np.arctan2(xyz[:, 2], xyz[:, 1])

    def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
        n_bins = max(1, int(n_bins))
        if values.size == 0 or n_bins == 1:
            return np.zeros(values.size, dtype=int)
        qs = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        return np.searchsorted(qs, values, side="right").astype(int)

    rb = quantile_bin(radius, radius_bins)
    zb = quantile_bin(z, z_bins)
    ab = np.floor(((angle + np.pi) / (2.0 * np.pi)) * max(1, int(angle_bins))).astype(int)
    ab = np.clip(ab, 0, max(1, int(angle_bins)) - 1)
    return [(int(a), int(b), int(c)) for a, b, c in zip(rb.tolist(), zb.tolist(), ab.tolist())]


def build_reachable_target_pool(
    *,
    beta_ranges_rad: dict[str, Any],
    pool_size: int,
    rng_seed: int,
    workspace_xyz_fn: Callable[[np.ndarray], np.ndarray],
    components: dict[str, float] | None = None,
    distal_biased_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    mixed_cfg = {
        "components": components or {"sobol_full": 0.5, "distal_biased": 0.5},
        "distal_biased": distal_biased_cfg or {"proximal_abs_ratio_max": 0.45, "distal_abs_ratio_min": 0.55},
    }
    tasks, plan = build_mixed_beta_tasks(
        beta_ranges_rad=beta_ranges_rad,
        num_samples=int(pool_size),
        mixed_cfg=mixed_cfg,
        rng_seed=int(rng_seed),
    )
    beta = np.asarray([task["beta6_rad"] for task in tasks], dtype=float).reshape(-1, 6)
    xyz = np.asarray(workspace_xyz_fn(beta), dtype=float).reshape(-1, 3)
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        row = {
            "source_pool_index": int(idx),
            "source_component": str(task.get("source_component", "unknown")),
            "source_component_idx": int(task.get("source_component_idx", idx)),
            "target_x_m": float(xyz[idx, 0]),
            "target_y_m": float(xyz[idx, 1]),
            "target_z_m": float(xyz[idx, 2]),
            "radius_m": float(np.linalg.norm(xyz[idx])),
        }
        for i in range(6):
            row[f"source_beta_{i + 1}_rad"] = float(beta[idx, i])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["component_counts"] = plan.component_counts
    return out


def select_workspace_balanced_targets(
    pool: pd.DataFrame,
    *,
    num_targets: int,
    radius_bins: int = 5,
    z_bins: int = 5,
    angle_bins: int = 8,
    seed: int = 20260608,
) -> pd.DataFrame:
    if int(num_targets) <= 0:
        raise ValueError("num_targets must be > 0")
    if pool.empty:
        raise ValueError("target pool is empty")
    work = pool.copy().reset_index(drop=True)
    xyz = work[TARGET_COLS].to_numpy(dtype=float)
    bins = _workspace_bins(xyz, radius_bins=radius_bins, z_bins=z_bins, angle_bins=angle_bins)
    work["_workspace_balance_bin"] = [f"{a},{b},{c}" for a, b, c in bins]
    work["_rand"] = np.random.default_rng(int(seed)).random(len(work))
    ordered = work.sort_values(["_workspace_balance_bin", "_rand"], ascending=[True, True], kind="mergesort")
    buckets = {key: block.index.to_list() for key, block in ordered.groupby("_workspace_balance_bin", sort=True)}
    keys = sorted(buckets)
    chosen: list[int] = []
    cursor = 0
    while len(chosen) < min(int(num_targets), len(work)) and keys:
        key = keys[cursor % len(keys)]
        vals = buckets[key]
        if vals:
            chosen.append(int(vals.pop(0)))
        if not vals:
            keys.remove(key)
            if not keys:
                break
            cursor = cursor % len(keys)
        else:
            cursor += 1
    out = work.loc[chosen].copy().reset_index(drop=True)
    out = out.drop(columns=[c for c in ["_workspace_balance_bin", "_rand"] if c in out.columns])
    out["active_target_id"] = np.arange(len(out), dtype=int)
    return out


def candidate_to_rows(
    *,
    sample_id: int,
    active_target_id: int,
    active_candidate_id: int,
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    beta6_rad: np.ndarray,
    theta30_rad: np.ndarray,
    tension12_n: np.ndarray,
    source_beta6_rad: np.ndarray,
    source_component: str,
    candidate_kind: str,
    xyz_err_m: float,
    rms_rnorm: float,
    mean_rnorm2: float,
    best_cost: float,
    iters_used: int,
    evals: int,
    pso_seed: int,
    elapsed_s: float,
    case_flag_12: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    achieved = np.asarray(achieved_xyz_m, dtype=float).reshape(3)
    beta = np.asarray(beta6_rad, dtype=float).reshape(6)
    theta = np.asarray(theta30_rad, dtype=float).reshape(30)
    tension = np.asarray(tension12_n, dtype=float).reshape(12)
    source_beta = np.asarray(source_beta6_rad, dtype=float).reshape(6)
    effective_beta = effective_beta_from_theta(theta.reshape(1, 30))[0]
    row: dict[str, Any] = {
        "sample_id": int(sample_id),
        "x_m": float(target[0]),
        "y_m": float(target[1]),
        "z_m": float(target[2]),
    }
    for i, value in enumerate(theta):
        row[f"theta_{i + 1}_rad"] = float(value)
    for j, value in enumerate(tension):
        row[f"tension_{j + 1}_n"] = float(value)
    meta: dict[str, Any] = {
        "sample_id": int(sample_id),
        "active_target_id": int(active_target_id),
        "active_candidate_id": int(active_candidate_id),
        "candidate_kind": str(candidate_kind),
        "source_component": str(source_component),
        "target_x_m": float(target[0]),
        "target_y_m": float(target[1]),
        "target_z_m": float(target[2]),
        "achieved_x_m": float(achieved[0]),
        "achieved_y_m": float(achieved[1]),
        "achieved_z_m": float(achieved[2]),
        "xyz_err_m": float(xyz_err_m),
        "rms_rnorm": float(rms_rnorm),
        "mean_rnorm2": float(mean_rnorm2),
        "max_tension": float(np.max(tension)),
        "best_cost": float(best_cost),
        "iters_used": int(iters_used),
        "evals": int(evals),
        "pso_seed": int(pso_seed),
        "elapsed_s": float(elapsed_s),
        "segmented_success": bool(np.isfinite(tension).all() and float(rms_rnorm) <= 0.06),
        "effective_beta_source": "theta_odd_even_mean",
    }
    for i, value in enumerate(beta):
        meta[f"beta{i + 1}_rad"] = float(value)
    for i, value in enumerate(source_beta):
        meta[f"source_beta_{i + 1}_rad"] = float(value)
    for i, value in enumerate(effective_beta):
        meta[f"effective_beta_{i + 1}_rad"] = float(value)
    flags = np.zeros(12, dtype=int) if case_flag_12 is None else np.asarray(case_flag_12, dtype=int).reshape(-1)
    for j in range(12):
        meta[f"case_{j + 1}"] = int(flags[j]) if j < flags.size else 0
    return row, meta


def _active_unary_score(meta_row: pd.Series) -> float:
    if "active_unary_score" in meta_row and np.isfinite(float(meta_row["active_unary_score"])):
        return float(meta_row["active_unary_score"])
    beta = meta_row[EFFECTIVE_BETA_COLS].to_numpy(dtype=float)
    b = beta / _beta_limit_scales()
    g1 = float(np.sqrt(np.mean(np.square(b[[0, 1]]))))
    g2 = float(np.sqrt(np.mean(np.square(b[[2, 3]]))))
    g3 = float(np.sqrt(np.mean(np.square(b[[4, 5]]))))
    xyz_err = float(meta_row.get("xyz_err_m", 0.0))
    max_t = float(meta_row.get("max_tension", 0.0))
    rms = float(meta_row.get("rms_rnorm", 0.0))
    return float(g3 - 0.65 * g1 - 0.45 * g2 - 0.6 * (xyz_err / 0.004) - 0.25 * (max_t / 2000.0) - 0.4 * (rms / 0.06))


def _pairwise_cost(
    a: _ActiveCandidate,
    b: _ActiveCandidate,
    *,
    lambda_theta: float,
    lambda_beta: float,
    lambda_tension: float,
) -> float:
    theta_rms_deg = float(np.sqrt(np.mean(np.square(a.theta - b.theta))) * (180.0 / np.pi))
    beta_dist = float(np.linalg.norm((a.beta - b.beta) / _beta_limit_scales()))
    tension_mae = float(np.mean(np.abs(a.tension - b.tension)))
    return float(
        lambda_theta * (theta_rms_deg / 3.0) ** 2
        + lambda_beta * beta_dist**2
        + lambda_tension * (tension_mae / 100.0) ** 2
    )


def _target_edges(target_xyz: np.ndarray, *, k_neighbors: int, radius_m: float) -> dict[int, list[int]]:
    n = int(len(target_xyz))
    edges: dict[int, list[int]] = {i: [] for i in range(n)}
    if n <= 1:
        return edges
    k = min(max(1, int(k_neighbors)) + 1, n)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(target_xyz)
    dist, idx = nn.kneighbors(target_xyz)
    seen: set[tuple[int, int]] = set()
    for row in range(n):
        for d, col in zip(dist[row, 1:].tolist(), idx[row, 1:].tolist()):
            if float(d) > float(radius_m):
                continue
            a, b = (row, int(col)) if row < int(col) else (int(col), row)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            edges[a].append(b)
            edges[b].append(a)
    return edges


def _component_ids(n: int, edges: dict[int, list[int]]) -> dict[int, int]:
    out: dict[int, int] = {}
    comp = 0
    for start in range(int(n)):
        if start in out:
            continue
        out[start] = comp
        stack = [start]
        while stack:
            cur = stack.pop()
            for neigh in edges.get(cur, []):
                if neigh in out:
                    continue
                out[neigh] = comp
                stack.append(neigh)
        comp += 1
    return out


def select_global_active_branch_frames(
    candidates_dataset: pd.DataFrame,
    candidates_meta: pd.DataFrame,
    *,
    k_neighbors: int = 16,
    radius_m: float = 0.02,
    max_icm_iters: int = 10,
    lambda_theta: float = 2.0,
    lambda_beta: float = 1.0,
    lambda_tension: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if candidates_dataset.empty or candidates_meta.empty:
        raise ValueError("candidate frames must not be empty")
    df = candidates_dataset.merge(candidates_meta, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_meta"))
    if "active_target_id" not in df.columns or "active_candidate_id" not in df.columns:
        raise ValueError("candidates_meta must contain active_target_id and active_candidate_id")
    target_ids = sorted(int(v) for v in df["active_target_id"].dropna().unique().tolist())
    target_to_pos = {target_id: pos for pos, target_id in enumerate(target_ids)}
    candidates_by_target: dict[int, list[_ActiveCandidate]] = {}
    target_xyz = np.zeros((len(target_ids), 3), dtype=float)
    for target_id in target_ids:
        block = df[df["active_target_id"].astype(int) == target_id].copy()
        if block.empty:
            continue
        pos = target_to_pos[target_id]
        target_xyz[pos] = block[["x_m", "y_m", "z_m"]].iloc[0].to_numpy(dtype=float)
        local: list[_ActiveCandidate] = []
        for row_idx, row in block.iterrows():
            local.append(
                _ActiveCandidate(
                    row_index=int(row_idx),
                    target_id=int(target_id),
                    candidate_id=int(row["active_candidate_id"]),
                    unary_score=_active_unary_score(row),
                    beta=row[EFFECTIVE_BETA_COLS].to_numpy(dtype=float),
                    theta=row[THETA_COLS].to_numpy(dtype=float),
                    tension=row[TENSION_COLS].to_numpy(dtype=float),
                )
            )
        candidates_by_target[pos] = local
    edges = _target_edges(target_xyz, k_neighbors=k_neighbors, radius_m=radius_m)
    selected: dict[int, _ActiveCandidate] = {
        pos: max(cands, key=lambda c: (c.unary_score, -c.candidate_id))
        for pos, cands in candidates_by_target.items()
    }
    iterations = 0
    for _ in range(max(0, int(max_icm_iters))):
        changed = False
        for pos in sorted(candidates_by_target):
            best = selected[pos]
            best_energy = -float("inf")
            for cand in candidates_by_target[pos]:
                energy = float(cand.unary_score)
                for neigh in edges.get(pos, []):
                    if neigh not in selected:
                        continue
                    energy -= _pairwise_cost(
                        cand,
                        selected[neigh],
                        lambda_theta=lambda_theta,
                        lambda_beta=lambda_beta,
                        lambda_tension=lambda_tension,
                    )
                if energy > best_energy:
                    best_energy = energy
                    best = cand
            if best.candidate_id != selected[pos].candidate_id:
                selected[pos] = best
                changed = True
        iterations += 1
        if not changed:
            break
    comp_by_pos = _component_ids(len(target_ids), edges)
    keep_indices = [selected[pos].row_index for pos in sorted(selected)]
    keep_sample_ids = df.loc[keep_indices, "sample_id"].astype(int).tolist()
    out_dataset = candidates_dataset[candidates_dataset["sample_id"].isin(keep_sample_ids)].copy()
    out_meta = candidates_meta[candidates_meta["sample_id"].isin(keep_sample_ids)].copy()
    out_dataset = out_dataset.set_index("sample_id").loc[keep_sample_ids].reset_index()
    out_meta = out_meta.set_index("sample_id").loc[keep_sample_ids].reset_index()
    out_meta["active_branch_selected"] = True
    out_meta["active_branch_policy_version"] = POLICY_VERSION
    out_meta["active_branch_score"] = [
        float(selected[target_to_pos[int(target_id)]].unary_score)
        for target_id in out_meta["active_target_id"].astype(int).tolist()
    ]
    out_meta["active_graph_component_id"] = [
        int(comp_by_pos[target_to_pos[int(target_id)]])
        for target_id in out_meta["active_target_id"].astype(int).tolist()
    ]
    out_meta["graph_component_id"] = out_meta["active_graph_component_id"].astype(int)
    out_dataset = out_dataset.reset_index(drop=True)
    out_meta = out_meta.reset_index(drop=True)
    out_dataset["sample_id"] = np.arange(len(out_dataset), dtype=int)
    out_meta["sample_id"] = np.arange(len(out_meta), dtype=int)
    summary = {
        "mode": "active_xyz_global_branch_selection",
        "branch_policy_version": POLICY_VERSION,
        "candidate_rows": int(len(candidates_dataset)),
        "selected_targets": int(len(out_dataset)),
        "target_graph_edges": int(sum(len(v) for v in edges.values()) // 2),
        "target_graph_components": int(len(set(comp_by_pos.values()))),
        "icm_iterations": int(iterations),
        "k_neighbors": int(k_neighbors),
        "radius_m": float(radius_m),
    }
    return out_dataset, out_meta, summary


def active_acceptance_gate_passed(
    gate: dict[str, Any],
    *,
    theta_target_deg: float = 3.0,
    tension_target_n: float = 100.0,
    beta_close_tension_target_n: float = 50.0,
    multi_branch_ratio_target: float = 0.25,
    xyz_err_p95_target_m: float = 0.004,
) -> bool:
    return bool(gate.get("hard_gate_passed", False)) and (
        float(gate.get("xyz_err_m_p95", float("inf"))) <= float(xyz_err_p95_target_m)
        and float(gate.get("all10_theta_p95_deg", float("inf"))) <= float(theta_target_deg)
        and float(gate.get("all10_tension_p95_n", float("inf"))) <= float(tension_target_n)
        and float(gate.get("beta_close_tension_p95_n", float("inf"))) <= float(beta_close_tension_target_n)
        and float(gate.get("multi_branch_ball_ratio", float("inf"))) <= float(multi_branch_ratio_target)
    )


def should_stop_before_relabel(gate: dict[str, Any], *, theta_stop_deg: float = 5.0) -> bool:
    return bool(gate.get("hard_gate_passed", False)) and float(gate.get("all10_theta_p95_deg", float("inf"))) > float(theta_stop_deg)


def _workspace_xyz_from_beta_rows(beta_rows: np.ndarray, inputs: Any, theta_sign: float) -> np.ndarray:
    beta_arr = np.asarray(beta_rows, dtype=float).reshape(-1, 6)
    xyz = np.zeros((beta_arr.shape[0], 3), dtype=float)
    for i, beta in enumerate(beta_arr):
        theta_raw = beta_to_theta(beta)
        p_xyz, _ = forward_kinematics(theta_raw, inputs.lengths_m, inputs.p_end_local_m, theta_sign=float(theta_sign))
        xyz[i] = np.asarray(p_xyz, dtype=float).reshape(3)
    return xyz


def _label_beta_candidate(
    *,
    sample_id: int,
    active_target_id: int,
    active_candidate_id: int,
    target_xyz: np.ndarray,
    beta: np.ndarray,
    source_beta: np.ndarray,
    source_component: str,
    candidate_kind: str,
    model: QuasiStaticModel,
    inputs: Any,
    pso_cfg: dict[str, Any],
    pso_seed: int,
    rms_thresh: float,
    best_cost: float = 0.0,
    iters_used: int = 0,
    evals: int = 0,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    t0 = time.time()
    beta = np.asarray(beta, dtype=float).reshape(6)
    theta_raw = beta_to_theta(beta)
    theta = theta_raw * float(model.theta_sign)
    achieved_xyz, _ = forward_kinematics(theta_raw, inputs.lengths_m, inputs.p_end_local_m, theta_sign=model.theta_sign)
    cache = model.build_cache(theta_raw)
    label = solve_tension_label(model=model, cache=cache, pso_cfg=pso_cfg, pso_seed=int(pso_seed), rms_thresh=float(rms_thresh))
    row, meta = candidate_to_rows(
        sample_id=int(sample_id),
        active_target_id=int(active_target_id),
        active_candidate_id=int(active_candidate_id),
        target_xyz_m=np.asarray(target_xyz, dtype=float).reshape(3),
        achieved_xyz_m=np.asarray(achieved_xyz, dtype=float).reshape(3),
        beta6_rad=beta,
        theta30_rad=theta,
        tension12_n=label.T_base_12,
        source_beta6_rad=np.asarray(source_beta, dtype=float).reshape(6),
        source_component=str(source_component),
        candidate_kind=str(candidate_kind),
        xyz_err_m=float(np.linalg.norm(np.asarray(achieved_xyz, dtype=float).reshape(3) - np.asarray(target_xyz, dtype=float).reshape(3))),
        rms_rnorm=float(label.meta.get("rms_rnorm", float("inf"))),
        mean_rnorm2=float(label.meta.get("mean_rnorm2", float("inf"))),
        best_cost=float(best_cost if best_cost != 0.0 else label.meta.get("best_cost", 0.0)),
        iters_used=int(iters_used + int(label.meta.get("iters_used", 0))),
        evals=int(evals + int(label.meta.get("evals", 0))),
        pso_seed=int(pso_seed),
        elapsed_s=float(time.time() - t0),
        case_flag_12=cache.case_flag_12,
    )
    ok = bool(label.ok and np.isfinite(label.T_base_12).all() and float(meta["rms_rnorm"]) <= float(rms_thresh))
    return ok, row, meta


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _diagnose_and_gate(dataset: pd.DataFrame, meta: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    continuity_mod = _load_script("eval_branch_aware_continuity")
    clustering_mod = _load_script("eval_branch_clustering")
    oracle_mod = _load_script("eval_oracle_floor")
    continuity = continuity_mod.evaluate_frames(dataset, meta)
    clustering = clustering_mod.evaluate_frames(dataset, meta)
    oracle = oracle_mod.evaluate_frames(dataset, meta, branch_column="active_graph_component_id")
    tension = dataset[TENSION_COLS].to_numpy(dtype=float)
    xyz_err = meta["xyz_err_m"].to_numpy(dtype=float) if "xyz_err_m" in meta.columns else np.full(len(meta), np.inf)
    rms = meta["rms_rnorm"].to_numpy(dtype=float) if "rms_rnorm" in meta.columns else np.full(len(meta), np.inf)
    hard_gate = bool(
        np.isfinite(tension).all()
        and (tension >= 0.0).all()
        and (tension <= 2000.0).all()
        and float(np.percentile(rms, 95)) <= 0.06
        and float(np.percentile(xyz_err, 95)) <= 0.004
    )
    all10 = continuity.get("groups", {}).get("all_xyz_<=10mm", {})
    beta_close = continuity.get("groups", {}).get("xyz_<=10mm_beta_close", {})
    gate = {
        "hard_gate_passed": hard_gate,
        "xyz_err_m_p95": float(np.percentile(xyz_err, 95)) if xyz_err.size else float("inf"),
        "xyz_err_m_max": float(np.max(xyz_err)) if xyz_err.size else float("inf"),
        "rms_rnorm_q95": float(np.percentile(rms, 95)) if rms.size else float("inf"),
        "all10_theta_p95_deg": float(all10.get("theta_rms_deg_p95", float("inf"))),
        "all10_tension_p95_n": float(all10.get("tension_mae_n_p95", float("inf"))),
        "beta_close_tension_p95_n": float(beta_close.get("tension_mae_n_p95", float("inf"))),
        "multi_branch_ball_ratio": float(clustering.get("multi_branch_ball_ratio", float("inf"))),
        "xyz_nn_tension_mae_n": float(oracle.get("xyz_nn_oracle", {}).get("tension_mae_n", float("inf"))),
        "beta_nn_tension_mae_n": float(oracle.get("beta_nn_oracle", {}).get("tension_mae_n", float("inf"))),
    }
    gate["acceptance_gate_passed"] = active_acceptance_gate_passed(gate)
    gate["stop_before_relabel"] = should_stop_before_relabel(gate)
    payload = {
        "gate": gate,
        "continuity": continuity,
        "branch_clustering": {k: v for k, v in clustering.items() if k != "per_ball"},
        "oracle_floor": oracle,
    }
    (out_dir / "active_diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def generate_active_dataset(
    *,
    cfg: dict[str, Any],
    num_targets: int,
    candidates_per_target: int,
    out_dir: Path,
    target_pool_size: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)
    sampling_cfg = cfg.get("sampling", {})
    active_cfg = cfg.get("active_generation", {})
    beta_ranges = sampling_cfg["beta_ranges_rad"]
    rng_seed = int(sampling_cfg.get("rng_seed", active_cfg.get("rng_seed", 20260608)))
    pool_size = int(target_pool_size or active_cfg.get("target_pool_size", max(50000, int(num_targets) * 20)))
    pool = build_reachable_target_pool(
        beta_ranges_rad=beta_ranges,
        pool_size=pool_size,
        rng_seed=rng_seed,
        workspace_xyz_fn=lambda beta_rows: _workspace_xyz_from_beta_rows(beta_rows, inputs, model.theta_sign),
        components=dict(active_cfg.get("target_components", {"sobol_full": 0.5, "distal_biased": 0.5})),
        distal_biased_cfg=dict(active_cfg.get("distal_biased", {"proximal_abs_ratio_max": 0.45, "distal_abs_ratio_min": 0.55})),
    )
    target_candidates = select_workspace_balanced_targets(
        pool,
        num_targets=int(num_targets) * int(active_cfg.get("target_oversample", 2)),
        radius_bins=int(active_cfg.get("radius_bins", 5)),
        z_bins=int(active_cfg.get("z_bins", 5)),
        angle_bins=int(active_cfg.get("angle_bins", 8)),
        seed=rng_seed + 17,
    )
    pso_cfg = cfg["pso"]
    inverse_cfg = dict(cfg.get("inverse_pso", {}))
    inverse_cfg["canonical_mode"] = "none"
    if "joint_preference" in inverse_cfg and isinstance(inverse_cfg["joint_preference"], dict):
        inverse_cfg["joint_preference"] = {**inverse_cfg["joint_preference"], "enabled": False}
    rms_thresh = float(cfg.get("dataset", {}).get("rms_rnorm_threshold", 0.06))
    xyz_thresh = float(cfg.get("dataset", {}).get("xyz_err_threshold_m", active_cfg.get("xyz_err_threshold_m", 0.006)))
    min_candidates = int(active_cfg.get("min_candidates_per_target", 2))
    warm_candidates = int(active_cfg.get("warm_start_candidates", 3))
    perturb_sigma = float(active_cfg.get("warm_start_perturb_sigma_rad", 0.01))
    rng = np.random.default_rng(rng_seed + 101)
    bounds = _beta_bounds(beta_ranges)

    cand_rows: list[dict[str, Any]] = []
    cand_meta: list[dict[str, Any]] = []
    accepted_targets = 0
    tried_targets = 0
    sample_id = 0
    t0 = time.time()
    for target_row in tqdm(target_candidates.itertuples(index=False), total=len(target_candidates), desc="active_targets"):
        if accepted_targets >= int(num_targets):
            break
        tried_targets += 1
        source_beta = np.asarray([getattr(target_row, f"source_beta_{i}_rad") for i in range(1, 7)], dtype=float)
        target_xyz = np.asarray([target_row.target_x_m, target_row.target_y_m, target_row.target_z_m], dtype=float)
        source_component = str(getattr(target_row, "source_component", "unknown"))
        local_rows: list[dict[str, Any]] = []
        local_meta: list[dict[str, Any]] = []
        ok, row, meta = _label_beta_candidate(
            sample_id=sample_id,
            active_target_id=accepted_targets,
            active_candidate_id=0,
            target_xyz=target_xyz,
            beta=source_beta,
            source_beta=source_beta,
            source_component=source_component,
            candidate_kind="source_beta",
            model=model,
            inputs=inputs,
            pso_cfg=pso_cfg,
            pso_seed=int(pso_cfg.get("rng_seed", 0)) + sample_id,
            rms_thresh=rms_thresh,
        )
        if ok and float(meta["xyz_err_m"]) <= xyz_thresh:
            local_rows.append(row)
            local_meta.append(meta)
            sample_id += 1
        for cand_id in range(1, int(candidates_per_target)):
            pso_seed = int(pso_cfg.get("rng_seed", 0)) + 100000 * accepted_targets + cand_id * 997
            warm = None
            kind = "inverse_global"
            if cand_id <= warm_candidates:
                noise = rng.normal(0.0, perturb_sigma, size=6)
                warm = np.clip(source_beta + noise, bounds[:, 0], bounds[:, 1])
                kind = "inverse_warm"
            try:
                res = solve_inverse_joint_pso(
                    model=model,
                    inputs=inputs,
                    xyz_target_m=target_xyz,
                    beta_ranges_rad=beta_ranges,
                    inverse_pso_cfg=inverse_cfg,
                    tension_pso_cfg=pso_cfg,
                    rng_seed=pso_seed,
                    warm_start_beta=warm,
                    continuity_ref_beta=None,
                )
            except Exception:
                continue
            if (not np.isfinite(res.beta6_rad).all()) or float(res.xyz_err_m) > xyz_thresh:
                continue
            ok, row, meta = _label_beta_candidate(
                sample_id=sample_id,
                active_target_id=accepted_targets,
                active_candidate_id=cand_id,
                target_xyz=target_xyz,
                beta=res.beta6_rad,
                source_beta=source_beta,
                source_component=source_component,
                candidate_kind=kind,
                model=model,
                inputs=inputs,
                pso_cfg=pso_cfg,
                pso_seed=pso_seed + 55001,
                rms_thresh=rms_thresh,
                best_cost=res.best_cost,
                iters_used=res.iters_used,
                evals=res.evals,
            )
            if ok and float(meta["xyz_err_m"]) <= xyz_thresh:
                local_rows.append(row)
                local_meta.append(meta)
                sample_id += 1
        if len(local_rows) < min_candidates:
            continue
        cand_rows.extend(local_rows)
        cand_meta.extend(local_meta)
        accepted_targets += 1

    if accepted_targets < int(num_targets):
        raise RuntimeError(
            f"active generation accepted {accepted_targets}/{num_targets} targets; "
            f"increase target_pool_size/target_oversample or relax thresholds"
        )
    candidates_dataset = pd.DataFrame(cand_rows)
    candidates_meta = pd.DataFrame(cand_meta)
    selected_dataset, selected_meta, selection_summary = select_global_active_branch_frames(
        candidates_dataset,
        candidates_meta,
        k_neighbors=int(active_cfg.get("graph_k_neighbors", 16)),
        radius_m=float(active_cfg.get("graph_radius_m", 0.02)),
        max_icm_iters=int(active_cfg.get("graph_max_icm_iters", 10)),
        lambda_theta=float(active_cfg.get("lambda_theta", 2.0)),
        lambda_beta=float(active_cfg.get("lambda_beta", 1.0)),
        lambda_tension=float(active_cfg.get("lambda_tension", 0.2)),
    )
    candidates_dataset.to_parquet(out_dir / "candidates.parquet", index=False)
    candidates_meta.to_parquet(out_dir / "candidates_meta.parquet", index=False)
    selected_dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    selected_meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)
    diagnostics = _diagnose_and_gate(selected_dataset, selected_meta, out_dir / "diagnostics")
    report = {
        "mode": "active_xyz_single_branch_distal_v3",
        "num_targets_requested": int(num_targets),
        "targets_tried": int(tried_targets),
        "targets_accepted": int(accepted_targets),
        "candidate_rows": int(len(candidates_dataset)),
        "selected_rows": int(len(selected_dataset)),
        "candidates_per_selected_target_mean": float(len(candidates_dataset) / max(len(selected_dataset), 1)),
        "elapsed_s": float(time.time() - t0),
        "sec_per_selected_target": float((time.time() - t0) / max(len(selected_dataset), 1)),
        "target_pool_size": int(pool_size),
        "selection_summary": selection_summary,
        "gate": diagnostics["gate"],
    }
    (out_dir / "active_generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--num-targets", type=int, default=2000)
    ap.add_argument("--candidates-per-target", type=int, default=8)
    ap.add_argument("--target-pool-size", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = args.out_dir or Path(cfg.get("dataset", {}).get("out_dir", "data/active_xyz_2k_single_branch_distal_v3"))
    report = generate_active_dataset(
        cfg=cfg,
        num_targets=int(args.num_targets),
        candidates_per_target=int(args.candidates_per_target),
        out_dir=out_dir,
        target_pool_size=args.target_pool_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
