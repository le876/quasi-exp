#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.model.sampling import effective_beta_from_theta  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
POLICY_VERSION = "distal_v2_graph"


class BranchCandidate(NamedTuple):
    voxel_key: str
    branch_label: int
    sample_indices: np.ndarray
    branch_size: int
    unary_score: float
    beta_mean: np.ndarray
    theta_mean: np.ndarray
    tension_mean: np.ndarray


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def theta_cols(df: pd.DataFrame) -> list[str]:
    cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    if len(cols) != 30:
        raise ValueError(f"expected 30 theta columns, got {len(cols)}")
    return cols


def tension_cols(df: pd.DataFrame) -> list[str]:
    cols = _numbered_cols(list(df.columns), "tension_", "_n")
    if len(cols) != 12:
        raise ValueError(f"expected 12 tension columns, got {len(cols)}")
    return cols


def ensure_effective_beta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    missing = [c for c in EFFECTIVE_BETA_COLS if c not in out.columns]
    if missing:
        theta = dataset[theta_cols(dataset)].to_numpy(dtype=float)
        beta = effective_beta_from_theta(theta)
        for i, col in enumerate(EFFECTIVE_BETA_COLS):
            out[col] = beta[:, i]
        out["effective_beta_source"] = "theta_odd_even_mean"
    elif "effective_beta_source" not in out.columns:
        out["effective_beta_source"] = "existing"
    return out


def _voxel_info(xyz: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, list[str]]:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be > 0")
    vox = np.floor(np.asarray(xyz, dtype=float) / float(voxel_size_m)).astype(int)
    keys = [f"{int(a)},{int(b)},{int(c)}" for a, b, c in vox.tolist()]
    return vox, keys


def _beta_limit_scales() -> np.ndarray:
    return np.asarray([np.deg2rad(5.0)] * 4 + [np.deg2rad(10.0)] * 2, dtype=float)


def _branch_score(
    beta_mean: np.ndarray,
    *,
    max_tension: float,
    rms_rnorm: float,
    branch_size: int,
    w3: float,
    w2: float,
    w1: float,
    w_tension: float,
    w_rms: float,
    w_size: float,
) -> float:
    scale = _beta_limit_scales()
    b = np.asarray(beta_mean, dtype=float).reshape(6) / scale
    b1 = float(np.linalg.norm(b[[0, 1]]))
    b2 = float(np.linalg.norm(b[[2, 3]]))
    b3 = float(np.linalg.norm(b[[4, 5]]))
    return float(
        w3 * b3
        - w2 * b2
        - w1 * b1
        - w_tension * (float(max_tension) / 2000.0)
        - w_rms * (float(rms_rnorm) / 0.06)
        + w_size * np.log1p(max(1, int(branch_size)))
    )


def _merge_dataset_meta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in dataset.columns or "sample_id" not in meta.columns:
        raise ValueError("dataset and meta must contain sample_id")
    meta_eff = ensure_effective_beta(dataset, meta)
    keep = ["sample_id"] + [c for c in meta_eff.columns if c != "sample_id"]
    df = dataset.merge(meta_eff[keep], on="sample_id", how="left", validate="one_to_one")
    if "source_component" not in df.columns:
        df["source_component"] = "unknown"
    if "rms_rnorm" not in df.columns:
        df["rms_rnorm"] = 0.0
    if "max_tension" not in df.columns:
        df["max_tension"] = df[tension_cols(df)].max(axis=1)
    return df


def _hard_mask(df: pd.DataFrame) -> np.ndarray:
    t = df[tension_cols(df)].to_numpy(dtype=float)
    mask = np.isfinite(t).all(axis=1) & (t >= 0.0).all(axis=1) & (t <= 2000.0).all(axis=1)
    mask &= df["rms_rnorm"].astype(float).to_numpy() <= 0.06
    mask &= df["max_tension"].astype(float).to_numpy() <= 2000.0
    for col in ["segmented_success", "canonical_success", "integrated_success"]:
        if col in df.columns:
            mask &= df[col].astype(bool).to_numpy()
    return mask


def _build_candidates(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float,
    beta_eps_norm: float,
    w3: float,
    w2: float,
    w1: float,
    w_tension: float,
    w_rms: float,
    w_size: float,
) -> tuple[pd.DataFrame, dict[str, list[BranchCandidate]], np.ndarray]:
    df = _merge_dataset_meta(dataset, meta)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    vox, keys = _voxel_info(xyz, voxel_size_m)
    out = df.copy()
    out["workspace_voxel_x"] = vox[:, 0]
    out["workspace_voxel_y"] = vox[:, 1]
    out["workspace_voxel_z"] = vox[:, 2]
    out["workspace_voxel_key"] = keys
    out["source_sample_id"] = out["sample_id"].astype(int)

    beta = out[EFFECTIVE_BETA_COLS].to_numpy(dtype=float)
    beta_norm = beta / _beta_limit_scales().reshape(1, 6)
    theta = out[theta_cols(out)].to_numpy(dtype=float)
    tension = out[tension_cols(out)].to_numpy(dtype=float)

    candidates_by_voxel: dict[str, list[BranchCandidate]] = {}
    hard = _hard_mask(out)
    for voxel_key, block in out.groupby("workspace_voxel_key", sort=True):
        idx = block.index.to_numpy(dtype=int)
        idx = idx[hard[idx]]
        if idx.size == 0:
            continue
        if idx.size == 1:
            labels = np.asarray([0], dtype=int)
        else:
            labels = DBSCAN(eps=float(beta_eps_norm), min_samples=1).fit_predict(beta_norm[idx])
        for row_idx, label in zip(idx.tolist(), labels.tolist()):
            out.loc[row_idx, "branch_label_in_voxel"] = int(label)
        cand: list[BranchCandidate] = []
        for label in sorted(set(int(v) for v in labels.tolist())):
            local = idx[labels == label]
            beta_mean = beta[local].mean(axis=0)
            theta_mean = theta[local].mean(axis=0)
            tension_mean = tension[local].mean(axis=0)
            unary = _branch_score(
                beta_mean,
                max_tension=float(out.loc[local, "max_tension"].astype(float).mean()),
                rms_rnorm=float(out.loc[local, "rms_rnorm"].astype(float).mean()),
                branch_size=int(local.size),
                w3=w3,
                w2=w2,
                w1=w1,
                w_tension=w_tension,
                w_rms=w_rms,
                w_size=w_size,
            )
            cand.append(
                BranchCandidate(
                    voxel_key=str(voxel_key),
                    branch_label=int(label),
                    sample_indices=local.copy(),
                    branch_size=int(local.size),
                    unary_score=float(unary),
                    beta_mean=beta_mean,
                    theta_mean=theta_mean,
                    tension_mean=tension_mean,
                )
            )
        candidates_by_voxel[str(voxel_key)] = cand
    return out, candidates_by_voxel, vox


def _graph_edges(voxel_keys: list[str], voxel_size_m: float, ball_radius_m: float) -> dict[str, list[str]]:
    coords = {key: np.asarray([int(v) for v in key.split(",")], dtype=float) for key in voxel_keys}
    edges: dict[str, list[str]] = {key: [] for key in voxel_keys}
    coord_to_key = {tuple(int(v) for v in value.astype(int).tolist()): key for key, value in coords.items()}
    radius_vox = int(np.ceil(float(ball_radius_m) / max(float(voxel_size_m), 1.0e-12)))
    offsets = []
    for dx in range(-radius_vox, radius_vox + 1):
        for dy in range(-radius_vox, radius_vox + 1):
            for dz in range(-radius_vox, radius_vox + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offset = np.asarray([dx, dy, dz], dtype=float)
                if float(np.linalg.norm(offset * float(voxel_size_m))) <= float(ball_radius_m) + 1.0e-12:
                    offsets.append((dx, dy, dz))
    seen: set[tuple[str, str]] = set()
    for key in voxel_keys:
        base = tuple(int(v) for v in coords[key].astype(int).tolist())
        for dx, dy, dz in offsets:
            neigh_key = coord_to_key.get((base[0] + dx, base[1] + dy, base[2] + dz))
            if neigh_key is None:
                continue
            a, b = (key, neigh_key) if key < neigh_key else (neigh_key, key)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            edges[a].append(b)
            edges[b].append(a)
    return edges


def _connected_component_ids(voxel_keys: list[str], edges: dict[str, list[str]]) -> dict[str, int]:
    component_by_key: dict[str, int] = {}
    component_id = 0
    for start in sorted(voxel_keys):
        if start in component_by_key:
            continue
        stack = [start]
        component_by_key[start] = component_id
        while stack:
            key = stack.pop()
            for neigh in edges.get(key, []):
                if neigh in component_by_key:
                    continue
                component_by_key[neigh] = component_id
                stack.append(neigh)
        component_id += 1
    return component_by_key


def _pairwise_cost(
    a: BranchCandidate,
    b: BranchCandidate,
    *,
    lambda_theta: float,
    lambda_beta: float,
    lambda_tension: float,
) -> float:
    theta_rms_deg = float(np.sqrt(np.mean(np.square(a.theta_mean - b.theta_mean))) * (180.0 / np.pi))
    beta_dist = float(np.linalg.norm((a.beta_mean - b.beta_mean) / _beta_limit_scales()))
    tension_mae = float(np.mean(np.abs(a.tension_mean - b.tension_mean)))
    return float(
        lambda_theta * (theta_rms_deg / 3.0) ** 2
        + lambda_beta * beta_dist**2
        + lambda_tension * (tension_mae / 100.0) ** 2
    )


def _select_candidates_icm(
    candidates_by_voxel: dict[str, list[BranchCandidate]],
    edges: dict[str, list[str]],
    *,
    max_icm_iters: int,
    lambda_theta: float,
    lambda_beta: float,
    lambda_tension: float,
) -> tuple[dict[str, BranchCandidate], int]:
    selected = {
        key: max(candidates, key=lambda c: (c.unary_score, -c.branch_label))
        for key, candidates in candidates_by_voxel.items()
    }
    iterations = 0
    for _ in range(max(0, int(max_icm_iters))):
        changed = False
        for key in sorted(candidates_by_voxel):
            best = selected[key]
            best_energy = -float("inf")
            for cand in candidates_by_voxel[key]:
                energy = float(cand.unary_score)
                for neigh in edges.get(key, []):
                    if neigh in selected:
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
            if best.branch_label != selected[key].branch_label:
                selected[key] = best
                changed = True
        iterations += 1
        if not changed:
            break
    return selected, iterations


def _reset_sample_ids(dataset: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset.sort_values("source_sample_id", kind="mergesort").reset_index(drop=True)
    meta = meta.sort_values("source_sample_id", kind="mergesort").reset_index(drop=True)
    dataset["sample_id"] = np.arange(len(dataset), dtype=int)
    meta["sample_id"] = np.arange(len(meta), dtype=int)
    return dataset, meta


def select_canonical_branch_graph_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float = 0.01,
    ball_radius_m: float = 0.015,
    beta_eps_norm: float = 0.15,
    max_icm_iters: int = 10,
    w3: float = 1.0,
    w2: float = 0.45,
    w1: float = 0.65,
    w_tension: float = 0.25,
    w_rms: float = 0.40,
    w_size: float = 0.05,
    lambda_theta: float = 1.0,
    lambda_beta: float = 0.5,
    lambda_tension: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df, candidates_by_voxel, _ = _build_candidates(
        dataset,
        meta,
        voxel_size_m=voxel_size_m,
        beta_eps_norm=beta_eps_norm,
        w3=w3,
        w2=w2,
        w1=w1,
        w_tension=w_tension,
        w_rms=w_rms,
        w_size=w_size,
    )
    voxel_keys = sorted(candidates_by_voxel)
    edges = _graph_edges(voxel_keys, voxel_size_m=voxel_size_m, ball_radius_m=ball_radius_m)
    selected, icm_iters = _select_candidates_icm(
        candidates_by_voxel,
        edges,
        max_icm_iters=max_icm_iters,
        lambda_theta=lambda_theta,
        lambda_beta=lambda_beta,
        lambda_tension=lambda_tension,
    )
    keep_indices: list[int] = []
    rows = []
    component_id_by_voxel = _connected_component_ids(voxel_keys, edges)
    for key, cand in selected.items():
        keep_indices.extend(int(v) for v in cand.sample_indices.tolist())
        rows.append(
            {
                "workspace_voxel_key": key,
                "selected_branch_label": int(cand.branch_label),
                "branch_size": int(cand.branch_size),
                "unary_score": float(cand.unary_score),
            }
        )
    keep_indices = sorted(set(keep_indices))
    keep = df.loc[keep_indices].copy()
    keep["graph_branch_selected"] = True
    keep["graph_branch_label"] = [
        f"{row.workspace_voxel_key}:b{int(row.branch_label_in_voxel)}" for row in keep.itertuples(index=False)
    ]
    keep["graph_component_id"] = [component_id_by_voxel[str(v)] for v in keep["workspace_voxel_key"].astype(str).tolist()]
    keep["branch_policy_version"] = POLICY_VERSION
    keep["branch_policy_score"] = [
        float(selected[str(row.workspace_voxel_key)].unary_score) for row in keep.itertuples(index=False)
    ]
    dataset_cols = [c for c in dataset.columns if c in keep.columns] + ["source_sample_id"]
    meta_extra_cols = [
        "source_sample_id",
        "workspace_voxel_x",
        "workspace_voxel_y",
        "workspace_voxel_z",
        "workspace_voxel_key",
        "branch_label_in_voxel",
        "graph_branch_selected",
        "graph_branch_label",
        "graph_component_id",
        "branch_policy_version",
        "branch_policy_score",
    ] + EFFECTIVE_BETA_COLS + ["effective_beta_source"]
    out_dataset = keep[dataset_cols].copy()
    out_meta_cols = [c for c in meta.columns if c in keep.columns]
    for col in meta_extra_cols:
        if col not in out_meta_cols and col in keep.columns:
            out_meta_cols.append(col)
    out_meta = keep[out_meta_cols].copy()
    out_dataset, out_meta = _reset_sample_ids(out_dataset, out_meta)
    summary = {
        "mode": "select_canonical_branch_graph",
        "branch_policy_version": POLICY_VERSION,
        "rows_in": int(len(dataset)),
        "rows_out": int(len(out_dataset)),
        "retention_ratio": float(len(out_dataset) / max(len(dataset), 1)),
        "voxel_size_m": float(voxel_size_m),
        "ball_radius_m": float(ball_radius_m),
        "beta_eps_norm": float(beta_eps_norm),
        "voxels": int(len(voxel_keys)),
        "graph_edges": int(sum(len(v) for v in edges.values()) // 2),
        "graph_components": int(len(set(component_id_by_voxel.values()))),
        "largest_graph_component_voxels": int(
            max(
                (sum(1 for value in component_id_by_voxel.values() if value == comp_id) for comp_id in set(component_id_by_voxel.values())),
                default=0,
            )
        ),
        "total_branch_candidates": int(sum(len(v) for v in candidates_by_voxel.values())),
        "multi_branch_voxel_ratio": float(
            np.mean([len(v) > 1 for v in candidates_by_voxel.values()]) if candidates_by_voxel else 0.0
        ),
        "icm_iterations": int(icm_iters),
        "workspace_coverage_shrinkage": float(1.0 - len(voxel_keys) / max(df["workspace_voxel_key"].nunique(), 1)),
        "selected_branches": rows[:200],
    }
    return out_dataset, out_meta, summary


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_report(out_dir: Path, summary: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    cont = diagnostics.get("continuity", {}).get("groups", {})
    oracle = diagnostics.get("oracle_floor", {})
    clustering = diagnostics.get("branch_clustering", {})
    lines = [
        "# Graph Branch Policy Report",
        "",
        f"- rows in/out: {summary['rows_in']} -> {summary['rows_out']}",
        f"- retention ratio: {summary['retention_ratio']:.4f}",
        f"- voxels: {summary['voxels']}",
        f"- graph edges: {summary['graph_edges']}",
        f"- total branch candidates: {summary['total_branch_candidates']}",
        f"- multi-branch voxel ratio: {summary['multi_branch_voxel_ratio']:.4f}",
        f"- ICM iterations: {summary['icm_iterations']}",
        "",
        "## Continuity",
        "",
        "| group | pairs | theta p95 deg | T p95 N |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ["all_xyz_<=10mm", "xyz_<=10mm_beta_close", "xyz_<=10mm_beta_far"]:
        row = cont.get(key, {})
        lines.append(
            f"| {key} | {row.get('pairs', 0)} | "
            f"{float(row.get('theta_rms_deg_p95', float('nan'))):.3f} | "
            f"{float(row.get('tension_mae_n_p95', float('nan'))):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Branch Clustering",
            "",
            f"- multi-branch ball ratio: {float(clustering.get('multi_branch_ball_ratio', float('nan'))):.4f}",
            "",
            "## Oracle Floor",
            "",
            f"- xyz-NN T MAE: {float(oracle.get('xyz_nn_oracle', {}).get('tension_mae_n', float('nan'))):.2f} N",
            f"- branch-aware xyz-NN T MAE: {float(oracle.get('branch_aware_xyz_nn_oracle', {}).get('tension_mae_n', float('nan'))):.2f} N",
            f"- beta-NN T MAE: {float(oracle.get('beta_nn_oracle', {}).get('tension_mae_n', float('nan'))):.2f} N",
        ]
    )
    (out_dir / "branch_policy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_dataset(dataset_path: Path, meta_path: Path, out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    out_dataset, out_meta, summary = select_canonical_branch_graph_frames(dataset, meta, **kwargs)
    out_dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    out_meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)
    diagnostics: dict[str, Any] = {}
    try:
        continuity_mod = _load_script("eval_branch_aware_continuity")
        clustering_mod = _load_script("eval_branch_clustering")
        oracle_mod = _load_script("eval_oracle_floor")
        diagnostics["continuity"] = continuity_mod.evaluate_frames(out_dataset, out_meta)
        diagnostics["branch_clustering"] = {
            k: v for k, v in clustering_mod.evaluate_frames(out_dataset, out_meta).items() if k != "per_ball"
        }
        diagnostics["oracle_floor"] = oracle_mod.evaluate_frames(out_dataset, out_meta, branch_column="graph_component_id")
    except Exception as exc:  # noqa: BLE001
        diagnostics["diagnostics_error"] = repr(exc)
    payload = {"summary": summary, "diagnostics": diagnostics}
    (out_dir / "selection_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(out_dir, summary, diagnostics)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", "--out-dir", dest="out_dir", required=True, type=Path)
    ap.add_argument("--beta-source", default="effective_theta", choices=["effective_theta"])
    ap.add_argument("--voxel-mm", type=float, default=10.0)
    ap.add_argument("--ball-mm", type=float, default=15.0)
    ap.add_argument("--beta-dbscan-eps", type=float, default=0.15)
    ap.add_argument("--policy", default=POLICY_VERSION)
    ap.add_argument("--max-icm-iters", type=int, default=10)
    args = ap.parse_args()
    payload = select_dataset(
        args.dataset,
        args.meta,
        args.out_dir,
        voxel_size_m=float(args.voxel_mm) / 1000.0,
        ball_radius_m=float(args.ball_mm) / 1000.0,
        beta_eps_norm=float(args.beta_dbscan_eps),
        max_icm_iters=int(args.max_icm_iters),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
