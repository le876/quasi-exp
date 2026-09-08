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
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.model.sampling import effective_beta_from_theta  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
POLICY_VERSION = "priority_grid_layer_field_v1"


class LayerCandidate(NamedTuple):
    voxel_key: str
    layer_label: str
    s1: float
    s2: float
    sample_indices: np.ndarray
    branch_size: int
    unary_score: float
    beta_mean: np.ndarray
    theta_mean: np.ndarray
    tension_mean: np.ndarray


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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


def layer_label_from_values(s1: float, s2: float) -> str:
    return f"s1_{int(round(float(s1) * 1000)):04d}_s2_{int(round(float(s2) * 1000)):04d}"


def _voxel_info(xyz: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, list[str]]:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be > 0")
    vox = np.floor(np.asarray(xyz, dtype=float) / float(voxel_size_m)).astype(int)
    keys = [f"{int(a)},{int(b)},{int(c)}" for a, b, c in vox.tolist()]
    return vox, keys


def _beta_limit_scales() -> np.ndarray:
    return np.asarray([np.deg2rad(5.0)] * 4 + [np.deg2rad(10.0)] * 2, dtype=float)


def ensure_effective_beta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    missing_eff = [c for c in EFFECTIVE_BETA_COLS if c not in out.columns]
    if missing_eff:
        if all(c in out.columns for c in BETA_COLS):
            for raw, eff in zip(BETA_COLS, EFFECTIVE_BETA_COLS):
                out[eff] = out[raw].astype(float)
            out["effective_beta_source"] = out.get("effective_beta_source", "raw_beta")
        else:
            theta = dataset[theta_cols(dataset)].to_numpy(dtype=float)
            beta = effective_beta_from_theta(theta)
            for i, col in enumerate(EFFECTIVE_BETA_COLS):
                out[col] = beta[:, i]
            out["effective_beta_source"] = "theta_odd_even_mean"
    elif "effective_beta_source" not in out.columns:
        out["effective_beta_source"] = "existing"
    for raw, eff in zip(BETA_COLS, EFFECTIVE_BETA_COLS):
        if raw not in out.columns:
            out[raw] = out[eff].astype(float)
    return out


def _merge_dataset_meta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in dataset.columns or "sample_id" not in meta.columns:
        raise ValueError("dataset and meta must contain sample_id")
    for col in ["s1", "s2"]:
        if col not in meta.columns:
            raise ValueError(f"priority-grid meta must contain {col}")
    meta_eff = ensure_effective_beta(dataset, meta)
    keep = ["sample_id"] + [c for c in meta_eff.columns if c != "sample_id"]
    df = dataset.merge(meta_eff[keep], on="sample_id", how="left", validate="one_to_one")
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


def _layer_unary_score(
    *,
    s1: float,
    s2: float,
    max_tension: float,
    rms_rnorm: float,
    branch_size: int,
    w_s1: float,
    w_s2: float,
    w_tension: float,
    w_rms: float,
    w_size: float,
    tension_scale_n: float,
) -> float:
    cost = (
        float(w_s1) * float(s1)
        + float(w_s2) * float(s2)
        + float(w_tension) * (float(max_tension) / max(float(tension_scale_n), 1.0e-9))
        + float(w_rms) * (float(rms_rnorm) / 0.06)
        - float(w_size) * np.log1p(max(1, int(branch_size)))
    )
    return -float(cost)


def _build_candidates(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float,
    w_s1: float,
    w_s2: float,
    w_tension: float,
    w_rms: float,
    w_size: float,
    tension_scale_n: float,
) -> tuple[pd.DataFrame, dict[str, list[LayerCandidate]], np.ndarray, np.ndarray]:
    df = _merge_dataset_meta(dataset, meta)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    vox, keys = _voxel_info(xyz, voxel_size_m)
    out = df.copy()
    out["workspace_voxel_x"] = vox[:, 0]
    out["workspace_voxel_y"] = vox[:, 1]
    out["workspace_voxel_z"] = vox[:, 2]
    out["workspace_voxel_key"] = keys
    out["source_sample_id"] = out["sample_id"].astype(int)
    out["layer_label"] = [layer_label_from_values(row.s1, row.s2) for row in out.itertuples(index=False)]

    beta = out[EFFECTIVE_BETA_COLS].to_numpy(dtype=float)
    theta = out[theta_cols(out)].to_numpy(dtype=float)
    tension = out[tension_cols(out)].to_numpy(dtype=float)
    hard = _hard_mask(out)
    candidates_by_voxel: dict[str, list[LayerCandidate]] = {}
    for voxel_key, block in out.groupby("workspace_voxel_key", sort=True):
        idx = block.index.to_numpy(dtype=int)
        idx = idx[hard[idx]]
        if idx.size == 0:
            continue
        cand: list[LayerCandidate] = []
        local = out.loc[idx]
        for (s1, s2), group in local.groupby(["s1", "s2"], sort=True):
            local_idx = group.index.to_numpy(dtype=int)
            beta_mean = beta[local_idx].mean(axis=0)
            theta_mean = theta[local_idx].mean(axis=0)
            tension_mean = tension[local_idx].mean(axis=0)
            unary = _layer_unary_score(
                s1=float(s1),
                s2=float(s2),
                max_tension=float(out.loc[local_idx, "max_tension"].astype(float).mean()),
                rms_rnorm=float(out.loc[local_idx, "rms_rnorm"].astype(float).mean()),
                branch_size=int(local_idx.size),
                w_s1=w_s1,
                w_s2=w_s2,
                w_tension=w_tension,
                w_rms=w_rms,
                w_size=w_size,
                tension_scale_n=tension_scale_n,
            )
            cand.append(
                LayerCandidate(
                    voxel_key=str(voxel_key),
                    layer_label=layer_label_from_values(float(s1), float(s2)),
                    s1=float(s1),
                    s2=float(s2),
                    sample_indices=local_idx.copy(),
                    branch_size=int(local_idx.size),
                    unary_score=float(unary),
                    beta_mean=beta_mean,
                    theta_mean=theta_mean,
                    tension_mean=tension_mean,
                )
            )
        candidates_by_voxel[str(voxel_key)] = cand
    return out, candidates_by_voxel, vox, hard


def _graph_edges(
    voxel_keys: list[str],
    voxel_centers: dict[str, np.ndarray],
    *,
    graph_radius_m: float,
    graph_k: int,
) -> dict[str, list[str]]:
    edges: dict[str, set[str]] = {key: set() for key in voxel_keys}
    if len(voxel_keys) < 2:
        return {key: [] for key in voxel_keys}
    coords = np.vstack([voxel_centers[key] for key in voxel_keys])
    k = min(max(2, int(graph_k) + 1), len(voxel_keys))
    nn = NearestNeighbors(n_neighbors=k, radius=float(graph_radius_m), algorithm="auto").fit(coords)
    dist, idx = nn.kneighbors(coords)
    for row, key in enumerate(voxel_keys):
        for d, col in zip(dist[row, 1:], idx[row, 1:]):
            if float(d) > float(graph_radius_m):
                continue
            neigh = voxel_keys[int(col)]
            edges[key].add(neigh)
            edges[neigh].add(key)
    if graph_radius_m > 0:
        neigh_radius = nn.radius_neighbors(coords, return_distance=False)
        for row, raw in enumerate(neigh_radius):
            key = voxel_keys[row]
            for col in raw.tolist():
                if int(col) == row:
                    continue
                neigh = voxel_keys[int(col)]
                edges[key].add(neigh)
                edges[neigh].add(key)
    return {key: sorted(values) for key, values in edges.items()}


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
    a: LayerCandidate,
    b: LayerCandidate,
    *,
    lambda_same_layer: float,
    lambda_s: float,
    lambda_theta: float,
    lambda_beta: float,
    lambda_tension: float,
) -> float:
    theta_rms_deg = float(np.sqrt(np.mean(np.square(a.theta_mean - b.theta_mean))) * (180.0 / np.pi))
    beta_dist = float(np.linalg.norm((a.beta_mean - b.beta_mean) / _beta_limit_scales()))
    tension_mae = float(np.mean(np.abs(a.tension_mean - b.tension_mean)))
    layer_change = 0.0 if a.layer_label == b.layer_label else 1.0
    s_dist = abs(float(a.s1) - float(b.s1)) + abs(float(a.s2) - float(b.s2))
    return float(
        lambda_same_layer * layer_change
        + lambda_s * s_dist
        + lambda_theta * (theta_rms_deg / 3.0) ** 2
        + lambda_beta * beta_dist**2
        + lambda_tension * (tension_mae / 100.0) ** 2
    )


def _selection_energy(
    selected: dict[str, LayerCandidate],
    edges: dict[str, list[str]],
    *,
    lambda_same_layer: float,
    lambda_s: float,
    lambda_theta: float,
    lambda_beta: float,
    lambda_tension: float,
) -> float:
    energy = sum(float(c.unary_score) for c in selected.values())
    seen: set[tuple[str, str]] = set()
    for key, neighs in edges.items():
        for neigh in neighs:
            a, b = (key, neigh) if key < neigh else (neigh, key)
            if (a, b) in seen or a not in selected or b not in selected:
                continue
            seen.add((a, b))
            energy -= _pairwise_cost(
                selected[a],
                selected[b],
                lambda_same_layer=lambda_same_layer,
                lambda_s=lambda_s,
                lambda_theta=lambda_theta,
                lambda_beta=lambda_beta,
                lambda_tension=lambda_tension,
            )
    return float(energy)


def _initial_selection(
    candidates_by_voxel: dict[str, list[LayerCandidate]],
    *,
    preferred_layer: str | None,
) -> dict[str, LayerCandidate]:
    selected: dict[str, LayerCandidate] = {}
    for key, candidates in candidates_by_voxel.items():
        if preferred_layer is not None:
            matches = [cand for cand in candidates if cand.layer_label == preferred_layer]
            if matches:
                selected[key] = max(matches, key=lambda c: (c.unary_score, -c.s1, -c.s2))
                continue
        selected[key] = max(candidates, key=lambda c: (c.unary_score, -c.s1, -c.s2))
    return selected


def _select_candidates_icm(
    candidates_by_voxel: dict[str, list[LayerCandidate]],
    edges: dict[str, list[str]],
    *,
    max_iters: int,
    lambda_same_layer: float,
    lambda_s: float,
    lambda_theta: float,
    lambda_beta: float,
    lambda_tension: float,
    restart_fixed_layers: bool,
) -> tuple[dict[str, LayerCandidate], int, str, float]:
    layers = sorted({cand.layer_label for values in candidates_by_voxel.values() for cand in values})
    restarts: list[str | None] = [None]
    if restart_fixed_layers:
        restarts.extend(layers)
    best_selected: dict[str, LayerCandidate] | None = None
    best_energy = -float("inf")
    best_restart = "local_unary"
    best_iters = 0
    for restart in restarts:
        selected = _initial_selection(candidates_by_voxel, preferred_layer=restart)
        iterations = 0
        for _ in range(max(0, int(max_iters))):
            changed = False
            for key in sorted(candidates_by_voxel):
                current = selected[key]
                best = current
                best_local_energy = -float("inf")
                for cand in candidates_by_voxel[key]:
                    local_energy = float(cand.unary_score)
                    for neigh in edges.get(key, []):
                        if neigh not in selected:
                            continue
                        local_energy -= _pairwise_cost(
                            cand,
                            selected[neigh],
                            lambda_same_layer=lambda_same_layer,
                            lambda_s=lambda_s,
                            lambda_theta=lambda_theta,
                            lambda_beta=lambda_beta,
                            lambda_tension=lambda_tension,
                        )
                    if local_energy > best_local_energy:
                        best_local_energy = local_energy
                        best = cand
                if best.layer_label != current.layer_label:
                    selected[key] = best
                    changed = True
            iterations += 1
            if not changed:
                break
        energy = _selection_energy(
            selected,
            edges,
            lambda_same_layer=lambda_same_layer,
            lambda_s=lambda_s,
            lambda_theta=lambda_theta,
            lambda_beta=lambda_beta,
            lambda_tension=lambda_tension,
        )
        if energy > best_energy:
            best_energy = float(energy)
            best_selected = dict(selected)
            best_restart = restart or "local_unary"
            best_iters = int(iterations)
    if best_selected is None:
        return {}, 0, "none", -float("inf")
    return best_selected, best_iters, best_restart, best_energy


def _reset_sample_ids(dataset: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset.sort_values("source_sample_id", kind="mergesort").reset_index(drop=True)
    meta = meta.sort_values("source_sample_id", kind="mergesort").reset_index(drop=True)
    dataset["sample_id"] = np.arange(len(dataset), dtype=int)
    meta["sample_id"] = np.arange(len(meta), dtype=int)
    return dataset, meta


def select_priority_grid_layer_field_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float = 0.010,
    graph_radius_m: float = 0.020,
    graph_k: int = 16,
    max_iters: int = 30,
    w_s1: float = 1.0,
    w_s2: float = 0.7,
    w_tension: float = 0.03,
    w_rms: float = 0.4,
    w_size: float = 0.05,
    tension_scale_n: float = 100.0,
    lambda_same_layer: float = 1.0,
    lambda_s: float = 2.0,
    lambda_theta: float = 1.0,
    lambda_beta: float = 0.5,
    lambda_tension: float = 0.15,
    restart_fixed_layers: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df, candidates_by_voxel, vox, hard = _build_candidates(
        dataset,
        meta,
        voxel_size_m=voxel_size_m,
        w_s1=w_s1,
        w_s2=w_s2,
        w_tension=w_tension,
        w_rms=w_rms,
        w_size=w_size,
        tension_scale_n=tension_scale_n,
    )
    voxel_keys = sorted(candidates_by_voxel)
    voxel_centers = {
        key: (np.asarray([int(v) for v in key.split(",")], dtype=float) + 0.5) * float(voxel_size_m)
        for key in voxel_keys
    }
    edges = _graph_edges(voxel_keys, voxel_centers, graph_radius_m=graph_radius_m, graph_k=graph_k)
    selected, icm_iters, restart, energy = _select_candidates_icm(
        candidates_by_voxel,
        edges,
        max_iters=max_iters,
        lambda_same_layer=lambda_same_layer,
        lambda_s=lambda_s,
        lambda_theta=lambda_theta,
        lambda_beta=lambda_beta,
        lambda_tension=lambda_tension,
        restart_fixed_layers=restart_fixed_layers,
    )
    component_id_by_voxel = _connected_component_ids(voxel_keys, edges)
    keep_indices: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    for key, cand in selected.items():
        keep_indices.extend(int(v) for v in cand.sample_indices.tolist())
        selected_rows.append(
            {
                "workspace_voxel_key": key,
                "selected_s1": float(cand.s1),
                "selected_s2": float(cand.s2),
                "layer_label": cand.layer_label,
                "branch_size": int(cand.branch_size),
                "unary_score": float(cand.unary_score),
                "component_id": int(component_id_by_voxel.get(key, -1)),
            }
        )
    keep_indices = sorted(set(keep_indices))
    keep = df.loc[keep_indices].copy()
    selected_by_key = {key: cand for key, cand in selected.items()}
    keep["selected_s1"] = [float(selected_by_key[str(v)].s1) for v in keep["workspace_voxel_key"].astype(str).tolist()]
    keep["selected_s2"] = [float(selected_by_key[str(v)].s2) for v in keep["workspace_voxel_key"].astype(str).tolist()]
    keep["layer_label"] = [selected_by_key[str(v)].layer_label for v in keep["workspace_voxel_key"].astype(str).tolist()]
    keep["source_component"] = keep["layer_label"].astype(str)
    keep["layer_field_component_id"] = [
        int(component_id_by_voxel.get(str(v), -1)) for v in keep["workspace_voxel_key"].astype(str).tolist()
    ]
    keep["layer_field_selected"] = True
    keep["layer_field_policy_version"] = POLICY_VERSION
    keep["layer_field_policy_score"] = [
        float(selected_by_key[str(v)].unary_score) for v in keep["workspace_voxel_key"].astype(str).tolist()
    ]

    dataset_cols = [c for c in dataset.columns if c in keep.columns] + ["source_sample_id"]
    meta_extra_cols = [
        "source_sample_id",
        "workspace_voxel_x",
        "workspace_voxel_y",
        "workspace_voxel_z",
        "workspace_voxel_key",
        "selected_s1",
        "selected_s2",
        "layer_label",
        "layer_field_component_id",
        "layer_field_selected",
        "layer_field_policy_version",
        "layer_field_policy_score",
    ] + BETA_COLS + EFFECTIVE_BETA_COLS + ["effective_beta_source"]
    out_dataset = keep[dataset_cols].copy()
    out_meta_cols = [c for c in meta.columns if c in keep.columns]
    if "source_component" in keep.columns and "source_component" not in out_meta_cols:
        out_meta_cols.append("source_component")
    for col in meta_extra_cols:
        if col not in out_meta_cols and col in keep.columns:
            out_meta_cols.append(col)
    out_meta = keep[out_meta_cols].copy()
    out_dataset, out_meta = _reset_sample_ids(out_dataset, out_meta)

    layer_counts = out_meta["layer_label"].value_counts().sort_index().to_dict() if len(out_meta) else {}
    summary = {
        "mode": "select_priority_grid_layer_field",
        "policy_version": POLICY_VERSION,
        "rows_in": int(len(dataset)),
        "rows_out": int(len(out_dataset)),
        "retention_ratio": float(len(out_dataset) / max(len(dataset), 1)),
        "hard_gate_rows": int(np.sum(hard)),
        "rejected_rows": int(len(dataset) - np.sum(hard)),
        "voxel_size_m": float(voxel_size_m),
        "graph_radius_m": float(graph_radius_m),
        "graph_k": int(graph_k),
        "voxels": int(len(voxel_keys)),
        "graph_edges": int(sum(len(v) for v in edges.values()) // 2),
        "graph_components": int(len(set(component_id_by_voxel.values()))) if component_id_by_voxel else 0,
        "largest_graph_component_voxels": int(
            max(
                (sum(1 for value in component_id_by_voxel.values() if value == comp_id) for comp_id in set(component_id_by_voxel.values())),
                default=0,
            )
        ),
        "total_layer_candidates": int(sum(len(v) for v in candidates_by_voxel.values())),
        "multi_layer_voxel_ratio": float(
            np.mean([len(v) > 1 for v in candidates_by_voxel.values()]) if candidates_by_voxel else 0.0
        ),
        "icm_iterations": int(icm_iters),
        "best_restart": str(restart),
        "best_energy": float(energy),
        "selected_layer_counts": {str(k): int(v) for k, v in layer_counts.items()},
        "selected_voxels_sample": selected_rows[:200],
    }
    return out_dataset, out_meta, summary


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _diagnostics(dataset: pd.DataFrame, meta: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    try:
        continuity_mod = _load_script("eval_branch_aware_continuity")
        clustering_mod = _load_script("eval_branch_clustering")
        oracle_mod = _load_script("eval_oracle_floor")
        diagnostics["continuity"] = continuity_mod.evaluate_frames(dataset, meta)
        diagnostics["branch_clustering"] = {
            k: v for k, v in clustering_mod.evaluate_frames(dataset, meta).items() if k != "per_ball"
        }
        diagnostics["oracle_floor"] = oracle_mod.evaluate_frames(dataset, meta, branch_column="layer_label")
    except Exception as exc:  # noqa: BLE001
        diagnostics["diagnostics_error"] = repr(exc)
    return diagnostics


def _gate_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    cont = diagnostics.get("continuity", {}).get("groups", {})
    clustering = diagnostics.get("branch_clustering", {})
    oracle = diagnostics.get("oracle_floor", {})
    all10 = cont.get("all_xyz_<=10mm", {})
    beta_close = cont.get("xyz_<=10mm_beta_close", {})
    return {
        "hard_gate_passed": True,
        "all10_theta_p95_deg": all10.get("theta_rms_deg_p95"),
        "all10_tension_p95_n": all10.get("tension_mae_n_p95"),
        "beta_close_tension_p95_n": beta_close.get("tension_mae_n_p95"),
        "multi_branch_ball_ratio": clustering.get("multi_branch_ball_ratio"),
        "xyz_nn_tension_mae_n": oracle.get("xyz_nn_oracle", {}).get("tension_mae_n"),
        "beta_nn_tension_mae_n": oracle.get("beta_nn_oracle", {}).get("tension_mae_n"),
    }


def select_dataset(dataset_path: Path, meta_path: Path, out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    out_dataset, out_meta, summary = select_priority_grid_layer_field_frames(dataset, meta, **kwargs)
    out_dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    out_meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)
    diagnostics = _diagnostics(out_dataset, out_meta)
    diagnostics["gate"] = _gate_from_diagnostics(diagnostics)
    (out_dir / "selection_summary.json").write_text(
        json.dumps({"summary": summary, "diagnostics": diagnostics}, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "layer_field_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return {"summary": summary, "diagnostics": diagnostics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--voxel-mm", type=float, default=10.0)
    ap.add_argument("--graph-radius-mm", type=float, default=20.0)
    ap.add_argument("--graph-k", type=int, default=16)
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--w-s1", type=float, default=1.0)
    ap.add_argument("--w-s2", type=float, default=0.7)
    ap.add_argument("--w-tension", type=float, default=0.03)
    ap.add_argument("--w-rms", type=float, default=0.4)
    ap.add_argument("--w-size", type=float, default=0.05)
    ap.add_argument("--tension-scale-n", type=float, default=100.0)
    ap.add_argument("--lambda-same-layer", type=float, default=1.0)
    ap.add_argument("--lambda-s", type=float, default=2.0)
    ap.add_argument("--lambda-theta", type=float, default=1.0)
    ap.add_argument("--lambda-beta", type=float, default=0.5)
    ap.add_argument("--lambda-tension", type=float, default=0.15)
    ap.add_argument("--no-restart-fixed-layers", action="store_true")
    args = ap.parse_args()
    payload = select_dataset(
        args.dataset,
        args.meta,
        args.out_dir,
        voxel_size_m=float(args.voxel_mm) / 1000.0,
        graph_radius_m=float(args.graph_radius_mm) / 1000.0,
        graph_k=int(args.graph_k),
        max_iters=int(args.max_iters),
        w_s1=float(args.w_s1),
        w_s2=float(args.w_s2),
        w_tension=float(args.w_tension),
        w_rms=float(args.w_rms),
        w_size=float(args.w_size),
        tension_scale_n=float(args.tension_scale_n),
        lambda_same_layer=float(args.lambda_same_layer),
        lambda_s=float(args.lambda_s),
        lambda_theta=float(args.lambda_theta),
        lambda_beta=float(args.lambda_beta),
        lambda_tension=float(args.lambda_tension),
        restart_fixed_layers=not bool(args.no_restart_fixed_layers),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
