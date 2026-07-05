#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def _theta_cols(df: pd.DataFrame) -> list[str]:
    return _numbered_cols(list(df.columns), "theta_", "_rad")


def _tension_cols(df: pd.DataFrame) -> list[str]:
    return _numbered_cols(list(df.columns), "tension_", "_n")


def _merge_meta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in dataset.columns or "sample_id" not in meta.columns:
        raise ValueError("dataset and meta must both contain sample_id")
    keep = ["sample_id"] + [c for c in BETA_COLS + ["source_component"] if c in meta.columns]
    out = dataset.merge(meta[keep], on="sample_id", how="left", validate="one_to_one")
    if out[BETA_COLS].isna().any().any():
        raise ValueError("meta is missing beta columns for at least one dataset row")
    if "source_component" not in out.columns:
        out["source_component"] = "unknown"
    out["source_component"] = out["source_component"].fillna("unknown").astype(str)
    return out


def _beta_scales(beta: np.ndarray) -> np.ndarray:
    return np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)


def _normalized_beta(beta: np.ndarray) -> np.ndarray:
    return beta / _beta_scales(beta).reshape(1, -1)


def _metric_block(
    *,
    i: np.ndarray,
    j: np.ndarray,
    xyz: np.ndarray,
    theta: np.ndarray,
    tension: np.ndarray | None,
) -> dict[str, Any]:
    i = np.asarray(i, dtype=int).reshape(-1)
    j = np.asarray(j, dtype=int).reshape(-1)
    if i.size == 0:
        return {"pairs": 0}
    dxyz_mm = np.linalg.norm(xyz[i] - xyz[j], axis=1) * 1000.0
    dtheta_rms_deg = np.sqrt(np.mean(np.square(theta[i] - theta[j]), axis=1)) * (180.0 / np.pi)
    block: dict[str, Any] = {
        "pairs": int(i.size),
        "xyz_dist_mm_p50": float(np.percentile(dxyz_mm, 50)),
        "xyz_dist_mm_p90": float(np.percentile(dxyz_mm, 90)),
        "xyz_dist_mm_p95": float(np.percentile(dxyz_mm, 95)),
        "theta_rms_deg_p50": float(np.percentile(dtheta_rms_deg, 50)),
        "theta_rms_deg_p90": float(np.percentile(dtheta_rms_deg, 90)),
        "theta_rms_deg_p95": float(np.percentile(dtheta_rms_deg, 95)),
    }
    if tension is not None:
        t_mae = np.mean(np.abs(tension[i] - tension[j]), axis=1)
        t_max_abs = np.max(np.abs(tension[i] - tension[j]), axis=1)
        block.update(
            {
                "tension_mae_n_p50": float(np.percentile(t_mae, 50)),
                "tension_mae_n_p90": float(np.percentile(t_mae, 90)),
                "tension_mae_n_p95": float(np.percentile(t_mae, 95)),
                "tension_mae_n_max": float(np.max(t_mae)),
                "tension_max_abs_n_p95": float(np.percentile(t_max_abs, 95)),
            }
        )
    return block


def _radius_pairs(xyz: np.ndarray, radius_m: float, k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    neigh = nn.radius_neighbors(xyz, return_distance=False)
    pairs: set[tuple[int, int]] = set()
    for row, raw in enumerate(neigh):
        vals = [int(v) for v in raw.tolist() if int(v) != row]
        if k_neighbors > 0 and len(vals) > k_neighbors:
            dist = np.linalg.norm(xyz[vals] - xyz[row], axis=1)
            vals = [vals[int(k)] for k in np.argsort(dist)[:k_neighbors]]
        for col in vals:
            a, b = (row, col) if row < col else (col, row)
            pairs.add((a, b))
    if not pairs:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    arr = np.asarray(sorted(pairs), dtype=int)
    return arr[:, 0], arr[:, 1]


def _beta_knn_pairs(beta_norm: np.ndarray, k: int, components: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if len(beta_norm) < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    pairs: set[tuple[int, int]] = set()
    if components is None:
        blocks = [(np.arange(len(beta_norm)), beta_norm)]
    else:
        blocks = [(np.where(components == comp)[0], beta_norm[components == comp]) for comp in np.unique(components)]
    for global_idx, block_beta in blocks:
        if len(global_idx) < 2:
            continue
        kk = min(int(k) + 1, len(global_idx))
        nn = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(block_beta)
        _, idx = nn.kneighbors(block_beta)
        for local_row, neigh in enumerate(idx):
            row = int(global_idx[local_row])
            for local_col in neigh[1:]:
                col = int(global_idx[int(local_col)])
                a, b = (row, col) if row < col else (col, row)
                pairs.add((a, b))
    if not pairs:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    arr = np.asarray(sorted(pairs), dtype=int)
    return arr[:, 0], arr[:, 1]


def evaluate_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    radii_m: list[float] | None = None,
    beta_k_values: list[int] | None = None,
    xyz_k_neighbors: int = 80,
    beta_close_threshold_norm: float = 0.15,
) -> dict[str, Any]:
    radii = [0.005, 0.01, 0.02] if radii_m is None else [float(v) for v in radii_m]
    beta_ks = [8, 16, 32] if beta_k_values is None else [int(v) for v in beta_k_values]
    df = _merge_meta(dataset, meta)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    beta = df[BETA_COLS].to_numpy(dtype=float)
    beta_norm = _normalized_beta(beta)
    theta_cols = _theta_cols(df)
    if len(theta_cols) != 30:
        raise ValueError(f"expected 30 theta columns, got {len(theta_cols)}")
    theta = df[theta_cols].to_numpy(dtype=float)
    tension_cols = _tension_cols(df)
    tension = df[tension_cols].to_numpy(dtype=float) if len(tension_cols) == 12 else None
    components = df["source_component"].astype(str).to_numpy()

    groups: dict[str, Any] = {}
    for k in beta_ks:
        i, j = _beta_knn_pairs(beta_norm, k)
        groups[f"same_beta_knn_k{k}"] = _metric_block(i=i, j=j, xyz=xyz, theta=theta, tension=tension)
        i, j = _beta_knn_pairs(beta_norm, k, components=components)
        groups[f"same_component_beta_knn_k{k}"] = _metric_block(i=i, j=j, xyz=xyz, theta=theta, tension=tension)

    for radius in radii:
        label = f"<={int(round(radius * 1000.0))}mm"
        i, j = _radius_pairs(xyz, radius, xyz_k_neighbors)
        same = components[i] == components[j] if i.size else np.asarray([], dtype=bool)
        beta_dist = np.sqrt(np.mean(np.square(beta_norm[i] - beta_norm[j]), axis=1)) if i.size else np.asarray([])
        beta_close = beta_dist <= float(beta_close_threshold_norm)
        groups[f"all_xyz_{label}"] = _metric_block(i=i, j=j, xyz=xyz, theta=theta, tension=tension)
        groups[f"same_component_xyz_{label}"] = _metric_block(i=i[same], j=j[same], xyz=xyz, theta=theta, tension=tension)
        groups[f"cross_component_xyz_{label}"] = _metric_block(i=i[~same], j=j[~same], xyz=xyz, theta=theta, tension=tension)
        groups[f"xyz_{label}_beta_close"] = _metric_block(i=i[beta_close], j=j[beta_close], xyz=xyz, theta=theta, tension=tension)
        groups[f"xyz_{label}_beta_far"] = _metric_block(i=i[~beta_close], j=j[~beta_close], xyz=xyz, theta=theta, tension=tension)

    return {
        "rows": int(len(df)),
        "beta_close_threshold_norm": float(beta_close_threshold_norm),
        "radii_m": radii,
        "beta_k_values": beta_ks,
        "groups": groups,
    }


def evaluate(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--radii-mm", default="5,10,20")
    ap.add_argument("--beta-k", default="8,16,32")
    ap.add_argument("--xyz-k-neighbors", type=int, default=80)
    ap.add_argument("--beta-close-threshold-norm", type=float, default=0.15)
    args = ap.parse_args()

    radii_m = [float(v.strip()) / 1000.0 for v in args.radii_mm.split(",") if v.strip()]
    beta_k = [int(v.strip()) for v in args.beta_k.split(",") if v.strip()]
    payload = evaluate(
        args.dataset,
        args.meta,
        radii_m=radii_m,
        beta_k_values=beta_k,
        xyz_k_neighbors=args.xyz_k_neighbors,
        beta_close_threshold_norm=args.beta_close_threshold_norm,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

