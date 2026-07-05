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


def _merge_meta(dataset: pd.DataFrame, meta: pd.DataFrame, *, branch_column: str | None = None) -> pd.DataFrame:
    extra = ["source_component", "branch_label"]
    if branch_column:
        extra.append(str(branch_column))
    keep = ["sample_id"] + [c for c in BETA_COLS + extra if c in meta.columns]
    keep = list(dict.fromkeys(keep))
    out = dataset.merge(meta[keep], on="sample_id", how="left", validate="one_to_one")
    if "source_component" not in out.columns:
        out["source_component"] = "unknown"
    return out


def _normalized_beta(beta: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    return beta / scale.reshape(1, -1)


def _nearest_leave_one_out(space: np.ndarray) -> np.ndarray:
    if len(space) < 2:
        return np.zeros(len(space), dtype=int)
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(space)
    _, idx = nn.kneighbors(space)
    return idx[:, 1].astype(int)


def _nearest_same_branch(space: np.ndarray, branch: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(space), dtype=int)
    for pos in range(len(space)):
        mask = np.where(branch == branch[pos])[0]
        mask = mask[mask != pos]
        if mask.size == 0:
            pred[pos] = pos
            continue
        dist = np.linalg.norm(space[mask] - space[pos], axis=1)
        pred[pos] = int(mask[int(np.argmin(dist))])
    return pred


def _metrics(df: pd.DataFrame, pred_idx: np.ndarray) -> dict[str, Any]:
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    tension_cols = _numbered_cols(list(df.columns), "tension_", "_n")
    theta = df[theta_cols].to_numpy(dtype=float)
    tension = df[tension_cols].to_numpy(dtype=float)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    dtheta = np.mean(np.abs(theta - theta[pred_idx]), axis=1) * (180.0 / np.pi)
    dtension = np.mean(np.abs(tension - tension[pred_idx]), axis=1)
    dee = np.linalg.norm(xyz - xyz[pred_idx], axis=1) * 1000.0
    return {
        "theta_mae_deg": float(np.mean(dtheta)),
        "theta_p95_deg": float(np.percentile(dtheta, 95)),
        "tension_mae_n": float(np.mean(dtension)),
        "tension_p95_n": float(np.percentile(dtension, 95)),
        "ee_pos_mae_mm": float(np.mean(dee)),
        "ee_pos_p95_mm": float(np.percentile(dee, 95)),
    }


def evaluate_frames(dataset: pd.DataFrame, meta: pd.DataFrame, *, branch_column: str | None = None) -> dict[str, Any]:
    df = _merge_meta(dataset, meta, branch_column=branch_column)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    beta_norm = _normalized_beta(df[BETA_COLS].to_numpy(dtype=float))
    branch_col = branch_column
    if branch_col is None:
        branch_col = "branch_label" if "branch_label" in df.columns else "source_component"
    if branch_col not in df.columns:
        raise ValueError(f"branch column not found: {branch_col}")
    branch = df[branch_col].astype(str).to_numpy()

    xyz_pred = _nearest_leave_one_out(xyz)
    branch_xyz_pred = _nearest_same_branch(xyz, branch)
    beta_pred = _nearest_leave_one_out(beta_norm)
    return {
        "rows": int(len(df)),
        "branch_column": str(branch_col),
        "xyz_nn_oracle": _metrics(df, xyz_pred),
        "branch_aware_xyz_nn_oracle": _metrics(df, branch_xyz_pred),
        "beta_nn_oracle": _metrics(df, beta_pred),
    }


def evaluate(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--branch-column", default=None)
    args = ap.parse_args()
    payload = evaluate(args.dataset, args.meta, branch_column=args.branch_column)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
