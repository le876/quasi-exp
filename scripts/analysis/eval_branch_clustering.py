#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
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


def _merge_meta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keep = ["sample_id"] + [c for c in BETA_COLS + ["source_component"] if c in meta.columns]
    out = dataset.merge(meta[keep], on="sample_id", how="left", validate="one_to_one")
    if "source_component" not in out.columns:
        out["source_component"] = "unknown"
    out["source_component"] = out["source_component"].fillna("unknown").astype(str)
    return out


def _normalized_beta(beta: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    return beta / scale.reshape(1, -1)


def _pair_mae(values: np.ndarray, labels: np.ndarray, want_same: bool) -> np.ndarray:
    out: list[float] = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            same = labels[a] == labels[b]
            if same != want_same:
                continue
            out.append(float(np.mean(np.abs(values[a] - values[b]))))
    return np.asarray(out, dtype=float)


def _safe_percentile(values: list[float], q: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def evaluate_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    radius_m: float = 0.01,
    min_ball_size: int = 6,
    beta_eps_norm: float = 0.15,
) -> dict[str, Any]:
    df = _merge_meta(dataset, meta)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    beta_norm = _normalized_beta(df[BETA_COLS].to_numpy(dtype=float))
    tension_cols = _numbered_cols(list(df.columns), "tension_", "_n")
    if len(tension_cols) != 12:
        raise ValueError(f"expected 12 tension columns, got {len(tension_cols)}")
    tension = df[tension_cols].to_numpy(dtype=float)

    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    neigh = nn.radius_neighbors(xyz, return_distance=False)

    rows: list[dict[str, Any]] = []
    for center, raw_idx in enumerate(neigh):
        idx = np.asarray(sorted(set(int(v) for v in raw_idx.tolist())), dtype=int)
        if idx.size < int(min_ball_size):
            continue
        labels = DBSCAN(eps=float(beta_eps_norm), min_samples=1).fit_predict(beta_norm[idx])
        branch_count = int(len(set(int(v) for v in labels.tolist())))
        counts = np.asarray([np.sum(labels == lab) for lab in sorted(set(labels.tolist()))], dtype=float)
        purity = float(np.max(counts) / max(np.sum(counts), 1.0))
        within = _pair_mae(tension[idx], labels, True)
        between = _pair_mae(tension[idx], labels, False)
        within_mean = float(np.mean(within)) if within.size else 0.0
        between_mean = float(np.mean(between)) if between.size else 0.0
        denom = within_mean * within_mean + between_mean * between_mean
        variance_ratio = float((between_mean * between_mean) / denom) if denom > 0 else 0.0
        rows.append(
            {
                "center_sample_id": int(df.iloc[center]["sample_id"]),
                "ball_size": int(idx.size),
                "branch_count": branch_count,
                "branch_purity": purity,
                "within_branch_tension_mae_n": within_mean,
                "between_branch_tension_mae_n": between_mean,
                "between_branch_variance_ratio": variance_ratio,
            }
        )

    if not rows:
        return {
            "rows": int(len(df)),
            "radius_m": float(radius_m),
            "balls_evaluated": 0,
            "multi_branch_ball_ratio": 0.0,
            "per_ball": [],
        }

    branch_counts = [float(r["branch_count"]) for r in rows]
    return {
        "rows": int(len(df)),
        "radius_m": float(radius_m),
        "min_ball_size": int(min_ball_size),
        "beta_eps_norm": float(beta_eps_norm),
        "balls_evaluated": int(len(rows)),
        "multi_branch_ball_ratio": float(np.mean(np.asarray(branch_counts) > 1.0)),
        "branch_count_p50": _safe_percentile(branch_counts, 50),
        "branch_count_p90": _safe_percentile(branch_counts, 90),
        "branch_purity_p50": _safe_percentile([float(r["branch_purity"]) for r in rows], 50),
        "within_branch_tension_mae_n_p50": _safe_percentile(
            [float(r["within_branch_tension_mae_n"]) for r in rows], 50
        ),
        "within_branch_tension_mae_n_p95": _safe_percentile(
            [float(r["within_branch_tension_mae_n"]) for r in rows], 95
        ),
        "between_branch_tension_mae_n_p50": _safe_percentile(
            [float(r["between_branch_tension_mae_n"]) for r in rows], 50
        ),
        "between_branch_tension_mae_n_p95": _safe_percentile(
            [float(r["between_branch_tension_mae_n"]) for r in rows], 95
        ),
        "between_branch_variance_ratio_p50": _safe_percentile(
            [float(r["between_branch_variance_ratio"]) for r in rows], 50
        ),
        "between_branch_variance_ratio_p95": _safe_percentile(
            [float(r["between_branch_variance_ratio"]) for r in rows], 95
        ),
        "per_ball": rows,
    }


def evaluate(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--radius-mm", type=float, default=10.0)
    ap.add_argument("--min-ball-size", type=int, default=6)
    ap.add_argument("--beta-eps-norm", type=float, default=0.15)
    args = ap.parse_args()
    payload = evaluate(
        args.dataset,
        args.meta,
        radius_m=float(args.radius_mm) / 1000.0,
        min_ball_size=args.min_ball_size,
        beta_eps_norm=args.beta_eps_norm,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
