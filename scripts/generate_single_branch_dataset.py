#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from select_canonical_branch_graph import (  # noqa: E402
    EFFECTIVE_BETA_COLS,
    ensure_effective_beta,
    select_canonical_branch_graph_frames,
)


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def _reset_sample_ids(dataset: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset.reset_index(drop=True).copy()
    meta = meta.reset_index(drop=True).copy()
    dataset["sample_id"] = np.arange(len(dataset), dtype=int)
    meta["sample_id"] = np.arange(len(meta), dtype=int)
    return dataset, meta


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
        if n_bins == 1 or values.size == 0:
            return np.zeros(values.size, dtype=int)
        qs = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        return np.searchsorted(qs, values, side="right").astype(int)

    rb = quantile_bin(radius, radius_bins)
    zb = quantile_bin(z, z_bins)
    ab = np.floor(((angle + np.pi) / (2.0 * np.pi)) * max(1, int(angle_bins))).astype(int)
    ab = np.clip(ab, 0, max(1, int(angle_bins)) - 1)
    return [(int(a), int(b), int(c)) for a, b, c in zip(rb.tolist(), zb.tolist(), ab.tolist())]


def _distal_score(meta: pd.DataFrame) -> np.ndarray:
    beta = meta[EFFECTIVE_BETA_COLS].to_numpy(dtype=float)
    scale = np.asarray([np.deg2rad(5.0)] * 4 + [np.deg2rad(10.0)] * 2, dtype=float)
    b = beta / scale.reshape(1, 6)
    b1 = np.linalg.norm(b[:, [0, 1]], axis=1)
    b2 = np.linalg.norm(b[:, [2, 3]], axis=1)
    b3 = np.linalg.norm(b[:, [4, 5]], axis=1)
    max_t = meta["max_tension"].to_numpy(dtype=float) if "max_tension" in meta.columns else np.zeros(len(meta))
    rms = meta["rms_rnorm"].to_numpy(dtype=float) if "rms_rnorm" in meta.columns else np.zeros(len(meta))
    return b3 - 0.45 * b2 - 0.65 * b1 - 0.25 * max_t / 2000.0 - 0.40 * rms / 0.06


def build_single_branch_from_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    num_samples: int,
    radius_bins: int = 5,
    z_bins: int = 5,
    angle_bins: int = 8,
    seed: int = 20260608,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    meta_eff = ensure_effective_beta(dataset, meta)
    if "source_sample_id" not in meta_eff.columns:
        meta_eff["source_sample_id"] = meta_eff["sample_id"].astype(int)
    if "graph_branch_selected" in meta_eff.columns:
        mask = meta_eff["graph_branch_selected"].astype(bool).to_numpy()
    else:
        mask = np.ones(len(meta_eff), dtype=bool)
    selected_dataset = dataset.loc[mask].copy().reset_index(drop=True)
    selected_meta = meta_eff.loc[mask].copy().reset_index(drop=True)
    if selected_dataset.empty:
        raise ValueError("no selected candidate rows")

    xyz = selected_dataset[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    bins = _workspace_bins(xyz, radius_bins=radius_bins, z_bins=z_bins, angle_bins=angle_bins)
    selected_meta["_workspace_balance_bin"] = [f"{a},{b},{c}" for a, b, c in bins]
    selected_meta["_distal_score"] = _distal_score(selected_meta)
    selected_meta["_rand"] = np.random.default_rng(int(seed)).random(len(selected_meta))

    ordered = selected_meta.sort_values(
        ["_workspace_balance_bin", "_distal_score", "_rand"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    buckets = {
        key: block.index.to_list()
        for key, block in ordered.groupby("_workspace_balance_bin", sort=True)
    }
    keys = sorted(buckets)
    chosen: list[int] = []
    cursor = 0
    while len(chosen) < min(int(num_samples), len(selected_meta)) and keys:
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

    out_dataset = selected_dataset.loc[chosen].copy()
    out_meta = selected_meta.loc[chosen].copy()
    out_meta = out_meta.drop(columns=[c for c in ["_workspace_balance_bin", "_distal_score", "_rand"] if c in out_meta.columns])
    out_dataset, out_meta = _reset_sample_ids(out_dataset, out_meta)
    summary = {
        "mode": "candidate_pool_single_branch",
        "rows_in": int(len(dataset)),
        "selected_rows_in": int(len(selected_dataset)),
        "rows_out": int(len(out_dataset)),
        "requested_rows": int(num_samples),
        "retention_ratio": float(len(out_dataset) / max(len(dataset), 1)),
        "workspace_balance": {
            "radius_bins": int(radius_bins),
            "z_bins": int(z_bins),
            "angle_bins": int(angle_bins),
        },
    }
    return out_dataset, out_meta, summary


def _load_candidate_dir(path: Path, pool_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_parquet(path / "dataset.parquet")
    meta = pd.read_parquet(path / "dataset_meta.parquet")
    dataset = dataset.copy()
    meta = ensure_effective_beta(dataset, meta)
    dataset["pool_name"] = str(pool_name)
    meta["pool_name"] = str(pool_name)
    meta["source_sample_id"] = meta.get("source_sample_id", meta["sample_id"]).astype(int)
    offset_cols = {c: c for c in dataset.columns}
    _ = offset_cols
    return dataset, meta


def _concat_pools(pool_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = []
    metas = []
    next_id = 0
    for pos, path in enumerate(pool_dirs):
        dataset, meta = _load_candidate_dir(path, pool_name=path.name or f"pool{pos}")
        n = len(dataset)
        dataset = dataset.copy()
        meta = meta.copy()
        dataset["pool_row_id"] = dataset["sample_id"].astype(int)
        meta["pool_row_id"] = meta["sample_id"].astype(int)
        dataset["sample_id"] = np.arange(next_id, next_id + n, dtype=int)
        meta["sample_id"] = np.arange(next_id, next_id + n, dtype=int)
        next_id += n
        datasets.append(dataset)
        metas.append(meta)
    if not datasets:
        raise ValueError("no candidate pool dirs")
    return pd.concat(datasets, axis=0, ignore_index=True), pd.concat(metas, axis=0, ignore_index=True)


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def generate_from_candidate_pools(
    pool_dirs: list[Path],
    out_dir: Path,
    *,
    num_samples: int,
    voxel_size_m: float = 0.01,
    ball_radius_m: float = 0.015,
    beta_eps_norm: float = 0.15,
    seed: int = 20260608,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dataset, pool_meta = _concat_pools(pool_dirs)
    graph_dataset, graph_meta, graph_summary = select_canonical_branch_graph_frames(
        pool_dataset,
        pool_meta,
        voxel_size_m=voxel_size_m,
        ball_radius_m=ball_radius_m,
        beta_eps_norm=beta_eps_norm,
    )
    out_dataset, out_meta, summary = build_single_branch_from_frames(
        graph_dataset,
        graph_meta,
        num_samples=num_samples,
        seed=seed,
    )
    out_dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    out_meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)
    report = {"candidate_pool": summary, "graph_policy": graph_summary}
    (out_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Single-Branch Candidate Pool Report",
        "",
        f"- pool rows: {summary['rows_in']}",
        f"- graph-selected rows: {summary['selected_rows_in']}",
        f"- output rows: {summary['rows_out']}",
        f"- graph policy retention: {graph_summary['retention_ratio']:.4f}",
    ]
    (out_dir / "branch_policy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-pool", required=True, help="Comma-separated candidate dataset directories")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--voxel-mm", type=float, default=10.0)
    ap.add_argument("--ball-mm", type=float, default=15.0)
    ap.add_argument("--beta-dbscan-eps", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260608)
    args = ap.parse_args()
    pools = [Path(v.strip()) for v in str(args.candidate_pool).split(",") if v.strip()]
    report = generate_from_candidate_pools(
        pools,
        args.out_dir,
        num_samples=int(args.num_samples),
        voxel_size_m=float(args.voxel_mm) / 1000.0,
        ball_radius_m=float(args.ball_mm) / 1000.0,
        beta_eps_norm=float(args.beta_dbscan_eps),
        seed=int(args.seed),
    )
    print(json.dumps(report["candidate_pool"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
