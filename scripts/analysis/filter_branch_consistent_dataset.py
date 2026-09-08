#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
POLICY_VERSION = "distal_v1"


def _normalized_beta(beta: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    return beta / scale.reshape(1, -1)


def _voxel_keys(xyz: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, list[str]]:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be > 0")
    vox = np.floor(np.asarray(xyz, dtype=float) / float(voxel_size_m)).astype(int)
    keys = [f"{int(a)},{int(b)},{int(c)}" for a, b, c in vox.tolist()]
    return vox, keys


def score_branch_candidates(
    branches: pd.DataFrame,
    *,
    w_tension: float = 0.02,
    w_rms: float = 0.50,
    w_size: float = 0.03,
) -> pd.DataFrame:
    out = branches.copy()
    required = {"distal_preference_score", "max_tension", "rms_rnorm", "branch_size"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"branch candidate table missing columns: {missing}")
    out["branch_score"] = (
        out["distal_preference_score"].astype(float)
        + float(w_tension) * (out["max_tension"].astype(float) / 2000.0)
        + float(w_rms) * (out["rms_rnorm"].astype(float) / 0.06)
        - float(w_size) * np.log1p(out["branch_size"].astype(float))
    )
    return out


def _merge_meta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in dataset.columns or "sample_id" not in meta.columns:
        raise ValueError("dataset and meta must contain sample_id")
    if any(c not in meta.columns for c in BETA_COLS):
        raise ValueError("meta must contain beta1_rad..beta6_rad")
    return dataset[["sample_id", "x_m", "y_m", "z_m"]].merge(meta, on="sample_id", how="left", validate="one_to_one")


def _branch_summary(block: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in block.groupby("branch_label_in_voxel", sort=True):
        rows.append(
            {
                "branch_id": int(label),
                "branch_label_in_voxel": int(label),
                "branch_size": int(len(group)),
                "distal_preference_score": float(group["distal_preference_score"].mean()),
                "max_tension": float(group["max_tension"].mean()),
                "rms_rnorm": float(group["rms_rnorm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _prepare_meta_columns(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if "source_component" not in out.columns:
        out["source_component"] = "unknown"
    if "rms_rnorm" not in out.columns:
        out["rms_rnorm"] = 0.0
    if "max_tension" not in out.columns:
        out["max_tension"] = 0.0
    if "distal_preference_score" not in out.columns:
        if {"beta_group1_norm", "beta_group2_norm", "beta_group3_norm"}.issubset(out.columns):
            out["distal_preference_score"] = (
                out["beta_group1_norm"].astype(float) ** 2
                + out["beta_group2_norm"].astype(float) ** 2
                - 1.25 * out["beta_group3_norm"].astype(float) ** 2
            )
        else:
            beta = out[BETA_COLS].to_numpy(dtype=float)
            scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
            norm = beta / scale.reshape(1, 6)
            g1 = np.sqrt(np.mean(np.square(norm[:, [0, 1]]), axis=1))
            g2 = np.sqrt(np.mean(np.square(norm[:, [2, 3]]), axis=1))
            g3 = np.sqrt(np.mean(np.square(norm[:, [4, 5]]), axis=1))
            out["distal_preference_score"] = g1**2 + g2**2 - 1.25 * g3**2
    return out


def filter_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float = 0.01,
    beta_eps_norm: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    meta_in = _prepare_meta_columns(meta)
    joined = _merge_meta(dataset, meta_in)
    xyz = joined[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    vox, keys = _voxel_keys(xyz, voxel_size_m)
    beta_norm = _normalized_beta(joined[BETA_COLS].to_numpy(dtype=float))
    joined["workspace_voxel_x"] = vox[:, 0].astype(int)
    joined["workspace_voxel_y"] = vox[:, 1].astype(int)
    joined["workspace_voxel_z"] = vox[:, 2].astype(int)
    joined["workspace_voxel_key"] = keys
    joined["source_sample_id"] = joined["sample_id"].astype(int)

    selected_source_ids: list[int] = []
    branch_rows: list[dict[str, Any]] = []
    total_branches = 0
    multi_branch_voxels = 0
    for voxel_key, block in joined.groupby("workspace_voxel_key", sort=True):
        idx = block.index.to_numpy(dtype=int)
        if len(idx) == 1:
            labels = np.asarray([0], dtype=int)
        else:
            labels = DBSCAN(eps=float(beta_eps_norm), min_samples=1).fit_predict(beta_norm[idx])
        joined.loc[idx, "branch_label_in_voxel"] = labels.astype(int)
        candidate = _branch_summary(joined.loc[idx])
        candidate = score_branch_candidates(candidate)
        chosen = candidate.sort_values(["branch_score", "branch_id"], kind="mergesort").iloc[0]
        chosen_label = int(chosen["branch_label_in_voxel"])
        chosen_idx = idx[labels == chosen_label]
        total_branches += int(len(candidate))
        if len(candidate) > 1:
            multi_branch_voxels += 1
        for _, row in candidate.iterrows():
            branch_rows.append(
                {
                    "workspace_voxel_key": voxel_key,
                    "branch_label_in_voxel": int(row["branch_label_in_voxel"]),
                    "branch_size": int(row["branch_size"]),
                    "branch_score": float(row["branch_score"]),
                    "selected": bool(int(row["branch_label_in_voxel"]) == chosen_label),
                }
            )
        selected_source_ids.extend(joined.loc[chosen_idx, "source_sample_id"].astype(int).tolist())

    selected_source_ids = sorted(selected_source_ids)
    keep_dataset = dataset.loc[dataset["sample_id"].astype(int).isin(selected_source_ids)].copy()
    keep_meta = meta_in.loc[meta_in["sample_id"].astype(int).isin(selected_source_ids)].copy()
    branch_info = joined[
        [
            "source_sample_id",
            "workspace_voxel_x",
            "workspace_voxel_y",
            "workspace_voxel_z",
            "workspace_voxel_key",
            "branch_label_in_voxel",
        ]
    ].copy()
    branch_table = pd.DataFrame(branch_rows)
    selected_branch_info = branch_table.loc[branch_table["selected"]].copy()
    branch_info = branch_info.merge(
        selected_branch_info[["workspace_voxel_key", "branch_label_in_voxel", "branch_score", "branch_size"]],
        on=["workspace_voxel_key", "branch_label_in_voxel"],
        how="left",
    )
    keep_meta = keep_meta.merge(branch_info, left_on="sample_id", right_on="source_sample_id", how="left", validate="one_to_one")
    keep_meta["canonical_branch_selected"] = True
    keep_meta["filter_policy_version"] = POLICY_VERSION

    keep_dataset = keep_dataset.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    keep_meta = keep_meta.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    keep_dataset["sample_id"] = np.arange(len(keep_dataset), dtype=int)
    keep_meta["sample_id"] = np.arange(len(keep_meta), dtype=int)

    summary = {
        "mode": "branch_consistent_filter",
        "filter_policy_version": POLICY_VERSION,
        "voxel_size_m": float(voxel_size_m),
        "beta_eps_norm": float(beta_eps_norm),
        "rows_in": int(len(dataset)),
        "rows_out": int(len(keep_dataset)),
        "retention_ratio": float(len(keep_dataset) / max(len(dataset), 1)),
        "voxels": int(joined["workspace_voxel_key"].nunique()),
        "total_branches": int(total_branches),
        "multi_branch_voxels": int(multi_branch_voxels),
        "multi_branch_voxel_ratio": float(multi_branch_voxels / max(joined["workspace_voxel_key"].nunique(), 1)),
        "source_counts": {str(k): int(v) for k, v in keep_meta["source_component"].astype(str).value_counts().sort_index().to_dict().items()},
    }
    return keep_dataset, keep_meta, summary


def filter_dataset(dataset_path: Path, meta_path: Path, out_dir: Path, *, voxel_size_m: float, beta_eps_norm: float) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    out_dataset, out_meta, summary = filter_frames(dataset, meta, voxel_size_m=voxel_size_m, beta_eps_norm=beta_eps_norm)
    out_dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    out_meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)
    (out_dir / "filter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--voxel-size-mm", type=float, default=10.0)
    ap.add_argument("--beta-eps-norm", type=float, default=0.15)
    args = ap.parse_args()
    summary = filter_dataset(
        args.dataset,
        args.meta,
        args.out_dir,
        voxel_size_m=float(args.voxel_size_mm) / 1000.0,
        beta_eps_norm=float(args.beta_eps_norm),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

