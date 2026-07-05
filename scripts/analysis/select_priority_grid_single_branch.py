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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quasi_exp.model.sampling import effective_beta_from_theta  # noqa: E402


THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
EFFECTIVE_BETA_COLS = [f"effective_beta_{i}_rad" for i in range(1, 7)]
POLICY_VERSION = "priority_grid_single_branch_v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ensure_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _voxel_keys(xyz: np.ndarray, voxel_size_m: float) -> list[str]:
    if float(voxel_size_m) <= 0.0:
        raise ValueError("voxel_size_m must be > 0")
    vox = np.floor(np.asarray(xyz, dtype=float).reshape(-1, 3) / float(voxel_size_m)).astype(int)
    return [f"{int(x)},{int(y)},{int(z)}" for x, y, z in vox.tolist()]


def ensure_effective_beta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if not set(EFFECTIVE_BETA_COLS).issubset(out.columns):
        _ensure_columns(dataset, THETA_COLS, "dataset")
        beta = effective_beta_from_theta(dataset[THETA_COLS].to_numpy(dtype=float))
        for i, col in enumerate(EFFECTIVE_BETA_COLS):
            out[col] = beta[:, i]
        out["effective_beta_source"] = "theta_odd_even_mean"
    return out


def priority_ratio_layer_score(row: pd.Series) -> tuple[float, float, float, float, float]:
    return (
        float(row["s1"]),
        float(row["s2"]),
        float(row.get("rms_rnorm", 0.0)),
        float(row.get("max_tension", 0.0)),
        -float(row.get("branch_size", 1.0)),
    )


def _merge_dataset_meta(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(dataset, ["sample_id", "x_m", "y_m", "z_m"] + THETA_COLS + TENSION_COLS, "dataset")
    _ensure_columns(meta, ["sample_id", "s1", "s2"], "meta")
    meta_eff = ensure_effective_beta(dataset, meta)
    keep = ["sample_id"] + [c for c in meta_eff.columns if c != "sample_id"]
    df = dataset.merge(meta_eff[keep], on="sample_id", how="left", validate="one_to_one")
    if "max_tension" not in df.columns:
        df["max_tension"] = df[TENSION_COLS].max(axis=1)
    if "rms_rnorm" not in df.columns:
        df["rms_rnorm"] = 0.0
    return df


def _hard_mask(df: pd.DataFrame) -> np.ndarray:
    tension = df[TENSION_COLS].to_numpy(dtype=float)
    mask = np.isfinite(tension).all(axis=1) & (tension >= 0.0).all(axis=1) & (tension <= 2000.0).all(axis=1)
    mask &= df["rms_rnorm"].astype(float).to_numpy() <= 0.06
    mask &= df["max_tension"].astype(float).to_numpy() <= 2000.0
    for col in ["segmented_success", "canonical_success", "integrated_success"]:
        if col in df.columns:
            mask &= df[col].astype(bool).to_numpy()
    return mask


def filter_priority_grid_single_branch_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float = 0.010,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = _merge_dataset_meta(dataset, meta)
    df["workspace_voxel_key"] = _voxel_keys(df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float), float(voxel_size_m))
    df["source_sample_id"] = df["sample_id"].astype(int)
    hard = _hard_mask(df)
    selected_indices: list[int] = []
    branch_rows: list[dict[str, Any]] = []

    for voxel_key, block in df.groupby("workspace_voxel_key", sort=True):
        block = block.loc[hard[block.index.to_numpy(dtype=int)]].copy()
        if block.empty:
            continue
        grouped = []
        for (s1, s2), layer in block.groupby(["s1", "s2"], sort=True):
            candidate = pd.Series(
                {
                    "s1": float(s1),
                    "s2": float(s2),
                    "max_tension": float(layer["max_tension"].astype(float).mean()),
                    "rms_rnorm": float(layer["rms_rnorm"].astype(float).mean()),
                    "branch_size": int(len(layer)),
                }
            )
            grouped.append((priority_ratio_layer_score(candidate), float(s1), float(s2), layer))
        _score, chosen_s1, chosen_s2, chosen = sorted(grouped, key=lambda item: item[0])[0]
        selected_indices.extend(int(i) for i in chosen.index.tolist())
        branch_rows.append(
            {
                "workspace_voxel_key": str(voxel_key),
                "selected_s1": float(chosen_s1),
                "selected_s2": float(chosen_s2),
                "selected_rows": int(len(chosen)),
                "candidate_layers": int(len(grouped)),
                "candidate_rows": int(len(block)),
            }
        )

    out = df.loc[sorted(selected_indices)].copy().reset_index(drop=True)
    out_dataset = out[["sample_id", "x_m", "y_m", "z_m"] + THETA_COLS + TENSION_COLS].copy()
    out_meta_cols = [c for c in out.columns if c not in set(["x_m", "y_m", "z_m"] + THETA_COLS + TENSION_COLS)]
    out_meta = out[out_meta_cols].copy()
    out_dataset["sample_id"] = np.arange(len(out_dataset), dtype=int)
    out_meta["sample_id"] = np.arange(len(out_meta), dtype=int)
    out_meta["priority_single_branch_selected"] = True
    out_meta["priority_single_branch_policy_version"] = POLICY_VERSION
    out_meta["active_graph_component_id"] = 0
    out_meta["graph_component_id"] = 0

    layer_counts = out_meta.groupby(["s1", "s2"]).size().reset_index(name="rows") if len(out_meta) else pd.DataFrame()
    summary = {
        "policy_version": POLICY_VERSION,
        "voxel_size_m": float(voxel_size_m),
        "input_rows": int(len(dataset)),
        "rows_out": int(len(out_dataset)),
        "retention_ratio": float(len(out_dataset) / max(len(dataset), 1)),
        "workspace_voxels": int(len(branch_rows)),
        "multi_layer_voxels": int(sum(1 for row in branch_rows if int(row["candidate_layers"]) > 1)),
        "selected_layer_counts": layer_counts.to_dict(orient="records"),
        "branches": branch_rows,
    }
    return out_dataset, out_meta, summary


def diagnose_and_gate(dataset: pd.DataFrame, meta: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    continuity_mod = _load_script("eval_branch_aware_continuity")
    clustering_mod = _load_script("eval_branch_clustering")
    oracle_mod = _load_script("eval_oracle_floor")
    continuity = continuity_mod.evaluate_frames(dataset, meta)
    clustering = clustering_mod.evaluate_frames(dataset, meta)
    oracle = oracle_mod.evaluate_frames(dataset, meta, branch_column="active_graph_component_id")
    all10 = continuity.get("groups", {}).get("all_xyz_<=10mm", {})
    beta_close = continuity.get("groups", {}).get("xyz_<=10mm_beta_close", {})
    tension = dataset[TENSION_COLS].to_numpy(dtype=float)
    rms = meta["rms_rnorm"].to_numpy(dtype=float) if "rms_rnorm" in meta.columns else np.full(len(meta), np.inf)
    gate = {
        "hard_gate_passed": bool(
            np.isfinite(tension).all()
            and (tension >= 0.0).all()
            and (tension <= 2000.0).all()
            and float(np.percentile(rms, 95)) <= 0.06
        ),
        "rms_rnorm_q95": float(np.percentile(rms, 95)) if rms.size else float("inf"),
        "all10_theta_p95_deg": float(all10.get("theta_rms_deg_p95", float("inf"))),
        "all10_tension_p95_n": float(all10.get("tension_mae_n_p95", float("inf"))),
        "beta_close_tension_p95_n": float(beta_close.get("tension_mae_n_p95", float("inf"))),
        "multi_branch_ball_ratio": float(clustering.get("multi_branch_ball_ratio", float("inf"))),
        "xyz_nn_tension_mae_n": float(oracle.get("xyz_nn_oracle", {}).get("tension_mae_n", float("inf"))),
        "beta_nn_tension_mae_n": float(oracle.get("beta_nn_oracle", {}).get("tension_mae_n", float("inf"))),
    }
    payload = {
        "gate": gate,
        "continuity": continuity,
        "branch_clustering": {k: v for k, v in clustering.items() if k != "per_ball"},
        "oracle_floor": oracle,
    }
    (out_dir / "priority_grid_single_branch_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return payload


def filter_priority_grid_single_branch_dataset(
    *,
    dataset_path: Path,
    meta_path: Path,
    out_dir: Path,
    voxel_size_m: float = 0.010,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    out_dataset, out_meta, summary = filter_priority_grid_single_branch_frames(dataset, meta, voxel_size_m=float(voxel_size_m))
    out_dataset.to_parquet(out_dir / "dataset.parquet", index=False)
    out_meta.to_parquet(out_dir / "dataset_meta.parquet", index=False)
    diagnostics = diagnose_and_gate(out_dataset, out_meta, out_dir / "diagnostics") if len(out_dataset) else {}
    branch_rows = summary.pop("branches", [])
    if branch_rows:
        pd.DataFrame(branch_rows).to_csv(out_dir / "priority_grid_single_branch_voxel_decisions.csv", index=False)
        summary["voxel_decisions_path"] = str(out_dir / "priority_grid_single_branch_voxel_decisions.csv")
        summary["voxel_decision_sample"] = branch_rows[:20]
    payload = {**summary, "gate": diagnostics.get("gate", {})}
    (out_dir / "priority_grid_single_branch_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--voxel-size-m", type=float, default=0.010)
    args = ap.parse_args()
    payload = filter_priority_grid_single_branch_dataset(
        dataset_path=args.dataset,
        meta_path=args.meta,
        out_dir=args.out_dir,
        voxel_size_m=float(args.voxel_size_m),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
