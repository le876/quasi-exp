#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from true_sinsincos_tube_utils import (  # noqa: E402
    BETA_COLS,
    THETA_COLS,
    U_DEG_COLS,
    beta_from_u_deg,
    infer_u_mapping_params,
    theta_from_beta_batch,
)


XYZ_COLS = ["x_m", "y_m", "z_m"]


def split_iid(n: int, *, seed: int, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n), dtype=np.int64)
    rng.shuffle(idx)
    n_test = int(round(int(n) * float(test_size)))
    n_val = int(round(int(n) * float(val_size)))
    n_train = int(n) - n_val - n_test
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def split_by_column(df: pd.DataFrame, col: str, *, seed: int, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if col not in df.columns:
        raise ValueError(f"split column not present: {col}")
    rng = np.random.default_rng(int(seed))
    vals = np.asarray(pd.unique(df[col]), dtype=object)
    rng.shuffle(vals)
    n_test = max(1, int(round(len(vals) * float(test_size))))
    n_val = max(1, int(round(len(vals) * float(val_size)))) if len(vals) - n_test > 2 else 0
    test_vals = set(vals[:n_test].tolist())
    val_vals = set(vals[n_test : n_test + n_val].tolist())
    arr = df[col].to_numpy()
    test_idx = np.flatnonzero(np.isin(arr, list(test_vals))).astype(np.int64)
    val_idx = np.flatnonzero(np.isin(arr, list(val_vals))).astype(np.int64)
    train_idx = np.flatnonzero(~np.isin(arr, list(test_vals | val_vals))).astype(np.int64)
    return train_idx, val_idx, test_idx


def make_split(df: pd.DataFrame, split: str, *, seed: int, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = str(split)
    if split == "iid":
        return split_iid(len(df), seed=seed, val_size=val_size, test_size=test_size)
    if split == "angle_holdout":
        return split_by_column(df, "angle_idx", seed=seed, val_size=val_size, test_size=test_size)
    if split == "offset_holdout":
        col = "tube_idx" if "tube_idx" in df.columns else "delta_n1_mm"
        return split_by_column(df, col, seed=seed, val_size=val_size, test_size=test_size)
    if split == "radius_holdout":
        return split_by_column(df, "amp_xy_mm", seed=seed, val_size=val_size, test_size=test_size)
    if split == "center_holdout":
        col = "center_id" if "center_id" in df.columns else "center_x"
        return split_by_column(df, col, seed=seed, val_size=val_size, test_size=test_size)
    raise ValueError(f"unsupported split: {split}")


def model_for(name: str, *, seed: int, max_iter: int, batch_size: int) -> MLPRegressor:
    if name == "mlp_u":
        hidden = (256, 128, 64, 32)
    elif name == "mlp_u_large":
        hidden = (512, 256, 128, 64)
    else:
        raise ValueError(f"unsupported model: {name}")
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1.0e-6,
        batch_size=int(batch_size),
        learning_rate_init=1.0e-3,
        max_iter=int(max_iter),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=int(seed),
        verbose=False,
    )


def _metric_deg(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    err = np.abs(np.asarray(true, dtype=float) - np.asarray(pred, dtype=float))
    return float(np.mean(err)), float(np.percentile(err, 95))


def run_one(
    df: pd.DataFrame,
    args: argparse.Namespace,
    split: str,
    model_name: str,
    *,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    mapping: dict[str, float],
) -> dict[str, Any]:
    train_idx, val_idx, test_idx = make_split(df, split, seed=int(args.seed), val_size=float(args.val_size), test_size=float(args.test_size))
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    features, feature_names = build_features(xyz, feature_set=str(args.feature_set))
    target_u = df[U_DEG_COLS].to_numpy(dtype=float)
    model = model_for(model_name, seed=int(args.seed), max_iter=int(args.max_iter), batch_size=int(args.batch_size))
    x_scaler = StandardScaler().fit(features[train_idx])
    y_scaler = StandardScaler().fit(target_u[train_idx])
    t0 = time.perf_counter()
    model.fit(x_scaler.transform(features[train_idx]), y_scaler.transform(target_u[train_idx]))
    fit_s = float(time.perf_counter() - t0)

    metrics_by_split: dict[str, dict[str, float]] = {}
    preds_to_save: dict[str, pd.DataFrame] = {}
    for part_name, idx in {"val": val_idx, "test": test_idx}.items():
        pred_u = y_scaler.inverse_transform(model.predict(x_scaler.transform(features[idx])))
        beta_pred, _s1, _s2, _q = beta_from_u_deg(pred_u, **mapping)
        theta_pred = theta_from_beta_batch(beta_pred, theta_sign=theta_sign)
        xyz_pred = fk_dh_batch(theta_pred, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
        ee = np.linalg.norm(xyz_pred - xyz[idx], axis=1) * 1000.0
        u_mae, u_p95 = _metric_deg(target_u[idx], pred_u)
        metrics_by_split[part_name] = {
            "rows": int(len(idx)),
            "fit_s": fit_s,
            "u_mae": u_mae,
            "u_p95": u_p95,
            "ee_mean_mm": float(np.mean(ee)),
            "ee_p95_mm": float(np.percentile(ee, 95)),
            "ee_max_mm": float(np.max(ee)),
        }
        pred_df = df.iloc[idx][XYZ_COLS].copy().reset_index(drop=True)
        for i, col in enumerate(U_DEG_COLS):
            pred_df[f"pred_{col}"] = pred_u[:, i]
        for i, col in enumerate(BETA_COLS):
            pred_df[f"pred_{col}"] = beta_pred[:, i]
        for i, col in enumerate(THETA_COLS):
            pred_df[f"pred_{col}"] = theta_pred[:, i]
        preds_to_save[part_name] = pred_df
    return {
        "split": split,
        "model": model_name,
        "sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "feature_names": feature_names,
        "metrics": metrics_by_split,
        "package": {
            "kind": "u_pose",
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "input_cols": XYZ_COLS,
            "feature_set": str(args.feature_set),
            "feature_names": feature_names,
            "target_cols": U_DEG_COLS,
            "u_mapping": mapping,
            "beta_cols": BETA_COLS,
            "theta_cols": THETA_COLS,
            "theta_sign": float(theta_sign),
            "robot_config": str(args.robot_config),
            "dataset": str(args.dataset),
            "split": split,
            "model_name": model_name,
        },
        "predictions": preds_to_save,
    }


def load_u_pose_model_package(path: Path) -> dict[str, Any]:
    package = load(path)
    if not isinstance(package, dict) or package.get("kind") != "u_pose":
        raise ValueError("model package kind must be 'u_pose'")
    for key in ["model", "x_scaler", "y_scaler", "target_cols", "u_mapping"]:
        if key not in package:
            raise ValueError(f"u_pose model package missing {key}")
    return package


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "model_checkpoints"
    pred_dir = out_dir / "prediction_parquets"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    df = pd.read_parquet(args.dataset)
    if int(args.max_rows) > 0 and len(df) > int(args.max_rows):
        df = df.sample(n=int(args.max_rows), random_state=int(args.seed)).reset_index(drop=True)
    missing = [c for c in XYZ_COLS + U_DEG_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"dataset missing columns: {missing}")
    mapping = infer_u_mapping_params(df)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for split in [s.strip() for s in str(args.splits).split(",") if s.strip()]:
        for model_name in [m.strip() for m in str(args.models).split(",") if m.strip()]:
            result = run_one(
                df,
                args,
                split,
                model_name,
                lengths_m=inputs.lengths_m,
                p_end_local_m=inputs.p_end_local_m,
                theta_sign=theta_sign,
                mapping=mapping,
            )
            key = f"{split}_{model_name}"
            model_dir = ckpt_dir / key
            model_dir.mkdir(parents=True, exist_ok=True)
            dump(result["package"], model_dir / "model.joblib")
            for pred_split, pred_df in result["predictions"].items():
                pred_df.to_parquet(pred_dir / f"{key}_{pred_split}.parquet", index=False, compression="zstd")
            details[key] = {k: v for k, v in result.items() if k not in {"package", "predictions"}}
            for eval_split, m in result["metrics"].items():
                rows.append({"split": split, "model": model_name, "eval_split": eval_split, **m})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_flat.csv", index=False)
    (out_dir / "all_metrics.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": str(out_dir), "metric_rows": len(rows), "rows": rows}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train xyz->u pose baselines.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="mlp_u,mlp_u_large")
    ap.add_argument("--splits", default="iid,angle_holdout,offset_holdout")
    ap.add_argument("--feature-set", default="poly_heavy", choices=["raw", "poly_medium", "poly_heavy"])
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260709)
    return ap.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({"out_dir": payload["out_dir"], "metric_rows": payload["metric_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

