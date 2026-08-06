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
from joblib import dump
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402


XYZ_COLS = ["x_m", "y_m", "z_m"]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]


def theta_from_beta_batch(beta: np.ndarray, *, theta_sign: float) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1, 6)
    theta = np.empty((len(beta), 30), dtype=float)
    for section, (odd_col, even_col) in enumerate(((0, 1), (2, 3), (4, 5))):
        start = section * 10
        theta[:, start : start + 10 : 2] = beta[:, odd_col : odd_col + 1]
        theta[:, start + 1 : start + 10 : 2] = beta[:, even_col : even_col + 1]
    return theta * float(theta_sign)


def split_iid(n: int, *, seed: int, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n), dtype=np.int64)
    rng.shuffle(idx)
    n_test = int(round(n * float(test_size)))
    n_val = int(round(n * float(val_size)))
    n_train = int(n) - n_val - n_test
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def split_x_slab(xyz: np.ndarray, *, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.argsort(np.asarray(xyz, dtype=float)[:, 0]).astype(np.int64)
    n = len(idx)
    n_test = int(round(n * float(test_size)))
    n_val = int(round(n * float(val_size)))
    n_train = n - n_val - n_test
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def split_by_column(df: pd.DataFrame, col: str, *, seed: int, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if col not in df.columns:
        raise ValueError(f"split column not present: {col}")
    rng = np.random.default_rng(int(seed))
    vals = np.asarray(pd.unique(df[col]), dtype=object)
    rng.shuffle(vals)
    n = len(vals)
    n_test = max(1, int(round(n * float(test_size))))
    n_val = max(1, int(round(n * float(val_size)))) if n - n_test > 2 else 0
    test_vals = set(vals[:n_test].tolist())
    val_vals = set(vals[n_test : n_test + n_val].tolist())
    arr = df[col].to_numpy()
    test_idx = np.flatnonzero(np.isin(arr, list(test_vals))).astype(np.int64)
    val_idx = np.flatnonzero(np.isin(arr, list(val_vals))).astype(np.int64)
    train_idx = np.flatnonzero(~np.isin(arr, list(test_vals | val_vals))).astype(np.int64)
    return train_idx, val_idx, test_idx


def make_split(df: pd.DataFrame, split: str, *, seed: int, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = str(split)
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    if split == "iid":
        return split_iid(len(df), seed=seed, val_size=val_size, test_size=test_size)
    if split == "x_slab":
        return split_x_slab(xyz, val_size=val_size, test_size=test_size)
    if split == "ellipse_holdout":
        return split_by_column(df, "ellipse_id", seed=seed, val_size=val_size, test_size=test_size)
    if split == "radius_holdout":
        col = "amp_xy_mm" if "amp_xy_mm" in df.columns else "amp_xy_m"
        return split_by_column(df, col, seed=seed, val_size=val_size, test_size=test_size)
    if split == "center_holdout":
        col = "center_id" if "center_id" in df.columns else "center_x"
        return split_by_column(df, col, seed=seed, val_size=val_size, test_size=test_size)
    raise ValueError(f"unsupported split: {split}")


def model_for(name: str, *, seed: int, max_iter: int, batch_size: int) -> MLPRegressor:
    if name == "mlp_beta6":
        hidden = (256, 128, 64, 32)
    elif name == "mlp_beta6_large":
        hidden = (512, 256, 128, 64)
    elif name == "mlp_theta30_direct":
        hidden = (256, 128, 64, 32)
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


def fit_scaled(model: MLPRegressor, X_train: np.ndarray, y_train: np.ndarray) -> tuple[StandardScaler, StandardScaler, MLPRegressor, float]:
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    t0 = time.perf_counter()
    model.fit(x_scaler.transform(X_train), y_scaler.transform(y_train))
    fit_s = float(time.perf_counter() - t0)
    return x_scaler, y_scaler, model, fit_s


def predict_scaled(x_scaler: StandardScaler, y_scaler: StandardScaler, model: MLPRegressor, X: np.ndarray) -> np.ndarray:
    return y_scaler.inverse_transform(model.predict(x_scaler.transform(X)))


def axis_metrics(target_xyz: np.ndarray, achieved_xyz: np.ndarray) -> dict[str, Any]:
    err = (np.asarray(achieved_xyz, dtype=float) - np.asarray(target_xyz, dtype=float)) * 1000.0
    out: dict[str, Any] = {}
    fixed = []
    p95s = []
    for i, axis in enumerate("xyz"):
        vals = err[:, i]
        p95 = float(np.percentile(np.abs(vals), 95))
        out[f"{axis}err_mean_mm"] = float(np.mean(vals))
        out[f"{axis}err_p95_abs_mm"] = p95
        out[f"{axis}err_same_sign_ratio"] = float(max(np.mean(vals > 0.0), np.mean(vals < 0.0)))
        out[f"fixed_{axis}_bias_gt2mm"] = bool(np.all(vals > 2.0) or np.all(vals < -2.0))
        fixed.append(out[f"fixed_{axis}_bias_gt2mm"])
        p95s.append(p95)
    out["axiserr_max_p95_abs_mm"] = float(max(p95s))
    out["fixed_any_axis_bias_gt2mm"] = bool(any(fixed))
    return out


def metrics(
    *,
    beta_true: np.ndarray,
    beta_pred: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    xyz_true: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
) -> dict[str, Any]:
    beta_err = np.abs(beta_true - beta_pred) * 180.0 / math.pi
    theta_err = np.abs(theta_true - theta_pred) * 180.0 / math.pi
    xyz_pred = fk_dh_batch(theta_pred, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    ee = np.linalg.norm(xyz_pred - xyz_true, axis=1) * 1000.0
    out: dict[str, Any] = {
        "beta_mae_deg": float(np.mean(beta_err)),
        "beta_p95_deg": float(np.percentile(beta_err, 95)),
        "theta_mae_deg": float(np.mean(theta_err)),
        "theta_p95_deg": float(np.percentile(theta_err, 95)),
        "ee_mean_mm": float(np.mean(ee)),
        "ee_p95_mm": float(np.percentile(ee, 95)),
        "ee_max_mm": float(np.max(ee)),
    }
    out.update(axis_metrics(xyz_true, xyz_pred))
    return out


def run_one(df: pd.DataFrame, args: argparse.Namespace, split: str, model_name: str, *, lengths_m: np.ndarray, p_end_local_m: np.ndarray, theta_sign: float) -> dict[str, Any]:
    train_idx, val_idx, test_idx = make_split(df, split, seed=int(args.seed), val_size=float(args.val_size), test_size=float(args.test_size))
    xyz = df[XYZ_COLS].to_numpy(dtype=float)
    features, feature_names = build_features(xyz, feature_set=str(args.feature_set))
    beta_true = df[BETA_COLS].to_numpy(dtype=float)
    theta_true = df[THETA_COLS].to_numpy(dtype=float)
    target = theta_true if model_name == "mlp_theta30_direct" else beta_true
    model = model_for(model_name, seed=int(args.seed), max_iter=int(args.max_iter), batch_size=int(args.batch_size))
    x_scaler, y_scaler, model, fit_s = fit_scaled(model, features[train_idx], target[train_idx])
    pred_target = {name: predict_scaled(x_scaler, y_scaler, model, features[idx]) for name, idx in {"val": val_idx, "test": test_idx}.items()}
    metrics_by_split = {}
    preds_to_save: dict[str, pd.DataFrame] = {}
    for part_name, idx in {"val": val_idx, "test": test_idx}.items():
        if model_name == "mlp_theta30_direct":
            theta_pred = pred_target[part_name]
            beta_pred = np.full((len(idx), 6), np.nan, dtype=float)
        else:
            beta_pred = pred_target[part_name]
            theta_pred = theta_from_beta_batch(beta_pred, theta_sign=theta_sign)
        metrics_by_split[part_name] = metrics(
            beta_true=beta_true[idx],
            beta_pred=beta_pred if model_name != "mlp_theta30_direct" else beta_true[idx],
            theta_true=theta_true[idx],
            theta_pred=theta_pred,
            xyz_true=xyz[idx],
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
        )
        metrics_by_split[part_name]["rows"] = int(len(idx))
        metrics_by_split[part_name]["fit_s"] = fit_s
        pred_df = df.iloc[idx][XYZ_COLS].copy().reset_index(drop=True)
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
            "kind": "beta6_pose" if model_name != "mlp_theta30_direct" else "theta30_direct_pose",
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "input_cols": XYZ_COLS,
            "feature_set": str(args.feature_set),
            "feature_names": feature_names,
            "target_cols": THETA_COLS if model_name == "mlp_theta30_direct" else BETA_COLS,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "model_checkpoints"
    pred_dir = out_dir / "prediction_parquets"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.robot_config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    df = pd.read_parquet(args.dataset)
    if int(args.max_rows) > 0 and len(df) > int(args.max_rows):
        df = df.sample(n=int(args.max_rows), random_state=int(args.seed)).reset_index(drop=True)
    missing = [c for c in XYZ_COLS + BETA_COLS + THETA_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"dataset missing columns: {missing}")
    rows = []
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
            )
            key = f"{split}_{model_name}"
            (ckpt_dir / key).mkdir(parents=True, exist_ok=True)
            dump(result["package"], ckpt_dir / key / "model.joblib")
            for pred_split, pred_df in result["predictions"].items():
                pred_path = pred_dir / f"{key}_{pred_split}.parquet"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_df.to_parquet(pred_path, index=False, compression="zstd")
            details[key] = {k: v for k, v in result.items() if k not in {"package", "predictions"}}
            for eval_split, m in result["metrics"].items():
                rows.append({"split": split, "model": model_name, "eval_split": eval_split, **m})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_flat.csv", index=False)
    (out_dir / "all_metrics.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": str(out_dir), "rows": rows}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train xyz->beta6->theta30 pose baselines.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="mlp_beta6,mlp_beta6_large,mlp_theta30_direct")
    ap.add_argument("--splits", default="iid,x_slab")
    ap.add_argument("--feature-set", default="poly_heavy", choices=["raw", "poly_medium", "poly_heavy"])
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260707)
    return ap.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({"out_dir": payload["out_dir"], "metric_rows": len(payload["rows"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
