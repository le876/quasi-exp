#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from splits import (  # noqa: E402
    split_angular_sector_holdout,
    split_beta_block_holdout,
    split_iid,
    split_radius_holdout,
)


XYZ_COLS = ["x_m", "y_m", "z_m"]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]


def _numbered_cols(cols: list[str], prefix: str, suffix: str) -> list[str]:
    out = [c for c in cols if c.startswith(prefix) and c.endswith(suffix)]

    def key(c: str) -> int:
        mid = c[len(prefix) : -len(suffix)]
        try:
            return int(mid.strip("_"))
        except Exception:
            return 10**9

    return sorted(out, key=key)


def split_x_slab_holdout(X_xyz_m: np.ndarray, *, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X_xyz_m, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X_xyz_m must have shape (N,3)")
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("n must be > 0")
    if float(val_size) < 0.0 or float(test_size) < 0.0 or float(val_size) + float(test_size) >= 1.0:
        raise ValueError("val_size+test_size must be in [0,1)")
    idx = np.argsort(X[:, 0]).astype(np.int64)
    n_test = int(round(n * float(test_size)))
    n_val = int(round(n * float(val_size)))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("train split would be empty")
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def theta_metrics(theta_true: np.ndarray, theta_pred: np.ndarray, ee_err_mm: np.ndarray | None = None) -> dict[str, float]:
    true_deg = np.asarray(theta_true, dtype=float) * 180.0 / math.pi
    pred_deg = np.asarray(theta_pred, dtype=float) * 180.0 / math.pi
    err_deg = pred_deg - true_deg
    out = {
        "theta_mae_deg": float(np.mean(np.abs(err_deg))),
        "theta_rmse_deg": float(np.sqrt(np.mean(np.square(err_deg)))),
        "theta_p95_deg": float(np.percentile(np.mean(np.abs(err_deg), axis=1), 95)),
    }
    if ee_err_mm is not None:
        ee = np.asarray(ee_err_mm, dtype=float).reshape(-1)
        out.update(
            {
                "ee_pos_mae_mm": float(np.mean(np.abs(ee))),
                "ee_pos_rmse_mm": float(np.sqrt(np.mean(np.square(ee)))),
                "ee_pos_p95_mm": float(np.percentile(ee, 95)),
                "ee_pos_max_mm": float(np.max(ee)),
            }
        )
    return out


def make_split(
    split: str,
    *,
    X: np.ndarray,
    beta: np.ndarray,
    seed: int,
    val_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = str(split)
    if split == "iid":
        return split_iid(len(X), seed=seed, val_size=val_size, test_size=test_size)
    if split == "x_slab":
        return split_x_slab_holdout(X, val_size=val_size, test_size=test_size)
    if split == "radius":
        return split_radius_holdout(X, val_size=val_size, test_size=test_size)
    if split == "beta_block":
        return split_beta_block_holdout(beta, val_size=val_size, test_size=test_size)
    if split == "angular_sector":
        return split_angular_sector_holdout(X, val_size=val_size, test_size=test_size)
    raise ValueError(f"unsupported split: {split}")


def _model(
    name: str,
    *,
    seed: int,
    max_iter: int,
    alpha: float,
    batch_size: int,
    learning_rate_init: float,
    n_jobs: int = 1,
    lgbm_n_estimators: int = 1200,
    lgbm_learning_rate: float = 0.03,
    lgbm_num_leaves: int = 63,
):
    sizes = {
        "mlp": (256, 128, 64, 32),
        "mlp_l2": (512, 256, 128, 64),
        "mlp_large": (1024, 512, 256, 128),
    }
    if name in sizes:
        return MLPRegressor(
            hidden_layer_sizes=sizes[name],
            activation="relu",
            solver="adam",
            alpha=float(alpha),
            batch_size=int(batch_size),
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            random_state=int(seed),
            early_stopping=True,
            n_iter_no_change=20,
            verbose=False,
        )
    if name == "lgbm":
        from sklearn.multioutput import MultiOutputRegressor

        try:
            import lightgbm as lgb  # type: ignore

            base = lgb.LGBMRegressor(
                n_estimators=int(lgbm_n_estimators),
                learning_rate=float(lgbm_learning_rate),
                num_leaves=int(lgbm_num_leaves),
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=int(seed),
                n_jobs=int(n_jobs),
                verbosity=-1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor

            base = HistGradientBoostingRegressor(
                learning_rate=float(lgbm_learning_rate),
                max_iter=max(20, int(lgbm_n_estimators)),
                max_leaf_nodes=int(lgbm_num_leaves),
                random_state=int(seed),
                l2_regularization=float(alpha),
            )
        return MultiOutputRegressor(base, n_jobs=1)
    raise ValueError(f"unsupported model: {name}")


def _load_dataset(path: Path, *, max_rows: int = 0, seed: int = 20260705) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    if int(max_rows) > 0 and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=int(seed)).sort_index().reset_index(drop=True)
    cols = list(df.columns)
    theta_cols = _numbered_cols(cols, "theta_", "_rad")
    if len(theta_cols) != 30:
        raise ValueError("dataset must contain 30 theta_*_rad columns")
    missing = [c for c in [*XYZ_COLS, *BETA_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    X = df[XYZ_COLS].to_numpy(dtype=float)
    theta = df[theta_cols].to_numpy(dtype=float)
    beta = df[BETA_COLS].to_numpy(dtype=float)
    return df, X, theta, beta


def load_theta_model_package(path: Path) -> dict[str, Any]:
    package = load(path)
    if not isinstance(package, dict):
        raise ValueError("theta model package must be a dict")
    if package.get("kind") != "theta_only":
        raise ValueError("model package kind must be 'theta_only'")
    for key in ("model", "input_cols", "theta_cols"):
        if key not in package:
            raise ValueError(f"theta model package missing {key}")
    return package


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, X, theta, beta = _load_dataset(Path(args.dataset), max_rows=int(args.max_rows), seed=int(args.seed))
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    cfg = load_config(Path(args.robot_config))
    inputs = load_robot_inputs(cfg)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]

    rows: list[dict[str, Any]] = []
    for split in splits:
        train_idx, val_idx, test_idx = make_split(split, X=X, beta=beta, seed=int(args.seed), val_size=float(args.val_size), test_size=float(args.test_size))
        for model_name in models:
            t0 = time.perf_counter()
            reg = _model(
                model_name,
                seed=int(args.seed),
                max_iter=int(args.max_iter),
                alpha=float(args.alpha),
                batch_size=int(args.batch_size),
                learning_rate_init=float(args.learning_rate_init),
                n_jobs=int(getattr(args, "n_jobs", 1)),
                lgbm_n_estimators=int(getattr(args, "lgbm_n_estimators", 1200)),
                lgbm_learning_rate=float(getattr(args, "lgbm_learning_rate", 0.03)),
                lgbm_num_leaves=int(getattr(args, "lgbm_num_leaves", 63)),
            )
            model = Pipeline([("x_scaler", StandardScaler()), ("mlp", reg)])
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(theta[train_idx])
            model.fit(X[train_idx], y_train)
            fit_s = float(time.perf_counter() - t0)
            split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
            split_metrics: dict[str, dict[str, float]] = {}
            for split_name, idx in split_indices.items():
                pred = y_scaler.inverse_transform(model.predict(X[idx]))
                ee_err = None
                if split_name == "test" and not bool(args.skip_fk):
                    achieved = fk_dh_batch(pred, lengths_m=inputs.lengths_m, p_end_local_m=inputs.p_end_local_m)
                    ee_err = np.linalg.norm(achieved - X[idx], axis=1) * 1000.0
                split_metrics[split_name] = theta_metrics(theta[idx], pred, ee_err)
            row = {
                "split": split,
                "model": model_name,
                "fit_s": fit_s,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "train": split_metrics["train"],
                "val": split_metrics["val"],
                "test": split_metrics["test"],
            }
            rows.append(row)
            model_dir = out_dir / split / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            if bool(args.save_model):
                dump(
                    {
                        "kind": "theta_only",
                        "model": model,
                        "y_scaler": y_scaler,
                        "input_cols": XYZ_COLS,
                        "theta_cols": theta_cols,
                        "dataset": str(args.dataset),
                        "robot_config": str(args.robot_config),
                        "model_name": model_name,
                        "split": split,
                    },
                    model_dir / "model.joblib",
                )

    flat = []
    for row in rows:
        flat.append(
            {
                "split": row["split"],
                "model": row["model"],
                "fit_s": row["fit_s"],
                **{f"test_{k}": v for k, v in row["test"].items()},
            }
        )
    payload = {"dataset": str(args.dataset), "rows": int(len(X)), "results": rows, "flat": flat}
    (out_dir / "all_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(flat).to_csv(out_dir / "metrics_flat.csv", index=False)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train theta-only xyz->theta MLP baselines and evaluate FK end-effector error.")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", default="iid,x_slab,radius,beta_block,angular_sector")
    ap.add_argument("--models", default="mlp_l2")
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--learning-rate-init", type=float, default=1e-3)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--lgbm-n-estimators", type=int, default=1200)
    ap.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    ap.add_argument("--lgbm-num-leaves", type=int, default=63)
    ap.add_argument("--skip-fk", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    return ap.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
