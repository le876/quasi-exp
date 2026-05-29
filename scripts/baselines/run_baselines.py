#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lgbm_device_fallback import (
    force_lgbm_params_to_cpu,
    is_lgbm_gpu_requested,
    is_lgbm_gpu_runtime_error,
    probe_lgbm_gpu_available,
)
from features import build_features
from tf_models import TFDualHeadRegressor, configure_tf_runtime, tensorflow_gpu_status


def _try_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _read_parquet(path):
    import pyarrow.parquet as pq  # local import for env flexibility

    table = pq.read_table(str(path))
    return table.to_pandas()

def _write_parquet(df, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path), compression="zstd")


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def _col_group(cols, prefix, suffix):
    out = [c for c in cols if c.startswith(prefix) and c.endswith(suffix)]

    def key(c):
        mid = c[len(prefix) : -len(suffix)]
        try:
            return int(mid.strip("_"))
        except Exception:
            return 10**9

    return sorted(out, key=key)


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.square(a - b))))


def _mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def _package_versions():
    import importlib

    pkgs = ["numpy", "scipy", "pandas", "pyarrow", "sklearn"]
    out = {}
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            out[p] = getattr(m, "__version__", "unknown")
        except Exception:
            out[p] = None
    # optional
    for p in ["lightgbm", "torch", "tensorflow"]:
        try:
            m = importlib.import_module(p)
            out[p] = getattr(m, "__version__", "unknown")
        except Exception:
            out[p] = None
    return out


class XYScaler(object):
    """
    Z-score for X, and group-wise scaling for Y:
      - theta: z-score per-dim
      - tension: first divide by tmax, then z-score per-dim
    """

    def __init__(self, tmax):
        self.tmax = float(tmax)
        self.x_mean_ = None
        self.x_std_ = None
        self.th_mean_ = None
        self.th_std_ = None
        self.t_mean_ = None
        self.t_std_ = None

    def fit(self, X, theta, tension):
        X = np.asarray(X, dtype=float)
        theta = np.asarray(theta, dtype=float)
        tension = np.asarray(tension, dtype=float)

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-12

        self.th_mean_ = theta.mean(axis=0)
        self.th_std_ = theta.std(axis=0) + 1e-12

        t_scaled = tension / self.tmax
        self.t_mean_ = t_scaled.mean(axis=0)
        self.t_std_ = t_scaled.std(axis=0) + 1e-12
        return self

    def transform_X(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.x_mean_) / self.x_std_

    def transform_Y(self, theta, tension):
        theta = np.asarray(theta, dtype=float)
        tension = np.asarray(tension, dtype=float)

        th = (theta - self.th_mean_) / self.th_std_
        t_scaled = tension / self.tmax
        tt = (t_scaled - self.t_mean_) / self.t_std_
        return th, tt

    def inverse_Y(self, th_norm, tt_norm):
        th_norm = np.asarray(th_norm, dtype=float)
        tt_norm = np.asarray(tt_norm, dtype=float)
        theta = th_norm * self.th_std_ + self.th_mean_
        t_scaled = tt_norm * self.t_std_ + self.t_mean_
        tension = t_scaled * self.tmax
        return theta, tension


def _load_split_npz(path):
    d = np.load(str(path), allow_pickle=False)
    return d["train_idx"].astype(np.int64), d["val_idx"].astype(np.int64), d["test_idx"].astype(np.int64)


def _save_split_npz(path, train_idx, val_idx, test_idx):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), train_idx=np.asarray(train_idx, dtype=np.int64), val_idx=np.asarray(val_idx, dtype=np.int64), test_idx=np.asarray(test_idx, dtype=np.int64))


def _load_robot_paths(robot_config, lengths_csv, end_effector_csv):
    if lengths_csv and end_effector_csv:
        return Path(lengths_csv), Path(end_effector_csv)

    if robot_config:
        y = _try_import_yaml()
        if y is None:
            raise SystemExit("需要 PyYAML 才能从 --robot-config 解析 lengths/end_effector 路径；或直接传 --lengths-csv/--end-effector-csv")
        cfg = y.safe_load(Path(robot_config).read_text(encoding="utf-8"))
        lengths_csv = cfg["paths"]["lengths_csv"]
        end_effector_csv = cfg["paths"]["end_effector_csv"]
        return Path(lengths_csv), Path(end_effector_csv)

    raise SystemExit("必须提供 --robot-config 或同时提供 --lengths-csv/--end-effector-csv")


def _read_lengths_end(lengths_csv, end_effector_csv):
    ldf = pd.read_csv(lengths_csv)
    emap = dict(zip(ldf["name"].astype(str), ldf["value_m"].astype(float)))
    # infer kD from max l index
    kD = max(int(k[1:]) for k in emap.keys() if k.startswith("l"))
    lengths = np.zeros(kD + 1, dtype=float)
    for i in range(kD + 1):
        lengths[i] = float(emap["l%d" % i])

    edf = pd.read_csv(end_effector_csv)
    p_end = np.array([float(edf["p_end_x_m"][0]), float(edf["p_end_y_m"][0]), float(edf["p_end_z_m"][0]), 1.0], dtype=float)
    return lengths, p_end


def _ee_errors_mm(theta_pred_rad, X_xyz_m, lengths_m, p_end_local_m):
    from fk_dh_numpy import fk_dh_batch  # local file import

    p_pred = fk_dh_batch(theta_pred_rad, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    d = p_pred - np.asarray(X_xyz_m, dtype=float)
    return np.linalg.norm(d, axis=1) * 1000.0


def _subsample(X, y, seed, max_n):
    n = int(X.shape[0])
    if max_n is None:
        idx = np.arange(n, dtype=np.int64)
        return X, y, idx
    max_n = int(max_n)
    if n <= max_n:
        idx = np.arange(n, dtype=np.int64)
        return X, y, idx
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=max_n, replace=False).astype(np.int64)
    return X[idx], y[idx], idx


def _fit_predict(model, X_train, y_train, X_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    t_pred = time.perf_counter() - t1
    return y_pred, t_fit, t_pred


def _model_builders(seed, n_jobs, wrapper_n_jobs, backend, tf_context):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    out = {}

    if backend in {"classic", "both"}:
        out["mlp"] = lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-6,
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=int(seed),
            early_stopping=True,
            n_iter_no_change=20,
            verbose=True,
        )
        out["mlp_large"] = lambda: MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-6,
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=int(seed),
            early_stopping=True,
            n_iter_no_change=20,
            verbose=True,
        )

        out["knn"] = lambda: KNeighborsRegressor(n_neighbors=10, weights="distance")

        out["rf"] = lambda: RandomForestRegressor(
            n_estimators=400,
            random_state=int(seed),
            n_jobs=int(n_jobs),
            max_features="sqrt",
            min_samples_leaf=1,
        )

        out["svr_rbf"] = lambda: MultiOutputRegressor(
            SVR(kernel="rbf", C=10.0, epsilon=0.02, gamma="scale"),
            n_jobs=int(wrapper_n_jobs),
        )

        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        out["gpr_rbf"] = lambda: MultiOutputRegressor(
            GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, random_state=int(seed)),
            n_jobs=int(wrapper_n_jobs),
        )

        try:
            import lightgbm as lgb  # type: ignore

            out["lgbm"] = lambda: MultiOutputRegressor(
                lgb.LGBMRegressor(
                    n_estimators=2000,
                    learning_rate=0.03,
                    num_leaves=63,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=int(seed),
                    n_jobs=int(n_jobs),
                ),
                n_jobs=int(wrapper_n_jobs),
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor

            out["lgbm"] = lambda: MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=400,
                    random_state=int(seed),
                ),
                n_jobs=int(wrapper_n_jobs),
            )

    if backend in {"tf", "both"}:
        theta_dim = int(tf_context["theta_dim"])
        tension_dim = int(tf_context["tension_dim"])
        tension_lower_norm = np.asarray(tf_context["tension_lower_norm"], dtype=float)
        tension_upper_norm = np.asarray(tf_context["tension_upper_norm"], dtype=float)
        loss_lambda_theta = float(tf_context["loss_lambda_theta"])
        loss_lambda_tension = float(tf_context["loss_lambda_tension"])
        loss_lambda_phys = float(tf_context["loss_lambda_phys"])
        tf_epochs = int(tf_context["tf_epochs"])
        tf_batch_size = int(tf_context["tf_batch_size"])
        tf_patience = int(tf_context["tf_patience"])
        tf_tol = float(tf_context["tf_tol"])
        tf_verbose = int(tf_context["tf_verbose"])

        tf_small = lambda: TFDualHeadRegressor(
            hidden_layer_sizes=(128, 64, 32),
            alpha=1e-6,
            learning_rate_init=1e-3,
            batch_size=tf_batch_size,
            max_iter=tf_epochs,
            n_iter_no_change=tf_patience,
            tol=tf_tol,
            theta_dim=theta_dim,
            tension_dim=tension_dim,
            loss_lambda_theta=loss_lambda_theta,
            loss_lambda_tension=loss_lambda_tension,
            loss_lambda_phys=loss_lambda_phys,
            tension_lower_norm=tension_lower_norm,
            tension_upper_norm=tension_upper_norm,
            random_state=int(seed),
            validation_split=0.1,
            verbose=tf_verbose,
        )
        tf_large = lambda: TFDualHeadRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            alpha=1e-6,
            learning_rate_init=1e-3,
            batch_size=tf_batch_size,
            max_iter=tf_epochs,
            n_iter_no_change=tf_patience,
            tol=tf_tol,
            theta_dim=theta_dim,
            tension_dim=tension_dim,
            loss_lambda_theta=loss_lambda_theta,
            loss_lambda_tension=loss_lambda_tension,
            loss_lambda_phys=loss_lambda_phys,
            tension_lower_norm=tension_lower_norm,
            tension_upper_norm=tension_upper_norm,
            random_state=int(seed),
            validation_split=0.1,
            verbose=tf_verbose,
        )

        out["tf_mlp"] = tf_small
        out["tf_mlp_large"] = tf_large
        if backend == "tf":
            out["mlp"] = tf_small
            out["mlp_large"] = tf_large

    return out


def _apply_overrides(model, overrides):
    """
    Apply constructor-style overrides when possible.
    For sklearn estimators: set_params covers most cases.
    """
    if not overrides:
        return model
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a dict")
    try:
        model.set_params(**overrides)
    except Exception:
        # best-effort fallback: try attribute set
        for k, v in overrides.items():
            try:
                setattr(model, k, v)
            except Exception:
                pass
    return model


def _env_force_cpu() -> bool:
    return str(os.environ.get("LGBM_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_params_file(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise SystemExit("missing params file: %s" % path)
    if path.suffix.lower() in [".yaml", ".yml"]:
        y = _try_import_yaml()
        if y is None:
            raise SystemExit("需要 PyYAML 才能读取 YAML params 文件：%s" % path)
        return y.safe_load(path.read_text(encoding="utf-8")) or {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_block(th_true, th_pred, t_true, t_pred, tmax, fk_payload=None):
    th_true = np.asarray(th_true, dtype=float)
    th_pred = np.asarray(th_pred, dtype=float)
    t_true = np.asarray(t_true, dtype=float)
    t_pred = np.asarray(t_pred, dtype=float)

    th_true_deg = th_true * 180.0 / math.pi
    th_pred_deg = th_pred * 180.0 / math.pi

    m = {}
    m["theta_mae_rad"] = _mae(th_true, th_pred)
    m["theta_rmse_rad"] = _rmse(th_true, th_pred)
    m["theta_mae_deg"] = _mae(th_true_deg, th_pred_deg)
    m["theta_rmse_deg"] = _rmse(th_true_deg, th_pred_deg)
    m["tension_mae_n"] = _mae(t_true, t_pred)
    m["tension_rmse_n"] = _rmse(t_true, t_pred)
    m["tension_lt0_ratio"] = float(np.mean(t_pred < 0.0))
    m["tension_gt_tmax_ratio"] = float(np.mean(t_pred > float(tmax)))

    if fk_payload is not None:
        ee_err_mm = fk_payload
        m["ee_pos_rmse_mm"] = float(np.sqrt(np.mean(np.square(ee_err_mm))))
        m["ee_pos_p95_mm"] = float(np.quantile(ee_err_mm, 0.95))
        m["ee_pos_max_mm"] = float(np.max(ee_err_mm))

    # unified score (dimensionless) for tuning: theta normalized by 20deg, tension by tmax
    m["score_val_like"] = float(m["theta_mae_deg"] / 20.0 + m["tension_mae_n"] / float(tmax))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260207)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--tmax", type=float, default=2000.0)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for sklearn/lightgbm where supported (-1 = all cores).",
    )
    ap.add_argument(
        "--wrapper-n-jobs",
        type=int,
        default=1,
        help="MultiOutputRegressor n_jobs (recommend 1 to avoid process/thread explosion).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="iid",
        choices=["iid", "radius"],
        help="Dataset split strategy: iid=random; radius=hold out largest ||p|| as val/test.",
    )
    ap.add_argument("--split-file", type=str, default=None, help="NPZ with train_idx/val_idx/test_idx (fixed split for comparable trials).")
    ap.add_argument("--save-split-file", type=str, default=None, help="Where to save the generated split NPZ (only when --split-file not provided).")
    ap.add_argument(
        "--eval-splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to evaluate: train,val,test",
    )
    ap.add_argument(
        "--summary-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to store in top-level all_metrics.json for bar charts.",
    )
    ap.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="YAML/JSON mapping model_key -> dict of estimator params (applied via set_params).",
    )
    ap.add_argument("--tag", type=str, default=None, help="Optional tag recorded in meta.json (e.g. trial_01)")

    ap.add_argument("--robot-config", type=str, default=None, help="YAML config to locate lengths/end_effector CSVs (optional)")
    ap.add_argument("--lengths-csv", type=str, default=None)
    ap.add_argument("--end-effector-csv", type=str, default=None)
    ap.add_argument("--skip-fk", action="store_true", help="Skip end-effector FK back-check metrics")
    ap.add_argument(
        "--save-preds",
        type=int,
        default=2000,
        help="Save N test predictions per model for plotting (0 disables).",
    )
    ap.add_argument(
        "--save-preds-splits",
        type=str,
        default="val,test",
        help="Comma-separated splits to save preds for plotting: val,test,train",
    )
    ap.add_argument(
        "--save-curves",
        action="store_true",
        help="Save train curves if the model exposes them (MLP supported).",
    )
    ap.add_argument(
        "--feature-set",
        type=str,
        default="poly_heavy",
        choices=["raw", "poly_medium", "poly_heavy"],
        help="Feature engineering set for task-space input.",
    )
    ap.add_argument(
        "--feature-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used in feature engineering.",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="tf",
        choices=["tf", "classic", "both"],
        help="Training backend family.",
    )
    ap.add_argument(
        "--tf-device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="TensorFlow device preference.",
    )
    ap.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast when TensorFlow cannot see any GPU.",
    )
    ap.add_argument("--tf-epochs", type=int, default=400)
    ap.add_argument("--tf-batch-size", type=int, default=1024)
    ap.add_argument("--tf-patience", type=int, default=30)
    ap.add_argument("--tf-tol", type=float, default=1e-6)
    ap.add_argument("--tf-verbose", type=int, default=1)
    ap.add_argument("--loss-lambda-theta", type=float, default=1.0)
    ap.add_argument("--loss-lambda-tension", type=float, default=1.0)
    ap.add_argument("--loss-lambda-phys", type=float, default=0.05)

    ap.add_argument(
        "--models",
        type=str,
        default="tf_mlp,tf_mlp_large",
        help="Comma-separated model keys.",
    )
    ap.add_argument("--svr-max-train", type=int, default=20000)
    ap.add_argument(
        "--gpr-max-train",
        type=int,
        default=800,
        help="GPR训练子采样上限（极耗时；建议<=800，并且通常只作为小数据基线）",
    )

    args = ap.parse_args()
    _ensure_dir(args.out_dir)

    eval_splits = [s.strip() for s in str(args.eval_splits).split(",") if s.strip()]
    for s in eval_splits:
        if s not in ["train", "val", "test"]:
            raise SystemExit("invalid --eval-splits: %s" % args.eval_splits)
    save_pred_splits = [s.strip() for s in str(args.save_preds_splits).split(",") if s.strip()]
    for s in save_pred_splits:
        if s not in ["train", "val", "test"]:
            raise SystemExit("invalid --save-preds-splits: %s" % args.save_preds_splits)

    params_map = _read_params_file(args.params_file)
    if params_map and not isinstance(params_map, dict):
        raise SystemExit("--params-file must contain a dict mapping model_key -> params dict")

    df = _read_parquet(args.dataset)
    cols = list(df.columns)
    theta_cols = _col_group(cols, "theta_", "_rad")
    tension_cols = _col_group(cols, "tension_", "_n")
    if len(theta_cols) != 30 or len(tension_cols) != 12:
        raise SystemExit("dataset列不符合预期：需要30个theta_*_rad与12个tension_*_n")

    X_raw = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    X, feature_names = build_features(X_raw, feature_set=args.feature_set, eps=float(args.feature_eps))
    theta = df[theta_cols].to_numpy(dtype=float)
    tension = df[tension_cols].to_numpy(dtype=float)

    from splits import split_iid, split_radius_holdout

    if args.split_file:
        train_idx, val_idx, test_idx = _load_split_npz(args.split_file)
    else:
        if args.split == "iid":
            train_idx, val_idx, test_idx = split_iid(
                len(df),
                seed=args.seed,
                val_size=args.val_size,
                test_size=args.test_size,
            )
        else:
            train_idx, val_idx, test_idx = split_radius_holdout(
                X_raw,
                val_size=args.val_size,
                test_size=args.test_size,
            )
        if args.save_split_file:
            _save_split_npz(args.save_split_file, train_idx, val_idx, test_idx)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    X_raw_train, X_raw_val, X_raw_test = X_raw[train_idx], X_raw[val_idx], X_raw[test_idx]
    th_train, th_val, th_test = theta[train_idx], theta[val_idx], theta[test_idx]
    t_train, t_val, t_test = tension[train_idx], tension[val_idx], tension[test_idx]

    scaler = XYScaler(tmax=args.tmax).fit(X_train, th_train, t_train)
    X_train_n = scaler.transform_X(X_train)
    X_val_n = scaler.transform_X(X_val)
    X_test_n = scaler.transform_X(X_test)
    th_train_n, t_train_n = scaler.transform_Y(th_train, t_train)
    th_val_n, t_val_n = scaler.transform_Y(th_val, t_val)
    th_test_n, t_test_n = scaler.transform_Y(th_test, t_test)

    y_train_n = np.concatenate([th_train_n, t_train_n], axis=1)
    y_val_n = np.concatenate([th_val_n, t_val_n], axis=1)
    y_test_n = np.concatenate([th_test_n, t_test_n], axis=1)

    tf_status = None
    tf_runtime_error = None
    if args.backend in {"tf", "both"}:
        try:
            _ = tensorflow_gpu_status()
            tf_status = configure_tf_runtime(
                tf_device=str(args.tf_device),
                require_gpu=bool(args.require_gpu),
                seed=int(args.seed),
            )
        except Exception as exc:
            tf_runtime_error = f"{type(exc).__name__}: {exc}"
            if args.backend == "tf":
                raise SystemExit(f"TensorFlow runtime init failed: {tf_runtime_error}")

    tension_lower_norm = (0.0 - scaler.t_mean_) / scaler.t_std_
    tension_upper_norm = (1.0 - scaler.t_mean_) / scaler.t_std_
    tf_context = {
        "theta_dim": 30,
        "tension_dim": 12,
        "tension_lower_norm": np.asarray(tension_lower_norm, dtype=float),
        "tension_upper_norm": np.asarray(tension_upper_norm, dtype=float),
        "loss_lambda_theta": float(args.loss_lambda_theta),
        "loss_lambda_tension": float(args.loss_lambda_tension),
        "loss_lambda_phys": float(args.loss_lambda_phys),
        "tf_epochs": int(args.tf_epochs),
        "tf_batch_size": int(args.tf_batch_size),
        "tf_patience": int(args.tf_patience),
        "tf_tol": float(args.tf_tol),
        "tf_verbose": int(args.tf_verbose),
    }

    # robot lengths/end for FK back-check
    lengths_m = None
    p_end_local_m = None
    if not args.skip_fk:
        lengths_csv, ee_csv = _load_robot_paths(args.robot_config, args.lengths_csv, args.end_effector_csv)
        lengths_m, p_end_local_m = _read_lengths_end(lengths_csv, ee_csv)

    builders = _model_builders(
        seed=args.seed,
        n_jobs=args.n_jobs,
        wrapper_n_jobs=args.wrapper_n_jobs,
        backend=str(args.backend),
        tf_context=tf_context,
    )
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.backend == "tf":
        non_tf = [m for m in model_keys if not (m.startswith("tf_") or m in {"mlp", "mlp_large"})]
        if non_tf:
            raise SystemExit(f"backend=tf does not support non-TF model keys: {non_tf}")

    meta = {
        "dataset": str(args.dataset),
        "out_dir": str(args.out_dir),
        "seed": int(args.seed),
        "tag": args.tag,
        "split": {
            "type": str(args.split),
            "val_size": float(args.val_size),
            "test_size": float(args.test_size),
            "split_file": args.split_file,
            "saved_split_file": args.save_split_file,
        },
        "sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "tmax": float(args.tmax),
        "summary_split": str(args.summary_split),
        "eval_splits": eval_splits,
        "save_preds_splits": save_pred_splits,
        "n_jobs": int(args.n_jobs),
        "wrapper_n_jobs": int(args.wrapper_n_jobs),
        "params_file": args.params_file,
        "backend": str(args.backend),
        "feature": {
            "feature_set": str(args.feature_set),
            "feature_eps": float(args.feature_eps),
            "feature_dim": int(X.shape[1]),
            "feature_names": feature_names,
        },
        "tensorflow": {
            "tf_device": str(args.tf_device),
            "require_gpu": bool(args.require_gpu),
            "gpu_count": None if tf_status is None else int(tf_status.gpu_count),
            "gpu_names": [] if tf_status is None else list(tf_status.gpu_names),
            "runtime_error": tf_runtime_error,
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # save scalers for reuse
    try:
        import joblib

        joblib.dump(scaler, args.out_dir / "scaler.joblib")
    except Exception:
        pass

    results = {}
    lgbm_gpu_probe_done = False
    lgbm_gpu_probe_ok = True
    lgbm_gpu_probe_reason = "not_checked"
    force_cpu = _env_force_cpu()

    for key in model_keys:
        if key not in builders:
            print("SKIP unknown model:", key)
            continue

        print("\n=== MODEL:", key, "===")
        out_m = args.out_dir / key
        _ensure_dir(out_m)

        model = builders[key]()
        overrides = params_map.get(key, None) if isinstance(params_map, dict) else None
        effective_overrides = overrides
        if key == "lgbm":
            if force_cpu and isinstance(effective_overrides, dict):
                effective_overrides = force_lgbm_params_to_cpu(effective_overrides)
            elif isinstance(effective_overrides, dict) and is_lgbm_gpu_requested(effective_overrides):
                if not lgbm_gpu_probe_done:
                    lgbm_gpu_probe_ok, lgbm_gpu_probe_reason = probe_lgbm_gpu_available()
                    lgbm_gpu_probe_done = True
                if not lgbm_gpu_probe_ok:
                    print(
                        "[WARN] LightGBM GPU unavailable, switch to CPU fallback:",
                        lgbm_gpu_probe_reason,
                    )
                    effective_overrides = force_lgbm_params_to_cpu(effective_overrides)

        if effective_overrides:
            model = _apply_overrides(model, effective_overrides)

        # heavy models: subsample training
        max_train = None
        if key.startswith("svr"):
            max_train = args.svr_max_train
        if key.startswith("gpr"):
            max_train = args.gpr_max_train

        Xtr, ytr, idx_tr = _subsample(X_train_n, y_train_n, seed=args.seed + 11, max_n=max_train)
        Xtr_phys = X_raw_train[idx_tr]
        # fit
        t0 = time.perf_counter()
        try:
            model.fit(Xtr, ytr)
        except Exception as exc:
            can_retry = (
                key == "lgbm"
                and (
                    force_cpu
                    or is_lgbm_gpu_runtime_error(exc)
                )
            )
            if not can_retry:
                raise

            if not force_cpu and not lgbm_gpu_probe_done:
                lgbm_gpu_probe_ok, lgbm_gpu_probe_reason = probe_lgbm_gpu_available()
                lgbm_gpu_probe_done = True
            retry_reason = "LGBM_FORCE_CPU is set" if force_cpu else str(exc)
            print("[WARN] LightGBM fit failed on GPU path, retry with CPU fallback:", retry_reason)
            model = builders[key]()
            retry_overrides = force_lgbm_params_to_cpu(effective_overrides)
            if retry_overrides:
                model = _apply_overrides(model, retry_overrides)
            model.fit(Xtr, ytr)
            effective_overrides = retry_overrides
        t_fit_s = time.perf_counter() - t0

        # pred timing (on test, for comparability)
        t1 = time.perf_counter()
        _ = model.predict(X_test_n)
        t_pred_s = time.perf_counter() - t1

        # evaluate splits
        metrics_by_split = {}
        preds_cache = {}

        split_payloads = {
            "train": (Xtr, ytr),  # train metrics on the actually-fitted subset
            "val": (X_val_n, y_val_n),
            "test": (X_test_n, y_test_n),
        }
        split_truth_phys = {"val": (th_val, t_val), "test": (th_test, t_test)}

        for sp in eval_splits:
            Xn, yn = split_payloads[sp]
            y_pred_n = model.predict(Xn)
            th_pred_n = y_pred_n[:, :30]
            t_pred_n = y_pred_n[:, 30:]
            th_pred, t_pred = scaler.inverse_Y(th_pred_n, t_pred_n)

            if sp == "train":
                th_true_phys, t_true_phys = scaler.inverse_Y(yn[:, :30], yn[:, 30:])  # recover physical truth for fitted subset
            else:
                th_true_phys, t_true_phys = split_truth_phys[sp]

            fk_payload = None
            if (not args.skip_fk) and sp == "test":
                fk_payload = _ee_errors_mm(th_pred, X_raw_test, lengths_m=lengths_m, p_end_local_m=p_end_local_m)

            mb = _metrics_block(th_true_phys, th_pred, t_true_phys, t_pred, tmax=args.tmax, fk_payload=fk_payload)
            mb["rows"] = int(Xn.shape[0])
            metrics_by_split[sp] = mb
            preds_cache[sp] = (th_true_phys, th_pred, t_true_phys, t_pred, fk_payload)

        # persist metrics
        out_metrics = {
            "train_rows_used": int(Xtr.shape[0]),
            "fit_time_s": float(t_fit_s),
            "pred_time_s": float(t_pred_s),
            "pred_time_ms_per_sample": float(1000.0 * t_pred_s / max(1, X_test_n.shape[0])),
            "metrics": metrics_by_split,
            "overrides": effective_overrides,
        }
        (out_m / "metrics.json").write_text(json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        # choose summary split for all_metrics.json
        summary = dict(metrics_by_split.get(args.summary_split, {}))
        summary["train_rows_used"] = int(Xtr.shape[0])
        summary["fit_time_s"] = float(t_fit_s)
        summary["pred_time_s"] = float(t_pred_s)
        summary["pred_time_ms_per_sample"] = float(1000.0 * t_pred_s / max(1, X_test_n.shape[0]))
        results[key] = summary

        # Save predictions for plotting
        if int(args.save_preds) > 0:
            rng = np.random.default_rng(int(args.seed) + 1000 + (abs(hash(key)) % 100000))
            for sp in save_pred_splits:
                if sp not in preds_cache:
                    continue
                if sp == "train":
                    X_phys = Xtr_phys
                elif sp == "val":
                    X_phys = X_raw_val
                else:
                    X_phys = X_raw_test

                th_true_sp, th_pred_sp, t_true_sp, t_pred_sp, fk_payload = preds_cache[sp]

                n_rows = int(X_phys.shape[0])
                n_save = min(int(args.save_preds), n_rows)
                pick = rng.choice(n_rows, size=n_save, replace=False)

                cols_out = {}
                cols_out["x_m"] = X_phys[pick, 0]
                cols_out["y_m"] = X_phys[pick, 1]
                cols_out["z_m"] = X_phys[pick, 2]

                for i in range(30):
                    cols_out["theta_true_%d_rad" % (i + 1)] = th_true_sp[pick, i]
                    cols_out["theta_pred_%d_rad" % (i + 1)] = th_pred_sp[pick, i]
                for j in range(12):
                    cols_out["tension_true_%d_n" % (j + 1)] = t_true_sp[pick, j]
                    cols_out["tension_pred_%d_n" % (j + 1)] = t_pred_sp[pick, j]
                if fk_payload is not None and sp == "test":
                    cols_out["ee_err_mm"] = fk_payload[pick]

                _write_parquet(pd.DataFrame(cols_out), out_m / ("preds_%s.parquet" % sp))

        # Save train curves when available
        if bool(args.save_curves):
            curves = {}
            try:
                # sklearn MLPRegressor
                if hasattr(model, "loss_curve_"):
                    curves["loss_curve"] = [float(x) for x in getattr(model, "loss_curve_")]
                if hasattr(model, "validation_scores_"):
                    curves["validation_scores"] = [float(x) for x in getattr(model, "validation_scores_")]
                if hasattr(model, "n_iter_"):
                    curves["n_iter"] = int(getattr(model, "n_iter_"))
            except Exception:
                curves = {"error": "failed_to_extract"}
            (out_m / "curves.json").write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            import joblib

            joblib.dump(model, out_m / "model.joblib")
        except Exception:
            pass

    (args.out_dir / "all_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nWROTE:", args.out_dir / "all_metrics.json")


if __name__ == "__main__":
    main()
