#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lgbm_device_fallback import (
    force_lgbm_params_to_cpu,
    is_lgbm_gpu_requested,
    probe_lgbm_gpu_available,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _read_parquet(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pandas()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path), compression="zstd")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _col_group(cols: list[str], prefix: str, suffix: str) -> list[str]:
    out = [c for c in cols if c.startswith(prefix) and c.endswith(suffix)]

    def key(name: str) -> int:
        token = name[len(prefix) : -len(suffix)]
        try:
            return int(token.strip("_"))
        except Exception:
            return 10**9

    return sorted(out, key=key)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _load_split_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(str(path), allow_pickle=False)
    return d["train_idx"].astype(np.int64), d["val_idx"].astype(np.int64), d["test_idx"].astype(np.int64)


def _read_lengths_end(robot_config: Path) -> tuple[np.ndarray, np.ndarray]:
    import yaml

    cfg = yaml.safe_load(robot_config.read_text(encoding="utf-8"))
    lengths_csv = Path(cfg["paths"]["lengths_csv"])
    end_csv = Path(cfg["paths"]["end_effector_csv"])

    ldf = pd.read_csv(lengths_csv)
    lmap = dict(zip(ldf["name"].astype(str), ldf["value_m"].astype(float)))
    kD = max(int(key[1:]) for key in lmap if key.startswith("l"))
    lengths = np.zeros(kD + 1, dtype=float)
    for i in range(kD + 1):
        lengths[i] = float(lmap[f"l{i}"])

    edf = pd.read_csv(end_csv)
    p_end = np.array(
        [float(edf["p_end_x_m"][0]), float(edf["p_end_y_m"][0]), float(edf["p_end_z_m"][0]), 1.0],
        dtype=float,
    )
    return lengths, p_end


def _ee_errors_mm(theta_pred_rad: np.ndarray, xyz_m: np.ndarray, lengths_m: np.ndarray, p_end_local_m: np.ndarray) -> np.ndarray:
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from fk_dh_numpy import fk_dh_batch

    xyz_pred = fk_dh_batch(theta_pred_rad, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    err = xyz_pred - np.asarray(xyz_m, dtype=float)
    return np.linalg.norm(err, axis=1) * 1000.0


def _read_params_file(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("params file must be a dict")
    return data


def _package_versions() -> dict[str, str | None]:
    import importlib

    versions: dict[str, str | None] = {}
    for package in ["numpy", "pandas", "pyarrow", "sklearn", "lightgbm"]:
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = None
    return versions


@dataclass
class XYScaler:
    tmax: float

    def fit(self, X: np.ndarray, theta: np.ndarray, tension: np.ndarray) -> "XYScaler":
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-12
        self.theta_mean_ = theta.mean(axis=0)
        self.theta_std_ = theta.std(axis=0) + 1e-12
        t_scaled = tension / self.tmax
        self.tension_mean_ = t_scaled.mean(axis=0)
        self.tension_std_ = t_scaled.std(axis=0) + 1e-12
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean_) / self.x_std_

    def transform_theta(self, theta: np.ndarray) -> np.ndarray:
        return (theta - self.theta_mean_) / self.theta_std_

    def inverse_theta(self, theta_scaled: np.ndarray) -> np.ndarray:
        return theta_scaled * self.theta_std_ + self.theta_mean_

    def transform_tension(self, tension: np.ndarray) -> np.ndarray:
        t_scaled = tension / self.tmax
        return (t_scaled - self.tension_mean_) / self.tension_std_

    def inverse_tension(self, tension_scaled: np.ndarray) -> np.ndarray:
        t_scaled = tension_scaled * self.tension_std_ + self.tension_mean_
        return t_scaled * self.tmax


def _theta_encode(theta_rad: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return theta_rad
    if mode == "sincos":
        s = np.sin(theta_rad)
        c = np.cos(theta_rad)
        return np.concatenate([s, c], axis=1)
    raise ValueError(f"Unsupported theta_target: {mode}")


def _theta_decode(encoded: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return encoded
    if mode == "sincos":
        half = encoded.shape[1] // 2
        s = encoded[:, :half]
        c = encoded[:, half:]
        return np.arctan2(s, c)
    raise ValueError(f"Unsupported theta_target: {mode}")


def _tension_encode(tension_n: np.ndarray, mode: str, tmax: float) -> np.ndarray:
    if mode == "raw":
        return tension_n
    if mode == "log1p":
        x = np.clip(tension_n, 0.0, tmax)
        return np.log1p(x)
    raise ValueError(f"Unsupported tension_target: {mode}")


def _tension_decode(encoded: np.ndarray, mode: str, tmax: float) -> np.ndarray:
    if mode == "raw":
        return encoded
    if mode == "log1p":
        x = np.expm1(encoded)
        return np.clip(x, 0.0, tmax)
    raise ValueError(f"Unsupported tension_target: {mode}")


def _sample_weights(
    X_xyz_m: np.ndarray,
    theta_rad: np.ndarray,
    tension_n: np.ndarray,
    tmax: float,
    cfg: dict[str, Any],
) -> np.ndarray:
    weights_cfg = cfg.get("weights", {})
    if not isinstance(weights_cfg, dict):
        weights_cfg = {}

    weight = np.ones(X_xyz_m.shape[0], dtype=float)

    theta_coef = float(weights_cfg.get("theta_coef", 0.0))
    theta_power = float(weights_cfg.get("theta_power", 1.0))
    theta_ref_deg = float(weights_cfg.get("theta_ref_deg", 20.0))
    if theta_coef > 0:
        theta_norm = np.mean(np.abs(theta_rad), axis=1) / max(theta_ref_deg * np.pi / 180.0, 1e-9)
        weight += theta_coef * np.power(np.clip(theta_norm, 0.0, None), theta_power)

    tension_coef = float(weights_cfg.get("tension_coef", 0.0))
    tension_power = float(weights_cfg.get("tension_power", 1.0))
    if tension_coef > 0:
        t_norm = np.mean(np.clip(tension_n / max(tmax, 1e-9), 0.0, None), axis=1)
        weight += tension_coef * np.power(t_norm, tension_power)

    edge_coef = float(weights_cfg.get("edge_coef", 0.0))
    edge_power = float(weights_cfg.get("edge_power", 1.0))
    if edge_coef > 0:
        radius = np.linalg.norm(X_xyz_m, axis=1)
        r_max = max(float(np.max(radius)), 1e-9)
        r_norm = radius / r_max
        weight += edge_coef * np.power(r_norm, edge_power)

    return np.clip(weight, 1e-6, None)


def _expand_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(mode).strip().lower()
    if mode in {"none", ""}:
        return X_train, X_val, X_test, {"mode": "none", "n_features_in": int(X_train.shape[1]), "n_features_out": int(X_train.shape[1])}
    if mode in {"poly2", "poly3"}:
        from sklearn.preprocessing import PolynomialFeatures

        deg = 2 if mode == "poly2" else 3
        transformer = PolynomialFeatures(degree=deg, include_bias=False)
        Xt = transformer.fit_transform(X_train)
        Xv = transformer.transform(X_val)
        Xs = transformer.transform(X_test)
        return Xt, Xv, Xs, {"mode": mode, "degree": deg, "n_features_in": int(X_train.shape[1]), "n_features_out": int(Xt.shape[1])}
    raise ValueError(f"Unsupported feature_expansion: {mode}")


def _metrics_block(
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    tension_true: np.ndarray,
    tension_pred: np.ndarray,
    tmax: float,
    ee_err_mm: np.ndarray | None = None,
) -> dict[str, float]:
    theta_true_deg = theta_true * 180.0 / math.pi
    theta_pred_deg = theta_pred * 180.0 / math.pi
    out: dict[str, float] = {
        "theta_mae_rad": _mae(theta_true, theta_pred),
        "theta_rmse_rad": _rmse(theta_true, theta_pred),
        "theta_mae_deg": _mae(theta_true_deg, theta_pred_deg),
        "theta_rmse_deg": _rmse(theta_true_deg, theta_pred_deg),
        "tension_mae_n": _mae(tension_true, tension_pred),
        "tension_rmse_n": _rmse(tension_true, tension_pred),
        "tension_lt0_ratio": float(np.mean(tension_pred < 0.0)),
        "tension_gt_tmax_ratio": float(np.mean(tension_pred > float(tmax))),
    }
    if ee_err_mm is not None:
        out["ee_pos_rmse_mm"] = float(np.sqrt(np.mean(np.square(ee_err_mm))))
        out["ee_pos_p95_mm"] = float(np.quantile(ee_err_mm, 0.95))
        out["ee_pos_max_mm"] = float(np.max(ee_err_mm))
    out["score_val_like"] = float(out["theta_mae_deg"] / 20.0 + out["tension_mae_n"] / max(float(tmax), 1e-9))
    return out


def _make_regressor(base_params: dict[str, Any], n_jobs: int):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    params = dict(base_params)
    params.setdefault("n_jobs", n_jobs)
    reg = LGBMRegressor(**params)
    return MultiOutputRegressor(reg, n_jobs=1)


def _fit_predict_head(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    sample_weight: np.ndarray | None,
    params_stage1: dict[str, Any],
    params_stage2: dict[str, Any] | None,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    stats: dict[str, float] = {}

    model1 = _make_regressor(params_stage1, n_jobs=n_jobs)
    t0 = time.perf_counter()
    if sample_weight is None:
        model1.fit(X_train, y_train)
    else:
        model1.fit(X_train, y_train, sample_weight=sample_weight)
    stats["fit_stage1_s"] = time.perf_counter() - t0

    tp = time.perf_counter()
    pred_train = model1.predict(X_train)
    stats["pred_train_stage1_s"] = time.perf_counter() - tp
    tp = time.perf_counter()
    pred_val = model1.predict(X_val)
    stats["pred_val_stage1_s"] = time.perf_counter() - tp
    tp = time.perf_counter()
    pred_test = model1.predict(X_test)
    stats["pred_test_stage1_s"] = time.perf_counter() - tp

    if params_stage2 is None:
        return pred_train, pred_val, pred_test, stats

    residual_train = y_train - pred_train
    model2 = _make_regressor(params_stage2, n_jobs=n_jobs)
    t1 = time.perf_counter()
    if sample_weight is None:
        model2.fit(X_train, residual_train)
    else:
        model2.fit(X_train, residual_train, sample_weight=sample_weight)
    stats["fit_stage2_s"] = time.perf_counter() - t1

    tp = time.perf_counter()
    pred_train = pred_train + model2.predict(X_train)
    stats["pred_train_stage2_s"] = time.perf_counter() - tp
    tp = time.perf_counter()
    pred_val = pred_val + model2.predict(X_val)
    stats["pred_val_stage2_s"] = time.perf_counter() - tp
    tp = time.perf_counter()
    pred_test = pred_test + model2.predict(X_test)
    stats["pred_test_stage2_s"] = time.perf_counter() - tp
    return pred_train, pred_val, pred_test, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--robot-config", required=True, type=Path)
    ap.add_argument("--split-file", required=True, type=Path)
    ap.add_argument("--params-file", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260209)
    ap.add_argument("--tmax", type=float, default=2000.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save-preds", type=int, default=2000)
    ap.add_argument("--skip-fk", action="store_true")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    _log(f"START out_dir={args.out_dir}")

    params_all = _read_params_file(args.params_file)
    exp_cfg = params_all.get("experiment", {})
    theta_target = str(exp_cfg.get("theta_target", "raw")).strip().lower()
    tension_target = str(exp_cfg.get("tension_target", "raw")).strip().lower()
    use_stage2 = bool(exp_cfg.get("use_stage2", False))
    feature_mode = str(exp_cfg.get("feature_expansion", "none")).strip().lower()

    model_cfg = params_all.get("model", {})
    if not isinstance(model_cfg, dict):
        raise SystemExit("params.model must be a dict")
    theta_params = dict(model_cfg.get("theta", {}))
    tension_params = dict(model_cfg.get("tension", {}))
    if not theta_params:
        raise SystemExit("params.model.theta is required")
    if not tension_params:
        raise SystemExit("params.model.tension is required")
    theta_stage2_params = dict(model_cfg.get("theta_stage2", {})) if use_stage2 else None
    tension_stage2_params = dict(model_cfg.get("tension_stage2", {})) if use_stage2 else None

    force_cpu = str(os.environ.get("LGBM_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    gpu_requested = any(
        is_lgbm_gpu_requested(p)
        for p in (
            theta_params,
            tension_params,
            theta_stage2_params,
            tension_stage2_params,
        )
        if isinstance(p, dict)
    )
    gpu_probe_ok: bool | None = None
    gpu_probe_reason = "not_requested"
    if force_cpu:
        gpu_probe_ok = False
        gpu_probe_reason = "LGBM_FORCE_CPU is set"
    elif gpu_requested:
        gpu_probe_ok, gpu_probe_reason = probe_lgbm_gpu_available()

    def _with_cpu_fallback(name: str, params: dict[str, Any] | None) -> dict[str, Any] | None:
        if params is None:
            return None
        if not is_lgbm_gpu_requested(params):
            return params
        if gpu_probe_ok is False:
            out = force_lgbm_params_to_cpu(params)
            _log(f"{name}: GPU unavailable, fallback to CPU ({gpu_probe_reason})")
            return out
        return params

    theta_params = _with_cpu_fallback("theta_stage1", theta_params) or {}
    tension_params = _with_cpu_fallback("tension_stage1", tension_params) or {}
    theta_stage2_params = _with_cpu_fallback("theta_stage2", theta_stage2_params)
    tension_stage2_params = _with_cpu_fallback("tension_stage2", tension_stage2_params)

    _log(f"LOAD dataset={args.dataset}")
    df = _read_parquet(args.dataset)
    cols = list(df.columns)
    theta_cols = _col_group(cols, "theta_", "_rad")
    tension_cols = _col_group(cols, "tension_", "_n")
    if len(theta_cols) != 30 or len(tension_cols) != 12:
        raise SystemExit("dataset requires 30 theta_*_rad and 12 tension_*_n columns")

    X = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    theta = df[theta_cols].to_numpy(dtype=float)
    tension = df[tension_cols].to_numpy(dtype=float)

    train_idx, val_idx, test_idx = _load_split_npz(args.split_file)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    theta_train, theta_val, theta_test = theta[train_idx], theta[val_idx], theta[test_idx]
    tension_train, tension_val, tension_test = tension[train_idx], tension[val_idx], tension[test_idx]

    scaler = XYScaler(tmax=float(args.tmax)).fit(X_train, theta_train, tension_train)
    X_train_n = scaler.transform_X(X_train)
    X_val_n = scaler.transform_X(X_val)
    X_test_n = scaler.transform_X(X_test)
    X_train_n, X_val_n, X_test_n, feature_info = _expand_features(X_train_n, X_val_n, X_test_n, feature_mode)
    _log(
        f"DATA ready train/val/test={X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]} "
        f"feature={feature_info.get('mode')} dim={feature_info.get('n_features_in')}->{feature_info.get('n_features_out')}"
    )

    theta_train_enc = _theta_encode(theta_train, theta_target)
    theta_val_enc = _theta_encode(theta_val, theta_target)
    theta_test_enc = _theta_encode(theta_test, theta_target)

    tension_train_enc = _tension_encode(tension_train, tension_target, float(args.tmax))
    tension_val_enc = _tension_encode(tension_val, tension_target, float(args.tmax))
    tension_test_enc = _tension_encode(tension_test, tension_target, float(args.tmax))

    weights = _sample_weights(X_train, theta_train, tension_train, float(args.tmax), params_all)

    t_fit_start = time.perf_counter()
    _log("FIT theta head begin")
    theta_pred_train_enc, theta_pred_val_enc, theta_pred_test_enc, theta_stats = _fit_predict_head(
        y_train=theta_train_enc,
        y_val=theta_val_enc,
        y_test=theta_test_enc,
        X_train=X_train_n,
        X_val=X_val_n,
        X_test=X_test_n,
        sample_weight=weights,
        params_stage1=theta_params,
        params_stage2=theta_stage2_params,
        n_jobs=int(args.n_jobs),
    )
    _log(
        f"FIT theta head done stage1={theta_stats.get('fit_stage1_s', 0.0):.2f}s "
        f"stage2={theta_stats.get('fit_stage2_s', 0.0):.2f}s"
    )

    _log("FIT tension head begin")
    tension_pred_train_enc, tension_pred_val_enc, tension_pred_test_enc, tension_stats = _fit_predict_head(
        y_train=tension_train_enc,
        y_val=tension_val_enc,
        y_test=tension_test_enc,
        X_train=X_train_n,
        X_val=X_val_n,
        X_test=X_test_n,
        sample_weight=weights,
        params_stage1=tension_params,
        params_stage2=tension_stage2_params,
        n_jobs=int(args.n_jobs),
    )
    _log(
        f"FIT tension head done stage1={tension_stats.get('fit_stage1_s', 0.2):.2f}s "
        f"stage2={tension_stats.get('fit_stage2_s', 0.0):.2f}s"
    )
    fit_time_s = time.perf_counter() - t_fit_start
    _log(f"FIT total={fit_time_s:.2f}s")

    theta_pred_train = _theta_decode(theta_pred_train_enc, theta_target)
    theta_pred_val = _theta_decode(theta_pred_val_enc, theta_target)
    theta_pred_test = _theta_decode(theta_pred_test_enc, theta_target)

    tension_pred_train = _tension_decode(tension_pred_train_enc, tension_target, float(args.tmax))
    tension_pred_val = _tension_decode(tension_pred_val_enc, tension_target, float(args.tmax))
    tension_pred_test = _tension_decode(tension_pred_test_enc, tension_target, float(args.tmax))
    tension_pred_train = np.clip(tension_pred_train, 0.0, float(args.tmax))
    tension_pred_val = np.clip(tension_pred_val, 0.0, float(args.tmax))
    tension_pred_test = np.clip(tension_pred_test, 0.0, float(args.tmax))

    lengths_m = None
    p_end_local_m = None
    if not args.skip_fk:
        lengths_m, p_end_local_m = _read_lengths_end(args.robot_config)

    pred_time_s = (
        float(theta_stats.get("pred_test_stage1_s", 0.0))
        + float(theta_stats.get("pred_test_stage2_s", 0.0))
        + float(tension_stats.get("pred_test_stage1_s", 0.0))
        + float(tension_stats.get("pred_test_stage2_s", 0.0))
    )
    pred_ms_per_sample = 1000.0 * pred_time_s / max(1, X_test.shape[0])

    ee_train = None
    ee_val = None
    ee_test = None
    if lengths_m is not None and p_end_local_m is not None:
        ee_train = _ee_errors_mm(theta_pred_train, X_train, lengths_m, p_end_local_m)
        ee_val = _ee_errors_mm(theta_pred_val, X_val, lengths_m, p_end_local_m)
        ee_test = _ee_errors_mm(theta_pred_test, X_test, lengths_m, p_end_local_m)

    metrics_train = _metrics_block(theta_train, theta_pred_train, tension_train, tension_pred_train, float(args.tmax), ee_train)
    metrics_val = _metrics_block(theta_val, theta_pred_val, tension_val, tension_pred_val, float(args.tmax), ee_val)
    metrics_test = _metrics_block(theta_test, theta_pred_test, tension_test, tension_pred_test, float(args.tmax), ee_test)
    metrics_train["rows"] = int(X_train.shape[0])
    metrics_val["rows"] = int(X_val.shape[0])
    metrics_test["rows"] = int(X_test.shape[0])

    out_metrics = {
        "fit_time_s": float(fit_time_s),
        "pred_time_s": float(pred_time_s),
        "pred_time_ms_per_sample": float(pred_ms_per_sample),
        "metrics": {"train": metrics_train, "val": metrics_val, "test": metrics_test},
        "theta_target": theta_target,
        "tension_target": tension_target,
        "use_stage2": use_stage2,
        "feature_info": feature_info,
        "theta_fit_stats": theta_stats,
        "tension_fit_stats": tension_stats,
        "params_file": str(args.params_file),
    }

    model_dir = args.out_dir / "lgbm_extreme"
    _ensure_dir(model_dir)
    (model_dir / "metrics.json").write_text(json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics = {
        "lgbm_extreme": {
            **metrics_test,
            "fit_time_s": float(fit_time_s),
            "pred_time_s": float(pred_time_s),
            "pred_time_ms_per_sample": float(pred_ms_per_sample),
            "train_rows_used": int(X_train.shape[0]),
        }
    }
    (args.out_dir / "all_metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "dataset": str(args.dataset),
        "robot_config": str(args.robot_config),
        "split_file": str(args.split_file),
        "params_file": str(args.params_file),
        "seed": int(args.seed),
        "sizes": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])},
        "env": {"packages": _package_versions()},
        "theta_target": theta_target,
        "tension_target": tension_target,
        "use_stage2": use_stage2,
        "feature_info": feature_info,
        "lgbm_gpu_probe": {
            "gpu_requested": bool(gpu_requested),
            "available": gpu_probe_ok,
            "reason": gpu_probe_reason,
            "force_cpu": force_cpu,
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_preds > 0:
        rng = np.random.default_rng(int(args.seed) + 2026)
        n = int(X_test.shape[0])
        pick_n = min(int(args.save_preds), n)
        pick = rng.choice(n, size=pick_n, replace=False)
        cols_out: dict[str, Any] = {
            "x_m": X_test[pick, 0],
            "y_m": X_test[pick, 1],
            "z_m": X_test[pick, 2],
        }
        for i in range(30):
            cols_out[f"theta_true_{i+1}_rad"] = theta_test[pick, i]
            cols_out[f"theta_pred_{i+1}_rad"] = theta_pred_test[pick, i]
        for j in range(12):
            cols_out[f"tension_true_{j+1}_n"] = tension_test[pick, j]
            cols_out[f"tension_pred_{j+1}_n"] = tension_pred_test[pick, j]
        if ee_test is not None:
            cols_out["ee_err_mm"] = ee_test[pick]
        _write_parquet(pd.DataFrame(cols_out), model_dir / "preds_test.parquet")

    _log(f"WROTE {args.out_dir / 'all_metrics.json'}")


if __name__ == "__main__":
    main()
