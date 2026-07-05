#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.model.sampling import effective_beta_from_theta  # noqa: E402
from splits import split_angular_sector_holdout, split_beta_block_holdout, split_iid, split_radius_holdout  # noqa: E402


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def beta_consistency_mae_rad(theta_pred_rad: np.ndarray, beta_pred_rad: np.ndarray) -> float:
    beta_from_theta = effective_beta_from_theta(np.asarray(theta_pred_rad, dtype=float))
    return float(np.mean(np.abs(beta_from_theta - np.asarray(beta_pred_rad, dtype=float))))


def _metrics(
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    tension_true: np.ndarray,
    tension_pred: np.ndarray,
    beta_true: np.ndarray,
    beta_pred: np.ndarray,
    xyz_true: np.ndarray,
    *,
    lengths_m: np.ndarray | None,
    p_end_local_m: np.ndarray | None,
) -> dict[str, Any]:
    theta_err = np.abs(theta_true - theta_pred) * (180.0 / np.pi)
    tension_err = np.abs(tension_true - tension_pred)
    beta_err = np.abs(beta_true - beta_pred) * (180.0 / np.pi)
    out = {
        "theta_mae_deg": float(np.mean(theta_err)),
        "theta_rmse_deg": float(np.sqrt(np.mean(np.square(theta_err)))),
        "tension_mae_n": float(np.mean(tension_err)),
        "tension_rmse_n": float(np.sqrt(np.mean(np.square(tension_err)))),
        "tension_p95_n": float(np.percentile(np.mean(tension_err, axis=1), 95)),
        "beta_mae_deg": float(np.mean(beta_err)),
        "beta_consistency_mae_rad": beta_consistency_mae_rad(theta_pred, beta_pred),
    }
    if lengths_m is not None and p_end_local_m is not None:
        pred_xyz = fk_dh_batch(theta_pred, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
        ee = np.linalg.norm(pred_xyz - xyz_true, axis=1) * 1000.0
        out["ee_pos_rmse_mm"] = float(np.sqrt(np.mean(np.square(ee))))
        out["ee_pos_p95_mm"] = float(np.percentile(ee, 95))
    else:
        out["ee_pos_rmse_mm"] = float("nan")
        out["ee_pos_p95_mm"] = float("nan")
    return out


def _load_robot_paths(robot_config: Path | None) -> tuple[Path | None, Path | None]:
    if robot_config is None:
        return None, None
    import yaml

    cfg = yaml.safe_load(robot_config.read_text(encoding="utf-8"))
    return Path(cfg["paths"]["lengths_csv"]), Path(cfg["paths"]["end_effector_csv"])


def _read_lengths_end(lengths_csv: Path | None, end_effector_csv: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if lengths_csv is None or end_effector_csv is None:
        return None, None
    ldf = pd.read_csv(lengths_csv)
    emap = dict(zip(ldf["name"].astype(str), ldf["value_m"].astype(float)))
    kD = max(int(k[1:]) for k in emap if k.startswith("l"))
    lengths = np.zeros(kD + 1, dtype=float)
    for i in range(kD + 1):
        lengths[i] = float(emap[f"l{i}"])
    edf = pd.read_csv(end_effector_csv)
    p_end = np.array([float(edf["p_end_x_m"][0]), float(edf["p_end_y_m"][0]), float(edf["p_end_z_m"][0]), 1.0], dtype=float)
    return lengths, p_end


def make_split(split: str, xyz: np.ndarray, beta: np.ndarray, *, seed: int, val_size: float, test_size: float):
    if split == "iid":
        return split_iid(len(xyz), seed=seed, val_size=val_size, test_size=test_size)
    if split == "radius":
        return split_radius_holdout(xyz, val_size=val_size, test_size=test_size)
    if split == "beta_block":
        return split_beta_block_holdout(beta, val_size=val_size, test_size=test_size)
    if split == "angular_sector":
        return split_angular_sector_holdout(xyz, val_size=val_size, test_size=test_size)
    raise ValueError(f"unsupported split: {split}")


def run_beta_aux(
    dataset_path: Path,
    out_dir: Path,
    *,
    split: str,
    seed: int = 20260608,
    val_size: float = 0.1,
    test_size: float = 0.1,
    feature_set: str = "poly_heavy",
    lambda_beta: float = 0.1,
    lambda_cons: float = 0.1,
    epochs: int = 200,
    batch_size: int = 1024,
    robot_config: Path | None = None,
) -> dict[str, Any]:
    import tensorflow as tf  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    tension_cols = _numbered_cols(list(df.columns), "tension_", "_n")
    theta = df[theta_cols].to_numpy(dtype=float)
    tension = df[tension_cols].to_numpy(dtype=float)
    beta = effective_beta_from_theta(theta)
    X, feature_names = build_features(xyz, feature_set=feature_set)
    train_idx, val_idx, test_idx = make_split(split, xyz, beta, seed=seed, val_size=val_size, test_size=test_size)

    x_scaler = StandardScaler().fit(X[train_idx])
    th_scaler = StandardScaler().fit(theta[train_idx])
    t_scaler = StandardScaler().fit(tension[train_idx] / 2000.0)
    b_scaler = StandardScaler().fit(beta[train_idx])

    def xs(idx):
        return x_scaler.transform(X[idx]).astype(np.float32)

    y_train = {
        "theta": th_scaler.transform(theta[train_idx]).astype(np.float32),
        "tension": t_scaler.transform(tension[train_idx] / 2000.0).astype(np.float32),
        "beta": b_scaler.transform(beta[train_idx]).astype(np.float32),
    }
    y_val = {
        "theta": th_scaler.transform(theta[val_idx]).astype(np.float32),
        "tension": t_scaler.transform(tension[val_idx] / 2000.0).astype(np.float32),
        "beta": b_scaler.transform(beta[val_idx]).astype(np.float32),
    }

    tf.random.set_seed(int(seed))
    inputs = tf.keras.Input(shape=(X.shape[1],), name="input")
    x = inputs
    for units in (256, 128, 64, 32):
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    theta_out = tf.keras.layers.Dense(30, name="theta")(x)
    tension_out = tf.keras.layers.Dense(12, name="tension")(x)
    beta_out = tf.keras.layers.Dense(6, name="beta")(x)
    model = tf.keras.Model(inputs=inputs, outputs={"theta": theta_out, "tension": tension_out, "beta": beta_out})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-3),
        loss={"theta": "mse", "tension": "mse", "beta": "mse"},
        loss_weights={"theta": 1.0, "tension": 1.0, "beta": float(lambda_beta) + float(lambda_cons)},
    )
    t0 = time.perf_counter()
    hist = model.fit(
        xs(train_idx),
        y_train,
        validation_data=(xs(val_idx), y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
    )
    fit_s = float(time.perf_counter() - t0)

    lengths_csv, ee_csv = _load_robot_paths(robot_config)
    lengths_m, p_end_local_m = _read_lengths_end(lengths_csv, ee_csv)

    metrics_by_split: dict[str, Any] = {}
    for name, idx in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        pred = model.predict(xs(idx), batch_size=int(batch_size), verbose=0)
        theta_pred = th_scaler.inverse_transform(np.asarray(pred["theta"], dtype=float))
        tension_pred = t_scaler.inverse_transform(np.asarray(pred["tension"], dtype=float)) * 2000.0
        beta_pred = b_scaler.inverse_transform(np.asarray(pred["beta"], dtype=float))
        metrics_by_split[name] = _metrics(
            theta[idx],
            theta_pred,
            tension[idx],
            tension_pred,
            beta[idx],
            beta_pred,
            xyz[idx],
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
        )
        metrics_by_split[name]["rows"] = int(len(idx))

    payload = {
        "dataset": str(dataset_path),
        "split": str(split),
        "feature": {"feature_set": feature_set, "feature_dim": int(X.shape[1]), "feature_names": feature_names},
        "lambda_beta": float(lambda_beta),
        "lambda_cons": float(lambda_cons),
        "fit_time_s": fit_s,
        "epochs_run": int(len(hist.history.get("loss", []))),
        "metrics": metrics_by_split,
    }
    (out_dir / "all_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", default="iid,radius,beta_block,angular_sector")
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--feature-set", default="poly_heavy", choices=["raw", "poly_medium", "poly_heavy"])
    ap.add_argument("--lambda-beta", type=float, default=0.1)
    ap.add_argument("--lambda-cons", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--robot-config", type=Path, default=None)
    args = ap.parse_args()
    summaries: dict[str, Any] = {}
    for split in [s.strip() for s in str(args.splits).split(",") if s.strip()]:
        payload = run_beta_aux(
            args.dataset,
            args.out_dir / split,
            split=split,
            seed=int(args.seed),
            feature_set=str(args.feature_set),
            lambda_beta=float(args.lambda_beta),
            lambda_cons=float(args.lambda_cons),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            robot_config=args.robot_config,
        )
        summaries[split] = payload["metrics"]["test"]
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
