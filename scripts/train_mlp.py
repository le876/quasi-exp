#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.kinematics import forward_kinematics  # noqa: E402


@dataclass(frozen=True)
class Metrics:
    n_train: int
    n_test: int
    rmse_theta_deg: float
    mae_theta_deg: float
    rmse_tension_n: float
    mae_tension_n: float
    ee_pos_rmse_mm: float
    ee_pos_p95_mm: float
    ee_pos_max_mm: float


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.square(a - b))))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def _col_group(cols: list[str], prefix: str, suffix: str) -> list[str]:
    out = [c for c in cols if c.startswith(prefix) and c.endswith(suffix)]
    # sort by numeric index if possible
    def key(c: str) -> int:
        s = c[len(prefix) : -len(suffix)]
        try:
            return int(s.strip("_"))
        except Exception:
            return 10**9

    return sorted(out, key=key)


def _load_xy(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    table = pq.read_table(dataset_path)
    df = table.to_pandas()
    cols = list(df.columns)

    if not {"x_m", "y_m", "z_m"}.issubset(cols):
        raise SystemExit("dataset必须包含列: x_m,y_m,z_m")

    theta_cols = _col_group(cols, "theta_", "_rad")
    tension_cols = _col_group(cols, "tension_", "_n")
    if len(theta_cols) != 30:
        raise SystemExit(f"theta列数量应为30，实际={len(theta_cols)}")
    if len(tension_cols) != 12:
        raise SystemExit(f"tension列数量应为12，实际={len(tension_cols)}")

    X = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    Y = df[theta_cols + tension_cols].to_numpy(dtype=float)
    return X, Y, theta_cols, tension_cols


def _ee_errors_mm(
    cfg_path: Path,
    X_xyz_m: np.ndarray,
    theta_pred_rad: np.ndarray,
) -> np.ndarray:
    """
    用 forward_kinematics(theta_pred) 回代位置，与输入 X 的 xyz 比较。

    注意：dataset里的 theta 已经是“对齐 theta_sign 后”的 theta。
    因此这里调用 FK 时使用 theta_sign=+1，避免重复翻转。
    """
    cfg = load_config(str(cfg_path))
    inputs = load_robot_inputs(cfg)
    lengths_m = inputs.lengths_m
    p_end_local_m = inputs.p_end_local_m

    errs = np.zeros(theta_pred_rad.shape[0], dtype=float)
    for i in range(theta_pred_rad.shape[0]):
        p_xyz, _ = forward_kinematics(
            theta_pred_rad[i],
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            theta_sign=1.0,
        )
        d = p_xyz - X_xyz_m[i]
        errs[i] = float(np.linalg.norm(d)) * 1000.0
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path, help="dataset.parquet")
    ap.add_argument(
        "--robot-config",
        required=True,
        type=Path,
        help="用于 FK 回代误差评估的机器人config（例如 configs/robot_rods_only_20deg_penalty_100k.yaml）",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260207)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--alpha", type=float, default=1e-6, help="L2正则")
    ap.add_argument("--learning-rate-init", type=float, default=1e-3)
    ap.add_argument("--early-stopping", action="store_true", help="使用sklearn内部early stopping")
    ap.add_argument("--hidden-sizes", type=str, default="128,64,32")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, Y, theta_cols, tension_cols = _load_xy(args.dataset)
    hidden = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())
    if not hidden:
        raise SystemExit("--hidden-sizes 不能为空")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        shuffle=True,
    )

    # Paper-like MLP: 3 hidden layers (128,64,32), ReLU, Adam.
    # 为了数值稳定：对 X/Y 都做标准化。
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    reg = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=float(args.alpha),
        batch_size=int(args.batch_size),
        learning_rate_init=float(args.learning_rate_init),
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
        early_stopping=bool(args.early_stopping),
        n_iter_no_change=20,
        verbose=True,
    )

    # Pipeline: X 标准化 -> MLP（y 标准化手动做）
    model = Pipeline([("x_scaler", x_scaler), ("mlp", reg)])

    Y_train_s = y_scaler.fit_transform(Y_train)
    model.fit(X_train, Y_train_s)

    Y_pred_s = model.predict(X_test)
    Y_pred = y_scaler.inverse_transform(Y_pred_s)

    # split outputs
    theta_true = Y_test[:, :30]
    tension_true = Y_test[:, 30:]
    theta_pred = Y_pred[:, :30]
    tension_pred = Y_pred[:, 30:]

    theta_true_deg = theta_true * 180.0 / math.pi
    theta_pred_deg = theta_pred * 180.0 / math.pi

    ee_err_mm = _ee_errors_mm(args.robot_config, X_test, theta_pred)

    metrics = Metrics(
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        rmse_theta_deg=_rmse(theta_true_deg, theta_pred_deg),
        mae_theta_deg=_mae(theta_true_deg, theta_pred_deg),
        rmse_tension_n=_rmse(tension_true, tension_pred),
        mae_tension_n=_mae(tension_true, tension_pred),
        ee_pos_rmse_mm=float(np.sqrt(np.mean(np.square(ee_err_mm)))),
        ee_pos_p95_mm=float(np.quantile(ee_err_mm, 0.95)),
        ee_pos_max_mm=float(np.max(ee_err_mm)),
    )

    # save artifacts
    dump(
        {
            "model": model,
            "y_scaler": y_scaler,
            "theta_cols": theta_cols,
            "tension_cols": tension_cols,
            "dataset": str(args.dataset),
            "robot_config": str(args.robot_config),
            "seed": int(args.seed),
            "hidden_sizes": hidden,
        },
        args.out_dir / "mlp.joblib",
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

