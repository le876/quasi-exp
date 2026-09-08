#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "scripts" / "baselines"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from run_theta_fk_baseline import _load_dataset, make_split  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data" / "hierarchical_beta_fk_x1p0_1p2_v1" / "balanced_100k.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "visualizations" / "theta_training_curves_hierarchical_beta_fk_v1"
DEFAULT_MLP_ROOT = REPO_ROOT / "runs" / "baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_mlp_v1"
DEFAULT_LGBM_ROOT = REPO_ROOT / "runs" / "baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_lgbm_v1"


def _inner_estimator(package: dict[str, Any]) -> Any:
    model = package["model"]
    if hasattr(model, "named_steps") and "mlp" in model.named_steps:
        return model.named_steps["mlp"]
    return model


def extract_mlp_history(package: dict[str, Any], model_id: str) -> pd.DataFrame:
    inner = _inner_estimator(package)
    loss = [float(v) for v in getattr(inner, "loss_curve_", [])]
    validation = [float(v) for v in getattr(inner, "validation_scores_", [])]
    n = max(len(loss), len(validation))
    if n <= 0:
        return pd.DataFrame(columns=["model_id", "epoch", "train_loss", "validation_score"])
    rows = []
    for i in range(n):
        rows.append(
            {
                "model_id": str(model_id),
                "epoch": int(i + 1),
                "train_loss": loss[i] if i < len(loss) else float("nan"),
                "validation_score": validation[i] if i < len(validation) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _predict_lgbm_iteration(package: dict[str, Any], X: np.ndarray, iteration: int) -> np.ndarray:
    model = package["model"]
    if not hasattr(model, "named_steps"):
        raise ValueError("expected sklearn Pipeline with x_scaler and mlp steps")
    x_scaler = model.named_steps["x_scaler"]
    inner = model.named_steps["mlp"]
    if not hasattr(inner, "estimators_"):
        raise ValueError("expected MultiOutputRegressor-like estimator with estimators_")
    Xn = x_scaler.transform(np.asarray(X, dtype=float))
    cols = []
    for estimator in inner.estimators_:
        cols.append(np.asarray(estimator.predict(Xn, num_iteration=int(iteration)), dtype=float).reshape(-1))
    pred_scaled = np.column_stack(cols)
    y_scaler = package.get("y_scaler")
    if y_scaler is not None:
        return np.asarray(y_scaler.inverse_transform(pred_scaled), dtype=float)
    return pred_scaled


def lgbm_theta_mae_curve(
    package: dict[str, Any],
    model_id: str,
    X_by_split: dict[str, np.ndarray],
    theta_by_split: dict[str, np.ndarray],
    iterations: list[int],
) -> pd.DataFrame:
    rows = []
    for split_name, X in X_by_split.items():
        theta_true = np.asarray(theta_by_split[split_name], dtype=float)
        for iteration in iterations:
            theta_pred = _predict_lgbm_iteration(package, X, int(iteration))
            mae_deg = float(np.mean(np.abs(theta_pred - theta_true)) * 180.0 / math.pi)
            rows.append(
                {
                    "model_id": str(model_id),
                    "split": str(split_name),
                    "iteration": int(iteration),
                    "theta_mae_deg": mae_deg,
                }
            )
    return pd.DataFrame(rows)


def _load_package(path: Path) -> dict[str, Any]:
    package = load(path)
    if not isinstance(package, dict) or package.get("kind") != "theta_only":
        raise ValueError(f"{path} is not a theta_only model package")
    return package


def _iteration_grid(package: dict[str, Any], max_points: int = 40) -> list[int]:
    inner = _inner_estimator(package)
    estimators = getattr(inner, "estimators_", None)
    if not estimators:
        return []
    n_estimators = max(int(getattr(est, "n_estimators_", getattr(est, "n_estimators", 0))) for est in estimators)
    if n_estimators <= 0:
        return []
    values = np.unique(np.linspace(1, n_estimators, min(max_points, n_estimators), dtype=int))
    return [int(v) for v in values]


def _sample_idx(idx: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if int(max_rows) <= 0 or len(idx) <= int(max_rows):
        return idx
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(idx, size=int(max_rows), replace=False)).astype(np.int64)


def _prepare_split_arrays(dataset: Path, split: str, max_curve_rows: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    _df, X, theta, beta = _load_dataset(Path(dataset), max_rows=0, seed=int(seed))
    train_idx, val_idx, test_idx = make_split(
        str(split),
        X=X,
        beta=beta,
        seed=int(seed),
        val_size=0.1,
        test_size=0.1,
    )
    out_X = {}
    out_theta = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        pick = _sample_idx(idx, max_curve_rows, seed + len(name))
        out_X[name] = X[pick]
        out_theta[name] = theta[pick]
    return out_X, out_theta


def _plot_mlp_histories(curves: pd.DataFrame, out_path: Path) -> None:
    if curves.empty:
        return
    model_ids = list(curves["model_id"].drop_duplicates())
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=False)
    for model_id in model_ids:
        sub = curves[curves["model_id"] == model_id]
        axes[0].plot(sub["epoch"], sub["train_loss"], lw=1.7, label=model_id)
        axes[1].plot(sub["epoch"], sub["validation_score"], lw=1.7, label=model_id)
    axes[0].set_ylabel("scaled train loss")
    axes[0].set_title("MLP optimizer training loss")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation R2 score")
    axes[1].set_title("MLP early-stopping validation score")
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_lgbm_curves(curves: pd.DataFrame, out_path: Path) -> None:
    if curves.empty:
        return
    model_ids = list(curves["model_id"].drop_duplicates())
    n = len(model_ids)
    ncols = min(3, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for i, model_id in enumerate(model_ids):
        ax = axes.reshape(-1)[i]
        ax.axis("on")
        sub = curves[curves["model_id"] == model_id]
        for split_name in ("train", "val", "test"):
            part = sub[sub["split"] == split_name]
            if part.empty:
                continue
            ax.plot(part["iteration"], part["theta_mae_deg"], lw=1.6, label=split_name)
        ax.set_title(model_id)
        ax.set_xlabel("boosting iteration")
        ax.set_ylabel("theta MAE (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("LGBM staged theta MAE curves", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_final_compare(mlp_curves: pd.DataFrame, lgbm_curves: pd.DataFrame, out_path: Path) -> None:
    rows = []
    if not mlp_curves.empty:
        for model_id, sub in mlp_curves.groupby("model_id"):
            rows.append({"model_id": model_id, "curve_type": "mlp_final_train_loss", "value": float(sub["train_loss"].dropna().iloc[-1])})
            rows.append({"model_id": model_id, "curve_type": "mlp_final_validation_score", "value": float(sub["validation_score"].dropna().iloc[-1])})
    if not lgbm_curves.empty:
        for model_id, sub in lgbm_curves.groupby("model_id"):
            test = sub[sub["split"] == "test"].sort_values("iteration")
            if not test.empty:
                rows.append({"model_id": model_id, "curve_type": "lgbm_final_test_theta_mae_deg", "value": float(test["theta_mae_deg"].iloc[-1])})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.38 * len(df))))
    labels = df["model_id"] + " / " + df["curve_type"]
    y = np.arange(len(df))
    ax.barh(y, df["value"], color="#4c78a8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("final curve value")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Final training-curve values")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_readme(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Theta Model Training Curves",
        "",
        "这些图用于判断 MLP 和 LGBM 是否欠拟合。",
        "",
        "- `mlp_loss_validation_curves.png`: sklearn MLP 保存的 optimizer train loss 和 early-stopping validation R2。train loss 下降后仍停在较高值，或 validation R2 很低，说明 MLP 容量/训练不足。",
        "- `lgbm_theta_mae_iteration_curves.png`: 用已训练 LGBM 在不同 boosting iteration 下重算 train/val/test theta MAE。train 和 val/test 一起高且下降缓慢，更像欠拟合；train 很低但 val/test 高才是过拟合。",
        "- `final_curve_values.png`: 曲线末端值的快速索引。",
        "",
        "## Inputs",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- MLP root: `{payload['mlp_root']}`",
        f"- LGBM root: `{payload['lgbm_root']}`",
        f"- Splits: `{', '.join(payload['splits'])}`",
        f"- LGBM curve sample rows per train/val/test split: `{payload['max_curve_rows']}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    mlp_frames = []
    for split in splits:
        path = Path(args.mlp_root) / split / "mlp" / "model.joblib"
        if not path.exists():
            continue
        package = _load_package(path)
        mlp_frames.append(extract_mlp_history(package, model_id=f"{split}_mlp"))
    mlp_curves = pd.concat(mlp_frames, ignore_index=True) if mlp_frames else pd.DataFrame()
    if not mlp_curves.empty:
        mlp_curves.to_csv(out_dir / "mlp_history_curves.csv", index=False)

    lgbm_frames = []
    for split in splits:
        path = Path(args.lgbm_root) / split / "lgbm" / "model.joblib"
        if not path.exists():
            continue
        package = _load_package(path)
        iterations = _iteration_grid(package, max_points=int(args.max_iteration_points))
        if not iterations:
            continue
        X_by_split, theta_by_split = _prepare_split_arrays(Path(args.dataset), split, int(args.max_curve_rows), int(args.seed))
        lgbm_frames.append(
            lgbm_theta_mae_curve(
                package,
                model_id=f"{split}_lgbm",
                X_by_split=X_by_split,
                theta_by_split=theta_by_split,
                iterations=iterations,
            )
        )
    lgbm_curves = pd.concat(lgbm_frames, ignore_index=True) if lgbm_frames else pd.DataFrame()
    if not lgbm_curves.empty:
        lgbm_curves.to_csv(out_dir / "lgbm_staged_theta_mae_curves.csv", index=False)

    images = []
    if not mlp_curves.empty:
        _plot_mlp_histories(mlp_curves, out_dir / "mlp_loss_validation_curves.png")
        images.append("mlp_loss_validation_curves.png")
    if not lgbm_curves.empty:
        _plot_lgbm_curves(lgbm_curves, out_dir / "lgbm_theta_mae_iteration_curves.png")
        images.append("lgbm_theta_mae_iteration_curves.png")
    _plot_final_compare(mlp_curves, lgbm_curves, out_dir / "final_curve_values.png")
    if (out_dir / "final_curve_values.png").exists():
        images.append("final_curve_values.png")

    payload = {
        "dataset": str(args.dataset),
        "mlp_root": str(args.mlp_root),
        "lgbm_root": str(args.lgbm_root),
        "splits": splits,
        "max_curve_rows": int(args.max_curve_rows),
        "images": images,
        "mlp_rows": int(len(mlp_curves)),
        "lgbm_rows": int(len(lgbm_curves)),
    }
    (out_dir / "training_curve_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(out_dir, payload)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot theta-only MLP/LGBM training curves for hierarchical beta FK baselines.")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--mlp-root", type=Path, default=DEFAULT_MLP_ROOT)
    ap.add_argument("--lgbm-root", type=Path, default=DEFAULT_LGBM_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--splits", default="iid,x_slab,radius")
    ap.add_argument("--max-curve-rows", type=int, default=5000)
    ap.add_argument("--max-iteration-points", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260705)
    return ap.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
