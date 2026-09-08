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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.model.sampling import beta_to_theta  # noqa: E402
from splits import split_angular_sector_holdout, split_beta_block_holdout, split_iid, split_radius_holdout  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def beta_matrix(meta: pd.DataFrame) -> np.ndarray:
    missing = [c for c in BETA_COLS if c not in meta.columns]
    if missing:
        raise ValueError(f"meta missing beta columns: {missing}")
    return meta[BETA_COLS].to_numpy(dtype=float)


def effective_beta_from_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 30:
        raise ValueError("theta must have shape [N, 30]")
    cols = []
    for section in range(3):
        start = section * 10
        odd_idx = [start + i for i in range(0, 10, 2)]
        even_idx = [start + i for i in range(1, 10, 2)]
        cols.append(theta[:, odd_idx].mean(axis=1))
        cols.append(theta[:, even_idx].mean(axis=1))
    return np.stack(cols, axis=1)


def beta_target(meta: pd.DataFrame, theta: np.ndarray, beta_source: str) -> np.ndarray:
    beta_source = str(beta_source)
    if beta_source == "meta":
        return beta_matrix(meta)
    if beta_source == "neg_meta":
        return -beta_matrix(meta)
    if beta_source == "effective_theta":
        return effective_beta_from_theta(theta)
    raise ValueError(f"unsupported beta_source: {beta_source}")


def theta_from_beta_batch(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).reshape(-1, 6)
    return np.vstack([beta_to_theta(row) for row in beta])


def make_split(
    split: str,
    xyz: np.ndarray,
    beta: np.ndarray,
    *,
    seed: int,
    val_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = str(split)
    if split == "iid":
        return split_iid(len(xyz), seed=seed, val_size=val_size, test_size=test_size)
    if split == "radius":
        return split_radius_holdout(xyz, val_size=val_size, test_size=test_size)
    if split == "beta_block":
        return split_beta_block_holdout(beta, val_size=val_size, test_size=test_size)
    if split == "angular_sector":
        return split_angular_sector_holdout(xyz, val_size=val_size, test_size=test_size)
    raise ValueError(f"unsupported split: {split}")


def family_gate_decision(family_summary: dict[str, Any], *, accuracy_gate: float = 0.70) -> dict[str, Any]:
    k = family_summary.get("recommended_k")
    if k is None:
        return {"enabled": False, "reason": "no recommended_k"}
    row = family_summary.get("families", {}).get(f"k{int(k)}", {})
    clf = row.get("classifier", {})
    acc = max(float(clf.get("rf_accuracy", 0.0)), float(clf.get("knn_accuracy", 0.0)))
    if acc < float(accuracy_gate):
        return {"enabled": False, "reason": f"family classifier accuracy {acc:.3f} < {accuracy_gate:.3f}", "k": int(k)}
    return {"enabled": True, "reason": "family classifier accuracy passed", "k": int(k), "accuracy": acc}


def _model(name: str, *, seed: int):
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=8, weights="distance")
    if name == "rf":
        return RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=int(seed), n_jobs=-1)
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=int(seed),
            learning_rate_init=1.0e-3,
        )
    raise ValueError(f"unsupported model: {name}")


def _fit_scaled_regressor(model, X_train: np.ndarray, y_train: np.ndarray):
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    model.fit(x_scaler.transform(X_train), y_scaler.transform(y_train))
    return x_scaler, y_scaler, model


def _predict_scaled(x_scaler: StandardScaler, y_scaler: StandardScaler, model, X: np.ndarray) -> np.ndarray:
    return y_scaler.inverse_transform(model.predict(x_scaler.transform(X)))


def _beta_knn_tension(beta_train: np.ndarray, tension_train: np.ndarray, beta_query: np.ndarray, k: int = 8) -> np.ndarray:
    kk = min(max(1, int(k)), len(beta_train))
    nn = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(beta_train)
    dist, idx = nn.kneighbors(beta_query)
    weights = 1.0 / np.maximum(dist, 1.0e-9)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return np.einsum("nk,nkd->nd", weights, tension_train[idx], optimize=True)


def _metrics(
    *,
    beta_true: np.ndarray,
    beta_pred: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    tension_true: np.ndarray,
    tension_pred: np.ndarray,
    xyz_true: np.ndarray,
    tmax: float,
    lengths_m: np.ndarray | None = None,
    p_end_local_m: np.ndarray | None = None,
) -> dict[str, Any]:
    beta_err_deg = np.abs(beta_true - beta_pred) * (180.0 / np.pi)
    theta_err_deg = np.abs(theta_true - theta_pred) * (180.0 / np.pi)
    tension_err = np.abs(tension_true - tension_pred)
    out = {
        "beta_mae_deg": float(np.mean(beta_err_deg)),
        "beta_rmse_deg": float(np.sqrt(np.mean(np.square(beta_err_deg)))),
        "theta_mae_deg": float(np.mean(theta_err_deg)),
        "theta_rmse_deg": float(np.sqrt(np.mean(np.square(theta_err_deg)))),
        "tension_mae_n": float(np.mean(tension_err)),
        "tension_rmse_n": float(np.sqrt(np.mean(np.square(tension_err)))),
        "tension_p95_n": float(np.percentile(np.mean(tension_err, axis=1), 95)),
        "tension_lt0_ratio": float(np.mean(tension_pred < 0.0)),
        "tension_gt_tmax_ratio": float(np.mean(tension_pred > float(tmax))),
    }
    if lengths_m is not None and p_end_local_m is not None:
        pred_xyz = fk_dh_batch(theta_pred, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
        ee = np.linalg.norm(pred_xyz - xyz_true, axis=1) * 1000.0
        out["ee_pos_rmse_mm"] = float(np.sqrt(np.mean(np.square(ee))))
        out["ee_pos_p95_mm"] = float(np.percentile(ee, 95))
        out["ee_pos_max_mm"] = float(np.max(ee))
    else:
        out["ee_pos_rmse_mm"] = float("nan")
        out["ee_pos_p95_mm"] = float("nan")
        out["ee_pos_max_mm"] = float("nan")
    out["score_val_like"] = float(out["theta_mae_deg"] / 20.0 + out["tension_mae_n"] / float(tmax))
    return out


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


def _load_robot_paths(robot_config: Path | None, lengths_csv: Path | None, end_effector_csv: Path | None):
    if lengths_csv is not None and end_effector_csv is not None:
        return lengths_csv, end_effector_csv
    if robot_config is None:
        return None, None
    import yaml

    cfg = yaml.safe_load(robot_config.read_text(encoding="utf-8"))
    return Path(cfg["paths"]["lengths_csv"]), Path(cfg["paths"]["end_effector_csv"])


def evaluate_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    split: str,
    models: list[str],
    tension_modes: list[str],
    seed: int = 20260608,
    val_size: float = 0.1,
    test_size: float = 0.1,
    feature_set: str = "poly_heavy",
    tmax: float = 2000.0,
    lengths_m: np.ndarray | None = None,
    p_end_local_m: np.ndarray | None = None,
    beta_source: str = "effective_theta",
) -> dict[str, Any]:
    xyz = dataset[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    X, feature_names = build_features(xyz, feature_set=feature_set)
    theta_cols = _numbered_cols(list(dataset.columns), "theta_", "_rad")
    tension_cols = _numbered_cols(list(dataset.columns), "tension_", "_n")
    theta_true = dataset[theta_cols].to_numpy(dtype=float)
    tension = dataset[tension_cols].to_numpy(dtype=float)
    beta = beta_target(meta, theta_true, beta_source=beta_source)
    train_idx, val_idx, test_idx = make_split(split, xyz, beta, seed=seed, val_size=val_size, test_size=test_size)
    split_idx = {"train": train_idx, "val": val_idx, "test": test_idx}

    out: dict[str, Any] = {
        "rows": int(len(dataset)),
        "split": str(split),
        "beta_source": str(beta_source),
        "sizes": {k: int(len(v)) for k, v in split_idx.items()},
        "feature": {"feature_set": str(feature_set), "feature_dim": int(X.shape[1]), "feature_names": feature_names},
        "models": {},
    }

    for model_name in models:
        beta_model = _model(model_name, seed=seed)
        t0 = time.perf_counter()
        bx, by, beta_model = _fit_scaled_regressor(beta_model, X[train_idx], beta[train_idx])
        beta_fit_s = float(time.perf_counter() - t0)
        pred_beta = {sp: _predict_scaled(bx, by, beta_model, X[idx]) for sp, idx in split_idx.items()}
        pred_theta = {sp: theta_from_beta_batch(pred_beta[sp]) for sp in split_idx}

        for tension_mode in tension_modes:
            key = f"{model_name}__{tension_mode}"
            fit_s = beta_fit_s
            pred_tension: dict[str, np.ndarray] = {}
            if tension_mode == "direct":
                train_aug = np.concatenate([X[train_idx], beta[train_idx]], axis=1)
                tension_model = _model(model_name, seed=seed + 17)
                t0 = time.perf_counter()
                tx, ty, tension_model = _fit_scaled_regressor(tension_model, train_aug, tension[train_idx])
                fit_s += float(time.perf_counter() - t0)
                for sp, idx in split_idx.items():
                    aug = np.concatenate([X[idx], pred_beta[sp]], axis=1)
                    pred_tension[sp] = _predict_scaled(tx, ty, tension_model, aug)
            elif tension_mode == "beta_knn":
                for sp in split_idx:
                    pred_tension[sp] = _beta_knn_tension(beta[train_idx], tension[train_idx], pred_beta[sp], k=8)
            else:
                raise ValueError(f"unsupported tension mode: {tension_mode}")

            metrics_by_split = {}
            for sp, idx in split_idx.items():
                metrics_by_split[sp] = _metrics(
                    beta_true=beta[idx],
                    beta_pred=pred_beta[sp],
                    theta_true=theta_true[idx],
                    theta_pred=pred_theta[sp],
                    tension_true=tension[idx],
                    tension_pred=pred_tension[sp],
                    xyz_true=xyz[idx],
                    tmax=tmax,
                    lengths_m=lengths_m,
                    p_end_local_m=p_end_local_m,
                )
                metrics_by_split[sp]["rows"] = int(len(idx))
            out["models"][key] = {
                "train_rows_used": int(len(train_idx)),
                "fit_time_s": float(fit_s),
                "metrics": metrics_by_split,
            }
    return out


def evaluate(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def _write_report(payload: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Beta-First Baseline Report",
        "",
        f"- split: `{payload['split']}`",
        f"- beta_source: `{payload.get('beta_source', 'unknown')}`",
        f"- rows: {payload['rows']}",
        "",
        "| model | T MAE N | T p95 N | beta MAE deg | theta MAE deg | EE p95 mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in sorted(payload["models"].items()):
        m = row["metrics"]["test"]
        lines.append(
            f"| {key} | {m['tension_mae_n']:.2f} | {m['tension_p95_n']:.2f} | "
            f"{m['beta_mae_deg']:.3f} | {m['theta_mae_deg']:.3f} | {m['ee_pos_p95_mm']:.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_predictions(
    out_dir: Path,
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    payload: dict[str, Any],
    max_rows: int = 0,
) -> None:
    # Reserved for future plotting; keep current output compact.
    _ = (out_dir, dataset, meta, payload, max_rows)


def run_one_split(
    *,
    dataset_path: Path,
    meta_path: Path,
    out_dir: Path,
    split: str,
    models: list[str],
    tension_modes: list[str],
    seed: int,
    val_size: float,
    test_size: float,
    feature_set: str,
    tmax: float,
    lengths_m: np.ndarray | None,
    p_end_local_m: np.ndarray | None,
    beta_source: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    payload = evaluate_frames(
        dataset,
        meta,
        split=split,
        models=models,
        tension_modes=tension_modes,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        feature_set=feature_set,
        tmax=tmax,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        beta_source=beta_source,
    )
    (out_dir / "all_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload, out_dir / "BETA_FIRST_REPORT.md")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--splits", default="iid,radius,beta_block,angular_sector")
    ap.add_argument("--models", default="mlp,rf,knn")
    ap.add_argument("--tension-modes", default="direct,beta_knn")
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--feature-set", default="poly_heavy", choices=["raw", "poly_medium", "poly_heavy"])
    ap.add_argument("--beta-source", default="effective_theta", choices=["effective_theta", "meta", "neg_meta"])
    ap.add_argument("--tmax", type=float, default=2000.0)
    ap.add_argument("--robot-config", type=Path, default=None)
    ap.add_argument("--lengths-csv", type=Path, default=None)
    ap.add_argument("--end-effector-csv", type=Path, default=None)
    ap.add_argument("--family-json", type=Path, default=None)
    ap.add_argument("--family-accuracy-gate", type=float, default=0.70)
    args = ap.parse_args()

    if args.family_json is not None:
        decision = family_gate_decision(json.loads(args.family_json.read_text(encoding="utf-8")), accuracy_gate=args.family_accuracy_gate)
        if not bool(decision["enabled"]):
            out_dir = Path(f"{args.out_prefix}_family_gated_skipped")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "family_gate_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"family_gated": decision}, ensure_ascii=False))
            return 0

    lengths_csv, ee_csv = _load_robot_paths(args.robot_config, args.lengths_csv, args.end_effector_csv)
    lengths_m, p_end_local_m = _read_lengths_end(lengths_csv, ee_csv)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    models = [s.strip() for s in str(args.models).split(",") if s.strip()]
    tension_modes = [s.strip() for s in str(args.tension_modes).split(",") if s.strip()]

    summaries: dict[str, Any] = {}
    for split in splits:
        out_dir = Path(f"{args.out_prefix}_{split}_fast4")
        payload = run_one_split(
            dataset_path=args.dataset,
            meta_path=args.meta,
            out_dir=out_dir,
            split=split,
            models=models,
            tension_modes=tension_modes,
            seed=int(args.seed),
            val_size=float(args.val_size),
            test_size=float(args.test_size),
            feature_set=str(args.feature_set),
            tmax=float(args.tmax),
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            beta_source=str(args.beta_source),
        )
        best_key, best_row = min(payload["models"].items(), key=lambda kv: float(kv[1]["metrics"]["test"]["tension_mae_n"]))
        summaries[split] = {"best_model": best_key, "best_test_metrics": best_row["metrics"]["test"]}

    summary_path = Path(f"{args.out_prefix}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "splits": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
