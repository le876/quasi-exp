#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


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


def _normalized_beta(beta: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.nanmax(np.abs(beta), axis=0), 1.0e-9)
    return beta / scale.reshape(1, -1)


def _voxel_key(xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    vox = np.floor(np.asarray(xyz, dtype=float) / float(voxel_size_m)).astype(int)
    return np.asarray([f"{int(a)},{int(b)},{int(c)}" for a, b, c in vox.tolist()], dtype=object)


def _merge(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keep = ["sample_id"] + [c for c in BETA_COLS + ["source_component"] if c in meta.columns]
    df = dataset.merge(meta[keep], on="sample_id", how="left", validate="one_to_one")
    if df[BETA_COLS].isna().any().any():
        raise ValueError("meta is missing beta columns for at least one dataset row")
    return df


def build_branch_labels(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float = 0.01,
    beta_eps_norm: float = 0.15,
    min_branch_size: int = 3,
) -> np.ndarray:
    df = _merge(dataset, meta)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    beta_norm = _normalized_beta(df[BETA_COLS].to_numpy(dtype=float))
    vox = _voxel_key(xyz, voxel_size_m)
    labels = np.empty(len(df), dtype=object)
    for key in sorted(set(vox.tolist())):
        idx = np.where(vox == key)[0]
        if idx.size == 1:
            cluster = np.asarray([0], dtype=int)
        else:
            cluster = DBSCAN(eps=float(beta_eps_norm), min_samples=1).fit_predict(beta_norm[idx])
        for row_idx, branch_id in zip(idx.tolist(), cluster.tolist()):
            labels[row_idx] = f"{key}#b{int(branch_id)}"

    counts = pd.Series(labels).value_counts()
    min_branch_size = max(1, int(min_branch_size))
    labels = np.asarray([str(v) if int(counts[str(v)]) >= min_branch_size else "other_branch" for v in labels], dtype=object)
    return labels


def branch_constrained_nearest_indices(
    *,
    train_xyz: np.ndarray,
    train_labels: np.ndarray,
    query_xyz: np.ndarray,
    pred_labels: np.ndarray,
) -> np.ndarray:
    train_xyz = np.asarray(train_xyz, dtype=float).reshape(-1, 3)
    query_xyz = np.asarray(query_xyz, dtype=float).reshape(-1, 3)
    train_labels = np.asarray(train_labels, dtype=object).reshape(-1)
    pred_labels = np.asarray(pred_labels, dtype=object).reshape(-1)
    global_nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(train_xyz)
    global_idx = global_nn.kneighbors(query_xyz, return_distance=False)[:, 0].astype(int)
    out = np.empty(len(query_xyz), dtype=int)
    cache: dict[str, tuple[np.ndarray, NearestNeighbors]] = {}
    for row, label in enumerate(pred_labels.tolist()):
        label = str(label)
        if label not in cache:
            idx = np.where(train_labels == label)[0]
            if idx.size == 0:
                cache[label] = (idx, global_nn)
            else:
                cache[label] = (idx, NearestNeighbors(n_neighbors=1, algorithm="auto").fit(train_xyz[idx]))
        idx, nn = cache[label]
        if idx.size == 0:
            out[row] = int(global_idx[row])
        else:
            local = int(nn.kneighbors(query_xyz[row : row + 1], return_distance=False)[0, 0])
            out[row] = int(idx[local])
    return out


def _metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xyz_true: np.ndarray,
    xyz_pred: np.ndarray,
    theta_dim: int,
) -> dict[str, Any]:
    theta_true = y_true[:, :theta_dim]
    theta_pred = y_pred[:, :theta_dim]
    tension_true = y_true[:, theta_dim:]
    tension_pred = y_pred[:, theta_dim:]
    theta_mae = np.mean(np.abs(theta_true - theta_pred), axis=1) * (180.0 / np.pi)
    tension_mae = np.mean(np.abs(tension_true - tension_pred), axis=1)
    ee = np.linalg.norm(xyz_true - xyz_pred, axis=1) * 1000.0
    return {
        "theta_mae_deg": float(np.mean(theta_mae)),
        "theta_p95_deg": float(np.percentile(theta_mae, 95)),
        "tension_mae_n": float(np.mean(tension_mae)),
        "tension_p95_n": float(np.percentile(tension_mae, 95)),
        "ee_pos_mae_mm": float(np.mean(ee)),
        "ee_pos_p95_mm": float(np.percentile(ee, 95)),
    }


def evaluate_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    voxel_size_m: float = 0.01,
    beta_eps_norm: float = 0.15,
    min_branch_size: int = 3,
    test_size: float = 0.2,
    seed: int = 20260608,
    classifier_k: int = 5,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    if sample_limit is not None and int(sample_limit) > 0 and len(dataset) > int(sample_limit):
        rng = np.random.default_rng(int(seed))
        pick = np.sort(rng.choice(np.arange(len(dataset)), size=int(sample_limit), replace=False))
        dataset = dataset.iloc[pick].reset_index(drop=True).copy()
        meta = meta.iloc[pick].reset_index(drop=True).copy()
        dataset["sample_id"] = np.arange(len(dataset), dtype=int)
        meta["sample_id"] = np.arange(len(meta), dtype=int)

    labels = build_branch_labels(
        dataset,
        meta,
        voxel_size_m=voxel_size_m,
        beta_eps_norm=beta_eps_norm,
        min_branch_size=min_branch_size,
    )
    theta_cols = _numbered_cols(list(dataset.columns), "theta_", "_rad")
    tension_cols = _numbered_cols(list(dataset.columns), "tension_", "_n")
    if len(theta_cols) != 30 or len(tension_cols) != 12:
        raise ValueError("dataset must contain 30 theta columns and 12 tension columns")
    xyz = dataset[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    y = dataset[theta_cols + tension_cols].to_numpy(dtype=float)
    all_idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(all_idx, test_size=float(test_size), random_state=int(seed), shuffle=True)
    train_xyz = xyz[train_idx]
    test_xyz = xyz[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    clf = KNeighborsClassifier(n_neighbors=min(max(1, int(classifier_k)), len(train_idx)))
    clf.fit(train_xyz, train_labels)
    pred_labels = clf.predict(test_xyz)
    global_idx_local = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(train_xyz).kneighbors(test_xyz, return_distance=False)[:, 0]
    global_train_idx = train_idx[global_idx_local]
    pred_train_idx = train_idx[
        branch_constrained_nearest_indices(
            train_xyz=train_xyz,
            train_labels=train_labels,
            query_xyz=test_xyz,
            pred_labels=pred_labels,
        )
    ]
    oracle_train_idx = train_idx[
        branch_constrained_nearest_indices(
            train_xyz=train_xyz,
            train_labels=train_labels,
            query_xyz=test_xyz,
            pred_labels=test_labels,
        )
    ]

    label_counts = pd.Series(labels).value_counts()
    metrics = {
        "rows": int(len(dataset)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "voxel_size_m": float(voxel_size_m),
        "beta_eps_norm": float(beta_eps_norm),
        "min_branch_size": int(min_branch_size),
        "branch_count": int(label_counts.size),
        "other_branch_ratio": float(np.mean(labels == "other_branch")),
        "classifier": {
            "accuracy": float(np.mean(pred_labels == test_labels)),
            "k": int(clf.n_neighbors),
        },
        "xyz_nn": _metrics(
            y_true=y[test_idx],
            y_pred=y[global_train_idx],
            xyz_true=xyz[test_idx],
            xyz_pred=xyz[global_train_idx],
            theta_dim=30,
        ),
        "classifier_expert": _metrics(
            y_true=y[test_idx],
            y_pred=y[pred_train_idx],
            xyz_true=xyz[test_idx],
            xyz_pred=xyz[pred_train_idx],
            theta_dim=30,
        ),
        "true_branch_oracle": _metrics(
            y_true=y[test_idx],
            y_pred=y[oracle_train_idx],
            xyz_true=xyz[test_idx],
            xyz_pred=xyz[oracle_train_idx],
            theta_dim=30,
        ),
    }
    return metrics


def evaluate(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def _write_report(payload: dict[str, Any], out_path: Path) -> None:
    def fmt(v: Any, digits: int = 3) -> str:
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return str(v)

    lines = [
        "# Branch-Aware Baseline Report",
        "",
        f"- rows: {payload['rows']}",
        f"- branch_count: {payload['branch_count']}",
        f"- other_branch_ratio: {fmt(payload['other_branch_ratio'])}",
        f"- classifier accuracy: {fmt(payload['classifier']['accuracy'])}",
        "",
        "| method | T MAE N | T p95 N | theta MAE deg | theta p95 deg | EE p95 mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ["xyz_nn", "classifier_expert", "true_branch_oracle"]:
        row = payload[key]
        lines.append(
            f"| {key} | {fmt(row['tension_mae_n'], 2)} | {fmt(row['tension_p95_n'], 2)} | "
            f"{fmt(row['theta_mae_deg'], 3)} | {fmt(row['theta_p95_deg'], 3)} | {fmt(row['ee_pos_p95_mm'], 2)} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--voxel-size-mm", type=float, default=10.0)
    ap.add_argument("--beta-eps-norm", type=float, default=0.15)
    ap.add_argument("--min-branch-size", type=int, default=3)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--classifier-k", type=int, default=5)
    ap.add_argument("--sample-limit", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluate(
        args.dataset,
        args.meta,
        voxel_size_m=float(args.voxel_size_mm) / 1000.0,
        beta_eps_norm=float(args.beta_eps_norm),
        min_branch_size=int(args.min_branch_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
        classifier_k=int(args.classifier_k),
        sample_limit=int(args.sample_limit) if int(args.sample_limit) > 0 else None,
    )
    (args.out_dir / "branch_aware_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload, args.out_dir / "BRANCH_AWARE_REPORT.md")
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

