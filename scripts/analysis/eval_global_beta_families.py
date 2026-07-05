#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from features import build_features  # noqa: E402


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


def _merge(dataset: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keep = ["sample_id"] + [c for c in BETA_COLS + ["source_component"] if c in meta.columns]
    df = dataset.merge(meta[keep], on="sample_id", how="left", validate="one_to_one")
    if df[BETA_COLS].isna().any().any():
        raise ValueError("meta is missing beta columns for at least one row")
    return df


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


def beta_matrix(df: pd.DataFrame, theta: np.ndarray, beta_source: str) -> np.ndarray:
    beta_source = str(beta_source)
    if beta_source == "meta":
        return df[BETA_COLS].to_numpy(dtype=float)
    if beta_source == "neg_meta":
        return -df[BETA_COLS].to_numpy(dtype=float)
    if beta_source == "effective_theta":
        return effective_beta_from_theta(theta)
    raise ValueError(f"unsupported beta_source: {beta_source}")


def _normalized_beta(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.max(np.abs(beta), axis=0), 1.0e-9)
    return beta / scale.reshape(1, 6), scale


def _radius_pairs(xyz: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    nn = NearestNeighbors(radius=float(radius_m), algorithm="auto").fit(xyz)
    neigh = nn.radius_neighbors(xyz, return_distance=False)
    pairs: set[tuple[int, int]] = set()
    for row, vals in enumerate(neigh):
        for col in vals.tolist():
            col = int(col)
            if col == row:
                continue
            a, b = (row, col) if row < col else (col, row)
            pairs.add((a, b))
    if not pairs:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    arr = np.asarray(sorted(pairs), dtype=int)
    return arr[:, 0], arr[:, 1]


def _metric_block(i: np.ndarray, j: np.ndarray, theta: np.ndarray, tension: np.ndarray) -> dict[str, Any]:
    i = np.asarray(i, dtype=int).reshape(-1)
    j = np.asarray(j, dtype=int).reshape(-1)
    if i.size == 0:
        return {"pairs": 0}
    theta_rms = np.sqrt(np.mean(np.square(theta[i] - theta[j]), axis=1)) * (180.0 / np.pi)
    tension_mae = np.mean(np.abs(tension[i] - tension[j]), axis=1)
    return {
        "pairs": int(i.size),
        "theta_rms_deg_p50": float(np.percentile(theta_rms, 50)),
        "theta_rms_deg_p95": float(np.percentile(theta_rms, 95)),
        "tension_mae_n_p50": float(np.percentile(tension_mae, 50)),
        "tension_mae_n_p90": float(np.percentile(tension_mae, 90)),
        "tension_mae_n_p95": float(np.percentile(tension_mae, 95)),
    }


def _classifier_metrics(X: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    counts = np.bincount(labels)
    can_stratify = bool(np.all(counts[counts > 0] >= 2) and len(labels) >= 10)
    idx = np.arange(len(labels), dtype=int)
    if can_stratify:
        train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=int(seed), stratify=labels)
    else:
        train_idx = idx
        test_idx = idx

    k = min(5, max(1, len(train_idx)))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[train_idx], labels[train_idx])
    pred_knn = knn.predict(X[test_idx])

    rf = RandomForestClassifier(n_estimators=80, random_state=int(seed), min_samples_leaf=2)
    rf.fit(X[train_idx], labels[train_idx])
    pred_rf = rf.predict(X[test_idx])
    return {
        "knn_accuracy": float(accuracy_score(labels[test_idx], pred_knn)),
        "knn_balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], pred_knn)),
        "rf_accuracy": float(accuracy_score(labels[test_idx], pred_rf)),
        "rf_balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], pred_rf)),
        "test_rows": int(len(test_idx)),
    }


def _family_oracle(xyz: np.ndarray, labels: np.ndarray, theta: np.ndarray, tension: np.ndarray) -> dict[str, Any]:
    global_idx = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(xyz).kneighbors(xyz, return_distance=False)[:, 1]
    same_idx = np.empty(len(xyz), dtype=int)
    for row in range(len(xyz)):
        mask = np.where(labels == labels[row])[0]
        mask = mask[mask != row]
        if mask.size == 0:
            same_idx[row] = int(global_idx[row])
        else:
            d = np.linalg.norm(xyz[mask] - xyz[row], axis=1)
            same_idx[row] = int(mask[int(np.argmin(d))])
    return {
        "global_xyz_nn": _metric_block(np.arange(len(xyz)), global_idx, theta, tension),
        "same_family_xyz_nn": _metric_block(np.arange(len(xyz)), same_idx, theta, tension),
    }


def evaluate_frames(
    dataset: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    k_values: list[int] | None = None,
    min_family_size_gate: int = 100,
    classifier_accuracy_gate: float = 0.70,
    same_family_tension_p95_gate_n: float = 70.0,
    radius_m: float = 0.01,
    seed: int = 20260608,
    feature_set: str = "poly_heavy",
    beta_source: str = "effective_theta",
) -> dict[str, Any]:
    k_values = [8, 16, 32, 64] if k_values is None else [int(v) for v in k_values]
    df = _merge(dataset, meta)
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    X, _names = build_features(xyz, feature_set=feature_set)
    theta_cols = _numbered_cols(list(df.columns), "theta_", "_rad")
    tension_cols = _numbered_cols(list(df.columns), "tension_", "_n")
    theta = df[theta_cols].to_numpy(dtype=float)
    tension = df[tension_cols].to_numpy(dtype=float)
    beta = beta_matrix(df, theta, beta_source=beta_source)
    beta_norm, beta_scale = _normalized_beta(beta)
    pi, pj = _radius_pairs(xyz, radius_m)

    families: dict[str, Any] = {}
    for k in k_values:
        if k <= 1 or k > len(df):
            continue
        km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed))
        labels = km.fit_predict(beta_norm)
        counts = np.bincount(labels, minlength=int(k))
        same = labels[pi] == labels[pj] if pi.size else np.asarray([], dtype=bool)
        same_block = _metric_block(pi[same], pj[same], theta, tension)
        cross_block = _metric_block(pi[~same], pj[~same], theta, tension)
        clf = _classifier_metrics(X, labels, seed=seed)
        oracle = _family_oracle(xyz, labels, theta, tension)
        min_size = int(np.min(counts))
        same_t95 = float(same_block.get("tension_mae_n_p95", float("inf")))
        best_acc = max(float(clf["rf_accuracy"]), float(clf["knn_accuracy"]))
        passed = bool(
            min_size >= int(min_family_size_gate)
            and best_acc >= float(classifier_accuracy_gate)
            and same_t95 <= float(same_family_tension_p95_gate_n)
        )
        families[f"k{k}"] = {
            "k": int(k),
            "passed": passed,
            "family_size_min": min_size,
            "family_size_p10": float(np.percentile(counts, 10)),
            "family_size_p50": float(np.percentile(counts, 50)),
            "family_size_p90": float(np.percentile(counts, 90)),
            "family_size_max": int(np.max(counts)),
            "same_family_10mm": same_block,
            "cross_family_10mm": cross_block,
            "classifier": clf,
            "oracle": oracle,
            "labels": labels.tolist(),
        }

    recommended = None
    passed_rows = [row for row in families.values() if bool(row["passed"])]
    if passed_rows:
        passed_rows.sort(
            key=lambda r: (
                -float(r["classifier"]["rf_accuracy"]),
                float(r["same_family_10mm"].get("tension_mae_n_p95", float("inf"))),
                int(r["k"]),
            )
        )
        recommended = int(passed_rows[0]["k"])
    return {
        "rows": int(len(df)),
        "beta_source": str(beta_source),
        "k_values": k_values,
        "radius_m": float(radius_m),
        "beta_scale": beta_scale.tolist(),
        "gates": {
            "min_family_size": int(min_family_size_gate),
            "classifier_accuracy": float(classifier_accuracy_gate),
            "same_family_tension_p95_n": float(same_family_tension_p95_gate_n),
        },
        "recommended_k": recommended,
        "families": families,
    }


def evaluate(dataset_path: Path, meta_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate_frames(pd.read_parquet(dataset_path), pd.read_parquet(meta_path), **kwargs)


def _strip_labels(payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    for row in out.get("families", {}).values():
        row.pop("labels", None)
    return out


def _write_report(payload: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Global Beta Families Report",
        "",
        f"- rows: {payload['rows']}",
        f"- beta_source: `{payload.get('beta_source', 'unknown')}`",
        f"- recommended_k: `{payload['recommended_k']}`",
        "",
        "| k | pass | min size | RF acc | KNN acc | same-family T p95 N | cross-family T p95 N |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in sorted(payload["families"].items(), key=lambda kv: int(kv[1]["k"])):
        lines.append(
            f"| {row['k']} | {bool(row['passed'])} | {row['family_size_min']} | "
            f"{float(row['classifier']['rf_accuracy']):.3f} | {float(row['classifier']['knn_accuracy']):.3f} | "
            f"{float(row['same_family_10mm'].get('tension_mae_n_p95', float('nan'))):.2f} | "
            f"{float(row['cross_family_10mm'].get('tension_mae_n_p95', float('nan'))):.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--k-values", default="8,16,32,64")
    ap.add_argument("--radius-mm", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--beta-source", default="effective_theta", choices=["effective_theta", "meta", "neg_meta"])
    ap.add_argument("--min-family-size-gate", type=int, default=100)
    ap.add_argument("--classifier-accuracy-gate", type=float, default=0.70)
    ap.add_argument("--same-family-tension-p95-gate-n", type=float, default=70.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(v.strip()) for v in str(args.k_values).split(",") if v.strip()]
    payload = evaluate(
        args.dataset,
        args.meta,
        k_values=k_values,
        radius_m=float(args.radius_mm) / 1000.0,
        seed=int(args.seed),
        beta_source=str(args.beta_source),
        min_family_size_gate=int(args.min_family_size_gate),
        classifier_accuracy_gate=float(args.classifier_accuracy_gate),
        same_family_tension_p95_gate_n=float(args.same_family_tension_p95_gate_n),
    )
    (args.out_dir / "global_beta_families_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "global_beta_families_summary_compact.json").write_text(
        json.dumps(_strip_labels(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(payload, args.out_dir / "GLOBAL_BETA_FAMILIES_REPORT.md")
    print(json.dumps({"out_dir": str(args.out_dir), "recommended_k": payload["recommended_k"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
