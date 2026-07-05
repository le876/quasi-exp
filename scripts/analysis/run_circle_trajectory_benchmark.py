#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / "scripts" / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    import joblib
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"joblib is required to load trained models: {exc}") from exc

from features import build_features  # noqa: E402
from fk_dh_numpy import fk_dh_batch  # noqa: E402
from run_baselines import XYScaler, _load_robot_paths, _read_lengths_end  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data" / "priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1" / "dataset.parquet"
DEFAULT_SPLIT_ROOT = REPO_ROOT / "runs" / "baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_v1"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "visualizations" / "circle_trajectory_benchmark_relabel_100k_v1"

SPLITS = ("iid", "radius", "beta_block", "angular_sector")
STRICT_XBIAS_OUT_DIR = REPO_ROOT / "runs" / "visualizations" / "circle_trajectory_benchmark_strict_xbias_relabel_100k_v1"


class CircleCandidate(NamedTuple):
    candidate_id: str
    x0_m: float
    center_y_m: float
    center_z_m: float
    radius_m: float
    n_points: int


class ModelSpec(NamedTuple):
    model_id: str
    family: str
    dataset_key: str
    train_split: str
    model_name: str
    model_path: Path
    scaler_path: Path
    feature_set: str = "poly_heavy"


class WorkspaceContext(NamedTuple):
    xyz_m: np.ndarray
    nn: NearestNeighbors
    split_membership: dict[str, np.ndarray]


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _boxplot_with_labels(ax, data, labels) -> None:
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:  # pragma: no cover - older matplotlib compatibility.
        ax.boxplot(data, labels=labels, showfliers=False)


def _parse_float_list(raw: str) -> list[float]:
    return [float(v.strip()) for v in str(raw).split(",") if v.strip()]


def _parse_float_range(raw: str) -> list[float]:
    raw = str(raw).strip()
    if not raw:
        return []
    parts = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(parts) != 3:
        raise ValueError("--x-range must be formatted as start,stop,step")
    start, stop, step = parts
    if step <= 0.0:
        raise ValueError("--x-range step must be positive")
    values = np.arange(start, stop + step * 0.5, step, dtype=float)
    return [float(np.round(v, 6)) for v in values]


def _parse_centers(raw: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in str(raw).split(";"):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(",", 1)
        out.append((float(left.strip()), float(right.strip())))
    return out


def _candidate_id(x0: float, center_y: float, center_z: float, radius: float, x_decimals: int = 2) -> str:
    return f"x{x0:.{int(x_decimals)}f}_cy{center_y:.3f}_cz{center_z:.3f}_r{radius:.3f}".replace("-", "m").replace(".", "p")


def make_offset_circle_points(
    x0: float,
    center_y: float,
    center_z: float,
    radius: float,
    n_points: int,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Create a circle in the y-z plane with fixed x and configurable y-z center."""
    if float(radius) <= 0.0:
        raise ValueError("radius must be positive")
    if int(n_points) < 8:
        raise ValueError("n_points must be at least 8")
    angles = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False, dtype=float) + float(phase_rad)
    points = np.zeros((int(n_points), 3), dtype=float)
    points[:, 0] = float(x0)
    points[:, 1] = float(center_y) + float(radius) * np.cos(angles)
    points[:, 2] = float(center_z) + float(radius) * np.sin(angles)
    return points


def build_candidate_grid(
    x_values: list[float],
    centers: list[tuple[float, float]],
    radii: list[float],
    n_points: int,
    x_id_decimals: int = 2,
) -> list[CircleCandidate]:
    candidates: list[CircleCandidate] = []
    seen: set[str] = set()
    for x0 in x_values:
        for center_y, center_z in centers:
            for radius in radii:
                cid = _candidate_id(float(x0), float(center_y), float(center_z), float(radius), x_decimals=int(x_id_decimals))
                if cid in seen:
                    continue
                seen.add(cid)
                candidates.append(
                    CircleCandidate(
                        candidate_id=cid,
                        x0_m=float(x0),
                        center_y_m=float(center_y),
                        center_z_m=float(center_z),
                        radius_m=float(radius),
                        n_points=int(n_points),
                    )
                )
    return candidates


def build_builtin_model_registry(repo_root: Path = REPO_ROOT) -> list[ModelSpec]:
    repo_root = Path(repo_root)
    specs: list[ModelSpec] = []
    best_root = (
        repo_root
        / "runs"
        / "mlp_capacity_fixed_layer_s1_0125_s2_0250_100k_relabel_v1"
        / "grid"
        / "relabel"
        / "iid"
        / "grid_relabel_iid_L2_alpha1em05_es"
    )
    specs.append(
        ModelSpec(
            model_id="best_capacity_relabel_iid_L2_mlp_large",
            family="direct_capacity",
            dataset_key="relabel",
            train_split="iid",
            model_name="mlp_large",
            model_path=best_root / "mlp_large" / "model.joblib",
            scaler_path=best_root / "scaler.joblib",
        )
    )

    baseline_root = repo_root / "runs" / "baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_v1"
    for split in SPLITS:
        for model_name in ("mlp", "mlp_large"):
            specs.append(
                ModelSpec(
                    model_id=f"relabel_{split}_{model_name}",
                    family="direct_baseline",
                    dataset_key="relabel",
                    train_split=split,
                    model_name=model_name,
                    model_path=baseline_root / split / model_name / "model.joblib",
                    scaler_path=baseline_root / split / "scaler.joblib",
                )
            )

    comparison_root = repo_root / "runs" / "baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_compare_v2"
    for split in SPLITS:
        for model_name in ("knn", "rf", "lgbm"):
            specs.append(
                ModelSpec(
                    model_id=f"relabel_{split}_{model_name}",
                    family="direct_comparison",
                    dataset_key="relabel",
                    train_split=split,
                    model_name=model_name,
                    model_path=comparison_root / split / model_name / model_name / "model.joblib",
                    scaler_path=comparison_root / split / model_name / "scaler.joblib",
                )
            )
    return specs


def filter_existing_models(specs: list[ModelSpec], requested: set[str] | None = None) -> list[ModelSpec]:
    out = []
    for spec in specs:
        if requested is not None and spec.model_id not in requested and spec.model_name not in requested:
            continue
        if spec.model_path.exists() and spec.scaler_path.exists():
            out.append(spec)
    return out


def split_membership_from_npz(split_path: Path, n_rows: int) -> np.ndarray:
    membership = np.full(int(n_rows), "unknown", dtype=object)
    data = np.load(str(split_path), allow_pickle=False)
    for key, label in (("train_idx", "train"), ("val_idx", "val"), ("test_idx", "test")):
        if key in data:
            membership[data[key].astype(np.int64)] = label
    return membership


def load_workspace_context(dataset_path: Path, split_root: Path) -> WorkspaceContext:
    df = pd.read_parquet(dataset_path, columns=["x_m", "y_m", "z_m"])
    xyz = df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(xyz)
    memberships: dict[str, np.ndarray] = {}
    for split in SPLITS:
        split_path = Path(split_root) / split / "split.npz"
        if split_path.exists():
            memberships[split] = split_membership_from_npz(split_path, len(xyz))
    return WorkspaceContext(xyz_m=xyz, nn=nn, split_membership=memberships)


def compute_support_metrics(nn_dist_m: np.ndarray, max_nn_p95_m: float) -> dict[str, float | bool]:
    dist = np.asarray(nn_dist_m, dtype=float).reshape(-1)
    if dist.size == 0:
        raise ValueError("nn_dist_m must be non-empty")
    p95_m = float(np.percentile(dist, 95))
    return {
        "nn_dist_mean_mm": float(np.mean(dist) * 1000.0),
        "nn_dist_p95_mm": float(p95_m * 1000.0),
        "nn_dist_max_mm": float(np.max(dist) * 1000.0),
        "supported": bool(p95_m <= float(max_nn_p95_m)),
    }


def has_fixed_axis_bias(values_mm: np.ndarray, threshold_mm: float) -> bool:
    values = np.asarray(values_mm, dtype=float).reshape(-1)
    if values.size == 0:
        return False
    threshold = float(threshold_mm)
    return bool(np.all(values > threshold) or np.all(values < -threshold))


def compute_x_error_metrics(x_error_mm: np.ndarray, fixed_bias_threshold_mm: float) -> dict[str, float | bool]:
    xerr = np.asarray(x_error_mm, dtype=float).reshape(-1)
    if xerr.size == 0:
        raise ValueError("x_error_mm must be non-empty")
    pos_ratio = float(np.mean(xerr > 0.0))
    neg_ratio = float(np.mean(xerr < 0.0))
    suffix = str(int(float(fixed_bias_threshold_mm))).replace("-", "m").replace(".", "p")
    return {
        "xerr_mean_mm": float(np.mean(xerr)),
        "xerr_p50_mm": float(np.percentile(xerr, 50)),
        "xerr_p95_abs_mm": float(np.percentile(np.abs(xerr), 95)),
        "xerr_min_mm": float(np.min(xerr)),
        "xerr_max_mm": float(np.max(xerr)),
        "xerr_same_sign_ratio": float(max(pos_ratio, neg_ratio)),
        f"fixed_x_bias_gt{suffix}mm": has_fixed_axis_bias(xerr, threshold_mm=float(fixed_bias_threshold_mm)),
    }


def compute_nearest_x_bias_metrics(
    target_xyz_m: np.ndarray,
    nearest_xyz_m: np.ndarray,
    fixed_bias_threshold_mm: float,
    max_nearest_x_mean_abs_mm: float | None = None,
) -> dict[str, float | bool]:
    target = np.asarray(target_xyz_m, dtype=float)
    nearest = np.asarray(nearest_xyz_m, dtype=float)
    if target.shape != nearest.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target_xyz_m and nearest_xyz_m must both be shaped (N, 3)")
    xdiff = (nearest[:, 0] - target[:, 0]) * 1000.0
    suffix = str(int(float(fixed_bias_threshold_mm))).replace("-", "m").replace(".", "p")
    mean_abs_limit = float("inf") if max_nearest_x_mean_abs_mm is None else float(max_nearest_x_mean_abs_mm)
    return {
        "nearest_x_diff_mean_mm": float(np.mean(xdiff)),
        "nearest_x_diff_p50_mm": float(np.percentile(xdiff, 50)),
        "nearest_x_diff_p95_abs_mm": float(np.percentile(np.abs(xdiff), 95)),
        "nearest_x_diff_min_mm": float(np.min(xdiff)),
        "nearest_x_diff_max_mm": float(np.max(xdiff)),
        f"fixed_nearest_x_bias_gt{suffix}mm": has_fixed_axis_bias(xdiff, threshold_mm=float(fixed_bias_threshold_mm)),
        "nearest_x_mean_abs_within_limit": bool(abs(float(np.mean(xdiff))) <= mean_abs_limit),
    }


def classify_split_region(
    nearest_idx: np.ndarray,
    split_membership: np.ndarray,
    majority_threshold: float = 0.60,
) -> dict[str, float | str]:
    idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
    labels = np.asarray(split_membership, dtype=object)[idx]
    ratios: dict[str, float] = {}
    for label in ("train", "val", "test", "unknown"):
        ratios[label] = float(np.mean(labels == label)) if labels.size else 0.0
    dominant = max(("train", "val", "test"), key=lambda label: ratios[label])
    region = f"{dominant}_majority" if ratios[dominant] >= float(majority_threshold) else "mixed"
    return {
        "region": region,
        "train_ratio": ratios["train"],
        "val_ratio": ratios["val"],
        "test_ratio": ratios["test"],
        "unknown_ratio": ratios["unknown"],
    }


def candidate_workspace_metadata(
    target_xyz_m: np.ndarray,
    context: WorkspaceContext,
    max_nn_p95_m: float,
    majority_threshold: float,
    max_nearest_x_mean_abs_mm: float | None = None,
    fixed_bias_threshold_mm: float = 2.0,
) -> tuple[dict[str, float | bool | str], np.ndarray]:
    dist, idx = context.nn.kneighbors(np.asarray(target_xyz_m, dtype=float), n_neighbors=1, return_distance=True)
    nearest_idx = idx[:, 0].astype(np.int64)
    out: dict[str, float | bool | str] = compute_support_metrics(dist[:, 0], max_nn_p95_m=max_nn_p95_m)
    nearest_xyz = context.xyz_m[nearest_idx]
    out.update(
        compute_nearest_x_bias_metrics(
            target_xyz_m,
            nearest_xyz,
            fixed_bias_threshold_mm=float(fixed_bias_threshold_mm),
            max_nearest_x_mean_abs_mm=max_nearest_x_mean_abs_mm,
        )
    )
    out["strict_geometry_supported"] = bool(
        bool(out["supported"]) and bool(out["nearest_x_mean_abs_within_limit"])
    )
    for split, membership in context.split_membership.items():
        region = classify_split_region(nearest_idx, membership, majority_threshold=majority_threshold)
        out[f"{split}_region"] = str(region["region"])
        out[f"{split}_train_ratio"] = float(region["train_ratio"])
        out[f"{split}_val_ratio"] = float(region["val_ratio"])
        out[f"{split}_test_ratio"] = float(region["test_ratio"])
        out[f"{split}_unknown_ratio"] = float(region["unknown_ratio"])
    return out, nearest_idx


def filter_strict_geometry_candidates(
    candidates: list[CircleCandidate],
    candidate_meta: dict[str, dict[str, float | bool | str]],
    min_radius_m: float,
    max_nearest_x_mean_abs_mm: float,
) -> list[CircleCandidate]:
    selected: list[CircleCandidate] = []
    for candidate in candidates:
        meta = candidate_meta.get(candidate.candidate_id, {})
        if float(candidate.radius_m) < float(min_radius_m):
            continue
        if not bool(meta.get("supported", False)):
            continue
        if abs(float(meta.get("nearest_x_diff_mean_mm", float("inf")))) > float(max_nearest_x_mean_abs_mm):
            continue
        selected.append(candidate)
    return selected


def _load_model_and_scaler(model_path: Path, scaler_path: Path):
    import __main__  # noqa: PLC0415

    if not hasattr(__main__, "XYScaler"):
        setattr(__main__, "XYScaler", XYScaler)
    return joblib.load(model_path), joblib.load(scaler_path)


def predict_theta_tension(model, scaler: XYScaler, xyz_m: np.ndarray, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    features, _ = build_features(np.asarray(xyz_m, dtype=float), feature_set=feature_set)
    pred_norm = np.asarray(model.predict(scaler.transform_X(features)), dtype=float)
    theta_dim = int(np.asarray(scaler.th_mean_).reshape(-1).shape[0])
    tension_dim = int(np.asarray(scaler.t_mean_).reshape(-1).shape[0])
    if pred_norm.ndim != 2 or pred_norm.shape[1] != theta_dim + tension_dim:
        raise ValueError(f"model prediction shape {pred_norm.shape} does not match scaler output dims")
    theta, tension = scaler.inverse_Y(pred_norm[:, :theta_dim], pred_norm[:, theta_dim:])
    return np.asarray(theta, dtype=float), np.asarray(tension, dtype=float)


def compute_tracking_metrics(
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    tension_n: np.ndarray,
    tension_upper_n: float,
    fixed_bias_threshold_mm: float = 2.0,
) -> dict[str, float]:
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    tension = np.asarray(tension_n, dtype=float)
    err_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
    tension_f = tension.reshape(-1)
    metrics = {
        "ee_mean_mm": float(np.mean(err_mm)),
        "ee_rmse_mm": float(np.sqrt(np.mean(np.square(err_mm)))),
        "ee_p50_mm": float(np.percentile(err_mm, 50)),
        "ee_p95_mm": float(np.percentile(err_mm, 95)),
        "ee_max_mm": float(np.max(err_mm)),
        "tension_min_n": float(np.min(tension_f)),
        "tension_max_n": float(np.max(tension_f)),
        "tension_mean_n": float(np.mean(tension_f)),
        "negative_tension_ratio": float(np.mean(tension_f < 0.0)),
        "over_upper_tension_ratio": float(np.mean(tension_f > float(tension_upper_n))),
    }
    xerr_mm = (achieved[:, 0] - target[:, 0]) * 1000.0
    metrics.update(compute_x_error_metrics(xerr_mm, fixed_bias_threshold_mm=fixed_bias_threshold_mm))
    return metrics


def evaluate_model_candidate(
    spec: ModelSpec,
    candidate: CircleCandidate,
    target_xyz_m: np.ndarray,
    workspace_meta: dict[str, float | bool | str],
    model,
    scaler: XYScaler,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    tension_upper_n: float,
    fixed_bias_threshold_mm: float = 2.0,
) -> tuple[dict[str, float | str | bool], pd.DataFrame]:
    t0 = time.perf_counter()
    theta, tension = predict_theta_tension(model, scaler, target_xyz_m, feature_set=spec.feature_set)
    pred_time_s = float(time.perf_counter() - t0)
    achieved = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    metrics = compute_tracking_metrics(
        target_xyz_m,
        achieved,
        tension,
        tension_upper_n=tension_upper_n,
        fixed_bias_threshold_mm=fixed_bias_threshold_mm,
    )
    row: dict[str, float | str | bool] = {
        "model_id": spec.model_id,
        "family": spec.family,
        "dataset_key": spec.dataset_key,
        "train_split": spec.train_split,
        "model_name": spec.model_name,
        "candidate_id": candidate.candidate_id,
        "x0_m": candidate.x0_m,
        "center_y_m": candidate.center_y_m,
        "center_z_m": candidate.center_z_m,
        "radius_m": candidate.radius_m,
        "n_points": candidate.n_points,
        "pred_time_s": pred_time_s,
        "pred_time_ms_per_point": pred_time_s * 1000.0 / float(candidate.n_points),
        **workspace_meta,
        **metrics,
    }
    point_data = {
        "model_id": [spec.model_id] * candidate.n_points,
        "candidate_id": [candidate.candidate_id] * candidate.n_points,
        "angle_deg": np.linspace(0.0, 360.0, candidate.n_points, endpoint=False, dtype=float),
        "target_x_m": target_xyz_m[:, 0],
        "target_y_m": target_xyz_m[:, 1],
        "target_z_m": target_xyz_m[:, 2],
        "achieved_x_m": achieved[:, 0],
        "achieved_y_m": achieved[:, 1],
        "achieved_z_m": achieved[:, 2],
        "ee_error_mm": np.linalg.norm(achieved - target_xyz_m, axis=1) * 1000.0,
        "x_error_mm": (achieved[:, 0] - target_xyz_m[:, 0]) * 1000.0,
    }
    for i in range(theta.shape[1]):
        point_data[f"theta{i + 1}_rad"] = theta[:, i]
    for i in range(tension.shape[1]):
        point_data[f"T{i + 1}_N"] = tension[:, i]
    return row, pd.DataFrame(point_data)


def select_best_row(
    rows: list[dict[str, float | str | bool]],
    min_radius_m: float,
    max_negative_tension_ratio: float = 0.0,
    max_over_upper_tension_ratio: float = 0.0,
    require_supported: bool = True,
    require_no_fixed_x_bias: bool = False,
    fixed_bias_threshold_mm: float = 2.0,
) -> dict[str, float | str | bool]:
    valid = []
    fixed_key = f"fixed_x_bias_gt{str(int(float(fixed_bias_threshold_mm))).replace('-', 'm').replace('.', 'p')}mm"
    for row in rows:
        if float(row["radius_m"]) < float(min_radius_m):
            continue
        if require_supported and not bool(row.get("supported", False)):
            continue
        if require_no_fixed_x_bias and bool(row.get(fixed_key, False)):
            continue
        if float(row["negative_tension_ratio"]) > float(max_negative_tension_ratio):
            continue
        if float(row["over_upper_tension_ratio"]) > float(max_over_upper_tension_ratio):
            continue
        valid.append(row)
    if not valid:
        raise ValueError("no valid benchmark row after support/radius/tension filters")
    return min(
        valid,
        key=lambda r: (
            float(r["ee_p95_mm"]),
            float(r.get("xerr_p95_abs_mm", 0.0)),
            float(r["ee_rmse_mm"]),
            -float(r["radius_m"]),
        ),
    )


def select_representative_rows(
    rows: list[dict[str, float | str | bool]],
    min_radius_m: float,
    require_no_fixed_x_bias: bool = False,
    fixed_bias_threshold_mm: float = 2.0,
) -> dict[str, dict[str, float | str | bool]]:
    reps: dict[str, dict[str, float | str | bool]] = {}
    try:
        reps["global_best"] = select_best_row(
            rows,
            min_radius_m=min_radius_m,
            require_no_fixed_x_bias=require_no_fixed_x_bias,
            fixed_bias_threshold_mm=fixed_bias_threshold_mm,
        )
    except ValueError:
        reps["global_best"] = min(rows, key=lambda r: (float(r["ee_p95_mm"]), float(r["ee_rmse_mm"])))
    supported = [r for r in rows if bool(r.get("supported", False)) and float(r["radius_m"]) >= min_radius_m]
    if supported:
        reps["supported_worst"] = max(supported, key=lambda r: float(r["ee_p95_mm"]))
    for model_id in sorted({str(r["model_id"]) for r in rows}):
        model_rows = [r for r in rows if str(r["model_id"]) == model_id]
        try:
            reps[f"best_model__{model_id}"] = select_best_row(
                model_rows,
                min_radius_m=min_radius_m,
                require_no_fixed_x_bias=require_no_fixed_x_bias,
                fixed_bias_threshold_mm=fixed_bias_threshold_mm,
            )
        except ValueError:
            continue
    for split in SPLITS:
        col = f"{split}_region"
        for region in ("train_majority", "val_majority", "test_majority", "mixed"):
            region_rows = [r for r in rows if str(r.get(col, "")) == region and bool(r.get("supported", False))]
            if region_rows:
                reps[f"best_{split}_{region}"] = min(region_rows, key=lambda r: float(r["ee_p95_mm"]))
                reps[f"worst_{split}_{region}"] = max(region_rows, key=lambda r: float(r["ee_p95_mm"]))
    return reps


def _plot_model_error_boxplot(summary: pd.DataFrame, out_path: Path) -> None:
    data = [g["ee_p95_mm"].to_numpy(dtype=float) for _, g in summary.groupby("model_id", sort=True)]
    labels = [str(k) for k, _ in summary.groupby("model_id", sort=True)]
    fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 0.45), 5.8))
    _boxplot_with_labels(ax, data, labels)
    ax.set_ylabel("EE p95 error (mm)")
    ax.set_title("Circular trajectory error by model")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=70, labelsize=8)
    _savefig(fig, out_path)


def _plot_region_boxplot(summary: pd.DataFrame, split: str, out_path: Path) -> None:
    col = f"{split}_region"
    order = ["train_majority", "val_majority", "test_majority", "mixed"]
    data = [summary.loc[summary[col] == region, "ee_p95_mm"].to_numpy(dtype=float) for region in order if region in set(summary[col].astype(str))]
    labels = [region for region in order if region in set(summary[col].astype(str))]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _boxplot_with_labels(ax, data, labels)
    ax.set_ylabel("EE p95 error (mm)")
    ax.set_title(f"Error by {split} workspace region")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    _savefig(fig, out_path)


def _plot_heatmap_best_model(summary: pd.DataFrame, best_model_id: str, out_path: Path) -> None:
    sub = summary[(summary["model_id"] == best_model_id) & (np.isclose(summary["center_y_m"], 0.0)) & (np.isclose(summary["center_z_m"], 0.0))]
    if sub.empty:
        return
    piv = sub.pivot_table(index="radius_m", columns="x0_m", values="ee_p95_mm", aggfunc="min")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    im = ax.imshow(piv.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in piv.index])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("radius (m)")
    ax.set_title(f"Center (0,0) x-radius scan: {best_model_id}")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("EE p95 error (mm)")
    _savefig(fig, out_path)


def _plot_center_scatter(summary: pd.DataFrame, out_path: Path) -> None:
    center_err = summary.groupby(["center_y_m", "center_z_m"], as_index=False)["ee_p95_mm"].median()
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    sc = ax.scatter(center_err["center_y_m"], center_err["center_z_m"], c=center_err["ee_p95_mm"], s=130, cmap="viridis_r", edgecolors="black", linewidths=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("circle center y (m)")
    ax.set_ylabel("circle center z (m)")
    ax.set_title("Median trajectory error by y-z center offset")
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("median EE p95 error (mm)")
    _savefig(fig, out_path)


def _plot_support_vs_error(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    supported = summary["supported"].astype(bool)
    ax.scatter(summary.loc[~supported, "nn_dist_p95_mm"], summary.loc[~supported, "ee_p95_mm"], s=28, alpha=0.45, label="out of support", color="#9a9a9a")
    ax.scatter(summary.loc[supported, "nn_dist_p95_mm"], summary.loc[supported, "ee_p95_mm"], s=28, alpha=0.55, label="supported", color="#1f77b4")
    ax.set_xlabel("workspace NN distance p95 (mm)")
    ax.set_ylabel("EE p95 error (mm)")
    ax.set_title("Workspace support vs circular tracking error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _savefig(fig, out_path)


def _axis_limits_for_points(points: pd.DataFrame, pad_frac: float = 0.12) -> dict[str, tuple[float, float]]:
    limits = {}
    for axis in ("x", "y", "z"):
        vals = np.concatenate([points[f"target_{axis}_m"].to_numpy(dtype=float), points[f"achieved_{axis}_m"].to_numpy(dtype=float)])
        lo, hi = float(np.min(vals)), float(np.max(vals))
        span = max(hi - lo, 0.05)
        pad = span * float(pad_frac)
        limits[axis] = (lo - pad, hi + pad)
    return limits


def _plot_trajectory_yz(points: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.8))
    ax.plot(points["target_y_m"], points["target_z_m"], color="#1f4e79", lw=2.2, label="target")
    ax.plot(points["achieved_y_m"], points["achieved_z_m"], color="#c7511f", lw=2.0, label="model + FK")
    ax.scatter(points["target_y_m"].iloc[0], points["target_z_m"].iloc[0], s=42, color="#1f4e79")
    ax.scatter(points["achieved_y_m"].iloc[0], points["achieved_z_m"].iloc[0], s=42, color="#c7511f", marker="x")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _savefig(fig, out_path)


def _plot_trajectory_3d(points: pd.DataFrame, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points["target_x_m"], points["target_y_m"], points["target_z_m"], color="#1f4e79", lw=2.2, label="target")
    ax.plot(points["achieved_x_m"], points["achieved_y_m"], points["achieved_z_m"], color="#c7511f", lw=2.0, label="model + FK")
    limits = _axis_limits_for_points(points)
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-62)
    ax.legend(frameon=False)
    _savefig(fig, out_path)


def write_readme(out_dir: Path, payload: dict[str, object], images: list[str]) -> None:
    best = payload.get("global_best", {})
    lines = [
        "# Circle Trajectory Benchmark",
        "",
        "This benchmark evaluates open-loop circular tracking as `target xyz -> model -> theta/T -> FK -> achieved xyz`.",
        "",
        "## Global Best",
        "",
    ]
    if isinstance(best, dict) and best:
        lines.extend(
            [
                f"- model: `{best.get('model_id')}`",
                f"- candidate: `{best.get('candidate_id')}`",
                f"- EE p95: `{float(best.get('ee_p95_mm', float('nan'))):.3f} mm`",
                f"- EE max: `{float(best.get('ee_max_mm', float('nan'))):.3f} mm`",
                f"- radius: `{float(best.get('radius_m', float('nan'))):.4f} m`",
                f"- supported: `{best.get('supported')}`",
                f"- tension min/max: `{float(best.get('tension_min_n', float('nan'))):.2f} / {float(best.get('tension_max_n', float('nan'))):.2f} N`",
            ]
        )
    lines.extend(["", "## Files", ""])
    for image in images:
        lines.append(f"- `{image}`")
    lines.extend(
        [
            "- `trajectory_benchmark_summary.csv`: model-candidate metrics.",
            "- `trajectory_benchmark_summary.json`: rollup and representative rows.",
            "- `selected_trajectory_points.parquet`: point-level data for representative trajectories.",
            "",
            "Note: this is not a closed-loop controller simulation.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = load_workspace_context(Path(args.dataset), Path(args.split_root))
    lengths_csv, ee_csv = _load_robot_paths(args.robot_config, None, None)
    lengths_m, p_end_local_m = _read_lengths_end(lengths_csv, ee_csv)

    use_x_range = bool(str(args.x_range).strip())
    x_values = _parse_float_range(args.x_range) if use_x_range else _parse_float_list(args.x_values)
    candidates = build_candidate_grid(
        x_values=x_values,
        centers=_parse_centers(args.centers),
        radii=_parse_float_list(args.radii),
        n_points=int(args.n_points),
        x_id_decimals=3 if use_x_range else 2,
    )
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    if not candidates:
        raise SystemExit("no candidates generated")

    requested_models = {m.strip() for m in str(args.models).split(",") if m.strip()} if str(args.models).strip() else None
    specs = filter_existing_models(build_builtin_model_registry(REPO_ROOT), requested=requested_models)
    if int(args.max_models) > 0:
        specs = specs[: int(args.max_models)]
    if not specs:
        raise SystemExit("no existing models selected")

    candidate_targets: dict[str, np.ndarray] = {}
    candidate_meta: dict[str, dict[str, float | bool | str]] = {}
    for candidate in candidates:
        target = make_offset_circle_points(candidate.x0_m, candidate.center_y_m, candidate.center_z_m, candidate.radius_m, candidate.n_points)
        meta, _ = candidate_workspace_metadata(
            target,
            context,
            max_nn_p95_m=float(args.max_nn_p95_m),
            majority_threshold=float(args.majority_threshold),
            max_nearest_x_mean_abs_mm=float(args.max_nearest_x_mean_abs_mm) if float(args.max_nearest_x_mean_abs_mm) >= 0.0 else None,
            fixed_bias_threshold_mm=float(args.fixed_bias_threshold_mm),
        )
        candidate_targets[candidate.candidate_id] = target
        candidate_meta[candidate.candidate_id] = meta
    candidates_generated = len(candidates)
    if bool(args.require_strict_geometry):
        candidates = filter_strict_geometry_candidates(
            candidates,
            candidate_meta,
            min_radius_m=float(args.min_showcase_radius_m),
            max_nearest_x_mean_abs_mm=float(args.max_nearest_x_mean_abs_mm),
        )
        if int(args.max_candidates) > 0:
            candidates = candidates[: int(args.max_candidates)]
        if not candidates:
            raise SystemExit(
                "no candidates remain after strict geometry filtering; relax --max-nn-p95-m/--max-nearest-x-mean-abs-mm or expand --x-range"
            )

    rows: list[dict[str, float | str | bool]] = []
    representative_points: list[pd.DataFrame] = []
    point_frames: dict[tuple[str, str], pd.DataFrame] = {}
    load_errors: list[dict[str, str]] = []

    for spec in specs:
        try:
            model, scaler = _load_model_and_scaler(spec.model_path, spec.scaler_path)
        except Exception as exc:
            load_errors.append({"model_id": spec.model_id, "error": repr(exc)})
            continue
        for candidate in candidates:
            row, points = evaluate_model_candidate(
                spec=spec,
                candidate=candidate,
                target_xyz_m=candidate_targets[candidate.candidate_id],
                workspace_meta=candidate_meta[candidate.candidate_id],
                model=model,
                scaler=scaler,
                lengths_m=lengths_m,
                p_end_local_m=p_end_local_m,
                tension_upper_n=float(args.tension_upper_n),
                fixed_bias_threshold_mm=float(args.fixed_bias_threshold_mm),
            )
            rows.append(row)
            point_frames[(spec.model_id, candidate.candidate_id)] = points
    if not rows:
        raise SystemExit(f"no model-candidate rows evaluated; load_errors={load_errors}")

    summary = pd.DataFrame(rows).sort_values(["ee_p95_mm", "ee_rmse_mm", "radius_m"], ascending=[True, True, False])
    summary_csv = out_dir / "trajectory_benchmark_summary.csv"
    summary.to_csv(summary_csv, index=False)

    reps = select_representative_rows(
        rows,
        min_radius_m=float(args.min_showcase_radius_m),
        require_no_fixed_x_bias=bool(args.require_no_fixed_x_bias_for_best),
        fixed_bias_threshold_mm=float(args.fixed_bias_threshold_mm),
    )
    for label, row in reps.items():
        key = (str(row["model_id"]), str(row["candidate_id"]))
        frame = point_frames.get(key)
        if frame is not None:
            representative_points.append(frame.assign(representative_label=label))
    points_path = out_dir / "selected_trajectory_points.parquet"
    if representative_points:
        pd.concat(representative_points, ignore_index=True).to_parquet(points_path, index=False)

    best = reps.get("global_best", {})
    by_model = {}
    for model_id, group in summary.groupby("model_id", sort=True):
        model_rows = group.to_dict(orient="records")
        try:
            best_row = select_best_row(
                model_rows,
                min_radius_m=float(args.min_showcase_radius_m),
                require_no_fixed_x_bias=bool(args.require_no_fixed_x_bias_for_best),
                fixed_bias_threshold_mm=float(args.fixed_bias_threshold_mm),
            )
            best_row = {**best_row, "selection_note": "supported_radius_tension_filtered"}
        except ValueError:
            fallback = group.sort_values(["ee_p95_mm", "ee_rmse_mm", "radius_m"], ascending=[True, True, False]).iloc[0].to_dict()
            best_row = {**fallback, "selection_note": "fallback_unfiltered_no_supported_valid_row"}
        by_model[str(model_id)] = {k: _jsonable(v) for k, v in best_row.items()}

    payload: dict[str, object] = {
        "mode": "circle_trajectory_benchmark",
        "generated_at": time.strftime("%F %T"),
        "dataset": str(args.dataset),
        "models_evaluated": len({r["model_id"] for r in rows}),
        "candidates_generated": candidates_generated,
        "candidates_evaluated": len(candidates),
        "rows": len(rows),
        "load_errors": load_errors,
        "candidate_filters": {
            "x_values": x_values,
            "x_range": str(args.x_range),
            "radii": _parse_float_list(args.radii),
            "centers": _parse_centers(args.centers),
            "max_nn_p95_m": float(args.max_nn_p95_m),
            "max_nearest_x_mean_abs_mm": float(args.max_nearest_x_mean_abs_mm),
            "min_showcase_radius_m": float(args.min_showcase_radius_m),
            "require_strict_geometry": bool(args.require_strict_geometry),
            "require_no_fixed_x_bias_for_best": bool(args.require_no_fixed_x_bias_for_best),
            "fixed_bias_threshold_mm": float(args.fixed_bias_threshold_mm),
        },
        "global_best": {k: _jsonable(v) for k, v in best.items()} if isinstance(best, dict) else {},
        "best_by_model": by_model,
    }
    summary_json = out_dir / "trajectory_benchmark_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    images: list[str] = []
    _plot_model_error_boxplot(summary, out_dir / "model_error_boxplot.png")
    images.append("model_error_boxplot.png")
    for split in SPLITS:
        img = f"{split}_region_error_boxplot.png"
        _plot_region_boxplot(summary, split, out_dir / img)
        if (out_dir / img).exists():
            images.append(img)
    best_model_id = str(best.get("model_id", summary.iloc[0]["model_id"])) if isinstance(best, dict) else str(summary.iloc[0]["model_id"])
    _plot_heatmap_best_model(summary, best_model_id, out_dir / "x_radius_heatmap_best_model.png")
    if (out_dir / "x_radius_heatmap_best_model.png").exists():
        images.append("x_radius_heatmap_best_model.png")
    _plot_center_scatter(summary, out_dir / "center_offset_scatter.png")
    images.append("center_offset_scatter.png")
    _plot_support_vs_error(summary, out_dir / "candidate_support_vs_error.png")
    images.append("candidate_support_vs_error.png")

    for label, img_prefix in (("global_best", "best_global"), ("supported_worst", "worst_supported")):
        row = reps.get(label)
        if not row:
            continue
        frame = point_frames.get((str(row["model_id"]), str(row["candidate_id"])))
        if frame is None:
            continue
        title = f"{label}: {row['model_id']} / {row['candidate_id']} / p95={float(row['ee_p95_mm']):.2f} mm"
        _plot_trajectory_yz(frame, title, out_dir / f"{img_prefix}_yz.png")
        _plot_trajectory_3d(frame, title, out_dir / f"{img_prefix}_3d.png")
        images.extend([f"{img_prefix}_yz.png", f"{img_prefix}_3d.png"])

    write_readme(out_dir, payload, images)
    return payload


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark open-loop circular trajectory tracking across direct inverse models.")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    ap.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--x-values", default="0.96,1.00,1.05,1.10,1.15,1.20")
    ap.add_argument("--x-range", default="", help="Optional dense x scan formatted as start,stop,step; overrides --x-values.")
    ap.add_argument("--centers", default="0,0;0.08,0;-0.08,0;0,0.08;0,-0.08;0.12,0.12;-0.12,0.12;0.12,-0.12;-0.12,-0.12")
    ap.add_argument("--radii", default="0.08,0.12,0.1512,0.16,0.20,0.26,0.32")
    ap.add_argument("--n-points", type=int, default=240)
    ap.add_argument("--models", default="", help="Comma-separated model ids or model names. Empty means all existing built-in direct models.")
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--max-nn-p95-m", type=float, default=0.03)
    ap.add_argument("--max-nearest-x-mean-abs-mm", type=float, default=-1.0, help="Strict geometry x-bias gate; negative disables it.")
    ap.add_argument("--fixed-bias-threshold-mm", type=float, default=2.0)
    ap.add_argument("--require-strict-geometry", action="store_true", help="Evaluate only candidates passing support, radius, and nearest-x mean gates.")
    ap.add_argument("--require-no-fixed-x-bias-for-best", action="store_true", help="Exclude model-candidate rows with fixed x bias from best-row selection.")
    ap.add_argument("--majority-threshold", type=float, default=0.60)
    ap.add_argument("--tension-upper-n", type=float, default=2000.0)
    ap.add_argument("--min-showcase-radius-m", type=float, default=0.10)
    return ap.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
