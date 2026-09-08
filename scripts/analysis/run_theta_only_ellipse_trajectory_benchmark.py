#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "scripts" / "baselines"
SRC_DIR = REPO_ROOT / "src"
for path in (str(BASELINE_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402


XYZ_COLS = ["x_m", "y_m", "z_m"]
BASE_AMP_XY_M = 0.200


@dataclass(frozen=True)
class EllipseCandidate:
    candidate_id: str
    center_x_m: float
    center_y_m: float
    center_z_m: float
    amp_xy_m: float
    amp_z_m: float
    n_points: int


@dataclass(frozen=True)
class WorkspaceContext:
    xyz_m: np.ndarray
    nn: NearestNeighbors


def _parse_float_list(text: str) -> list[float]:
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def _parse_float_range(text: str) -> list[float]:
    vals = _parse_float_list(text)
    if len(vals) != 3:
        raise ValueError("range must be formatted as start,stop,step")
    start, stop, step = vals
    if step <= 0.0:
        raise ValueError("range step must be positive")
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [round(start + i * step, 10) for i in range(max(0, n))]


def _parse_centers(text: str) -> list[tuple[float, float]]:
    out = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(v.strip()) for v in item.split(",")]
        if len(parts) != 2:
            raise ValueError("centers must be formatted as y,z;y,z")
        out.append((parts[0], parts[1]))
    return out


def _candidate_id(center_x: float, center_y: float, center_z: float, amp_xy: float, amp_z: float) -> str:
    raw = f"ell_cx{center_x:.3f}_cy{center_y:.3f}_cz{center_z:.3f}_a{amp_xy:.3f}_z{amp_z:.3f}"
    return raw.replace("-", "m").replace(".", "p")


def make_diagonal_ellipse_points(
    center_x: float,
    center_y: float,
    center_z: float,
    amp_xy: float,
    n_points: int,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Create center + k*(200 sin, 200 sin, 300 cos) mm in xyz space."""
    amp_xy = float(amp_xy)
    if amp_xy <= 0.0:
        raise ValueError("amp_xy must be positive")
    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2")
    angles = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False, dtype=float) + float(phase_rad)
    sin_v = np.sin(angles)
    points = np.zeros((int(n_points), 3), dtype=float)
    points[:, 0] = float(center_x) + amp_xy * sin_v
    points[:, 1] = float(center_y) + amp_xy * sin_v
    points[:, 2] = float(center_z) + 1.5 * amp_xy * np.cos(angles)
    return points


def build_ellipse_candidate_grid(
    center_x_values: list[float],
    centers_yz: list[tuple[float, float]],
    amp_xy_values: list[float],
    n_points: int,
) -> list[EllipseCandidate]:
    candidates = []
    seen: set[str] = set()
    for center_x in center_x_values:
        for center_y, center_z in centers_yz:
            for amp_xy in amp_xy_values:
                amp_z = 1.5 * float(amp_xy)
                cid = _candidate_id(float(center_x), float(center_y), float(center_z), float(amp_xy), amp_z)
                if cid in seen:
                    continue
                seen.add(cid)
                candidates.append(
                    EllipseCandidate(
                        candidate_id=cid,
                        center_x_m=float(center_x),
                        center_y_m=float(center_y),
                        center_z_m=float(center_z),
                        amp_xy_m=float(amp_xy),
                        amp_z_m=amp_z,
                        n_points=int(n_points),
                    )
                )
    return candidates


def has_fixed_axis_bias(values_mm: np.ndarray, threshold_mm: float = 2.0) -> bool:
    values = np.asarray(values_mm, dtype=float).reshape(-1)
    if values.size == 0:
        return False
    return bool(np.all(values > float(threshold_mm)) or np.all(values < -float(threshold_mm)))


def compute_axis_error_metrics(target_xyz_m: np.ndarray, achieved_xyz_m: np.ndarray, fixed_bias_threshold_mm: float) -> dict[str, float | bool]:
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    if target.shape != achieved.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target_xyz_m and achieved_xyz_m must both be shaped (N,3)")
    err_mm = (achieved - target) * 1000.0
    out: dict[str, float | bool] = {}
    p95_values = []
    fixed_values = []
    for axis_idx, axis in enumerate(("x", "y", "z")):
        values = err_mm[:, axis_idx]
        pos_ratio = float(np.mean(values > 0.0))
        neg_ratio = float(np.mean(values < 0.0))
        p95_abs = float(np.percentile(np.abs(values), 95))
        fixed = has_fixed_axis_bias(values, threshold_mm=float(fixed_bias_threshold_mm))
        out.update(
            {
                f"{axis}err_mean_mm": float(np.mean(values)),
                f"{axis}err_p50_mm": float(np.percentile(values, 50)),
                f"{axis}err_p95_abs_mm": p95_abs,
                f"{axis}err_min_mm": float(np.min(values)),
                f"{axis}err_max_mm": float(np.max(values)),
                f"{axis}err_same_sign_ratio": float(max(pos_ratio, neg_ratio)),
                f"fixed_{axis}_bias_gt2mm": fixed,
            }
        )
        p95_values.append(p95_abs)
        fixed_values.append(fixed)
    out["axiserr_max_p95_abs_mm"] = float(max(p95_values))
    out["fixed_any_axis_bias_gt2mm"] = bool(any(fixed_values))
    return out


def compute_tracking_metrics(target_xyz_m: np.ndarray, achieved_xyz_m: np.ndarray, fixed_bias_threshold_mm: float) -> dict[str, float | bool]:
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    err_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
    metrics: dict[str, float | bool] = {
        "ee_mean_mm": float(np.mean(err_mm)),
        "ee_rmse_mm": float(np.sqrt(np.mean(np.square(err_mm)))),
        "ee_p50_mm": float(np.percentile(err_mm, 50)),
        "ee_p95_mm": float(np.percentile(err_mm, 95)),
        "ee_max_mm": float(np.max(err_mm)),
    }
    metrics.update(compute_axis_error_metrics(target, achieved, fixed_bias_threshold_mm=float(fixed_bias_threshold_mm)))
    return metrics


def load_workspace_context(dataset: Path) -> WorkspaceContext:
    xyz = pd.read_parquet(dataset, columns=XYZ_COLS)[XYZ_COLS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(xyz)
    return WorkspaceContext(xyz_m=xyz, nn=nn)


def compute_support_metrics(nn_dist_m: np.ndarray, max_nn_p95_m: float) -> dict[str, float | bool]:
    dist = np.asarray(nn_dist_m, dtype=float).reshape(-1)
    p95_m = float(np.percentile(dist, 95))
    return {
        "supported": bool(p95_m <= float(max_nn_p95_m)),
        "nn_dist_mean_mm": float(np.mean(dist) * 1000.0),
        "nn_dist_p95_mm": float(p95_m * 1000.0),
        "nn_dist_max_mm": float(np.max(dist) * 1000.0),
    }


def candidate_workspace_metadata(
    target_xyz_m: np.ndarray,
    context: WorkspaceContext,
    max_nn_p95_m: float,
    max_nearest_x_mean_abs_mm: float,
    fixed_bias_threshold_mm: float,
) -> tuple[dict[str, float | bool], np.ndarray]:
    distances, indices = context.nn.kneighbors(np.asarray(target_xyz_m, dtype=float), n_neighbors=1)
    nearest_idx = indices[:, 0].astype(np.int64)
    nearest = context.xyz_m[nearest_idx]
    meta = compute_support_metrics(distances[:, 0], max_nn_p95_m=float(max_nn_p95_m))
    x_diff_mm = (nearest[:, 0] - np.asarray(target_xyz_m, dtype=float)[:, 0]) * 1000.0
    nearest_x_mean = float(np.mean(x_diff_mm))
    meta.update(
        {
            "nearest_x_diff_mean_mm": nearest_x_mean,
            "nearest_x_diff_p95_abs_mm": float(np.percentile(np.abs(x_diff_mm), 95)),
            "fixed_nearest_x_bias_gt2mm": has_fixed_axis_bias(x_diff_mm, threshold_mm=float(fixed_bias_threshold_mm)),
            "nearest_x_mean_abs_within_limit": bool(abs(nearest_x_mean) <= float(max_nearest_x_mean_abs_mm)),
        }
    )
    meta["strict_geometry_supported"] = bool(meta["supported"] and meta["nearest_x_mean_abs_within_limit"])
    return meta, nearest_idx


def filter_workspace_supported_candidates(
    candidates: list[EllipseCandidate],
    candidate_meta: dict[str, dict[str, float | bool]],
    max_nearest_x_mean_abs_mm: float,
    max_candidates_per_amp: int,
) -> list[EllipseCandidate]:
    grouped: dict[float, list[EllipseCandidate]] = {}
    for candidate in candidates:
        meta = candidate_meta.get(candidate.candidate_id, {})
        if not bool(meta.get("supported", False)):
            continue
        if abs(float(meta.get("nearest_x_diff_mean_mm", float("inf")))) > float(max_nearest_x_mean_abs_mm):
            continue
        grouped.setdefault(round(float(candidate.amp_xy_m), 6), []).append(candidate)
    selected: list[EllipseCandidate] = []
    for amp in sorted(grouped):
        group = sorted(
            grouped[amp],
            key=lambda c: (
                float(candidate_meta[c.candidate_id].get("nn_dist_p95_mm", float("inf"))),
                abs(float(candidate_meta[c.candidate_id].get("nearest_x_diff_mean_mm", float("inf")))),
                abs(float(c.center_y_m)) + abs(float(c.center_z_m)),
                float(c.center_x_m),
            ),
        )
        selected.extend(group[: max(1, int(max_candidates_per_amp))])
    return selected


def load_theta_model_package(path: Path) -> dict[str, Any]:
    package = load(path)
    if not isinstance(package, dict):
        raise ValueError("theta-only model package must be a dict")
    if package.get("kind") != "theta_only":
        raise ValueError("model package kind must be theta_only")
    for key in ("model", "input_cols", "theta_cols"):
        if key not in package:
            raise ValueError(f"model package missing {key}")
    return package


def predict_theta(package: dict[str, Any], xyz_m: np.ndarray) -> np.ndarray:
    model = package["model"]
    pred = np.asarray(model.predict(np.asarray(xyz_m, dtype=float)), dtype=float)
    y_scaler = package.get("y_scaler")
    if y_scaler is not None:
        pred = np.asarray(y_scaler.inverse_transform(pred), dtype=float)
    return pred


def evaluate_model_candidate(
    *,
    model_id: str,
    model_name: str,
    split: str,
    candidate: EllipseCandidate,
    target_xyz_m: np.ndarray,
    workspace_meta: dict[str, float | bool],
    package: dict[str, Any],
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    fixed_bias_threshold_mm: float,
) -> tuple[dict[str, float | str | bool], pd.DataFrame]:
    t0 = time.perf_counter()
    theta = predict_theta(package, target_xyz_m)
    pred_time_s = float(time.perf_counter() - t0)
    achieved = fk_dh_batch(theta, lengths_m=lengths_m, p_end_local_m=p_end_local_m)
    metrics = compute_tracking_metrics(target_xyz_m, achieved, fixed_bias_threshold_mm=float(fixed_bias_threshold_mm))
    row: dict[str, float | str | bool] = {
        "model_id": str(model_id),
        "model_name": str(model_name),
        "split": str(split),
        "candidate_id": candidate.candidate_id,
        "center_x_m": float(candidate.center_x_m),
        "center_y_m": float(candidate.center_y_m),
        "center_z_m": float(candidate.center_z_m),
        "amp_xy_m": float(candidate.amp_xy_m),
        "amp_z_m": float(candidate.amp_z_m),
        "scale_k": float(candidate.amp_xy_m / BASE_AMP_XY_M),
        "n_points": int(candidate.n_points),
        "pred_time_s": pred_time_s,
        "pred_time_ms_per_point": pred_time_s * 1000.0 / max(1, int(candidate.n_points)),
        **workspace_meta,
        **metrics,
    }
    point_data: dict[str, Any] = {
        "model_id": [str(model_id)] * int(candidate.n_points),
        "model_name": [str(model_name)] * int(candidate.n_points),
        "split": [str(split)] * int(candidate.n_points),
        "candidate_id": [candidate.candidate_id] * int(candidate.n_points),
        "angle_deg": np.linspace(0.0, 360.0, int(candidate.n_points), endpoint=False, dtype=float),
        "target_x_m": target_xyz_m[:, 0],
        "target_y_m": target_xyz_m[:, 1],
        "target_z_m": target_xyz_m[:, 2],
        "achieved_x_m": achieved[:, 0],
        "achieved_y_m": achieved[:, 1],
        "achieved_z_m": achieved[:, 2],
        "ee_error_mm": np.linalg.norm(achieved - target_xyz_m, axis=1) * 1000.0,
        "x_error_mm": (achieved[:, 0] - target_xyz_m[:, 0]) * 1000.0,
        "y_error_mm": (achieved[:, 1] - target_xyz_m[:, 1]) * 1000.0,
        "z_error_mm": (achieved[:, 2] - target_xyz_m[:, 2]) * 1000.0,
    }
    for i in range(theta.shape[1]):
        point_data[f"theta{i + 1}_rad"] = theta[:, i]
    return row, pd.DataFrame(point_data)


def _is_qualified(row: dict[str, Any], quality_ee_p95_mm: float, quality_axis_p95_mm: float) -> bool:
    return bool(
        bool(row.get("supported", False))
        and not bool(row.get("fixed_any_axis_bias_gt2mm", False))
        and float(row.get("ee_p95_mm", float("inf"))) <= float(quality_ee_p95_mm)
        and float(row.get("axiserr_max_p95_abs_mm", float("inf"))) <= float(quality_axis_p95_mm)
    )


def select_showcase_rows(rows: list[dict[str, Any]], quality_ee_p95_mm: float, quality_axis_p95_mm: float) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("rows must not be empty")
    qualified = [r for r in rows if _is_qualified(r, quality_ee_p95_mm, quality_axis_p95_mm)]
    if qualified:
        largest = max(
            qualified,
            key=lambda r: (
                float(r["amp_xy_m"]),
                -float(r["ee_p95_mm"]),
                -float(r["axiserr_max_p95_abs_mm"]),
                -float(r["ee_rmse_mm"]),
            ),
        )
        largest = {**largest, "selection_note": "largest_qualified"}
    else:
        largest = min(
            rows,
            key=lambda r: (
                float(r.get("ee_p95_mm", float("inf"))),
                float(r.get("axiserr_max_p95_abs_mm", float("inf"))),
                -float(r.get("amp_xy_m", 0.0)),
            ),
        )
        largest = {**largest, "selection_note": "fallback_no_qualified"}
    lowest = min(
        [r for r in rows if bool(r.get("supported", False))] or rows,
        key=lambda r: (
            float(r.get("ee_p95_mm", float("inf"))),
            float(r.get("axiserr_max_p95_abs_mm", float("inf"))),
            float(r.get("ee_rmse_mm", float("inf"))),
            -float(r.get("amp_xy_m", 0.0)),
        ),
    )
    return {"largest_qualified": largest, "lowest_error": {**lowest, "selection_note": "lowest_error"}}


def _scoreboard(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    columns = [
        "selection_note",
        "model_id",
        "model_name",
        "split",
        "candidate_id",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "amp_xy_m",
        "amp_z_m",
        "scale_k",
        "nn_dist_p95_mm",
        "nearest_x_diff_mean_mm",
        "ee_mean_mm",
        "ee_rmse_mm",
        "ee_p95_mm",
        "ee_max_mm",
        "axiserr_max_p95_abs_mm",
        "xerr_p95_abs_mm",
        "yerr_p95_abs_mm",
        "zerr_p95_abs_mm",
        "fixed_any_axis_bias_gt2mm",
    ]
    return df[[c for c in columns if c in df.columns]].sort_values(["selection_note", "ee_p95_mm"]).reset_index(drop=True)


def _points_for_rows(all_points: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in rows.iterrows():
        sub = all_points[
            (all_points["model_id"].astype(str) == str(row["model_id"]))
            & (all_points["candidate_id"].astype(str) == str(row["candidate_id"]))
        ]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_3d(points: pd.DataFrame, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8.0, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points["target_x_m"], points["target_y_m"], points["target_z_m"], color="#1f4e79", lw=2.0, label="target")
    ax.plot(points["achieved_x_m"], points["achieved_y_m"], points["achieved_z_m"], color="#c7511f", lw=1.8, label="model + FK")
    ax.scatter(points["target_x_m"].iloc[0], points["target_y_m"].iloc[0], points["target_z_m"].iloc[0], s=28, color="#1f4e79")
    ax.scatter(points["achieved_x_m"].iloc[0], points["achieved_y_m"].iloc[0], points["achieved_z_m"].iloc[0], s=28, color="#c7511f", marker="x")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=24, azim=-58)
    ax.legend(frameon=False)
    ax.set_title(title)
    _savefig(fig, out_path)


def _plot_projection(points: pd.DataFrame, title: str, out_path: Path, left: str, right: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(points[f"target_{left}_m"], points[f"target_{right}_m"], color="#1f4e79", lw=2.0, label="target")
    ax.plot(points[f"achieved_{left}_m"], points[f"achieved_{right}_m"], color="#c7511f", lw=1.8, label="model + FK")
    ax.set_xlabel(f"{left} (m)")
    ax.set_ylabel(f"{right} (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(title)
    _savefig(fig, out_path)


def _plot_axis_errors(points: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for col, label in (("x_error_mm", "x"), ("y_error_mm", "y"), ("z_error_mm", "z")):
        ax.plot(points["angle_deg"], points[col], lw=1.3, label=label)
    ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.5)
    ax.axhspan(-2.0, 2.0, color="#54a24b", alpha=0.08, linewidth=0)
    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel("ellipse angle (deg)")
    ax.set_ylabel("axis error (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_title(title)
    _savefig(fig, out_path)


def _plot_all_model_3d_grid(points: pd.DataFrame, scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if points.empty or scoreboard.empty:
        return
    n = len(scoreboard)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(4.6 * ncols, 3.8 * nrows))
    for idx, row in scoreboard.reset_index(drop=True).iterrows():
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        sub = points[
            (points["model_id"].astype(str) == str(row["model_id"]))
            & (points["candidate_id"].astype(str) == str(row["candidate_id"]))
        ]
        if sub.empty:
            continue
        ax.plot(sub["target_x_m"], sub["target_y_m"], sub["target_z_m"], color="#1f4e79", lw=1.4)
        ax.plot(sub["achieved_x_m"], sub["achieved_y_m"], sub["achieved_z_m"], color="#c7511f", lw=1.3)
        ax.view_init(elev=24, azim=-58)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_xlabel("x", fontsize=7, labelpad=-2)
        ax.set_ylabel("y", fontsize=7, labelpad=-2)
        ax.set_zlabel("z", fontsize=7, labelpad=-2)
        ax.set_title(
            f"{idx + 1}. {row['model_id']}\nEE95={float(row['ee_p95_mm']):.2f} axis95={float(row['axiserr_max_p95_abs_mm']):.2f} a={float(row['amp_xy_m']):.3f}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=14)
    _savefig(fig, out_path)


def _plot_metric_bars(scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if scoreboard.empty:
        return
    plot_df = scoreboard.sort_values("ee_p95_mm", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(12.0, max(4.8, 0.4 * len(plot_df))))
    ax.barh(y - 0.13, plot_df["ee_p95_mm"], height=0.24, color="#4c78a8", label="EE p95")
    ax.barh(y + 0.13, plot_df["axiserr_max_p95_abs_mm"], height=0.24, color="#f58518", label="axis p95 max")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model_id"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("error (mm)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(title)
    _savefig(fig, out_path)


def _markdown_table(df: pd.DataFrame, title: str, max_rows: int = 20) -> str:
    if df.empty:
        return f"## {title}\n\nNo rows.\n"
    cols = [c for c in ["selection_note", "model_id", "candidate_id", "amp_xy_m", "amp_z_m", "ee_p95_mm", "axiserr_max_p95_abs_mm", "nn_dist_p95_mm", "nearest_x_diff_mean_mm"] if c in df.columns]
    rows = df[cols].head(max_rows)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in rows.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    out = [f"## {title}", "", header, sep, *body, ""]
    return "\n".join(out)


def write_readme(out_dir: Path, payload: dict[str, Any], images: list[str], model_best: pd.DataFrame, global_rows: pd.DataFrame) -> None:
    lines = [
        "# Theta-only ellipse trajectory benchmark",
        "",
        "This run evaluates `xyz -> theta -> FK -> xyz` tracking on diagonal ellipses with the fixed 200:200:300 ratio.",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Models evaluated: `{payload['models_evaluated']}`",
        f"- Candidates generated/evaluated: `{payload['candidates_generated']}` / `{payload['candidates_evaluated']}`",
        "",
        _markdown_table(model_best, "Per-model showcase rows").rstrip(),
        "",
        _markdown_table(global_rows, "Global showcase rows").rstrip(),
        "",
        "## Images",
        "",
    ]
    for image in images:
        lines.append(f"- `{image}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def discover_model_paths(model_root: Path, models: list[str], splits: list[str]) -> list[tuple[str, str, str, Path]]:
    out = []
    for split in splits:
        for model_name in models:
            path = model_root / split / model_name / "model.joblib"
            if path.exists():
                out.append((f"{split}_{model_name}", model_name, split, path))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    center_x_values = _parse_float_range(args.center_x_range)
    centers_yz = _parse_centers(args.centers_yz)
    amp_xy_values = _parse_float_list(args.amp_xy_values)
    candidates = build_ellipse_candidate_grid(center_x_values, centers_yz, amp_xy_values, int(args.n_points))
    candidates_generated = len(candidates)

    context = load_workspace_context(Path(args.dataset))
    candidate_targets: dict[str, np.ndarray] = {}
    candidate_meta: dict[str, dict[str, float | bool]] = {}
    for candidate in candidates:
        target = make_diagonal_ellipse_points(
            candidate.center_x_m,
            candidate.center_y_m,
            candidate.center_z_m,
            candidate.amp_xy_m,
            candidate.n_points,
        )
        candidate_targets[candidate.candidate_id] = target
        meta, _ = candidate_workspace_metadata(
            target,
            context,
            max_nn_p95_m=float(args.max_nn_p95_m),
            max_nearest_x_mean_abs_mm=float(args.max_nearest_x_mean_abs_mm),
            fixed_bias_threshold_mm=float(args.fixed_bias_threshold_mm),
        )
        candidate_meta[candidate.candidate_id] = meta

    candidates = filter_workspace_supported_candidates(
        candidates,
        candidate_meta,
        max_nearest_x_mean_abs_mm=float(args.max_nearest_x_mean_abs_mm),
        max_candidates_per_amp=int(args.max_candidates_per_amp),
    )
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    if not candidates:
        raise SystemExit("no candidates remain after workspace support filtering")

    cfg = load_config(Path(args.robot_config))
    inputs = load_robot_inputs(cfg)
    model_specs = discover_model_paths(
        Path(args.model_root),
        [m.strip() for m in str(args.models).split(",") if m.strip()],
        [s.strip() for s in str(args.splits).split(",") if s.strip()],
    )
    if not model_specs:
        raise SystemExit("no theta-only model.joblib files found")

    rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    load_errors: list[dict[str, str]] = []
    for model_id, model_name, split, model_path in model_specs:
        try:
            package = load_theta_model_package(model_path)
        except Exception as exc:
            load_errors.append({"model_id": model_id, "error": repr(exc)})
            continue
        for candidate in candidates:
            row, points = evaluate_model_candidate(
                model_id=model_id,
                model_name=model_name,
                split=split,
                candidate=candidate,
                target_xyz_m=candidate_targets[candidate.candidate_id],
                workspace_meta=candidate_meta[candidate.candidate_id],
                package=package,
                lengths_m=inputs.lengths_m,
                p_end_local_m=inputs.p_end_local_m,
                fixed_bias_threshold_mm=float(args.fixed_bias_threshold_mm),
            )
            rows.append(row)
            point_frames.append(points)
    if not rows:
        raise SystemExit(f"no model-candidate rows evaluated; load_errors={load_errors}")

    summary = pd.DataFrame(rows).sort_values(["ee_p95_mm", "axiserr_max_p95_abs_mm", "amp_xy_m"], ascending=[True, True, False])
    summary.to_csv(out_dir / "trajectory_benchmark_summary.csv", index=False)
    all_points = pd.concat(point_frames, ignore_index=True)

    showcase_by_model: list[dict[str, Any]] = []
    for model_id in sorted(summary["model_id"].astype(str).unique()):
        model_rows = [r for r in rows if str(r["model_id"]) == model_id]
        selected = select_showcase_rows(model_rows, float(args.quality_ee_p95_mm), float(args.quality_axis_p95_mm))
        showcase_by_model.extend(selected.values())
    model_best = _scoreboard(showcase_by_model)
    model_best.to_csv(out_dir / "model_showcase_scoreboard.csv", index=False)
    model_points = _points_for_rows(all_points, model_best)
    if not model_points.empty:
        model_points.to_parquet(out_dir / "model_showcase_points.parquet", index=False)

    global_selected = select_showcase_rows(rows, float(args.quality_ee_p95_mm), float(args.quality_axis_p95_mm))
    global_rows = _scoreboard(list(global_selected.values()))
    global_rows.to_csv(out_dir / "global_showcase_scoreboard.csv", index=False)
    global_points = _points_for_rows(all_points, global_rows)
    if not global_points.empty:
        global_points.to_parquet(out_dir / "global_showcase_points.parquet", index=False)

    images: list[str] = []
    for label, row in global_rows.iterrows():
        prefix = str(row["selection_note"])
        sub = all_points[
            (all_points["model_id"].astype(str) == str(row["model_id"]))
            & (all_points["candidate_id"].astype(str) == str(row["candidate_id"]))
        ]
        title = f"{prefix}: {row['model_id']} / {row['candidate_id']} / EE95={float(row['ee_p95_mm']):.2f} mm"
        if sub.empty:
            continue
        img = f"best_{prefix}_3d.png"
        _plot_trajectory_3d(sub, title, out_dir / img)
        images.append(img)
        for left, right in (("x", "y"), ("x", "z"), ("y", "z")):
            img = f"best_{prefix}_{left}{right}.png"
            _plot_projection(sub, title, out_dir / img, left, right)
            images.append(img)
        img = f"best_{prefix}_axis_errors.png"
        _plot_axis_errors(sub, title, out_dir / img)
        images.append(img)

    largest_rows = model_best[model_best["selection_note"].astype(str).isin(["largest_qualified", "fallback_no_qualified"])]
    lowest_rows = model_best[model_best["selection_note"].astype(str) == "lowest_error"]
    _plot_all_model_3d_grid(model_points, largest_rows, "All models: largest qualified ellipse", out_dir / "all_models_largest_qualified_3d_grid.png")
    images.append("all_models_largest_qualified_3d_grid.png")
    _plot_all_model_3d_grid(model_points, lowest_rows, "All models: lowest error ellipse", out_dir / "all_models_lowest_error_3d_grid.png")
    images.append("all_models_lowest_error_3d_grid.png")
    _plot_metric_bars(model_best, "Theta-only ellipse benchmark", out_dir / "model_metric_bars.png")
    images.append("model_metric_bars.png")

    payload: dict[str, Any] = {
        "mode": "theta_only_ellipse_trajectory_benchmark",
        "generated_at": time.strftime("%F %T"),
        "dataset": str(args.dataset),
        "model_root": str(args.model_root),
        "models_evaluated": int(len({r["model_id"] for r in rows})),
        "candidates_generated": int(candidates_generated),
        "candidates_evaluated": int(len(candidates)),
        "rows": int(len(rows)),
        "load_errors": load_errors,
        "candidate_filters": {
            "center_x_range": str(args.center_x_range),
            "centers_yz": centers_yz,
            "amp_xy_values": amp_xy_values,
            "max_nn_p95_m": float(args.max_nn_p95_m),
            "max_nearest_x_mean_abs_mm": float(args.max_nearest_x_mean_abs_mm),
            "max_candidates_per_amp": int(args.max_candidates_per_amp),
            "quality_ee_p95_mm": float(args.quality_ee_p95_mm),
            "quality_axis_p95_mm": float(args.quality_axis_p95_mm),
        },
        "global_showcase": [_jsonable(row) for row in global_rows.to_dict(orient="records")],
    }
    (out_dir / "trajectory_benchmark_summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir, payload, images, model_best, global_rows)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark theta-only xyz->theta models on fixed-ratio diagonal ellipses.")
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "hierarchical_beta_fk_x1p0_1p2_v1" / "balanced_100k.parquet")
    ap.add_argument("--robot-config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="mlp_large,lgbm")
    ap.add_argument("--splits", default="iid,x_slab,radius,beta_block,angular_sector")
    ap.add_argument("--center-x-range", default="1.04,1.16,0.005")
    ap.add_argument("--centers-yz", default="-0.12,-0.12;-0.12,-0.08;-0.12,-0.04;-0.12,0;-0.12,0.04;-0.12,0.08;-0.12,0.12;-0.08,-0.12;-0.08,-0.08;-0.08,-0.04;-0.08,0;-0.08,0.04;-0.08,0.08;-0.08,0.12;-0.04,-0.12;-0.04,-0.08;-0.04,-0.04;-0.04,0;-0.04,0.04;-0.04,0.08;-0.04,0.12;0,-0.12;0,-0.08;0,-0.04;0,0;0,0.04;0,0.08;0,0.12;0.04,-0.12;0.04,-0.08;0.04,-0.04;0.04,0;0.04,0.04;0.04,0.08;0.04,0.12;0.08,-0.12;0.08,-0.08;0.08,-0.04;0.08,0;0.08,0.04;0.08,0.08;0.08,0.12;0.12,-0.12;0.12,-0.08;0.12,-0.04;0.12,0;0.12,0.04;0.12,0.08;0.12,0.12")
    ap.add_argument("--amp-xy-values", default="0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.10")
    ap.add_argument("--n-points", type=int, default=360)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--max-candidates-per-amp", type=int, default=12)
    ap.add_argument("--max-nn-p95-m", type=float, default=0.008)
    ap.add_argument("--max-nearest-x-mean-abs-mm", type=float, default=2.0)
    ap.add_argument("--fixed-bias-threshold-mm", type=float, default=2.0)
    ap.add_argument("--quality-ee-p95-mm", type=float, default=20.0)
    ap.add_argument("--quality-axis-p95-mm", type=float, default=10.0)
    return ap.parse_args()


def main() -> int:
    print(json.dumps(_jsonable(run(parse_args())), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
