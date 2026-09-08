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
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_circle_trajectory_benchmark import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_SPLIT_ROOT,
    REPO_ROOT as CIRCLE_REPO_ROOT,
    SPLITS,
    WorkspaceContext,
    _jsonable,
    _load_model_and_scaler,
    _parse_centers,
    _parse_float_list,
    _parse_float_range,
    build_builtin_model_registry,
    candidate_workspace_metadata,
    filter_existing_models,
    fk_dh_batch,
    has_fixed_axis_bias,
    load_workspace_context,
    predict_theta_tension,
    _load_robot_paths,
    _read_lengths_end,
)


DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "visualizations" / "ellipse_trajectory_benchmark_strict_support_relabel_100k_v1"
BASE_AMP_XY_M = 0.200
BASE_AMP_Z_M = 0.300


class EllipseCandidate(NamedTuple):
    candidate_id: str
    center_x_m: float
    center_y_m: float
    center_z_m: float
    amp_xy_m: float
    amp_z_m: float
    n_points: int


def _suffix_from_threshold(threshold_mm: float) -> str:
    value = float(threshold_mm)
    if value.is_integer():
        return str(int(value)).replace("-", "m")
    return str(value).replace("-", "m").replace(".", "p")


def _ellipse_candidate_id(
    center_x: float,
    center_y: float,
    center_z: float,
    amp_xy: float,
    amp_z: float,
    x_decimals: int = 3,
) -> str:
    raw = (
        f"ell_cx{center_x:.{int(x_decimals)}f}_cy{center_y:.3f}_cz{center_z:.3f}"
        f"_a{amp_xy:.3f}_z{amp_z:.3f}"
    )
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
    if int(n_points) < 8:
        raise ValueError("n_points must be at least 8")
    amp_z = 1.5 * amp_xy
    angles = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False, dtype=float) + float(phase_rad)
    sin_v = np.sin(angles)
    points = np.zeros((int(n_points), 3), dtype=float)
    points[:, 0] = float(center_x) + amp_xy * sin_v
    points[:, 1] = float(center_y) + amp_xy * sin_v
    points[:, 2] = float(center_z) + amp_z * np.cos(angles)
    return points


def build_ellipse_candidate_grid(
    center_x_values: list[float],
    centers_yz: list[tuple[float, float]],
    amp_xy_values: list[float],
    n_points: int,
    x_id_decimals: int = 3,
) -> list[EllipseCandidate]:
    candidates: list[EllipseCandidate] = []
    seen: set[str] = set()
    for center_x in center_x_values:
        for center_y, center_z in centers_yz:
            for amp_xy in amp_xy_values:
                amp_z = 1.5 * float(amp_xy)
                cid = _ellipse_candidate_id(
                    float(center_x),
                    float(center_y),
                    float(center_z),
                    float(amp_xy),
                    amp_z,
                    x_decimals=int(x_id_decimals),
                )
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


def compute_axis_error_metrics(
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    fixed_bias_threshold_mm: float,
) -> dict[str, float | bool]:
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    if target.shape != achieved.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target_xyz_m and achieved_xyz_m must both be shaped (N, 3)")
    err_mm = (achieved - target) * 1000.0
    suffix = _suffix_from_threshold(fixed_bias_threshold_mm)
    out: dict[str, float | bool] = {}
    p95_values: list[float] = []
    fixed_values: list[bool] = []
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
                f"fixed_{axis}_bias_gt{suffix}mm": fixed,
            }
        )
        p95_values.append(p95_abs)
        fixed_values.append(fixed)
    out["axiserr_max_p95_abs_mm"] = float(max(p95_values))
    out[f"fixed_any_axis_bias_gt{suffix}mm"] = bool(any(fixed_values))
    out["fixed_any_axis_bias_gt2mm"] = bool(any(fixed_values)) if suffix == "2" else bool(out[f"fixed_any_axis_bias_gt{suffix}mm"])
    return out


def compute_tracking_metrics(
    target_xyz_m: np.ndarray,
    achieved_xyz_m: np.ndarray,
    tension_n: np.ndarray,
    tension_upper_n: float,
    fixed_bias_threshold_mm: float = 2.0,
) -> dict[str, float | bool]:
    target = np.asarray(target_xyz_m, dtype=float)
    achieved = np.asarray(achieved_xyz_m, dtype=float)
    tension = np.asarray(tension_n, dtype=float)
    err_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
    tension_f = tension.reshape(-1)
    metrics: dict[str, float | bool] = {
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
    metrics.update(compute_axis_error_metrics(target, achieved, fixed_bias_threshold_mm=fixed_bias_threshold_mm))
    return metrics


def select_geometry_candidates_by_amp_band(
    candidates: list[EllipseCandidate],
    candidate_meta: dict[str, dict[str, float | bool | str]],
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


def evaluate_model_candidate(
    spec,
    candidate: EllipseCandidate,
    target_xyz_m: np.ndarray,
    workspace_meta: dict[str, float | bool | str],
    model,
    scaler,
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
        "center_x_m": candidate.center_x_m,
        "center_y_m": candidate.center_y_m,
        "center_z_m": candidate.center_z_m,
        "amp_xy_m": candidate.amp_xy_m,
        "amp_z_m": candidate.amp_z_m,
        "scale_k": candidate.amp_xy_m / BASE_AMP_XY_M,
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
        "y_error_mm": (achieved[:, 1] - target_xyz_m[:, 1]) * 1000.0,
        "z_error_mm": (achieved[:, 2] - target_xyz_m[:, 2]) * 1000.0,
    }
    for i in range(theta.shape[1]):
        point_data[f"theta{i + 1}_rad"] = theta[:, i]
    for i in range(tension.shape[1]):
        point_data[f"T{i + 1}_N"] = tension[:, i]
    return row, pd.DataFrame(point_data)


def select_showcase_row(
    rows: list[dict[str, float | str | bool]],
    quality_ee_p95_mm: float,
    quality_axis_p95_mm: float,
    max_negative_tension_ratio: float = 0.0,
    max_over_upper_tension_ratio: float = 0.0,
) -> dict[str, float | str | bool]:
    quality = []
    for row in rows:
        if not bool(row.get("supported", False)):
            continue
        if float(row.get("negative_tension_ratio", 0.0)) > float(max_negative_tension_ratio):
            continue
        if float(row.get("over_upper_tension_ratio", 0.0)) > float(max_over_upper_tension_ratio):
            continue
        if bool(row.get("fixed_any_axis_bias_gt2mm", False)):
            continue
        if float(row["ee_p95_mm"]) > float(quality_ee_p95_mm):
            continue
        if float(row["axiserr_max_p95_abs_mm"]) > float(quality_axis_p95_mm):
            continue
        quality.append(row)
    if quality:
        best = max(
            quality,
            key=lambda r: (
                float(r["amp_xy_m"]),
                -float(r["ee_p95_mm"]),
                -float(r["axiserr_max_p95_abs_mm"]),
                -float(r["ee_rmse_mm"]),
            ),
        )
        return {**best, "selection_note": "display_largest_under_quality"}
    fallback = min(
        rows,
        key=lambda r: (
            float(r["ee_p95_mm"]),
            float(r.get("axiserr_max_p95_abs_mm", float("inf"))),
            float(r["ee_rmse_mm"]),
            -float(r["amp_xy_m"]),
        ),
    )
    return {**fallback, "selection_note": "fallback_best_error_no_quality_row"}


def select_best_by_model(
    rows: list[dict[str, float | str | bool]],
    quality_ee_p95_mm: float,
    quality_axis_p95_mm: float,
) -> dict[str, dict[str, float | str | bool]]:
    out: dict[str, dict[str, float | str | bool]] = {}
    for model_id in sorted({str(r["model_id"]) for r in rows}):
        model_rows = [r for r in rows if str(r["model_id"]) == model_id]
        out[model_id] = select_showcase_row(
            model_rows,
            quality_ee_p95_mm=quality_ee_p95_mm,
            quality_axis_p95_mm=quality_axis_p95_mm,
        )
    return out


def _plot_trajectory_3d(points: pd.DataFrame, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8.0, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points["target_x_m"], points["target_y_m"], points["target_z_m"], color="#1f4e79", lw=2.0, label="target")
    ax.plot(points["achieved_x_m"], points["achieved_y_m"], points["achieved_z_m"], color="#c7511f", lw=1.8, label="model + FK")
    ax.scatter(points["target_x_m"].iloc[0], points["target_y_m"].iloc[0], points["target_z_m"].iloc[0], s=32, color="#1f4e79")
    ax.scatter(points["achieved_x_m"].iloc[0], points["achieved_y_m"].iloc[0], points["achieved_z_m"].iloc[0], s=32, color="#c7511f", marker="x")
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
    for col, label, color in (
        ("x_error_mm", "x error", "#4c78a8"),
        ("y_error_mm", "y error", "#f58518"),
        ("z_error_mm", "z error", "#54a24b"),
    ):
        ax.plot(points["angle_deg"], points[col], lw=1.5, label=label, color=color)
    ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.5)
    ax.axhspan(-2.0, 2.0, color="#54a24b", alpha=0.08, linewidth=0)
    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel("theta (deg)")
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
        sub = points[(points["model_id"].astype(str) == str(row["model_id"])) & (points["candidate_id"].astype(str) == str(row["candidate_id"]))]
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


def _plot_all_model_axis_errors(points: pd.DataFrame, scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if points.empty or scoreboard.empty:
        return
    n = len(scoreboard)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows), squeeze=False)
    max_abs = max(float(np.nanmax(np.abs(points[["x_error_mm", "y_error_mm", "z_error_mm"]].to_numpy(dtype=float)))), 2.5)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for idx, row in scoreboard.reset_index(drop=True).iterrows():
        ax = axes.reshape(-1)[idx]
        sub = points[(points["model_id"].astype(str) == str(row["model_id"])) & (points["candidate_id"].astype(str) == str(row["candidate_id"]))]
        if sub.empty:
            continue
        ax.axis("on")
        ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.6)
        ax.axhspan(-2.0, 2.0, color="#54a24b", alpha=0.08, linewidth=0)
        ax.plot(sub["angle_deg"], sub["x_error_mm"], lw=1.0, label="x")
        ax.plot(sub["angle_deg"], sub["y_error_mm"], lw=1.0, label="y")
        ax.plot(sub["angle_deg"], sub["z_error_mm"], lw=1.0, label="z")
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-max_abs * 1.08, max_abs * 1.08)
        ax.set_xticks([0, 180, 360])
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{idx + 1}. {row['model_id']}\naxis95={float(row['axiserr_max_p95_abs_mm']):.2f} mm", fontsize=8)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.suptitle(title, fontsize=14)
    _savefig(fig, out_path)


def _plot_metric_bars(scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if scoreboard.empty:
        return
    plot_df = scoreboard.sort_values("ee_p95_mm", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(13.0, max(6.0, 0.36 * len(plot_df))))
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


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scoreboard(rows: list[dict[str, float | str | bool]]) -> pd.DataFrame:
    columns = [
        "model_id",
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
        "pred_time_ms_per_point",
        "tension_min_n",
        "tension_max_n",
        "negative_tension_ratio",
        "over_upper_tension_ratio",
        "supported",
        "selection_note",
    ]
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df:
            df[col] = np.nan
    return df[columns].sort_values(["ee_p95_mm", "axiserr_max_p95_abs_mm", "model_id"]).reset_index(drop=True)


def _points_for_rows(points: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    if points.empty or rows.empty:
        return pd.DataFrame()
    keys = rows[["model_id", "candidate_id"]].drop_duplicates().astype(str)
    merged = points.copy()
    merged["model_id"] = merged["model_id"].astype(str)
    merged["candidate_id"] = merged["candidate_id"].astype(str)
    return merged.merge(keys, on=["model_id", "candidate_id"], how="inner").sort_values(["model_id", "candidate_id", "angle_deg"]).reset_index(drop=True)


def _markdown_table(df: pd.DataFrame, title: str, top_n: int = 25) -> str:
    lines = [f"## {title}", ""]
    if df.empty:
        return "\n".join(lines + ["No rows."])
    cols = [
        "model_id",
        "candidate_id",
        "amp_xy_m",
        "amp_z_m",
        "scale_k",
        "nn_dist_p95_mm",
        "nearest_x_diff_mean_mm",
        "ee_p95_mm",
        "axiserr_max_p95_abs_mm",
        "fixed_any_axis_bias_gt2mm",
        "selection_note",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.head(top_n).iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if col in {"amp_xy_m", "amp_z_m", "scale_k"}:
                    values.append(f"{value:.4f}")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(
    out_dir: Path,
    payload: dict[str, object],
    images: list[str],
    model_best: pd.DataFrame,
    same_candidate: pd.DataFrame,
) -> None:
    best = payload.get("global_best", {})
    if not isinstance(best, dict):
        best = {}
    diagnostics = payload.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    lines = [
        "# Ellipse Trajectory Benchmark",
        "",
        "This benchmark evaluates open-loop tracking for `center + k*(200 sin(theta), 200 sin(theta), 300 cos(theta)) mm`.",
        "",
        "## Global Showcase",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- model: `{best.get('model_id')}`",
                f"- candidate: `{best.get('candidate_id')}`",
                f"- center xyz: `{float(best.get('center_x_m', float('nan'))):.4f}, {float(best.get('center_y_m', float('nan'))):.4f}, {float(best.get('center_z_m', float('nan'))):.4f} m`",
                f"- amplitudes xyz: `{float(best.get('amp_xy_m', float('nan'))):.4f}, {float(best.get('amp_xy_m', float('nan'))):.4f}, {float(best.get('amp_z_m', float('nan'))):.4f} m`",
                f"- scale k: `{float(best.get('scale_k', float('nan'))):.3f}`",
                f"- EE p95/max: `{float(best.get('ee_p95_mm', float('nan'))):.3f} / {float(best.get('ee_max_mm', float('nan'))):.3f} mm`",
                f"- axiserr max p95: `{float(best.get('axiserr_max_p95_abs_mm', float('nan'))):.3f} mm`",
                f"- nn_dist_p95: `{float(best.get('nn_dist_p95_mm', float('nan'))):.3f} mm`",
                f"- nearest_x_diff_mean: `{float(best.get('nearest_x_diff_mean_mm', float('nan'))):.3f} mm`",
                f"- fixed_any_axis_bias_gt2mm: `{best.get('fixed_any_axis_bias_gt2mm')}`",
                f"- selection note: `{best.get('selection_note')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Search Diagnostics",
            "",
            f"- candidates generated/evaluated: `{payload.get('candidates_generated')} / {payload.get('candidates_evaluated')}`",
            f"- evaluated amplitudes xy: `{diagnostics.get('evaluated_amp_xy_values_m')}`",
            f"- quality rows with EE<=5mm and axis<=3mm: `{diagnostics.get('quality_rows_axis3')}`",
            f"- quality rows with EE<=5mm and axis<=5mm: `{diagnostics.get('quality_rows_axis5')}`",
            "",
        ]
    )
    lines.extend(
        [
            "## Files",
            "",
        ]
    )
    for image in images:
        lines.append(f"- `{image}`")
    lines.extend(
        [
            "- `trajectory_benchmark_summary.csv`: all model-candidate metrics.",
            "- `trajectory_benchmark_summary.json`: rollup and selected showcase rows.",
            "- `model_best_numeric_scoreboard.csv`: per-model selected ellipse metrics.",
            "- `same_global_candidate_numeric_scoreboard.csv`: every model on the global showcase ellipse.",
            "- `model_best_points.parquet`: point-level data for per-model selected ellipses.",
            "- `same_global_candidate_points.parquet`: point-level data for same-target model comparison.",
            "",
            "The unscaled 200/200/300 mm ellipse is not assumed reachable; this run searches a common scale `k` while preserving that ratio.",
            "",
            _markdown_table(model_best, "Best display ellipse by model").rstrip(),
            "",
            _markdown_table(same_candidate, "Same global showcase ellipse by model").rstrip(),
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = load_workspace_context(Path(args.dataset), Path(args.split_root))
    lengths_csv, ee_csv = _load_robot_paths(args.robot_config, None, None)
    lengths_m, p_end_local_m = _read_lengths_end(lengths_csv, ee_csv)

    candidates = build_ellipse_candidate_grid(
        center_x_values=_parse_float_range(args.center_x_range),
        centers_yz=_parse_centers(args.centers_yz),
        amp_xy_values=_parse_float_list(args.amp_xy_values),
        n_points=int(args.n_points),
        x_id_decimals=3,
    )
    if not candidates:
        raise SystemExit("no ellipse candidates generated")

    candidate_targets: dict[str, np.ndarray] = {}
    candidate_meta: dict[str, dict[str, float | bool | str]] = {}
    for candidate in candidates:
        target = make_diagonal_ellipse_points(
            candidate.center_x_m,
            candidate.center_y_m,
            candidate.center_z_m,
            candidate.amp_xy_m,
            candidate.n_points,
        )
        meta, _ = candidate_workspace_metadata(
            target,
            context,
            max_nn_p95_m=float(args.max_nn_p95_m),
            majority_threshold=float(args.majority_threshold),
            max_nearest_x_mean_abs_mm=float(args.max_nearest_x_mean_abs_mm),
            fixed_bias_threshold_mm=float(args.fixed_bias_threshold_mm),
        )
        candidate_targets[candidate.candidate_id] = target
        candidate_meta[candidate.candidate_id] = meta

    candidates_generated = len(candidates)
    candidates = select_geometry_candidates_by_amp_band(
        candidates,
        candidate_meta,
        max_nearest_x_mean_abs_mm=float(args.max_nearest_x_mean_abs_mm),
        max_candidates_per_amp=int(args.max_candidates_per_amp),
    )
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    if not candidates:
        raise SystemExit("no candidates remain after workspace geometry filtering")

    requested_models = {m.strip() for m in str(args.models).split(",") if m.strip()} if str(args.models).strip() else None
    specs = filter_existing_models(build_builtin_model_registry(CIRCLE_REPO_ROOT), requested=requested_models)
    if int(args.max_models) > 0:
        specs = specs[: int(args.max_models)]
    if not specs:
        raise SystemExit("no existing models selected")

    rows: list[dict[str, float | str | bool]] = []
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
            point_frames[(str(spec.model_id), str(candidate.candidate_id))] = points
    if not rows:
        raise SystemExit(f"no model-candidate rows evaluated; load_errors={load_errors}")

    summary = pd.DataFrame(rows).sort_values(
        ["ee_p95_mm", "axiserr_max_p95_abs_mm", "amp_xy_m"],
        ascending=[True, True, False],
    )
    summary.to_csv(out_dir / "trajectory_benchmark_summary.csv", index=False)
    quality_common = (
        summary["supported"].astype(bool)
        & (summary["negative_tension_ratio"].astype(float) <= 0.0)
        & (summary["over_upper_tension_ratio"].astype(float) <= 0.0)
        & (~summary["fixed_any_axis_bias_gt2mm"].astype(bool))
        & (summary["ee_p95_mm"].astype(float) <= 5.0)
    )

    global_best = select_showcase_row(
        rows,
        quality_ee_p95_mm=float(args.quality_ee_p95_mm),
        quality_axis_p95_mm=float(args.quality_axis_p95_mm),
    )
    best_by_model = select_best_by_model(
        rows,
        quality_ee_p95_mm=float(args.quality_ee_p95_mm),
        quality_axis_p95_mm=float(args.quality_axis_p95_mm),
    )
    model_best_df = _scoreboard(list(best_by_model.values()))
    model_best_df.to_csv(out_dir / "model_best_numeric_scoreboard.csv", index=False)
    same_rows = [
        {**r, "selection_note": "same_global_candidate"}
        for r in rows
        if str(r["candidate_id"]) == str(global_best["candidate_id"])
    ]
    same_df = _scoreboard(same_rows)
    same_df.to_csv(out_dir / "same_global_candidate_numeric_scoreboard.csv", index=False)

    all_points = pd.concat(point_frames.values(), ignore_index=True)
    model_best_points = _points_for_rows(all_points, model_best_df)
    same_points = _points_for_rows(all_points, same_df)
    if not model_best_points.empty:
        model_best_points.to_parquet(out_dir / "model_best_points.parquet", index=False)
    if not same_points.empty:
        same_points.to_parquet(out_dir / "same_global_candidate_points.parquet", index=False)

    payload: dict[str, object] = {
        "mode": "ellipse_trajectory_benchmark",
        "generated_at": time.strftime("%F %T"),
        "dataset": str(args.dataset),
        "models_evaluated": len({r["model_id"] for r in rows}),
        "candidates_generated": candidates_generated,
        "candidates_evaluated": len(candidates),
        "rows": len(rows),
        "load_errors": load_errors,
        "candidate_filters": {
            "center_x_range": str(args.center_x_range),
            "centers_yz": _parse_centers(args.centers_yz),
            "amp_xy_values": _parse_float_list(args.amp_xy_values),
            "max_nn_p95_m": float(args.max_nn_p95_m),
            "max_nearest_x_mean_abs_mm": float(args.max_nearest_x_mean_abs_mm),
            "max_candidates_per_amp": int(args.max_candidates_per_amp),
            "fixed_bias_threshold_mm": float(args.fixed_bias_threshold_mm),
            "quality_ee_p95_mm": float(args.quality_ee_p95_mm),
            "quality_axis_p95_mm": float(args.quality_axis_p95_mm),
            "selection_policy": str(args.selection_policy),
        },
        "diagnostics": {
            "evaluated_amp_xy_values_m": sorted(float(v) for v in summary["amp_xy_m"].dropna().unique()),
            "quality_rows_axis3": int((quality_common & (summary["axiserr_max_p95_abs_mm"].astype(float) <= 3.0)).sum()),
            "quality_rows_axis5": int((quality_common & (summary["axiserr_max_p95_abs_mm"].astype(float) <= 5.0)).sum()),
        },
        "global_best": {k: _jsonable(v) for k, v in global_best.items()},
        "best_by_model": {model_id: {k: _jsonable(v) for k, v in row.items()} for model_id, row in best_by_model.items()},
    }
    (out_dir / "trajectory_benchmark_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    images: list[str] = []
    best_points = point_frames[(str(global_best["model_id"]), str(global_best["candidate_id"]))]
    title = f"global showcase: {global_best['model_id']} / {global_best['candidate_id']} / EE95={float(global_best['ee_p95_mm']):.2f} mm"
    _plot_trajectory_3d(best_points, title, out_dir / "best_global_3d.png")
    images.append("best_global_3d.png")
    for left, right in (("x", "y"), ("x", "z"), ("y", "z")):
        img = f"best_global_{left}{right}.png"
        _plot_projection(best_points, title, out_dir / img, left=left, right=right)
        images.append(img)
    _plot_axis_errors(best_points, title, out_dir / "best_global_axis_errors.png")
    images.append("best_global_axis_errors.png")
    _plot_all_model_3d_grid(model_best_points, model_best_df, "All models: selected display ellipse by model", out_dir / "all_models_best_3d_grid.png")
    images.append("all_models_best_3d_grid.png")
    _plot_all_model_axis_errors(model_best_points, model_best_df, "All models: selected display ellipse axis errors", out_dir / "all_models_best_axis_errors_grid.png")
    images.append("all_models_best_axis_errors_grid.png")
    _plot_all_model_3d_grid(same_points, same_df, f"All models on one ellipse: {global_best['candidate_id']}", out_dir / "all_models_same_global_candidate_3d_grid.png")
    images.append("all_models_same_global_candidate_3d_grid.png")
    _plot_all_model_axis_errors(same_points, same_df, f"All models on one ellipse axis errors: {global_best['candidate_id']}", out_dir / "all_models_same_global_candidate_axis_errors_grid.png")
    images.append("all_models_same_global_candidate_axis_errors_grid.png")
    _plot_metric_bars(model_best_df, "Best display ellipse by model", out_dir / "model_best_metric_bars.png")
    images.append("model_best_metric_bars.png")
    _plot_metric_bars(same_df, "Same global showcase ellipse by model", out_dir / "same_global_candidate_metric_bars.png")
    images.append("same_global_candidate_metric_bars.png")

    write_readme(out_dir, payload, images, model_best_df, same_df)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark open-loop diagonal ellipse tracking across direct inverse models.")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    ap.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--center-x-range", default="1.190,1.215,0.001")
    ap.add_argument("--centers-yz", default="0,0;0.02,0;-0.02,0;0,0.02;0,-0.02;0.02,0.02;-0.02,0.02;0.02,-0.02;-0.02,-0.02;0.04,0;-0.04,0")
    ap.add_argument("--amp-xy-values", default="0.005,0.0075,0.010,0.015,0.020,0.030,0.040,0.060,0.080,0.100,0.120,0.140")
    ap.add_argument("--n-points", type=int, default=360)
    ap.add_argument("--models", default="", help="Comma-separated model ids or model names. Empty means all existing built-in direct models.")
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--max-candidates-per-amp", type=int, default=12)
    ap.add_argument("--max-nn-p95-m", type=float, default=0.008)
    ap.add_argument("--max-nearest-x-mean-abs-mm", type=float, default=2.0)
    ap.add_argument("--fixed-bias-threshold-mm", type=float, default=2.0)
    ap.add_argument("--majority-threshold", type=float, default=0.60)
    ap.add_argument("--tension-upper-n", type=float, default=2000.0)
    ap.add_argument("--quality-ee-p95-mm", type=float, default=5.0)
    ap.add_argument("--quality-axis-p95-mm", type=float, default=3.0)
    ap.add_argument("--selection-policy", default="display_largest_under_quality", choices=["display_largest_under_quality"])
    return ap.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
