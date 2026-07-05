#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from run_circle_trajectory_benchmark import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DEFAULT_ROBOT_CONFIG,
    CircleCandidate,
    ModelSpec,
    _load_model_and_scaler,
    _load_robot_paths,
    _read_lengths_end,
    build_builtin_model_registry,
    evaluate_model_candidate,
    filter_existing_models,
    make_offset_circle_points,
)


SCOREBOARD_COLUMNS = [
    "model_id",
    "family",
    "train_split",
    "model_name",
    "candidate_id",
    "x0_m",
    "center_y_m",
    "center_z_m",
    "radius_m",
    "supported",
    "nn_dist_mean_mm",
    "nn_dist_p95_mm",
    "nn_dist_max_mm",
    "ee_mean_mm",
    "ee_rmse_mm",
    "ee_p50_mm",
    "ee_p95_mm",
    "ee_max_mm",
    "xerr_mean_mm",
    "xerr_p50_mm",
    "xerr_p95_abs_mm",
    "xerr_min_mm",
    "xerr_max_mm",
    "xerr_same_sign_ratio",
    "fixed_x_bias_gt2mm",
    "nearest_x_diff_mean_mm",
    "nearest_x_diff_p95_abs_mm",
    "strict_geometry_supported",
    "pred_time_ms_per_point",
    "tension_min_n",
    "tension_max_n",
    "tension_mean_n",
    "negative_tension_ratio",
    "over_upper_tension_ratio",
    "selection_note",
]

METRIC_COLUMNS = ["ee_mean_mm", "ee_rmse_mm", "ee_p95_mm", "ee_max_mm"]
TRAJECTORY_3D_VIEW = {"elev": 22, "azim": -62}
TRAJECTORY_3D_BOX_ASPECT = (1.0, 1.0, 1.0)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_payload(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ordered_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in SCOREBOARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col in [
        "x0_m",
        "center_y_m",
        "center_z_m",
        "radius_m",
        "nn_dist_mean_mm",
        "nn_dist_p95_mm",
        "nn_dist_max_mm",
        "ee_mean_mm",
        "ee_rmse_mm",
        "ee_p50_mm",
        "ee_p95_mm",
        "ee_max_mm",
        "xerr_mean_mm",
        "xerr_p50_mm",
        "xerr_p95_abs_mm",
        "xerr_min_mm",
        "xerr_max_mm",
        "xerr_same_sign_ratio",
        "nearest_x_diff_mean_mm",
        "nearest_x_diff_p95_abs_mm",
        "pred_time_ms_per_point",
        "tension_min_n",
        "tension_max_n",
        "tension_mean_n",
        "negative_tension_ratio",
        "over_upper_tension_ratio",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[SCOREBOARD_COLUMNS].sort_values(["ee_p95_mm", "ee_rmse_mm", "ee_max_mm", "model_id"], ascending=True)
    return df.reset_index(drop=True)


def best_by_model_scoreboard(payload: dict[str, object]) -> pd.DataFrame:
    """Return the per-model selected trajectory rows from benchmark JSON."""
    raw = payload.get("best_by_model", {})
    if not isinstance(raw, dict):
        return pd.DataFrame(columns=SCOREBOARD_COLUMNS)
    rows = []
    for model_id, row in raw.items():
        if isinstance(row, dict):
            rows.append({"model_id": str(row.get("model_id", model_id)), **row})
    return _ordered_scoreboard(pd.DataFrame(rows))


def same_candidate_scoreboard(summary: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    sub = summary.loc[summary["candidate_id"].astype(str) == str(candidate_id)].copy()
    sub["selection_note"] = "same_global_candidate"
    return _ordered_scoreboard(sub)


def select_unique_points_for_rows(points: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    if points.empty or rows.empty:
        return pd.DataFrame(columns=[c for c in points.columns if c != "representative_label"])
    keys = rows[["model_id", "candidate_id"]].drop_duplicates().astype(str)
    merged = points.copy()
    merged["model_id"] = merged["model_id"].astype(str)
    merged["candidate_id"] = merged["candidate_id"].astype(str)
    selected = merged.merge(keys, on=["model_id", "candidate_id"], how="inner")
    drop_cols = ["model_id", "candidate_id", "angle_deg"]
    selected = selected.drop_duplicates(subset=[c for c in drop_cols if c in selected.columns])
    if "representative_label" in selected.columns:
        selected = selected.drop(columns=["representative_label"])
    return selected.sort_values(["model_id", "candidate_id", "angle_deg"]).reset_index(drop=True)


def scoreboard_to_markdown(scoreboard: pd.DataFrame, title: str, top_n: int = 25) -> str:
    if scoreboard.empty:
        return f"## {title}\n\nNo rows.\n"
    cols = [
        "model_id",
        "candidate_id",
        "radius_m",
        "nn_dist_p95_mm",
        "nearest_x_diff_mean_mm",
        "ee_mean_mm",
        "ee_rmse_mm",
        "ee_p95_mm",
        "ee_max_mm",
        "xerr_mean_mm",
        "xerr_p95_abs_mm",
        "fixed_x_bias_gt2mm",
        "pred_time_ms_per_point",
        "tension_min_n",
        "tension_max_n",
        "negative_tension_ratio",
        "over_upper_tension_ratio",
        "supported",
    ]
    table = scoreboard.loc[:, [c for c in cols if c in scoreboard.columns]].head(int(top_n)).copy()
    for col in table.select_dtypes(include=[np.number]).columns:
        table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body = []
    for _, row in table.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in table.columns) + " |")
    return f"## {title}\n\n" + "\n".join([header, sep, *body]) + "\n"


def _model_spec_map() -> dict[str, ModelSpec]:
    return {spec.model_id: spec for spec in filter_existing_models(build_builtin_model_registry(REPO_ROOT))}


def _candidate_from_row(row: pd.Series | dict[str, object]) -> CircleCandidate:
    return CircleCandidate(
        candidate_id=str(row["candidate_id"]),
        x0_m=float(row["x0_m"]),
        center_y_m=float(row["center_y_m"]),
        center_z_m=float(row["center_z_m"]),
        radius_m=float(row["radius_m"]),
        n_points=int(row.get("n_points", 240)),
    )


def _empty_workspace_meta() -> dict[str, float | bool | str]:
    return {
        "nn_dist_mean_mm": np.nan,
        "nn_dist_p95_mm": np.nan,
        "nn_dist_max_mm": np.nan,
        "supported": True,
    }


def evaluate_rows_to_points(
    rows: pd.DataFrame,
    robot_config: Path,
    tension_upper_n: float,
    allowed_model_ids: set[str] | None = None,
) -> pd.DataFrame:
    specs = _model_spec_map()
    lengths_csv, ee_csv = _load_robot_paths(str(robot_config), None, None)
    lengths_m, p_end_local_m = _read_lengths_end(lengths_csv, ee_csv)
    frames: list[pd.DataFrame] = []
    model_cache: dict[str, tuple[object, object]] = {}
    for _, row in rows.iterrows():
        model_id = str(row["model_id"])
        if allowed_model_ids is not None and model_id not in allowed_model_ids:
            continue
        spec = specs.get(model_id)
        if spec is None:
            continue
        if model_id not in model_cache:
            model_cache[model_id] = _load_model_and_scaler(spec.model_path, spec.scaler_path)
        model, scaler = model_cache[model_id]
        candidate = _candidate_from_row(row)
        target = make_offset_circle_points(
            candidate.x0_m,
            candidate.center_y_m,
            candidate.center_z_m,
            candidate.radius_m,
            candidate.n_points,
        )
        _, points = evaluate_model_candidate(
            spec=spec,
            candidate=candidate,
            target_xyz_m=target,
            workspace_meta=_empty_workspace_meta(),
            model=model,
            scaler=scaler,
            lengths_m=lengths_m,
            p_end_local_m=p_end_local_m,
            tension_upper_n=float(tension_upper_n),
        )
        frames.append(points)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _point_axis_limits(points: pd.DataFrame, axes: tuple[str, str] = ("y", "z"), pad_frac: float = 0.08) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for axis in axes:
        vals = np.concatenate(
            [
                points[f"target_{axis}_m"].to_numpy(dtype=float),
                points[f"achieved_{axis}_m"].to_numpy(dtype=float),
            ]
        )
        lo, hi = float(np.min(vals)), float(np.max(vals))
        span = max(hi - lo, 0.05)
        pad = span * float(pad_frac)
        limits[axis] = (lo - pad, hi + pad)
    return limits


def _set_3d_common_limits(ax: plt.Axes, limits: dict[str, tuple[float, float]]) -> None:
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    try:
        ax.set_box_aspect(TRAJECTORY_3D_BOX_ASPECT)
    except Exception:
        pass


def plot_all_model_yz_grid(points: pd.DataFrame, scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if points.empty or scoreboard.empty:
        return
    n = len(scoreboard)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    limits = _point_axis_limits(points, ("y", "z"))
    for ax in axes.reshape(-1):
        ax.axis("off")
    for idx, row in scoreboard.reset_index(drop=True).iterrows():
        ax = axes.reshape(-1)[idx]
        model_id = str(row["model_id"])
        candidate_id = str(row["candidate_id"])
        sub = points[(points["model_id"].astype(str) == model_id) & (points["candidate_id"].astype(str) == candidate_id)]
        if sub.empty:
            continue
        ax.axis("on")
        ax.plot(sub["target_y_m"], sub["target_z_m"], color="#1f4e79", lw=1.8, label="target")
        ax.plot(sub["achieved_y_m"], sub["achieved_z_m"], color="#c7511f", lw=1.6, label="model+FK")
        ax.set_xlim(*limits["y"])
        ax.set_ylim(*limits["z"])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        supported = str(row.get("supported", ""))
        ax.set_title(
            f"{idx + 1}. {model_id}\np95={float(row['ee_p95_mm']):.2f} max={float(row['ee_max_mm']):.2f} mm | r={float(row['radius_m']):.3f} | sup={supported}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=14, y=0.995)
    _savefig(fig, out_path)


def plot_all_model_3d_grid(points: pd.DataFrame, scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if points.empty or scoreboard.empty:
        return
    n = len(scoreboard)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(4.6 * ncols, 4.2 * nrows))
    limits = _point_axis_limits(points, ("x", "y", "z"), pad_frac=0.12)
    axes = [fig.add_subplot(nrows, ncols, idx + 1, projection="3d") for idx in range(nrows * ncols)]
    for ax in axes:
        ax.set_axis_off()
    for idx, row in scoreboard.reset_index(drop=True).iterrows():
        ax = axes[idx]
        model_id = str(row["model_id"])
        candidate_id = str(row["candidate_id"])
        sub = points[(points["model_id"].astype(str) == model_id) & (points["candidate_id"].astype(str) == candidate_id)]
        if sub.empty:
            continue
        ax.set_axis_on()
        ax.plot(sub["target_x_m"], sub["target_y_m"], sub["target_z_m"], color="#1f4e79", lw=1.8, label="target")
        ax.plot(sub["achieved_x_m"], sub["achieved_y_m"], sub["achieved_z_m"], color="#c7511f", lw=1.6, label="model+FK")
        ax.scatter(sub["target_x_m"].iloc[0], sub["target_y_m"].iloc[0], sub["target_z_m"].iloc[0], s=16, color="#1f4e79")
        ax.scatter(
            sub["achieved_x_m"].iloc[0],
            sub["achieved_y_m"].iloc[0],
            sub["achieved_z_m"].iloc[0],
            s=18,
            color="#c7511f",
            marker="x",
        )
        _set_3d_common_limits(ax, limits)
        ax.view_init(**TRAJECTORY_3D_VIEW)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("x", fontsize=7, labelpad=-2)
        ax.set_ylabel("y", fontsize=7, labelpad=-2)
        ax.set_zlabel("z", fontsize=7, labelpad=-2)
        ax.tick_params(labelsize=6, pad=-2)
        x_abs_err_mm = (sub["achieved_x_m"].to_numpy(dtype=float) - sub["target_x_m"].to_numpy(dtype=float))
        x95 = float(np.percentile(np.abs(x_abs_err_mm) * 1000.0, 95))
        supported = str(row.get("supported", ""))
        ax.set_title(
            f"{idx + 1}. {model_id}\np95={float(row['ee_p95_mm']):.2f} max={float(row['ee_max_mm']):.2f} mm | x95={x95:.2f} mm | sup={supported}",
            fontsize=8,
            y=0.98,
        )
    fig.suptitle(title, fontsize=14, y=0.995)
    _savefig(fig, out_path)


def _x_error_trajectory_limits(points: pd.DataFrame, pad_frac: float = 0.08) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for axis in ("y", "z"):
        vals = np.concatenate(
            [
                points[f"target_{axis}_m"].to_numpy(dtype=float) * 1000.0,
                points[f"achieved_{axis}_m"].to_numpy(dtype=float) * 1000.0,
            ]
        )
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        span = max(hi - lo, 20.0)
        pad = span * float(pad_frac)
        limits[axis] = (lo - pad, hi + pad)

    x_err_mm = (points["achieved_x_m"].to_numpy(dtype=float) - points["target_x_m"].to_numpy(dtype=float)) * 1000.0
    max_abs = max(float(np.nanmax(np.abs(x_err_mm))) if len(x_err_mm) else 0.0, 5.0)
    limits["x_error"] = (-max_abs * 1.12, max_abs * 1.12)
    return limits


def plot_all_model_x_error_trajectory_3d_grid(
    points: pd.DataFrame,
    scoreboard: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    if points.empty or scoreboard.empty:
        return
    n = len(scoreboard)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(4.8 * ncols, 4.4 * nrows))
    limits = _x_error_trajectory_limits(points)
    axes = [fig.add_subplot(nrows, ncols, idx + 1, projection="3d") for idx in range(nrows * ncols)]
    for ax in axes:
        ax.set_axis_off()
    for idx, row in scoreboard.reset_index(drop=True).iterrows():
        ax = axes[idx]
        model_id = str(row["model_id"])
        candidate_id = str(row["candidate_id"])
        sub = points[(points["model_id"].astype(str) == model_id) & (points["candidate_id"].astype(str) == candidate_id)]
        if sub.empty:
            continue

        target_y_mm = sub["target_y_m"].to_numpy(dtype=float) * 1000.0
        target_z_mm = sub["target_z_m"].to_numpy(dtype=float) * 1000.0
        achieved_y_mm = sub["achieved_y_m"].to_numpy(dtype=float) * 1000.0
        achieved_z_mm = sub["achieved_z_m"].to_numpy(dtype=float) * 1000.0
        x_err_mm = (sub["achieved_x_m"].to_numpy(dtype=float) - sub["target_x_m"].to_numpy(dtype=float)) * 1000.0

        ax.set_axis_on()
        ax.plot(target_y_mm, target_z_mm, np.zeros_like(target_y_mm), color="#1f4e79", lw=1.8, label="target x plane")
        ax.plot(achieved_y_mm, achieved_z_mm, x_err_mm, color="#c7511f", lw=1.6, label="model x drift")
        ax.scatter(target_y_mm[0], target_z_mm[0], 0.0, s=16, color="#1f4e79")
        ax.scatter(achieved_y_mm[0], achieved_z_mm[0], x_err_mm[0], s=18, color="#c7511f", marker="x")
        ax.set_xlim(*limits["y"])
        ax.set_ylim(*limits["z"])
        ax.set_zlim(*limits["x_error"])
        try:
            ax.set_box_aspect((1.0, 1.0, 0.65))
        except Exception:
            pass
        ax.view_init(**TRAJECTORY_3D_VIEW)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("y mm", fontsize=7, labelpad=-2)
        ax.set_ylabel("z mm", fontsize=7, labelpad=-2)
        ax.set_zlabel("x err mm", fontsize=7, labelpad=-1)
        ax.tick_params(labelsize=6, pad=-2)
        x95 = float(np.percentile(np.abs(x_err_mm), 95))
        x_max = float(np.max(np.abs(x_err_mm)))
        supported = str(row.get("supported", ""))
        ax.set_title(
            f"{idx + 1}. {model_id}\nEE95={float(row['ee_p95_mm']):.2f} mm | x95={x95:.2f} xMax={x_max:.2f} mm | sup={supported}",
            fontsize=8,
            y=0.98,
        )
    fig.suptitle(title, fontsize=14, y=0.995)
    _savefig(fig, out_path)


def plot_all_model_x_error_angle_grid(points: pd.DataFrame, scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if points.empty or scoreboard.empty:
        return
    plot_points = points.copy()
    if "x_error_mm" not in plot_points.columns:
        plot_points["x_error_mm"] = (plot_points["achieved_x_m"] - plot_points["target_x_m"]) * 1000.0
    n = len(scoreboard)
    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    max_abs = max(float(np.nanmax(np.abs(plot_points["x_error_mm"].to_numpy(dtype=float)))) if len(plot_points) else 0.0, 2.5)
    ylim = (-max_abs * 1.12, max_abs * 1.12)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for idx, row in scoreboard.reset_index(drop=True).iterrows():
        ax = axes.reshape(-1)[idx]
        model_id = str(row["model_id"])
        candidate_id = str(row["candidate_id"])
        sub = plot_points[(plot_points["model_id"].astype(str) == model_id) & (plot_points["candidate_id"].astype(str) == candidate_id)]
        if sub.empty:
            continue
        ax.axis("on")
        ax.axhline(0.0, color="#1f4e79", lw=1.0, alpha=0.7)
        ax.axhspan(-2.0, 2.0, color="#54a24b", alpha=0.08, linewidth=0)
        ax.plot(sub["angle_deg"], sub["x_error_mm"], color="#c7511f", lw=1.5)
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 180, 360])
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        x95 = float(row["xerr_p95_abs_mm"]) if "xerr_p95_abs_mm" in row and not pd.isna(row["xerr_p95_abs_mm"]) else float(np.percentile(np.abs(sub["x_error_mm"]), 95))
        raw_fixed = row.get("fixed_x_bias_gt2mm", False)
        fixed = False if pd.isna(raw_fixed) else bool(raw_fixed)
        ax.set_title(
            f"{idx + 1}. {model_id}\nEE95={float(row['ee_p95_mm']):.2f} mm | x95={x95:.2f} mm | fixed={fixed}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=14, y=0.995)
    _savefig(fig, out_path)


def plot_metric_bars(scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if scoreboard.empty:
        return
    plot_df = scoreboard.sort_values("ee_p95_mm", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(15.0, max(6.0, 0.36 * len(plot_df))), gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax = axes[0]
    offsets = [-0.24, -0.08, 0.08, 0.24]
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756"]
    for metric, offset, color in zip(METRIC_COLUMNS, offsets, colors, strict=True):
        ax.barh(y + offset, plot_df[metric].to_numpy(dtype=float), height=0.14, color=color, label=metric)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model_id"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("EE error (mm)")
    ax.set_title("Accuracy")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax2 = axes[1]
    ax2.barh(y, plot_df["pred_time_ms_per_point"].to_numpy(dtype=float), color="#54a24b", height=0.44)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Prediction time (ms/point)")
    ax2.set_title("Speed")
    ax2.grid(True, axis="x", alpha=0.25)
    fig.suptitle(title, fontsize=14)
    _savefig(fig, out_path)


def plot_speed_vs_accuracy(scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if scoreboard.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    colors = {"mlp": "#4c78a8", "mlp_large": "#72b7b2", "knn": "#f58518", "rf": "#54a24b", "lgbm": "#e45756"}
    for _, row in scoreboard.iterrows():
        color = colors.get(str(row.get("model_name", "")), "#777777")
        ax.scatter(float(row["pred_time_ms_per_point"]), float(row["ee_p95_mm"]), s=72, color=color, edgecolors="black", linewidths=0.4)
        ax.annotate(str(row["model_id"]).replace("relabel_", ""), (float(row["pred_time_ms_per_point"]), float(row["ee_p95_mm"])), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Prediction time (ms/point)")
    ax.set_ylabel("EE p95 error (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path)


def plot_tension_validity(scoreboard: pd.DataFrame, title: str, out_path: Path) -> None:
    if scoreboard.empty:
        return
    plot_df = scoreboard.sort_values("ee_p95_mm", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(15.0, max(6.0, 0.36 * len(plot_df))), gridspec_kw={"width_ratios": [1.1, 1.0]})
    axes[0].hlines(y, plot_df["tension_min_n"], plot_df["tension_max_n"], color="#4c78a8", lw=3)
    axes[0].scatter(plot_df["tension_min_n"], y, s=22, color="#1f4e79", label="min")
    axes[0].scatter(plot_df["tension_max_n"], y, s=22, color="#c7511f", label="max")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["model_id"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Tension range (N)")
    axes[0].set_title("Predicted tension range")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    bad_ratio = plot_df["negative_tension_ratio"].to_numpy(dtype=float) + plot_df["over_upper_tension_ratio"].to_numpy(dtype=float)
    axes[1].barh(y, bad_ratio, color="#e45756", height=0.44)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("negative + over-upper ratio")
    axes[1].set_title("Tension validity")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.suptitle(title, fontsize=14)
    _savefig(fig, out_path)


def update_readme(out_dir: Path, image_names: list[str], model_best: pd.DataFrame, same_candidate: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    original = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Circle Trajectory Benchmark\n"
    marker = "\n## All-Model Overview\n"
    original = original.split(marker, 1)[0].rstrip()
    lines = [
        original,
        "",
        "## All-Model Overview",
        "",
        "These files compare every direct `xyz -> theta/T` model evaluated by the circular trajectory benchmark.",
        "",
        "### New Figures",
        "",
    ]
    descriptions = {
        "all_models_best_yz_grid.png": "Each model's own best supported trajectory, ranked by EE p95 error.",
        "all_models_best_3d_grid.png": "3D view of each model's own best supported trajectory, with x-axis error visible in the subplot title.",
        "all_models_best_x_error_trajectory_3d_grid.png": "3D trajectory view where the circle plane is y-z and the third axis is x-axis error in mm.",
        "all_models_best_xerr_vs_angle_grid.png": "x-axis error versus circle angle for each model's own best trajectory; the green band marks +/-2 mm.",
        "all_models_same_global_candidate_yz_grid.png": "All models on the same global-best target circle for a fair visual comparison.",
        "all_models_same_global_candidate_3d_grid.png": "3D view of all models on the same global-best target circle, intended to expose x-axis drift hidden by YZ plots.",
        "all_models_same_global_candidate_x_error_trajectory_3d_grid.png": "Same-target 3D trajectory view with x-axis error as the third axis, making forward/backward drift visible.",
        "all_models_same_global_candidate_xerr_vs_angle_grid.png": "Same-target x-axis error versus circle angle; the green band marks +/-2 mm.",
        "model_best_metric_bars.png": "Best-by-model numerical accuracy and speed ranking.",
        "same_global_candidate_metric_bars.png": "Same-target numerical accuracy and speed ranking.",
        "model_speed_vs_accuracy.png": "Speed versus EE p95 accuracy for best-by-model rows.",
        "model_tension_validity_overview.png": "Predicted tension range and invalid-tension ratios for best-by-model rows.",
    }
    for name in image_names:
        lines.append(f"- `{name}`: {descriptions.get(name, 'Generated overview figure.')}")
    lines.extend(
        [
            "",
            "- `model_best_numeric_scoreboard.csv`: per-model best supported trajectory metrics.",
            "- `same_global_candidate_numeric_scoreboard.csv`: every model evaluated on the same global-best circle.",
            "- `same_global_candidate_points.parquet`: point-level trajectories for same-target model comparison.",
            "",
            scoreboard_to_markdown(model_best, "Best supported trajectory by model", top_n=25).rstrip(),
            "",
            scoreboard_to_markdown(same_candidate, "Same global-best circle by model", top_n=25).rstrip(),
            "",
        ]
    )
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    summary = pd.read_csv(out_dir / "trajectory_benchmark_summary.csv")
    payload = _load_payload(out_dir / "trajectory_benchmark_summary.json")
    selected_points_path = out_dir / "selected_trajectory_points.parquet"
    selected_points = pd.read_parquet(selected_points_path) if selected_points_path.exists() else pd.DataFrame()

    model_best = best_by_model_scoreboard(payload)
    model_best.to_csv(out_dir / "model_best_numeric_scoreboard.csv", index=False)
    model_best_points = select_unique_points_for_rows(selected_points, model_best)
    missing_keys = set(map(tuple, model_best[["model_id", "candidate_id"]].astype(str).to_numpy()))
    if not model_best_points.empty:
        present_keys = set(map(tuple, model_best_points[["model_id", "candidate_id"]].astype(str).drop_duplicates().to_numpy()))
        missing_keys -= present_keys
    if missing_keys:
        missing_rows = model_best[
            model_best.apply(lambda r: (str(r["model_id"]), str(r["candidate_id"])) in missing_keys, axis=1)
        ]
        extra = evaluate_rows_to_points(missing_rows, Path(args.robot_config), tension_upper_n=float(args.tension_upper_n))
        if not extra.empty:
            model_best_points = pd.concat([model_best_points, extra], ignore_index=True)

    global_best = payload.get("global_best", {})
    if not isinstance(global_best, dict) or not global_best:
        raise SystemExit("trajectory_benchmark_summary.json does not contain global_best")
    same_scoreboard = same_candidate_scoreboard(summary, str(global_best["candidate_id"]))
    same_candidate = _candidate_from_row(global_best)
    same_points = evaluate_rows_to_points(same_scoreboard, Path(args.robot_config), tension_upper_n=float(args.tension_upper_n))
    same_scoreboard.to_csv(out_dir / "same_global_candidate_numeric_scoreboard.csv", index=False)
    if not same_points.empty:
        same_points.to_parquet(out_dir / "same_global_candidate_points.parquet", index=False)

    image_names: list[str] = []
    plot_all_model_yz_grid(model_best_points, model_best, "All models: each model's best supported circular trajectory", out_dir / "all_models_best_yz_grid.png")
    image_names.append("all_models_best_yz_grid.png")
    plot_all_model_3d_grid(model_best_points, model_best, "All models 3D: each model's best supported circular trajectory", out_dir / "all_models_best_3d_grid.png")
    image_names.append("all_models_best_3d_grid.png")
    plot_all_model_x_error_trajectory_3d_grid(
        model_best_points,
        model_best,
        "All models: y-z circle with x-axis error trajectory",
        out_dir / "all_models_best_x_error_trajectory_3d_grid.png",
    )
    image_names.append("all_models_best_x_error_trajectory_3d_grid.png")
    plot_all_model_x_error_angle_grid(
        model_best_points,
        model_best,
        "All models: x-axis error versus circle angle",
        out_dir / "all_models_best_xerr_vs_angle_grid.png",
    )
    image_names.append("all_models_best_xerr_vs_angle_grid.png")
    plot_all_model_yz_grid(same_points, same_scoreboard, f"All models on one circle: {same_candidate.candidate_id}", out_dir / "all_models_same_global_candidate_yz_grid.png")
    image_names.append("all_models_same_global_candidate_yz_grid.png")
    plot_all_model_3d_grid(same_points, same_scoreboard, f"All models 3D on one circle: {same_candidate.candidate_id}", out_dir / "all_models_same_global_candidate_3d_grid.png")
    image_names.append("all_models_same_global_candidate_3d_grid.png")
    plot_all_model_x_error_trajectory_3d_grid(
        same_points,
        same_scoreboard,
        f"All models on one circle: y-z circle with x-axis error trajectory ({same_candidate.candidate_id})",
        out_dir / "all_models_same_global_candidate_x_error_trajectory_3d_grid.png",
    )
    image_names.append("all_models_same_global_candidate_x_error_trajectory_3d_grid.png")
    plot_all_model_x_error_angle_grid(
        same_points,
        same_scoreboard,
        f"All models on one circle: x-axis error versus angle ({same_candidate.candidate_id})",
        out_dir / "all_models_same_global_candidate_xerr_vs_angle_grid.png",
    )
    image_names.append("all_models_same_global_candidate_xerr_vs_angle_grid.png")
    plot_metric_bars(model_best, "Best supported trajectory by model", out_dir / "model_best_metric_bars.png")
    image_names.append("model_best_metric_bars.png")
    plot_metric_bars(same_scoreboard, "Same global-best circle by model", out_dir / "same_global_candidate_metric_bars.png")
    image_names.append("same_global_candidate_metric_bars.png")
    plot_speed_vs_accuracy(model_best, "Best-by-model speed vs accuracy", out_dir / "model_speed_vs_accuracy.png")
    image_names.append("model_speed_vs_accuracy.png")
    plot_tension_validity(model_best, "Best-by-model tension validity", out_dir / "model_tension_validity_overview.png")
    image_names.append("model_tension_validity_overview.png")
    update_readme(out_dir, image_names, model_best, same_scoreboard)

    return {
        "out_dir": str(out_dir),
        "models": int(model_best["model_id"].nunique()),
        "same_candidate": same_candidate.candidate_id,
        "best_by_model_csv": str(out_dir / "model_best_numeric_scoreboard.csv"),
        "same_candidate_csv": str(out_dir / "same_global_candidate_numeric_scoreboard.csv"),
        "images": image_names,
        "best_model": str(model_best.iloc[0]["model_id"]) if not model_best.empty else "",
        "best_model_ee_p95_mm": float(model_best.iloc[0]["ee_p95_mm"]) if not model_best.empty else float("nan"),
        "same_candidate_best_model": str(same_scoreboard.iloc[0]["model_id"]) if not same_scoreboard.empty else "",
        "same_candidate_best_ee_p95_mm": float(same_scoreboard.iloc[0]["ee_p95_mm"]) if not same_scoreboard.empty else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create all-model visual and numeric overviews for circle trajectory benchmark results.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    ap.add_argument("--tension-upper-n", type=float, default=2000.0)
    return ap.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
