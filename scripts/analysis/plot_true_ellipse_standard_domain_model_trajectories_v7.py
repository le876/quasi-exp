#!/usr/bin/env python3
"""Render V7 model trajectories with the established tilted SVD camera policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
for candidate in (str(REPO_ROOT / "src"), str(ANALYSIS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_true_ellipse_standard_domain_v7 as existing_plot  # noqa: E402
import run_true_ellipse_standard_domain_diagnostic_training_v7 as training  # noqa: E402


DEFAULT_RUN_DIR = training.DEFAULT_OUT_DIR
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "05_visualization"
TARGET_COLOR = existing_plot.TARGET_COLOR
ACHIEVED_COLOR = existing_plot.ACHIEVED_COLOR
IDENTITY_COLOR = "#4c78a8"
BOUNDED_COLOR = "#e45756"
ERROR_COLORS = ("#4c78a8", "#f58518", "#54a24b")
FIGURE_DPI = 150
VIEW_OFFSET_DEG = 12.0
TRAJECTORY_LINK_COLORS = {
    "identity": ACHIEVED_COLOR,
    "tanh_bounds": BOUNDED_COLOR,
}
FORMAL_CENTERLINE_LABELS = (
    "validation_integer_centerline",
    "validation_half_phase",
    "test_integer_centerline",
    "test_half_phase",
)
METRIC_LINK_COLORS = {
    "identity": IDENTITY_COLOR,
    "tanh_bounds": BOUNDED_COLOR,
}

read_json = existing_plot.read_json
write_json = existing_plot.write_json
sha256_file = existing_plot.sha256_file
_closed = existing_plot._closed
_set_3d_equal = existing_plot._set_3d_equal


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(str(column) for column in frame.columns).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def trajectory_view_from_xyz(
    xyz_m: np.ndarray,
    *,
    azimuth_offset_deg: float = VIEW_OFFSET_DEG,
) -> dict[str, float]:
    view = existing_plot.trajectory_view_from_xyz(
        xyz_m, azimuth_offset_deg=float(azimuth_offset_deg)
    )
    return {
        **view,
        "azimuth_offset_deg": float(azimuth_offset_deg),
    }


def camera_direction_from_view(view: Mapping[str, float]) -> np.ndarray:
    return existing_plot.camera_direction_from_view(dict(view))


def _target_achieved(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target = curve[["x_target_m", "y_target_m", "z_target_m"]].to_numpy(dtype=float)
    achieved = curve[["achieved_x_m", "achieved_y_m", "achieved_z_m"]].to_numpy(
        dtype=float
    )
    if target.shape != achieved.shape or not np.isfinite(np.vstack([target, achieved])).all():
        raise ValueError("model trajectory target/achieved xyz must be finite and aligned")
    return target, achieved


def _principal_plane(target: np.ndarray, values: np.ndarray) -> np.ndarray:
    center = np.mean(target, axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(target - center, full_matrices=False)
    if len(singular) < 2 or singular[1] <= max(singular[0], 1.0) * 1.0e-10:
        raise ValueError("trajectory is degenerate in its fitted plane")
    return (np.asarray(values, dtype=float) - center) @ vh[:2].T * 1000.0


def _plot_3d(ax: Any, curve: pd.DataFrame, *, achieved_color: str = ACHIEVED_COLOR) -> None:
    target, achieved = _target_achieved(curve)
    ax.plot(*_closed(target).T, color=TARGET_COLOR, lw=2.0, ls="--", label="target")
    ax.plot(*_closed(achieved).T, color=achieved_color, lw=1.35, label="model + FK")
    stride = max(1, len(curve) // 24)
    ax.scatter(*achieved[::stride].T, s=8, color=achieved_color, alpha=0.72)
    _set_3d_equal(ax, np.vstack([target, achieved]))
    view = trajectory_view_from_xyz(target)
    ax.view_init(elev=view["elev"], azim=view["azim"])
    ax.set_xlabel("x (m)", labelpad=1)
    ax.set_ylabel("y (m)", labelpad=1)
    ax.set_zlabel("z (m)", labelpad=1)
    ax.tick_params(labelsize=6, pad=0)


def _plot_principal_plane(
    ax: Any,
    curve: pd.DataFrame,
    *,
    achieved_color: str = ACHIEVED_COLOR,
) -> None:
    target, achieved = _target_achieved(curve)
    target_2d = _principal_plane(target, target)
    achieved_2d = _principal_plane(target, achieved)
    ax.plot(
        *_closed(target_2d).T,
        color=TARGET_COLOR,
        lw=2.0,
        ls="--",
        label="target",
    )
    ax.plot(
        *_closed(achieved_2d).T,
        color=achieved_color,
        lw=1.35,
        label="model + FK",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("principal axis 1 (mm)")
    ax.set_ylabel("principal axis 2 (mm)")
    ax.grid(True, alpha=0.22)


def _plot_axis_error(ax: Any, curve: pd.DataFrame) -> None:
    angle_deg = np.mod(np.rad2deg(curve["angle_rad"].to_numpy(dtype=float)), 360.0)
    for axis, color in zip("xyz", ERROR_COLORS, strict=True):
        ax.plot(
            angle_deg,
            curve[f"{axis}_error_mm"].to_numpy(dtype=float),
            lw=1.0,
            color=color,
            label=f"{axis} error",
        )
    residual = curve["ee_err_mm"].to_numpy(dtype=float)
    ax.axhline(0.0, color="#333333", lw=0.7, alpha=0.6)
    ax.text(
        0.02,
        0.97,
        f"EE P95 {np.percentile(residual, 95):.3f} mm\nmax {np.max(residual):.3f} mm",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    ax.set_xlim(0.0, 360.0)
    ax.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
    ax.set_xlabel("ellipse phase (deg)")
    ax.set_ylabel("axis error (mm)")
    ax.grid(True, alpha=0.22)


def render_model_composite(
    *,
    config_id: str,
    curves: Mapping[float, pd.DataFrame],
    output_path: str | Path,
) -> dict[str, Any]:
    radii = sorted(float(radius) for radius in curves)
    if not radii:
        raise ValueError("model composite requires at least one radius")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    link = str(next(iter(curves.values())).get("output_link_id", pd.Series([""])).iloc[0])
    achieved_color = TRAJECTORY_LINK_COLORS.get(link, ACHIEVED_COLOR)
    fig = plt.figure(figsize=(14.4, 3.75 * len(radii)))
    for row_index, radius in enumerate(radii):
        curve = curves[radius].sort_values("angle_rad", kind="stable").reset_index(drop=True)
        ax3d = fig.add_subplot(len(radii), 3, row_index * 3 + 1, projection="3d")
        _plot_3d(ax3d, curve, achieved_color=achieved_color)
        ax3d.set_title(f"{radius:g} mm — face-on 3D", fontsize=9)
        if row_index == 0:
            ax3d.legend(frameon=False, fontsize=7, loc="upper left")
        ax_plane = fig.add_subplot(len(radii), 3, row_index * 3 + 2)
        _plot_principal_plane(ax_plane, curve, achieved_color=achieved_color)
        ax_plane.set_title("trajectory-plane projection", fontsize=9)
        if row_index == 0:
            ax_plane.legend(frameon=False, fontsize=7)
        ax_error = fig.add_subplot(len(radii), 3, row_index * 3 + 3)
        _plot_axis_error(ax_error, curve)
        ax_error.set_title("Cartesian tracking errors", fontsize=9)
        if row_index == 0:
            ax_error.legend(frameon=False, fontsize=6, ncol=3, loc="lower left")
    fig.suptitle(
        f"V7-D1 DIAGNOSTIC ONLY — {config_id}\n"
        "target vs static xyz→beta6 model→FK; no change to formal V7 gate",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        **training.diagnostic_claims(),
        "figure_kind": "individual_model_composite",
        "config_id": str(config_id),
        "radii_mm": radii,
        "view_policy": "svd_plane_normal_plus_12deg",
        "source_curve_hashes": {
            f"{radius:g}": _frame_fingerprint(curves[radius]) for radius in radii
        },
        "png_path": str(output.resolve()),
        "png_sha256": sha256_file(output),
    }


def _save_figure(fig: plt.Figure, path: Path, *, kind: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        **training.diagnostic_claims(),
        "figure_kind": str(kind),
        "png_path": str(path.resolve()),
        "png_sha256": sha256_file(path),
    }


def _short_config(config_id: str) -> str:
    base = config_id.split("__", 1)[0]
    return base.replace("mlp_beta6_", "").replace("poly_", "p_")


def _config_with_link_label(config_id: str) -> str:
    parts = str(config_id).split("__", 1)
    return _short_config(config_id) + (f"__{parts[1]}" if len(parts) == 2 else "")


def _render_contact_sheet(
    *,
    curves: Mapping[str, pd.DataFrame],
    link_id: str,
    output_path: Path,
) -> dict[str, Any]:
    items = sorted(curves.items())
    columns = 4
    rows = max(1, math.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15.0, 3.2 * rows), squeeze=False)
    color = TRAJECTORY_LINK_COLORS.get(str(link_id), ACHIEVED_COLOR)
    for ax, (config_id, curve) in zip(axes.flat, items):
        _plot_principal_plane(ax, curve, achieved_color=color)
        p95 = float(np.percentile(curve["ee_err_mm"].to_numpy(dtype=float), 95))
        ax.set_title(f"{_short_config(config_id)}\nEE P95 {p95:.3f} mm", fontsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(items) :]:
        ax.axis("off")
    fig.suptitle(
        f"V7-D1 {link_id} models at 104 mm — DIAGNOSTIC ONLY",
        fontsize=13,
    )
    record = _save_figure(fig, output_path, kind=f"contact_sheet_{link_id}")
    record["model_count"] = len(items)
    record["radius_mm"] = 104.0
    return record


def _render_metric_bars(metrics: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    table = metrics.loc[np.isclose(metrics["radius_mm"], 104.0)].copy()
    table = table.sort_values("ee_p95_mm", ascending=True, kind="stable").reset_index(drop=True)
    colors = [
        METRIC_LINK_COLORS.get(value, IDENTITY_COLOR)
        for value in table["output_link_id"].astype(str)
    ]
    height = max(8.0, 0.28 * len(table) + 2.0)
    fig, (ax_ee, ax_beta) = plt.subplots(1, 2, figsize=(16.0, height), sharey=True)
    y = np.arange(len(table))
    labels = [_short_config(value) + "__" + link for value, link in zip(
        table["config_id"].astype(str), table["output_link_id"].astype(str), strict=True
    )]
    ax_ee.barh(y, table["ee_p95_mm"], color=colors, alpha=0.86)
    ax_beta.barh(y, table["beta_p95_deg"], color=colors, alpha=0.86)
    ax_ee.set_yticks(y, labels=labels, fontsize=6)
    ax_ee.set_xlabel("EE P95 (mm)")
    ax_beta.set_xlabel("beta P95 (deg)")
    for ax in (ax_ee, ax_beta):
        ax.grid(True, axis="x", alpha=0.22)
    fig.suptitle("V7-D1 104 mm model metrics — DIAGNOSTIC ONLY")
    record = _save_figure(fig, output_path, kind="screen_model_metric_bars")
    record["model_count"] = len(table)
    return record


def _render_error_heatmap(metrics: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    pivot = metrics.pivot(index="config_id", columns="radius_mm", values="ee_p95_mm")
    pivot = pivot.sort_index().sort_index(axis=1)
    height = max(8.0, 0.24 * len(pivot) + 2.5)
    fig, ax = plt.subplots(figsize=(10.5, height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[f"{value:g}" for value in pivot.columns])
    ax.set_yticks(
        np.arange(len(pivot.index)),
        labels=[_config_with_link_label(value) for value in pivot.index],
        fontsize=6,
    )
    ax.set_xlabel("accepted continuous radius (mm)")
    ax.set_title("V7-D1 EE P95 heatmap (mm) — DIAGNOSTIC ONLY")
    fig.colorbar(image, ax=ax, label="EE P95 (mm)", pad=0.015)
    record = _save_figure(fig, output_path, kind="screen_model_error_heatmap")
    record["shape"] = [int(value) for value in pivot.shape]
    return record


def _render_link_pairs(metrics: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    table = metrics.loc[np.isclose(metrics["radius_mm"], 104.0)].copy()
    table["base_config_id"] = table["config_id"].astype(str).str.split("__").str[0]
    pivot = table.pivot(index="base_config_id", columns="output_link_id", values="ee_p95_mm")
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    if {"identity", "tanh_bounds"}.issubset(pivot.columns):
        ax.scatter(
            pivot["identity"],
            pivot["tanh_bounds"],
            color="#6f4e7c",
            s=34,
            alpha=0.82,
        )
        maximum = float(np.nanmax(pivot[["identity", "tanh_bounds"]].to_numpy()))
        maximum = max(maximum, 1.0e-6)
        ax.plot([0.0, maximum], [0.0, maximum], color="#333333", ls="--", lw=1.0)
    ax.set_xlabel("identity EE P95 at 104 mm (mm)")
    ax.set_ylabel("tanh_bounds EE P95 at 104 mm (mm)")
    ax.grid(True, alpha=0.22)
    ax.set_title("V7-D1 output-link paired comparison — DIAGNOSTIC ONLY")
    record = _save_figure(fig, output_path, kind="identity_vs_tanh_bounds")
    record["paired_base_config_count"] = int(len(pivot))
    return record


def _render_seed_comparison(
    *,
    curves: Mapping[int, Mapping[float, pd.DataFrame]],
    output_path: Path,
) -> dict[str, Any]:
    seeds = sorted(curves)
    radii = sorted({radius for values in curves.values() for radius in values})
    fig, axes = plt.subplots(
        max(1, len(seeds)), max(1, len(radii)),
        figsize=(4.2 * max(1, len(radii)), 3.25 * max(1, len(seeds))),
        squeeze=False,
    )
    for row, seed in enumerate(seeds):
        for column, radius in enumerate(radii):
            ax = axes[row, column]
            curve = curves[seed][radius]
            color = TRAJECTORY_LINK_COLORS.get(
                str(curve["output_link_id"].iloc[0]), ACHIEVED_COLOR
            )
            _plot_principal_plane(ax, curve, achieved_color=color)
            p95 = float(np.percentile(curve["ee_err_mm"], 95))
            ax.set_title(f"seed {seed} · {radius:g} mm · EE P95 {p95:.3f}", fontsize=8)
            ax.tick_params(labelsize=6)
            if row != len(seeds) - 1:
                ax.set_xlabel("")
            if column != 0:
                ax.set_ylabel("")
    fig.suptitle("V7-D1 selected configuration across seeds — DIAGNOSTIC ONLY")
    record = _save_figure(fig, output_path, kind="selected_config_seed_comparison")
    record["seeds"] = seeds
    record["radii_mm"] = radii
    return record


def _render_unsupported(
    *,
    curves: Mapping[float, pd.DataFrame],
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    radii = sorted(curves)
    columns = 2
    rows = math.ceil(len(radii) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11.5, 4.7 * rows), squeeze=False)
    for ax, radius in zip(axes.flat, radii):
        curve = curves[radius]
        color = TRAJECTORY_LINK_COLORS.get(
            str(curve["output_link_id"].iloc[0]), ACHIEVED_COLOR
        )
        _plot_principal_plane(ax, curve, achieved_color=color)
        p95 = float(np.percentile(curve["ee_err_mm"], 95))
        violations = int(curve["prediction_any_joint_out_of_bounds"].astype(bool).sum())
        ax.set_title(
            f"{radius:g} mm unsupported extrapolation\nEE P95 {p95:.3f} mm · OOB rows {violations}",
            fontsize=9,
        )
    for ax in axes.flat[len(radii) :]:
        ax.axis("off")
    fig.suptitle(
        f"V7-D1 selected model seed {seed}: target-only extrapolation\n"
        "NO beta truth · NO formal radius claim",
        fontsize=13,
    )
    record = _save_figure(fig, output_path, kind="selected_config_unsupported")
    record["seed"] = int(seed)
    record["radii_mm"] = radii
    record["beta_truth_available"] = False
    return record


def _load_prediction(entry: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(entry["path"])
    if sha256_file(path) != str(entry["sha256"]):
        raise ValueError(f"trajectory prediction hash changed: {path}")
    return pd.read_parquet(path)


def _normalize_formal_prediction(
    frame: pd.DataFrame,
    *,
    output_link_id: str,
) -> pd.DataFrame:
    required = {
        "angle_rad",
        "radius_mm",
        "x_target_m",
        "y_target_m",
        "z_target_m",
        "achieved_x_m",
        "achieved_y_m",
        "achieved_z_m",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"formal trajectory prediction is missing columns: {missing}")
    normalized = frame.copy()
    for axis in "xyz":
        normalized[f"{axis}_error_mm"] = (
            normalized[f"achieved_{axis}_m"].to_numpy(dtype=float)
            - normalized[f"{axis}_target_m"].to_numpy(dtype=float)
        ) * 1000.0
    if "ee_err_mm" not in normalized:
        normalized["ee_err_mm"] = np.linalg.norm(
            normalized[[f"{axis}_error_mm" for axis in "xyz"]].to_numpy(dtype=float),
            axis=1,
        )
    normalized["output_link_id"] = str(output_link_id)
    return normalized.sort_values("angle_rad", kind="stable").reset_index(drop=True)


def _formal_claims(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_only": False,
        "formal_claims_allowed": bool(report.get("formal_claims_allowed", False)),
        "evidence_only": bool(report.get("evidence_only", False)),
        "model_evidence_gate_pass": bool(report.get("model_evidence_gate_pass", False)),
        "formal_model_gate_pass": bool(report.get("formal_model_gate_pass", False)),
        "changes_v7_formal_gate": False,
        "visualization_only": True,
    }


def render_formal_seed_composite(
    *,
    seed: int,
    config_id: str,
    curves: Mapping[str, pd.DataFrame],
    source_hashes: Mapping[str, str],
    training_report: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    labels = [label for label in FORMAL_CENTERLINE_LABELS if label in curves]
    if labels != list(FORMAL_CENTERLINE_LABELS):
        missing = sorted(set(FORMAL_CENTERLINE_LABELS) - set(labels))
        raise ValueError(f"formal seed {seed} is missing centerline predictions: {missing}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.4, 3.75 * len(labels)))
    for row_index, label in enumerate(labels):
        curve = curves[label]
        radius = float(curve["radius_mm"].iloc[0])
        link = str(curve["output_link_id"].iloc[0])
        achieved_color = TRAJECTORY_LINK_COLORS.get(link, ACHIEVED_COLOR)
        label_text = label.replace("_", " ")
        ax3d = fig.add_subplot(len(labels), 3, row_index * 3 + 1, projection="3d")
        _plot_3d(ax3d, curve, achieved_color=achieved_color)
        ax3d.set_title(f"{label_text} · {radius:g} mm — tilted 3D", fontsize=9)
        if row_index == 0:
            ax3d.legend(frameon=False, fontsize=7, loc="upper left")
        ax_plane = fig.add_subplot(len(labels), 3, row_index * 3 + 2)
        _plot_principal_plane(ax_plane, curve, achieved_color=achieved_color)
        ax_plane.set_title("trajectory-plane projection", fontsize=9)
        if row_index == 0:
            ax_plane.legend(frameon=False, fontsize=7)
        ax_error = fig.add_subplot(len(labels), 3, row_index * 3 + 3)
        _plot_axis_error(ax_error, curve)
        ax_error.set_title("Cartesian tracking errors", fontsize=9)
        if row_index == 0:
            ax_error.legend(frameon=False, fontsize=6, ncol=3, loc="lower left")
    scope = "EVIDENCE ONLY" if bool(training_report.get("evidence_only", False)) else "FORMAL"
    fig.suptitle(
        f"V7 {scope} — {config_id} — seed {seed}\n"
        "target vs static xyz→beta6 model→FK; SVD plane-normal view + 12° offset",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        **_formal_claims(training_report),
        "figure_kind": "formal_seed_centerline_composite",
        "seed": int(seed),
        "config_id": str(config_id),
        "evaluation_labels": labels,
        "radii_mm": [float(curves[label]["radius_mm"].iloc[0]) for label in labels],
        "view_policy": "svd_plane_normal_plus_12deg",
        "source_prediction_sha256": {
            label: str(source_hashes[label]) for label in labels
        },
        "png_path": str(output.resolve()),
        "png_sha256": sha256_file(output),
    }


def render_formal_training_trajectory_suite(
    *,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    final_dir = run / "03_final_models"
    final_report_path = final_dir / "final_training_report.json"
    final_report = read_json(final_report_path)
    if str(final_report.get("preset", "")) != "formal":
        raise ValueError("formal trajectory rendering requires a formal training report")
    output = Path(out_dir).resolve() if out_dir is not None else run / "04_visualization"
    output.mkdir(parents=True, exist_ok=True)
    config_id = str(final_report["selected_config_id"])
    output_link_id = config_id.rsplit("__", 1)[-1] if "__" in config_id else "identity"
    seeds = [int(seed) for seed in final_report.get("seeds", [])]
    if not seeds or len(seeds) != int(final_report.get("seed_count", -1)):
        raise ValueError("formal training report seed protocol is incomplete")

    prediction_sources: list[dict[str, Any]] = []
    seed_figures: list[dict[str, Any]] = []
    for seed in seeds:
        curves: dict[str, pd.DataFrame] = {}
        source_hashes: dict[str, str] = {}
        for label in FORMAL_CENTERLINE_LABELS:
            path = final_dir / "holdout_predictions" / f"seed_{seed}" / f"{label}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"formal trajectory prediction is missing: {path}")
            source_hash = sha256_file(path)
            curve = _normalize_formal_prediction(
                pd.read_parquet(path), output_link_id=output_link_id
            )
            curves[label] = curve
            source_hashes[label] = source_hash
            prediction_sources.append(
                {
                    "seed": int(seed),
                    "evaluation_label": label,
                    "radius_mm": float(curve["radius_mm"].iloc[0]),
                    "rows": int(len(curve)),
                    "path": str(path.resolve()),
                    "sha256": source_hash,
                }
            )
        seed_figures.append(
            render_formal_seed_composite(
                seed=seed,
                config_id=config_id,
                curves=curves,
                source_hashes=source_hashes,
                training_report=final_report,
                output_path=output / "seed_trajectories" / f"seed_{seed}.png",
            )
        )

    expected_prediction_count = len(seeds) * len(FORMAL_CENTERLINE_LABELS)
    checks = {
        "seed_count_matches_training_report": len(seed_figures)
        == int(final_report["seed_count"]),
        "every_seed_has_all_centerline_holdouts": len(prediction_sources)
        == expected_prediction_count,
        "all_prediction_sources_still_match_hash": all(
            sha256_file(entry["path"]) == entry["sha256"] for entry in prediction_sources
        ),
        "all_pngs_written_and_hashed": all(
            Path(record["png_path"]).is_file()
            and sha256_file(record["png_path"]) == record["png_sha256"]
            for record in seed_figures
        ),
        "dynamic_camera_policy": all(
            record["view_policy"] == "svd_plane_normal_plus_12deg"
            for record in seed_figures
        ),
    }
    report = {
        "visualization_schema_version": 2,
        **_formal_claims(final_report),
        "view_policy": "svd_plane_normal_plus_12deg",
        "training_report_path": str(final_report_path.resolve()),
        "training_report_sha256": sha256_file(final_report_path),
        "selected_config_id": config_id,
        "seeds": seeds,
        "prediction_sources": prediction_sources,
        "seed_figures": seed_figures,
        "checks": checks,
        "visualization_integrity_gate_pass": bool(all(checks.values())),
    }
    write_json(output / "visualization_report.json", report)
    return report


def render_model_trajectory_suite(
    *,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    output = Path(out_dir).resolve() if out_dir is not None else run / "05_visualization"
    output.mkdir(parents=True, exist_ok=True)
    evaluation_path = run / "04_evaluation" / "evaluation_report.json"
    evaluation = read_json(evaluation_path)
    manifest_path = Path(evaluation["prediction_manifest_path"])
    manifest = read_json(manifest_path)
    metrics = pd.read_csv(evaluation["screen_model_metrics_path"])

    screen_grouped: dict[str, dict[float, pd.DataFrame]] = defaultdict(dict)
    screen_entry_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in manifest["screen_models"]:
        screen_grouped[str(entry["config_id"])][float(entry["radius_mm"])] = _load_prediction(entry)
        screen_entry_grouped[str(entry["config_id"])].append(entry)
    individual_records = []
    individual_dir = output / "individual_models"
    for config_id, curves in sorted(screen_grouped.items()):
        individual_records.append(
            render_model_composite(
                config_id=config_id,
                curves=curves,
                output_path=individual_dir / f"{config_id}.png",
            )
        )

    at_104_by_link: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    for config_id, curves in screen_grouped.items():
        curve = curves[104.0]
        link = str(curve["output_link_id"].iloc[0])
        at_104_by_link[link][config_id] = curve
    summary_figures = []
    for link in ("identity", "tanh_bounds"):
        if at_104_by_link.get(link):
            summary_figures.append(
                _render_contact_sheet(
                    curves=at_104_by_link[link],
                    link_id=link,
                    output_path=output / f"contact_sheet_{link}.png",
                )
            )
    summary_figures.extend(
        [
            _render_metric_bars(metrics, output / "screen_model_metric_bars.png"),
            _render_error_heatmap(metrics, output / "screen_model_error_heatmap.png"),
            _render_link_pairs(metrics, output / "identity_vs_tanh_bounds.png"),
        ]
    )

    seed_grouped: dict[int, dict[float, pd.DataFrame]] = defaultdict(dict)
    for entry in manifest["selected_config_stability"]:
        seed_grouped[int(entry["seed"])][float(entry["radius_mm"])] = _load_prediction(entry)
    selected_seed_figure = _render_seed_comparison(
        curves=seed_grouped,
        output_path=output / "selected_config_seed_comparison.png",
    )
    summary_figures.append(selected_seed_figure)

    best_seed = int(evaluation["best_visual_seed"])
    unsupported_curves = {
        float(entry["radius_mm"]): _load_prediction(entry)
        for entry in manifest["selected_config_unsupported"]
        if int(entry["seed"]) == best_seed
    }
    unsupported_figure = _render_unsupported(
        curves=unsupported_curves,
        seed=best_seed,
        output_path=output / "selected_config_unsupported_extrapolation.png",
    )
    summary_figures.append(unsupported_figure)

    all_figures = [*individual_records, *summary_figures]
    prediction_records = [
        *manifest.get("screen_models", []),
        *manifest.get("selected_config_stability", []),
        *manifest.get("selected_config_unsupported", []),
    ]
    checks = {
        "evaluation_is_diagnostic_only": bool(evaluation.get("diagnostic_only", False)),
        "evaluation_has_no_formal_claim": not bool(
            evaluation.get("formal_claims_allowed", True)
        ),
        "prediction_manifest_is_diagnostic_only": bool(
            manifest.get("diagnostic_only", False)
            and not manifest.get("formal_claims_allowed", True)
            and not manifest.get("changes_v7_formal_gate", True)
            and manifest.get("static_inverse_claim_radius_mm") is None
        ),
        "prediction_records_keep_claim_boundary": bool(
            prediction_records
            and all(
                record.get("diagnostic_only") is True
                and record.get("formal_claims_allowed") is False
                and record.get("changes_v7_formal_gate") is False
                and record.get("static_inverse_claim_radius_mm") is None
                for record in prediction_records
            )
        ),
        "individual_model_count_matches_evaluation": len(individual_records)
        == int(evaluation["screen_model_count"]),
        "every_model_has_three_display_radii": all(
            record["radii_mm"] == list(training.DISPLAY_RADII_MM)
            for record in individual_records
        ),
        "dynamic_camera_policy": all(
            record["view_policy"] == "svd_plane_normal_plus_12deg"
            for record in individual_records
        ),
        "unsupported_never_has_beta_truth": unsupported_figure[
            "beta_truth_available"
        ]
        is False,
        "all_pngs_written_and_hashed": all(
            Path(record["png_path"]).is_file()
            and sha256_file(record["png_path"]) == record["png_sha256"]
            for record in all_figures
        ),
    }
    report = {
        "visualization_schema_version": 1,
        **training.diagnostic_claims(),
        "visualization_only": True,
        "view_policy": "svd_plane_normal_plus_12deg",
        "evaluation_report_path": str(evaluation_path.resolve()),
        "evaluation_report_sha256": sha256_file(evaluation_path),
        "prediction_manifest_path": str(manifest_path.resolve()),
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "individual_models": individual_records,
        "summary_figures": summary_figures,
        "best_visual_seed": best_seed,
        "checks": checks,
        "visualization_integrity_gate_pass": bool(all(checks.values())),
    }
    write_json(output / "visualization_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render V7 diagnostic or formal model trajectory figures."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = Path(args.run_dir)
    if (run / "03_final_models" / "final_training_report.json").is_file():
        report = render_formal_training_trajectory_suite(
            run_dir=run, out_dir=args.out_dir
        )
    else:
        report = render_model_trajectory_suite(run_dir=run, out_dir=args.out_dir)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
