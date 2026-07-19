#!/usr/bin/env python3
"""Render auditable trajectory views for the frozen V7 radial experiment.

The figures are deliberately post-processing only.  They separate registered
strict checkpoints, an intermediate continuous pass, failed continuation
attempts, and exploratory pointwise IK so no plot can silently promote
exploratory evidence into a formal geometry or model claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
SRC_DIR = REPO_ROOT / "src"
for candidate in (str(SRC_DIR), str(ANALYSIS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
import true_ellipse_atlas_utils as atlas  # noqa: E402
import true_ellipse_radial_bundle_engine as engine  # noqa: E402


DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "true_ellipse_standard_domain_v7"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "05_visualization"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robot_rods_only_standard_100k.yaml"

STRICT_COLOR = "#1f77b4"
CONTINUOUS_COLOR = "#2ca02c"
FAILED_COLOR = "#d62728"
TARGET_COLOR = "#172b4d"
ACHIEVED_COLOR = "#f28e2b"
EXPLORATORY_COLOR = "#9467bd"
SUCCESS_COLOR = "#2ca02c"

FIGURE_IDS = (
    "strict_tracking_overview",
    "strict_joint_profiles",
    "conditioning_frontier",
    "exploratory_pointwise_overview",
)
FIGURE_DPI = 180
TRAJECTORY_VIEW_OFFSET_DEG = 12.0
STRICT_OVERVIEW_PANELS = ("trajectory_3d", "xy", "yz", "axis_errors")
CONDITIONING_LOG_SCALE = True


class EvidenceClass(StrEnum):
    REGISTERED_STRICT_CHECKPOINT = "registered_strict_checkpoint"
    CONTINUOUS_PASS = "continuous_pass"
    FAILED_CONTINUATION = "failed_continuation"
    EXPLORATORY_POINTWISE = "exploratory_pointwise"


@dataclass(frozen=True)
class TrajectoryEvidence:
    radius_mm: float
    evidence_class: EvidenceClass
    path: Path
    report_path: Path | None
    schedule: str | None = None
    strategy: str | None = None
    kappa_p95: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))

    @property
    def is_continuous_path_pass(self) -> bool:
        return self.evidence_class in {
            EvidenceClass.REGISTERED_STRICT_CHECKPOINT,
            EvidenceClass.CONTINUOUS_PASS,
        }


@dataclass(frozen=True)
class TrajectoryInventory:
    run_dir: Path
    radial_report_path: Path
    tube_report_path: Path
    protocol_fingerprint: str
    radial_task_fingerprint: str
    joint_domain_id: str
    family_id: str
    radial_strict_rmax_mm: float
    strict_geometry_rmax_mm: float | None
    last_continuous_pass_mm: float
    first_strict_failure_mm: float
    exploratory_rescue_rmax_mm: float
    formal_radial_gate_pass: bool
    formal_tube_gate_pass: bool
    strict: tuple[TrajectoryEvidence, ...]
    failed: tuple[TrajectoryEvidence, ...]
    exploratory: tuple[TrajectoryEvidence, ...]


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float_csv(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(float(part) for part in parts)
    return tuple(float(item) for item in value)


def radius_tag(radius_mm: float) -> str:
    return f"r{float(radius_mm):06.2f}".replace(".", "p")


def _require_file(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_file():
        raise FileNotFoundError(f"missing visualization input: {output}")
    return output.resolve()


def _registered_path(value: str | Path | None, fallback: Path) -> Path:
    if value:
        registered = Path(value)
        if registered.is_file():
            return registered.resolve()
    return _require_file(fallback)


def _last_continuous_pass(report: dict[str, Any]) -> float:
    radii = [
        float(attempt["target_radius_mm"])
        for attempt in report.get("attempts", [])
        if bool(attempt.get("radial_bundle_gate_pass", False))
        and attempt.get("target_radius_mm") is not None
    ]
    if not radii:
        raise ValueError("radial report contains no passing continuous path")
    return float(max(radii))


def _formal_checkpoint_radii(report: dict[str, Any]) -> tuple[float, ...]:
    values = report.get("protocol", {}).get("formal_checkpoints_mm")
    if not isinstance(values, list) or not values:
        raise ValueError("radial report omits registered formal checkpoints")
    radii = tuple(float(value) for value in values)
    if not all(np.isfinite(radius) for radius in radii):
        raise ValueError("radial report contains a non-finite formal checkpoint")
    return radii


def _best_failed_direct_attempt(job_report: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        attempt
        for attempt in job_report.get("attempts", [])
        if str(attempt.get("strategy", "")) == "direct_joint_corrector"
        and not bool(attempt.get("centerline_gate_pass", False))
        and np.isfinite(float(attempt.get("kappa_p95", np.inf)))
    ]
    if not candidates:
        raise ValueError("failed continuation job has no auditable direct attempt")
    return min(
        candidates,
        key=lambda item: (
            float(item.get("kappa_p95", np.inf)),
            float(item.get("residual_p95_mm", np.inf)),
            str(item.get("anchor_schedule", "")),
        ),
    )


def discover_trajectory_evidence(
    run_dir: str | Path,
    *,
    strict_radii: Sequence[float],
    failed_radii: Sequence[float],
    exploratory_radii: Sequence[float],
) -> TrajectoryInventory:
    root = Path(run_dir).resolve()
    radial_dir = root / "01_radial"
    radial_report_path = _require_file(radial_dir / "radial_report.json")
    tube_report_path = _require_file(root / "02_tube" / "tube_report.json")
    radial_report = read_json(radial_report_path)
    tube_report = read_json(tube_report_path)

    radial_strict_rmax = float(
        tube_report.get(
            "radial_strict_rmax_mm", radial_report["strict_geometry_rmax_mm"]
        )
    )
    strict_geometry_value = tube_report.get("strict_geometry_rmax_mm")
    strict_geometry_rmax = (
        None if strict_geometry_value is None else float(strict_geometry_value)
    )
    continuous_rmax = _last_continuous_pass(radial_report)
    formal_checkpoints = _formal_checkpoint_radii(radial_report)
    first_failure = float(radial_report["first_strict_failure_mm"])
    exploratory_rmax = float(radial_report["exploratory_rescue_rmax_mm"])

    strict: list[TrajectoryEvidence] = []
    for radius in parse_float_csv(strict_radii):
        if radius > continuous_rmax + 1.0e-9:
            raise ValueError(
                "strict display radius exceeds continuous pass frontier: "
                f"{radius:g} > {continuous_rmax:g} mm"
            )
        radius_dir = radial_dir / radius_tag(radius)
        path = _require_file(radius_dir / "selected_centerline_360.parquet")
        per_radius_report = radius_dir / "radius_bundle_report.json"
        is_registered_checkpoint = radius <= radial_strict_rmax + 1.0e-9 and any(
            np.isclose(radius, checkpoint, atol=1.0e-8, rtol=0.0)
            for checkpoint in formal_checkpoints
        )
        evidence_class = (
            EvidenceClass.REGISTERED_STRICT_CHECKPOINT
            if is_registered_checkpoint
            else EvidenceClass.CONTINUOUS_PASS
        )
        strict.append(
            TrajectoryEvidence(
                radius_mm=float(radius),
                evidence_class=evidence_class,
                path=path,
                report_path=per_radius_report.resolve() if per_radius_report.is_file() else None,
            )
        )

    failed: list[TrajectoryEvidence] = []
    for radius in parse_float_csv(failed_radii):
        if radius <= continuous_rmax + 1.0e-9:
            raise ValueError(
                "failed display radius does not lie beyond continuous pass frontier: "
                f"{radius:g} <= {continuous_rmax:g} mm"
            )
        job_dir = radial_dir / radius_tag(radius) / "jobs" / "parent_copy" / "cut_000"
        job_report_path = _require_file(job_dir / "job_report.json")
        job_report = read_json(job_report_path)
        if bool(job_report.get("centerline_gate_pass", False)) or bool(
            job_report.get("job_gate_pass", False)
        ):
            raise ValueError(f"registered failed trajectory unexpectedly passes at {radius:g} mm")
        attempt = _best_failed_direct_attempt(job_report)
        schedule = str(attempt["anchor_schedule"])
        failed.append(
            TrajectoryEvidence(
                radius_mm=float(radius),
                evidence_class=EvidenceClass.FAILED_CONTINUATION,
                path=_registered_path(attempt.get("path"), job_dir / f"{schedule}.parquet"),
                report_path=_registered_path(
                    attempt.get("report_path"),
                    job_dir / f"{schedule}.json",
                ),
                schedule=schedule,
                strategy=str(attempt.get("strategy", "direct_joint_corrector")),
                kappa_p95=float(attempt["kappa_p95"]),
            )
        )

    exploratory: list[TrajectoryEvidence] = []
    for radius in parse_float_csv(exploratory_radii):
        if radius > exploratory_rmax + 1.0e-9:
            raise ValueError(
                "exploratory display radius exceeds registered admission frontier: "
                f"{radius:g} > {exploratory_rmax:g} mm"
            )
        stem = radial_dir / "exploratory_pointwise" / radius_tag(radius)
        report_path = _require_file(stem.with_suffix(".json"))
        point_report = read_json(report_path)
        if not bool(point_report.get("rescue_admission_pass", False)):
            raise ValueError(f"exploratory pointwise evidence is not admitted at {radius:g} mm")
        exploratory.append(
            TrajectoryEvidence(
                radius_mm=float(radius),
                evidence_class=EvidenceClass.EXPLORATORY_POINTWISE,
                path=_require_file(stem.with_suffix(".parquet")),
                report_path=report_path,
            )
        )

    return TrajectoryInventory(
        run_dir=root,
        radial_report_path=radial_report_path,
        tube_report_path=tube_report_path,
        protocol_fingerprint=str(radial_report.get("protocol_fingerprint", "")),
        radial_task_fingerprint=str(radial_report.get("task_fingerprint", "")),
        joint_domain_id=str(radial_report.get("joint_domain_id", "")),
        family_id=str(radial_report.get("family_id", "")),
        radial_strict_rmax_mm=radial_strict_rmax,
        strict_geometry_rmax_mm=strict_geometry_rmax,
        last_continuous_pass_mm=continuous_rmax,
        first_strict_failure_mm=first_failure,
        exploratory_rescue_rmax_mm=exploratory_rmax,
        formal_radial_gate_pass=bool(radial_report.get("formal_radial_gate_pass", False)),
        formal_tube_gate_pass=bool(tube_report.get("formal_tube_gate_pass", False)),
        strict=tuple(strict),
        failed=tuple(failed),
        exploratory=tuple(exploratory),
    )


def prepare_curve(
    frame: pd.DataFrame,
    *,
    fk_from_beta: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    required = {
        "angle_rad",
        "x_target_m",
        "y_target_m",
        "z_target_m",
        *atlas.BETA_COLS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trajectory visualization input missing columns: {missing}")
    if frame.empty:
        raise ValueError("trajectory visualization input is empty")
    sort_columns = [column for column in ("angle_idx", "angle_rad") if column in frame]
    output = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True).copy()
    target = output[["x_target_m", "y_target_m", "z_target_m"]].to_numpy(dtype=float)
    achieved_columns = ["x_m", "y_m", "z_m"]
    if not all(column in output for column in achieved_columns):
        if fk_from_beta is None:
            raise ValueError("trajectory input has no achieved xyz and no FK callback")
        achieved = np.asarray(
            fk_from_beta(output[atlas.BETA_COLS].to_numpy(dtype=float)),
            dtype=float,
        )
        if achieved.shape != target.shape:
            raise ValueError("FK callback must return achieved xyz with shape [N, 3]")
        output[achieved_columns] = achieved
    achieved = output[achieved_columns].to_numpy(dtype=float)
    if not np.isfinite(target).all() or not np.isfinite(achieved).all():
        raise ValueError("trajectory target/achieved xyz must be finite")
    error_mm = (achieved - target) * 1000.0
    output["angle_deg"] = np.mod(np.rad2deg(output["angle_rad"].to_numpy(dtype=float)), 360.0)
    output["x_error_mm"] = error_mm[:, 0]
    output["y_error_mm"] = error_mm[:, 1]
    output["z_error_mm"] = error_mm[:, 2]
    output["xyz_residual_recomputed_mm"] = np.linalg.norm(error_mm, axis=1)
    return output


def conditioning_summary(curve: pd.DataFrame) -> dict[str, float]:
    required = {"xyz_residual_recomputed_mm", "kappa", "sigma3_m"}
    missing = sorted(required - set(curve.columns))
    if missing:
        raise ValueError(f"conditioning summary missing columns: {missing}")
    residual = curve["xyz_residual_recomputed_mm"].to_numpy(dtype=float)
    kappa = curve["kappa"].to_numpy(dtype=float)
    sigma3 = curve["sigma3_m"].to_numpy(dtype=float)
    return {
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        "kappa_p95": float(np.percentile(kappa, 95)),
        "kappa_max": float(np.max(kappa)),
        "sigma3_p05_m": float(np.percentile(sigma3, 5)),
        "sigma3_min_m": float(np.min(sigma3)),
    }


def _closed(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if len(array) == 0:
        return array
    return np.concatenate([array, array[:1]], axis=0)


def _set_3d_equal(ax: Any, xyz: np.ndarray) -> None:
    values = np.asarray(xyz, dtype=float).reshape(-1, 3)
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    center = 0.5 * (low + high)
    half = 0.5 * float(np.max(np.maximum(high - low, 1.0e-6)))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def camera_direction_from_view(view: dict[str, float]) -> np.ndarray:
    elev = math.radians(float(view["elev"]))
    azim = math.radians(float(view["azim"]))
    return np.asarray(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ],
        dtype=float,
    )


def trajectory_view_from_xyz(
    xyz_m: np.ndarray,
    *,
    azimuth_offset_deg: float = TRAJECTORY_VIEW_OFFSET_DEG,
) -> dict[str, float]:
    """Choose a near face-on view from the rank-two trajectory plane."""
    xyz = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    if len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("trajectory view requires at least three finite xyz points")
    centered = xyz - np.mean(xyz, axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= max(singular[0], 1.0) * 1.0e-10:
        raise ValueError("trajectory view requires a non-degenerate rank-two curve")
    normal = np.asarray(vh[-1], dtype=float)
    if normal[2] < 0.0:
        normal = -normal
    elev = math.degrees(math.asin(float(np.clip(normal[2], -1.0, 1.0))))
    azim = math.degrees(math.atan2(float(normal[1]), float(normal[0])))
    return {
        "elev": float(elev),
        "azim": float(azim + azimuth_offset_deg),
    }


def _plot_trajectory_3d(
    ax: Any,
    curve: pd.DataFrame,
    *,
    achieved_line: bool,
    point_colors: np.ndarray | None = None,
) -> None:
    target = curve[["x_target_m", "y_target_m", "z_target_m"]].to_numpy(dtype=float)
    achieved = curve[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    ax.plot(*_closed(target).T, color=TARGET_COLOR, lw=2.4, ls="--", label="target")
    if achieved_line:
        ax.plot(*_closed(achieved).T, color=ACHIEVED_COLOR, lw=1.6, label="IK + FK")
        stride = max(1, len(curve) // 18)
        ax.scatter(*achieved[::stride].T, color=ACHIEVED_COLOR, s=12, alpha=0.8)
    else:
        colors = point_colors if point_colors is not None else EXPLORATORY_COLOR
        ax.scatter(*achieved.T, c=colors, s=20, alpha=0.85, depthshade=False)
    _set_3d_equal(ax, np.vstack([target, achieved]))
    ax.view_init(**trajectory_view_from_xyz(target))
    ax.set_xlabel("x (m)", labelpad=2)
    ax.set_ylabel("y (m)", labelpad=2)
    ax.set_zlabel("z (m)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)


def _save_figure(fig: plt.Figure, out_dir: Path, figure_id: str) -> dict[str, Any]:
    png = out_dir / f"{figure_id}.png"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "png_path": str(png.resolve()),
        "png_sha256": sha256_file(png),
    }


def _plot_strict_tracking(
    rows: Sequence[tuple[TrajectoryEvidence, pd.DataFrame]],
    out_dir: Path,
) -> dict[str, Any]:
    nrows = len(rows)
    ncols = len(STRICT_OVERVIEW_PANELS)
    fig = plt.figure(figsize=(20.0, 4.8 * nrows))
    for row_idx, (item, curve) in enumerate(rows):
        ax3d = fig.add_subplot(nrows, ncols, row_idx * ncols + 1, projection="3d")
        _plot_trajectory_3d(ax3d, curve, achieved_line=True)
        role = (
            "registered strict checkpoint"
            if item.evidence_class == "registered_strict_checkpoint"
            else "continuous pass; not a registered checkpoint"
        )
        ax3d.set_title(f"{item.radius_mm:g} mm — {role}", fontsize=10)
        if row_idx == 0:
            ax3d.legend(frameon=False, fontsize=8, loc="upper left")

        for offset, (left, right, label) in enumerate(
            (("x", "y", "x-y projection"), ("y", "z", "y-z projection")),
            start=2,
        ):
            ax = fig.add_subplot(nrows, ncols, row_idx * ncols + offset)
            ax.plot(
                _closed(curve[f"{left}_target_m"].to_numpy(dtype=float)),
                _closed(curve[f"{right}_target_m"].to_numpy(dtype=float)),
                color=TARGET_COLOR,
                lw=2.4,
                ls="--",
                label="target",
            )
            ax.plot(
                _closed(curve[f"{left}_m"].to_numpy(dtype=float)),
                _closed(curve[f"{right}_m"].to_numpy(dtype=float)),
                color=ACHIEVED_COLOR,
                lw=1.5,
                label="IK + FK",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"{left} (m)")
            ax.set_ylabel(f"{right} (m)")
            ax.grid(True, alpha=0.22)
            ax.set_title(label, fontsize=10)

        ax_error = fig.add_subplot(nrows, ncols, row_idx * ncols + 4)
        for column, label, color in (
            ("x_error_mm", "x error", "#4c78a8"),
            ("y_error_mm", "y error", "#f58518"),
            ("z_error_mm", "z error", "#54a24b"),
        ):
            ax_error.plot(curve["angle_deg"], curve[column], lw=1.2, label=label, color=color)
        ax_error.axhline(0.0, color="#333333", lw=0.8, alpha=0.6)
        ax_error.axhspan(-2.0, 2.0, color="#54a24b", alpha=0.06, linewidth=0)
        residual = curve["xyz_residual_recomputed_mm"].to_numpy(dtype=float)
        ax_error.text(
            0.02,
            0.96,
            f"residual P95={np.percentile(residual, 95):.4f} mm\nmax={np.max(residual):.4f} mm",
            transform=ax_error.transAxes,
            va="top",
            fontsize=8,
        )
        ax_error.set_xlim(0.0, 360.0)
        ax_error.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
        ax_error.set_xlabel("ellipse phase (deg)")
        ax_error.set_ylabel("axis error (mm)")
        ax_error.grid(True, alpha=0.22)
        ax_error.set_title("axis errors", fontsize=10)
        if row_idx == 0:
            ax_error.legend(frameon=False, ncol=3, fontsize=7, loc="lower left")
    fig.suptitle(
        "V7 strict/continuous true-ellipse tracking (target vs solved branch)",
        fontsize=15,
    )
    return _save_figure(fig, out_dir, "strict_tracking_overview")


def _plot_strict_joint_profiles(
    rows: Sequence[tuple[TrajectoryEvidence, pd.DataFrame]],
    *,
    joint_domain_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    domain = engine.registered_joint_domain(joint_domain_id)
    colors = [STRICT_COLOR, "#17becf", CONTINUOUS_COLOR, "#8c564b"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True)
    for joint_idx, ax in enumerate(axes.reshape(-1)):
        lower, upper = domain.bounds_deg[joint_idx]
        ax.axhspan(lower, upper, color="#d9e8f5", alpha=0.35, label="registered domain")
        ax.axhline(lower, color="#777777", lw=0.8, ls="--")
        ax.axhline(upper, color="#777777", lw=0.8, ls="--")
        for curve_idx, (item, curve) in enumerate(rows):
            beta_deg = np.rad2deg(curve[f"beta{joint_idx + 1}_rad"].to_numpy(dtype=float))
            style = "-" if item.evidence_class == "registered_strict_checkpoint" else "-."
            ax.plot(
                curve["angle_deg"],
                beta_deg,
                color=colors[curve_idx % len(colors)],
                lw=1.5,
                ls=style,
                label=f"{item.radius_mm:g} mm",
            )
        ax.set_title(f"beta{joint_idx + 1}  [{lower:g}°, {upper:g}°]")
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
        ax.set_xlabel("phase (deg)")
        ax.set_ylabel("joint angle (deg)")
        ax.grid(True, alpha=0.22)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(
        "V7 solved branch joint profiles — standard domain remains comfortably interior",
        fontsize=15,
    )
    return _save_figure(fig, out_dir, "strict_joint_profiles")


def select_conditioning_cases(
    strict_rows: Sequence[tuple[TrajectoryEvidence, pd.DataFrame]],
    failed_rows: Sequence[tuple[TrajectoryEvidence, pd.DataFrame]],
) -> list[tuple[TrajectoryEvidence, pd.DataFrame]]:
    if not strict_rows:
        raise ValueError("conditioning plot requires a passing continuous trajectory")
    frontier_pass = max(strict_rows, key=lambda row: row[0].radius_mm)
    return [frontier_pass, *failed_rows]


def _plot_conditioning_frontier(
    strict_rows: Sequence[tuple[TrajectoryEvidence, pd.DataFrame]],
    failed_rows: Sequence[tuple[TrajectoryEvidence, pd.DataFrame]],
    out_dir: Path,
) -> dict[str, Any]:
    cases = select_conditioning_cases(strict_rows, failed_rows)
    fig, axes = plt.subplots(3, len(cases), figsize=(5.2 * len(cases), 10.5))
    if len(cases) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for col, (item, curve) in enumerate(cases):
        color = CONTINUOUS_COLOR if item.is_continuous_path_pass else FAILED_COLOR
        label = "PASS" if item.is_continuous_path_pass else "FAIL conditioning"
        summary = conditioning_summary(curve)
        axes[0, col].plot(
            curve["angle_deg"],
            curve["xyz_residual_recomputed_mm"],
            color=color,
            lw=1.5,
        )
        axes[0, col].axhline(2.0, color="#555555", ls="--", lw=1.0, label="residual P95 gate 2 mm")
        axes[0, col].set_title(f"{item.radius_mm:g} mm — {label}")
        axes[0, col].set_ylabel("Cartesian residual (mm)")
        axes[0, col].legend(frameon=False, fontsize=7)
        axes[0, col].text(
            0.02,
            0.94,
            f"P95={summary['residual_p95_mm']:.4f} mm\nmax={summary['residual_max_mm']:.4f} mm",
            transform=axes[0, col].transAxes,
            va="top",
            fontsize=8,
        )

        axes[1, col].plot(
            curve["angle_deg"],
            np.maximum(curve["kappa"].to_numpy(dtype=float), 1.0e-9),
            color=color,
            lw=1.5,
        )
        axes[1, col].axhline(150.0, color="#111111", ls="--", lw=1.2, label="P95 threshold 150")
        axes[1, col].set_ylabel("Jacobian kappa")
        if CONDITIONING_LOG_SCALE:
            axes[1, col].set_yscale("log")
        axes[1, col].legend(frameon=False, fontsize=7)
        axes[1, col].text(
            0.02,
            0.94,
            f"P95={summary['kappa_p95']:.1f}\npointwise max={summary['kappa_max']:.1f}",
            transform=axes[1, col].transAxes,
            va="top",
            fontsize=8,
        )

        axes[2, col].plot(
            curve["angle_deg"],
            np.maximum(curve["sigma3_m"].to_numpy(dtype=float), 1.0e-9),
            color=color,
            lw=1.5,
        )
        axes[2, col].axhline(0.0015, color="#111111", ls="--", lw=1.2, label="P05 threshold 0.0015 m")
        axes[2, col].set_ylabel("sigma3 (m)")
        if CONDITIONING_LOG_SCALE:
            axes[2, col].set_yscale("log")
        axes[2, col].set_xlabel("phase (deg)")
        axes[2, col].legend(frameon=False, fontsize=7)
        axes[2, col].text(
            0.02,
            0.94,
            f"P05={summary['sigma3_p05_m']:.5f} m\npointwise min={summary['sigma3_min_m']:.5f} m",
            transform=axes[2, col].transAxes,
            va="top",
            fontsize=8,
        )
        for row in range(3):
            axes[row, col].set_xlim(0.0, 360.0)
            axes[row, col].set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
            axes[row, col].grid(True, alpha=0.22)
    fig.suptitle(
        "Conditioning frontier: Cartesian tracking remains close while the Jacobian becomes singular",
        fontsize=15,
    )
    return _save_figure(fig, out_dir, "conditioning_frontier")


def _plot_exploratory_pointwise(
    rows: Sequence[tuple[TrajectoryEvidence, pd.DataFrame]],
    out_dir: Path,
) -> dict[str, Any]:
    ncols = min(2, max(1, len(rows)))
    nrows = int(math.ceil(len(rows) / ncols))
    fig = plt.figure(figsize=(7.2 * ncols, 5.8 * nrows))
    for idx, (item, curve) in enumerate(rows):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        residual = curve["xyz_residual_recomputed_mm"].to_numpy(dtype=float)
        colors = np.where(residual <= 2.0, SUCCESS_COLOR, FAILED_COLOR)
        _plot_trajectory_3d(ax, curve, achieved_line=False, point_colors=colors)
        ratio = float(np.mean(residual <= 2.0))
        ax.set_title(
            f"{item.radius_mm:g} mm pointwise only\n"
            f"recomputed residual <=2 mm: {ratio:.3f}",
            fontsize=10,
        )
    fig.suptitle(
        "EXPLORATORY pointwise IK — dots are independent solutions, not a continuous branch",
        fontsize=15,
        color=EXPLORATORY_COLOR,
    )
    return _save_figure(fig, out_dir, "exploratory_pointwise_overview")


def _evidence_report(
    item: TrajectoryEvidence,
    curve: pd.DataFrame,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    stored = (
        curve["xyz_residual_mm"].to_numpy(dtype=float)
        if "xyz_residual_mm" in curve
        else np.asarray([], dtype=float)
    )
    recomputed = curve["xyz_residual_recomputed_mm"].to_numpy(dtype=float)
    payload: dict[str, Any] = {
        "radius_mm": float(item.radius_mm),
        "evidence_class": item.evidence_class,
        "source_path": str(item.path),
        "source_path_relative_to_run": (
            str(item.path.relative_to(run_dir)) if item.path.is_relative_to(run_dir) else None
        ),
        "source_sha256": sha256_file(item.path),
        "report_path": str(item.report_path) if item.report_path else None,
        "report_sha256": sha256_file(item.report_path) if item.report_path else None,
        "rows": int(len(curve)),
        "schedule": item.schedule,
        "strategy": item.strategy,
        "residual_recomputed_p95_mm": float(np.percentile(recomputed, 95)),
        "residual_recomputed_max_mm": float(np.max(recomputed)),
        "stored_vs_recomputed_residual_max_abs_mm": (
            float(np.max(np.abs(stored - recomputed))) if len(stored) == len(recomputed) else None
        ),
        "kappa_p95": (
            float(np.percentile(curve["kappa"].to_numpy(dtype=float), 95))
            if "kappa" in curve
            else item.kappa_p95
        ),
        "sigma3_p05_m": (
            float(np.percentile(curve["sigma3_m"].to_numpy(dtype=float), 5))
            if "sigma3_m" in curve
            else None
        ),
    }
    return payload


def render_v7_trajectory_visualizations(
    *,
    run_dir: str | Path,
    out_dir: str | Path,
    strict_radii: Sequence[float],
    failed_radii: Sequence[float],
    exploratory_radii: Sequence[float],
    exploratory_fk: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    inventory = discover_trajectory_evidence(
        run_dir,
        strict_radii=strict_radii,
        failed_radii=failed_radii,
        exploratory_radii=exploratory_radii,
    )
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    strict_rows = [
        (item, prepare_curve(pd.read_parquet(item.path))) for item in inventory.strict
    ]
    failed_rows = [
        (item, prepare_curve(pd.read_parquet(item.path))) for item in inventory.failed
    ]
    exploratory_rows = [
        (
            item,
            prepare_curve(pd.read_parquet(item.path), fk_from_beta=exploratory_fk),
        )
        for item in inventory.exploratory
    ]

    figures = [
        _plot_strict_tracking(strict_rows, output),
        _plot_strict_joint_profiles(
            strict_rows,
            joint_domain_id=inventory.joint_domain_id,
            out_dir=output,
        ),
        _plot_conditioning_frontier(strict_rows, failed_rows, output),
        _plot_exploratory_pointwise(exploratory_rows, output),
    ]

    evidence_rows = [
        *(
            _evidence_report(item, curve, run_dir=inventory.run_dir)
            for item, curve in strict_rows
        ),
        *(
            _evidence_report(item, curve, run_dir=inventory.run_dir)
            for item, curve in failed_rows
        ),
        *(
            _evidence_report(item, curve, run_dir=inventory.run_dir)
            for item, curve in exploratory_rows
        ),
    ]
    source_paths = [str(row["source_path"]) for row in evidence_rows]
    evidence_classes = {str(row["evidence_class"]) for row in evidence_rows}
    report: dict[str, Any] = {
        "visualization_schema_version": 1,
        "visualization_only": True,
        "changes_formal_gate": False,
        "strict_model_claim_made": False,
        "protocol_fingerprint": inventory.protocol_fingerprint,
        "radial_task_fingerprint": inventory.radial_task_fingerprint,
        "joint_domain_id": inventory.joint_domain_id,
        "family_id": inventory.family_id,
        "radial_strict_rmax_mm": inventory.radial_strict_rmax_mm,
        "strict_geometry_rmax_mm": inventory.strict_geometry_rmax_mm,
        "last_continuous_pass_mm": inventory.last_continuous_pass_mm,
        "first_strict_failure_mm": inventory.first_strict_failure_mm,
        "exploratory_rescue_rmax_mm": inventory.exploratory_rescue_rmax_mm,
        "formal_radial_gate_pass": inventory.formal_radial_gate_pass,
        "formal_tube_gate_pass": inventory.formal_tube_gate_pass,
        "radial_report_path": str(inventory.radial_report_path),
        "radial_report_sha256": sha256_file(inventory.radial_report_path),
        "tube_report_path": str(inventory.tube_report_path),
        "tube_report_sha256": sha256_file(inventory.tube_report_path),
        "evidence": evidence_rows,
        "figures": figures,
        "checks": {
            "evidence_classes_disjoint": bool(
                evidence_classes
                == {
                    "registered_strict_checkpoint",
                    "continuous_pass",
                    "failed_continuation",
                    "exploratory_pointwise",
                }
                and len(source_paths) == len(set(source_paths))
            ),
            "strict_radii_do_not_exceed_continuous_frontier": bool(
                all(
                    item.radius_mm <= inventory.last_continuous_pass_mm + 1.0e-9
                    for item in inventory.strict
                )
            ),
            "failed_radii_exceed_continuous_frontier": bool(
                all(
                    item.radius_mm > inventory.last_continuous_pass_mm + 1.0e-9
                    for item in inventory.failed
                )
            ),
            "exploratory_never_labeled_strict": bool(
                all(
                    item.evidence_class == "exploratory_pointwise"
                    for item in inventory.exploratory
                )
            ),
            "no_geometry_claim_without_tube_gate": bool(
                inventory.formal_tube_gate_pass
                or inventory.strict_geometry_rmax_mm is None
            ),
            "all_input_hashes_present": bool(
                all(bool(row["source_sha256"]) for row in evidence_rows)
            ),
            "all_figures_written": bool(
                len(figures) == len(FIGURE_IDS)
                and {item["figure_id"] for item in figures} == set(FIGURE_IDS)
                and all(
                    Path(item[path_key]).is_file()
                    for item in figures
                    for path_key in ("png_path",)
                )
            ),
        },
    }
    report["visualization_integrity_gate_pass"] = bool(all(report["checks"].values()))
    write_json(output / "visualization_report.json", report)
    return report


def _robot_fk(robot_config: Path) -> Callable[[np.ndarray], np.ndarray]:
    config = load_config(robot_config)
    inputs = load_robot_inputs(config)
    theta_sign = float(config.get("kinematics", {}).get("theta_sign", -1.0))

    def evaluate(beta_rad: np.ndarray) -> np.ndarray:
        return atlas.fk_from_beta_batch(
            beta_rad,
            lengths_m=inputs.lengths_m,
            p_end_local_m=inputs.p_end_local_m,
            theta_sign=theta_sign,
        )

    return evaluate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render non-gating V7 strict, failed-frontier, and exploratory "
            "true-ellipse trajectory visualizations."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--strict-radii-mm", default="100,102.5,104")
    parser.add_argument("--failed-radii-mm", default="104.25,105")
    parser.add_argument("--exploratory-radii-mm", default="105,110,115,120")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = render_v7_trajectory_visualizations(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        strict_radii=parse_float_csv(args.strict_radii_mm),
        failed_radii=parse_float_csv(args.failed_radii_mm),
        exploratory_radii=parse_float_csv(args.exploratory_radii_mm),
        exploratory_fk=_robot_fk(args.robot_config),
    )
    print(
        json.dumps(
            {
                "out_dir": str(Path(args.out_dir).resolve()),
                "figure_count": len(report["figures"]),
                "visualization_integrity_gate_pass": report[
                    "visualization_integrity_gate_pass"
                ],
                "changes_formal_gate": report["changes_formal_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
