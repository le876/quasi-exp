#!/usr/bin/env python3
"""Visualize authoritative teacher tracking for the V10 meter-scale ellipses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHARED_PROJECT_ROOT = (
    REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT
)
DEFAULT_INPUT_ROOT = (
    SHARED_PROJECT_ROOT
    / "runs"
    / "trajectory_canonical_teacher_v10"
    / "07_large_scale_challenge"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "plots"
FIGURE_DPI = 220
TARGET_COLOR = "#202020"
VARIANT_COLORS = {"T1": "#f28e2b", "T3": "#4e79a7", "T4": "#59a14f"}


@dataclass(frozen=True)
class ScaleSpec:
    slug: str
    evidence_tier: str
    variants: tuple[str, ...]
    phase_count: int


@dataclass(frozen=True)
class VariantEvidence:
    curve: pd.DataFrame
    target: np.ndarray
    achieved: np.ndarray
    residual_mm: np.ndarray
    phase_deg: np.ndarray
    p95_mm: float
    max_mm: float
    report: dict
    curve_path: Path
    report_path: Path


@dataclass(frozen=True)
class LoadedScale:
    spec: ScaleSpec
    pose: dict
    center: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    target: np.ndarray
    variants: dict[str, VariantEvidence]
    input_paths: tuple[Path, ...]


SCALE_SPECS = (
    ScaleSpec("a0p500m", "formal", ("T3", "T4"), 180),
    ScaleSpec("a0p750m", "screen", ("T1", "T3", "T4"), 24),
    ScaleSpec("a1p000m", "screen", ("T1", "T3", "T4"), 24),
)


def _unit_direction(values: Sequence[float], *, name: str) -> np.ndarray:
    direction = np.asarray(values, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or norm <= 1.0e-12:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return direction / norm


def project_to_ellipse_plane(
    xyz_m: np.ndarray,
    *,
    center_m: Sequence[float],
    major_direction: Sequence[float],
    minor_direction: Sequence[float],
) -> np.ndarray:
    """Project world xyz into the fitted ellipse's intrinsic major/minor frame."""

    points = np.asarray(xyz_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("xyz_m must be a finite N×3 array")
    center = np.asarray(center_m, dtype=float).reshape(3)
    major = _unit_direction(major_direction, name="major_direction")
    minor = _unit_direction(minor_direction, name="minor_direction")
    if abs(float(np.dot(major, minor))) > 1.0e-6:
        raise ValueError("ellipse major and minor directions must be orthogonal")
    centered = points - center[None, :]
    return np.column_stack([centered @ major, centered @ minor])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _xyz(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = [f"{prefix}_{axis}_m" for axis in "xyz"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"missing trajectory columns: {missing}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{prefix} trajectory contains non-finite xyz")
    return values


def _closed(values: np.ndarray) -> np.ndarray:
    return np.vstack([values, values[0]])


def _set_3d_equal(ax: object, values: np.ndarray) -> None:
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1.0e-3)
    half = 0.56 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _camera_from_pose(pose: dict) -> tuple[float, float]:
    major = _unit_direction(pose["major_direction"], name="major_direction")
    minor = _unit_direction(pose["minor_direction"], name="minor_direction")
    normal = _unit_direction(np.cross(major, minor), name="ellipse_plane_normal")
    camera = _unit_direction(
        normal + 0.28 * major + 0.12 * minor,
        name="camera_direction",
    )
    elevation = float(np.rad2deg(np.arcsin(np.clip(camera[2], -1.0, 1.0))))
    azimuth = float(np.rad2deg(np.arctan2(camera[1], camera[0])))
    return elevation, azimuth


def _load_scale(input_root: Path, spec: ScaleSpec) -> LoadedScale:
    scale_dir = input_root / spec.slug
    pose_path = scale_dir / "pose_report.json"
    pose = _read_json(pose_path)
    center = np.asarray(pose["center_m"], dtype=float)
    major = _unit_direction(pose["major_direction"], name="major_direction")
    minor = _unit_direction(pose["minor_direction"], name="minor_direction")
    if abs(float(np.dot(major, minor))) > 1.0e-6:
        raise ValueError(f"{spec.slug}: pose axes are not orthogonal")

    variants: dict[str, VariantEvidence] = {}
    input_paths = [pose_path]
    reference_target: np.ndarray | None = None
    for variant in spec.variants:
        variant_dir = scale_dir / spec.evidence_tier / variant
        curve_path = variant_dir / "centerline.parquet"
        report_path = variant_dir / "centerline_report.json"
        curve = pd.read_parquet(curve_path).sort_values("phase_idx", kind="stable")
        curve = curve.reset_index(drop=True)
        if len(curve) != spec.phase_count:
            raise ValueError(
                f"{spec.slug}/{variant}: expected {spec.phase_count} phases, got {len(curve)}"
            )
        target = _xyz(curve, "target")
        achieved = _xyz(curve, "achieved")
        residual = np.linalg.norm(achieved - target, axis=1) * 1000.0
        if "teacher_fk_residual_mm" in curve:
            stored = curve["teacher_fk_residual_mm"].to_numpy(dtype=float)
            np.testing.assert_allclose(stored, residual, rtol=1.0e-8, atol=1.0e-8)
        if reference_target is None:
            reference_target = target
        else:
            np.testing.assert_allclose(reference_target, target, rtol=0.0, atol=1.0e-10)
        phase_deg = np.mod(np.rad2deg(curve["phase_rad"].to_numpy(dtype=float)), 360.0)
        variants[variant] = VariantEvidence(
            curve=curve,
            target=target,
            achieved=achieved,
            residual_mm=residual,
            phase_deg=phase_deg,
            p95_mm=float(np.percentile(residual, 95)),
            max_mm=float(np.max(residual)),
            report=_read_json(report_path),
            curve_path=curve_path,
            report_path=report_path,
        )
        input_paths.extend([curve_path, report_path])
    assert reference_target is not None
    target_plane = project_to_ellipse_plane(
        reference_target,
        center_m=center,
        major_direction=major,
        minor_direction=minor,
    )
    actual_semimajor = float(np.max(np.abs(target_plane[:, 0])))
    expected_semimajor = float(pose["major_semiaxis_m"])
    if not np.isclose(actual_semimajor, expected_semimajor, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"{spec.slug}: target major semiaxis {actual_semimajor} disagrees with pose "
            f"{expected_semimajor}"
        )
    return LoadedScale(
        spec=spec,
        pose=pose,
        center=center,
        major=major,
        minor=minor,
        target=reference_target,
        variants=variants,
        input_paths=tuple(input_paths),
    )


def _plot_intrinsic(
    ax: object, scale: LoadedScale, *, variants: Sequence[str]
) -> None:
    pose = scale.pose
    target_plane = project_to_ellipse_plane(
        scale.target,
        center_m=pose["center_m"],
        major_direction=pose["major_direction"],
        minor_direction=pose["minor_direction"],
    )
    ax.plot(
        *_closed(target_plane).T,
        color=TARGET_COLOR,
        lw=2.3,
        ls="--",
        label="target ellipse",
        zorder=5,
    )
    for variant in variants:
        record = scale.variants[variant]
        achieved_plane = project_to_ellipse_plane(
            record.achieved,
            center_m=pose["center_m"],
            major_direction=pose["major_direction"],
            minor_direction=pose["minor_direction"],
        )
        ax.plot(
            *_closed(achieved_plane).T,
            color=VARIANT_COLORS[variant],
            lw=1.25,
            label=f"{variant} teacher + FK",
        )
        ax.scatter(
            achieved_plane[0, 0],
            achieved_plane[0, 1],
            s=22,
            color=VARIANT_COLORS[variant],
            marker="o",
            zorder=7,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("major-axis coordinate (m)")
    ax.set_ylabel("minor-axis coordinate (m)")
    ax.grid(True, alpha=0.22)


def _plot_world_3d(ax: object, scale: LoadedScale) -> None:
    target = scale.target
    ax.plot(
        *_closed(target).T,
        color=TARGET_COLOR,
        lw=2.2,
        ls="--",
        label="target ellipse",
    )
    all_values = [target]
    for variant, record in scale.variants.items():
        achieved = record.achieved
        ax.plot(
            *_closed(achieved).T,
            color=VARIANT_COLORS[variant],
            lw=1.15,
            label=f"{variant} teacher + FK",
        )
        stride = max(1, len(achieved) // 24)
        ax.scatter(
            *achieved[::stride].T,
            color=VARIANT_COLORS[variant],
            s=8,
            alpha=0.72,
        )
        all_values.append(achieved)
    _set_3d_equal(ax, np.vstack(all_values))
    elevation, azimuth = _camera_from_pose(scale.pose)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel("x (m)", labelpad=2)
    ax.set_ylabel("y (m)", labelpad=2)
    ax.set_zlabel("z (m)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)


def _plot_error(ax: object, scale: LoadedScale) -> None:
    for variant, record in scale.variants.items():
        ax.plot(
            record.phase_deg,
            record.residual_mm,
            color=VARIANT_COLORS[variant],
            lw=1.35,
            marker="o" if len(record.phase_deg) <= 24 else None,
            ms=2.5,
            label=f"{variant} FK residual",
        )
    ax.axhline(
        1.0,
        color="#777777",
        lw=0.9,
        ls="--",
        label="1 mm P95 reference (aggregate)",
    )
    ax.axhline(3.0, color="#b23a48", lw=0.9, ls=":", label="max gate 3 mm")
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    ax.set_xlim(0.0, 360.0)
    ax.set_xticks([0.0, 90.0, 180.0, 270.0, 360.0])
    ax.set_xlabel("ellipse phase (deg)")
    ax.set_ylabel("teacher FK residual (mm, symlog)")
    ax.grid(True, which="both", alpha=0.22)


def _plot_metric_bars(ax: object, scale: LoadedScale) -> None:
    variants = list(scale.variants)
    y = np.arange(len(variants), dtype=float)
    p95 = [scale.variants[variant].p95_mm for variant in variants]
    maximum = [scale.variants[variant].max_mm for variant in variants]
    ax.barh(y - 0.17, p95, height=0.32, color="#76b7b2", label="P95")
    ax.barh(y + 0.17, maximum, height=0.32, color="#e15759", label="max")
    ax.set_yticks(y, labels=variants)
    ax.set_xscale("log")
    ax.set_xlabel("teacher FK residual (mm, log)")
    ax.grid(True, axis="x", which="both", alpha=0.22)
    for row, (p95_value, max_value) in enumerate(zip(p95, maximum, strict=True)):
        ax.text(p95_value, row - 0.17, f" {p95_value:.3f}", va="center", fontsize=7)
        ax.text(max_value, row + 0.17, f" {max_value:.3f}", va="center", fontsize=7)
    ax.invert_yaxis()


def _save_figure(fig: plt.Figure, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _render_scale(scale: LoadedScale, output_path: Path) -> dict:
    spec = scale.spec
    major_m = float(scale.pose["major_semiaxis_m"])
    fig = plt.figure(figsize=(15.2, 9.6))
    intrinsic = fig.add_subplot(2, 2, 1)
    _plot_intrinsic(intrinsic, scale, variants=spec.variants)
    intrinsic.set_title("Ellipse-plane face-on view (shape-preserving)")
    intrinsic.legend(frameon=False, fontsize=8, loc="best")

    world = fig.add_subplot(2, 2, 2, projection="3d")
    _plot_world_3d(world, scale)
    world.set_title("World xyz — plane-normal view with slight tilt")
    world.legend(frameon=False, fontsize=7, loc="upper left")

    error = fig.add_subplot(2, 2, 3)
    _plot_error(error, scale)
    error.set_title("Phase-wise end-effector tracking residual")
    error.legend(frameon=False, fontsize=7, ncol=2, loc="best")

    metrics = fig.add_subplot(2, 2, 4)
    _plot_metric_bars(metrics, scale)
    metrics.set_title("Tracking summary (lower is better)")
    metrics.legend(frameon=False, fontsize=8, loc="best")

    tier_label = "formal evidence" if spec.evidence_tier == "formal" else "coarse-screen evidence"
    fig.suptitle(
        f"V10 teacher tracking — actual major semiaxis {major_m:.2f} m\n"
        f"{tier_label}, {spec.phase_count} phases; achieved curve = authoritative teacher beta → FK",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.012,
        "Visualization only. These are solver/teacher trajectories, not neural-student model predictions; "
        "plots do not alter any formal gate decision.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.925))
    return _save_figure(fig, output_path)


def _render_overview(
    scales: Sequence[LoadedScale], output_path: Path
) -> tuple[dict, dict[str, str]]:
    fig = plt.figure(figsize=(15.0, 12.6))
    selected: dict[str, str] = {}
    for row, scale in enumerate(scales):
        spec = scale.spec
        best = min(
            spec.variants,
            key=lambda variant: scale.variants[variant].p95_mm,
        )
        selected[spec.slug] = best
        intrinsic = fig.add_subplot(len(scales), 2, row * 2 + 1)
        _plot_intrinsic(intrinsic, scale, variants=(best,))
        intrinsic.set_title(
            f"a={float(scale.pose['major_semiaxis_m']):.2f} m — target vs lowest-P95 {best}"
        )
        intrinsic.legend(frameon=False, fontsize=8, loc="best")
        error = fig.add_subplot(len(scales), 2, row * 2 + 2)
        _plot_error(error, scale)
        error.set_title(
            f"{spec.evidence_tier}, {spec.phase_count} phases — all available teachers"
        )
        error.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    fig.suptitle(
        "V10 meter-scale ellipse teacher tracking overview\n"
        "Intrinsic plane views preserve ellipse geometry; error axes use symmetric-log scaling",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.009,
        "Teacher/solver evidence only — no trained neural student was evaluated on these three trajectories.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.94))
    return _save_figure(fig, output_path), selected


def render_tracking_evidence(
    *,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Render all registered scales and write a provenance-rich plot manifest."""

    source = Path(input_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scales = [_load_scale(source, spec) for spec in SCALE_SPECS]
    figures = []
    for scale in scales:
        spec = scale.spec
        figures.append(_render_scale(scale, output / f"{spec.slug}_teacher_tracking.png"))
    overview, selected = _render_overview(
        scales,
        output / "large_scale_teacher_tracking_overview.png",
    )
    figures.append(overview)

    input_hashes = {
        str(path.resolve()): _sha256_file(path)
        for scale in scales
        for path in scale.input_paths
    }
    scale_records = {}
    for scale in scales:
        spec = scale.spec
        scale_records[spec.slug] = {
            "actual_major_semiaxis_m": float(scale.pose["major_semiaxis_m"]),
            "actual_minor_semiaxis_m": float(scale.pose["minor_semiaxis_m"]),
            "evidence_tier": spec.evidence_tier,
            "phase_count": spec.phase_count,
            "variants": {
                variant: {
                    "teacher_fk_residual_p95_mm": scale.variants[variant].p95_mm,
                    "teacher_fk_residual_max_mm": scale.variants[variant].max_mm,
                    "centerline_gate_pass": bool(
                        scale.variants[variant].report.get("centerline_gate_pass", False)
                    ),
                }
                for variant in spec.variants
            },
            "overview_lowest_p95_variant": selected[spec.slug],
        }
    manifest = {
        "protocol_id": "large-scale-teacher-tracking-visualization-v10.1",
        "tracking_source": "teacher_solver_fk_not_neural_student",
        "visualization_only": True,
        "changes_formal_gate": False,
        "input_root": str(source),
        "worker_code": str(Path(__file__).resolve()),
        "worker_code_sha256": _sha256_file(Path(__file__).resolve()),
        "input_sha256": input_hashes,
        "view_policy": {
            "intrinsic": "registered_pose_major_minor_axes_equal_aspect",
            "world_3d": "registered_plane_normal_plus_major_minor_tilt_equal_xyz_scale",
        },
        "scales": scale_records,
        "figures": figures,
    }
    _write_json(output / "plot_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = render_tracking_evidence(
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
