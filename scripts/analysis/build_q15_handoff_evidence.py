#!/usr/bin/env python3
"""Build deterministic Q15 evidence about the current dataset geometry.

This script is intentionally read-only with respect to the formal V12.15 and
V12.16 artifacts.  It writes a compact profile, a sealed-path scale audit, and
one static endpoint-distribution figure to a separate handoff-evidence root.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "runs/gpt5pro_handoff_evidence/Q15"
DATASET = (
    PROJECT_ROOT
    / "runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731"
    / "01_multichart_dataset/train_validation.parquet"
)
PATH_SOURCES = {
    "chart_A": {
        "catalog": (
            PROJECT_ROOT
            / "runs/bacra_v12_15_single_student_formal_margin_tail_retry1_20260730"
            / "07_sealed_trajectory_test/trajectory_catalog.csv"
        ),
        "reference": (
            PROJECT_ROOT
            / "runs/bacra_v12_15_single_student_formal_margin_tail_retry1_20260730"
            / "07_sealed_trajectory_test/teacher_reference.parquet"
        ),
    },
    "chart_B": {
        "catalog": (
            PROJECT_ROOT
            / "runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731"
            / "05_locked_evaluation/chart_b_path_catalog.csv"
        ),
        "reference": (
            PROJECT_ROOT
            / "runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731"
            / "05_locked_evaluation/chart_b_path_teacher_reference.parquet"
        ),
    },
}
XYZ_COLUMNS = ["x_m", "y_m", "z_m"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_key_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts = frame.groupby(columns, dropna=False, observed=True).size()
    return {
        "/".join(map(str, key if isinstance(key, tuple) else (key,))): int(value)
        for key, value in counts.items()
    }


def coordinate_profile(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    quantiles = frame[XYZ_COLUMNS].quantile(
        [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    )
    names = ["min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"]
    return {
        column: {
            name: float(quantiles.loc[level, column])
            for name, level in zip(names, quantiles.index, strict=True)
        }
        for column in XYZ_COLUMNS
    }


def block_audit(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for chart_id, chart in frame.groupby("chart_id", sort=True):
        split_blocks = {
            str(split): set(map(int, part["spatial_block_key"].unique()))
            for split, part in chart.groupby("v12_15_split", sort=True)
        }
        names = sorted(split_blocks)
        overlaps = {
            f"{left}/{right}": len(split_blocks[left] & split_blocks[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        }
        result[str(chart_id)] = {
            "unique_spatial_blocks_by_split": {
                name: len(values) for name, values in split_blocks.items()
            },
            "cross_split_block_overlap_counts": overlaps,
        }
    return result


def build_path_scale_audit() -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    for chart_id, paths in PATH_SOURCES.items():
        catalog_path = paths["catalog"]
        reference_path = paths["reference"]
        catalog = pd.read_csv(catalog_path)
        reference = pd.read_parquet(
            reference_path, columns=["family_id", "phase_idx", *XYZ_COLUMNS]
        )
        sources[chart_id] = {
            "catalog": {
                "path": str(catalog_path.relative_to(PROJECT_ROOT)),
                "bytes": catalog_path.stat().st_size,
                "sha256": sha256_file(catalog_path),
                "rows": int(len(catalog)),
            },
            "reference": {
                "path": str(reference_path.relative_to(PROJECT_ROOT)),
                "bytes": reference_path.stat().st_size,
                "sha256": sha256_file(reference_path),
                "rows": int(len(reference)),
            },
        }
        type_map = catalog.set_index("family_id")["trajectory_type"].astype(str)
        for family_id, group in reference.groupby("family_id", sort=True):
            xyz_mm = group.sort_values("phase_idx")[XYZ_COLUMNS].to_numpy(
                dtype=float
            ) * 1000.0
            center = xyz_mm.mean(axis=0)
            _u, _s, axes = np.linalg.svd(xyz_mm - center, full_matrices=False)
            local = (xyz_mm - center) @ axes.T
            half_extents = np.ptp(local, axis=0) / 2.0
            bbox_extents = np.ptp(xyz_mm, axis=0)
            rows.append(
                {
                    "chart_id": chart_id,
                    "family_id": str(family_id),
                    "trajectory_type": str(type_map.loc[family_id]),
                    "phase_count": int(len(group)),
                    "pca_half_extent_1_mm": float(half_extents[0]),
                    "pca_half_extent_2_mm": float(half_extents[1]),
                    "pca_half_extent_3_mm": float(half_extents[2]),
                    "bbox_x_extent_mm": float(bbox_extents[0]),
                    "bbox_y_extent_mm": float(bbox_extents[1]),
                    "bbox_z_extent_mm": float(bbox_extents[2]),
                    "center_x_mm": float(center[0]),
                    "center_y_mm": float(center[1]),
                    "center_z_mm": float(center[2]),
                }
            )
    return pd.DataFrame(rows), sources


def set_equal_2d(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16, linewidth=0.5)


def build_distribution_figure(frame: pd.DataFrame, dataset_sha: str) -> Path:
    colors = {"chart_A": "#2563eb", "chart_B": "#dc2626"}
    labels = {"chart_A": "Chart A", "chart_B": "Chart B"}
    coordinates = frame.copy()
    for column in XYZ_COLUMNS:
        coordinates[f"{column}_mm"] = coordinates[column] * 1000.0

    figure = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
    ]
    projections = [
        ("y_m_mm", "z_m_mm", "Y-Z projection"),
        ("x_m_mm", "y_m_mm", "X-Y projection"),
        ("x_m_mm", "z_m_mm", "X-Z projection"),
    ]
    for ax, (x_column, y_column, title) in zip(
        axes, projections, strict=True
    ):
        for chart_id, group in coordinates.groupby("chart_id", sort=True):
            ax.scatter(
                group[x_column],
                group[y_column],
                s=1.0,
                alpha=0.13,
                linewidths=0,
                color=colors[str(chart_id)],
                label=labels[str(chart_id)],
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_xlabel(x_column[0].upper() + " (mm)")
        ax.set_ylabel(y_column[0].upper() + " (mm)")
        set_equal_2d(ax)
        ax.legend(markerscale=7, frameon=False)

    ax3d = figure.add_subplot(grid[1, 1], projection="3d")
    for chart_id, group in coordinates.groupby("chart_id", sort=True):
        ax3d.scatter(
            group["x_m_mm"],
            group["y_m_mm"],
            group["z_m_mm"],
            s=0.8,
            alpha=0.08,
            linewidths=0,
            color=colors[str(chart_id)],
            label=labels[str(chart_id)],
            rasterized=True,
        )
    ax3d.set_title("Robot base-frame endpoint distribution")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.view_init(elev=18, azim=7)
    ax3d.legend(markerscale=8, frameon=False)

    figure.suptitle(
        "V12.16C full train/validation endpoint distribution\n"
        f"All {len(frame):,} rows; no sampling; dataset SHA256 {dataset_sha[:16]}...",
        fontsize=14,
    )
    output = OUTPUT_ROOT / "q15_chart_ab_endpoint_distribution.png"
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_sha = sha256_file(DATASET)
    parquet = pq.ParquetFile(DATASET)
    frame = pd.read_parquet(
        DATASET,
        columns=[
            "chart_id",
            "v12_15_split",
            "quality_class",
            "sampling_bucket",
            "spatial_block_key",
            *XYZ_COLUMNS,
        ],
    )
    path_audit, path_sources = build_path_scale_audit()
    path_audit_path = OUTPUT_ROOT / "q15_holdout_geometry_scale_audit.csv"
    path_audit.to_csv(path_audit_path, index=False)
    plot_path = build_distribution_figure(frame, dataset_sha)

    chart_profiles = {
        str(chart_id): {
            "row_count": int(len(group)),
            "coordinate_profile_m": coordinate_profile(group),
        }
        for chart_id, group in frame.groupby("chart_id", sort=True)
    }
    ellipse = path_audit.loc[path_audit["trajectory_type"].eq("ellipse")]
    profile = {
        "schema_version": 1,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "generator": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "method": {
            "dataset_profile": "all rows; no sampling",
            "distribution_figure": "all rows; no sampling; equal-aspect 2D projections",
            "path_scale_audit": (
                "per-family target XYZ centered, SVD/PCA axes, half peak-to-peak "
                "extent in each local axis"
            ),
            "random_seed": None,
        },
        "dataset": {
            "path": str(DATASET.relative_to(PROJECT_ROOT)),
            "bytes": DATASET.stat().st_size,
            "sha256": dataset_sha,
            "rows": int(parquet.metadata.num_rows),
            "columns": int(parquet.metadata.num_columns),
            "row_groups": int(parquet.metadata.num_row_groups),
            "column_names": parquet.schema_arrow.names,
            "chart_split_counts": json_key_counts(
                frame, ["chart_id", "v12_15_split"]
            ),
            "chart_quality_counts": json_key_counts(
                frame, ["chart_id", "quality_class"]
            ),
            "chart_sampling_bucket_counts": json_key_counts(
                frame, ["chart_id", "sampling_bucket"]
            ),
            "chart_profiles": chart_profiles,
            "spatial_block_audit": block_audit(frame),
        },
        "sealed_path_scale_audit": {
            "sources": path_sources,
            "csv": str(path_audit_path.relative_to(PROJECT_ROOT)),
            "path_count": int(len(path_audit)),
            "ellipse_count": int(len(ellipse)),
            "ellipse_pca_half_extent_1_mm_min": float(
                ellipse["pca_half_extent_1_mm"].min()
            ),
            "ellipse_pca_half_extent_1_mm_max": float(
                ellipse["pca_half_extent_1_mm"].max()
            ),
            "ellipse_pca_half_extent_2_mm_min": float(
                ellipse["pca_half_extent_2_mm"].min()
            ),
            "ellipse_pca_half_extent_2_mm_max": float(
                ellipse["pca_half_extent_2_mm"].max()
            ),
        },
        "derived_outputs": {
            "profile_json": "q15_current_dataset_profile.json",
            "path_scale_csv": path_audit_path.name,
            "distribution_png": plot_path.name,
        },
        "limitations": [
            "Axis-aligned bounds and projections do not prove that every point inside the bounding box is supported.",
            "The PCA half extents describe the registered sealed test trajectories, not the full dataset or a guaranteed shell thickness.",
            "This profile does not define a shell coordinate system or infer an unregistered ellipse-radius domain.",
        ],
    }
    profile_path = OUTPUT_ROOT / "q15_current_dataset_profile.json"
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(profile_path)
    print(path_audit_path)
    print(plot_path)


if __name__ == "__main__":
    main()
