"""Preregistered 36-family workspace trajectory catalog for BACRA V14."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.spatial import cKDTree

from .workspace_reach import CellKey


XYZ_COLUMNS = ("x_m", "y_m", "z_m")


@dataclass(frozen=True)
class WorkspaceTrajectorySuite:
    catalog: pd.DataFrame
    targets: pd.DataFrame


def _maximin_indices(points: np.ndarray, count: int) -> tuple[int, ...]:
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(values) or int(count) < 1:
        return ()
    selected = [int(np.lexsort((values[:, 2], values[:, 1], values[:, 0]))[0])]
    minimum = np.linalg.norm(values - values[selected[0]], axis=1)
    while len(selected) < min(int(count), len(values)):
        index = min(
            np.flatnonzero(minimum == np.max(minimum)),
            key=lambda value: tuple(values[int(value)]),
        )
        selected.append(int(index))
        minimum = np.minimum(minimum, np.linalg.norm(values - values[index], axis=1))
    return tuple(selected)


def _fourier_curve(
    center: np.ndarray,
    phase: np.ndarray,
    *,
    amplitude_m: float,
    frequencies: tuple[int, int, int],
    offsets: tuple[float, float, float],
) -> np.ndarray:
    return np.column_stack(
        [
            center[index]
            + float(amplitude_m)
            * np.sin(int(frequencies[index]) * phase + float(offsets[index]))
            for index in range(3)
        ]
    )


def _bspline_curve(center: np.ndarray, phase_count: int, variant: int) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * math.pi, 7, endpoint=False)
    amplitude = 0.006 + 0.001 * (variant % 3)
    control = np.column_stack(
        [
            center[0] + amplitude * np.sin(angle + 0.3 * variant),
            center[1] + amplitude * np.cos(angle),
            center[2] + 0.7 * amplitude * np.sin(2.0 * angle + 0.2 * variant),
        ]
    )
    closed = np.vstack([control, control[0]])
    parameter = np.linspace(0.0, 1.0, len(closed))
    spline = make_interp_spline(parameter, closed, k=3, bc_type="periodic")
    return np.asarray(
        spline(np.linspace(0.0, 1.0, int(phase_count), endpoint=False)), dtype=float
    )


def _cell_keys(frame: pd.DataFrame) -> list[CellKey]:
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    indices = np.floor(xyz / 0.010).astype(np.int64)
    return [CellKey(10, int(row[0]), int(row[1]), int(row[2])) for row in indices]


def build_workspace_trajectory_suite(
    capability: pd.DataFrame,
    classification: pd.DataFrame,
    historical_final8: pd.DataFrame,
    *,
    phase_count: int = 120,
    projection_max_mm: float = 5.0,
) -> WorkspaceTrajectorySuite:
    """Build exactly 8+8+6+6+4+4 deterministic trajectory families.

    New analytic curves are projected only to actual capability points inside
    classified labelable cells, so their final targets have direct empirical
    reach evidence.  The ideal-to-capability projection distance is persisted
    and determines ``teacher_feasible``; a missing chart-transition stratum is
    recorded rather than silently replaced by an ordinary path.
    """

    count = int(phase_count)
    if count < 8:
        raise ValueError("trajectory phase_count must be at least eight")
    if float(projection_max_mm) <= 0.0:
        raise ValueError("projection_max_mm must be positive")
    required = {*XYZ_COLUMNS}
    if missing := sorted(required - set(capability.columns)):
        raise ValueError(f"capability trajectory source missing columns: {missing}")
    allowed_rows = classification[
        classification["domain_class"].isin(
            ("resolved_single_under_budget", "resolved_multichart")
        )
    ]
    allowed = {
        CellKey(10, int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in allowed_rows.itertuples(index=False)
    }
    source = capability.copy()
    source["_cell"] = _cell_keys(source)
    source = source[source["_cell"].isin(allowed)].reset_index(drop=True)
    if len(source) < 8:
        raise ValueError("trajectory suite needs at least eight labelable capability points")
    xyz = source.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    tree = cKDTree(xyz)
    anchors = xyz[list(_maximin_indices(xyz, 28))]
    if len(anchors) < 28:
        anchors = np.vstack([anchors[index % len(anchors)] for index in range(28)])
    phase = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)

    family_specs: list[dict[str, Any]] = []
    for index in range(8):
        center = anchors[index]
        major = 0.004 + 0.001 * (index % 4)
        minor = 0.6 * major
        ideal = np.column_stack(
            [
                center[0] + 0.3 * major * np.sin(phase),
                center[1] + major * np.cos(phase + 0.2 * index),
                center[2] + minor * np.sin(phase + math.pi / 3.0),
            ]
        )
        family_specs.append(
            {"family_id": f"new_ellipse_{index:02d}", "family_type": "new_ellipse_circle", "ideal": ideal, "major_semiaxis_m": major}
        )
    for index in range(6):
        ideal = _fourier_curve(
            anchors[8 + index], phase,
            amplitude_m=0.004 + 0.0005 * index,
            frequencies=(2, 3, 4 + index % 2),
            offsets=(0.0, 0.4, 0.8),
        )
        family_specs.append(
            {"family_id": f"lissajous_{index:02d}", "family_type": "lissajous", "ideal": ideal, "major_semiaxis_m": 0.004 + 0.0005 * index}
        )
    for index in range(6):
        ideal = _bspline_curve(anchors[14 + index], count, index)
        family_specs.append(
            {"family_id": f"bspline3d_{index:02d}", "family_type": "bspline_3d_closed", "ideal": ideal, "major_semiaxis_m": 0.008}
        )
    high_x_indices = np.argsort(-xyz[:, 0], kind="stable")[:4]
    for index, source_index in enumerate(high_x_indices):
        ideal = _fourier_curve(
            xyz[int(source_index)], phase,
            amplitude_m=0.0025 + 0.0005 * index,
            frequencies=(1, 2, 3),
            offsets=(math.pi, 0.0, 0.4),
        )
        family_specs.append(
            {"family_id": f"high_x_tip_{index:02d}", "family_type": "high_x_near_zero_pose", "ideal": ideal, "major_semiaxis_m": 0.004}
        )

    multichart_cells = {
        CellKey(10, int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in classification[
            classification["domain_class"].eq("resolved_multichart")
        ].itertuples(index=False)
    }
    multichart_xyz = source[source["_cell"].isin(multichart_cells)].loc[
        :, XYZ_COLUMNS
    ].to_numpy(dtype=float)
    cross_available = len(multichart_xyz) >= 2
    cross_anchors = (
        multichart_xyz[list(_maximin_indices(multichart_xyz, 8))]
        if cross_available
        else anchors[20:28]
    )
    for index in range(4):
        left = cross_anchors[(2 * index) % len(cross_anchors)]
        right = cross_anchors[(2 * index + 1) % len(cross_anchors)]
        half = count // 2
        forward = np.linspace(left, right, half, endpoint=False)
        reverse = np.linspace(right, left, count - half, endpoint=False)
        family_specs.append(
            {"family_id": f"cross_chart_{index:02d}", "family_type": "cross_chart", "ideal": np.vstack([forward, reverse]), "major_semiaxis_m": float(np.linalg.norm(right - left) / 2.0), "cross_chart_available": cross_available}
        )

    target_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    for spec in family_specs:
        ideal = np.asarray(spec["ideal"], dtype=float)
        distance, nearest = tree.query(ideal, k=1)
        projected = xyz[np.asarray(nearest, dtype=int)]
        feasible = bool(
            np.max(distance) * 1000.0 <= float(projection_max_mm)
            and spec.get("cross_chart_available", True)
        )
        catalog_rows.append(
            {
                "family_id": spec["family_id"],
                "family_type": spec["family_type"],
                "source": "analytic_projected_to_labelable_capability",
                "phase_count": len(projected),
                "major_semiaxis_m": float(spec["major_semiaxis_m"]),
                "teacher_feasible": feasible,
                "projection_p95_mm": float(np.percentile(distance * 1000.0, 95)),
                "projection_max_mm": float(np.max(distance) * 1000.0),
                "reason": (
                    "ok"
                    if feasible
                    else "cross_chart_stratum_missing"
                    if not spec.get("cross_chart_available", True)
                    else "capability_projection_above_max"
                ),
            }
        )
        for phase_index, (raw, target, gap) in enumerate(
            zip(ideal, projected, distance, strict=True)
        ):
            target_rows.append(
                {
                    "family_id": spec["family_id"],
                    "family_type": spec["family_type"],
                    "phase_idx": phase_index,
                    "x_m": target[0], "y_m": target[1], "z_m": target[2],
                    "ideal_x_m": raw[0], "ideal_y_m": raw[1], "ideal_z_m": raw[2],
                    "capability_projection_mm": float(gap * 1000.0),
                    "teacher_feasible": feasible,
                }
            )

    required_historical = {"family_id", "phase_idx", *XYZ_COLUMNS}
    if missing := sorted(required_historical - set(historical_final8.columns)):
        raise ValueError(f"historical final8 source missing columns: {missing}")
    historical_ids = tuple(sorted(historical_final8["family_id"].astype(str).unique()))
    if len(historical_ids) != 8:
        raise ValueError("historical final8 source must contain exactly eight families")
    for family_id in historical_ids:
        family = historical_final8[
            historical_final8["family_id"].astype(str).eq(family_id)
        ].sort_values("phase_idx", kind="stable")
        major = (
            float(family["major_semiaxis_m"].iloc[0])
            if "major_semiaxis_m" in family
            else math.nan
        )
        catalog_rows.append(
            {
                "family_id": f"historical_{family_id}",
                "family_type": "historical_final8",
                "source": "locked_historical_teacher_targets",
                "phase_count": len(family),
                "major_semiaxis_m": major,
                "teacher_feasible": True,
                "projection_p95_mm": 0.0,
                "projection_max_mm": 0.0,
                "reason": "locked_historical_reference",
            }
        )
        for row in family.itertuples(index=False):
            target_rows.append(
                {
                    "family_id": f"historical_{family_id}",
                    "family_type": "historical_final8",
                    "phase_idx": int(row.phase_idx),
                    "x_m": float(row.x_m), "y_m": float(row.y_m), "z_m": float(row.z_m),
                    "ideal_x_m": float(row.x_m), "ideal_y_m": float(row.y_m), "ideal_z_m": float(row.z_m),
                    "capability_projection_mm": 0.0,
                    "teacher_feasible": True,
                }
            )
    catalog = pd.DataFrame.from_records(catalog_rows).sort_values(
        ["family_type", "family_id"], kind="stable"
    ).reset_index(drop=True)
    targets = pd.DataFrame.from_records(target_rows).sort_values(
        ["family_id", "phase_idx"], kind="stable"
    ).reset_index(drop=True)
    expected_counts = {
        "historical_final8": 8,
        "new_ellipse_circle": 8,
        "lissajous": 6,
        "bspline_3d_closed": 6,
        "high_x_near_zero_pose": 4,
        "cross_chart": 4,
    }
    actual_counts = catalog["family_type"].value_counts().to_dict()
    if actual_counts != expected_counts or len(catalog) != 36:
        raise RuntimeError("workspace trajectory suite did not close the 36-family contract")
    return WorkspaceTrajectorySuite(catalog=catalog, targets=targets)
