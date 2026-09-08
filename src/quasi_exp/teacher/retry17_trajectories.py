"""Post-lock maximal cross-layer trajectory generation for retry17."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .retry12_symmetry import XYZ_COLUMNS, stable_id
from .retry17_continuity_fill import continuous_annular_sobol, points_inside_profile


PLANE_SPECS = tuple(
    {"alpha_deg": alpha, "psi_deg": psi}
    for alpha in (30.0, 45.0, 60.0)
    for psi in (0.0, 45.0, 90.0, 135.0)
)


def plane_basis(alpha_deg: float, psi_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha, psi = np.deg2rad([alpha_deg, psi_deg])
    normal = np.asarray([math.cos(alpha), math.sin(alpha) * math.cos(psi), math.sin(alpha) * math.sin(psi)])
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    e1 = np.cross(normal, reference); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1); e2 /= np.linalg.norm(e2)
    return normal, e1, e2


def frozen_shape_objective_registry() -> pd.DataFrame:
    rows = []
    roles = {
        "ellipse": ("mid_stable", "outer_boundary", "maximum_axial_span"),
        "rounded_rectangle": ("largest_distinct_plane_0", "largest_distinct_plane_1", "largest_distinct_plane_2"),
        "rounded_star": ("largest", "different_plane", "boundary_near"),
    }
    for shape_class, objectives in roles.items():
        for slot, objective in enumerate(objectives):
            rows.append(
                {
                    "slot_id": f"retry17_{shape_class}_{slot}",
                    "shape_class": shape_class,
                    "slot_index": slot,
                    "selection_objective": objective,
                    "required": True,
                    "minimum_axial_span_mm": 150.0,
                    "maximum_support_distance_mm": 30.0,
                    "erosion_mm": 15.0,
                }
            )
    return pd.DataFrame(rows)


def _ellipse_unit(count: int) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, int(count), endpoint=False)
    return np.column_stack([np.cos(phase), 0.6 * np.sin(phase)])


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack([np.asarray(points, dtype=float), np.asarray(points, dtype=float)[:1]])
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    query = np.linspace(0.0, cumulative[-1], int(count), endpoint=False)
    indices = np.searchsorted(cumulative, query, side="right") - 1
    indices = np.clip(indices, 0, len(segment) - 1)
    fraction = (query - cumulative[indices]) / np.maximum(segment[indices], 1.0e-15)
    return (1.0 - fraction[:, None]) * closed[indices] + fraction[:, None] * closed[indices + 1]


def _rounded_rectangle_unit(count: int) -> np.ndarray:
    # A 2:1 superellipse provides a deterministic C1 rounded-rectangle trace.
    phase = np.linspace(0.0, 2.0 * np.pi, int(count), endpoint=False)
    exponent = 5.0
    x = np.sign(np.cos(phase)) * np.abs(np.cos(phase)) ** (2.0 / exponent)
    y = 0.5 * np.sign(np.sin(phase)) * np.abs(np.sin(phase)) ** (2.0 / exponent)
    return _resample_closed(np.column_stack([x, y]), count)


def _star_unit(count: int, *, inner_ratio: float = 0.45, rounded: bool = True) -> np.ndarray:
    vertices = []
    for index in range(10):
        angle = 0.5 * np.pi + index * np.pi / 5.0
        radius = 1.0 if index % 2 == 0 else float(inner_ratio)
        vertices.append([radius * math.cos(angle), radius * math.sin(angle)])
    vertices = np.asarray(vertices)
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    allocation = np.maximum(2, np.rint(count * lengths / lengths.sum()).astype(int))
    points = []
    for index, segment_count in enumerate(allocation):
        t = np.linspace(0.0, 1.0, int(segment_count), endpoint=False)
        points.append((1.0 - t[:, None]) * vertices[index] + t[:, None] * vertices[(index + 1) % 10])
    result = np.concatenate(points, axis=0)
    if rounded:
        for _ in range(2):
            result = 0.25 * np.roll(result, 1, axis=0) + 0.5 * result + 0.25 * np.roll(result, -1, axis=0)
    return _resample_closed(result, count)


def unit_shape(shape_class: str, count: int, *, rounded: bool = True) -> np.ndarray:
    if shape_class == "ellipse":
        return _ellipse_unit(count)
    if shape_class == "rounded_rectangle":
        return _rounded_rectangle_unit(count) if rounded else _sharp_rectangle_unit(count)
    if shape_class == "rounded_star":
        return _star_unit(count, rounded=rounded)
    raise ValueError(f"unknown retry17 shape {shape_class}")


def _sharp_rectangle_unit(count: int) -> np.ndarray:
    corners = np.asarray([[1.0, 0.5], [-1.0, 0.5], [-1.0, -0.5], [1.0, -0.5]])
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    allocation = np.maximum(2, np.rint(count * lengths / lengths.sum()).astype(int))
    points = []
    for index, segment_count in enumerate(allocation):
        t = np.linspace(0, 1, int(segment_count), endpoint=False)
        points.append((1 - t[:, None]) * corners[index] + t[:, None] * corners[(index + 1) % 4])
    joined = np.concatenate(points)
    return _resample_closed(joined, count)


def _curve_xyz(center: np.ndarray, e1: np.ndarray, e2: np.ndarray, unit: np.ndarray, scale_mm: float) -> np.ndarray:
    return center[None, :] + (scale_mm / 1000.0) * (unit[:, :1] * e1[None, :] + unit[:, 1:] * e2[None, :])


def _maximal_scale(
    center: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    unit: np.ndarray,
    profile: pd.DataFrame,
    *,
    zero_x_m: float,
    erosion_mm: float,
    maximum_scale_mm: float = 1000.0,
) -> tuple[float, np.ndarray] | None:
    def valid(scale: float) -> bool:
        xyz = _curve_xyz(center, e1, e2, unit, scale)
        return bool(points_inside_profile(xyz, profile, zero_x_m=zero_x_m, erosion_mm=erosion_mm).all())

    low, high = 0.0, 10.0
    while high <= maximum_scale_mm and valid(high):
        low, high = high, 2.0 * high
    high = min(high, maximum_scale_mm)
    if low == 0.0 and not valid(1.0):
        return None
    for _ in range(24):
        mid = 0.5 * (low + high)
        if valid(mid): low = mid
        else: high = mid
    xyz = _curve_xyz(center, e1, e2, unit, low)
    return float(low), xyz


def generate_maximal_shape_registry(
    profile: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    zero_x_m: float,
    center_seed: int,
    center_power: int = 16,
    center_count: int = 200,
    waypoint_count: int = 720,
    erosion_mm: float = 15.0,
    support_maximum_mm: float = 30.0,
    minimum_axial_span_mm: float = 150.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve the frozen 3+3+3 slots only after dataset/model hashes are locked."""

    objectives = frozen_shape_objective_registry()
    centers = continuous_annular_sobol(
        profile, power=center_power, seed=center_seed, zero_x_m=zero_x_m, pool_id="retry17_shape_centers"
    )
    inside = points_inside_profile(centers.loc[:, XYZ_COLUMNS].to_numpy(float), profile, zero_x_m=zero_x_m, erosion_mm=erosion_mm)
    centers = centers.loc[inside].reset_index(drop=True)
    if len(centers) > center_count:
        # Deterministic maximin selection from the first Sobol point.
        xyz = centers.loc[:, XYZ_COLUMNS].to_numpy(float)
        chosen = [0]; distance = np.linalg.norm(xyz - xyz[0], axis=1)
        for _ in range(1, center_count):
            index = int(np.argmax(distance)); chosen.append(index)
            distance = np.minimum(distance, np.linalg.norm(xyz - xyz[index], axis=1))
        centers = centers.iloc[chosen].reset_index(drop=True)
    data_tree = cKDTree(dataset.loc[:, XYZ_COLUMNS].to_numpy(float))
    candidates: list[dict[str, Any]] = []
    waypoints: dict[str, pd.DataFrame] = {}
    for shape_class in objectives["shape_class"].unique():
        unit = unit_shape(str(shape_class), waypoint_count)
        for plane_index, plane in enumerate(PLANE_SPECS):
            _normal, e1, e2 = plane_basis(plane["alpha_deg"], plane["psi_deg"])
            for center_index, center_row in centers.iterrows():
                center = center_row.loc[list(XYZ_COLUMNS)].to_numpy(float)
                solved = _maximal_scale(center, e1, e2, unit, profile, zero_x_m=zero_x_m, erosion_mm=erosion_mm)
                if solved is None: continue
                scale, xyz = solved
                axial_span = float(np.ptp(1000.0 * (zero_x_m - xyz[:, 0])))
                support = data_tree.query(xyz, k=1)[0] * 1000.0
                support_max = float(np.max(support))
                if axial_span < minimum_axial_span_mm or support_max > support_maximum_mm:
                    continue
                candidate_id = stable_id("retry17_shape_candidate", shape_class, plane_index, center_index)
                candidates.append(
                    {
                        "candidate_id": candidate_id, "shape_class": shape_class,
                        "plane_index": plane_index, **plane, "center_index": center_index,
                        "center_x_m": center[0], "center_y_m": center[1], "center_z_m": center[2],
                        "scale_mm": scale, "axial_span_mm": axial_span,
                        "support_maximum_mm": support_max, "support_p95_mm": float(np.percentile(support, 95)),
                    }
                )
                frame = pd.DataFrame(xyz, columns=XYZ_COLUMNS)
                frame["waypoint_index"] = np.arange(len(frame), dtype=np.int64)
                frame["candidate_id"] = candidate_id
                waypoints[candidate_id] = frame
    candidate_frame = pd.DataFrame(candidates)
    selected_rows, selected_waypoints = [], []
    used_planes: dict[str, set[int]] = {}
    for objective in objectives.to_dict("records"):
        subset = candidate_frame[candidate_frame["shape_class"].eq(objective["shape_class"])].copy()
        unavailable = used_planes.setdefault(str(objective["shape_class"]), set())
        distinct = subset[~subset["plane_index"].isin(unavailable)]
        if not distinct.empty: subset = distinct
        if subset.empty:
            selected_rows.append({**objective, "selection_status": "unavailable"})
            continue
        if objective["selection_objective"] in ("outer_boundary", "boundary_near"):
            chosen = subset.sort_values(["support_maximum_mm", "scale_mm", "candidate_id"], ascending=[False, False, True], kind="stable").iloc[0]
        elif objective["selection_objective"] == "maximum_axial_span":
            chosen = subset.sort_values(["axial_span_mm", "scale_mm", "candidate_id"], ascending=[False, False, True], kind="stable").iloc[0]
        else:
            chosen = subset.sort_values(["scale_mm", "axial_span_mm", "candidate_id"], ascending=[False, False, True], kind="stable").iloc[0]
        trajectory_id = str(objective["slot_id"])
        used_planes[str(objective["shape_class"])].add(int(chosen["plane_index"]))
        selected_rows.append({**objective, **chosen.to_dict(), "trajectory_id": trajectory_id, "selection_status": "selected", "largest_found_not_global_optimum": True})
        frame = waypoints[str(chosen["candidate_id"])].copy()
        frame["trajectory_id"] = trajectory_id
        frame["shape_class"] = objective["shape_class"]
        selected_waypoints.append(frame)
    selected = pd.DataFrame(selected_rows)
    waypoint_frame = pd.concat(selected_waypoints, ignore_index=True, sort=False) if selected_waypoints else pd.DataFrame()
    # Sharp rectangle is an explicitly non-gating diagnostic at the first selected rounded rectangle pose.
    rectangle = selected[selected["shape_class"].eq("rounded_rectangle") & selected["selection_status"].eq("selected")]
    if len(rectangle):
        row = rectangle.iloc[0]
        _normal, e1, e2 = plane_basis(float(row["alpha_deg"]), float(row["psi_deg"]))
        center = np.asarray([row["center_x_m"], row["center_y_m"], row["center_z_m"]])
        sharp = _curve_xyz(center, e1, e2, unit_shape("rounded_rectangle", waypoint_count, rounded=False), float(row["scale_mm"]))
        diagnostic = pd.DataFrame(sharp, columns=XYZ_COLUMNS)
        diagnostic["waypoint_index"] = np.arange(len(diagnostic)); diagnostic["trajectory_id"] = "retry17_sharp_rectangle_diagnostic"
        diagnostic["shape_class"] = "sharp_rectangle_diagnostic"; diagnostic["candidate_id"] = row["candidate_id"]
        waypoint_frame = pd.concat([waypoint_frame, diagnostic], ignore_index=True, sort=False)
    return selected, waypoint_frame, candidate_frame


__all__ = [
    "PLANE_SPECS", "frozen_shape_objective_registry", "generate_maximal_shape_registry",
    "plane_basis", "unit_shape",
]
