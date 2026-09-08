"""Retry19 signed-workspace geometry, coverage, anchors, and Graph Teacher.

The module owns the contracts that differ scientifically from retry18:

* mixed-resolution observations remain their real 5/10 mm boxes;
* primary connectivity is positive-area face connectivity;
* exact zero has separate geometric and canonical-beta attachments;
* historical labels are hard, soft, or diagnostic candidates;
* Graph Teacher starts from an MST/tree-DP solution and refines the complete
  density-normalized graph with deterministic ICM orders.

Proposal beta values are intentionally absent from every public Teacher API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .exploration_qualification import weighted_beta_rms_deg
from .retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS, stable_id
from .retry15_canonical_graph import legal_candidate_clusters


BOX_MIN_COLUMNS = ("x_min_mm", "y_min_mm", "z_min_mm")
BOX_MAX_COLUMNS = ("x_max_mm", "y_max_mm", "z_max_mm")
CELL_REQUIRED_COLUMNS = {
    "cell_id",
    "cell_size_mm",
    "volume_mm3",
    *BOX_MIN_COLUMNS,
    *BOX_MAX_COLUMNS,
}


@dataclass(frozen=True)
class FullspaceCoveragePolicy:
    service_radius_mm: float = 10.0
    root_registration_mm: float = 5.0
    volume_coverage_minimum: float = 0.90
    fine_coverage_minimum: float = 0.85
    coarse_coverage_minimum: float = 0.90
    maximum_hole_fraction: float = 0.02
    minimum_axial_slice_coverage: float = 0.80
    path_success_minimum: float = 0.95
    stretch_p95_maximum: float = 2.0

    def __post_init__(self) -> None:
        if self.service_radius_mm <= 0 or self.root_registration_mm <= 0:
            raise ValueError("coverage distances must be positive")
        fractions = (
            self.volume_coverage_minimum,
            self.fine_coverage_minimum,
            self.coarse_coverage_minimum,
            self.maximum_hole_fraction,
            self.minimum_axial_slice_coverage,
            self.path_success_minimum,
        )
        if any(not 0 <= float(value) <= 1 for value in fractions):
            raise ValueError("coverage fractions must be in [0, 1]")
        if self.stretch_p95_maximum < 1:
            raise ValueError("stretch threshold cannot be below one")


@dataclass(frozen=True)
class Retry19TeacherPolicy:
    k: int = 16
    sigma_mm: float = 10.0
    pairwise_lambda: float = 4.0
    huber_delta_deg: float = 7.0
    soft_anchor_bonus: float = 0.25
    maximum_icm_sweeps: int = 30
    beta_weights: tuple[float, ...] = (4, 4, 2, 2, 1, 1)

    def __post_init__(self) -> None:
        if self.k < 1 or self.sigma_mm <= 0 or self.pairwise_lambda < 0:
            raise ValueError("invalid Graph Teacher neighbourhood policy")
        if self.huber_delta_deg <= 0 or self.maximum_icm_sweeps < 1:
            raise ValueError("invalid Graph Teacher optimizer policy")
        if self.soft_anchor_bonus < 0 or len(self.beta_weights) != 6:
            raise ValueError("invalid Graph Teacher unary policy")


def _validated_cells(cells: pd.DataFrame) -> pd.DataFrame:
    if not CELL_REQUIRED_COLUMNS <= set(cells):
        raise ValueError("mixed-resolution cell frame is missing box columns")
    result = cells.copy()
    if result["cell_id"].astype(str).duplicated().any():
        raise ValueError("cell_id must be unique")
    lower = result.loc[:, BOX_MIN_COLUMNS].to_numpy(float)
    upper = result.loc[:, BOX_MAX_COLUMNS].to_numpy(float)
    size = result["cell_size_mm"].to_numpy(float)
    if not np.isfinite(np.column_stack((lower, upper, size))).all():
        raise ValueError("cell geometry must be finite")
    extent = upper - lower
    if np.any(extent <= 0) or np.any(~np.isin(size, (5.0, 10.0))):
        raise ValueError("retry19 cells must be positive 5/10 mm boxes")
    if not np.allclose(extent, size[:, None], atol=1.0e-9, rtol=0):
        raise ValueError("cell_size_mm must equal every box extent")
    expected_volume = np.prod(extent, axis=1)
    if not np.allclose(result["volume_mm3"].to_numpy(float), expected_volume, atol=1.0e-6, rtol=0):
        raise ValueError("volume_mm3 must describe the real box")
    return result


def build_mixed_resolution_cell_registry(
    pool_a_xyz_m: np.ndarray,
    pool_b_xyz_m: np.ndarray,
    *,
    zero_xyz_m: Sequence[float],
    primary_u_maximum_mm: float = 200.0,
    audit_u_maximum_mm: float = 220.0,
    fine_root_radius_mm: float = 40.0,
) -> pd.DataFrame:
    """Register only actually observed fine/coarse boxes from two FK pools.

    Ten-millimetre parent boxes whose box-to-zero distance is within the
    registered root radius are geometrically refined, but only fine children
    that a pool truly hits receive that pool's observation flag.
    """

    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    zero_mm = zero * 1000.0

    def observed(points_m: np.ndarray, pool: str) -> pd.DataFrame:
        xyz_mm = np.asarray(points_m, dtype=float).reshape(-1, 3) * 1000.0
        u_mm = zero_mm[0] - xyz_mm[:, 0]
        keep = (u_mm >= -1.0e-9) & (u_mm <= float(audit_u_maximum_mm) + 1.0e-9)
        xyz_mm, u_mm = xyz_mm[keep], u_mm[keep]
        coarse_index = np.floor(xyz_mm / 10.0).astype(np.int64)
        coarse_min = coarse_index * 10.0
        coarse_max = coarse_min + 10.0
        coarse_delta = np.maximum(np.maximum(coarse_min - zero_mm, zero_mm - coarse_max), 0.0)
        refined = np.linalg.norm(coarse_delta, axis=1) <= float(fine_root_radius_mm) + 1.0e-9
        size = np.where(refined, 5, 10).astype(np.int64)
        index = np.floor(xyz_mm / size[:, None]).astype(np.int64)

        # Deduplicate the compact integer key before materializing strings or
        # pandas rows.  The formal pools contain more than a million proposal
        # hits but only about a hundred thousand observed boxes per pool.
        keys = np.unique(np.column_stack((size, index)), axis=0)
        unique_size = keys[:, 0]
        unique_index = keys[:, 1:]
        lower = unique_index * unique_size[:, None]
        upper = lower + unique_size[:, None]
        records = pd.DataFrame(
            {
                "cell_id": [
                    f"box{int(s)}:{x}:{y}:{z}"
                    for s, (x, y, z) in zip(unique_size, unique_index, strict=True)
                ],
                "x_min_mm": lower[:, 0],
                "y_min_mm": lower[:, 1],
                "z_min_mm": lower[:, 2],
                "x_max_mm": upper[:, 0],
                "y_max_mm": upper[:, 1],
                "z_max_mm": upper[:, 2],
                "cell_size_mm": unique_size.astype(float),
            }
        )
        records[f"pool_{pool}_observed"] = True
        return records

    a = observed(pool_a_xyz_m, "a")
    b = observed(pool_b_xyz_m, "b")
    combined = a.merge(b, on=["cell_id", *BOX_MIN_COLUMNS, *BOX_MAX_COLUMNS, "cell_size_mm"], how="outer")
    combined["pool_a_observed"] = combined["pool_a_observed"].notna()
    combined["pool_b_observed"] = combined["pool_b_observed"].notna()
    combined["volume_mm3"] = np.power(combined["cell_size_mm"].to_numpy(float), 3)
    centre_mm = 0.5 * (
        combined.loc[:, BOX_MIN_COLUMNS].to_numpy(float) + combined.loc[:, BOX_MAX_COLUMNS].to_numpy(float)
    )
    u_centre = zero_mm[0] - centre_mm[:, 0]
    combined["u_center_mm"] = u_centre
    combined["domain_class"] = np.where(u_centre <= primary_u_maximum_mm + 1.0e-9, "primary", "audit_only")
    combined["probe_x_m"] = centre_mm[:, 0] / 1000.0
    combined["probe_y_m"] = centre_mm[:, 1] / 1000.0
    combined["probe_z_m"] = centre_mm[:, 2] / 1000.0
    combined["required"] = combined["pool_a_observed"] & combined["pool_b_observed"]
    combined["served"] = False
    combined["proposal_beta_used_as_label_or_hint"] = False
    return _validated_cells(combined).sort_values("cell_id", kind="stable").reset_index(drop=True)


def target_knn_edges(targets: pd.DataFrame, *, k: int = 16) -> pd.DataFrame:
    if k < 1 or not {"target_id", *XYZ_COLUMNS} <= set(targets):
        raise ValueError("target graph requires target_id, xyz, and positive k")
    frame = targets.drop_duplicates("target_id").reset_index(drop=True)
    if len(frame) < 2:
        return pd.DataFrame(columns=["edge_id", "left_target_id", "right_target_id", "distance_mm"])
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float)
    ids = frame["target_id"].astype(str).tolist()
    _distance, neighbours = cKDTree(xyz).query(xyz, k=min(len(frame), int(k) + 1))
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for left, local in enumerate(np.atleast_2d(neighbours)):
        for right in np.atleast_1d(local)[1:]:
            right = int(right)
            if left == right:
                continue
            pair = tuple(sorted((ids[left], ids[right])))
            rows[pair] = {
                "edge_id": stable_id("retry19_target_edge", *pair),
                "left_target_id": pair[0],
                "right_target_id": pair[1],
                "distance_mm": float(np.linalg.norm(xyz[left] - xyz[right]) * 1000.0),
            }
    return pd.DataFrame(rows.values()).sort_values(["left_target_id", "right_target_id"], kind="stable").reset_index(drop=True)


def mixed_resolution_face_edges(cells: pd.DataFrame, *, tolerance_mm: float = 1.0e-9) -> pd.DataFrame:
    """Return only positive-area shared-face adjacency.

    A coarse observation remains one coarse node.  No fine children are
    materialized by this function or credited by its output.
    """

    frame = _validated_cells(cells)
    ids = frame["cell_id"].astype(str).tolist()
    lower = frame.loc[:, BOX_MIN_COLUMNS].to_numpy(float)
    upper = frame.loc[:, BOX_MAX_COLUMNS].to_numpy(float)

    # Face tiles are an adjacency index only.  They do not create observed
    # cells or receive coverage credit.  Bucketing bounds candidate pairs so
    # the formal registry does not require an O(N^2) box comparison.
    minima: dict[tuple[int, int, int, int], int] = {}
    maxima: dict[tuple[int, int, int, int], int] = {}
    quantum = 5.0
    for index in range(len(frame)):
        for axis in range(3):
            other = [value for value in range(3) if value != axis]
            first_start = math.floor((lower[index, other[0]] + tolerance_mm) / quantum)
            first_stop = math.ceil((upper[index, other[0]] - tolerance_mm) / quantum)
            second_start = math.floor((lower[index, other[1]] + tolerance_mm) / quantum)
            second_stop = math.ceil((upper[index, other[1]] - tolerance_mm) / quantum)
            for side, coordinate in (("min", lower[index, axis]), ("max", upper[index, axis])):
                plane = int(round(coordinate / tolerance_mm))
                destination = minima if side == "min" else maxima
                for first in range(first_start, first_stop):
                    for second in range(second_start, second_stop):
                        destination[(axis, plane, first, second)] = index
    candidate_pairs: set[tuple[int, int, int]] = set()
    for key, left in maxima.items():
        right = minima.get(key)
        if right is not None and left != right:
            candidate_pairs.add((min(left, right), max(left, right), key[0]))
    rows: list[dict[str, Any]] = []
    for left, right, axis in sorted(candidate_pairs):
        if not (
            abs(upper[left, axis] - lower[right, axis]) <= tolerance_mm
            or abs(upper[right, axis] - lower[left, axis]) <= tolerance_mm
        ):
            continue
        other = [value for value in range(3) if value != axis]
        lengths = [
            max(0.0, min(upper[left, value], upper[right, value]) - max(lower[left, value], lower[right, value]))
            for value in other
        ]
        area = float(lengths[0] * lengths[1])
        if area <= tolerance_mm:
            continue
        rows.append({
            "edge_id": stable_id("retry19_face", ids[left], ids[right]),
            "left_cell_id": min(ids[left], ids[right]),
            "right_cell_id": max(ids[left], ids[right]),
            "face_axis": "xyz"[axis],
            "shared_face_area_mm2": area,
            "adjacency": "positive_area_face",
        })
    columns = [
        "edge_id",
        "left_cell_id",
        "right_cell_id",
        "face_axis",
        "shared_face_area_mm2",
        "adjacency",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["left_cell_id", "right_cell_id"], kind="stable"
    ).reset_index(drop=True)


def mixed_resolution_26_edges(cells: pd.DataFrame) -> pd.DataFrame:
    """Return Q31 26-neighbour adjacency without inventing observed cells.

    The 5 mm lattice is used only as a compact adjacency index.  A 10 mm box
    occupies eight lattice slots but remains one observed node and receives
    coverage credit only once.
    """

    frame = _validated_cells(cells)
    ids = frame["cell_id"].astype(str).tolist()
    lower = np.rint(frame.loc[:, BOX_MIN_COLUMNS].to_numpy(float) / 5.0).astype(np.int64)
    upper = np.rint(frame.loc[:, BOX_MAX_COLUMNS].to_numpy(float) / 5.0).astype(np.int64)
    occupancy: dict[tuple[int, int, int], int] = {}
    for index in range(len(frame)):
        for x in range(int(lower[index, 0]), int(upper[index, 0])):
            for y in range(int(lower[index, 1]), int(upper[index, 1])):
                for z in range(int(lower[index, 2]), int(upper[index, 2])):
                    key = (x, y, z)
                    owner = occupancy.setdefault(key, index)
                    if owner != index:
                        raise ValueError("mixed-resolution observed boxes must not overlap")

    pairs: set[tuple[int, int]] = set()
    for (x, y, z), left in occupancy.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    right = occupancy.get((x + dx, y + dy, z + dz))
                    if right is not None and left != right:
                        pairs.add((min(left, right), max(left, right)))
    rows = [
        {
            "edge_id": stable_id("retry19_26_neighbor", ids[left], ids[right]),
            "left_cell_id": min(ids[left], ids[right]),
            "right_cell_id": max(ids[left], ids[right]),
            "adjacency": "q31_26_neighbor",
        }
        for left, right in sorted(pairs)
    ]
    return pd.DataFrame(
        rows,
        columns=["edge_id", "left_cell_id", "right_cell_id", "adjacency"],
    ).sort_values(["left_cell_id", "right_cell_id"], kind="stable").reset_index(drop=True)


def box_to_point_distance_mm(cells: pd.DataFrame, point_mm: Sequence[float]) -> np.ndarray:
    frame = _validated_cells(cells)
    point = np.asarray(point_mm, dtype=float).reshape(3)
    lower = frame.loc[:, BOX_MIN_COLUMNS].to_numpy(float)
    upper = frame.loc[:, BOX_MAX_COLUMNS].to_numpy(float)
    delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return np.linalg.norm(delta, axis=1)


def geometric_zero_attachment(
    cells: pd.DataFrame,
    face_edges: pd.DataFrame,
    *,
    zero_cell_id: str,
    zero_point_mm: Sequence[float],
    registration_threshold_mm: float = 5.0,
) -> tuple[pd.DataFrame, set[str]]:
    """Attach exact zero only through a face or registered box distance."""

    frame = _validated_cells(cells)
    ids = set(frame["cell_id"].astype(str))
    if zero_cell_id not in ids:
        raise ValueError("exact-zero cell must be deterministically inserted")
    root_edges = face_edges[
        face_edges["left_cell_id"].astype(str).eq(zero_cell_id)
        | face_edges["right_cell_id"].astype(str).eq(zero_cell_id)
    ].copy()
    existing = set(root_edges.get("left_cell_id", pd.Series(dtype=str)).astype(str)) | set(
        root_edges.get("right_cell_id", pd.Series(dtype=str)).astype(str)
    )
    distances = box_to_point_distance_mm(frame, zero_point_mm)
    additions: list[dict[str, Any]] = []
    for index, distance in enumerate(distances):
        other = str(frame.iloc[index]["cell_id"])
        if other == zero_cell_id or other in existing or distance > registration_threshold_mm + 1.0e-9:
            continue
        left, right = sorted((zero_cell_id, other))
        additions.append(
            {
                "edge_id": stable_id("retry19_root_registration", left, right),
                "left_cell_id": left,
                "right_cell_id": right,
                "face_axis": "registered_distance",
                "shared_face_area_mm2": 0.0,
                "adjacency": "root_box_distance",
                "root_box_distance_mm": float(distance),
            }
        )
    if additions and len(face_edges):
        edges = pd.concat([face_edges, pd.DataFrame(additions)], ignore_index=True, sort=False)
    elif additions:
        edges = pd.DataFrame(additions)
    else:
        edges = face_edges.copy()
    adjacency = {cell_id: set() for cell_id in ids}
    for row in edges.to_dict("records"):
        left, right = str(row["left_cell_id"]), str(row["right_cell_id"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    component: set[str] = set()
    stack = [zero_cell_id]
    while stack:
        current = stack.pop()
        if current in component:
            continue
        component.add(current)
        stack.extend(sorted(adjacency[current] - component, reverse=True))
    return edges.sort_values(["left_cell_id", "right_cell_id"], kind="stable").reset_index(drop=True), component


def teacher_zero_attachment(
    labels: pd.DataFrame,
    target_edges: pd.DataFrame,
    *,
    zero_target_id: str,
    residual_maximum_mm: float = 3.0,
    beta_gap_maximum_deg: float = 3.0,
    weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
) -> tuple[bool, set[str], pd.DataFrame]:
    """Return canonical-beta root connectivity, independent of cell geometry."""

    if zero_target_id not in set(labels.get("target_id", pd.Series(dtype=str)).astype(str)):
        return False, set(), pd.DataFrame()
    lookup = labels.set_index(labels["target_id"].astype(str), drop=False)
    accepted: list[dict[str, Any]] = []
    adjacency = {target: set() for target in lookup.index}
    for row in target_edges.to_dict("records"):
        left, right = str(row["left_target_id"]), str(row["right_target_id"])
        if left not in lookup.index or right not in lookup.index:
            continue
        a, b = lookup.loc[left], lookup.loc[right]
        residual_ok = float(a.get("fk_residual_mm", math.inf)) <= residual_maximum_mm and float(
            b.get("fk_residual_mm", math.inf)
        ) <= residual_maximum_mm
        gap = float(
            weighted_beta_rms_deg(
                a.loc[list(BETA_COLUMNS)].to_numpy(float),
                b.loc[list(BETA_COLUMNS)].to_numpy(float),
                weights,
            )
        )
        legal = bool(residual_ok and gap <= beta_gap_maximum_deg + 1.0e-12)
        accepted.append({**row, "weighted_beta_gap_deg": gap, "teacher_edge_legal": legal})
        if legal:
            adjacency[left].add(right)
            adjacency[right].add(left)
    component: set[str] = set()
    stack = [zero_target_id]
    while stack:
        current = stack.pop()
        if current in component:
            continue
        component.add(current)
        stack.extend(sorted(adjacency[current] - component, reverse=True))
    return len(component) > 1, component, pd.DataFrame(accepted)


def mark_served_cells(
    cells: pd.DataFrame,
    accepted_labels: pd.DataFrame,
    *,
    teacher_zero_connected_ids: Iterable[str],
    service_radius_mm: float = 10.0,
) -> pd.DataFrame:
    """Mark a real cell served from its frozen evaluation probe."""

    frame = _validated_cells(cells)
    probe_columns = ("probe_x_m", "probe_y_m", "probe_z_m")
    if not set(probe_columns) <= set(frame):
        lower = frame.loc[:, BOX_MIN_COLUMNS].to_numpy(float)
        upper = frame.loc[:, BOX_MAX_COLUMNS].to_numpy(float)
        probes = (lower + upper) / 2000.0
    else:
        probes = frame.loc[:, probe_columns].to_numpy(float)
    connected = set(map(str, teacher_zero_connected_ids))
    eligible = accepted_labels[accepted_labels["target_id"].astype(str).isin(connected)].copy()
    result = frame.copy()
    if eligible.empty:
        result["nearest_teacher_target_id"] = ""
        result["service_distance_mm"] = math.inf
        result["served"] = False
        return result
    distance_m, nearest = cKDTree(eligible.loc[:, XYZ_COLUMNS].to_numpy(float)).query(probes, k=1)
    result["nearest_teacher_target_id"] = eligible.iloc[np.asarray(nearest, dtype=int)]["target_id"].astype(str).to_numpy()
    result["service_distance_mm"] = np.asarray(distance_m, dtype=float) * 1000.0
    required = result.get("required", pd.Series(True, index=result.index)).astype(bool)
    result["served"] = required & result["service_distance_mm"].le(float(service_radius_mm) + 1.0e-9)
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return math.inf
    values, weights = values[finite], weights[finite]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), quantile * np.sum(weights), side="left")])


def mixed_resolution_coverage_metrics(cells: pd.DataFrame) -> Mapping[str, float | int]:
    frame = _validated_cells(cells)
    required = frame.get("required", pd.Series(True, index=frame.index)).astype(bool)
    served = frame.get("served", pd.Series(False, index=frame.index)).astype(bool) & required
    volume = frame["volume_mm3"].to_numpy(float)

    def fraction(mask: np.ndarray) -> float:
        denominator = float(np.sum(volume[mask]))
        return float(np.sum(volume[mask & served.to_numpy()]) / denominator) if denominator > 0 else 0.0

    required_mask = required.to_numpy()
    fine = required_mask & frame["cell_size_mm"].eq(5.0).to_numpy()
    coarse = required_mask & frame["cell_size_mm"].eq(10.0).to_numpy()
    distance = frame.get("service_distance_mm", pd.Series(math.inf, index=frame.index)).to_numpy(float)
    return {
        "required_cell_count": int(required.sum()),
        "served_cell_count": int(served.sum()),
        "cell_count_coverage": float(served.sum() / required.sum()) if required.any() else 0.0,
        "volume_coverage": fraction(required_mask),
        "fine_5mm_coverage": fraction(fine),
        "coarse_10mm_coverage": fraction(coarse),
        "service_distance_volume_weighted_p95_mm": _weighted_quantile(
            distance[required_mask], volume[required_mask], 0.95
        ),
    }


def largest_unserved_component_fraction(cells: pd.DataFrame, face_edges: pd.DataFrame) -> float:
    frame = _validated_cells(cells)
    required = frame.get("required", pd.Series(True, index=frame.index)).astype(bool)
    served = frame.get("served", pd.Series(False, index=frame.index)).astype(bool)
    unserved_ids = set(frame.loc[required & ~served, "cell_id"].astype(str))
    total_volume = float(frame.loc[required, "volume_mm3"].sum())
    if not unserved_ids or total_volume <= 0:
        return 0.0
    volume = frame.set_index(frame["cell_id"].astype(str))["volume_mm3"].to_dict()
    adjacency = {cell_id: set() for cell_id in unserved_ids}
    primary = face_edges[face_edges["adjacency"].eq("positive_area_face")]
    for row in primary.to_dict("records"):
        left, right = str(row["left_cell_id"]), str(row["right_cell_id"])
        if left in unserved_ids and right in unserved_ids:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen: set[str] = set()
    largest = 0.0
    for root in sorted(unserved_ids):
        if root in seen:
            continue
        component: set[str] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen |= component
        largest = max(largest, sum(float(volume[cell_id]) for cell_id in component))
    return float(largest / total_volume)


def axial_slice_coverage(
    cells: pd.DataFrame,
    *,
    zero_x_mm: float,
    bin_edges_u_mm: Sequence[float],
) -> pd.DataFrame:
    """Use exact box/slice intersection volume, not cell-centre binning."""

    frame = _validated_cells(cells)
    edges = np.asarray(bin_edges_u_mm, dtype=float)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("axial bin edges must be strictly increasing")
    required = frame.get("required", pd.Series(True, index=frame.index)).astype(bool).to_numpy()
    served = frame.get("served", pd.Series(False, index=frame.index)).astype(bool).to_numpy()
    x_min = frame["x_min_mm"].to_numpy(float)
    x_max = frame["x_max_mm"].to_numpy(float)
    u_min = zero_x_mm - x_max
    u_max = zero_x_mm - x_min
    yz_area = (frame["y_max_mm"] - frame["y_min_mm"]).to_numpy(float) * (
        frame["z_max_mm"] - frame["z_min_mm"]
    ).to_numpy(float)
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        overlap = np.maximum(0.0, np.minimum(u_max, right) - np.maximum(u_min, left)) * yz_area
        denominator = float(np.sum(overlap[required]))
        numerator = float(np.sum(overlap[required & served]))
        rows.append(
            {
                "u_bin_index": index,
                "u_min_mm": float(left),
                "u_max_mm": float(right),
                "required_volume_mm3": denominator,
                "served_volume_mm3": numerator,
                "volume_coverage": numerator / denominator if denominator > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def deterministic_path_query_registry(
    probes: pd.DataFrame,
    *,
    pair_count: int,
    seed: int,
) -> pd.DataFrame:
    if pair_count < 1 or len(probes) < 2:
        raise ValueError("path query registry requires at least two probes")
    required = probes[probes.get("required", pd.Series(True, index=probes.index)).astype(bool)].copy()
    ids = sorted(required["cell_id"].astype(str).unique())
    total_pairs = len(ids) * (len(ids) - 1) // 2
    selected_count = min(int(pair_count), total_pairs)
    selected_pairs: set[tuple[int, int]] = set()

    # Exhaustion is cheap only for genuinely small registries.  The formal
    # workspace contains O(1e5) cells, where enumerating all O(N^2) pairs just
    # to retain 1,000 queries is both unnecessary and memory-fatal.
    if total_pairs <= max(10_000, 10 * selected_count):
        ranked = []
        for left_index, left in enumerate(ids):
            for right_index in range(left_index + 1, len(ids)):
                right = ids[right_index]
                digest = hashlib.sha256(f"{seed}:{left}:{right}".encode()).hexdigest()
                ranked.append((digest, left_index, right_index))
        selected_pairs = {(left, right) for _digest, left, right in sorted(ranked)[:selected_count]}
    else:
        counter = 0
        while len(selected_pairs) < selected_count:
            digest = hashlib.sha256(f"{seed}:pair_sample:{counter}".encode()).digest()
            counter += 1
            left_index = int.from_bytes(digest[:8], "big") % len(ids)
            right_index = int.from_bytes(digest[8:16], "big") % len(ids)
            if left_index == right_index:
                continue
            selected_pairs.add(tuple(sorted((left_index, right_index))))

    selected = sorted(
        (
            hashlib.sha256(f"{seed}:{ids[left]}:{ids[right]}".encode()).hexdigest(),
            ids[left],
            ids[right],
        )
        for left, right in selected_pairs
    )
    return pd.DataFrame(
        [
            {
                "query_id": stable_id("retry19_path_query", seed, left, right),
                "left_cell_id": left,
                "right_cell_id": right,
                "selection_digest": digest,
            }
            for digest, left, right in selected
        ]
    )


def path_query_metrics(
    cells: pd.DataFrame,
    accepted_targets: pd.DataFrame,
    target_edges: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    service_radius_mm: float = 10.0,
) -> tuple[Mapping[str, float | int], pd.DataFrame]:
    frame = _validated_cells(cells).set_index("cell_id", drop=False)
    targets = accepted_targets.reset_index(drop=True)
    if targets.empty:
        audit = queries.assign(success=False, failure_reason="no_accepted_targets", stretch=math.inf)
        return {"query_count": len(audit), "success_rate": 0.0, "stretch_p95": math.inf}, audit
    target_ids = targets["target_id"].astype(str).tolist()
    target_index = {target: index for index, target in enumerate(target_ids)}
    xyz = targets.loc[:, XYZ_COLUMNS].to_numpy(float)
    edge_weights: dict[tuple[int, int], float] = {}
    for row in target_edges.itertuples(index=False):
        left, right = str(row.left_target_id), str(row.right_target_id)
        if left not in target_index or right not in target_index:
            continue
        i, j = target_index[left], target_index[right]
        if i == j:
            continue
        distance = float(
            getattr(row, "distance_mm", np.linalg.norm(xyz[i] - xyz[j]) * 1000.0)
        )
        key = tuple(sorted((i, j)))
        edge_weights[key] = min(edge_weights.get(key, math.inf), distance)
    if edge_weights:
        undirected_edges = sorted(edge_weights.items())
        left_indices = [key[0] for key, _distance in undirected_edges]
        right_indices = [key[1] for key, _distance in undirected_edges]
        distances = [distance for _key, distance in undirected_edges]
        graph = csr_matrix(
            (
                np.asarray(distances + distances, dtype=float),
                (
                    np.asarray(left_indices + right_indices, dtype=np.int64),
                    np.asarray(right_indices + left_indices, dtype=np.int64),
                ),
            ),
            shape=(len(targets), len(targets)),
        )
    else:
        graph = csr_matrix((len(targets), len(targets)), dtype=float)
    probe_xyz = np.column_stack(
        [
            (frame[lower].to_numpy(float) + frame[upper].to_numpy(float)) / 2000.0
            for lower, upper in zip(BOX_MIN_COLUMNS, BOX_MAX_COLUMNS, strict=True)
        ]
    )
    distance_m, nearest = cKDTree(xyz).query(probe_xyz, k=1)
    projection = {
        cell_id: (int(nearest[index]), float(distance_m[index] * 1000.0))
        for index, cell_id in enumerate(frame.index.astype(str))
    }
    query_records = queries.to_dict("records")
    source_indices = sorted(
        {
            projection[str(row["left_cell_id"])][0]
            for row in query_records
            if str(row["left_cell_id"]) in projection and str(row["right_cell_id"]) in projection
        }
    )
    shortest = (
        np.atleast_2d(dijkstra(graph, directed=False, indices=source_indices))
        if source_indices
        else np.empty((0, len(targets)), dtype=float)
    )
    shortest_row = {source: row for row, source in enumerate(source_indices)}
    probe_position = {cell_id: index for index, cell_id in enumerate(frame.index.astype(str))}
    rows: list[dict[str, Any]] = []
    for row in query_records:
        left, right = str(row["left_cell_id"]), str(row["right_cell_id"])
        result = dict(row)
        if left not in projection or right not in projection:
            result.update(success=False, failure_reason="unknown_probe", graph_distance_mm=math.inf, stretch=math.inf)
        else:
            li, ld = projection[left]
            ri, rd = projection[right]
            direct = float(
                np.linalg.norm(probe_xyz[probe_position[left]] - probe_xyz[probe_position[right]])
                * 1000.0
            )
            graph_distance = float(shortest[shortest_row[li], ri])
            if ld > service_radius_mm or rd > service_radius_mm:
                result.update(success=False, failure_reason="projection_too_far", graph_distance_mm=math.inf, stretch=math.inf)
            elif not math.isfinite(graph_distance):
                result.update(success=False, failure_reason="disconnected", graph_distance_mm=math.inf, stretch=math.inf)
            elif direct <= 1.0e-12:
                result.update(success=False, failure_reason="degenerate_pair", graph_distance_mm=graph_distance, stretch=math.inf)
            else:
                result.update(success=True, failure_reason="", graph_distance_mm=graph_distance, stretch=graph_distance / direct)
        rows.append(result)
    audit = pd.DataFrame(rows)
    successes = audit[audit["success"].astype(bool)] if len(audit) else audit
    return {
        "query_count": len(audit),
        "success_count": len(successes),
        "success_rate": float(len(successes) / len(audit)) if len(audit) else 0.0,
        "stretch_p95": float(np.percentile(successes["stretch"], 95)) if len(successes) else math.inf,
    }, audit


def classify_old_anchors(
    labels: pd.DataFrame,
    *,
    exact_zero_target_id: str,
    macroblock_mm: float = 40.0,
) -> pd.DataFrame:
    """Create a deterministic hard/soft/diagnostic historical-label registry."""

    frame = labels.copy()
    if not {"target_id", *XYZ_COLUMNS} <= set(frame):
        raise ValueError("old labels require target_id and signed xyz")
    bins = np.floor(frame.loc[:, XYZ_COLUMNS].to_numpy(float) * 1000.0 / float(macroblock_mm)).astype(np.int64)
    frame["macroblock_id"] = [f"smacro:{x}:{y}:{z}" for x, y, z in bins]
    frame["fresh_fk_residual_mm"] = frame.get("fresh_fk_residual_mm", frame.get("fk_residual_mm", math.inf))
    frame["repeat_pass"] = frame.get("repeat_pass", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame["multiparent_consistency_pass"] = frame.get(
        "multiparent_consistency_pass", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    frame["seam_or_conflict_risk"] = frame.get(
        "seam_or_conflict_risk", pd.Series(True, index=frame.index)
    ).fillna(True).astype(bool)
    frame["u_mm"] = frame.get("u_mm", np.nan)
    fresh = frame["fresh_fk_residual_mm"].le(3.0)
    stable = fresh & frame["repeat_pass"] & frame["multiparent_consistency_pass"] & ~frame["seam_or_conflict_risk"]
    frame["anchor_class"] = np.where(fresh & ~frame["seam_or_conflict_risk"], "soft", "diagnostic")
    frame["classification_reasons"] = np.where(
        frame["anchor_class"].eq("soft"), "fresh_fk_legal_no_registered_conflict", "missing_or_failed_fresh_stability_evidence"
    )
    frame.loc[frame["target_id"].astype(str).eq(exact_zero_target_id), ["anchor_class", "classification_reasons"]] = [
        "hard",
        "exact_zero",
    ]
    candidates = frame[stable & ~frame["target_id"].astype(str).eq(exact_zero_target_id)].copy()
    candidates = candidates.sort_values(
        ["macroblock_id", "fresh_fk_residual_mm", "target_id"], kind="stable"
    ).drop_duplicates("macroblock_id", keep="first")
    hard_ids = set(candidates["target_id"].astype(str))
    frame.loc[frame["target_id"].astype(str).isin(hard_ids), ["anchor_class", "classification_reasons"]] = [
        "hard",
        "fresh_repeat_multiparent_stable_macroblock_representative",
    ]
    columns = [
        "target_id",
        "u_mm",
        "macroblock_id",
        "anchor_class",
        "classification_reasons",
        "fresh_fk_residual_mm",
        "repeat_pass",
        "multiparent_consistency_pass",
        "seam_or_conflict_risk",
    ]
    return frame.loc[:, columns].sort_values("target_id", kind="stable").reset_index(drop=True)


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= delta, 0.5 * np.square(absolute), delta * (absolute - 0.5 * delta))


def select_fullspace_graph_teacher(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    edges: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    policy: Retry19TeacherPolicy,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """MST/tree-DP initialization followed by deterministic full-graph ICM."""

    anchor_class = anchors.set_index(anchors["target_id"].astype(str))["anchor_class"].to_dict() if len(anchors) else {}
    hard_targets = {target for target, value in anchor_class.items() if value == "hard"}
    candidate_frame = candidates.copy()
    if hard_targets:
        target_ids = candidate_frame["target_id"].astype(str)
        old_mask = candidate_frame.get(
            "old_label_candidate", pd.Series(False, index=candidate_frame.index)
        ).fillna(False).astype(bool)
        for target in sorted(hard_targets):
            matching = candidate_frame[target_ids.eq(target) & old_mask]
            if len(matching) != 1:
                raise ValueError(f"hard anchor {target} must have exactly one old-label candidate before clustering")
        candidate_frame = candidate_frame[~target_ids.isin(hard_targets) | old_mask].copy()
    legal = legal_candidate_clusters(candidate_frame, weights=policy.beta_weights)
    grouped = {
        str(target): frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        for target, frame in legal.groupby("target_id", sort=True)
    }
    if not grouped:
        return legal, {"status": "no_legal_candidates", "energy": math.inf}
    target_xyz = targets.set_index(targets["target_id"].astype(str)).loc[:, XYZ_COLUMNS]
    beta = {target: frame.loc[:, BETA_COLUMNS].to_numpy(float) for target, frame in grouped.items()}
    unary: dict[str, np.ndarray] = {}
    hard_state: dict[str, int] = {}
    for target, values in beta.items():
        frame = grouped[target]
        if "unary_cost" in frame:
            cost = frame["unary_cost"].to_numpy(float)
        else:
            cost = np.square(
                np.asarray(weighted_beta_rms_deg(values, np.zeros_like(values), policy.beta_weights), dtype=float)
            )
        if anchor_class.get(target) == "soft":
            old_mask = frame.get("old_label_candidate", pd.Series(False, index=frame.index)).astype(bool).to_numpy()
            cost = cost - policy.soft_anchor_bonus * old_mask.astype(float)
        if anchor_class.get(target) == "hard":
            old_mask = frame.get("old_label_candidate", pd.Series(False, index=frame.index)).astype(bool).to_numpy()
            matching = np.flatnonzero(old_mask)
            if len(matching) != 1:
                raise ValueError(f"hard anchor {target} must have exactly one old-label candidate")
            hard_state[target] = int(matching[0])
            cost = np.where(np.arange(len(cost)) == hard_state[target], cost, math.inf)
        unary[target] = np.asarray(cost, dtype=float)

    raw_edges: list[tuple[str, str, float]] = []
    for row in edges.to_dict("records"):
        left, right = str(row["left_target_id"]), str(row["right_target_id"])
        if left not in grouped or right not in grouped or left == right:
            continue
        if "distance_mm" in row:
            distance = float(row["distance_mm"])
        else:
            distance = float(np.linalg.norm(target_xyz.loc[left] - target_xyz.loc[right]) * 1000.0)
        raw_edges.append((*sorted((left, right)), distance))
    raw_edges = sorted(set(raw_edges), key=lambda value: (value[2], value[0], value[1]))
    weighted_degree = {target: 0.0 for target in grouped}
    gaussian: dict[tuple[str, str], float] = {}
    for left, right, distance in raw_edges:
        value = math.exp(-(distance * distance) / (2.0 * policy.sigma_mm * policy.sigma_mm))
        gaussian[(left, right)] = value
        weighted_degree[left] += value
        weighted_degree[right] += value
    coefficients = {
        pair: value / math.sqrt(max(weighted_degree[pair[0]] * weighted_degree[pair[1]], 1.0e-24))
        for pair, value in gaussian.items()
    }
    pair_cost: dict[tuple[str, str], np.ndarray] = {}
    adjacency: dict[str, set[str]] = {target: set() for target in grouped}
    for left, right, _distance in raw_edges:
        gap = np.empty((len(beta[left]), len(beta[right])), dtype=float)
        for index, left_beta in enumerate(beta[left]):
            gap[index] = weighted_beta_rms_deg(
                np.broadcast_to(left_beta, beta[right].shape), beta[right], policy.beta_weights
            )
        pair_cost[(left, right)] = coefficients[(left, right)] * _huber(gap, policy.huber_delta_deg)
        adjacency[left].add(right)
        adjacency[right].add(left)

    parent = {target: target for target in grouped}

    def find(target: str) -> str:
        while parent[target] != target:
            parent[target] = parent[parent[target]]
            target = parent[target]
        return target

    tree = {target: set() for target in grouped}
    for left, right, _distance in raw_edges:
        a, b = find(left), find(right)
        if a == b:
            continue
        parent[b] = a
        tree[left].add(right)
        tree[right].add(left)

    initial: dict[str, int] = {}
    bfs_orders: list[list[str]] = []
    seen: set[str] = set()
    preferred_roots = sorted(grouped, key=lambda target: (target not in hard_state, target))
    for root in preferred_roots:
        if root in seen:
            continue
        traversal: list[str] = []
        tree_parent: dict[str, str | None] = {root: None}
        queue = [root]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            traversal.append(current)
            for child in sorted(tree[current] - {tree_parent[current]}):
                tree_parent[child] = current
                queue.append(child)
        bfs_orders.append(traversal)
        subtree: dict[str, np.ndarray] = {}
        backtrack: dict[tuple[str, str], np.ndarray] = {}
        for target in reversed(traversal):
            cost = unary[target].copy()
            for child in sorted(tree[target] - {tree_parent[target]}):
                pair = tuple(sorted((target, child)))
                matrix = pair_cost[pair] if target == pair[0] else pair_cost[pair].T
                total = subtree[child][None, :] + policy.pairwise_lambda * matrix
                choice = np.argmin(total, axis=1).astype(int)
                cost += total[np.arange(len(cost)), choice]
                backtrack[(target, child)] = choice
            subtree[target] = cost
        initial[root] = hard_state.get(root, int(np.argmin(subtree[root])))
        for target in traversal:
            for child in sorted(tree[target] - {tree_parent[target]}):
                initial[child] = int(backtrack[(target, child)][initial[target]])

    node_ids = sorted(grouped)

    def objective(selection: Mapping[str, int]) -> float:
        value = sum(float(unary[target][selection[target]]) for target in node_ids)
        for left, right, _distance in raw_edges:
            value += policy.pairwise_lambda * float(pair_cost[(left, right)][selection[left], selection[right]])
        return value

    bfs_order = [target for traversal in bfs_orders for target in traversal]
    orders = (
        node_ids,
        sorted(node_ids, key=lambda target: (-weighted_degree[target], target)),
        bfs_order,
    )
    solutions: list[tuple[float, tuple[str, ...], dict[str, int], int]] = []
    for order_index, order in enumerate(orders):
        selection = dict(initial)
        for _ in range(policy.maximum_icm_sweeps):
            changed = False
            for target in order:
                if target in hard_state:
                    continue
                costs = unary[target].copy()
                for neighbour in adjacency[target]:
                    pair = tuple(sorted((target, neighbour)))
                    matrix = pair_cost[pair]
                    costs += policy.pairwise_lambda * (
                        matrix[:, selection[neighbour]]
                        if target == pair[0]
                        else matrix[selection[neighbour], :]
                    )
                chosen = int(np.argmin(costs))
                changed = changed or chosen != selection[target]
                selection[target] = chosen
            if not changed:
                break
        signature = tuple(str(grouped[target].iloc[selection[target]]["candidate_id"]) for target in node_ids)
        solutions.append((objective(selection), signature, selection, order_index))
    energy, signature, selected, order_index = min(solutions, key=lambda value: (value[0], value[1]))
    rows = [grouped[target].iloc[[selected[target]]].copy() for target in node_ids]
    result = pd.concat(rows, ignore_index=True, sort=False)
    result["teacher"] = "retry19_mst_dp_full_graph_icm"
    result["pairwise_lambda"] = policy.pairwise_lambda
    result["graph_energy"] = energy
    return result, {
        "status": "complete",
        "energy": energy,
        "assignment_signature": list(signature),
        "selected_icm_order": ("target_id", "degree", "root_bfs")[order_index],
        "mst_dp_initialized": True,
        "hard_anchor_count": len(hard_state),
        "soft_anchor_bonus": policy.soft_anchor_bonus,
        "pairwise_normalization": "w_ij/sqrt(weighted_degree_i*weighted_degree_j)",
    }


def overlap_compatibility(
    direct_labels: pd.DataFrame,
    retry18_labels: pd.DataFrame,
    *,
    weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
    p95_maximum_deg: float = 3.0,
    gt7_rate_maximum: float = 0.02,
) -> tuple[Mapping[str, Any], pd.DataFrame]:
    merged = direct_labels.merge(
        retry18_labels,
        on="target_id",
        suffixes=("_direct", "_retry18"),
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return {"compatible": False, "reason": "no_exact_target_overlap", "target_count": 0}, merged
    left = merged[[f"{column}_direct" for column in BETA_COLUMNS]].to_numpy(float)
    right = merged[[f"{column}_retry18" for column in BETA_COLUMNS]].to_numpy(float)
    gap = np.asarray(weighted_beta_rms_deg(left, right, weights), dtype=float)
    merged["weighted_beta_gap_deg"] = gap
    p95 = float(np.percentile(gap, 95))
    rate = float(np.mean(gap > 7.0))
    return {
        "compatible": bool(p95 <= p95_maximum_deg and rate <= gt7_rate_maximum),
        "target_count": len(merged),
        "weighted_gap_p95_deg": p95,
        "weighted_gap_gt7_rate": rate,
        "p95_maximum_deg": p95_maximum_deg,
        "gt7_rate_maximum": gt7_rate_maximum,
    }, merged


def data_gate(
    coverage: Mapping[str, float | int],
    *,
    maximum_hole_fraction: float,
    minimum_slice_coverage: float,
    path_metrics: Mapping[str, float | int],
    geometric_zero_connected: bool,
    teacher_zero_connected: bool,
    nearest_zero_nonzero_mm: float = math.inf,
    policy: FullspaceCoveragePolicy = FullspaceCoveragePolicy(),
) -> Mapping[str, Any]:
    checks = {
        "volume_coverage": float(coverage["volume_coverage"]) >= policy.volume_coverage_minimum,
        "service_distance_p95": (
            float(coverage["service_distance_volume_weighted_p95_mm"]) <= policy.service_radius_mm
            or math.isclose(
                float(coverage["service_distance_volume_weighted_p95_mm"]),
                policy.service_radius_mm,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ),
        "nearest_zero_nonzero": float(nearest_zero_nonzero_mm) <= policy.root_registration_mm,
        "geometric_zero_connected": bool(geometric_zero_connected),
        "teacher_zero_connected": bool(teacher_zero_connected),
    }
    diagnostics = {
        "fine_5mm_coverage": float(coverage["fine_5mm_coverage"]),
        "coarse_10mm_coverage": float(coverage["coarse_10mm_coverage"]),
        "maximum_hole_fraction": float(maximum_hole_fraction),
        "minimum_axial_slice_coverage": float(minimum_slice_coverage),
        "path_success_rate": float(path_metrics["success_rate"]),
        "path_stretch_p95": float(path_metrics["stretch_p95"]),
    }
    return {
        "status": "green" if all(checks.values()) else "red",
        "checks": checks,
        "diagnostics": diagnostics,
        **checks,
    }


__all__ = [
    "BOX_MAX_COLUMNS",
    "BOX_MIN_COLUMNS",
    "FullspaceCoveragePolicy",
    "Retry19TeacherPolicy",
    "axial_slice_coverage",
    "box_to_point_distance_mm",
    "build_mixed_resolution_cell_registry",
    "classify_old_anchors",
    "data_gate",
    "deterministic_path_query_registry",
    "geometric_zero_attachment",
    "largest_unserved_component_fraction",
    "mark_served_cells",
    "mixed_resolution_coverage_metrics",
    "mixed_resolution_26_edges",
    "mixed_resolution_face_edges",
    "overlap_compatibility",
    "path_query_metrics",
    "select_fullspace_graph_teacher",
    "teacher_zero_attachment",
    "target_knn_edges",
]
