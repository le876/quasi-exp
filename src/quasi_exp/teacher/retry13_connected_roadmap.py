"""Deterministic connected-roadmap contracts for BACRA V14.3R retry13.

The module owns graph construction, component selection, tree orientation,
connected leaf pruning, and symmetry expansion.  Numerical continuation and
artifact I/O stay in the runner so their cost and provenance remain explicit.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.exploration_qualification import weighted_beta_rms_deg
from quasi_exp.teacher.retry12_symmetry import (
    BETA_COLUMNS,
    SYMMETRY_BETA_SIGNS,
    THETA_COLUMNS,
    XYZ_COLUMNS,
    stable_id,
    transform_beta,
    transform_xyz,
)


def radius_edges(frame: pd.DataFrame, *, radius_mm: float) -> pd.DataFrame:
    """Return deterministic undirected task-space edges within ``radius_mm``."""

    if frame.empty:
        return pd.DataFrame(columns=["left_position", "right_position", "xyz_length_mm"])
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float)
    pairs = sorted(cKDTree(xyz).query_pairs(float(radius_mm) / 1000.0))
    if not pairs:
        return pd.DataFrame(columns=["left_position", "right_position", "xyz_length_mm"])
    array = np.asarray(pairs, dtype=np.int64)
    length = np.linalg.norm(xyz[array[:, 0]] - xyz[array[:, 1]], axis=1) * 1000.0
    return pd.DataFrame(
        {
            "left_position": array[:, 0],
            "right_position": array[:, 1],
            "xyz_length_mm": length,
        }
    )


def annotate_beta_gaps(frame: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    result = edges.copy()
    if result.empty:
        result["weighted_beta_gap_deg"] = pd.Series(dtype=float)
        result["raw_beta_gap_deg"] = pd.Series(dtype=float)
        return result
    beta = frame.loc[:, BETA_COLUMNS].to_numpy(float)
    left = result["left_position"].to_numpy(np.int64)
    right = result["right_position"].to_numpy(np.int64)
    result["weighted_beta_gap_deg"] = weighted_beta_rms_deg(beta[left], beta[right])
    result["raw_beta_gap_deg"] = np.max(np.abs(np.degrees(beta[left] - beta[right])), axis=1)
    return result


def component_registry(
    node_count: int,
    edges: pd.DataFrame,
    *,
    left_column: str = "left_position",
    right_column: str = "right_position",
) -> tuple[np.ndarray, np.ndarray]:
    """Return component labels and component sizes, retaining isolated nodes."""

    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    if node_count == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    if edges.empty:
        return np.arange(node_count, dtype=np.int64), np.ones(node_count, dtype=np.int64)
    left = edges[left_column].to_numpy(np.int64)
    right = edges[right_column].to_numpy(np.int64)
    row = np.concatenate([left, right])
    column = np.concatenate([right, left])
    graph = coo_matrix((np.ones(len(row)), (row, column)), shape=(node_count, node_count)).tocsr()
    count, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64), np.bincount(labels, minlength=int(count)).astype(np.int64)


def zero_compatible_component(
    frame: pd.DataFrame,
    *,
    zero_position: int,
    maximum_edge_mm: float,
    weighted_max_deg: float,
    raw_max_deg: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Select the zero-containing local-consistency component."""

    edges = annotate_beta_gaps(frame, radius_edges(frame, radius_mm=maximum_edge_mm))
    admitted = edges[
        edges["weighted_beta_gap_deg"].le(float(weighted_max_deg) + 1.0e-12)
        & edges["raw_beta_gap_deg"].le(float(raw_max_deg) + 1.0e-12)
    ].copy()
    labels, sizes = component_registry(len(frame), admitted)
    zero_label = int(labels[int(zero_position)])
    keep = labels == zero_label
    report = {
        "node_count": int(len(frame)),
        "candidate_edge_count": int(len(edges)),
        "compatible_edge_count": int(len(admitted)),
        "component_count": int(len(sizes)),
        "zero_component_row_count": int(np.sum(keep)),
        "rejected_component_row_count": int(len(frame) - np.sum(keep)),
        "component_sizes_descending": sorted(map(int, sizes), reverse=True),
    }
    admitted["left_component_id"] = labels[admitted["left_position"].to_numpy(np.int64)]
    admitted["right_component_id"] = labels[admitted["right_position"].to_numpy(np.int64)]
    return keep, admitted.reset_index(drop=True), report


def existing_parent_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover retry12 continuation edges whose parent is in ``frame``."""

    id_to_position = {
        str(value): int(position)
        for position, value in enumerate(frame["physical_point_id"].astype(str))
    }
    rows: list[dict[str, Any]] = []
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float)
    beta = frame.loc[:, BETA_COLUMNS].to_numpy(float)
    for child_position, row in frame.iterrows():
        parent_id = row.get("source_physical_point_id")
        if pd.isna(parent_id) or str(parent_id) not in id_to_position:
            continue
        parent_position = id_to_position[str(parent_id)]
        raw_gap = float(np.max(np.abs(np.degrees(beta[parent_position] - beta[int(child_position)]))))
        rows.append(
            {
                "fundamental_edge_id": stable_id(
                    "retry13_inherited_edge",
                    frame.iloc[parent_position]["physical_point_id"],
                    row["physical_point_id"],
                ),
                "left_position": parent_position,
                "right_position": int(child_position),
                "source_physical_point_id": str(frame.iloc[parent_position]["physical_point_id"]),
                "target_physical_point_id": str(row["physical_point_id"]),
                "xyz_length_mm": float(np.linalg.norm(xyz[parent_position] - xyz[int(child_position)]) * 1000.0),
                "weighted_beta_gap_deg": float(weighted_beta_rms_deg(beta[parent_position], beta[int(child_position)])),
                "raw_beta_gap_deg": raw_gap,
                "reverse_gap_deg": float(row.get("reverse_gap_deg", np.inf)),
                "certificate_origin": "retry12_continuation",
                "certificate_status": "inherited",
            }
        )
    return pd.DataFrame.from_records(rows)


def bridge_candidates(
    frame: pd.DataFrame,
    parent_edges: pd.DataFrame,
    *,
    maximum_edge_mm: float,
    weighted_max_deg: float,
    raw_max_deg: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return compatible cross-forest bridge candidates in stable order."""

    forest_labels, forest_sizes = component_registry(len(frame), parent_edges)
    edges = annotate_beta_gaps(frame, radius_edges(frame, radius_mm=maximum_edge_mm))
    left = edges["left_position"].to_numpy(np.int64)
    right = edges["right_position"].to_numpy(np.int64)
    mask = (
        (forest_labels[left] != forest_labels[right])
        & edges["weighted_beta_gap_deg"].le(float(weighted_max_deg) + 1.0e-12).to_numpy(bool)
        & edges["raw_beta_gap_deg"].le(float(raw_max_deg) + 1.0e-12).to_numpy(bool)
    )
    result = edges[mask].copy()
    result["left_component_id"] = forest_labels[result["left_position"].to_numpy(np.int64)]
    result["right_component_id"] = forest_labels[result["right_position"].to_numpy(np.int64)]
    result["left_physical_point_id"] = frame.iloc[result["left_position"].to_numpy(np.int64)][
        "physical_point_id"
    ].astype(str).to_numpy()
    result["right_physical_point_id"] = frame.iloc[result["right_position"].to_numpy(np.int64)][
        "physical_point_id"
    ].astype(str).to_numpy()
    result = result.sort_values(
        [
            "weighted_beta_gap_deg",
            "raw_beta_gap_deg",
            "xyz_length_mm",
            "left_physical_point_id",
            "right_physical_point_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return result, forest_labels, forest_sizes


def minimum_spanning_component_bridges(
    certified: pd.DataFrame,
    *,
    component_ids: Sequence[int],
) -> pd.DataFrame:
    """Kruskal-select a deterministic bridge tree over parent components."""

    components = sorted(set(map(int, component_ids)))
    position = {component: index for index, component in enumerate(components)}
    parent = list(range(len(components)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> bool:
        a = find(left)
        b = find(right)
        if a == b:
            return False
        parent[b] = a
        return True

    ordered = certified.sort_values(
        [
            "weighted_beta_gap_deg",
            "raw_beta_gap_deg",
            "xyz_length_mm",
            "source_physical_point_id",
            "target_physical_point_id",
        ],
        kind="stable",
    )
    selected: list[dict[str, Any]] = []
    for row in ordered.to_dict("records"):
        left = int(row["left_component_id"])
        right = int(row["right_component_id"])
        if left not in position or right not in position:
            continue
        if union(position[left], position[right]):
            selected.append(row)
    return pd.DataFrame.from_records(selected, columns=certified.columns)


def orient_certified_tree(
    frame: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    zero_physical_point_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orient an undirected certified tree from exact zero."""

    ids = frame["physical_point_id"].astype(str).tolist()
    id_to_position = {value: position for position, value in enumerate(ids)}
    if zero_physical_point_id not in id_to_position:
        raise ValueError("zero physical point is absent")
    adjacency: dict[int, list[tuple[int, int]]] = {position: [] for position in range(len(frame))}
    for edge_position, row in edges.reset_index(drop=True).iterrows():
        left = int(row["left_position"])
        right = int(row["right_position"])
        adjacency[left].append((right, int(edge_position)))
        adjacency[right].append((left, int(edge_position)))
    root = id_to_position[str(zero_physical_point_id)]
    parent_position = np.full(len(frame), -2, dtype=np.int64)
    parent_edge = np.full(len(frame), -1, dtype=np.int64)
    depth = np.full(len(frame), -1, dtype=np.int64)
    parent_position[root] = -1
    depth[root] = 0
    queue: deque[int] = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor, edge_position in sorted(adjacency[current], key=lambda item: ids[item[0]]):
            if parent_position[neighbor] != -2:
                continue
            parent_position[neighbor] = current
            parent_edge[neighbor] = edge_position
            depth[neighbor] = depth[current] + 1
            queue.append(neighbor)
    if np.any(parent_position == -2):
        raise ValueError("certified graph is not connected to zero")
    nodes = frame.copy()
    nodes["retry13_parent_physical_point_id"] = [
        None if value < 0 else ids[int(value)] for value in parent_position
    ]
    nodes["retry13_parent_edge_id"] = [
        None if value < 0 else str(edges.iloc[int(value)]["fundamental_edge_id"])
        for value in parent_edge
    ]
    nodes["retry13_lineage_depth"] = depth
    oriented = edges.copy().reset_index(drop=True)
    orientation: list[str] = ["unused"] * len(oriented)
    for child_position, edge_position in enumerate(parent_edge):
        if edge_position < 0:
            continue
        row = oriented.iloc[int(edge_position)]
        parent_id = ids[int(parent_position[child_position])]
        child_id = ids[int(child_position)]
        orientation[int(edge_position)] = (
            "stored" if str(row["source_physical_point_id"]) == parent_id else "reversed"
        )
        oriented.loc[int(edge_position), "oriented_source_physical_point_id"] = parent_id
        oriented.loc[int(edge_position), "oriented_target_physical_point_id"] = child_id
    oriented["tree_orientation"] = orientation
    return nodes, oriented


def connected_leaf_prune(
    representatives: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    maximum_expanded_rows: int,
    protected_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Remove only unprotected leaf orbits until the expanded budget is met."""

    nodes = representatives.copy().reset_index(drop=True)
    links = edges.copy().reset_index(drop=True)
    protected = set(map(str, protected_ids))
    removed: list[str] = []
    while int(nodes["orbit_size"].sum()) > int(maximum_expanded_rows):
        degree = pd.concat(
            [links["source_physical_point_id"], links["target_physical_point_id"]]
        ).value_counts()
        candidates = nodes[
            nodes["physical_point_id"].astype(str).map(degree).fillna(0).le(1)
            & ~nodes["physical_point_id"].astype(str).isin(protected)
        ].sort_values(
            ["zero_radius_mm", "physical_point_id"], ascending=[False, True], kind="stable"
        )
        if candidates.empty:
            break
        victim = str(candidates.iloc[0]["physical_point_id"])
        removed.append(victim)
        nodes = nodes[~nodes["physical_point_id"].astype(str).eq(victim)].copy()
        links = links[
            ~links["source_physical_point_id"].astype(str).eq(victim)
            & ~links["target_physical_point_id"].astype(str).eq(victim)
        ].copy()
    return nodes.reset_index(drop=True), links.reset_index(drop=True), removed


def expand_connected_orbits(
    representatives: pd.DataFrame,
    fundamental_edges: pd.DataFrame,
    environment: Any,
    *,
    zero_physical_point_id: str,
    fk_max_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expand nodes and certified edges under G4 while merging exact zero."""

    group = tuple(SYMMETRY_BETA_SIGNS)
    bounds = np.asarray(environment.bounds, dtype=float)
    member_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    member_map: dict[tuple[str, str], str] = {}
    for row in representatives.to_dict("records"):
        representative_id = str(row["physical_point_id"])
        xyz = np.asarray([row[name] for name in XYZ_COLUMNS], dtype=float)
        beta = np.asarray([row[name] for name in BETA_COLUMNS], dtype=float)
        unique: dict[tuple[float, ...], str] = {}
        staged: list[tuple[str, dict[str, Any], tuple[float, ...]]] = []
        for element in group:
            member_xyz = transform_xyz(xyz, element)
            member_beta = transform_beta(beta, element)
            key = tuple(np.round(np.concatenate([member_xyz, member_beta]), 12))
            if key not in unique:
                unique[key] = stable_id("retry13_full_node", representative_id, element)
            member_id = unique[key]
            member_map[(representative_id, element)] = member_id
            if any(existing[2] == key for existing in staged):
                continue
            observed = np.asarray(environment.fk(member_beta), dtype=float).reshape(-1, 3)[0]
            residual_mm = float(np.linalg.norm(observed - member_xyz) * 1000.0)
            valid = bool(
                np.isfinite(member_beta).all()
                and np.all(member_beta >= bounds[:, 0] - 1.0e-12)
                and np.all(member_beta <= bounds[:, 1] + 1.0e-12)
                and residual_mm <= float(fk_max_mm) + 1.0e-12
            )
            staged.append(
                (
                    element,
                    {
                        **row,
                        **dict(zip(XYZ_COLUMNS, member_xyz, strict=True)),
                        **dict(zip(BETA_COLUMNS, member_beta, strict=True)),
                        **dict(zip(THETA_COLUMNS, beta_to_theta(member_beta), strict=True)),
                        "physical_point_id": member_id,
                        "fundamental_physical_point_id": representative_id,
                        "symmetry_orbit_id": stable_id("retry13_orbit", representative_id),
                        "symmetry_element": element,
                        "fk_residual_mm": residual_mm,
                        "orbit_size": len(unique),
                    },
                    key,
                )
            )
            if not valid:
                rejected.append(
                    {
                        "fundamental_physical_point_id": representative_id,
                        "symmetry_element": element,
                        "reason": "invalid_orbit_member",
                        "fk_residual_mm": residual_mm,
                    }
                )
        orbit_size = len(unique)
        for _, member, _ in staged:
            member["orbit_size"] = orbit_size
            member_rows.append(member)
        for element in group:
            transformed_xyz = transform_xyz(xyz, element)
            transformed_beta = transform_beta(beta, element)
            key = tuple(np.round(np.concatenate([transformed_xyz, transformed_beta]), 12))
            member_map[(representative_id, element)] = unique[key]
    expanded = pd.DataFrame.from_records(member_rows)
    edge_rows: list[dict[str, Any]] = []
    for edge in fundamental_edges.to_dict("records"):
        source_id = str(edge["oriented_source_physical_point_id"])
        target_id = str(edge["oriented_target_physical_point_id"])
        for element in group:
            full_source = member_map[(source_id, element)]
            full_target = member_map[(target_id, element)]
            if full_source == full_target:
                continue
            edge_rows.append(
                {
                    **edge,
                    "full_edge_id": stable_id("retry13_full_edge", edge["fundamental_edge_id"], element),
                    "source_physical_point_id": full_source,
                    "target_physical_point_id": full_target,
                    "symmetry_element": element,
                }
            )
    full_edges = pd.DataFrame.from_records(edge_rows)
    if not full_edges.empty:
        undirected = full_edges.apply(
            lambda row: "|".join(sorted((str(row["source_physical_point_id"]), str(row["target_physical_point_id"])))),
            axis=1,
        )
        full_edges = full_edges.loc[~undirected.duplicated()].reset_index(drop=True)
    rejected_frame = pd.DataFrame.from_records(rejected)
    if not rejected_frame.empty:
        return expanded, full_edges, rejected_frame
    zero_rows = expanded[expanded["fundamental_physical_point_id"].astype(str).eq(str(zero_physical_point_id))]
    if len(zero_rows) != 1:
        raise ValueError("expanded graph must contain exactly one shared zero node")
    return expanded.reset_index(drop=True), full_edges.reset_index(drop=True), rejected_frame


__all__ = [
    "annotate_beta_gaps",
    "bridge_candidates",
    "component_registry",
    "connected_leaf_prune",
    "existing_parent_edges",
    "expand_connected_orbits",
    "minimum_spanning_component_bridges",
    "orient_certified_tree",
    "radius_edges",
    "zero_compatible_component",
]
