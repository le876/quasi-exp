"""Zero-component label classification and bounded bridge contracts for retry11."""

from __future__ import annotations

from collections import deque
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .retry10 import (
    BETA_COLUMNS,
    XYZ_COLUMNS,
    normalized_weighted_beta_deg,
    raw_beta_max_deg,
)
from .retry11_sampling import stable_row_id


FAILURE_CLASSES = (
    "solver_failure",
    "local_consistency_failure",
    "branch_conflict",
    "target_infeasible_observed",
    "budget_exceeded",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def discover_graph_components(
    task_nodes: pd.DataFrame,
    task_edges: pd.DataFrame,
    *,
    zero_xyz_m: Sequence[float],
    outer_task_node_ids: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Discover the graph components containing zero and the outer atlas."""

    _require_columns(task_nodes, ("task_node_id", *XYZ_COLUMNS), "task nodes")
    _require_columns(task_edges, ("left_node_id", "right_node_id"), "task edges")
    nodes = task_nodes.drop_duplicates("task_node_id").copy()
    node_ids = tuple(sorted(nodes["task_node_id"].astype(int)))
    known = set(node_ids)
    adjacency = {node: [] for node in node_ids}
    for row in task_edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left in known and right in known and left != right:
            adjacency[left].append(right)
            adjacency[right].append(left)
    component_by_node: dict[int, int] = {}
    components: list[list[int]] = []
    for start in node_ids:
        if start in component_by_node:
            continue
        component_index = len(components)
        queue = deque([start])
        component_by_node[start] = component_index
        members: list[int] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in component_by_node:
                    component_by_node[neighbor] = component_index
                    queue.append(neighbor)
        components.append(sorted(members))

    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    xyz = nodes.loc[:, XYZ_COLUMNS].to_numpy(float)
    distances = np.linalg.norm(xyz - zero.reshape(1, 3), axis=1)
    nearest_position = min(
        range(len(nodes)),
        key=lambda index: (float(distances[index]), int(nodes.iloc[index]["task_node_id"])),
    )
    nearest_zero_node = int(nodes.iloc[nearest_position]["task_node_id"])
    zero_component_index = int(component_by_node[nearest_zero_node])
    outer_ids = [int(value) for value in outer_task_node_ids if int(value) in component_by_node]
    if not outer_ids:
        raise ValueError("outer atlas has no task nodes in the registered graph")
    outer_counts = pd.Series([component_by_node[value] for value in outer_ids]).value_counts()
    outer_component_index = int(outer_counts.index[0])

    registry = nodes.copy()
    registry["graph_component_index"] = registry["task_node_id"].astype(int).map(component_by_node)
    registry["graph_component_id"] = registry["graph_component_index"].map(
        lambda value: f"graph_component_{int(value):03d}"
    )
    registry["is_zero_component"] = registry["graph_component_index"].eq(zero_component_index)
    registry["is_outer_component"] = registry["graph_component_index"].eq(outer_component_index)
    registry["distance_to_zero_mm"] = distances * 1000.0

    zero_rows = registry[registry["is_zero_component"]]
    outer_rows = registry[registry["is_outer_component"]]
    outer_label_rows = registry[registry["task_node_id"].astype(int).isin(outer_ids)]
    outer_tree = cKDTree(outer_label_rows.loc[:, XYZ_COLUMNS].to_numpy(float))
    nearest_distance, nearest_outer_index = outer_tree.query(
        zero_rows.loc[:, XYZ_COLUMNS].to_numpy(float), k=1
    )
    nearest_zero_position = int(np.argmin(nearest_distance))
    nearest_zero_row = zero_rows.iloc[nearest_zero_position]
    nearest_outer_row = outer_label_rows.iloc[int(nearest_outer_index[nearest_zero_position])]
    report: dict[str, object] = {
        "graph_component_count": len(components),
        "zero_component_graph_id": f"graph_component_{zero_component_index:03d}",
        "zero_component_node_count": len(components[zero_component_index]),
        "outer_component_graph_id": f"graph_component_{outer_component_index:03d}",
        "outer_component_node_count": len(components[outer_component_index]),
        "outer_frozen_label_count": len(outer_ids),
        "outer_internal_coverage_fraction": len(set(outer_ids))
        / max(1, len(components[outer_component_index])),
        "zero_nearest_task_node_id": nearest_zero_node,
        "zero_nearest_task_distance_mm": float(distances[nearest_position] * 1000.0),
        "zero_outer_same_graph_component": zero_component_index == outer_component_index,
        "nearest_zero_boundary_task_node_id": int(nearest_zero_row["task_node_id"]),
        "nearest_outer_frozen_task_node_id": int(nearest_outer_row["task_node_id"]),
        "nearest_zero_outer_boundary_distance_mm": float(
            nearest_distance[nearest_zero_position] * 1000.0
        ),
    }
    return registry.sort_values("task_node_id", kind="stable").reset_index(drop=True), report


def zero_seed_candidate_registry(
    component_registry: pd.DataFrame,
    *,
    candidate_count: int = 8,
) -> pd.DataFrame:
    _require_columns(
        component_registry,
        ("task_node_id", "is_zero_component", "distance_to_zero_mm", *XYZ_COLUMNS),
        "component registry",
    )
    count = int(candidate_count)
    if count < 1:
        raise ValueError("candidate_count must be positive")
    zero = component_registry[component_registry["is_zero_component"].astype(bool)].copy()
    result = zero.sort_values(
        ["distance_to_zero_mm", "task_node_id"], kind="stable"
    ).head(count)
    result["zero_seed_candidate_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def _successful_attempts(frame: pd.DataFrame, *, fk_max_mm: float) -> pd.DataFrame:
    success = frame[
        frame["solver_success"].astype(bool)
        & frame["actual_bounds"].astype(bool)
        & frame["fk_residual_mm"].astype(float).le(float(fk_max_mm) + 1.0e-12)
    ].copy()
    if success.empty:
        return success
    finite = np.isfinite(success.loc[:, BETA_COLUMNS].to_numpy(float)).all(axis=1)
    finite &= np.isfinite(success["fk_residual_mm"].to_numpy(float))
    return success.loc[finite].copy()


def _gold_from_attempts(group: pd.DataFrame) -> tuple[dict[str, object] | None, str | None]:
    success = _successful_attempts(group, fk_max_mm=5.0)
    if len(success) < 2 or success["source_path_id"].astype(str).nunique() < 2:
        return None, None
    ordered = success.sort_values("source_path_id", kind="stable").reset_index(drop=True)
    beta = ordered.loc[:, BETA_COLUMNS].to_numpy(float)
    weighted = np.zeros((len(beta), len(beta)), dtype=float)
    raw = np.zeros_like(weighted)
    for left in range(len(beta)):
        for right in range(left + 1, len(beta)):
            weighted[left, right] = weighted[right, left] = normalized_weighted_beta_deg(
                beta[left], beta[right]
            )
            raw[left, right] = raw[right, left] = raw_beta_max_deg(beta[left], beta[right])
    raw_max = float(np.max(raw))
    weighted_max = float(np.max(weighted))
    if raw_max > 5.0 + 1.0e-12:
        return None, "branch_conflict"
    if weighted_max > 2.0 + 1.0e-12:
        return None, "same_point_weighted_incompatible"
    chosen_index = min(
        range(len(ordered)),
        key=lambda index: (
            float(np.sum(weighted[index])),
            float(ordered.iloc[index]["fk_residual_mm"]),
            str(ordered.iloc[index]["source_path_id"]),
        ),
    )
    chosen = ordered.iloc[chosen_index]
    record = chosen.to_dict()
    record.update(
        {
            "label_quality": "Gold",
            "same_point_weighted_gap_deg": weighted_max,
            "same_point_raw_gap_deg": raw_max,
            "supporting_source_path_count": int(ordered["source_path_id"].astype(str).nunique()),
            "wide_silver_reverse_gap_deg": None,
            "wide_silver_neighbor_weighted_max_deg": None,
            "wide_silver_neighbor_raw_max_deg": None,
        }
    )
    return record, None


def _wide_silver_from_attempt(
    group: pd.DataFrame,
    retained: pd.DataFrame,
) -> tuple[dict[str, object] | None, str | None]:
    success = _successful_attempts(group, fk_max_mm=10.0)
    if len(success) != 1:
        return None, None
    chosen = success.iloc[0]
    reverse_gap = float(chosen.get("reverse_gap_deg", math.inf))
    if reverse_gap > 1.0 + 1.0e-12:
        return None, "reverse_return_failed"
    if len(retained) < 2:
        return None, "insufficient_neighbors"
    target_xyz = chosen.loc[list(XYZ_COLUMNS)].to_numpy(float)
    neighbor = retained.copy()
    neighbor["_distance"] = np.linalg.norm(
        neighbor.loc[:, XYZ_COLUMNS].to_numpy(float) - target_xyz.reshape(1, 3), axis=1
    )
    neighbor = neighbor.sort_values(["_distance", "physical_point_id"], kind="stable").head(2)
    if len(neighbor) < 2:
        return None, "insufficient_neighbors"
    beta = chosen.loc[list(BETA_COLUMNS)].to_numpy(float)
    weighted = [
        normalized_weighted_beta_deg(beta, row)
        for row in neighbor.loc[:, BETA_COLUMNS].to_numpy(float)
    ]
    raw = [
        raw_beta_max_deg(beta, row)
        for row in neighbor.loc[:, BETA_COLUMNS].to_numpy(float)
    ]
    if max(weighted) > 3.0 + 1.0e-12 or max(raw) > 7.0 + 1.0e-12:
        return None, "local_consistency_failure"
    record = chosen.to_dict()
    record.update(
        {
            "label_quality": "Wide-Silver",
            "same_point_weighted_gap_deg": None,
            "same_point_raw_gap_deg": None,
            "supporting_source_path_count": 1,
            "wide_silver_reverse_gap_deg": reverse_gap,
            "wide_silver_neighbor_weighted_max_deg": float(max(weighted)),
            "wide_silver_neighbor_raw_max_deg": float(max(raw)),
        }
    )
    return record, None


def classify_retry11_targets(
    attempts: pd.DataFrame,
    retained_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify Gold first, then iteratively admit Wide-Silver labels."""

    required = (
        "target_id",
        "source_path_id",
        "solver_success",
        "actual_bounds",
        "fk_residual_mm",
        "reverse_gap_deg",
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
    )
    _require_columns(attempts, required, "retry11 attempts")
    _require_columns(retained_labels, ("physical_point_id", *XYZ_COLUMNS, *BETA_COLUMNS), "retained labels")
    accepted: list[dict[str, object]] = []
    rejected_reason: dict[object, str] = {}
    groups = {key: group.copy() for key, group in attempts.groupby("target_id", sort=True)}

    for target_id, group in groups.items():
        record, reason = _gold_from_attempts(group)
        if record is not None:
            accepted.append(record)
        elif reason is not None:
            rejected_reason[target_id] = reason

    retained = pd.concat(
        [retained_labels.copy(), pd.DataFrame.from_records(accepted)],
        ignore_index=True,
        sort=False,
    )
    pending = [key for key in sorted(groups, key=str) if key not in {row["target_id"] for row in accepted} and key not in rejected_reason]
    while pending:
        progressed = False
        next_pending: list[object] = []
        for target_id in pending:
            record, reason = _wide_silver_from_attempt(groups[target_id], retained)
            if record is not None:
                accepted.append(record)
                retained = pd.concat([retained, pd.DataFrame.from_records([record])], ignore_index=True, sort=False)
                progressed = True
            elif reason == "insufficient_neighbors":
                next_pending.append(target_id)
            else:
                rejected_reason[target_id] = reason or "not_gold_or_wide_silver"
        if not progressed:
            for target_id in next_pending:
                rejected_reason[target_id] = "insufficient_neighbors"
            break
        pending = next_pending

    labels = pd.DataFrame.from_records(accepted)
    if not labels.empty:
        if "physical_point_id" not in labels:
            labels["physical_point_id"] = labels["target_id"].astype(str).map(
                lambda value: stable_row_id("retry11_zero", value)
            )
        labels = labels.sort_values("target_id", kind="stable").reset_index(drop=True)
    rejects = pd.DataFrame.from_records(
        [
            {
                "target_id": target_id,
                "label_quality": "Reject",
                "rejection_reason": reason,
                "attempt_count": len(groups[target_id]),
                "successful_path_count": int(
                    _successful_attempts(groups[target_id], fk_max_mm=10.0)["source_path_id"]
                    .astype(str)
                    .nunique()
                ),
            }
            for target_id, reason in sorted(rejected_reason.items(), key=lambda item: str(item[0]))
        ]
    )
    return labels, rejects


def zero_seed_status(*, exact_anchor_valid: bool, valid_count: int) -> str:
    if not bool(exact_anchor_valid) or int(valid_count) < 2:
        return "red"
    if int(valid_count) < 4:
        return "yellow"
    return "green"


def corridor_path_registry(
    zero_xyz_m: Sequence[float],
    outer_xyz_m: Sequence[float],
    *,
    maximum_step_mm: float = 10.0,
    maximum_probes: int = 32,
    diversity_ray_count: int = 3,
) -> pd.DataFrame:
    """Build one straight path plus deterministic bounded diversity rays."""

    start = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    end = np.asarray(outer_xyz_m, dtype=float).reshape(3)
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(start).all() or not np.isfinite(end).all() or distance <= 0.0:
        raise ValueError("corridor endpoints must be distinct finite points")
    count = int(math.ceil(distance * 1000.0 / float(maximum_step_mm))) + 1
    if count > int(maximum_probes):
        count = int(maximum_probes)
    if count < 2:
        raise ValueError("corridor requires at least two probes")
    direction = delta / distance
    basis_seed = np.eye(3)[int(np.argmin(np.abs(direction)))]
    lateral_u = np.cross(direction, basis_seed)
    lateral_u /= np.linalg.norm(lateral_u)
    lateral_v = np.cross(direction, lateral_u)
    amplitude = min(0.010, 0.05 * distance)
    ray_offsets = [
        ("straight", np.zeros(3)),
        ("diversity_u_plus", amplitude * lateral_u),
        ("diversity_u_minus", -amplitude * lateral_u),
        ("diversity_v_plus", amplitude * lateral_v),
    ][: 1 + int(diversity_ray_count)]
    rows: list[dict[str, object]] = []
    for path_id, offset in ray_offsets:
        for index, t in enumerate(np.linspace(0.0, 1.0, count)):
            taper = math.sin(math.pi * float(t))
            xyz = start + float(t) * delta + taper * offset
            rows.append(
                {
                    "bridge_path_id": path_id,
                    "probe_index": int(index),
                    "probe_count": count,
                    "path_fraction": float(t),
                    "x_m": float(xyz[0]),
                    "y_m": float(xyz[1]),
                    "z_m": float(xyz[2]),
                }
            )
    return pd.DataFrame.from_records(rows)


def evaluate_bridge_certificate(probe_audit: pd.DataFrame) -> dict[str, object]:
    """Apply the frozen retry11 POC thresholds and single-path semantics."""

    required = (
        "bridge_path_id",
        "probe_index",
        "solver_success",
        "same_point_raw_gap_deg",
        "local_weighted_gap_deg",
        "neighbor_raw_gap_deg",
        "persistent_edge_failure",
        "fk_residual_mm",
        "branch_conflict",
        "forward_reverse_conflict",
    )
    _require_columns(probe_audit, required, "bridge probe audit")
    frame = probe_audit.copy()
    finite_success = frame[
        frame["solver_success"].astype(bool)
        & np.isfinite(frame["fk_residual_mm"].to_numpy(float))
    ].copy()
    straight = frame[frame["bridge_path_id"].astype(str).eq("straight")]
    straight_verified = bool(len(straight) and straight["solver_success"].astype(bool).all())
    if finite_success.empty:
        return {
            "bridge_status": "not_verified",
            "straight_corridor_verified": False,
            "straight_corridor_not_verified": True,
            "straight_corridor_failure_class": "solver_failure",
            "failure_classification": "solver_failure",
            "zero_outer_branch_compatible": "unknown",
            "bridge_gate_pass": False,
        }
    raw_same_max = float(np.nanmax(finite_success["same_point_raw_gap_deg"].to_numpy(float)))
    local_p95 = float(np.nanpercentile(finite_success["local_weighted_gap_deg"].to_numpy(float), 95))
    neighbor_catastrophic_rate = float(
        np.mean(finite_success["neighbor_raw_gap_deg"].to_numpy(float) > 7.0)
    )
    persistent_rate = float(frame["persistent_edge_failure"].astype(bool).mean())
    fk = finite_success["fk_residual_mm"].to_numpy(float)
    fk_p95, fk_p99, fk_max = (
        float(np.percentile(fk, 95)),
        float(np.percentile(fk, 99)),
        float(np.max(fk)),
    )
    threshold_checks = {
        "same_point_raw_pass": raw_same_max <= 5.0 + 1.0e-12,
        "local_weighted_p95_pass": local_p95 <= 3.0 + 1.0e-12,
        "neighbor_catastrophic_rate_pass": neighbor_catastrophic_rate <= 0.02 + 1.0e-12,
        "persistent_edge_rate_pass": persistent_rate <= 0.005 + 1.0e-12,
        "fk_p95_pass": fk_p95 <= 8.0 + 1.0e-12,
        "fk_p99_pass": fk_p99 <= 15.0 + 1.0e-12,
        "fk_max_pass": fk_max <= 20.0 + 1.0e-12,
    }
    branch_rows = frame[frame["branch_conflict"].astype(bool)]
    adjacent_conflict = False
    for _path, group in branch_rows.groupby("bridge_path_id", sort=True):
        indices = sorted(group["probe_index"].astype(int).unique())
        adjacent_conflict = adjacent_conflict or any(
            right - left == 1 for left, right in zip(indices, indices[1:])
        )
    cross_path_conflict = bool(
        len(branch_rows)
        and branch_rows.groupby("probe_index", sort=True)["bridge_path_id"]
        .nunique()
        .ge(2)
        .any()
    )
    stable_branch_conflict = bool(
        adjacent_conflict or cross_path_conflict
    )
    passed = bool(straight_verified and all(threshold_checks.values()))
    if passed:
        compatibility = "true"
        status = "verified"
        failure_class = None
    elif stable_branch_conflict:
        compatibility = "false"
        status = "branch_conflict"
        failure_class = "branch_conflict"
    else:
        compatibility = "unknown"
        status = "not_verified"
        if not straight_verified or not frame["solver_success"].astype(bool).all():
            failure_class = "solver_failure"
        elif len(branch_rows):
            failure_class = "branch_conflict"
        else:
            failure_class = "local_consistency_failure"
    return {
        "bridge_status": status,
        "bridge_gate_pass": passed,
        "straight_corridor_verified": straight_verified,
        "straight_corridor_not_verified": not straight_verified,
        "straight_corridor_failure_class": failure_class,
        "failure_classification": failure_class,
        "zero_outer_branch_compatible": compatibility,
        "stable_branch_conflict": stable_branch_conflict,
        "same_point_raw_max_deg": raw_same_max,
        "local_weighted_p95_deg": local_p95,
        "neighbor_raw_catastrophic_rate": neighbor_catastrophic_rate,
        "persistent_edge_rate": persistent_rate,
        "fk_p95_mm": fk_p95,
        "fk_p99_mm": fk_p99,
        "fk_max_mm": fk_max,
        "threshold_checks": threshold_checks,
    }


__all__ = [
    "FAILURE_CLASSES",
    "classify_retry11_targets",
    "corridor_path_registry",
    "discover_graph_components",
    "evaluate_bridge_certificate",
    "zero_seed_candidate_registry",
    "zero_seed_status",
]
