"""Coverage-first qualification primitives for BACRA V14.3R retry9.

The historical formal audit remains untouched.  This module owns the separate
exploration metric, deterministic bounded sampling, and the fresh execution
schema that persists endpoint beta values needed for exact weighted evidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical_atlas import AtlasCandidate, AtlasTaskNode, ContinuationOutcome
from .holonomy_diagnostics import physical_repeat_perturbation
from .section_atlas_repair import (
    AuditV2Policy,
    _bind_retry_adapter,
    _retry_trace,
)
from .section_first_atlas import RootedSectionChart, SectionGrowthResult


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
ENDPOINT_BETA_COLUMNS = tuple(
    f"endpoint_beta{index}_rad" for index in range(1, 7)
)
DEFAULT_BETA_WEIGHTS = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)


@dataclass(frozen=True)
class ExplorationThresholds:
    minimum_overlap_nodes: int = 8
    minimum_overlap_fraction: float = 0.002
    minimum_overlap_spread_mm: float = 20.0
    label_weighted_p95_max_deg: float = 2.0
    transition_weighted_p95_max_deg: float = 2.0
    cycle_weighted_p95_max_deg: float = 3.0
    repeat_weighted_p95_max_deg: float = 1.0
    catastrophic_jump_deg: float = 5.0
    catastrophic_jump_rate_max: float = 0.01
    persistent_failure_rate_max: float = 0.001
    residual_p95_max_mm: float = 5.0
    residual_p99_max_mm: float = 10.0


def _validated_weights(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(weights), dtype=float)
    if (
        values.shape != (6,)
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
    ):
        raise ValueError("beta weights must contain six finite positive values")
    return values


def weighted_beta_rms_deg(
    left_beta_rad: np.ndarray,
    right_beta_rad: np.ndarray,
    weights: Sequence[float] = DEFAULT_BETA_WEIGHTS,
) -> float | np.ndarray:
    """Return normalized weighted beta RMS in degrees.

    The normalization is ``sum(w_i**2)``.  This intentionally differs from the
    older section-ranking helper, whose denominator is the coordinate count.
    """

    left = np.asarray(left_beta_rad, dtype=float)
    right = np.asarray(right_beta_rad, dtype=float)
    if left.shape != right.shape or left.shape[-1:] != (6,):
        raise ValueError("weighted beta inputs must have matching (..., 6) shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("weighted beta inputs must be finite")
    w = _validated_weights(weights)
    delta_deg = np.degrees(left - right)
    result = np.sqrt(np.sum(np.square(delta_deg * w), axis=-1) / np.sum(np.square(w)))
    return float(result) if result.ndim == 0 else result


def conservative_weighted_upper_bound_deg(
    raw_rms_deg: float | np.ndarray,
    weights: Sequence[float] = DEFAULT_BETA_WEIGHTS,
) -> float | np.ndarray:
    """Bound the retry9 metric using only historical unweighted RMS evidence."""

    raw = np.asarray(raw_rms_deg, dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("raw RMS values must be finite and non-negative")
    w = _validated_weights(weights)
    factor = float(np.max(w) * math.sqrt(6.0 / np.sum(np.square(w))))
    result = raw * factor
    return float(result) if result.ndim == 0 else result


def percentile(values: Sequence[float] | np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.percentile(finite, quantile)) if len(finite) else math.inf


def overlap_pair_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    thresholds: ExplorationThresholds | None = None,
    weights: Sequence[float] = DEFAULT_BETA_WEIGHTS,
) -> dict[str, Any]:
    """Measure a label-only chart overlap without numerical re-execution."""

    active = ExplorationThresholds() if thresholds is None else thresholds
    required = {"task_node_id", "x_m", "y_m", "z_m", *BETA_COLUMNS}
    if missing := required - set(left.columns):
        raise ValueError(f"left chart labels missing columns: {sorted(missing)}")
    if missing := required - set(right.columns):
        raise ValueError(f"right chart labels missing columns: {sorted(missing)}")
    if left["task_node_id"].duplicated().any() or right["task_node_id"].duplicated().any():
        raise ValueError("chart labels must be unique by task_node_id")
    joined = left.merge(
        right,
        on="task_node_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    count = len(joined)
    fraction = count / max(1, min(len(left), len(right)))
    if count:
        xyz = joined.loc[:, ["x_m_left", "y_m_left", "z_m_left"]].to_numpy(float)
        spread_mm = float(np.linalg.norm(np.max(xyz, axis=0) - np.min(xyz, axis=0)) * 1000.0)
        left_beta = joined.loc[:, [f"{column}_left" for column in BETA_COLUMNS]].to_numpy(float)
        right_beta = joined.loc[:, [f"{column}_right" for column in BETA_COLUMNS]].to_numpy(float)
        weighted = np.asarray(weighted_beta_rms_deg(left_beta, right_beta, weights), dtype=float)
        raw = np.sqrt(np.mean(np.square(np.degrees(left_beta - right_beta)), axis=1))
    else:
        spread_mm = 0.0
        weighted = np.asarray([], dtype=float)
        raw = np.asarray([], dtype=float)
    weighted_p95 = percentile(weighted, 95)
    raw_jump_rate = float(np.mean(raw > active.catastrophic_jump_deg)) if len(raw) else 1.0
    weighted_jump_rate = (
        float(np.mean(weighted > active.catastrophic_jump_deg)) if len(weighted) else 1.0
    )
    gate = bool(
        count >= active.minimum_overlap_nodes
        and fraction >= active.minimum_overlap_fraction
        and spread_mm >= active.minimum_overlap_spread_mm
        and weighted_p95 <= active.label_weighted_p95_max_deg
        and raw_jump_rate <= active.catastrophic_jump_rate_max
        and weighted_jump_rate <= active.catastrophic_jump_rate_max
    )
    return {
        "common_node_count": int(count),
        "overlap_fraction": float(fraction),
        "overlap_spread_mm": spread_mm,
        "weighted_gap_p95_deg": weighted_p95,
        "weighted_gap_max_deg": float(np.max(weighted)) if len(weighted) else math.inf,
        "raw_jump_gt_5deg_rate": raw_jump_rate,
        "weighted_jump_gt_5deg_rate": weighted_jump_rate,
        "label_overlap_gate": gate,
    }


def select_transition_edges(
    common_node_ids: Sequence[int],
    task_edges: pd.DataFrame,
    node_xyz_m: Mapping[int, np.ndarray],
    node_gap_deg: Mapping[int, float],
    *,
    maximum_edges: int = 64,
    high_gap_edges: int = 32,
) -> tuple[tuple[int, int], ...]:
    """Select high-disagreement edges followed by deterministic XYZ maximin."""

    common = frozenset(map(int, common_node_ids))
    candidates = sorted(
        {
            tuple(sorted((int(row.left_node_id), int(row.right_node_id))))
            for row in task_edges.itertuples(index=False)
            if int(row.left_node_id) in common and int(row.right_node_id) in common
        }
    )
    if not candidates:
        return ()
    ranked = sorted(
        candidates,
        key=lambda edge: (
            -max(float(node_gap_deg.get(edge[0], 0.0)), float(node_gap_deg.get(edge[1], 0.0))),
            edge,
        ),
    )
    selected = ranked[: min(int(high_gap_edges), int(maximum_edges), len(ranked))]
    if len(selected) >= min(int(maximum_edges), len(candidates)):
        return tuple(selected)
    candidate_index = {edge: index for index, edge in enumerate(candidates)}
    midpoints = np.asarray(
        [
            0.5
            * (
                np.asarray(node_xyz_m[edge[0]], dtype=float)
                + np.asarray(node_xyz_m[edge[1]], dtype=float)
            )
            for edge in candidates
        ]
    )
    available = np.ones(len(candidates), dtype=bool)
    selected_indices = [candidate_index[edge] for edge in selected]
    available[selected_indices] = False
    if selected_indices:
        deltas = midpoints[:, None, :] - midpoints[np.asarray(selected_indices)][None, :, :]
        minimum_distance = np.min(np.linalg.norm(deltas, axis=2), axis=1)
    else:
        minimum_distance = np.full(len(candidates), math.inf)
    while np.any(available) and len(selected) < int(maximum_edges):
        score = np.where(available, minimum_distance, -math.inf)
        chosen_index = int(np.argmax(score))
        selected.append(candidates[chosen_index])
        available[chosen_index] = False
        distances = np.linalg.norm(midpoints - midpoints[chosen_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distances)
    return tuple(selected)


def _canonical_cycle(nodes: Sequence[int]) -> tuple[int, ...]:
    cycle = tuple(map(int, nodes))
    rotations = []
    for direction in (cycle, tuple(reversed(cycle))):
        rotations.extend(direction[index:] + direction[:index] for index in range(len(direction)))
    return min(rotations)


def bounded_fundamental_cycles(
    common_node_ids: Sequence[int],
    task_edges: pd.DataFrame,
    *,
    maximum_cycles: int = 16,
    maximum_length: int = 16,
) -> tuple[tuple[int, ...], ...]:
    """Find deterministic short cycles without an optional graph dependency."""

    common = frozenset(map(int, common_node_ids))
    edges = sorted(
        {
            tuple(sorted((int(row.left_node_id), int(row.right_node_id))))
            for row in task_edges.itertuples(index=False)
            if int(row.left_node_id) in common and int(row.right_node_id) in common
        }
    )
    adjacency: dict[int, list[int]] = {node: [] for node in common}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for values in adjacency.values():
        values.sort()
    parent: dict[int, int] = {}
    depth: dict[int, int] = {}
    tree_edges: set[tuple[int, int]] = set()
    for root in sorted(common):
        if root in parent:
            continue
        parent[root] = -1
        depth[root] = 0
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in parent:
                    continue
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                tree_edges.add(tuple(sorted((node, neighbor))))
                queue.append(neighbor)
    found: set[tuple[int, ...]] = set()
    for left, right in edges:
        if (left, right) in tree_edges:
            continue
        left_path = [left]
        right_path = [right]
        left_cursor, right_cursor = left, right
        while depth[left_cursor] > depth[right_cursor]:
            left_cursor = parent[left_cursor]
            left_path.append(left_cursor)
        while depth[right_cursor] > depth[left_cursor]:
            right_cursor = parent[right_cursor]
            right_path.append(right_cursor)
        while left_cursor != right_cursor:
            left_cursor = parent[left_cursor]
            right_cursor = parent[right_cursor]
            left_path.append(left_cursor)
            right_path.append(right_cursor)
        cycle = left_path + list(reversed(right_path[:-1]))
        if 3 <= len(cycle) <= int(maximum_length):
            found.add(_canonical_cycle(cycle))
        if len(found) >= int(maximum_cycles):
            break
    return tuple(cycle + (cycle[0],) for cycle in sorted(found)[: int(maximum_cycles)])


def execute_exploration_audit_schedules(
    growth: SectionGrowthResult,
    schedules: pd.DataFrame,
    continuation: Callable[..., ContinuationOutcome],
    policy: AuditV2Policy,
    *,
    retry_continuation: Callable[..., ContinuationOutcome] | None,
    weights: Sequence[float] = DEFAULT_BETA_WEIGHTS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Execute bounded fresh schedules while persisting exact endpoint beta."""

    chart_by_id = {chart.chart_id: chart for chart in growth.charts}
    node_by_id = {node.node_id: node for node in growth.task_nodes}
    raw_adapter = continuation if retry_continuation is None else retry_continuation
    adapter = _bind_retry_adapter(raw_adapter)
    rows: list[dict[str, Any]] = []
    records = schedules.to_dict(orient="records")
    for completed, schedule in enumerate(records, start=1):
        chart = chart_by_id[str(schedule["chart_id"])]
        selected = {node: item.candidate for node, item in chart.selected_by_node.items()}
        path = tuple(int(value) for value in schedule["path_node_ids"])
        endpoints: list[np.ndarray | None] = []
        row_indices: list[int] = []
        for repeat_index in range(policy.repeats_per_direction):
            perturbation = physical_repeat_perturbation(
                str(schedule["physical_entity_id"]),
                "canonical_forward",
                repeat_index,
                policy.repeat_perturbation_rad,
            )
            trace = _retry_trace(
                path,
                selected,
                node_by_id,
                adapter,
                policy,
                source_perturbation_rad=perturbation,
            )
            endpoint = trace.get("endpoint_beta")
            endpoint_array = None if endpoint is None else np.asarray(endpoint, dtype=float)
            endpoints.append(endpoint_array)
            expected = selected[path[-1]].beta_rad if path and path[-1] in selected else None
            weighted_gap = (
                math.nan
                if endpoint_array is None or expected is None
                else float(weighted_beta_rms_deg(endpoint_array, expected, weights))
            )
            row: dict[str, Any] = {
                "schedule_id": str(schedule["schedule_id"]),
                "pair_id": str(schedule["pair_id"]),
                "audit_kind": str(schedule["audit_kind"]),
                "chart_id": str(schedule["chart_id"]),
                "physical_entity_id": str(schedule["physical_entity_id"]),
                "repeat_index": int(repeat_index),
                "repeat_perturbation_l2_rad": float(np.linalg.norm(perturbation)),
                "solver_success": bool(trace["solver_success"]),
                "geometry_gap_deg": float(trace["geometry_gap_deg"]),
                "weighted_geometry_gap_deg": weighted_gap,
                "weighted_repeat_gap_deg": math.nan,
                "residual_mm": float(trace["residual_mm"]),
                "retry_tier": str(trace["retry_tier"]),
                "registered_solver_chain": str(trace["registered_solver_chain"]),
                "executed_solver_chain": str(trace["executed_solver_chain"]),
                "solver_chain_sha256": str(trace["solver_chain_sha256"]),
                "classification": str(trace["classification"]),
                "failure_source_node": trace["failure_source_node"],
                "failure_target_node": trace["failure_target_node"],
                "exact_bounds": bool(trace["solver_success"]),
                "oracle_used_for_pass": False,
            }
            for column, value in zip(
                ENDPOINT_BETA_COLUMNS,
                ([math.nan] * 6 if endpoint_array is None else endpoint_array),
                strict=True,
            ):
                row[column] = float(value)
            rows.append(row)
            row_indices.append(len(rows) - 1)
        reference = next((value for value in endpoints if value is not None), None)
        repeat_gap = (
            max(
                (
                    float(weighted_beta_rms_deg(reference, value, weights))
                    for value in endpoints
                    if value is not None
                ),
                default=0.0,
            )
            if reference is not None
            else math.nan
        )
        for index in row_indices:
            rows[index]["weighted_repeat_gap_deg"] = repeat_gap
        if progress_callback is not None:
            progress_callback(completed, len(records))
    return pd.DataFrame.from_records(rows)


def summarize_exploration_pair(
    executions: pd.DataFrame,
    *,
    thresholds: ExplorationThresholds | None = None,
) -> dict[str, Any]:
    active = ExplorationThresholds() if thresholds is None else thresholds
    if len(executions) == 0:
        return {"fresh_audit_gate": False, "reason": "no_fresh_executions"}
    success = executions[executions["solver_success"].astype(bool)]
    transition = success[success["audit_kind"].eq("transition")]
    cycles = success[success["audit_kind"].eq("cycle")]
    persistent_rate = float(
        executions["classification"].astype(str).eq("persistent_numerical").mean()
    )
    raw_jump_rate = float(
        (success["geometry_gap_deg"].astype(float) > active.catastrophic_jump_deg).mean()
    ) if len(success) else 1.0
    weighted_jump_rate = float(
        (success["weighted_geometry_gap_deg"].astype(float) > active.catastrophic_jump_deg).mean()
    ) if len(success) else 1.0
    metrics = {
        "execution_count": int(len(executions)),
        "successful_execution_count": int(len(success)),
        "transition_execution_count": int(len(transition)),
        "cycle_execution_count": int(len(cycles)),
        "persistent_failure_rate": persistent_rate,
        "transition_weighted_p95_deg": percentile(
            transition["weighted_geometry_gap_deg"].to_numpy(float), 95
        ),
        "cycle_weighted_p95_deg": percentile(
            cycles["weighted_geometry_gap_deg"].to_numpy(float), 95
        ),
        "repeat_weighted_p95_deg": percentile(
            success["weighted_repeat_gap_deg"].to_numpy(float), 95
        ),
        "raw_jump_gt_5deg_rate": raw_jump_rate,
        "weighted_jump_gt_5deg_rate": weighted_jump_rate,
        "residual_p95_mm": percentile(success["residual_mm"].to_numpy(float), 95),
        "residual_p99_mm": percentile(success["residual_mm"].to_numpy(float), 99),
        "exact_bounds_all": bool(success["exact_bounds"].astype(bool).all()) if len(success) else False,
    }
    metrics["fresh_audit_gate"] = bool(
        metrics["transition_execution_count"] > 0
        and metrics["cycle_execution_count"] > 0
        and metrics["persistent_failure_rate"] <= active.persistent_failure_rate_max
        and metrics["transition_weighted_p95_deg"] <= active.transition_weighted_p95_max_deg
        and metrics["cycle_weighted_p95_deg"] <= active.cycle_weighted_p95_max_deg
        and metrics["repeat_weighted_p95_deg"] <= active.repeat_weighted_p95_max_deg
        and metrics["raw_jump_gt_5deg_rate"] <= active.catastrophic_jump_rate_max
        and metrics["weighted_jump_gt_5deg_rate"] <= active.catastrophic_jump_rate_max
        and metrics["residual_p95_mm"] <= active.residual_p95_max_mm
        and metrics["residual_p99_mm"] <= active.residual_p99_max_mm
        and metrics["exact_bounds_all"]
    )
    return metrics


__all__ = [
    "BETA_COLUMNS",
    "DEFAULT_BETA_WEIGHTS",
    "ENDPOINT_BETA_COLUMNS",
    "ExplorationThresholds",
    "bounded_fundamental_cycles",
    "conservative_weighted_upper_bound_deg",
    "execute_exploration_audit_schedules",
    "overlap_pair_metrics",
    "select_transition_edges",
    "summarize_exploration_pair",
    "weighted_beta_rms_deg",
]
