"""Path, loop, direction, repeat and overlap audits for BACRA V12 charts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import heapq
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .canonical import beta_rms_deg
from .canonical_atlas import CanonicalAtlas, CanonicalChart


@dataclass(frozen=True)
class AtlasAuditPolicy:
    endpoint_count: int = 100
    paths_per_endpoint: int = 3
    loop_count: int = 100
    path_p95_deg: float = 0.5
    loop_p95_deg: float = 0.5
    direction_p95_deg: float = 0.5
    overlap_p95_deg: float = 0.5
    repeat_p95_deg: float = 0.2
    common_max_deg: float = 1.0

    def __post_init__(self) -> None:
        if self.endpoint_count < 1 or self.paths_per_endpoint < 2 or self.loop_count < 1:
            raise ValueError("endpoint_count, paths_per_endpoint and loop_count must be positive")
        if min(
            self.path_p95_deg,
            self.loop_p95_deg,
            self.direction_p95_deg,
            self.overlap_p95_deg,
            self.repeat_p95_deg,
            self.common_max_deg,
        ) <= 0.0:
            raise ValueError("all audit thresholds must be positive")


@dataclass(frozen=True)
class PathTrace:
    """Configuration values associated with every task node on one traversal."""

    node_ids: tuple[int, ...]
    beta_rad_by_node: np.ndarray
    success: bool
    status: str = "unknown"

    def __post_init__(self) -> None:
        node_ids = tuple(int(value) for value in self.node_ids)
        beta = np.asarray(self.beta_rad_by_node, dtype=float).reshape(-1, 6)
        if len(node_ids) != len(beta):
            raise ValueError("trace node inventory must align with beta rows")
        if not np.isfinite(beta).all():
            raise ValueError("trace beta values must be finite")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "beta_rad_by_node", beta.copy())
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "status", str(self.status))


PathExecutor = Callable[[CanonicalChart, tuple[int, ...]], PathTrace]
RepeatExecutor = Callable[[CanonicalChart, tuple[int, ...]], Sequence[PathTrace]]


@dataclass(frozen=True)
class AuditMetric:
    sample_count: int
    p95_deg: float
    max_deg: float
    gate_pass: bool
    missing_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "p95_deg": float(self.p95_deg),
            "max_deg": float(self.max_deg),
            "missing_count": int(self.missing_count),
            "gate_pass": bool(self.gate_pass),
        }


@dataclass(frozen=True)
class AtlasAuditReport:
    path: AuditMetric
    loop: AuditMetric
    direction: AuditMetric
    repeat: AuditMetric
    overlap: AuditMetric
    multi_chart_overlap_valid: bool
    representation_decision: str
    gate_pass: bool
    endpoint_count: int
    loop_count: int
    evidence_limitations: tuple[str, ...]
    checks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_limitations", tuple(self.evidence_limitations))
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))


def replay_product_graph_path(atlas: CanonicalAtlas, chart: CanonicalChart, path: tuple[int, ...]) -> PathTrace:
    """Replay saved directed continuation outcomes for a selected chart path.

    This is useful for structural diagnostics.  A formal repeat audit must
    still pass a separate ``repeat_executor`` that executes independent solves;
    replay alone is never labelled independent repeat evidence.
    """

    selected = chart.selection_by_node
    directed = atlas.product_graph.directed_edge_by_keys
    if not path or any(node not in selected for node in path):
        return PathTrace((), np.zeros((0, 6)), False, "unknown_chart_node")
    beta = [atlas.product_graph.candidate_by_key[selected[path[0]]].beta_rad]
    for left_node, right_node in zip(path, path[1:]):
        edge = directed.get((selected[left_node], selected[right_node]))
        if edge is None:
            return PathTrace(tuple(path), np.vstack(beta), False, "missing_directed_edge")
        beta.append(edge.continuation_beta_rad)
    return PathTrace(tuple(path), np.vstack(beta), True, "saved_directed_continuation_replay")


def audit_atlas(
    atlas: CanonicalAtlas,
    *,
    policy: AtlasAuditPolicy | None = None,
    path_executor: PathExecutor | None = None,
    repeat_executor: RepeatExecutor | None = None,
) -> AtlasAuditReport:
    """Run deterministic multi-path and loop checks and decide label form.

    ``path_executor`` receives every explicitly generated path.  Supplying it
    is how a runner requests fresh continuation execution.  Without one, the
    saved product-graph edge outcomes are replayed for path/loop/direction,
    but the repeat Gate fails closed because no independent repetition exists.
    """

    active_policy = policy or AtlasAuditPolicy()
    limitations: list[str] = []
    if path_executor is None:
        path_executor = lambda chart, path: replay_product_graph_path(atlas, chart, path)
        limitations.append("path_loop_direction_use_saved_directed_continuation_replay")
    if repeat_executor is None:
        limitations.append("independent_repeat_executor_not_supplied")

    path_values: list[float] = []
    loop_values: list[float] = []
    direction_values: list[float] = []
    repeat_values: list[float] = []
    path_missing = 0
    loop_missing = 0
    direction_missing = 0
    repeat_missing = 0
    endpoints_seen = 0
    loops_seen = 0

    for chart in atlas.charts:
        adjacency = _selected_task_adjacency(atlas, chart)
        endpoints = _audit_endpoints(chart, adjacency, active_policy.endpoint_count)
        endpoints_seen += len(endpoints)
        if len(endpoints) < active_policy.endpoint_count:
            path_missing += active_policy.endpoint_count - len(endpoints)
        for endpoint in endpoints:
            paths = _edge_penalized_paths(
                adjacency,
                chart.root_key[0],
                endpoint,
                active_policy.paths_per_endpoint,
            )
            if len(paths) < active_policy.paths_per_endpoint:
                path_missing += 1
                direction_missing += 1
                loop_missing += 1
                if repeat_executor is None:
                    repeat_missing += 1
                continue
            traces = [_run_path_executor(path_executor, chart, path) for path in paths]
            if any(not _trace_matches(trace, path) for trace, path in zip(traces, paths)):
                path_missing += 1
                direction_missing += 1
                loop_missing += 1
                continue
            for left_index, left in enumerate(traces):
                for right in traces[left_index + 1 :]:
                    path_values.append(beta_rms_deg(left.beta_rad_by_node[-1], right.beta_rad_by_node[-1]))
            for path, trace in zip(paths, traces):
                reverse_path = tuple(reversed(path))
                reverse_trace = _run_path_executor(path_executor, chart, reverse_path)
                if not _trace_matches(reverse_trace, reverse_path):
                    direction_missing += 1
                else:
                    aligned_reverse = reverse_trace.beta_rad_by_node[::-1]
                    direction_values.extend(
                        beta_rms_deg(left, right)
                        for left, right in zip(trace.beta_rad_by_node, aligned_reverse)
                    )
            for alternative in paths[1:]:
                if loops_seen >= active_policy.loop_count:
                    break
                loop_path = paths[0] + tuple(reversed(alternative))[1:]
                loop_trace = _run_path_executor(path_executor, chart, loop_path)
                if not _trace_matches(loop_trace, loop_path):
                    loop_missing += 1
                else:
                    loop_values.append(
                        beta_rms_deg(loop_trace.beta_rad_by_node[0], loop_trace.beta_rad_by_node[-1])
                    )
                    loops_seen += 1
            if repeat_executor is None:
                repeat_missing += 1
            else:
                repeats = tuple(_run_repeat_executor(repeat_executor, chart, paths[0]))
                if len(repeats) < 2 or any(not _trace_matches(trace, paths[0]) for trace in repeats[:2]):
                    repeat_missing += 1
                else:
                    repeat_values.extend(
                        beta_rms_deg(left, right)
                        for left, right in zip(
                            repeats[0].beta_rad_by_node, repeats[1].beta_rad_by_node
                        )
                    )

    if loops_seen < active_policy.loop_count:
        loop_missing += active_policy.loop_count - loops_seen
    path_metric = _metric(path_values, active_policy.path_p95_deg, active_policy.common_max_deg, path_missing)
    loop_metric = _metric(loop_values, active_policy.loop_p95_deg, active_policy.common_max_deg, loop_missing)
    direction_metric = _metric(
        direction_values,
        active_policy.direction_p95_deg,
        active_policy.common_max_deg,
        direction_missing,
    )
    repeat_metric = _metric(
        repeat_values,
        active_policy.repeat_p95_deg,
        active_policy.common_max_deg,
        repeat_missing,
    )
    overlap_metric, multi_chart_overlap_valid = _overlap_metric(atlas, active_policy)
    stability_pass = (
        path_metric.gate_pass
        and loop_metric.gate_pass
        and direction_metric.gate_pass
        and repeat_metric.gate_pass
    )
    representation = _representation_decision(
        chart_count=len(atlas.charts),
        stability_pass=stability_pass,
        path_metric=path_metric,
        loop_metric=loop_metric,
        direction_metric=direction_metric,
        repeat_metric=repeat_metric,
        overlap_metric=overlap_metric,
        multi_chart_overlap_valid=multi_chart_overlap_valid,
    )
    gate_pass = bool(
        stability_pass
        and (overlap_metric.gate_pass or (len(atlas.charts) > 1 and multi_chart_overlap_valid))
        and representation != "reject_insufficient_atlas_evidence"
    )
    checks = {
        "path": path_metric.as_dict(),
        "loop": loop_metric.as_dict(),
        "direction": direction_metric.as_dict(),
        "repeat": repeat_metric.as_dict(),
        "overlap": overlap_metric.as_dict(),
        "multi_chart_overlap_valid": {"gate_pass": bool(multi_chart_overlap_valid)},
        "representation_decision": {"value": representation},
    }
    return AtlasAuditReport(
        path=path_metric,
        loop=loop_metric,
        direction=direction_metric,
        repeat=repeat_metric,
        overlap=overlap_metric,
        multi_chart_overlap_valid=multi_chart_overlap_valid,
        representation_decision=representation,
        gate_pass=gate_pass,
        endpoint_count=endpoints_seen,
        loop_count=loops_seen,
        evidence_limitations=tuple(limitations),
        checks=checks,
    )


def _selected_task_adjacency(
    atlas: CanonicalAtlas, chart: CanonicalChart
) -> Mapping[int, tuple[tuple[int, float], ...]]:
    selected = chart.selection_by_node
    rows: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in atlas.product_graph.robust_edges:
        left_node, right_node = edge.left_key[0], edge.right_key[0]
        if selected.get(left_node) != edge.left_key or selected.get(right_node) != edge.right_key:
            continue
        rows[left_node].append((right_node, edge.cost))
        rows[right_node].append((left_node, edge.cost))
    return MappingProxyType(
        {
            node: tuple(sorted(values, key=lambda item: (item[1], item[0])))
            for node, values in sorted(rows.items())
        }
    )


def _audit_endpoints(
    chart: CanonicalChart,
    adjacency: Mapping[int, tuple[tuple[int, float], ...]],
    count: int,
) -> tuple[int, ...]:
    root = chart.root_key[0]
    distance = _node_dijkstra(adjacency, root)
    options = [node for node in chart.selection_by_node if node != root and node in distance]
    ordered = sorted(options, key=lambda node: (-distance[node], node))
    return tuple(ordered[: int(count)])


def _node_dijkstra(
    adjacency: Mapping[int, tuple[tuple[int, float], ...]], root: int, penalties: Mapping[tuple[int, int], float] | None = None
) -> Mapping[int, float]:
    if root not in adjacency:
        return MappingProxyType({root: 0.0})
    penalty = penalties or {}
    distance: dict[int, float] = {root: 0.0}
    queue: list[tuple[float, int]] = [(0.0, root)]
    while queue:
        value, current = heapq.heappop(queue)
        if value > distance.get(current, math.inf) + 1.0e-12:
            continue
        for neighbor, cost in adjacency.get(current, ()):
            pair = (current, neighbor) if current < neighbor else (neighbor, current)
            candidate = value + float(cost) + float(penalty.get(pair, 0.0))
            if candidate < distance.get(neighbor, math.inf) - 1.0e-12:
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return MappingProxyType(distance)


def _shortest_path(
    adjacency: Mapping[int, tuple[tuple[int, float], ...]],
    start: int,
    end: int,
    penalties: Mapping[tuple[int, int], float],
) -> tuple[int, ...] | None:
    if start == end:
        return (start,)
    distance: dict[int, float] = {start: 0.0}
    parent: dict[int, int | None] = {start: None}
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        value, current = heapq.heappop(queue)
        if current == end:
            break
        if value > distance.get(current, math.inf) + 1.0e-12:
            continue
        for neighbor, base_cost in adjacency.get(current, ()):
            pair = (current, neighbor) if current < neighbor else (neighbor, current)
            candidate = value + float(base_cost) + float(penalties.get(pair, 0.0))
            old = distance.get(neighbor, math.inf)
            if candidate < old - 1.0e-12 or (
                abs(candidate - old) <= 1.0e-12 and current < (parent.get(neighbor) or current)
            ):
                distance[neighbor] = candidate
                parent[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if end not in parent:
        return None
    path = [end]
    while path[-1] != start:
        previous = parent[path[-1]]
        if previous is None:
            return None
        path.append(previous)
    return tuple(reversed(path))


def _edge_penalized_paths(
    adjacency: Mapping[int, tuple[tuple[int, float], ...]],
    start: int,
    end: int,
    count: int,
) -> tuple[tuple[int, ...], ...]:
    """Deterministic edge-penalised paths; no duplicate is counted as a path."""

    if start not in adjacency or end not in adjacency:
        return ()
    base_max = max((cost for rows in adjacency.values() for _, cost in rows), default=1.0)
    penalty_scale = max(1.0, float(base_max)) * (len(adjacency) + 1)
    penalties: dict[tuple[int, int], float] = {}
    paths: list[tuple[int, ...]] = []
    for _ in range(max(1, int(count)) * max(4, len(adjacency))):
        path = _shortest_path(adjacency, start, end, penalties)
        if path is None:
            break
        if path not in paths:
            paths.append(path)
            if len(paths) >= int(count):
                break
        for left, right in zip(path, path[1:]):
            pair = (left, right) if left < right else (right, left)
            penalties[pair] = penalties.get(pair, 0.0) + penalty_scale
    return tuple(paths)


def _run_path_executor(executor: PathExecutor, chart: CanonicalChart, path: tuple[int, ...]) -> PathTrace:
    try:
        raw = executor(chart, path)
        if isinstance(raw, PathTrace):
            return raw
        if isinstance(raw, Mapping):
            return PathTrace(
                tuple(raw["node_ids"]), raw["beta_rad_by_node"], bool(raw["success"]), str(raw.get("status", "mapping"))
            )
        raise TypeError("path executor must return PathTrace or mapping")
    except Exception as error:
        return PathTrace(path, np.zeros((len(path), 6)), False, f"executor_exception:{type(error).__name__}")


def _run_repeat_executor(
    executor: RepeatExecutor, chart: CanonicalChart, path: tuple[int, ...]
) -> Sequence[PathTrace]:
    try:
        return tuple(
            trace if isinstance(trace, PathTrace) else PathTrace(
                tuple(trace["node_ids"]), trace["beta_rad_by_node"], bool(trace["success"]), str(trace.get("status", "mapping"))
            )
            for trace in executor(chart, path)
        )
    except Exception:
        return ()


def _trace_matches(trace: PathTrace, path: tuple[int, ...]) -> bool:
    return bool(trace.success and trace.node_ids == tuple(path) and len(trace.beta_rad_by_node) == len(path))


def _metric(values: Sequence[float], p95_limit: float, max_limit: float, missing: int) -> AuditMetric:
    data = np.asarray(values, dtype=float)
    if len(data) == 0 or int(missing) > 0:
        p95 = float(np.percentile(data, 95)) if len(data) else math.inf
        maximum = float(np.max(data)) if len(data) else math.inf
        return AuditMetric(len(data), p95, maximum, False, int(missing))
    p95 = float(np.percentile(data, 95))
    maximum = float(np.max(data))
    return AuditMetric(
        len(data),
        p95,
        maximum,
        bool(p95 <= p95_limit + 1.0e-12 and maximum <= max_limit + 1.0e-12),
        0,
    )


def _overlap_metric(atlas: CanonicalAtlas, policy: AtlasAuditPolicy) -> tuple[AuditMetric, bool]:
    reports = atlas.overlap_reports
    if not reports:
        return AuditMetric(0, 0.0, 0.0, True, 0), True
    values: list[float] = []
    ambiguous = False
    for report in reports:
        values.extend([report.gap_p95_deg, report.gap_max_deg])
        if report.resolution == "ambiguous_keep_separate":
            ambiguous = True
    metric = _metric(values, policy.overlap_p95_deg, policy.common_max_deg, 0)
    # Distinct branches are valid only when explicitly more than the one-degree
    # split threshold, not when they lie in the ambiguous 0.5--1 degree band.
    multi_chart_valid = bool(
        not ambiguous
        and all(report.resolution != "mergeable_duplicate_retained_separately" or metric.gate_pass for report in reports)
    )
    return metric, multi_chart_valid


def _representation_decision(
    *,
    chart_count: int,
    stability_pass: bool,
    path_metric: AuditMetric,
    loop_metric: AuditMetric,
    direction_metric: AuditMetric,
    repeat_metric: AuditMetric,
    overlap_metric: AuditMetric,
    multi_chart_overlap_valid: bool,
) -> str:
    if chart_count == 0:
        return "reject_insufficient_atlas_evidence"
    if stability_pass:
        if chart_count == 1 and overlap_metric.gate_pass:
            return "static_xyz_to_beta6"
        if chart_count > 1 and (overlap_metric.gate_pass or multi_chart_overlap_valid):
            return "multi_chart_xyz_chart_id_to_beta6"
    # Statefulness is a representation conclusion only when actual path/loop/
    # direction discrepancies, rather than missing evidence, caused the fail.
    path_dependent = any(
        metric.sample_count > 0 and metric.missing_count == 0 and not metric.gate_pass
        for metric in (path_metric, loop_metric, direction_metric)
    )
    if path_dependent and repeat_metric.gate_pass:
        return "stateful_xyz_beta_prev_to_delta_beta"
    return "reject_insufficient_atlas_evidence"
