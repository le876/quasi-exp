"""Fresh audits for the one-candidate-per-node section selected by V14.2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical import beta_rms_deg
from .canonical_atlas import AtlasCandidate, AtlasTaskNode, ContinuationAdapter, ContinuationOutcome
from .section_first_atlas import RootedSectionChart, SectionGrowthResult


@dataclass(frozen=True)
class SelectedSectionAuditPolicy:
    path_p95_max_deg: float = 0.5
    cycle_p95_max_deg: float = 0.5
    repeat_p95_max_deg: float = 0.2
    common_max_deg: float = 1.0
    continuation_residual_max_mm: float = 3.0

    def __post_init__(self) -> None:
        values = (
            self.path_p95_max_deg,
            self.cycle_p95_max_deg,
            self.repeat_p95_max_deg,
            self.common_max_deg,
            self.continuation_residual_max_mm,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            raise ValueError("selected-section audit thresholds must be finite and positive")


@dataclass(frozen=True)
class SelectedSectionAuditReport:
    gate_pass: bool
    chart_gate_by_id: Mapping[str, bool]
    metrics: Mapping[str, Mapping[str, Any]]
    frames: Mapping[str, pd.DataFrame]
    execution_count: int
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chart_gate_by_id", MappingProxyType(dict(self.chart_gate_by_id)))
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType({key: MappingProxyType(dict(value)) for key, value in self.metrics.items()}),
        )
        object.__setattr__(self, "frames", MappingProxyType({key: value.copy() for key, value in self.frames.items()}))
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))


@dataclass(frozen=True)
class _Trace:
    path: tuple[int, ...]
    beta_by_node: tuple[np.ndarray, ...]
    residuals_mm: tuple[float, ...]
    selected_gaps_deg: tuple[float, ...]
    success: bool
    failure_reason: str


def audit_selected_sections(
    growth: SectionGrowthResult,
    continuation: ContinuationAdapter,
    *,
    policy: SelectedSectionAuditPolicy | None = None,
    patch_id: str,
    method: str,
) -> SelectedSectionAuditReport:
    """Execute fresh edge, path, cycle, multipath, and repeat audits.

    Alternative beam hypotheses are intentionally absent from every schedule.
    The result therefore measures the exact object that a primary Student can
    learn rather than the raw candidate bank.
    """

    active = SelectedSectionAuditPolicy() if policy is None else policy
    node_by_id = {node.node_id: node for node in growth.task_nodes}
    edge_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    multipath_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    execution_count = 0
    chart_gate: dict[str, bool] = {}

    for chart in growth.charts:
        selected = {node_id: item.candidate for node_id, item in chart.selected_by_node.items()}
        adjacency = _adjacency(chart.selected_edges)
        tree_parent, tree_paths, tree_edges, connected = _spanning_tree(
            adjacency,
            root=chart.root_key[0],
            selected_node_ids=set(selected),
        )
        chords = tuple(sorted(set(chart.selected_edges) - set(tree_edges)))

        chart_edge_rows: list[dict[str, Any]] = []
        chart_repeat_rows: list[dict[str, Any]] = []
        for edge_index, (left, right) in enumerate(chart.selected_edges):
            pair_results: dict[str, _Trace] = {}
            pair_repeats: dict[str, _Trace] = {}
            for direction, path in (("forward", (left, right)), ("reverse", (right, left))):
                first = _trace(path, selected, node_by_id, continuation, active)
                second = _trace(path, selected, node_by_id, continuation, active)
                execution_count += 2
                pair_results[direction] = first
                pair_repeats[direction] = second
                repeat_gap = _endpoint_gap(first, second)
                repeat_pass = bool(
                    first.success
                    and second.success
                    and repeat_gap <= active.repeat_p95_max_deg + 1e-12
                )
                chart_repeat_rows.append(
                    _base_row(patch_id, method, chart.chart_id, f"edge_{edge_index:06d}_{direction}")
                    | {
                        "audit_kind": "edge_repeat",
                        "edge_or_cycle_length": 1,
                        "source_node": path[0],
                        "target_node": path[-1],
                        "forward_residual": _last(first.residuals_mm),
                        "reverse_residual": _last(second.residuals_mm),
                        "return_beta_gap": None,
                        "path_endpoint_gap": _last(first.selected_gaps_deg),
                        "repeat_gap": repeat_gap,
                        "passed": repeat_pass,
                        "failure_reason": "" if repeat_pass else _failure(first, second, "repeat_gap"),
                        "uses_propagated_candidate": True,
                        "candidate_lineage": chart.root_key[1],
                        "candidate_was_selected": True,
                    }
                )
            for direction, path in (("forward", (left, right)), ("reverse", (right, left))):
                trace = pair_results[direction]
                opposite = pair_results["reverse" if direction == "forward" else "forward"]
                endpoint_gap = _last(trace.selected_gaps_deg)
                passed = bool(trace.success and endpoint_gap <= active.common_max_deg + 1e-12)
                chart_edge_rows.append(
                    _base_row(patch_id, method, chart.chart_id, f"edge_{edge_index:06d}_{direction}")
                    | {
                        "audit_kind": "selected_edge",
                        "edge_or_cycle_length": 1,
                        "source_node": path[0],
                        "target_node": path[-1],
                        "forward_residual": _last(trace.residuals_mm),
                        "reverse_residual": _last(opposite.residuals_mm),
                        "return_beta_gap": _last(opposite.selected_gaps_deg),
                        "path_endpoint_gap": endpoint_gap,
                        "repeat_gap": _endpoint_gap(trace, pair_repeats[direction]),
                        "passed": passed,
                        "failure_reason": "" if passed else trace.failure_reason or "selected_endpoint_gap",
                        "uses_propagated_candidate": True,
                        "candidate_lineage": chart.root_key[1],
                        "candidate_was_selected": True,
                    }
                )
        edge_rows.extend(chart_edge_rows)
        repeat_rows.extend(chart_repeat_rows)

        chart_path_rows: list[dict[str, Any]] = []
        for path_index, path in enumerate(
            sorted((path for path in tree_paths.values() if len(path) > 1), key=lambda value: (len(value), value))
        ):
            first = _trace(path, selected, node_by_id, continuation, active)
            second = _trace(path, selected, node_by_id, continuation, active)
            execution_count += 2
            endpoint_gap = _last(first.selected_gaps_deg)
            repeat_gap = _endpoint_gap(first, second)
            passed = bool(first.success and endpoint_gap <= active.common_max_deg + 1e-12)
            chart_path_rows.append(
                _base_row(patch_id, method, chart.chart_id, f"path_{path_index:06d}")
                | {
                    "audit_kind": "root_path",
                    "edge_or_cycle_length": len(path) - 1,
                    "source_node": path[0],
                    "target_node": path[-1],
                    "forward_residual": _last(first.residuals_mm),
                    "reverse_residual": None,
                    "return_beta_gap": None,
                    "path_endpoint_gap": endpoint_gap,
                    "repeat_gap": repeat_gap,
                    "passed": passed,
                    "failure_reason": "" if passed else first.failure_reason or "path_endpoint_gap",
                    "uses_propagated_candidate": True,
                    "candidate_lineage": chart.root_key[1],
                    "candidate_was_selected": True,
                }
            )
        path_rows.extend(chart_path_rows)

        chart_cycle_rows: list[dict[str, Any]] = []
        chart_multipath_rows: list[dict[str, Any]] = []
        for cycle_index, (left, right) in enumerate(chords):
            tree_path = _tree_path(left, right, tree_parent)
            direct_path = (left, right)
            cycle_path = tree_path + (left,)
            tree_trace = _trace(tree_path, selected, node_by_id, continuation, active)
            direct_trace = _trace(direct_path, selected, node_by_id, continuation, active)
            cycle_trace = _trace(cycle_path, selected, node_by_id, continuation, active)
            cycle_repeat = _trace(cycle_path, selected, node_by_id, continuation, active)
            execution_count += 4
            loop_gap = (
                beta_rms_deg(cycle_trace.beta_by_node[0], cycle_trace.beta_by_node[-1])
                if cycle_trace.success else math.inf
            )
            cycle_pass = bool(cycle_trace.success and loop_gap <= active.common_max_deg + 1e-12)
            chart_cycle_rows.append(
                _base_row(patch_id, method, chart.chart_id, f"cycle_{cycle_index:06d}")
                | {
                    "audit_kind": "fundamental_cycle",
                    "edge_or_cycle_length": len(cycle_path) - 1,
                    "source_node": cycle_path[0],
                    "target_node": cycle_path[-1],
                    "forward_residual": _last(cycle_trace.residuals_mm),
                    "reverse_residual": None,
                    "return_beta_gap": loop_gap,
                    "path_endpoint_gap": _last(cycle_trace.selected_gaps_deg),
                    "repeat_gap": _endpoint_gap(cycle_trace, cycle_repeat),
                    "passed": cycle_pass,
                    "failure_reason": "" if cycle_pass else cycle_trace.failure_reason or "cycle_return_gap",
                    "uses_propagated_candidate": True,
                    "candidate_lineage": chart.root_key[1],
                    "candidate_was_selected": True,
                }
            )
            multipath_gap = (
                beta_rms_deg(tree_trace.beta_by_node[-1], direct_trace.beta_by_node[-1])
                if tree_trace.success and direct_trace.success else math.inf
            )
            multipath_pass = bool(
                tree_trace.success
                and direct_trace.success
                and multipath_gap <= active.common_max_deg + 1e-12
            )
            chart_multipath_rows.append(
                _base_row(patch_id, method, chart.chart_id, f"multipath_{cycle_index:06d}")
                | {
                    "audit_kind": "fundamental_multipath",
                    "edge_or_cycle_length": len(tree_path) - 1,
                    "source_node": left,
                    "target_node": right,
                    "forward_residual": _last(tree_trace.residuals_mm),
                    "reverse_residual": _last(direct_trace.residuals_mm),
                    "return_beta_gap": None,
                    "path_endpoint_gap": multipath_gap,
                    "repeat_gap": None,
                    "passed": multipath_pass,
                    "failure_reason": "" if multipath_pass else _failure(tree_trace, direct_trace, "multipath_gap"),
                    "uses_propagated_candidate": True,
                    "candidate_lineage": chart.root_key[1],
                    "candidate_was_selected": True,
                }
            )
        cycle_rows.extend(chart_cycle_rows)
        multipath_rows.extend(chart_multipath_rows)

        local_metrics = {
            "edge": _metric([row["path_endpoint_gap"] for row in chart_edge_rows], active.path_p95_max_deg, active.common_max_deg, missing=0 if chart_edge_rows else 1),
            "path": _metric([row["path_endpoint_gap"] for row in chart_path_rows], active.path_p95_max_deg, active.common_max_deg, missing=0 if chart_path_rows else 1),
            "cycle": _metric([row["return_beta_gap"] for row in chart_cycle_rows], active.cycle_p95_max_deg, active.common_max_deg, missing=0 if chart_cycle_rows and connected else 1),
            "multipath": _metric([row["path_endpoint_gap"] for row in chart_multipath_rows], active.path_p95_max_deg, active.common_max_deg, missing=0 if chart_multipath_rows and connected else 1),
            "repeat": _metric([row["repeat_gap"] for row in chart_repeat_rows], active.repeat_p95_max_deg, active.common_max_deg, missing=0 if chart_repeat_rows else 1),
        }
        chart_gate[chart.chart_id] = bool(
            all(value["gate_pass"] for value in local_metrics.values())
            and all(row["passed"] for row in chart_edge_rows)
            and all(row["passed"] for row in chart_cycle_rows)
        )

    frames = {
        "fresh_edge_audit": _frame(edge_rows),
        "fresh_path_audit": _frame(path_rows),
        "fresh_cycle_audit": _frame(cycle_rows),
        "fresh_multipath_audit": _frame(multipath_rows),
        "fresh_repeat_direction_audit": _frame(repeat_rows),
        "cap_hit_events": pd.DataFrame.from_records(
            [
                {
                    "patch_id": patch_id,
                    "method": method,
                    "chart_id": item.chart_id,
                    "schedule_id": f"cap_{index:06d}",
                    "task_node_id": item.node_id,
                    "wave": item.wave,
                    "cap_name": item.cap_name,
                    "candidate_count": item.candidate_count,
                    "retained_count": item.retained_count,
                    "pruned_candidate_ids": list(item.pruned_candidate_ids),
                }
                for index, item in enumerate(growth.cap_hit_events)
            ]
        ),
    }
    metrics = {
        "edge": _metric([row["path_endpoint_gap"] for row in edge_rows], active.path_p95_max_deg, active.common_max_deg, missing=0 if edge_rows else 1),
        "path": _metric([row["path_endpoint_gap"] for row in path_rows], active.path_p95_max_deg, active.common_max_deg, missing=0 if path_rows else 1),
        "cycle": _metric([row["return_beta_gap"] for row in cycle_rows], active.cycle_p95_max_deg, active.common_max_deg, missing=0 if cycle_rows else 1),
        "multipath": _metric([row["path_endpoint_gap"] for row in multipath_rows], active.path_p95_max_deg, active.common_max_deg, missing=0 if multipath_rows else 1),
        "repeat": _metric([row["repeat_gap"] for row in repeat_rows], active.repeat_p95_max_deg, active.common_max_deg, missing=0 if repeat_rows else 1),
    }
    failures = tuple(
        f"{name}_audit_failed" for name, metric in metrics.items() if not metric["gate_pass"]
    ) + tuple(
        f"{chart_id}_audit_failed" for chart_id, passed in chart_gate.items() if not passed
    )
    return SelectedSectionAuditReport(
        gate_pass=bool(chart_gate and all(chart_gate.values()) and all(value["gate_pass"] for value in metrics.values())),
        chart_gate_by_id=chart_gate,
        metrics=metrics,
        frames=frames,
        execution_count=execution_count,
        failure_reasons=failures,
    )


def _trace(
    path: Sequence[int],
    selected: Mapping[int, AtlasCandidate],
    node_by_id: Mapping[int, AtlasTaskNode],
    continuation: ContinuationAdapter,
    policy: SelectedSectionAuditPolicy,
) -> _Trace:
    route = tuple(map(int, path))
    if not route or any(node_id not in selected for node_id in route):
        return _Trace(route, (), (), (), False, "missing_selected_candidate")
    beta = [selected[route[0]].beta_rad.copy()]
    residuals: list[float] = []
    gaps: list[float] = []
    current = selected[route[0]]
    for node_id in route[1:]:
        outcome = continuation(current, node_by_id[node_id])
        if not isinstance(outcome, ContinuationOutcome):
            raise TypeError("selected-section continuation must return ContinuationOutcome")
        residuals.append(float(outcome.residual_mm))
        gap = beta_rms_deg(outcome.beta_rad, selected[node_id].beta_rad)
        gaps.append(gap)
        beta.append(outcome.beta_rad.copy())
        if not (
            outcome.success
            and outcome.actual_bounds
            and outcome.residual_mm <= policy.continuation_residual_max_mm + 1e-12
        ):
            return _Trace(route, tuple(beta), tuple(residuals), tuple(gaps), False, outcome.status)
        current = AtlasCandidate(
            node_id=node_id,
            candidate_id=f"fresh_{current.candidate_id}_{node_id}",
            beta_rad=outcome.beta_rad,
            residual_mm=outcome.residual_mm,
            min_margin_deg=(
                outcome.minimum_margin_deg
                if outcome.minimum_margin_deg is not None
                else selected[node_id].min_margin_deg
            ),
            normalized_min_margin=selected[node_id].normalized_min_margin,
            posture_cost=selected[node_id].posture_cost,
            condition_number=selected[node_id].condition_number,
            quality=selected[node_id].quality,
        )
    return _Trace(route, tuple(beta), tuple(residuals), tuple(gaps), True, "")


def _adjacency(edges: Sequence[tuple[int, int]]) -> Mapping[int, tuple[int, ...]]:
    output: dict[int, set[int]] = {}
    for left, right in edges:
        output.setdefault(left, set()).add(right)
        output.setdefault(right, set()).add(left)
    return {key: tuple(sorted(values)) for key, values in output.items()}


def _spanning_tree(
    adjacency: Mapping[int, Sequence[int]],
    *,
    root: int,
    selected_node_ids: set[int],
) -> tuple[Mapping[int, int | None], Mapping[int, tuple[int, ...]], tuple[tuple[int, int], ...], bool]:
    if root not in selected_node_ids:
        return {}, {}, (), False
    parent: dict[int, int | None] = {root: None}
    paths: dict[int, tuple[int, ...]] = {root: (root,)}
    queue = deque([root])
    tree_edges: set[tuple[int, int]] = set()
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            paths[neighbor] = paths[node] + (neighbor,)
            tree_edges.add(_ordered_edge(node, neighbor))
            queue.append(neighbor)
    return parent, paths, tuple(sorted(tree_edges)), set(parent) == selected_node_ids


def _tree_path(left: int, right: int, parent: Mapping[int, int | None]) -> tuple[int, ...]:
    def ancestors(node: int) -> list[int]:
        output = [node]
        while parent.get(node) is not None:
            node = int(parent[node])
            output.append(node)
        return output

    left_path = ancestors(left)
    right_path = ancestors(right)
    right_positions = {node: index for index, node in enumerate(right_path)}
    common_index = next(index for index, node in enumerate(left_path) if node in right_positions)
    common = left_path[common_index]
    return tuple(left_path[: common_index + 1] + list(reversed(right_path[: right_positions[common]])))


def _metric(values: Sequence[Any], p95_limit: float, max_limit: float, *, missing: int) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))])
    p95 = float(np.percentile(finite, 95)) if len(finite) else math.inf
    maximum = float(np.max(finite)) if len(finite) else math.inf
    return {
        "sample_count": int(len(finite)),
        "p95_deg": p95,
        "max_deg": maximum,
        "missing_count": int(missing + len(values) - len(finite)),
        "gate_pass": bool(len(finite) and missing == 0 and len(finite) == len(values) and p95 <= p95_limit + 1e-12 and maximum <= max_limit + 1e-12),
    }


def _base_row(patch_id: str, method: str, chart_id: str, schedule_id: str) -> dict[str, Any]:
    return {
        "patch_id": str(patch_id),
        "method": str(method),
        "chart_id": str(chart_id),
        "schedule_id": str(schedule_id),
    }


def _frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = (
        "patch_id", "method", "chart_id", "schedule_id", "audit_kind",
        "edge_or_cycle_length", "source_node", "target_node", "forward_residual",
        "reverse_residual", "return_beta_gap", "path_endpoint_gap", "repeat_gap",
        "passed", "failure_reason", "uses_propagated_candidate", "candidate_lineage",
        "candidate_was_selected",
    )
    return pd.DataFrame.from_records(rows, columns=columns)


def _endpoint_gap(left: _Trace, right: _Trace) -> float:
    if not left.success or not right.success or not left.beta_by_node or not right.beta_by_node:
        return math.inf
    return beta_rms_deg(left.beta_by_node[-1], right.beta_by_node[-1])


def _failure(left: _Trace, right: _Trace, fallback: str) -> str:
    return left.failure_reason or right.failure_reason or fallback


def _last(values: Sequence[float]) -> float | None:
    return float(values[-1]) if values else None


def _ordered_edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)
