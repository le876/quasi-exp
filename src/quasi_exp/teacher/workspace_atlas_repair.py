"""BACRA V14.1 cross-cell canonical-atlas repair primitives.

V14 deliberately kept a static candidate bank: a successful continuation was
accepted only when it landed within 0.5 degree of a candidate that already
existed at the target task node.  The scientific Pilot showed that this made
the product graph almost entirely cell-local.  This module preserves that
behaviour as :class:`RepairAblation.E0` and adds the registered E1--E4 repair
experiment without changing the historical V14 implementation.

The central invariant is intentionally strict: a propagated endpoint may be
stored as diagnostic evidence after a successful forward solve, but it enters
the next propagation wave only after an immediate reverse continuation also
returns to the source candidate within the registered beta tolerance.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .canonical import beta_rms_deg
from .canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    CanonicalAtlas,
    ContinuationAdapter,
    ContinuationOutcome,
    DirectedContinuationEdge,
    ProductGraph,
    assemble_product_graph,
    build_product_graph,
)


CandidateKey = tuple[int, str]
CellIdentity = tuple[int, int, int, int]


class RepairAblation(str, Enum):
    """Nested repair variants evaluated on exactly the same task patches."""

    E0_LEGACY = "E0_legacy"
    E1_SHARED_FACE = "E1_shared_face"
    E2_SEGMENTED = "E2_segmented"
    E3_DYNAMIC_INSERTION = "E3_dynamic_insertion"
    E4_FRONTIER_ENRICHMENT = "E4_frontier_enrichment"


@dataclass(frozen=True)
class CrossCellRepairPolicy:
    """Registered numerical and propagation budget for V14.1."""

    shared_face_probe_count: int = 4
    cartesian_step_max_mm: float = 5.0
    propagated_cluster_deg: float = 0.5
    reverse_return_max_deg: float = 0.5
    continuation_residual_max_mm: float = 3.0
    maximum_waves: int = 3
    maximum_propagated_candidates_per_node: int = 12
    maximum_candidates_per_lineage_per_node: int = 2
    icm_max_sweeps: int = 50
    stitchable_p95_deg: float = 0.5
    stitchable_max_deg: float = 1.0
    patch_size: int = 64
    patch_count: int = 12
    development_patch_count: int = 8
    severe_enrichment_start_count: int = 32
    ordinary_enrichment_start_count: int = 16
    severe_enrichment_fraction_max: float = 0.25
    enrichment_fraction_max: float = 0.10
    enrichment_node_max: int = 1024

    def __post_init__(self) -> None:
        integer_fields = (
            self.shared_face_probe_count,
            self.maximum_waves,
            self.maximum_propagated_candidates_per_node,
            self.maximum_candidates_per_lineage_per_node,
            self.icm_max_sweeps,
            self.patch_size,
            self.patch_count,
            self.development_patch_count,
            self.severe_enrichment_start_count,
            self.ordinary_enrichment_start_count,
            self.enrichment_node_max,
        )
        if any(int(value) < 1 for value in integer_fields):
            raise ValueError("repair integer budgets must be positive")
        positive = (
            self.cartesian_step_max_mm,
            self.propagated_cluster_deg,
            self.reverse_return_max_deg,
            self.continuation_residual_max_mm,
            self.stitchable_p95_deg,
            self.stitchable_max_deg,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("repair thresholds must be finite and positive")
        fractions = (
            self.severe_enrichment_fraction_max,
            self.enrichment_fraction_max,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in fractions):
            raise ValueError("repair enrichment limits must be fractions")
        if self.development_patch_count >= self.patch_count:
            raise ValueError("repair requires at least one confirmation patch")


@dataclass(frozen=True)
class PropagationRecord:
    wave: int
    source_key: CandidateKey
    target_node_id: int
    lineage_id: str
    status: str
    forward_success: bool
    reverse_success: bool
    inserted: bool
    propagated_next_wave: bool
    residual_mm: float
    reverse_return_gap_deg: float | None
    target_candidate_id: str | None


@dataclass(frozen=True)
class CrossCellGraphMetrics:
    task_cross_cell_pair_count: int
    robust_cross_cell_pair_count: int
    cross_cell_neighbor_pair_connection_rate: float
    connected_cell_count: int
    total_cell_count: int
    connected_cell_ratio: float
    largest_component_cell_count: int
    largest_component_ratio: float
    robust_cross_cell_edge_count: int


@dataclass(frozen=True)
class CrossCellRepairResult:
    ablation: RepairAblation
    graph: ProductGraph
    task_edges: pd.DataFrame
    propagation_records: tuple[PropagationRecord, ...]
    metrics: CrossCellGraphMetrics
    budget_exhausted: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ablation", RepairAblation(self.ablation))
        object.__setattr__(self, "task_edges", self.task_edges.copy())
        object.__setattr__(self, "propagation_records", tuple(self.propagation_records))
        object.__setattr__(self, "budget_exhausted", bool(self.budget_exhausted))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class PiecewisePartitionResult:
    assignment_by_node: Mapping[int, str | None]
    candidate_sections_by_node: Mapping[int, tuple[str, ...]]
    abstained_node_ids: frozenset[int]
    primary_section_ids: tuple[str, ...]
    required_transitions: tuple[tuple[str, str], ...]
    icm_sweeps: int
    boundary_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assignment_by_node",
            MappingProxyType({int(key): value for key, value in self.assignment_by_node.items()}),
        )
        object.__setattr__(
            self,
            "candidate_sections_by_node",
            MappingProxyType(
                {int(key): tuple(value) for key, value in self.candidate_sections_by_node.items()}
            ),
        )
        object.__setattr__(self, "abstained_node_ids", frozenset(self.abstained_node_ids))
        object.__setattr__(self, "primary_section_ids", tuple(self.primary_section_ids))
        object.__setattr__(self, "required_transitions", tuple(self.required_transitions))
        object.__setattr__(self, "boundary_counts", MappingProxyType(dict(self.boundary_counts)))


@dataclass(frozen=True)
class CanonicalSelectionObservation:
    node_id: int
    budget: int
    selected_beta_rad: np.ndarray
    selected_component_id: str
    outgoing_neighbor_ids: frozenset[int]

    def __post_init__(self) -> None:
        beta = np.asarray(self.selected_beta_rad, dtype=float).reshape(6)
        if not np.isfinite(beta).all() or int(self.budget) < 1:
            raise ValueError("selection observations require finite beta and positive budget")
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "budget", int(self.budget))
        object.__setattr__(self, "selected_beta_rad", beta.copy())
        object.__setattr__(self, "selected_component_id", str(self.selected_component_id))
        object.__setattr__(self, "outgoing_neighbor_ids", frozenset(map(int, self.outgoing_neighbor_ids)))


@dataclass(frozen=True)
class CanonicalSelectionStabilityReport:
    audited_node_count: int
    beta_gap_p95_deg: float
    beta_gap_max_deg: float
    component_switch_ratio: float
    outgoing_neighbor_change_ratio: float
    gate_pass: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RepairPatchInventory:
    assignments: pd.DataFrame
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", self.assignments.copy())
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def _cell_identity(row: Any) -> CellIdentity:
    return (
        int(getattr(row, "cell_level_mm")),
        int(getattr(row, "cell_ix")),
        int(getattr(row, "cell_iy")),
        int(getattr(row, "cell_iz")),
    )


def _task_frame_index(task_nodes: pd.DataFrame) -> tuple[dict[int, Any], dict[int, CellIdentity]]:
    required = {
        "task_node_id",
        "x_m",
        "y_m",
        "z_m",
        "cell_level_mm",
        "cell_ix",
        "cell_iy",
        "cell_iz",
        "is_representative",
    }
    missing = sorted(required - set(task_nodes.columns))
    if missing:
        raise ValueError(f"repair task table missing columns: {missing}")
    if task_nodes["task_node_id"].duplicated().any():
        raise ValueError("repair task node IDs must be unique")
    rows = {int(row.task_node_id): row for row in task_nodes.itertuples(index=False)}
    return rows, {node_id: _cell_identity(row) for node_id, row in rows.items()}


def _canonical_edge_frame(edges: Iterable[tuple[int, int, str]]) -> pd.DataFrame:
    unique: dict[tuple[int, int], str] = {}
    priority = {
        "cell_internal": 0,
        "legacy_representative": 1,
        "shared_face": 2,
        "shared_face_medoid": 3,
    }
    for raw_left, raw_right, adjacency in edges:
        left, right = sorted((int(raw_left), int(raw_right)))
        if left == right:
            continue
        old = unique.get((left, right))
        if old is None or priority.get(adjacency, 99) < priority.get(old, 99):
            unique[(left, right)] = str(adjacency)
    return pd.DataFrame.from_records(
        [
            {"left_node_id": left, "right_node_id": right, "adjacency": adjacency}
            for (left, right), adjacency in sorted(unique.items())
        ],
        columns=["left_node_id", "right_node_id", "adjacency"],
    )


def build_shared_face_task_edges(
    task_nodes: pd.DataFrame,
    legacy_task_edges: pd.DataFrame,
    *,
    policy: CrossCellRepairPolicy | None = None,
) -> pd.DataFrame:
    """Add deterministic multi-probe edges for every legacy neighbour cell pair.

    The historical representative pair is retained.  Up to four probes closest
    to the shared face are matched one-to-one by Cartesian distance, and a
    separate XYZ-medoid pair is added.  No cell adjacency is invented: the
    legacy cross-cell representative graph remains the authority for which
    cells are neighbours.
    """

    active = CrossCellRepairPolicy() if policy is None else policy
    rows, cell_by_node = _task_frame_index(task_nodes)
    required_edges = {"left_node_id", "right_node_id"}
    missing = sorted(required_edges - set(legacy_task_edges.columns))
    if missing:
        raise ValueError(f"legacy task edge table missing columns: {missing}")
    nodes_by_cell: dict[CellIdentity, list[int]] = defaultdict(list)
    for node_id, cell in cell_by_node.items():
        nodes_by_cell[cell].append(node_id)
    edges: list[tuple[int, int, str]] = []
    neighbor_cells: set[tuple[CellIdentity, CellIdentity]] = set()
    for edge in legacy_task_edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        if left not in rows or right not in rows:
            raise ValueError("legacy task edge references unknown task node")
        left_cell, right_cell = cell_by_node[left], cell_by_node[right]
        if left_cell == right_cell:
            edges.append((left, right, "cell_internal"))
        else:
            edges.append((left, right, "legacy_representative"))
            neighbor_cells.add(tuple(sorted((left_cell, right_cell))))

    def xyz(node_id: int) -> np.ndarray:
        row = rows[node_id]
        return np.asarray([row.x_m, row.y_m, row.z_m], dtype=float)

    for left_cell, right_cell in sorted(neighbor_cells):
        left_nodes = tuple(sorted(nodes_by_cell[left_cell]))
        right_nodes = tuple(sorted(nodes_by_cell[right_cell]))
        if not left_nodes or not right_nodes:
            continue
        delta = np.asarray(right_cell[1:], dtype=int) - np.asarray(left_cell[1:], dtype=int)
        if np.count_nonzero(delta) == 1:
            axis = int(np.flatnonzero(delta)[0])
            direction = int(np.sign(delta[axis]))
            left_ranked = sorted(left_nodes, key=lambda node: (-direction * xyz(node)[axis], node))
            right_ranked = sorted(right_nodes, key=lambda node: (direction * xyz(node)[axis], node))
        else:
            # The registered graph can contain parent/child or diagonal links.
            # In that case closest-to-other-centroid is the deterministic face proxy.
            left_centroid = np.mean(np.vstack([xyz(node) for node in left_nodes]), axis=0)
            right_centroid = np.mean(np.vstack([xyz(node) for node in right_nodes]), axis=0)
            left_ranked = sorted(left_nodes, key=lambda node: (np.linalg.norm(xyz(node) - right_centroid), node))
            right_ranked = sorted(right_nodes, key=lambda node: (np.linalg.norm(xyz(node) - left_centroid), node))
        count = min(int(active.shared_face_probe_count), len(left_ranked), len(right_ranked))
        selected_left = left_ranked[:count]
        selected_right = right_ranked[:count]
        if count:
            cost = np.asarray(
                [
                    [float(np.linalg.norm(xyz(left) - xyz(right))) for right in selected_right]
                    for left in selected_left
                ],
                dtype=float,
            )
            # A tiny stable index term resolves exactly equal Cartesian costs.
            cost += np.arange(cost.size, dtype=float).reshape(cost.shape) * 1.0e-15
            left_indices, right_indices = linear_sum_assignment(cost)
            for left_index, right_index in zip(left_indices, right_indices, strict=True):
                edges.append(
                    (selected_left[int(left_index)], selected_right[int(right_index)], "shared_face")
                )

        all_nodes = (*left_nodes, *right_nodes)
        cloud = np.vstack([xyz(node) for node in all_nodes])
        center = np.mean(cloud, axis=0)
        left_medoid = min(left_nodes, key=lambda node: (np.linalg.norm(xyz(node) - center), node))
        right_medoid = min(right_nodes, key=lambda node: (np.linalg.norm(xyz(node) - center), node))
        edges.append((left_medoid, right_medoid, "shared_face_medoid"))
    return _canonical_edge_frame(edges)


def atlas_nodes_from_frames(task_nodes: pd.DataFrame, task_edges: pd.DataFrame) -> tuple[AtlasTaskNode, ...]:
    """Convert V14 task frames to immutable nodes for repaired graph builds."""

    rows, _cell_by_node = _task_frame_index(task_nodes)
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in rows}
    for edge in task_edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        if left not in adjacency or right not in adjacency:
            raise ValueError("repair task edge references unknown task node")
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(
        AtlasTaskNode(
            node_id=node_id,
            xyz_m=np.asarray([row.x_m, row.y_m, row.z_m], dtype=float),
            neighbor_node_ids=tuple(sorted(adjacency[node_id])),
            core_safe=bool(getattr(row, "core_safe", True)),
        )
        for node_id, row in sorted(rows.items())
    )


def make_segmented_continuation(
    continuation: ContinuationAdapter,
    node_by_id: Mapping[int, AtlasTaskNode],
    cell_by_node: Mapping[int, CellIdentity],
    *,
    step_max_mm: float = 5.0,
) -> ContinuationAdapter:
    """Subdivide only cross-cell moves; cell-internal K5 edges stay direct."""

    if not math.isfinite(float(step_max_mm)) or step_max_mm <= 0.0:
        raise ValueError("segmented continuation step must be finite and positive")

    def segmented(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
        source_node = node_by_id[source.node_id]
        if cell_by_node[source.node_id] == cell_by_node[target.node_id]:
            return continuation(source, target)
        distance_mm = float(np.linalg.norm(target.xyz_m - source_node.xyz_m) * 1000.0)
        waypoint_count = max(1, int(math.ceil(distance_mm / float(step_max_mm))))
        current = source
        total_iterations = 0
        minimum_margin: float | None = None
        statuses: list[str] = []
        last: ContinuationOutcome | None = None
        for waypoint in range(1, waypoint_count + 1):
            fraction = waypoint / waypoint_count
            xyz = source_node.xyz_m + fraction * (target.xyz_m - source_node.xyz_m)
            waypoint_node = AtlasTaskNode(
                node_id=target.node_id,
                xyz_m=xyz,
                neighbor_node_ids=(),
                core_safe=target.core_safe,
            )
            outcome = continuation(current, waypoint_node)
            last = outcome
            total_iterations += outcome.corrector_iterations
            statuses.append(outcome.status)
            if outcome.minimum_margin_deg is not None:
                minimum_margin = (
                    outcome.minimum_margin_deg
                    if minimum_margin is None
                    else min(minimum_margin, outcome.minimum_margin_deg)
                )
            if not outcome.success or not outcome.actual_bounds:
                return ContinuationOutcome(
                    beta_rad=outcome.beta_rad,
                    residual_mm=outcome.residual_mm,
                    success=False,
                    actual_bounds=outcome.actual_bounds,
                    corrector_iterations=total_iterations,
                    status=f"segmented_failed:{waypoint}/{waypoint_count}:{outcome.status}",
                    minimum_margin_deg=minimum_margin,
                    waypoint_count=waypoint,
                )
            current = AtlasCandidate(
                node_id=target.node_id,
                candidate_id=source.candidate_id,
                beta_rad=outcome.beta_rad,
                residual_mm=outcome.residual_mm,
                min_margin_deg=(
                    source.min_margin_deg
                    if outcome.minimum_margin_deg is None
                    else outcome.minimum_margin_deg
                ),
                normalized_min_margin=source.normalized_min_margin,
                posture_cost=source.posture_cost,
                condition_number=source.condition_number,
                quality=source.quality,
                solver_success=True,
                actual_bounds=True,
                cluster_id=source.cluster_id,
                diagnostics=source.diagnostics,
            )
        assert last is not None
        return ContinuationOutcome(
            beta_rad=last.beta_rad,
            residual_mm=last.residual_mm,
            success=True,
            actual_bounds=True,
            corrector_iterations=total_iterations,
            status="segmented:" + "|".join(statuses),
            minimum_margin_deg=minimum_margin,
            waypoint_count=waypoint_count,
        )

    return segmented


def _lineage_id(candidate: AtlasCandidate) -> str:
    value = candidate.diagnostics.get("lineage_id")
    if value is not None and str(value):
        return str(value)
    return f"initial:{candidate.node_id}:{candidate.candidate_id}"


def _propagated_candidate_id(
    wave: int, source: AtlasCandidate, target_node_id: int, beta: np.ndarray
) -> str:
    digest = hashlib.sha256(
        np.asarray(beta, dtype="<f8").tobytes()
        + repr((wave, source.node_id, source.candidate_id, target_node_id)).encode("utf-8")
    ).hexdigest()[:16]
    return f"prop_w{wave:02d}_{digest}"


def _directed_edge(
    source: AtlasCandidate,
    target: AtlasCandidate,
    outcome: ContinuationOutcome,
) -> DirectedContinuationEdge:
    return DirectedContinuationEdge(
        source_key=source.key,
        target_key=target.key,
        continuation_beta_rad=outcome.beta_rad,
        match_gap_deg=beta_rms_deg(outcome.beta_rad, target.beta_rad),
        residual_mm=outcome.residual_mm,
        corrector_iterations=outcome.corrector_iterations,
        status=outcome.status,
        minimum_margin_deg=outcome.minimum_margin_deg,
        waypoint_count=outcome.waypoint_count,
    )


def build_dynamic_product_graph(
    task_nodes: Sequence[AtlasTaskNode],
    candidates: Sequence[AtlasCandidate],
    continuation: ContinuationAdapter,
    *,
    atlas_policy: AtlasPolicy | None = None,
    repair_policy: CrossCellRepairPolicy | None = None,
) -> tuple[ProductGraph, tuple[PropagationRecord, ...], bool]:
    """Build a graph while inserting novel, reverse-verified endpoints."""

    graph_policy = AtlasPolicy() if atlas_policy is None else atlas_policy
    active = CrossCellRepairPolicy() if repair_policy is None else repair_policy
    nodes = tuple(task_nodes)
    node_by_id = {node.node_id: node for node in nodes}
    retained = [candidate for candidate in candidates if candidate.is_strict_feasible]
    candidates_by_node: dict[int, list[AtlasCandidate]] = defaultdict(list)
    for candidate in retained:
        if candidate.node_id not in node_by_id:
            raise ValueError("dynamic candidate references unknown task node")
        candidates_by_node[candidate.node_id].append(candidate)
    initial_keys = {candidate.key for candidate in retained}
    frontier = tuple(sorted(retained, key=lambda item: item.key))
    directed: list[DirectedContinuationEdge] = []
    records: list[PropagationRecord] = []
    attempts = 0
    rejected = 0
    budget_exhausted = False
    lineage_node_count: dict[tuple[str, int], int] = defaultdict(int)
    propagated_node_count: dict[int, int] = defaultdict(int)

    for wave in range(1, int(active.maximum_waves) + 1):
        next_frontier: list[AtlasCandidate] = []
        if not frontier:
            break
        for source in frontier:
            source_node = node_by_id[source.node_id]
            lineage = _lineage_id(source)
            for target_node_id in source_node.neighbor_node_ids:
                target_node = node_by_id[target_node_id]
                attempts += 1
                forward = continuation(source, target_node)
                if not (
                    forward.success
                    and forward.actual_bounds
                    and forward.residual_mm <= active.continuation_residual_max_mm + 1.0e-12
                ):
                    rejected += 1
                    records.append(
                        PropagationRecord(
                            wave, source.key, target_node_id, lineage,
                            "forward_failed", False, False, False, False,
                            forward.residual_mm, None, None,
                        )
                    )
                    continue
                target_candidates = candidates_by_node[target_node_id]
                matched = tuple(
                    sorted(
                        (
                            (beta_rms_deg(forward.beta_rad, target.beta_rad), target)
                            for target in target_candidates
                            if beta_rms_deg(forward.beta_rad, target.beta_rad)
                            <= active.propagated_cluster_deg + 1.0e-12
                        ),
                        key=lambda item: (item[0], item[1].candidate_id),
                    )
                )
                inserted = False
                if matched:
                    target = matched[0][1]
                else:
                    if (
                        propagated_node_count[target_node_id]
                        >= active.maximum_propagated_candidates_per_node
                        or lineage_node_count[(lineage, target_node_id)]
                        >= active.maximum_candidates_per_lineage_per_node
                    ):
                        budget_exhausted = True
                        rejected += 1
                        records.append(
                            PropagationRecord(
                                wave, source.key, target_node_id, lineage,
                                "insertion_budget_exhausted", True, False, False, False,
                                forward.residual_mm, None, None,
                            )
                        )
                        continue
                    candidate_id = _propagated_candidate_id(
                        wave, source, target_node_id, forward.beta_rad
                    )
                    target = AtlasCandidate(
                        node_id=target_node_id,
                        candidate_id=candidate_id,
                        beta_rad=forward.beta_rad,
                        residual_mm=forward.residual_mm,
                        min_margin_deg=(
                            source.min_margin_deg
                            if forward.minimum_margin_deg is None
                            else forward.minimum_margin_deg
                        ),
                        normalized_min_margin=source.normalized_min_margin,
                        posture_cost=float(np.linalg.norm(forward.beta_rad)),
                        condition_number=source.condition_number,
                        quality=source.quality,
                        solver_success=True,
                        actual_bounds=True,
                        diagnostics={
                            **dict(source.diagnostics),
                            "lineage_id": lineage,
                            "propagation_wave": wave,
                            "propagated_from": source.key,
                        },
                    )
                    candidates_by_node[target_node_id].append(target)
                    retained.append(target)
                    propagated_node_count[target_node_id] += 1
                    lineage_node_count[(lineage, target_node_id)] += 1
                    inserted = True
                directed.append(_directed_edge(source, target, forward))

                attempts += 1
                reverse = continuation(target, source_node)
                reverse_gap = beta_rms_deg(reverse.beta_rad, source.beta_rad)
                reverse_ok = bool(
                    reverse.success
                    and reverse.actual_bounds
                    and reverse.residual_mm <= active.continuation_residual_max_mm + 1.0e-12
                    and reverse_gap <= active.reverse_return_max_deg + 1.0e-12
                )
                if reverse_ok:
                    directed.append(_directed_edge(target, source, reverse))
                    if inserted and target.key not in initial_keys:
                        next_frontier.append(target)
                else:
                    rejected += 1
                records.append(
                    PropagationRecord(
                        wave=wave,
                        source_key=source.key,
                        target_node_id=target_node_id,
                        lineage_id=lineage,
                        status="bidirectional_robust" if reverse_ok else "forward_only",
                        forward_success=True,
                        reverse_success=reverse_ok,
                        inserted=inserted,
                        propagated_next_wave=bool(reverse_ok and inserted),
                        residual_mm=forward.residual_mm,
                        reverse_return_gap_deg=reverse_gap,
                        target_candidate_id=target.candidate_id,
                    )
                )
        frontier = tuple(
            sorted({candidate.key: candidate for candidate in next_frontier}.values(), key=lambda item: item.key)
        )
    graph = assemble_product_graph(
        nodes,
        retained,
        directed,
        continuation_attempt_count=attempts,
        rejected_continuation_count=rejected,
        policy=graph_policy,
    )
    return graph, tuple(records), budget_exhausted


def cross_cell_graph_metrics(
    graph: ProductGraph,
    cell_by_node: Mapping[int, CellIdentity],
) -> CrossCellGraphMetrics:
    task_pairs: set[tuple[CellIdentity, CellIdentity]] = set()
    for node in graph.task_nodes:
        for neighbor in node.neighbor_node_ids:
            if neighbor <= node.node_id:
                continue
            left_cell, right_cell = cell_by_node[node.node_id], cell_by_node[neighbor]
            if left_cell != right_cell:
                task_pairs.add(tuple(sorted((left_cell, right_cell))))
    robust_pairs: set[tuple[CellIdentity, CellIdentity]] = set()
    cell_adjacency: dict[CellIdentity, set[CellIdentity]] = {
        cell: set() for cell in set(cell_by_node.values())
    }
    robust_cross_edges = 0
    for edge in graph.robust_edges:
        left_cell = cell_by_node[edge.left_key[0]]
        right_cell = cell_by_node[edge.right_key[0]]
        if left_cell == right_cell:
            continue
        robust_cross_edges += 1
        pair = tuple(sorted((left_cell, right_cell)))
        robust_pairs.add(pair)
        cell_adjacency[left_cell].add(right_cell)
        cell_adjacency[right_cell].add(left_cell)
    component_sizes: list[int] = []
    reached: set[CellIdentity] = set()
    for root in sorted(cell_adjacency):
        if root in reached:
            continue
        pending: deque[CellIdentity] = deque([root])
        size = 0
        while pending:
            current = pending.popleft()
            if current in reached:
                continue
            reached.add(current)
            size += 1
            pending.extend(sorted(cell_adjacency[current] - reached))
        component_sizes.append(size)
    total = len(cell_adjacency)
    connected = sum(bool(neighbors) for neighbors in cell_adjacency.values())
    largest = max(component_sizes, default=0)
    return CrossCellGraphMetrics(
        task_cross_cell_pair_count=len(task_pairs),
        robust_cross_cell_pair_count=len(robust_pairs),
        cross_cell_neighbor_pair_connection_rate=float(len(robust_pairs) / max(1, len(task_pairs))),
        connected_cell_count=connected,
        total_cell_count=total,
        connected_cell_ratio=float(connected / max(1, total)),
        largest_component_cell_count=largest,
        largest_component_ratio=float(largest / max(1, total)),
        robust_cross_cell_edge_count=robust_cross_edges,
    )


def run_cross_cell_repair_ablation(
    ablation: RepairAblation,
    task_frame: pd.DataFrame,
    legacy_task_edges: pd.DataFrame,
    candidates: Sequence[AtlasCandidate],
    continuation: ContinuationAdapter,
    *,
    atlas_policy: AtlasPolicy | None = None,
    repair_policy: CrossCellRepairPolicy | None = None,
) -> CrossCellRepairResult:
    """Run one nested E0--E4 graph ablation and compute registered metrics."""

    selected = RepairAblation(ablation)
    active = CrossCellRepairPolicy() if repair_policy is None else repair_policy
    edges = (
        legacy_task_edges.copy()
        if selected is RepairAblation.E0_LEGACY
        else build_shared_face_task_edges(task_frame, legacy_task_edges, policy=active)
    )
    nodes = atlas_nodes_from_frames(task_frame, edges)
    _rows, cell_by_node = _task_frame_index(task_frame)
    node_by_id = {node.node_id: node for node in nodes}
    active_continuation = continuation
    if selected in {
        RepairAblation.E2_SEGMENTED,
        RepairAblation.E3_DYNAMIC_INSERTION,
        RepairAblation.E4_FRONTIER_ENRICHMENT,
    }:
        active_continuation = make_segmented_continuation(
            continuation,
            node_by_id,
            cell_by_node,
            step_max_mm=active.cartesian_step_max_mm,
        )
    if selected in {
        RepairAblation.E3_DYNAMIC_INSERTION,
        RepairAblation.E4_FRONTIER_ENRICHMENT,
    }:
        graph, propagation, exhausted = build_dynamic_product_graph(
            nodes,
            candidates,
            active_continuation,
            atlas_policy=atlas_policy,
            repair_policy=active,
        )
    else:
        graph = build_product_graph(
            nodes,
            candidates,
            active_continuation,
            policy=atlas_policy,
        )
        propagation = ()
        exhausted = False
    metrics = cross_cell_graph_metrics(graph, cell_by_node)
    return CrossCellRepairResult(
        ablation=selected,
        graph=graph,
        task_edges=edges,
        propagation_records=propagation,
        metrics=metrics,
        budget_exhausted=exhausted,
        diagnostics={
            "candidate_count": len(graph.candidates),
            "directed_edge_count": len(graph.directed_edges),
            "robust_edge_count": len(graph.robust_edges),
            "continuation_attempt_count": graph.continuation_attempt_count,
            "rejected_continuation_count": graph.rejected_continuation_count,
        },
    )


def _ordered_section_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def piecewise_primary_partition(
    nodes: Sequence[AtlasTaskNode],
    atlas: CanonicalAtlas,
    *,
    valid_chart_ids: Iterable[int],
    section_id_by_chart: Mapping[int, str],
    overlap_kind_by_pair: Mapping[tuple[str, str], str],
    icm_max_sweeps: int = 50,
    stitchable_edge_max_deg: float = 0.5,
) -> PiecewisePartitionResult:
    """Cover a task component by several connected chart regions.

    The objective is lexicographic and contains no tunable weighted sum:
    minimise hard incompatible neighbours, then chart switches, then global
    canonical chart rank.  Non-stitchable or indeterminate switch endpoints
    and their one-hop task neighbours become an explicit abstention band.
    """

    node_by_id = {node.node_id: node for node in nodes}
    valid = set(map(int, valid_chart_ids))
    candidates_by_node: dict[int, tuple[int, ...]] = {}
    for node_id in sorted(node_by_id):
        candidates_by_node[node_id] = tuple(
            chart.chart_id
            for chart in atlas.charts
            if chart.chart_id in valid and node_id in chart.selection_by_node
        )
    robust_pairs = {
        tuple(sorted((edge.left_key, edge.right_key))) for edge in atlas.product_graph.robust_edges
    }

    def compatible(left_node: int, left_chart: int, right_node: int, right_chart: int) -> bool:
        left = next(chart for chart in atlas.charts if chart.chart_id == left_chart)
        right = next(chart for chart in atlas.charts if chart.chart_id == right_chart)
        left_key = left.selection_by_node[left_node]
        right_key = right.selection_by_node[right_node]
        return tuple(sorted((left_key, right_key))) in robust_pairs

    assignment: dict[int, int | None] = {
        node_id: (choices[0] if choices else None)
        for node_id, choices in candidates_by_node.items()
    }
    sweeps = 0
    for sweep in range(int(icm_max_sweeps)):
        changed = False
        for node_id in sorted(assignment):
            choices = candidates_by_node[node_id]
            if not choices:
                continue
            scores: list[tuple[tuple[int, int, int], int]] = []
            for chart_id in choices:
                incompatible = 0
                switches = 0
                for neighbor_id in node_by_id[node_id].neighbor_node_ids:
                    neighbor_chart = assignment.get(neighbor_id)
                    if neighbor_chart is None:
                        continue
                    if not compatible(node_id, chart_id, neighbor_id, neighbor_chart):
                        incompatible += 1
                    if neighbor_chart != chart_id:
                        switches += 1
                scores.append(((incompatible, switches, chart_id), chart_id))
            best = min(scores)[1]
            if best != assignment[node_id]:
                assignment[node_id] = best
                changed = True
        sweeps = sweep + 1
        if not changed:
            break

    abstain: set[int] = {node_id for node_id, chart_id in assignment.items() if chart_id is None}
    required: set[tuple[str, str]] = set()
    boundary_counts: dict[str, int] = defaultdict(int)
    boundary_endpoints: set[int] = set()
    for node_id, chart_id in sorted(assignment.items()):
        if chart_id is None:
            continue
        for neighbor_id in node_by_id[node_id].neighbor_node_ids:
            if neighbor_id <= node_id:
                continue
            neighbor_chart = assignment.get(neighbor_id)
            if neighbor_chart is None or neighbor_chart == chart_id:
                continue
            left_section = section_id_by_chart[chart_id]
            right_section = section_id_by_chart[neighbor_chart]
            pair = _ordered_section_pair(left_section, right_section)
            kind_value = overlap_kind_by_pair.get(pair)
            if kind_value is None and compatible(
                node_id, chart_id, neighbor_id, neighbor_chart
            ):
                left_chart = atlas.chart_by_id[chart_id]
                right_chart = atlas.chart_by_id[neighbor_chart]
                left_candidate = atlas.product_graph.candidate_by_key[
                    left_chart.selection_by_node[node_id]
                ]
                right_candidate = atlas.product_graph.candidate_by_key[
                    right_chart.selection_by_node[neighbor_id]
                ]
                gap = beta_rms_deg(left_candidate.beta_rad, right_candidate.beta_rad)
                kind_value = (
                    "stitchable"
                    if gap <= float(stitchable_edge_max_deg) + 1.0e-12
                    else "non_stitchable"
                    if gap > 1.0 + 1.0e-12
                    else "indeterminate"
                )
            kind = str("indeterminate" if kind_value is None else kind_value)
            boundary_counts[kind] += 1
            if kind == "stitchable" and compatible(node_id, chart_id, neighbor_id, neighbor_chart):
                required.add(pair)
            else:
                boundary_endpoints.update((node_id, neighbor_id))
    for endpoint in boundary_endpoints:
        abstain.add(endpoint)
        abstain.update(node_by_id[endpoint].neighbor_node_ids)

    assignment_text: dict[int, str | None] = {}
    for node_id, chart_id in assignment.items():
        assignment_text[node_id] = (
            None if node_id in abstain or chart_id is None else section_id_by_chart[chart_id]
        )
    primary = tuple(sorted({value for value in assignment_text.values() if value is not None}))
    candidates_text = {
        node_id: tuple(section_id_by_chart[chart_id] for chart_id in chart_ids)
        for node_id, chart_ids in candidates_by_node.items()
    }
    return PiecewisePartitionResult(
        assignment_by_node=assignment_text,
        candidate_sections_by_node=candidates_text,
        abstained_node_ids=frozenset(abstain),
        primary_section_ids=primary,
        required_transitions=tuple(sorted(required)),
        icm_sweeps=sweeps,
        boundary_counts=dict(sorted(boundary_counts.items())),
    )


def evaluate_canonical_selection_stability(
    observations: Sequence[CanonicalSelectionObservation],
    *,
    lower_budget: int = 16,
    upper_budget: int = 32,
    beta_p95_max_deg: float = 0.5,
    beta_max_deg: float = 1.0,
    component_switch_ratio_max: float = 0.05,
    outgoing_neighbor_change_ratio_max: float = 0.05,
) -> CanonicalSelectionStabilityReport:
    """Evaluate the registered B16-vs-B32 canonical stability Gate."""

    by_key = {(row.node_id, row.budget): row for row in observations}
    node_ids = sorted(
        node_id
        for node_id, budget in by_key
        if budget == int(lower_budget) and (node_id, int(upper_budget)) in by_key
    )
    if not node_ids:
        return CanonicalSelectionStabilityReport(
            0, math.inf, math.inf, 1.0, 1.0, False, ("missing_paired_budget_observations",)
        )
    gaps = np.asarray(
        [
            beta_rms_deg(
                by_key[(node_id, int(lower_budget))].selected_beta_rad,
                by_key[(node_id, int(upper_budget))].selected_beta_rad,
            )
            for node_id in node_ids
        ],
        dtype=float,
    )
    component_switch = float(
        np.mean(
            [
                by_key[(node_id, int(lower_budget))].selected_component_id
                != by_key[(node_id, int(upper_budget))].selected_component_id
                for node_id in node_ids
            ]
        )
    )
    neighbor_change = float(
        np.mean(
            [
                by_key[(node_id, int(lower_budget))].outgoing_neighbor_ids
                != by_key[(node_id, int(upper_budget))].outgoing_neighbor_ids
                for node_id in node_ids
            ]
        )
    )
    p95 = float(np.percentile(gaps, 95))
    maximum = float(np.max(gaps))
    checks = {
        "beta_p95": p95 <= float(beta_p95_max_deg) + 1.0e-12,
        "beta_max": maximum <= float(beta_max_deg) + 1.0e-12,
        "component_switch": component_switch <= float(component_switch_ratio_max) + 1.0e-12,
        "outgoing_neighbor_change": neighbor_change
        <= float(outgoing_neighbor_change_ratio_max) + 1.0e-12,
    }
    return CanonicalSelectionStabilityReport(
        audited_node_count=len(node_ids),
        beta_gap_p95_deg=p95,
        beta_gap_max_deg=maximum,
        component_switch_ratio=component_switch,
        outgoing_neighbor_change_ratio=neighbor_change,
        gate_pass=all(checks.values()),
        reasons=tuple(sorted(key for key, passed in checks.items() if not passed)),
    )


def select_frontier_enrichment_nodes(
    task_nodes: pd.DataFrame,
    graph: ProductGraph,
    *,
    condition_column: str = "condition_number",
    policy: CrossCellRepairPolicy | None = None,
) -> pd.DataFrame:
    """Select deterministic E4-only difficult nodes and register start budgets."""

    active = CrossCellRepairPolicy() if policy is None else policy
    rows, cell_by_node = _task_frame_index(task_nodes)
    cross_degree: dict[int, int] = defaultdict(int)
    lineage_collisions: dict[int, int] = defaultdict(int)
    lineages_by_node: dict[int, set[str]] = defaultdict(set)
    for candidate in graph.candidates:
        lineages_by_node[candidate.node_id].add(_lineage_id(candidate))
    for node_id, lineages in lineages_by_node.items():
        lineage_collisions[node_id] = max(0, len(lineages) - 1)
    for edge in graph.robust_edges:
        left, right = edge.left_key[0], edge.right_key[0]
        if cell_by_node[left] != cell_by_node[right]:
            cross_degree[left] += 1
            cross_degree[right] += 1
    condition_values = np.asarray(
        [float(getattr(row, condition_column, 0.0)) for row in rows.values()], dtype=float
    )
    high_condition = float(np.quantile(condition_values, 2.0 / 3.0)) if len(condition_values) else 0.0
    candidates: list[dict[str, Any]] = []
    for node_id, row in sorted(rows.items()):
        flags: list[str] = []
        if cross_degree[node_id] == 0:
            flags.append("no_cross_edge")
        if lineage_collisions[node_id] > 0:
            flags.append("lineage_collision")
        condition = float(getattr(row, condition_column, 0.0))
        if condition >= high_condition:
            flags.append("top_condition_tertile")
        if not flags:
            continue
        severe = len(flags) >= 2 or "lineage_collision" in flags
        candidates.append(
            {
                "task_node_id": node_id,
                "flags": "|".join(sorted(flags)),
                "severe": severe,
                "start_budget": (
                    active.severe_enrichment_start_count
                    if severe
                    else active.ordinary_enrichment_start_count
                ),
                "condition_number": condition,
            }
        )
    maximum = min(
        int(math.floor(len(rows) * active.enrichment_fraction_max)),
        int(active.enrichment_node_max),
    )
    maximum = max(0, maximum)
    ordered = sorted(
        candidates,
        key=lambda item: (
            not item["severe"],
            -len(str(item["flags"]).split("|")),
            -float(item["condition_number"]),
            int(item["task_node_id"]),
        ),
    )
    severe_cap = int(math.floor(maximum * active.severe_enrichment_fraction_max))
    selected: list[dict[str, Any]] = []
    severe_count = 0
    for row in ordered:
        if len(selected) >= maximum:
            break
        if row["severe"] and severe_count >= severe_cap:
            continue
        selected.append(row)
        severe_count += int(bool(row["severe"]))
    return pd.DataFrame.from_records(
        selected,
        columns=["task_node_id", "flags", "severe", "start_budget", "condition_number"],
    )


def select_connected_repair_patches(
    parent_nodes: pd.DataFrame,
    parent_edges: pd.DataFrame,
    *,
    condition_by_node: Mapping[int, float],
    existing_cross_edge_node_ids: Iterable[int] = (),
    policy: CrossCellRepairPolicy | None = None,
) -> RepairPatchInventory:
    """Select the registered twelve disjoint connected 64-cell patches.

    Patch category is defined by its seed cell.  Growth is deterministic BFS
    ordered by distance from the seed, x difference, then node ID.  The
    inventory fails closed unless all exact size/connectivity and aggregate
    stratum quotas are met.
    """

    active = CrossCellRepairPolicy() if policy is None else policy
    required_nodes = {"node_id", "x_m", "strata"}
    required_edges = {"left_node_id", "right_node_id"}
    missing_nodes = sorted(required_nodes - set(parent_nodes.columns))
    missing_edges = sorted(required_edges - set(parent_edges.columns))
    if missing_nodes or missing_edges:
        raise ValueError(
            f"patch inventory missing columns: nodes={missing_nodes}, edges={missing_edges}"
        )
    if parent_nodes["node_id"].duplicated().any():
        raise ValueError("patch parent node IDs must be unique")
    rows = {int(row.node_id): row for row in parent_nodes.itertuples(index=False)}
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in rows}
    for edge in parent_edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    if any(node_id not in condition_by_node for node_id in rows):
        missing = sorted(set(rows) - set(condition_by_node))
        raise ValueError(f"patch condition inventory missing nodes: {missing[:10]}")

    node_ids = np.asarray(sorted(rows), dtype=int)
    x_values = np.asarray([float(rows[node_id].x_m) for node_id in node_ids], dtype=float)
    condition_values = np.asarray([float(condition_by_node[node_id]) for node_id in node_ids], dtype=float)
    x_cuts = np.quantile(x_values, (1.0 / 3.0, 2.0 / 3.0))
    condition_cuts = np.quantile(condition_values, (1.0 / 3.0, 2.0 / 3.0))

    def tertile(value: float, cuts: np.ndarray) -> int:
        return int(np.searchsorted(cuts, value, side="right"))

    cross_nodes = set(map(int, existing_cross_edge_node_ids))
    metadata: dict[int, dict[str, Any]] = {}
    for node_id in node_ids:
        strata = frozenset(str(rows[int(node_id)].strata).split("|"))
        metadata[int(node_id)] = {
            "x_tertile": tertile(float(rows[int(node_id)].x_m), x_cuts),
            "condition_tertile": tertile(float(condition_by_node[int(node_id)]), condition_cuts),
            "boundary": "boundary" in strata,
            "retention": "retention" in strata,
            "tip": "tip" in strata,
            "existing_cross_edge": int(node_id) in cross_nodes,
            "no_existing_cross_edge": int(node_id) not in cross_nodes,
        }

    # Development contains all tertiles; confirmation independently contains
    # every tertile and one extra special no-edge patch.
    target_x = (0, 0, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2)
    requested_role = (
        "boundary",
        "retention",
        "existing_cross_edge",
        "boundary",
        "retention",
        "existing_cross_edge",
        "no_existing_cross_edge",
        "tip",
        "boundary",
        "no_existing_cross_edge",
        "existing_cross_edge",
        "tip",
    )
    target_condition = (0, 1, 2, 0, 1, 2, 0, 2, 1, 0, 1, 2)
    used: set[int] = set()
    assignments: list[dict[str, Any]] = []

    def seed_matches(node_id: int, x_target: int, condition_target: int, role: str) -> bool:
        values = metadata[node_id]
        if (
            values["x_tertile"] != x_target
            or values["condition_tertile"] != condition_target
        ):
            return False
        return bool(values.get(role, False))

    def grow(seed: int) -> tuple[int, ...]:
        selected: list[int] = []
        seen: set[int] = set()
        pending: list[tuple[int, float, int]] = [(0, 0.0, seed)]
        seed_x = float(rows[seed].x_m)
        while pending and len(selected) < int(active.patch_size):
            distance, _x_gap, current = pending.pop(0)
            if current in seen or current in used:
                continue
            seen.add(current)
            selected.append(current)
            frontier = [
                (
                    distance + 1,
                    abs(float(rows[neighbor].x_m) - seed_x),
                    neighbor,
                )
                for neighbor in adjacency[current]
                if neighbor not in seen and neighbor not in used
            ]
            pending.extend(frontier)
            pending.sort()
        return tuple(selected)

    for patch_index, (x_target, condition_target, role) in enumerate(
        zip(target_x, target_condition, requested_role, strict=True)
    ):
        candidates = sorted(
            (
                node_id
                for node_id in rows
                if node_id not in used
                and seed_matches(node_id, x_target, condition_target, role)
            ),
            key=lambda node_id: (
                -len(adjacency[node_id]),
                -float(condition_by_node[node_id]),
                node_id,
            ),
        )
        chosen_seed: int | None = None
        chosen_patch: tuple[int, ...] = ()
        for seed in candidates:
            patch = grow(seed)
            if len(patch) == int(active.patch_size):
                chosen_seed, chosen_patch = seed, patch
                break
        if chosen_seed is None:
            raise RuntimeError(
                f"unable to construct exact connected patch {patch_index:02d} "
                f"for x_tertile={x_target}, condition_tertile={condition_target}, "
                f"role={role}"
            )
        patch_id = f"patch_{patch_index:02d}"
        split = "development" if patch_index < active.development_patch_count else "confirmation"
        for node_id in chosen_patch:
            assignments.append(
                {
                    "patch_id": patch_id,
                    "patch_index": patch_index,
                    "patch_split": split,
                    "parent_node_id": node_id,
                    "seed_node_id": chosen_seed,
                    "is_seed": node_id == chosen_seed,
                    "seed_role": role,
                    **metadata[node_id],
                }
            )
        used.update(chosen_patch)
    frame = pd.DataFrame.from_records(assignments)

    seed_frame = frame[frame["is_seed"]].copy()
    x_counts = seed_frame["x_tertile"].value_counts().to_dict()
    condition_counts = seed_frame["condition_tertile"].value_counts().to_dict()
    quota_checks = {
        "exact_patch_count": frame["patch_id"].nunique() == active.patch_count,
        "exact_patch_size": bool(frame.groupby("patch_id").size().eq(active.patch_size).all()),
        "unique_nonoverlap": not frame["parent_node_id"].duplicated().any(),
        "x_tertiles_four_each": all(int(x_counts.get(index, 0)) == 4 for index in range(3)),
        "boundary_seed_patches": int(seed_frame["boundary"].sum()) >= 3,
        "retention_seed_patches": int(seed_frame["retention"].sum()) >= 2,
        "tip_seed_patches": int(seed_frame["tip"].sum()) >= 2,
        "condition_tertiles_three_each": all(
            int(condition_counts.get(index, 0)) >= 3 for index in range(3)
        ),
        "existing_cross_edge_seed_patches": int(seed_frame["existing_cross_edge"].sum()) >= 2,
        "no_edge_seed_patches": int(seed_frame["no_existing_cross_edge"].sum()) >= 2,
        "confirmation_covers_x_tertiles": set(
            seed_frame.loc[seed_frame["patch_split"].eq("confirmation"), "x_tertile"]
        ) == {0, 1, 2},
        "confirmation_has_special_patch": bool(
            seed_frame.loc[
                seed_frame["patch_split"].eq("confirmation"), "no_existing_cross_edge"
            ].any()
        ),
    }
    # Recheck connectivity from the actual selected induced subgraph.
    connected_checks: dict[str, bool] = {}
    for patch_id, group in frame.groupby("patch_id", sort=True):
        patch_nodes = set(group["parent_node_id"].astype(int))
        reached: set[int] = set()
        pending: deque[int] = deque([min(patch_nodes)])
        while pending:
            current = pending.popleft()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(sorted((adjacency[current] & patch_nodes) - reached))
        connected_checks[str(patch_id)] = reached == patch_nodes
    quota_checks["all_patches_connected"] = all(connected_checks.values())
    audit = {
        "gate_pass": all(quota_checks.values()),
        "checks": quota_checks,
        "patch_count": int(frame["patch_id"].nunique()),
        "cell_count": len(frame),
        "x_tertile_seed_counts": {str(key): int(value) for key, value in x_counts.items()},
        "condition_tertile_seed_counts": {
            str(key): int(value) for key, value in condition_counts.items()
        },
        "connectivity_by_patch": connected_checks,
    }
    if not audit["gate_pass"]:
        failed = sorted(key for key, value in quota_checks.items() if not value)
        raise RuntimeError(f"repair patch inventory failed closed: {failed}")
    return RepairPatchInventory(frame, audit)


def evaluate_patch_ablation_gate(metrics: pd.DataFrame) -> Mapping[str, Any]:
    """Freeze E4 when it passes, otherwise E3; E1/E2 stay diagnostic-only."""

    required = {
        "patch_id",
        "patch_split",
        "ablation",
        "cross_cell_neighbor_pair_connection_rate",
        "largest_component_ratio",
        "audit_gate_pass",
        "budget_exhausted",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"patch gate table missing columns: {missing}")
    by_key = {
        (str(row.patch_id), str(row.ablation)): row
        for row in metrics.itertuples(index=False)
    }

    def method_report(method: RepairAblation) -> dict[str, Any]:
        rows = metrics[metrics["ablation"].eq(method.value)]
        checks_by_patch: dict[str, bool] = {}
        largest_by_split = {"development": 0, "confirmation": 0}
        for row in rows.itertuples(index=False):
            baseline = by_key.get((str(row.patch_id), RepairAblation.E0_LEGACY.value))
            if baseline is None:
                checks_by_patch[str(row.patch_id)] = False
                continue
            base_rate = float(baseline.cross_cell_neighbor_pair_connection_rate)
            rate = float(row.cross_cell_neighbor_pair_connection_rate)
            improvement = math.inf if base_rate <= 0.0 and rate > 0.0 else (
                rate / base_rate if base_rate > 0.0 else 0.0
            )
            micro = bool(rate >= 0.05 and improvement >= 20.0)
            audit = bool(row.audit_gate_pass) and not bool(row.budget_exhausted)
            checks_by_patch[str(row.patch_id)] = micro and audit
            if float(row.largest_component_ratio) >= 0.30 and micro and audit:
                largest_by_split[str(row.patch_split)] += 1
        gate = bool(
            len(rows) == 12
            and all(checks_by_patch.values())
            and largest_by_split["development"] >= 6
            and largest_by_split["confirmation"] >= 3
        )
        return {
            "gate_pass": gate,
            "patch_checks": checks_by_patch,
            "largest_component_pass_counts": largest_by_split,
        }

    e4 = method_report(RepairAblation.E4_FRONTIER_ENRICHMENT)
    e3 = method_report(RepairAblation.E3_DYNAMIC_INSERTION)
    selected = (
        RepairAblation.E4_FRONTIER_ENRICHMENT.value
        if e4["gate_pass"]
        else RepairAblation.E3_DYNAMIC_INSERTION.value
        if e3["gate_pass"]
        else None
    )
    return MappingProxyType(
        {
            "gate_pass": selected is not None,
            "selected_method": selected,
            "E3": e3,
            "E4": e4,
            "E1_E2_diagnostic_only": True,
        }
    )


def evaluate_repaired_pilot_gate(
    *,
    cross_cell_connection_rate: float,
    labelable_measure_ratio: float,
    largest_primary_region_measure_ratio: float,
    unresolved_abstain_ratio: float,
    selection_stability_gate: bool,
    audit_gate: bool,
    x_tertile_labelable_counts: Mapping[int, int],
) -> Mapping[str, Any]:
    checks = {
        "cross_cell_connection_rate": float(cross_cell_connection_rate) >= 0.05,
        "labelable_measure_ratio": float(labelable_measure_ratio) >= 0.30,
        "largest_primary_region_measure_ratio": float(largest_primary_region_measure_ratio) >= 0.20,
        "unresolved_abstain_ratio": float(unresolved_abstain_ratio) <= 0.70,
        "selection_stability": bool(selection_stability_gate),
        "all_atlas_audits": bool(audit_gate),
        "all_x_tertiles_nonzero": all(int(x_tertile_labelable_counts.get(index, 0)) > 0 for index in range(3)),
    }
    return MappingProxyType({"gate_pass": all(checks.values()), "checks": checks})


def evaluate_formal_admission_gate(
    *,
    reach_convergence_gate: bool,
    selection_stability_gate: bool,
    labelable_measure_ratio: float,
    minimum_x_bin_coverage: float,
    unresolved_ratio: float,
    abstention_ratio: float,
    single_cell_chart_measure_ratio: float,
    largest_region_x_bin_count: int,
    atlas_audit_gate: bool,
    representation_frozen: bool,
    n_min: int,
    student_gate: bool,
) -> Mapping[str, Any]:
    """Formal admission criteria; no threshold relaxation or row padding."""

    checks = {
        "reach_convergence": bool(reach_convergence_gate),
        "canonical_selection_stability": bool(selection_stability_gate),
        "labelable_measure": float(labelable_measure_ratio) >= 0.60,
        "minimum_x_bin_coverage": float(minimum_x_bin_coverage) >= 0.60,
        "unresolved_ratio": float(unresolved_ratio) <= 0.20,
        "abstention_ratio": float(abstention_ratio) <= 0.20,
        "single_cell_chart_measure": float(single_cell_chart_measure_ratio) <= 0.20,
        "largest_region_spans_three_x_bins": int(largest_region_x_bin_count) >= 3,
        "atlas_audits": bool(atlas_audit_gate),
        "representation_frozen": bool(representation_frozen),
        "real_budget_feasible": int(n_min) <= 200_000,
        "student_gates": bool(student_gate),
    }
    return MappingProxyType(
        {
            "gate_pass": all(checks.values()),
            "checks": checks,
            "formal_rows": 200_000,
            "padding_authorized": False,
            "deployment_claim": False,
        }
    )
