"""Branch-aware task/configuration product graphs and canonical V12 charts.

This module deliberately has a small in-memory seam.  Callers supply task
nodes, a bank of already classified IK candidates and a continuation adapter.
The module then owns the fail-closed graph construction, deterministic section
selection and chart identity rules.  In particular, a small beta distance is
never treated as proof of a graph edge: a directed edge exists only after the
adapter reports a successful predictor-corrector continuation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import heapq
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import least_squares

from .canonical import beta_rms_deg, weighted_damped_pinv


# These numbers are protocol constants, not user-tuned scoring knobs.  The
# lexicographic chart ranking below is the scientific selection rule; this
# cost is only used to make a local product-graph path deterministic.
FROZEN_EDGE_COST_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "transition": 1.0,
        "posture": 0.05,
        "condition": 0.01,
        "margin": 0.10,
    }
)

CandidateKey = tuple[int, str]


@dataclass(frozen=True)
class AtlasPolicy:
    """Frozen V12 graph/chart semantics.

    ``edge_cost_weights`` is accepted only to validate a serialised protocol;
    values different from :data:`FROZEN_EDGE_COST_WEIGHTS` are rejected.  This
    prevents a later runner from silently changing canonical ordering.
    """

    edge_match_deg: float = 0.5
    continuation_residual_max_mm: float = 3.0
    root_count: int = 8
    icm_max_sweeps: int = 50
    top_section_count: int = 8
    split_gap_deg: float = 1.0
    merge_overlap_p95_deg: float = 0.5
    merge_overlap_max_deg: float = 1.0
    edge_cost_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(FROZEN_EDGE_COST_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if self.edge_match_deg <= 0.0 or self.continuation_residual_max_mm <= 0.0:
            raise ValueError("edge match and continuation residual thresholds must be positive")
        if self.root_count < 1 or self.icm_max_sweeps < 1 or self.top_section_count < 1:
            raise ValueError("root_count, icm_max_sweeps and top_section_count must be positive")
        expected = dict(FROZEN_EDGE_COST_WEIGHTS)
        supplied = {str(name): float(value) for name, value in self.edge_cost_weights.items()}
        if supplied != expected:
            raise ValueError(
                "BACRA V12 edge cost weights are frozen: "
                f"expected {expected}, got {supplied}"
            )
        object.__setattr__(self, "edge_cost_weights", MappingProxyType(expected))


@dataclass(frozen=True)
class AtlasTaskNode:
    """One task-space node and its undirected task-graph neighbours."""

    node_id: int
    xyz_m: np.ndarray
    neighbor_node_ids: tuple[int, ...] = ()
    core_safe: bool = True

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=float).reshape(3)
        if not np.isfinite(xyz).all():
            raise ValueError("task-node xyz_m must be finite")
        neighbors = tuple(
            sorted({int(value) for value in self.neighbor_node_ids if int(value) != int(self.node_id)})
        )
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "xyz_m", xyz.copy())
        object.__setattr__(self, "neighbor_node_ids", neighbors)
        object.__setattr__(self, "core_safe", bool(self.core_safe))


@dataclass(frozen=True)
class AtlasCandidate:
    """A strict IK candidate exposed to the product-graph module.

    ``quality`` intentionally preserves ``Gold``/``Silver``.  Both are
    physically usable candidates for an atlas diagnostic, while Gold coverage
    remains a primary chart-ranking and downstream admission metric.
    """

    node_id: int
    candidate_id: str
    beta_rad: np.ndarray
    residual_mm: float
    min_margin_deg: float
    normalized_min_margin: float
    posture_cost: float = 0.0
    condition_number: float = 0.0
    quality: str = "Gold"
    solver_success: bool = True
    actual_bounds: bool = True
    cluster_id: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta_rad, dtype=float).reshape(6)
        finite_scalars = (
            self.residual_mm,
            self.min_margin_deg,
            self.normalized_min_margin,
            self.posture_cost,
            self.condition_number,
        )
        if not np.isfinite(beta).all() or not all(np.isfinite(value) for value in finite_scalars):
            raise ValueError("candidate values must be finite")
        if self.residual_mm < 0.0 or self.condition_number < 0.0:
            raise ValueError("candidate residual and condition number must be non-negative")
        quality = _quality_text(self.quality)
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "beta_rad", beta.copy())
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "solver_success", bool(self.solver_success))
        object.__setattr__(self, "actual_bounds", bool(self.actual_bounds))
        object.__setattr__(
            self, "cluster_id", None if self.cluster_id is None else int(self.cluster_id)
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def key(self) -> CandidateKey:
        return (self.node_id, self.candidate_id)

    @property
    def is_gold(self) -> bool:
        return self.quality == "Gold"

    @property
    def is_strict_feasible(self) -> bool:
        return (
            self.quality != "Reject"
            and self.solver_success
            and self.actual_bounds
            and np.isfinite(self.beta_rad).all()
            and self.residual_mm <= 3.0 + 1.0e-12
        )


@dataclass(frozen=True)
class ContinuationOutcome:
    """One actual directed local predictor-corrector result."""

    beta_rad: np.ndarray
    residual_mm: float
    success: bool
    actual_bounds: bool
    corrector_iterations: int = 0
    status: str = "unknown"
    target_cluster_id: int | None = None

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta_rad, dtype=float).reshape(6)
        if not np.isfinite(beta).all() or not np.isfinite(self.residual_mm):
            raise ValueError("continuation outcome must be finite")
        object.__setattr__(self, "beta_rad", beta.copy())
        object.__setattr__(self, "residual_mm", float(self.residual_mm))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "actual_bounds", bool(self.actual_bounds))
        object.__setattr__(self, "corrector_iterations", int(self.corrector_iterations))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(
            self,
            "target_cluster_id",
            None if self.target_cluster_id is None else int(self.target_cluster_id),
        )


ContinuationAdapter = Callable[[AtlasCandidate, AtlasTaskNode], ContinuationOutcome]


@dataclass(frozen=True)
class DirectedContinuationEdge:
    """A successful directed continuation matched to a target candidate."""

    source_key: CandidateKey
    target_key: CandidateKey
    continuation_beta_rad: np.ndarray
    match_gap_deg: float
    residual_mm: float
    corrector_iterations: int
    status: str

    def __post_init__(self) -> None:
        beta = np.asarray(self.continuation_beta_rad, dtype=float).reshape(6)
        if not np.isfinite(beta).all():
            raise ValueError("directed continuation beta must be finite")
        object.__setattr__(self, "continuation_beta_rad", beta.copy())
        object.__setattr__(self, "match_gap_deg", float(self.match_gap_deg))
        object.__setattr__(self, "residual_mm", float(self.residual_mm))
        object.__setattr__(self, "corrector_iterations", int(self.corrector_iterations))
        object.__setattr__(self, "status", str(self.status))


@dataclass(frozen=True)
class RobustBidirectionalEdge:
    """An edge admitted to canonical sections only after both directions pass."""

    left_key: CandidateKey
    right_key: CandidateKey
    forward: DirectedContinuationEdge
    reverse: DirectedContinuationEdge
    cost: float
    transition_deg: float
    posture_term: float
    condition_term: float
    margin_term: float

    def __post_init__(self) -> None:
        if self.left_key >= self.right_key:
            raise ValueError("robust edge keys must use stable canonical order")
        if self.forward.source_key != self.left_key or self.forward.target_key != self.right_key:
            raise ValueError("forward directed edge does not match robust orientation")
        if self.reverse.source_key != self.right_key or self.reverse.target_key != self.left_key:
            raise ValueError("reverse directed edge does not match robust orientation")


@dataclass(frozen=True)
class ProductGraph:
    """Immutable product-graph evidence, including rejected continuation count."""

    task_nodes: tuple[AtlasTaskNode, ...]
    candidates: tuple[AtlasCandidate, ...]
    directed_edges: tuple[DirectedContinuationEdge, ...]
    robust_edges: tuple[RobustBidirectionalEdge, ...]
    continuation_attempt_count: int
    rejected_continuation_count: int

    @property
    def node_by_id(self) -> Mapping[int, AtlasTaskNode]:
        return MappingProxyType({node.node_id: node for node in self.task_nodes})

    @property
    def candidate_by_key(self) -> Mapping[CandidateKey, AtlasCandidate]:
        return MappingProxyType({candidate.key: candidate for candidate in self.candidates})

    @property
    def directed_edge_by_keys(self) -> Mapping[tuple[CandidateKey, CandidateKey], DirectedContinuationEdge]:
        return MappingProxyType(
            {(edge.source_key, edge.target_key): edge for edge in self.directed_edges}
        )

    @property
    def robust_adjacency(self) -> Mapping[CandidateKey, tuple[tuple[CandidateKey, float], ...]]:
        adjacency: dict[CandidateKey, list[tuple[CandidateKey, float]]] = defaultdict(list)
        for edge in self.robust_edges:
            adjacency[edge.left_key].append((edge.right_key, float(edge.cost)))
            adjacency[edge.right_key].append((edge.left_key, float(edge.cost)))
        return MappingProxyType(
            {
                key: tuple(sorted(values, key=lambda item: (item[1], item[0])))
                for key, values in sorted(adjacency.items())
            }
        )


@dataclass(frozen=True)
class SectionProposal:
    """A deterministic single-valued section before duplicate-chart merging."""

    root_key: CandidateKey
    selections: tuple[tuple[int, str], ...]
    icm_sweeps: int
    metrics: Mapping[str, float]
    rank_key: tuple[Any, ...]

    def __post_init__(self) -> None:
        selections = tuple(sorted((int(node), str(candidate)) for node, candidate in self.selections))
        if len({node for node, _ in selections}) != len(selections):
            raise ValueError("a section may select at most one candidate per task node")
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def selection_by_node(self) -> Mapping[int, CandidateKey]:
        return MappingProxyType({node: (node, candidate) for node, candidate in self.selections})


@dataclass(frozen=True)
class CanonicalChart:
    """Final chart with an ID assigned only after stable global ranking."""

    chart_id: int
    root_key: CandidateKey
    selections: tuple[tuple[int, str], ...]
    metrics: Mapping[str, float]
    merged_root_keys: tuple[CandidateKey, ...] = ()

    def __post_init__(self) -> None:
        selections = tuple(sorted((int(node), str(candidate)) for node, candidate in self.selections))
        if len({node for node, _ in selections}) != len(selections):
            raise ValueError("a chart may select at most one candidate per task node")
        roots = tuple(sorted({self.root_key, *self.merged_root_keys}))
        object.__setattr__(self, "chart_id", int(self.chart_id))
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "merged_root_keys", roots)

    @property
    def selection_by_node(self) -> Mapping[int, CandidateKey]:
        return MappingProxyType({node: (node, candidate) for node, candidate in self.selections})


@dataclass(frozen=True)
class ChartOverlap:
    chart_a_id: int
    chart_b_id: int
    shared_node_ids: tuple[int, ...]
    gap_p95_deg: float
    gap_max_deg: float
    resolution: str


@dataclass(frozen=True)
class ChartMergeDecision:
    kept_root_key: CandidateKey
    merged_root_key: CandidateKey
    shared_node_count: int
    gap_p95_deg: float
    gap_max_deg: float
    resolution: str


@dataclass(frozen=True)
class CanonicalAtlas:
    product_graph: ProductGraph
    charts: tuple[CanonicalChart, ...]
    root_node_ids: tuple[int, ...]
    merge_decisions: tuple[ChartMergeDecision, ...]
    overlap_reports: tuple[ChartOverlap, ...]
    policy: AtlasPolicy

    @property
    def chart_by_id(self) -> Mapping[int, CanonicalChart]:
        return MappingProxyType({chart.chart_id: chart for chart in self.charts})


class _ForwardEnvironment(Protocol):
    bounds: np.ndarray

    def fk(self, beta: np.ndarray) -> np.ndarray: ...


def make_predictor_corrector_continuation(
    environment: _ForwardEnvironment,
    *,
    damping: float = 1.0e-3,
    beta_weights: Sequence[float] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
    max_corrector_iterations: int = 100,
    residual_tolerance_mm: float = 3.0,
) -> ContinuationAdapter:
    """Return a bounded predictor-corrector adapter for a real forward model.

    The initial predictor is never post-hoc clipped.  If it leaves mechanical
    bounds, the already strict-feasible source beta becomes the corrector
    initializer and the event is retained in ``status``.  The final solver is
    bounded directly by the authoritative environment bounds.
    """

    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    weights = np.asarray(beta_weights, dtype=float).reshape(6)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("environment bounds must have shape (6, 2) and positive spans")
    if max_corrector_iterations < 1 or residual_tolerance_mm <= 0.0:
        raise ValueError("corrector iteration and residual limits must be positive")

    def continuation(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
        try:
            jacobian = _environment_jacobian(environment, source.beta_rad)
            source_xyz = _fk_one(environment, source.beta_rad)
            predictor = source.beta_rad + weighted_damped_pinv(
                jacobian, damping=float(damping), weights=weights
            ) @ (target.xyz_m - source_xyz)
            predictor_in_bounds = bool(
                np.all(predictor >= bounds[:, 0] - 1.0e-12)
                and np.all(predictor <= bounds[:, 1] + 1.0e-12)
            )
            initial = predictor if predictor_in_bounds else source.beta_rad.copy()

            def residual(beta: np.ndarray) -> np.ndarray:
                return (_fk_one(environment, beta) - target.xyz_m) / 0.001

            solved = least_squares(
                residual,
                initial,
                bounds=(bounds[:, 0], bounds[:, 1]),
                max_nfev=int(max_corrector_iterations),
            )
            beta = np.asarray(solved.x, dtype=float).reshape(6)
            residual_mm = float(np.linalg.norm(_fk_one(environment, beta) - target.xyz_m) * 1000.0)
            actual_bounds = bool(
                np.all(beta >= bounds[:, 0] - 1.0e-12)
                and np.all(beta <= bounds[:, 1] + 1.0e-12)
            )
            return ContinuationOutcome(
                beta_rad=beta,
                residual_mm=residual_mm,
                success=bool(solved.success and residual_mm <= residual_tolerance_mm and actual_bounds),
                actual_bounds=actual_bounds,
                corrector_iterations=int(solved.nfev),
                status=(
                    str(solved.status)
                    if predictor_in_bounds
                    else f"predictor_out_of_bounds_fallback:{solved.status}"
                ),
            )
        except Exception as error:  # Continuation failure is graph evidence, not a runner crash.
            return ContinuationOutcome(
                beta_rad=source.beta_rad,
                residual_mm=float("inf") if False else 1.0e300,
                success=False,
                actual_bounds=False,
                corrector_iterations=0,
                status=f"exception:{type(error).__name__}",
            )

    return continuation


def build_product_graph(
    task_nodes: Sequence[AtlasTaskNode] | Any,
    candidates: Sequence[AtlasCandidate] | Any,
    continuation: ContinuationAdapter,
    *,
    policy: AtlasPolicy | None = None,
) -> ProductGraph:
    """Build directed continuation evidence and robust bidirectional edges.

    ``continuation`` is invoked once for every source candidate and directed
    neighbouring task node.  A returned beta is matched against the target
    node's retained candidate clusters.  No edge is inferred from raw
    candidate proximity without this actual continuation result.
    """

    active_policy = policy or AtlasPolicy()
    nodes = _coerce_task_nodes(task_nodes)
    candidate_rows = _coerce_candidates(candidates)
    node_by_id = {node.node_id: node for node in nodes}
    if not nodes:
        raise ValueError("at least one task node is required")
    for node in nodes:
        unknown = set(node.neighbor_node_ids) - set(node_by_id)
        if unknown:
            raise ValueError(f"task node {node.node_id} references unknown neighbours {sorted(unknown)}")

    retained = tuple(candidate for candidate in candidate_rows if candidate.is_strict_feasible)
    by_node: dict[int, tuple[AtlasCandidate, ...]] = {}
    for node_id in sorted(node_by_id):
        by_node[node_id] = tuple(
            sorted(
                (candidate for candidate in retained if candidate.node_id == node_id),
                key=lambda candidate: candidate.candidate_id,
            )
        )
    candidate_by_key = {candidate.key: candidate for candidate in retained}
    directed: list[DirectedContinuationEdge] = []
    attempt_count = 0
    rejected_count = 0
    for left_node_id, right_node_id in _undirected_task_edges(nodes):
        for source_node_id, target_node_id in ((left_node_id, right_node_id), (right_node_id, left_node_id)):
            target = node_by_id[target_node_id]
            target_candidates = by_node[target_node_id]
            for source in by_node[source_node_id]:
                attempt_count += 1
                outcome = _run_continuation(continuation, source, target)
                accepted = False
                if (
                    outcome.success
                    and outcome.actual_bounds
                    and outcome.residual_mm <= active_policy.continuation_residual_max_mm + 1.0e-12
                ):
                    for target_candidate in target_candidates:
                        if (
                            outcome.target_cluster_id is not None
                            and target_candidate.cluster_id is not None
                            and outcome.target_cluster_id != target_candidate.cluster_id
                        ):
                            continue
                        gap = beta_rms_deg(outcome.beta_rad, target_candidate.beta_rad)
                        if gap <= active_policy.edge_match_deg + 1.0e-12:
                            directed.append(
                                DirectedContinuationEdge(
                                    source_key=source.key,
                                    target_key=target_candidate.key,
                                    continuation_beta_rad=outcome.beta_rad,
                                    match_gap_deg=gap,
                                    residual_mm=outcome.residual_mm,
                                    corrector_iterations=outcome.corrector_iterations,
                                    status=outcome.status,
                                )
                            )
                            accepted = True
                if not accepted:
                    rejected_count += 1

    # A duplicate indicates an adapter made the same source/target solve look
    # like multiple edges.  Keep the best numerical match deterministically.
    directed_by_key: dict[tuple[CandidateKey, CandidateKey], DirectedContinuationEdge] = {}
    for edge in sorted(
        directed,
        key=lambda item: (
            item.source_key,
            item.target_key,
            item.match_gap_deg,
            item.residual_mm,
            item.corrector_iterations,
        ),
    ):
        directed_by_key.setdefault((edge.source_key, edge.target_key), edge)
    directed_rows = tuple(
        directed_by_key[key] for key in sorted(directed_by_key)
    )
    robust: list[RobustBidirectionalEdge] = []
    for (source_key, target_key), forward in directed_by_key.items():
        if source_key >= target_key:
            continue
        reverse = directed_by_key.get((target_key, source_key))
        if reverse is None:
            continue
        left = candidate_by_key[source_key]
        right = candidate_by_key[target_key]
        transition, posture, condition, margin, cost = _robust_edge_cost(left, right)
        robust.append(
            RobustBidirectionalEdge(
                left_key=source_key,
                right_key=target_key,
                forward=forward,
                reverse=reverse,
                cost=cost,
                transition_deg=transition,
                posture_term=posture,
                condition_term=condition,
                margin_term=margin,
            )
        )
    robust_rows = tuple(sorted(robust, key=lambda edge: (edge.left_key, edge.right_key)))
    return ProductGraph(
        task_nodes=nodes,
        candidates=retained,
        directed_edges=directed_rows,
        robust_edges=robust_rows,
        continuation_attempt_count=attempt_count,
        rejected_continuation_count=rejected_count,
    )


def deterministic_root_nodes(
    task_nodes: Sequence[AtlasTaskNode] | Any, *, count: int
) -> tuple[int, ...]:
    """Task-space farthest-point roots with a node-id deterministic first root."""

    nodes = _coerce_task_nodes(task_nodes)
    if count < 1:
        raise ValueError("count must be positive")
    if not nodes:
        return ()
    ordered = tuple(sorted(nodes, key=lambda node: node.node_id))
    selected = [ordered[0]]
    remaining = {node.node_id: node for node in ordered[1:]}
    while remaining and len(selected) < min(int(count), len(ordered)):
        best = max(
            remaining.values(),
            key=lambda node: (
                min(float(np.linalg.norm(node.xyz_m - root.xyz_m)) for root in selected),
                -node.node_id,
            ),
        )
        selected.append(best)
        del remaining[best.node_id]
    return tuple(node.node_id for node in selected)


def rank_root_candidates(graph: ProductGraph, node_id: int) -> tuple[AtlasCandidate, ...]:
    """Frozen root ordering: Gold, margin, residual, posture, condition, ID."""

    return tuple(
        sorted(
            (candidate for candidate in graph.candidates if candidate.node_id == int(node_id)),
            key=lambda candidate: (
                0 if candidate.is_gold else 1,
                -candidate.min_margin_deg,
                candidate.residual_mm,
                candidate.posture_cost,
                candidate.condition_number,
                candidate.candidate_id,
            ),
        )
    )


def build_canonical_atlas(
    task_nodes: Sequence[AtlasTaskNode] | Any,
    candidates: Sequence[AtlasCandidate] | Any,
    continuation: ContinuationAdapter,
    *,
    policy: AtlasPolicy | None = None,
) -> CanonicalAtlas:
    """Construct V12 product graph, deterministic sections and final chart IDs."""

    active_policy = policy or AtlasPolicy()
    graph = build_product_graph(task_nodes, candidates, continuation, policy=active_policy)
    root_ids = deterministic_root_nodes(graph.task_nodes, count=active_policy.root_count)
    proposals: list[SectionProposal] = []
    for root_node_id in root_ids:
        for root in rank_root_candidates(graph, root_node_id):
            proposal = _extract_section(graph, root.key, active_policy)
            if proposal is not None:
                proposals.append(proposal)
    proposals = sorted(proposals, key=lambda proposal: proposal.rank_key)
    retained, merge_decisions = _deduplicate_mergeable_proposals(proposals, graph, active_policy)
    final_proposals = retained[: active_policy.top_section_count]
    charts = tuple(
        CanonicalChart(
            chart_id=index,
            root_key=proposal.root_key,
            selections=proposal.selections,
            metrics=proposal.metrics,
            merged_root_keys=tuple(
                decision.merged_root_key
                for decision in merge_decisions
                if decision.kept_root_key == proposal.root_key
            ),
        )
        for index, proposal in enumerate(final_proposals)
    )
    overlap_reports = _chart_overlaps(charts, graph, active_policy)
    return CanonicalAtlas(
        product_graph=graph,
        charts=charts,
        root_node_ids=root_ids,
        merge_decisions=tuple(merge_decisions),
        overlap_reports=overlap_reports,
        policy=active_policy,
    )


def _quality_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw)
    normalized = text.lower()
    if normalized == "gold":
        return "Gold"
    if normalized == "silver":
        return "Silver"
    if normalized == "reject":
        return "Reject"
    raise ValueError(f"unknown candidate quality {value!r}")


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _coerce_task_nodes(values: Sequence[AtlasTaskNode] | Any) -> tuple[AtlasTaskNode, ...]:
    raw = _field(values, "nodes", "task_nodes", default=values)
    rows: list[AtlasTaskNode] = []
    for item in raw:
        if isinstance(item, AtlasTaskNode):
            rows.append(item)
            continue
        node_id = _field(item, "node_id", "task_node_id", "id")
        xyz = _field(item, "xyz_m", "target_xyz_m", "target_xyz")
        neighbors = _field(item, "neighbor_node_ids", "neighbors", "neighbor_ids", default=())
        core_safe = _field(item, "core_safe", "is_core_safe", default=True)
        if node_id is None or xyz is None:
            raise ValueError("task node requires node_id and xyz_m")
        rows.append(AtlasTaskNode(node_id, xyz, tuple(neighbors), bool(core_safe)))
    ordered = tuple(sorted(rows, key=lambda node: node.node_id))
    if len({node.node_id for node in ordered}) != len(ordered):
        raise ValueError("task-node IDs must be unique")
    return ordered


def _coerce_candidates(values: Sequence[AtlasCandidate] | Any) -> tuple[AtlasCandidate, ...]:
    raw = _field(values, "records", "candidates", default=values)
    rows: list[AtlasCandidate] = []
    for item in raw:
        if isinstance(item, AtlasCandidate):
            rows.append(item)
            continue
        diagnostics = _field(item, "diagnostics", default={}) or {}
        node_id = _field(item, "node_id", "task_node_id")
        candidate_id = _field(item, "candidate_id", "id")
        beta = _field(item, "beta_rad", "beta")
        if node_id is None or candidate_id is None or beta is None:
            raise ValueError("candidate requires node_id, candidate_id and beta_rad")
        rows.append(
            AtlasCandidate(
                node_id=node_id,
                candidate_id=candidate_id,
                beta_rad=beta,
                residual_mm=float(_field(item, "residual_mm", default=math.inf)),
                min_margin_deg=float(_field(item, "min_margin_deg", "minimum_margin_deg", default=-math.inf)),
                normalized_min_margin=float(
                    _field(item, "normalized_min_margin", "normalized_margin", default=0.0)
                ),
                posture_cost=float(
                    _field(item, "posture_cost", default=diagnostics.get("posture_cost", 0.0))
                ),
                condition_number=float(
                    _field(
                        item,
                        "condition_number",
                        "kappa",
                        default=diagnostics.get("kappa", diagnostics.get("condition_number", 0.0)),
                    )
                ),
                quality=_field(item, "quality", "quality_tier", default="Reject"),
                solver_success=bool(_field(item, "solver_success", default=False)),
                actual_bounds=bool(
                    _field(item, "actual_bounds", default=diagnostics.get("actual_bounds", True))
                ),
                cluster_id=_field(item, "cluster_id", default=diagnostics.get("cluster_id")),
                diagnostics=diagnostics,
            )
        )
    ordered = tuple(sorted(rows, key=lambda candidate: candidate.key))
    if len({candidate.key for candidate in ordered}) != len(ordered):
        raise ValueError("candidate IDs must be unique within each task node")
    return ordered


def _undirected_task_edges(nodes: Sequence[AtlasTaskNode]) -> tuple[tuple[int, int], ...]:
    edges = {
        tuple(sorted((node.node_id, neighbor)))
        for node in nodes
        for neighbor in node.neighbor_node_ids
    }
    return tuple(sorted(edge for edge in edges if edge[0] != edge[1]))


def _run_continuation(
    continuation: ContinuationAdapter, source: AtlasCandidate, target: AtlasTaskNode
) -> ContinuationOutcome:
    try:
        raw = continuation(source, target)
        if isinstance(raw, ContinuationOutcome):
            return raw
        if isinstance(raw, Mapping):
            return ContinuationOutcome(
                beta_rad=raw["beta_rad"],
                residual_mm=float(raw["residual_mm"]),
                success=bool(raw["success"]),
                actual_bounds=bool(raw["actual_bounds"]),
                corrector_iterations=int(raw.get("corrector_iterations", 0)),
                status=str(raw.get("status", "mapping")),
                target_cluster_id=raw.get("target_cluster_id"),
            )
        raise TypeError("continuation adapter must return ContinuationOutcome or mapping")
    except Exception as error:
        # A failed corrector is valid negative evidence.  Keep it out of the
        # graph while allowing the rest of a large candidate bank to proceed.
        return ContinuationOutcome(
            beta_rad=source.beta_rad,
            residual_mm=1.0e300,
            success=False,
            actual_bounds=False,
            corrector_iterations=0,
            status=f"adapter_exception:{type(error).__name__}",
        )


def _robust_edge_cost(
    left: AtlasCandidate, right: AtlasCandidate
) -> tuple[float, float, float, float, float]:
    transition = beta_rms_deg(left.beta_rad, right.beta_rad)
    posture = 0.5 * (left.posture_cost + right.posture_cost)
    condition = 0.5 * (
        math.log1p(max(left.condition_number, 0.0))
        + math.log1p(max(right.condition_number, 0.0))
    )
    margin = 1.0 - min(
        1.0,
        max(0.0, min(left.normalized_min_margin, right.normalized_min_margin)),
    )
    weights = FROZEN_EDGE_COST_WEIGHTS
    cost = (
        weights["transition"] * transition
        + weights["posture"] * posture
        + weights["condition"] * condition
        + weights["margin"] * margin
    )
    return float(transition), float(posture), float(condition), float(margin), float(cost)


def _environment_jacobian(environment: Any, beta: np.ndarray) -> np.ndarray:
    if hasattr(environment, "jacobian"):
        return np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6)
    if hasattr(environment, "numerical_jacobian"):
        return np.asarray(environment.numerical_jacobian(beta), dtype=float).reshape(3, 6)
    raise TypeError("forward environment must provide jacobian or numerical_jacobian")


def _fk_one(environment: _ForwardEnvironment, beta: np.ndarray) -> np.ndarray:
    return np.asarray(environment.fk(np.asarray(beta, dtype=float).reshape(1, 6)), dtype=float).reshape(-1, 3)[0]


def _dijkstra(
    adjacency: Mapping[CandidateKey, tuple[tuple[CandidateKey, float], ...]], root: CandidateKey
) -> tuple[dict[CandidateKey, float], dict[CandidateKey, CandidateKey | None]]:
    distance: dict[CandidateKey, float] = {root: 0.0}
    parent: dict[CandidateKey, CandidateKey | None] = {root: None}
    queue: list[tuple[float, CandidateKey]] = [(0.0, root)]
    while queue:
        value, current = heapq.heappop(queue)
        if value > distance.get(current, math.inf) + 1.0e-12:
            continue
        for neighbor, cost in adjacency.get(current, ()):
            candidate_distance = value + float(cost)
            old_distance = distance.get(neighbor, math.inf)
            old_parent = parent.get(neighbor)
            if (
                candidate_distance < old_distance - 1.0e-12
                or (
                    abs(candidate_distance - old_distance) <= 1.0e-12
                    and (old_parent is None or current < old_parent)
                )
            ):
                distance[neighbor] = candidate_distance
                parent[neighbor] = current
                heapq.heappush(queue, (candidate_distance, neighbor))
    return distance, parent


def _extract_section(
    graph: ProductGraph, root_key: CandidateKey, policy: AtlasPolicy
) -> SectionProposal | None:
    adjacency = graph.robust_adjacency
    if root_key not in adjacency:
        return None
    candidates = graph.candidate_by_key
    nodes = graph.node_by_id
    distance, _parent = _dijkstra(adjacency, root_key)
    choices: dict[int, tuple[CandidateKey, ...]] = {}
    for key in sorted(distance):
        choices.setdefault(key[0], tuple())
    for node_id in list(choices):
        choices[node_id] = tuple(sorted(key for key in distance if key[0] == node_id))
    if root_key[0] not in choices:
        return None
    selected: dict[int, CandidateKey] = {}
    for node_id, candidates_at_node in sorted(choices.items()):
        selected[node_id] = min(
            candidates_at_node,
            key=lambda key: (distance[key], key[1]),
        )
    selected[root_key[0]] = root_key
    pair_cost = _robust_pair_cost(graph)
    sweeps = 0
    for sweep in range(policy.icm_max_sweeps):
        changed = False
        for node_id in sorted(selected):
            if node_id == root_key[0]:
                continue
            current = selected[node_id]
            candidate_scores: list[tuple[float, CandidateKey]] = []
            for candidate_key in choices[node_id]:
                score = _candidate_unary_cost(candidates[candidate_key])
                compatible = True
                for neighbor_node_id in nodes[node_id].neighbor_node_ids:
                    neighbor_key = selected.get(neighbor_node_id)
                    if neighbor_key is None:
                        continue
                    edge_cost = pair_cost.get(_ordered_pair(candidate_key, neighbor_key))
                    if edge_cost is None:
                        compatible = False
                        break
                    score += edge_cost
                if compatible:
                    candidate_scores.append((score, candidate_key))
            if candidate_scores:
                best_score, best_key = min(candidate_scores, key=lambda item: (item[0], item[1]))
                current_score = next(
                    (score for score, key in candidate_scores if key == current), math.inf
                )
                if best_key != current and best_score <= current_score + 1.0e-12:
                    selected[node_id] = best_key
                    changed = True
        sweeps = sweep + 1
        if not changed:
            break
    metrics = _section_metrics(selected, graph, root_key)
    if metrics["task_connectivity_ratio"] < 1.0 - 1.0e-12:
        return None
    if len(selected) < 2 and len(graph.task_nodes) > 1:
        return None
    rank_key = _section_rank_key(metrics, root_key)
    return SectionProposal(
        root_key=root_key,
        selections=tuple((node_id, key[1]) for node_id, key in sorted(selected.items())),
        icm_sweeps=sweeps,
        metrics=metrics,
        rank_key=rank_key,
    )


def _candidate_unary_cost(candidate: AtlasCandidate) -> float:
    weights = FROZEN_EDGE_COST_WEIGHTS
    return float(
        weights["posture"] * candidate.posture_cost
        + weights["condition"] * math.log1p(max(candidate.condition_number, 0.0))
        + weights["margin"] * (1.0 - min(1.0, max(0.0, candidate.normalized_min_margin)))
    )


def _ordered_pair(left: CandidateKey, right: CandidateKey) -> tuple[CandidateKey, CandidateKey]:
    return (left, right) if left < right else (right, left)


def _robust_pair_cost(graph: ProductGraph) -> Mapping[tuple[CandidateKey, CandidateKey], float]:
    return MappingProxyType(
        {
            (edge.left_key, edge.right_key): float(edge.cost)
            for edge in graph.robust_edges
        }
    )


def _section_metrics(
    selected: Mapping[int, CandidateKey], graph: ProductGraph, root_key: CandidateKey
) -> dict[str, float]:
    candidates = graph.candidate_by_key
    nodes = graph.node_by_id
    pair_cost = _robust_pair_cost(graph)
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in selected}
    edge_gaps: list[float] = []
    robust_edge_count = 0
    task_edge_count = 0
    for node_id, key in sorted(selected.items()):
        for neighbor_id in nodes[node_id].neighbor_node_ids:
            if neighbor_id <= node_id or neighbor_id not in selected:
                continue
            task_edge_count += 1
            neighbor_key = selected[neighbor_id]
            if _ordered_pair(key, neighbor_key) not in pair_cost:
                continue
            adjacency[node_id].add(neighbor_id)
            adjacency[neighbor_id].add(node_id)
            robust_edge_count += 1
            edge_gaps.append(beta_rms_deg(candidates[key].beta_rad, candidates[neighbor_key].beta_rad))
    reached: set[int] = set()
    pending: deque[int] = deque([root_key[0]])
    while pending:
        current = pending.popleft()
        if current in reached or current not in adjacency:
            continue
        reached.add(current)
        pending.extend(sorted(adjacency[current] - reached))
    selected_rows = [candidates[key] for _, key in sorted(selected.items())]
    core_safe_total = max(1, sum(node.core_safe for node in graph.task_nodes))
    gold_core = sum(
        1
        for node_id, key in selected.items()
        if nodes[node_id].core_safe and candidates[key].is_gold
    )
    residual = np.asarray([candidate.residual_mm for candidate in selected_rows], dtype=float)
    margin = np.asarray([candidate.min_margin_deg for candidate in selected_rows], dtype=float)
    posture = np.asarray([candidate.posture_cost for candidate in selected_rows], dtype=float)
    condition = np.asarray([candidate.condition_number for candidate in selected_rows], dtype=float)
    gaps = np.asarray(edge_gaps, dtype=float)
    return {
        "node_coverage_ratio": float(len(selected) / max(1, len(graph.task_nodes))),
        "gold_node_coverage_ratio": float(gold_core / core_safe_total),
        "task_connectivity_ratio": float(len(reached) / max(1, len(selected))),
        "robust_edge_ratio": float(robust_edge_count / max(1, task_edge_count)),
        "minimum_joint_margin_deg": float(np.min(margin)),
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "local_smoothness_p95_deg": float(np.percentile(gaps, 95)) if len(gaps) else 0.0,
        "posture_mean": float(np.mean(posture)),
        "condition_mean": float(np.mean(condition)),
    }


def _section_rank_key(metrics: Mapping[str, float], root_key: CandidateKey) -> tuple[Any, ...]:
    # Exact V12 lexicographic order.  The robust-edge ratio is the pre-audit
    # proxy for cycle/path consistency; audit later supplies the hard evidence.
    return (
        -float(metrics["gold_node_coverage_ratio"]),
        -float(metrics["task_connectivity_ratio"]),
        -float(metrics["robust_edge_ratio"]),
        -float(metrics["minimum_joint_margin_deg"]),
        float(metrics["residual_p95_mm"]),
        float(metrics["local_smoothness_p95_deg"]),
        float(metrics["posture_mean"]),
        float(metrics["condition_mean"]),
        root_key,
    )


def _proposal_overlap(
    left: SectionProposal, right: SectionProposal, graph: ProductGraph
) -> tuple[tuple[int, ...], float, float]:
    candidates = graph.candidate_by_key
    left_map = left.selection_by_node
    right_map = right.selection_by_node
    shared = tuple(sorted(set(left_map) & set(right_map)))
    if not shared:
        return shared, 0.0, 0.0
    gaps = np.asarray(
        [
            beta_rms_deg(candidates[left_map[node]].beta_rad, candidates[right_map[node]].beta_rad)
            for node in shared
        ],
        dtype=float,
    )
    return shared, float(np.percentile(gaps, 95)), float(np.max(gaps))


def _deduplicate_mergeable_proposals(
    proposals: Sequence[SectionProposal], graph: ProductGraph, policy: AtlasPolicy
) -> tuple[list[SectionProposal], list[ChartMergeDecision]]:
    retained: list[SectionProposal] = []
    decisions: list[ChartMergeDecision] = []
    for proposal in proposals:
        matched_index: int | None = None
        matched_overlap: tuple[tuple[int, ...], float, float] | None = None
        for index, current in enumerate(retained):
            shared, p95, maximum = _proposal_overlap(current, proposal, graph)
            if not shared:
                continue
            if (
                p95 <= policy.merge_overlap_p95_deg + 1.0e-12
                and maximum <= policy.merge_overlap_max_deg + 1.0e-12
            ):
                matched_index = index
                matched_overlap = (shared, p95, maximum)
                break
        if matched_index is None:
            retained.append(proposal)
            continue
        current = retained[matched_index]
        shared, p95, maximum = matched_overlap  # type: ignore[misc]
        # The higher-ranked proposal remains the representative.  This is a
        # deduplication merge, not an unverified union of two graph sections.
        kept, merged = (
            (current, proposal)
            if current.rank_key <= proposal.rank_key
            else (proposal, current)
        )
        retained[matched_index] = kept
        decisions.append(
            ChartMergeDecision(
                kept_root_key=kept.root_key,
                merged_root_key=merged.root_key,
                shared_node_count=len(shared),
                gap_p95_deg=p95,
                gap_max_deg=maximum,
                resolution="mergeable_deduplicated",
            )
        )
    return sorted(retained, key=lambda proposal: proposal.rank_key), decisions


def _chart_overlaps(
    charts: Sequence[CanonicalChart], graph: ProductGraph, policy: AtlasPolicy
) -> tuple[ChartOverlap, ...]:
    candidates = graph.candidate_by_key
    reports: list[ChartOverlap] = []
    for left_index, left in enumerate(charts):
        left_map = left.selection_by_node
        for right in charts[left_index + 1 :]:
            right_map = right.selection_by_node
            shared = tuple(sorted(set(left_map) & set(right_map)))
            if not shared:
                continue
            gaps = np.asarray(
                [
                    beta_rms_deg(candidates[left_map[node]].beta_rad, candidates[right_map[node]].beta_rad)
                    for node in shared
                ],
                dtype=float,
            )
            p95 = float(np.percentile(gaps, 95))
            maximum = float(np.max(gaps))
            if p95 <= policy.merge_overlap_p95_deg + 1.0e-12 and maximum <= policy.merge_overlap_max_deg + 1.0e-12:
                resolution = "mergeable_duplicate_retained_separately"
            elif maximum > policy.split_gap_deg + 1.0e-12:
                resolution = "separate_distinct_branch"
            else:
                resolution = "ambiguous_keep_separate"
            reports.append(
                ChartOverlap(
                    chart_a_id=left.chart_id,
                    chart_b_id=right.chart_id,
                    shared_node_ids=shared,
                    gap_p95_deg=p95,
                    gap_max_deg=maximum,
                    resolution=resolution,
                )
            )
    return tuple(reports)
