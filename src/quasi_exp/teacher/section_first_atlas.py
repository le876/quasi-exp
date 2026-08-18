"""Rooted, beam-bounded construction of a canonical inverse section.

The V14.1 repair grows a product graph containing every retained propagated
candidate and only afterwards extracts a section.  This module reverses that
order.  Each root owns an independent chart, each ``(chart, task node)`` keeps
at most ``beam_width`` root-consistent hypotheses, and inconsistent
multi-parent endpoints are rejected while the chart is being grown.

The public interface deliberately accepts the existing immutable atlas types
and a continuation adapter.  Solver choice, FK implementation, persistence,
and experiment scheduling stay outside this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical import beta_rms_deg
from .canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    CanonicalAtlas,
    CandidateKey,
    ContinuationAdapter,
    ContinuationOutcome,
)


@dataclass(frozen=True)
class RootedSectionPolicy:
    """Registered growth, pruning, and local-consistency policy."""

    beam_width: int = 4
    root_count: int = 8
    maximum_growth_waves: int = 16
    parent_consensus_gold_deg: float = 0.5
    parent_consensus_silver_deg: float = 1.0
    continuation_residual_max_mm: float = 3.0
    reverse_return_max_deg: float = 0.5
    minimum_alternative_chart_cells: int = 8
    beta_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)
    posture_weight: float = 0.05
    margin_weight: float = 0.01
    online_cycle_repair: bool = False

    def __post_init__(self) -> None:
        if min(
            int(self.beam_width),
            int(self.root_count),
            int(self.maximum_growth_waves),
            int(self.minimum_alternative_chart_cells),
        ) < 1:
            raise ValueError("section-first integer budgets must be positive")
        if len(self.beta_weights) != 6 or any(float(value) <= 0 for value in self.beta_weights):
            raise ValueError("section-first beta_weights must contain six positive values")
        thresholds = (
            self.parent_consensus_gold_deg,
            self.parent_consensus_silver_deg,
            self.continuation_residual_max_mm,
            self.reverse_return_max_deg,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in thresholds):
            raise ValueError("section-first thresholds must be finite and positive")
        if self.parent_consensus_gold_deg > self.parent_consensus_silver_deg:
            raise ValueError("Gold parent consensus cannot exceed Silver consensus")


@dataclass(frozen=True)
class SectionHypothesis:
    chart_id: str
    root_key: CandidateKey
    node_id: int
    candidate: AtlasCandidate
    parent_keys: tuple[CandidateKey, ...]
    score: float
    wave: int
    boundary_risk: bool = False
    uses_propagated_candidate: bool = True

    def __post_init__(self) -> None:
        if self.candidate.node_id != int(self.node_id):
            raise ValueError("section hypothesis candidate must belong to its task node")
        if not math.isfinite(float(self.score)):
            raise ValueError("section hypothesis score must be finite")
        object.__setattr__(self, "chart_id", str(self.chart_id))
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "parent_keys", tuple(sorted(set(self.parent_keys))))
        object.__setattr__(self, "wave", int(self.wave))

    @property
    def key(self) -> CandidateKey:
        return self.candidate.key


@dataclass(frozen=True)
class SectionGrowthEvent:
    chart_id: str
    node_id: int
    wave: int
    reason: str
    parent_node_ids: tuple[int, ...] = ()
    endpoint_gap_deg: float | None = None
    candidate_count: int = 0
    retained_count: int = 0
    pre_pruning_proposal_count: int = 0
    pre_pruning_cluster_count: int = 0
    post_pruning_hypothesis_count: int = 0


@dataclass(frozen=True)
class CapHitEvent:
    chart_id: str
    node_id: int
    wave: int
    cap_name: str
    candidate_count: int
    retained_count: int
    pruned_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class RootedSectionChart:
    chart_id: str
    root_key: CandidateKey
    hypotheses_by_node: Mapping[int, tuple[SectionHypothesis, ...]]
    selected_by_node: Mapping[int, SectionHypothesis]
    selected_edges: tuple[tuple[int, int], ...]
    boundary_risk_node_ids: frozenset[int]
    frontier_node_ids: frozenset[int]

    def __post_init__(self) -> None:
        hypotheses = {
            int(node_id): tuple(values)
            for node_id, values in self.hypotheses_by_node.items()
        }
        selected = {int(node_id): value for node_id, value in self.selected_by_node.items()}
        if set(selected) - set(hypotheses):
            raise ValueError("selected section references nodes without hypotheses")
        if any(value.node_id != node_id for node_id, value in selected.items()):
            raise ValueError("selected section node identity mismatch")
        object.__setattr__(self, "hypotheses_by_node", MappingProxyType(hypotheses))
        object.__setattr__(self, "selected_by_node", MappingProxyType(selected))
        object.__setattr__(
            self,
            "selected_edges",
            tuple(sorted({_ordered_edge(*edge) for edge in self.selected_edges})),
        )
        object.__setattr__(self, "boundary_risk_node_ids", frozenset(self.boundary_risk_node_ids))
        object.__setattr__(self, "frontier_node_ids", frozenset(self.frontier_node_ids))


@dataclass(frozen=True)
class SectionGrowthResult:
    task_nodes: tuple[AtlasTaskNode, ...]
    charts: tuple[RootedSectionChart, ...]
    primary_chart_by_node: Mapping[int, str | None]
    abstained_node_ids: frozenset[int]
    events: tuple[SectionGrowthEvent, ...]
    cap_hit_events: tuple[CapHitEvent, ...]
    policy: RootedSectionPolicy
    continuation_attempt_count: int
    rejected_continuation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "primary_chart_by_node",
            MappingProxyType({int(key): value for key, value in self.primary_chart_by_node.items()}),
        )
        object.__setattr__(self, "abstained_node_ids", frozenset(self.abstained_node_ids))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "cap_hit_events", tuple(self.cap_hit_events))

    @property
    def covered_node_ids(self) -> frozenset[int]:
        return frozenset(
            node_id
            for node_id, chart_id in self.primary_chart_by_node.items()
            if chart_id is not None
        )

    @property
    def raw_cap_hit(self) -> bool:
        return bool(self.cap_hit_events)

    @property
    def chart_by_id(self) -> Mapping[str, RootedSectionChart]:
        return MappingProxyType({chart.chart_id: chart for chart in self.charts})

    def frames(self) -> Mapping[str, pd.DataFrame]:
        """Return stable, primitive projections for sealing and review."""

        hypotheses = []
        sections = []
        for chart in self.charts:
            for node_id, values in chart.hypotheses_by_node.items():
                selected_key = chart.selected_by_node[node_id].key
                for value in values:
                    hypotheses.append(
                        {
                            "chart_id": chart.chart_id,
                            "root_node_id": chart.root_key[0],
                            "root_candidate_id": chart.root_key[1],
                            "task_node_id": node_id,
                            "candidate_id": value.candidate.candidate_id,
                            "selected": value.key == selected_key,
                            "score": value.score,
                            "wave": value.wave,
                            "boundary_risk": value.boundary_risk,
                            # A CandidateKey mixes an integer task-node id and a
                            # string candidate id.  Keeping those pairs as one
                            # nested object column makes PyArrow infer a single
                            # scalar type for both tuple positions.  Store the
                            # two typed projections instead; matching positions
                            # still reconstruct the exact lineage.
                            "parent_node_ids": [int(key[0]) for key in value.parent_keys],
                            "parent_candidate_ids": [str(key[1]) for key in value.parent_keys],
                            "residual_mm": float(value.candidate.residual_mm),
                            "min_margin_deg": float(value.candidate.min_margin_deg),
                            "normalized_min_margin": float(
                                value.candidate.normalized_min_margin
                            ),
                            "posture_cost": float(value.candidate.posture_cost),
                            "condition_number": float(
                                value.candidate.condition_number
                            ),
                            "quality": str(value.candidate.quality),
                            "solver_success": bool(value.candidate.solver_success),
                            "actual_bounds": bool(value.candidate.actual_bounds),
                            "cluster_id": value.candidate.cluster_id,
                            **{
                                f"beta{index + 1}_rad": float(value.candidate.beta_rad[index])
                                for index in range(6)
                            },
                        }
                    )
            sections.append(
                {
                    "chart_id": chart.chart_id,
                    "root_node_id": chart.root_key[0],
                    "root_candidate_id": chart.root_key[1],
                    "selection_count": len(chart.selected_by_node),
                    "selected_edge_count": len(chart.selected_edges),
                    "boundary_risk_count": len(chart.boundary_risk_node_ids),
                    "frontier_count": len(chart.frontier_node_ids),
                }
            )
        primary = [
            {
                "task_node_id": node.node_id,
                "primary_chart_id": self.primary_chart_by_node.get(node.node_id),
                "abstained": node.node_id in self.abstained_node_ids,
            }
            for node in self.task_nodes
        ]
        events = [vars(item) for item in self.events]
        caps = [vars(item) for item in self.cap_hit_events]
        selected_edges = [
            {
                "chart_id": chart.chart_id,
                "left_node_id": left,
                "right_node_id": right,
            }
            for chart in self.charts
            for left, right in chart.selected_edges
        ]
        return MappingProxyType(
            {
                "section_hypotheses": pd.DataFrame.from_records(hypotheses),
                "section_charts": pd.DataFrame.from_records(sections),
                "primary_section": pd.DataFrame.from_records(primary),
                "growth_events": pd.DataFrame.from_records(events),
                "cap_hit_events": pd.DataFrame.from_records(caps),
                "selected_edges": pd.DataFrame.from_records(
                    selected_edges,
                    columns=("chart_id", "left_node_id", "right_node_id"),
                ),
            }
        )


@dataclass(frozen=True)
class _EndpointProposal:
    source: SectionHypothesis
    beta_rad: np.ndarray
    residual_mm: float
    minimum_margin_deg: float
    reverse_return_gap_deg: float


def build_section_first_atlas(
    task_nodes: Sequence[AtlasTaskNode],
    seed_candidates: Sequence[AtlasCandidate],
    continuation: ContinuationAdapter,
    *,
    root_keys: Sequence[CandidateKey] | None = None,
    policy: RootedSectionPolicy | None = None,
) -> SectionGrowthResult:
    """Grow deterministic, root-consistent inverse charts.

    A continuation endpoint is admitted only after a fresh reverse solve.  At
    a multi-parent node, non-stitchable disagreement terminates that chart's
    frontier instead of inserting both inverse families into one component.
    """

    active = RootedSectionPolicy() if policy is None else policy
    nodes = tuple(sorted(task_nodes, key=lambda item: item.node_id))
    if not nodes:
        raise ValueError("section-first growth requires task nodes")
    node_by_id = {node.node_id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("section-first task node IDs must be unique")
    for node in nodes:
        unknown = set(node.neighbor_node_ids) - set(node_by_id)
        if unknown:
            raise ValueError(f"task node {node.node_id} references unknown neighbours {unknown}")

    candidates = tuple(candidate for candidate in seed_candidates if candidate.is_strict_feasible)
    candidate_by_key = {candidate.key: candidate for candidate in candidates}
    if len(candidate_by_key) != len(candidates):
        raise ValueError("section-first seed candidate keys must be unique")
    selected_roots = _select_root_keys(candidates, root_keys, active)
    if not selected_roots:
        raise ValueError("section-first growth requires at least one feasible root candidate")

    charts: list[RootedSectionChart] = []
    all_events: list[SectionGrowthEvent] = []
    all_caps: list[CapHitEvent] = []
    attempts = 0
    rejected = 0
    for chart_index, root_key in enumerate(selected_roots):
        chart_id = f"chart_{chart_index:03d}"
        root = candidate_by_key[root_key]
        chart, events, caps, chart_attempts, chart_rejected = _grow_chart(
            chart_id,
            root,
            nodes,
            node_by_id,
            continuation,
            active,
        )
        charts.append(chart)
        all_events.extend(events)
        all_caps.extend(caps)
        attempts += chart_attempts
        rejected += chart_rejected

    primary: dict[int, str | None] = {}
    abstained: set[int] = set()
    for node in nodes:
        covering = [chart for chart in charts if node.node_id in chart.selected_by_node]
        if not covering:
            primary[node.node_id] = None
            abstained.add(node.node_id)
            continue
        covering.sort(
            key=lambda chart: (
                chart.selected_by_node[node.node_id].score,
                chart.chart_id,
            )
        )
        primary[node.node_id] = covering[0].chart_id

    return SectionGrowthResult(
        task_nodes=nodes,
        charts=tuple(charts),
        primary_chart_by_node=primary,
        abstained_node_ids=frozenset(abstained),
        events=tuple(all_events),
        cap_hit_events=tuple(all_caps),
        policy=active,
        continuation_attempt_count=attempts,
        rejected_continuation_count=rejected,
    )


def _select_root_keys(
    candidates: Sequence[AtlasCandidate],
    root_keys: Sequence[CandidateKey] | None,
    policy: RootedSectionPolicy,
) -> tuple[CandidateKey, ...]:
    by_key = {candidate.key: candidate for candidate in candidates}
    if root_keys is not None:
        requested = tuple(dict.fromkeys((int(key[0]), str(key[1])) for key in root_keys))
        missing = [key for key in requested if key not in by_key]
        if missing:
            raise ValueError(f"unknown or infeasible section roots: {missing}")
        return requested[: policy.root_count]
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item.is_gold else 1,
            item.posture_cost,
            -item.min_margin_deg,
            item.node_id,
            item.candidate_id,
        ),
    )
    return tuple(item.key for item in ranked[: policy.root_count])


def _grow_chart(
    chart_id: str,
    root: AtlasCandidate,
    nodes: Sequence[AtlasTaskNode],
    node_by_id: Mapping[int, AtlasTaskNode],
    continuation: ContinuationAdapter,
    policy: RootedSectionPolicy,
) -> tuple[RootedSectionChart, list[SectionGrowthEvent], list[CapHitEvent], int, int]:
    root_hypothesis = SectionHypothesis(
        chart_id=chart_id,
        root_key=root.key,
        node_id=root.node_id,
        candidate=root,
        parent_keys=(),
        score=_unary_score(root.beta_rad, root, root, policy),
        wave=0,
        uses_propagated_candidate=False,
    )
    hypotheses: dict[int, tuple[SectionHypothesis, ...]] = {root.node_id: (root_hypothesis,)}
    selected: dict[int, SectionHypothesis] = {root.node_id: root_hypothesis}
    events: list[SectionGrowthEvent] = []
    caps: list[CapHitEvent] = []
    boundary_risk: set[int] = set()
    blocked: set[int] = set()
    attempts = 0
    rejected = 0

    for wave in range(1, policy.maximum_growth_waves + 1):
        progressed = False
        accepted_before_wave = frozenset(selected)
        for target_node in nodes:
            target_id = target_node.node_id
            if target_id in selected or target_id in blocked:
                continue
            parents = [
                selected[parent_id]
                for parent_id in target_node.neighbor_node_ids
                if parent_id in accepted_before_wave
            ]
            if not parents:
                continue
            proposals: list[_EndpointProposal] = []
            for parent in sorted(parents, key=lambda item: item.key):
                attempts += 2
                forward = _coerce_outcome(continuation(parent.candidate, target_node))
                if not _usable(forward, policy):
                    rejected += 1
                    events.append(
                        SectionGrowthEvent(
                            chart_id, target_id, wave, "forward_continuation_failed",
                            (parent.node_id,), None, 1, 0,
                        )
                    )
                    continue
                candidate = _endpoint_candidate(
                    chart_id,
                    root.key,
                    target_id,
                    forward,
                    (parent.key,),
                    wave,
                    root,
                    policy,
                )
                reverse = _coerce_outcome(continuation(candidate, node_by_id[parent.node_id]))
                reverse_gap = beta_rms_deg(reverse.beta_rad, parent.candidate.beta_rad)
                if not _usable(reverse, policy) or reverse_gap > policy.reverse_return_max_deg + 1e-12:
                    rejected += 1
                    events.append(
                        SectionGrowthEvent(
                            chart_id, target_id, wave, "reverse_continuation_failed",
                            (parent.node_id,), reverse_gap, 1, 0,
                        )
                    )
                    continue
                proposals.append(
                    _EndpointProposal(
                        source=parent,
                        beta_rad=forward.beta_rad,
                        residual_mm=forward.residual_mm,
                        minimum_margin_deg=(
                            forward.minimum_margin_deg
                            if forward.minimum_margin_deg is not None
                            else parent.candidate.min_margin_deg
                        ),
                        reverse_return_gap_deg=reverse_gap,
                    )
                )
            if not proposals:
                continue

            parent_representatives = _best_by_parent_node(proposals)
            maximum_parent_gap = _maximum_endpoint_gap(parent_representatives)
            if (
                len(parent_representatives) >= 2
                and maximum_parent_gap > policy.parent_consensus_silver_deg + 1e-12
            ):
                blocked.add(target_id)
                events.append(
                    SectionGrowthEvent(
                        chart_id,
                        target_id,
                        wave,
                        "nonstitchable_parent_disagreement",
                        tuple(sorted(parent_representatives)),
                        maximum_parent_gap,
                        len(proposals),
                        0,
                    )
                )
                continue
            is_boundary = bool(
                len(parent_representatives) >= 2
                and maximum_parent_gap > policy.parent_consensus_gold_deg + 1e-12
            )
            if is_boundary:
                boundary_risk.add(target_id)
                if policy.online_cycle_repair:
                    repaired, repair_attempts = _repair_parent_consensus(
                        chart_id,
                        root,
                        target_node,
                        parent_representatives,
                        node_by_id,
                        continuation,
                        wave,
                        policy,
                    )
                    attempts += repair_attempts
                    if repaired is not None:
                        proposals = list(repaired)
                        parent_representatives = _best_by_parent_node(proposals)
                        maximum_parent_gap = 0.0
                        is_boundary = False
                        boundary_risk.discard(target_id)
                        events.append(
                            SectionGrowthEvent(
                                chart_id,
                                target_id,
                                wave,
                                "online_cycle_repaired",
                                tuple(sorted(parent.node_id for parent in parents)),
                                0.0,
                                len(parents),
                                1,
                            )
                        )

            clustered = _cluster_proposals(proposals, policy.parent_consensus_gold_deg)
            retained: list[SectionHypothesis] = []
            for cluster_index, cluster in enumerate(clustered):
                parent_keys = tuple(sorted({item.source.key for item in cluster}))
                beta = min(
                    cluster,
                    key=lambda item: (
                        _proposal_consensus_score(item, cluster, root, policy),
                        item.source.key,
                    ),
                ).beta_rad
                margin = min(item.minimum_margin_deg for item in cluster)
                residual = max(item.residual_mm for item in cluster)
                candidate = _endpoint_candidate_from_values(
                    chart_id,
                    root.key,
                    target_id,
                    beta,
                    residual,
                    margin,
                    parent_keys,
                    wave,
                    root,
                    policy,
                )
                retained.append(
                    SectionHypothesis(
                        chart_id=chart_id,
                        root_key=root.key,
                        node_id=target_id,
                        candidate=candidate,
                        parent_keys=parent_keys,
                        score=_proposal_consensus_score(
                            _EndpointProposal(
                                source=cluster[0].source,
                                beta_rad=beta,
                                residual_mm=residual,
                                minimum_margin_deg=margin,
                                reverse_return_gap_deg=max(
                                    item.reverse_return_gap_deg for item in cluster
                                ),
                            ),
                            cluster,
                            root,
                            policy,
                        ),
                        wave=wave,
                        boundary_risk=is_boundary,
                    )
                )
            retained.sort(key=lambda item: (item.score, item.candidate.candidate_id))
            if len(retained) > policy.beam_width:
                pruned = retained[policy.beam_width :]
                caps.append(
                    CapHitEvent(
                        chart_id,
                        target_id,
                        wave,
                        "beam_width",
                        len(retained),
                        policy.beam_width,
                        tuple(item.candidate.candidate_id for item in pruned),
                    )
                )
                retained = retained[: policy.beam_width]
            hypotheses[target_id] = tuple(retained)
            selected[target_id] = retained[0]
            progressed = True
            events.append(
                SectionGrowthEvent(
                    chart_id,
                    target_id,
                    wave,
                    "accepted_boundary_risk" if is_boundary else "accepted",
                    tuple(sorted(parent_representatives)),
                    maximum_parent_gap,
                    len(proposals),
                    len(retained),
                    len(proposals),
                    len(clustered),
                    len(retained),
                )
            )
        if not progressed:
            break

    selected_edges = {
        _ordered_edge(parent_key[0], hypothesis.node_id)
        for hypothesis in selected.values()
        for parent_key in hypothesis.parent_keys
        if parent_key[0] in selected
    }
    frontier = {
        node.node_id
        for node in nodes
        if node.node_id not in selected
        and any(neighbor in selected for neighbor in node.neighbor_node_ids)
    }
    return (
        RootedSectionChart(
            chart_id=chart_id,
            root_key=root.key,
            hypotheses_by_node=hypotheses,
            selected_by_node=selected,
            selected_edges=tuple(selected_edges),
            boundary_risk_node_ids=frozenset(boundary_risk),
            frontier_node_ids=frozenset(frontier),
        ),
        events,
        caps,
        attempts,
        rejected,
    )


def section_growth_from_canonical_atlas(
    atlas: CanonicalAtlas,
    *,
    policy: RootedSectionPolicy | None = None,
) -> SectionGrowthResult:
    """Adapt a V14.1 canonical atlas to the selected-section audit seam.

    This adapter never invents graph edges.  It retains exactly those robust
    product edges whose endpoint candidates are selected by the chart.
    """

    active = RootedSectionPolicy(beam_width=1) if policy is None else policy
    candidate_by_key = atlas.product_graph.candidate_by_key
    charts: list[RootedSectionChart] = []
    for source_chart in atlas.charts:
        chart_id = f"chart_{source_chart.chart_id:03d}"
        selected_keys = source_chart.selection_by_node
        selected_edges = {
            _ordered_edge(edge.left_key[0], edge.right_key[0])
            for edge in atlas.product_graph.robust_edges
            if selected_keys.get(edge.left_key[0]) == edge.left_key
            and selected_keys.get(edge.right_key[0]) == edge.right_key
        }
        parent_keys_by_node: dict[int, list[CandidateKey]] = defaultdict(list)
        for left, right in selected_edges:
            parent_keys_by_node[left].append(selected_keys[right])
            parent_keys_by_node[right].append(selected_keys[left])
        hypotheses: dict[int, tuple[SectionHypothesis, ...]] = {}
        selected: dict[int, SectionHypothesis] = {}
        for node_id, key in selected_keys.items():
            candidate = candidate_by_key[key]
            hypothesis = SectionHypothesis(
                chart_id=chart_id,
                root_key=source_chart.root_key,
                node_id=node_id,
                candidate=candidate,
                parent_keys=tuple(parent_keys_by_node.get(node_id, ())),
                score=0.0,
                wave=0,
                uses_propagated_candidate=str(candidate.diagnostics.get("source", "")).startswith("propagated"),
            )
            hypotheses[node_id] = (hypothesis,)
            selected[node_id] = hypothesis
        charts.append(
            RootedSectionChart(
                chart_id=chart_id,
                root_key=source_chart.root_key,
                hypotheses_by_node=hypotheses,
                selected_by_node=selected,
                selected_edges=tuple(selected_edges),
                boundary_risk_node_ids=frozenset(),
                frontier_node_ids=frozenset(),
            )
        )
    primary: dict[int, str | None] = {}
    abstained: set[int] = set()
    for node in atlas.product_graph.task_nodes:
        covering = [chart for chart in charts if node.node_id in chart.selected_by_node]
        if not covering:
            primary[node.node_id] = None
            abstained.add(node.node_id)
        else:
            primary[node.node_id] = min(covering, key=lambda item: item.chart_id).chart_id
    return SectionGrowthResult(
        task_nodes=atlas.product_graph.task_nodes,
        charts=tuple(charts),
        primary_chart_by_node=primary,
        abstained_node_ids=frozenset(abstained),
        events=(),
        cap_hit_events=(),
        policy=active,
        continuation_attempt_count=atlas.product_graph.continuation_attempt_count,
        rejected_continuation_count=atlas.product_graph.rejected_continuation_count,
    )


def _best_by_parent_node(proposals: Sequence[_EndpointProposal]) -> Mapping[int, _EndpointProposal]:
    output: dict[int, _EndpointProposal] = {}
    for proposal in proposals:
        node_id = proposal.source.node_id
        current = output.get(node_id)
        if current is None or (proposal.residual_mm, proposal.source.key) < (
            current.residual_mm,
            current.source.key,
        ):
            output[node_id] = proposal
    return output


def _repair_parent_consensus(
    chart_id: str,
    root: AtlasCandidate,
    target_node: AtlasTaskNode,
    proposals_by_parent: Mapping[int, _EndpointProposal],
    node_by_id: Mapping[int, AtlasTaskNode],
    continuation: ContinuationAdapter,
    wave: int,
    policy: RootedSectionPolicy,
) -> tuple[tuple[_EndpointProposal, ...] | None, int]:
    proposals = tuple(proposals_by_parent.values())
    mean_beta = np.mean(np.vstack([item.beta_rad for item in proposals]), axis=0)
    provisional = _endpoint_candidate_from_values(
        chart_id,
        root.key,
        target_node.node_id,
        mean_beta,
        max(item.residual_mm for item in proposals),
        min(item.minimum_margin_deg for item in proposals),
        tuple(item.source.key for item in proposals),
        wave,
        root,
        policy,
    )
    corrected = _coerce_outcome(continuation(provisional, target_node))
    attempts = 1
    if not _usable(corrected, policy):
        return None, attempts
    corrected_candidate = _endpoint_candidate(
        chart_id,
        root.key,
        target_node.node_id,
        corrected,
        tuple(item.source.key for item in proposals),
        wave,
        root,
        policy,
    )
    repaired: list[_EndpointProposal] = []
    for item in proposals:
        reverse = _coerce_outcome(
            continuation(corrected_candidate, node_by_id[item.source.node_id])
        )
        attempts += 1
        gap = beta_rms_deg(reverse.beta_rad, item.source.candidate.beta_rad)
        if not _usable(reverse, policy) or gap > policy.reverse_return_max_deg + 1e-12:
            return None, attempts
        repaired.append(
            _EndpointProposal(
                source=item.source,
                beta_rad=corrected.beta_rad,
                residual_mm=corrected.residual_mm,
                minimum_margin_deg=(
                    corrected.minimum_margin_deg
                    if corrected.minimum_margin_deg is not None
                    else item.minimum_margin_deg
                ),
                reverse_return_gap_deg=gap,
            )
        )
    return tuple(repaired), attempts


def _maximum_endpoint_gap(proposals: Mapping[int, _EndpointProposal]) -> float:
    values = list(proposals.values())
    return max(
        (
            beta_rms_deg(left.beta_rad, right.beta_rad)
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        ),
        default=0.0,
    )


def _cluster_proposals(
    proposals: Sequence[_EndpointProposal], threshold_deg: float
) -> tuple[tuple[_EndpointProposal, ...], ...]:
    clusters: list[list[_EndpointProposal]] = []
    for proposal in sorted(proposals, key=lambda item: item.source.key):
        for cluster in clusters:
            if all(
                beta_rms_deg(proposal.beta_rad, existing.beta_rad) <= threshold_deg + 1e-12
                for existing in cluster
            ):
                cluster.append(proposal)
                break
        else:
            clusters.append([proposal])
    return tuple(tuple(cluster) for cluster in clusters)


def _proposal_consensus_score(
    proposal: _EndpointProposal,
    cluster: Sequence[_EndpointProposal],
    root: AtlasCandidate,
    policy: RootedSectionPolicy,
) -> float:
    weights = np.asarray(policy.beta_weights, dtype=float)
    gaps = [
        _weighted_beta_rms_deg(proposal.beta_rad, item.beta_rad, weights)
        for item in cluster
    ]
    posture = _weighted_beta_rms_deg(proposal.beta_rad, root.beta_rad, weights)
    margin_penalty = 1.0 / max(1e-6, proposal.minimum_margin_deg)
    return float(
        np.mean(gaps)
        + policy.posture_weight * posture
        + policy.margin_weight * margin_penalty
        + 1e-3 * proposal.residual_mm
    )


def _unary_score(
    beta: np.ndarray,
    candidate: AtlasCandidate,
    root: AtlasCandidate,
    policy: RootedSectionPolicy,
) -> float:
    weights = np.asarray(policy.beta_weights, dtype=float)
    return float(
        policy.posture_weight * _weighted_beta_rms_deg(beta, root.beta_rad, weights)
        + policy.margin_weight / max(candidate.min_margin_deg, 1e-6)
        + 1e-3 * candidate.residual_mm
    )


def _weighted_beta_rms_deg(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    delta_deg = np.degrees(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    return float(np.sqrt(np.mean(np.square(weights * delta_deg))))


def _endpoint_candidate(
    chart_id: str,
    root_key: CandidateKey,
    node_id: int,
    outcome: ContinuationOutcome,
    parent_keys: Sequence[CandidateKey],
    wave: int,
    root: AtlasCandidate,
    policy: RootedSectionPolicy,
) -> AtlasCandidate:
    margin = outcome.minimum_margin_deg if outcome.minimum_margin_deg is not None else root.min_margin_deg
    return _endpoint_candidate_from_values(
        chart_id,
        root_key,
        node_id,
        outcome.beta_rad,
        outcome.residual_mm,
        margin,
        parent_keys,
        wave,
        root,
        policy,
    )


def _endpoint_candidate_from_values(
    chart_id: str,
    root_key: CandidateKey,
    node_id: int,
    beta_rad: np.ndarray,
    residual_mm: float,
    margin_deg: float,
    parent_keys: Sequence[CandidateKey],
    wave: int,
    root: AtlasCandidate,
    policy: RootedSectionPolicy,
) -> AtlasCandidate:
    beta = np.asarray(beta_rad, dtype=float).reshape(6)
    digest = hashlib.sha256(
        b"|".join(
            (
                chart_id.encode(),
                str(root_key).encode(),
                str(node_id).encode(),
                beta.tobytes(),
                str(tuple(sorted(parent_keys))).encode(),
            )
        )
    ).hexdigest()[:16]
    candidate = AtlasCandidate(
        node_id=node_id,
        candidate_id=f"sf_w{wave:02d}_{digest}",
        beta_rad=beta,
        residual_mm=float(residual_mm),
        min_margin_deg=float(margin_deg),
        normalized_min_margin=max(0.0, float(margin_deg)) / 180.0,
        posture_cost=_weighted_beta_rms_deg(
            beta, root.beta_rad, np.asarray(policy.beta_weights, dtype=float)
        ),
        condition_number=root.condition_number,
        quality="Gold" if margin_deg >= 1.5 else "Silver",
        solver_success=True,
        actual_bounds=True,
        diagnostics={"chart_id": chart_id, "root_key": root_key, "wave": wave},
    )
    return candidate


def _usable(outcome: ContinuationOutcome, policy: RootedSectionPolicy) -> bool:
    return bool(
        outcome.success
        and outcome.actual_bounds
        and outcome.residual_mm <= policy.continuation_residual_max_mm + 1e-12
    )


def _coerce_outcome(value: ContinuationOutcome) -> ContinuationOutcome:
    if not isinstance(value, ContinuationOutcome):
        raise TypeError("section-first continuation must return ContinuationOutcome")
    return value


def _ordered_edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)
