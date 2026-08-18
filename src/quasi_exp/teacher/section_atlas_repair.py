"""Audit-v2 and stitched-primary repair for rooted inverse sections.

This module owns the scientific seam introduced by BACRA V14.2R.  It keeps
three questions separate:

* did a fresh numerical continuation finish (``solver_gate``),
* do the completed continuations agree geometrically (``geometry_gate``), and
* are repeated source perturbations stable (``repeat_gate``), and
* is the exact retained primary atlas fully supported by all three kinds of
  evidence (``certificate_gate``)?

The input remains :class:`SectionGrowthResult`; no historical artifact type is
mutated.  A repaired primary is a stitchable chart component (a singleton is a
valid component), never an unconstrained pointwise mixture of inverse branches.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical import beta_rms_deg
from .canonical_gauge import (
    AnchorComponentSelection,
    CanonicalAnchorPolicy,
    select_anchor_locked_component,
)
from .holonomy_diagnostics import (
    physical_audit_direction,
    physical_audit_entity_id,
    physical_repeat_perturbation,
)
from .canonical_atlas import AtlasCandidate, AtlasTaskNode, ContinuationAdapter, ContinuationOutcome
from .section_first_atlas import (
    RootedSectionChart,
    RootedSectionPolicy,
    SectionGrowthResult,
    SectionHypothesis,
)


@dataclass(frozen=True)
class RetryTier:
    tier_id: str
    maximum_step_mm: float
    maximum_iterations: int
    solver_chain: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.maximum_step_mm <= 0 or self.maximum_iterations < 1 or not self.solver_chain:
            raise ValueError("retry tier budgets must be positive")


def _default_retry_tiers() -> tuple[RetryTier, ...]:
    return (
        RetryTier("R0", 5.0, 100, ("predictor", "bounded_ls")),
        RetryTier("R1", 2.5, 200, ("weighted_dls", "bounded_ls")),
        RetryTier("R2", 1.25, 400, ("weighted_dls", "bounded_ls", "slsqp")),
    )


@dataclass(frozen=True)
class AuditV2Policy:
    geometry_p95_max_deg: float = 0.5
    geometry_max_deg: float = 1.0
    repeat_p95_max_deg: float = 0.2
    continuation_residual_max_mm: float = 3.0
    repeats_per_direction: int = 3
    repeat_perturbation_rad: float = 1.0e-8
    retry_tiers: tuple[RetryTier, ...] = field(default_factory=_default_retry_tiers)

    def __post_init__(self) -> None:
        values = (
            self.geometry_p95_max_deg,
            self.geometry_max_deg,
            self.repeat_p95_max_deg,
            self.continuation_residual_max_mm,
            self.repeat_perturbation_rad,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            raise ValueError("audit-v2 thresholds must be finite and positive")
        if self.repeats_per_direction < 1 or not self.retry_tiers:
            raise ValueError("audit-v2 requires repeats and retry tiers")


@dataclass(frozen=True)
class AtlasRepairPolicy:
    audit: AuditV2Policy = field(default_factory=AuditV2Policy)
    minimum_chart_cells: int = 8
    minimum_chart_fraction: float = 0.0
    minimum_chart_spread_mm: float = 0.0
    minimum_stitch_overlap_cells: int = 8
    stitch_p95_max_deg: float = 0.5
    stitch_max_deg: float = 1.0
    minimum_boundary_transition_edges: int = 2
    minimum_overlap_fraction: float = 0.0
    minimum_overlap_spread_mm: float = 0.0
    abstention_hops: int = 1
    abstention_radius_mm: float = 10.0

    def __post_init__(self) -> None:
        if self.minimum_chart_cells < 1 or self.minimum_stitch_overlap_cells < 1 or self.minimum_boundary_transition_edges < 1:
            raise ValueError("chart and overlap cell counts must be positive")
        if self.abstention_hops < 0:
            raise ValueError("abstention_hops cannot be negative")
        if self.stitch_p95_max_deg <= 0 or self.stitch_max_deg <= 0:
            raise ValueError("stitch thresholds must be positive")
        if not 0.0 <= self.minimum_overlap_fraction <= 1.0:
            raise ValueError("minimum_overlap_fraction must be a fraction")
        if not 0.0 <= self.minimum_chart_fraction <= 1.0:
            raise ValueError("minimum_chart_fraction must be a fraction")
        if (
            self.minimum_chart_spread_mm < 0
            or self.minimum_overlap_spread_mm < 0
            or self.abstention_radius_mm < 0
        ):
            raise ValueError("spatial stitch/abstention widths cannot be negative")


@dataclass(frozen=True)
class RootedArtifactDiagnostic:
    geometry_gate: bool
    solver_gate: bool
    repeat_gate: bool
    certificate_gate: bool
    geometry_metrics: Mapping[str, Any]
    solver_metrics: Mapping[str, Any]
    repeat_metrics: Mapping[str, Any]
    chart_gate_by_id: Mapping[str, bool]
    chart_metrics_by_id: Mapping[str, Mapping[str, Any]]
    schedules: pd.DataFrame
    executions: pd.DataFrame
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_metrics", MappingProxyType(dict(self.geometry_metrics)))
        object.__setattr__(self, "solver_metrics", MappingProxyType(dict(self.solver_metrics)))
        object.__setattr__(self, "repeat_metrics", MappingProxyType(dict(self.repeat_metrics)))
        object.__setattr__(self, "chart_gate_by_id", MappingProxyType(dict(self.chart_gate_by_id)))
        object.__setattr__(
            self,
            "chart_metrics_by_id",
            MappingProxyType({key: MappingProxyType(dict(value)) for key, value in self.chart_metrics_by_id.items()}),
        )
        object.__setattr__(self, "schedules", self.schedules.copy())
        object.__setattr__(self, "executions", self.executions.copy())


@dataclass
class RepairedSectionAtlas:
    growth: SectionGrowthResult
    diagnostic: RootedArtifactDiagnostic
    qualified_chart_ids: tuple[str, ...]
    stitch_edges: tuple[tuple[str, str], ...]
    selected_stitch_component: tuple[str, ...]
    primary_chart_by_node: dict[int, str]
    primary_beta_by_node: dict[int, np.ndarray]
    abstained_node_ids: frozenset[int]
    coverage_ratio: float
    largest_coherent_region_ratio: float
    geometry_gate: bool
    solver_gate: bool
    repeat_gate: bool
    certificate_gate: bool
    induced_edge_count: int
    audited_edge_count: int
    edge_completeness_ratio: float
    pre_abstention_cycle_rank: int
    retained_cycle_rank: int
    audited_fundamental_cycle_count: int
    cycle_coverage_ratio: float
    canonical_anchor_root_key: CandidateKey | None
    canonical_anchor_qualified: bool
    canonical_anchor_component_selected: bool
    canonical_anchor_component_coverage: float
    canonical_anchor_fallback_reason: str | None
    selected_anchor_rank: int | None
    frames: Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class PrimaryAtlasStability:
    gate_pass: bool
    coverage_jaccard: float
    beta_p95_deg: float
    beta_max_deg: float
    beta_disagreement_ratio_gt_1deg: float
    compared_node_count: int


def section_growth_from_frames(
    task_nodes: Sequence[AtlasTaskNode],
    section_hypotheses: pd.DataFrame,
    selected_edges: pd.DataFrame,
    *,
    policy: RootedSectionPolicy | None = None,
) -> SectionGrowthResult:
    """Rehydrate the exact selected rooted charts from parquet-safe frames.

    This loader is intentionally limited to the selected section needed by
    audit-v2 and stitching.  It does not pretend to recover solver diagnostics
    that were never persisted in the V14.2 hypothesis frame.
    """

    required = {
        "chart_id", "root_node_id", "root_candidate_id", "task_node_id",
        "candidate_id", "selected", "score", "wave",
        "parent_node_ids", "parent_candidate_ids",
        *(f"beta{index}_rad" for index in range(1, 7)),
    }
    missing = required - set(section_hypotheses)
    if missing:
        raise ValueError(f"section hypothesis frame missing columns: {sorted(missing)}")
    nodes = tuple(sorted(task_nodes, key=lambda item: item.node_id))
    charts: list[RootedSectionChart] = []
    for chart_id, rows in section_hypotheses.groupby("chart_id", sort=True):
        hypotheses_by_node: dict[int, tuple[SectionHypothesis, ...]] = {}
        selected_by_node: dict[int, SectionHypothesis] = {}
        first = rows.iloc[0]
        root_key = (int(first["root_node_id"]), str(first["root_candidate_id"]))
        for node_id, node_rows in rows.groupby("task_node_id", sort=True):
            values: list[SectionHypothesis] = []
            for row in node_rows.itertuples(index=False):
                def optional(name: str, default: Any) -> Any:
                    value = getattr(row, name, default)
                    return default if pd.isna(value) else value

                candidate = AtlasCandidate(
                    node_id=int(node_id),
                    candidate_id=str(row.candidate_id),
                    beta_rad=np.asarray([getattr(row, f"beta{index}_rad") for index in range(1, 7)]),
                    residual_mm=float(optional("residual_mm", 0.0)),
                    min_margin_deg=float(optional("min_margin_deg", 1.0)),
                    normalized_min_margin=float(
                        optional("normalized_min_margin", 0.1)
                    ),
                    posture_cost=float(optional("posture_cost", 0.0)),
                    condition_number=float(optional("condition_number", 0.0)),
                    quality=str(optional("quality", "Gold")),
                    solver_success=bool(optional("solver_success", True)),
                    actual_bounds=bool(optional("actual_bounds", True)),
                    cluster_id=(
                        None
                        if optional("cluster_id", None) is None
                        else int(optional("cluster_id", None))
                    ),
                    diagnostics={"rehydrated_selected_section": True},
                )
                parent_nodes = tuple(map(int, row.parent_node_ids))
                parent_candidates = tuple(map(str, row.parent_candidate_ids))
                if len(parent_nodes) != len(parent_candidates):
                    raise ValueError("persisted parent lineage columns have different lengths")
                hypothesis = SectionHypothesis(
                    chart_id=str(chart_id),
                    root_key=root_key,
                    node_id=int(node_id),
                    candidate=candidate,
                    parent_keys=tuple(zip(parent_nodes, parent_candidates, strict=True)),
                    score=float(row.score),
                    wave=int(row.wave),
                    boundary_risk=bool(getattr(row, "boundary_risk", False)),
                    uses_propagated_candidate=int(node_id) != root_key[0],
                )
                values.append(hypothesis)
                if bool(row.selected):
                    if int(node_id) in selected_by_node:
                        raise ValueError("persisted section has multiple selected hypotheses at one chart/node")
                    selected_by_node[int(node_id)] = hypothesis
            hypotheses_by_node[int(node_id)] = tuple(values)
        edge_rows = selected_edges[selected_edges["chart_id"].astype(str).eq(str(chart_id))]
        edges = tuple(
            (int(row.left_node_id), int(row.right_node_id))
            for row in edge_rows.itertuples(index=False)
            if int(row.left_node_id) in selected_by_node and int(row.right_node_id) in selected_by_node
        )
        charts.append(
            RootedSectionChart(
                chart_id=str(chart_id),
                root_key=root_key,
                hypotheses_by_node=hypotheses_by_node,
                selected_by_node=selected_by_node,
                selected_edges=edges,
                boundary_risk_node_ids=frozenset(
                    node for node, item in selected_by_node.items() if item.boundary_risk
                ),
                frontier_node_ids=frozenset(),
            )
        )
    primary: dict[int, str | None] = {}
    for node in nodes:
        covering = [chart.chart_id for chart in charts if node.node_id in chart.selected_by_node]
        primary[node.node_id] = min(covering) if covering else None
    return SectionGrowthResult(
        task_nodes=nodes,
        charts=tuple(charts),
        primary_chart_by_node=primary,
        abstained_node_ids=frozenset(node for node, chart in primary.items() if chart is None),
        events=(),
        cap_hit_events=(),
        policy=RootedSectionPolicy() if policy is None else policy,
        continuation_attempt_count=0,
        rejected_continuation_count=0,
    )


def diagnose_rooted_section_artifacts(
    growth: SectionGrowthResult,
    *,
    schedules: Sequence[Mapping[str, Any]] | pd.DataFrame,
    executions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    policy: AuditV2Policy | None = None,
    primary_chart_ids: Sequence[str] | None = None,
) -> RootedArtifactDiagnostic:
    """Diagnose persisted audit-v2 records without conflating missing solves.

    Missing executions increase ``missing_count`` and fail ``solver_gate``.
    They are deliberately excluded from the distribution used by the geometry
    Gate.  Thus an all-missing diagnostic has a vacuously true geometry Gate,
    but can never obtain a certificate.
    """

    active = AuditV2Policy() if policy is None else policy
    schedule_frame = _records_frame(schedules)
    execution_frame = _records_frame(executions)
    if schedule_frame.empty:
        geometry = {
            "sample_count": 0,
            "missing_count": 0,
            "p95_deg": math.nan,
            "max_deg": math.nan,
            "gate_pass": True,
        }
        solver = {
            "execution_count": 0,
            "success_count": 0,
            "missing_count": 0,
            "recoverable_numerical_count": 0,
            "persistent_numerical_count": 0,
            "geometric_branch_disagreement_count": 0,
            "geometric_branch_disagreement_execution_count": 0,
            "geometric_branch_disagreement_unique_entity_count": 0,
            "local_feasibility_failure_count": 0,
            "gate_pass": False,
        }
        repeat = {
            "sample_count": 0,
            "missing_count": 0,
            "p95_deg": math.nan,
            "gate_pass": True,
        }
        return RootedArtifactDiagnostic(
            geometry_gate=True,
            solver_gate=False,
            repeat_gate=True,
            certificate_gate=False,
            geometry_metrics=geometry,
            solver_metrics=solver,
            repeat_metrics=repeat,
            chart_gate_by_id={},
            chart_metrics_by_id={},
            schedules=schedule_frame,
            executions=execution_frame,
            failure_reasons=("empty_schedule_set", "solver_gate_failed"),
        )
    required_schedule_columns = {
        "schedule_id", "unique_entity_id", "chart_id", "audit_kind",
        "path_node_ids", "primary_usage",
    }
    missing_columns = required_schedule_columns - set(schedule_frame)
    if missing_columns:
        raise ValueError(f"audit-v2 schedules missing columns: {sorted(missing_columns)}")
    if schedule_frame["schedule_id"].duplicated().any():
        raise ValueError("audit-v2 schedule_id must be unique")
    if not execution_frame.empty and "schedule_id" not in execution_frame:
        raise ValueError("audit-v2 executions require schedule_id")
    if execution_frame.empty and "schedule_id" not in execution_frame:
        execution_frame["schedule_id"] = pd.Series(dtype=str)
    unknown = set(execution_frame.get("schedule_id", ())) - set(schedule_frame["schedule_id"])
    if unknown:
        raise ValueError(f"audit-v2 executions reference unknown schedules: {sorted(unknown)}")

    execution_frame = execution_frame.copy()
    for column, default in (
        ("solver_success", False),
        ("geometry_gap_deg", math.nan),
        ("repeat_gap_deg", math.nan),
        ("residual_mm", math.nan),
        ("classification", "persistent_numerical"),
    ):
        if column not in execution_frame:
            execution_frame[column] = default
    execution_payload = execution_frame.drop(columns=["chart_id"], errors="ignore")
    merged = schedule_frame[["schedule_id", "chart_id", "primary_usage"]].merge(
        execution_payload, on="schedule_id", how="left", validate="one_to_many"
    )
    if primary_chart_ids is None:
        primary_mask = merged["primary_usage"].fillna(False).astype(bool)
    else:
        primary_mask = merged["chart_id"].astype(str).isin({str(value) for value in primary_chart_ids})
    primary = merged.loc[primary_mask].copy()
    if primary.empty:
        primary = merged.copy()

    geometry = _geometry_metrics(primary, active)
    solver = _solver_metrics(primary, active)
    repeat = _repeat_metrics(primary, active)
    chart_gate: dict[str, bool] = {}
    chart_metrics: dict[str, dict[str, Any]] = {}
    for chart_id, rows in merged.groupby("chart_id", sort=True):
        chart_geometry = _geometry_metrics(rows, active)
        chart_solver = _solver_metrics(rows, active)
        chart_repeat = _repeat_metrics(rows, active)
        gate = bool(
            chart_geometry["gate_pass"]
            and chart_solver["gate_pass"]
            and chart_repeat["gate_pass"]
        )
        chart_gate[str(chart_id)] = gate
        chart_metrics[str(chart_id)] = {
            "geometry_gate": bool(chart_geometry["gate_pass"]),
            "solver_gate": bool(chart_solver["gate_pass"]),
            "repeat_gate": bool(chart_repeat["gate_pass"]),
            "certificate_gate": gate,
            "geometry_p95_deg": chart_geometry["p95_deg"],
            "geometry_max_deg": chart_geometry["max_deg"],
            "missing_count": chart_solver["missing_count"],
            "persistent_numerical_count": chart_solver["persistent_numerical_count"],
            "repeat_p95_deg": chart_repeat["p95_deg"],
            "repeat_missing_count": chart_repeat["missing_count"],
        }
    certificate = bool(
        geometry["gate_pass"] and solver["gate_pass"] and repeat["gate_pass"]
    )
    reasons: list[str] = []
    if not geometry["gate_pass"]:
        reasons.append("geometry_gate_failed")
    if not solver["gate_pass"]:
        reasons.append("solver_gate_failed")
    if not repeat["gate_pass"]:
        reasons.append("repeat_gate_failed")
    return RootedArtifactDiagnostic(
        geometry_gate=bool(geometry["gate_pass"]),
        solver_gate=bool(solver["gate_pass"]),
        repeat_gate=bool(repeat["gate_pass"]),
        certificate_gate=certificate,
        geometry_metrics=geometry,
        solver_metrics=solver,
        repeat_metrics=repeat,
        chart_gate_by_id=chart_gate,
        chart_metrics_by_id=chart_metrics,
        schedules=schedule_frame,
        executions=execution_frame,
        failure_reasons=tuple(reasons),
    )


def repair_rooted_section_atlas(
    growth: SectionGrowthResult,
    continuation: ContinuationAdapter,
    *,
    patch_id: str,
    method: str,
    policy: AtlasRepairPolicy | None = None,
    retry_continuation: Callable[..., ContinuationOutcome] | None = None,
    schedule_executor: Callable[..., pd.DataFrame] | None = None,
    screening_first: bool = False,
    canonical_root_priority: Sequence[CandidateKey] = (),
    canonical_anchor_policy: CanonicalAnchorPolicy | None = None,
) -> RepairedSectionAtlas:
    """Freshly audit, qualify, stitch, and select one deployable primary atlas."""

    active = AtlasRepairPolicy() if policy is None else policy
    schedules = _build_schedules(growth, patch_id=patch_id, method=method)
    chart_audit_policy = active.audit
    if screening_first:
        schedules = schedules[
            schedules["audit_kind"].isin(("edge", "fundamental_cycle"))
        ].reset_index(drop=True)
        chart_audit_policy = AuditV2Policy(
            geometry_p95_max_deg=active.audit.geometry_p95_max_deg,
            geometry_max_deg=active.audit.geometry_max_deg,
            repeat_p95_max_deg=active.audit.repeat_p95_max_deg,
            continuation_residual_max_mm=active.audit.continuation_residual_max_mm,
            repeats_per_direction=1,
            repeat_perturbation_rad=active.audit.repeat_perturbation_rad,
            retry_tiers=active.audit.retry_tiers,
        )
    executions = _execute_phase(
        growth, schedules, continuation, chart_audit_policy,
        retry_continuation=retry_continuation,
        schedule_executor=schedule_executor,
        phase_id="chart_screening" if screening_first else "chart_initial",
    )
    initial = diagnose_rooted_section_artifacts(
        growth,
        schedules=schedules,
        executions=executions,
        policy=chart_audit_policy,
    )
    working_growth = _split_failed_chart_fragments(
        growth, schedules, executions, initial, active
    )
    if working_growth is not growth:
        schedules = _build_schedules(working_growth, patch_id=patch_id, method=f"{method}_fragment")
        if screening_first:
            schedules = schedules[
                schedules["audit_kind"].isin(("edge", "fundamental_cycle"))
            ].reset_index(drop=True)
        executions = _execute_phase(
            working_growth, schedules, continuation, chart_audit_policy,
            retry_continuation=retry_continuation,
            schedule_executor=schedule_executor,
            phase_id="fragment_screening" if screening_first else "fragment_reaudit",
        )
        initial = diagnose_rooted_section_artifacts(
            working_growth,
            schedules=schedules,
            executions=executions,
            policy=chart_audit_policy,
        )
    chart_by_id = {chart.chart_id: chart for chart in working_growth.charts}
    node_by_id = {node.node_id: node for node in working_growth.task_nodes}
    minimum_chart_support = max(
        active.minimum_chart_cells,
        int(math.ceil(active.minimum_chart_fraction * len(node_by_id))),
    )
    qualified = tuple(
        chart_id
        for chart_id in sorted(chart_by_id)
        if len(chart_by_id[chart_id].selected_by_node) >= minimum_chart_support
        and _node_spread_mm(chart_by_id[chart_id].selected_by_node, node_by_id)
        >= active.minimum_chart_spread_mm
        and initial.chart_gate_by_id.get(chart_id, False)
    )
    root_priority = tuple(
        dict.fromkeys((int(key[0]), str(key[1])) for key in canonical_root_priority)
    )
    qualified = _remove_dominated_charts(
        chart_by_id, qualified, active, canonical_root_priority=root_priority
    )
    stitch_rows, stitch_edges = _qualify_stitches(
        chart_by_id, qualified, active, working_growth.task_nodes, continuation
    )
    components = _chart_components(qualified, stitch_edges)
    if canonical_anchor_policy is not None:
        anchor_selection = select_anchor_locked_component(
            components,
            chart_by_id,
            working_growth.task_nodes,
            canonical_anchor_policy,
        )
        selected_component = anchor_selection.component
    else:
        selected_component = _select_component(
            components, chart_by_id, canonical_root_priority=root_priority
        )
        selected_roots = {
            chart_by_id[chart_id].root_key for chart_id in selected_component
        }
        selected_anchor = next(
            (key for key in root_priority if key in selected_roots), None
        )
        anchor_selection = AnchorComponentSelection(
            component=selected_component,
            anchor_root_key=selected_anchor,
            anchor_rank=(
                root_priority.index(selected_anchor)
                if selected_anchor is not None
                else None
            ),
            anchor_qualified=selected_anchor is not None,
            anchor_component_selected=selected_anchor is not None,
            component_coverage=(
                len(
                    set().union(
                        *(
                            set(chart_by_id[chart_id].selected_by_node)
                            for chart_id in selected_component
                        )
                    )
                )
                / max(1, len(working_growth.task_nodes))
                if selected_component
                else 0.0
            ),
            coherent_measure=0.0,
            fallback_reason=None,
        )
    primary_chart, primary_beta, primary_optimization = _materialize_primary(
        selected_component,
        chart_by_id,
        working_growth.task_nodes,
        stitch_edges,
        canonical_root_priority=root_priority,
    )
    pre_abstention_nodes = set(primary_chart)
    all_nodes = {node.node_id for node in working_growth.task_nodes}
    endpoint_abstention = set(all_nodes) - set(primary_chart)
    one_hop_abstention = _expand_abstention(
        endpoint_abstention, working_growth.task_nodes, 1
    ) if endpoint_abstention else set()
    radius_abstention = _radius_abstention(
        endpoint_abstention,
        working_growth.task_nodes,
        active.abstention_radius_mm,
    ) if endpoint_abstention else set()
    abstained = set(endpoint_abstention)
    if abstained and active.abstention_hops:
        abstained = _expand_abstention(abstained, working_growth.task_nodes, active.abstention_hops)
        for node_id in abstained:
            primary_chart.pop(node_id, None)
            primary_beta.pop(node_id, None)

    primary_growth = _primary_growth(
        working_growth, primary_chart, chart_by_id
    )
    primary_schedules = _build_schedules(
        primary_growth, patch_id=patch_id, method=f"{method}_primary_certificate"
    )
    primary_executions = _execute_phase(
        primary_growth, primary_schedules, continuation, active.audit,
        retry_continuation=retry_continuation,
        schedule_executor=schedule_executor,
        phase_id="primary_certificate",
    )
    final_diagnostic = diagnose_rooted_section_artifacts(
        primary_growth,
        schedules=primary_schedules,
        executions=primary_executions,
        policy=active.audit,
        primary_chart_ids=("primary_atlas",),
    )
    covered = set(primary_chart)
    largest_ratio = _largest_task_component_ratio(covered, working_growth.task_nodes)
    coverage = len(covered) / max(1, len(all_nodes))
    induced_edges = _induced_task_edges(covered, node_by_id)
    audited_edge_entities = set(
        primary_schedules.loc[
            primary_schedules["audit_kind"].eq("edge"), "unique_entity_id"
        ].astype(str)
    )
    edge_completeness = len(audited_edge_entities) / max(1, len(induced_edges))
    pre_edges = _induced_task_edges(pre_abstention_nodes, node_by_id)
    pre_cycle_rank = _cycle_rank(pre_abstention_nodes, pre_edges)
    retained_cycle_rank = _cycle_rank(covered, induced_edges)
    audited_cycles = int(primary_schedules["audit_kind"].eq("fundamental_cycle").sum())
    cycle_coverage = (
        audited_cycles / retained_cycle_rank if retained_cycle_rank else 1.0
    )
    certificate = bool(
        selected_component
        and final_diagnostic.certificate_gate
        and all(chart_id in qualified for chart_id in selected_component)
        and abs(edge_completeness - 1.0) <= 1e-12
        and abs(cycle_coverage - 1.0) <= 1e-12
        and (
            canonical_anchor_policy is None
            or not canonical_anchor_policy.require_anchor_selected
            or anchor_selection.anchor_component_selected
        )
    )
    assignment_rows = [
        {
            "task_node_id": node.node_id,
            "primary_chart_id": primary_chart.get(node.node_id),
            "abstained": node.node_id in abstained,
            **{
                f"beta{index + 1}_rad": (
                    float(primary_beta[node.node_id][index]) if node.node_id in primary_beta else math.nan
                )
                for index in range(6)
            },
        }
        for node in working_growth.task_nodes
    ]
    chart_rows = [
        {
            "chart_id": chart_id,
            "cell_count": len(chart_by_id[chart_id].selected_by_node),
            "support_node_count": len(chart_by_id[chart_id].selected_by_node),
            "support_fraction": len(chart_by_id[chart_id].selected_by_node)
            / max(1, len(node_by_id)),
            "support_spread_mm": _node_spread_mm(
                chart_by_id[chart_id].selected_by_node, node_by_id
            ),
            "minimum_registered_support_node_count": minimum_chart_support,
            "minimum_registered_support_fraction": active.minimum_chart_fraction,
            "minimum_registered_support_spread_mm": active.minimum_chart_spread_mm,
            "qualified": chart_id in qualified,
            "selected_component": chart_id in selected_component,
            **dict(initial.chart_metrics_by_id.get(chart_id, {})),
        }
        for chart_id in sorted(chart_by_id)
    ]
    abstention_rows = [
        {
            "task_node_id": node.node_id,
            "endpoint_only": node.node_id in endpoint_abstention,
            "endpoint_plus_one_hop": node.node_id in one_hop_abstention,
            "physical_radius_mm": active.abstention_radius_mm,
            "within_physical_radius": node.node_id in radius_abstention,
            "selected_by_registered_policy": node.node_id in abstained,
        }
        for node in working_growth.task_nodes
    ]
    frames = MappingProxyType(
        {
            "chart_audit_v2_schedules": schedules,
            "chart_audit_v2_executions": executions,
            "audit_v2_schedules": primary_schedules,
            "audit_v2_executions": primary_executions,
            "chart_qualification": pd.DataFrame.from_records(chart_rows),
            "chart_stitchability": stitch_rows,
            "primary_atlas": pd.DataFrame.from_records(assignment_rows),
            "abstention_sensitivity": pd.DataFrame.from_records(abstention_rows),
            "primary_optimization": primary_optimization,
        }
    )
    return RepairedSectionAtlas(
        growth=working_growth,
        diagnostic=final_diagnostic,
        qualified_chart_ids=qualified,
        stitch_edges=stitch_edges,
        selected_stitch_component=selected_component,
        primary_chart_by_node=primary_chart,
        primary_beta_by_node=primary_beta,
        abstained_node_ids=frozenset(abstained),
        coverage_ratio=float(coverage),
        largest_coherent_region_ratio=float(largest_ratio),
        geometry_gate=final_diagnostic.geometry_gate,
        solver_gate=final_diagnostic.solver_gate,
        repeat_gate=final_diagnostic.repeat_gate,
        certificate_gate=certificate,
        induced_edge_count=len(induced_edges),
        audited_edge_count=len(audited_edge_entities),
        edge_completeness_ratio=float(edge_completeness),
        pre_abstention_cycle_rank=int(pre_cycle_rank),
        retained_cycle_rank=int(retained_cycle_rank),
        audited_fundamental_cycle_count=audited_cycles,
        cycle_coverage_ratio=float(cycle_coverage),
        canonical_anchor_root_key=anchor_selection.anchor_root_key,
        canonical_anchor_qualified=anchor_selection.anchor_qualified,
        canonical_anchor_component_selected=anchor_selection.anchor_component_selected,
        canonical_anchor_component_coverage=anchor_selection.component_coverage,
        canonical_anchor_fallback_reason=anchor_selection.fallback_reason,
        selected_anchor_rank=anchor_selection.anchor_rank,
        frames=frames,
    )


def compare_stitched_primary_atlases(
    left: RepairedSectionAtlas,
    right: RepairedSectionAtlas,
    *,
    coverage_jaccard_min: float = 0.95,
    beta_p95_max_deg: float = 1.0,
    beta_max_deg: float = 2.0,
    assignment_change_max: float = 0.05,
) -> PrimaryAtlasStability:
    """Compare physical primary labels; chart identifier renames are irrelevant."""

    left_nodes = set(left.primary_beta_by_node)
    right_nodes = set(right.primary_beta_by_node)
    union = left_nodes | right_nodes
    common = sorted(left_nodes & right_nodes)
    jaccard = len(common) / max(1, len(union))
    gaps = np.asarray(
        [beta_rms_deg(left.primary_beta_by_node[node], right.primary_beta_by_node[node]) for node in common],
        dtype=float,
    )
    p95 = float(np.percentile(gaps, 95)) if len(gaps) else math.inf
    maximum = float(np.max(gaps)) if len(gaps) else math.inf
    changed = float(np.mean(gaps > 1.0 + 1e-12)) if len(gaps) else 1.0
    gate = bool(
        jaccard >= coverage_jaccard_min
        and p95 <= beta_p95_max_deg
        and maximum <= beta_max_deg
        and changed <= assignment_change_max
    )
    return PrimaryAtlasStability(gate, jaccard, p95, maximum, changed, len(common))


def _build_schedules(growth: SectionGrowthResult, *, patch_id: str, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    node_by_id = {node.node_id: node for node in growth.task_nodes}
    for chart in growth.charts:
        selected = set(chart.selected_by_node)
        edges = _induced_task_edges(selected, node_by_id)
        adjacency = _adjacency(edges)
        entities: list[tuple[str, tuple[int, ...]]] = []
        entities.extend(("edge", edge) for edge in edges)
        for component in _node_components(selected, edges):
            root = chart.root_key[0] if chart.root_key[0] in component else min(component)
            parent, paths, tree_edges = _spanning_tree(adjacency, root, set(component))
            entities.extend(("root_path", path) for node, path in sorted(paths.items()) if node != root)
            component_edges = {edge for edge in edges if edge[0] in component and edge[1] in component}
            for left, right in sorted(component_edges - tree_edges):
                tree_path = _tree_path(left, right, parent)
                entities.append(("fundamental_cycle", tree_path + (left,)))
                entities.append(("multipath_tree", tree_path))
                entities.append(("multipath_chord", (left, right)))
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for kind, path in entities:
            signature_path = _canonical_path_signature(path, cycle="cycle" in kind)
            entity = (kind, signature_path)
            if entity in seen:
                continue
            seen.add(entity)
            physical_entity = physical_audit_entity_id(patch_id, kind, path)
            schedule_identity = (
                f"{method}:{chart.chart_id}:{kind}:"
                f"{','.join(map(str, signature_path))}"
            )
            digest = hashlib.sha256(schedule_identity.encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "patch_id": str(patch_id),
                    "method": str(method),
                    "schedule_id": f"schedule_{digest}",
                    "unique_entity_id": physical_entity,
                    "physical_entity_id": physical_entity,
                    "chart_id": chart.chart_id,
                    "audit_kind": kind,
                    "path_node_ids": list(path),
                    "unordered_signature": list(signature_path),
                    "path_task_node_ids": list(dict.fromkeys(path)),
                    "measure_weight": float(len(set(path))),
                    "primary_usage": True,
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "patch_id",
            "method",
            "schedule_id",
            "unique_entity_id",
            "physical_entity_id",
            "chart_id",
            "audit_kind",
            "path_node_ids",
            "unordered_signature",
            "path_task_node_ids",
            "measure_weight",
            "primary_usage",
        ),
    )


def execute_audit_schedules(
    growth: SectionGrowthResult,
    schedules: pd.DataFrame,
    continuation: ContinuationAdapter,
    policy: AuditV2Policy,
    *,
    retry_continuation: Callable[..., ContinuationOutcome] | None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    chart_by_id = {chart.chart_id: chart for chart in growth.charts}
    node_by_id = {node.node_id: node for node in growth.task_nodes}
    raw_adapter = continuation if retry_continuation is None else retry_continuation
    adapter = _bind_retry_adapter(raw_adapter)
    rows: list[dict[str, Any]] = []
    schedule_records = schedules.to_dict(orient="records")
    total_schedule_count = len(schedule_records)
    for completed_schedule_count, schedule in enumerate(schedule_records, start=1):
        chart = chart_by_id[str(schedule["chart_id"])]
        selected = {node: item.candidate for node, item in chart.selected_by_node.items()}
        base_path = tuple(int(value) for value in schedule["path_node_ids"])
        for direction, path in (("forward", base_path), ("reverse", tuple(reversed(base_path)))):
            physical_entity = str(
                schedule.get(
                    "physical_entity_id",
                    schedule.get("unique_entity_id", schedule["schedule_id"]),
                )
            )
            physical_direction = physical_audit_direction(
                base_path,
                direction,
                cycle="cycle" in str(schedule["audit_kind"]),
            )
            endpoints: list[np.ndarray] = []
            for repeat_index in range(policy.repeats_per_direction):
                perturbation = physical_repeat_perturbation(
                    physical_entity, physical_direction, repeat_index,
                    policy.repeat_perturbation_rad,
                )
                trace = _retry_trace(
                    path, selected, node_by_id, adapter, policy,
                    source_perturbation_rad=perturbation,
                )
                endpoint = trace["endpoint_beta"]
                if endpoint is not None:
                    endpoints.append(endpoint)
                rows.append(
                    {
                        "schedule_id": schedule["schedule_id"],
                        "chart_id": schedule["chart_id"],
                        "direction": direction,
                        "physical_entity_id": physical_entity,
                        "physical_direction": physical_direction,
                        "repeat_index": repeat_index,
                        "repeat_perturbation_l2_rad": float(np.linalg.norm(perturbation)),
                        "repeat_perturbation_sha256": hashlib.sha256(
                            np.asarray(perturbation, dtype=np.float64).tobytes()
                        ).hexdigest(),
                        "solver_success": trace["solver_success"],
                        "geometry_gap_deg": trace["geometry_gap_deg"],
                        "repeat_gap_deg": math.nan,
                        "residual_mm": trace["residual_mm"],
                        "retry_tier": trace["retry_tier"],
                        "registered_solver_chain": trace[
                            "registered_solver_chain"
                        ],
                        "executed_solver_chain": trace["executed_solver_chain"],
                        "solver_chain_sha256": trace["solver_chain_sha256"],
                        "classification": trace["classification"],
                        "failure_source_node": trace["failure_source_node"],
                        "failure_target_node": trace["failure_target_node"],
                        "oracle_used_for_pass": False,
                    }
                )
            if len(endpoints) >= 1:
                reference = endpoints[0]
                repeat_gaps = [beta_rms_deg(reference, endpoint) for endpoint in endpoints[1:]]
                for row in rows[-policy.repeats_per_direction:]:
                    row["repeat_gap_deg"] = max(repeat_gaps, default=0.0)
        if progress_callback is not None:
            progress_callback(completed_schedule_count, total_schedule_count)
    return pd.DataFrame.from_records(rows)


def _execute_phase(
    growth: SectionGrowthResult,
    schedules: pd.DataFrame,
    continuation: ContinuationAdapter,
    policy: AuditV2Policy,
    *,
    retry_continuation: Callable[..., ContinuationOutcome] | None,
    schedule_executor: Callable[..., pd.DataFrame] | None,
    phase_id: str,
) -> pd.DataFrame:
    if schedule_executor is None:
        return execute_audit_schedules(
            growth,
            schedules,
            continuation,
            policy,
            retry_continuation=retry_continuation,
        )
    executions = schedule_executor(
        growth,
        schedules,
        continuation,
        policy,
        retry_continuation,
        str(phase_id),
    )
    if not isinstance(executions, pd.DataFrame):
        raise TypeError("schedule executor must return a pandas DataFrame")
    return executions


def _retry_trace(
    path: tuple[int, ...],
    selected: Mapping[int, AtlasCandidate],
    nodes: Mapping[int, AtlasTaskNode],
    adapter: Callable[..., ContinuationOutcome],
    policy: AuditV2Policy,
    *,
    source_perturbation_rad: np.ndarray,
) -> dict[str, Any]:
    if not path or path[0] not in selected or path[-1] not in selected:
        return _failed_trace("local_feasibility_failure")
    last_failed_edge: tuple[int, int] | None = None
    last_executed_chain: tuple[str, ...] = ()
    best_geometric_trace: dict[str, Any] | None = None
    saw_geometric_disagreement = False
    for tier in policy.retry_tiers:
        source = selected[path[0]]
        current = AtlasCandidate(
            node_id=source.node_id,
            candidate_id=f"{source.candidate_id}_repeat_perturbation",
            beta_rad=source.beta_rad + np.asarray(source_perturbation_rad, dtype=float),
            residual_mm=source.residual_mm,
            min_margin_deg=source.min_margin_deg,
            normalized_min_margin=source.normalized_min_margin,
            posture_cost=source.posture_cost,
            condition_number=source.condition_number,
            quality=source.quality,
            solver_success=source.solver_success,
            actual_bounds=source.actual_bounds,
            cluster_id=source.cluster_id,
            diagnostics=source.diagnostics,
        )
        last_residual = 0.0
        failed = False
        executed_chain: list[str] = []
        for target_id in path[1:]:
            outcome = _invoke_adapter(adapter, current, nodes[target_id], tier)
            outcome_chain = _outcome_solver_chain(outcome) or tier.solver_chain
            for solver in outcome_chain:
                if solver not in executed_chain:
                    executed_chain.append(solver)
            last_executed_chain = tuple(executed_chain)
            last_residual = float(outcome.residual_mm)
            if not (
                outcome.success
                and outcome.actual_bounds
                and last_residual <= policy.continuation_residual_max_mm + 1e-12
            ):
                last_failed_edge = (current.node_id, target_id)
                failed = True
                break
            current = _endpoint_candidate(current, target_id, outcome, tier.tier_id)
        if failed:
            continue
        gap = beta_rms_deg(current.beta_rad, selected[path[-1]].beta_rad)
        trace = {
            "solver_success": True,
            "geometry_gap_deg": gap,
            "residual_mm": last_residual,
            "retry_tier": tier.tier_id,
            "registered_solver_chain": ">".join(tier.solver_chain),
            "executed_solver_chain": ">".join(executed_chain),
            "solver_chain_sha256": hashlib.sha256(
                json.dumps(list(tier.solver_chain), separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "classification": "verified",
            "endpoint_beta": current.beta_rad,
            "failure_source_node": None,
            "failure_target_node": None,
        }
        if gap > policy.geometry_max_deg + 1e-12:
            saw_geometric_disagreement = True
            trace["classification"] = "geometric_branch_disagreement"
            if (
                best_geometric_trace is None
                or float(trace["geometry_gap_deg"])
                < float(best_geometric_trace["geometry_gap_deg"])
            ):
                best_geometric_trace = trace
            # A numerically successful solve on the wrong redundant fibre is
            # not a terminal success.  Continue through the registered kernels
            # without ever seeding from the stored target beta.
            continue
        if tier.tier_id != policy.retry_tiers[0].tier_id:
            trace["classification"] = (
                "recoverable_geometric"
                if saw_geometric_disagreement
                else "recoverable_numerical"
            )
        return trace
    if best_geometric_trace is not None:
        return best_geometric_trace
    last_tier = policy.retry_tiers[-1] if policy.retry_tiers else None
    return _failed_trace(
        "persistent_numerical",
        failure_edge=last_failed_edge,
        tier=last_tier,
        executed_solver_chain=last_executed_chain,
    )


def _repeat_perturbation(
    schedule_id: str, direction: str, repeat_index: int, magnitude_rad: float
) -> np.ndarray:
    if repeat_index == 0:
        return np.zeros(6, dtype=float)
    digest = hashlib.sha256(
        f"{schedule_id}:{direction}:{repeat_index}".encode("utf-8")
    ).digest()
    signs = np.asarray([1.0 if digest[index] & 1 else -1.0 for index in range(6)])
    return signs * float(magnitude_rad) * repeat_index


def _outcome_solver_chain(outcome: ContinuationOutcome) -> tuple[str, ...]:
    """Read V14.2R provenance without changing the sealed outcome schema."""

    marker = "audit_solver_chain["
    chains: list[str] = []
    remaining = str(outcome.status)
    while marker in remaining:
        remaining = remaining.split(marker, 1)[1]
        if "]" not in remaining:
            break
        encoded, remaining = remaining.split("]", 1)
        for solver in encoded.split(">"):
            if solver and solver not in chains:
                chains.append(solver)
    return tuple(chains)


def _bind_retry_adapter(
    adapter: Callable[..., ContinuationOutcome],
) -> Callable[[AtlasCandidate, AtlasTaskNode, RetryTier], ContinuationOutcome]:
    """Resolve the legacy two/three-argument seam once, outside hot loops."""

    try:
        signature = inspect.signature(adapter)
        accepts_tier = len(signature.parameters) >= 3 or any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_tier = False

    if accepts_tier:
        def bound(source: AtlasCandidate, target: AtlasTaskNode, tier: RetryTier) -> ContinuationOutcome:
            return adapter(source, target, tier)
    else:
        def bound(source: AtlasCandidate, target: AtlasTaskNode, tier: RetryTier) -> ContinuationOutcome:
            del tier
            return adapter(source, target)
    return bound


def _invoke_adapter(
    adapter: Callable[[AtlasCandidate, AtlasTaskNode, RetryTier], ContinuationOutcome],
    source: AtlasCandidate,
    target: AtlasTaskNode,
    tier: RetryTier,
) -> ContinuationOutcome:
    outcome = adapter(source, target, tier)
    if not isinstance(outcome, ContinuationOutcome):
        raise TypeError("continuation adapter must return ContinuationOutcome")
    return outcome


def _endpoint_candidate(
    source: AtlasCandidate,
    node_id: int,
    outcome: ContinuationOutcome,
    tier_id: str,
) -> AtlasCandidate:
    return AtlasCandidate(
        node_id=node_id,
        candidate_id=f"audit_{source.candidate_id}_{node_id}_{tier_id}",
        beta_rad=outcome.beta_rad,
        residual_mm=max(0.0, outcome.residual_mm),
        min_margin_deg=(outcome.minimum_margin_deg if outcome.minimum_margin_deg is not None else source.min_margin_deg),
        normalized_min_margin=source.normalized_min_margin,
        posture_cost=source.posture_cost,
        condition_number=source.condition_number,
        quality="Gold",
        actual_bounds=outcome.actual_bounds,
        solver_success=outcome.success,
    )


def _failed_trace(
    classification: str,
    *,
    failure_edge: tuple[int, int] | None = None,
    tier: RetryTier | None = None,
    executed_solver_chain: tuple[str, ...] = (),
) -> dict[str, Any]:
    chain = () if tier is None else tier.solver_chain
    return {
        "solver_success": False,
        "geometry_gap_deg": math.nan,
        "residual_mm": math.nan,
        "retry_tier": None if tier is None else tier.tier_id,
        "registered_solver_chain": ">".join(chain),
        "executed_solver_chain": ">".join(executed_solver_chain),
        "solver_chain_sha256": (
            ""
            if tier is None
            else hashlib.sha256(
                json.dumps(list(chain), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        ),
        "classification": classification,
        "endpoint_beta": None,
        "failure_source_node": None if failure_edge is None else failure_edge[0],
        "failure_target_node": None if failure_edge is None else failure_edge[1],
    }


def _geometry_metrics(rows: pd.DataFrame, policy: AuditV2Policy) -> dict[str, Any]:
    success = rows["solver_success"].fillna(False).astype(bool) if len(rows) else pd.Series(dtype=bool)
    gaps = pd.to_numeric(rows.loc[success, "geometry_gap_deg"], errors="coerce") if len(rows) else pd.Series(dtype=float)
    finite = gaps[np.isfinite(gaps.to_numpy(dtype=float))].to_numpy(dtype=float)
    p95 = float(np.percentile(finite, 95)) if len(finite) else math.nan
    maximum = float(np.max(finite)) if len(finite) else math.nan
    missing = int((~success).sum() + (success.sum() - len(finite)))
    gate = bool(
        not len(finite)
        or (p95 <= policy.geometry_p95_max_deg + 1e-12 and maximum <= policy.geometry_max_deg + 1e-12)
    )
    return {
        "sample_count": int(len(finite)),
        "missing_count": missing,
        "p95_deg": p95,
        "max_deg": maximum,
        "gate_pass": gate,
    }


def _solver_metrics(rows: pd.DataFrame, policy: AuditV2Policy) -> dict[str, Any]:
    success = rows["solver_success"].fillna(False).astype(bool) if len(rows) else pd.Series(dtype=bool)
    residual = pd.to_numeric(rows.get("residual_mm", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    complete = success & np.isfinite(residual.to_numpy(dtype=float)) & (residual <= policy.continuation_residual_max_mm + 1e-12)
    classifications = rows.get("classification", pd.Series(index=rows.index, dtype=str)).fillna("").astype(str)
    missing = int((~complete).sum())
    geometric = classifications == "geometric_branch_disagreement"
    entity_column = (
        "physical_entity_id" if "physical_entity_id" in rows else "schedule_id"
    )
    geometric_entities = (
        rows.loc[geometric, entity_column].astype(str).nunique()
        if entity_column in rows
        else int(geometric.sum())
    )
    return {
        "execution_count": int(len(rows)),
        "success_count": int(complete.sum()),
        "missing_count": missing,
        "recoverable_numerical_count": int((classifications == "recoverable_numerical").sum()),
        "persistent_numerical_count": int((classifications == "persistent_numerical").sum()),
        "geometric_branch_disagreement_count": int((classifications == "geometric_branch_disagreement").sum()),
        "geometric_branch_disagreement_execution_count": int(geometric.sum()),
        "geometric_branch_disagreement_unique_entity_count": int(geometric_entities),
        "local_feasibility_failure_count": int((classifications == "local_feasibility_failure").sum()),
        "gate_pass": bool(len(rows) > 0 and missing == 0),
    }


def _repeat_metrics(rows: pd.DataFrame, policy: AuditV2Policy) -> dict[str, Any]:
    """Evaluate repeat stability without reclassifying solver failures."""

    if not len(rows):
        return {
            "sample_count": 0,
            "missing_count": 0,
            "p95_deg": math.nan,
            "gate_pass": True,
        }
    solver_success = rows["solver_success"].fillna(False).astype(bool)
    values = pd.to_numeric(
        rows.get("repeat_gap_deg", pd.Series(index=rows.index, dtype=float)),
        errors="coerce",
    )
    eligible = values.loc[solver_success]
    finite = eligible[np.isfinite(eligible.to_numpy(dtype=float))].to_numpy(dtype=float)
    missing = int(len(eligible) - len(finite))
    p95 = float(np.percentile(finite, 95)) if len(finite) else math.nan
    return {
        "sample_count": int(len(finite)),
        "missing_count": missing,
        "p95_deg": p95,
        "gate_pass": bool(
            missing == 0
            and (
                not len(finite)
                or p95 <= policy.repeat_p95_max_deg + 1e-12
            )
        ),
    }


def _split_failed_chart_fragments(
    growth: SectionGrowthResult,
    schedules: pd.DataFrame,
    executions: pd.DataFrame,
    diagnostic: RootedArtifactDiagnostic,
    policy: AtlasRepairPolicy,
) -> SectionGrowthResult:
    """Cut persisted failed paths, retain nontrivial components, then re-audit."""

    if all(diagnostic.chart_gate_by_id.get(chart.chart_id, False) for chart in growth.charts):
        return growth
    payload = executions.drop(columns=["chart_id"], errors="ignore")
    merged = schedules[
        ["schedule_id", "chart_id", "audit_kind", "path_node_ids"]
    ].merge(
        payload, on="schedule_id", how="left"
    )
    success = merged.get("solver_success", pd.Series(False, index=merged.index)).fillna(False).astype(bool)
    gap = pd.to_numeric(merged.get("geometry_gap_deg"), errors="coerce")
    repeat = pd.to_numeric(merged.get("repeat_gap_deg"), errors="coerce")
    failed = merged[
        (~success)
        | gap.gt(policy.audit.geometry_max_deg)
        | repeat.gt(policy.audit.repeat_p95_max_deg)
    ]
    bad_edges_by_chart: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in failed.itertuples(index=False):
        path = tuple(map(int, row.path_node_ids))
        failure_source = getattr(row, "failure_source_node", None)
        failure_target = getattr(row, "failure_target_node", None)
        if (
            failure_source is not None
            and failure_target is not None
            and not pd.isna(failure_source)
            and not pd.isna(failure_target)
        ):
            bad_edges_by_chart[str(row.chart_id)].add(
                tuple(sorted((int(failure_source), int(failure_target))))
            )
            continue
        if str(row.audit_kind) == "edge" and len(path) == 2 and path[0] != path[1]:
            bad_edges_by_chart[str(row.chart_id)].add(tuple(sorted(path)))
    node_by_id = {node.node_id: node for node in growth.task_nodes}
    output: list[RootedSectionChart] = []
    changed = False
    for chart in growth.charts:
        if diagnostic.chart_gate_by_id.get(chart.chart_id, False):
            output.append(chart)
            continue
        bad = bad_edges_by_chart.get(chart.chart_id, set())
        if not bad:
            # A chart with an unlocalized cycle/path failure is not silently
            # promoted.  Qualification will remove it.
            output.append(chart)
            continue
        removed_nodes = _deterministic_failure_vertex_cover(
            bad, set(chart.selected_by_node), node_by_id, protected_root=chart.root_key[0]
        )
        retained_nodes = set(chart.selected_by_node) - removed_nodes
        retained_edges = _induced_task_edges(retained_nodes, node_by_id)
        components = _node_components(retained_nodes, retained_edges)
        fragments = [component for component in components if len(component) >= policy.minimum_chart_cells]
        if not fragments:
            output.append(chart)
            continue
        changed = True
        for fragment_index, component in enumerate(fragments):
            chart_id = f"{chart.chart_id}__fragment_{fragment_index:02d}"
            hypotheses = {
                node: tuple(
                    SectionHypothesis(
                        chart_id=chart_id,
                        root_key=item.root_key,
                        node_id=item.node_id,
                        candidate=item.candidate,
                        parent_keys=tuple(key for key in item.parent_keys if key[0] in component),
                        score=item.score,
                        wave=item.wave,
                        boundary_risk=item.boundary_risk,
                        uses_propagated_candidate=item.uses_propagated_candidate,
                    )
                    for item in chart.hypotheses_by_node[node]
                )
                for node in component
            }
            selected = {
                node: next(
                    item
                    for item in hypotheses[node]
                    if item.candidate.key == chart.selected_by_node[node].candidate.key
                )
                for node in component
            }
            output.append(
                RootedSectionChart(
                    chart_id=chart_id,
                    root_key=chart.root_key,
                    hypotheses_by_node=hypotheses,
                    selected_by_node=selected,
                    selected_edges=tuple(
                        edge for edge in retained_edges if edge[0] in component and edge[1] in component
                    ),
                    boundary_risk_node_ids=chart.boundary_risk_node_ids & component,
                    frontier_node_ids=chart.frontier_node_ids & component,
                )
            )
    if not changed:
        return growth
    primary: dict[int, str | None] = {}
    for node in growth.task_nodes:
        covering = [chart.chart_id for chart in output if node.node_id in chart.selected_by_node]
        primary[node.node_id] = min(covering) if covering else None
    return SectionGrowthResult(
        task_nodes=growth.task_nodes,
        charts=tuple(output),
        primary_chart_by_node=primary,
        abstained_node_ids=frozenset(node for node, chart in primary.items() if chart is None),
        events=growth.events,
        cap_hit_events=growth.cap_hit_events,
        policy=growth.policy,
        continuation_attempt_count=growth.continuation_attempt_count,
        rejected_continuation_count=growth.rejected_continuation_count,
    )


def _node_components(
    nodes: set[int], edges: Sequence[tuple[int, int]]
) -> tuple[frozenset[int], ...]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        if left in nodes and right in nodes:
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(nodes)
    output = []
    while remaining:
        root = min(remaining)
        queue = [root]
        component: set[int] = set()
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            queue.extend(adjacency[current] - component)
        remaining -= component
        output.append(frozenset(component))
    return tuple(output)


def _induced_task_edges(
    nodes: set[int], node_by_id: Mapping[int, AtlasTaskNode]
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                tuple(sorted((node, neighbor)))
                for node in nodes
                for neighbor in node_by_id[node].neighbor_node_ids
                if neighbor in nodes and neighbor != node
            }
        )
    )


def _deterministic_failure_vertex_cover(
    bad_edges: set[tuple[int, int]],
    selected_nodes: set[int],
    node_by_id: Mapping[int, AtlasTaskNode],
    *,
    protected_root: int,
) -> frozenset[int]:
    """Choose explicit abstention nodes; never retain both endpoints and drop an edge."""

    remaining = set(bad_edges)
    removed: set[int] = set()
    while remaining:
        endpoints = set().union(*(set(edge) for edge in remaining))
        ranked = sorted(
            endpoints,
            key=lambda node: (
                node == protected_root,
                -sum(node in edge for edge in remaining) / max(1, len(set(node_by_id[node].neighbor_node_ids) & selected_nodes)),
                len(set(node_by_id[node].neighbor_node_ids) & selected_nodes),
                node,
            ),
        )
        chosen = ranked[0]
        removed.add(chosen)
        remaining = {edge for edge in remaining if chosen not in edge}
    return frozenset(removed)


def _remove_dominated_charts(
    charts: Mapping[str, RootedSectionChart],
    qualified: Sequence[str],
    policy: AtlasRepairPolicy,
    *,
    canonical_root_priority: Sequence[CandidateKey] = (),
) -> tuple[str, ...]:
    rank_by_root = {
        (int(key[0]), str(key[1])): index
        for index, key in enumerate(canonical_root_priority)
    }

    def stable_rank(chart_id: str) -> tuple[int, CandidateKey]:
        root_key = charts[chart_id].root_key
        return rank_by_root.get(root_key, len(rank_by_root)), root_key

    retained = set(qualified)
    protected_roots = set(rank_by_root)
    for left_id in sorted(qualified):
        if left_id not in retained:
            continue
        left_nodes = set(charts[left_id].selected_by_node)
        if charts[left_id].root_key in protected_roots:
            continue
        for right_id in sorted(qualified):
            if left_id == right_id or right_id not in retained:
                continue
            right_nodes = set(charts[right_id].selected_by_node)
            if not left_nodes <= right_nodes or len(left_nodes) > len(right_nodes):
                continue
            gaps = np.asarray([
                beta_rms_deg(
                    charts[left_id].selected_by_node[node].candidate.beta_rad,
                    charts[right_id].selected_by_node[node].candidate.beta_rad,
                )
                for node in sorted(left_nodes)
            ])
            if (
                len(gaps)
                and float(np.percentile(gaps, 95)) <= policy.stitch_p95_max_deg + 1e-12
                and float(np.max(gaps)) <= policy.stitch_max_deg + 1e-12
            ):
                if len(left_nodes) < len(right_nodes) or stable_rank(left_id) > stable_rank(right_id):
                    retained.remove(left_id)
                    break
    return tuple(sorted(retained))


def _node_spread_mm(
    node_ids: Mapping[int, Any] | Sequence[int] | set[int],
    node_by_id: Mapping[int, AtlasTaskNode],
) -> float:
    """Return bounding-box diagonal of a chart support in physical task space."""

    identifiers = tuple(int(node) for node in node_ids)
    if len(identifiers) < 2:
        return 0.0
    xyz = np.vstack([node_by_id[node].xyz_m for node in identifiers])
    return float(np.linalg.norm(np.ptp(xyz, axis=0)) * 1000.0)


def _shortest_chart_path(
    source: int,
    target: int,
    selected_nodes: set[int],
    node_by_id: Mapping[int, AtlasTaskNode],
) -> tuple[int, ...] | None:
    if source == target:
        return (source,)
    queue = deque([source])
    parent: dict[int, int | None] = {source: None}
    while queue:
        current = queue.popleft()
        for neighbour in node_by_id[current].neighbor_node_ids:
            if neighbour not in selected_nodes or neighbour in parent:
                continue
            parent[neighbour] = current
            if neighbour == target:
                path = [target]
                while parent[path[-1]] is not None:
                    path.append(int(parent[path[-1]]))
                return tuple(reversed(path))
            queue.append(neighbour)
    return None


def _audit_cross_chart_cycle(
    path: Sequence[tuple[int, RootedSectionChart]],
    node_by_id: Mapping[int, AtlasTaskNode],
    continuation: ContinuationAdapter,
    policy: AtlasRepairPolicy,
    *,
    source_perturbation_rad: np.ndarray,
) -> tuple[bool, float, np.ndarray | None]:
    if len(path) < 2:
        return False, math.inf, None
    first_node, first_chart = path[0]
    first = first_chart.selected_by_node[first_node].candidate
    current = AtlasCandidate(
        node_id=first.node_id,
        candidate_id=f"{first.candidate_id}_stitch_cycle_perturbation",
        beta_rad=first.beta_rad + np.asarray(source_perturbation_rad, dtype=float),
        residual_mm=first.residual_mm,
        min_margin_deg=first.min_margin_deg,
        normalized_min_margin=first.normalized_min_margin,
        posture_cost=first.posture_cost,
        condition_number=first.condition_number,
        quality=first.quality,
        solver_success=first.solver_success,
        actual_bounds=first.actual_bounds,
        cluster_id=first.cluster_id,
        diagnostics=first.diagnostics,
    )
    for target_node, target_chart in path[1:]:
        outcome = continuation(current, node_by_id[target_node])
        if not (
            outcome.success
            and outcome.actual_bounds
            and outcome.residual_mm <= policy.audit.continuation_residual_max_mm + 1e-12
        ):
            return False, math.inf, None
        target_beta = target_chart.selected_by_node[target_node].candidate.beta_rad
        if beta_rms_deg(outcome.beta_rad, target_beta) > policy.stitch_max_deg + 1e-12:
            return False, math.inf, None
        current = _endpoint_candidate(current, target_node, outcome, "stitch_cycle")
    return (
        True,
        beta_rms_deg(current.beta_rad, first.beta_rad),
        current.beta_rad.copy(),
    )


def _qualify_stitches(
    charts: Mapping[str, RootedSectionChart],
    qualified: Sequence[str],
    policy: AtlasRepairPolicy,
    task_nodes: Sequence[AtlasTaskNode],
    continuation: ContinuationAdapter,
) -> tuple[pd.DataFrame, tuple[tuple[str, str], ...]]:
    node_by_id = {node.node_id: node for node in task_nodes}
    rows: list[dict[str, Any]] = []
    edges: list[tuple[str, str]] = []
    for index, left_id in enumerate(qualified):
        for right_id in qualified[index + 1:]:
            left = charts[left_id]
            right = charts[right_id]
            left_nodes = set(left.selected_by_node)
            right_nodes = set(right.selected_by_node)
            overlap = sorted(left_nodes & right_nodes)
            overlap_fraction = len(overlap) / max(1, min(len(left_nodes), len(right_nodes)))
            overlap_xyz = np.vstack([node_by_id[node].xyz_m for node in overlap]) if overlap else np.empty((0, 3))
            overlap_spread_mm = (
                float(np.linalg.norm(np.ptp(overlap_xyz, axis=0)) * 1000.0)
                if len(overlap_xyz) else 0.0
            )
            gaps = np.asarray(
                [
                    beta_rms_deg(
                        left.selected_by_node[node].candidate.beta_rad,
                        right.selected_by_node[node].candidate.beta_rad,
                    )
                    for node in overlap
                ],
                dtype=float,
            )
            p95 = float(np.percentile(gaps, 95)) if len(gaps) else math.inf
            maximum = float(np.max(gaps)) if len(gaps) else math.inf
            overlap_set = set(overlap)
            overlap_edges = sorted(
                {
                    tuple(sorted((node, neighbor)))
                    for node in overlap
                    for neighbor in node_by_id[node].neighbor_node_ids
                    if neighbor in overlap_set
                }
            )
            boundary_edges = sorted(
                {
                    (node, neighbor)
                    for node in left_nodes
                    for neighbor in node_by_id[node].neighbor_node_ids
                    if neighbor in right_nodes and (node not in overlap_set or neighbor not in overlap_set)
                }
            )
            overlap_evidence_sufficient = bool(
                len(overlap) >= policy.minimum_stitch_overlap_cells
                and overlap_fraction >= policy.minimum_overlap_fraction
                and overlap_spread_mm >= policy.minimum_overlap_spread_mm
            )
            transition_attempt_count = 0
            transition_failure_count = 0
            transition_gap_values: list[float] = []
            certification_mode = "none"
            candidate_edges: list[tuple[int, int]] = []
            if overlap_evidence_sufficient:
                certification_mode = "overlap"
                candidate_edges = overlap_edges
            elif len(boundary_edges) >= policy.minimum_boundary_transition_edges:
                certification_mode = "boundary"
                candidate_edges = boundary_edges
            elif boundary_edges:
                # Registered local extension is still executed to turn an
                # evidence shortage into explicit evidence.  Too few
                # independent boundary edges remains indeterminate and can
                # never pass merely because these probes succeeded.
                candidate_edges = boundary_edges
            overlap_gap_ok = bool(
                certification_mode != "overlap"
                or (
                    p95 <= policy.stitch_p95_max_deg + 1e-12
                    and maximum <= policy.stitch_max_deg + 1e-12
                )
            )
            if candidate_edges and overlap_gap_ok:
                for left_node, right_node in candidate_edges:
                    for source_chart, target_chart, source_node, target_node in (
                        (left, right, left_node, right_node),
                        (right, left, right_node, left_node),
                    ):
                        if source_node not in source_chart.selected_by_node or target_node not in target_chart.selected_by_node:
                            transition_failure_count += 1
                            continue
                        transition_attempt_count += 1
                        outcome = continuation(
                            source_chart.selected_by_node[source_node].candidate,
                            node_by_id[target_node],
                        )
                        target_beta = target_chart.selected_by_node[target_node].candidate.beta_rad
                        transition_gap = beta_rms_deg(outcome.beta_rad, target_beta)
                        transition_gap_values.append(transition_gap)
                        if not (
                            outcome.success
                            and outcome.actual_bounds
                            and outcome.residual_mm <= policy.audit.continuation_residual_max_mm + 1e-12
                            and transition_gap <= policy.stitch_max_deg + 1e-12
                        ):
                            transition_failure_count += 1
            cycle_candidates: list[list[tuple[int, RootedSectionChart]]] = []
            for edge_index, (left_one, right_one) in enumerate(candidate_edges[:16]):
                for left_two, right_two in candidate_edges[edge_index + 1 : 16]:
                    left_path = _shortest_chart_path(
                        left_two, left_one, left_nodes, node_by_id
                    )
                    right_path = _shortest_chart_path(
                        right_one, right_two, right_nodes, node_by_id
                    )
                    if left_path is None or right_path is None:
                        continue
                    cycle_candidates.append(
                        [
                            (left_one, left),
                            *[(node, right) for node in right_path],
                            *[(node, left) for node in left_path],
                        ]
                    )
                    if len(cycle_candidates) >= 8:
                        break
                if len(cycle_candidates) >= 8:
                    break
            cycle_results: list[tuple[bool, float, np.ndarray | None]] = []
            cycle_repeat_gaps: list[float] = []
            cycle_candidate_pass: list[bool] = []
            for cycle_index, path in enumerate(cycle_candidates):
                repeated = [
                    _audit_cross_chart_cycle(
                        path,
                        node_by_id,
                        continuation,
                        policy,
                        source_perturbation_rad=_repeat_perturbation(
                            f"stitch_cycle_{left_id}_{right_id}_{cycle_index}",
                            "forward",
                            repeat_index,
                            policy.audit.repeat_perturbation_rad,
                        ),
                    )
                    for repeat_index in range(policy.audit.repeats_per_direction)
                ]
                cycle_results.extend(repeated)
                endpoints = [endpoint for passed_cycle, _gap, endpoint in repeated if passed_cycle and endpoint is not None]
                if len(endpoints) >= 2:
                    cycle_repeat_gaps.extend(
                        beta_rms_deg(endpoints[0], endpoint) for endpoint in endpoints[1:]
                    )
                cycle_candidate_pass.append(all(result[0] for result in repeated))
            cycle_gaps = [gap for passed_cycle, gap, _endpoint in cycle_results if passed_cycle]
            boundary_cycle_candidate_count = len(cycle_candidates)
            boundary_cycle_failure_count = sum(not passed_cycle for passed_cycle in cycle_candidate_pass)
            cycle_repeat_p95 = (
                float(np.percentile(cycle_repeat_gaps, 95)) if cycle_repeat_gaps else 0.0
            )
            passed = bool(
                certification_mode != "none"
                and overlap_gap_ok
                and transition_attempt_count > 0
                and transition_failure_count == 0
                and boundary_cycle_candidate_count > 0
                and boundary_cycle_failure_count == 0
                and max(cycle_gaps, default=math.inf) <= policy.stitch_max_deg + 1e-12
                and cycle_repeat_p95 <= policy.audit.repeat_p95_max_deg + 1e-12
            )
            if passed:
                status = f"{certification_mode}_stitchable"
            elif certification_mode == "none" or (
                certification_mode == "overlap" and not candidate_edges
            ):
                status = "indeterminate"
            else:
                status = "nonstitchable"
            rows.append(
                {
                    "left_chart_id": left_id,
                    "right_chart_id": right_id,
                    "independent_overlap_cell_count": len(overlap),
                    "overlap_fraction_of_smaller_chart": overlap_fraction,
                    "overlap_spread_mm": overlap_spread_mm,
                    "overlap_evidence_sufficient": overlap_evidence_sufficient,
                    "boundary_edge_count": len(boundary_edges),
                    "boundary_cycle_candidate_count": boundary_cycle_candidate_count,
                    "verified_boundary_cycle_count": len(cycle_gaps),
                    "verified_boundary_cycle_failure_count": boundary_cycle_failure_count,
                    "boundary_cycle_beta_max_deg": max(cycle_gaps, default=math.inf),
                    "boundary_cycle_repeat_p95_deg": cycle_repeat_p95,
                    "certification_mode": certification_mode,
                    "beta_p95_deg": p95,
                    "beta_max_deg": maximum,
                    "transition_beta_p95_deg": (
                        float(np.percentile(transition_gap_values, 95)) if transition_gap_values else math.inf
                    ),
                    "transition_beta_max_deg": (
                        float(np.max(transition_gap_values)) if transition_gap_values else math.inf
                    ),
                    "verified_transition_attempt_count": transition_attempt_count,
                    "verified_transition_failure_count": transition_failure_count,
                    "verified_transition": bool(transition_attempt_count and transition_failure_count == 0),
                    "stitch_status": status,
                    "targeted_boundary_enrichment_executed": bool(
                        boundary_edges and transition_attempt_count
                    ),
                    "stitchable": passed,
                }
            )
            if passed:
                edges.append((left_id, right_id))
    return pd.DataFrame.from_records(rows), tuple(edges)


def _chart_components(
    chart_ids: Sequence[str], edges: Sequence[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    adjacency = {chart_id: set() for chart_id in chart_ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(chart_ids)
    output: list[tuple[str, ...]] = []
    while remaining:
        root = min(remaining)
        queue = [root]
        component: set[str] = set()
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            queue.extend(sorted(adjacency[current] - component, reverse=True))
        remaining -= component
        output.append(tuple(sorted(component)))
    return tuple(output)


def _select_component(
    components: Sequence[tuple[str, ...]],
    charts: Mapping[str, RootedSectionChart],
    *,
    canonical_root_priority: Sequence[CandidateKey] = (),
) -> tuple[str, ...]:
    if not components:
        return ()
    rank_by_root = {
        (int(key[0]), str(key[1])): index
        for index, key in enumerate(canonical_root_priority)
    }

    def component_signature(component: Sequence[str]) -> tuple[CandidateKey, ...]:
        return tuple(sorted(charts[chart].root_key for chart in component))

    return min(
        components,
        key=lambda component: (
            -len(set().union(*(set(charts[chart].selected_by_node) for chart in component))),
            len(component),
            min(
                (rank_by_root.get(charts[chart].root_key, len(rank_by_root)) for chart in component),
                default=len(rank_by_root),
            ),
            component_signature(component),
        ),
    )


def _materialize_primary(
    component: Sequence[str],
    charts: Mapping[str, RootedSectionChart],
    task_nodes: Sequence[AtlasTaskNode],
    stitch_edges: Sequence[tuple[str, str]],
    *,
    canonical_root_priority: Sequence[CandidateKey] = (),
) -> tuple[dict[int, str], dict[int, np.ndarray], pd.DataFrame]:
    if not component:
        return {}, {}, pd.DataFrame()
    rank_by_root = {
        (int(key[0]), str(key[1])): index
        for index, key in enumerate(canonical_root_priority)
    }
    ranked = sorted(
        component,
        key=lambda chart_id: (
            -len(charts[chart_id].selected_by_node),
            rank_by_root.get(charts[chart_id].root_key, len(rank_by_root)),
            charts[chart_id].root_key,
        ),
    )
    available: dict[int, tuple[str, ...]] = {
        node: tuple(chart for chart in ranked if node in charts[chart].selected_by_node)
        for node in sorted(
            set().union(*(set(charts[chart].selected_by_node) for chart in component))
        )
    }
    if len(component) == 1:
        chart_id = component[0]
        assignment = {node: chart_id for node in available}
        beta = {
            node: charts[chart_id].selected_by_node[node].candidate.beta_rad.copy()
            for node in available
        }
        return assignment, beta, pd.DataFrame.from_records(
            [
                {
                    "initialization": "qualified_singleton",
                    "objective": 0.0,
                    "sweeps": 0,
                    "assignment_change_ratio_to_selected": 0.0,
                    "selected": True,
                }
            ]
        )
    node_by_id = {node.node_id: node for node in task_nodes}
    retained_nodes = set(available)
    task_edges = _induced_task_edges(retained_nodes, node_by_id)
    adjacency = _adjacency(task_edges)
    stitch_pairs = {tuple(sorted(edge)) for edge in stitch_edges}

    def unary(node: int, chart_id: str) -> float:
        candidate = charts[chart_id].selected_by_node[node].candidate
        return (
            float(candidate.residual_mm) / 3.0
            - min(float(candidate.min_margin_deg), 10.0) / 100.0
            + min(float(candidate.condition_number), 1000.0) / 100000.0
        )

    def pairwise(left: int, left_chart: str, right: int, right_chart: str) -> float:
        if (
            left_chart != right_chart
            and tuple(sorted((left_chart, right_chart))) not in stitch_pairs
        ):
            return math.inf
        gap = beta_rms_deg(
            charts[left_chart].selected_by_node[left].candidate.beta_rad,
            charts[right_chart].selected_by_node[right].candidate.beta_rad,
        )
        return gap / 10.0 + (1.0 if left_chart != right_chart else 0.0)

    def objective(assignment: Mapping[int, str]) -> float:
        value = sum(unary(node, chart_id) for node, chart_id in assignment.items())
        for left, right in task_edges:
            value += pairwise(left, assignment[left], right, assignment[right])
        return float(value)

    initializations = {
        "coverage_ranked": {node: labels[0] for node, labels in available.items()},
        "reverse_coverage_ranked": {node: labels[-1] for node, labels in available.items()},
        "unary_best": {
            node: min(labels, key=lambda chart: (unary(node, chart), chart))
            for node, labels in available.items()
        },
    }
    solutions: list[tuple[str, dict[int, str], float, int]] = []
    for name, initial in initializations.items():
        assignment = dict(initial)
        sweeps = 0
        for sweep in range(50):
            changed = False
            for node in sorted(assignment):
                def local_cost(chart_id: str) -> tuple[float, str]:
                    cost = unary(node, chart_id)
                    for neighbour in adjacency.get(node, ()):
                        if neighbour in assignment:
                            cost += pairwise(
                                node, chart_id, neighbour, assignment[neighbour]
                            )
                    return cost, chart_id

                selected = min(available[node], key=local_cost)
                if selected != assignment[node]:
                    assignment[node] = selected
                    changed = True
            sweeps = sweep + 1
            if not changed:
                break
        solutions.append((name, assignment, objective(assignment), sweeps))
    selected_name, selected_assignment, selected_objective, _sweeps = min(
        solutions,
        key=lambda row: (row[2], row[0]),
    )
    rows = []
    for name, assignment, value, sweeps in solutions:
        changed = np.mean(
            [assignment[node] != selected_assignment[node] for node in selected_assignment]
        )
        gaps = np.asarray(
            [
                beta_rms_deg(
                    charts[assignment[node]].selected_by_node[node].candidate.beta_rad,
                    charts[selected_assignment[node]].selected_by_node[node].candidate.beta_rad,
                )
                for node in selected_assignment
            ]
        )
        rows.append(
            {
                "initialization": name,
                "objective": value,
                "objective_gap_to_selected": value - selected_objective,
                "sweeps": sweeps,
                "assignment_change_ratio_to_selected": float(changed),
                "beta_p95_deg_to_selected": float(np.percentile(gaps, 95)),
                "beta_max_deg_to_selected": float(np.max(gaps)),
                "selected": name == selected_name,
            }
        )
    beta = {
        node: charts[chart_id].selected_by_node[node].candidate.beta_rad.copy()
        for node, chart_id in selected_assignment.items()
    }
    return selected_assignment, beta, pd.DataFrame.from_records(rows)


def _primary_growth(
    growth: SectionGrowthResult,
    assignment: Mapping[int, str],
    charts: Mapping[str, RootedSectionChart],
) -> SectionGrowthResult:
    node_by_id = {node.node_id: node for node in growth.task_nodes}
    selected: dict[int, SectionHypothesis] = {}
    hypotheses: dict[int, tuple[SectionHypothesis, ...]] = {}
    for node_id, chart_id in sorted(assignment.items()):
        source = charts[chart_id].selected_by_node[node_id]
        item = SectionHypothesis(
            chart_id="primary_atlas",
            root_key=source.root_key,
            node_id=node_id,
            candidate=source.candidate,
            parent_keys=(),
            score=source.score,
            wave=source.wave,
            boundary_risk=source.boundary_risk,
            uses_propagated_candidate=source.uses_propagated_candidate,
        )
        selected[node_id] = item
        hypotheses[node_id] = (item,)
    retained_nodes = set(selected)
    edges = _induced_task_edges(retained_nodes, node_by_id)
    if selected:
        root_node = min(selected)
        root_key = selected[root_node].candidate.key
        chart_rows = (
            RootedSectionChart(
                chart_id="primary_atlas",
                root_key=root_key,
                hypotheses_by_node=hypotheses,
                selected_by_node=selected,
                selected_edges=edges,
                boundary_risk_node_ids=frozenset(
                    node for node, item in selected.items() if item.boundary_risk
                ),
                frontier_node_ids=frozenset(),
            ),
        )
    else:
        chart_rows = ()
    primary = {
        node.node_id: ("primary_atlas" if node.node_id in selected else None)
        for node in growth.task_nodes
    }
    return SectionGrowthResult(
        task_nodes=growth.task_nodes,
        charts=chart_rows,
        primary_chart_by_node=primary,
        abstained_node_ids=frozenset(node for node, chart in primary.items() if chart is None),
        events=(),
        cap_hit_events=(),
        policy=growth.policy,
        continuation_attempt_count=0,
        rejected_continuation_count=0,
    )


def _expand_abstention(
    initial: set[int], task_nodes: Sequence[AtlasTaskNode], hops: int
) -> set[int]:
    adjacency = {node.node_id: set(node.neighbor_node_ids) for node in task_nodes}
    output = set(initial)
    frontier = set(initial)
    for _ in range(hops):
        frontier = set().union(*(adjacency.get(node, set()) for node in frontier)) - output
        output |= frontier
    return output


def _radius_abstention(
    initial: set[int], task_nodes: Sequence[AtlasTaskNode], radius_mm: float
) -> set[int]:
    if not initial or radius_mm <= 0:
        return set(initial)
    node_by_id = {node.node_id: node for node in task_nodes}
    sources = np.vstack([node_by_id[node].xyz_m for node in sorted(initial)])
    radius_m = float(radius_mm) / 1000.0
    return {
        node.node_id
        for node in task_nodes
        if float(np.min(np.linalg.norm(sources - node.xyz_m, axis=1))) <= radius_m + 1e-12
    }


def _largest_task_component_ratio(covered: set[int], nodes: Sequence[AtlasTaskNode]) -> float:
    if not nodes or not covered:
        return 0.0
    adjacency = {node.node_id: set(node.neighbor_node_ids) & covered for node in nodes if node.node_id in covered}
    remaining = set(covered)
    largest = 0
    while remaining:
        root = next(iter(remaining))
        queue = [root]
        component: set[int] = set()
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            queue.extend(adjacency.get(current, set()) - component)
        remaining -= component
        largest = max(largest, len(component))
    return largest / len(nodes)


def _cycle_rank(nodes: set[int], edges: Sequence[tuple[int, int]]) -> int:
    if not nodes:
        return 0
    component_count = len(_node_components(nodes, edges))
    return max(0, len(edges) - len(nodes) + component_count)


def _adjacency(edges: Sequence[tuple[int, int]]) -> dict[int, set[int]]:
    output: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        output[int(left)].add(int(right))
        output[int(right)].add(int(left))
    return output


def _spanning_tree(
    adjacency: Mapping[int, set[int]], root: int, selected: set[int]
) -> tuple[dict[int, int | None], dict[int, tuple[int, ...]], set[tuple[int, int]]]:
    if root not in selected:
        root = min(selected)
    parent: dict[int, int | None] = {root: None}
    paths: dict[int, tuple[int, ...]] = {root: (root,)}
    tree_edges: set[tuple[int, int]] = set()
    queue: deque[int] = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, set()) & selected):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            paths[neighbor] = paths[current] + (neighbor,)
            tree_edges.add(tuple(sorted((current, neighbor))))
            queue.append(neighbor)
    return parent, paths, tree_edges


def _tree_path(left: int, right: int, parent: Mapping[int, int | None]) -> tuple[int, ...]:
    def ancestors(node: int) -> list[int]:
        values = [node]
        while parent.get(node) is not None:
            node = int(parent[node])
            values.append(node)
        return values

    left_values = ancestors(left)
    right_values = ancestors(right)
    right_position = {node: index for index, node in enumerate(right_values)}
    common_index = next(index for index, node in enumerate(left_values) if node in right_position)
    common = left_values[common_index]
    return tuple(left_values[: common_index + 1] + list(reversed(right_values[: right_position[common]])))


def _canonical_path_signature(path: Sequence[int], *, cycle: bool) -> tuple[int, ...]:
    values = tuple(int(value) for value in path)
    if cycle and len(values) > 1 and values[0] == values[-1]:
        body = values[:-1]
        rotations = [body[index:] + body[:index] for index in range(len(body))]
        reverse = tuple(reversed(body))
        rotations.extend(reverse[index:] + reverse[:index] for index in range(len(reverse)))
        return min(rotations)
    reverse = tuple(reversed(values))
    return min(values, reverse)


def _records_frame(records: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    return records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records)
