"""Bridge finite workspace tables to the V14 atlas evidence model.

This module is intentionally an in-memory adapter.  It accepts representative
task/candidate tables, constructs the branch-aware product graph using a real
continuation supplied by an environment, and then reduces the result into the
explicit cell/section evidence consumed by :mod:`workspace_atlas`.

Two rules are deliberately fail-closed here:

* a representative task node is not evidence that its whole workspace cell is
  labelable; unless the input explicitly records complete cell-measure
  evidence, an unresolved measurement witness is retained for that cell; and
* canonical-product-graph edges are only a structural seed.  Every selected
  directed edge, every fundamental cycle, both paths around each fundamental
  cycle, and independent repeat executions are freshly continued before a
  section can claim complete closure.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import heapq
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .atlas_audit import (
    AtlasAuditPolicy,
    AtlasAuditReport,
    AuditExecutionTask,
    AuditMetric,
    PathTrace,
)
from .canonical import beta_rms_deg
from .canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    CanonicalAtlas,
    CanonicalChart,
    ContinuationAdapter,
    ContinuationOutcome,
    ProductGraph,
    build_canonical_atlas,
    make_predictor_corrector_continuation,
)
from .workspace_atlas import (
    CellProbeEvidence,
    CellResolution,
    InverseStatus,
    OverlapAssessment,
    OverlapEvidence,
    OverlapKind,
    PhysicalStatus,
    RiskFlag,
    SectionEvidence,
    SectionSelection,
    TaskCellEvidence,
    WorkspaceAtlasBuilder,
    WorkspaceAtlasInput,
    WorkspaceAtlasPolicy,
    WorkspaceAtlasResult,
)
from .workspace_reach import CellKey


_TASK_XYZ_LAYOUTS = (
    ("x_m", "y_m", "z_m"),
    ("target_x_m", "target_y_m", "target_z_m"),
)
_BETA_LAYOUTS = (
    tuple(f"beta{index}_rad" for index in range(1, 7)),
    tuple(f"beta{index}_rad" for index in range(6)),
    tuple(f"beta_{index}_rad" for index in range(1, 7)),
    tuple(f"beta_{index}_rad" for index in range(6)),
)


@dataclass(frozen=True)
class WorkspaceAtlasIntegrationPolicy:
    """Frozen adapter choices that sit around the existing V12/V14 policies."""

    atlas_policy: AtlasPolicy = field(default_factory=AtlasPolicy)
    audit_policy: AtlasAuditPolicy = field(default_factory=AtlasAuditPolicy)
    workspace_policy: WorkspaceAtlasPolicy = field(default_factory=WorkspaceAtlasPolicy)
    candidate_cluster_deg: float = 0.5
    default_cell_level_mm: int = 10

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.candidate_cluster_deg)) or self.candidate_cluster_deg <= 0.0:
            raise ValueError("candidate_cluster_deg must be finite and positive")
        if int(self.default_cell_level_mm) <= 0:
            raise ValueError("default_cell_level_mm must be positive")
        object.__setattr__(self, "default_cell_level_mm", int(self.default_cell_level_mm))


@dataclass(frozen=True)
class FreshChartAudit:
    """Complete fresh-continuation evidence for one canonical chart."""

    chart_id: int
    selected_task_edge_count: int
    fundamental_cycle_count: int
    fresh_execution_count: int
    path: AuditMetric
    loop: AuditMetric
    direction: AuditMetric
    repeat: AuditMetric
    multipath: AuditMetric
    full_fundamental_closure_checked: bool

    @property
    def section_valid(self) -> bool:
        return bool(
            self.full_fundamental_closure_checked
            and self.path.gate_pass
            and self.loop.gate_pass
            and self.direction.gate_pass
            and self.repeat.gate_pass
            and self.multipath.gate_pass
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart_id": int(self.chart_id),
            "selected_task_edge_count": int(self.selected_task_edge_count),
            "fundamental_cycle_count": int(self.fundamental_cycle_count),
            "fresh_execution_count": int(self.fresh_execution_count),
            "path": self.path.as_dict(),
            "loop": self.loop.as_dict(),
            "direction": self.direction.as_dict(),
            "repeat": self.repeat.as_dict(),
            "multipath": self.multipath.as_dict(),
            "full_fundamental_closure_checked": bool(
                self.full_fundamental_closure_checked
            ),
            "section_valid": bool(self.section_valid),
        }


@dataclass(frozen=True)
class WorkspaceAtlasIntegrationResult:
    """Typed atlas evidence plus primitive DataFrame/report projections."""

    task_nodes: tuple[AtlasTaskNode, ...]
    candidates: tuple[AtlasCandidate, ...]
    product_graph: ProductGraph
    canonical_atlas: CanonicalAtlas
    chart_audits: tuple[FreshChartAudit, ...]
    audit_report: AtlasAuditReport
    workspace_input: WorkspaceAtlasInput
    workspace_result: WorkspaceAtlasResult
    frames: Mapping[str, pd.DataFrame]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", MappingProxyType(dict(self.frames)))
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe reports and record-oriented frame projections."""

        return {
            "report": _json_safe(dict(self.report)),
            "frames": {
                name: _json_safe(frame.to_dict(orient="records"))
                for name, frame in self.frames.items()
            },
        }


@dataclass(frozen=True)
class _TaskContext:
    node: AtlasTaskNode
    task_id: str
    physical_point_id: str
    cell: CellKey
    is_measure_probe: bool
    is_representative: bool
    cell_measure_complete: bool
    physical_status: PhysicalStatus
    risk_flags: frozenset[RiskFlag]


@dataclass(frozen=True)
class _FreshExecution:
    task: AuditExecutionTask
    schedule_id: str
    repetition: int
    trace: PathTrace


def task_nodes_from_frame(
    task_nodes: pd.DataFrame,
    *,
    task_edges: pd.DataFrame | None = None,
    default_cell_level_mm: int = 10,
) -> tuple[AtlasTaskNode, ...]:
    """Convert a representative task-node table into immutable graph nodes.

    The accepted node identifiers are ``task_node_id`` or ``node_id``.  XYZ
    may use either ``x_m/y_m/z_m`` or ``target_x_m/target_y_m/target_z_m``.
    Edges may be supplied as a separate ``left_node_id/right_node_id`` table
    or as a node-table neighbour column.
    """

    contexts = _adapt_task_nodes(
        task_nodes,
        task_edges=task_edges,
        default_cell_level_mm=default_cell_level_mm,
    )
    return tuple(context.node for context in contexts)


def clustered_candidates_from_frame(
    candidates: pd.DataFrame,
    *,
    environment: Any,
    candidate_cluster_deg: float = 0.5,
) -> tuple[AtlasCandidate, ...]:
    """Convert and deterministically cluster an IK candidate table.

    If the table already carries ``cluster_id``, that identity is preserved as
    a hard boundary (only representatives *within* an input cluster may be
    reduced).  Without it, the adapter forms stable beta-space clusters using
    the supplied threshold and the scientific candidate rank order.
    """

    rows, _frame = _adapt_candidates(
        candidates,
        environment=environment,
        candidate_cluster_deg=candidate_cluster_deg,
    )
    return rows


def build_workspace_atlas_integration(
    task_nodes: pd.DataFrame,
    candidates: pd.DataFrame,
    environment: Any,
    *,
    task_edges: pd.DataFrame | None = None,
    continuation: ContinuationAdapter | None = None,
    branch_audit: Sequence[Any] = (),
    policy: WorkspaceAtlasIntegrationPolicy | None = None,
) -> WorkspaceAtlasIntegrationResult:
    """Build a fail-closed V14 workspace-atlas evidence package.

    ``continuation`` is optional for environments that own a stricter
    waypoint-aware corrector.  In its absence the adapter creates the bounded
    predictor-corrector from ``environment``.  It is used both for product
    graph construction and again for every fresh audit execution; saved graph
    continuations are never replayed as repeat evidence here.
    """

    active = WorkspaceAtlasIntegrationPolicy() if policy is None else policy
    contexts = _adapt_task_nodes(
        task_nodes,
        task_edges=task_edges,
        default_cell_level_mm=active.default_cell_level_mm,
    )
    nodes = tuple(context.node for context in contexts)
    node_ids = {node.node_id for node in nodes}
    converted_candidates, candidate_frame = _adapt_candidates(
        candidates,
        environment=environment,
        candidate_cluster_deg=active.candidate_cluster_deg,
    )
    unknown_candidate_nodes = sorted(
        {candidate.node_id for candidate in converted_candidates} - node_ids
    )
    if unknown_candidate_nodes:
        raise ValueError(
            "candidate table references unknown task nodes: "
            f"{unknown_candidate_nodes}"
        )

    active_continuation = continuation
    if active_continuation is None:
        _environment_bounds(environment)
        active_continuation = make_predictor_corrector_continuation(
            environment,
            residual_tolerance_mm=active.atlas_policy.continuation_residual_max_mm,
        )

    canonical_atlas = build_canonical_atlas(
        nodes,
        converted_candidates,
        active_continuation,
        policy=active.atlas_policy,
    )
    chart_audits, executions = _run_complete_fresh_audit(
        canonical_atlas,
        active_continuation,
        active.audit_policy,
    )
    audit_by_chart = {item.chart_id: item for item in chart_audits}
    context_by_node = {context.node.node_id: context for context in contexts}
    candidate_by_key = canonical_atlas.product_graph.candidate_by_key
    section_id_by_chart = {
        chart.chart_id: _section_id(chart.chart_id)
        for chart in canonical_atlas.charts
    }
    probe_id_by_chart_node = {
        (chart.chart_id, node_id): _chart_probe_id(
            context_by_node[node_id].task_id,
            chart.chart_id,
        )
        for chart in canonical_atlas.charts
        for node_id in chart.selection_by_node
    }
    if len(set(probe_id_by_chart_node.values())) != len(probe_id_by_chart_node):
        raise RuntimeError("chart task probes must have globally unique task IDs")

    sections = _section_evidence(
        canonical_atlas,
        audit_by_chart,
        section_id_by_chart,
        probe_id_by_chart_node,
        active.audit_policy,
    )
    overlaps = _overlap_evidence(
        canonical_atlas,
        contexts=context_by_node,
        chart_audits=audit_by_chart,
        section_id_by_chart=section_id_by_chart,
    )
    overlap_assessments = tuple(
        _assess_overlap(item, active.workspace_policy) for item in overlaps
    )
    primary_section_ids, required_transitions, partition_by_node = _primary_partition(
        nodes,
        canonical_atlas,
        audit_by_chart,
        overlap_assessments,
        section_id_by_chart,
    )
    cells, probe_rows = _cell_evidence(
        contexts,
        canonical_atlas,
        candidate_by_key,
        section_id_by_chart,
        probe_id_by_chart_node,
    )
    workspace_input = WorkspaceAtlasInput(
        cells=cells,
        sections=sections,
        overlaps=overlaps,
        branch_audit=tuple(branch_audit),
        primary_section_ids=primary_section_ids,
        required_transitions=required_transitions,
    )
    workspace_result = WorkspaceAtlasBuilder(active.workspace_policy).build(
        workspace_input
    )
    audit_report = _aggregate_audit_report(
        canonical_atlas,
        chart_audits,
        overlap_assessments,
        primary_section_ids,
    )

    frames = _result_frames(
        contexts=contexts,
        candidates=candidate_frame,
        atlas=canonical_atlas,
        executions=executions,
        chart_audits=chart_audits,
        sections=sections,
        overlaps=overlap_assessments,
        partition_by_node=partition_by_node,
        probe_rows=probe_rows,
        workspace_result=workspace_result,
    )
    report = _integration_report(
        canonical_atlas=canonical_atlas,
        chart_audits=chart_audits,
        audit_report=audit_report,
        workspace_result=workspace_result,
        cells=cells,
        primary_section_ids=primary_section_ids,
        required_transitions=required_transitions,
        execution_count=len(executions),
    )
    return WorkspaceAtlasIntegrationResult(
        task_nodes=nodes,
        candidates=converted_candidates,
        product_graph=canonical_atlas.product_graph,
        canonical_atlas=canonical_atlas,
        chart_audits=chart_audits,
        audit_report=audit_report,
        workspace_input=workspace_input,
        workspace_result=workspace_result,
        frames=frames,
        report=report,
    )


# The shorter spelling is useful to callers that already use ``build_*`` for
# the two lower-level atlas modules.
integrate_workspace_atlas = build_workspace_atlas_integration


def _adapt_task_nodes(
    task_nodes: pd.DataFrame,
    *,
    task_edges: pd.DataFrame | None,
    default_cell_level_mm: int,
) -> tuple[_TaskContext, ...]:
    frame = _require_frame(task_nodes, "task_nodes")
    id_column = _first_column(frame, ("task_node_id", "node_id"), "task node ID")
    xyz_columns = _xyz_columns(frame, "task_nodes")
    neighbor_column = _optional_column(
        frame,
        ("neighbor_node_ids", "neighbors", "neighbor_ids"),
    )
    task_id_column = _optional_column(frame, ("task_id", "task_probe_id"))
    physical_id_column = _optional_column(
        frame,
        ("physical_point_id", "point_id", "target_id"),
    )
    core_column = _optional_column(frame, ("core_safe", "is_core_safe", "strict_gold"))
    measure_column = _optional_column(frame, ("is_measure_probe", "measure_probe"))
    representative_column = _optional_column(
        frame,
        ("is_representative", "representative_probe"),
    )
    coverage_column = _optional_column(
        frame,
        (
            "cell_measure_complete",
            "full_cell_measure_evidence",
            "cell_coverage_complete",
        ),
    )
    physical_status_column = _optional_column(frame, ("physical_status",))
    risk_column = _optional_column(frame, ("risk_flags",))

    raw: dict[int, dict[str, Any]] = {}
    for row_index, row in frame.iterrows():
        node_id = _integer(row[id_column], f"task node ID at row {row_index}")
        if node_id in raw:
            raise ValueError(f"task node IDs must be unique; duplicate {node_id}")
        xyz = np.asarray([row[column] for column in xyz_columns], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"task node {node_id} has non-finite xyz")
        raw[node_id] = {
            "xyz": xyz,
            "neighbors": set(
                _integer(value, f"neighbour of task node {node_id}")
                for value in _sequence(row[neighbor_column])
            )
            if neighbor_column is not None
            else set(),
            "task_id": _identifier(
                row[task_id_column] if task_id_column is not None else f"task_node_{node_id}",
                f"task ID for node {node_id}",
            ),
            "physical_point_id": _identifier(
                row[physical_id_column]
                if physical_id_column is not None
                else f"physical_node_{node_id}",
                f"physical point ID for node {node_id}",
            ),
            "core_safe": _as_bool(
                row[core_column] if core_column is not None else True,
                default=True,
                label=f"core_safe for node {node_id}",
            ),
            "is_measure_probe": _as_bool(
                row[measure_column] if measure_column is not None else True,
                default=True,
                label=f"is_measure_probe for node {node_id}",
            ),
            "is_representative": _as_bool(
                row[representative_column]
                if representative_column is not None
                else False,
                default=False,
                label=f"is_representative for node {node_id}",
            ),
            "cell_measure_complete": _as_bool(
                row[coverage_column] if coverage_column is not None else False,
                default=False,
                label=f"cell_measure_complete for node {node_id}",
            ),
            "cell": _cell_from_row(
                row,
                xyz,
                default_level_mm=default_cell_level_mm,
            ),
            "physical_status": _physical_status(
                row[physical_status_column]
                if physical_status_column is not None
                else PhysicalStatus.VALID,
            ),
            "risk_flags": _risk_flags(row[risk_column]) if risk_column is not None else frozenset(),
        }

    if task_edges is not None:
        edges = _require_frame(task_edges, "task_edges")
        left_column = _first_column(
            edges,
            ("left_node_id", "source_node_id", "node_a_id"),
            "task edge left node ID",
        )
        right_column = _first_column(
            edges,
            ("right_node_id", "target_node_id", "node_b_id"),
            "task edge right node ID",
        )
        for row_index, row in edges.iterrows():
            left = _integer(row[left_column], f"task edge left node at row {row_index}")
            right = _integer(row[right_column], f"task edge right node at row {row_index}")
            if left == right:
                raise ValueError("task edges cannot be self edges")
            if left not in raw or right not in raw:
                raise ValueError("task edge references an unknown task node")
            raw[left]["neighbors"].add(right)
            raw[right]["neighbors"].add(left)

    unknown_neighbors = {
        (node_id, neighbor)
        for node_id, item in raw.items()
        for neighbor in item["neighbors"]
        if neighbor not in raw
    }
    if unknown_neighbors:
        raise ValueError(
            "task node references unknown neighbours: "
            f"{sorted(unknown_neighbors)}"
        )
    task_ids = [item["task_id"] for item in raw.values()]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task-node task_id values must be globally unique")
    physical_xyz: dict[str, np.ndarray] = {}
    for item in raw.values():
        point_id = item["physical_point_id"]
        previous = physical_xyz.get(point_id)
        if previous is not None and not np.allclose(previous, item["xyz"], rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "one physical_point_id cannot name different target coordinates"
            )
        physical_xyz[point_id] = item["xyz"]

    contexts: list[_TaskContext] = []
    for node_id in sorted(raw):
        item = raw[node_id]
        node = AtlasTaskNode(
            node_id=node_id,
            xyz_m=item["xyz"],
            neighbor_node_ids=tuple(sorted(item["neighbors"])),
            core_safe=item["core_safe"],
        )
        contexts.append(
            _TaskContext(
                node=node,
                task_id=item["task_id"],
                physical_point_id=item["physical_point_id"],
                cell=item["cell"],
                is_measure_probe=item["is_measure_probe"],
                is_representative=item["is_representative"],
                cell_measure_complete=item["cell_measure_complete"],
                physical_status=item["physical_status"],
                risk_flags=item["risk_flags"],
            )
        )
    return tuple(contexts)


def _adapt_candidates(
    candidates: pd.DataFrame,
    *,
    environment: Any,
    candidate_cluster_deg: float,
) -> tuple[tuple[AtlasCandidate, ...], pd.DataFrame]:
    frame = _require_frame(candidates, "candidates")
    node_column = _first_column(frame, ("task_node_id", "node_id"), "candidate node ID")
    candidate_column = _first_column(frame, ("candidate_id", "id"), "candidate ID")
    beta_getter = _beta_getter(frame, "candidates")
    residual_column = _first_column(
        frame,
        ("residual_mm", "fk_residual_mm"),
        "candidate residual_mm",
    )
    margin_column = _optional_column(frame, ("min_margin_deg", "minimum_margin_deg"))
    normalized_margin_column = _optional_column(
        frame,
        ("normalized_min_margin", "normalized_margin"),
    )
    posture_column = _optional_column(frame, ("posture_cost",))
    condition_column = _optional_column(frame, ("condition_number", "kappa"))
    quality_column = _optional_column(frame, ("quality", "quality_tier"))
    solver_column = _optional_column(frame, ("solver_success", "success"))
    bounds_column = _optional_column(frame, ("actual_bounds",))
    cluster_column = _optional_column(frame, ("cluster_id",))
    bounds = _environment_bounds(environment)

    raw: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()
    for row_index, row in frame.iterrows():
        node_id = _integer(row[node_column], f"candidate node ID at row {row_index}")
        candidate_id = _identifier(
            row[candidate_column], f"candidate ID at row {row_index}"
        )
        key = (node_id, candidate_id)
        if key in seen_keys:
            raise ValueError("candidate IDs must be unique within a task node")
        seen_keys.add(key)
        beta = beta_getter(row)
        if not np.isfinite(beta).all():
            raise ValueError(f"candidate {key} beta values must be finite")
        residual = _finite_nonnegative(
            row[residual_column], f"residual_mm for candidate {key}"
        )
        margin = _finite_value(
            row[margin_column] if margin_column is not None else 0.0,
            f"minimum margin for candidate {key}",
        )
        normalized_margin = _finite_value(
            row[normalized_margin_column]
            if normalized_margin_column is not None
            else 0.0,
            f"normalized minimum margin for candidate {key}",
        )
        posture = _finite_value(
            row[posture_column] if posture_column is not None else 0.0,
            f"posture cost for candidate {key}",
        )
        condition = _finite_nonnegative(
            row[condition_column] if condition_column is not None else 0.0,
            f"condition number for candidate {key}",
        )
        quality = _quality(
            row[quality_column] if quality_column is not None else "Gold"
        )
        solver_success = _as_bool(
            row[solver_column] if solver_column is not None else True,
            default=True,
            label=f"solver_success for candidate {key}",
        )
        supplied_bounds = _as_bool(
            row[bounds_column] if bounds_column is not None else True,
            default=True,
            label=f"actual_bounds for candidate {key}",
        )
        actual_bounds = bool(
            supplied_bounds
            and np.all(beta >= bounds[:, 0] - 1.0e-12)
            and np.all(beta <= bounds[:, 1] + 1.0e-12)
        )
        input_cluster = None
        if cluster_column is not None and not _is_missing(row[cluster_column]):
            input_cluster = str(row[cluster_column]).strip()
            if not input_cluster:
                raise ValueError("candidate cluster_id cannot be empty")
        raw.append(
            {
                "node_id": node_id,
                "candidate_id": candidate_id,
                "beta_rad": beta,
                "residual_mm": residual,
                "min_margin_deg": margin,
                "normalized_min_margin": normalized_margin,
                "posture_cost": posture,
                "condition_number": condition,
                "quality": quality,
                "solver_success": solver_success,
                "actual_bounds": actual_bounds,
                "input_cluster": input_cluster,
            }
        )

    if cluster_column is not None:
        present = [item["input_cluster"] is not None for item in raw]
        if any(present) and not all(present):
            raise ValueError("candidate cluster_id must be present for every row or none")

    output: list[AtlasCandidate] = []
    frame_rows: list[dict[str, Any]] = []
    by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in raw:
        by_node[item["node_id"]].append(item)
    for node_id in sorted(by_node):
        groups = _candidate_groups(
            by_node[node_id],
            threshold_deg=float(candidate_cluster_deg),
        )
        for cluster_id, group in enumerate(groups):
            representative = min(group, key=_candidate_rank)
            source_ids = tuple(sorted(item["candidate_id"] for item in group))
            candidate = AtlasCandidate(
                node_id=node_id,
                candidate_id=representative["candidate_id"],
                beta_rad=representative["beta_rad"],
                residual_mm=representative["residual_mm"],
                min_margin_deg=representative["min_margin_deg"],
                normalized_min_margin=representative["normalized_min_margin"],
                posture_cost=representative["posture_cost"],
                condition_number=representative["condition_number"],
                quality=representative["quality"],
                solver_success=representative["solver_success"],
                actual_bounds=representative["actual_bounds"],
                cluster_id=cluster_id,
                diagnostics={
                    "source_candidate_ids": source_ids,
                    "cluster_size": len(group),
                    "input_cluster_id": representative["input_cluster"],
                },
            )
            output.append(candidate)
            frame_rows.append(
                {
                    "task_node_id": int(candidate.node_id),
                    "candidate_id": candidate.candidate_id,
                    "cluster_id": int(cluster_id),
                    "source_candidate_ids": list(source_ids),
                    "cluster_size": int(len(group)),
                    "quality": candidate.quality,
                    "solver_success": bool(candidate.solver_success),
                    "actual_bounds": bool(candidate.actual_bounds),
                    "residual_mm": float(candidate.residual_mm),
                    "min_margin_deg": float(candidate.min_margin_deg),
                    "normalized_min_margin": float(candidate.normalized_min_margin),
                    "posture_cost": float(candidate.posture_cost),
                    "condition_number": float(candidate.condition_number),
                    **{
                        f"beta{index + 1}_rad": float(candidate.beta_rad[index])
                        for index in range(6)
                    },
                }
            )
    converted = tuple(sorted(output, key=lambda item: item.key))
    return converted, pd.DataFrame.from_records(frame_rows)


def _candidate_groups(
    rows: Sequence[Mapping[str, Any]], *, threshold_deg: float
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    ordered = tuple(sorted(rows, key=_candidate_rank))
    if ordered and ordered[0]["input_cluster"] is not None:
        by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in ordered:
            by_cluster[str(item["input_cluster"])].append(item)
        return tuple(
            tuple(sorted(group, key=_candidate_rank))
            for _label, group in sorted(
                by_cluster.items(),
                key=lambda item: (_candidate_rank(min(item[1], key=_candidate_rank)), item[0]),
            )
        )
    groups: list[list[Mapping[str, Any]]] = []
    for item in ordered:
        for group in groups:
            representative = min(group, key=_candidate_rank)
            if beta_rms_deg(item["beta_rad"], representative["beta_rad"]) <= threshold_deg + 1.0e-12:
                group.append(item)
                break
        else:
            groups.append([item])
    return tuple(tuple(group) for group in groups)


def _candidate_rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        {"Gold": 0, "Silver": 1, "Reject": 2}[str(item["quality"])],
        -float(item["min_margin_deg"]),
        float(item["residual_mm"]),
        float(item["posture_cost"]),
        float(item["condition_number"]),
        str(item["candidate_id"]),
    )


def _run_complete_fresh_audit(
    atlas: CanonicalAtlas,
    continuation: ContinuationAdapter,
    policy: AtlasAuditPolicy,
) -> tuple[tuple[FreshChartAudit, ...], tuple[_FreshExecution, ...]]:
    """Freshly execute all selected edges and every fundamental cycle.

    The lower-level audit helper intentionally limits itself to a registered
    endpoint/loop sample.  This adapter has a different claim: it enumerates
    the full selected robust graph, uses a deterministic spanning forest, and
    checks every non-tree edge's fundamental cycle.  Every scheduled path is
    executed twice through ``continuation`` rather than replaying graph edges.
    """

    all_audits: list[FreshChartAudit] = []
    all_executions: list[_FreshExecution] = []
    for chart in atlas.charts:
        graph = _selected_chart_graph(atlas, chart)
        tree_paths, tree_edges, connected = _spanning_tree_paths(
            graph,
            root=chart.root_key[0],
        )
        selected_edges = tuple(sorted(graph["edges"]))
        chords = tuple(edge for edge in selected_edges if edge not in tree_edges)
        schedules: dict[str, tuple[str, tuple[int, ...]]] = {}
        direct_schedule: list[tuple[str, tuple[int, int]]] = []
        for edge_index, (left, right) in enumerate(selected_edges):
            for direction, path in (
                ("forward", (left, right)),
                ("reverse", (right, left)),
            ):
                schedule_id = (
                    f"chart{chart.chart_id:03d}_edge{edge_index:06d}_{direction}"
                )
                schedules[schedule_id] = ("edge", path)
                direct_schedule.append((schedule_id, (left, right)))
        fundamental_schedule: list[tuple[str, str, str]] = []
        for chord_index, (left, right) in enumerate(chords):
            left_path = tree_paths.get(left)
            right_path = tree_paths.get(right)
            if left_path is None or right_path is None:
                # The graph is disconnected; missing paths are retained below.
                continue
            path_a_id = f"chart{chart.chart_id:03d}_fundamental{chord_index:06d}_tree"
            path_b_id = f"chart{chart.chart_id:03d}_fundamental{chord_index:06d}_chord"
            cycle_id = f"chart{chart.chart_id:03d}_fundamental{chord_index:06d}_cycle"
            schedules[path_a_id] = ("multipath_tree", right_path)
            schedules[path_b_id] = ("multipath_chord", left_path + (right,))
            schedules[cycle_id] = (
                "fundamental_cycle",
                left_path + (right,) + tuple(reversed(right_path))[1:],
            )
            fundamental_schedule.append((path_a_id, path_b_id, cycle_id))

        executions_by_schedule: dict[str, list[_FreshExecution]] = defaultdict(list)
        for schedule_id, (kind, path) in schedules.items():
            for repetition in range(2):
                task = AuditExecutionTask(
                    task_id=f"{schedule_id}_fresh{repetition}",
                    chart_id=chart.chart_id,
                    kind=kind,
                    path=path,
                )
                trace = _fresh_trace(atlas, chart, path, continuation)
                execution = _FreshExecution(task, schedule_id, repetition, trace)
                executions_by_schedule[schedule_id].append(execution)
                all_executions.append(execution)

        selected = chart.selection_by_node
        candidates = atlas.product_graph.candidate_by_key
        path_values: list[float] = []
        direction_values: list[float] = []
        path_missing = 0
        direction_missing = 0
        for schedule_id, _edge in direct_schedule:
            first = _first_execution(executions_by_schedule[schedule_id])
            path = schedules[schedule_id][1]
            if not _complete_trace(first.trace, path):
                path_missing += 1
                direction_missing += 1
                continue
            target = candidates[selected[path[-1]]].beta_rad
            gap = beta_rms_deg(first.trace.beta_rad_by_node[-1], target)
            path_values.append(gap)
            direction_values.append(gap)
        if not direct_schedule:
            path_missing += 1
            direction_missing += 1

        loop_values: list[float] = []
        multipath_values: list[float] = []
        loop_missing = 0
        multipath_missing = 0
        for path_a_id, path_b_id, cycle_id in fundamental_schedule:
            path_a = schedules[path_a_id][1]
            path_b = schedules[path_b_id][1]
            cycle_path = schedules[cycle_id][1]
            first_a = _first_execution(executions_by_schedule[path_a_id]).trace
            first_b = _first_execution(executions_by_schedule[path_b_id]).trace
            first_cycle = _first_execution(executions_by_schedule[cycle_id]).trace
            if _complete_trace(first_a, path_a) and _complete_trace(first_b, path_b):
                multipath_values.append(
                    beta_rms_deg(
                        first_a.beta_rad_by_node[-1],
                        first_b.beta_rad_by_node[-1],
                    )
                )
            else:
                multipath_missing += 1
            if _complete_trace(first_cycle, cycle_path):
                loop_values.append(
                    beta_rms_deg(
                        first_cycle.beta_rad_by_node[0],
                        first_cycle.beta_rad_by_node[-1],
                    )
                )
            else:
                loop_missing += 1
        if not chords or len(fundamental_schedule) != len(chords) or not connected:
            # A tree has no observed loop closure.  Treat the absence as a
            # missing certificate rather than a vacuous pass for a global atlas.
            loop_missing += max(1, len(chords) - len(fundamental_schedule))
            multipath_missing += max(1, len(chords) - len(fundamental_schedule))

        repeat_values: list[float] = []
        repeat_missing = 0
        for schedule_id, (_kind, path) in schedules.items():
            first, second = _two_executions(executions_by_schedule[schedule_id])
            if not (_complete_trace(first.trace, path) and _complete_trace(second.trace, path)):
                repeat_missing += 1
                continue
            repeat_values.extend(
                beta_rms_deg(left, right)
                for left, right in zip(
                    first.trace.beta_rad_by_node,
                    second.trace.beta_rad_by_node,
                    strict=True,
                )
            )
        if not schedules:
            repeat_missing += 1

        path_metric = _metric(
            path_values,
            p95_limit=policy.path_p95_deg,
            max_limit=policy.common_max_deg,
            missing=path_missing,
        )
        loop_metric = _metric(
            loop_values,
            p95_limit=policy.loop_p95_deg,
            max_limit=policy.common_max_deg,
            missing=loop_missing,
        )
        direction_metric = _metric(
            direction_values,
            p95_limit=policy.direction_p95_deg,
            max_limit=policy.common_max_deg,
            missing=direction_missing,
        )
        repeat_metric = _metric(
            repeat_values,
            p95_limit=policy.repeat_p95_deg,
            max_limit=policy.common_max_deg,
            missing=repeat_missing,
        )
        multipath_metric = _metric(
            multipath_values,
            p95_limit=policy.path_p95_deg,
            max_limit=policy.common_max_deg,
            missing=multipath_missing,
        )
        all_audits.append(
            FreshChartAudit(
                chart_id=chart.chart_id,
                selected_task_edge_count=len(selected_edges),
                fundamental_cycle_count=len(chords),
                fresh_execution_count=sum(
                    len(values) for values in executions_by_schedule.values()
                ),
                path=path_metric,
                loop=loop_metric,
                direction=direction_metric,
                repeat=repeat_metric,
                multipath=multipath_metric,
                full_fundamental_closure_checked=bool(
                    connected
                    and len(chords) > 0
                    and len(fundamental_schedule) == len(chords)
                    and loop_missing == 0
                ),
            )
        )
    return tuple(all_audits), tuple(all_executions)


def _selected_chart_graph(
    atlas: CanonicalAtlas, chart: CanonicalChart
) -> Mapping[str, Any]:
    selected = chart.selection_by_node
    edges: set[tuple[int, int]] = set()
    adjacency: dict[int, list[tuple[int, float]]] = {
        node_id: [] for node_id in selected
    }
    for edge in atlas.product_graph.robust_edges:
        left, right = edge.left_key[0], edge.right_key[0]
        if selected.get(left) != edge.left_key or selected.get(right) != edge.right_key:
            continue
        pair = (left, right) if left < right else (right, left)
        edges.add(pair)
        adjacency[left].append((right, float(edge.cost)))
        adjacency[right].append((left, float(edge.cost)))
    return MappingProxyType(
        {
            "edges": tuple(sorted(edges)),
            "adjacency": MappingProxyType(
                {
                    node: tuple(sorted(values, key=lambda item: (item[1], item[0])))
                    for node, values in sorted(adjacency.items())
                }
            ),
        }
    )


def _spanning_tree_paths(
    graph: Mapping[str, Any], *, root: int
) -> tuple[Mapping[int, tuple[int, ...]], frozenset[tuple[int, int]], bool]:
    adjacency: Mapping[int, tuple[tuple[int, float], ...]] = graph["adjacency"]
    if root not in adjacency:
        return MappingProxyType({}), frozenset(), False
    distance: dict[int, float] = {root: 0.0}
    parent: dict[int, int | None] = {root: None}
    queue: list[tuple[float, int]] = [(0.0, root)]
    while queue:
        value, current = heapq.heappop(queue)
        if value > distance.get(current, math.inf) + 1.0e-12:
            continue
        for neighbor, cost in adjacency[current]:
            candidate = value + float(cost)
            previous = distance.get(neighbor, math.inf)
            old_parent = parent.get(neighbor)
            if (
                candidate < previous - 1.0e-12
                or (
                    abs(candidate - previous) <= 1.0e-12
                    and (old_parent is None or current < old_parent)
                )
            ):
                distance[neighbor] = candidate
                parent[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    paths: dict[int, tuple[int, ...]] = {}
    tree_edges: set[tuple[int, int]] = set()
    for node in sorted(parent):
        chain = [node]
        while chain[-1] != root:
            previous = parent[chain[-1]]
            if previous is None:
                raise RuntimeError("spanning-tree parent chain is incomplete")
            pair = (previous, chain[-1]) if previous < chain[-1] else (chain[-1], previous)
            tree_edges.add(pair)
            chain.append(previous)
        paths[node] = tuple(reversed(chain))
    return MappingProxyType(paths), frozenset(tree_edges), len(paths) == len(adjacency)


def _fresh_trace(
    atlas: CanonicalAtlas,
    chart: CanonicalChart,
    path: tuple[int, ...],
    continuation: ContinuationAdapter,
) -> PathTrace:
    selected = chart.selection_by_node
    candidates = atlas.product_graph.candidate_by_key
    nodes = atlas.product_graph.node_by_id
    if not path or any(node_id not in selected for node_id in path):
        return PathTrace((), np.zeros((0, 6)), False, "unknown_chart_node")
    beta_rows = [candidates[selected[path[0]]].beta_rad]
    current = candidates[selected[path[0]]]
    for source_node, target_node in zip(path, path[1:]):
        if current.node_id != source_node:
            return PathTrace(
                tuple(path[: len(beta_rows)]),
                np.vstack(beta_rows),
                False,
                "fresh_source_node_mismatch",
            )
        try:
            raw = continuation(current, nodes[target_node])
            outcome = _coerce_outcome(raw)
        except Exception as error:
            return PathTrace(
                tuple(path[: len(beta_rows)]),
                np.vstack(beta_rows),
                False,
                f"fresh_continuation_exception:{type(error).__name__}",
            )
        target = candidates[selected[target_node]]
        if not (
            outcome.success
            and outcome.actual_bounds
            and outcome.residual_mm
            <= atlas.policy.continuation_residual_max_mm + 1.0e-12
        ):
            return PathTrace(
                tuple(path[: len(beta_rows)]),
                np.vstack(beta_rows),
                False,
                f"fresh_continuation_failed:{outcome.status}",
            )
        if beta_rms_deg(outcome.beta_rad, target.beta_rad) > atlas.policy.edge_match_deg + 1.0e-12:
            return PathTrace(
                tuple(path[: len(beta_rows)]),
                np.vstack(beta_rows),
                False,
                "fresh_target_candidate_mismatch",
            )
        beta_rows.append(outcome.beta_rad)
        current = replace(target, beta_rad=outcome.beta_rad)
    return PathTrace(tuple(path), np.vstack(beta_rows), True, "fresh_continuation")


def _coerce_outcome(value: Any) -> ContinuationOutcome:
    if isinstance(value, ContinuationOutcome):
        return value
    if isinstance(value, Mapping):
        return ContinuationOutcome(
            beta_rad=value["beta_rad"],
            residual_mm=float(value["residual_mm"]),
            success=bool(value["success"]),
            actual_bounds=bool(value["actual_bounds"]),
            corrector_iterations=int(value.get("corrector_iterations", 0)),
            status=str(value.get("status", "mapping")),
            target_cluster_id=value.get("target_cluster_id"),
            minimum_margin_deg=value.get("minimum_margin_deg"),
            waypoint_count=int(value.get("waypoint_count", 1)),
        )
    raise TypeError("continuation must return ContinuationOutcome or mapping")


def _complete_trace(trace: PathTrace, path: tuple[int, ...]) -> bool:
    return bool(
        trace.success
        and trace.node_ids == tuple(path)
        and len(trace.beta_rad_by_node) == len(path)
    )


def _first_execution(values: Sequence[_FreshExecution]) -> _FreshExecution:
    matches = [value for value in values if value.repetition == 0]
    if len(matches) != 1:
        raise RuntimeError("every fresh schedule must contain exactly one first execution")
    return matches[0]


def _two_executions(values: Sequence[_FreshExecution]) -> tuple[_FreshExecution, _FreshExecution]:
    ordered = tuple(sorted(values, key=lambda item: item.repetition))
    if tuple(item.repetition for item in ordered) != (0, 1):
        raise RuntimeError("every fresh schedule must contain repetitions zero and one")
    return ordered[0], ordered[1]


def _metric(
    values: Sequence[float], *, p95_limit: float, max_limit: float, missing: int
) -> AuditMetric:
    data = np.asarray(values, dtype=float)
    if len(data) == 0 or int(missing) > 0:
        p95 = float(np.percentile(data, 95)) if len(data) else math.inf
        maximum = float(np.max(data)) if len(data) else math.inf
        return AuditMetric(len(data), p95, maximum, False, int(missing))
    p95 = float(np.percentile(data, 95))
    maximum = float(np.max(data))
    return AuditMetric(
        sample_count=len(data),
        p95_deg=p95,
        max_deg=maximum,
        gate_pass=bool(
            p95 <= float(p95_limit) + 1.0e-12
            and maximum <= float(max_limit) + 1.0e-12
        ),
        missing_count=0,
    )


def _section_evidence(
    atlas: CanonicalAtlas,
    audits: Mapping[int, FreshChartAudit],
    section_id_by_chart: Mapping[int, str],
    probe_id_by_chart_node: Mapping[tuple[int, int], str],
    audit_policy: AtlasAuditPolicy,
) -> tuple[SectionEvidence, ...]:
    output: list[SectionEvidence] = []
    for chart in atlas.charts:
        selected = chart.selection_by_node
        audit = audits[chart.chart_id]
        selected_edges: list[tuple[str, str]] = []
        for edge in atlas.product_graph.robust_edges:
            left, right = edge.left_key[0], edge.right_key[0]
            if selected.get(left) != edge.left_key or selected.get(right) != edge.right_key:
                continue
            selected_edges.append(
                (
                    probe_id_by_chart_node[(chart.chart_id, left)],
                    probe_id_by_chart_node[(chart.chart_id, right)],
                )
            )
        output.append(
            SectionEvidence(
                section_id=section_id_by_chart[chart.chart_id],
                selections=tuple(
                    SectionSelection(
                        probe_id_by_chart_node[(chart.chart_id, node_id)],
                        candidate_id,
                    )
                    for node_id, candidate_id in chart.selections
                ),
                # The two fields are intentionally identical: every task edge
                # admitted to a section was first found in the selected robust
                # product graph; the adapter never promotes a raw task edge.
                task_edges=tuple(selected_edges),
                robust_edges=tuple(selected_edges),
                cycle_p95_deg=_finite_section_metric(audit.loop, audit_policy.common_max_deg),
                cycle_max_deg=_finite_section_metric_max(audit.loop, audit_policy.common_max_deg),
                multipath_p95_deg=_finite_section_metric(
                    audit.multipath,
                    audit_policy.common_max_deg,
                ),
                multipath_max_deg=_finite_section_metric_max(
                    audit.multipath,
                    audit_policy.common_max_deg,
                ),
                full_fundamental_closure_checked=audit.full_fundamental_closure_checked,
            )
        )
    return tuple(output)


def _finite_section_metric(metric: AuditMetric, common_max_deg: float) -> float:
    return (
        float(metric.p95_deg)
        if math.isfinite(metric.p95_deg)
        else float(common_max_deg) * 10.0
    )


def _finite_section_metric_max(metric: AuditMetric, common_max_deg: float) -> float:
    return (
        float(metric.max_deg)
        if math.isfinite(metric.max_deg)
        else float(common_max_deg) * 10.0
    )


def _overlap_evidence(
    atlas: CanonicalAtlas,
    *,
    contexts: Mapping[int, _TaskContext],
    chart_audits: Mapping[int, FreshChartAudit],
    section_id_by_chart: Mapping[int, str],
) -> tuple[OverlapEvidence, ...]:
    candidates = atlas.product_graph.candidate_by_key
    output: list[OverlapEvidence] = []
    for left_index, left_chart in enumerate(atlas.charts):
        left_selection = left_chart.selection_by_node
        for right_chart in atlas.charts[left_index + 1 :]:
            right_selection = right_chart.selection_by_node
            shared = tuple(sorted(set(left_selection) & set(right_selection)))
            if not shared:
                continue
            probe_ids = tuple(contexts[node_id].physical_point_id for node_id in shared)
            gaps = tuple(
                beta_rms_deg(
                    candidates[left_selection[node_id]].beta_rad,
                    candidates[right_selection[node_id]].beta_rad,
                )
                for node_id in shared
            )
            left_audit = chart_audits[left_chart.chart_id]
            right_audit = chart_audits[right_chart.chart_id]
            output.append(
                OverlapEvidence(
                    left_section_id=section_id_by_chart[left_chart.chart_id],
                    right_section_id=section_id_by_chart[right_chart.chart_id],
                    shared_physical_probe_ids=probe_ids,
                    beta_gaps_deg=gaps,
                    cycle_gate_pass=bool(
                        left_audit.full_fundamental_closure_checked
                        and right_audit.full_fundamental_closure_checked
                        and left_audit.loop.gate_pass
                        and right_audit.loop.gate_pass
                        and left_audit.multipath.gate_pass
                        and right_audit.multipath.gate_pass
                    ),
                    repeatable=bool(
                        left_audit.repeat.gate_pass and right_audit.repeat.gate_pass
                    ),
                )
            )
    return tuple(output)


def _assess_overlap(
    evidence: OverlapEvidence, policy: WorkspaceAtlasPolicy
) -> OverlapAssessment:
    values = np.asarray(evidence.beta_gaps_deg, dtype=float)
    p50 = float(np.percentile(values, 50))
    p95 = float(np.percentile(values, 95))
    maximum = float(np.max(values))
    if (
        evidence.cycle_gate_pass
        and p95 <= policy.stitchable_p95_deg + 1.0e-12
        and maximum <= policy.stitchable_max_deg + 1.0e-12
    ):
        kind = OverlapKind.STITCHABLE
    elif evidence.repeatable and p95 > policy.nonstitchable_p95_deg:
        kind = OverlapKind.NON_STITCHABLE
    else:
        kind = OverlapKind.INDETERMINATE
    return OverlapAssessment(
        left_section_id=evidence.left_section_id,
        right_section_id=evidence.right_section_id,
        beta_gap_p50_deg=p50,
        beta_gap_p95_deg=p95,
        beta_gap_max_deg=maximum,
        kind=kind,
    )


def _primary_partition(
    nodes: Sequence[AtlasTaskNode],
    atlas: CanonicalAtlas,
    audits: Mapping[int, FreshChartAudit],
    overlaps: Sequence[OverlapAssessment],
    section_id_by_chart: Mapping[int, str],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], Mapping[int, Mapping[str, Any]]]:
    """Assign whole spatial components, never individual FK-preferred nodes."""

    component_by_node = _task_components(nodes)
    nodes_by_component: dict[int, set[int]] = defaultdict(set)
    for node_id, component_id in component_by_node.items():
        nodes_by_component[component_id].add(node_id)
    chart_nodes = {
        chart.chart_id: set(chart.selection_by_node)
        for chart in atlas.charts
        if audits[chart.chart_id].section_valid
    }
    kind_by_pair = {
        _ordered_section_pair(item.left_section_id, item.right_section_id): item.kind
        for item in overlaps
    }
    primary: set[str] = set()
    required: set[tuple[str, str]] = set()
    state_by_component: dict[int, Mapping[str, Any]] = {}
    for component_id, component_nodes in sorted(nodes_by_component.items()):
        covering = tuple(
            sorted(
                chart_id
                for chart_id, chart_nodes_set in chart_nodes.items()
                if component_nodes <= chart_nodes_set
            )
        )
        if not covering:
            state_by_component[component_id] = MappingProxyType(
                {
                    "kind": "uncovered_component",
                    "assigned_section_id": None,
                    "candidate_section_ids": (),
                }
            )
            continue
        section_ids = tuple(section_id_by_chart[chart_id] for chart_id in covering)
        if len(section_ids) == 1:
            primary.add(section_ids[0])
            state_by_component[component_id] = MappingProxyType(
                {
                    "kind": "single_section_component",
                    "assigned_section_id": section_ids[0],
                    "candidate_section_ids": section_ids,
                }
            )
            continue
        pairs = tuple(
            _ordered_section_pair(left, right)
            for position, left in enumerate(section_ids)
            for right in section_ids[position + 1 :]
        )
        if any(pair not in kind_by_pair for pair in pairs) or any(
            kind_by_pair.get(pair) is OverlapKind.INDETERMINATE for pair in pairs
        ):
            state_by_component[component_id] = MappingProxyType(
                {
                    "kind": "indeterminate_overlap_component",
                    "assigned_section_id": None,
                    "candidate_section_ids": section_ids,
                }
            )
            continue
        if any(kind_by_pair[pair] is OverlapKind.NON_STITCHABLE for pair in pairs):
            primary.update(section_ids)
            required.update(pairs)
            state_by_component[component_id] = MappingProxyType(
                {
                    "kind": "stateful_multichart_component",
                    "assigned_section_id": None,
                    "candidate_section_ids": section_ids,
                }
            )
            continue
        # The chart ID is a global canonical rank.  Selecting it once for the
        # complete spatial component avoids a local residual/FK tie-break.
        assigned = min(section_ids)
        primary.add(assigned)
        state_by_component[component_id] = MappingProxyType(
            {
                "kind": "stitchable_global_section_component",
                "assigned_section_id": assigned,
                "candidate_section_ids": section_ids,
            }
        )
    by_node = {
        node_id: state_by_component[component_id]
        for node_id, component_id in component_by_node.items()
    }
    return (
        tuple(sorted(primary)),
        tuple(sorted(required)),
        MappingProxyType(by_node),
    )


def _task_components(nodes: Sequence[AtlasTaskNode]) -> Mapping[int, int]:
    adjacency: dict[int, set[int]] = {node.node_id: set() for node in nodes}
    for node in nodes:
        for neighbor in node.neighbor_node_ids:
            if neighbor in adjacency:
                adjacency[node.node_id].add(neighbor)
                adjacency[neighbor].add(node.node_id)
    component: dict[int, int] = {}
    for root in sorted(adjacency):
        if root in component:
            continue
        component_id = len(set(component.values()))
        pending: deque[int] = deque([root])
        while pending:
            current = pending.popleft()
            if current in component:
                continue
            component[current] = component_id
            pending.extend(sorted(adjacency[current] - set(component)))
    return MappingProxyType(component)


def _cell_evidence(
    contexts: Sequence[_TaskContext],
    atlas: CanonicalAtlas,
    candidates: Mapping[tuple[int, str], AtlasCandidate],
    section_id_by_chart: Mapping[int, str],
    probe_id_by_chart_node: Mapping[tuple[int, int], str],
) -> tuple[tuple[TaskCellEvidence, ...], tuple[Mapping[str, Any], ...]]:
    selected_charts_by_node: dict[int, list[CanonicalChart]] = defaultdict(list)
    for chart in atlas.charts:
        for node_id in chart.selection_by_node:
            selected_charts_by_node[node_id].append(chart)
    candidate_count_by_node: dict[int, int] = defaultdict(int)
    for candidate in candidates.values():
        candidate_count_by_node[candidate.node_id] += 1
    context_by_node = {context.node.node_id: context for context in contexts}
    probes_by_cell: dict[CellKey, list[CellProbeEvidence]] = defaultdict(list)
    probe_rows: list[Mapping[str, Any]] = []
    probe_ids_by_node: dict[int, list[str]] = defaultdict(list)
    for context in contexts:
        node_id = context.node.node_id
        charts = tuple(sorted(selected_charts_by_node.get(node_id, ()), key=lambda item: item.chart_id))
        inverse_status = (
            InverseStatus.TEACHER_UNRESOLVED
            if not charts
            else InverseStatus.RESOLVED_SINGLE_UNDER_BUDGET
            if len(charts) == 1
            else InverseStatus.RESOLVED_MULTICHART
        )
        if not charts:
            probe_id = f"{context.task_id}:unresolved"
            probe = CellProbeEvidence(
                probe_id=probe_id,
                physical_point_id=context.physical_point_id,
                cell=context.cell,
                xyz_m=context.node.xyz_m,
                is_measure_probe=context.is_measure_probe,
                labelable=False,
                inverse_status=inverse_status,
                physical_status=PhysicalStatus.UNKNOWN,
                candidate_family_count=candidate_count_by_node[node_id],
                risk_flags=context.risk_flags,
            )
            probes_by_cell[context.cell].append(probe)
            probe_ids_by_node[node_id].append(probe_id)
            probe_rows.append(_probe_row(probe, node_id, None))
            continue
        for chart in charts:
            candidate = candidates[chart.selection_by_node[node_id]]
            labelable = context.physical_status is not PhysicalStatus.PHYSICALLY_INVALID
            probe = CellProbeEvidence(
                probe_id=probe_id_by_chart_node[(chart.chart_id, node_id)],
                physical_point_id=context.physical_point_id,
                cell=context.cell,
                xyz_m=context.node.xyz_m,
                is_measure_probe=context.is_measure_probe,
                labelable=labelable,
                inverse_status=inverse_status,
                physical_status=context.physical_status,
                selected_candidate_id=candidate.candidate_id if labelable else None,
                chart_id=section_id_by_chart[chart.chart_id] if labelable else None,
                candidate_family_count=candidate_count_by_node[node_id],
                risk_flags=context.risk_flags,
            )
            probes_by_cell[context.cell].append(probe)
            probe_ids_by_node[node_id].append(probe.probe_id)
            probe_rows.append(_probe_row(probe, node_id, chart.chart_id))

    contexts_by_cell: dict[CellKey, list[_TaskContext]] = defaultdict(list)
    for context in contexts:
        contexts_by_cell[context.cell].append(context)
    cells: list[TaskCellEvidence] = []
    for cell, cell_contexts in sorted(contexts_by_cell.items()):
        probes = list(probes_by_cell[cell])
        measure_complete = bool(
            cell_contexts and all(context.cell_measure_complete for context in cell_contexts)
        )
        has_measure = any(probe.is_measure_probe for probe in probes)
        if not measure_complete or not has_measure:
            # This is not an invented positive sample.  It is an explicit,
            # unlabelled coverage witness which prevents a representative-only
            # graph node from being promoted to CellResolution.FULL.
            xyz = np.mean(
                np.vstack([context.node.xyz_m for context in cell_contexts]),
                axis=0,
            )
            coverage_probe = CellProbeEvidence(
                probe_id=(
                    f"cell:{cell.level_mm}:{cell.ix}:{cell.iy}:{cell.iz}:"
                    "coverage_unverified"
                ),
                physical_point_id=(
                    f"cell:{cell.level_mm}:{cell.ix}:{cell.iy}:{cell.iz}:"
                    "coverage_unverified"
                ),
                cell=cell,
                xyz_m=xyz,
                is_measure_probe=True,
                labelable=False,
                inverse_status=InverseStatus.TEACHER_UNRESOLVED,
                physical_status=PhysicalStatus.UNKNOWN,
            )
            probes.append(coverage_probe)
            probe_rows.append(_probe_row(coverage_probe, None, None))
        representatives = [
            context for context in cell_contexts if context.is_representative
        ]
        if len(representatives) > 1:
            raise ValueError("a workspace cell cannot name multiple representatives")
        representative_context = (
            representatives[0]
            if representatives
            else min(cell_contexts, key=lambda item: item.node.node_id)
        )
        representative_probe_ids = probe_ids_by_node[representative_context.node.node_id]
        if not representative_probe_ids:
            raise RuntimeError("representative task node did not produce a task probe")
        cells.append(
            TaskCellEvidence(
                cell=cell,
                representative_probe_id=min(representative_probe_ids),
                probes=tuple(sorted(probes, key=lambda item: item.probe_id)),
            )
        )
    all_probe_ids = [row["task_probe_id"] for row in probe_rows]
    if len(set(all_probe_ids)) != len(all_probe_ids):
        raise RuntimeError("task probes must use globally unique task IDs")
    return tuple(cells), tuple(probe_rows)


def _probe_row(
    probe: CellProbeEvidence, node_id: int | None, chart_id: int | None
) -> Mapping[str, Any]:
    return {
        "task_probe_id": probe.probe_id,
        "task_node_id": None if node_id is None else int(node_id),
        "chart_numeric_id": None if chart_id is None else int(chart_id),
        "physical_point_id": probe.physical_point_id,
        "cell_level_mm": probe.cell.level_mm,
        "cell_ix": probe.cell.ix,
        "cell_iy": probe.cell.iy,
        "cell_iz": probe.cell.iz,
        "x_m": float(probe.xyz_m[0]),
        "y_m": float(probe.xyz_m[1]),
        "z_m": float(probe.xyz_m[2]),
        "is_measure_probe": bool(probe.is_measure_probe),
        "labelable": bool(probe.labelable),
        "inverse_status": probe.inverse_status.value,
        "physical_status": probe.physical_status.value,
        "selected_candidate_id": probe.selected_candidate_id,
        "chart_id": probe.chart_id,
        "candidate_family_count": int(probe.candidate_family_count),
        "risk_flags": sorted(flag.value for flag in probe.risk_flags),
    }


def _aggregate_audit_report(
    atlas: CanonicalAtlas,
    chart_audits: Sequence[FreshChartAudit],
    overlaps: Sequence[OverlapAssessment],
    primary_section_ids: Sequence[str],
) -> AtlasAuditReport:
    if chart_audits:
        path = _aggregate_metric(item.path for item in chart_audits)
        loop = _aggregate_metric(item.loop for item in chart_audits)
        direction = _aggregate_metric(item.direction for item in chart_audits)
        repeat = _aggregate_metric(item.repeat for item in chart_audits)
        multipath = _aggregate_metric(item.multipath for item in chart_audits)
    else:
        missing = AuditMetric(0, math.inf, math.inf, False, 1)
        path = loop = direction = repeat = multipath = missing
    if not overlaps:
        overlap = AuditMetric(0, 0.0, 0.0, True, 0)
        multi_chart_overlap_valid = True
    else:
        gaps = [
            value
            for item in overlaps
            for value in (item.beta_gap_p95_deg, item.beta_gap_max_deg)
        ]
        overlap = _metric(
            gaps,
            p95_limit=atlas.policy.merge_overlap_p95_deg,
            max_limit=atlas.policy.merge_overlap_max_deg,
            missing=0,
        )
        multi_chart_overlap_valid = bool(
            all(item.kind is not OverlapKind.INDETERMINATE for item in overlaps)
        )
    stability = bool(
        path.gate_pass
        and loop.gate_pass
        and direction.gate_pass
        and repeat.gate_pass
        and multipath.gate_pass
    )
    primary = tuple(primary_section_ids)
    if not atlas.charts or not primary:
        representation = "reject_insufficient_atlas_evidence"
    elif any(item.kind is OverlapKind.NON_STITCHABLE for item in overlaps):
        representation = (
            "stateful_xyz_beta_prev_to_delta_beta"
            if stability
            else "reject_insufficient_atlas_evidence"
        )
    elif len(primary) == 1:
        representation = (
            "static_xyz_to_beta6" if stability else "reject_insufficient_atlas_evidence"
        )
    else:
        representation = (
            "multi_chart_xyz_chart_id_to_beta6"
            if stability and multi_chart_overlap_valid
            else "reject_insufficient_atlas_evidence"
        )
    gate_pass = bool(
        stability
        and bool(primary)
        and (
            overlap.gate_pass
            or (len(atlas.charts) > 1 and multi_chart_overlap_valid)
        )
        and representation != "reject_insufficient_atlas_evidence"
    )
    checks: dict[str, Mapping[str, Any]] = {
        "path": path.as_dict(),
        "loop": loop.as_dict(),
        "direction": direction.as_dict(),
        "repeat": repeat.as_dict(),
        "multipath": multipath.as_dict(),
        "overlap": overlap.as_dict(),
        "multi_chart_overlap_valid": {"gate_pass": multi_chart_overlap_valid},
        "fundamental_cycle_scope": {
            "gate_pass": bool(
                chart_audits
                and all(item.full_fundamental_closure_checked for item in chart_audits)
            ),
            "chart_count": len(chart_audits),
            "fundamental_cycle_count": int(
                sum(item.fundamental_cycle_count for item in chart_audits)
            ),
            "fresh_execution_count": int(
                sum(item.fresh_execution_count for item in chart_audits)
            ),
            "scope": "all_selected_robust_edges_and_all_fundamental_cycles",
        },
        "representation_decision": {"value": representation},
    }
    return AtlasAuditReport(
        path=path,
        loop=loop,
        direction=direction,
        repeat=repeat,
        overlap=overlap,
        multi_chart_overlap_valid=multi_chart_overlap_valid,
        representation_decision=representation,
        gate_pass=gate_pass,
        endpoint_count=sum(item.selected_task_edge_count for item in chart_audits),
        loop_count=sum(item.fundamental_cycle_count for item in chart_audits),
        evidence_limitations=(),
        checks=checks,
    )


def _aggregate_metric(metrics: Sequence[AuditMetric]) -> AuditMetric:
    rows = tuple(metrics)
    count = sum(item.sample_count for item in rows)
    missing = sum(item.missing_count for item in rows)
    p95 = max((item.p95_deg for item in rows), default=math.inf)
    maximum = max((item.max_deg for item in rows), default=math.inf)
    return AuditMetric(
        sample_count=count,
        p95_deg=p95,
        max_deg=maximum,
        gate_pass=bool(rows and missing == 0 and all(item.gate_pass for item in rows)),
        missing_count=missing,
    )


def _result_frames(
    *,
    contexts: Sequence[_TaskContext],
    candidates: pd.DataFrame,
    atlas: CanonicalAtlas,
    executions: Sequence[_FreshExecution],
    chart_audits: Sequence[FreshChartAudit],
    sections: Sequence[SectionEvidence],
    overlaps: Sequence[OverlapAssessment],
    partition_by_node: Mapping[int, Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    workspace_result: WorkspaceAtlasResult,
) -> Mapping[str, pd.DataFrame]:
    task_frame = pd.DataFrame.from_records(
        [
            {
                "task_node_id": context.node.node_id,
                "task_id": context.task_id,
                "physical_point_id": context.physical_point_id,
                "x_m": float(context.node.xyz_m[0]),
                "y_m": float(context.node.xyz_m[1]),
                "z_m": float(context.node.xyz_m[2]),
                "neighbor_node_ids": list(context.node.neighbor_node_ids),
                "core_safe": bool(context.node.core_safe),
                "cell_level_mm": context.cell.level_mm,
                "cell_ix": context.cell.ix,
                "cell_iy": context.cell.iy,
                "cell_iz": context.cell.iz,
                "is_measure_probe": bool(context.is_measure_probe),
                "is_representative": bool(context.is_representative),
                "cell_measure_complete": bool(context.cell_measure_complete),
            }
            for context in contexts
        ]
    )
    product_edges = pd.DataFrame.from_records(
        [
            {
                "left_task_node_id": edge.left_key[0],
                "left_candidate_id": edge.left_key[1],
                "right_task_node_id": edge.right_key[0],
                "right_candidate_id": edge.right_key[1],
                "cost": float(edge.cost),
                "transition_deg": float(edge.transition_deg),
                "forward_status": edge.forward.status,
                "reverse_status": edge.reverse.status,
            }
            for edge in atlas.product_graph.robust_edges
        ]
    )
    audit_tasks = pd.DataFrame.from_records(
        [
            {
                "task_id": execution.task.task_id,
                "chart_id": execution.task.chart_id,
                "kind": execution.task.kind,
                "schedule_id": execution.schedule_id,
                "repetition": execution.repetition,
                "path": list(execution.task.path),
                "success": bool(execution.trace.success),
                "status": execution.trace.status,
                "executed_node_count": len(execution.trace.node_ids),
            }
            for execution in executions
        ]
    )
    section_by_id = {section.section_id: section for section in sections}
    audit_by_chart = {item.chart_id: item for item in chart_audits}
    section_frame = pd.DataFrame.from_records(
        [
            {
                "section_id": section.section_id,
                "chart_id": chart_id,
                "selection_count": len(section.selections),
                "task_edge_count": len(section.task_edges),
                "robust_edge_count": len(section.robust_edges),
                "cycle_p95_deg": float(section.cycle_p95_deg),
                "cycle_max_deg": float(section.cycle_max_deg),
                "multipath_p95_deg": float(section.multipath_p95_deg),
                "multipath_max_deg": float(section.multipath_max_deg),
                "full_fundamental_closure_checked": bool(
                    section.full_fundamental_closure_checked
                ),
                "section_valid": bool(audit_by_chart[chart_id].section_valid),
            }
            for chart_id, section in sorted(
                (
                    chart.chart_id,
                    section_by_id[_section_id(chart.chart_id)],
                )
                for chart in atlas.charts
            )
        ]
    )
    overlap_frame = pd.DataFrame.from_records(
        [
            {
                "left_section_id": item.left_section_id,
                "right_section_id": item.right_section_id,
                "beta_gap_p50_deg": float(item.beta_gap_p50_deg),
                "beta_gap_p95_deg": float(item.beta_gap_p95_deg),
                "beta_gap_max_deg": float(item.beta_gap_max_deg),
                "kind": item.kind.value,
            }
            for item in overlaps
        ]
    )
    partition_frame = pd.DataFrame.from_records(
        [
            {
                "task_node_id": node_id,
                "partition_kind": state["kind"],
                "assigned_section_id": state["assigned_section_id"],
                "candidate_section_ids": list(state["candidate_section_ids"]),
            }
            for node_id, state in sorted(partition_by_node.items())
        ]
    )
    cell_frame = pd.DataFrame.from_records(
        [
            {
                "cell_level_mm": assessment.cell.level_mm,
                "cell_ix": assessment.cell.ix,
                "cell_iy": assessment.cell.iy,
                "cell_iz": assessment.cell.iz,
                "representative_status": assessment.representative_status.value,
                "empirical_labelable_fraction": float(
                    assessment.empirical_labelable_fraction
                ),
                "probe_count": assessment.probe_count,
                "labelable_probe_count": assessment.labelable_probe_count,
                "resolution": assessment.resolution.value,
                "risk_flags": sorted(flag.value for flag in assessment.risk_flags),
            }
            for assessment in workspace_result.cell_assessments
        ]
    )
    return MappingProxyType(
        {
            "task_nodes": task_frame,
            "candidates": candidates.copy(),
            "product_edges": product_edges,
            "audit_tasks": audit_tasks,
            "sections": section_frame,
            "overlaps": overlap_frame,
            "primary_partition": partition_frame,
            "task_probes": pd.DataFrame.from_records(probe_rows),
            "cells": cell_frame,
        }
    )


def _integration_report(
    *,
    canonical_atlas: CanonicalAtlas,
    chart_audits: Sequence[FreshChartAudit],
    audit_report: AtlasAuditReport,
    workspace_result: WorkspaceAtlasResult,
    cells: Sequence[TaskCellEvidence],
    primary_section_ids: Sequence[str],
    required_transitions: Sequence[tuple[str, str]],
    execution_count: int,
) -> Mapping[str, Any]:
    full_cells = bool(cells) and all(
        item.resolution is CellResolution.FULL
        for item in workspace_result.cell_assessments
    )
    representative_only_cells = sum(
        any(probe.probe_id.endswith("coverage_unverified") for probe in cell.probes)
        for cell in cells
    )
    deployment_claim = bool(
        audit_report.gate_pass
        and workspace_result.representation.static_inverse_authorized
        and workspace_result.branch_saturation.gate_pass
        and full_cells
        and representative_only_cells == 0
    )
    return MappingProxyType(
        _json_safe(
            {
                "schema_version": 1,
                "claim_scope": "finite_workspace_atlas_integration",
                "canonical_chart_count": len(canonical_atlas.charts),
                "product_graph": {
                    "task_node_count": len(canonical_atlas.product_graph.task_nodes),
                    "candidate_count": len(canonical_atlas.product_graph.candidates),
                    "directed_edge_count": len(
                        canonical_atlas.product_graph.directed_edges
                    ),
                    "robust_edge_count": len(canonical_atlas.product_graph.robust_edges),
                    "continuation_attempt_count": int(
                        canonical_atlas.product_graph.continuation_attempt_count
                    ),
                    "rejected_continuation_count": int(
                        canonical_atlas.product_graph.rejected_continuation_count
                    ),
                },
                "fresh_audit": {
                    "execution_count": int(execution_count),
                    "scope": "all_selected_robust_edges_and_all_fundamental_cycles",
                    "charts": [item.as_dict() for item in chart_audits],
                    "aggregate": {
                        "gate_pass": bool(audit_report.gate_pass),
                        "checks": {
                            name: dict(value)
                            for name, value in audit_report.checks.items()
                        },
                    },
                },
                "primary_section_ids": list(primary_section_ids),
                "required_transitions": [list(item) for item in required_transitions],
                "workspace_representation": {
                    "mode": workspace_result.representation.mode.value,
                    "static_inverse_authorized": bool(
                        workspace_result.representation.static_inverse_authorized
                    ),
                    "stateful_inverse_authorized": bool(
                        workspace_result.representation.stateful_inverse_authorized
                    ),
                    "reasons": list(workspace_result.representation.reasons),
                },
                "cell_coverage": {
                    "full_cell_count": sum(
                        item.resolution is CellResolution.FULL
                        for item in workspace_result.cell_assessments
                    ),
                    "representative_only_or_incomplete_cell_count": representative_only_cells,
                    "full_cell_status_asserted_from_representative_alone": False,
                },
                "deployment_claim_gate_pass": deployment_claim,
                "evidence_limitations": [
                    "representative task nodes without explicit complete cell-measure evidence are retained as partial/unresolved cells",
                    "branch saturation remains fail-closed unless BranchAuditObservation evidence is supplied",
                ],
            }
        )
    )


def _require_frame(value: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    if value.empty:
        raise ValueError(f"{label} must be non-empty")
    return value.copy()


def _first_column(frame: pd.DataFrame, names: Sequence[str], label: str) -> str:
    column = _optional_column(frame, names)
    if column is None:
        raise ValueError(f"{label} is required; accepted columns are {list(names)}")
    return column


def _optional_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _xyz_columns(frame: pd.DataFrame, label: str) -> tuple[str, str, str]:
    for layout in _TASK_XYZ_LAYOUTS:
        if all(column in frame.columns for column in layout):
            return layout
    raise ValueError(
        f"{label} needs one XYZ layout: "
        + " or ".join(", ".join(layout) for layout in _TASK_XYZ_LAYOUTS)
    )


def _beta_getter(frame: pd.DataFrame, label: str) -> Any:
    vector_column = _optional_column(frame, ("beta_rad", "beta"))
    if vector_column is not None:
        def vector_getter(row: pd.Series) -> np.ndarray:
            try:
                return np.asarray(row[vector_column], dtype=float).reshape(6)
            except Exception as error:
                raise ValueError(f"{label}.{vector_column} must contain six beta values") from error

        return vector_getter
    for layout in _BETA_LAYOUTS:
        if all(column in frame.columns for column in layout):
            return lambda row, columns=layout: np.asarray(
                [row[column] for column in columns], dtype=float
            ).reshape(6)
    raise ValueError(f"{label} needs beta_rad/beta or six beta*_rad columns")


def _cell_from_row(
    row: pd.Series, xyz_m: np.ndarray, *, default_level_mm: int
) -> CellKey:
    if "cell" in row.index and not _is_missing(row["cell"]):
        value = row["cell"]
        if isinstance(value, CellKey):
            return value
        if isinstance(value, Mapping):
            return CellKey(
                _integer(value.get("level_mm"), "cell.level_mm"),
                _integer(value.get("ix"), "cell.ix"),
                _integer(value.get("iy"), "cell.iy"),
                _integer(value.get("iz"), "cell.iz"),
            )
        values = _sequence(value)
        if len(values) == 4:
            return CellKey(
                _integer(values[0], "cell level_mm"),
                _integer(values[1], "cell ix"),
                _integer(values[2], "cell iy"),
                _integer(values[3], "cell iz"),
            )
        raise ValueError("cell must be a CellKey, mapping, or four-value sequence")
    level_column = _optional_series_column(row, ("cell_level_mm", "level_mm"))
    ix_column = _optional_series_column(row, ("cell_ix", "ix"))
    iy_column = _optional_series_column(row, ("cell_iy", "iy"))
    iz_column = _optional_series_column(row, ("cell_iz", "iz"))
    supplied = (level_column, ix_column, iy_column, iz_column)
    if any(column is not None for column in supplied):
        if any(column is None for column in supplied):
            raise ValueError("explicit cell identity requires level_mm and ix/iy/iz")
        return CellKey(
            _integer(row[level_column], "cell level_mm"),
            _integer(row[ix_column], "cell ix"),
            _integer(row[iy_column], "cell iy"),
            _integer(row[iz_column], "cell iz"),
        )
    level = int(default_level_mm)
    step = level / 1000.0
    indices = np.floor(np.asarray(xyz_m, dtype=float) / step).astype(int)
    return CellKey(level, int(indices[0]), int(indices[1]), int(indices[2]))


def _optional_series_column(row: pd.Series, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in row.index), None)


def _environment_bounds(environment: Any) -> np.ndarray:
    if not hasattr(environment, "bounds"):
        raise TypeError("environment must expose authoritative bounds")
    bounds = np.asarray(environment.bounds, dtype=float)
    if (
        bounds.shape != (6, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("environment.bounds must be finite ordered shape (6, 2)")
    return bounds.copy()


def _integer(value: Any, label: str) -> int:
    if _is_missing(value):
        raise ValueError(f"{label} cannot be missing")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return integer
    if not math.isfinite(numeric) or not math.isclose(numeric, integer, abs_tol=1.0e-12):
        raise ValueError(f"{label} must be an integer")
    return integer


def _identifier(value: Any, label: str) -> str:
    if _is_missing(value):
        raise ValueError(f"{label} cannot be missing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} cannot be empty")
    return text


def _finite_value(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _finite_nonnegative(value: Any, label: str) -> float:
    numeric = _finite_value(value, label)
    if numeric < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return numeric


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, tuple, list, dict, np.ndarray, CellKey)):
        return False
    try:
        result = pd.isna(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _sequence(value: Any) -> tuple[Any, ...]:
    if _is_missing(value):
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError("string sequence must be JSON or comma-separated") from error
            return _sequence(decoded)
        return tuple(piece.strip() for piece in text.split(",") if piece.strip())
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _as_bool(value: Any, *, default: bool, label: str) -> bool:
    if _is_missing(value):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{label} must be boolean")


def _quality(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw).strip().lower()
    mapping = {"gold": "Gold", "silver": "Silver", "reject": "Reject"}
    if text not in mapping:
        raise ValueError(f"unknown candidate quality {value!r}")
    return mapping[text]


def _physical_status(value: Any) -> PhysicalStatus:
    raw = getattr(value, "value", value)
    try:
        return PhysicalStatus(str(raw))
    except ValueError as error:
        raise ValueError(f"unknown physical_status {value!r}") from error


def _risk_flags(value: Any) -> frozenset[RiskFlag]:
    values: list[Any] = []
    for item in _sequence(value):
        if isinstance(item, str) and "|" in item:
            values.extend(piece.strip() for piece in item.split("|") if piece.strip())
        else:
            values.append(item)
    try:
        return frozenset(RiskFlag(getattr(item, "value", item)) for item in values)
    except ValueError as error:
        raise ValueError(f"unknown risk flag in {value!r}") from error


def _section_id(chart_id: int) -> str:
    return f"chart_{int(chart_id):02d}"


def _chart_probe_id(task_id: str, chart_id: int) -> str:
    return f"{task_id}:chart_{int(chart_id):03d}"


def _ordered_section_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("section transition cannot be a self edge")
    return (left, right) if left < right else (right, left)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (PhysicalStatus, InverseStatus, OverlapKind, RiskFlag, CellResolution)):
        return value.value
    return value
