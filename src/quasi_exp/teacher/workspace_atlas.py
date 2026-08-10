"""Cell evidence, inverse sections, and representation gates for BACRA V14.

The module treats a product-graph component as evidence only.  A deployable
chart must provide an explicit one-candidate-per-task-probe section and pass
the registered connectivity, edge, cycle, and overlap checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .workspace_reach import CellKey


class InverseStatus(str, Enum):
    RESOLVED_SINGLE_UNDER_BUDGET = "resolved_single_under_budget"
    RESOLVED_MULTICHART = "resolved_multichart"
    TEACHER_UNRESOLVED = "teacher_unresolved"


class PhysicalStatus(str, Enum):
    VALID = "valid"
    PHYSICALLY_INVALID = "physically_invalid"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    LOW_MARGIN = "low_margin"
    ILL_CONDITIONED = "ill_conditioned"


class CellResolution(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class OverlapKind(str, Enum):
    STITCHABLE = "stitchable"
    NON_STITCHABLE = "non_stitchable"
    INDETERMINATE = "indeterminate"


class RepresentationMode(str, Enum):
    XYZ_GLOBAL = "xyz_global"
    XYZ_ROUTER_EXPERTS = "xyz_router_experts"
    STATEFUL_ROUTER_EXPERTS = "stateful_router_experts"
    BLOCKED = "blocked"


def normalized_jacobian(jacobian: np.ndarray, beta_bounds: np.ndarray) -> np.ndarray:
    """Return d(x)/d(beta_normalized) using each registered full beta span."""

    jac = np.asarray(jacobian, dtype=float)
    bounds = np.asarray(beta_bounds, dtype=float)
    if jac.shape != (3, 6) or not np.isfinite(jac).all():
        raise ValueError("jacobian must be finite shape (3, 6)")
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all():
        raise ValueError("beta_bounds must be finite shape (6, 2)")
    span = bounds[:, 1] - bounds[:, 0]
    if np.any(span <= 0.0):
        raise ValueError("beta bounds must have positive span")
    return jac @ np.diag(span)


def normalized_jacobian_metrics(
    jacobian: np.ndarray, beta_bounds: np.ndarray
) -> tuple[float, float]:
    singular = np.linalg.svd(
        normalized_jacobian(jacobian, beta_bounds), compute_uv=False
    )
    sigma3 = float(singular[-1])
    kappa = float(np.inf if sigma3 <= 0.0 else singular[0] / sigma3)
    return sigma3, kappa


@dataclass(frozen=True)
class CellProbeEvidence:
    probe_id: str
    physical_point_id: str
    cell: CellKey
    xyz_m: np.ndarray
    is_measure_probe: bool
    labelable: bool
    inverse_status: InverseStatus
    physical_status: PhysicalStatus
    counts_toward_primary_measure: bool = True
    selected_candidate_id: str | None = None
    chart_id: str | None = None
    candidate_family_count: int = 0
    risk_flags: frozenset[RiskFlag] = frozenset()

    def __post_init__(self) -> None:
        probe_id = str(self.probe_id).strip()
        physical_id = str(self.physical_point_id).strip()
        xyz = np.asarray(self.xyz_m, dtype=float).reshape(3)
        if not probe_id or not physical_id or not np.isfinite(xyz).all():
            raise ValueError("probe IDs must be non-empty and xyz_m finite")
        inverse_status = InverseStatus(self.inverse_status)
        physical_status = PhysicalStatus(self.physical_status)
        flags = frozenset(RiskFlag(value) for value in self.risk_flags)
        if int(self.candidate_family_count) < 0:
            raise ValueError("candidate_family_count must be non-negative")
        if self.labelable and not self.selected_candidate_id:
            raise ValueError("labelable probes require selected_candidate_id")
        if self.labelable and not self.chart_id:
            raise ValueError("labelable probes require chart_id")
        if not self.labelable and self.selected_candidate_id is not None:
            raise ValueError("unlabelable probes cannot select a candidate")
        object.__setattr__(self, "probe_id", probe_id)
        object.__setattr__(self, "physical_point_id", physical_id)
        object.__setattr__(self, "xyz_m", xyz.copy())
        object.__setattr__(self, "is_measure_probe", bool(self.is_measure_probe))
        object.__setattr__(
            self,
            "counts_toward_primary_measure",
            bool(self.counts_toward_primary_measure),
        )
        object.__setattr__(self, "labelable", bool(self.labelable))
        object.__setattr__(self, "inverse_status", inverse_status)
        object.__setattr__(self, "physical_status", physical_status)
        object.__setattr__(self, "candidate_family_count", int(self.candidate_family_count))
        object.__setattr__(self, "risk_flags", flags)


@dataclass(frozen=True)
class TaskCellEvidence:
    cell: CellKey
    representative_probe_id: str
    probes: tuple[CellProbeEvidence, ...]

    def __post_init__(self) -> None:
        representative = str(self.representative_probe_id).strip()
        probes = tuple(self.probes)
        if not representative or not probes:
            raise ValueError("task cell requires probes and a representative")
        if len({probe.probe_id for probe in probes}) != len(probes):
            raise ValueError("task cell probe IDs must be unique")
        if any(probe.cell != self.cell for probe in probes):
            raise ValueError("all probes must belong to the task cell")
        if representative not in {probe.probe_id for probe in probes}:
            raise ValueError("representative probe must belong to the task cell")
        if not any(probe.is_measure_probe for probe in probes):
            raise ValueError("task cell requires at least one measure probe")
        object.__setattr__(self, "representative_probe_id", representative)
        object.__setattr__(self, "probes", probes)


@dataclass(frozen=True)
class CellAssessment:
    cell: CellKey
    representative_status: InverseStatus
    empirical_labelable_fraction: float
    probe_count: int
    labelable_probe_count: int
    resolution: CellResolution
    risk_flags: frozenset[RiskFlag]
    labelable_measure_m3: float
    x_center_m: float


@dataclass(frozen=True)
class SectionSelection:
    task_probe_id: str
    candidate_id: str

    def __post_init__(self) -> None:
        probe = str(self.task_probe_id).strip()
        candidate = str(self.candidate_id).strip()
        if not probe or not candidate:
            raise ValueError("section selection IDs must be non-empty")
        object.__setattr__(self, "task_probe_id", probe)
        object.__setattr__(self, "candidate_id", candidate)


def _ordered_edge(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("task edges cannot be self edges")
    return (left, right) if left < right else (right, left)


@dataclass(frozen=True)
class SectionEvidence:
    section_id: str
    selections: tuple[SectionSelection, ...]
    task_edges: tuple[tuple[str, str], ...]
    robust_edges: tuple[tuple[str, str], ...]
    cycle_p95_deg: float
    cycle_max_deg: float
    multipath_p95_deg: float
    multipath_max_deg: float
    full_fundamental_closure_checked: bool

    def __post_init__(self) -> None:
        section_id = str(self.section_id).strip()
        if not section_id or not self.selections:
            raise ValueError("section requires an ID and selections")
        scalars = (
            self.cycle_p95_deg,
            self.cycle_max_deg,
            self.multipath_p95_deg,
            self.multipath_max_deg,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in scalars):
            raise ValueError("section audit metrics must be finite and non-negative")
        task_edges = tuple(_ordered_edge(*edge) for edge in self.task_edges)
        robust_edges = tuple(_ordered_edge(*edge) for edge in self.robust_edges)
        object.__setattr__(self, "section_id", section_id)
        object.__setattr__(self, "task_edges", task_edges)
        object.__setattr__(self, "robust_edges", robust_edges)
        object.__setattr__(
            self,
            "full_fundamental_closure_checked",
            bool(self.full_fundamental_closure_checked),
        )


@dataclass(frozen=True)
class SectionAssessment:
    section_id: str
    probe_ids: frozenset[str]
    cell_ids: frozenset[CellKey]
    valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OverlapEvidence:
    left_section_id: str
    right_section_id: str
    shared_physical_probe_ids: tuple[str, ...]
    beta_gaps_deg: tuple[float, ...]
    cycle_gate_pass: bool
    repeatable: bool

    def __post_init__(self) -> None:
        left = str(self.left_section_id).strip()
        right = str(self.right_section_id).strip()
        shared = tuple(str(value).strip() for value in self.shared_physical_probe_ids)
        gaps = tuple(float(value) for value in self.beta_gaps_deg)
        if not left or not right or left == right:
            raise ValueError("overlap requires two distinct section IDs")
        if not shared or not gaps or len(shared) != len(gaps):
            raise ValueError("overlap probes and beta gaps must be non-empty and aligned")
        if any(not value for value in shared) or any(
            not math.isfinite(value) or value < 0.0 for value in gaps
        ):
            raise ValueError("overlap evidence must be finite")
        object.__setattr__(self, "left_section_id", left)
        object.__setattr__(self, "right_section_id", right)
        object.__setattr__(self, "shared_physical_probe_ids", shared)
        object.__setattr__(self, "beta_gaps_deg", gaps)
        object.__setattr__(self, "cycle_gate_pass", bool(self.cycle_gate_pass))
        object.__setattr__(self, "repeatable", bool(self.repeatable))


@dataclass(frozen=True)
class OverlapAssessment:
    left_section_id: str
    right_section_id: str
    beta_gap_p50_deg: float
    beta_gap_p95_deg: float
    beta_gap_max_deg: float
    kind: OverlapKind


@dataclass(frozen=True)
class BranchAuditObservation:
    cell: CellKey
    stratum: str
    new_stable_branch: bool

    def __post_init__(self) -> None:
        stratum = str(self.stratum).strip()
        if not stratum:
            raise ValueError("branch audit stratum must be non-empty")
        object.__setattr__(self, "stratum", stratum)
        object.__setattr__(self, "new_stable_branch", bool(self.new_stable_branch))


@dataclass(frozen=True)
class BranchSaturationReport:
    sample_count: int
    new_branch_count: int
    observed_rate: float
    overall_wilson_upper: float
    stratum_wilson_upper: Mapping[str, float]
    gate_pass: bool


@dataclass(frozen=True)
class RepresentationDecision:
    mode: RepresentationMode
    primary_section_ids: tuple[str, ...]
    measure_coverage: float
    minimum_x_bin_coverage: float
    abstention_measure_ratio: float
    static_inverse_authorized: bool
    stateful_inverse_authorized: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class WorkspaceAtlasInput:
    cells: tuple[TaskCellEvidence, ...] = ()
    sections: tuple[SectionEvidence, ...] = ()
    overlaps: tuple[OverlapEvidence, ...] = ()
    branch_audit: tuple[BranchAuditObservation, ...] = ()
    primary_section_ids: tuple[str, ...] = ()
    required_transitions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class WorkspaceAtlasPolicy:
    stitchable_p95_deg: float = 0.5
    stitchable_max_deg: float = 1.0
    nonstitchable_p95_deg: float = 1.0
    cycle_p95_deg: float = 0.5
    cycle_max_deg: float = 1.0
    minimum_primary_measure_coverage: float = 0.95
    minimum_x_bin_coverage: float = 0.90
    maximum_abstention_measure_ratio: float = 0.05
    x_slab_low_m: float = 1.015498
    x_bin_width_m: float = 0.020
    branch_overall_wilson_upper: float = 0.02
    branch_stratum_wilson_upper: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.stitchable_p95_deg,
            self.stitchable_max_deg,
            self.nonstitchable_p95_deg,
            self.cycle_p95_deg,
            self.cycle_max_deg,
            self.x_bin_width_m,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise ValueError("atlas numerical thresholds must be finite and positive")
        fractions = (
            self.minimum_primary_measure_coverage,
            self.minimum_x_bin_coverage,
            self.maximum_abstention_measure_ratio,
            self.branch_overall_wilson_upper,
            self.branch_stratum_wilson_upper,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in fractions):
            raise ValueError("atlas coverage thresholds must be fractions")


@dataclass(frozen=True)
class WorkspaceAtlasResult:
    cell_assessments: tuple[CellAssessment, ...]
    section_assessments: tuple[SectionAssessment, ...]
    overlap_assessments: tuple[OverlapAssessment, ...]
    branch_saturation: BranchSaturationReport
    representation: RepresentationDecision


def _wilson_upper(successes: int, total: int, *, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float((center + radius) / denominator)


class WorkspaceAtlasBuilder:
    """Validate cell evidence and decide the deployable inverse representation."""

    def __init__(self, policy: WorkspaceAtlasPolicy | None = None) -> None:
        self.policy = WorkspaceAtlasPolicy() if policy is None else policy

    def build(self, evidence: WorkspaceAtlasInput) -> WorkspaceAtlasResult:
        cell_assessments, probes = self._assess_cells(evidence.cells)
        section_assessments = self._assess_sections(evidence.sections, probes)
        overlap_assessments = tuple(
            self._assess_overlap(item) for item in evidence.overlaps
        )
        saturation = self._assess_branch_saturation(evidence.branch_audit)
        representation = self._decide_representation(
            cell_assessments,
            section_assessments,
            overlap_assessments,
            evidence.primary_section_ids,
            evidence.required_transitions,
        )
        return WorkspaceAtlasResult(
            cell_assessments=cell_assessments,
            section_assessments=section_assessments,
            overlap_assessments=overlap_assessments,
            branch_saturation=saturation,
            representation=representation,
        )

    def _assess_cells(
        self, cells: Sequence[TaskCellEvidence]
    ) -> tuple[tuple[CellAssessment, ...], Mapping[str, CellProbeEvidence]]:
        if len({cell.cell for cell in cells}) != len(cells):
            raise ValueError("workspace task cells must be unique")
        probe_by_id: dict[str, CellProbeEvidence] = {}
        assessments: list[CellAssessment] = []
        for item in cells:
            for probe in item.probes:
                if probe.probe_id in probe_by_id:
                    raise ValueError("task probe IDs must be globally unique")
                probe_by_id[probe.probe_id] = probe
            representative = probe_by_id[item.representative_probe_id]
            measure = tuple(
                probe
                for probe in item.probes
                if probe.is_measure_probe and probe.counts_toward_primary_measure
            )
            if not measure:
                raise ValueError("task cell requires a primary-measure probe witness")
            accepted = sum(probe.labelable for probe in measure)
            fraction = float(accepted / len(measure))
            chart_ids = {probe.chart_id for probe in measure if probe.labelable}
            if accepted == 0:
                resolution = CellResolution.UNRESOLVED
            elif accepted == len(measure) and len(chart_ids) == 1:
                resolution = CellResolution.FULL
            else:
                resolution = CellResolution.PARTIAL
            flags = frozenset(
                flag for probe in item.probes for flag in probe.risk_flags
            )
            x_center = float(np.mean([probe.xyz_m[0] for probe in measure]))
            assessments.append(
                CellAssessment(
                    cell=item.cell,
                    representative_status=representative.inverse_status,
                    empirical_labelable_fraction=fraction,
                    probe_count=len(measure),
                    labelable_probe_count=accepted,
                    resolution=resolution,
                    risk_flags=flags,
                    labelable_measure_m3=item.cell.volume_m3 * fraction,
                    x_center_m=x_center,
                )
            )
        return tuple(sorted(assessments, key=lambda row: row.cell)), MappingProxyType(probe_by_id)

    def _assess_sections(
        self,
        sections: Sequence[SectionEvidence],
        probes: Mapping[str, CellProbeEvidence],
    ) -> tuple[SectionAssessment, ...]:
        if len({section.section_id for section in sections}) != len(sections):
            raise ValueError("section IDs must be unique")
        result: list[SectionAssessment] = []
        for section in sections:
            selection_ids = [row.task_probe_id for row in section.selections]
            if len(set(selection_ids)) != len(selection_ids):
                raise ValueError(
                    f"section {section.section_id} must select exactly one candidate per task probe"
                )
            reasons: list[str] = []
            selected = set(selection_ids)
            unknown = selected - set(probes)
            if unknown:
                reasons.append("unknown_task_probe")
            for row in section.selections:
                probe = probes.get(row.task_probe_id)
                if probe is None:
                    continue
                if not probe.labelable or probe.selected_candidate_id != row.candidate_id:
                    reasons.append("selection_not_labelable_or_not_registered")
            task_edges = set(section.task_edges)
            robust_edges = set(section.robust_edges)
            if not task_edges <= robust_edges:
                reasons.append("selected_task_edge_without_robust_product_edge")
            if selected and not self._connected(selected, task_edges):
                reasons.append("section_task_domain_not_connected")
            if not section.full_fundamental_closure_checked:
                reasons.append("fundamental_closure_not_checked")
            if (
                section.cycle_p95_deg > self.policy.cycle_p95_deg
                or section.cycle_max_deg > self.policy.cycle_max_deg
                or section.multipath_p95_deg > self.policy.cycle_p95_deg
                or section.multipath_max_deg > self.policy.cycle_max_deg
            ):
                reasons.append("cycle_or_multipath_gap")
            cell_ids = frozenset(
                probes[probe_id].cell for probe_id in selected if probe_id in probes
            )
            result.append(
                SectionAssessment(
                    section_id=section.section_id,
                    probe_ids=frozenset(selected),
                    cell_ids=cell_ids,
                    valid=not reasons,
                    reasons=tuple(sorted(set(reasons))),
                )
            )
        return tuple(result)

    @staticmethod
    def _connected(nodes: set[str], edges: set[tuple[str, str]]) -> bool:
        if len(nodes) <= 1:
            return True
        adjacency: dict[str, set[str]] = {node: set() for node in nodes}
        for left, right in edges:
            if left in nodes and right in nodes:
                adjacency[left].add(right)
                adjacency[right].add(left)
        reached: set[str] = set()
        pending: deque[str] = deque([min(nodes)])
        while pending:
            current = pending.popleft()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(sorted(adjacency[current] - reached))
        return reached == nodes

    def _assess_overlap(self, evidence: OverlapEvidence) -> OverlapAssessment:
        gaps = np.asarray(evidence.beta_gaps_deg, dtype=float)
        p50 = float(np.percentile(gaps, 50))
        p95 = float(np.percentile(gaps, 95))
        maximum = float(np.max(gaps))
        if (
            evidence.cycle_gate_pass
            and p95 <= self.policy.stitchable_p95_deg
            and maximum <= self.policy.stitchable_max_deg
        ):
            kind = OverlapKind.STITCHABLE
        elif evidence.repeatable and p95 > self.policy.nonstitchable_p95_deg:
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

    def _assess_branch_saturation(
        self, observations: Sequence[BranchAuditObservation]
    ) -> BranchSaturationReport:
        rows = tuple(observations)
        total = len(rows)
        new_count = sum(row.new_stable_branch for row in rows)
        overall_upper = _wilson_upper(new_count, total)
        by_stratum: dict[str, list[BranchAuditObservation]] = {}
        for row in rows:
            by_stratum.setdefault(row.stratum, []).append(row)
        stratum_upper = {
            key: _wilson_upper(
                sum(row.new_stable_branch for row in values), len(values)
            )
            for key, values in sorted(by_stratum.items())
        }
        gate = bool(
            total > 0
            and overall_upper <= self.policy.branch_overall_wilson_upper
            and all(
                value <= self.policy.branch_stratum_wilson_upper
                for value in stratum_upper.values()
            )
        )
        return BranchSaturationReport(
            sample_count=total,
            new_branch_count=int(new_count),
            observed_rate=float(new_count / total) if total else 0.0,
            overall_wilson_upper=overall_upper,
            stratum_wilson_upper=MappingProxyType(stratum_upper),
            gate_pass=gate,
        )

    def _decide_representation(
        self,
        cells: Sequence[CellAssessment],
        sections: Sequence[SectionAssessment],
        overlaps: Sequence[OverlapAssessment],
        primary_section_ids: Sequence[str],
        required_transitions: Sequence[tuple[str, str]],
    ) -> RepresentationDecision:
        primary_ids = tuple(str(value) for value in primary_section_ids)
        if not cells or not sections or not primary_ids:
            return self._blocked(primary_ids, "missing_cells_sections_or_primary_section")
        by_section = {section.section_id: section for section in sections}
        if any(section_id not in by_section for section_id in primary_ids):
            return self._blocked(primary_ids, "unknown_primary_section")
        if any(not by_section[section_id].valid for section_id in primary_ids):
            return self._blocked(primary_ids, "invalid_primary_section")

        total_measure = sum(cell.labelable_measure_m3 for cell in cells)
        if total_measure <= 0.0:
            return self._blocked(primary_ids, "no_labelable_measure")
        covered_cells = frozenset(
            cell
            for section_id in primary_ids
            for cell in by_section[section_id].cell_ids
        )
        covered_measure = sum(
            cell.labelable_measure_m3 for cell in cells if cell.cell in covered_cells
        )
        coverage = float(covered_measure / total_measure)
        abstention = float(1.0 - coverage)

        bin_total: dict[int, float] = {}
        bin_covered: dict[int, float] = {}
        for cell in cells:
            bin_id = int(
                math.floor(
                    (cell.x_center_m - self.policy.x_slab_low_m)
                    / self.policy.x_bin_width_m
                )
            )
            bin_total[bin_id] = bin_total.get(bin_id, 0.0) + cell.labelable_measure_m3
            if cell.cell in covered_cells:
                bin_covered[bin_id] = (
                    bin_covered.get(bin_id, 0.0) + cell.labelable_measure_m3
                )
        bin_ratios = [
            bin_covered.get(bin_id, 0.0) / measure
            for bin_id, measure in bin_total.items()
            if measure > 0.0
        ]
        minimum_bin = float(min(bin_ratios)) if bin_ratios else 0.0
        coverage_pass = bool(
            coverage >= self.policy.minimum_primary_measure_coverage
            and minimum_bin >= self.policy.minimum_x_bin_coverage
            and abstention <= self.policy.maximum_abstention_measure_ratio
        )
        if not coverage_pass:
            return RepresentationDecision(
                mode=RepresentationMode.BLOCKED,
                primary_section_ids=primary_ids,
                measure_coverage=coverage,
                minimum_x_bin_coverage=minimum_bin,
                abstention_measure_ratio=abstention,
                static_inverse_authorized=False,
                stateful_inverse_authorized=False,
                reasons=("primary_measure_coverage_gate_failed",),
            )

        overlap_by_pair = {
            _ordered_edge(row.left_section_id, row.right_section_id): row
            for row in overlaps
        }
        required = tuple(_ordered_edge(*pair) for pair in required_transitions)
        missing = [pair for pair in required if pair not in overlap_by_pair]
        if missing:
            return RepresentationDecision(
                mode=RepresentationMode.BLOCKED,
                primary_section_ids=primary_ids,
                measure_coverage=coverage,
                minimum_x_bin_coverage=minimum_bin,
                abstention_measure_ratio=abstention,
                static_inverse_authorized=False,
                stateful_inverse_authorized=False,
                reasons=("required_overlap_evidence_missing",),
            )
        kinds = [overlap_by_pair[pair].kind for pair in required]
        if any(kind is OverlapKind.INDETERMINATE for kind in kinds):
            mode = RepresentationMode.BLOCKED
            static = stateful = False
            reasons = ("required_transition_indeterminate",)
        elif any(kind is OverlapKind.NON_STITCHABLE for kind in kinds):
            mode = RepresentationMode.STATEFUL_ROUTER_EXPERTS
            static = False
            stateful = True
            reasons = ("required_transition_non_stitchable",)
        elif len(primary_ids) == 1:
            mode = RepresentationMode.XYZ_GLOBAL
            static = True
            stateful = False
            reasons = ()
        else:
            mode = RepresentationMode.XYZ_ROUTER_EXPERTS
            static = True
            stateful = False
            reasons = ()
        return RepresentationDecision(
            mode=mode,
            primary_section_ids=primary_ids,
            measure_coverage=coverage,
            minimum_x_bin_coverage=minimum_bin,
            abstention_measure_ratio=abstention,
            static_inverse_authorized=static,
            stateful_inverse_authorized=stateful,
            reasons=reasons,
        )

    @staticmethod
    def _blocked(primary_ids: Sequence[str], reason: str) -> RepresentationDecision:
        return RepresentationDecision(
            mode=RepresentationMode.BLOCKED,
            primary_section_ids=tuple(primary_ids),
            measure_coverage=0.0,
            minimum_x_bin_coverage=0.0,
            abstention_measure_ratio=1.0,
            static_inverse_authorized=False,
            stateful_inverse_authorized=False,
            reasons=(reason,),
        )
