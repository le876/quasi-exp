from __future__ import annotations

import math

import numpy as np
import pytest

from quasi_exp.teacher.workspace_atlas import (
    BranchAuditObservation,
    CellProbeEvidence,
    CellResolution,
    InverseStatus,
    OverlapEvidence,
    OverlapKind,
    PhysicalStatus,
    RepresentationMode,
    SectionEvidence,
    SectionSelection,
    TaskCellEvidence,
    WorkspaceAtlasBuilder,
    WorkspaceAtlasInput,
    WorkspaceAtlasPolicy,
    normalized_jacobian,
)
from quasi_exp.teacher.workspace_reach import CellKey


def _probe(
    probe_id: str,
    cell: CellKey,
    *,
    labelable: bool,
    measure: bool = True,
    chart_id: str | None = "chart_00",
) -> CellProbeEvidence:
    return CellProbeEvidence(
        probe_id=probe_id,
        physical_point_id=probe_id,
        cell=cell,
        xyz_m=np.asarray([1.01, 0.0, 0.0]),
        is_measure_probe=measure,
        labelable=labelable,
        inverse_status=(
            InverseStatus.RESOLVED_SINGLE_UNDER_BUDGET
            if labelable
            else InverseStatus.TEACHER_UNRESOLVED
        ),
        physical_status=PhysicalStatus.VALID if labelable else PhysicalStatus.UNKNOWN,
        selected_candidate_id=f"candidate_{probe_id}" if labelable else None,
        chart_id=chart_id if labelable else None,
    )


def _section(section_id: str, probe: CellProbeEvidence) -> SectionEvidence:
    return SectionEvidence(
        section_id=section_id,
        selections=(SectionSelection(probe.probe_id, probe.selected_candidate_id or "missing"),),
        task_edges=(),
        robust_edges=(),
        cycle_p95_deg=0.0,
        cycle_max_deg=0.0,
        multipath_p95_deg=0.0,
        multipath_max_deg=0.0,
        full_fundamental_closure_checked=True,
    )


def test_representative_success_cannot_mark_a_partially_labelable_cell_full() -> None:
    cell = CellKey(10, 101, 0, 0)
    evidence = TaskCellEvidence(
        cell=cell,
        representative_probe_id="center",
        probes=(
            _probe("center", cell, labelable=True, measure=False),
            _probe("child_a", cell, labelable=True),
            _probe("child_b", cell, labelable=False),
        ),
    )

    result = WorkspaceAtlasBuilder().build(WorkspaceAtlasInput(cells=(evidence,)))

    assessment = result.cell_assessments[0]
    assert assessment.representative_status is InverseStatus.RESOLVED_SINGLE_UNDER_BUDGET
    assert assessment.empirical_labelable_fraction == pytest.approx(0.5)
    assert assessment.resolution is CellResolution.PARTIAL
    assert result.representation.mode is RepresentationMode.BLOCKED


def test_section_rejects_multiple_candidates_for_one_task_probe() -> None:
    cell = CellKey(10, 101, 0, 0)
    probe = _probe("p0", cell, labelable=True)
    section = SectionEvidence(
        section_id="chart_00",
        selections=(
            SectionSelection("p0", "candidate_a"),
            SectionSelection("p0", "candidate_b"),
        ),
        task_edges=(),
        robust_edges=(),
        cycle_p95_deg=0.0,
        cycle_max_deg=0.0,
        multipath_p95_deg=0.0,
        multipath_max_deg=0.0,
        full_fundamental_closure_checked=True,
    )

    with pytest.raises(ValueError, match="exactly one candidate"):
        WorkspaceAtlasBuilder().build(
            WorkspaceAtlasInput(
                cells=(TaskCellEvidence(cell, "p0", (probe,)),),
                sections=(section,),
            )
        )


def test_stitchable_required_transition_authorizes_xyz_router() -> None:
    left_cell = CellKey(10, 101, 0, 0)
    right_cell = CellKey(10, 102, 0, 0)
    left = _probe("left", left_cell, labelable=True, chart_id="chart_a")
    right = _probe("right", right_cell, labelable=True, chart_id="chart_b")
    result = WorkspaceAtlasBuilder(
        WorkspaceAtlasPolicy(
            minimum_primary_measure_coverage=1.0,
            minimum_x_bin_coverage=1.0,
        )
    ).build(
        WorkspaceAtlasInput(
            cells=(
                TaskCellEvidence(left_cell, "left", (left,)),
                TaskCellEvidence(right_cell, "right", (right,)),
            ),
            sections=(_section("chart_a", left), _section("chart_b", right)),
            overlaps=(
                OverlapEvidence(
                    "chart_a",
                    "chart_b",
                    shared_physical_probe_ids=("overlap_0", "overlap_1"),
                    beta_gaps_deg=(0.2, 0.4),
                    cycle_gate_pass=True,
                    repeatable=True,
                ),
            ),
            primary_section_ids=("chart_a", "chart_b"),
            required_transitions=(("chart_a", "chart_b"),),
        )
    )

    assert result.overlap_assessments[0].kind is OverlapKind.STITCHABLE
    assert result.representation.mode is RepresentationMode.XYZ_ROUTER_EXPERTS
    assert result.representation.static_inverse_authorized is True


def test_required_nonstitchable_transition_selects_stateful_representation() -> None:
    left_cell = CellKey(10, 101, 0, 0)
    right_cell = CellKey(10, 102, 0, 0)
    left = _probe("left", left_cell, labelable=True, chart_id="chart_a")
    right = _probe("right", right_cell, labelable=True, chart_id="chart_b")
    result = WorkspaceAtlasBuilder().build(
        WorkspaceAtlasInput(
            cells=(
                TaskCellEvidence(left_cell, "left", (left,)),
                TaskCellEvidence(right_cell, "right", (right,)),
            ),
            sections=(_section("chart_a", left), _section("chart_b", right)),
            overlaps=(
                OverlapEvidence(
                    "chart_a",
                    "chart_b",
                    shared_physical_probe_ids=("overlap_0", "overlap_1"),
                    beta_gaps_deg=(2.0, 2.1),
                    cycle_gate_pass=True,
                    repeatable=True,
                ),
            ),
            primary_section_ids=("chart_a", "chart_b"),
            required_transitions=(("chart_a", "chart_b"),),
        )
    )

    assert result.overlap_assessments[0].kind is OverlapKind.NON_STITCHABLE
    assert result.representation.mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS
    assert result.representation.static_inverse_authorized is False
    assert result.representation.stateful_inverse_authorized is True


def test_branch_saturation_uses_one_sided_wilson_upper_bounds() -> None:
    cell = CellKey(10, 101, 0, 0)
    observations = tuple(
        BranchAuditObservation(
            cell=cell,
            stratum=f"stratum_{index % 5}",
            new_stable_branch=False,
        )
        for index in range(500)
    )
    passed = WorkspaceAtlasBuilder().build(
        WorkspaceAtlasInput(branch_audit=observations)
    )
    assert passed.branch_saturation.gate_pass is True
    assert passed.branch_saturation.overall_wilson_upper < 0.02

    failed = WorkspaceAtlasBuilder().build(
        WorkspaceAtlasInput(
            branch_audit=(
                BranchAuditObservation(cell, "tip", True),
                *tuple(BranchAuditObservation(cell, "tip", False) for _ in range(9)),
            )
        )
    )
    assert failed.branch_saturation.gate_pass is False


def test_normalized_jacobian_is_invariant_to_radian_or_degree_coordinates() -> None:
    jacobian_rad = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.5],
        ]
    )
    bounds_rad = np.tile(np.asarray([[-math.pi / 2.0, math.pi / 2.0]]), (6, 1))
    jacobian_deg = jacobian_rad * math.pi / 180.0
    bounds_deg = np.rad2deg(bounds_rad)

    assert np.allclose(
        normalized_jacobian(jacobian_rad, bounds_rad),
        normalized_jacobian(jacobian_deg, bounds_deg),
    )
