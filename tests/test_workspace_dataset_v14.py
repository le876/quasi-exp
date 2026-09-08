from __future__ import annotations

import numpy as np
import pytest

from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_dataset import (
    BudgetFeasibilityStatus,
    BudgetExceededError,
    FixedBudgetDemand,
    MacroblockSplitPolicy,
    SplitRole,
    SupervisionBudgetPolicy,
    SupervisionKind,
    SupervisionMaterializer,
    SupervisionPriority,
    SupervisionRecord,
    evaluate_fixed_budget_feasibility,
)
from quasi_exp.teacher.workspace_reach import CellKey


def _record(
    record_id: str,
    *,
    point_id: str | None = None,
    chart_id: str = "chart_00",
    cell: CellKey = CellKey(10, 101, 0, 0),
    beta0: float = 0.0,
    primary: bool = True,
    required: bool = False,
    priority: SupervisionPriority = SupervisionPriority.MAXIMIN,
) -> SupervisionRecord:
    return SupervisionRecord(
        record_id=record_id,
        kind=SupervisionKind.STATIC,
        physical_point_id=point_id or record_id,
        chart_id=chart_id,
        cell=cell,
        xyz_m=np.asarray([1.01, 0.001, 0.001]),
        beta_rad=np.asarray([beta0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        is_primary=primary,
        required=required,
        priority=priority,
        source_family="unit_test",
    )


def test_materializer_collapses_identical_reuse_but_rejects_conflicting_labels() -> None:
    policy = SupervisionBudgetPolicy(target_total=4, hard_max=6, active_reserve=0)
    materializer = SupervisionMaterializer(policy)
    first = _record("row_a", point_id="point", chart_id="chart")
    reused = _record("row_b", point_id="point", chart_id="chart")

    bundle = materializer.materialize(
        (first, reused), representation_mode=RepresentationMode.XYZ_GLOBAL
    )

    assert bundle.budget_report.unique_supervision_count == 1
    assert len(bundle.primary_canonical) == 1
    assert len(bundle.chart_expert) == 1

    conflicting = _record(
        "row_c", point_id="point", chart_id="chart", beta0=0.2
    )
    with pytest.raises(ValueError, match="conflicting supervision"):
        materializer.materialize(
            (first, conflicting), representation_mode=RepresentationMode.XYZ_GLOBAL
        )


def test_alternative_charts_share_macroblock_but_primary_view_stays_single_valued() -> None:
    materializer = SupervisionMaterializer(
        SupervisionBudgetPolicy(target_total=4, hard_max=6, active_reserve=0)
    )
    primary = _record("primary", point_id="point", chart_id="chart_a")
    alternative = _record(
        "alternative",
        point_id="point",
        chart_id="chart_b",
        beta0=0.01,
        primary=False,
    )

    bundle = materializer.materialize(
        (primary, alternative),
        representation_mode=RepresentationMode.XYZ_ROUTER_EXPERTS,
    )

    assert len(bundle.primary_canonical) == 1
    assert len(bundle.chart_expert) == 2
    roles = {row.split_role for row in bundle.chart_expert}
    macroblocks = {row.macroblock_id for row in bundle.chart_expert}
    assert len(roles) == 1
    assert len(macroblocks) == 1


def test_children_and_alternative_branches_inherit_one_40mm_macroblock_role() -> None:
    split = MacroblockSplitPolicy(macroblock_mm=40, seed=20260860)
    parent = CellKey(20, 50, -1, 0)
    children = (
        CellKey(10, 100, -2, 0),
        CellKey(10, 101, -1, 1),
        CellKey(5, 203, -1, 3),
    )

    parent_assignment = split.assignment_for_cell(parent)
    child_assignments = tuple(split.assignment_for_cell(cell) for cell in children)

    assert all(row.macroblock_id == parent_assignment.macroblock_id for row in child_assignments)
    assert all(row.split_role is parent_assignment.split_role for row in child_assignments)


def test_required_minimum_fails_before_consuming_active_reserve() -> None:
    materializer = SupervisionMaterializer(
        SupervisionBudgetPolicy(target_total=5, hard_max=6, active_reserve=2)
    )
    required = tuple(
        _record(f"required_{index}", required=True) for index in range(5)
    )

    with pytest.raises(BudgetExceededError, match="active reserve"):
        materializer.materialize(
            required, representation_mode=RepresentationMode.XYZ_GLOBAL
        )


def test_materializer_uses_priority_then_stable_id_without_duplicate_padding() -> None:
    materializer = SupervisionMaterializer(
        SupervisionBudgetPolicy(target_total=3, hard_max=5, active_reserve=0)
    )
    records = (
        _record("maximin", priority=SupervisionPriority.MAXIMIN),
        _record("base", priority=SupervisionPriority.BASE_COVERAGE),
        _record("refine", priority=SupervisionPriority.REFINEMENT),
        _record("retention", priority=SupervisionPriority.RETENTION),
    )

    bundle = materializer.materialize(
        records, representation_mode=RepresentationMode.XYZ_GLOBAL
    )

    assert [row.record_id for row in bundle.supervision_records] == [
        "base",
        "retention",
        "refine",
    ]
    assert bundle.budget_report.unique_supervision_count == 3
    assert bundle.budget_report.padding_count == 0


def test_stateful_budget_key_uses_directed_transition_and_branch() -> None:
    materializer = SupervisionMaterializer(
        SupervisionBudgetPolicy(target_total=4, hard_max=6, active_reserve=0)
    )
    common = dict(
        kind=SupervisionKind.STATEFUL,
        physical_point_id="target",
        chart_id="chart_a",
        cell=CellKey(10, 101, 0, 0),
        xyz_m=np.asarray([1.01, 0.0, 0.0]),
        beta_rad=np.zeros(6),
        previous_beta_rad=np.zeros(6),
        is_primary=False,
        required=True,
        priority=SupervisionPriority.BASE_COVERAGE,
        source_family="unit_test",
    )
    forward = SupervisionRecord(
        record_id="forward",
        source_probe_id="p0",
        target_probe_id="p1",
        branch_id="branch_a",
        **common,
    )
    reverse = SupervisionRecord(
        record_id="reverse",
        source_probe_id="p1",
        target_probe_id="p0",
        branch_id="branch_a",
        **common,
    )

    bundle = materializer.materialize(
        (forward, reverse),
        representation_mode=RepresentationMode.STATEFUL_ROUTER_EXPERTS,
    )

    assert len(bundle.stateful_transition) == 2
    assert bundle.budget_report.unique_supervision_count == 2
    assert {row.split_role for row in bundle.stateful_transition} <= set(SplitRole)


def test_budget_feasibility_counts_every_supervised_branch_and_refinement_row() -> None:
    demand = FixedBudgetDemand(
        labelable_base_rows=120_000,
        refinement_extra_rows=30_000,
        chart_expert_extra_rows=20_000,
        retention_rows=15_000,
        active_rows=15_000,
    )

    report = evaluate_fixed_budget_feasibility(demand, total_budget=200_000)

    assert report.minimum_supervised_rows == 200_000
    assert report.status is BudgetFeasibilityStatus.FITS_BASE_RESOLUTION
    assert report.gate_pass is True


def test_budget_infeasibility_never_silently_drops_domain_cells() -> None:
    demand = FixedBudgetDemand(
        labelable_base_rows=210_000,
        refinement_extra_rows=10_000,
        chart_expert_extra_rows=5_000,
        retention_rows=5_000,
        active_rows=0,
        coarsened_interior_rows=180_000,
    )
    coarsened = evaluate_fixed_budget_feasibility(demand, total_budget=200_000)
    assert coarsened.status is BudgetFeasibilityStatus.FITS_AFTER_INTERIOR_COARSENING
    assert coarsened.minimum_supervised_rows == 230_000
    assert coarsened.coarsened_minimum_rows == 200_000
    assert coarsened.gate_pass is True

    impossible = evaluate_fixed_budget_feasibility(
        FixedBudgetDemand(
            labelable_base_rows=210_000,
            refinement_extra_rows=10_000,
            chart_expert_extra_rows=5_000,
            retention_rows=5_000,
            active_rows=0,
            coarsened_interior_rows=195_000,
        ),
        total_budget=200_000,
    )
    assert impossible.status is BudgetFeasibilityStatus.INFEASIBLE_FIXED_BUDGET
    assert impossible.gate_pass is False


def test_unselected_active_probe_macroblock_cannot_leak_into_preactive_rows() -> None:
    split = MacroblockSplitPolicy(macroblock_mm=40, seed=20260860)
    active_cell = next(
        CellKey(10, ix, 0, 0)
        for ix in range(500)
        if split.assignment_for_cell(CellKey(10, ix, 0, 0)).split_role
        is SplitRole.ACTIVE_PROBE
    )
    active = _record("active_candidate", cell=active_cell)
    ordinary = _record("ordinary", cell=CellKey(10, 101, 0, 0), required=True)
    materializer = SupervisionMaterializer(
        SupervisionBudgetPolicy(target_total=2, hard_max=3, active_reserve=1), split
    )

    before = materializer.materialize(
        (ordinary, active), representation_mode=RepresentationMode.XYZ_GLOBAL
    )
    after = materializer.materialize(
        (ordinary, active),
        representation_mode=RepresentationMode.XYZ_GLOBAL,
        active_record_ids=(active.record_id,),
    )

    assert {row.record_id for row in before.supervision_records} == {"ordinary"}
    assert {row.record_id for row in after.supervision_records} == {
        "ordinary", "active_candidate"
    }
