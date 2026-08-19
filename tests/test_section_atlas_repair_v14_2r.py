from __future__ import annotations

import numpy as np

from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    ContinuationOutcome,
)
from quasi_exp.teacher.canonical_gauge import CanonicalAnchorPolicy
from quasi_exp.teacher.section_atlas_repair import (
    AtlasRepairPolicy,
    AuditV2Policy,
    _primary_growth,
    compare_stitched_primary_atlases,
    diagnose_rooted_section_artifacts,
    execute_audit_schedules,
    repair_rooted_section_atlas,
    section_growth_from_frames,
)
from quasi_exp.teacher.section_first_atlas import RootedSectionPolicy, build_section_first_atlas


def _nodes(count: int = 10) -> tuple[AtlasTaskNode, ...]:
    return tuple(
        AtlasTaskNode(
            node_id=index,
            xyz_m=np.asarray([0.001 * index, 0.0, 0.0]),
            neighbor_node_ids=tuple(
                neighbor for neighbor in (index - 1, index + 1) if 0 <= neighbor < count
            ),
        )
        for index in range(count)
    )


def _grid_nodes() -> tuple[AtlasTaskNode, ...]:
    xyz = ((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.0, 0.001, 0.0), (0.001, 0.001, 0.0))
    neighbours = ((1, 2), (0, 3), (0, 3), (1, 2))
    return tuple(
        AtlasTaskNode(index, np.asarray(point), neighbours[index])
        for index, point in enumerate(xyz)
    )


def _candidate(node_id: int, branch_deg: float, name: str) -> AtlasCandidate:
    beta = np.zeros(6, dtype=float)
    beta[0] = 0.001 * node_id
    beta[5] = np.radians(branch_deg)
    return AtlasCandidate(
        node_id=node_id,
        candidate_id=name,
        beta_rad=beta,
        residual_mm=0.0,
        min_margin_deg=5.0,
        normalized_min_margin=0.5,
        condition_number=1.0,
    )


def _affine(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
    beta = source.beta_rad.copy()
    beta[0] = target.xyz_m[0]
    return ContinuationOutcome(beta, 0.0, True, True, 1, "affine", minimum_margin_deg=5.0)


def _growth(branches: tuple[float, ...], count: int = 10):
    nodes = _nodes(count)
    roots = tuple(_candidate(0, branch, f"root_{index}") for index, branch in enumerate(branches))
    return build_section_first_atlas(
        nodes,
        roots,
        _affine,
        root_keys=tuple(root.key for root in roots),
        policy=RootedSectionPolicy(root_count=len(roots), beam_width=2, maximum_growth_waves=16),
    )


def test_missing_solver_execution_is_not_a_geometry_failure_but_blocks_certificate() -> None:
    growth = _growth((0.0,))
    audit = diagnose_rooted_section_artifacts(
        growth,
        schedules=[
            {
                "schedule_id": "edge_0_1",
                "unique_entity_id": "edge:0:1",
                "chart_id": "chart_000",
                "audit_kind": "edge",
                "path_node_ids": [0, 1],
                "primary_usage": True,
            }
        ],
        executions=[
            {
                "schedule_id": "edge_0_1",
                "direction": "forward",
                "repeat_index": 0,
                "solver_success": False,
                "classification": "persistent_numerical",
            }
        ],
    )

    assert audit.geometry_gate is True
    assert audit.solver_gate is False
    assert audit.repeat_gate is True
    assert audit.certificate_gate is False
    assert audit.geometry_metrics["missing_count"] == 1


def test_canonical_root_priority_is_invariant_to_chart_order() -> None:
    first = _growth((0.0, 8.0))
    second = build_section_first_atlas(
        first.task_nodes,
        tuple(
            chart.selected_by_node[0].candidate
            for chart in reversed(first.charts)
        ),
        _affine,
        root_keys=tuple(chart.root_key for chart in reversed(first.charts)),
        policy=RootedSectionPolicy(root_count=2, beam_width=2, maximum_growth_waves=16),
    )
    anchor = first.charts[0].root_key
    left = repair_rooted_section_atlas(
        first,
        _affine,
        patch_id="left",
        method="anchor",
        canonical_root_priority=(anchor,),
    )
    right = repair_rooted_section_atlas(
        second,
        _affine,
        patch_id="right",
        method="anchor",
        canonical_root_priority=(anchor,),
    )
    comparison = compare_stitched_primary_atlases(left, right)

    assert left.growth.charts[int(left.selected_stitch_component[0].split("_")[-1])].root_key == anchor
    assert right.growth.charts[int(right.selected_stitch_component[0].split("_")[-1])].root_key == anchor
    assert comparison.gate_pass is True


def test_qualified_canonical_anchor_cannot_be_replaced_by_higher_coverage_branch() -> None:
    growth = _growth((0.0, 8.0), count=10)
    anchor_chart, alternative_chart = growth.charts
    object.__setattr__(
        anchor_chart,
        "selected_by_node",
        type(anchor_chart.selected_by_node)(
            {
                node: item
                for node, item in anchor_chart.selected_by_node.items()
                if node < 9
            }
        ),
    )
    anchor = anchor_chart.root_key

    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_anchor_lock",
        method="anchor_lock",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=8,
            abstention_hops=0,
        ),
        canonical_root_priority=(anchor,),
        canonical_anchor_policy=CanonicalAnchorPolicy(
            ordered_anchor_root_keys=(anchor,),
            minimum_component_coverage=0.9,
            minimum_coherent_measure=0.6,
            allow_stitchable_extensions=True,
            require_anchor_selected=True,
        ),
    )

    selected_roots = {
        growth.charts[int(chart_id.split("_")[-1])].root_key
        for chart_id in repaired.selected_stitch_component
    }
    assert anchor in selected_roots
    assert alternative_chart.root_key not in selected_roots
    assert repaired.canonical_anchor_component_selected is True
    assert repaired.canonical_anchor_root_key == anchor
    assert repaired.canonical_anchor_component_coverage == 0.9


def test_unqualified_required_anchor_blocks_primary_instead_of_silent_fallback() -> None:
    growth = _growth((0.0, 8.0), count=10)
    anchor_chart = growth.charts[0]
    object.__setattr__(
        anchor_chart,
        "selected_by_node",
        type(anchor_chart.selected_by_node)(
            {
                node: item
                for node, item in anchor_chart.selected_by_node.items()
                if node < 8
            }
        ),
    )
    anchor = anchor_chart.root_key

    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_anchor_required",
        method="anchor_required",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=8,
            abstention_hops=0,
        ),
        canonical_root_priority=(anchor,),
        canonical_anchor_policy=CanonicalAnchorPolicy(
            ordered_anchor_root_keys=(anchor,),
            minimum_component_coverage=0.9,
            minimum_coherent_measure=0.6,
            require_anchor_selected=True,
        ),
    )

    assert repaired.canonical_anchor_qualified is False
    assert repaired.canonical_anchor_component_selected is False
    assert repaired.canonical_anchor_fallback_reason == (
        "no_registered_anchor_component_qualified"
    )
    assert repaired.selected_stitch_component == ()
    assert repaired.certificate_gate is False


def test_repeat_only_failure_blocks_chart_and_certificate() -> None:
    growth = _growth((0.0,))
    audit = diagnose_rooted_section_artifacts(
        growth,
        schedules=[
            {
                "schedule_id": "edge_0_1",
                "unique_entity_id": "edge:0:1",
                "chart_id": "chart_000",
                "audit_kind": "edge",
                "path_node_ids": [0, 1],
                "primary_usage": True,
            }
        ],
        executions=[
            {
                "schedule_id": "edge_0_1",
                "direction": direction,
                "repeat_index": repeat_index,
                "solver_success": True,
                "residual_mm": 0.0,
                "geometry_gap_deg": 0.0,
                "repeat_gap_deg": 0.3,
                "classification": "connected",
            }
            for direction in ("forward", "reverse")
            for repeat_index in range(3)
        ],
        policy=AuditV2Policy(repeats_per_direction=3),
    )

    assert audit.geometry_gate is True
    assert audit.solver_gate is True
    assert audit.repeat_gate is False
    assert audit.certificate_gate is False
    assert audit.chart_gate_by_id["chart_000"] is False
    assert audit.repeat_metrics["p95_deg"] == 0.3


def test_repeat_only_failure_triggers_fragment_reaudit() -> None:
    growth = _growth((0.0,))
    phases: list[str] = []

    def executor(
        local_growth,
        schedules,
        continuation,
        audit_policy,
        retry_continuation,
        phase_id,
    ):
        phases.append(phase_id)
        rows = execute_audit_schedules(
            local_growth,
            schedules,
            continuation,
            audit_policy,
            retry_continuation=retry_continuation,
        )
        if phase_id == "chart_initial":
            failed_schedules = set(
                schedules.loc[schedules["audit_kind"].eq("edge"), "schedule_id"]
                .astype(str)
                .head(2)
            )
            rows.loc[
                rows["schedule_id"].astype(str).isin(failed_schedules),
                "repeat_gap_deg",
            ] = 0.3
        return rows

    repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_repeat_failure",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=3),
            minimum_chart_cells=2,
        ),
        schedule_executor=executor,
    )

    assert phases[0] == "chart_initial"
    assert "fragment_reaudit" in phases
    assert phases[-1] == "primary_certificate"


def test_unlocalized_long_path_failure_does_not_cut_every_path_edge() -> None:
    growth = _growth((0.0,))
    phases: list[str] = []

    def executor(
        local_growth,
        schedules,
        continuation,
        audit_policy,
        retry_continuation,
        phase_id,
    ):
        phases.append(phase_id)
        rows = execute_audit_schedules(
            local_growth,
            schedules,
            continuation,
            audit_policy,
            retry_continuation=retry_continuation,
        )
        if phase_id == "chart_initial":
            failed = set(
                schedules.loc[
                    schedules["audit_kind"].eq("root_path"), "schedule_id"
                ].astype(str)
            )
            rows.loc[rows["schedule_id"].astype(str).isin(failed), "geometry_gap_deg"] = 2.0
            rows.loc[rows["schedule_id"].astype(str).isin(failed), "classification"] = (
                "geometric_branch_disagreement"
            )
        return rows

    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_unlocalized_path",
        method="unlocalized_path",
        policy=AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=1)),
        schedule_executor=executor,
    )

    assert "fragment_reaudit" not in phases
    assert "fragment_screening" not in phases
    assert repaired.selected_stitch_component == ()
    assert repaired.certificate_gate is False


def test_two_nonstitchable_full_charts_choose_one_static_singleton_primary() -> None:
    growth = _growth((0.0, 8.0))
    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_test",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=8,
            minimum_stitch_overlap_cells=8,
        ),
    )

    assert repaired.certificate_gate is True
    assert repaired.abstained_node_ids == frozenset()
    assert len(set(repaired.primary_chart_by_node.values())) == 1
    assert len(repaired.selected_stitch_component) == 1


def test_stitchable_partial_charts_form_a_union_but_nonstitchable_gap_abstains() -> None:
    growth = _growth((0.0, 0.2), count=10)
    # Make complementary partial charts with a three-cell overlap so two
    # independent transitions and a non-vacuous cross-chart cycle exist.
    left, right = growth.charts
    object.__setattr__(left, "selected_by_node", type(left.selected_by_node)({
        node: item for node, item in left.selected_by_node.items() if node <= 6
    }))
    object.__setattr__(right, "selected_by_node", type(right.selected_by_node)({
        node: item for node, item in right.selected_by_node.items() if node >= 4
    }))

    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_test",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=4,
            minimum_stitch_overlap_cells=2,
        ),
    )
    assert set(repaired.selected_stitch_component) == {"chart_000", "chart_001"}
    assert repaired.abstained_node_ids == frozenset()
    optimization = repaired.frames["primary_optimization"]
    assert set(optimization["initialization"]) == {
        "coverage_ranked",
        "reverse_coverage_ranked",
        "unary_best",
    }
    assert optimization["selected"].astype(bool).sum() == 1

    far = _growth((0.0, 8.0), count=10)
    left, right = far.charts
    object.__setattr__(left, "selected_by_node", type(left.selected_by_node)({
        node: item for node, item in left.selected_by_node.items() if node <= 5
    }))
    object.__setattr__(right, "selected_by_node", type(right.selected_by_node)({
        node: item for node, item in right.selected_by_node.items() if node >= 4
    }))
    blocked = repair_rooted_section_atlas(
        far,
        _affine,
        patch_id="patch_test",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=4,
            minimum_stitch_overlap_cells=2,
            abstention_hops=1,
        ),
    )
    # The retained singleton can still be certified; deployment coverage is
    # separately fail-closed by the mechanism Gate.
    assert blocked.certificate_gate is True
    assert blocked.coverage_ratio < 1.0
    assert blocked.abstained_node_ids


def test_stability_compares_physical_beta_not_chart_identifier() -> None:
    first = repair_rooted_section_atlas(
        _growth((0.0,)),
        _affine,
        patch_id="patch_test",
        method="S4",
        policy=AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=1)),
    )
    second = repair_rooted_section_atlas(
        _growth((0.0,)),
        _affine,
        patch_id="patch_test",
        method="S8",
        policy=AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=1)),
    )
    second.primary_chart_by_node = {
        node: "renamed_chart" for node in second.primary_chart_by_node
    }

    stability = compare_stitched_primary_atlases(first, second)

    assert stability.gate_pass is True
    assert stability.beta_p95_deg == 0.0
    assert stability.beta_disagreement_ratio_gt_1deg == 0.0


def test_registered_retry_only_advances_failed_schedules_and_never_uses_oracle() -> None:
    growth = _growth((0.0,))
    calls: list[str] = []

    def retry(source: AtlasCandidate, target: AtlasTaskNode, tier) -> ContinuationOutcome:
        calls.append(tier.tier_id)
        if tier.tier_id == "R0":
            return ContinuationOutcome(source.beta_rad, 9.0, False, True, 1, "forced_r0_failure")
        return _affine(source, target)

    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_test",
        method="S4",
        policy=AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=1)),
        retry_continuation=retry,
    )

    executions = repaired.frames["audit_v2_executions"]
    assert set(executions["retry_tier"]) == {"R1"}
    assert set(executions["classification"]) == {"recoverable_numerical"}
    assert not executions["oracle_used_for_pass"].any()
    assert "R2" not in calls


def test_geometric_disagreement_advances_registered_kernel_without_oracle_seed() -> None:
    growth = _growth((0.0,))
    calls: list[str] = []

    def retry(source: AtlasCandidate, target: AtlasTaskNode, tier) -> ContinuationOutcome:
        calls.append(tier.tier_id)
        if tier.tier_id == "R0":
            beta = source.beta_rad.copy()
            beta[0] = target.xyz_m[0]
            beta[5] += np.radians(3.0)
            return ContinuationOutcome(beta, 0.0, True, True, 1, "wrong_fiber")
        return _affine(source, target)

    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_geometric_retry",
        method="geometry_retry",
        policy=AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=1)),
        retry_continuation=retry,
    )

    executions = repaired.frames["audit_v2_executions"]
    assert set(executions["retry_tier"]) == {"R1"}
    assert set(executions["classification"]) == {"recoverable_geometric"}
    assert not executions["oracle_used_for_pass"].any()
    assert "R0" in calls and "R1" in calls


def test_parquet_safe_section_frames_rehydrate_the_same_selected_beta() -> None:
    original = _growth((0.0, 0.2))
    frames = original.frames()
    restored = section_growth_from_frames(
        original.task_nodes,
        frames["section_hypotheses"],
        frames["selected_edges"],
        policy=original.policy,
    )

    assert len(restored.charts) == len(original.charts)
    for left, right in zip(original.charts, restored.charts, strict=True):
        assert set(left.selected_by_node) == set(right.selected_by_node)
        for node_id in left.selected_by_node:
            np.testing.assert_allclose(
                left.selected_by_node[node_id].candidate.beta_rad,
                right.selected_by_node[node_id].candidate.beta_rad,
            )
            left_candidate = left.selected_by_node[node_id].candidate
            right_candidate = right.selected_by_node[node_id].candidate
            assert left_candidate.residual_mm == right_candidate.residual_mm
            assert left_candidate.min_margin_deg == right_candidate.min_margin_deg
            assert (
                left_candidate.normalized_min_margin
                == right_candidate.normalized_min_margin
            )
            assert left_candidate.posture_cost == right_candidate.posture_cost
            assert left_candidate.condition_number == right_candidate.condition_number
            assert left_candidate.quality == right_candidate.quality


def test_empty_primary_section_frames_keep_the_rehydration_schema() -> None:
    original = _growth((0.0,))
    empty = _primary_growth(original, {}, original.chart_by_id)
    frames = empty.frames()

    restored = section_growth_from_frames(
        empty.task_nodes,
        frames["section_hypotheses"],
        frames["selected_edges"],
        policy=empty.policy,
    )

    assert frames["section_hypotheses"].empty
    assert restored.charts == ()
    assert restored.covered_node_ids == frozenset()


def test_persistent_failed_edge_splits_and_reaudits_nontrivial_fragments() -> None:
    growth = _growth((0.0,), count=10)

    def localized(source: AtlasCandidate, target: AtlasTaskNode, tier) -> ContinuationOutcome:
        if {source.node_id, target.node_id} == {4, 5}:
            return ContinuationOutcome(source.beta_rad, 9.0, False, True, 1, f"{tier.tier_id}_fail")
        return _affine(source, target)

    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_test",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=4,
            abstention_hops=0,
        ),
        retry_continuation=localized,
    )

    assert {chart.chart_id for chart in repaired.growth.charts} == {
        "chart_000__fragment_00",
        "chart_000__fragment_01",
    }
    assert repaired.diagnostic.solver_gate is True
    assert repaired.certificate_gate is True
    assert repaired.coverage_ratio == 0.5
    assert repaired.edge_completeness_ratio == 1.0
    retained = set(repaired.primary_beta_by_node)
    assert not ({4, 5} <= retained)


def test_primary_certificate_audits_the_complete_induced_cycle_basis() -> None:
    nodes = _grid_nodes()
    root = _candidate(0, 0.0, "root")
    growth = build_section_first_atlas(
        nodes,
        (root,),
        _affine,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(root_count=1, beam_width=2),
    )
    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_cycle",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=4,
        ),
    )

    assert repaired.induced_edge_count == 4
    assert repaired.audited_edge_count == 4
    assert repaired.edge_completeness_ratio == 1.0
    assert repaired.retained_cycle_rank == 1
    assert repaired.audited_fundamental_cycle_count == 1
    assert repaired.cycle_coverage_ratio == 1.0


def test_boundary_stitch_and_indeterminate_are_distinct_evidence_states() -> None:
    nodes = _grid_nodes()
    roots = (_candidate(0, 0.0, "left"), _candidate(0, 0.2, "right"))
    growth = build_section_first_atlas(
        nodes,
        roots,
        _affine,
        root_keys=tuple(root.key for root in roots),
        policy=RootedSectionPolicy(root_count=2, beam_width=2),
    )
    left, right = growth.charts
    object.__setattr__(left, "selected_by_node", type(left.selected_by_node)({
        node: item for node, item in left.selected_by_node.items() if node in {0, 2}
    }))
    object.__setattr__(right, "selected_by_node", type(right.selected_by_node)({
        node: item for node, item in right.selected_by_node.items() if node in {1, 3}
    }))
    repaired = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_boundary",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=2,
            minimum_stitch_overlap_cells=2,
            minimum_boundary_transition_edges=2,
            abstention_hops=0,
        ),
    )
    row = repaired.frames["chart_stitchability"].iloc[0]
    assert row.stitch_status == "boundary_stitchable"
    assert row.targeted_boundary_enrichment_executed
    assert row.verified_boundary_cycle_count > 0
    assert row.verified_boundary_cycle_failure_count == 0
    assert row.boundary_cycle_repeat_p95_deg <= 0.2
    assert repaired.coverage_ratio == 1.0

    line = _growth((0.0, 0.2), count=6)
    left, right = line.charts
    object.__setattr__(left, "selected_by_node", type(left.selected_by_node)({
        node: item for node, item in left.selected_by_node.items() if node <= 2
    }))
    object.__setattr__(right, "selected_by_node", type(right.selected_by_node)({
        node: item for node, item in right.selected_by_node.items() if node >= 3
    }))
    unresolved = repair_rooted_section_atlas(
        line,
        _affine,
        patch_id="patch_indeterminate",
        method="S4",
        policy=AtlasRepairPolicy(
            audit=AuditV2Policy(repeats_per_direction=1),
            minimum_chart_cells=3,
            minimum_stitch_overlap_cells=2,
            minimum_boundary_transition_edges=2,
            abstention_hops=0,
        ),
    )
    unresolved_row = unresolved.frames["chart_stitchability"].iloc[0]
    assert unresolved_row.stitch_status == "indeterminate"
    assert unresolved_row.targeted_boundary_enrichment_executed
    assert unresolved.coverage_ratio == 0.5


def test_repeat_audit_uses_registered_nonzero_source_perturbations() -> None:
    repaired = repair_rooted_section_atlas(
        _growth((0.0,)),
        _affine,
        patch_id="patch_repeat",
        method="S4",
        policy=AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=3)),
    )
    executions = repaired.frames["audit_v2_executions"]
    assert executions.loc[executions["repeat_index"].eq(0), "repeat_perturbation_l2_rad"].eq(0.0).all()
    assert executions.loc[executions["repeat_index"].gt(0), "repeat_perturbation_l2_rad"].gt(0.0).all()
    assert set(executions["registered_solver_chain"]) == {
        "predictor>bounded_ls"
    }
    executed = executions["executed_solver_chain"].astype(str)
    assert set(executed) <= {"", "predictor>bounded_ls"}
    assert (executed == "predictor>bounded_ls").any()
    assert executions["solver_chain_sha256"].astype(str).str.len().eq(64).all()


def test_phase_executor_preserves_serial_audit_semantics() -> None:
    growth = _growth((0.0,))
    policy = AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=1))
    serial = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_executor",
        method="serial",
        policy=policy,
    )
    phases: list[str] = []

    def executor(
        local_growth,
        schedules,
        continuation,
        audit_policy,
        retry_continuation,
        phase_id,
    ):
        phases.append(phase_id)
        return execute_audit_schedules(
            local_growth,
            schedules,
            continuation,
            audit_policy,
            retry_continuation=retry_continuation,
        )

    delegated = repair_rooted_section_atlas(
        growth,
        _affine,
        patch_id="patch_executor",
        method="delegated",
        policy=policy,
        schedule_executor=executor,
    )

    assert phases == ["chart_initial", "primary_certificate"]
    columns = [
        "direction",
        "repeat_index",
        "solver_success",
        "geometry_gap_deg",
        "classification",
    ]
    left = serial.frames["audit_v2_executions"].sort_values(
        ["schedule_id", "direction", "repeat_index"]
    )
    right = delegated.frames["audit_v2_executions"].sort_values(
        ["schedule_id", "direction", "repeat_index"]
    )
    assert left[columns].reset_index(drop=True).equals(
        right[columns].reset_index(drop=True)
    )
    assert serial.certificate_gate == delegated.certificate_gate


def test_screening_is_lighter_but_primary_certificate_remains_full_and_fresh() -> None:
    phases: list[tuple[str, int, int, set[str]]] = []

    def executor(
        local_growth,
        schedules,
        continuation,
        audit_policy,
        retry_continuation,
        phase_id,
    ):
        phases.append(
            (
                phase_id,
                len(schedules),
                audit_policy.repeats_per_direction,
                set(schedules["audit_kind"].astype(str)),
            )
        )
        return execute_audit_schedules(
            local_growth,
            schedules,
            continuation,
            audit_policy,
            retry_continuation=retry_continuation,
        )

    repaired = repair_rooted_section_atlas(
        _growth((0.0,)),
        _affine,
        patch_id="patch_screen",
        method="screened",
        policy=AtlasRepairPolicy(audit=AuditV2Policy(repeats_per_direction=3)),
        schedule_executor=executor,
        screening_first=True,
    )

    assert repaired.certificate_gate is True
    assert phases[0][0] == "chart_screening"
    assert phases[0][2] == 1
    assert phases[0][3] <= {"edge", "fundamental_cycle"}
    assert phases[-1][0] == "primary_certificate"
    assert phases[-1][2] == 3
    assert "root_path" in phases[-1][3]
