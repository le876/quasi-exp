from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.partial_relay import (
    MethodSpec,
    RootKind,
    RootSpec,
    authorize_exploratory_stages,
    choose_macroblock_split,
    sample_screening_schedules,
    select_spatial_relay_roots,
)
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    ContinuationOutcome,
)
from quasi_exp.teacher.section_atlas_repair import (
    AuditV2Policy,
    build_registered_audit_schedules,
    execute_audit_schedules,
)
from quasi_exp.teacher.section_first_atlas import (
    RootedSectionPolicy,
    build_section_first_atlas,
)


def _root(
    root_id: str,
    kind: RootKind,
    node_id: int,
    xyz: tuple[float, float, float],
) -> RootSpec:
    return RootSpec(
        root_id=root_id,
        root_kind=kind,
        task_node_id=node_id,
        candidate_id=f"candidate_{node_id}",
        canonical_lineage_id="lineage_a",
        parent_root_id=None if kind is RootKind.BRANCH_ORIGIN else "origin",
        inherited_beta_rad=np.full(6, node_id / 100.0),
        xyz_m=np.asarray(xyz),
        inheritance_gap_deg=0.0,
    )


def test_raw_retry7_method_is_diagnostic_only_and_p0_is_selectable() -> None:
    diagnostic = MethodSpec.raw_retry7_diagnostic()
    fallback = MethodSpec.frozen_partial_singleton()

    assert diagnostic.method_id == "D0_raw_retry7"
    assert diagnostic.is_diagnostic_only is True
    assert diagnostic.is_selectable is False
    assert fallback.method_id == "P0_frozen_partial_singleton"
    assert fallback.is_diagnostic_only is False
    assert fallback.is_selectable is True


def test_branch_origin_and_spatial_relay_are_distinct_root_kinds() -> None:
    origin = _root("origin", RootKind.BRANCH_ORIGIN, 1, (0.0, 0.0, 0.0))
    relay = _root("relay", RootKind.SPATIAL_RELAY, 2, (0.02, 0.0, 0.0))

    assert origin.root_kind is RootKind.BRANCH_ORIGIN
    assert relay.root_kind is RootKind.SPATIAL_RELAY
    assert origin.task_node_id != relay.task_node_id

    with pytest.raises(ValueError, match="alternative branch"):
        replace(relay, root_kind=RootKind.ALTERNATIVE_BRANCH)


def test_relay_roots_must_be_spatially_distinct_and_inherit_the_lineage() -> None:
    labels = pd.DataFrame(
        {
            "task_node_id": [1, 2, 3, 4],
            "candidate_id": ["a", "b", "c", "d"],
            "x_m": [0.0, 0.005, 0.020, 0.040],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            **{
                f"beta{index}_rad": [0.0, 0.1, 0.2, 0.3]
                for index in range(1, 7)
            },
        }
    )

    roots = select_spatial_relay_roots(
        labels,
        origin_node_id=1,
        canonical_lineage_id="lineage_a",
        target_count=4,
        preferred_separation_mm=20.0,
        minimum_separation_mm=10.0,
    )

    assert roots[0].root_kind is RootKind.BRANCH_ORIGIN
    assert len({root.task_node_id for root in roots}) == len(roots)
    assert all(root.canonical_lineage_id == "lineage_a" for root in roots)
    assert all(root.inheritance_gap_deg <= 0.1 for root in roots)
    assert all(root.root_kind is not RootKind.ALTERNATIVE_BRANCH for root in roots)


def test_coverage_and_student_execution_authorizations_are_orthogonal() -> None:
    report = authorize_exploratory_stages(
        partial_certificate_pass=True,
        strict_unique_label_count=3200,
        relay_root_count=2,
        relay_coverage_gain=0.01,
        relay_stitch_pass=True,
        coverage_ratio=0.18,
        largest_component_ratio=0.12,
        smoke_pipeline_complete=True,
        teacher_label_integrity_failure=False,
        unresolved_training_implementation_failure=False,
        smoke_student_quality_pass=False,
    )

    assert report.data_legality_gate is True
    assert report.smoke_execution_authorized is True
    assert report.relay_method_validation_pass is True
    assert report.global_coverage_gate_pass is False
    assert report.five_k_teacher_execution_authorized is True
    assert report.student_claim_authorized is False
    assert report.formal_or_deployment_authorized is False


def test_integrity_failure_blocks_five_k_but_quality_miss_does_not() -> None:
    report = authorize_exploratory_stages(
        partial_certificate_pass=True,
        strict_unique_label_count=5000,
        relay_root_count=0,
        relay_coverage_gain=0.0,
        relay_stitch_pass=False,
        coverage_ratio=0.35,
        largest_component_ratio=0.25,
        smoke_pipeline_complete=True,
        teacher_label_integrity_failure=True,
        unresolved_training_implementation_failure=False,
        smoke_student_quality_pass=True,
    )

    assert report.relay_method_validation_status == "not_evaluated"
    assert report.five_k_teacher_execution_authorized is False


def test_macroblock_split_uses_coarsest_feasible_registered_size() -> None:
    points = []
    for index in range(20):
        for repeat in range(500):
            points.append(
                {
                    "x_m": index * 0.041,
                    "y_m": (index % 4) * 0.041,
                    "z_m": 0.0,
                    "physical_point_id": f"{index}:{repeat}",
                }
            )
    frame = pd.DataFrame.from_records(points)

    split = choose_macroblock_split(
        frame,
        block_sizes_mm=(40, 30, 20),
        split_seed=20260881,
        split_fractions=(0.70, 0.15, 0.15),
        minimum_train_blocks=10,
        minimum_validation_blocks=3,
        minimum_test_blocks=3,
        minimum_rows_per_split=500,
    )

    assert split.block_size_mm in (40, 30, 20)
    assert set(split.frame["split_role"]) == {"train_core", "validation", "test"}
    assert split.frame.groupby("physical_point_id")["split_role"].nunique().max() == 1


def test_sampled_screening_caps_interior_edges_and_cycles() -> None:
    schedules = pd.DataFrame.from_records(
        [
            {
                "schedule_id": f"edge_{index}",
                "audit_kind": "edge",
                "path_node_ids": [index, index + 1],
            }
            for index in range(20)
        ]
        + [
            {
                "schedule_id": f"cycle_{index}",
                "audit_kind": "fundamental_cycle",
                "path_node_ids": [index, index + 1, index + 2, index],
            }
            for index in range(10)
        ]
        + [
            {
                "schedule_id": "root_path_0",
                "audit_kind": "root_path",
                "path_node_ids": [0, 1, 2],
            }
        ]
    )

    selected = sample_screening_schedules(
        schedules,
        relay_node_ids=(0,),
        boundary_node_ids=(10,),
        maximum_interior_edges=4,
        maximum_cycles=3,
    )

    assert "root_path" not in set(selected["audit_kind"])
    assert selected["audit_kind"].eq("fundamental_cycle").sum() == 3
    assert set(selected.loc[selected["screen_reason"].eq("relay_transition"), "schedule_id"]) == {"edge_0"}
    assert selected["audit_kind"].eq("edge").sum() <= 7


def test_screening_audit_can_run_forward_only_with_two_repeats() -> None:
    nodes = (
        AtlasTaskNode(0, np.asarray([0.0, 0.0, 0.0]), (1,)),
        AtlasTaskNode(1, np.asarray([0.001, 0.0, 0.0]), (0,)),
    )
    root = AtlasCandidate(
        node_id=0,
        candidate_id="origin",
        beta_rad=np.zeros(6),
        residual_mm=0.0,
        min_margin_deg=5.0,
        normalized_min_margin=0.5,
        condition_number=1.0,
    )

    def affine(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
        beta = source.beta_rad.copy()
        beta[0] = target.xyz_m[0]
        return ContinuationOutcome(beta, 0.0, True, True, 1, "affine")

    growth = build_section_first_atlas(
        nodes,
        (root,),
        affine,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(root_count=1, maximum_growth_waves=2),
    )
    schedules = build_registered_audit_schedules(
        growth, patch_id="screen", method="screen"
    )
    executions = execute_audit_schedules(
        growth,
        schedules[schedules["audit_kind"].eq("edge")],
        affine,
        AuditV2Policy(
            repeats_per_direction=2,
            directions=("forward",),
        ),
        retry_continuation=None,
    )

    assert set(executions["direction"]) == {"forward"}
    assert len(executions) == 2
