from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.dense_chart_sampling import weighted_beta_gap_deg
from quasi_exp.teacher.retry10 import (
    BETA_COLUMNS,
    GateStatus,
    classify_frontier_taxonomy,
    compatible_candidate_medoid,
    component_boundary_audit,
    dataset_generation_decision,
    dynamic_frontier_root_registry,
    k2_retention_decision,
    preflight_generation_decision,
    student_quality_decision,
)
from quasi_exp.teacher.retry10_sampling import (
    exact_nested_dataset,
    parent_domain_target_registry,
)


def _beta_row(offsets_deg: list[float]) -> dict[str, float]:
    return {
        column: math.radians(value)
        for column, value in zip(BETA_COLUMNS, offsets_deg, strict=True)
    }


def test_retry10_metric_is_opt_in_and_does_not_change_legacy_default() -> None:
    left = np.zeros(6)
    right = np.deg2rad([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert weighted_beta_gap_deg(left, right) == pytest.approx(math.sqrt(4.0 / 14.0))
    assert weighted_beta_gap_deg(
        left, right, metric_version="normalized_weighted_v1"
    ) == pytest.approx(4.0 / math.sqrt(42.0))


def test_new_chart_candidates_require_complete_link_compatibility_before_medoid() -> None:
    candidates = pd.DataFrame.from_records(
        [
            {
                "task_node_id": 7,
                "chart_id": "a",
                "candidate_id": "a7",
                **_beta_row([0.0] * 6),
            },
            {
                "task_node_id": 7,
                "chart_id": "b",
                "candidate_id": "b7",
                **_beta_row([0.2] * 6),
            },
            {
                "task_node_id": 8,
                "chart_id": "a",
                "candidate_id": "a8",
                **_beta_row([0.0] * 6),
            },
            {
                "task_node_id": 8,
                "chart_id": "b",
                "candidate_id": "b8",
                **_beta_row([5.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
        ]
    )
    selected, audit = compatible_candidate_medoid(candidates)
    assert selected["task_node_id"].tolist() == [7]
    conflict = audit.set_index("task_node_id").loc[8]
    assert bool(conflict["new_node_branch_conflict"]) is True
    assert pd.isna(conflict["selected_candidate_id"])


def test_frontier_roots_use_current_parent_support_before_historical_support() -> None:
    nodes = pd.DataFrame.from_records(
        [
            {"task_node_id": 0, "source_parent_node_id": 10, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            {"task_node_id": 1, "source_parent_node_id": 11, "x_m": 0.1, "y_m": 0.0, "z_m": 0.0},
            {"task_node_id": 2, "source_parent_node_id": 12, "x_m": 0.0, "y_m": 0.1, "z_m": 0.0},
            {"task_node_id": 3, "source_parent_node_id": 13, "x_m": 0.1, "y_m": 0.1, "z_m": 0.0},
        ]
    )
    edges = pd.DataFrame(
        {"left_node_id": [0, 1], "right_node_id": [2, 3]}
    )
    labels = pd.DataFrame.from_records(
        [
            {"task_node_id": 0, "chart_id": "a", "x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            {"task_node_id": 0, "chart_id": "b", "x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            {"task_node_id": 1, "chart_id": "a", "x_m": 0.1, "y_m": 0.0, "z_m": 0.0},
        ]
    )
    roots = dynamic_frontier_root_registry(
        nodes,
        edges,
        labels,
        historical_support={0: 0, 1: 100},
        root_count=1,
        hops=1,
    )
    assert roots.iloc[0]["task_node_id"] == 1
    assert roots.iloc[0]["current_node_support_count"] == 1


def test_taxonomy_predicates_are_strictly_prioritized() -> None:
    base = {
        "feasible_label_count": 0,
        "raw_candidate_gap_max_deg": 0.0,
        "catastrophic_neighbor_jump": False,
        "cap_hit": False,
        "pruned_hypothesis_reachable": False,
        "registered_retained_neighbor": False,
        "local_continuation_succeeded": False,
        "local_continuation_exhausted_or_pending": False,
        "edge_predicate_satisfied_missing_edge": False,
        "finite_in_bounds_fk_valid_label": False,
        "certified_primary_path": False,
    }
    evidence = pd.DataFrame.from_records(
        [
            {"task_node_id": 0, **base, "feasible_label_count": 2, "raw_candidate_gap_max_deg": 6.0, "cap_hit": True, "pruned_hypothesis_reachable": True},
            {"task_node_id": 1, **base, "cap_hit": True, "pruned_hypothesis_reachable": True, "registered_retained_neighbor": True, "local_continuation_exhausted_or_pending": True},
            {"task_node_id": 2, **base, "registered_retained_neighbor": True, "local_continuation_exhausted_or_pending": True, "edge_predicate_satisfied_missing_edge": True},
            {"task_node_id": 3, **base, "edge_predicate_satisfied_missing_edge": True, "finite_in_bounds_fk_valid_label": True},
            {"task_node_id": 4, **base, "finite_in_bounds_fk_valid_label": True},
            {"task_node_id": 5, **base},
        ]
    )
    assert classify_frontier_taxonomy(evidence)["taxonomy"].tolist() == [
        "branch_conflict",
        "beam_limited",
        "frontier_reachable",
        "graph_limited",
        "disconnected_but_feasible",
        "no_feasible_solution_observed",
    ]


def test_valid_k2_labels_are_kept_even_when_policy_is_not_promoted() -> None:
    small = k2_retention_decision(recovered_node_count=80)
    large = k2_retention_decision(recovered_node_count=125)
    assert small["keep_valid_k2_labels"] is True
    assert small["promote_k2_policy"] is False
    assert large["promote_k2_policy"] is True


def test_preflight_authorization_is_distinct_from_exact_completion() -> None:
    good = pd.DataFrame.from_records(
        [
            {"stratum_id": "a", "target_weight": 0.5, "trial_count": 100, "gold_count": 75, "silver_count": 15},
            {"stratum_id": "b", "target_weight": 0.5, "trial_count": 100, "gold_count": 70, "silver_count": 20},
        ]
    )
    decision, report = preflight_generation_decision(
        good, required_new_rows=50_000, allowed_qualities=("Gold", "Silver")
    )
    assert decision.status is GateStatus.PASS
    assert decision.authorized is True
    assert report["estimated_attempts"] < 250_000
    assert "completion" not in report


def test_dataset_generation_is_independent_of_student_quality_and_formal_claims() -> None:
    decision, dataset_id = dataset_generation_decision(
        target_rows=50_000,
        atlas_frozen=True,
        exploration_hard_guards_pass=True,
        coverage_fraction=0.25,
        plateau_or_exhausted=True,
        seed_rows=10_000,
        seed_learning_curve_artifact_exists=True,
        prior_dataset_complete=False,
        taxonomy_complete=True,
        preflight_pass=True,
    )
    assert decision.authorized is True
    assert dataset_id == "pilot_domain_limited_candidate_dataset_50k"


def test_student_quality_preserves_registered_raw_or_dls_gate() -> None:
    decision = student_quality_decision(
        evaluated=True,
        no_nan=True,
        bounds_violation_count=0,
        raw_fk_p95_mm=40.0,
        dls_two_step_fk_p95_mm=0.01,
    )
    assert decision.authorized is True


def test_adjacent_components_must_pass_boundary_consistency() -> None:
    labels = pd.DataFrame.from_records(
        [
            {"task_node_id": 0, "component_id": "primary", **_beta_row([0.0] * 6)},
            {"task_node_id": 1, "component_id": "other", **_beta_row([6.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        ]
    )
    audit = component_boundary_audit(
        labels, pd.DataFrame({"left_node_id": [0], "right_node_id": [1]})
    )
    assert audit.iloc[0]["relation"] == "incompatible"
    assert bool(audit.iloc[0]["component_global_student_compatible"]) is False


def test_target_registry_never_crosses_registered_parent_domains() -> None:
    labels = pd.DataFrame.from_records(
        [
            {"task_node_id": 0, "source_parent_node_id": 10, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0, **_beta_row([0.0] * 6)},
            {"task_node_id": 1, "source_parent_node_id": 10, "x_m": 1.0, "y_m": 0.0, "z_m": 0.0, **_beta_row([0.0] * 6)},
            {"task_node_id": 2, "source_parent_node_id": 20, "x_m": 10.0, "y_m": 0.0, "z_m": 0.0, **_beta_row([0.0] * 6)},
            {"task_node_id": 3, "source_parent_node_id": 20, "x_m": 11.0, "y_m": 0.0, "z_m": 0.0, **_beta_row([0.0] * 6)},
        ]
    )
    targets = parent_domain_target_registry(labels, target_count=12)
    indexed = labels.set_index("task_node_id")
    for row in targets.itertuples(index=False):
        assert indexed.loc[row.source_node_1, "source_parent_node_id"] == row.source_parent_node_id
        assert indexed.loc[row.source_node_2, "source_parent_node_id"] == row.source_parent_node_id


def test_exact_dataset_underfill_is_reported_without_padding() -> None:
    existing = pd.DataFrame.from_records(
        [
            {"xyz_key": "a", "label_quality": "Gold", "source_parent_node_id": 1, "sampling_stratum": "s"},
            {"xyz_key": "a", "label_quality": "Gold", "source_parent_node_id": 1, "sampling_stratum": "s"},
        ]
    )
    generated = pd.DataFrame.from_records(
        [{"xyz_key": "b", "label_quality": "Silver", "source_parent_node_id": 1, "sampling_stratum": "s"}]
    )
    result = exact_nested_dataset(
        existing, generated, target_rows=5, allowed_qualities=("Gold", "Silver")
    )
    assert len(result) == 2
    assert result["xyz_key"].is_unique
