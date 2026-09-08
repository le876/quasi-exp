from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.retry11_bridge import (
    classify_retry11_targets,
    corridor_path_registry,
    evaluate_bridge_certificate,
    zero_seed_status,
)
from quasi_exp.teacher.retry11_sampling import (
    annotate_spatial_strata,
    build_macroblock_split_registry,
    hierarchical_replacement_candidates,
    select_parent_first_sparse_wide,
)


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))


def _row(parent: int, ordinal: int, *, quality: str = "Gold") -> dict[str, object]:
    value = parent * 0.01 + ordinal * 0.001
    return {
        "task_node_id": parent * 100 + ordinal,
        "physical_point_id": f"p{parent}_{ordinal}",
        "source_parent_node_id": parent,
        "label_quality": quality,
        "x_m": value,
        "y_m": (parent % 2) * 0.01,
        "z_m": 0.0,
        "shell_sector": f"shell_{ordinal % 2}:sector_{parent % 2}",
        **{name: 0.001 * (index + ordinal) for index, name in enumerate(BETA_COLUMNS)},
    }


def test_parent_first_serves_every_eligible_parent_before_round_two() -> None:
    mandatory = pd.DataFrame([_row(parent, 0) for parent in range(4)])
    candidates = pd.DataFrame(
        [_row(parent, ordinal) for parent in range(4) for ordinal in range(1, 5)]
    )

    selected, audit, report = select_parent_first_sparse_wide(
        mandatory, candidates, target_rows=12, minimum_rows=8
    )

    assert len(selected) == 12
    assert report["new_parent_service_coverage_round1"] == 1.0
    assert report["active_nonexhausted_parent_fairness_gap"] <= 1
    assert audit[audit["parent_round"].eq(1)]["source_parent_node_id"].nunique() == 4
    assert audit[audit["parent_round"].eq(2)]["source_parent_node_id"].nunique() == 4
    assert selected["selection_origin"].eq("mandatory_frozen_atlas").sum() == 4
    assert report["duplicate_xyz_count"] == 0
    assert report["row_padding_used"] is False


def test_shell_sector_denominator_counts_distinct_parents_not_candidates() -> None:
    mandatory = pd.DataFrame([_row(1, 0), _row(2, 0)])
    candidates = pd.DataFrame(
        [
            *[_row(1, ordinal) for ordinal in range(1, 8)],
            _row(2, 2),
        ]
    )
    candidates["shell_sector"] = "shell_0:sector_0"
    mandatory["shell_sector"] = "shell_0:sector_0"

    _selected, _audit, report = select_parent_first_sparse_wide(
        mandatory, candidates, target_rows=4, minimum_rows=4
    )

    assert report["stratum_eligible_parent_denominator"] == {"shell_0:sector_0": 2}


def test_spatial_strata_use_radial_shell_and_eight_octants() -> None:
    frame = pd.DataFrame(
        {
            "x_m": [0.011, -0.011],
            "y_m": [0.001, -0.001],
            "z_m": [0.001, -0.001],
        }
    )

    result = annotate_spatial_strata(frame, zero_xyz_m=[0, 0, 0], shell_width_mm=10)

    assert result["radial_shell_id"].tolist() == [1, 1]
    assert result["sector_id"].tolist() == [7, 0]


def test_macroblock_registry_rejects_panel_overlap() -> None:
    frame = pd.DataFrame({"x_m": [0.0], "y_m": [0.0], "z_m": [0.0]})
    with pytest.raises(ValueError, match="overlap"):
        build_macroblock_split_registry([frame], {"A": frame, "B": frame})


def test_replacement_is_hierarchical_quality_monotone_and_split_safe() -> None:
    base = {
        "physical_point_id": "failed",
        "source_parent_node_id": 10,
        "radial_shell_id": 4,
        "sector_id": 3,
        "canonical_component_id": "outer_component",
        "split_role": "train_core",
        "label_quality": "Silver",
        "x_m": 0.0,
        "y_m": 0.0,
        "z_m": 0.0,
    }
    candidates = pd.DataFrame.from_records(
        [
            {**base, "physical_point_id": "level0_silver", "label_quality": "Silver", "x_m": 0.03},
            {**base, "physical_point_id": "level0_gold", "label_quality": "Gold", "x_m": 0.01},
            {**base, "physical_point_id": "level1_silver", "source_parent_node_id": 11, "label_quality": "Silver", "x_m": 0.001},
            {**base, "physical_point_id": "wrong_component", "canonical_component_id": "zero_component"},
            {**base, "physical_point_id": "wrong_split", "split_role": "panel_A"},
        ]
    )

    queue = hierarchical_replacement_candidates(base, candidates)

    assert queue["physical_point_id"].tolist() == [
        "level0_silver",
        "level0_gold",
        "level1_silver",
    ]
    assert queue["replacement_level"].tolist() == ["level_0", "level_0", "level_1"]

    gold = dict(base, label_quality="Gold")
    gold_queue = hierarchical_replacement_candidates(gold, candidates)
    assert gold_queue["label_quality"].eq("Gold").all()


def _attempt(
    target: int,
    path: str,
    beta_deg: float,
    *,
    success: bool = True,
    reverse_deg: float = 0.5,
    fk_mm: float = 1.0,
) -> dict[str, object]:
    beta = np.radians(np.full(6, beta_deg))
    return {
        "target_id": target,
        "source_path_id": path,
        "solver_success": success,
        "actual_bounds": True,
        "fk_residual_mm": fk_mm,
        "reverse_gap_deg": reverse_deg,
        "x_m": target * 0.001,
        "y_m": 0.0,
        "z_m": 0.0,
        **{name: beta[index] for index, name in enumerate(BETA_COLUMNS)},
    }


def _retained() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "physical_point_id": "anchor",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                **{name: 0.0 for name in BETA_COLUMNS},
            },
            {
                "physical_point_id": "neighbor",
                "x_m": 0.001,
                "y_m": 0.0,
                "z_m": 0.0,
                **{name: np.radians(0.2) for name in BETA_COLUMNS},
            },
        ]
    )


def test_gold_and_wide_silver_are_fully_defined() -> None:
    attempts = pd.DataFrame.from_records(
        [
            _attempt(10, "path_a", 0.2),
            _attempt(10, "path_b", 0.3),
            _attempt(11, "path_only", 0.4),
        ]
    )

    labels, rejects = classify_retry11_targets(attempts, _retained())

    assert labels.set_index("target_id").loc[10, "label_quality"] == "Gold"
    assert labels.set_index("target_id").loc[11, "label_quality"] == "Wide-Silver"
    assert rejects.empty


def test_two_successful_paths_with_raw_gap_over_five_are_rejected_not_silver() -> None:
    attempts = pd.DataFrame.from_records(
        [_attempt(12, "path_a", 0.0), _attempt(12, "path_b", 6.0)]
    )

    labels, rejects = classify_retry11_targets(attempts, _retained())

    assert labels.empty
    assert rejects.iloc[0]["rejection_reason"] == "branch_conflict"


@pytest.mark.parametrize(
    ("anchor", "count", "expected"),
    [(True, 4, "green"), (True, 2, "yellow"), (True, 1, "red"), (False, 8, "red")],
)
def test_zero_seed_uses_four_two_thresholds(anchor: bool, count: int, expected: str) -> None:
    assert zero_seed_status(exact_anchor_valid=anchor, valid_count=count) == expected


def _bridge_frame(*, branch_conflict: bool = False, adjacent: bool = False, solver: bool = True) -> pd.DataFrame:
    rows = []
    for index in range(4):
        conflict = branch_conflict and (index == 1 or (adjacent and index == 2))
        rows.append(
            {
                "bridge_path_id": "straight",
                "probe_index": index,
                "solver_success": solver,
                "same_point_raw_gap_deg": 1.0 if not conflict else 6.0,
                "local_weighted_gap_deg": 1.0,
                "neighbor_raw_gap_deg": 2.0,
                "persistent_edge_failure": False,
                "fk_residual_mm": 1.0,
                "branch_conflict": conflict,
                "forward_reverse_conflict": False,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_straight_corridor_failure_is_unknown_not_disconnection() -> None:
    certificate = evaluate_bridge_certificate(_bridge_frame(solver=False))

    assert certificate["straight_corridor_not_verified"] is True
    assert certificate["straight_corridor_failure_class"] == "solver_failure"
    assert certificate["zero_outer_branch_compatible"] == "unknown"


def test_only_stable_branch_conflict_sets_compatibility_false() -> None:
    isolated = evaluate_bridge_certificate(_bridge_frame(branch_conflict=True, adjacent=False))
    stable = evaluate_bridge_certificate(_bridge_frame(branch_conflict=True, adjacent=True))

    assert isolated["zero_outer_branch_compatible"] == "unknown"
    assert stable["zero_outer_branch_compatible"] == "false"


def test_corridor_step_budget_and_diversity_paths_are_bounded() -> None:
    registry = corridor_path_registry([0, 0, 0], [0.15, 0, 0])

    assert registry["bridge_path_id"].nunique() == 4
    assert registry.groupby("bridge_path_id").size().max() <= 32
    straight = registry[registry["bridge_path_id"].eq("straight")].sort_values("probe_index")
    steps = np.linalg.norm(np.diff(straight[["x_m", "y_m", "z_m"]], axis=0), axis=1)
    assert np.max(steps) <= 0.010 + 1.0e-12
