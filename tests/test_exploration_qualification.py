from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.canonical_atlas import AtlasTaskNode, ContinuationOutcome
from quasi_exp.teacher.exploration_qualification import (
    BETA_COLUMNS,
    ENDPOINT_BETA_COLUMNS,
    ExplorationThresholds,
    bounded_fundamental_cycles,
    conservative_weighted_upper_bound_deg,
    execute_exploration_audit_schedules,
    overlap_pair_metrics,
    select_transition_edges,
    summarize_exploration_pair,
    weighted_beta_rms_deg,
)
from quasi_exp.teacher.section_atlas_repair import AuditV2Policy, RetryTier, section_growth_from_frames


def _labels(chart_id: str, beta_offset_deg: float = 0.0) -> pd.DataFrame:
    rows = []
    for node_id in range(4):
        row = {
            "chart_id": chart_id,
            "root_node_id": 0,
            "root_candidate_id": "retry8_lineage_000",
            "task_node_id": node_id,
            "candidate_id": f"{chart_id}_{node_id}",
            "selected": True,
            "score": 0.0,
            "wave": node_id,
            "boundary_risk": False,
            "parent_node_ids": [] if node_id == 0 else [node_id - 1],
            "parent_candidate_ids": [] if node_id == 0 else [f"{chart_id}_{node_id - 1}"],
            "residual_mm": 0.0,
            "min_margin_deg": 10.0,
            "normalized_min_margin": 1.0,
            "posture_cost": 0.0,
            "condition_number": 1.0,
            "quality": "Gold",
            "solver_success": True,
            "actual_bounds": True,
            "cluster_id": np.nan,
            "x_m": 0.02 * (node_id % 2),
            "y_m": 0.02 * (node_id // 2),
            "z_m": 0.0,
        }
        row.update({column: math.radians(beta_offset_deg) for column in BETA_COLUMNS})
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        {"left_node_id": [0, 0, 1, 2], "right_node_id": [1, 2, 3, 3],
         "adjacency": ["grid"] * 4}
    )


def test_weighted_metric_uses_sum_of_squared_weights_normalization() -> None:
    left = np.zeros(6)
    right = np.deg2rad([1, 1, 1, 1, 1, 1])
    assert weighted_beta_rms_deg(left, right) == pytest.approx(1.0)
    expected_factor = 4.0 * math.sqrt(6.0 / 42.0)
    assert conservative_weighted_upper_bound_deg(2.0) == pytest.approx(2.0 * expected_factor)
    with pytest.raises(ValueError, match="six finite positive"):
        weighted_beta_rms_deg(left, right, (1.0, 2.0))


def test_label_overlap_and_bounded_sampling_are_deterministic() -> None:
    left, right = _labels("chart_a"), _labels("chart_b", 0.1)
    thresholds = ExplorationThresholds(
        minimum_overlap_nodes=4, minimum_overlap_fraction=1.0,
        minimum_overlap_spread_mm=20.0,
    )
    metrics = overlap_pair_metrics(left, right, thresholds=thresholds)
    assert metrics["label_overlap_gate"] is True
    assert metrics["weighted_gap_p95_deg"] == pytest.approx(0.1)
    xyz = {
        int(row.task_node_id): np.asarray([row.x_m, row.y_m, row.z_m])
        for row in left.itertuples(index=False)
    }
    gaps = {0: 0.0, 1: 0.1, 2: 2.0, 3: 0.2}
    first = select_transition_edges(range(4), _edges(), xyz, gaps, maximum_edges=3, high_gap_edges=1)
    second = select_transition_edges(range(4), _edges(), xyz, gaps, maximum_edges=3, high_gap_edges=1)
    assert first == second
    assert first[0] in {(0, 2), (2, 3)}
    assert len(first) == 3


def test_short_cycle_finder_closes_the_registered_cycle() -> None:
    cycles = bounded_fundamental_cycles(range(4), _edges(), maximum_cycles=2, maximum_length=4)
    assert cycles
    assert cycles[0][0] == cycles[0][-1]
    assert set(cycles[0][:-1]) == {0, 1, 2, 3}


def test_fresh_execution_persists_endpoint_beta_and_exact_weighted_metrics() -> None:
    labels = _labels("chart_a")
    edges = _edges()
    adjacency = {node: [] for node in range(4)}
    for row in edges.itertuples(index=False):
        adjacency[int(row.left_node_id)].append(int(row.right_node_id))
        adjacency[int(row.right_node_id)].append(int(row.left_node_id))
    nodes = tuple(
        AtlasTaskNode(node, labels.loc[node, ["x_m", "y_m", "z_m"]].to_numpy(float), adjacency[node])
        for node in range(4)
    )
    selected_edges = edges.assign(chart_id="chart_a")
    growth = section_growth_from_frames(nodes, labels, selected_edges)

    def continuation(source, target):
        return ContinuationOutcome(
            beta_rad=np.zeros(6), residual_mm=0.0, success=True,
            actual_bounds=True, status="audit_solver_chain[predictor>bounded_ls]::test",
        )

    schedules = pd.DataFrame.from_records(
        [
            {"schedule_id": "edge", "pair_id": "a__b", "chart_id": "chart_a",
             "audit_kind": "transition", "path_node_ids": [0, 1],
             "physical_entity_id": "edge:0:1"},
            {"schedule_id": "cycle", "pair_id": "a__b", "chart_id": "chart_a",
             "audit_kind": "cycle", "path_node_ids": [0, 1, 3, 2, 0],
             "physical_entity_id": "cycle:0"},
        ]
    )
    policy = AuditV2Policy(
        geometry_p95_max_deg=2.0, geometry_max_deg=5.0,
        repeat_p95_max_deg=1.0, continuation_residual_max_mm=10.0,
        repeats_per_direction=2, directions=("forward",),
        retry_tiers=(RetryTier("R0", 5.0, 10, ("predictor", "bounded_ls")),),
    )
    executions = execute_exploration_audit_schedules(
        growth, schedules, continuation, policy, retry_continuation=None
    )
    assert len(executions) == 4
    assert executions.loc[:, ENDPOINT_BETA_COLUMNS].notna().all().all()
    assert executions["weighted_geometry_gap_deg"].eq(0.0).all()
    summary = summarize_exploration_pair(
        executions,
        thresholds=ExplorationThresholds(
            minimum_overlap_nodes=1, minimum_overlap_fraction=0.0,
            minimum_overlap_spread_mm=0.0,
        ),
    )
    assert summary["fresh_audit_gate"] is True


def test_pair_summary_does_not_upgrade_missing_cycle_evidence() -> None:
    executions = pd.DataFrame(
        {"audit_kind": ["transition"], "solver_success": [True],
         "classification": ["verified"], "geometry_gap_deg": [0.0],
         "weighted_geometry_gap_deg": [0.0], "weighted_repeat_gap_deg": [0.0],
         "residual_mm": [0.0], "exact_bounds": [True]}
    )
    assert summarize_exploration_pair(executions)["fresh_audit_gate"] is False
