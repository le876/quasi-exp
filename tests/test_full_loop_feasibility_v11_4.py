from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.full_loop_feasibility import (
    StrictCycleGate,
    candidate_search_budgets,
    classify_full_loop_outcome,
    cluster_full_loop_cycles,
    connectivity_sensitivity,
    formal_decision,
    filter_hard_feasible_candidates,
    geodesic_farthest_representatives,
    rank_viability,
    solve_sparse_cycle,
    strict_cycle_metrics,
    validated_mutual_knn_graph,
)


def test_candidate_search_budget_starts_at_registered_initial_budget() -> None:
    assert candidate_search_budgets(
        candidate_cap=16,
        initial_budget_floor=128,
        initial_budget_multiplier=4,
        targeted_cumulative_budgets=(128, 256, 512),
    ) == (128, 256, 512)
    assert candidate_search_budgets(
        candidate_cap=64,
        initial_budget_floor=128,
        initial_budget_multiplier=4,
        targeted_cumulative_budgets=(128, 256, 512),
    ) == (256, 512)


def test_full_loop_cluster_uses_curve_p95_not_only_root_distance() -> None:
    base = np.zeros((20, 6), dtype=float)
    locally_different = base.copy()
    locally_different[0, 5] = np.deg2rad(3.0)
    globally_different = base.copy()
    globally_different[:, 5] = np.deg2rad(3.0)

    clustered = cluster_full_loop_cycles(
        [base, locally_different, globally_different],
        threshold_deg=1.0,
        beta_weights=np.ones(6),
    )

    assert clustered.distance_deg.shape == (3, 3)
    assert clustered.distance_deg[0, 1] < 1.0
    assert clustered.component_labels[0] == clustered.component_labels[1]
    assert clustered.component_labels[2] != clustered.component_labels[0]


def test_formal_decision_never_relabels_audit_failure_as_relaxed_d() -> None:
    assert (
        formal_decision(
            numerical_outcome="A",
            audit_pass=True,
        )
        == "A"
    )
    assert (
        formal_decision(
            numerical_outcome="A",
            audit_pass=False,
        )
        == "A_AUDIT_FAILED"
    )
    assert (
        formal_decision(
            numerical_outcome="D",
            audit_pass=False,
        )
        == "D"
    )


def test_geodesic_representatives_do_not_reselect_disconnected_seed() -> None:
    beta = np.deg2rad(np.arange(24, dtype=float).reshape(4, 6))
    geodesic = np.full((4, 4), np.inf)
    np.fill_diagonal(geodesic, 0.0)

    selected = geodesic_farthest_representatives(
        beta,
        max_count=4,
        required_indices=[0],
        geodesic_distance_deg=geodesic,
    )

    assert selected.tolist() == [0, 1, 2, 3]


def test_hard_candidate_filter_never_raw_fills_to_requested_cap() -> None:
    beta = np.deg2rad(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            ]
        )
    )
    records = pd.DataFrame(
        {
            "candidate_source": ["lineage", "duplicate", "raw", "bad_margin", "good"],
            "corrector_success": [True, True, False, True, True],
            "residual_mm": [0.1, 0.2, 0.0, 0.1, 2.9],
            "joint_margin_deg": [2.0, 2.0, 10.0, 1.49, 2.0],
            "within_bounds": [True] * 5,
        }
    )

    selected = filter_hard_feasible_candidates(
        beta,
        records,
        cap=16,
        residual_max_mm=3.0,
        margin_min_deg=1.5,
        distinct_threshold_deg=0.25,
    )

    assert selected.table["candidate_source"].tolist() == ["lineage", "good"]
    assert selected.beta_rad.shape == (2, 6)
    assert selected.requested_cap == 16
    assert selected.shortfall == 14


def test_connectivity_reports_epsilon_sensitivity_and_validates_edges() -> None:
    beta = np.deg2rad(
        np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.4],
                [0, 0, 0, 0, 0, 0.8],
                [0, 0, 0, 0, 0, 4.0],
            ],
            dtype=float,
        )
    )
    sensitivity = connectivity_sensitivity(beta, thresholds_deg=[0.25, 0.5, 1.0, 5.0])
    assert sensitivity["connected_component_count"].tolist() == [2, 2, 1, 1]

    def interpolation_is_feasible(left: np.ndarray, right: np.ndarray, fraction: float) -> bool:
        midpoint = (1.0 - fraction) * left + fraction * right
        return bool(np.rad2deg(midpoint[5]) < 0.55)

    graph = validated_mutual_knn_graph(
        beta[:3],
        k=2,
        max_edge_deg=3.0,
        interpolation_fractions=(0.25, 0.5, 0.75),
        interpolation_is_feasible=interpolation_is_feasible,
    )
    assert graph.edge_table["validated"].all()
    assert set(map(tuple, graph.edge_table[["left_idx", "right_idx"]].to_numpy())) == {
        (0, 1)
    }
    assert graph.connected_component_count == 2


def test_viability_ranking_is_lexicographic_with_full_loop_first() -> None:
    table = pd.DataFrame(
        {
            "root_candidate_idx": [0, 1, 2],
            "completed_360": [False, True, True],
            "strict_feasible_ratio": [1.0, 0.8, 0.8],
            "joint_margin_min_deg": [10.0, 2.0, 3.0],
            "residual_max_mm": [0.01, 2.0, 2.0],
            "residual_p95_mm": [0.01, 0.5, 0.5],
            "velocity_p95_deg": [0.01, 0.2, 0.2],
            "acceleration_p95_deg": [0.01, 0.1, 0.1],
            "posture_cost": [0.0, 0.0, 100.0],
            "kappa_p95": [1.0, 10.0, 100.0],
        }
    )

    ranked = rank_viability(table)

    assert ranked["root_candidate_idx"].tolist() == [2, 1, 0]
    assert ranked["viability_rank"].tolist() == [1, 2, 3]


def test_sparse_cycle_finds_closed_hard_path_and_rejects_open_path() -> None:
    smooth = np.deg2rad(
        np.asarray(
            [
                [0, 0, 0, 0, 0, 0.0],
                [0, 0, 0, 0, 0, 0.4],
                [0, 0, 0, 0, 0, 0.0],
                [0, 0, 0, 0, 0, -0.4],
            ],
            dtype=float,
        )
    )
    distractor = smooth.copy()
    distractor[:, 5] += np.deg2rad([5.0, 7.0, 9.0, 11.0])
    layers = [np.vstack([smooth[index], distractor[index]]) for index in range(4)]
    residuals = [np.asarray([0.1, 0.0]) for _ in layers]

    solved = solve_sparse_cycle(
        layers,
        residuals,
        beta_weights=np.ones(6),
        edge_limit_deg=1.0,
        lambda_velocity=1.0,
        lambda_acceleration=0.5,
        top_m=8,
    )

    assert solved.success is True
    assert solved.beta_rad.shape == (4, 6)
    assert np.allclose(solved.beta_rad, smooth)
    assert solved.root_candidate_idx == 0
    assert solved.edge_limit_deg == 1.0

    broken = [layer.copy() for layer in layers]
    broken[-1][:, 5] += np.deg2rad(4.0)
    rejected = solve_sparse_cycle(
        broken,
        residuals,
        beta_weights=np.ones(6),
        edge_limit_deg=1.0,
        lambda_velocity=1.0,
        lambda_acceleration=0.5,
        top_m=8,
    )
    assert rejected.success is False
    assert rejected.reason == "no_closed_cycle"


def test_strict_metrics_and_decision_boundary_are_fail_closed() -> None:
    beta = np.zeros((8, 6))
    target = np.zeros((8, 3))
    achieved = target.copy()
    metrics = strict_cycle_metrics(
        beta,
        target_xyz_m=target,
        achieved_xyz_m=achieved,
        joint_margin_deg=np.full(8, 2.0),
    )
    gate = StrictCycleGate().evaluate(metrics)
    assert gate["gate_pass"] is True

    assert (
        classify_full_loop_outcome(
            strict_cycle_pass=True,
            empty_feasible_layer_indices=[],
            targeted_search_complete=True,
            diagnostic_relaxed_cycle_pass=True,
        )
        == "A"
    )
    assert (
        classify_full_loop_outcome(
            strict_cycle_pass=False,
            empty_feasible_layer_indices=[],
            targeted_search_complete=True,
            diagnostic_relaxed_cycle_pass=False,
        )
        == "B"
    )
    assert (
        classify_full_loop_outcome(
            strict_cycle_pass=False,
            empty_feasible_layer_indices=[3],
            targeted_search_complete=True,
            diagnostic_relaxed_cycle_pass=False,
        )
        == "C"
    )
    assert (
        classify_full_loop_outcome(
            strict_cycle_pass=False,
            empty_feasible_layer_indices=[],
            targeted_search_complete=True,
            diagnostic_relaxed_cycle_pass=True,
        )
        == "D"
    )
