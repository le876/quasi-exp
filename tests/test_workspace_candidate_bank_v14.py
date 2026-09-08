from __future__ import annotations

import numpy as np

from quasi_exp.teacher.workspace_candidate_bank import (
    CandidatePolicy,
    CandidateQuality,
    CandidateSearchMode,
    solve_candidate_bank,
)


class _RedundantAffineEnvironment:
    bounds = np.tile(np.asarray([[-1.0, 1.0]]), (6, 1))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3] + values[:, 3:]

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        return np.hstack([np.eye(3), np.eye(3)])


def test_diversity_mode_runs_every_solver_after_dls_success() -> None:
    policy = CandidatePolicy(
        candidate_budget_per_node=2,
        difficult_candidate_budget_per_node=2,
        candidate_seed_budget_per_node=1,
        difficult_seed_budget_per_node=1,
        nullspace_seed_budget_per_node=0,
        search_mode=CandidateSearchMode.DIVERSITY,
        solver_names=("weighted_dls", "bounded_least_squares", "slsqp"),
        silver_margin_deg=0.25,
    )

    bank = solve_candidate_bank(
        _RedundantAffineEnvironment(),
        np.zeros((1, 3)),
        policy,
        node_seed_beta_rad={0: np.zeros(6, dtype=float)},
    )

    rows = bank.for_node(0)
    assert [row.solver for row in rows] == [
        "weighted_dls",
        "bounded_least_squares",
        "slsqp",
    ]
    assert bank.node_reports[0]["source_seed_count"] == 1
    assert bank.node_reports[0]["solver_attempt_count"] == 3
    assert all(row.quality is CandidateQuality.GOLD for row in rows)
    assert {"fun", "nit", "nfev"} <= rows[-1].diagnostics.keys()


def test_correction_mode_stops_after_first_accepted_solver() -> None:
    policy = CandidatePolicy(
        candidate_budget_per_node=2,
        difficult_candidate_budget_per_node=2,
        candidate_seed_budget_per_node=1,
        difficult_seed_budget_per_node=1,
        nullspace_seed_budget_per_node=0,
        search_mode=CandidateSearchMode.CORRECTION,
        solver_names=("weighted_dls", "bounded_least_squares", "slsqp"),
    )

    bank = solve_candidate_bank(
        _RedundantAffineEnvironment(),
        np.zeros((1, 3)),
        policy,
        node_seed_beta_rad={0: np.zeros(6, dtype=float)},
    )

    assert [row.solver for row in bank.for_node(0)] == ["weighted_dls"]
    assert bank.node_reports[0]["source_seed_count"] == 1
    assert bank.node_reports[0]["solver_attempt_count"] == 1


def test_positive_low_margin_remains_silver_sensitivity_not_invalid() -> None:
    environment = _RedundantAffineEnvironment()
    seed = np.zeros(6, dtype=float)
    seed[0] = 0.999
    target = environment.fk(seed.reshape(1, 6))
    policy = CandidatePolicy(
        candidate_seed_budget_per_node=1,
        difficult_seed_budget_per_node=1,
        nullspace_seed_budget_per_node=0,
        search_mode=CandidateSearchMode.CORRECTION,
        solver_names=("weighted_dls",),
        gold_margin_deg=1.5,
        silver_margin_deg=0.0,
    )

    row = solve_candidate_bank(
        environment,
        target,
        policy,
        node_seed_beta_rad={0: seed},
    ).for_node(0)[0]

    assert row.quality is CandidateQuality.SILVER
    assert row.min_margin_deg > 0.0
