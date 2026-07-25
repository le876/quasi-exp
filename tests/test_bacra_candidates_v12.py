from __future__ import annotations

import math

import numpy as np

from quasi_exp.teacher.multi_ik_candidates import (
    CandidatePolicy,
    CandidateQuality,
    solve_candidate_bank,
    stable_cluster_representatives,
)


class _RedundantAffineEnvironment:
    bounds = np.deg2rad(
        np.asarray([[-5.0, 5.0], [-5.0, 5.0], [-10.0, 10.0], [-10.0, 10.0], [-15.0, 15.0], [-15.0, 15.0]])
    )

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3] + values[:, 3:]

    def theta(self, beta: np.ndarray) -> np.ndarray:
        return np.zeros((len(np.asarray(beta).reshape(-1, 6)), 30))

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        return np.hstack([np.eye(3), np.eye(3)])


def test_candidate_bank_emits_gold_silver_and_reject_without_posthoc_clipping() -> None:
    env = _RedundantAffineEnvironment()
    target = np.asarray([[math.radians(6.0), 0.0, 0.0]])
    bank = solve_candidate_bank(
        env,
        target,
        CandidatePolicy(candidate_budget_per_node=4, difficult_candidate_budget_per_node=4),
        capability_beta_rad=np.asarray([
            [math.radians(3.0), 0.0, 0.0, math.radians(3.0), 0.0, 0.0],
            [math.radians(4.75), 0.0, 0.0, math.radians(1.25), 0.0, 0.0],
        ]),
    )
    rows = bank.for_node(0)
    assert any(row.quality is CandidateQuality.GOLD for row in rows)
    assert any(row.quality is CandidateQuality.SILVER for row in rows)
    assert all(np.all(row.beta_rad >= env.bounds[:, 0] - 1.0e-12) or row.quality is CandidateQuality.REJECT for row in rows)
    # An explicitly out-of-range neighbour seed remains a reject attempt; it
    # must never be quietly projected onto +5 degrees.
    rejected = solve_candidate_bank(env, np.zeros((1, 3)), neighbor_beta_rad={0: np.full((1, 6), math.radians(20.0))})
    assert any(row.diagnostics.get("status") == "seed_out_of_bounds" for row in rejected.for_node(0))


def test_stable_clustering_and_threshold_sensitivity_are_deterministic() -> None:
    beta = np.deg2rad(np.asarray([[0, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], dtype=float))
    representatives, indices = stable_cluster_representatives(beta, scores=np.asarray([2.0, 1.0, 3.0]), threshold_deg=0.5)
    assert indices == (1, 2)
    assert np.allclose(representatives[0], beta[1])
    bank = solve_candidate_bank(_RedundantAffineEnvironment(), np.zeros((1, 3)), capability_beta_rad=beta)
    sensitivity = bank.node_reports[0]["cluster_threshold_sensitivity"]
    assert set(sensitivity) == {"0.25", "0.5", "1"}


def test_nullspace_seeds_and_solver_diagnostics_are_recorded() -> None:
    env = _RedundantAffineEnvironment()
    capability = np.zeros((2, 6))
    capability[1, 3] = math.radians(2.0)
    bank = solve_candidate_bank(
        env,
        np.zeros((1, 3)),
        CandidatePolicy(candidate_budget_per_node=12, difficult_candidate_budget_per_node=12),
        capability_beta_rad=capability,
        difficult_node_ids=(0,),
    )
    rows = bank.for_node(0)
    assert any(row.source.startswith("nullspace_") for row in rows)
    bounded = [row for row in rows if row.solver == "bounded_least_squares"]
    assert bounded and all({"cost", "optimality", "nfev"} <= row.diagnostics.keys() for row in bounded if row.solver_success)
    assert all("sigma1_m" in row.diagnostics for row in rows if np.isfinite(row.min_margin_deg))
