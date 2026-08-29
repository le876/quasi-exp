from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry15_candidate_solver import (
    Retry15CandidatePolicy,
    _constrained_least_squares,
    seed_bank_xyz,
    solve_target_candidates,
    stable_seed_sequence,
    teacher_seed_bank,
    weighted_linear_seed,
)
from quasi_exp.teacher.retry15_canonical_graph import (
    OMEGA600_X0_M,
    OMEGA600_X_MIN_M,
    Omega600Contract,
    ProbePolicy,
    build_target_registry,
    candidate_attempt_upper_bound,
    capped_profile,
    legal_candidate_clusters,
    objective_feasibility_documents,
    select_t1,
    select_t2,
    slope_clip,
    workspace_probe,
)


class LinearEnvironment:
    bounds = np.asarray([[-0.5, 0.5]] * 6, dtype=float)

    @staticmethod
    def fk(beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
        return np.column_stack(
            [OMEGA600_X0_M - beta[:, 0], beta[:, 1], beta[:, 2]]
        )

    @staticmethod
    def jacobian(beta_rad: np.ndarray) -> np.ndarray:
        del beta_rad
        result = np.zeros((3, 6), dtype=float)
        result[0, 0] = -1.0
        result[1, 1] = 1.0
        result[2, 2] = 1.0
        return result


def _profile(radius_mm: float = 130.0) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "u_index": np.arange(60),
            "u_center_mm": np.arange(60) * 10.0 + 5.0,
            "radius_mm": np.minimum(np.arange(60) * 20.0, radius_mm),
        }
    )
    result.loc[0, "radius_mm"] = 0.0
    return result


def _candidate_frame() -> pd.DataFrame:
    rows = []
    for target, x_m in (("a", 1.0), ("b", 0.99)):
        for ordinal, beta0 in enumerate((0.05, 0.25)):
            rows.append(
                {
                    "target_id": target,
                    "candidate_id": f"{target}:{ordinal}",
                    "x_m": x_m,
                    "y_m": 0.0,
                    "z_m": 0.0,
                    **dict(zip(BETA_COLUMNS, [beta0, 0, 0, 0, 0, 0], strict=True)),
                    "solver_success": True,
                    "bounds_pass": True,
                    "fk_residual_mm": 0.1,
                    "min_margin_deg": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_omega600_contract_is_exact_and_rejects_old_200mm_interval() -> None:
    contract = Omega600Contract()
    assert contract.x0_m == OMEGA600_X0_M
    assert contract.x_min_m == OMEGA600_X_MIN_M
    assert 1000.0 * (contract.x0_m - contract.x_min_m) == pytest.approx(600.0)
    with pytest.raises(ValueError, match="600 mm"):
        Omega600Contract(x_min_m=contract.x0_m - 0.2, length_mm=200.0)


def test_slope_clip_anchors_zero_and_enforces_twenty_mm_limit() -> None:
    clipped = slope_clip([100, 100, 100, 10], limit_mm=20.0)
    assert clipped[0] == 0.0
    assert np.max(np.abs(np.diff(clipped))) <= 20.0


def test_two_pool_probe_never_counts_single_pool_voxels_as_robust() -> None:
    x0 = OMEGA600_X0_M
    # One common point per full sector in the first axial/radial cell.
    phase = np.arange(16) * (2 * np.pi / 16) + 0.01
    common = np.column_stack(
        [np.full(16, x0 - 0.005), 0.005 * np.cos(phase), 0.005 * np.sin(phase)]
    )
    only_a = np.asarray([[x0 - 0.015, 0.015, 0.0]])
    support, profile = workspace_probe(
        np.vstack([common, only_a]),
        common,
        policy=ProbePolicy(radial_margin_mm=0.0, allow_single_voxel_closing=False),
    )
    single = support.loc[support["u_index"].eq(1)]
    assert not single["robust_supported"].any()
    assert profile.loc[0, "radius_mm"] == 0.0


def test_target_registry_keeps_axis_and_nine_large_circle_panel() -> None:
    profile = _profile(130.0)
    proposals = np.asarray(
        [[OMEGA600_X0_M - u / 1000.0, y / 1000.0, z / 1000.0]
         for u in np.arange(5, 600, 10)
         for y in (0, 50, 100)
         for z in (0, 50, 100)],
        dtype=float,
    )
    registry, edges, circles, summary = build_target_registry(
        profile, proposals, target_budget=5000
    )
    assert summary["complete_600mm_axis_core"]
    assert summary["registered_circle_count"] == 9
    assert summary["large_circle_count"] == 3
    assert registry.loc[registry["target_role"].eq("heldout_circle"), "mandatory"].all()
    assert set(edges["edge_type"]) == {"circle"}
    assert circles.groupby("circle_id")["held_out"].all().all()


def test_teacher_seed_banks_are_nested_and_seam_constraints_are_exact() -> None:
    environment = LinearEnvironment()
    bank = teacher_seed_bank(
        environment.bounds,
        size=128,
        seed=20260897,
        bank_id="retry15_teacher_full",
    )
    with_xyz = seed_bank_xyz(environment, bank)
    ids8, seeds8 = stable_seed_sequence(
        environment,
        np.asarray([OMEGA600_X0_M - 0.05, 0.02, 0.0]),
        with_xyz,
        policy=Retry15CandidatePolicy(seed_budget=8),
    )
    ids16, seeds16 = stable_seed_sequence(
        environment,
        np.asarray([OMEGA600_X0_M - 0.05, 0.02, 0.0]),
        with_xyz,
        policy=Retry15CandidatePolicy(seed_budget=16),
    )
    assert ids16[:8] == ids8
    assert np.array_equal(seeds16[:8], seeds8)
    # z=0 fixes beta2/beta4/beta6 in one-based notation.
    assert np.array_equal(seeds16[:, [1, 3, 5]], np.zeros((16, 3)))


def test_weighted_linear_seed_and_candidate_solver_do_not_need_proposal_beta() -> None:
    environment = LinearEnvironment()
    target = np.asarray([OMEGA600_X0_M - 0.05, 0.02, 0.03])
    linear = weighted_linear_seed(environment, target)
    assert np.linalg.norm(environment.fk(linear)[0] - target) < 1.0e-4
    bank = seed_bank_xyz(
        environment,
        teacher_seed_bank(
            environment.bounds,
            size=128,
            seed=20260897,
            bank_id="retry15_teacher_full",
        ),
    )
    candidates = solve_target_candidates(
        environment,
        {"target_id": "target", "target_role": "interior", **dict(zip(XYZ_COLUMNS, target, strict=True))},
        bank,
        policy=Retry15CandidatePolicy(seed_budget=8),
    )
    assert len(candidates) == 16
    assert not candidates["proposal_beta_used"].any()
    assert candidates["fk_residual_mm"].min() <= 3.0


def test_least_squares_clips_tolerated_dls_boundary_roundoff() -> None:
    environment = LinearEnvironment()
    seed = np.zeros(6, dtype=float)
    seed[0] = environment.bounds[0, 1] + 5.0e-13
    target = environment.fk(np.asarray([[environment.bounds[0, 1], 0, 0, 0, 0, 0]]))[0]
    beta, diagnostic = _constrained_least_squares(
        environment,
        target,
        seed,
        free=np.arange(6, dtype=int),
        policy=Retry15CandidatePolicy(seed_budget=8),
    )
    assert diagnostic["success"]
    assert beta[0] <= environment.bounds[0, 1]
    assert np.linalg.norm(environment.fk(beta)[0] - target) * 1000.0 <= 3.0


def test_candidate_legality_ignores_margin_as_hard_filter() -> None:
    frame = _candidate_frame()
    frame.loc[0, "min_margin_deg"] = -0.1
    legal = legal_candidate_clusters(frame)
    assert frame.loc[0, "candidate_id"] in set(legal["candidate_id"])


def test_t1_and_t2_select_only_existing_candidate_ids() -> None:
    candidates = _candidate_frame()
    edges = pd.DataFrame(
        [{"edge_id": "edge", "left_target_id": "a", "right_target_id": "b", "edge_type": "circle"}]
    )
    t1 = select_t1(candidates)
    t2_zero = select_t2(candidates, edges, pairwise_lambda=0.0)
    t2 = select_t2(candidates, edges, pairwise_lambda=4.0)
    assert set(t1["candidate_id"]) == set(t2_zero["candidate_id"])
    assert set(t2["candidate_id"]) <= set(candidates["candidate_id"])
    assert t2["target_id"].is_unique


def test_t2_mst_dynamic_programming_escapes_unary_icm_trap() -> None:
    rows = []
    candidates_by_target = {
        "a": ([0.05, 0, 0, 0, 0, 0], [0, 0.08, 0, 0, 0, 0]),
        "b": ([-0.05, 0, 0, 0, 0, 0], [0, 0.08, 0, 0, 0, 0]),
    }
    for target, candidate_betas in candidates_by_target.items():
        for ordinal, beta in enumerate(candidate_betas):
            rows.append(
                {
                    "target_id": target,
                    "candidate_id": f"{target}:{ordinal}",
                    **dict(zip(BETA_COLUMNS, beta, strict=True)),
                    "solver_success": True,
                    "bounds_pass": True,
                    "fk_residual_mm": 0.1,
                    "min_margin_deg": 1.0,
                }
            )
    candidates = pd.DataFrame(rows)
    edges = pd.DataFrame(
        [
            {
                "edge_id": "edge",
                "left_target_id": "a",
                "right_target_id": "b",
                "edge_type": "circle",
                "distance_mm": 1.0,
            }
        ]
    )
    assert set(select_t1(candidates)["candidate_id"]) == {"a:0", "b:0"}
    selected = select_t2(candidates, edges, pairwise_lambda=1.0)
    assert set(selected["candidate_id"]) == {"a:1", "b:1"}


def test_objective_feasibility_separates_target_and_solver_budgets() -> None:
    target_registry = {
        "path": "02_domain/targets.parquet",
        "sha256": "a" * 64,
        "id_column": "target_id",
        "row_count": 100,
    }
    circle_registry = {
        "path": "02_domain/circles.parquet",
        "sha256": "b" * 64,
        "id_column": "circle_id",
        "row_count": 9,
    }
    documents = objective_feasibility_documents(
        experiment_id="retry15_test",
        scientific_source_sha="c" * 40,
        config_sha256="d" * 64,
        target_registry=target_registry,
        circle_registry=circle_registry,
        target_budget=100,
        solver_attempt_budget=candidate_attempt_upper_bound(
            target_count=100, seed_budget=16, difficult_target_count=0
        ),
        candidate_seed_budget=16,
        difficult_target_count=0,
        source_artifacts=[{"path": "upstream.json", "sha256": "e" * 64}],
    )
    assert documents["gate.json"]["claim_bearing_run_authorized"]
    assert documents["budget_lower_bound.json"]["resources"]["solver_attempts"]["optimistic_minimum"] == 3200

    blocked = objective_feasibility_documents(
        experiment_id="retry15_test",
        scientific_source_sha="c" * 40,
        config_sha256="d" * 64,
        target_registry=target_registry,
        circle_registry=circle_registry,
        target_budget=99,
        solver_attempt_budget=3199,
        candidate_seed_budget=16,
        difficult_target_count=0,
        source_artifacts=[{"path": "upstream.json", "sha256": "e" * 64}],
    )
    assert blocked["gate.json"]["status"] == "diagnostic_only"
    assert not blocked["gate.json"]["claim_bearing_run_authorized"]


def test_teacher_target_schema_rejects_beta_columns() -> None:
    profile = capped_profile(_profile(), ceiling_mm=120, shrink_mm=0, slope_limit_mm=20)
    assert not any(column.startswith("be") for column in profile.columns)
    with pytest.raises(ValueError, match="must not contain beta"):
        from quasi_exp.teacher.retry15_candidate_solver import solve_candidate_shard

        targets = pd.DataFrame(
            [{"target_id": "bad", "x_m": 1.0, "y_m": 0.0, "z_m": 0.0, "be1_rad": 0.0}]
        )
        bank = pd.DataFrame(columns=["seed_id", *BETA_COLUMNS, *XYZ_COLUMNS])
        solve_candidate_shard(
            LinearEnvironment(), targets, bank, policy=Retry15CandidatePolicy(seed_budget=8)
        )
