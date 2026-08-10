from __future__ import annotations

import math

import numpy as np

from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_inverse import (
    refine_inverse_with_bounded_dls,
    InverseQuery,
    WorkspaceInverse,
    WorkspaceInversePolicy,
    WorkspaceRegistry,
)
from quasi_exp.teacher.workspace_reach import CellKey, WorkspaceGridSpec


BOUNDS = np.deg2rad(np.tile(np.asarray([[-10.0, 10.0]]), (6, 1)))


def _fk(beta: np.ndarray) -> np.ndarray:
    values = np.asarray(beta, dtype=float).reshape(-1, 6)
    return values[:, :3] + values[:, 3:]


def _constant(beta_deg: list[float]):
    value = np.deg2rad(np.asarray(beta_deg, dtype=float)).reshape(1, 6)

    def predict(features: np.ndarray) -> np.ndarray:
        return np.repeat(value, len(np.asarray(features)), axis=0)

    return predict


def _router(probabilities: list[float]):
    value = np.asarray(probabilities, dtype=float).reshape(1, -1)

    def predict(features: np.ndarray) -> np.ndarray:
        return np.repeat(value, len(np.asarray(features)), axis=0)

    return predict


def test_fk_tie_cannot_choose_between_nonstitchable_static_branches() -> None:
    inverse = WorkspaceInverse(
        mode=RepresentationMode.XYZ_ROUTER_EXPERTS,
        beta_bounds_rad=BOUNDS,
        fk=_fk,
        router_predictor=_router([0.6, 0.4]),
        expert_predictors={
            "chart_a": _constant([0, 0, 0, 0, 0, 0]),
            "chart_b": _constant([2, 0, 0, -2, 0, 0]),
        },
    )

    result = inverse.predict(InverseQuery(xyz_m=np.zeros((1, 3))))

    assert result.accepted.tolist() == [False]
    assert result.reasons == ("nonstitchable_static_ambiguity",)
    assert np.isnan(result.beta_rad[0]).all()


def test_fk_can_choose_a_more_accurate_candidate_inside_one_stitch_class() -> None:
    # Both candidates differ by much less than one degree; chart_b has zero FK
    # residual while chart_a misses the target by about 0.17 mm.
    inverse = WorkspaceInverse(
        mode=RepresentationMode.XYZ_ROUTER_EXPERTS,
        beta_bounds_rad=BOUNDS,
        fk=_fk,
        router_predictor=_router([0.6, 0.4]),
        expert_predictors={
            "chart_a": _constant([0.01, 0, 0, 0, 0, 0]),
            "chart_b": _constant([0, 0, 0, 0, 0, 0]),
        },
    )

    result = inverse.predict(InverseQuery(xyz_m=np.zeros((1, 3))))

    assert result.accepted.tolist() == [True]
    assert result.chart_ids == ("chart_b",)
    assert result.fk_residual_mm[0] == 0.0


def test_stateful_inverse_requires_previous_beta_and_preserves_nearest_branch() -> None:
    inverse = WorkspaceInverse(
        mode=RepresentationMode.STATEFUL_ROUTER_EXPERTS,
        beta_bounds_rad=BOUNDS,
        fk=_fk,
        router_predictor=_router([0.5, 0.5]),
        expert_predictors={
            "branch_a": _constant([0, 0, 0, 0, 0, 0]),
            "branch_b": _constant([2, 0, 0, -2, 0, 0]),
        },
    )

    missing = inverse.predict(InverseQuery(xyz_m=np.zeros((1, 3))))
    assert missing.accepted.tolist() == [False]
    assert missing.reasons == ("previous_beta_required",)

    previous = np.deg2rad(np.asarray([[1.9, 0, 0, -1.9, 0, 0]], dtype=float))
    result = inverse.predict(
        InverseQuery(xyz_m=np.zeros((1, 3)), previous_beta_rad=previous)
    )
    assert result.accepted.tolist() == [True]
    assert result.chart_ids == ("branch_b",)


def test_registry_abstains_outside_registered_cells_before_model_selection() -> None:
    grid = WorkspaceGridSpec(levels_mm=(20, 10, 5), x_slab_m=(1.0, 1.2))
    registry = WorkspaceRegistry(
        grid=grid,
        allowed_cells=frozenset({CellKey(10, 101, 0, 0)}),
    )
    inverse = WorkspaceInverse(
        mode=RepresentationMode.XYZ_GLOBAL,
        beta_bounds_rad=BOUNDS,
        fk=_fk,
        global_predictor=_constant([0, 0, 0, 0, 0, 0]),
        registry=registry,
        policy=WorkspaceInversePolicy(max_fk_residual_mm=3.0),
    )

    result = inverse.predict(InverseQuery(xyz_m=np.asarray([[1.05, 0.0, 0.0]])))

    assert result.accepted.tolist() == [False]
    assert result.reasons == ("outside_registered_domain",)


def test_global_inverse_rejects_out_of_bounds_output() -> None:
    inverse = WorkspaceInverse(
        mode=RepresentationMode.XYZ_GLOBAL,
        beta_bounds_rad=BOUNDS,
        fk=_fk,
        global_predictor=_constant([20, 0, 0, 0, 0, 0]),
    )

    result = inverse.predict(InverseQuery(xyz_m=np.zeros((1, 3))))

    assert result.accepted.tolist() == [False]
    assert result.reasons == ("prediction_out_of_bounds",)


def test_one_or_two_step_dls_fallback_improves_without_clipping() -> None:
    class Environment:
        bounds = np.tile(np.asarray([[-1.0, 1.0]]), (6, 1))

        @staticmethod
        def fk(beta: np.ndarray) -> np.ndarray:
            return np.asarray(beta, dtype=float).reshape(-1, 6)[:, :3]

        @staticmethod
        def jacobian(beta: np.ndarray) -> np.ndarray:
            del beta
            return np.hstack([np.eye(3), np.zeros((3, 3))])

    target = np.asarray([[0.02, -0.01, 0.005]])
    initial = np.zeros((1, 6))

    result = refine_inverse_with_bounded_dls(
        Environment(), target, initial, steps=1, acceptance_residual_mm=0.01
    )

    assert result.accepted.tolist() == [True]
    assert result.residual_mm[0] < 0.01
    assert np.all(result.beta_rad >= -1.0)
    assert np.all(result.beta_rad <= 1.0)
