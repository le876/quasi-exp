from __future__ import annotations

import numpy as np
import pytest

from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.forward import ForwardEnvironment


def _environment(*, bounds: np.ndarray | None = None) -> ForwardEnvironment:
    return ForwardEnvironment(
        lengths_m=np.full(31, 0.04, dtype=float),
        p_end_local_m=np.array([0.01, -0.02, 0.03, 1.0], dtype=float),
        theta_sign=-1.0,
        beta_bounds_rad=bounds,
    )


def test_fk_batch_matches_authoritative_scalar_forward_kinematics() -> None:
    environment = _environment()
    beta = np.deg2rad(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, -2.0, 0.5, 0.25, -3.0, 2.0],
            ]
        )
    )

    actual = environment.fk(beta)
    expected = np.vstack(
        [
            forward_kinematics(
                beta_to_theta(row),
                environment.lengths_m,
                environment.p_end_local_m,
                theta_sign=environment.theta_sign,
            )[0]
            for row in beta
        ]
    )

    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_forward_environment_exposes_the_canonical_teacher_adapter_interface() -> None:
    bounds = np.deg2rad(np.tile(np.array([[-5.0, 5.0]]), (6, 1)))
    environment = _environment(bounds=bounds)
    beta = np.zeros((2, 6), dtype=float)

    np.testing.assert_array_equal(environment.bounds, bounds)
    assert environment.theta(beta).shape == (2, 30)
    assert environment.jacobian(beta[0]).shape == (3, 6)


def test_fk_rejects_non_six_dimensional_or_non_finite_beta() -> None:
    environment = _environment()

    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        environment.fk(np.zeros((2, 5), dtype=float))
    with pytest.raises(ValueError, match="finite"):
        environment.fk(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, np.nan]]))


def test_numerical_jacobian_predicts_a_small_exact_fk_displacement() -> None:
    environment = _environment()
    beta = np.deg2rad(np.array([1.0, -1.5, 0.5, 0.25, -2.0, 1.0]))
    delta = np.array([2.0, -1.0, 0.5, 1.5, -0.25, 0.75]) * 1.0e-6

    jacobian = environment.numerical_jacobian(beta, eps_rad=1.0e-5)
    predicted = jacobian @ delta
    exact = environment.fk(beta + delta)[0] - environment.fk(beta)[0]

    assert jacobian.shape == (3, 6)
    np.testing.assert_allclose(predicted, exact, rtol=2.0e-4, atol=1.0e-10)


def test_jacobian_metrics_report_ordered_singular_values_and_conditioning() -> None:
    environment = _environment()
    jacobian = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    metrics = environment.jacobian_metrics(jacobian)

    assert metrics == {
        "sigma1_m": 3.0,
        "sigma2_m": 2.0,
        "sigma3_m": 1.0,
        "kappa": 3.0,
    }


def test_validate_jacobian_returns_replayable_per_scale_accuracy() -> None:
    environment = _environment()
    beta = np.deg2rad(
        np.array(
            [
                [1.0, -1.5, 0.5, 0.25, -2.0, 1.0],
                [-2.0, 1.0, 1.5, -0.5, 2.5, -1.0],
                [0.25, 0.5, -1.0, 1.25, -0.75, 1.5],
            ]
        )
    )

    result = environment.validate_jacobian(
        beta,
        perturbation_scales_rad=(1.0e-6, 1.0e-4),
        validation_scale_rad=1.0e-4,
        jacobian_eps_rad=1.0e-5,
        direction_seed=17,
        relative_error_p95_limit=0.01,
        absolute_error_p95_m_limit=5.0e-5,
    )
    replay = environment.validate_jacobian(
        beta,
        perturbation_scales_rad=(1.0e-6, 1.0e-4),
        validation_scale_rad=1.0e-4,
        jacobian_eps_rad=1.0e-5,
        direction_seed=17,
        relative_error_p95_limit=0.01,
        absolute_error_p95_m_limit=5.0e-5,
    )

    assert result.passed
    assert result.sample_count == 3
    assert result.valid_count == 3
    assert len(result.per_scale) == 2
    assert result.per_scale == replay.per_scale
    assert result.metrics == replay.metrics
    assert result.metrics["validation_scale_rad"] == 1.0e-4
    assert result.metrics["relative_error_p95"] <= 0.01
    assert result.metrics["absolute_error_p95_m"] <= 5.0e-5


def test_validate_jacobian_fails_when_the_selected_scale_misses_limits() -> None:
    environment = _environment()
    beta = np.deg2rad(np.array([[2.0, -1.0, 1.5, 0.5, -2.5, 1.0]]))

    result = environment.validate_jacobian(
        beta,
        perturbation_scales_rad=(1.0e-4,),
        validation_scale_rad=1.0e-4,
        direction_seed=3,
        relative_error_p95_limit=0.0,
        absolute_error_p95_m_limit=0.0,
    )

    assert not result.passed
    assert result.metrics["relative_error_p95"] > 0.0
    assert result.metrics["absolute_error_p95_m"] > 0.0


def test_evaluate_synthetic_recovery_accepts_exact_in_bounds_solutions() -> None:
    bounds = np.deg2rad(np.tile(np.array([[-5.0, 5.0]]), (6, 1)))
    environment = _environment(bounds=bounds)
    source = np.deg2rad(
        np.array(
            [
                [1.0, -1.0, 0.5, 0.25, -2.0, 1.0],
                [-2.0, 1.5, -0.5, 1.0, 2.0, -1.5],
            ]
        )
    )

    result = environment.evaluate_synthetic_recovery(source, source.copy())

    assert result.passed
    assert result.sample_count == 2
    assert result.valid_count == 2
    assert result.metrics["success_rate"] == 1.0
    assert result.metrics["residual_p95_m"] == 0.0
    assert result.metrics["residual_max_m"] == 0.0
    assert result.metrics["bounds_violation_count"] == 0.0


def test_evaluate_synthetic_recovery_reports_solver_and_bounds_failures() -> None:
    bounds = np.deg2rad(np.tile(np.array([[-5.0, 5.0]]), (6, 1)))
    environment = _environment(bounds=bounds)
    source = np.zeros((2, 6), dtype=float)
    recovered = source.copy()
    recovered[1, 0] = np.deg2rad(6.0)

    result = environment.evaluate_synthetic_recovery(
        source,
        recovered,
        solver_success=np.array([True, False]),
    )

    assert not result.passed
    assert result.sample_count == 2
    assert result.valid_count == 1
    assert result.metrics["success_rate"] == 0.5
    assert result.metrics["bounds_violation_count"] == 1.0
