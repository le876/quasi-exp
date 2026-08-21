from __future__ import annotations

import numpy as np

from quasi_exp.model.endpoint_kinematics import EndpointKinematics
from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.forward import ForwardEnvironment
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    make_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_continuation import (
    make_iterative_weighted_dls_continuation,
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_forward import optimized_forward


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linspace(0.008, 0.012, 31, dtype=float)
    endpoint = np.asarray([0.0, 0.0, lengths[-1], 1.0], dtype=float)
    return lengths, endpoint


def test_batch_fk_matches_reference_scalar_dh() -> None:
    lengths, endpoint = _inputs()
    rng = np.random.default_rng(20260812)
    beta = rng.uniform(-0.18, 0.18, size=(32, 6))
    expected = np.vstack(
        [
            forward_kinematics(
                beta_to_theta(row), lengths, endpoint, theta_sign=-1.0
            )[0]
            for row in beta
        ]
    )
    actual = EndpointKinematics(lengths, endpoint, theta_sign=-1.0).fk(beta)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


def test_analytic_jacobian_matches_central_difference_near_bounds() -> None:
    lengths, endpoint = _inputs()
    reference = ForwardEnvironment(
        lengths,
        endpoint,
        theta_sign=-1.0,
        beta_bounds_rad=np.tile(np.asarray([-0.2, 0.2]), (6, 1)),
    )
    environment = optimized_forward(reference)
    beta_rows = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.199, -0.199, 0.17, -0.18, 0.19, -0.16],
            [-0.198, 0.197, -0.15, 0.16, -0.19, 0.18],
        ],
        dtype=float,
    )
    for beta in beta_rows:
        analytic = environment.jacobian(beta)
        numerical = reference.numerical_jacobian(beta, eps_rad=1.0e-6)
        np.testing.assert_allclose(analytic, numerical, rtol=3.0e-7, atol=3.0e-9)


def test_evaluate_returns_fk_and_jacobian_in_one_batch() -> None:
    lengths, endpoint = _inputs()
    beta = np.zeros((5, 6), dtype=float)
    evaluation = EndpointKinematics(lengths, endpoint).evaluate(beta, jacobian=True)
    assert evaluation.xyz_m.shape == (5, 3)
    assert evaluation.jacobian_m is not None
    assert evaluation.jacobian_m.shape == (5, 3, 6)
    assert np.isfinite(evaluation.xyz_m).all()
    assert np.isfinite(evaluation.jacobian_m).all()


def test_optimized_environment_counts_fk_and_jacobian_rows() -> None:
    lengths, endpoint = _inputs()
    reference = ForwardEnvironment(
        lengths,
        endpoint,
        theta_sign=-1.0,
        beta_bounds_rad=np.tile(np.asarray([-0.2, 0.2]), (6, 1)),
    )
    environment = optimized_forward(reference)

    environment.fk(np.zeros((3, 6)))
    environment.fk_and_jacobian(np.zeros(6))

    assert environment.performance_counters() == {
        "fk_row_count": 4,
        "jacobian_row_count": 1,
        "kinematics_cache_hit_count": 0,
        "kinematics_cache_miss_count": 1,
        "kinematics_cache_entry_count": 1,
    }


def test_optimized_forward_cache_reuses_only_pure_kinematics() -> None:
    lengths, endpoint = _inputs()
    reference = ForwardEnvironment(
        lengths,
        endpoint,
        theta_sign=-1.0,
        beta_bounds_rad=np.tile(np.asarray([-0.2, 0.2]), (6, 1)),
    )
    environment = optimized_forward(reference)
    beta = np.zeros(6, dtype=float)

    first_xyz, first_jacobian = environment.fk_and_jacobian(beta)
    second_xyz, second_jacobian = environment.fk_and_jacobian(beta.copy())
    fk_xyz = environment.fk(beta.copy())

    np.testing.assert_array_equal(first_xyz, second_xyz)
    np.testing.assert_array_equal(first_jacobian, second_jacobian)
    np.testing.assert_array_equal(first_xyz, fk_xyz)
    counters = environment.performance_counters()
    assert counters["kinematics_cache_miss_count"] == 1
    assert counters["kinematics_cache_hit_count"] == 2
    assert counters["kinematics_cache_entry_count"] == 1


def test_explicit_jacobian_preserves_continuation_classification() -> None:
    lengths, endpoint = _inputs()
    reference = ForwardEnvironment(
        lengths,
        endpoint,
        theta_sign=-1.0,
        beta_bounds_rad=np.tile(np.asarray([-0.2, 0.2]), (6, 1)),
    )
    optimized = optimized_forward(reference)
    source_beta = np.zeros(6, dtype=float)
    target_beta = np.asarray([0.01, -0.008, 0.006, -0.004, 0.003, -0.002])
    source = AtlasCandidate(
        node_id=0,
        candidate_id="source",
        beta_rad=source_beta,
        residual_mm=0.0,
        min_margin_deg=5.0,
        normalized_min_margin=0.5,
    )
    target = AtlasTaskNode(1, reference.fk(target_beta)[0], (0,))

    historical = make_predictor_corrector_continuation(reference)(source, target)
    analytic = make_optimized_predictor_corrector_continuation(optimized)(source, target)

    assert historical.success is True
    assert analytic.success is True
    assert historical.actual_bounds is True
    assert analytic.actual_bounds is True
    assert historical.residual_mm <= 3.0
    assert analytic.residual_mm <= 3.0
    np.testing.assert_allclose(
        optimized.fk(analytic.beta_rad),
        reference.fk(historical.beta_rad),
        rtol=0.0,
        atol=3.0e-3,
    )


def test_iterative_weighted_dls_is_a_distinct_registered_kernel() -> None:
    lengths, endpoint = _inputs()
    reference = ForwardEnvironment(
        lengths,
        endpoint,
        theta_sign=-1.0,
        beta_bounds_rad=np.tile(np.asarray([-0.2, 0.2]), (6, 1)),
    )
    environment = optimized_forward(reference)
    source = AtlasCandidate(
        node_id=0,
        candidate_id="source",
        beta_rad=np.zeros(6),
        residual_mm=0.0,
        min_margin_deg=5.0,
        normalized_min_margin=0.5,
    )
    target_beta = np.asarray([0.006, -0.005, 0.004, -0.003, 0.002, -0.001])
    target = AtlasTaskNode(1, environment.fk(target_beta)[0], (0,))

    outcome = make_iterative_weighted_dls_continuation(environment)(source, target)

    assert outcome.success is True
    assert outcome.residual_mm <= 3.0
    assert outcome.status == "weighted_dls_converged"
