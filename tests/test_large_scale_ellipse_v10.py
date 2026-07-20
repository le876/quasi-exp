import numpy as np

from quasi_exp.teacher.large_scale import (
    REGISTERED_MAJOR_AXIS_GAIN,
    EllipseChallenge,
    ReachabilityAtlas,
    assess_chain_length_necessity,
    fit_ellipse_pose_to_atlas,
)


def test_requested_major_semiaxes_are_not_confused_with_legacy_radius_parameter():
    challenge = EllipseChallenge.from_major_semiaxis_m(0.5)

    assert np.isclose(REGISTERED_MAJOR_AXIS_GAIN, 1.953403629, atol=1.0e-9)
    assert np.isclose(challenge.radius_parameter_m, 0.255963485, atol=1.0e-9)
    assert np.isclose(challenge.major_diameter_m, 1.0, atol=1.0e-12)
    assert np.isclose(challenge.minor_semiaxis_m, 0.168666975, atol=1.0e-9)


def test_all_three_user_challenges_remain_in_the_chain_length_report():
    challenges = tuple(
        EllipseChallenge.from_major_semiaxis_m(value)
        for value in (0.5, 0.75, 1.0)
    )

    report = assess_chain_length_necessity(
        challenges,
        robot_max_reach_m=1.215498,
        tube_radius_m=0.001,
    )

    assert [row["major_semiaxis_m"] for row in report] == [0.5, 0.75, 1.0]
    assert [row["major_diameter_m"] for row in report] == [1.0, 1.5, 2.0]
    assert all(row["translation_independent_diameter_gate_pass"] for row in report)
    assert np.isclose(report[-1]["diameter_margin_m"], 0.430996, atol=1.0e-12)
    assert np.isclose(report[-1]["tube_diameter_margin_m"], 0.428996, atol=1.0e-12)


def test_target_generator_realises_the_requested_geometric_axes():
    challenge = EllipseChallenge.from_major_semiaxis_m(0.5)
    target = challenge.generate_targets(
        center_m=np.asarray([0.7, 0.0, 0.0]),
        major_direction=np.asarray([0.0, 1.0, 0.0]),
        minor_direction=np.asarray([0.0, 0.0, 1.0]),
        phase_count=4,
    )

    expected = np.asarray(
        [
            [0.7, 0.5, 0.0],
            [0.7, 0.0, 0.168666975],
            [0.7, -0.5, 0.0],
            [0.7, 0.0, -0.168666975],
        ]
    )
    assert np.allclose(target, expected, atol=1.0e-9)


def test_target_generator_rejects_non_orthogonal_axes():
    challenge = EllipseChallenge.from_major_semiaxis_m(0.5)

    with np.testing.assert_raises_regex(ValueError, "orthogonal"):
        challenge.generate_targets(
            center_m=np.zeros(3),
            major_direction=np.asarray([1.0, 0.0, 0.0]),
            minor_direction=np.asarray([1.0, 1.0, 0.0]),
            phase_count=24,
        )


def test_reachability_atlas_returns_the_initial_path_and_zero_gap_for_known_curve():
    challenge = EllipseChallenge.from_major_semiaxis_m(0.5)
    target = challenge.generate_targets(
        center_m=np.asarray([0.7, 0.0, 0.0]),
        major_direction=np.asarray([0.0, 1.0, 0.0]),
        minor_direction=np.asarray([0.0, 0.0, 1.0]),
        phase_count=8,
    )
    beta = np.arange(48, dtype=float).reshape(8, 6) / 100.0
    atlas = ReachabilityAtlas(xyz_m=target, beta_rad=beta)

    match = atlas.match_targets(target)

    assert np.allclose(match.initial_beta_path_rad, beta)
    assert np.allclose(match.nearest_distance_mm, 0.0, atol=1.0e-12)
    assert match.metrics == {
        "nearest_distance_p95_mm": 0.0,
        "nearest_distance_max_mm": 0.0,
    }


def test_pose_search_finds_a_known_reachable_ellipse_in_a_synthetic_atlas():
    challenge = EllipseChallenge.from_major_semiaxis_m(0.1)
    exact = challenge.generate_targets(
        center_m=np.asarray([0.7, 0.0, 0.0]),
        major_direction=np.asarray([0.0, 1.0, 0.0]),
        minor_direction=np.asarray([0.0, 0.0, 1.0]),
        phase_count=96,
    )
    beta = np.zeros((len(exact), 6), dtype=float)
    atlas = ReachabilityAtlas(xyz_m=exact, beta_rad=beta)

    fit = fit_ellipse_pose_to_atlas(
        challenge,
        atlas,
        phase_count=24,
        seed=7,
        max_iterations=80,
        population_size=10,
    )

    assert fit.match.metrics["nearest_distance_p95_mm"] <= 5.0
    assert fit.match.metrics["nearest_distance_max_mm"] <= 5.0
    assert np.isclose(fit.center_m[0], 0.7, atol=5.0e-3)
