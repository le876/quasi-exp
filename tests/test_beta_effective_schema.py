from __future__ import annotations

import numpy as np

from quasi_exp.model.sampling import beta_to_theta, effective_beta_from_theta


def test_effective_beta_from_theta_round_trips_beta_to_theta() -> None:
    beta = np.array([0.01, -0.02, 0.03, -0.04, 0.08, -0.09], dtype=float)
    theta = beta_to_theta(beta).reshape(1, 30)

    got = effective_beta_from_theta(theta)

    assert got.shape == (1, 6)
    assert np.max(np.abs(got[0] - beta)) < 1.0e-12
    assert np.max(np.abs(beta_to_theta(got[0]) - theta[0])) < 1.0e-12


def test_effective_beta_from_theta_rejects_non_theta30_matrix() -> None:
    bad = np.zeros((2, 29), dtype=float)

    try:
        effective_beta_from_theta(bad)
    except ValueError as exc:
        assert "shape [N, 30]" in str(exc)
    else:
        raise AssertionError("expected ValueError")
