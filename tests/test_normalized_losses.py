from __future__ import annotations

import math

from quasi_exp.opt.losses import normalized_pseudo_huber, theta_delta_rms_deg


def test_normalized_pseudo_huber_anchor_and_monotonic() -> None:
    assert math.isclose(normalized_pseudo_huber(0.0, scale=2.0, delta=0.5), 0.0, abs_tol=1e-12)
    assert math.isclose(normalized_pseudo_huber(2.0, scale=2.0, delta=0.5), 1.0, rel_tol=1e-9)
    assert normalized_pseudo_huber(0.5, scale=2.0, delta=0.5) < normalized_pseudo_huber(2.0, scale=2.0, delta=0.5)
    assert normalized_pseudo_huber(6.0, scale=2.0, delta=0.5) > normalized_pseudo_huber(2.0, scale=2.0, delta=0.5)
    assert normalized_pseudo_huber(6.0, scale=2.0, delta=0.5) < 9.0


def test_theta_delta_rms_deg_matches_expected_units() -> None:
    deg6 = math.radians(6.0)
    beta_a = [0.0] * 6
    beta_b = [deg6] * 6
    assert math.isclose(theta_delta_rms_deg(beta_a, beta_b), 6.0, rel_tol=1e-9)
