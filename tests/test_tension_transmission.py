from __future__ import annotations

import numpy as np

from quasi_exp.model.tension_transmission import transmit_tensions


def test_mu0_reduces_to_cos_ratio() -> None:
    theta = np.array([0.2, -0.1, 0.3] + [0.0] * 27, dtype=float)
    T_base = np.ones(12, dtype=float) * 100.0
    mu = 0.0
    end_disk_by_j = {j: 3 for j in range(1, 13)}
    L0 = np.ones(12, dtype=float)
    L = np.ones(12, dtype=float) * 2.0  # force Case1, but mu=0 两种 case 等价

    F, _case = transmit_tensions(theta, T_base, mu, end_disk_by_j, L0, L)
    # theta0=0 => F1 = F0 * cos(0)/cos(theta1/2)
    expected_F1 = T_base / np.cos(theta[0] / 2.0)
    assert np.allclose(F[1], expected_F1, rtol=0, atol=1e-12)

