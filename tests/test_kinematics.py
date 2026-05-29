from __future__ import annotations

import numpy as np

from quasi_exp.model.kinematics import forward_kinematics


def test_straight_pose_x_matches_chain_length() -> None:
    kD = 30
    lengths = np.ones(kD + 1, dtype=float) * 0.1
    theta = np.zeros(kD, dtype=float)
    p_end = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    p_xyz, _T = forward_kinematics(theta, lengths, p_end)
    expected_x = float(np.sum(lengths[:kD]))
    assert np.all(np.isfinite(p_xyz))
    assert abs(float(p_xyz[0]) - expected_x) < 1e-9
    assert abs(float(p_xyz[1])) < 1e-9
    assert abs(float(p_xyz[2])) < 1e-9


def test_rotation_orthonormal() -> None:
    kD = 30
    lengths = np.ones(kD + 1, dtype=float) * 0.05
    rng = np.random.default_rng(0)
    theta = rng.uniform(-0.5, 0.5, size=(kD,))
    _p, T_0_from_i = forward_kinematics(theta, lengths, None)
    for i in [0, 1, 2, 10, 30]:
        R = T_0_from_i[i][:3, :3]
        err = np.max(np.abs(R.T @ R - np.eye(3)))
        assert err < 1e-6

