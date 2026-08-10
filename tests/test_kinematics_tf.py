from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")
if not all(hasattr(tf, name) for name in ("zeros", "Variable", "GradientTape", "keras")):
    pytest.skip("TensorFlow namespace is incomplete", allow_module_level=True)

from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf
from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.sampling import beta_to_theta


def test_tensorflow_fk_maps_zero_beta_to_known_straight_chain() -> None:
    beta = tf.zeros((2, 6), dtype=tf.float32)
    lengths = np.full(31, 0.01, dtype=np.float32)
    endpoint = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    xyz = forward_xyz_from_beta_tf(
        beta,
        lengths_m=lengths,
        p_end_local_m=endpoint,
        theta_sign=-1.0,
    ).numpy()

    np.testing.assert_allclose(xyz, [[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], atol=1e-6)


def test_tensorflow_fk_has_finite_beta_gradient() -> None:
    beta = tf.Variable([[0.01, -0.02, 0.03, -0.04, 0.05, -0.06]])
    lengths = np.linspace(0.01, 0.03, 31, dtype=np.float32)
    endpoint = np.array([0.02, -0.01, 0.03, 1.0], dtype=np.float32)

    with tf.GradientTape() as tape:
        xyz = forward_xyz_from_beta_tf(
            beta,
            lengths_m=lengths,
            p_end_local_m=endpoint,
            theta_sign=-1.0,
        )
        loss = tf.reduce_sum(tf.square(xyz))
    gradient = tape.gradient(loss, beta)

    assert gradient is not None
    assert gradient.shape == (1, 6)
    assert np.isfinite(gradient.numpy()).all()


def test_tensorflow_fk_matches_authoritative_numpy_fk() -> None:
    beta = np.array(
        [
            [0.01, -0.02, 0.03, -0.04, 0.05, -0.06],
            [-0.04, 0.03, -0.02, 0.01, -0.06, 0.05],
        ],
        dtype=np.float32,
    )
    lengths = np.linspace(0.01, 0.04, 31, dtype=np.float32)
    endpoint = np.array([0.02, -0.01, 0.03, 1.0], dtype=np.float32)
    expected = np.vstack(
        [
            forward_kinematics(
                beta_to_theta(row),
                lengths,
                endpoint,
                theta_sign=-1.0,
            )[0]
            for row in beta
        ]
    )

    actual = forward_xyz_from_beta_tf(
        beta,
        lengths_m=lengths,
        p_end_local_m=endpoint,
        theta_sign=-1.0,
    ).numpy()

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
