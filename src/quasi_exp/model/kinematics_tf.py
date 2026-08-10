"""Differentiable TensorFlow forward kinematics for the registered beta6 robot."""

from __future__ import annotations

from typing import Any

import numpy as np


def _theta_index_map_30() -> np.ndarray:
    return np.asarray(
        [
            (section * 2) + (index % 2)
            for section in range(3)
            for index in range(10)
        ],
        dtype=np.int32,
    )


def _alpha_array_30() -> np.ndarray:
    values = np.empty(30, dtype=np.float64)
    for index in range(1, 31):
        values[index - 1] = (
            0.0
            if index == 1
            else (np.pi / 2.0 if index % 2 == 1 else -np.pi / 2.0)
        )
    return values


def forward_xyz_from_beta_tf(
    beta6: Any,
    *,
    lengths_m: Any,
    p_end_local_m: Any,
    theta_sign: float,
) -> Any:
    """Map an ``N×6`` beta tensor to ``N×3`` end-effector XYZ.

    The implementation matches :mod:`quasi_exp.model.kinematics` and remains
    differentiable with respect to ``beta6`` for use in FK-aware student losses.
    """

    import tensorflow as tf

    beta = tf.convert_to_tensor(beta6)
    if beta.shape.rank != 2 or beta.shape[-1] != 6:
        raise ValueError("beta6 must have shape (N, 6)")
    if not beta.dtype.is_floating:
        beta = tf.cast(beta, tf.float32)
    dtype = beta.dtype
    lengths = np.asarray(lengths_m, dtype=float).reshape(-1)
    endpoint = np.asarray(p_end_local_m, dtype=float).reshape(4)
    if len(lengths) != 31:
        raise ValueError("lengths_m must contain 31 registered segment lengths")

    theta_map = tf.constant(_theta_index_map_30(), dtype=tf.int32)
    theta = tf.gather(beta, theta_map, axis=1) * tf.cast(theta_sign, dtype)
    alpha = tf.constant(_alpha_array_30(), dtype=dtype)
    ca = tf.cos(alpha)
    sa = tf.sin(alpha)
    link = tf.constant(lengths[:30], dtype=dtype)
    endpoint_tf = tf.constant(endpoint, dtype=dtype)

    count = tf.shape(beta)[0]
    transform = tf.tile(tf.eye(4, dtype=dtype)[None, :, :], [count, 1, 1])
    zeros = tf.zeros((count,), dtype=dtype)
    ones = tf.ones((count,), dtype=dtype)
    for index in range(30):
        ct = tf.cos(theta[:, index])
        st = tf.sin(theta[:, index])
        row0 = tf.stack([ct, -st, zeros, tf.fill(tf.shape(ct), link[index])], axis=1)
        row1 = tf.stack(
            [st * ca[index], ct * ca[index], tf.fill(tf.shape(ct), -sa[index]), zeros],
            axis=1,
        )
        row2 = tf.stack(
            [st * sa[index], ct * sa[index], tf.fill(tf.shape(ct), ca[index]), zeros],
            axis=1,
        )
        row3 = tf.stack([zeros, zeros, zeros, ones], axis=1)
        transform = tf.matmul(transform, tf.stack([row0, row1, row2, row3], axis=1))
    return tf.linalg.matvec(transform, endpoint_tf)[:, :3]
