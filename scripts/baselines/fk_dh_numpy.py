#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def _alpha_rad(i):
    # i is 1-based joint index
    if i == 1:
        return 0.0
    return (np.pi / 2.0) if (i % 2 == 1) else (-np.pi / 2.0)


def fk_dh_batch(theta_rad, lengths_m, p_end_local_m):
    """
    Batch forward kinematics for this repo's DH convention.

    Parameters
    ----------
    theta_rad : (N, kD) array
        Joint angles already in the same sign convention as the dataset (do NOT apply theta_sign again).
    lengths_m : (kD+1,) array
        Link lengths [l0..lkD].
    p_end_local_m : (4,) array
        Homogeneous end point in frame {kD}, usually [l_kD,0,0,1].

    Returns
    -------
    p_xyz : (N,3) array
        End point coordinates in base frame {0}.
    """
    theta = np.asarray(theta_rad, dtype=float)
    if theta.ndim != 2:
        raise ValueError("theta_rad must be 2D (N,kD)")
    n, kD = theta.shape
    lengths = np.asarray(lengths_m, dtype=float).reshape(-1)
    if lengths.shape[0] != kD + 1:
        raise ValueError("lengths_m must have shape (kD+1,)")
    p_end = np.asarray(p_end_local_m, dtype=float).reshape(4)

    # T: (N,4,4)
    T = np.tile(np.eye(4, dtype=float), (n, 1, 1))

    for i in range(1, kD + 1):
        alpha = float(_alpha_rad(i))
        a = float(lengths[i - 1])
        d = 0.0

        ca = np.cos(alpha)
        sa = np.sin(alpha)

        th = theta[:, i - 1]
        ct = np.cos(th)
        st = np.sin(th)

        # Build A_i: (N,4,4)
        A = np.zeros((n, 4, 4), dtype=float)
        A[:, 0, 0] = ct
        A[:, 0, 1] = -st
        A[:, 0, 3] = a

        A[:, 1, 0] = st * ca
        A[:, 1, 1] = ct * ca
        A[:, 1, 2] = -sa
        A[:, 1, 3] = -d * sa

        A[:, 2, 0] = st * sa
        A[:, 2, 1] = ct * sa
        A[:, 2, 2] = ca
        A[:, 2, 3] = d * ca

        A[:, 3, 3] = 1.0

        # T = T @ A (batch)
        T = np.einsum("nij,njk->nik", T, A, optimize=True)

    p0 = np.einsum("nij,j->ni", T, p_end, optimize=True)
    return p0[:, :3].copy()

