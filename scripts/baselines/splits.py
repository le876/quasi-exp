#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def split_iid(n, seed, val_size, test_size):
    """
    IID random split.

    Returns: train_idx, val_idx, test_idx (np.int64 arrays)
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")
    val_size = float(val_size)
    test_size = float(test_size)
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1.0:
        raise ValueError("val_size+test_size must be in [0,1)")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("train split would be empty")

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def split_radius_holdout(X_xyz_m, val_size, test_size):
    """
    Workspace split by end-effector radius r=||p|| (OOD-style):
      - test: largest radii
      - val: next largest radii
      - train: remaining (smaller radii)

    Returns: train_idx, val_idx, test_idx (np.int64 arrays)
    """
    X = np.asarray(X_xyz_m, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X_xyz_m must have shape (N,3)")
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("n must be > 0")

    val_size = float(val_size)
    test_size = float(test_size)
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1.0:
        raise ValueError("val_size+test_size must be in [0,1)")

    r = np.linalg.norm(X, axis=1)
    idx = np.argsort(r).astype(np.int64)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("train split would be empty")

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def split_beta_block_holdout(beta6_rad, val_size, test_size):
    """
    OOD split by reduced joint-angle magnitude.

    The largest normalized 6D beta magnitudes are held out for test; the next
    largest block is validation. This tests extrapolation toward joint-limit
    combinations rather than IID interpolation.
    """
    B = np.asarray(beta6_rad, dtype=float)
    if B.ndim != 2 or B.shape[1] != 6:
        raise ValueError("beta6_rad must have shape (N,6)")
    n = int(B.shape[0])
    if n <= 0:
        raise ValueError("n must be > 0")

    val_size = float(val_size)
    test_size = float(test_size)
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1.0:
        raise ValueError("val_size+test_size must be in [0,1)")

    scale = np.maximum(np.max(np.abs(B), axis=0), 1e-12)
    beta_norm = np.sqrt(np.mean(np.square(B / scale[None, :]), axis=1))
    idx = np.argsort(beta_norm).astype(np.int64)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("train split would be empty")
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def split_angular_sector_holdout(X_xyz_m, val_size, test_size):
    """
    OOD split by contiguous end-effector direction sectors in the yz plane.

    Test holds out the highest contiguous angular sector and validation holds
    the adjacent sector. This checks directional workspace generalization.
    """
    X = np.asarray(X_xyz_m, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X_xyz_m must have shape (N,3)")
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("n must be > 0")

    val_size = float(val_size)
    test_size = float(test_size)
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1.0:
        raise ValueError("val_size+test_size must be in [0,1)")

    angle = np.arctan2(X[:, 2], X[:, 1])
    idx = np.argsort(angle).astype(np.int64)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("train split would be empty")
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
