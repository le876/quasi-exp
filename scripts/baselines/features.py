#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple

import numpy as np


def build_features(X_xyz: np.ndarray, feature_set: str = "poly_heavy", eps: float = 1e-8) -> Tuple[np.ndarray, list[str]]:
    X_xyz = np.asarray(X_xyz, dtype=float)
    if X_xyz.ndim != 2 or X_xyz.shape[1] != 3:
        raise ValueError("X_xyz must have shape [N, 3]")

    x = X_xyz[:, 0]
    y = X_xyz[:, 1]
    z = X_xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)

    if feature_set == "raw":
        feats = [x, y, z]
        names = ["x_m", "y_m", "z_m"]
        return np.stack(feats, axis=1), names

    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z
    r2 = r * r

    if feature_set == "poly_medium":
        feats = [x, y, z, r, xy, xz, yz, x2, y2, z2, r2]
        names = [
            "x_m",
            "y_m",
            "z_m",
            "r_m",
            "xy_m2",
            "xz_m2",
            "yz_m2",
            "x2_m2",
            "y2_m2",
            "z2_m2",
            "r2_m2",
        ]
        return np.stack(feats, axis=1), names

    if feature_set != "poly_heavy":
        raise ValueError(f"unsupported feature_set: {feature_set}")

    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    rho_xy = np.sqrt(x2 + y2)
    cos_theta = z / (r + eps)
    sin_theta = rho_xy / (r + eps)
    sin_phi = y / (rho_xy + eps)
    cos_phi = x / (rho_xy + eps)
    x_over_r = x / (r + eps)
    y_over_r = y / (r + eps)
    z_over_r = z / (r + eps)

    feats = [
        x,
        y,
        z,
        r,
        xy,
        xz,
        yz,
        x2,
        y2,
        z2,
        r2,
        x3,
        y3,
        z3,
        rho_xy,
        cos_theta,
        sin_theta,
        sin_phi,
        cos_phi,
        x_over_r,
        y_over_r,
        z_over_r,
    ]
    names = [
        "x_m",
        "y_m",
        "z_m",
        "r_m",
        "xy_m2",
        "xz_m2",
        "yz_m2",
        "x2_m2",
        "y2_m2",
        "z2_m2",
        "r2_m2",
        "x3_m3",
        "y3_m3",
        "z3_m3",
        "rho_xy_m",
        "cos_theta",
        "sin_theta",
        "sin_phi",
        "cos_phi",
        "x_over_r",
        "y_over_r",
        "z_over_r",
    ]
    return np.stack(feats, axis=1), names

