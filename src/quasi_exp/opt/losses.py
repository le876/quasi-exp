from __future__ import annotations

import numpy as np


def normalized_pseudo_huber(value: float | np.ndarray, scale: float, delta: float) -> float | np.ndarray:
    """Normalize an error by a task-specific scale and apply pseudo-Huber loss.

    The result is additionally normalized so that phi(scale) == 1. This makes
    the surrounding weights behave more like preference weights than raw numeric
    compensation terms.
    """
    safe_scale = float(scale) if np.isfinite(scale) and scale > 0.0 else 1.0
    safe_delta = float(delta) if np.isfinite(delta) and delta > 0.0 else 1.0

    z = np.abs(np.asarray(value, dtype=float)) / safe_scale
    base = (safe_delta**2) * (np.sqrt(1.0 + (1.0 / safe_delta) ** 2) - 1.0)
    if not np.isfinite(base) or base <= 0.0:
        base = 1.0
    out = (safe_delta**2) * (np.sqrt(1.0 + (z / safe_delta) ** 2) - 1.0) / base
    if np.isscalar(value):
        return float(out)
    return out


def theta_delta_rms_deg(beta_a: np.ndarray, beta_b: np.ndarray) -> float:
    """RMS delta in degrees over the 6 independent beta/theta dimensions."""
    a = np.asarray(beta_a, dtype=float).reshape(6)
    b = np.asarray(beta_b, dtype=float).reshape(6)
    rms_rad = float(np.sqrt(np.mean(np.square(a - b))))
    return float(np.rad2deg(rms_rad))
