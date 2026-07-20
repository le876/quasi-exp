"""Scale-normalised acceptance gate for learned trajectory tracking."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


TRACKING_GATE_ID = "relative-major-semiaxis-v1"
DEFAULT_TRACKING_RELATIVE_LIMIT = 0.02


def relative_tracking_gate_spec() -> dict[str, Any]:
    """Return the canonical metadata embedded in future experiment artifacts."""

    return {
        "gate_id": TRACKING_GATE_ID,
        "basis": "actual_major_semiaxis",
        "relative_limit": DEFAULT_TRACKING_RELATIVE_LIMIT,
        "aggregation": "trajectory_max",
    }


def require_relative_tracking_gate(payload: dict[str, Any], *, context: str) -> None:
    """Reject legacy caches instead of silently reinterpreting their outcome."""

    if payload.get("tracking_gate") != relative_tracking_gate_spec():
        raise RuntimeError(
            f"legacy or incompatible tracking gate in {context}; start the new "
            "relative-gate experiment in a fresh output root"
        )


def evaluate_relative_tracking_gate(
    residual_mm: Iterable[float] | np.ndarray,
    *,
    major_semiaxis_m: float,
    relative_limit: float = DEFAULT_TRACKING_RELATIVE_LIMIT,
) -> dict[str, Any]:
    """Evaluate a whole trajectory against a scale-relative error limit.

    The formal pass condition is fail-closed: every evaluated trajectory point
    must have Euclidean FK residual no greater than ``relative_limit`` times the
    *actual* major semiaxis.  Percentiles and the within-limit fraction remain
    diagnostics and never override a failed worst-point check.
    """

    residual = np.asarray(list(residual_mm), dtype=float).reshape(-1)
    semiaxis = float(major_semiaxis_m)
    limit = float(relative_limit)
    if residual.size == 0:
        raise ValueError("tracking residuals must not be empty")
    if not np.isfinite(semiaxis) or semiaxis <= 0.0:
        raise ValueError("major_semiaxis_m must be finite and positive")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("relative_limit must be finite and positive")
    if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
        raise ValueError("tracking residuals must be finite and non-negative")

    semiaxis_mm = semiaxis * 1000.0
    relative = residual / semiaxis_mm
    within = relative <= limit
    return {
        "tracking_gate_id": TRACKING_GATE_ID,
        "tracking_gate_aggregation": "trajectory_max",
        "major_semiaxis_m": semiaxis,
        "tracking_gate_relative_limit": limit,
        "tracking_gate_limit_pct": limit * 100.0,
        "tracking_gate_threshold_mm": semiaxis_mm * limit,
        "tracking_relative_error_p50_pct": float(np.percentile(relative, 50) * 100.0),
        "tracking_relative_error_p95_pct": float(np.percentile(relative, 95) * 100.0),
        "tracking_relative_error_max_pct": float(np.max(relative) * 100.0),
        "tracking_within_gate_rate": float(np.mean(within)),
        "tracking_gate_pass": bool(np.all(within)),
    }
