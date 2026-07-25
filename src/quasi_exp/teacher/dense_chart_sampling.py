"""Dense sampling inside a validated BACRA inverse chart.

The module deliberately owns the complete dense-label decision.  Callers
provide sparse chart nodes and an authoritative forward environment; they do
not reimplement predictor/corrector or admission checks in a runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

from .canonical import TeacherPolicy, _correct_target, beta_rms_deg, weighted_damped_pinv


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("x_m", "y_m", "z_m")


class DenseForwardEnvironment(Protocol):
    @property
    def bounds(self) -> np.ndarray: ...

    def fk(self, beta_rad: np.ndarray) -> np.ndarray: ...

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DenseSamplingPolicy:
    """Frozen V12 dense-label policy."""

    row_count: int
    attempt_multiplier: int = 3
    dual_anchor_gap_deg: float = 0.5
    residual_max_mm: float = 3.0
    gold_margin_deg: float = 1.5
    seed: int = 20260735
    candidate_policy: TeacherPolicy = TeacherPolicy()

    def __post_init__(self) -> None:
        if int(self.row_count) < 1:
            raise ValueError("row_count must be positive")
        if int(self.attempt_multiplier) < 1:
            raise ValueError("attempt_multiplier must be positive")
        for name in ("dual_anchor_gap_deg", "residual_max_mm", "gold_margin_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class DenseSamplingResult:
    rows: pd.DataFrame
    attempts: pd.DataFrame
    report: Mapping[str, Any]


def _as_chart_arrays(
    xyz_m: np.ndarray, beta_rad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_m, dtype=float)
    beta = np.asarray(beta_rad, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4:
        raise ValueError("chart xyz_m must have shape (N, 3), N >= 4")
    if beta.shape != (len(xyz), 6):
        raise ValueError(f"chart beta_rad must have shape ({len(xyz)}, 6)")
    if not np.isfinite(xyz).all() or not np.isfinite(beta).all():
        raise ValueError("chart arrays must be finite")
    return xyz, beta


def weighted_beta_gap_deg(
    left: np.ndarray,
    right: np.ndarray,
    *,
    weights: Sequence[float] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
) -> float:
    """Weighted RMS beta gap used by chart agreement checks."""

    delta = np.asarray(left, dtype=float).reshape(6) - np.asarray(
        right, dtype=float
    ).reshape(6)
    weight = np.asarray(weights, dtype=float).reshape(6)
    if np.any(weight <= 0.0):
        raise ValueError("weights must be positive")
    return float(
        np.rad2deg(np.sqrt(np.sum(weight * np.square(delta)) / np.sum(weight)))
    )


def _predict_from_anchor(
    environment: DenseForwardEnvironment,
    target_xyz_m: np.ndarray,
    anchor_xyz_m: np.ndarray,
    anchor_beta_rad: np.ndarray,
    policy: TeacherPolicy,
) -> np.ndarray:
    jacobian = np.asarray(environment.jacobian(anchor_beta_rad), dtype=float).reshape(3, 6)
    pinv = weighted_damped_pinv(
        jacobian,
        damping=float(policy.damping),
        weights=np.asarray(policy.beta_weights, dtype=float),
    )
    prediction = anchor_beta_rad + pinv @ (target_xyz_m - anchor_xyz_m)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    return np.clip(prediction, bounds[:, 0], bounds[:, 1])


def _margin_deg(beta_rad: np.ndarray, bounds_rad: np.ndarray) -> float:
    beta = np.asarray(beta_rad, dtype=float).reshape(6)
    bounds = np.asarray(bounds_rad, dtype=float).reshape(6, 2)
    return float(
        np.rad2deg(
            np.min(np.minimum(beta - bounds[:, 0], bounds[:, 1] - beta))
        )
    )


def _sample_barycentric(
    vertices: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    # Exponential variates give a uniform Dirichlet(1,1,1,1) draw.
    weights = rng.exponential(size=4)
    weights /= np.sum(weights)
    return weights @ vertices, weights


def _empty_rows() -> pd.DataFrame:
    columns = [
        "sample_id",
        "chart_id",
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
        "residual_mm",
        "joint_margin_deg",
        "dual_anchor_gap_deg",
        "anchor_a_idx",
        "anchor_b_idx",
        "tetrahedron_idx",
        "accepted",
        "reason",
    ]
    return pd.DataFrame(columns=columns)


def densify_chart(
    environment: DenseForwardEnvironment,
    chart_xyz_m: np.ndarray,
    chart_beta_rad: np.ndarray,
    *,
    chart_id: int | str,
    policy: DenseSamplingPolicy,
) -> DenseSamplingResult:
    """Generate strict Gold labels inside the convex hull of a sparse chart.

    Two geometrically independent sparse anchors predict and independently
    correct every target.  A row is admitted only if both corrections converge,
    agree, satisfy the hard FK residual and retain the registered Gold margin.
    No failed attempt is padded into the returned dataset.
    """

    xyz, beta = _as_chart_arrays(chart_xyz_m, chart_beta_rad)
    try:
        tessellation = Delaunay(xyz)
    except Exception as exc:
        return DenseSamplingResult(
            rows=_empty_rows(),
            attempts=pd.DataFrame(
                [{"accepted": False, "reason": f"delaunay_failed:{type(exc).__name__}"}]
            ),
            report={
                "gate_pass": False,
                "reason": "chart_has_no_3d_tessellation",
                "requested_rows": int(policy.row_count),
                "accepted_rows": 0,
                "attempt_count": 0,
                "acceptance_ratio": 0.0,
            },
        )
    simplices = np.asarray(tessellation.simplices, dtype=np.int64)
    if len(simplices) == 0:
        raise ValueError("chart tessellation contains no tetrahedra")
    volumes = np.abs(
        np.linalg.det(
            xyz[simplices[:, 1:]] - xyz[simplices[:, :1]]
        )
    ) / 6.0
    valid = np.isfinite(volumes) & (volumes > np.finfo(float).eps)
    simplices = simplices[valid]
    volumes = volumes[valid]
    if len(simplices) == 0:
        raise ValueError("chart tessellation contains no non-degenerate tetrahedra")
    probabilities = volumes / np.sum(volumes)
    rng = np.random.default_rng(int(policy.seed))
    tree = cKDTree(xyz)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    limit = int(policy.row_count) * int(policy.attempt_multiplier)

    for attempt_idx in range(limit):
        tetrahedron_idx = int(rng.choice(len(simplices), p=probabilities))
        vertex_ids = simplices[tetrahedron_idx]
        target, _weights = _sample_barycentric(xyz[vertex_ids], rng)
        nearest = np.asarray(tree.query(target, k=min(len(xyz), 8))[1], dtype=np.int64).reshape(-1)
        anchor_a_idx = int(nearest[0])
        distances = np.linalg.norm(xyz[nearest] - xyz[anchor_a_idx], axis=1)
        anchor_b_idx = int(nearest[int(np.argmax(distances))])

        corrected: list[np.ndarray] = []
        residuals: list[float] = []
        successes: list[bool] = []
        iterations: list[int] = []
        for anchor_idx in (anchor_a_idx, anchor_b_idx):
            initial = _predict_from_anchor(
                environment,
                target,
                xyz[anchor_idx],
                beta[anchor_idx],
                policy.candidate_policy,
            )
            value, residual_mm, iteration_count, success = _correct_target(
                environment, target, initial, policy.candidate_policy
            )
            corrected.append(value)
            residuals.append(float(residual_mm))
            iterations.append(int(iteration_count))
            successes.append(bool(success))

        gap_deg = weighted_beta_gap_deg(
            corrected[0],
            corrected[1],
            weights=policy.candidate_policy.beta_weights,
        )
        chosen_index = int(np.argmin(residuals))
        chosen = corrected[chosen_index]
        achieved = np.asarray(environment.fk(chosen.reshape(1, 6)), dtype=float)[0]
        verified_residual_mm = float(np.linalg.norm(achieved - target) * 1000.0)
        margin_deg = _margin_deg(chosen, bounds)

        reason = "accepted"
        if not all(successes):
            reason = "corrector_failed"
        elif gap_deg > float(policy.dual_anchor_gap_deg):
            reason = "dual_anchor_disagreement"
        elif verified_residual_mm > float(policy.residual_max_mm):
            reason = "fk_residual"
        elif margin_deg < float(policy.gold_margin_deg):
            reason = "gold_margin"
        elif np.any(chosen < bounds[:, 0]) or np.any(chosen > bounds[:, 1]):
            reason = "bounds"
        is_accepted = reason == "accepted"
        row: dict[str, Any] = {
            "sample_id": f"{chart_id}:{attempt_idx:08d}",
            "chart_id": chart_id,
            "x_m": float(target[0]),
            "y_m": float(target[1]),
            "z_m": float(target[2]),
            **{name: float(chosen[index]) for index, name in enumerate(BETA_COLUMNS)},
            "residual_mm": verified_residual_mm,
            "joint_margin_deg": margin_deg,
            "dual_anchor_gap_deg": gap_deg,
            "anchor_a_idx": anchor_a_idx,
            "anchor_b_idx": anchor_b_idx,
            "tetrahedron_idx": tetrahedron_idx,
            "corrector_a_iterations": iterations[0],
            "corrector_b_iterations": iterations[1],
            "accepted": bool(is_accepted),
            "reason": reason,
        }
        attempts.append(row)
        if is_accepted:
            accepted.append(row)
            if len(accepted) >= int(policy.row_count):
                break

    attempt_frame = pd.DataFrame(attempts)
    row_frame = pd.DataFrame(accepted)
    if row_frame.empty:
        row_frame = _empty_rows()
    attempt_count = len(attempt_frame)
    acceptance_ratio = float(len(row_frame) / attempt_count) if attempt_count else 0.0
    reason_counts = (
        attempt_frame["reason"].value_counts().sort_index().astype(int).to_dict()
        if attempt_count
        else {}
    )
    report = {
        "gate_pass": bool(len(row_frame) == int(policy.row_count)),
        "requested_rows": int(policy.row_count),
        "accepted_rows": int(len(row_frame)),
        "attempt_count": int(attempt_count),
        "acceptance_ratio": acceptance_ratio,
        "reason_counts": reason_counts,
        "residual_p95_mm": (
            float(np.percentile(row_frame["residual_mm"], 95))
            if len(row_frame)
            else float("inf")
        ),
        "residual_max_mm": (
            float(row_frame["residual_mm"].max()) if len(row_frame) else float("inf")
        ),
        "joint_margin_min_deg": (
            float(row_frame["joint_margin_deg"].min())
            if len(row_frame)
            else float("-inf")
        ),
        "dual_anchor_gap_p95_deg": (
            float(np.percentile(row_frame["dual_anchor_gap_deg"], 95))
            if len(row_frame)
            else float("inf")
        ),
    }
    return DenseSamplingResult(rows=row_frame, attempts=attempt_frame, report=report)


def chart_fill_distance_metrics(
    sample_xyz_m: np.ndarray, reference_xyz_m: np.ndarray
) -> dict[str, float]:
    """Return registered nearest-neighbour fill-distance diagnostics in mm."""

    sample = np.asarray(sample_xyz_m, dtype=float)
    reference = np.asarray(reference_xyz_m, dtype=float)
    if sample.ndim != 2 or sample.shape[1] != 3 or len(sample) == 0:
        raise ValueError("sample_xyz_m must have non-empty shape (N, 3)")
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
        raise ValueError("reference_xyz_m must have non-empty shape (M, 3)")
    distance_m = cKDTree(sample).query(reference, k=1)[0]
    distance_mm = distance_m * 1000.0
    return {
        "nn_p50_mm": float(np.percentile(distance_mm, 50)),
        "nn_p95_mm": float(np.percentile(distance_mm, 95)),
        "nn_max_mm": float(np.max(distance_mm)),
    }


def local_label_consistency(
    xyz_m: np.ndarray,
    beta_rad: np.ndarray,
    *,
    radii_mm: Sequence[float] = (5.0, 10.0),
) -> dict[str, float]:
    """Measure local beta discontinuity without assuming a trajectory order."""

    xyz, beta = _as_chart_arrays(xyz_m, beta_rad)
    tree = cKDTree(xyz)
    output: dict[str, float] = {}
    for radius_mm in radii_mm:
        gaps: list[float] = []
        for index, neighbours in enumerate(
            tree.query_ball_point(xyz, r=float(radius_mm) / 1000.0)
        ):
            for other in neighbours:
                if int(other) <= index:
                    continue
                gaps.append(beta_rms_deg(beta[index], beta[int(other)]))
        key = f"local_{float(radius_mm):g}mm_beta_gap_p95_deg"
        output[key] = float(np.percentile(gaps, 95)) if gaps else 0.0
    return output
