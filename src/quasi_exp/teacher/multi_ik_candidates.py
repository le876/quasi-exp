"""Deterministic multi-candidate inverse kinematics for BACRA-V12.

This module is deliberately an in-memory seam.  It turns a finite target set
and optional capability/graph anchors into auditable candidate banks without
letting callers reimplement seed ordering, bound handling, or Gold/Silver/
Reject classification.  Persistence belongs to the V12 runner.

There is intentionally no post-hoc clipping in this file.  A seed outside the
registered mechanical bounds is recorded as a reject, and DLS backtracks a
proposed step until it is legal instead of projecting it onto a joint limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .canonical import ForwardEnvironment, beta_rms_deg, weighted_damped_pinv


class CandidateQuality(str, Enum):
    GOLD = "Gold"
    SILVER = "Silver"
    REJECT = "Reject"


@dataclass(frozen=True)
class CandidatePolicy:
    """Frozen numerical and admission policy for a V12 candidate bank."""

    capability_nearest_count: int = 64
    capability_representative_count: int = 8
    candidate_budget_per_node: int = 16
    difficult_candidate_budget_per_node: int = 32
    gold_candidate_target: int = 4
    beta_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)
    damping: float = 1.0e-3
    max_rms_step_deg: float = 3.0
    max_corrector_iterations: int = 100
    tracking_tolerance_mm: float = 1.0
    max_residual_mm: float = 3.0
    gold_margin_deg: float = 1.5
    candidate_cluster_deg: float = 0.5
    cluster_sensitivity_deg: tuple[float, ...] = (0.25, 0.5, 1.0)
    nullspace_step_deg: float = 1.0
    solver_seed: int = 20260732

    def __post_init__(self) -> None:
        if self.capability_nearest_count < 1 or self.capability_representative_count < 1:
            raise ValueError("capability counts must be positive")
        if self.candidate_budget_per_node < 1 or self.difficult_candidate_budget_per_node < 1:
            raise ValueError("candidate budgets must be positive")
        if self.gold_candidate_target < 1:
            raise ValueError("gold_candidate_target must be positive")
        if len(self.beta_weights) != 6 or any(float(value) <= 0.0 for value in self.beta_weights):
            raise ValueError("beta_weights must contain six positive entries")
        if self.damping < 0.0 or self.max_rms_step_deg <= 0.0:
            raise ValueError("damping must be non-negative and max step positive")
        if self.max_corrector_iterations < 1 or self.tracking_tolerance_mm <= 0.0:
            raise ValueError("corrector iterations and tracking tolerance must be positive")
        if self.max_residual_mm < self.tracking_tolerance_mm:
            raise ValueError("max_residual_mm must not be less than tracking_tolerance_mm")
        if self.gold_margin_deg <= 0.0 or self.candidate_cluster_deg <= 0.0:
            raise ValueError("margin and cluster thresholds must be positive")
        if not self.cluster_sensitivity_deg or any(float(value) <= 0.0 for value in self.cluster_sensitivity_deg):
            raise ValueError("cluster_sensitivity_deg must contain positive thresholds")
        if self.nullspace_step_deg <= 0.0:
            raise ValueError("nullspace_step_deg must be positive")


@dataclass(frozen=True)
class CandidateRecord:
    node_id: int
    candidate_id: str
    source: str
    solver: str
    seed_rank: int
    beta_rad: np.ndarray
    achieved_xyz_m: np.ndarray
    residual_mm: float
    min_margin_deg: float
    normalized_min_margin: float
    quality: CandidateQuality
    solver_success: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta_rad, dtype=float).reshape(6)
        xyz = np.asarray(self.achieved_xyz_m, dtype=float).reshape(3)
        object.__setattr__(self, "beta_rad", beta.copy())
        object.__setattr__(self, "achieved_xyz_m", xyz.copy())
        object.__setattr__(self, "quality", CandidateQuality(self.quality))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class CandidateBank:
    """Stable candidate records plus task-level evidence and sensitivity."""

    records: tuple[CandidateRecord, ...]
    node_reports: Mapping[int, Mapping[str, Any]]
    policy: CandidatePolicy

    def for_node(self, node_id: int) -> tuple[CandidateRecord, ...]:
        return tuple(record for record in self.records if record.node_id == int(node_id))

    @property
    def gold_node_ratio(self) -> float:
        if not self.node_reports:
            return 0.0
        return float(np.mean([bool(report["gold_candidate_count"]) for report in self.node_reports.values()]))


def _bounds(environment: ForwardEnvironment) -> np.ndarray:
    bounds = np.asarray(environment.bounds, dtype=float)
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("environment.bounds must be finite shape (6, 2) ordered bounds")
    return bounds


def _jacobian(environment: ForwardEnvironment, beta: np.ndarray) -> np.ndarray:
    method = getattr(environment, "jacobian", None) or getattr(environment, "numerical_jacobian", None)
    if method is None:
        raise TypeError("environment must provide jacobian or numerical_jacobian")
    value = np.asarray(method(np.asarray(beta, dtype=float).reshape(6)), dtype=float)
    if value.shape != (3, 6) or not np.isfinite(value).all():
        raise ValueError("environment Jacobian must be finite shape (3, 6)")
    return value


def _inside_bounds(beta: np.ndarray, bounds: np.ndarray, *, atol: float = 1.0e-12) -> bool:
    values = np.asarray(beta, dtype=float).reshape(6)
    return bool(np.isfinite(values).all() and np.all(values >= bounds[:, 0] - atol) and np.all(values <= bounds[:, 1] + atol))


def _margin_metrics(beta: np.ndarray, bounds: np.ndarray) -> tuple[float, float]:
    margin = np.minimum(beta - bounds[:, 0], bounds[:, 1] - beta)
    half_span = np.maximum(0.5 * (bounds[:, 1] - bounds[:, 0]), np.finfo(float).eps)
    return (
        float(np.min(margin) * 180.0 / math.pi),
        float(np.min(margin / half_span)),
    )


def _fk_one(environment: ForwardEnvironment, beta: np.ndarray) -> np.ndarray:
    xyz = np.asarray(environment.fk(np.asarray(beta, dtype=float).reshape(1, 6)), dtype=float)
    return xyz.reshape(-1, 3)[0]


def _stable_cluster_indices(beta: np.ndarray, scores: np.ndarray, threshold_deg: float) -> tuple[int, ...]:
    """Greedy stable representatives ordered by (score, original index)."""
    rows = np.asarray(beta, dtype=float).reshape(-1, 6)
    values = np.asarray(scores, dtype=float).reshape(-1)
    order = sorted(range(len(rows)), key=lambda index: (float(values[index]), int(index)))
    kept: list[int] = []
    for index in order:
        if any(beta_rms_deg(rows[index], rows[previous]) < float(threshold_deg) for previous in kept):
            continue
        kept.append(index)
    return tuple(kept)


def stable_cluster_representatives(
    beta_rad: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    threshold_deg: float = 0.5,
    max_count: int | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return deterministic beta-space cluster representatives and source indices."""
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    if not len(beta) or not np.isfinite(beta).all():
        raise ValueError("beta_rad must be non-empty and finite with shape (N, 6)")
    if threshold_deg <= 0.0:
        raise ValueError("threshold_deg must be positive")
    values = np.arange(len(beta), dtype=float) if scores is None else np.asarray(scores, dtype=float).reshape(-1)
    if len(values) != len(beta) or not np.isfinite(values).all():
        raise ValueError("scores must be finite and align with beta_rad")
    indices = _stable_cluster_indices(beta, values, float(threshold_deg))
    if max_count is not None:
        indices = indices[: max(0, int(max_count))]
    return beta[np.asarray(indices, dtype=int)].copy(), indices


def _dls_correct(
    environment: ForwardEnvironment,
    target: np.ndarray,
    seed: np.ndarray,
    policy: CandidatePolicy,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Bound-respecting weighted DLS; rejected steps are backtracked, never clipped."""
    bounds = _bounds(environment)
    beta = np.asarray(seed, dtype=float).reshape(6).copy()
    if not _inside_bounds(beta, bounds):
        return beta, {"success": False, "status": "seed_out_of_bounds", "iterations": 0}
    limit = math.radians(float(policy.max_rms_step_deg))
    tolerance = float(policy.tracking_tolerance_mm) / 1000.0
    previous_residual = float("inf")
    for iteration in range(int(policy.max_corrector_iterations)):
        achieved = _fk_one(environment, beta)
        error = np.asarray(target, dtype=float).reshape(3) - achieved
        residual = float(np.linalg.norm(error))
        if residual <= tolerance:
            return beta, {"success": True, "status": "converged", "iterations": iteration, "residual_mm": residual * 1000.0}
        jac = _jacobian(environment, beta)
        step = weighted_damped_pinv(jac, damping=policy.damping, weights=policy.beta_weights) @ error
        rms = float(np.sqrt(np.mean(np.square(step))))
        if rms > limit:
            step *= limit / rms
        accepted = False
        factor = 1.0
        for _ in range(24):
            proposed = beta + factor * step
            if _inside_bounds(proposed, bounds):
                proposed_residual = float(np.linalg.norm(_fk_one(environment, proposed) - np.asarray(target).reshape(3)))
                if proposed_residual <= residual + 1.0e-15:
                    beta = proposed
                    accepted = True
                    previous_residual = proposed_residual
                    break
            factor *= 0.5
        if not accepted:
            return beta, {"success": False, "status": "bounds_or_descent_blocked", "iterations": iteration + 1, "residual_mm": residual * 1000.0}
        if previous_residual >= residual - 1.0e-15 and float(np.linalg.norm(step)) <= 1.0e-14:
            return beta, {"success": False, "status": "stagnated", "iterations": iteration + 1, "residual_mm": residual * 1000.0}
    residual = float(np.linalg.norm(_fk_one(environment, beta) - np.asarray(target).reshape(3)))
    return beta, {"success": False, "status": "max_iterations", "iterations": int(policy.max_corrector_iterations), "residual_mm": residual * 1000.0}


def _bounded_least_squares(
    environment: ForwardEnvironment,
    target: np.ndarray,
    seed: np.ndarray,
    policy: CandidatePolicy,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    bounds = _bounds(environment)
    initial = np.asarray(seed, dtype=float).reshape(6)
    if not _inside_bounds(initial, bounds):
        return initial.copy(), {"success": False, "status": "seed_out_of_bounds", "nfev": 0}

    def residual(beta: np.ndarray) -> np.ndarray:
        return (_fk_one(environment, beta) - np.asarray(target, dtype=float).reshape(3)) / 0.001

    result = least_squares(
        residual,
        initial,
        bounds=(bounds[:, 0], bounds[:, 1]),
        max_nfev=int(policy.max_corrector_iterations),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    beta = np.asarray(result.x, dtype=float).reshape(6)
    return beta, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
    }


def _quality(
    beta: np.ndarray,
    achieved: np.ndarray,
    target: np.ndarray,
    bounds: np.ndarray,
    solver_success: bool,
    policy: CandidatePolicy,
) -> tuple[CandidateQuality, float, float, float, str | None]:
    if not _inside_bounds(beta, bounds):
        return CandidateQuality.REJECT, float("nan"), float("nan"), float("nan"), "actual_bounds_failed"
    if not np.isfinite(achieved).all():
        return CandidateQuality.REJECT, float("nan"), float("nan"), float("nan"), "nonfinite_achieved_xyz"
    residual_mm = float(np.linalg.norm(np.asarray(achieved) - np.asarray(target)) * 1000.0)
    margin, normalized_margin = _margin_metrics(beta, bounds)
    if not solver_success:
        return CandidateQuality.REJECT, residual_mm, margin, normalized_margin, "solver_not_converged"
    if residual_mm > float(policy.max_residual_mm):
        return CandidateQuality.REJECT, residual_mm, margin, normalized_margin, "residual_above_max"
    if margin >= float(policy.gold_margin_deg):
        return CandidateQuality.GOLD, residual_mm, margin, normalized_margin, None
    if margin > 0.0:
        return CandidateQuality.SILVER, residual_mm, margin, normalized_margin, None
    return CandidateQuality.REJECT, residual_mm, margin, normalized_margin, "zero_or_negative_margin"


def _candidate_record(
    *,
    environment: ForwardEnvironment,
    node_id: int,
    ordinal: int,
    source: str,
    solver: str,
    seed_rank: int,
    target: np.ndarray,
    seed: np.ndarray,
    policy: CandidatePolicy,
) -> CandidateRecord:
    if solver == "weighted_dls":
        beta, diagnostic = _dls_correct(environment, target, seed, policy)
    elif solver == "bounded_least_squares":
        beta, diagnostic = _bounded_least_squares(environment, target, seed, policy)
    else:
        raise ValueError(f"unsupported solver {solver!r}")
    bounds = _bounds(environment)
    try:
        achieved = _fk_one(environment, beta) if _inside_bounds(beta, bounds) else np.full(3, np.nan)
    except (ValueError, FloatingPointError):
        achieved = np.full(3, np.nan)
    quality, residual_mm, margin_deg, normalized_margin, reason = _quality(
        beta, achieved, target, bounds, bool(diagnostic.get("success", False)), policy
    )
    payload = dict(diagnostic)
    payload.update({"seed_in_bounds": _inside_bounds(seed, bounds), "rejection_reason": reason})
    if _inside_bounds(beta, bounds):
        jac = _jacobian(environment, beta)
        singular = np.linalg.svd(jac, compute_uv=False)
        payload.update({
            "sigma1_m": float(singular[0]), "sigma2_m": float(singular[1]), "sigma3_m": float(singular[2]),
            "kappa": float(np.inf if singular[-1] <= 0.0 else singular[0] / singular[-1]),
        })
    return CandidateRecord(
        node_id=int(node_id), candidate_id=f"n{int(node_id):06d}_c{int(ordinal):03d}",
        source=source, solver=solver, seed_rank=int(seed_rank), beta_rad=beta,
        achieved_xyz_m=achieved, residual_mm=residual_mm, min_margin_deg=margin_deg,
        normalized_min_margin=normalized_margin, quality=quality,
        solver_success=bool(diagnostic.get("success", False)), diagnostics=payload,
    )


def _capability_seeds(
    environment: ForwardEnvironment,
    target: np.ndarray,
    capability_beta_rad: np.ndarray | None,
    capability_xyz_m: np.ndarray | None,
    policy: CandidatePolicy,
    capability_tree: cKDTree | None = None,
) -> tuple[list[np.ndarray], Mapping[str, Any]]:
    if capability_beta_rad is None:
        return [], {"capability_pool_count": 0, "selected_indices": []}
    beta = np.asarray(capability_beta_rad, dtype=float).reshape(-1, 6)
    if not len(beta) or not np.isfinite(beta).all():
        raise ValueError("capability_beta_rad must be finite shape (N, 6)")
    xyz = np.asarray(capability_xyz_m, dtype=float).reshape(-1, 3) if capability_xyz_m is not None else np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    if len(xyz) != len(beta) or not np.isfinite(xyz).all():
        raise ValueError("capability_xyz_m must be finite shape (N, 3) aligned with beta")
    target_row = np.asarray(target, dtype=float).reshape(3)
    nearest_count = min(len(beta), int(policy.capability_nearest_count))
    if capability_tree is None:
        distance = np.linalg.norm(xyz - target_row.reshape(1, 3), axis=1)
        nearest = np.asarray(
            sorted(
                range(len(beta)),
                key=lambda index: (float(distance[index]), int(index)),
            )[:nearest_count],
            dtype=int,
        )
    else:
        query_distance, query_index = capability_tree.query(
            target_row, k=nearest_count
        )
        query_index = np.atleast_1d(query_index).astype(int)
        query_distance = np.atleast_1d(query_distance).astype(float)
        stable = sorted(
            zip(query_index.tolist(), query_distance.tolist()),
            key=lambda pair: (float(pair[1]), int(pair[0])),
        )
        nearest = np.asarray([pair[0] for pair in stable], dtype=int)
        distance = np.full(len(beta), np.nan, dtype=float)
        distance[nearest] = np.asarray([pair[1] for pair in stable], dtype=float)
    representatives, relative = stable_cluster_representatives(beta[nearest], scores=distance[nearest], threshold_deg=policy.candidate_cluster_deg, max_count=policy.capability_representative_count)
    return [row.copy() for row in representatives], {
        "capability_pool_count": int(len(beta)), "nearest_count": int(len(nearest)),
        "selected_indices": [int(nearest[index]) for index in relative],
        "nearest_distance_mm": [float(distance[index] * 1000.0) for index in nearest],
    }


def _nullspace_seeds(environment: ForwardEnvironment, records: Sequence[CandidateRecord], bounds: np.ndarray, policy: CandidatePolicy) -> list[tuple[str, np.ndarray]]:
    result: list[tuple[str, np.ndarray]] = []
    step = math.radians(float(policy.nullspace_step_deg))
    for record in records:
        if record.quality not in {CandidateQuality.GOLD, CandidateQuality.SILVER}:
            continue
        _u, _s, vt = np.linalg.svd(_jacobian(environment, record.beta_rad), full_matrices=True)
        for axis, direction in enumerate(vt[3:]):
            for sign in (-1.0, 1.0):
                seed = record.beta_rad + sign * step * direction
                if _inside_bounds(seed, bounds):
                    result.append((f"nullspace_{record.candidate_id}_a{axis}_{'plus' if sign > 0 else 'minus'}", seed))
    return result


def solve_candidate_bank(
    environment: ForwardEnvironment,
    target_xyz_m: np.ndarray,
    policy: CandidatePolicy | None = None,
    *,
    capability_beta_rad: np.ndarray | None = None,
    capability_xyz_m: np.ndarray | None = None,
    neighbor_beta_rad: Mapping[int, np.ndarray] | None = None,
    difficult_node_ids: Sequence[int] = (),
) -> CandidateBank:
    """Solve an auditable, stable multi-IK bank for every target point.

    The return is entirely in memory.  Records include rejected attempts so a
    runner can persist full provenance without re-solving; no candidate is
    silently repaired by clipping after a solver returns.
    """
    frozen_policy = CandidatePolicy() if policy is None else policy
    targets = np.asarray(target_xyz_m, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) == 0 or not np.isfinite(targets).all():
        raise ValueError("target_xyz_m must be non-empty finite shape (N, 3)")
    bounds = _bounds(environment)
    difficult = {int(value) for value in difficult_node_ids}
    neighbours = {} if neighbor_beta_rad is None else {int(key): np.asarray(value, dtype=float).reshape(-1, 6) for key, value in neighbor_beta_rad.items()}
    records: list[CandidateRecord] = []
    reports: dict[int, Mapping[str, Any]] = {}
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    capability_tree: cKDTree | None = None
    if capability_beta_rad is not None and capability_xyz_m is not None:
        capability_xyz = np.asarray(capability_xyz_m, dtype=float).reshape(-1, 3)
        if len(capability_xyz):
            capability_tree = cKDTree(capability_xyz)

    for node_id, target in enumerate(targets):
        seeds, capability_report = _capability_seeds(
            environment,
            target,
            capability_beta_rad,
            capability_xyz_m,
            frozen_policy,
            capability_tree,
        )
        source_seeds: list[tuple[str, np.ndarray]] = [("capability", value) for value in seeds]
        for rank, value in enumerate(neighbours.get(node_id, np.empty((0, 6)))):
            source_seeds.append((f"neighbor_warm_start_{rank:03d}", value.copy()))
        if not source_seeds:
            source_seeds.append(("bounds_midpoint", midpoint.copy()))
        budget = frozen_policy.difficult_candidate_budget_per_node if node_id in difficult else frozen_policy.candidate_budget_per_node
        node_records: list[CandidateRecord] = []
        ordinal = 0
        for seed_rank, (source, seed) in enumerate(source_seeds[:budget]):
            for solver in ("weighted_dls", "bounded_least_squares"):
                node_records.append(_candidate_record(environment=environment, node_id=node_id, ordinal=ordinal, source=source, solver=solver, seed_rank=seed_rank, target=target, seed=seed, policy=frozen_policy))
                ordinal += 1
        # C4: exact correction of ± null-space perturbations from accepted roots.
        if len(node_records) < budget:
            for source, seed in _nullspace_seeds(environment, node_records, bounds, frozen_policy):
                if len(node_records) >= budget:
                    break
                node_records.append(_candidate_record(environment=environment, node_id=node_id, ordinal=ordinal, source=source, solver="bounded_least_squares", seed_rank=ordinal, target=target, seed=seed, policy=frozen_policy))
                ordinal += 1

        accepted = [record for record in node_records if record.quality is not CandidateQuality.REJECT]
        accepted.sort(key=lambda item: (0 if item.quality is CandidateQuality.GOLD else 1, -item.min_margin_deg, item.residual_mm, item.source, item.solver, item.candidate_id))
        sensitivity = {}
        if accepted:
            beta = np.vstack([record.beta_rad for record in accepted])
            scores = np.asarray([record.residual_mm for record in accepted])
            for threshold in sorted(set(float(value) for value in frozen_policy.cluster_sensitivity_deg)):
                _representatives, indices = stable_cluster_representatives(beta, scores=scores, threshold_deg=threshold)
                sensitivity[f"{threshold:g}"] = {"cluster_count": len(indices), "accepted_record_indices": [int(index) for index in indices]}
        gold_count = sum(record.quality is CandidateQuality.GOLD for record in node_records)
        reports[node_id] = {
            "node_id": node_id, "target_xyz_m": np.asarray(target, dtype=float).tolist(),
            "candidate_attempt_count": len(node_records), "gold_candidate_count": gold_count,
            "silver_candidate_count": sum(record.quality is CandidateQuality.SILVER for record in node_records),
            "reject_candidate_count": sum(record.quality is CandidateQuality.REJECT for record in node_records),
            "has_gold": bool(gold_count), "is_difficult": node_id in difficult,
            "capability": capability_report, "cluster_threshold_sensitivity": sensitivity,
        }
        records.extend(node_records)
    records.sort(key=lambda item: (item.node_id, item.candidate_id))
    return CandidateBank(records=tuple(records), node_reports=reports, policy=frozen_policy)
