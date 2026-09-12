"""Shared numerical helpers for trajectory evaluation.

These functions operate only on trajectory arrays and candidate rows.  Experiment
runners retain ownership of registries, stages, Gates, and persisted reports.
"""

from __future__ import annotations

import math
from typing import Any, Sequence, TypedDict

import numpy as np
import pandas as pd

from .exploration_qualification import percentile, weighted_beta_rms_deg
from .retry12_symmetry import BETA_COLUMNS, transform_beta
from .retry15_canonical_graph import legal_candidate_clusters


class PathMetrics(TypedDict):
    """Six metrics: residual/path excess in mm, raw steps in degrees, rate unitless."""

    fk_p95_mm: float
    fk_maximum_mm: float
    path_step_excess_p99_mm: float
    path_step_excess_maximum_mm: float
    raw_step_gt7_rate: float
    raw_step_maximum_deg: float


def _trajectory_arrays(
    beta: np.ndarray,
    xyz: np.ndarray,
    *,
    function: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return beta (N, 6) and target xyz (N, 3) without filtering values."""

    result_beta = np.asarray(beta, dtype=float)
    points = np.asarray(xyz, dtype=float)
    if result_beta.ndim != 2 or result_beta.shape[1] != 6:
        raise ValueError(f"{function} beta must have shape (N, 6)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{function} xyz must have shape (N, 3)")
    if len(result_beta) != len(points):
        raise ValueError(f"{function} beta and xyz must have the same row count")
    return result_beta, points


def two_step_dls(
    environment: Any,
    beta: np.ndarray,
    xyz: np.ndarray,
    *,
    zero_xyz: np.ndarray,
) -> np.ndarray:
    """Map beta radians (N, 6) to corrected beta (N, 6) for xyz metres (N, 3).

    ``zero_xyz`` is metres with shape (3,).
    """

    source_beta, points = _trajectory_arrays(beta, xyz, function="two_step_dls")
    zero = np.asarray(zero_xyz, dtype=float)
    if zero.shape != (3,):
        raise ValueError("two_step_dls zero_xyz must have shape (3,)")
    corrected = source_beta.copy()
    bounds = np.asarray(environment.bounds, dtype=float)
    weights = np.asarray([4, 4, 2, 2, 1, 1], dtype=float)
    for _ in range(2):
        for index, point in enumerate(points):
            if np.linalg.norm(point - zero) <= 1e-12:
                corrected[index] = 0.0
                continue
            freeze = {0, 2, 4} if abs(point[1]) <= 1e-12 else {1, 3, 5} if abs(point[2]) <= 1e-12 else set()
            free = np.asarray([axis for axis in range(6) if axis not in freeze], dtype=int)
            current = np.asarray(environment.fk(corrected[index])).reshape(-1, 3)[0]
            jacobian = np.asarray(environment.jacobian(corrected[index])).reshape(3, 6)[:, free]
            winv = np.diag(1.0 / weights[free])
            task = jacobian @ winv @ jacobian.T + 1e-6 * np.eye(3)
            pseudo = winv @ jacobian.T @ np.linalg.pinv(task, rcond=1e-12)
            corrected[index, free] = np.clip(
                corrected[index, free] + pseudo @ (point - current),
                bounds[free, 0],
                bounds[free, 1],
            )
            if freeze:
                corrected[index, sorted(freeze)] = 0.0
    return corrected


def symmetry_prediction(model: Any, xyz: np.ndarray, zero_xyz: np.ndarray) -> np.ndarray:
    """Return beta radians (N, 6) for xyz metres (N, 3) using retry17 seams.

    ``zero_xyz`` is metres with shape (3,), and the model must return (N, 6).
    """

    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("symmetry_prediction xyz must have shape (N, 3)")
    zero = np.asarray(zero_xyz, dtype=float)
    if zero.shape != (3,):
        raise ValueError("symmetry_prediction zero_xyz must have shape (3,)")
    canonical = points.copy()
    canonical[:, 1:] = np.abs(canonical[:, 1:])
    beta = np.asarray(model(canonical.astype(np.float32), training=False), dtype=float)
    if beta.ndim != 2 or beta.shape != (len(points), 6):
        raise ValueError("symmetry_prediction model output must have shape (N, 6)")
    for index, point in enumerate(points):
        if point[1] < 0:
            beta[index] = transform_beta(beta[index], "mirror_y")
        if point[2] < 0:
            beta[index] = transform_beta(beta[index], "mirror_z")
        if abs(point[1]) <= 1e-12:
            beta[index, [0, 2, 4]] = 0
        if abs(point[2]) <= 1e-12:
            beta[index, [1, 3, 5]] = 0
        if np.linalg.norm(point - zero) <= 1e-12:
            beta[index] = 0
    return beta


def cycle_teacher(
    candidates: pd.DataFrame,
    *,
    pairwise_lambda: float,
    weights: Sequence[float],
    tau_deg: float,
) -> pd.DataFrame:
    """Select one legal beta-radian candidate per waypoint with retry17 closed-cycle DP.

    ``candidates`` requires target/candidate IDs, ``waypoint_index``,
    ``BETA_COLUMNS``, and the legality fields ``solver_success``, ``bounds_pass``,
    ``fk_residual_mm``, and ``min_margin_deg``.
    """

    required = {
        "target_id",
        "candidate_id",
        "waypoint_index",
        "solver_success",
        "bounds_pass",
        "fk_residual_mm",
        "min_margin_deg",
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(candidates))
    if missing:
        raise ValueError(f"cycle_teacher candidate frame is missing required columns: {', '.join(missing)}")
    legal = legal_candidate_clusters(candidates, weights=weights)
    order = list(dict.fromkeys(legal.sort_values("waypoint_index", kind="stable")["target_id"].astype(str)))
    grouped = {
        target: legal[legal["target_id"].astype(str).eq(target)]
        .sort_values("candidate_id", kind="stable")
        .reset_index(drop=True)
        for target in order
    }
    if len(grouped) != len(order) or not order:
        return pd.DataFrame()
    beta = {target: grouped[target].loc[:, BETA_COLUMNS].to_numpy(float) for target in order}
    unary = {
        target: np.square(
            np.asarray(weighted_beta_rms_deg(beta[target], np.zeros_like(beta[target]), weights), dtype=float)
        )
        for target in order
    }

    def cost(left: str, right: str) -> np.ndarray:
        a, b = beta[left], beta[right]
        matrix = np.empty((len(a), len(b)))
        for i in range(len(a)):
            gaps = np.asarray(weighted_beta_rms_deg(np.broadcast_to(a[i], b.shape), b, weights), dtype=float)
            matrix[i] = np.minimum(np.square(gaps), tau_deg * tau_deg)
        return matrix

    transitions = [cost(order[i], order[i + 1]) for i in range(len(order) - 1)]
    closing = cost(order[-1], order[0])
    solutions = []
    for first_state in range(len(grouped[order[0]])):
        values = np.full(len(grouped[order[0]]), math.inf)
        values[first_state] = unary[order[0]][first_state]
        backs = []
        for index, matrix in enumerate(transitions):
            total = values[:, None] + pairwise_lambda * matrix
            back = np.argmin(total, axis=0)
            values = unary[order[index + 1]] + total[back, np.arange(total.shape[1])]
            backs.append(back)
        values = values + pairwise_lambda * closing[:, first_state]
        last = int(np.argmin(values))
        selection = [last]
        for back in reversed(backs):
            selection.append(int(back[selection[-1]]))
        selection.reverse()
        signature = tuple(
            grouped[target].iloc[state]["candidate_id"]
            for target, state in zip(order, selection, strict=True)
        )
        solutions.append((float(values[last]), signature, selection))
    _value, _signature, selection = min(solutions, key=lambda item: (item[0], item[1]))
    result = pd.concat(
        [grouped[target].iloc[[state]] for target, state in zip(order, selection, strict=True)],
        ignore_index=True,
        sort=False,
    )
    result["teacher"] = "retry17_cycle_dp"
    result["pairwise_lambda"] = pairwise_lambda
    return result


def path_metrics(
    xyz: np.ndarray,
    beta: np.ndarray,
    environment: Any,
    *,
    closed: bool = True,
) -> PathMetrics:
    """Return six retry18 metrics for xyz metres (N, 3) and beta radians (N, 6).

    The result excludes ``success_rate``; callers compute their own success
    criterion from FK residuals.
    """

    beta_values, target = _trajectory_arrays(beta, xyz, function="path_metrics")
    if not np.isfinite(beta_values).all():
        return {
            "fk_p95_mm": math.inf,
            "fk_maximum_mm": math.inf,
            "path_step_excess_p99_mm": math.inf,
            "path_step_excess_maximum_mm": math.inf,
            "raw_step_gt7_rate": 1.0,
            "raw_step_maximum_deg": math.inf,
        }
    achieved = np.asarray(environment.fk(beta_values)).reshape(-1, 3)
    residual = np.linalg.norm(achieved - target, axis=1) * 1000.0
    if closed:
        target = np.vstack([target, target[:1]])
        achieved = np.vstack([achieved, achieved[:1]])
        beta_values = np.vstack([beta_values, beta_values[:1]])
    target_step = np.linalg.norm(np.diff(target, axis=0), axis=1) * 1000.0
    achieved_step = np.linalg.norm(np.diff(achieved, axis=0), axis=1) * 1000.0
    excess = np.maximum(0.0, achieved_step - target_step)
    raw_step = np.max(np.abs(np.rad2deg(np.diff(beta_values, axis=0))), axis=1)
    return {
        "fk_p95_mm": percentile(residual, 95),
        "fk_maximum_mm": float(np.max(residual)),
        "path_step_excess_p99_mm": percentile(excess, 99),
        "path_step_excess_maximum_mm": float(np.max(excess)),
        "raw_step_gt7_rate": float(np.mean(raw_step > 7.0)),
        "raw_step_maximum_deg": float(np.max(raw_step)),
    }
