"""Independent-seed candidate discovery for retry15.

The solver accepts only frozen task targets plus a separately identified
Teacher seed bank.  Proposal beta values and proposal row identities are not
part of any public interface in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

from .exploration_qualification import weighted_beta_rms_deg
from .retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS, sobol_beta_samples, stable_id


@dataclass(frozen=True)
class Retry15CandidatePolicy:
    seed_budget: int = 16
    nearest_seed_count: int = 64
    beta_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)
    damping: float = 2.0e-3
    maximum_dls_iterations: int = 100
    maximum_least_squares_evaluations: int = 200
    maximum_step_rms_deg: float = 3.0
    residual_maximum_mm: float = 3.0
    difficult_slsqp_seed_count: int = 4

    def __post_init__(self) -> None:
        if self.seed_budget not in (8, 16, 32):
            raise ValueError("retry15 seed budget must be 8, 16 or 32")
        if self.nearest_seed_count < self.seed_budget - 2:
            raise ValueError("nearest seed pool is too small")
        if len(self.beta_weights) != 6 or any(float(value) <= 0 for value in self.beta_weights):
            raise ValueError("retry15 beta weights must contain six positive values")
        if self.damping <= 0 or self.maximum_dls_iterations < 1:
            raise ValueError("retry15 DLS policy is invalid")
        if self.maximum_least_squares_evaluations < 1 or self.maximum_step_rms_deg <= 0:
            raise ValueError("retry15 least-squares policy is invalid")
        if self.residual_maximum_mm <= 0 or self.difficult_slsqp_seed_count != 4:
            raise ValueError("retry15 residual/SLSQP policy is invalid")


def teacher_seed_bank(
    bounds_rad: np.ndarray,
    *,
    size: int,
    seed: int,
    bank_id: str,
    free_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    bounds = np.asarray(bounds_rad, dtype=float)
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds_rad must be finite ordered shape (6, 2)")
    power = int(round(math.log2(int(size))))
    if 2**power != int(size):
        raise ValueError("Teacher seed bank size must be a power of two")
    if not bank_id or "proposal" in str(bank_id).lower():
        raise ValueError("Teacher seed bank identity must be non-proposal")
    beta = sobol_beta_samples(bounds, power=power, seed=int(seed))
    if free_indices is not None:
        free = np.asarray(sorted(set(map(int, free_indices))), dtype=int)
        if len(free) == 0 or np.any(free < 0) or np.any(free >= 6):
            raise ValueError("free_indices must select beta coordinates")
        fixed = np.ones(6, dtype=bool)
        fixed[free] = False
        beta[:, fixed] = 0.0
    frame = pd.DataFrame(beta, columns=BETA_COLUMNS)
    frame.insert(0, "seed_id", [stable_id("retry15_teacher_seed", bank_id, index) for index in range(len(frame))])
    frame.insert(0, "seed_bank_id", str(bank_id))
    frame["seed"] = int(seed)
    frame["proposal_row_identity_shared"] = False
    return frame


def seed_bank_xyz(environment: Any, seed_bank: pd.DataFrame) -> pd.DataFrame:
    required = {"seed_bank_id", "seed_id", *BETA_COLUMNS}
    if not required <= set(seed_bank):
        raise ValueError("Teacher seed bank is missing identity or beta columns")
    if any("proposal" in str(value).lower() for value in seed_bank["seed_bank_id"].unique()):
        raise ValueError("proposal beta cannot enter Teacher seed bank")
    xyz = np.asarray(environment.fk(seed_bank.loc[:, BETA_COLUMNS].to_numpy(float)), dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError("Teacher seed-bank FK must be finite")
    result = seed_bank.copy()
    for index, name in enumerate(XYZ_COLUMNS):
        result[name] = xyz[:, index]
    return result


def _free_indices(target_xyz_m: np.ndarray) -> tuple[int, ...]:
    target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    y_zero = abs(float(target[1])) <= 1.0e-12
    z_zero = abs(float(target[2])) <= 1.0e-12
    if y_zero and z_zero:
        return tuple(range(6))
    if y_zero:
        return (1, 3, 5)
    if z_zero:
        return (0, 2, 4)
    return tuple(range(6))


def weighted_linear_seed(
    environment: Any,
    target_xyz_m: np.ndarray,
    *,
    weights: Sequence[float] = (4, 4, 2, 2, 1, 1),
    damping: float = 2.0e-3,
    free_indices: Sequence[int] | None = None,
) -> np.ndarray:
    target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    if not np.isfinite(target).all():
        raise ValueError("weighted linear target must be finite")
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    zero = np.zeros(6, dtype=float)
    free = np.asarray(tuple(range(6)) if free_indices is None else tuple(free_indices), dtype=int)
    jacobian = np.asarray(environment.jacobian(zero), dtype=float).reshape(3, 6)[:, free]
    delta = target - np.asarray(environment.fk(zero.reshape(1, 6)), dtype=float).reshape(3)
    weight = np.asarray(weights, dtype=float)[free]
    inverse_w2 = np.diag(1.0 / np.square(weight))
    system = jacobian @ inverse_w2 @ jacobian.T + float(damping) ** 2 * np.eye(3)
    try:
        step = inverse_w2 @ jacobian.T @ np.linalg.solve(system, delta)
    except np.linalg.LinAlgError:
        step = inverse_w2 @ jacobian.T @ np.linalg.pinv(system) @ delta
    beta = zero.copy()
    beta[free] = step
    scale = 1.0
    while scale >= 2.0**-20:
        candidate = zero.copy()
        candidate[free] = scale * step
        if np.all(candidate >= bounds[:, 0]) and np.all(candidate <= bounds[:, 1]):
            return candidate
        scale *= 0.5
    return zero


def stable_seed_sequence(
    environment: Any,
    target_xyz_m: np.ndarray,
    seed_bank_with_xyz: pd.DataFrame,
    *,
    policy: Retry15CandidatePolicy,
    seed_tree: cKDTree | None = None,
) -> tuple[list[str], np.ndarray]:
    required = {"seed_id", *BETA_COLUMNS, *XYZ_COLUMNS}
    if not required <= set(seed_bank_with_xyz):
        raise ValueError("seed bank with xyz is missing required columns")
    target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    free = _free_indices(target)
    zero = np.zeros(6, dtype=float)
    linear = weighted_linear_seed(
        environment,
        target,
        weights=policy.beta_weights,
        damping=policy.damping,
        free_indices=free,
    )
    xyz = seed_bank_with_xyz.loc[:, XYZ_COLUMNS].to_numpy(float)
    tree = cKDTree(xyz) if seed_tree is None else seed_tree
    _distance, nearest = tree.query(target, k=min(policy.nearest_seed_count, len(xyz)))
    indices = list(map(int, np.atleast_1d(nearest)))
    pool = seed_bank_with_xyz.iloc[indices].reset_index(drop=True)
    beta_pool = pool.loc[:, BETA_COLUMNS].to_numpy(float)
    selected_indices: list[int] = []
    selected_beta: list[np.ndarray] = [zero, linear]
    while len(selected_beta) < policy.seed_budget:
        best: tuple[float, str, int] | None = None
        for index, beta in enumerate(beta_pool):
            if index in selected_indices:
                continue
            distance = min(
                float(weighted_beta_rms_deg(beta, chosen, policy.beta_weights))
                for chosen in selected_beta
            )
            key = (distance, str(pool.iloc[index]["seed_id"]), index)
            if best is None or (-key[0], key[1], key[2]) < (-best[0], best[1], best[2]):
                best = key
        if best is None:
            raise RuntimeError("independent seed bank cannot satisfy the registered seed budget")
        selected_indices.append(best[2])
        selected_beta.append(beta_pool[best[2]].copy())
    ids = ["retry15_exact_zero_seed", "retry15_weighted_linear_seed"] + [
        str(pool.iloc[index]["seed_id"]) for index in selected_indices
    ]
    values = np.vstack(selected_beta)
    fixed = np.ones(6, dtype=bool)
    fixed[np.asarray(free, dtype=int)] = False
    values[:, fixed] = 0.0
    return ids, values


def _residual_mm(environment: Any, beta: np.ndarray, target: np.ndarray) -> float:
    achieved = np.asarray(environment.fk(np.asarray(beta, dtype=float).reshape(1, 6)), dtype=float).reshape(3)
    return float(np.linalg.norm(achieved - target) * 1000.0)


def _margin_deg(beta: np.ndarray, bounds: np.ndarray) -> float:
    margin = np.minimum(beta - bounds[:, 0], bounds[:, 1] - beta)
    return float(np.rad2deg(np.min(margin)))


def _bounded_step_scale(beta: np.ndarray, step: np.ndarray, bounds: np.ndarray) -> float:
    scale = 1.0
    for index, value in enumerate(step):
        if value > 0:
            scale = min(scale, float((bounds[index, 1] - beta[index]) / value))
        elif value < 0:
            scale = min(scale, float((bounds[index, 0] - beta[index]) / value))
    return max(0.0, min(1.0, scale))


def _constrained_dls(
    environment: Any,
    target: np.ndarray,
    seed: np.ndarray,
    *,
    free: np.ndarray,
    policy: Retry15CandidatePolicy,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    beta = np.asarray(seed, dtype=float).reshape(6).copy()
    fixed = np.ones(6, dtype=bool)
    fixed[free] = False
    beta[fixed] = 0.0
    if np.any(beta < bounds[:, 0]) or np.any(beta > bounds[:, 1]):
        return beta, {"success": False, "status": "seed_out_of_bounds", "iterations": 0}
    weights = np.asarray(policy.beta_weights, dtype=float)[free]
    inverse_w2 = np.diag(1.0 / np.square(weights))
    for iteration in range(policy.maximum_dls_iterations):
        achieved = np.asarray(environment.fk(beta.reshape(1, 6)), dtype=float).reshape(3)
        residual = target - achieved
        residual_mm = float(np.linalg.norm(residual) * 1000.0)
        if residual_mm <= policy.residual_maximum_mm:
            return beta, {"success": True, "status": "residual_reached", "iterations": iteration, "residual_mm": residual_mm}
        jacobian = np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6)[:, free]
        system = jacobian @ inverse_w2 @ jacobian.T + policy.damping**2 * np.eye(3)
        try:
            free_step = inverse_w2 @ jacobian.T @ np.linalg.solve(system, residual)
        except np.linalg.LinAlgError:
            free_step = inverse_w2 @ jacobian.T @ np.linalg.pinv(system) @ residual
        step = np.zeros(6, dtype=float)
        step[free] = free_step
        rms_deg = float(weighted_beta_rms_deg(step, np.zeros(6), policy.beta_weights))
        if rms_deg > policy.maximum_step_rms_deg:
            step *= policy.maximum_step_rms_deg / rms_deg
        scale = _bounded_step_scale(beta, step, bounds)
        accepted = False
        while scale >= 2.0**-12:
            trial = beta + scale * step
            trial[fixed] = 0.0
            if _residual_mm(environment, trial, target) < residual_mm:
                beta = trial
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return beta, {"success": False, "status": "line_search_stalled", "iterations": iteration + 1, "residual_mm": residual_mm}
    return beta, {"success": False, "status": "iteration_limit", "iterations": policy.maximum_dls_iterations, "residual_mm": _residual_mm(environment, beta, target)}


def _constrained_least_squares(
    environment: Any,
    target: np.ndarray,
    seed: np.ndarray,
    *,
    free: np.ndarray,
    policy: Retry15CandidatePolicy,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    template = np.zeros(6, dtype=float)
    # DLS uses a small tolerance when deciding whether a result remains within
    # bounds.  Floating-point roundoff can therefore leave a seed a few ulps
    # outside the exact interval accepted by scipy.optimize.least_squares,
    # which otherwise aborts the whole target shard with ``x0 is infeasible``.
    # Clipping changes no registered numerical policy; it maps only the
    # tolerated boundary overshoot back to the same physical bound.
    template[free] = np.clip(
        np.asarray(seed, dtype=float).reshape(6)[free],
        bounds[free, 0],
        bounds[free, 1],
    )

    def residual(free_beta: np.ndarray) -> np.ndarray:
        beta = template.copy()
        beta[free] = free_beta
        return (np.asarray(environment.fk(beta.reshape(1, 6)), dtype=float).reshape(3) - target) / 0.001

    result = least_squares(
        residual,
        template[free],
        bounds=(bounds[free, 0], bounds[free, 1]),
        max_nfev=policy.maximum_least_squares_evaluations,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    beta = template.copy()
    beta[free] = np.asarray(result.x, dtype=float)
    return beta, {"success": bool(result.success), "status": int(result.status), "message": str(result.message), "nfev": int(result.nfev), "residual_mm": _residual_mm(environment, beta, target)}


def _constrained_slsqp(
    environment: Any,
    target: np.ndarray,
    seed: np.ndarray,
    *,
    free: np.ndarray,
    policy: Retry15CandidatePolicy,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    template = np.zeros(6, dtype=float)

    def objective(free_beta: np.ndarray) -> float:
        beta = template.copy()
        beta[free] = free_beta
        residual = (np.asarray(environment.fk(beta.reshape(1, 6)), dtype=float).reshape(3) - target) / 0.001
        return float(np.dot(residual, residual))

    result = minimize(
        objective,
        np.asarray(seed, dtype=float).reshape(6)[free],
        method="SLSQP",
        bounds=[(float(bounds[index, 0]), float(bounds[index, 1])) for index in free],
        options={"maxiter": policy.maximum_least_squares_evaluations, "ftol": 1.0e-12, "disp": False},
    )
    beta = template.copy()
    beta[free] = np.asarray(result.x, dtype=float)
    return beta, {"success": bool(result.success), "status": int(result.status), "message": str(result.message), "nfev": int(result.nfev), "residual_mm": _residual_mm(environment, beta, target)}


def solve_target_candidates(
    environment: Any,
    target: Mapping[str, Any],
    seed_bank_with_xyz: pd.DataFrame,
    *,
    policy: Retry15CandidatePolicy,
    seed_tree: cKDTree | None = None,
) -> pd.DataFrame:
    if any(key not in target for key in ("target_id", *XYZ_COLUMNS)):
        raise ValueError("target is missing stable identity or xyz")
    target_id = str(target["target_id"])
    xyz = np.asarray([target[name] for name in XYZ_COLUMNS], dtype=float)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    if str(target.get("target_role", "")) == "exact_zero":
        achieved = np.asarray(environment.fk(np.zeros((1, 6))), dtype=float).reshape(3)
        residual = float(np.linalg.norm(achieved - xyz) * 1000.0)
        return pd.DataFrame(
            [{
                "target_id": target_id,
                "candidate_id": f"{target_id}:exact_zero",
                "seed_id": "retry15_exact_zero_seed",
                "seed_rank": 0,
                "solver": "fixed_exact_zero",
                **dict(zip(BETA_COLUMNS, np.zeros(6), strict=True)),
                "solver_success": bool(residual <= 1.0e-9),
                "bounds_pass": True,
                "fk_residual_mm": residual,
                "min_margin_deg": _margin_deg(np.zeros(6), bounds),
                "seam_class": "exact_zero",
                "proposal_beta_used": False,
                "diagnostic_status": "fixed",
            }]
        )
    free = np.asarray(_free_indices(xyz), dtype=int)
    seed_ids, seeds = stable_seed_sequence(
        environment, xyz, seed_bank_with_xyz, policy=policy, seed_tree=seed_tree
    )
    rows: list[dict[str, Any]] = []

    def record(seed_rank: int, seed_id: str, solver: str, beta: np.ndarray, diagnostic: Mapping[str, Any]) -> None:
        bounds_pass = bool(np.all(beta >= bounds[:, 0] - 1.0e-12) and np.all(beta <= bounds[:, 1] + 1.0e-12))
        residual = _residual_mm(environment, beta, xyz) if bounds_pass else math.inf
        rows.append(
            {
                "target_id": target_id,
                "candidate_id": f"{target_id}:seed:{seed_rank:02d}:{solver}",
                "seed_id": seed_id,
                "seed_rank": int(seed_rank),
                "solver": solver,
                **dict(zip(BETA_COLUMNS, beta, strict=True)),
                "solver_success": bool(diagnostic.get("success", False)),
                "bounds_pass": bounds_pass,
                "fk_residual_mm": residual,
                "min_margin_deg": _margin_deg(beta, bounds) if bounds_pass else -math.inf,
                "seam_class": "y_seam" if tuple(free) == (1, 3, 5) else "z_seam" if tuple(free) == (0, 2, 4) else "interior",
                "proposal_beta_used": False,
                "diagnostic_status": str(diagnostic.get("status", "")),
            }
        )

    for seed_rank, (seed_id, seed) in enumerate(zip(seed_ids, seeds, strict=True)):
        dls_beta, dls_diagnostic = _constrained_dls(environment, xyz, seed, free=free, policy=policy)
        record(seed_rank, seed_id, "weighted_dls", dls_beta, dls_diagnostic)
        ls_beta, ls_diagnostic = _constrained_least_squares(environment, xyz, dls_beta, free=free, policy=policy)
        record(seed_rank, seed_id, "bounded_least_squares", ls_beta, ls_diagnostic)
    legal = [row for row in rows if row["solver_success"] and row["bounds_pass"] and row["fk_residual_mm"] <= policy.residual_maximum_mm]
    if not legal:
        ranked = sorted(rows, key=lambda row: (row["fk_residual_mm"], row["seed_rank"], row["solver"]))
        used: set[int] = set()
        for row in ranked:
            rank = int(row["seed_rank"])
            if rank in used:
                continue
            used.add(rank)
            beta, diagnostic = _constrained_slsqp(environment, xyz, seeds[rank], free=free, policy=policy)
            record(rank, seed_ids[rank], "slsqp", beta, diagnostic)
            if len(used) >= policy.difficult_slsqp_seed_count:
                break
    return pd.DataFrame(rows)


def solve_candidate_shard(
    environment: Any,
    targets: pd.DataFrame,
    seed_bank_with_xyz: pd.DataFrame,
    *,
    policy: Retry15CandidatePolicy,
) -> pd.DataFrame:
    required = {"target_id", *XYZ_COLUMNS}
    if not required <= set(targets):
        raise ValueError("candidate shard targets are missing identity or xyz")
    if any(column.startswith("be") for column in targets.columns):
        raise ValueError("Teacher targets must not contain beta columns")
    seed_tree = cKDTree(seed_bank_with_xyz.loc[:, XYZ_COLUMNS].to_numpy(float))
    frames = [
        solve_target_candidates(
            environment, row, seed_bank_with_xyz, policy=policy, seed_tree=seed_tree
        )
        for row in targets.to_dict("records")
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


__all__ = [
    "Retry15CandidatePolicy",
    "seed_bank_xyz",
    "solve_candidate_shard",
    "solve_target_candidates",
    "stable_seed_sequence",
    "teacher_seed_bank",
    "weighted_linear_seed",
]
