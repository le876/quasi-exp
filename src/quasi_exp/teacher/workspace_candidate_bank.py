"""V14 correction/diversity candidate discovery over immutable V12 solvers.

The V12.12 formal fixed point locks ``multi_ik_candidates.py`` byte-for-byte.
V14 therefore adapts its stable numerical primitives here instead of changing
the historical module.  This module owns the new search-mode, solver-matrix,
seed-budget and low-margin semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .canonical import ForwardEnvironment
from . import multi_ik_candidates as _v12


CandidateQuality = _v12.CandidateQuality
CandidateRecord = _v12.CandidateRecord
stable_cluster_representatives = _v12.stable_cluster_representatives


class CandidateSearchMode(str, Enum):
    """Whether a seed is corrected once or explored by every solver."""

    CORRECTION = "correction"
    DIVERSITY = "diversity"


@dataclass(frozen=True)
class CandidatePolicy:
    """Frozen V14 seed and solver policy.

    The first block mirrors the immutable V12 policy so its bounded DLS,
    bounded least-squares, capability selection and null-space primitives can
    be reused without changing their historical source bytes.
    """

    capability_nearest_count: int = 64
    capability_representative_count: int = 8
    candidate_budget_per_node: int = 16
    difficult_candidate_budget_per_node: int = 32
    candidate_seed_budget_per_node: int | None = None
    difficult_seed_budget_per_node: int | None = None
    nullspace_seed_budget_per_node: int | None = None
    search_mode: CandidateSearchMode = CandidateSearchMode.DIVERSITY
    solver_names: tuple[str, ...] = ("weighted_dls", "bounded_least_squares")
    gold_candidate_target: int = 4
    beta_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)
    damping: float = 1.0e-3
    max_rms_step_deg: float = 3.0
    max_corrector_iterations: int = 100
    tracking_tolerance_mm: float = 1.0
    max_residual_mm: float = 3.0
    gold_margin_deg: float = 1.5
    silver_margin_deg: float = 0.0
    candidate_cluster_deg: float = 0.5
    cluster_sensitivity_deg: tuple[float, ...] = (0.25, 0.5, 1.0)
    nullspace_step_deg: float = 1.0
    solver_seed: int = 20260732

    def __post_init__(self) -> None:
        # Delegate all shared numerical invariants to the frozen V12 policy.
        _v12.CandidatePolicy(
            capability_nearest_count=self.capability_nearest_count,
            capability_representative_count=self.capability_representative_count,
            candidate_budget_per_node=self.candidate_budget_per_node,
            difficult_candidate_budget_per_node=self.difficult_candidate_budget_per_node,
            gold_candidate_target=self.gold_candidate_target,
            beta_weights=self.beta_weights,
            damping=self.damping,
            max_rms_step_deg=self.max_rms_step_deg,
            max_corrector_iterations=self.max_corrector_iterations,
            tracking_tolerance_mm=self.tracking_tolerance_mm,
            max_residual_mm=self.max_residual_mm,
            gold_margin_deg=self.gold_margin_deg,
            candidate_cluster_deg=self.candidate_cluster_deg,
            cluster_sensitivity_deg=self.cluster_sensitivity_deg,
            nullspace_step_deg=self.nullspace_step_deg,
            solver_seed=self.solver_seed,
        )
        for name, value in (
            ("candidate_seed_budget_per_node", self.candidate_seed_budget_per_node),
            ("difficult_seed_budget_per_node", self.difficult_seed_budget_per_node),
        ):
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when provided")
        if (
            self.nullspace_seed_budget_per_node is not None
            and int(self.nullspace_seed_budget_per_node) < 0
        ):
            raise ValueError("nullspace_seed_budget_per_node must be non-negative")
        mode = CandidateSearchMode(self.search_mode)
        solvers = tuple(str(value) for value in self.solver_names)
        supported = {"weighted_dls", "bounded_least_squares", "slsqp"}
        if (
            not solvers
            or len(set(solvers)) != len(solvers)
            or any(value not in supported for value in solvers)
        ):
            raise ValueError(
                f"solver_names must be unique members of {sorted(supported)}"
            )
        if (
            not math.isfinite(float(self.silver_margin_deg))
            or self.silver_margin_deg < 0.0
            or self.silver_margin_deg >= self.gold_margin_deg
        ):
            raise ValueError(
                "silver_margin_deg must be finite, non-negative and below gold_margin_deg"
            )
        object.__setattr__(self, "search_mode", mode)
        object.__setattr__(self, "solver_names", solvers)


@dataclass(frozen=True)
class CandidateBank:
    records: tuple[CandidateRecord, ...]
    node_reports: Mapping[int, Mapping[str, Any]]
    policy: CandidatePolicy = field(repr=False)

    def for_node(self, node_id: int) -> tuple[CandidateRecord, ...]:
        return tuple(
            record for record in self.records if record.node_id == int(node_id)
        )

    @property
    def gold_node_ratio(self) -> float:
        if not self.node_reports:
            return 0.0
        return float(
            np.mean(
                [
                    bool(report["gold_candidate_count"])
                    for report in self.node_reports.values()
                ]
            )
        )


def _slsqp(
    environment: ForwardEnvironment,
    target: np.ndarray,
    seed: np.ndarray,
    policy: CandidatePolicy,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    bounds = _v12._bounds(environment)
    initial = np.asarray(seed, dtype=float).reshape(6)
    if not _v12._inside_bounds(initial, bounds):
        return initial.copy(), {
            "success": False,
            "status": "seed_out_of_bounds",
            "nit": 0,
            "nfev": 0,
        }
    target_row = np.asarray(target, dtype=float).reshape(3)

    def objective(beta: np.ndarray) -> float:
        residual_mm = (_v12._fk_one(environment, beta) - target_row) / 0.001
        return float(np.dot(residual_mm, residual_mm))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(float(low), float(high)) for low, high in bounds],
        options={
            "maxiter": int(policy.max_corrector_iterations),
            "ftol": 1.0e-12,
            "disp": False,
        },
    )
    beta = np.asarray(result.x, dtype=float).reshape(6)
    return beta, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
        "nit": int(result.nit),
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
    if not _v12._inside_bounds(beta, bounds):
        return (
            CandidateQuality.REJECT,
            float("nan"),
            float("nan"),
            float("nan"),
            "actual_bounds_failed",
        )
    if not np.isfinite(achieved).all():
        return (
            CandidateQuality.REJECT,
            float("nan"),
            float("nan"),
            float("nan"),
            "nonfinite_achieved_xyz",
        )
    residual_mm = float(
        np.linalg.norm(np.asarray(achieved) - np.asarray(target)) * 1000.0
    )
    margin, normalized_margin = _v12._margin_metrics(beta, bounds)
    if not solver_success:
        return (
            CandidateQuality.REJECT,
            residual_mm,
            margin,
            normalized_margin,
            "solver_not_converged",
        )
    if residual_mm > float(policy.max_residual_mm):
        return (
            CandidateQuality.REJECT,
            residual_mm,
            margin,
            normalized_margin,
            "residual_above_max",
        )
    if margin >= float(policy.gold_margin_deg):
        return CandidateQuality.GOLD, residual_mm, margin, normalized_margin, None
    silver = float(policy.silver_margin_deg)
    if margin >= silver and (silver > 0.0 or margin > 0.0):
        return CandidateQuality.SILVER, residual_mm, margin, normalized_margin, None
    return (
        CandidateQuality.REJECT,
        residual_mm,
        margin,
        normalized_margin,
        "margin_below_silver",
    )


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
        beta, diagnostic = _v12._dls_correct(environment, target, seed, policy)
    elif solver == "bounded_least_squares":
        beta, diagnostic = _v12._bounded_least_squares(
            environment, target, seed, policy
        )
    elif solver == "slsqp":
        beta, diagnostic = _slsqp(environment, target, seed, policy)
    else:  # pragma: no cover - protected by policy validation
        raise ValueError(f"unsupported solver {solver!r}")
    bounds = _v12._bounds(environment)
    try:
        achieved = (
            _v12._fk_one(environment, beta)
            if _v12._inside_bounds(beta, bounds)
            else np.full(3, np.nan)
        )
    except (ValueError, FloatingPointError):
        achieved = np.full(3, np.nan)
    quality, residual_mm, margin_deg, normalized_margin, reason = _quality(
        beta,
        achieved,
        target,
        bounds,
        bool(diagnostic.get("success", False)),
        policy,
    )
    payload = dict(diagnostic)
    payload.update(
        {
            "seed_in_bounds": _v12._inside_bounds(seed, bounds),
            "rejection_reason": reason,
        }
    )
    if _v12._inside_bounds(beta, bounds):
        singular = np.linalg.svd(_v12._jacobian(environment, beta), compute_uv=False)
        payload.update(
            {
                "sigma1_m": float(singular[0]),
                "sigma2_m": float(singular[1]),
                "sigma3_m": float(singular[2]),
                "kappa": float(
                    np.inf
                    if singular[-1] <= 0.0
                    else singular[0] / singular[-1]
                ),
            }
        )
    return CandidateRecord(
        node_id=int(node_id),
        candidate_id=f"n{int(node_id):06d}_c{int(ordinal):03d}",
        source=source,
        solver=solver,
        seed_rank=int(seed_rank),
        beta_rad=beta,
        achieved_xyz_m=achieved,
        residual_mm=residual_mm,
        min_margin_deg=margin_deg,
        normalized_min_margin=normalized_margin,
        quality=quality,
        solver_success=bool(diagnostic.get("success", False)),
        diagnostics=payload,
    )


def solve_candidate_bank(
    environment: ForwardEnvironment,
    target_xyz_m: np.ndarray,
    policy: CandidatePolicy | None = None,
    *,
    capability_beta_rad: np.ndarray | None = None,
    capability_xyz_m: np.ndarray | None = None,
    neighbor_beta_rad: Mapping[int, np.ndarray] | None = None,
    node_seed_beta_rad: Mapping[int, np.ndarray] | None = None,
    difficult_node_ids: Sequence[int] = (),
) -> CandidateBank:
    """Run correction or independent diversity attempts for every task node."""

    active = CandidatePolicy() if policy is None else policy
    targets = np.asarray(target_xyz_m, dtype=float)
    if (
        targets.ndim != 2
        or targets.shape[1] != 3
        or len(targets) == 0
        or not np.isfinite(targets).all()
    ):
        raise ValueError("target_xyz_m must be non-empty finite shape (N, 3)")
    bounds = _v12._bounds(environment)
    difficult = {int(value) for value in difficult_node_ids}
    neighbours = (
        {}
        if neighbor_beta_rad is None
        else {
            int(key): np.asarray(value, dtype=float).reshape(-1, 6)
            for key, value in neighbor_beta_rad.items()
        }
    )
    exact_seeds = (
        {}
        if node_seed_beta_rad is None
        else {
            int(key): np.asarray(value, dtype=float).reshape(6)
            for key, value in node_seed_beta_rad.items()
        }
    )
    records: list[CandidateRecord] = []
    reports: dict[int, Mapping[str, Any]] = {}
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    capability_tree: cKDTree | None = None
    if capability_beta_rad is not None and capability_xyz_m is not None:
        capability_xyz = np.asarray(capability_xyz_m, dtype=float).reshape(-1, 3)
        if len(capability_xyz):
            capability_tree = cKDTree(capability_xyz)

    for node_id, target in enumerate(targets):
        seeds, capability_report = _v12._capability_seeds(
            environment,
            target,
            capability_beta_rad,
            capability_xyz_m,
            active,
            capability_tree,
        )
        source_seeds: list[tuple[str, np.ndarray]] = []
        if node_id in exact_seeds:
            source_seeds.append(("node_exact_seed", exact_seeds[node_id].copy()))
        source_seeds.extend(("capability", value) for value in seeds)
        for rank, value in enumerate(
            neighbours.get(node_id, np.empty((0, 6)))
        ):
            source_seeds.append((f"neighbor_warm_start_{rank:03d}", value.copy()))
        if not source_seeds:
            source_seeds.append(("bounds_midpoint", midpoint.copy()))
        legacy_budget = (
            active.difficult_candidate_budget_per_node
            if node_id in difficult
            else active.candidate_budget_per_node
        )
        explicit_seed_budget = (
            active.difficult_seed_budget_per_node
            if node_id in difficult
            else active.candidate_seed_budget_per_node
        )
        seed_budget = (
            legacy_budget
            if explicit_seed_budget is None
            else int(explicit_seed_budget)
        )
        node_records: list[CandidateRecord] = []
        ordinal = 0
        source_seed_count = 0

        def run_seed(source: str, seed: np.ndarray, seed_rank: int) -> None:
            nonlocal ordinal
            for solver in active.solver_names:
                record = _candidate_record(
                    environment=environment,
                    node_id=node_id,
                    ordinal=ordinal,
                    source=source,
                    solver=solver,
                    seed_rank=seed_rank,
                    target=target,
                    seed=seed,
                    policy=active,
                )
                node_records.append(record)
                ordinal += 1
                if (
                    active.search_mode is CandidateSearchMode.CORRECTION
                    and record.quality is not CandidateQuality.REJECT
                ):
                    break

        for seed_rank, (source, seed) in enumerate(source_seeds[:seed_budget]):
            run_seed(source, seed, seed_rank)
            source_seed_count += 1

        if explicit_seed_budget is None and len(node_records) < legacy_budget:
            for source, seed in _v12._nullspace_seeds(
                environment, node_records, bounds, active
            ):
                if len(node_records) >= legacy_budget:
                    break
                node_records.append(
                    _candidate_record(
                        environment=environment,
                        node_id=node_id,
                        ordinal=ordinal,
                        source=source,
                        solver="bounded_least_squares",
                        seed_rank=ordinal,
                        target=target,
                        seed=seed,
                        policy=active,
                    )
                )
                ordinal += 1
                source_seed_count += 1
        elif explicit_seed_budget is not None:
            nullspace_budget = int(active.nullspace_seed_budget_per_node or 0)
            nullspace = _v12._nullspace_seeds(
                environment, node_records, bounds, active
            )[:nullspace_budget]
            for offset, (source, seed) in enumerate(nullspace):
                run_seed(source, seed, source_seed_count + offset)
            source_seed_count += len(nullspace)

        accepted = [
            record
            for record in node_records
            if record.quality is not CandidateQuality.REJECT
        ]
        accepted.sort(
            key=lambda item: (
                0 if item.quality is CandidateQuality.GOLD else 1,
                -item.min_margin_deg,
                item.residual_mm,
                item.source,
                item.solver,
                item.candidate_id,
            )
        )
        sensitivity: dict[str, Any] = {}
        if accepted:
            beta = np.vstack([record.beta_rad for record in accepted])
            scores = np.asarray([record.residual_mm for record in accepted])
            for threshold in sorted(
                set(float(value) for value in active.cluster_sensitivity_deg)
            ):
                _representatives, indices = stable_cluster_representatives(
                    beta, scores=scores, threshold_deg=threshold
                )
                sensitivity[f"{threshold:g}"] = {
                    "cluster_count": len(indices),
                    "accepted_record_indices": [int(index) for index in indices],
                }
        gold_count = sum(
            record.quality is CandidateQuality.GOLD for record in node_records
        )
        reports[node_id] = {
            "node_id": node_id,
            "target_xyz_m": np.asarray(target, dtype=float).tolist(),
            "candidate_attempt_count": len(node_records),
            "gold_candidate_count": gold_count,
            "silver_candidate_count": sum(
                record.quality is CandidateQuality.SILVER
                for record in node_records
            ),
            "reject_candidate_count": sum(
                record.quality is CandidateQuality.REJECT
                for record in node_records
            ),
            "source_seed_count": int(source_seed_count),
            "solver_attempt_count": int(len(node_records)),
            "search_mode": active.search_mode.value,
            "solver_names": list(active.solver_names),
            "has_gold": bool(gold_count),
            "is_difficult": node_id in difficult,
            "capability": capability_report,
            "cluster_threshold_sensitivity": sensitivity,
        }
        records.extend(node_records)
    records.sort(key=lambda item: (item.node_id, item.candidate_id))
    return CandidateBank(
        records=tuple(records), node_reports=reports, policy=active
    )


__all__ = [
    "CandidateBank",
    "CandidatePolicy",
    "CandidateQuality",
    "CandidateRecord",
    "CandidateSearchMode",
    "solve_candidate_bank",
    "stable_cluster_representatives",
]
