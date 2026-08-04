"""Fail-closed inference facade for static and stateful BACRA V14 inverses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import numpy as np

from .canonical import beta_rms_deg
from .workspace_atlas import RepresentationMode
from .workspace_reach import CellKey, WorkspaceGridSpec


Predictor = Callable[[np.ndarray], np.ndarray]
ForwardKinematics = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WorkspaceInversePolicy:
    max_fk_residual_mm: float = 3.0
    stitchable_beta_gap_deg: float = 1.0
    bounds_atol: float = 1.0e-12

    def __post_init__(self) -> None:
        values = (
            self.max_fk_residual_mm,
            self.stitchable_beta_gap_deg,
            self.bounds_atol,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("workspace inverse thresholds must be finite and non-negative")
        if self.max_fk_residual_mm <= 0.0 or self.stitchable_beta_gap_deg <= 0.0:
            raise ValueError("residual and stitchability thresholds must be positive")


@dataclass(frozen=True)
class WorkspaceRegistry:
    """Registered empirical domain used for inference abstention."""

    grid: WorkspaceGridSpec
    allowed_cells: frozenset[CellKey]

    def __post_init__(self) -> None:
        cells = frozenset(self.allowed_cells)
        levels = {cell.level_mm for cell in cells}
        if len(levels) > 1:
            raise ValueError("workspace registry cells must use one grid level")
        if levels and not levels <= set(self.grid.levels_mm):
            raise ValueError("workspace registry uses an unregistered grid level")
        object.__setattr__(self, "allowed_cells", cells)

    @property
    def level_mm(self) -> int:
        if self.allowed_cells:
            return next(iter(self.allowed_cells)).level_mm
        return self.grid.convergence_level_mm

    def contains(self, xyz_m: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_m, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("registry xyz_m must be finite shape (N, 3)")
        result = np.zeros(len(xyz), dtype=bool)
        for index, row in enumerate(xyz):
            cells = self.grid.cells_for_points(row.reshape(1, 3), level_mm=self.level_mm)
            result[index] = bool(cells and next(iter(cells)) in self.allowed_cells)
        return result


@dataclass(frozen=True)
class InverseQuery:
    xyz_m: np.ndarray
    previous_beta_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=float)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, 3)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("xyz_m must be finite shape (N, 3)")
        previous = self.previous_beta_rad
        if previous is not None:
            previous = np.asarray(previous, dtype=float)
            if previous.ndim == 1:
                previous = previous.reshape(1, 6)
            if previous.shape != (len(xyz), 6) or not np.isfinite(previous).all():
                raise ValueError("previous_beta_rad must be finite shape (N, 6)")
        object.__setattr__(self, "xyz_m", xyz.copy())
        object.__setattr__(
            self, "previous_beta_rad", None if previous is None else previous.copy()
        )


@dataclass(frozen=True)
class PredictionResult:
    beta_rad: np.ndarray
    accepted: np.ndarray
    reasons: tuple[str, ...]
    chart_ids: tuple[str, ...]
    fk_residual_mm: np.ndarray
    representation_mode: RepresentationMode


class WorkspaceInverse:
    """One mode-locked inverse with explicit routing and abstention semantics."""

    def __init__(
        self,
        *,
        mode: RepresentationMode,
        beta_bounds_rad: np.ndarray,
        fk: ForwardKinematics,
        global_predictor: Predictor | None = None,
        router_predictor: Predictor | None = None,
        expert_predictors: Mapping[str, Predictor] | None = None,
        registry: WorkspaceRegistry | None = None,
        policy: WorkspaceInversePolicy | None = None,
    ) -> None:
        self.mode = RepresentationMode(mode)
        if self.mode is RepresentationMode.BLOCKED:
            raise ValueError("blocked representation cannot construct WorkspaceInverse")
        bounds = np.asarray(beta_bounds_rad, dtype=float)
        if (
            bounds.shape != (6, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("beta_bounds_rad must be finite ordered shape (6, 2)")
        self.beta_bounds_rad = bounds.copy()
        self.fk = fk
        self.global_predictor = global_predictor
        self.router_predictor = router_predictor
        self.expert_predictors = dict(expert_predictors or {})
        self.chart_ids = tuple(self.expert_predictors)
        self.registry = registry
        self.policy = WorkspaceInversePolicy() if policy is None else policy
        if self.mode is RepresentationMode.XYZ_GLOBAL:
            if self.global_predictor is None:
                raise ValueError("xyz_global requires global_predictor")
        else:
            if self.router_predictor is None or not self.expert_predictors:
                raise ValueError("router representations require router and expert predictors")

    def predict(self, query: InverseQuery) -> PredictionResult:
        xyz = query.xyz_m
        count = len(xyz)
        beta = np.full((count, 6), np.nan, dtype=float)
        accepted = np.zeros(count, dtype=bool)
        reasons = ["unprocessed"] * count
        charts = [""] * count
        residuals = np.full(count, np.nan, dtype=float)
        in_domain = (
            np.ones(count, dtype=bool)
            if self.registry is None
            else self.registry.contains(xyz)
        )
        for index, allowed in enumerate(in_domain):
            if not allowed:
                reasons[index] = "outside_registered_domain"

        if self.mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS and query.previous_beta_rad is None:
            for index in range(count):
                if in_domain[index]:
                    reasons[index] = "previous_beta_required"
            return self._result(beta, accepted, reasons, charts, residuals)

        features = (
            xyz
            if self.mode is not RepresentationMode.STATEFUL_ROUTER_EXPERTS
            else np.hstack([xyz, query.previous_beta_rad])
        )
        if self.mode is RepresentationMode.XYZ_GLOBAL:
            predictions = self._run_predictor(self.global_predictor, features, "global")
            for index in range(count):
                if not in_domain[index]:
                    continue
                valid, reason, residual = self._validate_candidate(
                    predictions[index], xyz[index]
                )
                residuals[index] = residual
                reasons[index] = reason
                if valid:
                    beta[index] = predictions[index]
                    accepted[index] = True
                    charts[index] = "global"
            return self._result(beta, accepted, reasons, charts, residuals)

        probabilities = np.asarray(self.router_predictor(features), dtype=float)
        if probabilities.shape != (count, len(self.chart_ids)) or not np.isfinite(probabilities).all():
            raise ValueError("router predictor must return finite shape (N, chart_count)")
        if np.any(probabilities < 0.0) or np.any(probabilities.sum(axis=1) <= 0.0):
            raise ValueError("router probabilities must be non-negative with positive row sums")
        expert_beta = {
            chart_id: self._run_predictor(predictor, features, chart_id)
            for chart_id, predictor in self.expert_predictors.items()
        }
        for index in range(count):
            if not in_domain[index]:
                continue
            order = np.argsort(-probabilities[index], kind="stable")
            if self.mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
                candidates: list[tuple[float, float, str, np.ndarray]] = []
                for chart_index in order:
                    chart_id = self.chart_ids[int(chart_index)]
                    candidate = expert_beta[chart_id][index]
                    valid, _reason, residual = self._validate_candidate(
                        candidate, xyz[index]
                    )
                    if valid:
                        distance = beta_rms_deg(
                            candidate, query.previous_beta_rad[index]
                        )
                        candidates.append((distance, residual, chart_id, candidate))
                if not candidates:
                    reasons[index] = "no_valid_stateful_candidate"
                    continue
                distance, residual, chart_id, candidate = min(
                    candidates, key=lambda item: (item[0], item[1], item[2])
                )
                del distance
                beta[index] = candidate
                accepted[index] = True
                reasons[index] = "ok"
                charts[index] = chart_id
                residuals[index] = residual
                continue

            top_indices = order[: min(2, len(order))]
            top = []
            for chart_index in top_indices:
                chart_id = self.chart_ids[int(chart_index)]
                candidate = expert_beta[chart_id][index]
                valid, reason, residual = self._validate_candidate(candidate, xyz[index])
                top.append((chart_id, candidate, valid, reason, residual))
            chosen = top[0]
            if len(top) == 2:
                first, second = top
                finite_pair = np.isfinite(first[1]).all() and np.isfinite(second[1]).all()
                gap = (
                    beta_rms_deg(first[1], second[1]) if finite_pair else math.inf
                )
                if first[2] and second[2] and gap > self.policy.stitchable_beta_gap_deg:
                    reasons[index] = "nonstitchable_static_ambiguity"
                    residuals[index] = min(first[4], second[4])
                    continue
                if first[2] and second[2]:
                    chosen = min((first, second), key=lambda item: (item[4], item[0]))
                elif not first[2] and second[2] and gap <= self.policy.stitchable_beta_gap_deg:
                    chosen = second
            chart_id, candidate, valid, reason, residual = chosen
            residuals[index] = residual
            reasons[index] = reason
            if valid:
                beta[index] = candidate
                accepted[index] = True
                charts[index] = chart_id
        return self._result(beta, accepted, reasons, charts, residuals)

    @staticmethod
    def _run_predictor(
        predictor: Predictor | None, features: np.ndarray, name: str
    ) -> np.ndarray:
        if predictor is None:
            raise ValueError(f"missing predictor {name}")
        values = np.asarray(predictor(np.asarray(features, dtype=float)), dtype=float)
        if values.shape != (len(features), 6):
            raise ValueError(f"predictor {name} must return shape (N, 6)")
        return values

    def _validate_candidate(
        self, candidate: np.ndarray, target_xyz_m: np.ndarray
    ) -> tuple[bool, str, float]:
        values = np.asarray(candidate, dtype=float).reshape(6)
        if not np.isfinite(values).all():
            return False, "prediction_nonfinite", math.nan
        atol = self.policy.bounds_atol
        if not (
            np.all(values >= self.beta_bounds_rad[:, 0] - atol)
            and np.all(values <= self.beta_bounds_rad[:, 1] + atol)
        ):
            return False, "prediction_out_of_bounds", math.nan
        achieved = np.asarray(self.fk(values.reshape(1, 6)), dtype=float)
        if achieved.shape != (1, 3) or not np.isfinite(achieved).all():
            return False, "fk_nonfinite_or_wrong_shape", math.nan
        residual = float(
            np.linalg.norm(achieved[0] - np.asarray(target_xyz_m, dtype=float))
            * 1000.0
        )
        if residual > self.policy.max_fk_residual_mm:
            return False, "fk_residual_above_max", residual
        return True, "ok", residual

    def _result(
        self,
        beta: np.ndarray,
        accepted: np.ndarray,
        reasons: list[str],
        charts: list[str],
        residuals: np.ndarray,
    ) -> PredictionResult:
        return PredictionResult(
            beta_rad=beta,
            accepted=accepted,
            reasons=tuple(reasons),
            chart_ids=tuple(charts),
            fk_residual_mm=residuals,
            representation_mode=self.mode,
        )


@dataclass(frozen=True)
class DLSRefinementResult:
    beta_rad: np.ndarray
    residual_mm: np.ndarray
    accepted: np.ndarray
    step_count: int


def refine_inverse_with_bounded_dls(
    environment: Any,
    xyz_m: np.ndarray,
    beta_rad: np.ndarray,
    *,
    steps: int = 1,
    damping: float = 1.0e-3,
    maximum_rms_step_deg: float = 3.0,
    acceptance_residual_mm: float = 10.0,
) -> DLSRefinementResult:
    """Apply one or two legal DLS corrections without post-hoc clipping."""

    targets = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6).copy()
    if (
        len(targets) != len(beta)
        or not np.isfinite(targets).all()
        or not np.isfinite(beta).all()
    ):
        raise ValueError("DLS targets and beta must be finite and aligned")
    count = int(steps)
    if count not in (1, 2):
        raise ValueError("bounded DLS fallback is registered for one or two steps")
    if float(damping) < 0.0 or float(maximum_rms_step_deg) <= 0.0:
        raise ValueError("DLS damping and step limit must be valid")
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    active = np.all(beta >= bounds[:, 0], axis=1) & np.all(
        beta <= bounds[:, 1], axis=1
    )
    max_rms_rad = math.radians(float(maximum_rms_step_deg))
    for _iteration in range(count):
        for index in np.flatnonzero(active):
            current_xyz = np.asarray(
                environment.fk(beta[index].reshape(1, 6)), dtype=float
            ).reshape(-1, 3)[0]
            error = targets[index] - current_xyz
            jacobian = np.asarray(environment.jacobian(beta[index]), dtype=float).reshape(
                3, 6
            )
            gram = jacobian @ jacobian.T + float(damping) ** 2 * np.eye(3)
            try:
                delta = jacobian.T @ np.linalg.solve(gram, error)
            except np.linalg.LinAlgError:
                active[index] = False
                continue
            rms = float(np.sqrt(np.mean(np.square(delta))))
            if rms > max_rms_rad:
                delta *= max_rms_rad / rms
            accepted_step = False
            scale = 1.0
            previous_error = float(np.linalg.norm(error))
            for _backtrack in range(12):
                proposal = beta[index] + scale * delta
                if np.all(proposal >= bounds[:, 0]) and np.all(
                    proposal <= bounds[:, 1]
                ):
                    proposal_xyz = np.asarray(
                        environment.fk(proposal.reshape(1, 6)), dtype=float
                    ).reshape(-1, 3)[0]
                    if (
                        float(np.linalg.norm(targets[index] - proposal_xyz))
                        <= previous_error + 1.0e-15
                    ):
                        beta[index] = proposal
                        accepted_step = True
                        break
                scale *= 0.5
            if not accepted_step:
                active[index] = False
    achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    residual = np.linalg.norm(achieved - targets, axis=1) * 1000.0
    accepted = (
        np.isfinite(residual)
        & (residual <= float(acceptance_residual_mm))
        & np.all(beta >= bounds[:, 0], axis=1)
        & np.all(beta <= bounds[:, 1], axis=1)
    )
    return DLSRefinementResult(beta, residual, accepted, count)
