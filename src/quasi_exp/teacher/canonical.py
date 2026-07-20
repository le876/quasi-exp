from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class TeacherVariant(str, Enum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


class ForwardEnvironment(Protocol):
    bounds: np.ndarray

    def fk(self, beta: np.ndarray) -> np.ndarray: ...

    def theta(self, beta: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class TrajectorySpec:
    trajectory_id: str
    family_id: str
    radius_mm: float
    target_xyz_m: np.ndarray
    phase_count: int | None = None
    tube_offsets_mm: tuple[float, ...] = ()
    traversal_direction: str = "forward"
    cyclic_cut: int = 0

    def __post_init__(self) -> None:
        targets = np.asarray(self.target_xyz_m, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) < 3:
            raise ValueError("target_xyz_m must have shape (N, 3) with N >= 3")
        if not np.isfinite(targets).all():
            raise ValueError("target_xyz_m contains non-finite values")
        direction = str(self.traversal_direction).lower()
        if direction not in {"forward", "reverse"}:
            raise ValueError("traversal_direction must be forward or reverse")
        object.__setattr__(self, "target_xyz_m", targets.copy())
        object.__setattr__(self, "traversal_direction", direction)
        object.__setattr__(
            self,
            "phase_count",
            int(len(targets) if self.phase_count is None else self.phase_count),
        )
        object.__setattr__(self, "cyclic_cut", int(self.cyclic_cut) % len(targets))


@dataclass(frozen=True)
class TeacherPolicy:
    variant: TeacherVariant = TeacherVariant.T3
    candidate_budget: int = 32
    damping: float = 1.0e-3
    nullspace_weight: float = 1.0e-3
    beta_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)
    lambda_velocity: float = 1.0
    lambda_acceleration: float = 0.1
    lambda_posture: float = 0.05
    lambda_conditioning: float = 0.01
    closure_weight: float = 5.0
    safe_joint_margin_deg: float = 1.5
    safe_margin_repulsion_step_deg: float = 0.0
    surface_neighbor_weight: float = 0.0
    surface_neighbor_scale_deg: float = 1.0
    max_corrector_iterations: int = 100
    tracking_tolerance_mm: float = 1.0
    solver_seed: int = 20260720
    candidate_cluster_deg: float = 0.5
    max_step_deg: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", TeacherVariant(self.variant))
        if int(self.candidate_budget) < 1:
            raise ValueError("candidate_budget must be positive")
        if len(self.beta_weights) != 6 or min(self.beta_weights) <= 0.0:
            raise ValueError("beta_weights must contain six positive values")
        if self.damping < 0.0 or self.tracking_tolerance_mm <= 0.0:
            raise ValueError("damping and tracking tolerance must be valid")
        if self.safe_joint_margin_deg <= 0.0:
            raise ValueError("safe_joint_margin_deg must be positive")
        if self.safe_margin_repulsion_step_deg < 0.0:
            raise ValueError("safe_margin_repulsion_step_deg must be non-negative")
        if self.surface_neighbor_weight < 0.0 or self.surface_neighbor_scale_deg <= 0.0:
            raise ValueError("surface-neighbor weight/scale must be non-negative/positive")

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["variant"] = self.variant.value
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TeacherTrajectory:
    beta_rad: np.ndarray
    theta_rad: np.ndarray
    achieved_xyz_m: np.ndarray
    target_xyz_m: np.ndarray
    chart_id: np.ndarray
    branch_id: np.ndarray
    metrics: Mapping[str, float]
    provenance: Mapping[str, Any]
    success: bool
    candidate_diagnostics: Mapping[str, Any] = field(default_factory=dict)


def weighted_damped_pinv(
    jacobian: np.ndarray,
    *,
    damping: float,
    weights: np.ndarray | Sequence[float],
) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=float).reshape(3, 6)
    weight = np.asarray(weights, dtype=float).reshape(6)
    if np.any(weight <= 0.0):
        raise ValueError("weights must be positive")
    if float(damping) < 0.0:
        raise ValueError("damping must be non-negative")
    w_inv = np.diag(1.0 / weight)
    task = jac @ w_inv @ jac.T + float(damping) ** 2 * np.eye(3)
    return w_inv @ jac.T @ np.linalg.pinv(task, rcond=1.0e-12)


def beta_rms_deg(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=float).reshape(6) - np.asarray(
        right, dtype=float
    ).reshape(6)
    return float(np.sqrt(np.mean(np.square(delta))) * 180.0 / math.pi)


def _pairwise_beta_rms_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=float).reshape(-1, 6)
    b = np.asarray(right, dtype=float).reshape(-1, 6)
    delta = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.mean(np.square(delta), axis=2)) * 180.0 / math.pi


def link_cyclic_candidates(
    candidate_layers: Sequence[np.ndarray],
    residual_mm: Sequence[np.ndarray],
    *,
    lambda_velocity: float,
    closure_weight: float,
    lambda_posture: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not candidate_layers or len(candidate_layers) != len(residual_mm):
        raise ValueError("candidate layers and residuals must be non-empty and aligned")
    layers = [np.asarray(values, dtype=float).reshape(-1, 6) for values in candidate_layers]
    residuals = [np.asarray(values, dtype=float).reshape(-1) for values in residual_mm]
    if any(len(layer) == 0 for layer in layers):
        return np.zeros((0, 6)), {"success": False, "reason": "empty_layer"}
    if any(len(layer) != len(err) for layer, err in zip(layers, residuals)):
        raise ValueError("each residual vector must match its candidate layer")

    unary: list[np.ndarray] = []
    posture_scale = np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0])
    for layer, error in zip(layers, residuals):
        posture = np.mean(np.square(layer) * posture_scale[None, :], axis=1)
        unary.append(np.square(error) + float(lambda_posture) * posture)
    transitions = [
        float(lambda_velocity)
        * np.square(_pairwise_beta_rms_deg(layers[index - 1], layers[index]))
        for index in range(1, len(layers))
    ]

    best_cost = float("inf")
    best_path: list[int] | None = None
    for start in range(len(layers[0])):
        cost = np.full(len(layers[0]), np.inf)
        cost[start] = unary[0][start]
        parents: list[np.ndarray] = []
        for index in range(1, len(layers)):
            values = cost[:, None] + transitions[index - 1]
            parent = np.argmin(values, axis=0).astype(np.int64)
            cost = values[parent, np.arange(values.shape[1])] + unary[index]
            parents.append(parent)
        seam = _pairwise_beta_rms_deg(layers[-1], layers[0][start : start + 1])[:, 0]
        total = cost + float(closure_weight) * np.square(seam)
        end = int(np.argmin(total))
        if float(total[end]) >= best_cost:
            continue
        path = [end]
        current = end
        for parent in reversed(parents):
            current = int(parent[current])
            path.append(current)
        best_path = list(reversed(path))
        best_cost = float(total[end])

    if best_path is None:
        return np.zeros((0, 6)), {"success": False, "reason": "no_closed_path"}
    selected = np.vstack([layers[i][best_path[i]] for i in range(len(layers))])
    delta = np.asarray(
        [beta_rms_deg(selected[i], selected[(i + 1) % len(selected)]) for i in range(len(selected))]
    )
    return selected, {
        "success": True,
        "cost": best_cost,
        "delta_beta_rms_p95_deg": float(np.percentile(delta, 95)),
        "delta_beta_rms_max_deg": float(np.max(delta)),
        "seam_beta_rms_deg": float(delta[-1]),
    }


def _environment_jacobian(environment: ForwardEnvironment, beta: np.ndarray) -> np.ndarray:
    if hasattr(environment, "jacobian"):
        return np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6)  # type: ignore[attr-defined]
    if hasattr(environment, "numerical_jacobian"):
        return np.asarray(environment.numerical_jacobian(beta), dtype=float).reshape(3, 6)  # type: ignore[attr-defined]
    raise TypeError("forward environment must provide jacobian or numerical_jacobian")


def _posture_margin_gradient(beta: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    values = np.asarray(beta, dtype=float).reshape(6)
    limits = np.asarray(bounds, dtype=float).reshape(6, 2)
    span = np.maximum(limits[:, 1] - limits[:, 0], 1.0e-12)
    mid = 0.5 * (limits[:, 0] + limits[:, 1])
    priority = np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0])
    return 2.0 * priority * (values - mid) / np.square(span)


def _correct_target(
    environment: ForwardEnvironment,
    target: np.ndarray,
    initial: np.ndarray,
    policy: TeacherPolicy,
) -> tuple[np.ndarray, float, int, bool]:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    beta = np.clip(np.asarray(initial, dtype=float).reshape(6), bounds[:, 0], bounds[:, 1])
    tolerance_m = float(policy.tracking_tolerance_mm) / 1000.0
    max_step = math.radians(float(policy.max_step_deg))
    safe_margin = math.radians(float(policy.safe_joint_margin_deg))
    repulsion_cap = math.radians(float(policy.safe_margin_repulsion_step_deg))
    for iteration in range(int(policy.max_corrector_iterations)):
        xyz = np.asarray(environment.fk(beta.reshape(1, 6)), dtype=float).reshape(-1, 3)[0]
        error = np.asarray(target, dtype=float).reshape(3) - xyz
        residual = float(np.linalg.norm(error))
        lower_margin = beta - bounds[:, 0]
        upper_margin = bounds[:, 1] - beta
        margin = np.minimum(lower_margin, upper_margin)
        margin_ok = bool(np.min(margin) >= safe_margin - 1.0e-10)
        if residual <= tolerance_m and (repulsion_cap == 0.0 or margin_ok):
            return beta, residual * 1000.0, iteration, True
        jacobian = _environment_jacobian(environment, beta)
        pinv = weighted_damped_pinv(
            jacobian,
            damping=float(policy.damping),
            weights=np.asarray(policy.beta_weights),
        )
        task_step = pinv @ error
        nullspace = np.eye(6) - pinv @ jacobian
        null_step = -float(policy.nullspace_weight) * (
            nullspace @ _posture_margin_gradient(beta, bounds)
        )
        if repulsion_cap > 0.0:
            deficit_fraction = np.clip((safe_margin - margin) / safe_margin, 0.0, 1.0)
            inward_direction = np.where(lower_margin <= upper_margin, 1.0, -1.0)
            repulsion = repulsion_cap * deficit_fraction * inward_direction
            null_step += nullspace @ repulsion
        step = task_step + null_step
        step_norm = float(np.sqrt(np.mean(np.square(step))))
        if step_norm > max_step:
            step *= max_step / step_norm
        beta = np.clip(beta + step, bounds[:, 0], bounds[:, 1])
    xyz = np.asarray(environment.fk(beta.reshape(1, 6)), dtype=float).reshape(-1, 3)[0]
    residual_mm = float(np.linalg.norm(xyz - np.asarray(target)) * 1000.0)
    return beta, residual_mm, int(policy.max_corrector_iterations), bool(
        residual_mm <= float(policy.tracking_tolerance_mm)
    )


def _deduplicate_candidates(
    candidates: list[tuple[np.ndarray, float]], threshold_deg: float, budget: int
) -> list[tuple[np.ndarray, float]]:
    ordered = sorted(candidates, key=lambda item: item[1])
    kept: list[tuple[np.ndarray, float]] = []
    for beta, residual in ordered:
        if any(beta_rms_deg(beta, previous) < float(threshold_deg) for previous, _ in kept):
            continue
        kept.append((beta, residual))
        if len(kept) >= int(budget):
            break
    return kept or ordered[:1]


def _trajectory_metrics(
    beta: np.ndarray, target: np.ndarray, achieved: np.ndarray
) -> dict[str, float]:
    residual = np.linalg.norm(np.asarray(achieved) - np.asarray(target), axis=1) * 1000.0
    velocity = np.roll(beta, -1, axis=0) - beta
    acceleration = np.roll(beta, -1, axis=0) - 2.0 * beta + np.roll(beta, 1, axis=0)
    delta_deg = np.sqrt(np.mean(np.square(velocity), axis=1)) * 180.0 / math.pi
    acceleration_deg = (
        np.sqrt(np.mean(np.square(acceleration), axis=1)) * 180.0 / math.pi
    )
    return {
        "residual_p95_mm": float(np.percentile(residual, 95)),
        "residual_max_mm": float(np.max(residual)),
        "delta_beta_rms_p95_deg": float(np.percentile(delta_deg, 95)),
        "delta_beta_rms_max_deg": float(np.max(delta_deg)),
        "acceleration_beta_rms_p95_deg": float(np.percentile(acceleration_deg, 95)),
        "seam_beta_rms_deg": float(delta_deg[-1]),
    }


class CanonicalTeacher:
    def __init__(self, environment: ForwardEnvironment):
        bounds = np.asarray(environment.bounds, dtype=float)
        if bounds.shape != (6, 2):
            raise ValueError("forward environment bounds must have shape (6, 2)")
        self._environment = environment

    def solve(
        self,
        spec: TrajectorySpec,
        policy: TeacherPolicy,
        *,
        root_beta: np.ndarray,
        initial_beta_path: np.ndarray | None = None,
        neighbor_anchor_path: np.ndarray | None = None,
    ) -> TeacherTrajectory:
        original_target = np.asarray(spec.target_xyz_m, dtype=float)
        order = np.roll(np.arange(len(original_target)), -int(spec.cyclic_cut))
        if spec.traversal_direction == "reverse":
            order = order[::-1]
        target = original_target[order]
        root = np.asarray(root_beta, dtype=float).reshape(6)
        initial_path = None
        if initial_beta_path is not None:
            initial_path = np.asarray(initial_beta_path, dtype=float)
            if initial_path.shape != (len(original_target), 6):
                raise ValueError(
                    f"initial_beta_path must have shape ({len(original_target)}, 6)"
                )
            if not np.isfinite(initial_path).all():
                raise ValueError("initial_beta_path must contain finite values")
            initial_path = initial_path[order]
        neighbor_anchor = None
        if neighbor_anchor_path is not None:
            neighbor_anchor = np.asarray(neighbor_anchor_path, dtype=float)
            if neighbor_anchor.shape != (len(original_target), 6):
                raise ValueError(
                    f"neighbor_anchor_path must have shape ({len(original_target)}, 6)"
                )
            if not np.isfinite(neighbor_anchor).all():
                raise ValueError("neighbor_anchor_path must contain finite values")
            neighbor_anchor = neighbor_anchor[order]
        rng = np.random.default_rng(int(policy.solver_seed))
        candidate_layers: list[np.ndarray] = []
        residual_layers: list[np.ndarray] = []
        previous: list[np.ndarray] = []
        total_corrector_iterations = 0

        pointwise_targets = (
            () if policy.variant == TeacherVariant.T2 else enumerate(target)
        )
        for index, point in pointwise_targets:
            seeds: list[np.ndarray] = []
            if initial_path is not None:
                seeds.append(initial_path[index])
            if previous:
                seeds.append(previous[-1])
                if len(previous) >= 2:
                    seeds.append(previous[-1] + (previous[-1] - previous[-2]))
                jacobian = _environment_jacobian(self._environment, previous[-1])
                predictor = previous[-1] + weighted_damped_pinv(
                    jacobian,
                    damping=float(policy.damping),
                    weights=np.asarray(policy.beta_weights),
                ) @ (point - target[index - 1])
                seeds.append(predictor)
            seeds.append(root)
            while len(seeds) < int(policy.candidate_budget):
                anchor = seeds[len(seeds) % len(seeds)].copy()
                anchor += rng.normal(0.0, math.radians(0.5), size=6)
                seeds.append(anchor)

            solved: list[tuple[np.ndarray, float]] = []
            for seed in seeds[: int(policy.candidate_budget)]:
                beta, residual_mm, iterations, _success = _correct_target(
                    self._environment, point, seed, policy
                )
                total_corrector_iterations += int(iterations)
                solved.append((beta, residual_mm))
            deduplicated = _deduplicate_candidates(
                solved,
                threshold_deg=float(policy.candidate_cluster_deg),
                budget=int(policy.candidate_budget),
            )
            layer = np.vstack([item[0] for item in deduplicated])
            errors = np.asarray([item[1] for item in deduplicated])
            candidate_layers.append(layer)
            residual_layers.append(errors)
            if previous:
                closest = int(
                    np.argmin([beta_rms_deg(value, previous[-1]) for value in layer])
                )
            else:
                closest = int(np.argmin(errors))
            previous.append(layer[closest])

        if policy.variant in {TeacherVariant.T0, TeacherVariant.T1}:
            selected = np.vstack(previous)
            graph_report: dict[str, Any] = {"success": True, "mode": "continuation"}
        elif policy.variant == TeacherVariant.T2:
            selected = (
                np.asarray(initial_path, dtype=float).copy()
                if initial_path is not None
                else np.tile(root, (len(target), 1))
            )
            graph_report = {"success": True, "mode": "direct_whole_trajectory"}
        else:
            selected, graph_report = link_cyclic_candidates(
                candidate_layers,
                residual_layers,
                lambda_velocity=float(policy.lambda_velocity),
                closure_weight=float(policy.closure_weight),
                lambda_posture=float(policy.lambda_posture),
            )
            if not graph_report.get("success", False):
                selected = np.vstack(previous)

        if policy.variant in {TeacherVariant.T2, TeacherVariant.T3, TeacherVariant.T4}:
            selected = self._optimize_trajectory(
                target, selected, policy, neighbor_anchor=neighbor_anchor
            )

        corrected: list[np.ndarray] = []
        for point, initial in zip(target, selected):
            beta, _residual, iterations, _success = _correct_target(
                self._environment, point, initial, policy
            )
            total_corrector_iterations += int(iterations)
            corrected.append(beta)
        selected = np.vstack(corrected)
        achieved = np.asarray(self._environment.fk(selected), dtype=float).reshape(-1, 3)

        restored_beta = np.empty_like(selected)
        restored_xyz = np.empty_like(achieved)
        restored_beta[order] = selected
        restored_xyz[order] = achieved
        theta = np.asarray(self._environment.theta(restored_beta), dtype=float)
        metrics = _trajectory_metrics(restored_beta, original_target, restored_xyz)
        bounds = np.asarray(self._environment.bounds, dtype=float)
        margin = np.minimum(
            restored_beta - bounds[:, 0][None, :],
            bounds[:, 1][None, :] - restored_beta,
        )
        metrics["joint_margin_min_deg"] = float(np.min(margin) * 180.0 / math.pi)
        metrics["corrector_iteration_count"] = float(total_corrector_iterations)
        multi_branch = []
        for layer, errors in zip(candidate_layers, residual_layers):
            feasible = layer[errors <= float(policy.tracking_tolerance_mm)]
            distinct = any(
                beta_rms_deg(feasible[left], feasible[right]) >= float(policy.candidate_cluster_deg)
                for left in range(len(feasible))
                for right in range(left + 1, len(feasible))
            )
            multi_branch.append(distinct)
        metrics["multi_branch_waypoint_ratio"] = (
            float(np.mean(multi_branch)) if multi_branch else 0.0
        )

        chart_id = np.zeros(len(restored_beta), dtype=np.int64)
        if policy.variant == TeacherVariant.T4:
            current_chart = 0
            for index in range(1, len(restored_beta)):
                if beta_rms_deg(restored_beta[index - 1], restored_beta[index]) > 0.5:
                    current_chart += 1
                chart_id[index] = current_chart
        success = bool(
            metrics["residual_max_mm"] <= float(policy.tracking_tolerance_mm)
            and np.all(margin >= -1.0e-12)
        )
        return TeacherTrajectory(
            beta_rad=restored_beta,
            theta_rad=theta,
            achieved_xyz_m=restored_xyz,
            target_xyz_m=original_target.copy(),
            chart_id=chart_id,
            branch_id=np.zeros(len(restored_beta), dtype=np.int64),
            metrics=metrics,
            provenance={
                "teacher_policy_id": policy.fingerprint,
                "teacher_variant": policy.variant.value,
                "trajectory_id": spec.trajectory_id,
                "family_id": spec.family_id,
                "radius_mm": float(spec.radius_mm),
                "solver_seed": int(policy.solver_seed),
                "traversal_direction": spec.traversal_direction,
                "cyclic_cut": int(spec.cyclic_cut),
                "surface_neighbor_coupled": bool(
                    neighbor_anchor is not None and policy.surface_neighbor_weight > 0.0
                ),
            },
            success=success,
            candidate_diagnostics={
                "layer_counts": [int(len(layer)) for layer in candidate_layers],
                "graph": graph_report,
            },
        )

    def _optimize_trajectory(
        self,
        target: np.ndarray,
        initial: np.ndarray,
        policy: TeacherPolicy,
        *,
        neighbor_anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        bounds = np.asarray(self._environment.bounds, dtype=float).reshape(6, 2)
        count = len(target)
        conditioning_indices = tuple(range(0, count, 6))
        velocity_scale = math.radians(1.0)
        acceleration_scale = math.radians(0.25)

        def residual(flat: np.ndarray) -> np.ndarray:
            beta = flat.reshape(count, 6)
            achieved = np.asarray(self._environment.fk(beta), dtype=float).reshape(-1, 3)
            parts = [((achieved - target) / 0.001).reshape(-1)]
            velocity = np.roll(beta, -1, axis=0) - beta
            parts.append(
                math.sqrt(max(float(policy.lambda_velocity), 0.0))
                * velocity.reshape(-1)
                / velocity_scale
            )
            if policy.lambda_acceleration > 0.0:
                acceleration = (
                    np.roll(beta, -1, axis=0)
                    - 2.0 * beta
                    + np.roll(beta, 1, axis=0)
                )
                parts.append(
                    math.sqrt(float(policy.lambda_acceleration))
                    * acceleration.reshape(-1)
                    / acceleration_scale
                )
            if policy.lambda_posture > 0.0:
                span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
                mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
                priority = np.sqrt(np.asarray(policy.beta_weights))
                parts.append(
                    math.sqrt(float(policy.lambda_posture))
                    * (((beta - mid) / span) * priority[None, :]).reshape(-1)
                )
            if policy.lambda_conditioning > 0.0:
                conditioning = []
                for point_index in conditioning_indices:
                    row_beta = beta[point_index]
                    sigma = np.linalg.svd(
                        _environment_jacobian(self._environment, row_beta),
                        compute_uv=False,
                    )
                    conditioning.append(0.01 / (float(sigma[-1]) + 0.01))
                parts.append(
                    math.sqrt(float(policy.lambda_conditioning))
                    * np.asarray(conditioning, dtype=float)
                )
            if neighbor_anchor is not None and policy.surface_neighbor_weight > 0.0:
                surface_scale = math.radians(float(policy.surface_neighbor_scale_deg))
                parts.append(
                    math.sqrt(float(policy.surface_neighbor_weight))
                    * (beta - neighbor_anchor).reshape(-1)
                    / surface_scale
                )
            safe_margin = math.radians(float(policy.safe_joint_margin_deg))
            lower_margin = beta - bounds[:, 0][None, :]
            upper_margin = bounds[:, 1][None, :] - beta
            margin = np.minimum(lower_margin, upper_margin)
            # Smoothness is unnecessary here: least_squares only needs a
            # stable high-weight one-sided shell penalty.
            parts.append(10.0 * np.maximum(safe_margin - margin, 0.0).reshape(-1) / safe_margin)
            parts.append(
                math.sqrt(max(float(policy.closure_weight), 0.0))
                * (beta[-1] - beta[0])
                / velocity_scale
            )
            return np.concatenate(parts)

        # The FK and posture rows are point-local; velocity/acceleration are
        # cyclic neighbours.  Supplying this pattern prevents a 6T-column
        # dense finite-difference sweep for every iteration.
        residual_count = count * 3 + count * 6
        if policy.lambda_acceleration > 0.0:
            residual_count += count * 6
        if policy.lambda_posture > 0.0:
            residual_count += count * 6
        if policy.lambda_conditioning > 0.0:
            residual_count += len(conditioning_indices)
        if neighbor_anchor is not None and policy.surface_neighbor_weight > 0.0:
            residual_count += count * 6
        residual_count += count * 6
        residual_count += 6
        sparsity = lil_matrix((residual_count, count * 6), dtype=np.int8)
        row = 0
        for point_index in range(count):
            sparsity[row : row + 3, point_index * 6 : (point_index + 1) * 6] = 1
            row += 3
        for point_index in range(count):
            for neighbour in (point_index, (point_index + 1) % count):
                sparsity[row : row + 6, neighbour * 6 : (neighbour + 1) * 6] = 1
            row += 6
        if policy.lambda_acceleration > 0.0:
            for point_index in range(count):
                for neighbour in ((point_index - 1) % count, point_index, (point_index + 1) % count):
                    sparsity[row : row + 6, neighbour * 6 : (neighbour + 1) * 6] = 1
                row += 6
        if policy.lambda_posture > 0.0:
            for point_index in range(count):
                sparsity[row : row + 6, point_index * 6 : (point_index + 1) * 6] = 1
                row += 6
        if policy.lambda_conditioning > 0.0:
            for point_index in conditioning_indices:
                sparsity[row, point_index * 6 : (point_index + 1) * 6] = 1
                row += 1
        if neighbor_anchor is not None and policy.surface_neighbor_weight > 0.0:
            for point_index in range(count):
                sparsity[
                    row : row + 6,
                    point_index * 6 : (point_index + 1) * 6,
                ] = 1
                row += 6
        for point_index in range(count):
            sparsity[row : row + 6, point_index * 6 : (point_index + 1) * 6] = 1
            row += 6
        sparsity[row : row + 6, 0:6] = 1
        sparsity[row : row + 6, (count - 1) * 6 : count * 6] = 1
        row += 6

        result = least_squares(
            residual,
            np.asarray(initial, dtype=float).reshape(-1),
            bounds=(
                np.tile(bounds[:, 0], count),
                np.tile(bounds[:, 1], count),
            ),
            max_nfev=40,
            jac_sparsity=sparsity.tocsr(),
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )
        return np.asarray(result.x, dtype=float).reshape(count, 6)
