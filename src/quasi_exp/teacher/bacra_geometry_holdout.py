"""Independent task-geometry closed-loop diagnostics for frozen BACRA Students."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from .capability_path_bridge import (
    analyze_hard_cyclic_reachability,
    independently_certify_hard_cyclic_path,
)
from .canonical import (
    TeacherPolicy,
    _correct_target,
    beta_rms_deg,
    link_cyclic_candidates,
    weighted_damped_pinv,
)
from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .multi_ik_candidates import (
    CandidatePolicy,
    CandidateQuality,
    solve_candidate_bank,
)
from .region import EllipseFamilySpec


class GeometryHoldoutEnvironment(Protocol):
    bounds: np.ndarray

    def fk(self, beta_rad: np.ndarray) -> np.ndarray: ...

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LoopReferencePolicy:
    """Frozen root-connection, label-validity, and path-invariance thresholds."""

    phase_count: int = 720
    max_tetrahedron_edge_mm: float = 12.99038105676658
    target_support_max_mm: float = 7.5
    continuation_step_mm: float = 2.0
    residual_p95_mm: float = 1.0
    residual_max_mm: float = 3.0
    joint_margin_min_deg: float = 1.5
    phase_beta_rms_p95_deg: float = 1.0
    phase_beta_rms_max_deg: float = 2.0
    acceleration_beta_rms_p95_deg: float = 0.5
    seam_beta_rms_deg: float = 1.0
    direction_gap_p95_deg: float = 0.5
    direction_gap_max_deg: float = 1.0
    candidate_nearest_count: int = 32
    candidate_representative_count: int = 8
    candidate_budget_per_phase: int = 16
    candidate_cluster_deg: float = 0.5
    gap_enrichment_rounds: int = 0
    gap_enrichment_sources_per_frontier: int = 2
    gap_enrichment_round_candidate_budget: int = 96
    gap_enrichment_max_candidates_per_phase: int = 64
    gap_enrichment_topology_dedup_deg: float = 1.0e-6
    teacher_policy: TeacherPolicy = TeacherPolicy()

    def __post_init__(self) -> None:
        if int(self.phase_count) < 4:
            raise ValueError("phase_count must be at least four")
        for name in (
            "max_tetrahedron_edge_mm",
            "target_support_max_mm",
            "continuation_step_mm",
            "residual_p95_mm",
            "residual_max_mm",
            "joint_margin_min_deg",
            "phase_beta_rms_p95_deg",
            "phase_beta_rms_max_deg",
            "acceleration_beta_rms_p95_deg",
            "seam_beta_rms_deg",
            "direction_gap_p95_deg",
            "direction_gap_max_deg",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(self.continuation_step_mm) <= 0.0:
            raise ValueError("continuation_step_mm must be positive")
        if (
            int(self.candidate_nearest_count) < 1
            or int(self.candidate_representative_count) < 1
            or int(self.candidate_budget_per_phase) < 1
        ):
            raise ValueError("candidate-bank counts must be positive")
        if float(self.candidate_cluster_deg) <= 0.0:
            raise ValueError("candidate_cluster_deg must be positive")
        if int(self.gap_enrichment_rounds) < 0:
            raise ValueError("gap_enrichment_rounds must be non-negative")
        if int(self.gap_enrichment_sources_per_frontier) < 1:
            raise ValueError(
                "gap_enrichment_sources_per_frontier must be positive"
            )
        if int(self.gap_enrichment_round_candidate_budget) < 1:
            raise ValueError(
                "gap_enrichment_round_candidate_budget must be positive"
            )
        if int(self.gap_enrichment_max_candidates_per_phase) < int(
            self.candidate_budget_per_phase
        ):
            raise ValueError(
                "gap_enrichment_max_candidates_per_phase must cover "
                "the initial candidate budget"
            )
        if (
            not math.isfinite(
                float(self.gap_enrichment_topology_dedup_deg)
            )
            or float(self.gap_enrichment_topology_dedup_deg) <= 0.0
        ):
            raise ValueError(
                "gap_enrichment_topology_dedup_deg must be finite "
                "and positive"
            )


def _scale(values: np.ndarray, limits: Sequence[float]) -> np.ndarray:
    low, high = map(float, limits)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("holdout range must contain finite low < high")
    return low + np.asarray(values, dtype=float) * (high - low)


def generate_geometry_holdout(
    anchor: EllipseFamilySpec,
    groups: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Generate a deterministic, pre-label family inventory from LHS designs."""

    rows: list[dict[str, Any]] = []
    global_order = 0
    anchor_axes = np.column_stack(
        [
            anchor.major_direction,
            anchor.minor_direction,
            anchor.plane_normal,
        ]
    )
    for group in groups:
        group_id = str(group["group_id"])
        count = int(group["count"])
        seed = int(group["seed"])
        if count < 1:
            raise ValueError("geometry holdout group count must be positive")
        sample = qmc.LatinHypercube(d=8, seed=seed).random(count)
        major = _scale(sample[:, 0], group["major_semiaxis_m"])
        ratio = _scale(sample[:, 1], group["axis_ratio"])
        q1_mm = _scale(sample[:, 2], group["center_q1_offset_mm"])
        q2_mm = _scale(sample[:, 3], group["center_q2_offset_mm"])
        normal_mm = _scale(
            sample[:, 4], group["center_normal_offset_mm"]
        )
        tilt_deg = np.column_stack(
            [
                _scale(sample[:, index], group["tilt_deg"])
                for index in range(5, 8)
            ]
        )
        for local_order in range(count):
            rotation = Rotation.from_rotvec(
                anchor_axes @ np.deg2rad(tilt_deg[local_order])
            )
            major_direction = rotation.apply(anchor.major_direction)
            minor_direction = rotation.apply(anchor.minor_direction)
            center = (
                anchor.center_m
                + q1_mm[local_order]
                / 1000.0
                * anchor.major_direction
                + q2_mm[local_order]
                / 1000.0
                * anchor.minor_direction
                + normal_mm[local_order]
                / 1000.0
                * anchor.plane_normal
            )
            family_id = f"{group_id}_{local_order:02d}"
            family = EllipseFamilySpec(
                family_id=family_id,
                center_m=center,
                major_direction=major_direction,
                minor_direction=minor_direction,
                major_semiaxis_m=float(major[local_order]),
                minor_semiaxis_m=float(
                    major[local_order] * ratio[local_order]
                ),
                metadata={
                    "group_id": group_id,
                    "generation_seed": seed,
                    "catalog_order": global_order,
                },
            )
            rows.append(
                {
                    "family_id": family_id,
                    "group_id": group_id,
                    "catalog_order": global_order,
                    "group_order": local_order,
                    "generation_seed": seed,
                    "major_semiaxis_m": family.major_semiaxis_m,
                    "axis_ratio": family.axis_ratio,
                    "center_x_m": float(center[0]),
                    "center_y_m": float(center[1]),
                    "center_z_m": float(center[2]),
                    "major_x": float(major_direction[0]),
                    "major_y": float(major_direction[1]),
                    "major_z": float(major_direction[2]),
                    "minor_x": float(minor_direction[0]),
                    "minor_y": float(minor_direction[1]),
                    "minor_z": float(minor_direction[2]),
                    "center_q1_offset_mm": float(q1_mm[local_order]),
                    "center_q2_offset_mm": float(q2_mm[local_order]),
                    "center_normal_offset_mm": float(
                        normal_mm[local_order]
                    ),
                    "tilt_q1_deg": float(tilt_deg[local_order, 0]),
                    "tilt_q2_deg": float(tilt_deg[local_order, 1]),
                    "tilt_normal_deg": float(tilt_deg[local_order, 2]),
                    "family_fingerprint": family.fingerprint,
                }
            )
            global_order += 1
    output = pd.DataFrame(rows).sort_values(
        "catalog_order", kind="stable"
    )
    if not output["family_id"].is_unique:
        raise ValueError("geometry holdout family ids must be unique")
    return output.reset_index(drop=True)


def family_from_row(row: Mapping[str, Any] | pd.Series) -> EllipseFamilySpec:
    """Reconstruct one frozen family from its materialized catalog row."""

    return EllipseFamilySpec(
        family_id=str(row["family_id"]),
        center_m=np.asarray(
            [row["center_x_m"], row["center_y_m"], row["center_z_m"]],
            dtype=float,
        ),
        major_direction=np.asarray(
            [row["major_x"], row["major_y"], row["major_z"]],
            dtype=float,
        ),
        minor_direction=np.asarray(
            [row["minor_x"], row["minor_y"], row["minor_z"]],
            dtype=float,
        ),
        major_semiaxis_m=float(row["major_semiaxis_m"]),
        minor_semiaxis_m=float(
            row["major_semiaxis_m"] * row["axis_ratio"]
        ),
        metadata={
            "group_id": str(row["group_id"]),
            "generation_seed": int(row["generation_seed"]),
            "catalog_order": int(row["catalog_order"]),
        },
    )


def weighted_beta_gap_deg(
    left: np.ndarray,
    right: np.ndarray,
    *,
    weights: Sequence[float],
) -> np.ndarray:
    """Return the per-row weighted RMS beta gap in degrees."""

    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.shape != rhs.shape or lhs.ndim != 2 or lhs.shape[1] != 6:
        raise ValueError("beta arrays must share shape (N, 6)")
    weight = np.asarray(weights, dtype=float).reshape(6)
    return np.rad2deg(
        np.sqrt(
            np.sum(weight[None, :] * np.square(lhs - rhs), axis=1)
            / np.sum(weight)
        )
    )


def cyclic_beta_metrics(beta_rad: np.ndarray) -> dict[str, float]:
    """Measure velocity, acceleration, and seam over a complete cyclic path."""

    beta = np.asarray(beta_rad, dtype=float)
    if beta.ndim != 2 or beta.shape[1] != 6 or len(beta) < 4:
        raise ValueError("beta_rad must have shape (N, 6), N >= 4")
    if not np.isfinite(beta).all():
        raise ValueError("cyclic beta path must be finite")
    velocity = np.roll(beta, -1, axis=0) - beta
    acceleration = (
        np.roll(beta, -1, axis=0)
        - 2.0 * beta
        + np.roll(beta, 1, axis=0)
    )
    velocity_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(velocity), axis=1))
    )
    acceleration_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(acceleration), axis=1))
    )
    return {
        "phase_beta_rms_p95_deg": float(
            np.percentile(velocity_deg, 95)
        ),
        "phase_beta_rms_max_deg": float(np.max(velocity_deg)),
        "acceleration_beta_rms_p95_deg": float(
            np.percentile(acceleration_deg, 95)
        ),
        "seam_beta_rms_deg": float(velocity_deg[-1]),
    }


def _margin_deg(beta_rad: np.ndarray, bounds_rad: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float)
    bounds = np.asarray(bounds_rad, dtype=float).reshape(6, 2)
    return np.rad2deg(
        np.min(
            np.minimum(
                beta - bounds[:, 0][None, :],
                bounds[:, 1][None, :] - beta,
            ),
            axis=1,
        )
    )


def _robust_correct_target(
    environment: GeometryHoldoutEnvironment,
    target: np.ndarray,
    initial: np.ndarray,
    policy: TeacherPolicy,
) -> tuple[np.ndarray, float, int, bool]:
    """Use the legacy corrector first, then its existing bounded-LS backend."""

    value, error_mm, iterations, ok = _correct_target(
        environment,
        target,
        initial,
        policy,
    )
    if ok:
        return value, error_mm, iterations, True

    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    seed = np.asarray(initial, dtype=float).reshape(6)
    xyz = np.asarray(environment.fk(seed.reshape(1, 6)), dtype=float).reshape(
        -1, 3
    )[0]
    jacobian = np.asarray(environment.jacobian(seed), dtype=float).reshape(
        3, 6
    )
    predictor = seed + weighted_damped_pinv(
        jacobian,
        damping=float(policy.damping),
        weights=policy.beta_weights,
    ) @ (np.asarray(target, dtype=float).reshape(3) - xyz)
    if np.all(predictor >= bounds[:, 0]) and np.all(
        predictor <= bounds[:, 1]
    ):
        seed = predictor

    def residual(beta: np.ndarray) -> np.ndarray:
        achieved = np.asarray(
            environment.fk(np.asarray(beta).reshape(1, 6)),
            dtype=float,
        ).reshape(-1, 3)[0]
        return (achieved - np.asarray(target, dtype=float).reshape(3)) / 0.001

    solved = least_squares(
        residual,
        seed,
        bounds=(bounds[:, 0], bounds[:, 1]),
        max_nfev=int(policy.max_corrector_iterations),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    beta = np.asarray(solved.x, dtype=float).reshape(6)
    residual_mm = float(np.linalg.norm(residual(beta)))
    return (
        beta,
        residual_mm,
        int(iterations) + int(solved.nfev),
        bool(
            solved.success
            and residual_mm <= float(policy.tracking_tolerance_mm)
        ),
    )


def _continue_segment(
    environment: GeometryHoldoutEnvironment,
    source_target: np.ndarray,
    destination_target: np.ndarray,
    start_beta: np.ndarray,
    policy: TeacherPolicy,
    *,
    step_mm: float,
    recovery: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, float, int, bool]
    ]
    | None = None,
) -> tuple[np.ndarray, float, int, int, float, int, bool]:
    """Correct along a deterministic straight task-space segment."""

    source = np.asarray(source_target, dtype=float).reshape(3)
    destination = np.asarray(destination_target, dtype=float).reshape(3)
    distance_mm = float(np.linalg.norm(destination - source) * 1000.0)
    substeps = max(1, int(math.ceil(distance_mm / float(step_mm))))
    current = np.asarray(start_beta, dtype=float).reshape(6)
    total_iterations = 0
    maximum_residual_mm = 0.0
    final_residual_mm = math.inf
    recovery_count = 0
    for substep in range(1, substeps + 1):
        target = source + (destination - source) * (
            float(substep) / float(substeps)
        )
        value, error_mm, n_iter, ok = _robust_correct_target(
            environment,
            target,
            current,
            policy,
        )
        current = value
        final_residual_mm = float(error_mm)
        maximum_residual_mm = max(maximum_residual_mm, final_residual_mm)
        total_iterations += int(n_iter)
        if ok:
            bounds = np.asarray(environment.bounds, dtype=float).reshape(
                6, 2
            )
            margin_deg = float(
                _margin_deg(current.reshape(1, 6), bounds)[0]
            )
            ok = bool(
                margin_deg >= float(policy.safe_joint_margin_deg)
            )
        if not ok and recovery is not None:
            value, error_mm, n_iter, ok = recovery(target, current)
            current = value
            final_residual_mm = float(error_mm)
            maximum_residual_mm = max(
                maximum_residual_mm, final_residual_mm
            )
            total_iterations += int(n_iter)
            recovery_count += int(ok)
        if not ok:
            return (
                current,
                final_residual_mm,
                total_iterations,
                substep,
                maximum_residual_mm,
                recovery_count,
                False,
            )
    return (
        current,
        final_residual_mm,
        total_iterations,
        substeps,
        maximum_residual_mm,
        recovery_count,
        True,
    )


def _walk_loop(
    environment: GeometryHoldoutEnvironment,
    targets: np.ndarray,
    root_index: int,
    root_beta: np.ndarray,
    order: Sequence[int],
    policy: TeacherPolicy,
    *,
    step_mm: float,
    recovery: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, float, int, bool]
    ]
    | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    count = len(targets)
    beta = np.full((count, 6), np.nan, dtype=float)
    residual = np.full(count, np.nan, dtype=float)
    iterations = np.zeros(count, dtype=np.int64)
    substeps = np.zeros(count, dtype=np.int64)
    maximum_substep_residual = np.full(count, np.nan, dtype=float)
    recovery_count = np.zeros(count, dtype=np.int64)
    success = np.zeros(count, dtype=bool)
    root_index = int(root_index)
    beta[root_index] = np.asarray(root_beta, dtype=float)
    residual[root_index] = float(
        np.linalg.norm(
            np.asarray(environment.fk(beta[root_index].reshape(1, 6)))[0]
            - targets[root_index]
        )
        * 1000.0
    )
    maximum_substep_residual[root_index] = residual[root_index]
    success[root_index] = (
        residual[root_index] <= float(policy.tracking_tolerance_mm)
    )
    current = beta[root_index]
    current_index = root_index
    for phase_idx in order:
        (
            value,
            error_mm,
            n_iter,
            n_substeps,
            max_residual_mm,
            n_recoveries,
            ok,
        ) = _continue_segment(
            environment,
            targets[current_index],
            targets[int(phase_idx)],
            current,
            policy,
            step_mm=step_mm,
            recovery=recovery,
        )
        beta[int(phase_idx)] = value
        residual[int(phase_idx)] = float(error_mm)
        iterations[int(phase_idx)] = int(n_iter)
        substeps[int(phase_idx)] = int(n_substeps)
        maximum_substep_residual[int(phase_idx)] = float(max_residual_mm)
        recovery_count[int(phase_idx)] = int(n_recoveries)
        success[int(phase_idx)] = bool(ok)
        if not ok:
            break
        current = value
        current_index = int(phase_idx)
    (
        closed,
        _error,
        _n_iter,
        _n_substeps,
        _max_residual_mm,
        _n_recoveries,
        closed_ok,
    ) = _continue_segment(
        environment,
        targets[current_index],
        targets[root_index],
        current,
        policy,
        step_mm=step_mm,
        recovery=recovery,
    )
    closure_gap_deg = float(
        weighted_beta_gap_deg(
            closed.reshape(1, 6),
            beta[root_index].reshape(1, 6),
            weights=policy.beta_weights,
        )[0]
    )
    if not closed_ok:
        closure_gap_deg = 1.0e300
    return (
        beta,
        residual,
        iterations,
        substeps,
        maximum_substep_residual,
        recovery_count,
        success,
        closure_gap_deg,
    )


def _gap_directed_enrich_layers(
    environment: GeometryHoldoutEnvironment,
    targets: np.ndarray,
    candidate_layers: Sequence[np.ndarray],
    residual_layers: Sequence[np.ndarray],
    candidate_sources: Sequence[Sequence[str]],
    policy: LoopReferencePolicy,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[list[str]],
    dict[str, Any],
]:
    """Propagate only root-identity frontiers until a hard cycle appears."""

    layers = [
        np.asarray(values, dtype=float).reshape(-1, 6).copy()
        for values in candidate_layers
    ]
    residuals = [
        np.asarray(values, dtype=float).reshape(-1).copy()
        for values in residual_layers
    ]
    sources = [list(map(str, values)) for values in candidate_sources]
    rounds: list[dict[str, Any]] = []
    total_added = 0
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    gold_shell = math.radians(float(policy.joint_margin_min_deg))
    gold_lower = bounds[:, 0] + gold_shell
    gold_upper = bounds[:, 1] - gold_shell
    hard_edge_limit_rad = math.radians(
        float(policy.phase_beta_rms_max_deg)
    )
    if np.any(gold_lower > gold_upper):
        raise ValueError("Gold shell is empty inside mechanical bounds")

    def hard_edge_correct(
        target: np.ndarray, source_beta: np.ndarray
    ) -> tuple[np.ndarray, float, int, bool]:
        """Solve one target inside the frozen Gold shell and 2° RMS ball."""

        source = np.asarray(source_beta, dtype=float).reshape(6)
        destination = np.asarray(target, dtype=float).reshape(3)

        def objective(beta: np.ndarray) -> float:
            achieved = np.asarray(
                environment.fk(np.asarray(beta).reshape(1, 6)),
                dtype=float,
            ).reshape(-1, 3)[0]
            residual_mm = (achieved - destination) / 0.001
            delta = (np.asarray(beta) - source) / hard_edge_limit_rad
            return float(
                np.dot(residual_mm, residual_mm)
                + 1.0e-6
                * np.dot(
                    np.asarray(policy.teacher_policy.beta_weights)
                    * delta,
                    delta,
                )
            )

        def edge_constraint(beta: np.ndarray) -> float:
            delta = np.asarray(beta, dtype=float) - source
            return float(
                hard_edge_limit_rad**2 - np.mean(np.square(delta))
            )

        solved = minimize(
            objective,
            np.clip(source, gold_lower, gold_upper),
            method="SLSQP",
            bounds=list(zip(gold_lower, gold_upper)),
            constraints={"type": "ineq", "fun": edge_constraint},
            options={
                "maxiter": int(
                    policy.teacher_policy.max_corrector_iterations
                ),
                "ftol": 1.0e-12,
                "disp": False,
            },
        )
        beta = np.asarray(solved.x, dtype=float).reshape(6)
        achieved = np.asarray(
            environment.fk(beta.reshape(1, 6)), dtype=float
        ).reshape(-1, 3)[0]
        residual_mm = float(
            np.linalg.norm(achieved - destination) * 1000.0
        )
        transition_deg = beta_rms_deg(source, beta)
        margin_deg = float(_margin_deg(beta.reshape(1, 6), bounds)[0])
        success = bool(
            np.isfinite(beta).all()
            and residual_mm <= float(policy.residual_max_mm)
            and transition_deg
            <= float(policy.phase_beta_rms_max_deg) + 1.0e-7
            and margin_deg
            >= float(policy.joint_margin_min_deg) - 1.0e-7
        )
        return beta, residual_mm, int(solved.nfev), success

    for round_index in range(int(policy.gap_enrichment_rounds) + 1):
        try:
            reachability = analyze_hard_cyclic_reachability(
                layers,
                max_transition_deg=float(
                    policy.phase_beta_rms_max_deg
                ),
                frontier_sources_per_root=int(
                    policy.gap_enrichment_sources_per_frontier
                ),
            )
        except ValueError as error:
            if "candidate layers must be non-empty" not in str(error):
                raise
            return (
                layers,
                residuals,
                sources,
                {
                    "method": (
                        "gap_directed_bidirectional_predictor_corrector"
                    ),
                    "hard_transition_limit_deg": float(
                        policy.phase_beta_rms_max_deg
                    ),
                    "round_budget": int(policy.gap_enrichment_rounds),
                    "round_candidate_budget": int(
                        policy.gap_enrichment_round_candidate_budget
                    ),
                    "max_candidates_per_phase": int(
                        policy.gap_enrichment_max_candidates_per_phase
                    ),
                    "total_added_candidate_count": 0,
                    "rounds": [],
                    "final_reachability": None,
                    "closed_cycle_exists": False,
                    "stopped_reason": "empty_gold_candidate_layer",
                    "empty_phase_indices": [
                        int(index)
                        for index, layer in enumerate(layers)
                        if len(layer) == 0
                    ],
                },
            )
        reachability_report = reachability.to_report()
        if reachability.closed_cycle_exists:
            rounds.append(
                {
                    "round_index": int(round_index),
                    "candidate_count_before": int(
                        sum(len(layer) for layer in layers)
                    ),
                    "candidate_additions": [],
                    "reachability": reachability_report,
                }
            )
            break
        if round_index >= int(policy.gap_enrichment_rounds):
            rounds.append(
                {
                    "round_index": int(round_index),
                    "candidate_count_before": int(
                        sum(len(layer) for layer in layers)
                    ),
                    "candidate_additions": [],
                    "reachability": reachability_report,
                }
            )
            break

        additions: list[dict[str, Any]] = []
        attempted: set[tuple[int, int, int, str]] = set()
        ordered_frontiers = sorted(
            reachability.frontiers,
            key=lambda item: (
                item.nearest_existing_gap_deg,
                item.source_phase_idx,
                item.destination_phase_idx,
                item.source_candidate_idx,
                item.direction,
                item.root_candidate_idx,
            ),
        )
        for frontier in ordered_frontiers:
            if len(additions) >= int(
                policy.gap_enrichment_round_candidate_budget
            ):
                break
            key = (
                int(frontier.source_phase_idx),
                int(frontier.destination_phase_idx),
                int(frontier.source_candidate_idx),
                str(frontier.direction),
            )
            if key in attempted:
                continue
            attempted.add(key)
            source_phase = int(frontier.source_phase_idx)
            destination_phase = int(frontier.destination_phase_idx)
            if len(layers[destination_phase]) >= int(
                policy.gap_enrichment_max_candidates_per_phase
            ):
                continue
            source_index = int(frontier.source_candidate_idx)
            if source_index >= len(layers[source_phase]):
                continue
            (
                beta,
                residual_mm,
                iterations,
                substeps,
                maximum_substep_residual_mm,
                _recovery_count,
                success,
            ) = _continue_segment(
                environment,
                targets[source_phase],
                targets[destination_phase],
                layers[source_phase][source_index],
                policy.teacher_policy,
                step_mm=float(policy.continuation_step_mm),
            )
            source_beta = layers[source_phase][source_index]
            source_transition_deg = beta_rms_deg(source_beta, beta)
            correction_method = "predictor_corrector"
            if (
                not success
                or source_transition_deg
                > float(policy.phase_beta_rms_max_deg) + 1.0e-7
            ):
                (
                    beta,
                    residual_mm,
                    constrained_nfev,
                    success,
                ) = hard_edge_correct(
                    targets[destination_phase], source_beta
                )
                iterations += int(constrained_nfev)
                source_transition_deg = beta_rms_deg(
                    source_beta, beta
                )
                correction_method = "hard_edge_constrained_corrector"
            minimum_existing_gap = float(
                min(
                    beta_rms_deg(beta, existing)
                    for existing in layers[destination_phase]
                )
            )
            accepted = bool(
                success
                and np.isfinite(beta).all()
                and source_transition_deg
                <= float(policy.phase_beta_rms_max_deg) + 1.0e-7
                and minimum_existing_gap
                >= float(policy.gap_enrichment_topology_dedup_deg)
            )
            entry = {
                "source_phase_idx": source_phase,
                "destination_phase_idx": destination_phase,
                "source_candidate_idx": source_index,
                "direction": str(frontier.direction),
                "root_candidate_idx": int(
                    frontier.root_candidate_idx
                ),
                "nearest_existing_gap_before_deg": float(
                    frontier.nearest_existing_gap_deg
                ),
                "corrected_candidate_nearest_gap_deg": (
                    minimum_existing_gap
                ),
                "source_transition_deg": float(
                    source_transition_deg
                ),
                "correction_method": correction_method,
                "residual_mm": float(residual_mm),
                "corrector_iterations": int(iterations),
                "continuation_substeps": int(substeps),
                "maximum_substep_residual_mm": float(
                    maximum_substep_residual_mm
                ),
                "accepted": accepted,
            }
            additions.append(entry)
            if not accepted:
                continue
            source_label = (
                f"gap_round_{round_index:02d}_{frontier.direction}_"
                f"p{source_phase:04d}_c{source_index:03d}"
            )
            layers[destination_phase] = np.vstack(
                [layers[destination_phase], beta]
            )
            residuals[destination_phase] = np.append(
                residuals[destination_phase], float(residual_mm)
            )
            sources[destination_phase].append(source_label)
            total_added += 1

        rounds.append(
            {
                "round_index": int(round_index),
                "candidate_count_before": int(
                    sum(len(layer) for layer in layers)
                    - sum(bool(item["accepted"]) for item in additions)
                ),
                "candidate_count_after": int(
                    sum(len(layer) for layer in layers)
                ),
                "candidate_additions": additions,
                "reachability": reachability_report,
            }
        )
        if not any(bool(item["accepted"]) for item in additions):
            break

    final = analyze_hard_cyclic_reachability(
        layers,
        max_transition_deg=float(policy.phase_beta_rms_max_deg),
        frontier_sources_per_root=int(
            policy.gap_enrichment_sources_per_frontier
        ),
    )
    return (
        layers,
        residuals,
        sources,
        {
            "method": (
                "gap_directed_bidirectional_predictor_corrector"
            ),
            "hard_transition_limit_deg": float(
                policy.phase_beta_rms_max_deg
            ),
            "round_budget": int(policy.gap_enrichment_rounds),
            "round_candidate_budget": int(
                policy.gap_enrichment_round_candidate_budget
            ),
            "max_candidates_per_phase": int(
                policy.gap_enrichment_max_candidates_per_phase
            ),
            "topology_dedup_deg": float(
                policy.gap_enrichment_topology_dedup_deg
            ),
            "total_added_candidate_count": int(total_added),
            "rounds": rounds,
            "final_reachability": final.to_report(),
            "closed_cycle_exists": bool(final.closed_cycle_exists),
        },
    )


def solve_loop_reference(
    environment: GeometryHoldoutEnvironment,
    chart_frame: pd.DataFrame,
    family: EllipseFamilySpec,
    *,
    group_id: str,
    policy: LoopReferencePolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Connect a frozen chart to a loop and build bidirectional evidence."""

    required = {*XYZ_COLUMNS, *BETA_COLUMNS}
    missing = sorted(required - set(chart_frame.columns))
    if missing:
        raise ValueError(f"canonical chart missing columns: {missing}")
    chart = chart_frame.sort_values(
        [column for column in ("chart_id", "node_id") if column in chart_frame],
        kind="stable",
    )
    chart_xyz = chart.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    chart_beta = chart.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    triangulation = Delaunay(chart_xyz)
    tree = cKDTree(chart_xyz)
    targets = family.centerline(phase_count=int(policy.phase_count))
    simplex_ids = triangulation.find_simplex(targets).astype(np.int64)
    support_mm = tree.query(targets, k=1)[0] * 1000.0
    tetrahedron_edge_mm = np.full(len(targets), np.nan, dtype=float)
    supported = np.zeros(len(targets), dtype=bool)

    for phase_idx, simplex_idx in enumerate(simplex_ids):
        if simplex_idx < 0:
            continue
        vertices = triangulation.simplices[int(simplex_idx)]
        vertex_xyz = chart_xyz[vertices]
        tetrahedron_edge_mm[phase_idx] = max(
            np.linalg.norm(vertex_xyz[left] - vertex_xyz[right])
            * 1000.0
            for left in range(4)
            for right in range(left + 1, 4)
        )
        supported[phase_idx] = bool(
            support_mm[phase_idx] <= float(policy.target_support_max_mm)
            and tetrahedron_edge_mm[phase_idx]
            <= float(policy.max_tetrahedron_edge_mm)
        )
        if not supported[phase_idx]:
            continue

    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)

    def recover_from_chart(
        target: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, float, int, bool]:
        """Recover a stalled local corrector from frozen chart seeds."""

        target = np.asarray(target, dtype=float).reshape(3)
        seeds: list[np.ndarray] = []
        simplex_index = int(triangulation.find_simplex(target))
        if simplex_index >= 0:
            vertices = triangulation.simplices[simplex_index]
            transform = triangulation.transform[simplex_index]
            first = transform[:3] @ (target - transform[3])
            barycentric = np.append(first, 1.0 - np.sum(first))
            seeds.append(barycentric @ chart_beta[vertices])
        _distance, nearest = tree.query(
            target, k=min(8, len(chart_xyz))
        )
        for chart_index in np.asarray(nearest, dtype=np.int64).reshape(-1):
            seeds.append(chart_beta[int(chart_index)])

        candidates: list[
            tuple[float, float, float, int, np.ndarray, int]
        ] = []
        for seed_order, seed in enumerate(seeds):
            value, error_mm, n_iter, ok = _robust_correct_target(
                environment,
                target,
                seed,
                policy.teacher_policy,
            )
            if not ok:
                continue
            gap_deg = float(
                weighted_beta_gap_deg(
                    value.reshape(1, 6),
                    np.asarray(current).reshape(1, 6),
                    weights=policy.teacher_policy.beta_weights,
                )[0]
            )
            candidate_margin_deg = float(
                _margin_deg(value.reshape(1, 6), bounds)[0]
            )
            if candidate_margin_deg < float(
                policy.joint_margin_min_deg
            ):
                continue
            candidates.append(
                (
                    gap_deg,
                    -candidate_margin_deg,
                    float(error_mm),
                    seed_order,
                    value,
                    int(n_iter),
                )
            )
        if not candidates:
            return (
                np.asarray(current, dtype=float).reshape(6),
                1.0e300,
                0,
                False,
            )
        selected = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )
        return (
            np.asarray(selected[4], dtype=float),
            float(selected[2]),
            int(selected[5]),
            True,
        )

    root_index = int(np.argmin(support_mm))
    root_simplex = int(simplex_ids[root_index])
    root_seed_source = "nearest_chart_anchor_continuation"
    root_anchor_index = int(tree.query(targets[root_index], k=1)[1])
    root_anchor_distance_mm = float(support_mm[root_index])
    root_waypoint_count = 0
    root_iterations = 0
    root_max_substep_residual_mm = math.inf
    if supported[root_index] and root_simplex >= 0:
        vertices = triangulation.simplices[root_simplex]
        transform = triangulation.transform[root_simplex]
        first = transform[:3] @ (targets[root_index] - transform[3])
        barycentric = np.append(first, 1.0 - np.sum(first))
        initial = barycentric @ chart_beta[vertices]
        (
            root_beta,
            root_residual_mm,
            root_iterations,
            root_success,
        ) = _robust_correct_target(
            environment,
            targets[root_index],
            initial,
            policy.teacher_policy,
        )
        root_seed_source = "local_tetrahedron"
        root_max_substep_residual_mm = float(root_residual_mm)
    else:
        (
            root_beta,
            root_residual_mm,
            root_iterations,
            root_waypoint_count,
            root_max_substep_residual_mm,
            _root_recovery_count,
            root_success,
        ) = _continue_segment(
            environment,
            chart_xyz[root_anchor_index],
            targets[root_index],
            chart_beta[root_anchor_index],
            policy.teacher_policy,
            step_mm=float(policy.continuation_step_mm),
        )
    root_margin_deg = float(
        _margin_deg(np.asarray(root_beta).reshape(1, 6), bounds)[0]
    )
    if (
        root_success
        and root_margin_deg < float(policy.joint_margin_min_deg)
    ):
        (
            root_beta,
            root_residual_mm,
            recovery_iterations,
            root_success,
        ) = recover_from_chart(targets[root_index], root_beta)
        root_iterations += int(recovery_iterations)
        root_seed_source = "chart_gold_recovery"
        root_margin_deg = float(
            _margin_deg(np.asarray(root_beta).reshape(1, 6), bounds)[0]
        )

    reference = np.full((len(targets), 6), np.nan, dtype=float)
    forward = np.full_like(reference, np.nan)
    reverse = np.full_like(reference, np.nan)
    forward_residual = np.full(len(targets), np.nan)
    reverse_residual = np.full(len(targets), np.nan)
    forward_iterations = np.zeros(len(targets), dtype=np.int64)
    reverse_iterations = np.zeros(len(targets), dtype=np.int64)
    forward_substeps = np.zeros(len(targets), dtype=np.int64)
    reverse_substeps = np.zeros(len(targets), dtype=np.int64)
    forward_max_substep_residual = np.full(len(targets), np.nan)
    reverse_max_substep_residual = np.full(len(targets), np.nan)
    forward_recovery_count = np.zeros(len(targets), dtype=np.int64)
    reverse_recovery_count = np.zeros(len(targets), dtype=np.int64)
    forward_success = np.zeros(len(targets), dtype=bool)
    reverse_success = np.zeros(len(targets), dtype=bool)
    forward_closure_gap = 1.0e300
    reverse_closure_gap = 1.0e300
    if root_success:
        forward_order = [
            (root_index + offset) % len(targets)
            for offset in range(1, len(targets))
        ]
        reverse_order = [
            (root_index - offset) % len(targets)
            for offset in range(1, len(targets))
        ]
        (
            forward,
            forward_residual,
            forward_iterations,
            forward_substeps,
            forward_max_substep_residual,
            forward_recovery_count,
            forward_success,
            forward_closure_gap,
        ) = _walk_loop(
            environment,
            targets,
            root_index,
            root_beta,
            forward_order,
            policy.teacher_policy,
            step_mm=float(policy.continuation_step_mm),
            recovery=recover_from_chart,
        )
        (
            reverse,
            reverse_residual,
            reverse_iterations,
            reverse_substeps,
            reverse_max_substep_residual,
            reverse_recovery_count,
            reverse_success,
            reverse_closure_gap,
        ) = _walk_loop(
            environment,
            targets,
            root_index,
            root_beta,
            reverse_order,
            policy.teacher_policy,
            step_mm=float(policy.continuation_step_mm),
            recovery=recover_from_chart,
        )
    raw_forward_complete = bool(
        np.all(forward_success) and np.isfinite(forward).all()
    )
    raw_reverse_complete = bool(
        np.all(reverse_success) and np.isfinite(reverse).all()
    )
    raw_direction_gap_p95 = None
    raw_direction_gap_max = None
    raw_forward_trajectory = (
        cyclic_beta_metrics(forward) if raw_forward_complete else None
    )
    raw_reverse_trajectory = (
        cyclic_beta_metrics(reverse) if raw_reverse_complete else None
    )
    raw_margin_min = None
    if raw_forward_complete and raw_reverse_complete:
        raw_gap = weighted_beta_gap_deg(
            forward,
            reverse,
            weights=policy.teacher_policy.beta_weights,
        )
        raw_direction_gap_p95 = float(np.percentile(raw_gap, 95))
        raw_direction_gap_max = float(np.max(raw_gap))
        raw_margin_min = float(
            np.min(
                np.concatenate(
                    [
                        _margin_deg(forward, bounds),
                        _margin_deg(reverse, bounds),
                    ]
                )
            )
        )
    raw_path_is_canonical = bool(
        raw_forward_trajectory is not None
        and raw_reverse_trajectory is not None
        and raw_margin_min is not None
        and raw_margin_min >= float(policy.joint_margin_min_deg)
        and raw_direction_gap_p95 is not None
        and raw_direction_gap_max is not None
        and raw_direction_gap_p95 <= float(policy.direction_gap_p95_deg)
        and raw_direction_gap_max <= float(policy.direction_gap_max_deg)
        and raw_forward_trajectory["phase_beta_rms_p95_deg"]
        <= float(policy.phase_beta_rms_p95_deg)
        and raw_forward_trajectory["phase_beta_rms_max_deg"]
        <= float(policy.phase_beta_rms_max_deg)
        and raw_reverse_trajectory["phase_beta_rms_p95_deg"]
        <= float(policy.phase_beta_rms_p95_deg)
        and raw_reverse_trajectory["phase_beta_rms_max_deg"]
        <= float(policy.phase_beta_rms_max_deg)
    )

    reference_method = "bidirectional_gold_continuation"
    canonical_reverse = reverse.copy()
    reference = forward.copy()
    candidate_layer_counts: list[int] = []
    candidate_attempt_count = 0
    candidate_sources: list[list[str]] = []
    gap_enrichment_report: dict[str, Any] | None = None
    gold_candidate_artifact: list[dict[str, Any]] = []
    primary_link_report: dict[str, Any] = {
        "success": raw_path_is_canonical
    }
    reverse_link_report: dict[str, Any] = {
        "success": raw_path_is_canonical
    }
    if not raw_path_is_canonical:
        reference_method = "global_gold_candidate_cyclic_dp"
        candidate_policy = CandidatePolicy(
            capability_nearest_count=int(
                policy.candidate_nearest_count
            ),
            capability_representative_count=int(
                policy.candidate_representative_count
            ),
            candidate_budget_per_node=int(
                policy.candidate_budget_per_phase
            ),
            difficult_candidate_budget_per_node=int(
                policy.candidate_budget_per_phase
            ),
            gold_candidate_target=1,
            beta_weights=policy.teacher_policy.beta_weights,
            damping=float(policy.teacher_policy.damping),
            max_rms_step_deg=float(policy.teacher_policy.max_step_deg),
            max_corrector_iterations=int(
                policy.teacher_policy.max_corrector_iterations
            ),
            tracking_tolerance_mm=float(
                policy.teacher_policy.tracking_tolerance_mm
            ),
            max_residual_mm=float(policy.residual_max_mm),
            gold_margin_deg=float(policy.joint_margin_min_deg),
            candidate_cluster_deg=float(policy.candidate_cluster_deg),
            solver_seed=int(policy.teacher_policy.solver_seed),
        )
        continuation_warm_starts: dict[int, np.ndarray] = {}
        for phase_idx in range(len(targets)):
            warm: list[np.ndarray] = []
            if forward_success[phase_idx] and np.isfinite(
                forward[phase_idx]
            ).all():
                warm.append(forward[phase_idx])
            if reverse_success[phase_idx] and np.isfinite(
                reverse[phase_idx]
            ).all():
                warm.append(reverse[phase_idx])
            if warm:
                continuation_warm_starts[phase_idx] = np.vstack(warm)
        bank = solve_candidate_bank(
            environment,
            targets,
            candidate_policy,
            capability_beta_rad=chart_beta,
            capability_xyz_m=chart_xyz,
            neighbor_beta_rad=continuation_warm_starts,
            difficult_node_ids=range(len(targets)),
        )
        layers: list[np.ndarray] = []
        layer_residuals: list[np.ndarray] = []
        for phase_idx in range(len(targets)):
            records = [
                record
                for record in bank.for_node(phase_idx)
                if record.quality is CandidateQuality.GOLD
            ]
            records.sort(
                key=lambda record: (
                    record.candidate_id,
                    record.solver,
                    record.source,
                )
            )
            candidate_layer_counts.append(len(records))
            candidate_sources.append(
                [
                    (
                        f"initial:{record.candidate_id}:"
                        f"{record.source}:{record.solver}"
                    )
                    for record in records
                ]
            )
            layers.append(
                np.vstack([record.beta_rad for record in records])
                if records
                else np.empty((0, 6), dtype=float)
            )
            layer_residuals.append(
                np.asarray(
                    [record.residual_mm for record in records],
                    dtype=float,
                )
            )
        candidate_attempt_count = int(len(bank.records))
        reference, primary_link_report = link_cyclic_candidates(
            layers,
            layer_residuals,
            lambda_velocity=float(
                policy.teacher_policy.lambda_velocity
            ),
            closure_weight=float(
                policy.teacher_policy.closure_weight
            ),
            lambda_posture=float(
                policy.teacher_policy.lambda_posture
            ),
            max_transition_deg=float(
                policy.phase_beta_rms_max_deg
            ),
        )
        if (
            not bool(primary_link_report.get("success"))
            and int(policy.gap_enrichment_rounds) > 0
        ):
            (
                layers,
                layer_residuals,
                candidate_sources,
                gap_enrichment_report,
            ) = _gap_directed_enrich_layers(
                environment,
                targets,
                layers,
                layer_residuals,
                candidate_sources,
                policy,
            )
            candidate_layer_counts = [
                int(len(layer)) for layer in layers
            ]
            candidate_attempt_count += int(
                gap_enrichment_report["total_added_candidate_count"]
            )
            reference, primary_link_report = link_cyclic_candidates(
                layers,
                layer_residuals,
                lambda_velocity=float(
                    policy.teacher_policy.lambda_velocity
                ),
                closure_weight=float(
                    policy.teacher_policy.closure_weight
                ),
                lambda_posture=float(
                    policy.teacher_policy.lambda_posture
                ),
                max_transition_deg=float(
                    policy.phase_beta_rms_max_deg
                ),
            )
        reversed_reference, reverse_link_report = link_cyclic_candidates(
            list(reversed(layers)),
            list(reversed(layer_residuals)),
            lambda_velocity=float(
                policy.teacher_policy.lambda_velocity
            ),
            closure_weight=float(
                policy.teacher_policy.closure_weight
            ),
            lambda_posture=float(
                policy.teacher_policy.lambda_posture
            ),
            max_transition_deg=float(
                policy.phase_beta_rms_max_deg
            ),
        )
        if bool(primary_link_report.get("success")):
            reference = np.asarray(reference, dtype=float).reshape(
                len(targets), 6
            )
        else:
            reference = np.full((len(targets), 6), np.nan, dtype=float)
        if bool(reverse_link_report.get("success")):
            canonical_reverse = np.asarray(
                reversed_reference[::-1], dtype=float
            ).reshape(len(targets), 6)
        else:
            canonical_reverse = np.full(
                (len(targets), 6), np.nan, dtype=float
            )
        if int(policy.gap_enrichment_rounds) > 0:
            for phase_idx, (
                layer,
                layer_residual,
                layer_source,
            ) in enumerate(
                zip(layers, layer_residuals, candidate_sources)
            ):
                for candidate_idx, (
                    beta,
                    residual_mm,
                    source,
                ) in enumerate(
                    zip(layer, layer_residual, layer_source)
                ):
                    candidate_margin_deg = float(
                        _margin_deg(
                            np.asarray(beta).reshape(1, 6), bounds
                        )[0]
                    )
                    gold_candidate_artifact.append(
                        {
                            "family_id": family.family_id,
                            "group_id": str(group_id),
                            "phase_idx": int(phase_idx),
                            "candidate_idx": int(candidate_idx),
                            "source": str(source),
                            "residual_mm": float(residual_mm),
                            "minimum_joint_margin_deg": (
                                candidate_margin_deg
                            ),
                            **{
                                name: float(beta[index])
                                for index, name in enumerate(
                                    BETA_COLUMNS
                                )
                            },
                        }
                    )

    reference_complete = bool(
        np.isfinite(reference).all()
        and np.isfinite(canonical_reverse).all()
        and bool(primary_link_report.get("success"))
        and bool(reverse_link_report.get("success"))
    )
    if reference_complete:
        achieved = np.asarray(
            environment.fk(reference), dtype=float
        ).reshape(len(targets), 3)
        residual = np.linalg.norm(achieved - targets, axis=1) * 1000.0
        margin = _margin_deg(reference, bounds)
        corrector_success = np.ones(len(targets), dtype=bool)
    else:
        residual = np.full(len(targets), np.nan, dtype=float)
        margin = np.full(len(targets), np.nan, dtype=float)
        corrector_success = np.zeros(len(targets), dtype=bool)
    iterations = np.zeros(len(targets), dtype=np.int64)

    hard_certificate: dict[str, Any] | None = None
    certificate_detail: dict[str, np.ndarray] = {}
    if reference_complete:
        hard_certificate, certificate_detail = (
            independently_certify_hard_cyclic_path(
                reference,
                targets,
                achieved,
                bounds,
                residual_p95_mm=float(policy.residual_p95_mm),
                residual_max_mm=float(policy.residual_max_mm),
                gold_margin_deg=float(policy.joint_margin_min_deg),
                max_transition_deg=float(
                    policy.phase_beta_rms_max_deg
                ),
            )
        )

    frame = pd.DataFrame(
        {
            "family_id": family.family_id,
            "group_id": str(group_id),
            "phase_idx": np.arange(len(targets), dtype=np.int64),
            **{
                name: targets[:, index]
                for index, name in enumerate(XYZ_COLUMNS)
            },
            "simplex_idx": simplex_ids,
            "chart_support_distance_mm": support_mm,
            "tetrahedron_max_edge_mm": tetrahedron_edge_mm,
            "chart_supported": supported,
            "reference_seed_source": reference_method,
            "reference_root_phase_idx": root_index,
            "reference_corrector_success": corrector_success,
            "reference_residual_mm": residual,
            "reference_corrector_iterations": iterations,
            "reference_joint_margin_deg": margin,
            **{
                name: reference[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
            **{
                f"forward_{name}": forward[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
            **{
                f"reverse_{name}": reverse[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
            **{
                f"canonical_reverse_{name}": canonical_reverse[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
            "forward_residual_mm": forward_residual,
            "reverse_residual_mm": reverse_residual,
            "forward_corrector_iterations": forward_iterations,
            "reverse_corrector_iterations": reverse_iterations,
            "forward_continuation_substeps": forward_substeps,
            "reverse_continuation_substeps": reverse_substeps,
            "forward_max_substep_residual_mm": (
                forward_max_substep_residual
            ),
            "reverse_max_substep_residual_mm": (
                reverse_max_substep_residual
            ),
            "forward_chart_recovery_count": forward_recovery_count,
            "reverse_chart_recovery_count": reverse_recovery_count,
            "forward_success": forward_success,
            "reverse_success": reverse_success,
            **certificate_detail,
        }
    )

    trajectory: dict[str, float] | None = None
    direction_gap_p95 = None
    direction_gap_max = None
    if reference_complete:
        trajectory = cyclic_beta_metrics(reference)
        canonical_order_gap = weighted_beta_gap_deg(
            reference,
            canonical_reverse,
            weights=policy.teacher_policy.beta_weights,
        )
        direction_gap_p95 = float(
            np.percentile(canonical_order_gap, 95)
        )
        direction_gap_max = float(np.max(canonical_order_gap))

    if reference_complete:
        reverse_achieved = np.asarray(
            environment.fk(canonical_reverse), dtype=float
        ).reshape(len(targets), 3)
        canonical_reverse_residual = (
            np.linalg.norm(reverse_achieved - targets, axis=1) * 1000.0
        )
    else:
        canonical_reverse_residual = np.full(
            len(targets), np.nan, dtype=float
        )
    combined_residual = np.concatenate(
        [residual, canonical_reverse_residual]
    )
    residual_p95 = (
        float(np.percentile(combined_residual, 95))
        if np.isfinite(combined_residual).all()
        else None
    )
    residual_max = (
        float(np.max(combined_residual))
        if np.isfinite(combined_residual).all()
        else None
    )
    combined_margin = (
        np.concatenate(
            [
                _margin_deg(reference, bounds),
                _margin_deg(canonical_reverse, bounds),
            ]
        )
        if np.isfinite(reference).all()
        and np.isfinite(canonical_reverse).all()
        else np.asarray([np.nan])
    )
    margin_min = (
        float(np.min(combined_margin))
        if np.isfinite(combined_margin).all()
        else None
    )
    forward_trajectory = (
        cyclic_beta_metrics(reference)
        if np.isfinite(reference).all()
        else None
    )
    reverse_trajectory = (
        cyclic_beta_metrics(canonical_reverse)
        if np.isfinite(canonical_reverse).all()
        else None
    )
    canonical_forward_closure_gap = (
        float(forward_trajectory["seam_beta_rms_deg"])
        if forward_trajectory is not None
        else 1.0e300
    )
    canonical_reverse_closure_gap = (
        float(reverse_trajectory["seam_beta_rms_deg"])
        if reverse_trajectory is not None
        else 1.0e300
    )
    checks = {
        "phase_count_complete": len(frame) == int(policy.phase_count),
        "root_chart_connection": bool(root_success),
        "canonical_candidate_layers_complete": bool(
            reference_complete
        ),
        "actual_bounds_and_margin": bool(
            margin_min is not None
            and margin_min >= float(policy.joint_margin_min_deg)
        ),
        "fk_residual": bool(
            residual_p95 is not None
            and residual_max is not None
            and residual_p95 <= float(policy.residual_p95_mm)
            and residual_max <= float(policy.residual_max_mm)
        ),
        "cyclic_label_continuity": bool(
            forward_trajectory is not None
            and reverse_trajectory is not None
            and forward_trajectory["phase_beta_rms_p95_deg"]
            <= float(policy.phase_beta_rms_p95_deg)
            and forward_trajectory["phase_beta_rms_max_deg"]
            <= float(policy.phase_beta_rms_max_deg)
            and forward_trajectory["acceleration_beta_rms_p95_deg"]
            <= float(policy.acceleration_beta_rms_p95_deg)
            and forward_trajectory["seam_beta_rms_deg"]
            <= float(policy.seam_beta_rms_deg)
            and reverse_trajectory["phase_beta_rms_p95_deg"]
            <= float(policy.phase_beta_rms_p95_deg)
            and reverse_trajectory["phase_beta_rms_max_deg"]
            <= float(policy.phase_beta_rms_max_deg)
            and reverse_trajectory["acceleration_beta_rms_p95_deg"]
            <= float(policy.acceleration_beta_rms_p95_deg)
            and reverse_trajectory["seam_beta_rms_deg"]
            <= float(policy.seam_beta_rms_deg)
        ),
        "canonical_order_audit_complete": bool(
            reference_complete
        ),
        "direction_and_reference_invariance": bool(
            direction_gap_p95 is not None
            and direction_gap_max is not None
            and direction_gap_p95 <= float(policy.direction_gap_p95_deg)
            and direction_gap_max <= float(policy.direction_gap_max_deg)
            and canonical_forward_closure_gap
            <= float(policy.seam_beta_rms_deg)
            and canonical_reverse_closure_gap
            <= float(policy.seam_beta_rms_deg)
        ),
        "independently_certified_hard_cyclic_path": bool(
            hard_certificate is not None
            and hard_certificate["gate_pass"]
        ),
    }
    report = {
        "family_id": family.family_id,
        "group_id": str(group_id),
        "family_fingerprint": family.fingerprint,
        "phase_count": int(len(frame)),
        "supported_phase_count": int(np.sum(supported)),
        "chart_support_is_diagnostic_only": True,
        "root_phase_idx": root_index,
        "root_seed_source": root_seed_source,
        "root_anchor_index": root_anchor_index,
        "root_anchor_distance_mm": root_anchor_distance_mm,
        "root_waypoint_count": int(root_waypoint_count),
        "root_corrector_iterations": int(root_iterations),
        "root_residual_mm": float(root_residual_mm),
        "root_joint_margin_deg": float(root_margin_deg),
        "root_max_substep_residual_mm": float(
            root_max_substep_residual_mm
        ),
        "root_connection_success": bool(root_success),
        "reference_method": reference_method,
        "candidate_attempt_count": int(candidate_attempt_count),
        "gap_enrichment_enabled": bool(
            int(policy.gap_enrichment_rounds) > 0
        ),
        "gap_enrichment": gap_enrichment_report,
        "continuation_warm_start_phase_count": int(
            sum(forward_success | reverse_success)
        ),
        "gold_candidate_layer_count": int(
            sum(count > 0 for count in candidate_layer_counts)
        ),
        "gold_candidate_layer_min_count": (
            int(min(candidate_layer_counts))
            if candidate_layer_counts
            else None
        ),
        "gold_candidate_layer_max_count": (
            int(max(candidate_layer_counts))
            if candidate_layer_counts
            else None
        ),
        "primary_link_report": primary_link_report,
        "reverse_link_report": reverse_link_report,
        "independent_hard_cyclic_certificate": hard_certificate,
        "raw_forward_complete": raw_forward_complete,
        "raw_reverse_complete": raw_reverse_complete,
        "raw_direction_gap_p95_deg": raw_direction_gap_p95,
        "raw_direction_gap_max_deg": raw_direction_gap_max,
        "raw_minimum_joint_margin_deg": raw_margin_min,
        "forward_chart_recovery_count": int(
            np.sum(forward_recovery_count)
        ),
        "reverse_chart_recovery_count": int(
            np.sum(reverse_recovery_count)
        ),
        "corrected_phase_count": int(np.sum(corrector_success)),
        "teacher_success_rate": float(np.mean(corrector_success)),
        "residual_p95_mm": residual_p95,
        "residual_max_mm": residual_max,
        "minimum_joint_margin_deg": margin_min,
        "trajectory": trajectory,
        "forward_trajectory": forward_trajectory,
        "reverse_trajectory": reverse_trajectory,
        "direction_gap_p95_deg": direction_gap_p95,
        "direction_gap_max_deg": direction_gap_max,
        "canonical_forward_closure_gap_deg": (
            float(canonical_forward_closure_gap)
            if canonical_forward_closure_gap < 1.0e299
            else None
        ),
        "canonical_reverse_closure_gap_deg": (
            float(canonical_reverse_closure_gap)
            if canonical_reverse_closure_gap < 1.0e299
            else None
        ),
        "forward_closure_gap_deg": (
            float(forward_closure_gap)
            if forward_closure_gap < 1.0e299
            else None
        ),
        "reverse_closure_gap_deg": (
            float(reverse_closure_gap)
            if reverse_closure_gap < 1.0e299
            else None
        ),
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
    }
    if gold_candidate_artifact:
        # Worker runners remove this private transport field before writing the
        # compact JSON report and persist it as a Parquet Gold-set artifact.
        report["_gold_candidate_artifact"] = gold_candidate_artifact
    return frame, report


def evaluate_student_loop(
    environment: GeometryHoldoutEnvironment,
    reference_frame: pd.DataFrame,
    prediction_rad: np.ndarray,
    *,
    teacher_reference_gate_pass: bool,
    admission: Mapping[str, float],
    trajectory_limits: Mapping[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one frozen Student on one complete geometry loop."""

    prediction = np.asarray(prediction_rad, dtype=float)
    if prediction.shape != (len(reference_frame), 6):
        raise ValueError("Student prediction must have shape (phase_count, 6)")
    if not np.isfinite(prediction).all():
        raise ValueError("Student prediction must be finite")
    targets = reference_frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    achieved = np.asarray(environment.fk(prediction), dtype=float).reshape(
        len(prediction), 3
    )
    fk_error_mm = np.linalg.norm(achieved - targets, axis=1) * 1000.0
    margins_deg = _margin_deg(prediction, environment.bounds)
    trajectory = cyclic_beta_metrics(prediction)
    truth = reference_frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    reference_available = bool(
        teacher_reference_gate_pass and np.isfinite(truth).all()
    )
    beta_abs_p95: list[float] | None = None
    beta_rms_deg = np.full(len(prediction), np.nan, dtype=float)
    if reference_available:
        delta = prediction - truth
        beta_abs_p95 = np.percentile(
            np.abs(np.rad2deg(delta)), 95, axis=0
        ).tolist()
        beta_rms_deg = np.rad2deg(
            np.sqrt(np.mean(np.square(delta), axis=1))
        )
    checks = {
        "teacher_reference_available": reference_available,
        "all_joint_abs_p95_below_limit": bool(
            beta_abs_p95 is not None
            and np.all(
                np.asarray(beta_abs_p95)
                < float(admission["joint_abs_p95_max_deg"])
            )
        ),
        "fk_p95_below_limit": bool(
            float(np.percentile(fk_error_mm, 95))
            < float(admission["fk_p95_max_mm"])
        ),
        "fk_max_below_limit": bool(
            float(np.max(fk_error_mm)) < float(admission["fk_max_mm"])
        ),
        "minimum_joint_margin_above_limit": bool(
            float(np.min(margins_deg))
            > float(admission["minimum_joint_margin_min_deg"])
        ),
        "trajectory_continuity": bool(
            trajectory["phase_beta_rms_p95_deg"]
            <= float(trajectory_limits["phase_beta_rms_p95_deg"])
            and trajectory["phase_beta_rms_max_deg"]
            <= float(trajectory_limits["phase_beta_rms_max_deg"])
            and trajectory["acceleration_beta_rms_p95_deg"]
            <= float(
                trajectory_limits["acceleration_beta_rms_p95_deg"]
            )
            and trajectory["seam_beta_rms_deg"]
            <= float(trajectory_limits["seam_beta_rms_deg"])
        ),
    }
    report = {
        "family_id": str(reference_frame.iloc[0]["family_id"]),
        "group_id": str(reference_frame.iloc[0]["group_id"]),
        "row_count": int(len(reference_frame)),
        "teacher_reference_gate_pass": bool(
            teacher_reference_gate_pass
        ),
        "reference_available": reference_available,
        "validation_beta_abs_p95_by_joint_deg": beta_abs_p95,
        "worst_joint_abs_p95_deg": (
            float(max(beta_abs_p95))
            if beta_abs_p95 is not None
            else None
        ),
        "validation_fk_p95_mm": float(
            np.percentile(fk_error_mm, 95)
        ),
        "validation_fk_max_mm": float(np.max(fk_error_mm)),
        "predicted_minimum_joint_margin_deg": float(
            np.min(margins_deg)
        ),
        **trajectory,
        "checks": checks,
        "student_family_gate_pass": bool(all(checks.values())),
    }
    diagnostics = pd.DataFrame(
        {
            "phase_idx": reference_frame["phase_idx"].to_numpy(
                dtype=np.int64
            ),
            **{
                f"predicted_{name}": prediction[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
            "achieved_x_m": achieved[:, 0],
            "achieved_y_m": achieved[:, 1],
            "achieved_z_m": achieved[:, 2],
            "fk_error_mm": fk_error_mm,
            "predicted_minimum_joint_margin_deg": margins_deg,
            "beta_error_rms_deg": beta_rms_deg,
        }
    )
    return report, diagnostics
