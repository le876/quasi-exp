"""Geometry protocol for large-span ellipse challenges.

The historical true-ellipse parameter ``R`` is not a geometric semiaxis.
This module keeps that lineage conversion explicit while making the public
challenge interface use the actual three-dimensional major semiaxis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# Singular values of the registered historical family coefficient matrix,
# divided by its scalar R parameter.  These values are frozen evidence from
# the V7 geometry audit, not values inferred from a new fitted trajectory.
REGISTERED_MAJOR_AXIS_GAIN = 1.953403629
REGISTERED_MINOR_AXIS_GAIN = 0.658949362
REGISTERED_MINOR_TO_MAJOR_RATIO = (
    REGISTERED_MINOR_AXIS_GAIN / REGISTERED_MAJOR_AXIS_GAIN
)


@dataclass(frozen=True)
class EllipseChallenge:
    """One challenge expressed in physical geometric semiaxes."""

    major_semiaxis_m: float
    minor_semiaxis_m: float
    radius_parameter_m: float

    def __post_init__(self) -> None:
        values = (
            float(self.major_semiaxis_m),
            float(self.minor_semiaxis_m),
            float(self.radius_parameter_m),
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("ellipse challenge dimensions must be finite and positive")
        if self.minor_semiaxis_m > self.major_semiaxis_m:
            raise ValueError("minor semiaxis cannot exceed major semiaxis")

    @classmethod
    def from_major_semiaxis_m(
        cls,
        major_semiaxis_m: float,
        *,
        minor_to_major_ratio: float = REGISTERED_MINOR_TO_MAJOR_RATIO,
    ) -> "EllipseChallenge":
        major = float(major_semiaxis_m)
        ratio = float(minor_to_major_ratio)
        if not math.isfinite(major) or major <= 0.0:
            raise ValueError("major_semiaxis_m must be finite and positive")
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError("minor_to_major_ratio must be in (0, 1]")
        return cls(
            major_semiaxis_m=major,
            minor_semiaxis_m=major * ratio,
            radius_parameter_m=major / REGISTERED_MAJOR_AXIS_GAIN,
        )

    @property
    def major_diameter_m(self) -> float:
        return 2.0 * float(self.major_semiaxis_m)

    @property
    def minor_diameter_m(self) -> float:
        return 2.0 * float(self.minor_semiaxis_m)

    def generate_targets(
        self,
        *,
        center_m: np.ndarray,
        major_direction: np.ndarray,
        minor_direction: np.ndarray,
        phase_count: int,
    ) -> np.ndarray:
        """Generate a planar 3-D ellipse with independently verified axes."""

        center = np.asarray(center_m, dtype=float)
        major = np.asarray(major_direction, dtype=float)
        minor = np.asarray(minor_direction, dtype=float)
        if center.shape != (3,) or major.shape != (3,) or minor.shape != (3,):
            raise ValueError("center and axis directions must each have shape (3,)")
        if not np.isfinite(center).all() or not np.isfinite(major).all() or not np.isfinite(minor).all():
            raise ValueError("center and axis directions must be finite")
        major_norm = float(np.linalg.norm(major))
        minor_norm = float(np.linalg.norm(minor))
        if major_norm <= 0.0 or minor_norm <= 0.0:
            raise ValueError("axis directions must be non-zero")
        major_unit = major / major_norm
        minor_unit = minor / minor_norm
        if not math.isclose(float(np.dot(major_unit, minor_unit)), 0.0, abs_tol=1.0e-10):
            raise ValueError("major and minor directions must be orthogonal")
        count = int(phase_count)
        if count < 4:
            raise ValueError("phase_count must be at least four")
        phase = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        return (
            center[None, :]
            + self.major_semiaxis_m * np.cos(phase)[:, None] * major_unit[None, :]
            + self.minor_semiaxis_m * np.sin(phase)[:, None] * minor_unit[None, :]
        )


@dataclass(frozen=True)
class AtlasMatch:
    """Nearest reachable witnesses for one requested target trajectory."""

    initial_beta_path_rad: np.ndarray
    nearest_distance_mm: np.ndarray
    metrics: dict[str, float]


@dataclass(frozen=True)
class ReachabilityAtlas:
    """Finite FK witness atlas used only for screening and initialisation."""

    xyz_m: np.ndarray
    beta_rad: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=float)
        beta = np.asarray(self.beta_rad, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
            raise ValueError("xyz_m must have shape (N, 3) with N > 0")
        if beta.shape != (len(xyz), 6):
            raise ValueError(f"beta_rad must have shape ({len(xyz)}, 6)")
        if not np.isfinite(xyz).all() or not np.isfinite(beta).all():
            raise ValueError("reachability atlas values must be finite")
        object.__setattr__(self, "xyz_m", xyz.copy())
        object.__setattr__(self, "beta_rad", beta.copy())

    def match_targets(
        self,
        target_xyz_m: np.ndarray,
        *,
        beta_bounds_rad: np.ndarray | None = None,
    ) -> AtlasMatch:
        target = np.asarray(target_xyz_m, dtype=float)
        if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
            raise ValueError("target_xyz_m must have shape (N, 3) with N > 0")
        if not np.isfinite(target).all():
            raise ValueError("target_xyz_m must be finite")
        distance_m, indices = cKDTree(self.xyz_m).query(target, k=1)
        distance_mm = np.asarray(distance_m, dtype=float) * 1000.0
        metrics = {
            "nearest_distance_p95_mm": float(np.percentile(distance_mm, 95)),
            "nearest_distance_max_mm": float(np.max(distance_mm)),
        }
        if beta_bounds_rad is not None:
            bounds = np.asarray(beta_bounds_rad, dtype=float)
            if bounds.shape != (6, 2) or not np.isfinite(bounds).all():
                raise ValueError("beta_bounds_rad must have finite shape (6, 2)")
            selected_beta = self.beta_rad[np.asarray(indices, dtype=np.int64)]
            margin_deg = np.rad2deg(
                np.min(
                    np.minimum(
                        selected_beta - bounds[:, 0][None, :],
                        bounds[:, 1][None, :] - selected_beta,
                    ),
                    axis=1,
                )
            )
            metrics.update(
                {
                    "nearest_joint_margin_min_deg": float(np.min(margin_deg)),
                    "nearest_joint_margin_p05_deg": float(np.percentile(margin_deg, 5)),
                }
            )
        return AtlasMatch(
            initial_beta_path_rad=self.beta_rad[np.asarray(indices, dtype=np.int64)].copy(),
            nearest_distance_mm=distance_mm,
            metrics=metrics,
        )


@dataclass(frozen=True)
class EllipsePoseFit:
    """Best finite-atlas pose witness found for one ellipse challenge."""

    center_m: np.ndarray
    major_direction: np.ndarray
    minor_direction: np.ndarray
    target_xyz_m: np.ndarray
    match: AtlasMatch
    objective_mm: float
    optimizer_success: bool
    optimizer_message: str
    evaluations: int


def fit_ellipse_pose_to_atlas(
    challenge: EllipseChallenge,
    atlas: ReachabilityAtlas,
    *,
    phase_count: int = 24,
    seed: int = 20260720,
    max_iterations: int = 120,
    population_size: int = 10,
    beta_bounds_rad: np.ndarray | None = None,
    safe_joint_margin_deg: float = 1.5,
    joint_margin_penalty_mm_per_deg: float = 20.0,
) -> EllipsePoseFit:
    """Fit ellipse center/orientation to a finite FK reachability atlas.

    This is deliberately a witness search, not a reachability proof.  Its
    nearest configurations initialise the exact teacher, whose FK and gate
    results remain authoritative.
    """

    count = int(phase_count)
    if count < 4:
        raise ValueError("phase_count must be at least four")
    if int(max_iterations) < 1 or int(population_size) < 4:
        raise ValueError("optimizer iteration and population sizes are too small")
    xyz_min = np.min(atlas.xyz_m, axis=0)
    xyz_max = np.max(atlas.xyz_m, axis=0)
    tree = cKDTree(atlas.xyz_m)
    bounds_rad = None
    if beta_bounds_rad is not None:
        bounds_rad = np.asarray(beta_bounds_rad, dtype=float)
        if bounds_rad.shape != (6, 2) or not np.isfinite(bounds_rad).all():
            raise ValueError("beta_bounds_rad must have finite shape (6, 2)")
        if np.any(bounds_rad[:, 0] > bounds_rad[:, 1]):
            raise ValueError("beta_bounds_rad must contain ordered bounds")
    safe_margin = float(safe_joint_margin_deg)
    margin_weight = float(joint_margin_penalty_mm_per_deg)
    if not math.isfinite(safe_margin) or safe_margin < 0.0:
        raise ValueError("safe_joint_margin_deg must be finite and non-negative")
    if not math.isfinite(margin_weight) or margin_weight < 0.0:
        raise ValueError("joint_margin_penalty_mm_per_deg must be finite and non-negative")

    def pose(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.asarray(values[:3], dtype=float)
        rotation = Rotation.from_euler("xyz", np.asarray(values[3:6], dtype=float)).as_matrix()
        return center, rotation[:, 0], rotation[:, 1]

    def objective(values: np.ndarray) -> float:
        center, major, minor = pose(values)
        target = challenge.generate_targets(
            center_m=center,
            major_direction=major,
            minor_direction=minor,
            phase_count=count,
        )
        distance_m, indices = tree.query(target, k=1)
        distance_mm = np.asarray(distance_m, dtype=float) * 1000.0
        score = float(np.percentile(distance_mm, 95) + 0.25 * np.max(distance_mm))
        if bounds_rad is not None and margin_weight > 0.0:
            selected_beta = atlas.beta_rad[np.asarray(indices, dtype=np.int64)]
            margin_deg = np.rad2deg(
                np.min(
                    np.minimum(
                        selected_beta - bounds_rad[:, 0][None, :],
                        bounds_rad[:, 1][None, :] - selected_beta,
                    ),
                    axis=1,
                )
            )
            deficit = np.maximum(safe_margin - margin_deg, 0.0)
            score += margin_weight * float(
                np.percentile(deficit, 95) + 0.25 * np.max(deficit)
            )
        return score

    bounds = [
        (float(xyz_min[index]), float(xyz_max[index])) for index in range(3)
    ] + [(-math.pi, math.pi)] * 3
    result = differential_evolution(
        objective,
        bounds,
        seed=int(seed),
        maxiter=int(max_iterations),
        popsize=int(population_size),
        tol=1.0e-7,
        polish=True,
        workers=1,
        updating="immediate",
    )
    center, major, minor = pose(np.asarray(result.x, dtype=float))
    target = challenge.generate_targets(
        center_m=center,
        major_direction=major,
        minor_direction=minor,
        phase_count=count,
    )
    return EllipsePoseFit(
        center_m=center,
        major_direction=major,
        minor_direction=minor,
        target_xyz_m=target,
        match=atlas.match_targets(target, beta_bounds_rad=bounds_rad),
        objective_mm=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        evaluations=int(result.nfev),
    )


def assess_chain_length_necessity(
    challenges: Iterable[EllipseChallenge],
    *,
    robot_max_reach_m: float,
    tube_radius_m: float = 0.0,
) -> list[dict[str, float | bool]]:
    """Apply the translation-independent endpoint-diameter necessary gate.

    Any two points reachable from one fixed base can be at most ``2 L``
    apart.  This gate is necessary, not sufficient: passing it does not imply
    that one center/orientation or one joint-bounded trajectory exists.
    """

    reach = float(robot_max_reach_m)
    tube = float(tube_radius_m)
    if not math.isfinite(reach) or reach <= 0.0:
        raise ValueError("robot_max_reach_m must be finite and positive")
    if not math.isfinite(tube) or tube < 0.0:
        raise ValueError("tube_radius_m must be finite and non-negative")
    available_diameter = 2.0 * reach
    rows: list[dict[str, float | bool]] = []
    for challenge in challenges:
        required = challenge.major_diameter_m
        tube_required = required + 2.0 * tube
        rows.append(
            {
                "major_semiaxis_m": float(challenge.major_semiaxis_m),
                "minor_semiaxis_m": float(challenge.minor_semiaxis_m),
                "radius_parameter_m": float(challenge.radius_parameter_m),
                "major_diameter_m": float(required),
                "robot_pair_diameter_upper_bound_m": float(available_diameter),
                "diameter_margin_m": float(available_diameter - required),
                "tube_diameter_margin_m": float(available_diameter - tube_required),
                "translation_independent_diameter_gate_pass": bool(
                    required <= available_diameter
                ),
                "translation_independent_tube_diameter_gate_pass": bool(
                    tube_required <= available_diameter
                ),
            }
        )
    return rows
