"""Canonical teacher over a solid tube around a three-dimensional ellipse.

The V10 teacher solves one cyclic trajectory.  This module deliberately puts
the next abstraction boundary one level higher: a region is a connected graph
of cyclic trajectories.  Every curve is still corrected by the exact V10
teacher, while Delaunay-neighbour initialisers couple the labels across the
normal cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.stats import qmc

from .canonical import CanonicalTeacher, TeacherPolicy, TrajectorySpec


class RegionEnvironment(Protocol):
    bounds: np.ndarray

    def fk(self, beta: np.ndarray) -> np.ndarray: ...

    def theta(self, beta: np.ndarray) -> np.ndarray: ...


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unit_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


@dataclass(frozen=True)
class EllipseFamilySpec:
    """Physical pose and dimensions of one ellipse family."""

    family_id: str
    center_m: np.ndarray
    major_direction: np.ndarray
    minor_direction: np.ndarray
    major_semiaxis_m: float
    minor_semiaxis_m: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family_id = str(self.family_id).strip()
        center = np.asarray(self.center_m, dtype=float).reshape(3)
        major = _unit_vector(self.major_direction, name="major_direction")
        minor = _unit_vector(self.minor_direction, name="minor_direction")
        if not family_id:
            raise ValueError("family_id must not be empty")
        if not np.isfinite(center).all():
            raise ValueError("center_m must be finite")
        if not math.isclose(float(np.dot(major, minor)), 0.0, abs_tol=1.0e-10):
            raise ValueError("major and minor directions must be orthogonal")
        major_radius = float(self.major_semiaxis_m)
        minor_radius = float(self.minor_semiaxis_m)
        if not (math.isfinite(major_radius) and math.isfinite(minor_radius)):
            raise ValueError("ellipse semiaxes must be finite")
        if minor_radius <= 0.0 or major_radius < minor_radius:
            raise ValueError("ellipse semiaxes must satisfy major >= minor > 0")
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "center_m", center.copy())
        object.__setattr__(self, "major_direction", major)
        object.__setattr__(self, "minor_direction", minor)
        object.__setattr__(self, "major_semiaxis_m", major_radius)
        object.__setattr__(self, "minor_semiaxis_m", minor_radius)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def plane_normal(self) -> np.ndarray:
        return _unit_vector(
            np.cross(self.major_direction, self.minor_direction),
            name="plane_normal",
        )

    @property
    def axis_ratio(self) -> float:
        return float(self.minor_semiaxis_m / self.major_semiaxis_m)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "family_id": self.family_id,
                "center_m": self.center_m.tolist(),
                "major_direction": self.major_direction.tolist(),
                "minor_direction": self.minor_direction.tolist(),
                "major_semiaxis_m": self.major_semiaxis_m,
                "minor_semiaxis_m": self.minor_semiaxis_m,
                "metadata": dict(self.metadata),
            }
        )

    def centerline(
        self, *, phase_count: int, phase_offset_rad: float = 0.0
    ) -> np.ndarray:
        count = int(phase_count)
        if count < 4:
            raise ValueError("phase_count must be at least four")
        phase = (
            np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
            + float(phase_offset_rad)
        )
        return (
            self.center_m[None, :]
            + self.major_semiaxis_m
            * np.cos(phase)[:, None]
            * self.major_direction[None, :]
            + self.minor_semiaxis_m
            * np.sin(phase)[:, None]
            * self.minor_direction[None, :]
        )

    def in_plane_normals(
        self, *, phase_count: int, phase_offset_rad: float = 0.0
    ) -> np.ndarray:
        count = int(phase_count)
        if count < 4:
            raise ValueError("phase_count must be at least four")
        phase = (
            np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
            + float(phase_offset_rad)
        )
        # Gradient of x^2/a^2 + y^2/b^2 is the geometric outward normal.
        normal = (
            np.cos(phase)[:, None]
            / self.major_semiaxis_m
            * self.major_direction[None, :]
            + np.sin(phase)[:, None]
            / self.minor_semiaxis_m
            * self.minor_direction[None, :]
        )
        return normal / np.linalg.norm(normal, axis=1, keepdims=True)

    def tube_targets(
        self,
        *,
        phase_count: int,
        normalized_offsets: np.ndarray,
        radial_radius_mm: float,
        plane_radius_mm: float,
        phase_offset_rad: float = 0.0,
    ) -> np.ndarray:
        offsets = np.asarray(normalized_offsets, dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 2 or len(offsets) == 0:
            raise ValueError("normalized_offsets must have shape (N, 2)")
        if not np.isfinite(offsets).all() or np.any(np.linalg.norm(offsets, axis=1) > 1.0 + 1e-12):
            raise ValueError("normalized_offsets must lie in the closed unit disk")
        radial = float(radial_radius_mm)
        plane = float(plane_radius_mm)
        if not (math.isfinite(radial) and radial >= 0.0):
            raise ValueError("radial_radius_mm must be finite and non-negative")
        if not (math.isfinite(plane) and plane >= 0.0):
            raise ValueError("plane_radius_mm must be finite and non-negative")
        centerline = self.centerline(
            phase_count=int(phase_count), phase_offset_rad=float(phase_offset_rad)
        )
        radial_normal = self.in_plane_normals(
            phase_count=int(phase_count), phase_offset_rad=float(phase_offset_rad)
        )
        return (
            centerline[None, :, :]
            + offsets[:, 0, None, None] * (radial / 1000.0) * radial_normal[None, :, :]
            + offsets[:, 1, None, None]
            * (plane / 1000.0)
            * self.plane_normal[None, None, :]
        )


def _delaunay_edges(points: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(values) < 3:
        return tuple((index, index + 1) for index in range(len(values) - 1))
    triangulation = Delaunay(values)
    edges: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for left, right in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(simplex[left]), int(simplex[right]))))
            edges.add(edge)
    return tuple(sorted(edges))


@dataclass(frozen=True)
class TubeCrossSection:
    """A deterministic, connected design in the normalized unit disk."""

    points: np.ndarray
    seed: int
    parent_fingerprint: str | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("cross-section points must have shape (N, 2), N >= 3")
        if not np.isfinite(points).all() or np.any(np.linalg.norm(points, axis=1) > 1.0 + 1e-12):
            raise ValueError("cross-section points must lie in the closed unit disk")
        if len(np.unique(np.round(points, 12), axis=0)) != len(points):
            raise ValueError("cross-section points must be unique")
        object.__setattr__(self, "points", points.copy())
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def master(cls, *, seed: int = 20260722) -> "TubeCrossSection":
        boundary_phase = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        boundary = np.column_stack([np.cos(boundary_phase), np.sin(boundary_phase)])
        unit = qmc.Sobol(d=2, scramble=True, seed=int(seed)).random_base2(6)
        sobol_disk = np.column_stack(
            [
                np.sqrt(unit[:, 0]) * np.cos(2.0 * math.pi * unit[:, 1]),
                np.sqrt(unit[:, 0]) * np.sin(2.0 * math.pi * unit[:, 1]),
            ]
        )
        # Prefixes intentionally expose centre/boundary support early while
        # retaining all 16 boundary and all 64 Sobol points in the master.
        points = np.vstack(
            [
                np.zeros((1, 2)),
                boundary[::2],
                boundary[1::2],
                sobol_disk[:8],
                sobol_disk[8:],
            ]
        )
        return cls(points=points, seed=int(seed))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "seed": self.seed,
                "points": np.round(self.points, 15).tolist(),
            }
        )

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return _delaunay_edges(self.points)

    def prefix(self, count: int) -> "TubeCrossSection":
        requested = int(count)
        if requested not in {9, 25, 49, 81}:
            raise ValueError("cross-section prefix must be one of 9, 25, 49, 81")
        if requested > len(self.points):
            raise ValueError("requested prefix exceeds this cross-section")
        return TubeCrossSection(
            points=self.points[:requested],
            seed=self.seed,
            parent_fingerprint=self.fingerprint,
        )


@dataclass(frozen=True)
class TeacherSurface:
    family: EllipseFamilySpec
    cross_section: TubeCrossSection
    radial_radius_mm: float
    plane_radius_mm: float
    beta_rad: np.ndarray
    theta_rad: np.ndarray
    achieved_xyz_m: np.ndarray
    target_xyz_m: np.ndarray
    success_mask: np.ndarray
    chart_id: np.ndarray
    metrics: Mapping[str, float]
    provenance: Mapping[str, Any]

    @property
    def complete(self) -> bool:
        expected = (len(self.cross_section.points), self.target_xyz_m.shape[1])
        return bool(self.success_mask.shape == expected and np.all(self.success_mask))

    def to_frame(self) -> pd.DataFrame:
        node_count, phase_count = self.beta_rad.shape[:2]
        node = np.repeat(np.arange(node_count), phase_count)
        phase = np.tile(np.arange(phase_count), node_count)
        offsets = self.cross_section.points[node]
        target = self.target_xyz_m.reshape(-1, 3)
        achieved = self.achieved_xyz_m.reshape(-1, 3)
        beta = self.beta_rad.reshape(-1, 6)
        frame = pd.DataFrame(
            {
                "family_id": self.family.family_id,
                "trajectory_id": [f"{self.family.family_id}:node{value:03d}" for value in node],
                "sample_id": [
                    f"{self.family.family_id}:node{n:03d}:phase{p:04d}"
                    for n, p in zip(node, phase)
                ],
                "cross_section_idx": node,
                "phase_idx": phase,
                "normalized_radial_offset": offsets[:, 0],
                "normalized_plane_offset": offsets[:, 1],
                "tube_n1_mm": offsets[:, 0] * float(self.radial_radius_mm),
                "tube_n2_mm": offsets[:, 1] * float(self.plane_radius_mm),
                "target_x_m": target[:, 0],
                "target_y_m": target[:, 1],
                "target_z_m": target[:, 2],
                "achieved_x_m": achieved[:, 0],
                "achieved_y_m": achieved[:, 1],
                "achieved_z_m": achieved[:, 2],
                "teacher_success": self.success_mask.reshape(-1),
                "chart_id": self.chart_id.reshape(-1),
            }
        )
        for index in range(6):
            frame[f"teacher_beta{index + 1}_rad"] = beta[:, index]
        return frame


def _edge_beta_rms_deg(beta: np.ndarray, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros(1, dtype=float)
    values = []
    for left, right in edges:
        gap = beta[left] - beta[right]
        values.extend(np.rad2deg(np.sqrt(np.mean(np.square(gap), axis=1))).tolist())
    return np.asarray(values, dtype=float)


class CanonicalRegionTeacher:
    """Solve a phase×cross-section surface through coupled curve sweeps."""

    def __init__(self, environment: RegionEnvironment):
        self._environment = environment
        self._trajectory_teacher = CanonicalTeacher(environment)

    def solve_surface(
        self,
        *,
        family: EllipseFamilySpec,
        cross_section: TubeCrossSection,
        phase_count: int,
        radial_radius_mm: float,
        plane_radius_mm: float,
        policy: TeacherPolicy,
        root_beta: np.ndarray,
        centerline_seed: np.ndarray | None = None,
        sweep_directions: Sequence[str] = ("outward", "inward"),
        traversal_direction: str = "forward",
        cyclic_cut: int = 0,
    ) -> TeacherSurface:
        count = int(phase_count)
        target = family.tube_targets(
            phase_count=count,
            normalized_offsets=cross_section.points,
            radial_radius_mm=float(radial_radius_mm),
            plane_radius_mm=float(plane_radius_mm),
        )
        directions = tuple(str(value).lower() for value in sweep_directions)
        if not directions or any(value not in {"outward", "inward"} for value in directions):
            raise ValueError("sweep_directions must contain outward and/or inward")
        node_count = len(cross_section.points)
        beta = np.empty((node_count, count, 6), dtype=float)
        theta = np.empty((node_count, count, 30), dtype=float)
        achieved = np.empty((node_count, count, 3), dtype=float)
        success = np.zeros((node_count, count), dtype=bool)
        chart = np.zeros((node_count, count), dtype=np.int64)
        solved = np.zeros(node_count, dtype=bool)
        radius_order = np.argsort(np.linalg.norm(cross_section.points, axis=1), kind="stable")
        center_idx = int(radius_order[0])
        neighbours: dict[int, list[int]] = {index: [] for index in range(node_count)}
        for left, right in cross_section.edges:
            neighbours[left].append(right)
            neighbours[right].append(left)

        def initializer(node_index: int) -> np.ndarray | None:
            available = [index for index in neighbours[node_index] if solved[index]]
            if available:
                return np.mean(beta[available], axis=0)
            if node_index == center_idx and centerline_seed is not None:
                seed = np.asarray(centerline_seed, dtype=float)
                if seed.shape != (count, 6):
                    raise ValueError(f"centerline_seed must have shape ({count}, 6)")
                return seed
            if np.any(solved):
                solved_indices = np.flatnonzero(solved)
                closest = solved_indices[
                    np.argmin(
                        np.linalg.norm(
                            cross_section.points[solved_indices]
                            - cross_section.points[node_index],
                            axis=1,
                        )
                    )
                ]
                return beta[int(closest)]
            return None

        def solve_node(node_index: int) -> None:
            initial = initializer(node_index)
            trajectory = self._trajectory_teacher.solve(
                TrajectorySpec(
                    trajectory_id=f"{family.family_id}:node{node_index:03d}",
                    family_id=family.family_id,
                    radius_mm=float(family.major_semiaxis_m * 1000.0),
                    target_xyz_m=target[node_index],
                    tube_offsets_mm=(
                        float(cross_section.points[node_index, 0] * radial_radius_mm),
                        float(cross_section.points[node_index, 1] * plane_radius_mm),
                    ),
                    traversal_direction=str(traversal_direction),
                    cyclic_cut=int(cyclic_cut),
                ),
                policy,
                root_beta=np.asarray(root_beta, dtype=float),
                initial_beta_path=initial,
            )
            beta[node_index] = trajectory.beta_rad
            theta[node_index] = trajectory.theta_rad
            achieved[node_index] = trajectory.achieved_xyz_m
            residual_mm = np.linalg.norm(
                trajectory.achieved_xyz_m - target[node_index], axis=1
            ) * 1000.0
            bounds = np.asarray(self._environment.bounds, dtype=float).reshape(6, 2)
            within = np.all(
                (trajectory.beta_rad >= bounds[:, 0][None, :] - 1e-12)
                & (trajectory.beta_rad <= bounds[:, 1][None, :] + 1e-12),
                axis=1,
            )
            success[node_index] = within & (residual_mm <= float(policy.tracking_tolerance_mm))
            chart[node_index] = trajectory.chart_id
            solved[node_index] = True

        # Bootstrap every node outward so no curve is ever solved without a
        # previously solved geometric neighbour or the centreline seed.
        for node_index in radius_order:
            solve_node(int(node_index))
        # Block-coordinate sweeps revisit curves with Delaunay-neighbour means.
        for direction in directions:
            order = radius_order if direction == "outward" else radius_order[::-1]
            for node_index in order:
                solve_node(int(node_index))

        residual_mm = np.linalg.norm(achieved - target, axis=2) * 1000.0
        velocity = np.roll(beta, -1, axis=1) - beta
        velocity_deg = np.rad2deg(np.sqrt(np.mean(np.square(velocity), axis=2)))
        acceleration = (
            np.roll(beta, -1, axis=1)
            - 2.0 * beta
            + np.roll(beta, 1, axis=1)
        )
        acceleration_deg = np.rad2deg(
            np.sqrt(np.mean(np.square(acceleration), axis=2))
        )
        edge_deg = _edge_beta_rms_deg(beta, cross_section.edges)
        bounds = np.asarray(self._environment.bounds, dtype=float).reshape(6, 2)
        margin_deg = np.rad2deg(
            np.minimum(
                beta - bounds[None, None, :, 0],
                bounds[None, None, :, 1] - beta,
            )
        )
        metrics = {
            "rows": float(node_count * count),
            "success_rate": float(np.mean(success)),
            "residual_p95_mm": float(np.percentile(residual_mm, 95)),
            "residual_max_mm": float(np.max(residual_mm)),
            "joint_margin_min_deg": float(np.min(margin_deg)),
            "phase_beta_rms_p95_deg": float(np.percentile(velocity_deg, 95)),
            "acceleration_beta_rms_p95_deg": float(
                np.percentile(acceleration_deg, 95)
            ),
            "seam_beta_rms_max_deg": float(np.max(velocity_deg[:, -1])),
            "surface_edge_beta_rms_p95_deg": float(np.percentile(edge_deg, 95)),
            "surface_edge_beta_rms_max_deg": float(np.max(edge_deg)),
        }
        return TeacherSurface(
            family=family,
            cross_section=cross_section,
            radial_radius_mm=float(radial_radius_mm),
            plane_radius_mm=float(plane_radius_mm),
            beta_rad=beta,
            theta_rad=theta,
            achieved_xyz_m=achieved,
            target_xyz_m=target,
            success_mask=success,
            chart_id=chart,
            metrics=metrics,
            provenance={
                "protocol_id": "generalized-ellipse-region-v11",
                "surface_strategy": "delaunay_neighbor_anchor_block_coordinate_v1",
                "family_fingerprint": family.fingerprint,
                "cross_section_fingerprint": cross_section.fingerprint,
                "teacher_policy_id": policy.fingerprint,
                "sweep_directions": list(directions),
                "traversal_direction": str(traversal_direction),
                "cyclic_cut": int(cyclic_cut) % count,
            },
        )
