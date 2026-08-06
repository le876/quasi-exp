"""Empirical workspace reach proxy for BACRA V14.

This module owns the distinction between finite forward-sampling evidence and
certified reachability.  It deliberately does not run FK or IK itself: callers
provide independent scrambled-replica samples and optional frontier evidence,
which keeps the numerical generators replaceable and the classification seam
fully testable in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


class ReachStatus(str, Enum):
    """Evidence status for one task-space cell."""

    EMPIRICAL_SUPPORTED = "empirical_supported"
    EMPIRICAL_UNSUPPORTED = "empirical_unsupported"
    UNREACHABLE_CERTIFIED = "unreachable_certified"


@dataclass(frozen=True, order=True)
class CellKey:
    """Stable origin-anchored workspace cell identity."""

    level_mm: int
    ix: int
    iy: int
    iz: int

    def __post_init__(self) -> None:
        if int(self.level_mm) <= 0:
            raise ValueError("cell level_mm must be positive")
        object.__setattr__(self, "level_mm", int(self.level_mm))
        object.__setattr__(self, "ix", int(self.ix))
        object.__setattr__(self, "iy", int(self.iy))
        object.__setattr__(self, "iz", int(self.iz))

    @property
    def volume_m3(self) -> float:
        return float((self.level_mm / 1000.0) ** 3)

    def face_neighbors(self) -> tuple[CellKey, ...]:
        offsets = (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        )
        return tuple(
            CellKey(
                self.level_mm,
                self.ix + dx,
                self.iy + dy,
                self.iz + dz,
            )
            for dx, dy, dz in offsets
        )


@dataclass(frozen=True)
class WorkspaceGridSpec:
    """Nested workspace grid with a globally stable metric origin."""

    levels_mm: tuple[int, ...] = (20, 10, 5)
    x_slab_m: tuple[float, float] = (1.015498, 1.215498)
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        levels = tuple(int(value) for value in self.levels_mm)
        if not levels or any(value <= 0 for value in levels):
            raise ValueError("levels_mm must contain positive levels")
        if tuple(sorted(levels, reverse=True)) != levels:
            raise ValueError("levels_mm must be ordered coarse to fine")
        if any(left % right for left, right in zip(levels, levels[1:], strict=False)):
            raise ValueError("workspace grid levels must be nested integer divisors")
        x_low, x_high = (float(value) for value in self.x_slab_m)
        if not (math.isfinite(x_low) and math.isfinite(x_high) and x_low < x_high):
            raise ValueError("x_slab_m must be finite and ordered")
        origin = tuple(float(value) for value in self.origin_m)
        if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
            raise ValueError("origin_m must contain three finite coordinates")
        object.__setattr__(self, "levels_mm", levels)
        object.__setattr__(self, "x_slab_m", (x_low, x_high))
        object.__setattr__(self, "origin_m", origin)

    @property
    def convergence_level_mm(self) -> int:
        return int(self.levels_mm[0])

    def cells_for_points(self, xyz_m: np.ndarray, *, level_mm: int) -> frozenset[CellKey]:
        points = np.asarray(xyz_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("xyz_m must be finite shape (N, 3)")
        level = int(level_mm)
        if level not in self.levels_mm:
            raise ValueError(f"unregistered grid level {level}")
        x_low, x_high = self.x_slab_m
        inside = points[(points[:, 0] >= x_low) & (points[:, 0] <= x_high)]
        if not len(inside):
            return frozenset()
        step = level / 1000.0
        indices = np.floor(
            (inside - np.asarray(self.origin_m, dtype=float).reshape(1, 3)) / step
        ).astype(np.int64)
        return frozenset(
            CellKey(level, int(row[0]), int(row[1]), int(row[2]))
            for row in indices
        )


@dataclass(frozen=True)
class ReachReplica:
    """One cumulative independently scrambled forward-sampling replica."""

    replica_id: str
    scramble_seeds: tuple[int, ...]
    xyz_m: np.ndarray

    def __post_init__(self) -> None:
        replica_id = str(self.replica_id).strip()
        seeds = tuple(int(value) for value in self.scramble_seeds)
        xyz = np.asarray(self.xyz_m, dtype=float)
        if not replica_id:
            raise ValueError("replica_id must be non-empty")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("scramble_seeds must be non-empty and unique")
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("xyz_m must be finite shape (N, 3)")
        object.__setattr__(self, "replica_id", replica_id)
        object.__setattr__(self, "scramble_seeds", seeds)
        object.__setattr__(self, "xyz_m", xyz.copy())


@dataclass(frozen=True)
class ReachReplicaRound:
    round_id: int
    replica_a: ReachReplica
    replica_b: ReachReplica

    def __post_init__(self) -> None:
        if int(self.round_id) < 1:
            raise ValueError("round_id must be positive")
        if self.replica_a.replica_id == self.replica_b.replica_id:
            raise ValueError("replica IDs must be distinct")
        if set(self.replica_a.scramble_seeds) & set(self.replica_b.scramble_seeds):
            raise ValueError("reach replicas must use independent scrambled seeds")
        object.__setattr__(self, "round_id", int(self.round_id))


@dataclass(frozen=True)
class FrontierProbeEvidence:
    """Active task-space evidence for one convergence-level cell."""

    cell: CellKey
    found_valid_inverse: bool
    unreachable_certificate: str | None = None
    round_id: int | None = None

    def __post_init__(self) -> None:
        certificate = (
            None
            if self.unreachable_certificate is None
            else str(self.unreachable_certificate).strip()
        )
        if self.found_valid_inverse and certificate:
            raise ValueError("a frontier cell cannot be both reachable and certified unreachable")
        if self.round_id is not None and int(self.round_id) < 1:
            raise ValueError("frontier round_id must be positive")
        object.__setattr__(self, "found_valid_inverse", bool(self.found_valid_inverse))
        object.__setattr__(self, "unreachable_certificate", certificate or None)
        object.__setattr__(
            self, "round_id", None if self.round_id is None else int(self.round_id)
        )


@dataclass(frozen=True)
class ReachProxyConfig:
    grid: WorkspaceGridSpec = field(default_factory=WorkspaceGridSpec)
    minimum_weighted_jaccard: float = 0.95
    maximum_new_volume_ratio: float = 0.01
    maximum_boundary_change_ratio: float = 0.02
    maximum_frontier_new_volume_ratio: float = 0.01
    required_consecutive_rounds: int = 2

    def __post_init__(self) -> None:
        fractions = (
            self.minimum_weighted_jaccard,
            self.maximum_new_volume_ratio,
            self.maximum_boundary_change_ratio,
            self.maximum_frontier_new_volume_ratio,
        )
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in fractions):
            raise ValueError("reach proxy thresholds must be finite fractions")
        if int(self.required_consecutive_rounds) < 1:
            raise ValueError("required_consecutive_rounds must be positive")
        object.__setattr__(self, "required_consecutive_rounds", int(self.required_consecutive_rounds))


@dataclass(frozen=True)
class ReachReplicaMetric:
    round_id: int
    replica_a_cell_count: int
    replica_b_cell_count: int
    intersection_cell_count: int
    union_cell_count: int
    raw_jaccard: float
    volume_weighted_jaccard: float
    new_volume_ratio: float
    boundary_change_ratio: float
    frontier_new_volume_ratio: float
    gate_pass: bool


@dataclass(frozen=True)
class ReachProxyResult:
    proxy_lower_cells: frozenset[CellKey]
    proxy_upper_cells: frozenset[CellKey]
    certified_unreachable_cells: frozenset[CellKey]
    status_by_cell: Mapping[CellKey, ReachStatus]
    occupied_by_level: Mapping[int, frozenset[CellKey]]
    replica_metrics: tuple[ReachReplicaMetric, ...]
    convergence_gate: bool


def _jaccard(left: frozenset[CellKey], right: frozenset[CellKey]) -> float:
    union = left | right
    return 1.0 if not union else float(len(left & right) / len(union))


def _weighted_jaccard(left: frozenset[CellKey], right: frozenset[CellKey]) -> float:
    union = left | right
    if not union:
        return 1.0
    numerator = sum(cell.volume_m3 for cell in left & right)
    denominator = sum(cell.volume_m3 for cell in union)
    return float(numerator / denominator)


def _boundary(cells: frozenset[CellKey]) -> frozenset[CellKey]:
    return frozenset(
        cell for cell in cells if any(neighbor not in cells for neighbor in cell.face_neighbors())
    )


def _change_ratio(current: frozenset[CellKey], previous: frozenset[CellKey]) -> float:
    if not previous:
        return 1.0 if current else 0.0
    return float(len(current - previous) / len(previous))


def _symmetric_change_ratio(current: frozenset[CellKey], previous: frozenset[CellKey]) -> float:
    union = current | previous
    return 0.0 if not union else float(len(current ^ previous) / len(union))


def measure_weighted_boundary_change_ratio(
    current_cells: frozenset[CellKey],
    previous_cells: frozenset[CellKey],
) -> float:
    """Measure boundary change relative to occupied workspace measure.

    The historical diagnostic divides the symmetric difference by the union
    of the two *boundary* sets.  Thin boundary motion can therefore look large
    even when it changes little of the represented workspace.  V14.2 retains
    that diagnostic and adds this separately named, measure-aware quantity:

    ``measure(boundary_t xor boundary_t-1) / measure(omega_t union omega_t-1)``.
    """

    current = frozenset(current_cells)
    previous = frozenset(previous_cells)
    occupied = current | previous
    if not occupied:
        return 0.0
    changed_boundary = _boundary(current) ^ _boundary(previous)
    numerator = sum(cell.volume_m3 for cell in changed_boundary)
    denominator = sum(cell.volume_m3 for cell in occupied)
    return float(numerator / denominator)


class ReachProxyBuilder:
    """Reduce independent forward replicas and explicit frontier evidence."""

    def __init__(self, config: ReachProxyConfig | None = None) -> None:
        self.config = ReachProxyConfig() if config is None else config

    def build(
        self,
        rounds: Sequence[ReachReplicaRound],
        *,
        frontier_evidence: Sequence[FrontierProbeEvidence] = (),
        supplemental_supported_cells: Sequence[CellKey] = (),
    ) -> ReachProxyResult:
        ordered = tuple(sorted(rounds, key=lambda item: item.round_id))
        if not ordered:
            raise ValueError("at least one independent replica round is required")
        if len({item.round_id for item in ordered}) != len(ordered):
            raise ValueError("reach replica round IDs must be unique")

        level = self.config.grid.convergence_level_mm
        evidence = tuple(frontier_evidence)
        supplemental = frozenset(supplemental_supported_cells)
        for row in evidence:
            if row.cell.level_mm != level:
                raise ValueError("frontier evidence must use the convergence grid level")
        if any(cell.level_mm != level for cell in supplemental):
            raise ValueError("supplemental support must use the convergence grid level")

        metrics: list[ReachReplicaMetric] = []
        previous_union: frozenset[CellKey] = frozenset()
        previous_boundary: frozenset[CellKey] = frozenset()
        latest_a: frozenset[CellKey] = frozenset()
        latest_b: frozenset[CellKey] = frozenset()

        for pair in ordered:
            # Re-run validation here so objects loaded through alternate adapters
            # cannot bypass the cross-replica independence invariant.
            if set(pair.replica_a.scramble_seeds) & set(pair.replica_b.scramble_seeds):
                raise ValueError("reach replicas must use independent scrambled seeds")
            latest_a = self.config.grid.cells_for_points(
                pair.replica_a.xyz_m, level_mm=level
            )
            latest_b = self.config.grid.cells_for_points(
                pair.replica_b.xyz_m, level_mm=level
            )
            current_union = latest_a | latest_b
            current_boundary = _boundary(current_union)
            active_successes = {
                row.cell
                for row in evidence
                if row.found_valid_inverse
                and (row.round_id is None or row.round_id == pair.round_id)
            }
            frontier_new_ratio = float(
                len(active_successes - current_union) / max(1, len(current_union))
            )
            weighted = _weighted_jaccard(latest_a, latest_b)
            new_ratio = _change_ratio(current_union, previous_union)
            boundary_ratio = _symmetric_change_ratio(current_boundary, previous_boundary)
            gate = bool(
                weighted >= self.config.minimum_weighted_jaccard
                and new_ratio <= self.config.maximum_new_volume_ratio
                and boundary_ratio <= self.config.maximum_boundary_change_ratio
                and frontier_new_ratio <= self.config.maximum_frontier_new_volume_ratio
            )
            metrics.append(
                ReachReplicaMetric(
                    round_id=pair.round_id,
                    replica_a_cell_count=len(latest_a),
                    replica_b_cell_count=len(latest_b),
                    intersection_cell_count=len(latest_a & latest_b),
                    union_cell_count=len(current_union),
                    raw_jaccard=_jaccard(latest_a, latest_b),
                    volume_weighted_jaccard=weighted,
                    new_volume_ratio=new_ratio,
                    boundary_change_ratio=boundary_ratio,
                    frontier_new_volume_ratio=frontier_new_ratio,
                    gate_pass=gate,
                )
            )
            previous_union = current_union
            previous_boundary = current_boundary

        certified = frozenset(
            row.cell for row in evidence if row.unreachable_certificate is not None
        )
        frontier_supported = frozenset(
            row.cell for row in evidence if row.found_valid_inverse
        )
        frontier_unresolved = frozenset(
            row.cell
            for row in evidence
            if not row.found_valid_inverse and row.unreachable_certificate is None
        )
        latest_union = latest_a | latest_b
        lower = frozenset(
            ((latest_a & latest_b) | frontier_supported | supplemental) - certified
        )
        upper = frozenset(
            (
                latest_union
                | frontier_supported
                | frontier_unresolved
                | supplemental
            )
            - certified
        )
        status_cells = (
            latest_union
            | frontier_supported
            | frontier_unresolved
            | supplemental
            | certified
        )
        status: dict[CellKey, ReachStatus] = {}
        for cell in sorted(status_cells):
            if cell in certified:
                status[cell] = ReachStatus.UNREACHABLE_CERTIFIED
            elif cell in latest_union or cell in frontier_supported or cell in supplemental:
                status[cell] = ReachStatus.EMPIRICAL_SUPPORTED
            else:
                status[cell] = ReachStatus.EMPIRICAL_UNSUPPORTED

        occupied_by_level = {
            registered_level: frozenset(
                self.config.grid.cells_for_points(
                    np.vstack([ordered[-1].replica_a.xyz_m, ordered[-1].replica_b.xyz_m]),
                    level_mm=registered_level,
                )
            )
            for registered_level in self.config.grid.levels_mm
        }
        required = self.config.required_consecutive_rounds
        convergence = bool(
            len(metrics) >= required
            and all(metric.gate_pass for metric in metrics[-required:])
        )
        return ReachProxyResult(
            proxy_lower_cells=lower,
            proxy_upper_cells=upper,
            certified_unreachable_cells=certified,
            status_by_cell=MappingProxyType(status),
            occupied_by_level=MappingProxyType(occupied_by_level),
            replica_metrics=tuple(metrics),
            convergence_gate=convergence,
        )
