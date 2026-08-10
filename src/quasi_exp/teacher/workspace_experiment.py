"""Numerical generation primitives for the BACRA V14 workspace experiment.

The module owns the two sampling details that are easy to accidentally weaken
in a runner: reach replicas are independently scrambled (never slices of one
Sobol pool), and the near-zero-pose pool is supplemental reach evidence rather
than an allegedly independent replica.  Artifact persistence and IK frontier
solves remain orchestration concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np
from scipy.stats import qmc

from .capability_map import batch_fk, nested_sobol_beta
from .workspace_reach import (
    CellKey,
    ReachReplica,
    ReachReplicaRound,
    WorkspaceGridSpec,
)


class _ForwardEnvironment(Protocol):
    def fk(self, beta_rad: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ReachSamplingRoundSpec:
    power: int
    seed_a: int
    seed_b: int

    def __post_init__(self) -> None:
        if int(self.power) < 1:
            raise ValueError("reach sampling power must be positive")
        if int(self.seed_a) == int(self.seed_b):
            raise ValueError("A/B reach replicas require distinct scramble seeds")
        object.__setattr__(self, "power", int(self.power))
        object.__setattr__(self, "seed_a", int(self.seed_a))
        object.__setattr__(self, "seed_b", int(self.seed_b))


@dataclass(frozen=True)
class GeneratedReachSamples:
    rounds: tuple[ReachReplicaRound, ...]
    beta_a: np.ndarray
    xyz_a: np.ndarray
    beta_b: np.ndarray
    xyz_b: np.ndarray

    def __post_init__(self) -> None:
        for name in ("a", "b"):
            beta = np.asarray(getattr(self, f"beta_{name}"), dtype=float)
            xyz = np.asarray(getattr(self, f"xyz_{name}"), dtype=float)
            if (
                beta.ndim != 2
                or beta.shape[1] != 6
                or xyz.shape != (len(beta), 3)
                or not np.isfinite(beta).all()
                or not np.isfinite(xyz).all()
            ):
                raise ValueError("generated reach arrays must be finite aligned beta6/xyz")
            object.__setattr__(self, f"beta_{name}", beta.copy())
            object.__setattr__(self, f"xyz_{name}", xyz.copy())


def _validate_bounds(bounds_rad: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds_rad, dtype=float)
    if (
        bounds.shape != (6, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("bounds_rad must be finite ordered shape (6, 2)")
    return bounds


def _slab_filter(
    beta_rad: np.ndarray, xyz_m: np.ndarray, grid: WorkspaceGridSpec
) -> tuple[np.ndarray, np.ndarray]:
    x_low, x_high = grid.x_slab_m
    selected = (xyz_m[:, 0] >= x_low) & (xyz_m[:, 0] <= x_high)
    return beta_rad[selected], xyz_m[selected]


def generate_independent_reach_samples(
    environment: _ForwardEnvironment,
    bounds_rad: np.ndarray,
    *,
    grid: WorkspaceGridSpec,
    rounds: Sequence[ReachSamplingRoundSpec],
    chunk_rows: int,
) -> GeneratedReachSamples:
    """Generate cumulative A/B replicas from disjoint scrambled sequences.

    Each registered round adds a fresh power-of-two batch to each replica.
    Only slab points are retained in memory because points outside the target
    predicate cannot affect its occupancy.  The scramble-seed lineage remains
    attached to every cumulative round.
    """

    bounds = _validate_bounds(bounds_rad)
    specs = tuple(rounds)
    if not specs:
        raise ValueError("at least one reach sampling round is required")
    seeds_a = [item.seed_a for item in specs]
    seeds_b = [item.seed_b for item in specs]
    if (
        len(set(seeds_a)) != len(seeds_a)
        or len(set(seeds_b)) != len(seeds_b)
        or set(seeds_a) & set(seeds_b)
    ):
        raise ValueError("all A/B reach scramble seeds must be globally disjoint")
    if int(chunk_rows) < 1:
        raise ValueError("chunk_rows must be positive")

    beta_parts: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    xyz_parts: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    seed_lineage: dict[str, list[int]] = {"A": [], "B": []}
    generated_rounds: list[ReachReplicaRound] = []
    for round_id, spec in enumerate(specs, start=1):
        for replica_id, seed in (("A", spec.seed_a), ("B", spec.seed_b)):
            beta = nested_sobol_beta(bounds, power=spec.power, seed=seed)
            xyz = batch_fk(environment, beta, chunk_rows=int(chunk_rows))
            beta, xyz = _slab_filter(beta, xyz, grid)
            beta_parts[replica_id].append(beta)
            xyz_parts[replica_id].append(xyz)
            seed_lineage[replica_id].append(seed)
        cumulative_xyz_a = np.vstack(xyz_parts["A"])
        cumulative_xyz_b = np.vstack(xyz_parts["B"])
        generated_rounds.append(
            ReachReplicaRound(
                round_id=round_id,
                replica_a=ReachReplica(
                    replica_id="A",
                    scramble_seeds=tuple(seed_lineage["A"]),
                    xyz_m=cumulative_xyz_a,
                ),
                replica_b=ReachReplica(
                    replica_id="B",
                    scramble_seeds=tuple(seed_lineage["B"]),
                    xyz_m=cumulative_xyz_b,
                ),
            )
        )
    beta_a = np.vstack(beta_parts["A"])
    beta_b = np.vstack(beta_parts["B"])
    return GeneratedReachSamples(
        rounds=tuple(generated_rounds),
        beta_a=beta_a,
        xyz_a=np.vstack(xyz_parts["A"]),
        beta_b=beta_b,
        xyz_b=np.vstack(xyz_parts["B"]),
    )


def tip_focused_beta(
    bounds_rad: np.ndarray,
    *,
    power: int,
    seed: int,
    scale_max: float = 0.25,
) -> np.ndarray:
    """Sample independently scaled perturbations around the bounds midpoint."""

    bounds = _validate_bounds(bounds_rad)
    if int(power) < 1:
        raise ValueError("tip Sobol power must be positive")
    if not math.isfinite(float(scale_max)) or not 0.0 < float(scale_max) <= 1.0:
        raise ValueError("scale_max must be in (0, 1]")
    unit = qmc.Sobol(d=12, scramble=True, seed=int(seed)).random_base2(int(power))
    direction = 2.0 * unit[:, :6] - 1.0
    scale = float(scale_max) * unit[:, 6:]
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])
    return midpoint[None, :] + direction * scale * halfspan[None, :]


def cell_centers(
    cells: Sequence[CellKey], *, grid: WorkspaceGridSpec
) -> np.ndarray:
    """Return metric centers for stable origin-anchored cell identities."""

    rows = tuple(cells)
    result = np.empty((len(rows), 3), dtype=float)
    origin = np.asarray(grid.origin_m, dtype=float)
    for index, cell in enumerate(rows):
        if cell.level_mm not in grid.levels_mm:
            raise ValueError("cell uses an unregistered workspace level")
        step = cell.level_mm / 1000.0
        result[index] = origin + step * (
            np.asarray([cell.ix, cell.iy, cell.iz], dtype=float) + 0.5
        )
    return result


def frontier_candidates(
    supported_cells: Sequence[CellKey],
    *,
    grid: WorkspaceGridSpec,
    maximum_count: int,
) -> tuple[CellKey, ...]:
    """Select deterministic maximin unsupported face-neighbour frontier cells."""

    supported = frozenset(supported_cells)
    if int(maximum_count) < 0:
        raise ValueError("maximum_count must be non-negative")
    if not supported or maximum_count == 0:
        return ()
    levels = {cell.level_mm for cell in supported}
    if levels != {grid.convergence_level_mm}:
        raise ValueError("frontier discovery requires convergence-level cells")
    raw = {
        neighbour
        for cell in supported
        for neighbour in cell.face_neighbors()
        if neighbour not in supported
    }
    ordered = tuple(sorted(raw))
    if not ordered:
        return ()
    centers = cell_centers(ordered, grid=grid)
    x_low, x_high = grid.x_slab_m
    eligible_indices = [
        index
        for index, xyz in enumerate(centers)
        if x_low <= float(xyz[0]) <= x_high
    ]
    if len(eligible_indices) <= int(maximum_count):
        return tuple(ordered[index] for index in eligible_indices)

    # Stable farthest-point traversal.  The first lexicographic cell is fixed,
    # then geometry dominates with the stable index as final tie-break.
    selected = [eligible_indices[0]]
    pending = set(eligible_indices[1:])
    while pending and len(selected) < int(maximum_count):
        best = max(
            pending,
            key=lambda index: (
                min(
                    float(np.linalg.norm(centers[index] - centers[previous]))
                    for previous in selected
                ),
                -index,
            ),
        )
        selected.append(best)
        pending.remove(best)
    return tuple(ordered[index] for index in selected)
