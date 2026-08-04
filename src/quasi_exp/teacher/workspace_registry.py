"""Empirical task-space registry and deterministic BACRA V14 Pilot sampler.

The registry is deliberately a *forward-sample* data structure.  It groups
given capability rows into the nested 20/10/5 mm grid, but it does not invoke
FK, IK, or a reachability solver.  Consequently, a selected cell only means
that the supplied finite sample pool supported it; neither unobserved portions
of a cell nor cells marked unsupported become query targets by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .workspace_reach import CellKey, ReachStatus, WorkspaceGridSpec


class CellStratum(str, Enum):
    """Named, non-exclusive registry strata.

    A cell may appear in several strata.  In particular, retaining a legacy
    point does not erase its boundary, tip, density, or conditioning evidence.
    """

    BOUNDARY = "boundary"
    TIP = "tip"
    LOW_DENSITY = "low_density"
    ILL_CONDITIONED = "ill_conditioned"
    INTERIOR = "interior"
    RETENTION = "retention"


@dataclass(frozen=True)
class CapabilityColumns:
    """Column contract for a capability DataFrame.

    ``sample_id`` must identify one real FK sample.  The defaults match the
    capability-map materialisation, while callers with another frozen schema
    can pass an explicit ``CapabilityColumns`` instance.
    """

    sample_id: str = "sample_id"
    xyz_m: tuple[str, str, str] = ("x_m", "y_m", "z_m")
    beta_rad: tuple[str, str, str, str, str, str] = (
        "beta1_rad",
        "beta2_rad",
        "beta3_rad",
        "beta4_rad",
        "beta5_rad",
        "beta6_rad",
    )
    reach_status: str | None = "reach_status"
    retention: str | None = "retention"
    tip: str | None = "tip"
    ill_conditioned: str | None = "ill_conditioned"
    sigma3_m: str | None = "sigma3_m"
    kappa: str | None = "kappa"

    def __post_init__(self) -> None:
        names = (self.sample_id, *self.xyz_m, *self.beta_rad)
        if any(not str(name).strip() for name in names):
            raise ValueError("capability column names must be non-empty")
        if len(self.xyz_m) != 3 or len(self.beta_rad) != 6:
            raise ValueError("capability schema requires xyz=3 and beta=6 columns")


@dataclass(frozen=True)
class WorkspaceRegistryPolicy:
    """Registered sampling policy; all thresholds have empirical semantics."""

    beta_bounds_rad: np.ndarray
    pilot_level_mm: int = 10
    pilot_x_bins: int = 4
    measure_probe_count: int = 3
    max_seed_betas: int = 4
    seed_cluster_radius_normalized: float = 0.025
    low_density_max_samples: int = 8
    tip_x_fraction: float = 0.10
    ill_conditioned_sigma3_m_max: float | None = None
    ill_conditioned_kappa_min: float | None = None
    bounds_atol: float = 1.0e-12
    selection_seed: int = 20260805

    def __post_init__(self) -> None:
        bounds = np.asarray(self.beta_bounds_rad, dtype=float)
        if (
            bounds.shape != (6, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError("beta_bounds_rad must be finite ordered shape (6, 2)")
        if int(self.pilot_level_mm) <= 0 or int(self.pilot_x_bins) < 1:
            raise ValueError("pilot grid level and x bin count must be positive")
        if not 2 <= int(self.measure_probe_count) <= 4:
            raise ValueError("measure_probe_count must be in [2, 4]")
        if not 1 <= int(self.max_seed_betas) <= 4:
            raise ValueError("max_seed_betas must be in [1, 4]")
        finite_positive = (
            self.seed_cluster_radius_normalized,
            self.tip_x_fraction,
            self.bounds_atol,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in finite_positive):
            raise ValueError("registry policy fractions must be finite and non-negative")
        if self.tip_x_fraction > 1.0:
            raise ValueError("tip_x_fraction must not exceed one")
        if int(self.low_density_max_samples) < 1:
            raise ValueError("low_density_max_samples must be positive")
        for value in (
            self.ill_conditioned_sigma3_m_max,
            self.ill_conditioned_kappa_min,
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError("ill-conditioning thresholds must be finite and non-negative")
        object.__setattr__(self, "beta_bounds_rad", bounds.copy())
        object.__setattr__(self, "pilot_level_mm", int(self.pilot_level_mm))
        object.__setattr__(self, "pilot_x_bins", int(self.pilot_x_bins))
        object.__setattr__(self, "measure_probe_count", int(self.measure_probe_count))
        object.__setattr__(self, "max_seed_betas", int(self.max_seed_betas))
        object.__setattr__(self, "low_density_max_samples", int(self.low_density_max_samples))
        object.__setattr__(self, "selection_seed", int(self.selection_seed))


@dataclass(frozen=True)
class RegisteredCell:
    """One empirically supported cell and the real samples registered to it."""

    cell: CellKey
    sample_ids: tuple[str, ...]
    sample_count: int
    occupancy_fraction: float | None
    strata: frozenset[CellStratum]
    x_center_m: float

    def __post_init__(self) -> None:
        ids = tuple(str(value).strip() for value in self.sample_ids)
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("registered cells require unique non-empty sample IDs")
        if int(self.sample_count) != len(ids):
            raise ValueError("sample_count must equal the number of sample IDs")
        occupancy = self.occupancy_fraction
        if occupancy is not None and (
            not math.isfinite(float(occupancy)) or not 0.0 <= float(occupancy) <= 1.0
        ):
            raise ValueError("occupancy_fraction must be a finite fraction")
        if not math.isfinite(float(self.x_center_m)):
            raise ValueError("x_center_m must be finite")
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "strata", frozenset(CellStratum(value) for value in self.strata))


@dataclass(frozen=True)
class RegisteredSample:
    """A real supported capability row retained for probe and seed selection."""

    sample_id: str
    xyz_m: np.ndarray
    beta_rad: np.ndarray

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        xyz = np.asarray(self.xyz_m, dtype=float).reshape(3)
        beta = np.asarray(self.beta_rad, dtype=float).reshape(6)
        if not sample_id or not np.isfinite(xyz).all() or not np.isfinite(beta).all():
            raise ValueError("registered samples need a non-empty ID and finite xyz/beta")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "xyz_m", xyz.copy())
        object.__setattr__(self, "beta_rad", beta.copy())


@dataclass(frozen=True, eq=False)
class PilotCell:
    """A no-synthesis Pilot request assembled only from registered FK rows."""

    cell: CellKey
    strata: frozenset[CellStratum]
    x_bin: int
    representative_sample_id: str
    measure_sample_ids: tuple[str, ...]
    seed_sample_ids: tuple[str, ...]
    seed_beta_rad: np.ndarray

    def __post_init__(self) -> None:
        representative = str(self.representative_sample_id).strip()
        measure = tuple(str(value).strip() for value in self.measure_sample_ids)
        seed_ids = tuple(str(value).strip() for value in self.seed_sample_ids)
        seed_beta = np.asarray(self.seed_beta_rad, dtype=float)
        if not representative or len(measure) < 2 or len(measure) > 4:
            raise ValueError("Pilot cells require a representative and 2-4 measure samples")
        if any(not value for value in measure + seed_ids) or len(set(measure)) != len(measure):
            raise ValueError("Pilot sample IDs must be non-empty and measure IDs unique")
        if representative in measure:
            raise ValueError("representative and measure probes must be distinct real samples")
        if len(seed_ids) != len(set(seed_ids)) or len(seed_ids) != len(seed_beta):
            raise ValueError("seed IDs and seed beta rows must align uniquely")
        if (
            not 1 <= len(seed_ids) <= 4
            or seed_beta.shape != (len(seed_ids), 6)
            or not np.isfinite(seed_beta).all()
        ):
            raise ValueError("Pilot seed betas must be finite shape (1..4, 6)")
        object.__setattr__(self, "strata", frozenset(CellStratum(value) for value in self.strata))
        object.__setattr__(self, "representative_sample_id", representative)
        object.__setattr__(self, "measure_sample_ids", measure)
        object.__setattr__(self, "seed_sample_ids", seed_ids)
        object.__setattr__(self, "seed_beta_rad", seed_beta.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PilotCell):
            return NotImplemented
        return bool(
            self.cell == other.cell
            and self.strata == other.strata
            and self.x_bin == other.x_bin
            and self.representative_sample_id == other.representative_sample_id
            and self.measure_sample_ids == other.measure_sample_ids
            and self.seed_sample_ids == other.seed_sample_ids
            and np.array_equal(self.seed_beta_rad, other.seed_beta_rad)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.cell,
                self.strata,
                self.x_bin,
                self.representative_sample_id,
                self.measure_sample_ids,
                self.seed_sample_ids,
                self.seed_beta_rad.dtype.str,
                self.seed_beta_rad.shape,
                self.seed_beta_rad.tobytes(),
            )
        )


@dataclass(frozen=True)
class PilotSelection:
    """Selected Pilot cells and their stable six-face task graph."""

    cells: tuple[PilotCell, ...]
    task_edges: tuple[tuple[CellKey, CellKey], ...]
    requested_max_cells: int
    eligible_cell_count: int

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        keys = tuple(item.cell for item in cells)
        edges = tuple(_ordered_edge(*edge) for edge in self.task_edges)
        if len(keys) != len(set(keys)):
            raise ValueError("Pilot cells must have unique cell keys")
        if any(left not in keys or right not in keys for left, right in edges):
            raise ValueError("task graph edges must connect selected cells")
        if any(right not in left.face_neighbors() for left, right in edges):
            raise ValueError("task graph must contain only six-face neighbors")
        if len(edges) != len(set(edges)):
            raise ValueError("task graph edges must be unique")
        if int(self.requested_max_cells) < 0 or int(self.eligible_cell_count) < len(cells):
            raise ValueError("invalid Pilot selection counts")
        object.__setattr__(self, "cells", tuple(sorted(cells, key=lambda item: item.cell)))
        object.__setattr__(self, "task_edges", tuple(sorted(edges)))
        object.__setattr__(self, "requested_max_cells", int(self.requested_max_cells))
        object.__setattr__(self, "eligible_cell_count", int(self.eligible_cell_count))


@dataclass(frozen=True)
class WorkspaceRegistryResult:
    """Nested empirical support cells plus registered real capability samples."""

    grid: WorkspaceGridSpec
    policy: WorkspaceRegistryPolicy
    cells_by_level: Mapping[int, Mapping[CellKey, RegisteredCell]]
    samples_by_id: Mapping[str, RegisteredSample]

    @property
    def pilot_cells(self) -> Mapping[CellKey, RegisteredCell]:
        return self.cells_by_level[self.policy.pilot_level_mm]

    def select_pilot(self, max_cells: int) -> PilotSelection:
        """Choose exactly ``min(max_cells, eligible)`` graph-aware Pilot cells.

        The seed set first covers named strata and x bins.  Remaining quota is
        filled from six-face neighbours of selected cells, preferring cells
        that close more graph edges and underrepresented strata/bins.  This
        yields connected local patches suitable for continuation audits rather
        than a spatially scattered collection of individually useful cells.
        """

        limit = int(max_cells)
        if limit < 0:
            raise ValueError("max_cells must be non-negative")
        eligible = {
            key: cell
            for key, cell in self.pilot_cells.items()
            if cell.sample_count >= self.policy.measure_probe_count + 1
        }
        if limit == 0 or not eligible:
            return PilotSelection((), (), limit, len(eligible))
        selected: list[CellKey] = []
        target = min(limit, len(eligible))
        atomic_strata = sorted(
            {stratum for cell in eligible.values() for stratum in cell.strata},
            key=lambda item: item.value,
        )
        x_bins = sorted({self._x_bin(cell.x_center_m) for cell in eligible.values()})

        def add_best(candidates: Sequence[CellKey]) -> None:
            available = [key for key in candidates if key not in selected]
            if available and len(selected) < target:
                selected.append(min(available, key=self._selection_rank))

        for stratum in atomic_strata:
            add_best([key for key, cell in eligible.items() if stratum in cell.strata])
        for x_bin in x_bins:
            add_best(
                [
                    key
                    for key, cell in eligible.items()
                    if self._x_bin(cell.x_center_m) == x_bin
                ]
            )

        while len(selected) < target:
            selected_set = set(selected)
            frontier = {
                neighbor
                for key in selected
                for neighbor in key.face_neighbors()
                if neighbor in eligible and neighbor not in selected_set
            }
            candidates = frontier or (set(eligible) - selected_set)
            stratum_counts = {
                stratum: sum(stratum in eligible[key].strata for key in selected)
                for stratum in atomic_strata
            }
            bin_counts = {
                x_bin: sum(
                    self._x_bin(eligible[key].x_center_m) == x_bin for key in selected
                )
                for x_bin in x_bins
            }

            def expansion_rank(key: CellKey) -> tuple[object, ...]:
                cell = eligible[key]
                connected_edges = sum(
                    neighbor in selected_set for neighbor in key.face_neighbors()
                )
                least_stratum = min(
                    (stratum_counts[value] for value in cell.strata), default=0
                )
                x_bin = self._x_bin(cell.x_center_m)
                stable = self._selection_rank(key)
                return (-connected_edges, least_stratum, bin_counts[x_bin], *stable)

            selected.append(min(candidates, key=expansion_rank))
        pilots = tuple(self._pilot_cell(eligible[key]) for key in selected)
        selected_keys = frozenset(selected)
        edges = tuple(
            (key, neighbor)
            for key in sorted(selected_keys)
            for neighbor in key.face_neighbors()
            if neighbor in selected_keys and key < neighbor
        )
        return PilotSelection(pilots, edges, limit, len(eligible))

    def _x_bin(self, x_m: float) -> int:
        low, high = self.grid.x_slab_m
        relative = (float(x_m) - low) / (high - low)
        return min(
            self.policy.pilot_x_bins - 1,
            max(0, int(math.floor(relative * self.policy.pilot_x_bins))),
        )

    def _selection_rank(self, cell: CellKey) -> tuple[bytes, CellKey]:
        payload = f"{self.policy.selection_seed}:{cell.level_mm}:{cell.ix}:{cell.iy}:{cell.iz}"
        return hashlib.sha256(payload.encode("utf-8")).digest(), cell

    def _pilot_cell(self, cell: RegisteredCell) -> PilotCell:
        samples = tuple(self.samples_by_id[sample_id] for sample_id in cell.sample_ids)
        center = _cell_center_m(cell.cell, self.grid)
        representative = min(
            samples,
            key=lambda item: (float(np.linalg.norm(item.xyz_m - center)), item.sample_id),
        )
        remaining = tuple(item for item in samples if item.sample_id != representative.sample_id)
        measure = _maximin_samples(remaining, self.policy.measure_probe_count, space="xyz")
        seeds = _clustered_maximin_seed_samples(
            samples,
            representative_sample_id=representative.sample_id,
            beta_bounds_rad=self.policy.beta_bounds_rad,
            cluster_radius=self.policy.seed_cluster_radius_normalized,
            max_count=self.policy.max_seed_betas,
        )
        return PilotCell(
            cell=cell.cell,
            strata=cell.strata,
            x_bin=self._x_bin(cell.x_center_m),
            representative_sample_id=representative.sample_id,
            measure_sample_ids=tuple(item.sample_id for item in measure),
            seed_sample_ids=tuple(item.sample_id for item in seeds),
            seed_beta_rad=np.vstack([item.beta_rad for item in seeds]),
        )


class WorkspaceRegistryBuilder:
    """Build a nested registry from frozen capability rows without solving."""

    def __init__(
        self,
        grid: WorkspaceGridSpec,
        policy: WorkspaceRegistryPolicy,
        columns: CapabilityColumns | None = None,
    ) -> None:
        if not {20, 10, 5}.issubset(set(grid.levels_mm)):
            raise ValueError("workspace registry requires registered 20/10/5 mm levels")
        if policy.pilot_level_mm not in grid.levels_mm:
            raise ValueError("pilot_level_mm must be registered by WorkspaceGridSpec")
        self.grid = grid
        self.policy = policy
        self.columns = CapabilityColumns() if columns is None else columns

    def build(self, capability_rows: pd.DataFrame) -> WorkspaceRegistryResult:
        """Register finite, empirically supported capability rows in the x slab.

        A ``reach_status`` column, when present, is a filter rather than a
        request to investigate missing rows.  Only ``empirical_supported``
        records contribute samples, occupancy, strata, or Pilot probes.
        """

        if not isinstance(capability_rows, pd.DataFrame):
            raise TypeError("capability_rows must be a pandas DataFrame")
        frame = capability_rows.copy()
        self._require_columns(frame)
        sample_ids = frame[self.columns.sample_id].astype(str).str.strip()
        if not len(frame) or (sample_ids == "").any() or sample_ids.duplicated().any():
            raise ValueError("capability_rows require unique non-empty sample_id values")
        numeric_columns = (*self.columns.xyz_m, *self.columns.beta_rad)
        numeric = frame.loc[:, list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("capability xyz and beta columns must be finite")
        frame.loc[:, list(numeric_columns)] = numeric
        support = self._empirical_support_mask(frame)
        frame = frame.loc[support].copy()
        frame[self.columns.sample_id] = sample_ids.loc[support]
        xyz = frame.loc[:, list(self.columns.xyz_m)].to_numpy(dtype=float)
        beta = frame.loc[:, list(self.columns.beta_rad)].to_numpy(dtype=float)
        self._validate_registered_bounds(beta)
        inside = (xyz[:, 0] >= self.grid.x_slab_m[0]) & (xyz[:, 0] <= self.grid.x_slab_m[1])
        frame = frame.loc[inside].copy().reset_index(drop=True)
        xyz = xyz[inside]
        beta = beta[inside]
        samples = {
            str(frame.iloc[index][self.columns.sample_id]): RegisteredSample(
                sample_id=str(frame.iloc[index][self.columns.sample_id]),
                xyz_m=xyz[index],
                beta_rad=beta[index],
            )
            for index in range(len(frame))
        }
        cells_by_level: dict[int, dict[CellKey, RegisteredCell]] = {}
        for level in self.grid.levels_mm:
            membership = self._membership(frame, xyz, level)
            cells_by_level[level] = self._registered_cells(level, membership, frame, xyz)
        return WorkspaceRegistryResult(
            grid=self.grid,
            policy=self.policy,
            cells_by_level=MappingProxyType(
                {
                    level: MappingProxyType(dict(sorted(cells.items())))
                    for level, cells in cells_by_level.items()
                }
            ),
            samples_by_id=MappingProxyType(dict(sorted(samples.items()))),
        )

    def _require_columns(self, frame: pd.DataFrame) -> None:
        required = {self.columns.sample_id, *self.columns.xyz_m, *self.columns.beta_rad}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"capability_rows missing required columns: {missing}")

    def _empirical_support_mask(self, frame: pd.DataFrame) -> np.ndarray:
        column = self.columns.reach_status
        if column is None or column not in frame.columns:
            return np.ones(len(frame), dtype=bool)
        status = frame[column].astype(str).str.strip()
        known = {item.value for item in ReachStatus}
        unknown = sorted(set(status) - known)
        if unknown:
            raise ValueError(f"unknown reach status values: {unknown}")
        return status.eq(ReachStatus.EMPIRICAL_SUPPORTED.value).to_numpy(dtype=bool)

    def _validate_registered_bounds(self, beta: np.ndarray) -> None:
        lower = self.policy.beta_bounds_rad[:, 0]
        upper = self.policy.beta_bounds_rad[:, 1]
        if np.any(beta < lower[None, :] - self.policy.bounds_atol) or np.any(
            beta > upper[None, :] + self.policy.bounds_atol
        ):
            raise ValueError("capability beta lies outside registered beta bounds")

    def _membership(
        self, frame: pd.DataFrame, xyz: np.ndarray, level: int
    ) -> Mapping[CellKey, tuple[int, ...]]:
        result: dict[CellKey, list[int]] = {}
        # ``cells_for_points`` deliberately returns a set, so compute aligned
        # membership here instead of accidentally losing row-to-sample identity.
        step = level / 1000.0
        indices = np.floor(
            (xyz - np.asarray(self.grid.origin_m).reshape(1, 3)) / step
        ).astype(np.int64)
        for row_index, row in enumerate(indices):
            key = CellKey(level, int(row[0]), int(row[1]), int(row[2]))
            result.setdefault(key, []).append(row_index)
        return MappingProxyType({key: tuple(value) for key, value in result.items()})

    def _registered_cells(
        self,
        level: int,
        membership: Mapping[CellKey, tuple[int, ...]],
        frame: pd.DataFrame,
        xyz: np.ndarray,
    ) -> dict[CellKey, RegisteredCell]:
        fine_cells = set()
        if level != 5:
            fine_membership = self._membership(frame, xyz, 5)
            fine_cells = set(fine_membership)
        ten_membership = membership if level == 10 else self._membership(frame, xyz, 10)
        ten_cells = set(ten_membership)
        centers = {
            key: _cell_center_m(key, self.grid)[0]
            for key in membership
        }
        tip_cutoff = self._tip_cutoff(tuple(centers.values()))
        result: dict[CellKey, RegisteredCell] = {}
        for key, rows in membership.items():
            row_indices = np.asarray(rows, dtype=int)
            ids = tuple(
                sorted(str(frame.iloc[index][self.columns.sample_id]) for index in row_indices)
            )
            occupancy = self._occupancy(key, fine_cells) if level == 10 else None
            strata = self._strata_for_cell(
                key,
                row_indices,
                frame,
                ten_cells,
                x_center=centers[key],
                tip_cutoff=tip_cutoff,
            ) if level == 10 else frozenset()
            result[key] = RegisteredCell(
                cell=key,
                sample_ids=ids,
                sample_count=len(ids),
                occupancy_fraction=occupancy,
                strata=strata,
                x_center_m=centers[key],
            )
        return result

    def _tip_cutoff(self, centers: Sequence[float]) -> float:
        if not centers:
            return math.inf
        minimum, maximum = min(centers), max(centers)
        return float(maximum - (maximum - minimum) * self.policy.tip_x_fraction)

    def _occupancy(self, parent: CellKey, fine_cells: set[CellKey]) -> float:
        children = {
            CellKey(5, parent.ix * 2 + dx, parent.iy * 2 + dy, parent.iz * 2 + dz)
            for dx in (0, 1)
            for dy in (0, 1)
            for dz in (0, 1)
        }
        return float(len(children & fine_cells) / 8.0)

    def _strata_for_cell(
        self,
        cell: CellKey,
        rows: np.ndarray,
        frame: pd.DataFrame,
        ten_cells: set[CellKey],
        *,
        x_center: float,
        tip_cutoff: float,
    ) -> frozenset[CellStratum]:
        strata: set[CellStratum] = set()
        if any(neighbor not in ten_cells for neighbor in cell.face_neighbors()):
            strata.add(CellStratum.BOUNDARY)
        else:
            strata.add(CellStratum.INTERIOR)
        if x_center >= tip_cutoff:
            strata.add(CellStratum.TIP)
        if len(rows) <= self.policy.low_density_max_samples:
            strata.add(CellStratum.LOW_DENSITY)
        if self._any_true(frame, rows, self.columns.retention):
            strata.add(CellStratum.RETENTION)
        if self._is_ill_conditioned(frame, rows):
            strata.add(CellStratum.ILL_CONDITIONED)
        return frozenset(strata)

    @staticmethod
    def _any_true(frame: pd.DataFrame, rows: np.ndarray, column: str | None) -> bool:
        if column is None or column not in frame.columns:
            return False
        values = frame.iloc[rows][column]
        if values.dtype == bool:
            return bool(values.any())
        normalized = values.astype(str).str.strip().str.lower()
        known = {"true", "false", "1", "0", "yes", "no"}
        if not set(normalized).issubset(known):
            raise ValueError(f"boolean registry column {column!r} contains invalid values")
        return bool(normalized.isin(("true", "1", "yes")).any())

    def _is_ill_conditioned(self, frame: pd.DataFrame, rows: np.ndarray) -> bool:
        if self._any_true(frame, rows, self.columns.ill_conditioned):
            return True
        sigma_column = self.columns.sigma3_m
        if self.policy.ill_conditioned_sigma3_m_max is not None and sigma_column in frame.columns:
            values = pd.to_numeric(
                frame.iloc[rows][sigma_column], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("sigma3 values must be finite when used for strata")
            if np.any(values <= self.policy.ill_conditioned_sigma3_m_max):
                return True
        kappa_column = self.columns.kappa
        if self.policy.ill_conditioned_kappa_min is not None and kappa_column in frame.columns:
            values = pd.to_numeric(
                frame.iloc[rows][kappa_column], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("kappa values must be finite when used for strata")
            if np.any(values >= self.policy.ill_conditioned_kappa_min):
                return True
        return False


def _cell_center_m(cell: CellKey, grid: WorkspaceGridSpec) -> np.ndarray:
    step = cell.level_mm / 1000.0
    return np.asarray(grid.origin_m, dtype=float) + step * np.asarray(
        (cell.ix + 0.5, cell.iy + 0.5, cell.iz + 0.5), dtype=float
    )


def _ordered_edge(left: CellKey, right: CellKey) -> tuple[CellKey, CellKey]:
    if left == right:
        raise ValueError("task graph cannot have self edges")
    return (left, right) if left < right else (right, left)


def _maximin_samples(
    samples: Sequence[RegisteredSample], count: int, *, space: str
) -> tuple[RegisteredSample, ...]:
    """Deterministically choose far-apart real samples from a non-empty pool."""

    candidates = tuple(sorted(samples, key=lambda item: item.sample_id))
    if len(candidates) < count:
        raise ValueError("not enough registered samples for requested measure probes")
    values = np.vstack([item.xyz_m if space == "xyz" else item.beta_rad for item in candidates])
    centroid = np.mean(values, axis=0)
    first = min(
        range(len(candidates)),
        key=lambda index: (
            -float(np.linalg.norm(values[index] - centroid)),
            candidates[index].sample_id,
        ),
    )
    chosen = [first]
    while len(chosen) < count:
        remaining = [index for index in range(len(candidates)) if index not in chosen]
        next_index = min(
            remaining,
            key=lambda index: (
                -float(np.min(np.linalg.norm(values[index] - values[chosen], axis=1))),
                candidates[index].sample_id,
            ),
        )
        chosen.append(next_index)
    return tuple(candidates[index] for index in chosen)


def _clustered_maximin_seed_samples(
    samples: Sequence[RegisteredSample],
    *,
    representative_sample_id: str,
    beta_bounds_rad: np.ndarray,
    cluster_radius: float,
    max_count: int,
) -> tuple[RegisteredSample, ...]:
    """Collapse near-identical normalized betas, then retain maximin seeds."""

    ordered = tuple(sorted(samples, key=lambda item: item.sample_id))
    beta = np.vstack([item.beta_rad for item in ordered])
    normalized = (beta - beta_bounds_rad[:, 0]) / (beta_bounds_rad[:, 1] - beta_bounds_rad[:, 0])
    # Connected components make clustering independent of accidental row order.
    unassigned = set(range(len(ordered)))
    clusters: list[list[int]] = []
    while unassigned:
        component = {min(unassigned)}
        frontier = list(component)
        unassigned -= component
        while frontier:
            current = frontier.pop()
            adjacent = {
                index
                for index in tuple(unassigned)
                if float(np.linalg.norm(normalized[index] - normalized[current])) <= cluster_radius
            }
            frontier.extend(sorted(adjacent))
            component |= adjacent
            unassigned -= adjacent
        clusters.append(sorted(component))
    representatives: list[int] = []
    for component in clusters:
        centroid = np.mean(normalized[component], axis=0)
        representatives.append(
            min(
                component,
                key=lambda index: (
                    float(np.linalg.norm(normalized[index] - centroid)),
                    ordered[index].sample_id,
                ),
            )
        )
    representatives.sort(key=lambda index: ordered[index].sample_id)
    start = next(
        (
            index
            for index in representatives
            if ordered[index].sample_id == representative_sample_id
        ),
        min(
            representatives,
            key=lambda index: (
                float(np.linalg.norm(normalized[index] - 0.5)),
                ordered[index].sample_id,
            ),
        ),
    )
    chosen = [start]
    while len(chosen) < min(max_count, len(representatives)):
        remaining = [index for index in representatives if index not in chosen]
        chosen.append(
            min(
                remaining,
                key=lambda index: (
                    -float(np.min(np.linalg.norm(normalized[index] - normalized[chosen], axis=1))),
                    ordered[index].sample_id,
                ),
            )
        )
    return tuple(ordered[index] for index in chosen)
