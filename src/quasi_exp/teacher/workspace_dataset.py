"""Leakage-resistant fixed-budget supervision materialization for BACRA V14."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
import hashlib
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .workspace_atlas import RepresentationMode
from .workspace_reach import CellKey


class BudgetExceededError(RuntimeError):
    """Raised when required supervision cannot fit below the hard cap."""


class BudgetFeasibilityStatus(str, Enum):
    FITS_BASE_RESOLUTION = "fits_base_resolution"
    FITS_AFTER_INTERIOR_COARSENING = "fits_after_interior_coarsening"
    INFEASIBLE_FIXED_BUDGET = "infeasible_fixed_budget"


@dataclass(frozen=True)
class FixedBudgetDemand:
    """Minimum unique supervised rows; Teacher-only artifacts are excluded."""

    labelable_base_rows: int
    refinement_extra_rows: int
    chart_expert_extra_rows: int
    retention_rows: int
    active_rows: int
    coarsened_interior_rows: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.labelable_base_rows,
            self.refinement_extra_rows,
            self.chart_expert_extra_rows,
            self.retention_rows,
            self.active_rows,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError("fixed-budget demand counts must be non-negative")
        for name, value in zip(
            (
                "labelable_base_rows",
                "refinement_extra_rows",
                "chart_expert_extra_rows",
                "retention_rows",
                "active_rows",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, int(value))
        if self.coarsened_interior_rows is not None:
            coarsened = int(self.coarsened_interior_rows)
            if coarsened < 0 or coarsened > self.labelable_base_rows:
                raise ValueError(
                    "coarsened_interior_rows must be between zero and labelable_base_rows"
                )
            object.__setattr__(self, "coarsened_interior_rows", coarsened)

    @property
    def minimum_supervised_rows(self) -> int:
        return int(
            self.labelable_base_rows
            + self.refinement_extra_rows
            + self.chart_expert_extra_rows
            + self.retention_rows
            + self.active_rows
        )

    @property
    def coarsened_minimum_rows(self) -> int | None:
        if self.coarsened_interior_rows is None:
            return None
        return int(
            self.coarsened_interior_rows
            + self.refinement_extra_rows
            + self.chart_expert_extra_rows
            + self.retention_rows
            + self.active_rows
        )


@dataclass(frozen=True)
class BudgetFeasibilityReport:
    status: BudgetFeasibilityStatus
    minimum_supervised_rows: int
    coarsened_minimum_rows: int | None
    total_budget: int
    shortfall_rows: int
    gate_pass: bool


def evaluate_fixed_budget_feasibility(
    demand: FixedBudgetDemand, *, total_budget: int
) -> BudgetFeasibilityReport:
    """Apply the pre-materialization Gate without deleting workspace cells."""

    budget = int(total_budget)
    if budget < 1:
        raise ValueError("total_budget must be positive")
    minimum = demand.minimum_supervised_rows
    coarsened = demand.coarsened_minimum_rows
    if minimum <= budget:
        status = BudgetFeasibilityStatus.FITS_BASE_RESOLUTION
        chosen = minimum
        passed = True
    elif coarsened is not None and coarsened <= budget:
        status = BudgetFeasibilityStatus.FITS_AFTER_INTERIOR_COARSENING
        chosen = coarsened
        passed = True
    else:
        status = BudgetFeasibilityStatus.INFEASIBLE_FIXED_BUDGET
        chosen = minimum if coarsened is None else min(minimum, coarsened)
        passed = False
    return BudgetFeasibilityReport(
        status=status,
        minimum_supervised_rows=minimum,
        coarsened_minimum_rows=coarsened,
        total_budget=budget,
        shortfall_rows=max(0, chosen - budget),
        gate_pass=passed,
    )


class SplitRole(str, Enum):
    TRAIN_CORE = "train_core"
    ACTIVE_PROBE = "active_probe"
    VALIDATION = "validation"
    SEALED = "sealed"


class SupervisionKind(str, Enum):
    STATIC = "static"
    STATEFUL = "stateful"


class SupervisionPriority(IntEnum):
    BASE_COVERAGE = 0
    RETENTION = 1
    CHART_EXPERT = 2
    REFINEMENT = 3
    ACTIVE = 4
    MAXIMIN = 5


@dataclass(frozen=True)
class MacroblockAssignment:
    macroblock_id: str
    split_role: SplitRole
    ix: int
    iy: int
    iz: int


@dataclass(frozen=True)
class MacroblockSplitPolicy:
    macroblock_mm: int = 40
    seed: int = 20260860
    train_core_fraction: float = 0.625
    active_probe_fraction: float = 0.075
    validation_fraction: float = 0.15
    sealed_fraction: float = 0.15

    def __post_init__(self) -> None:
        if int(self.macroblock_mm) <= 0:
            raise ValueError("macroblock_mm must be positive")
        fractions = np.asarray(
            [
                self.train_core_fraction,
                self.active_probe_fraction,
                self.validation_fraction,
                self.sealed_fraction,
            ],
            dtype=float,
        )
        if not np.isfinite(fractions).all() or np.any(fractions < 0.0):
            raise ValueError("split fractions must be finite and non-negative")
        if not np.isclose(float(fractions.sum()), 1.0, atol=1.0e-12):
            raise ValueError("split fractions must sum to one")
        object.__setattr__(self, "macroblock_mm", int(self.macroblock_mm))
        object.__setattr__(self, "seed", int(self.seed))

    def assignment_for_cell(self, cell: CellKey) -> MacroblockAssignment:
        macro = int(self.macroblock_mm)
        indices = tuple(
            math.floor(index * cell.level_mm / macro)
            for index in (cell.ix, cell.iy, cell.iz)
        )
        macroblock_id = f"m{macro}_x{indices[0]}_y{indices[1]}_z{indices[2]}"
        digest = hashlib.sha256(
            f"{self.seed}:{indices[0]}:{indices[1]}:{indices[2]}".encode("utf-8")
        ).digest()
        value = int.from_bytes(digest[:8], "big") / float(1 << 64)
        train_limit = self.train_core_fraction
        active_limit = train_limit + self.active_probe_fraction
        validation_limit = active_limit + self.validation_fraction
        if value < train_limit:
            role = SplitRole.TRAIN_CORE
        elif value < active_limit:
            role = SplitRole.ACTIVE_PROBE
        elif value < validation_limit:
            role = SplitRole.VALIDATION
        else:
            role = SplitRole.SEALED
        return MacroblockAssignment(
            macroblock_id=macroblock_id,
            split_role=role,
            ix=indices[0],
            iy=indices[1],
            iz=indices[2],
        )


@dataclass(frozen=True)
class SupervisionRecord:
    record_id: str
    kind: SupervisionKind
    physical_point_id: str
    chart_id: str
    cell: CellKey
    xyz_m: np.ndarray
    beta_rad: np.ndarray
    is_primary: bool
    required: bool
    priority: SupervisionPriority
    source_family: str
    previous_beta_rad: np.ndarray | None = None
    source_probe_id: str | None = None
    target_probe_id: str | None = None
    branch_id: str | None = None
    sample_weight: float = 1.0
    quality_class: str = "Gold"
    residual_mm: float = 0.0
    actual_bounds: bool = True
    macroblock_id: str | None = None
    split_role: SplitRole | None = None

    def __post_init__(self) -> None:
        record_id = str(self.record_id).strip()
        point_id = str(self.physical_point_id).strip()
        chart_id = str(self.chart_id).strip()
        source_family = str(self.source_family).strip()
        kind = SupervisionKind(self.kind)
        priority = SupervisionPriority(self.priority)
        xyz = np.asarray(self.xyz_m, dtype=float).reshape(3)
        beta = np.asarray(self.beta_rad, dtype=float).reshape(6)
        previous = (
            None
            if self.previous_beta_rad is None
            else np.asarray(self.previous_beta_rad, dtype=float).reshape(6)
        )
        if not record_id or not point_id or not chart_id or not source_family:
            raise ValueError("supervision record IDs and source_family must be non-empty")
        if not np.isfinite(xyz).all() or not np.isfinite(beta).all():
            raise ValueError("supervision xyz and beta must be finite")
        if previous is not None and not np.isfinite(previous).all():
            raise ValueError("previous_beta_rad must be finite")
        if not math.isfinite(float(self.sample_weight)) or self.sample_weight <= 0.0:
            raise ValueError("sample_weight must be finite and positive")
        if not math.isfinite(float(self.residual_mm)) or self.residual_mm < 0.0:
            raise ValueError("residual_mm must be finite and non-negative")
        if kind is SupervisionKind.STATIC:
            if any(
                value is not None
                for value in (self.source_probe_id, self.target_probe_id, self.branch_id)
            ):
                raise ValueError("static supervision cannot carry transition identity")
            if previous is not None:
                raise ValueError("static supervision cannot carry previous_beta_rad")
        else:
            if not self.source_probe_id or not self.target_probe_id or not self.branch_id:
                raise ValueError("stateful supervision requires directed transition identity")
            if self.source_probe_id == self.target_probe_id:
                raise ValueError("stateful transition cannot be a self edge")
            if previous is None:
                raise ValueError("stateful supervision requires previous_beta_rad")
            if self.is_primary:
                raise ValueError("stateful transitions are not primary point labels")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "physical_point_id", point_id)
        object.__setattr__(self, "chart_id", chart_id)
        object.__setattr__(self, "source_family", source_family)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "xyz_m", xyz.copy())
        object.__setattr__(self, "beta_rad", beta.copy())
        object.__setattr__(self, "previous_beta_rad", None if previous is None else previous.copy())
        object.__setattr__(self, "is_primary", bool(self.is_primary))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "actual_bounds", bool(self.actual_bounds))
        if self.split_role is not None:
            object.__setattr__(self, "split_role", SplitRole(self.split_role))

    @property
    def supervision_key(self) -> tuple[str, ...]:
        if self.kind is SupervisionKind.STATIC:
            return (self.kind.value, self.physical_point_id, self.chart_id)
        return (
            self.kind.value,
            str(self.source_probe_id),
            str(self.target_probe_id),
            str(self.branch_id),
        )


@dataclass(frozen=True)
class SupervisionBudgetPolicy:
    target_total: int = 200_000
    hard_max: int = 300_000
    active_reserve: int = 15_000

    def __post_init__(self) -> None:
        target = int(self.target_total)
        hard = int(self.hard_max)
        active = int(self.active_reserve)
        if target < 1 or hard < target or active < 0 or active >= hard:
            raise ValueError("supervision budget must satisfy 0 <= active < target <= hard")
        object.__setattr__(self, "target_total", target)
        object.__setattr__(self, "hard_max", hard)
        object.__setattr__(self, "active_reserve", active)

    @property
    def preactive_target(self) -> int:
        return max(0, self.target_total - self.active_reserve)

    @property
    def preactive_hard_max(self) -> int:
        return self.hard_max - self.active_reserve


@dataclass(frozen=True)
class SupervisionBudgetReport:
    candidate_count: int
    deduplicated_candidate_count: int
    required_count: int
    preactive_count: int
    active_count: int
    unique_supervision_count: int
    target_total: int
    hard_max: int
    padding_count: int
    gate_pass: bool


@dataclass(frozen=True)
class DatasetBundle:
    supervision_records: tuple[SupervisionRecord, ...]
    primary_canonical: tuple[SupervisionRecord, ...]
    chart_expert: tuple[SupervisionRecord, ...]
    stateful_transition: tuple[SupervisionRecord, ...]
    supervision_index: pd.DataFrame
    budget_report: SupervisionBudgetReport


def _same_label(left: SupervisionRecord, right: SupervisionRecord) -> bool:
    return bool(
        np.allclose(left.xyz_m, right.xyz_m, rtol=0.0, atol=1.0e-12)
        and np.allclose(left.beta_rad, right.beta_rad, rtol=0.0, atol=1.0e-12)
        and (
            left.previous_beta_rad is None
            and right.previous_beta_rad is None
            or left.previous_beta_rad is not None
            and right.previous_beta_rad is not None
            and np.allclose(
                left.previous_beta_rad,
                right.previous_beta_rad,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    )


class SupervisionMaterializer:
    """Select and split unique supervision without padding or hidden expansion."""

    def __init__(
        self,
        budget_policy: SupervisionBudgetPolicy | None = None,
        split_policy: MacroblockSplitPolicy | None = None,
    ) -> None:
        self.budget_policy = (
            SupervisionBudgetPolicy() if budget_policy is None else budget_policy
        )
        self.split_policy = MacroblockSplitPolicy() if split_policy is None else split_policy

    def materialize(
        self,
        candidates: Sequence[SupervisionRecord],
        *,
        representation_mode: RepresentationMode,
        active_record_ids: Sequence[str] = (),
    ) -> DatasetBundle:
        mode = RepresentationMode(representation_mode)
        if mode is RepresentationMode.BLOCKED:
            raise ValueError("cannot materialize supervision for a blocked representation")
        rows = tuple(candidates)
        if mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
            eligible = [row for row in rows if row.kind is SupervisionKind.STATEFUL]
        elif mode is RepresentationMode.XYZ_GLOBAL:
            eligible = [
                row
                for row in rows
                if row.kind is SupervisionKind.STATIC and row.is_primary
            ]
        else:
            eligible = [row for row in rows if row.kind is SupervisionKind.STATIC]

        deduplicated = self._deduplicate(eligible)
        active_ids = {str(value) for value in active_record_ids}
        known_ids = {row.record_id for row in deduplicated}
        unknown_active = active_ids - known_ids
        if unknown_active:
            raise ValueError(f"unknown active supervision IDs: {sorted(unknown_active)}")

        assigned = tuple(self._assign_split(row) for row in deduplicated)
        active_rows = [row for row in assigned if row.record_id in active_ids]
        if len(active_rows) > self.budget_policy.active_reserve:
            raise BudgetExceededError("selected active supervision exceeds active reserve")
        if any(row.split_role is not SplitRole.ACTIVE_PROBE for row in active_rows):
            raise ValueError("active supervision must come from active_probe macroblocks")

        # ACTIVE_PROBE macroblocks are a sealed acquisition pool.  Their rows
        # cannot leak into the pre-active Student merely because they were not
        # selected in this round.
        initial_pool = [
            row
            for row in assigned
            if row.record_id not in active_ids
            and row.split_role is not SplitRole.ACTIVE_PROBE
        ]
        required = sorted(
            (row for row in initial_pool if row.required), key=self._priority_key
        )
        if len(required) > self.budget_policy.preactive_hard_max:
            raise BudgetExceededError(
                "required supervision cannot fit below the hard cap while preserving active reserve"
            )
        optional = sorted(
            (row for row in initial_pool if not row.required), key=self._priority_key
        )
        initial_target = max(self.budget_policy.preactive_target, len(required))
        initial_target = min(initial_target, self.budget_policy.preactive_hard_max)
        chosen_initial = required + optional[: max(0, initial_target - len(required))]
        chosen = chosen_initial + sorted(active_rows, key=self._priority_key)
        if len(chosen) > self.budget_policy.hard_max:
            raise BudgetExceededError("unique supervision exceeds hard_max")
        chosen = sorted(chosen, key=self._priority_key)

        primary = tuple(row for row in chosen if row.kind is SupervisionKind.STATIC and row.is_primary)
        primary_ids = [row.physical_point_id for row in primary]
        if len(set(primary_ids)) != len(primary_ids):
            raise ValueError("primary canonical dataset must be single valued per physical point")
        expert = tuple(row for row in chosen if row.kind is SupervisionKind.STATIC)
        stateful = tuple(row for row in chosen if row.kind is SupervisionKind.STATEFUL)
        index = self._index(chosen)
        report = SupervisionBudgetReport(
            candidate_count=len(rows),
            deduplicated_candidate_count=len(deduplicated),
            required_count=len(required),
            preactive_count=len(chosen_initial),
            active_count=len(active_rows),
            unique_supervision_count=len(chosen),
            target_total=self.budget_policy.target_total,
            hard_max=self.budget_policy.hard_max,
            padding_count=0,
            gate_pass=len(chosen) <= self.budget_policy.hard_max,
        )
        return DatasetBundle(
            supervision_records=tuple(chosen),
            primary_canonical=primary,
            chart_expert=expert,
            stateful_transition=stateful,
            supervision_index=index,
            budget_report=report,
        )

    def _deduplicate(
        self, rows: Sequence[SupervisionRecord]
    ) -> tuple[SupervisionRecord, ...]:
        by_key: dict[tuple[str, ...], SupervisionRecord] = {}
        for row in sorted(rows, key=lambda item: item.record_id):
            previous = by_key.get(row.supervision_key)
            if previous is None:
                by_key[row.supervision_key] = row
                continue
            if not _same_label(previous, row):
                raise ValueError(
                    f"conflicting supervision for key {row.supervision_key}"
                )
            by_key[row.supervision_key] = replace(
                previous,
                is_primary=previous.is_primary or row.is_primary,
                required=previous.required or row.required,
                priority=min(previous.priority, row.priority),
                sample_weight=max(previous.sample_weight, row.sample_weight),
            )
        return tuple(by_key[key] for key in sorted(by_key))

    def _assign_split(self, row: SupervisionRecord) -> SupervisionRecord:
        assignment = self.split_policy.assignment_for_cell(row.cell)
        if row.macroblock_id is not None and row.macroblock_id != assignment.macroblock_id:
            raise ValueError("preassigned macroblock_id disagrees with split policy")
        if row.split_role is not None and row.split_role is not assignment.split_role:
            raise ValueError("preassigned split_role disagrees with split policy")
        return replace(
            row,
            macroblock_id=assignment.macroblock_id,
            split_role=assignment.split_role,
        )

    @staticmethod
    def _priority_key(row: SupervisionRecord) -> tuple[int, str]:
        return (int(row.priority), row.record_id)

    @staticmethod
    def _index(rows: Sequence[SupervisionRecord]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "record_id": row.record_id,
                    "supervision_key": "|".join(row.supervision_key),
                    "kind": row.kind.value,
                    "physical_point_id": row.physical_point_id,
                    "chart_id": row.chart_id,
                    "macroblock_id": row.macroblock_id,
                    "split_role": None if row.split_role is None else row.split_role.value,
                    "priority": row.priority.name.lower(),
                    "required": row.required,
                    "is_primary": row.is_primary,
                    "source_family": row.source_family,
                }
                for row in rows
            ]
        )
