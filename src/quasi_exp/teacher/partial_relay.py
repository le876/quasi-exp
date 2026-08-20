"""Partial canonical sections and same-lineage spatial relays for V14 retry8.

The module keeps four decisions independent:

* root identity (branch origin, spatial relay, alternative branch),
* whether a partial canonical section is scientifically valid,
* whether there are enough labels to execute an exploratory Student, and
* whether broad-coverage or deployment claims are authorized.

It contains no solver or experiment scheduling code.  Runners consume these
small deterministic contracts and persist their decisions as explicit fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Sequence

import numpy as np
import pandas as pd


class RootKind(str, Enum):
    BRANCH_ORIGIN = "branch_origin"
    SPATIAL_RELAY = "spatial_relay"
    ALTERNATIVE_BRANCH = "alternative_branch"


@dataclass(frozen=True)
class RootSpec:
    root_id: str
    root_kind: RootKind
    task_node_id: int
    candidate_id: str
    canonical_lineage_id: str
    parent_root_id: str | None
    inherited_beta_rad: np.ndarray
    xyz_m: np.ndarray
    inheritance_gap_deg: float

    def __post_init__(self) -> None:
        beta = np.asarray(self.inherited_beta_rad, dtype=float)
        xyz = np.asarray(self.xyz_m, dtype=float)
        if beta.shape != (6,) or not np.isfinite(beta).all():
            raise ValueError("root beta must contain six finite radians")
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("root xyz must contain three finite coordinates")
        if not self.root_id or not self.candidate_id or not self.canonical_lineage_id:
            raise ValueError("root identity fields cannot be empty")
        if int(self.task_node_id) < 0:
            raise ValueError("root task node ID cannot be negative")
        if not math.isfinite(float(self.inheritance_gap_deg)):
            raise ValueError("root inheritance gap must be finite")
        if self.root_kind is RootKind.BRANCH_ORIGIN:
            if self.parent_root_id is not None:
                raise ValueError("branch origin cannot have a parent root")
        elif self.root_kind is RootKind.SPATIAL_RELAY:
            if not self.parent_root_id:
                raise ValueError("spatial relay must name its branch origin")
            if float(self.inheritance_gap_deg) > 0.1 + 1.0e-12:
                raise ValueError("spatial relay must inherit the frozen lineage")
        else:
            if self.parent_root_id is not None:
                raise ValueError(
                    "alternative branch cannot masquerade as a same-lineage relay"
                )
        object.__setattr__(self, "root_kind", RootKind(self.root_kind))
        object.__setattr__(self, "task_node_id", int(self.task_node_id))
        object.__setattr__(self, "inherited_beta_rad", beta.copy())
        object.__setattr__(self, "xyz_m", xyz.copy())
        object.__setattr__(self, "inheritance_gap_deg", float(self.inheritance_gap_deg))


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    is_diagnostic_only: bool
    is_selectable: bool
    relay_root_count: int
    maximum_growth_waves: int

    @classmethod
    def raw_retry7_diagnostic(cls) -> "MethodSpec":
        return cls("D0_raw_retry7", True, False, 0, 16)

    @classmethod
    def frozen_partial_singleton(cls) -> "MethodSpec":
        return cls("P0_frozen_partial_singleton", False, True, 0, 16)


@dataclass(frozen=True)
class ExploratoryAuthorization:
    data_legality_gate: bool
    relay_method_validation_pass: bool
    relay_method_validation_status: str
    global_coverage_gate_pass: bool
    smoke_execution_authorized: bool
    five_k_teacher_execution_authorized: bool
    student_claim_authorized: bool
    formal_or_deployment_authorized: bool
    teacher_label_integrity_failure: bool


@dataclass(frozen=True)
class MacroblockSplit:
    block_size_mm: int
    frame: pd.DataFrame
    block_counts: dict[str, int]
    row_counts: dict[str, int]


def _beta_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    alternatives = (
        tuple(f"beta{index}_rad" for index in range(1, 7)),
        tuple(f"beta_{index}" for index in range(6)),
    )
    for columns in alternatives:
        if set(columns).issubset(frame.columns):
            return columns
    raise ValueError("frozen lineage labels require six beta columns")


def select_spatial_relay_roots(
    labels: pd.DataFrame,
    *,
    origin_node_id: int,
    canonical_lineage_id: str,
    target_count: int,
    preferred_separation_mm: float,
    minimum_separation_mm: float,
) -> tuple[RootSpec, ...]:
    """Choose deterministic task-space maximin roots from one frozen lineage.

    Preferred separation is attempted first.  If it yields fewer roots than
    requested, the registered minimum separation is used.  The function never
    invents an IK solution: every beta is copied exactly from ``labels``.
    """

    required = {"task_node_id", "candidate_id", "x_m", "y_m", "z_m"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"frozen lineage labels missing columns: {sorted(missing)}")
    if target_count < 1:
        raise ValueError("target root count must be positive")
    if not 0.0 < minimum_separation_mm <= preferred_separation_mm:
        raise ValueError("relay separations must be positive and ordered")
    beta_columns = _beta_columns(labels)
    ordered = labels.sort_values("task_node_id", kind="stable").drop_duplicates(
        "task_node_id", keep="first"
    )
    origin_rows = ordered[ordered["task_node_id"].astype(int).eq(int(origin_node_id))]
    if len(origin_rows) != 1:
        raise ValueError("frozen lineage must contain exactly one branch origin row")
    origin_row = origin_rows.iloc[0]

    def spec(row: pd.Series, *, origin: bool) -> RootSpec:
        node_id = int(row["task_node_id"])
        return RootSpec(
            root_id="origin" if origin else f"relay_{node_id}",
            root_kind=(RootKind.BRANCH_ORIGIN if origin else RootKind.SPATIAL_RELAY),
            task_node_id=node_id,
            candidate_id=str(row["candidate_id"]),
            canonical_lineage_id=str(canonical_lineage_id),
            parent_root_id=None if origin else "origin",
            inherited_beta_rad=row.loc[list(beta_columns)].to_numpy(dtype=float),
            xyz_m=row.loc[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
            inheritance_gap_deg=0.0,
        )

    origin_spec = spec(origin_row, origin=True)
    candidates = [
        row
        for _, row in ordered.iterrows()
        if int(row["task_node_id"]) != int(origin_node_id)
    ]

    def fill(selected: list[pd.Series], separation_mm: float) -> None:
        threshold_m = float(separation_mm) / 1000.0
        while len(selected) < int(target_count):
            selected_xyz = [origin_spec.xyz_m] + [
                row.loc[["x_m", "y_m", "z_m"]].to_numpy(float)
                for row in selected
            ]
            eligible: list[tuple[float, int, pd.Series]] = []
            selected_ids = {int(row["task_node_id"]) for row in selected}
            for row in candidates:
                node_id = int(row["task_node_id"])
                if node_id in selected_ids:
                    continue
                xyz = row.loc[["x_m", "y_m", "z_m"]].to_numpy(float)
                distance = min(float(np.linalg.norm(xyz - other)) for other in selected_xyz)
                if distance + 1.0e-12 >= threshold_m:
                    eligible.append((distance, node_id, row))
            if not eligible:
                return
            _distance, _node, chosen = max(
                eligible, key=lambda item: (item[0], -item[1])
            )
            selected.append(chosen)

    selected_rows: list[pd.Series] = []
    fill(selected_rows, preferred_separation_mm)
    fill(selected_rows, minimum_separation_mm)
    roots = (origin_spec, *(spec(row, origin=False) for row in selected_rows))
    node_ids = [root.task_node_id for root in roots]
    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError("relay registry contains duplicate task nodes")
    return tuple(roots)


def authorize_exploratory_stages(
    *,
    partial_certificate_pass: bool,
    strict_unique_label_count: int,
    relay_root_count: int,
    relay_coverage_gain: float,
    relay_stitch_pass: bool,
    coverage_ratio: float,
    largest_component_ratio: float,
    smoke_pipeline_complete: bool,
    teacher_label_integrity_failure: bool,
    unresolved_training_implementation_failure: bool,
    smoke_student_quality_pass: bool,
) -> ExploratoryAuthorization:
    data_legal = bool(partial_certificate_pass)
    smoke = bool(data_legal and int(strict_unique_label_count) >= 3000)
    if int(relay_root_count) == 0:
        relay_status = "not_evaluated"
        relay_pass = False
    elif int(relay_root_count) == 1:
        relay_status = "availability_limited"
        relay_pass = False
    else:
        relay_pass = bool(float(relay_coverage_gain) > 0.0 and relay_stitch_pass)
        relay_status = "pass" if relay_pass else "fail"
    global_coverage = bool(
        float(coverage_ratio) >= 0.30
        and float(largest_component_ratio) >= 0.20
    )
    five_k = bool(
        smoke
        and smoke_pipeline_complete
        and not teacher_label_integrity_failure
        and not unresolved_training_implementation_failure
    )
    student_claim = bool(
        data_legal
        and smoke_pipeline_complete
        and not teacher_label_integrity_failure
        and not unresolved_training_implementation_failure
        and smoke_student_quality_pass
    )
    return ExploratoryAuthorization(
        data_legality_gate=data_legal,
        relay_method_validation_pass=relay_pass,
        relay_method_validation_status=relay_status,
        global_coverage_gate_pass=global_coverage,
        smoke_execution_authorized=smoke,
        five_k_teacher_execution_authorized=five_k,
        student_claim_authorized=student_claim,
        formal_or_deployment_authorized=False,
        teacher_label_integrity_failure=bool(teacher_label_integrity_failure),
    )


def sample_screening_schedules(
    schedules: pd.DataFrame,
    *,
    relay_node_ids: Sequence[int],
    boundary_node_ids: Sequence[int],
    maximum_interior_edges: int,
    maximum_cycles: int,
) -> pd.DataFrame:
    """Build the registered Level-B audit without certifying a primary atlas."""

    required = {"schedule_id", "audit_kind", "path_node_ids"}
    if missing := required - set(schedules.columns):
        raise ValueError(f"schedule frame missing columns: {sorted(missing)}")
    relay_nodes = set(map(int, relay_node_ids))
    boundary_nodes = set(map(int, boundary_node_ids))
    rows: list[dict[str, object]] = []
    interiors: list[dict[str, object]] = []
    cycles: list[dict[str, object]] = []
    for record in schedules.to_dict(orient="records"):
        kind = str(record["audit_kind"])
        path = set(map(int, record["path_node_ids"]))
        output = dict(record)
        if kind == "edge" and path & relay_nodes:
            output["screen_reason"] = "relay_transition"
            rows.append(output)
        elif kind == "edge" and path & boundary_nodes:
            output["screen_reason"] = "new_coverage_boundary"
            rows.append(output)
        elif kind == "edge":
            output["screen_reason"] = "stratified_interior"
            interiors.append(output)
        elif kind == "fundamental_cycle":
            output["screen_reason"] = "cycle_sample"
            cycles.append(output)

    def stable_sample(
        values: list[dict[str, object]], maximum: int
    ) -> list[dict[str, object]]:
        return sorted(
            values,
            key=lambda row: (
                hashlib.sha256(str(row["schedule_id"]).encode()).digest(),
                str(row["schedule_id"]),
            ),
        )[: max(0, int(maximum))]

    rows.extend(stable_sample(interiors, maximum_interior_edges))
    rows.extend(stable_sample(cycles, maximum_cycles))
    if not rows:
        return pd.DataFrame(columns=(*schedules.columns, "screen_reason"))
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values(
        ["audit_kind", "schedule_id"], kind="stable"
    ).reset_index(drop=True)


def choose_macroblock_split(
    frame: pd.DataFrame,
    *,
    block_sizes_mm: Sequence[int],
    split_seed: int,
    split_fractions: Sequence[float],
    minimum_train_blocks: int,
    minimum_validation_blocks: int,
    minimum_test_blocks: int,
    minimum_rows_per_split: int,
) -> MacroblockSplit:
    """Select the coarsest registered spatial split that meets row/block minima."""

    required = {"x_m", "y_m", "z_m"}
    if missing := required - set(frame.columns):
        raise ValueError(f"split frame missing columns: {sorted(missing)}")
    fractions = np.asarray(tuple(split_fractions), dtype=float)
    if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("split fractions must contain three positive values summing to one")
    roles = ("train_core", "validation", "test")
    minima = (minimum_train_blocks, minimum_validation_blocks, minimum_test_blocks)
    for block_size_mm in block_sizes_mm:
        block_size_m = int(block_size_mm) / 1000.0
        candidate = frame.copy()
        coordinates = np.floor(
            candidate.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(float)
            / block_size_m
        ).astype(int)
        candidate["macroblock"] = [str(tuple(row)) for row in coordinates]
        blocks = sorted(candidate["macroblock"].unique())
        if len(blocks) < sum(minima):
            continue
        ranked = sorted(
            blocks,
            key=lambda block: (
                hashlib.sha256(f"{int(split_seed)}:{block}".encode()).digest(),
                block,
            ),
        )
        counts = [max(int(minima[index]), int(round(fractions[index] * len(ranked)))) for index in range(3)]
        while sum(counts) > len(ranked):
            reducible = [index for index in range(3) if counts[index] > int(minima[index])]
            if not reducible:
                break
            index = max(reducible, key=lambda item: (counts[item] - minima[item], counts[item], -item))
            counts[index] -= 1
        while sum(counts) < len(ranked):
            index = max(range(3), key=lambda item: (fractions[item] - counts[item] / len(ranked), -item))
            counts[index] += 1
        if sum(counts) != len(ranked):
            continue
        mapping: dict[str, str] = {}
        offset = 0
        for role, count in zip(roles, counts, strict=True):
            mapping.update({block: role for block in ranked[offset : offset + count]})
            offset += count
        candidate["split_role"] = candidate["macroblock"].map(mapping)
        row_counts = {
            role: int(candidate["split_role"].eq(role).sum()) for role in roles
        }
        block_counts = {
            role: int(candidate.loc[candidate["split_role"].eq(role), "macroblock"].nunique())
            for role in roles
        }
        if any(block_counts[role] < minimum for role, minimum in zip(roles, minima, strict=True)):
            continue
        if any(row_counts[role] < int(minimum_rows_per_split) for role in roles):
            continue
        return MacroblockSplit(
            block_size_mm=int(block_size_mm),
            frame=candidate,
            block_counts=block_counts,
            row_counts=row_counts,
        )
    raise ValueError("no registered macroblock size satisfies spatial split minima")
