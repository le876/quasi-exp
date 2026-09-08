"""Deterministic retry8 Smoke and repaired-5k progression contracts."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("x_m", "y_m", "z_m")


def _maximin_ids(
    frame: pd.DataFrame,
    *,
    id_column: str,
    count: int,
    initial_ids: Iterable[int] = (),
) -> list[int]:
    ordered = frame.sort_values(id_column, kind="stable").drop_duplicates(id_column)
    xyz_by_id = {
        int(row[id_column]): row.loc[list(XYZ_COLUMNS)].to_numpy(float)
        for _, row in ordered.iterrows()
    }
    selected = [int(value) for value in initial_ids if int(value) in xyz_by_id]
    selected = list(dict.fromkeys(selected))
    while len(selected) < min(int(count), len(xyz_by_id)):
        remaining = [value for value in xyz_by_id if value not in set(selected)]
        if not selected:
            chosen = min(remaining)
        else:
            chosen = max(
                remaining,
                key=lambda value: (
                    min(
                        float(np.linalg.norm(xyz_by_id[value] - xyz_by_id[other]))
                        for other in selected
                    ),
                    -value,
                ),
            )
        selected.append(chosen)
    return selected[: int(count)]


def select_balanced_smoke_domain(
    parents: pd.DataFrame,
    tasks: pd.DataFrame,
    frozen_labels: pd.DataFrame,
    *,
    hard_parent_ids: set[int],
    parent_count: int,
    hard_parent_target: int,
    hard_parent_maximum: int,
) -> tuple[int, ...]:
    """Keep the certified P0 support, a bounded hard subset, then easy cells."""

    required_parent = {"node_id", *XYZ_COLUMNS}
    if missing := required_parent - set(parents.columns):
        raise ValueError(f"parent registry missing columns: {sorted(missing)}")
    p0_ids = set(frozen_labels["source_parent_node_id"].astype(int))
    available = set(parents["node_id"].astype(int))
    if not p0_ids <= available:
        raise ValueError("frozen P0 parents are absent from the 5k proxy")
    if len(p0_ids) > int(hard_parent_maximum):
        raise ValueError("P0 support exceeds the registered hard-subset maximum")
    hard_candidates = parents[parents["node_id"].astype(int).isin(hard_parent_ids)]
    hard_count = min(
        int(hard_parent_maximum),
        max(int(hard_parent_target), len(p0_ids)),
    )
    selected = _maximin_ids(
        hard_candidates,
        id_column="node_id",
        count=hard_count,
        initial_ids=sorted(p0_ids),
    )
    needed = int(parent_count) - len(selected)
    if needed < 0:
        raise ValueError("registered Smoke domain is smaller than P0 support")
    task_safety = (
        tasks.groupby("source_parent_node_id")["core_safe"].mean()
        if "core_safe" in tasks.columns
        else pd.Series(dtype=float)
    )
    easy = parents[
        ~parents["node_id"].astype(int).isin(set(selected))
        & ~parents["node_id"].astype(int).isin(hard_parent_ids)
    ].copy()
    easy["_core_safe"] = easy["node_id"].map(task_safety).fillna(0.0)
    easy = easy.sort_values(
        ["_core_safe", "node_id"],
        ascending=[False, True],
        kind="stable",
    )
    if len(easy) < needed:
        hard_room = int(hard_parent_maximum) - len(selected)
        extra_hard = parents[
            ~parents["node_id"].astype(int).isin(set(selected))
            & parents["node_id"].astype(int).isin(hard_parent_ids)
        ].sort_values("node_id", kind="stable").head(max(0, hard_room))
        easy = pd.concat([easy, extra_hard], ignore_index=True)
    pool_size = max(needed, min(len(easy), needed * 3))
    easy_pool = easy.head(pool_size)
    easy_ids = _maximin_ids(easy_pool, id_column="node_id", count=needed)
    output = tuple(selected + easy_ids)
    if len(output) != int(parent_count) or len(set(output)) != len(output):
        raise RuntimeError("balanced Smoke parent selection is incomplete")
    hard_selected = len(set(output) & set(map(int, hard_parent_ids)))
    if not len(p0_ids) <= hard_selected <= int(hard_parent_maximum):
        raise RuntimeError("balanced Smoke hard subset violates registered bounds")
    return output


def map_frozen_lineage_roots(
    frozen_labels: pd.DataFrame,
    tasks: pd.DataFrame,
    *,
    root_count: int,
) -> pd.DataFrame:
    """Map exact frozen xyz labels into a new task graph and choose maximin roots."""

    required = {"candidate_id", "canonical_lineage_id", *XYZ_COLUMNS, *BETA_COLUMNS}
    if missing := required - set(frozen_labels.columns):
        raise ValueError(f"frozen lineage missing columns: {sorted(missing)}")
    task_lookup = {
        tuple(np.round(row.loc[list(XYZ_COLUMNS)].to_numpy(float), 12)): int(
            row["task_node_id"]
        )
        for _, row in tasks.iterrows()
    }
    rows = []
    for _, row in frozen_labels.iterrows():
        node_id = task_lookup.get(
            tuple(np.round(row.loc[list(XYZ_COLUMNS)].to_numpy(float), 12))
        )
        if node_id is None:
            continue
        rows.append({**row.to_dict(), "task_node_id": node_id})
    mapped = pd.DataFrame.from_records(rows)
    if mapped.empty:
        raise ValueError("no frozen lineage label maps into the registered task graph")
    if mapped["canonical_lineage_id"].astype(str).nunique() != 1:
        raise ValueError("root registry would mix canonical lineages")
    root_ids = _maximin_ids(
        mapped,
        id_column="task_node_id",
        count=min(int(root_count), len(mapped)),
    )
    selected = mapped.set_index("task_node_id").loc[root_ids].reset_index()
    selected.insert(0, "root_index", np.arange(len(selected), dtype=int))
    selected["root_kind"] = ["branch_origin", *(["spatial_relay"] * (len(selected) - 1))]
    selected["parent_root_id"] = [None, *(["root_000"] * (len(selected) - 1))]
    selected["root_id"] = selected["root_index"].map(lambda value: f"root_{int(value):03d}")
    selected["inherited_from_frozen_lineage"] = True
    return selected


def authorize_five_k_after_smoke(
    *,
    smoke_pipeline_operationally_complete: bool,
    teacher_label_integrity_failure: bool,
    unresolved_training_implementation_failure: bool,
    smoke_student_quality_pass: bool,
) -> dict[str, bool]:
    five_k = bool(
        smoke_pipeline_operationally_complete
        and not teacher_label_integrity_failure
        and not unresolved_training_implementation_failure
    )
    return {
        "five_k_teacher_execution_authorized": five_k,
        "student_claim_authorized": bool(five_k and smoke_student_quality_pass),
        "formal_or_deployment_authorized": False,
    }
