"""Deterministic sparse-wide sampling and replacement contracts for retry11.

The module deliberately owns only in-memory scientific decisions.  It never
reads an experiment artifact and never launches a numerical solver, which
keeps the Parent-first and replacement invariants directly testable.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .retry10 import BETA_COLUMNS, XYZ_COLUMNS, macroblock_ids


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _xyz_keys(frame: pd.DataFrame) -> pd.Series:
    return frame.loc[:, XYZ_COLUMNS].round(12).astype(str).agg("|".join, axis=1)


def stable_row_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(str(value).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def graph_shell_width_mm(
    task_nodes: pd.DataFrame,
    task_edges: pd.DataFrame,
    *,
    percentile: float = 95.0,
    multiplier: float = 2.0,
    minimum_mm: float = 10.0,
    maximum_mm: float = 30.0,
) -> tuple[float, float]:
    """Return ``(shell_width_mm, edge_length_p95_mm)`` from registered edges."""

    _require_columns(task_nodes, ("task_node_id", *XYZ_COLUMNS), "task nodes")
    _require_columns(task_edges, ("left_node_id", "right_node_id"), "task edges")
    indexed = (
        task_nodes.drop_duplicates("task_node_id")
        .set_index("task_node_id")
        .loc[:, XYZ_COLUMNS]
    )
    left = indexed.reindex(task_edges["left_node_id"].astype(int)).to_numpy(float)
    right = indexed.reindex(task_edges["right_node_id"].astype(int)).to_numpy(float)
    lengths = np.linalg.norm(left - right, axis=1) * 1000.0
    lengths = lengths[np.isfinite(lengths)]
    if not len(lengths):
        raise ValueError("registered task graph has no finite edge lengths")
    edge_p95 = float(np.percentile(lengths, float(percentile)))
    width = float(np.clip(float(multiplier) * edge_p95, minimum_mm, maximum_mm))
    return width, edge_p95


def annotate_spatial_strata(
    frame: pd.DataFrame,
    *,
    zero_xyz_m: Sequence[float],
    shell_width_mm: float,
) -> pd.DataFrame:
    """Attach zero-radius, radial shell, octant sector and boundary role."""

    _require_columns(frame, XYZ_COLUMNS, "spatial frame")
    width = float(shell_width_mm)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("shell_width_mm must be finite and positive")
    zero = np.asarray(zero_xyz_m, dtype=float).reshape(3)
    if not np.isfinite(zero).all():
        raise ValueError("zero_xyz_m must be finite shape (3,)")
    result = frame.copy()
    delta = result.loc[:, XYZ_COLUMNS].to_numpy(float) - zero.reshape(1, 3)
    radius = np.linalg.norm(delta, axis=1) * 1000.0
    shell = np.floor(radius / width).astype(np.int64)
    positive = delta >= 0.0
    sector = (
        positive[:, 0].astype(np.int64) * 4
        + positive[:, 1].astype(np.int64) * 2
        + positive[:, 2].astype(np.int64)
    )
    fraction = radius / width - shell
    result["zero_radius_mm"] = radius
    result["radial_shell_id"] = shell
    result["sector_id"] = sector
    result["shell_sector"] = [
        f"shell_{int(shell_id):03d}:sector_{int(sector_id)}"
        for shell_id, sector_id in zip(shell, sector, strict=True)
    ]
    result["boundary_role"] = np.where(
        (fraction <= 0.15) | (fraction >= 0.85), "boundary", "interior"
    )
    return result


def build_macroblock_split_registry(
    frames: Sequence[pd.DataFrame],
    panels: Mapping[str, pd.DataFrame],
    *,
    block_size_mm: int = 40,
    seed: int = 20260881,
) -> pd.DataFrame:
    """Freeze one no-leakage split role for every known 40 mm macroblock."""

    if not frames:
        raise ValueError("at least one dataset/candidate frame is required")
    known_blocks: set[str] = set()
    for frame in frames:
        known_blocks.update(macroblock_ids(frame, block_size_mm=block_size_mm).astype(str))
    panel_role_by_block: dict[str, str] = {}
    for panel_id, panel in sorted(panels.items()):
        role = f"panel_{str(panel_id)}"
        for block in macroblock_ids(panel, block_size_mm=block_size_mm).astype(str):
            previous = panel_role_by_block.get(block)
            if previous is not None and previous != role:
                raise ValueError(
                    f"frozen evaluation panels overlap at macroblock {block}: "
                    f"{previous} versus {role}"
                )
            panel_role_by_block[block] = role
            known_blocks.add(block)

    records: list[dict[str, object]] = []
    for block in sorted(known_blocks):
        if block in panel_role_by_block:
            role = panel_role_by_block[block]
        else:
            rank = int(
                hashlib.sha256(f"{int(seed)}:{block}".encode()).hexdigest(), 16
            ) / float(2**256)
            role = "train_core" if rank < 0.70 else "validation" if rank < 0.90 else "test_internal"
        records.append(
            {
                "macroblock_id": block,
                "split_role": role,
                "block_size_mm": int(block_size_mm),
                "split_seed": int(seed),
            }
        )
    return pd.DataFrame.from_records(records)


def apply_macroblock_split_registry(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    block_size_mm: int = 40,
) -> pd.DataFrame:
    _require_columns(registry, ("macroblock_id", "split_role"), "macroblock registry")
    if registry["macroblock_id"].astype(str).duplicated().any():
        raise ValueError("macroblock registry IDs must be unique")
    result = frame.copy()
    result["macroblock_id"] = macroblock_ids(result, block_size_mm=block_size_mm)
    roles = registry.set_index("macroblock_id")["split_role"]
    result["split_role"] = result["macroblock_id"].astype(str).map(roles)
    if result["split_role"].isna().any():
        missing = sorted(result.loc[result["split_role"].isna(), "macroblock_id"].astype(str).unique())
        raise ValueError(f"macroblock registry does not cover blocks: {missing[:8]}")
    return result


def _mandatory_row_hashes(frame: pd.DataFrame) -> pd.Series:
    columns = [name for name in (*XYZ_COLUMNS, *BETA_COLUMNS, "task_node_id", "source_parent_node_id", "label_quality") if name in frame]
    canonical = frame.loc[:, columns].astype(str).agg("|".join, axis=1)
    return canonical.map(lambda value: hashlib.sha256(value.encode()).hexdigest())


def select_parent_first_sparse_wide(
    mandatory_rows: pd.DataFrame,
    accepted_candidates: pd.DataFrame,
    *,
    target_rows: int = 20_000,
    minimum_rows: int = 18_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select mandatory rows plus fair parent rounds and local XYZ maximin.

    Parent round is the primary ordering.  At each parent's turn the least
    serviced shell-sector is selected using a denominator frozen as distinct
    eligible parents, then the point farthest from retained rows of that parent
    is chosen.  No padding is possible.
    """

    required = ("source_parent_node_id", "label_quality", "shell_sector", *XYZ_COLUMNS, *BETA_COLUMNS)
    _require_columns(mandatory_rows, required, "mandatory rows")
    _require_columns(accepted_candidates, required, "accepted candidates")
    target = int(target_rows)
    minimum = int(minimum_rows)
    if not 0 < minimum <= target:
        raise ValueError("minimum_rows and target_rows must be positive and ordered")
    if len(mandatory_rows) > target:
        raise ValueError("mandatory rows exceed retry11 target")

    mandatory = mandatory_rows.copy().reset_index(drop=True)
    candidates = accepted_candidates.copy().reset_index(drop=True)
    if "xyz_key" not in mandatory:
        mandatory["xyz_key"] = _xyz_keys(mandatory)
    if "xyz_key" not in candidates:
        candidates["xyz_key"] = _xyz_keys(candidates)
    if mandatory["xyz_key"].astype(str).duplicated().any():
        raise ValueError("mandatory rows contain duplicate xyz keys")
    mandatory_keys = set(mandatory["xyz_key"].astype(str))
    candidates = candidates[~candidates["xyz_key"].astype(str).isin(mandatory_keys)].copy()
    candidates = (
        candidates.sort_values(
            ["source_parent_node_id", "shell_sector", "xyz_key"], kind="stable"
        )
        .drop_duplicates("xyz_key", keep="first")
        .reset_index(drop=True)
    )
    if "physical_point_id" not in mandatory:
        mandatory["physical_point_id"] = mandatory["xyz_key"].astype(str).map(
            lambda value: stable_row_id("retry11_mandatory", value)
        )
    if "physical_point_id" not in candidates:
        candidates["physical_point_id"] = candidates["xyz_key"].astype(str).map(
            lambda value: stable_row_id("retry11_candidate", value)
        )
    if mandatory["physical_point_id"].astype(str).duplicated().any():
        raise ValueError("mandatory physical point IDs must be unique")
    if candidates["physical_point_id"].astype(str).duplicated().any():
        raise ValueError("candidate physical point IDs must be unique")
    candidate_index_by_id = {
        str(point_id): int(index)
        for index, point_id in candidates["physical_point_id"].items()
    }

    eligible_parents = tuple(sorted(candidates["source_parent_node_id"].astype(int).unique()))
    denominator = (
        candidates.groupby("shell_sector", sort=True)["source_parent_node_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    selected_by_stratum = {str(key): 0 for key in denominator}
    new_count = {int(parent): 0 for parent in eligible_parents}
    retained_xyz: dict[int, list[np.ndarray]] = {int(parent): [] for parent in eligible_parents}
    for row in mandatory.itertuples(index=False):
        parent = int(row.source_parent_node_id)
        retained_xyz.setdefault(parent, []).append(
            np.asarray([row.x_m, row.y_m, row.z_m], dtype=float)
        )
    queues = {
        int(parent): group.copy().reset_index(drop=True)
        for parent, group in candidates.groupby("source_parent_node_id", sort=True)
    }
    selected_indices: list[int] = []
    audit_rows: list[dict[str, object]] = []
    round_index = 1
    while len(mandatory) + len(selected_indices) < target:
        active = [parent for parent in eligible_parents if len(queues[parent])]
        if not active:
            break
        progressed = False
        for parent in active:
            if len(mandatory) + len(selected_indices) >= target:
                break
            queue = queues[parent]
            if queue.empty:
                continue
            strata = sorted(queue["shell_sector"].astype(str).unique())
            chosen_stratum = min(
                strata,
                key=lambda value: (
                    selected_by_stratum.get(value, 0) / max(1, int(denominator[value])),
                    value,
                ),
            )
            available = queue[queue["shell_sector"].astype(str).eq(chosen_stratum)]
            prior = retained_xyz.get(parent, [])
            choices: list[tuple[float, str, int]] = []
            for index, row in available.iterrows():
                xyz = row.loc[list(XYZ_COLUMNS)].to_numpy(float)
                distance = (
                    min(float(np.linalg.norm(xyz - other)) for other in prior)
                    if prior
                    else math.inf
                )
                choices.append((distance, str(row["xyz_key"]), int(index)))
            _distance, _key, chosen_index = min(
                choices,
                key=lambda item: (-item[0], item[1], item[2]),
            )
            chosen = queue.loc[chosen_index]
            global_index = int(chosen.name)
            # DataFrame group queues have local indices; recover the unique
            # physical point from the frozen candidate table.
            point_id = str(chosen["physical_point_id"])
            candidate_index = candidate_index_by_id[point_id]
            selected_indices.append(candidate_index)
            retained_xyz.setdefault(parent, []).append(chosen.loc[list(XYZ_COLUMNS)].to_numpy(float))
            new_count[parent] += 1
            selected_by_stratum[chosen_stratum] = selected_by_stratum.get(chosen_stratum, 0) + 1
            audit_rows.append(
                {
                    "selection_order": len(selected_indices),
                    "parent_round": int(round_index),
                    "source_parent_node_id": int(parent),
                    "shell_sector": chosen_stratum,
                    "relative_deficit_before": (
                        (selected_by_stratum[chosen_stratum] - 1)
                        / max(1, int(denominator[chosen_stratum]))
                    ),
                    "parent_local_min_distance_mm": (
                        None if not math.isfinite(_distance) else float(_distance * 1000.0)
                    ),
                    "physical_point_id": point_id,
                    "xyz_key": str(chosen["xyz_key"]),
                }
            )
            queues[parent] = queue.drop(index=chosen_index).reset_index(drop=True)
            progressed = True
        if not progressed:
            break
        round_index += 1

    selected_new = candidates.loc[selected_indices].copy().reset_index(drop=True)
    selected_new["selection_origin"] = "accepted_candidate_pool"
    selected_new["new_parent_round"] = [int(row["parent_round"]) for row in audit_rows]
    mandatory["selection_origin"] = "mandatory_frozen_atlas"
    mandatory["new_parent_round"] = 0
    mandatory["mandatory_source_row_sha256"] = _mandatory_row_hashes(mandatory)
    selected_new["mandatory_source_row_sha256"] = None
    dataset = pd.concat([mandatory, selected_new], ignore_index=True, sort=False)

    round1_parents = {
        int(row["source_parent_node_id"])
        for row in audit_rows
        if int(row["parent_round"]) == 1
    }
    remaining_active = [parent for parent in eligible_parents if len(queues[parent])]
    active_counts = [new_count[parent] for parent in remaining_active]
    fairness_gap = max(active_counts) - min(active_counts) if active_counts else 0
    report: dict[str, object] = {
        "target_rows": target,
        "minimum_rows": minimum,
        "mandatory_row_count": len(mandatory),
        "selected_new_row_count": len(selected_new),
        "actual_rows": len(dataset),
        "eligible_parent_count": len(eligible_parents),
        "round1_served_parent_count": len(round1_parents),
        "new_parent_service_coverage_round1": (
            len(round1_parents) / len(eligible_parents) if eligible_parents else 1.0
        ),
        "new_parent_service_coverage": (
            sum(count >= 1 for count in new_count.values()) / len(eligible_parents)
            if eligible_parents
            else 1.0
        ),
        "total_parent_coverage": dataset["source_parent_node_id"].astype(int).nunique()
        / max(1, len(set(mandatory["source_parent_node_id"].astype(int)) | set(eligible_parents))),
        "active_nonexhausted_parent_fairness_gap": int(fairness_gap),
        "duplicate_xyz_count": int(dataset["xyz_key"].astype(str).duplicated().sum()),
        "duplicate_physical_point_count": int(
            dataset["physical_point_id"].astype(str).duplicated().sum()
        ),
        "row_padding_used": False,
        "minimum_rows_pass": len(dataset) >= minimum,
        "exact_target_pass": len(dataset) == target,
        "parent_first_scheduler_invariant": bool(fairness_gap <= 1),
        "stratum_eligible_parent_denominator": {
            str(key): int(value) for key, value in sorted(denominator.items())
        },
        "new_row_count_by_parent_min": min(new_count.values()) if new_count else 0,
        "new_row_count_by_parent_max": max(new_count.values()) if new_count else 0,
    }
    return dataset.reset_index(drop=True), pd.DataFrame.from_records(audit_rows), report


def hierarchical_replacement_candidates(
    failed_row: Mapping[str, object] | pd.Series,
    candidates: pd.DataFrame,
    *,
    used_physical_point_ids: Sequence[str] = (),
) -> pd.DataFrame:
    """Return the registered Level 0/1 replacement queue for one failed row."""

    required = (
        "physical_point_id",
        "source_parent_node_id",
        "radial_shell_id",
        "sector_id",
        "canonical_component_id",
        "split_role",
        "label_quality",
        *XYZ_COLUMNS,
    )
    _require_columns(candidates, required, "replacement candidates")
    source = dict(failed_row)
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"failed row missing replacement fields: {missing}")
    primary_quality = str(source["label_quality"])
    allowed_quality = ("Gold",) if primary_quality == "Gold" else (primary_quality, "Gold")
    used = set(map(str, used_physical_point_ids)) | {str(source["physical_point_id"])}
    eligible = candidates[
        candidates["radial_shell_id"].astype(int).eq(int(source["radial_shell_id"]))
        & candidates["sector_id"].astype(int).eq(int(source["sector_id"]))
        & candidates["canonical_component_id"].astype(str).eq(str(source["canonical_component_id"]))
        & candidates["split_role"].astype(str).eq(str(source["split_role"]))
        & candidates["label_quality"].astype(str).isin(allowed_quality)
        & ~candidates["physical_point_id"].astype(str).isin(used)
    ].copy()
    if eligible.empty:
        eligible["replacement_level"] = pd.Series(dtype="object")
        return eligible
    same_parent = eligible["source_parent_node_id"].astype(int).eq(
        int(source["source_parent_node_id"])
    )
    eligible["replacement_level"] = np.where(same_parent, "level_0", "level_1")
    eligible["_level_rank"] = np.where(same_parent, 0, 1)
    eligible["_quality_rank"] = np.where(
        eligible["label_quality"].astype(str).eq(primary_quality), 0, 1
    )
    source_xyz = np.asarray([source[name] for name in XYZ_COLUMNS], dtype=float)
    eligible["replacement_distance_mm"] = (
        np.linalg.norm(eligible.loc[:, XYZ_COLUMNS].to_numpy(float) - source_xyz, axis=1)
        * 1000.0
    )
    return (
        eligible.sort_values(
            [
                "_level_rank",
                "_quality_rank",
                "replacement_distance_mm",
                "source_parent_node_id",
                "physical_point_id",
            ],
            kind="stable",
        )
        .drop(columns=["_level_rank", "_quality_rank"])
        .reset_index(drop=True)
    )


__all__ = [
    "annotate_spatial_strata",
    "apply_macroblock_split_registry",
    "build_macroblock_split_registry",
    "graph_shell_width_mm",
    "hierarchical_replacement_candidates",
    "select_parent_first_sparse_wide",
    "stable_row_id",
]
