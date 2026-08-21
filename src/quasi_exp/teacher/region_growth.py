"""Region-first canonical Teacher primitives for BACRA V12.14.

This module owns the numerical meaning of a region label.  A caller may plan
targets or shard independent corrector calls, but it cannot admit a label
without going through :func:`reduce_parent_candidates`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree

from .canonical import TeacherPolicy, _correct_target, weighted_damped_pinv
from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS


VOXEL_COLUMNS = ("voxel_i", "voxel_j", "voxel_k")
JACOBIAN_COLUMNS = tuple(
    f"jacobian_{row}_{column}" for row in range(3) for column in range(6)
)
PACK_OFFSET = 1 << 20
PACK_MASK = (1 << 21) - 1


class RegionEnvironment(Protocol):
    bounds: np.ndarray

    def fk(self, beta_rad: np.ndarray) -> np.ndarray: ...

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RegionLabelPolicy:
    """Frozen V12.14 local-continuation and label-admission policy."""

    parent_distance_max_mm: float = 15.0
    parent_attempt_count: int = 5
    successful_parent_minimum: int = 3
    continuation_substep_mm: float = 5.0
    gold_residual_max_mm: float = 1.0
    gold_candidate_gap_max_deg: float = 0.5
    silver_residual_max_mm: float = 3.0
    silver_candidate_gap_max_deg: float = 1.0
    edge_distance_max_mm: float = 10.0
    teacher_policy: TeacherPolicy = TeacherPolicy(
        tracking_tolerance_mm=3.0,
        max_corrector_iterations=80,
        safe_margin_repulsion_step_deg=0.0,
    )

    def __post_init__(self) -> None:
        values = (
            self.parent_distance_max_mm,
            self.continuation_substep_mm,
            self.gold_residual_max_mm,
            self.gold_candidate_gap_max_deg,
            self.silver_residual_max_mm,
            self.silver_candidate_gap_max_deg,
            self.edge_distance_max_mm,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("region thresholds must be finite and positive")
        if self.parent_attempt_count < self.successful_parent_minimum:
            raise ValueError("parent attempts must cover the required successes")
        if self.successful_parent_minimum < 2:
            raise ValueError("at least two independent parents are required")
        if self.gold_residual_max_mm > self.silver_residual_max_mm:
            raise ValueError("Gold residual cannot be looser than Silver")
        if self.gold_candidate_gap_max_deg > self.silver_candidate_gap_max_deg:
            raise ValueError("Gold agreement cannot be looser than Silver")

    @property
    def fingerprint(self) -> str:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if key != "teacher_policy"
        }
        payload["teacher_policy"] = self.teacher_policy.fingerprint
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()


def beta_rms_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Unweighted six-dimensional beta RMS, in degrees."""

    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    return np.rad2deg(np.sqrt(np.mean(np.square(lhs - rhs), axis=-1)))


def pack_voxels(voxels: np.ndarray) -> np.ndarray:
    values = np.asarray(voxels, dtype=np.int64).reshape(-1, 3)
    shifted = values + PACK_OFFSET
    if np.any(shifted < 0) or np.any(shifted > PACK_MASK):
        raise ValueError("voxel coordinate exceeds deterministic 21-bit packing")
    return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]


def unpack_voxels(keys: np.ndarray) -> np.ndarray:
    values = np.asarray(keys, dtype=np.int64).reshape(-1)
    return np.column_stack(
        [
            ((values >> 42) & PACK_MASK) - PACK_OFFSET,
            ((values >> 21) & PACK_MASK) - PACK_OFFSET,
            (values & PACK_MASK) - PACK_OFFSET,
        ]
    ).astype(np.int64)


def voxel_indices(xyz_m: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    scale = float(voxel_size_mm) / 1000.0
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("voxel_size_mm must be finite and positive")
    return np.floor(np.asarray(xyz_m, dtype=float).reshape(-1, 3) / scale).astype(
        np.int64
    )


def canonicalize_seed_set(
    frame: pd.DataFrame,
    *,
    source_priority: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact task/phase duplicates without averaging inverse labels.

    Core and bridge labels precede tube labels.  Exact ``u=0,v=0`` tube copies
    are excluded before de-duplication so their independently solved beta does
    not silently replace the canonical centerline label.
    """

    required = {
        "family_id",
        "phase_idx",
        "dataset_source",
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"D3 is missing canonicalization columns: {missing}")
    work = frame.copy()
    zero_tube = work["family_id"].astype(str).str.endswith("_u+0_v+0")
    registered_priority = (
        {"D0_v12_11": 0, "D2_bridge": 1, "D3_tube": 2}
        if source_priority is None
        else {str(key): int(value) for key, value in source_priority.items()}
    )
    priority = work["dataset_source"].map(registered_priority)
    if priority.isna().any():
        raise ValueError("D3 contains an unregistered dataset_source")
    work["_source_priority"] = priority.astype(np.int8)
    work["_zero_tube"] = zero_tube
    rounded = np.round(work.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), 12)
    for index, column in enumerate(XYZ_COLUMNS):
        work[f"_key_{column}"] = rounded[:, index]
    key = [f"_key_{column}" for column in XYZ_COLUMNS] + ["phase_idx"]
    ordered = work.sort_values(
        [*key, "_zero_tube", "_source_priority", "family_id"],
        kind="stable",
    )
    keep = ~ordered.duplicated(key, keep="first")
    clean = ordered.loc[keep & ~ordered["_zero_tube"]].copy()
    duplicate = ordered.loc[~keep | ordered["_zero_tube"]].copy()
    selected = ordered.loc[keep, [*key, "family_id", *BETA_COLUMNS]].rename(
        columns={
            "family_id": "selected_family_id",
            **{name: f"selected_{name}" for name in BETA_COLUMNS},
        }
    )
    duplicate = duplicate.merge(selected, on=key, how="left", validate="many_to_one")
    beta = duplicate.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    chosen = duplicate.loc[
        :, [f"selected_{name}" for name in BETA_COLUMNS]
    ].to_numpy(dtype=float)
    duplicate["beta_rms_to_selected_deg"] = beta_rms_deg(beta, chosen)
    duplicate["duplicate_reason"] = np.where(
        duplicate["_zero_tube"],
        "zero_offset_tube_copy",
        "exact_task_phase_duplicate",
    )
    private = [
        column
        for column in clean.columns
        if column.startswith("_key_") or column.startswith("_source_") or column == "_zero_tube"
    ]
    clean = clean.drop(columns=private).sort_values(
        ["family_id", "phase_idx"], kind="stable"
    )
    duplicate = duplicate.drop(
        columns=[
            column
            for column in duplicate.columns
            if column.startswith("_key_") or column.startswith("_source_")
        ]
    )
    return clean.reset_index(drop=True), duplicate.reset_index(drop=True)


def farthest_point_seed_indices(
    xyz_m: np.ndarray, count: int, *, seed: int
) -> np.ndarray:
    """Deterministic Euclidean farthest-point sampling."""

    xyz = np.asarray(xyz_m, dtype=float).reshape(-1, 3)
    if count < 1 or count > len(xyz):
        raise ValueError("seed count must be within the input row count")
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, len(xyz)))
    selected = np.empty(int(count), dtype=np.int64)
    selected[0] = first
    distance2 = np.sum(np.square(xyz - xyz[first]), axis=1)
    distance2[first] = -1.0
    for index in range(1, int(count)):
        chosen = int(np.argmax(distance2))
        selected[index] = chosen
        candidate = np.sum(np.square(xyz - xyz[chosen]), axis=1)
        distance2 = np.minimum(distance2, candidate)
        distance2[selected[: index + 1]] = -1.0
    return selected


def aggregate_capability_voxels(
    pool: pd.DataFrame,
    *,
    voxel_size_mm: float,
    conditioning_max: float,
) -> pd.DataFrame:
    """Reaggregate the branch-agnostic pool to the frozen formal voxel grid."""

    required = {*XYZ_COLUMNS, "kappa", "minimum_margin_deg"}
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"capability pool missing columns: {missing}")
    work = pool.loc[np.isfinite(pool[list(required)]).all(axis=1)].copy()
    work = work.loc[work["kappa"].le(float(conditioning_max))].copy()
    voxels = voxel_indices(work.loc[:, XYZ_COLUMNS].to_numpy(), voxel_size_mm)
    for index, column in enumerate(VOXEL_COLUMNS):
        work[column] = voxels[:, index]
    grouped = (
        work.groupby(list(VOXEL_COLUMNS), sort=True, observed=True)
        .agg(
            support_count=("kappa", "size"),
            x_m=("x_m", "mean"),
            y_m=("y_m", "mean"),
            z_m=("z_m", "mean"),
            kappa_min=("kappa", "min"),
            kappa_mean=("kappa", "mean"),
            kappa_max=("kappa", "max"),
            support_margin_min_deg=("minimum_margin_deg", "min"),
            support_margin_max_deg=("minimum_margin_deg", "max"),
        )
        .reset_index()
    )
    grouped["voxel_key"] = pack_voxels(grouped.loc[:, VOXEL_COLUMNS].to_numpy())
    return grouped.sort_values("voxel_key", kind="stable").reset_index(drop=True)


def select_capability_region(
    voxel_map: pd.DataFrame,
    seed_xyz_m: np.ndarray,
    *,
    voxel_size_mm: float,
    thickness_layers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select all occupied 26-neighbour voxels within a frozen graph depth."""

    if int(thickness_layers) < 0:
        raise ValueError("thickness_layers must be non-negative")
    occupied_keys = voxel_map["voxel_key"].to_numpy(dtype=np.int64)
    occupied_xyz = voxel_map.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    tree = cKDTree(occupied_xyz)
    _distance, indices = tree.query(np.asarray(seed_xyz_m, dtype=float), k=1)
    seed_keys = np.unique(occupied_keys[np.asarray(indices, dtype=np.int64)])
    visited = seed_keys.copy()
    frontier = seed_keys.copy()
    wave_frames = [
        pd.DataFrame({"voxel_key": seed_keys, "region_wave": 0})
    ]
    offsets = np.asarray(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)
        ],
        dtype=np.int64,
    )
    for wave in range(1, int(thickness_layers) + 1):
        candidates = (
            unpack_voxels(frontier)[:, None, :] + offsets[None, :, :]
        ).reshape(-1, 3)
        candidate_keys = np.unique(pack_voxels(candidates))
        candidate_keys = np.intersect1d(
            candidate_keys, occupied_keys, assume_unique=True
        )
        frontier = np.setdiff1d(candidate_keys, visited, assume_unique=True)
        if len(frontier) == 0:
            break
        visited = np.union1d(visited, frontier)
        wave_frames.append(
            pd.DataFrame({"voxel_key": frontier, "region_wave": wave})
        )
    waves = pd.concat(wave_frames, ignore_index=True)
    selected = voxel_map.merge(
        waves, on="voxel_key", how="inner", validate="one_to_one"
    ).sort_values(["region_wave", "voxel_key"], kind="stable")
    seed_rows = pd.DataFrame(
        {
            "seed_order": np.arange(len(seed_xyz_m), dtype=np.int64),
            "voxel_key": occupied_keys[np.asarray(indices, dtype=np.int64)],
            "support_distance_mm": np.asarray(_distance) * 1000.0,
        }
    )
    return selected.reset_index(drop=True), seed_rows


def select_sparse_targets(
    region: pd.DataFrame,
    seed_xyz_m: np.ndarray,
    *,
    count: int,
    seed: int,
    parent_distance_max_mm: float = 15.0,
    required_parent_support_count: int = 3,
    initial_parent_xyz_m: np.ndarray | None = None,
) -> pd.DataFrame:
    """Freeze a topology-preserving, parent-reachable sparse inventory.

    Half of the inventory establishes the connected backbone used by the
    topology Gate.  The remainder is admitted in deterministic frontier
    batches: every selected fill point already has the required number of
    earlier targets within the registered parent distance.  The solver still
    attempts the separately configured five nearest parents, but now it grows
    the chart batch by batch instead of attempting unsupported random points in
    one pass.
    """

    if count < 1 or count > len(region):
        raise ValueError("sparse target count exceeds selected region")
    work = region.copy()
    distance, _ = cKDTree(np.asarray(seed_xyz_m, dtype=float)).query(
        work.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), k=1
    )
    work["seed_distance_mm"] = distance * 1000.0
    maximum_wave = int(work["region_wave"].max())
    work["target_stratum"] = "interior"
    work.loc[work["region_wave"].ge(maximum_wave), "target_stratum"] = "boundary"
    far_cut = float(np.quantile(work["seed_distance_mm"], 0.90))
    work.loc[work["seed_distance_mm"].ge(far_cut), "target_stratum"] = "far"
    occupied = set(map(int, work["voxel_key"]))
    start_row = work.sort_values(
        ["seed_distance_mm", "voxel_key"], kind="stable"
    ).iloc[0]
    backbone_target = max(1, int(count) // 2)
    queue = [int(start_row["voxel_key"])]
    queued = set(queue)
    backbone: list[int] = []
    offsets = [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ]
    cursor = 0
    while cursor < len(queue) and len(backbone) < backbone_target:
        key_value = queue[cursor]
        cursor += 1
        backbone.append(key_value)
        coordinate = unpack_voxels(np.asarray([key_value]))[0]
        neighbor_keys = pack_voxels(
            coordinate[None, :] + np.asarray(offsets, dtype=np.int64)
        )
        for neighbor in sorted(map(int, neighbor_keys)):
            if neighbor in occupied and neighbor not in queued:
                queued.add(neighbor)
                queue.append(neighbor)
    backbone_frame = work.loc[work["voxel_key"].isin(backbone)].copy()
    backbone_frame["selection_role"] = "connected_backbone"
    backbone_frame["growth_batch"] = backbone_frame["region_wave"].astype(int)
    backbone_frame["planned_parent_support_count"] = np.nan
    backbone_frame["planned_farthest_parent_distance_mm"] = np.nan
    quotas = {
        "interior": int(round(count * 0.70)),
        "boundary": int(round(count * 0.20)),
    }
    quotas["far"] = int(count - quotas["interior"] - quotas["boundary"])
    backbone_counts = {
        stratum: int(backbone_frame["target_stratum"].eq(stratum).sum())
        for stratum in quotas
    }
    effective_quotas = {
        stratum: max(quotas[stratum], backbone_counts[stratum])
        for stratum in quotas
    }
    excess = int(sum(effective_quotas.values()) - count)
    for stratum in ("interior", "boundary", "far"):
        reducible = effective_quotas[stratum] - backbone_counts[stratum]
        reduction = min(excess, reducible)
        effective_quotas[stratum] -= reduction
        excess -= reduction
    if excess:
        raise ValueError("connected backbone exceeds sparse inventory size")

    rng = np.random.default_rng(int(seed))
    work["_selection_rank"] = rng.permutation(len(work))
    selected: list[pd.DataFrame] = [backbone_frame]
    used: set[int] = set(backbone)
    selected_counts = dict(backbone_counts)
    next_batch = int(backbone_frame["growth_batch"].max()) + 1
    batch_capacity = max(1, min(500, count // 10))
    required_support = min(
        int(required_parent_support_count), len(backbone_frame)
    )
    while len(used) < count:
        earlier = work.loc[work["voxel_key"].isin(used)]
        earlier_xyz = earlier.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
        if initial_parent_xyz_m is not None:
            earlier_xyz = np.vstack(
                [
                    np.asarray(initial_parent_xyz_m, dtype=float).reshape(-1, 3),
                    earlier_xyz,
                ]
            )
        remaining = work.loc[~work["voxel_key"].isin(used)].copy()
        support_distance, _ = cKDTree(earlier_xyz).query(
            remaining.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
            k=required_support,
        )
        if required_support == 1:
            farthest_support = np.asarray(support_distance, dtype=float)
        else:
            farthest_support = np.asarray(support_distance, dtype=float)[:, -1]
        remaining["planned_parent_support_count"] = required_support
        remaining["planned_farthest_parent_distance_mm"] = (
            farthest_support * 1000.0
        )
        eligible = remaining.loc[
            remaining["planned_farthest_parent_distance_mm"].le(
                float(parent_distance_max_mm) + 1.0e-9
            )
        ]
        capacity = min(batch_capacity, count - len(used))
        picked: list[pd.DataFrame] = []
        for stratum in ("far", "boundary", "interior"):
            needed = effective_quotas[stratum] - selected_counts[stratum]
            if needed <= 0 or capacity <= 0:
                continue
            subset = eligible.loc[eligible["target_stratum"].eq(stratum)].sort_values(
                ["_selection_rank", "voxel_key"], kind="stable"
            )
            part = subset.iloc[: min(needed, capacity)].copy()
            if len(part):
                picked.append(part)
                capacity -= len(part)
        if not picked:
            raise ValueError(
                "parent-reachable frontier cannot satisfy sparse target quotas"
            )
        batch = pd.concat(picked, ignore_index=True)
        batch["selection_role"] = "parent_reachable_frontier"
        batch["growth_batch"] = next_batch
        selected.append(batch)
        used.update(map(int, batch["voxel_key"]))
        for stratum, stratum_count in batch["target_stratum"].value_counts().items():
            selected_counts[str(stratum)] += int(stratum_count)
        next_batch += 1

    chosen = pd.concat(selected, ignore_index=True)
    chosen = chosen.drop(columns=["_selection_rank"], errors="ignore")
    chosen = chosen.sort_values(
        ["growth_batch", "voxel_key"], kind="stable"
    ).reset_index(drop=True)
    chosen.insert(0, "target_id", np.arange(len(chosen), dtype=np.int64))
    return chosen


def predictor_from_anchor(
    environment: RegionEnvironment,
    target_xyz_m: np.ndarray,
    anchor_xyz_m: np.ndarray,
    anchor_beta_rad: np.ndarray,
    policy: TeacherPolicy,
) -> np.ndarray:
    jacobian = np.asarray(environment.jacobian(anchor_beta_rad), dtype=float).reshape(
        3, 6
    )
    pinv = weighted_damped_pinv(
        jacobian,
        damping=float(policy.damping),
        weights=np.asarray(policy.beta_weights, dtype=float),
    )
    prediction = np.asarray(anchor_beta_rad, dtype=float) + pinv @ (
        np.asarray(target_xyz_m, dtype=float)
        - np.asarray(anchor_xyz_m, dtype=float)
    )
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    return np.clip(prediction, bounds[:, 0], bounds[:, 1])


def continue_from_parent(
    environment: RegionEnvironment,
    target_xyz_m: np.ndarray,
    parent_xyz_m: np.ndarray,
    parent_beta_rad: np.ndarray,
    *,
    policy: RegionLabelPolicy,
) -> dict[str, Any]:
    """Run one short predictor-corrector continuation from one parent."""

    target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    origin = np.asarray(parent_xyz_m, dtype=float).reshape(3)
    beta = np.asarray(parent_beta_rad, dtype=float).reshape(6)
    distance_mm = float(np.linalg.norm(target - origin) * 1000.0)
    if distance_mm > policy.parent_distance_max_mm + 1.0e-9:
        return {
            "success": False,
            "reason": "parent_too_far",
            "distance_mm": distance_mm,
        }
    steps = max(1, int(math.ceil(distance_mm / policy.continuation_substep_mm)))
    total_iterations = 0
    residual_mm = math.inf
    for step in range(1, steps + 1):
        point = origin + (target - origin) * (float(step) / float(steps))
        prediction = predictor_from_anchor(
            environment, point, origin, beta, policy.teacher_policy
        )
        beta, residual_mm, iterations, success = _correct_target(
            environment, point, prediction, policy.teacher_policy
        )
        total_iterations += int(iterations)
        if not success or residual_mm > policy.silver_residual_max_mm:
            return {
                "success": False,
                "reason": "corrector_failed",
                "distance_mm": distance_mm,
                "residual_mm": float(residual_mm),
                "corrector_iterations": total_iterations,
            }
        origin = point
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    actual_bounds = bool(
        np.all(beta >= bounds[:, 0] - 1.0e-12)
        and np.all(beta <= bounds[:, 1] + 1.0e-12)
    )
    return {
        "success": bool(actual_bounds),
        "reason": "success" if actual_bounds else "actual_bounds",
        "distance_mm": distance_mm,
        "residual_mm": float(residual_mm),
        "corrector_iterations": total_iterations,
        **{name: float(beta[index]) for index, name in enumerate(BETA_COLUMNS)},
    }


def reduce_parent_candidates(
    candidates: pd.DataFrame,
    *,
    policy: RegionLabelPolicy,
    required_successes: int | None = None,
) -> dict[str, Any]:
    """Choose a medoid and assign RegionGold/RegionSilver/Reject."""

    minimum = (
        int(policy.successful_parent_minimum)
        if required_successes is None
        else int(required_successes)
    )
    successful = candidates.loc[candidates["success"].astype(bool)].copy()
    if len(successful) < minimum:
        return {
            "accepted": False,
            "quality_class": "Reject",
            "reason": "insufficient_independent_parent_success",
            "successful_parent_count": int(len(successful)),
            "candidate_gap_max_deg": math.inf,
        }
    raw_beta = successful.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    raw_pairwise = beta_rms_deg(
        raw_beta[:, None, :], raw_beta[None, :, :]
    )
    consensus_indices: tuple[int, ...] | None = None
    consensus_gap = math.inf
    for threshold in (
        policy.gold_candidate_gap_max_deg,
        policy.silver_candidate_gap_max_deg,
    ):
        for subset_size in range(len(successful), minimum - 1, -1):
            options = []
            for indices in combinations(range(len(successful)), subset_size):
                local = raw_pairwise[np.ix_(indices, indices)]
                gap = float(np.max(local))
                if gap <= threshold:
                    residual = float(
                        successful.iloc[list(indices)]["residual_mm"].sum()
                    )
                    options.append((gap, residual, indices))
            if options:
                consensus_gap, _residual, consensus_indices = min(options)
                break
        if consensus_indices is not None:
            break
    if consensus_indices is None:
        return {
            "accepted": False,
            "quality_class": "Reject",
            "reason": "no_silver_consensus_clique",
            "successful_parent_count": int(len(successful)),
            "consensus_parent_count": 0,
            "excluded_conflicting_parent_count": int(len(successful)),
            "raw_candidate_gap_max_deg": float(np.max(raw_pairwise)),
            "candidate_gap_max_deg": math.inf,
        }
    consensus = successful.iloc[list(consensus_indices)].copy()
    beta = consensus.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    pairwise = beta_rms_deg(beta[:, None, :], beta[None, :, :])
    medoid = int(np.argmin(np.sum(pairwise, axis=1)))
    gap = float(np.max(pairwise))
    residual_max = float(consensus["residual_mm"].max())
    if (
        residual_max <= policy.gold_residual_max_mm
        and gap <= policy.gold_candidate_gap_max_deg
    ):
        quality = "RegionGold"
    elif (
        residual_max <= policy.silver_residual_max_mm
        and gap <= policy.silver_candidate_gap_max_deg
    ):
        quality = "RegionSilver"
    else:
        quality = "Reject"
    row = consensus.iloc[medoid]
    return {
        "accepted": quality != "Reject",
        "quality_class": quality,
        "reason": "accepted" if quality != "Reject" else "candidate_conflict",
        "successful_parent_count": int(len(successful)),
        "consensus_parent_count": int(len(consensus)),
        "excluded_conflicting_parent_count": int(
            len(successful) - len(consensus)
        ),
        "raw_candidate_gap_max_deg": float(np.max(raw_pairwise)),
        "candidate_gap_p95_deg": float(np.percentile(pairwise, 95)),
        "candidate_gap_max_deg": gap,
        "residual_max_mm": residual_max,
        "selected_candidate_row": int(row.name),
        "consensus_candidate_rows": [
            int(successful.iloc[index].name)
            for index in consensus_indices
        ],
        **{name: float(row[name]) for name in BETA_COLUMNS},
    }


def plan_parent_rows(
    targets: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    policy: RegionLabelPolicy,
) -> pd.DataFrame:
    """Choose up to the frozen number of nearest distinct accepted parents."""

    if len(targets) == 0 or len(anchors) == 0:
        return pd.DataFrame()
    targets = targets.reset_index(drop=True)
    anchors = anchors.reset_index(drop=True)
    tree = cKDTree(anchors.loc[:, XYZ_COLUMNS].to_numpy(dtype=float))
    k = min(int(policy.parent_attempt_count), len(anchors))
    distance, index = tree.query(
        targets.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), k=k
    )
    distance = np.asarray(distance).reshape(len(targets), k)
    index = np.asarray(index).reshape(len(targets), k)
    rows: list[dict[str, Any]] = []
    for target_order, target in targets.iterrows():
        for parent_rank in range(k):
            distance_mm = float(distance[target_order, parent_rank] * 1000.0)
            if distance_mm > policy.parent_distance_max_mm:
                continue
            parent = anchors.iloc[int(index[target_order, parent_rank])]
            rows.append(
                {
                    "target_id": int(target["target_id"]),
                    "target_x_m": float(target["x_m"]),
                    "target_y_m": float(target["y_m"]),
                    "target_z_m": float(target["z_m"]),
                    "parent_rank": parent_rank,
                    "parent_id": str(parent["node_id"]),
                    "parent_x_m": float(parent["x_m"]),
                    "parent_y_m": float(parent["y_m"]),
                    "parent_z_m": float(parent["z_m"]),
                    "parent_distance_mm": distance_mm,
                    **{name: float(parent[name]) for name in BETA_COLUMNS},
                }
            )
    return pd.DataFrame(rows)


def build_consistent_edges(
    nodes: pd.DataFrame, *, distance_max_mm: float, gap_max_deg: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut conflicting local edges and retain the largest connected component."""

    xyz = nodes.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    beta = nodes.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    pairs = np.asarray(
        sorted(cKDTree(xyz).query_pairs(float(distance_max_mm) / 1000.0)),
        dtype=np.int64,
    ).reshape(-1, 2)
    if len(pairs) == 0:
        empty = pd.DataFrame(
            columns=["left_node_id", "right_node_id", "distance_mm", "beta_gap_deg", "accepted"]
        )
        return empty, nodes.iloc[:0].copy()
    distance = (
        np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1) * 1000.0
    )
    gap = beta_rms_deg(beta[pairs[:, 0]], beta[pairs[:, 1]])
    accepted = gap <= float(gap_max_deg)
    edges = pd.DataFrame(
        {
            "left_node_id": nodes.iloc[pairs[:, 0]]["node_id"].to_numpy(),
            "right_node_id": nodes.iloc[pairs[:, 1]]["node_id"].to_numpy(),
            "distance_mm": distance,
            "beta_gap_deg": gap,
            "accepted": accepted,
        }
    )
    good = pairs[accepted]
    adjacency = coo_matrix(
        (
            np.ones(len(good) * 2, dtype=np.int8),
            (
                np.concatenate([good[:, 0], good[:, 1]]),
                np.concatenate([good[:, 1], good[:, 0]]),
            ),
        ),
        shape=(len(nodes), len(nodes)),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    counts = np.bincount(labels, minlength=component_count)
    largest = int(np.argmax(counts))
    selected = nodes.loc[labels == largest].copy()
    selected["component_id"] = labels[labels == largest]
    return edges, selected.reset_index(drop=True)


def consistency_audit(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    triangle_count: int,
    two_path_count: int,
    seed: int,
    environment: RegionEnvironment | None = None,
    policy: RegionLabelPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Materialize deterministic triangle-return and multi-parent evidence."""

    rng = np.random.default_rng(int(seed))
    node_lookup = {
        str(row["node_id"]): row
        for _, row in nodes.iterrows()
    }
    valid_node_ids = set(node_lookup)
    good = edges.loc[edges["accepted"].astype(bool)].copy()
    good = good.loc[
        good["left_node_id"].astype(str).isin(valid_node_ids)
        & good["right_node_id"].astype(str).isin(valid_node_ids)
    ].copy()
    neighbor: dict[str, set[str]] = {}
    for left, right in good[
        ["left_node_id", "right_node_id"]
    ].itertuples(index=False, name=None):
        neighbor.setdefault(str(left), set()).add(str(right))
        neighbor.setdefault(str(right), set()).add(str(left))
    triangles: set[tuple[str, str, str]] = set()
    for left in sorted(neighbor):
        for right in sorted(neighbor[left]):
            if left >= right:
                continue
            for third in neighbor[left].intersection(neighbor[right]):
                triangles.add(tuple(sorted((left, right, third))))
    triangle_list = sorted(triangles)
    if len(triangle_list) > triangle_count:
        chosen = rng.choice(len(triangle_list), size=triangle_count, replace=False)
        triangle_list = [triangle_list[index] for index in sorted(chosen)]
    gap_lookup = {
        frozenset((str(left), str(right))): float(gap)
        for left, right, gap in good[
            ["left_node_id", "right_node_id", "beta_gap_deg"]
        ].itertuples(index=False, name=None)
    }
    triangle_rows = []
    for index, (a, b, c) in enumerate(triangle_list):
        gaps = [
            gap_lookup[frozenset((a, b))],
            gap_lookup[frozenset((b, c))],
            gap_lookup[frozenset((c, a))],
        ]
        row: dict[str, Any] = {
            "audit_id": index,
            "node_a": a,
            "node_b": b,
            "node_c": c,
            "loop_edge_gap_p95_deg": float(np.percentile(gaps, 95)),
            "loop_edge_gap_max_deg": float(max(gaps)),
        }
        if environment is not None and policy is not None:
            start = node_lookup[a]
            beta_start = start.loc[list(BETA_COLUMNS)].to_numpy(dtype=float)
            beta = beta_start.copy()
            origin = start.loc[list(XYZ_COLUMNS)].to_numpy(dtype=float)
            total_iterations = 0
            maximum_residual_mm = 0.0
            success = True
            failure_reason = "success"
            for destination_id in (b, c, a):
                destination = node_lookup[destination_id]
                target = destination.loc[list(XYZ_COLUMNS)].to_numpy(dtype=float)
                result = continue_from_parent(
                    environment,
                    target,
                    origin,
                    beta,
                    policy=policy,
                )
                total_iterations += int(result.get("corrector_iterations", 0))
                if not bool(result["success"]):
                    success = False
                    failure_reason = str(result["reason"])
                    break
                maximum_residual_mm = max(
                    maximum_residual_mm, float(result["residual_mm"])
                )
                beta = np.asarray(
                    [result[name] for name in BETA_COLUMNS], dtype=float
                )
                origin = target
            row.update(
                {
                    "loop_continuation_success": success,
                    "loop_failure_reason": failure_reason,
                    "loop_return_gap_deg": (
                        float(beta_rms_deg(beta, beta_start))
                        if success
                        else np.nan
                    ),
                    "loop_corrector_iterations": total_iterations,
                    "loop_residual_max_mm": (
                        maximum_residual_mm if success else np.nan
                    ),
                }
            )
        else:
            row.update(
                {
                    "loop_continuation_success": True,
                    "loop_failure_reason": "edge_gap_proxy_without_environment",
                    "loop_return_gap_deg": float(np.percentile(gaps, 95)),
                    "loop_corrector_iterations": 0,
                    "loop_residual_max_mm": np.nan,
                }
            )
        triangle_rows.append(row)
    triangle_frame = pd.DataFrame(triangle_rows)
    successful = candidates.loc[candidates["success"].astype(bool)].copy()
    if "consensus_selected" in successful.columns:
        successful = successful.loc[
            successful["consensus_selected"].astype(bool)
        ].copy()
    path_rows = []
    for target_id, group in successful.groupby("target_id", sort=True):
        if len(group) < 2:
            continue
        beta = group.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
        gaps = beta_rms_deg(beta[:, None, :], beta[None, :, :])
        path_rows.append(
            {
                "target_id": int(target_id),
                "independent_path_count": int(len(group)),
                "two_path_gap_p95_deg": float(np.percentile(gaps, 95)),
                "two_path_gap_max_deg": float(np.max(gaps)),
            }
        )
    path_frame = pd.DataFrame(path_rows)
    if len(path_frame) > two_path_count:
        chosen = rng.choice(len(path_frame), size=two_path_count, replace=False)
        path_frame = path_frame.iloc[sorted(chosen)].reset_index(drop=True)
    successful_loops = triangle_frame.loc[
        triangle_frame.get(
            "loop_continuation_success",
            pd.Series(dtype=bool),
        ).astype(bool)
    ]
    report = {
        "triangle_available": int(len(triangle_frame)),
        "triangle_successful": int(len(successful_loops)),
        "triangle_failed": int(len(triangle_frame) - len(successful_loops)),
        "two_path_available": int(len(path_frame)),
        "triangle_p95_deg": (
            float(np.percentile(successful_loops["loop_return_gap_deg"], 95))
            if len(successful_loops)
            else None
        ),
        "triangle_max_deg": (
            float(successful_loops["loop_return_gap_deg"].max())
            if len(successful_loops)
            else None
        ),
        "two_path_p95_deg": (
            float(np.percentile(path_frame["two_path_gap_max_deg"], 95))
            if len(path_frame)
            else None
        ),
    }
    return triangle_frame, path_frame, report


def plan_dense_targets(
    region: pd.DataFrame,
    sparse_nodes: pd.DataFrame,
    *,
    attempt_count: int,
    seed: int,
    voxel_size_mm: float = 5.0,
    parent_distance_max_mm: float = 15.0,
    minimum_anchor_count: int = 2,
) -> pd.DataFrame:
    """Sample balanced continuous points inside supported capability voxels."""

    candidates = region.copy().reset_index(drop=True)
    if len(candidates) == 0:
        raise ValueError("dense sampling requires at least one capability voxel")
    if minimum_anchor_count < 1 or minimum_anchor_count > len(sparse_nodes):
        raise ValueError("invalid dense minimum anchor count")
    rng = np.random.default_rng(int(seed))
    sparse_xyz = sparse_nodes.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    sparse_tree = cKDTree(sparse_xyz)
    selected_parts: list[pd.DataFrame] = []
    selected_count = 0
    sampling_round = 0
    voxel_size_m = float(voxel_size_mm) / 1000.0
    while selected_count < int(attempt_count):
        order = rng.permutation(len(candidates))
        part = candidates.iloc[order].copy().reset_index(drop=True)
        voxel = part.loc[:, VOXEL_COLUMNS].to_numpy(dtype=float)
        center = (voxel + 0.5) * voxel_size_m
        target = center + rng.uniform(
            -0.5 * voxel_size_m,
            0.5 * voxel_size_m,
            size=(len(part), 3),
        )
        support_distance, _ = sparse_tree.query(
            target, k=int(minimum_anchor_count)
        )
        support_distance = np.asarray(support_distance, dtype=float)
        if minimum_anchor_count > 1:
            support_distance = support_distance[:, -1]
        keep = support_distance * 1000.0 <= (
            float(parent_distance_max_mm) + 1.0e-9
        )
        part = part.loc[keep].copy()
        if len(part) == 0:
            raise ValueError(
                "supported capability voxels yield no continuous dense targets"
            )
        part.loc[:, XYZ_COLUMNS] = target[keep]
        part["voxel_sample_round"] = sampling_round
        part["planned_anchor_support_count"] = int(minimum_anchor_count)
        part["planned_farthest_anchor_distance_mm"] = (
            support_distance[keep] * 1000.0
        )
        selected_parts.append(part)
        selected_count += len(part)
        sampling_round += 1
    chosen = pd.concat(selected_parts, ignore_index=True).iloc[
        : int(attempt_count)
    ].copy()
    chosen = chosen.sort_values(
        ["voxel_key", "voxel_sample_round"], kind="stable"
    ).reset_index(drop=True)
    chosen.insert(0, "target_id", np.arange(len(chosen), dtype=np.int64))
    xyz = sparse_xyz
    beta = sparse_nodes.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    tree = cKDTree(xyz)
    distance, anchor = tree.query(
        chosen.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        k=min(4, len(sparse_nodes)),
    )
    distance = np.asarray(distance).reshape(len(chosen), -1)
    anchor = np.asarray(anchor).reshape(len(chosen), -1)
    for rank in range(anchor.shape[1]):
        chosen[f"anchor_{rank}_node_id"] = sparse_nodes.iloc[
            anchor[:, rank]
        ]["node_id"].to_numpy()
        chosen[f"anchor_{rank}_distance_mm"] = distance[:, rank] * 1000.0
        for column in XYZ_COLUMNS:
            chosen[f"anchor_{rank}_{column}"] = sparse_nodes.iloc[anchor[:, rank]][
                column
            ].to_numpy(dtype=float)
        for column in BETA_COLUMNS:
            chosen[f"anchor_{rank}_{column}"] = sparse_nodes.iloc[anchor[:, rank]][
                column
            ].to_numpy(dtype=float)
    chosen["tetrahedral_prediction_available"] = False
    if len(sparse_nodes) >= 4:
        try:
            tessellation = Delaunay(xyz)
            simplex = tessellation.find_simplex(
                chosen.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
            )
            valid = simplex >= 0
            chosen["tetrahedral_prediction_available"] = valid
            chosen["tetrahedron_idx"] = simplex
            predictions = np.full((len(chosen), 6), np.nan)
            valid_index = np.flatnonzero(valid)
            transform = tessellation.transform[simplex[valid_index]]
            target = chosen.loc[
                valid_index, XYZ_COLUMNS
            ].to_numpy(dtype=float)
            delta = target - transform[:, 3, :]
            bary_first = np.einsum(
                "nij,nj->ni", transform[:, :3, :], delta
            )
            bary = np.column_stack(
                [bary_first, 1.0 - bary_first.sum(axis=1)]
            )
            vertices = tessellation.simplices[simplex[valid_index]]
            predictions[valid_index] = np.einsum(
                "ni,nij->nj", bary, beta[vertices]
            )
            for column_index, column in enumerate(BETA_COLUMNS):
                chosen[f"tetrahedral_{column}"] = predictions[:, column_index]
        except Exception:
            chosen["tetrahedron_idx"] = -1
    return chosen


def add_jacobians(
    frame: pd.DataFrame, environment: RegionEnvironment
) -> pd.DataFrame:
    output = frame.copy()
    values = np.empty((len(output), 18), dtype=float)
    for index, beta in enumerate(output.loc[:, BETA_COLUMNS].to_numpy(dtype=float)):
        values[index] = np.asarray(environment.jacobian(beta), dtype=float).reshape(-1)
    for index, column in enumerate(JACOBIAN_COLUMNS):
        output[column] = values[:, index]
    return output


def spatial_train_validation_split(
    frame: pd.DataFrame,
    *,
    macro_voxel_mm: float,
    validation_fraction: float,
    buffer_mm: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign whole 20-mm macro voxels and remove a 5-mm train-side buffer."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0,1)")
    output = frame.copy()
    macro = voxel_indices(
        output.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), macro_voxel_mm
    )
    key = pack_voxels(macro)
    unique = np.unique(key)
    hashed = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}:{int(value)}".encode()).digest()[:8],
                "big",
            )
            for value in unique
        ],
        dtype=np.uint64,
    )
    order = np.argsort(hashed, kind="stable")
    count = max(1, int(round(len(unique) * validation_fraction)))
    validation_keys = unique[order[:count]]
    validation = np.isin(key, validation_keys)
    output["split"] = np.where(validation, "validation", "train")
    if validation.any() and (~validation).any() and buffer_mm > 0.0:
        distance, _ = cKDTree(
            output.loc[validation, XYZ_COLUMNS].to_numpy(dtype=float)
        ).query(output.loc[~validation, XYZ_COLUMNS].to_numpy(dtype=float), k=1)
        train_index = output.index[~validation]
        buffered = train_index[np.asarray(distance) * 1000.0 < float(buffer_mm)]
        output.loc[buffered, "split"] = "buffer_excluded"
    report = {
        "row_count": int(len(output)),
        "macro_voxel_count": int(len(unique)),
        "validation_macro_voxel_count": int(len(validation_keys)),
        "train_rows": int(output["split"].eq("train").sum()),
        "validation_rows": int(output["split"].eq("validation").sum()),
        "buffer_excluded_rows": int(output["split"].eq("buffer_excluded").sum()),
    }
    return output, report
