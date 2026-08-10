"""BACRA dataset assembly, conflict checks, and sealed spatial splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS


@dataclass(frozen=True)
class SpatialSplitPolicy:
    macro_voxel_mm: float = 20.0
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    sealed_fraction: float = 0.15
    buffer_mm: float = 5.0
    seed: int = 20260736

    def __post_init__(self) -> None:
        fractions = np.asarray(
            [self.train_fraction, self.validation_fraction, self.sealed_fraction],
            dtype=float,
        )
        if not np.isfinite(fractions).all() or np.any(fractions <= 0.0):
            raise ValueError("split fractions must be finite and positive")
        if not np.isclose(float(np.sum(fractions)), 1.0, atol=1.0e-12):
            raise ValueError("split fractions must sum to one")
        if float(self.macro_voxel_mm) <= 0.0 or float(self.buffer_mm) < 0.0:
            raise ValueError("macro voxel must be positive and buffer non-negative")


@dataclass(frozen=True)
class SealedSplit:
    public_rows: pd.DataFrame
    _sealed_rows: pd.DataFrame
    manifest: pd.DataFrame
    seal_token: str

    def open_sealed(self, token: str) -> pd.DataFrame:
        if str(token) != self.seal_token:
            raise PermissionError("sealed BACRA split requires the frozen token")
        return self._sealed_rows.copy(deep=True)


def _require_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*XYZ_COLUMNS, *BETA_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    if len(frame) == 0:
        raise ValueError("dataset must contain at least one row")
    values = frame[[*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("dataset coordinates and labels must be finite")
    return frame.copy(deep=True).reset_index(drop=True)


def _block_key(values: np.ndarray, voxel_m: float) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=float) / float(voxel_m)).astype(np.int64)


def _stable_uniform(block: Sequence[int], seed: int) -> float:
    payload = f"{int(seed)}:{int(block[0])}:{int(block[1])}:{int(block[2])}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_spatial_splits(
    frame: pd.DataFrame, policy: SpatialSplitPolicy
) -> SealedSplit:
    """Assign complete macro voxels and seal the test rows.

    Rows within ``buffer_mm`` of a macro-voxel face are retained as explicit
    ``buffer`` evidence and excluded from model fitting.  This prevents nearly
    coincident points across train/validation/test boundaries.
    """

    output = _require_dataset(frame)
    xyz = output[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    voxel_m = float(policy.macro_voxel_mm) / 1000.0
    blocks = _block_key(xyz, voxel_m)
    block_ids = [f"{x}:{y}:{z}" for x, y, z in blocks]
    output["spatial_block_id"] = block_ids
    unique_blocks = sorted({tuple(map(int, row)) for row in blocks})
    roles: dict[tuple[int, int, int], str] = {}
    train_cut = float(policy.train_fraction)
    validation_cut = train_cut + float(policy.validation_fraction)
    for block in unique_blocks:
        draw = _stable_uniform(block, int(policy.seed))
        roles[block] = (
            "train"
            if draw < train_cut
            else "validation"
            if draw < validation_cut
            else "sealed_test"
        )
    output["split_role"] = [roles[tuple(map(int, row))] for row in blocks]

    local = np.mod(xyz, voxel_m)
    buffer_m = float(policy.buffer_mm) / 1000.0
    is_buffer = np.zeros(len(output), dtype=bool)
    # Exclude only points near a face whose adjacent block belongs to another
    # split.  Faces between same-role blocks are ordinary interior evidence.
    for row_index, (block_row, local_row) in enumerate(zip(blocks, local)):
        block = tuple(map(int, block_row))
        own_role = roles[block]
        for axis in range(3):
            lower = list(block)
            lower[axis] -= 1
            upper = list(block)
            upper[axis] += 1
            if (
                local_row[axis] < buffer_m
                and tuple(lower) in roles
                and roles[tuple(lower)] != own_role
            ):
                is_buffer[row_index] = True
            if (
                voxel_m - local_row[axis] < buffer_m
                and tuple(upper) in roles
                and roles[tuple(upper)] != own_role
            ):
                is_buffer[row_index] = True
    output["split_base_role"] = output["split_role"]
    output.loc[is_buffer, "split_role"] = "buffer"
    output["split_buffer"] = is_buffer

    manifest_rows: list[dict[str, Any]] = []
    for block in unique_blocks:
        block_id = f"{block[0]}:{block[1]}:{block[2]}"
        mask = output["spatial_block_id"].eq(block_id)
        manifest_rows.append(
            {
                "spatial_block_id": block_id,
                "block_x": block[0],
                "block_y": block[1],
                "block_z": block[2],
                "assigned_role": roles[block],
                "row_count": int(np.count_nonzero(mask)),
                "buffer_row_count": int(np.count_nonzero(mask & is_buffer)),
            }
        )
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["block_x", "block_y", "block_z"], kind="stable"
    )
    seal_payload = {
        "policy": policy.__dict__,
        "sealed_blocks": manifest.loc[
            manifest["assigned_role"].eq("sealed_test"), "spatial_block_id"
        ].tolist(),
    }
    seal_token = hashlib.sha256(
        json.dumps(seal_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # Sealing is owned by the macro block, not by the row's buffer role.
    # A row on the sealed side of a split boundary must remain sealed even
    # though it is excluded from evaluation as ``buffer``.  Otherwise its beta
    # label leaks into the public artifact merely because it is near a face.
    sealed_mask = output["split_base_role"].eq("sealed_test")
    sealed = output.loc[sealed_mask].copy()
    public = output.loc[~sealed_mask].copy()
    return SealedSplit(
        public_rows=public,
        _sealed_rows=sealed,
        manifest=manifest.reset_index(drop=True),
        seal_token=seal_token,
    )


def detect_cross_chart_conflicts(
    frame: pd.DataFrame,
    *,
    voxel_mm: float = 2.0,
    beta_gap_deg: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find incompatible labels at the same Cartesian voxel."""

    data = _require_dataset(frame)
    if "chart_id" not in data:
        data["chart_id"] = 0
    xyz = data[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    beta = data[list(BETA_COLUMNS)].to_numpy(dtype=float)
    voxels = _block_key(xyz, float(voxel_mm) / 1000.0)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, value in enumerate(voxels):
        groups.setdefault(tuple(map(int, value)), []).append(index)
    rows: list[dict[str, Any]] = []
    conflict_voxels = 0
    eligible_voxels = 0
    for voxel, indices in sorted(groups.items()):
        charts = data.iloc[indices]["chart_id"].astype(str).to_numpy()
        if len(set(charts)) < 2:
            continue
        eligible_voxels += 1
        maximum = 0.0
        pair: tuple[int, int] | None = None
        for local_left, left in enumerate(indices):
            for right in indices[local_left + 1 :]:
                if str(data.iloc[left]["chart_id"]) == str(data.iloc[right]["chart_id"]):
                    continue
                gap = float(
                    np.rad2deg(
                        np.sqrt(np.mean(np.square(beta[left] - beta[right])))
                    )
                )
                if gap > maximum:
                    maximum = gap
                    pair = (left, right)
        is_conflict = maximum > float(beta_gap_deg)
        conflict_voxels += int(is_conflict)
        rows.append(
            {
                "voxel_x": voxel[0],
                "voxel_y": voxel[1],
                "voxel_z": voxel[2],
                "row_count": len(indices),
                "chart_count": len(set(charts)),
                "beta_gap_max_deg": maximum,
                "conflict": bool(is_conflict),
                "left_row": pair[0] if pair else -1,
                "right_row": pair[1] if pair else -1,
            }
        )
    report = {
        "gate_pass": bool(conflict_voxels == 0),
        "eligible_cross_chart_voxels": int(eligible_voxels),
        "conflict_voxels": int(conflict_voxels),
        "conflict_voxel_ratio": (
            float(conflict_voxels / eligible_voxels) if eligible_voxels else 0.0
        ),
        "voxel_mm": float(voxel_mm),
        "beta_gap_deg": float(beta_gap_deg),
    }
    return pd.DataFrame(rows), report


def local_fill_distance_by_role(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute held-out nearest training distance without opening sealed labels."""

    if "split_role" not in frame:
        raise ValueError("frame requires split_role")
    data = _require_dataset(frame)
    train = data[data["split_role"].eq("train")]
    if len(train) == 0:
        raise ValueError("split contains no training rows")
    tree = cKDTree(train[list(XYZ_COLUMNS)].to_numpy(dtype=float))
    rows: list[dict[str, Any]] = []
    for role, group in data.groupby("split_role", sort=True):
        distance_mm = (
            tree.query(group[list(XYZ_COLUMNS)].to_numpy(dtype=float), k=1)[0] * 1000.0
        )
        rows.append(
            {
                "split_role": str(role),
                "row_count": int(len(group)),
                "fill_distance_p50_mm": float(np.percentile(distance_mm, 50)),
                "fill_distance_p95_mm": float(np.percentile(distance_mm, 95)),
                "fill_distance_max_mm": float(np.max(distance_mm)),
            }
        )
    return pd.DataFrame(rows)


def make_nested_datasets(
    frame: pd.DataFrame, sizes: Sequence[int], *, seed: int
) -> Mapping[int, pd.DataFrame]:
    """Return deterministic nested prefixes after a single frozen permutation."""

    data = _require_dataset(frame)
    requested = tuple(sorted({int(value) for value in sizes}))
    if not requested or requested[0] <= 0:
        raise ValueError("sizes must contain positive integers")
    if requested[-1] > len(data):
        raise ValueError(
            f"largest nested size {requested[-1]} exceeds available rows {len(data)}"
        )
    permutation = np.random.default_rng(int(seed)).permutation(len(data))
    ordered = data.iloc[permutation].reset_index(drop=True)
    ordered["nested_order"] = np.arange(len(ordered), dtype=np.int64)
    return {size: ordered.iloc[:size].copy() for size in requested}


def equal_count_ablation(
    methods: Mapping[str, pd.DataFrame], *, row_count: int, seed: int
) -> Mapping[str, pd.DataFrame]:
    """Select equal deterministic counts for D0/D1/D2/D3 comparisons."""

    required = {"D0", "D1", "D2", "D3"}
    missing = sorted(required - set(methods))
    if missing:
        raise ValueError(f"ablation methods missing: {missing}")
    output: dict[str, pd.DataFrame] = {}
    for index, method in enumerate(sorted(required)):
        data = _require_dataset(methods[method])
        if len(data) < int(row_count):
            raise ValueError(f"{method} has {len(data)} rows, needs {row_count}")
        order = np.random.default_rng(int(seed) + index).permutation(len(data))
        selected = data.iloc[order[: int(row_count)]].copy().reset_index(drop=True)
        selected["data_method"] = method
        output[method] = selected
    return output
