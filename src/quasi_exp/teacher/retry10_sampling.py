"""Deterministic parent-domain sampling contracts for retry10."""

from __future__ import annotations

import hashlib
import itertools
from typing import Sequence

import numpy as np
import pandas as pd

from .retry10 import BETA_COLUMNS, XYZ_COLUMNS, stable_hash_rank


def _van_der_corput(index: int, base: int = 2) -> float:
    value = 0.0
    denominator = 1.0
    active = int(index) + 1
    while active:
        active, remainder = divmod(active, int(base))
        denominator *= int(base)
        value += remainder / denominator
    return value


def parent_domain_target_registry(
    labels: pd.DataFrame,
    *,
    target_count: int,
    shard_count: int = 48,
    seed: int = 20260881,
    excluded_xyz_keys: Sequence[str] = (),
) -> pd.DataFrame:
    """Sample within frozen parent domains using convex combinations of labels.

    The registry never invents cell geometry.  A target belongs to the exact
    ``source_parent_node_id`` shared by its two certified endpoints, and a
    convex combination remains inside that registered parent domain proxy.
    """

    required = {
        "task_node_id",
        "source_parent_node_id",
        *XYZ_COLUMNS,
        *BETA_COLUMNS,
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"parent-domain labels missing columns: {missing}")
    if int(target_count) < 0 or int(shard_count) < 1:
        raise ValueError("target count and shard count must be valid")
    grouped: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    indexed = labels.drop_duplicates("task_node_id").set_index("task_node_id")
    for parent_id, group in labels.groupby("source_parent_node_id", sort=True):
        node_ids = sorted(set(group["task_node_id"].astype(int)))
        pairs = tuple(itertools.combinations(node_ids, 2))
        if pairs:
            grouped.append((int(parent_id), pairs))
    if not grouped and int(target_count):
        return pd.DataFrame()
    grouped.sort(key=lambda item: stable_hash_rank(seed, str(item[0])))
    seen = set(map(str, excluded_xyz_keys))
    seen.update(
        labels.loc[:, XYZ_COLUMNS].round(12).astype(str).agg("|".join, axis=1)
    )
    records: list[dict[str, object]] = []
    attempt = 0
    maximum_attempts = max(10_000, int(target_count) * 50)
    while len(records) < int(target_count) and attempt < maximum_attempts:
        parent_id, pairs = grouped[attempt % len(grouped)]
        cycle = attempt // len(grouped)
        left_id, right_id = pairs[cycle % len(pairs)]
        fraction = 0.025 + 0.95 * _van_der_corput(
            cycle // len(pairs), base=(2, 3, 5, 7)[attempt % 4]
        )
        left = indexed.loc[left_id]
        right = indexed.loc[right_id]
        xyz = (
            (1.0 - fraction) * left.loc[list(XYZ_COLUMNS)].to_numpy(float)
            + fraction * right.loc[list(XYZ_COLUMNS)].to_numpy(float)
        )
        key = "|".join(map(str, np.round(xyz, 12)))
        attempt += 1
        if key in seen:
            continue
        seen.add(key)
        target_id = len(records)
        stratum = f"parent:{parent_id}:f{min(4, int(fraction * 5))}"
        records.append(
            {
                "target_id": int(target_id),
                "source_parent_node_id": int(parent_id),
                "source_node_1": int(left_id),
                "source_node_2": int(right_id),
                "interpolation_fraction": float(fraction),
                "sampling_stratum": stratum,
                "xyz_key": key,
                "shard_id": int(
                    hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16
                )
                % int(shard_count),
                **{name: float(value) for name, value in zip(XYZ_COLUMNS, xyz, strict=True)},
            }
        )
    return pd.DataFrame.from_records(records)


def exact_nested_dataset(
    existing: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    target_rows: int,
    allowed_qualities: Sequence[str],
) -> pd.DataFrame:
    """Select a deterministic exact prefix without padding or duplicates."""

    allowed = set(map(str, allowed_qualities))
    combined = pd.concat([existing, generated], ignore_index=True, sort=False)
    combined = combined[combined["label_quality"].astype(str).isin(allowed)].copy()
    if "xyz_key" not in combined:
        combined["xyz_key"] = (
            combined.loc[:, XYZ_COLUMNS].round(12).astype(str).agg("|".join, axis=1)
        )
    quality_rank = combined["label_quality"].map({"Gold": 0, "Silver": 1}).fillna(9)
    combined = (
        combined.assign(_quality_rank=quality_rank)
        .sort_values(
            ["_quality_rank", "source_parent_node_id", "sampling_stratum", "xyz_key"],
            kind="stable",
        )
        .drop_duplicates("xyz_key", keep="first")
        .drop(columns="_quality_rank")
    )
    if len(combined) < int(target_rows):
        return combined.reset_index(drop=True)
    return combined.head(int(target_rows)).reset_index(drop=True)
