"""Leakage-safe feature contracts for the V10 probe students."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd


class StudentKind(str, Enum):
    S0 = "S0"
    S2 = "S2"
    S4 = "S4"


XYZ_COLUMNS = ("target_x_m", "target_y_m", "target_z_m")
BETA_COLUMNS = tuple(f"teacher_beta{index}_rad" for index in range(1, 7))
PREVIOUS_BETA_COLUMNS = tuple(f"previous_beta{index}_rad" for index in range(1, 7))


@dataclass(frozen=True)
class StudentArrays:
    features: np.ndarray
    targets: np.ndarray
    chart_labels: np.ndarray | None = None


def split_by_trajectory(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 20260720,
) -> Mapping[str, pd.DataFrame]:
    if "trajectory_id" not in frame:
        raise ValueError("student dataset must contain trajectory_id")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum below one")
    ids = np.asarray(sorted(frame["trajectory_id"].astype(str).unique()))
    if len(ids) < 3:
        raise ValueError("at least three trajectories are required for leakage-safe splits")
    rng = np.random.default_rng(int(seed))
    ids = ids[rng.permutation(len(ids))]
    validation_count = max(1, int(round(len(ids) * validation_fraction)))
    test_count = max(1, int(round(len(ids) * test_fraction)))
    if validation_count + test_count >= len(ids):
        validation_count = test_count = 1
    validation_ids = set(ids[:validation_count])
    test_ids = set(ids[validation_count : validation_count + test_count])
    train_ids = set(ids[validation_count + test_count :])
    trajectory = frame["trajectory_id"].astype(str)
    return {
        "train": frame[trajectory.isin(train_ids)].copy().reset_index(drop=True),
        "validation": frame[trajectory.isin(validation_ids)].copy().reset_index(drop=True),
        "test": frame[trajectory.isin(test_ids)].copy().reset_index(drop=True),
    }


def build_student_arrays(frame: pd.DataFrame, kind: StudentKind | str) -> StudentArrays:
    student_kind = StudentKind(kind)
    required = {"trajectory_id", *XYZ_COLUMNS, *BETA_COLUMNS}
    if student_kind == StudentKind.S2:
        required.update(PREVIOUS_BETA_COLUMNS)
    if student_kind == StudentKind.S4:
        required.add("chart_id")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"student dataset is missing columns: {sorted(missing)}")
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float64)
    beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=np.float64)
    if student_kind in {StudentKind.S0, StudentKind.S4}:
        return StudentArrays(
            features=xyz,
            targets=beta,
            chart_labels=(
                frame["chart_id"].to_numpy(dtype=np.int64)
                if student_kind == StudentKind.S4
                else None
            ),
        )
    previous_beta = frame.loc[:, PREVIOUS_BETA_COLUMNS].to_numpy(dtype=np.float64)
    previous_xyz = (
        frame.groupby("trajectory_id", sort=False, group_keys=False)[list(XYZ_COLUMNS)]
        .shift(1)
        .to_numpy(dtype=np.float64)
    )
    first = np.isnan(previous_xyz).any(axis=1)
    previous_xyz[first] = xyz[first]
    delta_xyz = xyz - previous_xyz
    return StudentArrays(
        features=np.column_stack([xyz, delta_xyz, previous_beta]),
        targets=beta - previous_beta,
    )
