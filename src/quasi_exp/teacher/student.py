"""Leakage-safe feature contracts for the V10 probe students."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quasi_exp.teacher.experiment import atomic_write_json, sha256_file


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


def periodic_beta_interpolation(
    frame: pd.DataFrame,
    *,
    phase_count: int,
) -> np.ndarray:
    """Interpolate a cyclic teacher beta path onto an endpoint-free grid."""

    missing = {"phase_rad", *BETA_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"teacher initialisation frame is missing columns: {sorted(missing)}")
    ordered = frame.sort_values("phase_rad", kind="stable")
    phase = ordered["phase_rad"].to_numpy(dtype=float)
    if len(phase) < 4 or not np.isfinite(phase).all():
        raise ValueError("teacher initialisation requires at least four finite phases")
    query = np.linspace(0.0, 2.0 * np.pi, int(phase_count), endpoint=False)
    return np.column_stack(
        [
            np.interp(
                query,
                phase,
                ordered[column].to_numpy(dtype=float),
                period=2.0 * np.pi,
            )
            for column in BETA_COLUMNS
        ]
    )


def assign_dense_phase_split(
    frame: pd.DataFrame,
    *,
    expected_phase_count: int = 720,
    residual_limit_mm: float = 3.0,
) -> pd.DataFrame:
    """Assign the frozen interleaved train/validation/test split.

    Only finite, in-bounds teacher labels at or below the registered residual
    limit may be consumed by training.  Validation and test labels remain
    ineligible for fitting even when their teacher solution is valid.
    """

    required = {"phase_idx", "teacher_fk_residual_mm", "within_joint_bounds"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dense student frame is missing columns: {sorted(missing)}")
    ordered = frame.sort_values("phase_idx", kind="stable").reset_index(drop=True).copy()
    expected = int(expected_phase_count)
    phase = ordered["phase_idx"].to_numpy(dtype=np.int64)
    if len(ordered) != expected or not np.array_equal(phase, np.arange(expected)):
        raise ValueError(f"dense student frame must contain phase_idx 0..{expected - 1}")
    residual = ordered["teacher_fk_residual_mm"].to_numpy(dtype=float)
    eligible = (
        np.isfinite(residual)
        & (residual <= float(residual_limit_mm))
        & ordered["within_joint_bounds"].astype(bool).to_numpy()
    )
    modulo = phase % 4
    ordered["split"] = np.where(
        modulo % 2 == 0,
        "train",
        np.where(modulo == 1, "validation", "test"),
    )
    ordered["label_eligible"] = eligible
    ordered["used_for_training"] = eligible & ordered["split"].eq("train").to_numpy()
    return ordered


def materialize_phase_splits(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write train/validation/test labels as separately hashed artifacts."""

    required = {"phase_idx", "split", "label_eligible", "used_for_training"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"assigned student frame is missing columns: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "train.parquet",
        "validation": "validation.parquet",
        "test": "sealed_test.parquet",
    }
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split, filename in names.items():
        part = frame.loc[frame["split"].astype(str).eq(split)].copy()
        if part.empty:
            raise ValueError(f"student split {split} is empty")
        path = output / filename
        part.to_parquet(path, index=False, compression="zstd")
        paths[split] = path
        counts[split] = int(len(part))
    if set(frame.loc[frame["used_for_training"].astype(bool), "split"]) != {"train"}:
        raise ValueError("used_for_training contains validation or test labels")
    report: dict[str, object] = {
        "split_counts": counts,
        "eligible_counts": {
            split: int(
                frame.loc[frame["split"].astype(str).eq(split), "label_eligible"]
                .astype(bool)
                .sum()
            )
            for split in names
        },
        "split_paths": {split: str(path.resolve()) for split, path in paths.items()},
        "split_sha256": {split: sha256_file(path) for split, path in paths.items()},
        "sealed_test_label_access": "post_selection_only",
    }
    atomic_write_json(output / "split_report.json", report)
    return report


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
