from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.bacra_dataset import (
    SpatialSplitPolicy,
    assign_spatial_splits,
    detect_cross_chart_conflicts,
    equal_count_ablation,
    make_nested_datasets,
)
from quasi_exp.teacher.canonical import TeacherPolicy
from quasi_exp.teacher.dense_chart_sampling import (
    BETA_COLUMNS,
    DenseSamplingPolicy,
    chart_fill_distance_metrics,
    densify_chart,
)


class LinearEnvironment:
    def __init__(self) -> None:
        self._bounds = np.column_stack(
            [np.full(6, -0.5, dtype=float), np.full(6, 0.5, dtype=float)]
        )

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    def fk(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
        return beta[:, :3]

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
        del beta_rad
        return np.column_stack([np.eye(3), np.zeros((3, 3))])


def _dataset(count: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    xyz = rng.uniform(0.01, 0.19, size=(count, 3))
    beta = np.column_stack([xyz, np.zeros((count, 3))])
    return pd.DataFrame(
        np.column_stack([xyz, beta]),
        columns=["x_m", "y_m", "z_m", *BETA_COLUMNS],
    )


def test_dense_dual_anchor_corrector_emits_only_verified_gold_rows() -> None:
    environment = LinearEnvironment()
    xyz = np.asarray(
        [
            [0.04, 0.04, 0.04],
            [0.08, 0.04, 0.04],
            [0.04, 0.08, 0.04],
            [0.04, 0.04, 0.08],
            [0.08, 0.08, 0.08],
        ]
    )
    beta = np.column_stack([xyz, np.zeros((len(xyz), 3))])
    policy = DenseSamplingPolicy(
        row_count=40,
        attempt_multiplier=2,
        seed=11,
        candidate_policy=TeacherPolicy(
            tracking_tolerance_mm=0.01,
            safe_joint_margin_deg=1.5,
            safe_margin_repulsion_step_deg=0.0,
            max_corrector_iterations=10,
        ),
    )
    result = densify_chart(
        environment, xyz, beta, chart_id=3, policy=policy
    )
    assert result.report["gate_pass"] is True
    assert len(result.rows) == 40
    assert result.rows["accepted"].all()
    assert result.rows["residual_mm"].max() <= policy.residual_max_mm
    assert result.rows["joint_margin_deg"].min() >= policy.gold_margin_deg
    assert result.rows["dual_anchor_gap_deg"].max() <= policy.dual_anchor_gap_deg
    fill = chart_fill_distance_metrics(
        result.rows[["x_m", "y_m", "z_m"]].to_numpy(), xyz
    )
    assert fill["nn_max_mm"] >= 0.0


def test_spatial_blocks_are_atomic_and_sealed_rows_require_token() -> None:
    data = _dataset(300)
    split = assign_spatial_splits(
        data,
        SpatialSplitPolicy(
            macro_voxel_mm=20.0,
            train_fraction=0.5,
            validation_fraction=0.25,
            sealed_fraction=0.25,
            buffer_mm=0.0,
            seed=23,
        ),
    )
    combined = pd.concat(
        [split.public_rows, split.open_sealed(split.seal_token)], ignore_index=True
    )
    assert len(combined) == len(data)
    assert combined.groupby("spatial_block_id")["split_base_role"].nunique().max() == 1
    with pytest.raises(PermissionError):
        split.open_sealed("wrong-token")


def test_conflicts_and_nested_ablation_are_fail_closed_and_deterministic() -> None:
    data = _dataset(80)
    conflicting = pd.concat([data.iloc[:1], data.iloc[:1]], ignore_index=True)
    conflicting["chart_id"] = [0, 1]
    conflicting.loc[1, "beta1_rad"] += np.deg2rad(3.0)
    rows, report = detect_cross_chart_conflicts(
        conflicting, voxel_mm=2.0, beta_gap_deg=1.0
    )
    assert report["gate_pass"] is False
    assert bool(rows.iloc[0]["conflict"]) is True

    nested = make_nested_datasets(data, [20, 40], seed=31)
    assert nested[20]["nested_order"].tolist() == nested[40]["nested_order"].iloc[:20].tolist()

    methods = {name: data.copy() for name in ("D0", "D1", "D2", "D3")}
    equal = equal_count_ablation(methods, row_count=30, seed=41)
    assert {name: len(frame) for name, frame in equal.items()} == {
        "D0": 30,
        "D1": 30,
        "D2": 30,
        "D3": 30,
    }
