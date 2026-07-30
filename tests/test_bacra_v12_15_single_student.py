from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _block_fixture() -> pd.DataFrame:
    rows = []
    for block in range(12):
        for local in range(4):
            rows.append(
                {
                    "x_m": 0.020 * block + 0.0002 * local,
                    "y_m": 0.001 * local,
                    "z_m": 0.0,
                    "sampling_bucket": (
                        "old" if local == 0 else "interior"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_whole_spatial_block_partition_is_deterministic_and_disjoint() -> None:
    from quasi_exp.teacher.retention_distillation import (
        SpatialBlockPolicy,
        assign_whole_spatial_blocks,
    )

    frame = _block_fixture()
    region = frame["sampling_bucket"].ne("old").to_numpy()
    policy = SpatialBlockPolicy(
        macro_voxel_mm=15.0,
        validation_fraction=0.2,
        sealed_holdout_fraction=0.2,
        train_buffer_mm=0.0,
    )
    first, report = assign_whole_spatial_blocks(
        frame, region_mask=region, policy=policy, seed=17
    )
    second, second_report = assign_whole_spatial_blocks(
        frame, region_mask=region, policy=policy, seed=17
    )
    assert first["v12_15_split"].tolist() == second["v12_15_split"].tolist()
    assert report["sealed_block_keys"] == second_report["sealed_block_keys"]
    roles_per_block = first.groupby("spatial_block_key")[
        "v12_15_split"
    ].nunique()
    assert int(roles_per_block.max()) == 1
    sealed = set(report["sealed_block_keys"])
    validation = set(report["validation_block_keys"])
    assert sealed
    assert validation
    assert sealed.isdisjoint(validation)


def test_training_materialization_excludes_every_sealed_block() -> None:
    from quasi_exp.teacher.retention_distillation import (
        SpatialBlockPolicy,
        assign_whole_spatial_blocks,
    )

    frame = _block_fixture()
    partitioned, report = assign_whole_spatial_blocks(
        frame,
        region_mask=frame["sampling_bucket"].ne("old").to_numpy(),
        policy=SpatialBlockPolicy(train_buffer_mm=0.0),
        seed=23,
    )
    usable = partitioned.loc[
        partitioned["v12_15_split"].isin(["train", "validation"])
    ]
    assert not usable["spatial_block_key"].isin(
        report["sealed_block_keys"]
    ).any()


def test_registered_point_gate_requires_fk_margin_and_bounds() -> None:
    from scripts.analysis.run_bacra_v12_15_single_student import (
        _apply_point_gate,
    )

    config = {
        "v12_15": {
            "selection": {
                "fk_p95_mm": 5.0,
                "fk_max_mm": 10.0,
                "minimum_joint_margin_deg": 1.5,
            }
        }
    }
    metrics = pd.DataFrame(
        [
            {
                "fk_p95_mm": 4.9,
                "fk_max_mm": 9.9,
                "minimum_joint_margin_deg": 1.5,
                "actual_bounds": True,
            },
            {
                "fk_p95_mm": 5.1,
                "fk_max_mm": 9.9,
                "minimum_joint_margin_deg": 1.5,
                "actual_bounds": True,
            },
            {
                "fk_p95_mm": 4.9,
                "fk_max_mm": 9.9,
                "minimum_joint_margin_deg": 1.49,
                "actual_bounds": True,
            },
        ]
    )
    gated = _apply_point_gate(metrics, config)
    assert gated["seed_gate_pass"].tolist() == [True, False, False]


def test_v12_15_config_freezes_single_chart_and_sealed_holdout() -> None:
    import run_bacra_v12 as v12

    root = Path(__file__).resolve().parents[1]
    config = v12.load_protocol_config(
        root / "configs/bacra_v12_15_single_student.yaml", "formal"
    )
    assert (
        config["claim_scope"]
        == "simulation_single_chart_spatial_block_generalization"
    )
    assert config["deployment_claim_gate_pass"] is False
    assert (
        config["v12_15"]["spatial_holdout"]["sealed_holdout_fraction"]
        == 0.15
    )
    assert config["v12_15"]["parallel"]["gpu_training_workers"] == 1
    assert set(config["v12_15"]["distillation"]["variants"]) == {
        "S0_balanced",
        "S1_teacher_anchored",
    }


def test_final8_reference_joins_registered_semiaxis_catalog() -> None:
    import run_bacra_v12 as v12
    from scripts.analysis.run_bacra_v12_15_single_student import (
        _final8_reference,
    )

    root = Path(__file__).resolve().parents[1]
    project = Path("/mnt/ML_projects/quasi_exp")
    config = v12.load_protocol_config(
        root / "configs/bacra_v12_15_single_student.yaml", "formal"
    )
    reference = _final8_reference(config, project)
    assert len(reference) == 8 * 720
    assert reference["family_id"].nunique() == 8
    assert np.isfinite(reference["major_semiaxis_m"]).all()
