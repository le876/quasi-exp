from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_v12_14_config_and_runner_register_frozen_stages() -> None:
    root = Path(__file__).resolve().parents[1]
    config = (root / "configs/bacra_v12_14_region_growth.yaml").read_text()
    runner = (
        root / "scripts/analysis/run_bacra_v12_14_region_growth.py"
    ).read_text()
    assert "source_capability_pool" in config
    assert "expected_canonical_seed_rows: 84960" in config
    assert "capability_pool_beta_role" in runner
    assert "canonical_seed_dataset.parquet" in runner
    assert '"sparse_growth"' in runner
    assert '"dense_generation"' in runner
    assert "second_anchor_support_distance_mm" in runner
    assert "seed_support_mapping_complete_and_finite" in runner
    assert '"development_random"' in runner
    assert '"active_enrich"' in runner
    assert '"final_test"' in runner
    assert 'config["preset"] == "smoke"' in runner
    assert "requested_workers=1" in runner
    assert 'dense.groupby("voxel_key", sort=True)' in runner


def test_v12_14_batch_proportions_are_explicit() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    values = yaml.safe_load(
        (root / "configs/bacra_v12_14_region_growth.yaml").read_text()
    )["v12_14"]["dataset"]["batch_proportions"]
    assert values["R0"] == {"old": 1.0}
    assert values["R1_full"] == {"interior": 0.75, "boundary": 0.25}
    assert values["R2"] == {
        "old": 0.20,
        "interior": 0.60,
        "boundary": 0.20,
    }
