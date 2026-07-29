from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_residual_composite_predictor_blends_outputs() -> None:
    from scripts.analysis.run_bacra_v12_14_region_growth import (
        _ResidualCompositePredictor,
    )

    class Predictor:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(
            self, target: np.ndarray, *, batch_size: int, verbose: int
        ) -> np.ndarray:
            assert batch_size == 64
            assert verbose == 0
            return np.full((len(target), 2), self.value)

    predictor = _ResidualCompositePredictor(
        Predictor(2.0), Predictor(5.0), alpha=0.25
    )
    actual = predictor.predict(np.zeros((3, 4)), batch_size=64, verbose=0)
    np.testing.assert_allclose(actual, np.full((3, 2), 2.75))


def test_family_seed_gate_uses_registered_two_of_three_semantics() -> None:
    from scripts.analysis.run_bacra_v12_14_region_growth import (
        _family_seed_gate,
    )

    metrics = pd.DataFrame(
        {
            "family_id": ["a", "a", "a", "b", "b", "b"],
            "seed_gate_pass": [True, True, False, True, False, False],
        }
    )
    actual = _family_seed_gate(
        metrics,
        family_column="family_id",
        pass_column="seed_gate_pass",
        required_seed_passes=2,
    )
    assert actual.to_dict() == {"a": True, "b": False}
