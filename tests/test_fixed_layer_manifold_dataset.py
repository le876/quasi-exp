from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "generate_fixed_layer_manifold_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_fixed_layer_manifold_dataset", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixed_layer_pool_2k_uses_grid_sobol_lhs_and_preserves_layer_formula() -> None:
    mod = _load_module()

    pool = mod.build_fixed_layer_pool(
        s1=0.125,
        s2=0.5,
        num_samples=2000,
        beta5_deg=(-10.0, 10.0),
        beta6_deg=(-10.0, 10.0),
        seed=123,
    )

    assert len(pool) == 2000
    assert pool["sample_source"].value_counts().to_dict() == {
        "grid": 1681,
        "sobol": 160,
        "lhs": 159,
    }
    assert set(pool["layer_label"]) == {"s1_0125_s2_0500"}
    assert np.allclose(pool["beta1_rad"], 0.125 * pool["beta5_rad"])
    assert np.allclose(pool["beta2_rad"], 0.125 * pool["beta6_rad"])
    assert np.allclose(pool["beta3_rad"], 0.5 * pool["beta5_rad"])
    assert np.allclose(pool["beta4_rad"], 0.5 * pool["beta6_rad"])
    assert float(np.rad2deg(pool["beta5_rad"].abs().max())) <= 10.0 + 1.0e-9
    assert float(np.rad2deg(pool["beta6_rad"].abs().max())) <= 10.0 + 1.0e-9


def test_smoke_fixed_layer_pool_splits_non_2k_counts_deterministically() -> None:
    mod = _load_module()

    first = mod.build_fixed_layer_pool(s1=0.0, s2=0.25, num_samples=40, seed=2026)
    second = mod.build_fixed_layer_pool(s1=0.0, s2=0.25, num_samples=40, seed=2026)

    assert len(first) == 40
    assert first["sample_source"].value_counts().to_dict() == {
        "grid": 34,
        "sobol": 3,
        "lhs": 3,
    }
    assert first[[f"beta{i}_rad" for i in range(1, 7)]].equals(second[[f"beta{i}_rad" for i in range(1, 7)]])
