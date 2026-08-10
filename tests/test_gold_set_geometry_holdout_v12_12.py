from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/analysis"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_bacra_v12_12_independent_gold_set_holdout",
    SCRIPT_DIR / "run_bacra_v12_12_independent_gold_set_holdout.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _report(seed: int, *, passed: bool, fk_p95: float, margin: float):
    family = {
        "validation_fk_p95_mm": fk_p95,
        "validation_fk_max_mm": fk_p95 + 0.5,
        "predicted_minimum_joint_margin_deg": margin,
        "validation_beta_abs_p95_by_joint_deg": [0.2] * 6,
    }
    return {
        "seed": seed,
        "seed_admission_gate_pass": passed,
        "families": [family, family],
    }


def test_model_ranking_uses_only_pre_holdout_v12_11_metrics() -> None:
    ranked = runner._rank_accepted_models(
        [
            _report(4, passed=False, fk_p95=0.1, margin=5.0),
            _report(2, passed=True, fk_p95=2.0, margin=1.8),
            _report(1, passed=True, fk_p95=2.0, margin=1.7),
            _report(3, passed=True, fk_p95=3.0, margin=2.0),
        ]
    )
    assert [row["seed"] for row in ranked] == [2, 1, 3]
    assert all(row["seed"] != 4 for row in ranked)


def test_multiresolution_seed_preserves_cyclic_coarse_knots() -> None:
    coarse = np.arange(24, dtype=float).reshape(4, 6)
    fine = runner._upsample_cyclic_beta(coarse, 16)
    assert fine.shape == (16, 6)
    np.testing.assert_allclose(fine[::4], coarse)
    np.testing.assert_allclose(
        fine[-1], 0.75 * coarse[0] + 0.25 * coarse[-1]
    )
