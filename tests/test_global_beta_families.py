from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "eval_global_beta_families.py"
    spec = importlib.util.spec_from_file_location("eval_global_beta_families", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    for sample_id in range(12):
        family = sample_id // 4
        beta = np.ones(6) * float(family)
        beta += (sample_id % 4) * 0.01
        xyz = (float(family), float(sample_id % 4) * 0.01, 0.0)
        tension = 100.0 + family * 100.0 + (sample_id % 4)
        row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
        for j in range(30):
            row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
        for j in range(12):
            row[f"tension_{j + 1}_n"] = tension
        rows.append(row)
        m = {"sample_id": sample_id, "source_component": "toy", "rms_rnorm": 0.01, "max_tension": tension}
        for j in range(6):
            m[f"beta{j + 1}_rad"] = float(beta[j])
        meta.append(m)
    return pd.DataFrame(rows), pd.DataFrame(meta)


def test_global_beta_family_sweep_recommends_supported_classifiable_k() -> None:
    mod = _load_module()
    dataset, meta = _toy_frames()

    result = mod.evaluate_frames(
        dataset,
        meta,
        k_values=[3, 6],
        min_family_size_gate=3,
        classifier_accuracy_gate=0.5,
        same_family_tension_p95_gate_n=20.0,
        radius_m=0.05,
        seed=7,
        beta_source="meta",
    )

    assert result["recommended_k"] == 3
    assert result["families"]["k3"]["passed"] is True
    assert result["families"]["k6"]["passed"] is False
    assert result["families"]["k3"]["family_size_min"] >= 3
    assert result["families"]["k3"]["same_family_10mm"]["tension_mae_n_p95"] <= 20.0


def test_effective_beta_from_theta_uses_odd_even_sections() -> None:
    mod = _load_module()
    theta = np.zeros((2, 30), dtype=float)
    expected = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
    )
    for row, beta in enumerate(expected):
        for section in range(3):
            start = section * 10
            theta[row, start : start + 10 : 2] = beta[2 * section]
            theta[row, start + 1 : start + 10 : 2] = beta[2 * section + 1]

    got = mod.effective_beta_from_theta(theta)

    np.testing.assert_allclose(got, expected)
