from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "baselines" / "run_beta_first_baselines.py"
    spec = importlib.util.spec_from_file_location("run_beta_first_baselines", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_frames(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    for sample_id in range(n):
        x = sample_id / float(n)
        xyz = (x, x * x, 0.1 * x)
        beta = np.array([x, -x, 0.5 * x, -0.5 * x, 2.0 * x, -2.0 * x], dtype=float) * 0.05
        tension = np.linspace(100.0, 210.0, 12) + beta[4] * 100.0
        row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
        theta = mod_beta_to_theta(beta)
        for j in range(30):
            row[f"theta_{j + 1}_rad"] = float(theta[j])
        for j in range(12):
            row[f"tension_{j + 1}_n"] = float(tension[j])
        rows.append(row)
        m = {"sample_id": sample_id, "source_component": "toy", "rms_rnorm": 0.01, "max_tension": float(tension.max())}
        for j in range(6):
            m[f"beta{j + 1}_rad"] = float(beta[j])
        meta.append(m)
    return pd.DataFrame(rows), pd.DataFrame(meta)


def mod_beta_to_theta(beta: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(beta, dtype=float), 5)


def test_beta_first_pipeline_outputs_beta_theta_tension_metrics(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "beta_to_theta", mod_beta_to_theta)
    dataset, meta = _toy_frames()

    result = mod.evaluate_frames(
        dataset,
        meta,
        split="iid",
        models=["knn"],
        tension_modes=["beta_knn"],
        seed=42,
        val_size=0.1,
        test_size=0.2,
        feature_set="poly_medium",
        beta_source="meta",
    )

    metrics = result["models"]["knn__beta_knn"]["metrics"]["test"]
    assert "beta_mae_deg" in metrics
    assert "theta_mae_deg" in metrics
    assert "tension_mae_n" in metrics
    assert "ee_pos_p95_mm" in metrics
    assert metrics["tension_gt_tmax_ratio"] == 0.0


def test_effective_beta_target_can_ignore_signed_meta() -> None:
    mod = _load_module()
    theta = np.zeros((1, 30), dtype=float)
    expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    for section in range(3):
        start = section * 10
        theta[:, start : start + 10 : 2] = expected[:, 2 * section : 2 * section + 1]
        theta[:, start + 1 : start + 10 : 2] = expected[:, 2 * section + 1 : 2 * section + 2]
    meta = pd.DataFrame({f"beta{i}_rad": [-expected[0, i - 1]] for i in range(1, 7)})

    got = mod.beta_target(meta, theta, beta_source="effective_theta")

    np.testing.assert_allclose(got, expected)
