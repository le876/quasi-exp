from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "eval_source_ablation.py"
    spec = importlib.util.spec_from_file_location("eval_source_ablation", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(sample_id: int, source: str, z_m: float, beta: np.ndarray, tension: float) -> tuple[dict, dict]:
    row = {"sample_id": sample_id, "x_m": 0.0, "y_m": 0.0, "z_m": z_m}
    for j in range(30):
        row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
    for j in range(12):
        row[f"tension_{j + 1}_n"] = tension
    meta = {"sample_id": sample_id, "source_component": source, "rms_rnorm": 0.01, "max_tension": tension}
    for j in range(6):
        meta[f"beta{j + 1}_rad"] = float(beta[j])
    return row, meta


def test_source_ablation_ranks_cross_source_beta_far_mixing() -> None:
    mod = _load_module()
    rows = []
    meta = []
    specs = [
        ("sobol_full", 0.000, np.zeros(6), 100.0),
        ("sobol_full", 0.004, np.ones(6) * 0.01, 105.0),
        ("workspace_balanced", 0.005, np.ones(6) * 0.9, 500.0),
        ("workspace_balanced", 0.006, np.ones(6) * 0.95, 520.0),
        ("distal_biased", 0.100, np.ones(6) * 0.02, 120.0),
    ]
    for sample_id, spec in enumerate(specs):
        row, item = _row(sample_id, *spec)
        rows.append(row)
        meta.append(item)

    result = mod.evaluate_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        radius_m=0.01,
        beta_close_threshold_norm=0.2,
    )

    assert result["sources"]["sobol_full"]["within_source"]["pairs"] == 1
    assert result["source_pair_matrix"]["sobol_full|workspace_balanced"]["pairs"] == 4
    assert result["source_pair_matrix"]["sobol_full|workspace_balanced"]["beta_far_ratio"] == 1.0
    assert result["source_risk_ranking"][0]["source"] == "workspace_balanced"
    assert result["source_risk_ranking"][0]["risk_score"] > result["source_risk_ranking"][-1]["risk_score"]

