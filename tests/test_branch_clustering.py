from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "eval_branch_clustering.py"
    spec = importlib.util.spec_from_file_location("eval_branch_clustering", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_branch_clustering_reports_within_and_between_branch_tension() -> None:
    mod = _load_module()
    rows = []
    meta = []
    betas = [np.zeros(6), np.ones(6) * 0.02, np.ones(6), np.ones(6) * 1.02]
    tensions = [100.0, 104.0, 500.0, 508.0]
    for i, beta in enumerate(betas):
        row = {"sample_id": i, "x_m": 0.0, "y_m": 0.0, "z_m": float(i) * 0.001}
        for j in range(30):
            row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
        for j in range(12):
            row[f"tension_{j + 1}_n"] = tensions[i]
        rows.append(row)
        m = {"sample_id": i, "source_component": "mixed", "rms_rnorm": 0.01, "max_tension": tensions[i]}
        for j in range(6):
            m[f"beta{j + 1}_rad"] = float(beta[j])
        meta.append(m)

    result = mod.evaluate_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        radius_m=0.01,
        min_ball_size=4,
        beta_eps_norm=0.1,
    )

    assert result["balls_evaluated"] == 4
    assert result["branch_count_p50"] == 2.0
    assert result["multi_branch_ball_ratio"] == 1.0
    assert result["within_branch_tension_mae_n_p50"] <= 8.0
    assert result["between_branch_tension_mae_n_p50"] >= 396.0
    assert result["between_branch_variance_ratio_p50"] > 0.95

