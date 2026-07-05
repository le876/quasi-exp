from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "eval_branch_aware_continuity.py"
    spec = importlib.util.spec_from_file_location("eval_branch_aware_continuity", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    xyz = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.004),
        (0.0, 0.0, 0.006),
        (0.1, 0.0, 0.0),
    ]
    beta = [
        np.zeros(6),
        np.ones(6) * 0.01,
        np.ones(6) * 0.8,
        np.ones(6) * 0.02,
    ]
    comps = ["sobol_full", "sobol_full", "lhs_full", "sobol_full"]
    tensions = [
        np.ones(12) * 100.0,
        np.ones(12) * 110.0,
        np.ones(12) * 500.0,
        np.ones(12) * 120.0,
    ]
    for i in range(4):
        row = {"sample_id": i, "x_m": xyz[i][0], "y_m": xyz[i][1], "z_m": xyz[i][2]}
        for j in range(30):
            row[f"theta_{j + 1}_rad"] = float(beta[i][j // 5])
        for j in range(12):
            row[f"tension_{j + 1}_n"] = float(tensions[i][j])
        rows.append(row)

        m = {"sample_id": i, "source_component": comps[i], "rms_rnorm": 0.01, "max_tension": float(tensions[i].max())}
        for j in range(6):
            m[f"beta{j + 1}_rad"] = float(beta[i][j])
        meta.append(m)
    return pd.DataFrame(rows), pd.DataFrame(meta)


def test_branch_aware_continuity_separates_same_component_and_beta_far_pairs() -> None:
    mod = _load_module()
    dataset, meta = _toy_frames()

    result = mod.evaluate_frames(
        dataset,
        meta,
        radii_m=[0.01],
        beta_k_values=[2],
        xyz_k_neighbors=4,
        beta_close_threshold_norm=0.1,
    )

    groups = result["groups"]
    assert groups["same_component_xyz_<=10mm"]["pairs"] == 1
    assert groups["cross_component_xyz_<=10mm"]["pairs"] == 2
    assert groups["xyz_<=10mm_beta_close"]["pairs"] == 1
    assert groups["xyz_<=10mm_beta_far"]["pairs"] == 2
    assert groups["same_component_xyz_<=10mm"]["tension_mae_n_p50"] == 10.0
    assert groups["cross_component_xyz_<=10mm"]["tension_mae_n_p50"] >= 390.0

