from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "eval_oracle_floor.py"
    spec = importlib.util.spec_from_file_location("eval_oracle_floor", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_row(sample_id: int, xyz: tuple[float, float, float], beta: np.ndarray, tension: float, branch: int) -> tuple[dict, dict]:
    row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
    for j in range(30):
        row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
    for j in range(12):
        row[f"tension_{j + 1}_n"] = tension
    meta = {
        "sample_id": sample_id,
        "source_component": f"branch_{branch}",
        "branch_label": branch,
        "rms_rnorm": 0.01,
        "max_tension": tension,
    }
    for j in range(6):
        meta[f"beta{j + 1}_rad"] = float(beta[j])
    return row, meta


def test_branch_aware_oracle_uses_branch_label_instead_of_nearest_xyz_only() -> None:
    mod = _load_module()
    rows = []
    meta = []
    specs = [
        ((0.000, 0.0, 0.0), np.zeros(6), 100.0, 0),
        ((0.001, 0.0, 0.0), np.ones(6), 500.0, 1),
        ((0.002, 0.0, 0.0), np.ones(6) * 0.02, 105.0, 0),
        ((0.100, 0.0, 0.0), np.ones(6) * 1.02, 510.0, 1),
    ]
    for i, spec in enumerate(specs):
        row, m = _make_row(i, *spec)
        rows.append(row)
        meta.append(m)

    result = mod.evaluate_frames(pd.DataFrame(rows), pd.DataFrame(meta), branch_column="branch_label")

    assert result["xyz_nn_oracle"]["tension_mae_n"] > 100.0
    assert result["branch_aware_xyz_nn_oracle"]["tension_mae_n"] < result["xyz_nn_oracle"]["tension_mae_n"]
    assert result["beta_nn_oracle"]["tension_mae_n"] <= result["branch_aware_xyz_nn_oracle"]["tension_mae_n"]

