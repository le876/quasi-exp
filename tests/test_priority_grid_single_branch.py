from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analysis" / "select_priority_grid_single_branch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_priority_grid_single_branch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample(sample_id: int, xyz: tuple[float, float, float], s1: float, s2: float, tension: float) -> tuple[dict, dict]:
    row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
    beta = np.asarray([s1 * 0.1, s1 * 0.1, s2 * 0.1, s2 * 0.1, 0.1, 0.1], dtype=float)
    theta = np.zeros(30, dtype=float)
    theta[0:10:2] = beta[0]
    theta[1:10:2] = beta[1]
    theta[10:20:2] = beta[2]
    theta[11:20:2] = beta[3]
    theta[20:30:2] = beta[4]
    theta[21:30:2] = beta[5]
    for i, value in enumerate(theta):
        row[f"theta_{i + 1}_rad"] = float(value)
    for j in range(12):
        row[f"tension_{j + 1}_n"] = float(tension)
    meta = {
        "sample_id": sample_id,
        "s1": float(s1),
        "s2": float(s2),
        "priority_ratio_pair_id": int(round(1000 * s1 + 100 * s2)),
        "priority_grid_id": sample_id,
        "rms_rnorm": 0.01,
        "max_tension": float(tension),
        "segmented_success": True,
    }
    for i, value in enumerate(beta):
        meta[f"effective_beta_{i + 1}_rad"] = float(value)
    return row, meta


def test_filter_keeps_one_priority_ratio_layer_per_workspace_voxel():
    mod = load_module()
    rows = []
    meta = []
    specs = [
        ((0.001, 0.001, 0.001), 0.125, 0.25, 100.0),
        ((0.002, 0.001, 0.001), 0.0, 0.25, 120.0),
        ((0.003, 0.001, 0.001), 0.0, 0.5, 90.0),
        ((0.021, 0.001, 0.001), 0.125, 0.25, 80.0),
        ((0.022, 0.001, 0.001), 0.25, 0.5, 70.0),
    ]
    for sample_id, spec in enumerate(specs):
        row, item = _sample(sample_id, *spec)
        rows.append(row)
        meta.append(item)

    out_dataset, out_meta, summary = mod.filter_priority_grid_single_branch_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        voxel_size_m=0.01,
    )

    assert summary["input_rows"] == 5
    assert summary["workspace_voxels"] == 2
    assert summary["rows_out"] == 2
    assert set(out_meta["source_sample_id"].tolist()) == {1, 3}
    first_voxel = out_meta[out_meta["workspace_voxel_key"] == "0,0,0"]
    assert set(first_voxel["s1"].round(6).tolist()) == {0.0}
    assert first_voxel["s2"].iloc[0] == 0.25
    assert out_meta["priority_single_branch_selected"].all()


def test_priority_layer_score_prefers_smaller_s1_then_s2_before_tension():
    mod = load_module()
    candidate = pd.DataFrame(
        {
            "s1": [0.0, 0.0, 0.125],
            "s2": [0.5, 0.25, 0.25],
            "max_tension": [50.0, 150.0, 10.0],
            "rms_rnorm": [0.01, 0.01, 0.01],
            "branch_size": [1, 1, 1],
        }
    )

    scores = [mod.priority_ratio_layer_score(row) for _, row in candidate.iterrows()]

    assert scores[1] < scores[0]
    assert scores[1] < scores[2]
