from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "filter_branch_consistent_dataset.py"
    spec = importlib.util.spec_from_file_location("filter_branch_consistent_dataset", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filtered_dataset_schema_preserves_theta_tension_and_reindexes() -> None:
    mod = _load_module()
    rows = []
    meta = []
    for sample_id in range(3):
        beta = np.ones(6) * (0.01 * sample_id)
        row = {"sample_id": sample_id, "x_m": 0.0, "y_m": 0.0, "z_m": float(sample_id) * 0.02}
        for j in range(30):
            row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
        for j in range(12):
            row[f"tension_{j + 1}_n"] = 100.0 + sample_id
        rows.append(row)
        item = {
            "sample_id": sample_id,
            "source_component": "sobol_full",
            "rms_rnorm": 0.01,
            "max_tension": 100.0 + sample_id,
            "segmented_success": True,
            "beta_group1_norm": 0.1,
            "beta_group2_norm": 0.1,
            "beta_group3_norm": 0.8,
            "distal_preference_score": -0.5,
        }
        for j in range(6):
            item[f"beta{j + 1}_rad"] = float(beta[j])
        meta.append(item)

    out_dataset, out_meta, summary = mod.filter_frames(pd.DataFrame(rows), pd.DataFrame(meta), voxel_size_m=0.01)

    assert out_dataset["sample_id"].tolist() == list(range(len(out_dataset)))
    assert out_meta["sample_id"].tolist() == list(range(len(out_meta)))
    assert out_meta["source_sample_id"].tolist() == [0, 1, 2]
    assert len([c for c in out_dataset.columns if c.startswith("theta_") and c.endswith("_rad")]) == 30
    assert len([c for c in out_dataset.columns if c.startswith("tension_") and c.endswith("_n")]) == 12
    assert summary["rows_in"] == 3
    assert summary["rows_out"] == 3

