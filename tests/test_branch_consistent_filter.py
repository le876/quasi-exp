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


def _sample(sample_id: int, beta: np.ndarray, tension: float, rms: float = 0.01) -> tuple[dict, dict]:
    row = {"sample_id": sample_id, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0}
    for j in range(30):
        row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
    for j in range(12):
        row[f"tension_{j + 1}_n"] = tension
    g1 = float(np.sqrt(np.mean(np.square(beta[[0, 1]]))))
    g2 = float(np.sqrt(np.mean(np.square(beta[[2, 3]]))))
    g3 = float(np.sqrt(np.mean(np.square(beta[[4, 5]]))))
    meta = {
        "sample_id": sample_id,
        "source_component": "mixed",
        "rms_rnorm": rms,
        "max_tension": tension,
        "segmented_success": True,
        "beta_group1_norm": g1,
        "beta_group2_norm": g2,
        "beta_group3_norm": g3,
        "distal_preference_score": g1**2 + g2**2 - 1.25 * g3**2,
    }
    for j in range(6):
        meta[f"beta{j + 1}_rad"] = float(beta[j])
    return row, meta


def test_branch_filter_keeps_all_samples_from_lowest_score_branch() -> None:
    mod = _load_module()
    distal = np.array([0.01, 0.01, 0.01, 0.01, 0.8, 0.8])
    proximal = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1])
    rows = []
    meta = []
    for sample_id, beta in enumerate([distal, distal + 0.01, proximal, proximal + 0.01]):
        row, item = _sample(sample_id, beta, tension=100.0 + sample_id)
        rows.append(row)
        meta.append(item)

    out_dataset, out_meta, summary = mod.filter_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        voxel_size_m=0.01,
        beta_eps_norm=0.2,
    )

    assert len(out_dataset) == 2
    assert summary["rows_out"] == 2
    assert set(out_meta["source_sample_id"].tolist()) == {0, 1}
    assert out_meta["canonical_branch_selected"].all()
    assert set(out_meta["filter_policy_version"].tolist()) == {"distal_v1"}
    assert {"workspace_voxel_key", "branch_label_in_voxel", "branch_score"}.issubset(out_meta.columns)

