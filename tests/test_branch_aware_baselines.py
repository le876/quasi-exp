from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "baselines" / "run_branch_aware_baselines.py"
    spec = importlib.util.spec_from_file_location("run_branch_aware_baselines", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    specs = [
        ((0.000, 0.0, 0.0), np.zeros(6), 100.0),
        ((0.001, 0.0, 0.0), np.ones(6) * 0.01, 102.0),
        ((0.002, 0.0, 0.0), np.ones(6) * 0.02, 104.0),
        ((0.000, 0.0, 0.0), np.ones(6), 500.0),
        ((0.001, 0.0, 0.0), np.ones(6) * 1.01, 502.0),
        ((0.002, 0.0, 0.0), np.ones(6) * 1.02, 504.0),
    ]
    for sample_id, (xyz, beta, tension) in enumerate(specs):
        row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
        for j in range(30):
            row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
        for j in range(12):
            row[f"tension_{j + 1}_n"] = tension
        rows.append(row)
        item = {"sample_id": sample_id, "source_component": "toy", "rms_rnorm": 0.01, "max_tension": tension}
        for j in range(6):
            item[f"beta{j + 1}_rad"] = float(beta[j])
        meta.append(item)
    return pd.DataFrame(rows), pd.DataFrame(meta)


def test_branch_labels_split_same_voxel_by_beta_cluster() -> None:
    mod = _load_module()
    dataset, meta = _make_frames()

    labels = mod.build_branch_labels(dataset, meta, voxel_size_m=0.01, beta_eps_norm=0.1, min_branch_size=1)

    assert len(set(labels.tolist())) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_branch_expert_uses_predicted_branch_candidates_only() -> None:
    mod = _load_module()
    dataset, meta = _make_frames()
    labels = mod.build_branch_labels(dataset, meta, voxel_size_m=0.01, beta_eps_norm=0.1, min_branch_size=1)
    y = dataset[[f"tension_{i}_n" for i in range(1, 13)]].to_numpy(dtype=float)

    pred_idx = mod.branch_constrained_nearest_indices(
        train_xyz=dataset[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        train_labels=labels,
        query_xyz=np.asarray([[0.001, 0.0, 0.0]], dtype=float),
        pred_labels=np.asarray([labels[3]]),
    )

    assert int(pred_idx[0]) in {3, 4, 5}
    assert np.mean(np.abs(y[pred_idx[0]] - 502.0)) <= 2.0

