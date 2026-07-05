from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from quasi_exp.model.sampling import beta_to_theta


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "generate_single_branch_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_single_branch_dataset", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample(sample_id: int, x: float, beta: np.ndarray) -> tuple[dict, dict]:
    theta = beta_to_theta(beta)
    row = {"sample_id": sample_id, "x_m": x, "y_m": 0.0, "z_m": 0.0}
    for i, value in enumerate(theta):
        row[f"theta_{i + 1}_rad"] = float(value)
    for j in range(12):
        row[f"tension_{j + 1}_n"] = 100.0 + sample_id
    meta = {
        "sample_id": sample_id,
        "source_component": "pool",
        "rms_rnorm": 0.01,
        "max_tension": 100.0 + sample_id,
        "graph_branch_selected": True,
        "graph_branch_label": "v0:b0",
    }
    for i, value in enumerate(beta):
        meta[f"beta{i + 1}_rad"] = float(value)
    return row, meta


def test_candidate_pool_builder_balances_workspace_and_preserves_source_ids() -> None:
    mod = _load_module()
    rows: list[dict] = []
    meta: list[dict] = []
    for sample_id in range(12):
        beta = np.array([0.0, 0.0, 0.01, 0.01, 0.08 + sample_id * 0.0001, 0.08], dtype=float)
        row, item = _sample(sample_id, x=0.001 * sample_id, beta=beta)
        rows.append(row)
        meta.append(item)

    out_dataset, out_meta, summary = mod.build_single_branch_from_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        num_samples=6,
        radius_bins=3,
        z_bins=1,
        angle_bins=1,
        seed=123,
    )

    assert len(out_dataset) == 6
    assert out_dataset["sample_id"].tolist() == list(range(6))
    assert out_meta["sample_id"].tolist() == list(range(6))
    assert "source_sample_id" in out_meta.columns
    assert set(out_meta["source_sample_id"]).issubset(set(range(12)))
    assert summary["rows_out"] == 6
    assert summary["workspace_balance"]["radius_bins"] == 3
