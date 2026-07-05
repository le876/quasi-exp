from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from quasi_exp.model.sampling import beta_to_theta


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "select_canonical_branch_graph.py"
    spec = importlib.util.spec_from_file_location("select_canonical_branch_graph", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(sample_id: int, xyz: tuple[float, float, float], beta: np.ndarray, tension: float) -> tuple[dict, dict]:
    theta = beta_to_theta(beta)
    row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
    for i, value in enumerate(theta):
        row[f"theta_{i + 1}_rad"] = float(value)
    for j in range(12):
        row[f"tension_{j + 1}_n"] = float(tension)
    meta = {
        "sample_id": sample_id,
        "source_component": "test",
        "rms_rnorm": 0.01,
        "max_tension": float(tension),
        "segmented_success": True,
    }
    for i, value in enumerate(beta):
        meta[f"beta{i + 1}_rad"] = float(value)
    return row, meta


def test_graph_policy_pairwise_can_override_local_unary_for_neighbor_voxel() -> None:
    mod = _load_module()
    branch_a = np.array([0.01, 0.01, 0.01, 0.01, 0.08, 0.08], dtype=float)
    branch_b = np.array([0.08, 0.08, 0.08, 0.08, 0.01, 0.01], dtype=float)
    rows: list[dict] = []
    meta: list[dict] = []
    specs = [
        ((0.000, 0.0, 0.0), branch_a, 120.0),
        ((0.000, 0.0, 0.0), branch_b, 100.0),
        ((0.011, 0.0, 0.0), branch_a + 0.001, 125.0),
        ((0.011, 0.0, 0.0), branch_b + 0.001, 90.0),
    ]
    for sample_id, (xyz, beta, tension) in enumerate(specs):
        row, item = _row(sample_id, xyz, beta, tension)
        rows.append(row)
        meta.append(item)

    out_dataset, out_meta, summary = mod.select_canonical_branch_graph_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        voxel_size_m=0.01,
        ball_radius_m=0.015,
        beta_eps_norm=0.20,
        max_icm_iters=8,
        lambda_theta=20.0,
        lambda_beta=20.0,
        lambda_tension=0.0,
    )

    assert set(out_meta["source_sample_id"].tolist()) == {0, 2}
    assert summary["rows_out"] == 2
    assert summary["icm_iterations"] >= 1
    assert summary["graph_edges"] >= 1
    assert summary["graph_components"] == 1
    assert out_dataset["sample_id"].tolist() == [0, 1]
    assert out_meta["graph_component_id"].nunique() == 1
    assert {"effective_beta_1_rad", "graph_branch_label", "branch_policy_version"}.issubset(out_meta.columns)
