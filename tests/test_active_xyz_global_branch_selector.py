from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from quasi_exp.model.sampling import beta_to_theta


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "generate_active_single_branch_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_active_single_branch_dataset", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _candidate(sample_id: int, target_id: int, cand_id: int, xyz: tuple[float, float, float], beta: np.ndarray, unary: float):
    theta = beta_to_theta(beta)
    row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
    for i, value in enumerate(theta):
        row[f"theta_{i + 1}_rad"] = float(value)
    for j in range(12):
        row[f"tension_{j + 1}_n"] = 100.0 + 5.0 * cand_id
    meta = {
        "sample_id": sample_id,
        "active_target_id": target_id,
        "active_candidate_id": cand_id,
        "active_unary_score": unary,
        "xyz_err_m": 0.001,
        "rms_rnorm": 0.01,
        "max_tension": 120.0,
        "segmented_success": True,
    }
    for i, value in enumerate(beta):
        meta[f"effective_beta_{i + 1}_rad"] = float(value)
    return row, meta


def test_global_selector_pairwise_continuity_can_override_local_unary() -> None:
    mod = _load_module()
    branch_a0 = np.array([0.01, 0.01, 0.01, 0.01, 0.08, 0.08], dtype=float)
    branch_a1 = branch_a0 + 0.001
    branch_b0 = np.array([-0.08, -0.08, -0.07, -0.07, 0.01, 0.01], dtype=float)
    branch_b1 = -branch_b0
    rows: list[dict] = []
    meta: list[dict] = []
    specs = [
        (0, 0, 0, (1.0, 0.0, 0.0), branch_a0, 4.0),
        (1, 0, 1, (1.0, 0.0, 0.0), branch_b0, 0.0),
        (2, 1, 0, (1.0, 0.005, 0.0), branch_a1, 0.0),
        (3, 1, 1, (1.0, 0.005, 0.0), branch_b1, 4.0),
    ]
    for spec in specs:
        row, item = _candidate(*spec)
        rows.append(row)
        meta.append(item)

    out_dataset, out_meta, summary = mod.select_global_active_branch_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        k_neighbors=1,
        radius_m=0.02,
        lambda_theta=20.0,
        lambda_beta=20.0,
        lambda_tension=0.0,
        max_icm_iters=8,
    )

    assert summary["selected_targets"] == 2
    assert out_meta["active_candidate_id"].tolist() == [0, 0]
    assert out_meta["active_graph_component_id"].nunique() == 1
    assert out_meta["graph_component_id"].tolist() == out_meta["active_graph_component_id"].tolist()
    assert out_dataset["sample_id"].tolist() == [0, 1]
