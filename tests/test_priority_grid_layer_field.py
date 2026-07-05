from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analysis" / "select_priority_grid_layer_field.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_priority_grid_layer_field", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample(
    sample_id: int,
    xyz: tuple[float, float, float],
    s1: float,
    s2: float,
    tension: float,
    rms_rnorm: float = 0.01,
) -> tuple[dict, dict]:
    beta = np.asarray([s1 * 0.1, s1 * 0.1, s2 * 0.1, s2 * 0.1, 0.1, 0.1], dtype=float)
    theta = np.zeros(30, dtype=float)
    theta[0:10:2] = beta[0]
    theta[1:10:2] = beta[1]
    theta[10:20:2] = beta[2]
    theta[11:20:2] = beta[3]
    theta[20:30:2] = beta[4]
    theta[21:30:2] = beta[5]
    row = {"sample_id": sample_id, "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]}
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
        "rms_rnorm": float(rms_rnorm),
        "max_tension": float(tension),
        "segmented_success": True,
    }
    for i, value in enumerate(beta):
        meta[f"beta{i + 1}_rad"] = float(value)
        meta[f"effective_beta_{i + 1}_rad"] = float(value)
    return row, meta


def test_pairwise_smoothness_can_override_local_layer_preference() -> None:
    mod = load_module()
    rows: list[dict] = []
    meta: list[dict] = []
    specs = [
        ((0.001, 0.0, 0.0), 0.0, 0.25, 140.0),
        ((0.001, 0.0, 0.0), 0.25, 0.50, 60.0),
        ((0.012, 0.0, 0.0), 0.0, 0.25, 145.0),
        ((0.012, 0.0, 0.0), 0.25, 0.50, 55.0),
    ]
    for sample_id, spec in enumerate(specs):
        row, item = _sample(sample_id, *spec)
        rows.append(row)
        meta.append(item)

    out_dataset, out_meta, summary = mod.select_priority_grid_layer_field_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        voxel_size_m=0.01,
        graph_radius_m=0.02,
        graph_k=4,
        max_iters=20,
        w_s1=0.2,
        w_s2=0.2,
        w_tension=2.0,
        lambda_same_layer=50.0,
        lambda_s=50.0,
        lambda_theta=0.0,
        lambda_beta=0.0,
        lambda_tension=0.0,
        restart_fixed_layers=False,
    )

    assert set(out_meta["source_sample_id"].tolist()) == {1, 3}
    assert out_dataset["sample_id"].tolist() == [0, 1]
    assert out_meta["layer_label"].nunique() == 1
    assert out_meta["source_component"].nunique() == 1
    assert out_meta["layer_field_component_id"].nunique() == 1
    assert summary["rows_out"] == 2
    assert summary["graph_edges"] >= 1
    assert summary["icm_iterations"] >= 1


def test_hard_gate_rejects_bad_quality_layer_candidate() -> None:
    mod = load_module()
    rows: list[dict] = []
    meta: list[dict] = []
    specs = [
        ((0.001, 0.0, 0.0), 0.0, 0.25, 40.0, 0.20),
        ((0.001, 0.0, 0.0), 0.125, 0.25, 80.0, 0.01),
    ]
    for sample_id, spec in enumerate(specs):
        row, item = _sample(sample_id, *spec)
        rows.append(row)
        meta.append(item)

    _out_dataset, out_meta, summary = mod.select_priority_grid_layer_field_frames(
        pd.DataFrame(rows),
        pd.DataFrame(meta),
        voxel_size_m=0.01,
        graph_radius_m=0.02,
    )

    assert out_meta["source_sample_id"].tolist() == [1]
    assert out_meta["layer_field_selected"].all()
    assert summary["hard_gate_rows"] == 1
    assert summary["rejected_rows"] == 1
