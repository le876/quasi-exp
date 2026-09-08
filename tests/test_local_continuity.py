from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_local_continuity_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "eval_local_continuity.py"
    spec = importlib.util.spec_from_file_location("eval_local_continuity", mod_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_continuity_reports_tension_neighbor_metrics(tmp_path: Path) -> None:
    module = _load_local_continuity_module()
    n = 12
    rows = []
    for i in range(n):
        row = {
            "sample_id": i,
            "x_m": float(i) * 0.001,
            "y_m": 0.0,
            "z_m": 0.0,
        }
        row.update({f"theta_{j+1}_rad": float(i) * 1e-4 for j in range(30)})
        row.update({f"tension_{j+1}_n": 100.0 + float(i) for j in range(12)})
        rows.append(row)

    dataset = tmp_path / "dataset.parquet"
    pd.DataFrame(rows).to_parquet(dataset)

    res = module.evaluate(dataset, radii_m=[0.01], k_neighbors=6)
    local = res["local"]["<= 10mm"]

    assert local["pairs"] > 0
    assert local["tension_mae_n_p95"] > 0.0
    assert local["tension_max_abs_n_p95"] > 0.0
    assert np.isfinite(local["tension_mae_n_p50"])
