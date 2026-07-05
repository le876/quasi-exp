from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "audit_tension_discontinuity_sources.py"
    spec = importlib.util.spec_from_file_location("audit_tension_discontinuity_sources", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dataset() -> pd.DataFrame:
    rows = []
    xyz = [(0.0, 0.0, 0.0), (0.0, 0.004, 0.0), (0.0, 0.008, 0.0), (0.0, 0.012, 0.0)]
    tensions = [
        np.ones(12) * 100.0,
        np.ones(12) * 110.0,
        np.r_[np.ones(4) * 100.0, np.ones(4) * 400.0, np.ones(4) * 100.0],
        np.ones(12) * 1999.5,
    ]
    for sample_id, ((x, y, z), tension) in enumerate(zip(xyz, tensions)):
        row = {"sample_id": sample_id, "x_m": x, "y_m": y, "z_m": z}
        for i in range(30):
            row[f"theta_{i + 1}_rad"] = 0.001 * sample_id
        for j, value in enumerate(tension):
            row[f"tension_{j + 1}_n"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _meta() -> pd.DataFrame:
    rows = []
    for sample_id in range(4):
        row = {
            "sample_id": sample_id,
            "source_component": "s1_0000_s2_0250",
            "layer_label": "s1_0000_s2_0250",
            "s1": 0.0,
            "s2": 0.25,
        }
        for i in range(6):
            row[f"beta{i + 1}_rad"] = 0.001 * sample_id
            row[f"effective_beta_{i + 1}_rad"] = 0.001 * sample_id
        for j in range(12):
            row[f"case_{j + 1}"] = 1
        rows.append(row)
    rows[2]["case_5"] = 2
    return pd.DataFrame(rows)


def test_audit_groups_pairs_by_case_and_active_set_flips() -> None:
    mod = _load_module()

    payload = mod.audit_frames(
        _dataset(),
        _meta(),
        radius_m=0.010,
        beta_close_threshold_norm=2.0,
        k_neighbors=8,
    )

    assert payload["pair_count"] > 0
    assert payload["groups"]["no_flip"]["pairs"] >= 1
    assert payload["groups"]["case_flip_only"]["pairs"] >= 1
    assert payload["groups"]["active_set_flip_only"]["pairs"] >= 1
    assert payload["groups"]["case_flip_only"]["segment_causing_jump_counts"]["third"] >= 1
    assert payload["groups"]["case_flip_only"]["tension_mae_n_p95"] > payload["groups"]["no_flip"]["tension_mae_n_p95"]
