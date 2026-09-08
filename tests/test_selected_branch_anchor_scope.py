from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "relabel_anchor_canonical.py"
    spec = importlib.util.spec_from_file_location("relabel_anchor_canonical", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selected_branch_effective_beta_anchors_do_not_cross_graph_component() -> None:
    mod = _load_module()
    meta = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "source_sample_id": [0, 1, 2, 3],
            "source_component": ["same"] * 4,
            "graph_component_id": [10, 10, 20, 20],
            "graph_branch_selected": [True] * 4,
            "effective_beta_1_rad": [0.0, 0.01, 0.0, 0.01],
            "effective_beta_2_rad": [0.0, 0.0, 0.0, 0.0],
            "effective_beta_3_rad": [0.0, 0.0, 0.0, 0.0],
            "effective_beta_4_rad": [0.0, 0.0, 0.0, 0.0],
            "effective_beta_5_rad": [0.0, 0.0, 0.0, 0.0],
            "effective_beta_6_rad": [0.0, 0.0, 0.0, 0.0],
        }
    )
    for i in range(1, 7):
        meta[f"beta{i}_rad"] = meta[f"effective_beta_{i}_rad"]
    tension = np.arange(4 * 12, dtype=float).reshape(4, 12)

    anchors = mod.build_selected_branch_graph_anchors(
        pool_meta=meta,
        pool_tension=tension,
        selected_meta=meta.iloc[[0, 2]].reset_index(drop=True),
        k_neighbors=2,
        distance_space="effective_beta",
        no_cross_branch_anchor=True,
    )

    assert anchors[0].neighbor_sample_ids == [1]
    assert anchors[1].neighbor_sample_ids == [3]
    assert np.allclose(anchors[0].anchor_tension_n, tension[1])
    assert np.allclose(anchors[1].anchor_tension_n, tension[3])
