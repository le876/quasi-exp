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


def test_same_component_beta_knn_excludes_other_components_and_self() -> None:
    mod = _load_module()
    meta = pd.DataFrame(
        {
            "sample_id": [10, 11, 12, 20, 21],
            "source_component": ["a", "a", "a", "b", "b"],
            "beta1_rad": [0.0, 0.01, -0.01, 0.0, 0.01],
            "beta2_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
            "beta3_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
            "beta4_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
            "beta5_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
            "beta6_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    tension = np.arange(5 * 12, dtype=float).reshape(5, 12)
    selected = meta.iloc[[0, 3]].reset_index(drop=True)

    anchors = mod.build_same_component_beta_knn_anchors(
        pool_meta=meta,
        pool_tension=tension,
        selected_meta=selected,
        k_neighbors=2,
    )

    assert anchors[0].neighbor_sample_ids == [11, 12]
    assert anchors[1].neighbor_sample_ids == [21]
    assert np.allclose(anchors[0].anchor_tension_n, np.mean(tension[[1, 2]], axis=0))
    assert np.allclose(anchors[1].anchor_tension_n, tension[4])
