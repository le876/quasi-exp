from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "relabel_tension_graph_canonical.py"
    spec = importlib.util.spec_from_file_location("relabel_tension_graph_canonical", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_graph_anchors_do_not_cross_layer_labels() -> None:
    mod = _load_module()
    meta = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "source_sample_id": [0, 1, 2, 3],
            "layer_label": ["a", "a", "b", "b"],
            "x_m": [0.0, 0.001, 0.0, 0.001],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    for i in range(6):
        meta[f"effective_beta_{i + 1}_rad"] = [0.0, 0.01, 0.0, 0.01]
        meta[f"beta{i + 1}_rad"] = meta[f"effective_beta_{i + 1}_rad"]
    tension = np.arange(4 * 12, dtype=float).reshape(4, 12)

    anchors = mod.build_layer_graph_anchors(
        pool_meta=meta,
        pool_tension=tension,
        selected_meta=meta.iloc[[0, 2]].reset_index(drop=True),
        k_neighbors=1,
        distance_space="effective_beta_xyz",
        anchor_stat="median",
    )

    assert anchors[0].neighbor_sample_ids == [1]
    assert anchors[1].neighbor_sample_ids == [3]
    assert np.allclose(anchors[0].anchor_tension_n, tension[1])
    assert np.allclose(anchors[1].anchor_tension_n, tension[3])


def test_huber_anchor_reduces_outlier_influence() -> None:
    mod = _load_module()
    values = np.asarray([[100.0] * 12, [110.0] * 12, [1000.0] * 12], dtype=float)

    huber = mod.robust_anchor(values, stat="huber_mean")
    mean = values.mean(axis=0)

    assert np.all(huber < mean)
    assert np.all(huber > 100.0)
