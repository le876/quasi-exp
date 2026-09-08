from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_active_beta_manifold_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_active_beta_manifold_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_injectivity_filter_rejects_workspace_close_beta_far_points():
    mod = load_module()
    scale = np.asarray([5, 5, 5, 5, 10, 10], dtype=float) * np.pi / 180.0
    beta = np.zeros((3, 6), dtype=float)
    beta[1, 5] = scale[5] * 0.5
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.005],
            [0.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    pool = pd.DataFrame(beta, columns=mod.BETA_COLS)
    pool[["target_x_m", "target_y_m", "target_z_m"]] = xyz

    filtered, report = mod.apply_local_injectivity_filter(
        pool,
        beta_scale_rad=scale,
        radius_m=0.01,
        beta_far_threshold_norm=0.15,
        min_neighbors=1,
    )

    assert report["rejected_beta_far_count"] == 2
    assert len(filtered) == 1
    assert np.allclose(filtered[mod.BETA_COLS].to_numpy(dtype=float), beta[[2]])


def test_local_injectivity_filter_keeps_workspace_close_beta_close_points():
    mod = load_module()
    scale = np.asarray([5, 5, 5, 5, 10, 10], dtype=float) * np.pi / 180.0
    beta = np.zeros((3, 6), dtype=float)
    beta[1, 5] = scale[5] * 0.05
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.005],
            [0.1, 0.0, 0.0],
        ],
        dtype=float,
    )
    pool = pd.DataFrame(beta, columns=mod.BETA_COLS)
    pool[["target_x_m", "target_y_m", "target_z_m"]] = xyz

    filtered, report = mod.apply_local_injectivity_filter(
        pool,
        beta_scale_rad=scale,
        radius_m=0.01,
        beta_far_threshold_norm=0.15,
        min_neighbors=1,
    )

    assert report["rejected_beta_far_count"] == 0
    assert len(filtered) == 3
