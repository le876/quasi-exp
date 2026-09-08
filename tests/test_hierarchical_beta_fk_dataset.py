from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "generate_hierarchical_beta_fk_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_hierarchical_beta_fk_dataset", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grid_levels_include_continuous_beta1_to_beta4_with_hierarchical_steps() -> None:
    mod = _load_module()

    spec = mod.HierarchicalGridSpec(
        beta12_deg=(-5.0, 5.0, 2.5),
        beta34_deg=(-5.0, 5.0, 1.25),
        beta56_deg=(-15.0, 15.0, 0.5),
    )
    levels = mod.grid_levels_deg(spec)

    assert np.allclose(levels["beta1"], [-5.0, -2.5, 0.0, 2.5, 5.0])
    assert np.allclose(levels["beta2"], levels["beta1"])
    assert len(levels["beta3"]) == 9
    assert len(levels["beta4"]) == 9
    assert len(levels["beta5"]) == 61
    assert len(levels["beta6"]) == 61
    assert spec.step_deg("joint3") < spec.step_deg("joint2") < spec.step_deg("joint1")
    assert mod.expected_grid_rows(spec) == 7_535_025


def test_generate_beta_chunk_scans_beta5_beta6_fastest_and_keeps_beta1_to_beta4_variable() -> None:
    mod = _load_module()
    spec = mod.HierarchicalGridSpec(
        beta12_deg=(-5.0, 5.0, 5.0),
        beta34_deg=(-5.0, 5.0, 5.0),
        beta56_deg=(-1.0, 1.0, 1.0),
    )

    beta = mod.generate_beta_chunk_rad(spec, start=0, stop=20)

    assert beta.shape == (20, 6)
    # First beta1..4 combination sweeps the full beta5/beta6 plane.
    assert np.unique(np.round(np.rad2deg(beta[:9, 0]), 6)).tolist() == [-5.0]
    assert np.unique(np.round(np.rad2deg(beta[:9, 2]), 6)).tolist() == [-5.0]
    assert set(np.round(np.rad2deg(beta[:9, 4]), 6)) == {-1.0, 0.0, 1.0}
    assert set(np.round(np.rad2deg(beta[:9, 5]), 6)) == {-1.0, 0.0, 1.0}
    # Shortly after the first 3x3 distal plane, a proximal value changes too.
    assert len(np.unique(np.round(np.rad2deg(beta[:, 3]), 6))) > 1


def test_workspace_metrics_report_x_thickness_and_uniformity() -> None:
    mod = _load_module()
    xyz = np.array(
        [
            [1.00, 0.00, 0.00],
            [1.04, 0.00, 0.00],
            [1.08, 0.00, 0.00],
            [1.12, 0.02, 0.00],
            [1.16, 0.02, 0.00],
            [1.20, 0.02, 0.00],
        ],
        dtype=float,
    )

    metrics = mod.workspace_metrics(xyz, x_range=(1.0, 1.2), x_bin_mm=50.0, yz_cell_mm=30.0, voxel_mm=20.0)

    assert metrics["rows"] == 6
    assert metrics["x_bin_nonempty_ratio"] > 0.0
    assert metrics["yz_cell_x_range_p95_mm"] >= 40.0
    assert metrics["voxel_count_3d"] >= 4
    assert 0.0 <= metrics["x_bin_count_cv"] < 2.0


def test_select_balanced_subset_prefers_small_proximal_and_large_distal_within_voxel() -> None:
    mod = _load_module()
    df = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "x_m": [1.001, 1.002, 1.101, 1.102],
            "y_m": [0.001, 0.002, 0.001, 0.002],
            "z_m": [0.001, 0.002, 0.001, 0.002],
            "beta1_rad": np.deg2rad([5.0, 0.0, 0.0, 2.5]),
            "beta2_rad": np.deg2rad([5.0, 0.0, 0.0, 2.5]),
            "beta3_rad": np.deg2rad([5.0, 0.0, 0.0, 2.5]),
            "beta4_rad": np.deg2rad([5.0, 0.0, 0.0, 2.5]),
            "beta5_rad": np.deg2rad([5.0, 15.0, 10.0, 15.0]),
            "beta6_rad": np.deg2rad([5.0, 15.0, 10.0, 15.0]),
        }
    )

    selected = mod.select_workspace_balanced_subset(
        df,
        max_rows=2,
        x_range=(1.0, 1.2),
        x_bin_mm=100.0,
        voxel_mm=10.0,
        seed=7,
    )

    assert selected["sample_id"].tolist() == [1, 2]
    assert "hierarchical_priority_score" in selected.columns
    assert "workspace_voxel_id" in selected.columns
