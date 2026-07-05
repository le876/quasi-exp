from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_priority_grid_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_priority_grid_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_priority_grid_builds_strict_third_joint_first_grid():
    mod = load_module()

    grid = mod.build_priority_grid_pool(
        beta5_levels=41,
        beta6_levels=41,
        beta5_deg=(-10.0, 10.0),
        beta6_deg=(-10.0, 10.0),
        s2_values=(0.0, 0.25, 0.5),
        s1_values=(0.0, 0.125, 0.25),
        strict_s1_le_s2=True,
    )

    assert len(grid) == 41 * 41 * 7
    assert grid["priority_ratio_pair_id"].nunique() == 7
    assert np.all(grid["s1"].to_numpy(dtype=float) <= grid["s2"].to_numpy(dtype=float) + 1.0e-12)
    assert np.isclose(grid["beta5_rad"].min(), -np.deg2rad(10.0))
    assert np.isclose(grid["beta5_rad"].max(), np.deg2rad(10.0))
    assert np.isclose(grid["beta6_rad"].min(), -np.deg2rad(10.0))
    assert np.isclose(grid["beta6_rad"].max(), np.deg2rad(10.0))

    beta = grid[mod.BETA_COLS].to_numpy(dtype=float)
    assert np.allclose(beta[:, 0], grid["s1"].to_numpy(dtype=float) * beta[:, 4])
    assert np.allclose(beta[:, 1], grid["s1"].to_numpy(dtype=float) * beta[:, 5])
    assert np.allclose(beta[:, 2], grid["s2"].to_numpy(dtype=float) * beta[:, 4])
    assert np.allclose(beta[:, 3], grid["s2"].to_numpy(dtype=float) * beta[:, 5])


def test_priority_reference_selector_rejects_targets_outside_workspace():
    mod = load_module()
    reference = pd.DataFrame(
        {
            "x_m": [1.0, 1.0],
            "y_m": [0.0, 0.01],
            "z_m": [0.0, 0.0],
            "effective_beta_1_rad": [0.0, 0.0],
            "effective_beta_2_rad": [0.0, 0.0],
            "effective_beta_3_rad": [0.0, 0.0],
            "effective_beta_4_rad": [0.0, 0.0],
            "effective_beta_5_rad": [0.1, 0.1],
            "effective_beta_6_rad": [0.0, 0.0],
        }
    )

    assert mod.target_inside_priority_workspace(
        np.asarray([1.0, 0.004, 0.0], dtype=float),
        reference,
        radius_m=0.010,
    )
    assert not mod.target_inside_priority_workspace(
        np.asarray([1.0, 0.05, 0.0], dtype=float),
        reference,
        radius_m=0.010,
    )


def test_priority_reference_selector_prefers_first_then_second_joint_match():
    mod = load_module()
    reference = pd.DataFrame(
        {
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "effective_beta_1_rad": [0.0],
            "effective_beta_2_rad": [0.0],
            "effective_beta_3_rad": [0.02],
            "effective_beta_4_rad": [0.02],
            "effective_beta_5_rad": [0.10],
            "effective_beta_6_rad": [0.10],
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": [0, 1, 2],
            "target_x_m": [1.0, 1.0, 1.0],
            "target_y_m": [0.0, 0.0, 0.0],
            "target_z_m": [0.0, 0.0, 0.0],
            "xyz_err_m": [0.001, 0.001, 0.0005],
            "rms_rnorm": [0.01, 0.01, 0.02],
            "max_tension": [120.0, 120.0, 100.0],
            "effective_beta_1_rad": [0.03, 0.0, 0.0],
            "effective_beta_2_rad": [0.03, 0.0, 0.0],
            "effective_beta_3_rad": [0.02, 0.07, 0.02],
            "effective_beta_4_rad": [0.02, 0.07, 0.02],
            "effective_beta_5_rad": [0.10, 0.10, 0.12],
            "effective_beta_6_rad": [0.10, 0.10, 0.12],
        }
    )

    selected = mod.select_priority_reference_candidate(
        candidates,
        reference,
        target_xyz=np.asarray([1.0, 0.0, 0.0], dtype=float),
        workspace_radius_m=0.010,
    )

    assert int(selected["candidate_id"]) == 2
    assert selected["priority_reference_neighbor_count"] == 1
    assert selected["priority_reference_joint1_dist"] == 0.0
    assert selected["priority_reference_joint2_dist"] == 0.0
