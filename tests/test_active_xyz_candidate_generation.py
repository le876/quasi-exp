from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "generate_active_single_branch_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_active_single_branch_dataset", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidate_row_uses_target_xyz_and_meta_keeps_achieved_xyz() -> None:
    mod = _load_module()
    beta = np.array([0.01, -0.01, 0.02, -0.02, 0.08, -0.08], dtype=float)
    theta = np.repeat(beta, 5)[:30]
    target_xyz = np.array([1.0, 0.1, -0.2], dtype=float)
    achieved_xyz = np.array([1.001, 0.101, -0.199], dtype=float)
    tension = np.linspace(100.0, 210.0, 12)

    row, meta = mod.candidate_to_rows(
        sample_id=5,
        active_target_id=2,
        active_candidate_id=3,
        target_xyz_m=target_xyz,
        achieved_xyz_m=achieved_xyz,
        beta6_rad=beta,
        theta30_rad=theta,
        tension12_n=tension,
        source_beta6_rad=beta + 0.001,
        source_component="unit",
        candidate_kind="inverse_warm",
        xyz_err_m=float(np.linalg.norm(achieved_xyz - target_xyz)),
        rms_rnorm=0.02,
        mean_rnorm2=0.0004,
        best_cost=1.5,
        iters_used=10,
        evals=200,
        pso_seed=1234,
        elapsed_s=0.25,
    )

    assert [row["x_m"], row["y_m"], row["z_m"]] == target_xyz.tolist()
    assert [meta["achieved_x_m"], meta["achieved_y_m"], meta["achieved_z_m"]] == achieved_xyz.tolist()
    assert meta["active_target_id"] == 2
    assert meta["active_candidate_id"] == 3
    assert meta["candidate_kind"] == "inverse_warm"
    assert meta["source_component"] == "unit"
    assert meta["effective_beta_source"] == "theta_odd_even_mean"
    assert np.isclose(meta["xyz_err_m"], np.linalg.norm(achieved_xyz - target_xyz))
