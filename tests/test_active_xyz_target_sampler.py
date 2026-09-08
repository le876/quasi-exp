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


def _ranges() -> dict[str, tuple[float, float]]:
    return {f"beta{i + 1}": (-0.1, 0.1) for i in range(6)}


def test_reachable_target_pool_preserves_source_beta_and_workspace_xyz() -> None:
    mod = _load_module()

    def workspace_xyz(beta_rows: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rows, dtype=float).reshape(-1, 6)
        return np.column_stack([1.0 + beta[:, 4], beta[:, 0] + beta[:, 2], beta[:, 1] + beta[:, 5]])

    pool = mod.build_reachable_target_pool(
        beta_ranges_rad=_ranges(),
        pool_size=12,
        rng_seed=123,
        workspace_xyz_fn=workspace_xyz,
        components={"sobol_full": 0.5, "distal_biased": 0.5},
    )

    assert len(pool) == 12
    assert {"target_x_m", "target_y_m", "target_z_m", "source_beta_1_rad", "source_component"}.issubset(pool.columns)
    first_beta = pool[[f"source_beta_{i + 1}_rad" for i in range(6)]].iloc[0].to_numpy(dtype=float)
    first_xyz = pool[["target_x_m", "target_y_m", "target_z_m"]].iloc[0].to_numpy(dtype=float)
    assert np.allclose(first_xyz, workspace_xyz(first_beta.reshape(1, 6))[0])


def test_workspace_balanced_targets_are_reindexed_and_keep_source_beta() -> None:
    mod = _load_module()
    pool = mod.build_reachable_target_pool(
        beta_ranges_rad=_ranges(),
        pool_size=20,
        rng_seed=456,
        workspace_xyz_fn=lambda beta: np.asarray(beta, dtype=float).reshape(-1, 6)[:, :3],
        components={"sobol_full": 1.0},
    )

    targets = mod.select_workspace_balanced_targets(pool, num_targets=7, radius_bins=3, z_bins=2, angle_bins=4, seed=99)

    assert len(targets) == 7
    assert targets["active_target_id"].tolist() == list(range(7))
    assert targets["source_pool_index"].is_unique
    assert not targets[[f"source_beta_{i + 1}_rad" for i in range(6)]].isna().any().any()
