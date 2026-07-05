from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_active_beta_manifold_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_active_beta_manifold_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifold_3d_full_angle_is_distal_preferred_and_in_bounds():
    mod = load_module()
    bounds = mod.beta_bounds_from_ranges(
        {
            "beta1": [-np.deg2rad(5), np.deg2rad(5)],
            "beta2": [-np.deg2rad(5), np.deg2rad(5)],
            "beta3": [-np.deg2rad(5), np.deg2rad(5)],
            "beta4": [-np.deg2rad(5), np.deg2rad(5)],
            "beta5": [-np.deg2rad(10), np.deg2rad(10)],
            "beta6": [-np.deg2rad(10), np.deg2rad(10)],
        }
    )

    df = mod.build_beta_manifold_pool(
        variant="manifold_3d_full_angle",
        pool_size=256,
        beta_bounds=bounds,
        seed=123,
    )

    beta = df[mod.BETA_COLS].to_numpy(dtype=float)
    assert len(df) == 256
    assert np.all(beta >= bounds[:, 0][None, :] - 1e-12)
    assert np.all(beta <= bounds[:, 1][None, :] + 1e-12)

    proximal = np.linalg.norm(beta[:, :4], axis=1)
    distal = np.linalg.norm(beta[:, 4:], axis=1)
    assert np.percentile(distal, 50) > np.percentile(proximal, 90)
    assert set(["latent_rho", "latent_phi", "latent_q", "manifold_variant"]).issubset(df.columns)
    assert set(df["manifold_variant"]) == {"manifold_3d_full_angle"}


def test_manifold_3d_quadrant_keeps_distal_positive_quadrant():
    mod = load_module()
    bounds = np.asarray(
        [
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(10), np.deg2rad(10)],
            [-np.deg2rad(10), np.deg2rad(10)],
        ],
        dtype=float,
    )

    df = mod.build_beta_manifold_pool(
        variant="manifold_3d_quadrant",
        pool_size=128,
        beta_bounds=bounds,
        seed=456,
    )

    assert np.all(df["beta5_rad"].to_numpy(dtype=float) >= -1e-12)
    assert np.all(df["beta6_rad"].to_numpy(dtype=float) >= -1e-12)
    assert float(df["latent_phi"].min()) >= -1e-12
    assert float(df["latent_phi"].max()) <= np.pi / 2.0 + 1e-12


def test_filtered_6d_distal_cloud_respects_distal_ratio_policy():
    mod = load_module()
    bounds = np.asarray(
        [
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(5), np.deg2rad(5)],
            [-np.deg2rad(10), np.deg2rad(10)],
            [-np.deg2rad(10), np.deg2rad(10)],
        ],
        dtype=float,
    )

    df = mod.build_beta_manifold_pool(
        variant="filtered_6d_distal_cloud",
        pool_size=200,
        beta_bounds=bounds,
        seed=789,
        proximal_abs_ratio_max=0.45,
        distal_abs_ratio_min=0.55,
    )

    beta = df[mod.BETA_COLS].to_numpy(dtype=float)
    scale = np.asarray([5, 5, 5, 5, 10, 10], dtype=float) * np.pi / 180.0
    assert np.max(np.abs(beta[:, :4]) / scale[:4]) <= 0.45 + 1e-12
    assert np.min(np.abs(beta[:, 4:]) / scale[4:]) >= 0.55 - 1e-12
