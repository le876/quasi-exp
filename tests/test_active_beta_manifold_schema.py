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


def test_beta_manifold_to_rows_schema_uses_target_xyz_and_effective_beta():
    mod = load_module()
    beta = np.asarray([0.01, -0.01, 0.02, -0.02, 0.08, -0.08], dtype=float)
    theta = np.zeros(30, dtype=float)
    theta[0::2] = [0.01] * 5 + [0.02] * 5 + [0.08] * 5
    theta[1::2] = [-0.01] * 5 + [-0.02] * 5 + [-0.08] * 5
    tension = np.linspace(100.0, 1200.0, 12)

    row, meta = mod.beta_manifold_to_rows(
        sample_id=7,
        beta=beta,
        theta=theta,
        tension=tension,
        xyz=np.asarray([1.0, 2.0, 3.0], dtype=float),
        rms_rnorm=0.02,
        mean_rnorm2=0.0004,
        best_cost=1.5,
        iters_used=9,
        evals=99,
        pso_seed=1234,
        elapsed_s=0.5,
        variant="manifold_3d_full_angle",
        latent={"latent_rho": 0.8, "latent_phi": 1.0, "latent_q": 0.2},
        injectivity={"injectivity_neighbor_count": 3, "injectivity_max_beta_dist_norm": 0.1},
    )

    assert row["sample_id"] == 7
    assert row["x_m"] == 1.0
    assert row["y_m"] == 2.0
    assert row["z_m"] == 3.0
    assert meta["manifold_variant"] == "manifold_3d_full_angle"
    assert meta["source_component"] == "manifold_3d_full_angle"
    assert meta["xyz_err_m"] == 0.0
    assert meta["beta1_rad"] == meta["effective_beta_1_rad"]
    assert meta["beta6_rad"] == meta["effective_beta_6_rad"]
    assert meta["latent_rho"] == 0.8
    assert meta["injectivity_neighbor_count"] == 3
    assert all(f"theta_{i}_rad" in row for i in range(1, 31))
    assert all(f"tension_{i}_n" in row for i in range(1, 13))
