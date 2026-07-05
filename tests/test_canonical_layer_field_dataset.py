from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_canonical_layer_field_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_canonical_layer_field_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_u_grid_uses_61_by_61_by_21_default_pilot() -> None:
    mod = load_module()

    grid = mod.build_u_grid(
        a_deg=(-15.0, 15.0, 0.5),
        b_deg=(-15.0, 15.0, 0.5),
        eta_count=21,
    )

    assert len(grid) == 61 * 61 * 21
    assert np.isclose(grid["u_a_deg"].min(), -15.0)
    assert np.isclose(grid["u_a_deg"].max(), 15.0)
    assert np.isclose(grid["u_b_deg"].min(), -15.0)
    assert np.isclose(grid["u_b_deg"].max(), 15.0)
    assert np.isclose(grid["u_eta"].min(), -1.0)
    assert np.isclose(grid["u_eta"].max(), 1.0)


def test_path_b_maps_u_to_redistributed_distal_preferred_beta() -> None:
    mod = load_module()
    path = mod.layer_paths()["path_b_redistribute_12"]
    a = np.deg2rad(10.0)
    b = np.deg2rad(-5.0)

    beta, s1, s2 = mod.beta_from_u(a, b, 1.0, path)

    assert np.isclose(s1, 0.200)
    assert np.isclose(s2, 0.150)
    assert np.allclose(beta, [0.2 * a, 0.2 * b, 0.15 * a, 0.15 * b, a, b])


def test_safety_filter_rejects_out_of_bounds_layer_path_values() -> None:
    mod = load_module()
    path = mod.LayerPath(name="bad", s10=0.125, s20=0.250, ds1=-0.250, ds2=0.100)
    grid = mod.build_u_grid(a_deg=(-1.0, 1.0, 1.0), b_deg=(-1.0, 1.0, 1.0), eta_count=3)

    pool = mod.build_layer_field_pool(grid, path)

    assert len(pool) < len(grid)
    assert float(pool["s1"].min()) >= 0.0
    assert float(pool["s1"].max()) <= 0.35
    assert float(pool["s2"].min()) >= 0.05
    assert float(pool["s2"].max()) <= 0.60


def test_theta_from_beta_batch_expands_six_betas_to_three_ten_disk_sections() -> None:
    mod = load_module()
    beta = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=float)

    theta = mod.theta_from_beta_batch(beta, theta_sign=1.0)

    assert theta.shape == (1, 30)
    assert np.all(theta[0, 0:10:2] == 1.0)
    assert np.all(theta[0, 1:10:2] == 2.0)
    assert np.all(theta[0, 10:20:2] == 3.0)
    assert np.all(theta[0, 11:20:2] == 4.0)
    assert np.all(theta[0, 20:30:2] == 5.0)
    assert np.all(theta[0, 21:30:2] == 6.0)


def test_numerical_jacobian_reports_expected_singular_values_for_linear_map() -> None:
    mod = load_module()

    def h(u_bar: np.ndarray) -> np.ndarray:
        return np.asarray([u_bar[0], 2.0 * u_bar[1], 3.0 * u_bar[2]], dtype=float)

    J, singular, kappa = mod.numerical_jacobian(h, np.asarray([0.1, -0.2, 0.3]), delta=1e-5)

    assert np.allclose(J, np.diag([1.0, 2.0, 3.0]), atol=1e-9)
    assert np.allclose(singular, [3.0, 2.0, 1.0], atol=1e-9)
    assert np.isclose(kappa, 3.0)
