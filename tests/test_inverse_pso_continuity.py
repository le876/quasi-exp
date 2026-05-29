from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.opt.pso_inverse import _theta_delta_mse, solve_inverse_joint_pso


@dataclass(frozen=True)
class _Inputs:
    lengths_m: np.ndarray
    holes_local_m: np.ndarray
    masses_kg: np.ndarray
    com_local_m: np.ndarray
    E_pa: np.ndarray
    Iz_m4: np.ndarray
    p_end_local_m: np.ndarray


def _toy_inputs() -> _Inputs:
    kD = 30
    lengths = np.ones(kD + 1, dtype=float) * 0.05
    lengths[0] = 0.1
    holes = np.zeros((kD + 1, 12, 2, 4), dtype=float)
    holes[:, :, :, 3] = 1.0
    masses = np.zeros(kD + 1, dtype=float)
    masses[1:] = 0.2
    com = np.zeros((kD + 1, 4), dtype=float)
    com[:, 3] = 1.0
    E = np.zeros(kD + 1, dtype=float)
    Iz = np.zeros(kD + 1, dtype=float)
    E[1:] = 3e10
    Iz[1:] = 1e-10
    p_end = np.array([0.05, 0.0, 0.0, 1.0], dtype=float)
    return _Inputs(lengths, holes, masses, com, E, Iz, p_end)


def _toy_cfg() -> dict:
    return {
        "robot": {"kD": 30},
        "paths": {},
        "physics": {"g": 9.81},
        "friction": {"mu_shaft": 0.1, "r_shaft_m": 0.004, "mu_cable": 0.0},
        "tension": {"bounds_n": [0.0, 2000.0]},
        "cables": {"end_disk_by_j": {str(j): 30 for j in range(1, 13)}},
        "pso": {
            "n_particles": 12,
            "iters": 3,
            "inertia": 0.5,
            "c1": 1.2,
            "c2": 1.2,
            "w_resid": 1.0,
            "lambda_max": 1.0,
            "early_stop_mean_rnorm2": 0.0,
            "rng_seed": 1,
        },
    }


def test_theta_delta_mse_basic() -> None:
    beta = np.array([0.1, -0.1, 0.05, -0.05, 0.02, -0.02], dtype=float)
    assert _theta_delta_mse(beta, beta) == 0.0
    assert _theta_delta_mse(beta, -beta) > 0.0


def test_inverse_solver_accepts_warm_start_and_continuity() -> None:
    cfg = _toy_cfg()
    inputs = _toy_inputs()
    model = QuasiStaticModel(cfg, inputs)
    theta0 = np.zeros(30, dtype=float)
    target_xyz, _ = forward_kinematics(theta0, inputs.lengths_m, inputs.p_end_local_m, theta_sign=model.theta_sign)

    beta_ranges = {f"beta{i+1}": [-0.2, 0.2] for i in range(6)}
    inverse_cfg = {
        "backend": "numpy",
        "n_particles": 10,
        "iters": 4,
        "inertia": 0.6,
        "c1": 1.2,
        "c2": 1.2,
        "w_xyz": 10000.0,
        "w_beta_l2": 1.0,
        "w_continuity": 200.0,
        "continuity_max_ratio": 0.3,
        "continuity_relax_if_err_ratio": 1.2,
        "continuity_relax_scale": 0.3,
        "warm_start": True,
        "warm_start_particles": 4,
        "warm_start_sigma": 0.05,
        "n_restarts": 2,
        "restart_seed_stride": 123,
        "canonical_mode": "none",
        "early_stop_xyz_err_m": 1e-6,
    }

    warm = np.array([0.05, 0.05, -0.03, -0.03, 0.01, 0.01], dtype=float)
    res = solve_inverse_joint_pso(
        model=model,
        inputs=inputs,
        xyz_target_m=target_xyz,
        beta_ranges_rad=beta_ranges,
        inverse_pso_cfg=inverse_cfg,
        tension_pso_cfg=cfg["pso"],
        rng_seed=12345,
        warm_start_beta=warm,
        continuity_ref_beta=warm,
    )

    assert res.beta6_rad.shape == (6,)
    assert res.theta_rad.shape == (30,)
    assert res.T_base_12.shape == (12,)
    assert np.isfinite(res.theta_rad).all()
    assert np.isfinite(res.T_base_12).all()


def test_inverse_solver_accepts_normalized_objectives() -> None:
    cfg = _toy_cfg()
    cfg["pso"].update({
        "normalized_objectives": True,
        "w_resid": 2.0,
        "lambda_max": 0.8,
        "w_tension_mean": 0.2,
        "w_tension_soft_cap": 0.5,
        "residual_cost_scale": 0.045,
        "residual_cost_delta": 0.8,
    })
    inputs = _toy_inputs()
    model = QuasiStaticModel(cfg, inputs)
    theta0 = np.zeros(30, dtype=float)
    target_xyz, _ = forward_kinematics(theta0, inputs.lengths_m, inputs.p_end_local_m, theta_sign=model.theta_sign)

    beta_ranges = {f"beta{i+1}": [-0.2, 0.2] for i in range(6)}
    inverse_cfg = {
        "backend": "numpy",
        "n_particles": 10,
        "iters": 4,
        "inertia": 0.6,
        "c1": 1.2,
        "c2": 1.2,
        "normalized_objectives": True,
        "w_xyz": 1.0,
        "w_beta_l2": 0.05,
        "w_continuity": 1.8,
        "xyz_cost_scale_m": 0.015,
        "xyz_cost_delta": 0.6,
        "cont_cost_scale_deg": 6.0,
        "cont_cost_delta": 0.4,
        "continuity_max_ratio": 0.3,
        "continuity_relax_if_err_ratio": 1.2,
        "continuity_relax_scale": 0.3,
        "warm_start": True,
        "warm_start_particles": 4,
        "warm_start_sigma": 0.05,
        "n_restarts": 2,
        "restart_seed_stride": 123,
        "canonical_mode": "none",
        "early_stop_xyz_err_m": 1e-6,
    }

    warm = np.array([0.05, 0.05, -0.03, -0.03, 0.01, 0.01], dtype=float)
    res = solve_inverse_joint_pso(
        model=model,
        inputs=inputs,
        xyz_target_m=target_xyz,
        beta_ranges_rad=beta_ranges,
        inverse_pso_cfg=inverse_cfg,
        tension_pso_cfg=cfg["pso"],
        rng_seed=12345,
        warm_start_beta=warm,
        continuity_ref_beta=warm,
    )

    assert res.beta6_rad.shape == (6,)
    assert np.isfinite(res.best_cost)
    assert np.isfinite(res.theta_rad).all()
