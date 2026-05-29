from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.opt.pso import _tension_objective_cost, solve_tensions_pso


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
    # disk0 requires dist holes; here prox/dist all at origin is fine for direction via frame translations
    masses = np.zeros(kD + 1, dtype=float)
    masses[1:] = 0.2
    com = np.zeros((kD + 1, 4), dtype=float)
    com[:, 3] = 1.0
    E = np.zeros(kD + 1, dtype=float)
    Iz = np.zeros(kD + 1, dtype=float)
    E[1:] = 3e10
    Iz[1:] = 1e-10
    p_end = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
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
            "n_particles": 16,
            "iters": 5,
            "inertia": 0.5,
            "c1": 1.2,
            "c2": 1.2,
            "w_resid": 1.0,
            "lambda_max": 1.0,
            "early_stop_mean_rnorm2": 0.0,
            "rng_seed": 1,
        },
        "dataset": {"rms_rnorm_threshold": 1.0, "shard_rows": 10, "out_dir": "data"},
        "sampling": {"rng_seed": 0, "beta_ranges_rad": {}},
        "parallel": {"workers": 1},
    }


def test_residual_shape_and_finite() -> None:
    cfg = _toy_cfg()
    inputs = _toy_inputs()
    model = QuasiStaticModel(cfg, inputs)
    theta = np.zeros(30, dtype=float)
    cache = model.build_cache(theta)
    T_base = np.ones(12, dtype=float) * 100.0
    rnorm, _ = model.residual_norm(cache, T_base)
    assert rnorm.shape == (30,)
    assert np.isfinite(rnorm).all()


def test_pso_deterministic() -> None:
    cfg = _toy_cfg()
    inputs = _toy_inputs()
    model = QuasiStaticModel(cfg, inputs)
    theta = np.zeros(30, dtype=float)
    cache = model.build_cache(theta)
    r1 = solve_tensions_pso(model, cache, cfg["pso"], rng_seed=123)
    r2 = solve_tensions_pso(model, cache, cfg["pso"], rng_seed=123)
    assert np.allclose(r1.T_base_12, r2.T_base_12)



def test_pso_deterministic_with_normalized_objectives() -> None:
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
    theta = np.zeros(30, dtype=float)
    cache = model.build_cache(theta)
    r1 = solve_tensions_pso(model, cache, cfg["pso"], rng_seed=321)
    r2 = solve_tensions_pso(model, cache, cfg["pso"], rng_seed=321)
    assert np.allclose(r1.T_base_12, r2.T_base_12)
    assert np.isfinite(r1.best_cost)


def test_paper_constraint_objective_ignores_mean_and_soft_cap_weights() -> None:
    tensions = np.array([100.0, 250.0, 900.0] + [500.0] * 9, dtype=float)
    cfg_a = {
        "objective": "paper_constraint",
        "paper_feasible_rms_rnorm": 0.06,
        "lambda_max": 5000.0,
        "w_tension_mean": 2000.0,
        "w_tension_soft_cap": 1200000.0,
        "tension_soft_cap_ratio": 0.25,
        "max_tension_power": 4.0,
    }
    cfg_b = {
        "objective": "paper_constraint",
        "paper_feasible_rms_rnorm": 0.06,
        "lambda_max": 0.0,
        "w_tension_mean": 0.0,
        "w_tension_soft_cap": 0.0,
        "tension_soft_cap_ratio": 1.0,
        "max_tension_power": 1.0,
    }

    score_a = _tension_objective_cost(tensions, mean_r2=0.03**2, tmax=2000.0, pso_cfg=cfg_a)
    score_b = _tension_objective_cost(tensions, mean_r2=0.03**2, tmax=2000.0, pso_cfg=cfg_b)

    assert score_a == score_b


def test_paper_constraint_objective_prefers_lower_max_tension_when_feasible() -> None:
    cfg = {"objective": "paper_constraint", "paper_feasible_rms_rnorm": 0.06}
    low_max = np.array([1000.0] + [100.0] * 11, dtype=float)
    high_max = np.array([1500.0] + [100.0] * 11, dtype=float)

    low_score = _tension_objective_cost(low_max, mean_r2=0.03**2, tmax=2000.0, pso_cfg=cfg)
    high_score = _tension_objective_cost(high_max, mean_r2=0.03**2, tmax=2000.0, pso_cfg=cfg)

    assert low_score < high_score
