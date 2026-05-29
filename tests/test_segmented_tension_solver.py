from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.model.tension_transmission import transmit_tensions
from quasi_exp.opt.segmented_tension import (
    boundary_tensions_to_base,
    solve_tensions_segmented,
    unit_transmission_factors,
)


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
    radius_m = 0.003
    for j in range(12):
        angle = 2.0 * np.pi * j / 12.0
        y = radius_m * np.cos(angle)
        z = radius_m * np.sin(angle)
        holes[:, j, :, 1] = y
        holes[:, j, :, 2] = z

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
        "friction": {"mu_shaft": 0.1, "r_shaft_m": 0.004, "mu_cable": 0.03},
        "tension": {"bounds_n": [0.0, 2000.0]},
        "cables": {
            "end_disk_by_j": {
                "1": 10,
                "2": 10,
                "3": 20,
                "4": 20,
                "5": 30,
                "6": 30,
                "7": 30,
                "8": 30,
                "9": 20,
                "10": 20,
                "11": 10,
                "12": 10,
            }
        },
    }


def test_boundary_tension_conversion_reproduces_requested_interface_values() -> None:
    cfg = _toy_cfg()
    model = QuasiStaticModel(cfg, _toy_inputs())
    theta = np.linspace(-0.08, 0.08, 30, dtype=float)
    cache = model.build_cache(theta)
    unit = unit_transmission_factors(model, cache)

    cable_indices = np.array([4, 5, 6, 7], dtype=int)
    desired_boundary = np.array([120.0, 240.0, 360.0, 480.0], dtype=float)
    base_template = np.ones(12, dtype=float) * 300.0

    T_base = boundary_tensions_to_base(
        desired_boundary,
        unit,
        boundary_i=20,
        cable_indices=cable_indices,
        t_min=model.t_min,
        t_max=model.t_max,
        base_template=base_template,
    )
    F_ct, _case = transmit_tensions(
        theta_rad=cache.theta_rad,
        T_base_12=T_base,
        mu_cable=model.mu_cable,
        end_disk_by_j=model.end_disk_by_j,
        L0_12=model.L0_12,
        L_12=cache.L_12,
    )

    assert np.allclose(F_ct[20, cable_indices], desired_boundary, rtol=1e-10, atol=1e-10)
    assert np.allclose(T_base[[0, 1, 2, 3, 8, 9, 10, 11]], base_template[[0, 1, 2, 3, 8, 9, 10, 11]])


def test_segmented_solver_is_deterministic_and_reports_section_metrics() -> None:
    cfg = _toy_cfg()
    model = QuasiStaticModel(cfg, _toy_inputs())
    theta = np.linspace(-0.04, 0.05, 30, dtype=float)
    cache = model.build_cache(theta)
    solver_cfg = {
        "max_nfev": 12,
        "t_ref_n": 300.0,
        "w_ref": 0.01,
        "w_base_norm": 0.001,
        "feasible_rms_rnorm": 0.2,
    }

    res1 = solve_tensions_segmented(model, cache, solver_cfg)
    res2 = solve_tensions_segmented(model, cache, solver_cfg)

    assert res1.T_base_12.shape == (12,)
    assert np.isfinite(res1.T_base_12).all()
    assert np.all(res1.T_base_12 >= model.t_min)
    assert np.all(res1.T_base_12 <= model.t_max)
    assert np.allclose(res1.T_base_12, res2.T_base_12)
    assert set(res1.section_rms_rnorm) == {"third", "second", "first"}
    assert set(res1.section_elapsed_s) == {"third", "second", "first"}
    assert set(res1.section_nfev) == {"third", "second", "first"}
    assert np.isfinite(res1.rms_rnorm)
    assert res1.elapsed_s >= 0.0
