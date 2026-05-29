from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasi_exp.opt.tension_canonical import canonicalize_tension


@dataclass
class _SumResidualModel:
    t_min: float = 0.0
    t_max: float = 2000.0
    target_sum: float = 9600.0
    residual_scale: float = 9600.0

    def residual_norm(self, cache, T_base_12):
        T = np.asarray(T_base_12, dtype=float).reshape(12)
        residual = (float(np.sum(T)) - self.target_sum) / self.residual_scale
        return np.full(30, residual, dtype=float), {}


def test_canonicalize_tension_converges_to_reference_from_different_feasible_starts() -> None:
    model = _SumResidualModel()
    low_high = np.array([100.0, 1900.0] * 6, dtype=float)
    high_low = np.array([1900.0, 100.0] * 6, dtype=float)
    cfg = {
        "enabled": True,
        "t_ref_n": 800.0,
        "w_ref": 1.0,
        "w_max": 0.0,
        "rms_rnorm_threshold": 1.0e-8,
        "maxiter": 80,
        "ftol": 1.0e-9,
    }

    r1 = canonicalize_tension(model, cache=None, initial_tensions=[low_high], cfg=cfg)
    r2 = canonicalize_tension(model, cache=None, initial_tensions=[high_low], cfg=cfg)

    assert r1.success is True
    assert r2.success is True
    assert np.mean(np.abs(r1.T_base_12 - r2.T_base_12)) < 1.0e-3
    assert np.allclose(r1.T_base_12, np.full(12, 800.0), atol=1.0e-3)
    assert r1.rms_rnorm <= 1.0e-8


def test_canonicalize_tension_falls_back_when_constraint_is_not_met() -> None:
    model = _SumResidualModel(target_sum=1.0e9)
    initial = np.full(12, 500.0, dtype=float)
    cfg = {
        "enabled": True,
        "t_ref_n": 800.0,
        "w_ref": 1.0,
        "w_max": 0.0,
        "rms_rnorm_threshold": 1.0e-8,
        "maxiter": 5,
    }

    res = canonicalize_tension(model, cache=None, initial_tensions=[initial], cfg=cfg)

    assert res.success is False
    assert res.method == "fallback_pso"
    assert np.allclose(res.T_base_12, initial)
