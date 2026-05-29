from __future__ import annotations

import numpy as np
import pytest

from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.opt import tension_labeler
from tests.test_segmented_tension_solver import _toy_cfg, _toy_inputs


def test_segmented_canonical_labeler_does_not_call_pso(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _toy_cfg()
    cfg["tension_labeler"] = {"method": "segmented_canonical"}
    cfg["segmented_tension"] = {
        "max_nfev": 12,
        "t_ref_n": 300.0,
        "feasible_rms_rnorm": 0.2,
    }
    model = QuasiStaticModel(cfg, _toy_inputs())
    cache = model.build_cache(np.linspace(-0.04, 0.05, 30, dtype=float))

    def _fail_if_pso_called(*_args, **_kwargs):
        raise AssertionError("segmented_canonical must not call PSO")

    monkeypatch.setattr(tension_labeler, "solve_tensions_pso", _fail_if_pso_called)

    res = tension_labeler.solve_tension_label(
        model=model,
        cache=cache,
        pso_cfg={"rng_seed": 123},
        pso_seed=123,
        rms_thresh=0.2,
    )

    assert res.ok is True
    assert res.T_base_12.shape == (12,)
    assert res.meta["tension_solver_method"] == "segmented_canonical"
    assert res.meta["canonical_enabled"] is True
    assert res.meta["canonical_adopted"] is True
    assert res.meta["segmented_success"] is True
    assert res.meta["segmented_section_third_rms_rnorm"] >= 0.0
    assert res.meta["pso_elapsed_s"] == 0.0
    assert res.meta["evals"] == (
        res.meta["segmented_section_third_nfev"]
        + res.meta["segmented_section_second_nfev"]
        + res.meta["segmented_section_first_nfev"]
    )


def test_default_labeler_preserves_pso_canonical_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _toy_cfg()
    model = QuasiStaticModel(cfg, _toy_inputs())
    cache = model.build_cache(np.zeros(30, dtype=float))

    def _fail_if_segmented_called(*_args, **_kwargs):
        raise AssertionError("default labeler must not call segmented solver")

    monkeypatch.setattr(tension_labeler, "solve_tensions_segmented", _fail_if_segmented_called)

    res = tension_labeler.solve_tension_label(
        model=model,
        cache=cache,
        pso_cfg={
            "n_particles": 8,
            "iters": 2,
            "inertia": 0.5,
            "c1": 1.2,
            "c2": 1.2,
            "w_resid": 1.0,
            "lambda_max": 1.0,
            "early_stop_mean_rnorm2": 0.0,
            "rng_seed": 1,
        },
        pso_seed=123,
        rms_thresh=1.0,
    )

    assert res.meta["tension_solver_method"] == "pso_canonical"
    assert res.meta["canonical_enabled"] is False
    assert res.meta["canonical_adopted"] is False
    assert res.meta["pso_elapsed_s"] > 0.0
    assert np.isfinite(res.T_base_12).all()
