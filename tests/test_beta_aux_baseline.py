from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from quasi_exp.model.sampling import beta_to_theta


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "baselines" / "run_beta_aux_baselines.py"
    spec = importlib.util.spec_from_file_location("run_beta_aux_baselines", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_beta_aux_consistency_error_uses_effective_beta_from_predicted_theta() -> None:
    mod = _load_module()
    beta = np.array([[0.01, -0.02, 0.03, -0.04, 0.08, -0.09]], dtype=float)
    theta = np.vstack([beta_to_theta(beta[0])])

    same = mod.beta_consistency_mae_rad(theta, beta)
    shifted = mod.beta_consistency_mae_rad(theta, beta + 0.1)

    assert same < 1.0e-12
    assert shifted > 0.09
