from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_beta6_large_ellipse_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_beta6_large_ellipse_benchmark", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=float)


class ZeroUModel:
    def predict(self, x):
        return np.zeros((len(x), 3), dtype=float)


def test_select_candidates_uses_true_support_when_requested(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "stale_good_true_bad",
                "amp_xy_mm": 100.0,
                "nn_p95_mm": 1.0,
                "support_gate_pass": True,
                "true_nn_p95_mm": 100.0,
                "true_support_gate_pass": False,
                "nearest_x_mean_diff_mm": 0.0,
            },
            {
                "candidate_id": "true_good",
                "amp_xy_mm": 87.5,
                "nn_p95_mm": 999.0,
                "support_gate_pass": False,
                "true_nn_p95_mm": 2.0,
                "true_support_gate_pass": True,
                "nearest_x_mean_diff_mm": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    selected = mod.select_candidates(path, max_candidates=5, require_true_support=True)

    assert selected["candidate_id"].tolist() == ["true_good"]


def test_predict_pose_supports_u_pose_package() -> None:
    mod = _load_module()
    package = {
        "kind": "u_pose",
        "feature_set": "raw",
        "model": ZeroUModel(),
        "x_scaler": IdentityScaler(),
        "y_scaler": IdentityScaler(),
        "theta_sign": -1.0,
        "u_mapping": {"s10": 0.125, "s20": 0.25, "ds1": 0.0, "ds2": 0.0, "dq": 0.0},
    }

    beta, theta = mod.predict_pose(package, np.zeros((4, 3), dtype=float))

    assert beta.shape == (4, 6)
    assert theta.shape == (4, 30)
    assert np.allclose(beta, 0.0)
    assert np.allclose(theta, 0.0)

