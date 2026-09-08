from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyMlp:
    loss_curve_ = [3.0, 2.0, 1.0]
    validation_scores_ = [0.1, 0.2, 0.3]


class DummyPipeline:
    def __init__(self, inner):
        self.named_steps = {"mlp": inner}


class DummyEstimator:
    def __init__(self, coef: float):
        self.coef = float(coef)
        self.n_estimators_ = 4

    def predict(self, X: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        iteration = 4 if num_iteration is None else int(num_iteration)
        return np.full(X.shape[0], self.coef * iteration, dtype=float)


class DummyMultiOutput:
    def __init__(self):
        self.estimators_ = [DummyEstimator(0.1), DummyEstimator(0.2)]


class IdentityScaler:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)


class DummyLgbmPipeline:
    def __init__(self):
        self.named_steps = {"x_scaler": IdentityScaler(), "mlp": DummyMultiOutput()}


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "plot_theta_training_curves.py"
    spec = importlib.util.spec_from_file_location("plot_theta_training_curves", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_mlp_history_returns_epoch_loss_and_validation_score() -> None:
    mod = _load_module()
    package = {"model": DummyPipeline(DummyMlp())}

    curve = mod.extract_mlp_history(package, model_id="radius_mlp")

    assert curve["model_id"].tolist() == ["radius_mlp"] * 3
    assert curve["epoch"].tolist() == [1, 2, 3]
    assert curve["train_loss"].tolist() == [3.0, 2.0, 1.0]
    assert curve["validation_score"].tolist() == [0.1, 0.2, 0.3]


def test_lgbm_theta_mae_curve_uses_num_iteration_and_inverse_scaler() -> None:
    mod = _load_module()
    package = {"model": DummyLgbmPipeline(), "y_scaler": IdentityScaler()}
    X = np.zeros((3, 3), dtype=float)
    theta_true = np.zeros((3, 2), dtype=float)

    curve = mod.lgbm_theta_mae_curve(
        package,
        model_id="radius_lgbm",
        X_by_split={"train": X, "val": X},
        theta_by_split={"train": theta_true, "val": theta_true},
        iterations=[1, 4],
    )

    assert set(curve["split"]) == {"train", "val"}
    assert curve["iteration"].tolist() == [1, 4, 1, 4]
    assert np.isclose(curve[curve["iteration"] == 1]["theta_mae_deg"].iloc[0], np.rad2deg(0.15))
    assert np.isclose(curve[curve["iteration"] == 4]["theta_mae_deg"].iloc[0], np.rad2deg(0.60))
