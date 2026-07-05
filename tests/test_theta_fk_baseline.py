from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "baselines" / "run_theta_fk_baseline.py"
    spec = importlib.util.spec_from_file_location("run_theta_fk_baseline", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_x_slab_split_holds_out_largest_x_values() -> None:
    mod = _load_module()
    xyz = np.column_stack([np.arange(10, dtype=float), np.zeros((10, 2), dtype=float)])

    train_idx, val_idx, test_idx = mod.split_x_slab_holdout(xyz, val_size=0.2, test_size=0.2)

    assert train_idx.tolist() == [0, 1, 2, 3, 4, 5]
    assert val_idx.tolist() == [6, 7]
    assert test_idx.tolist() == [8, 9]


def test_theta_metrics_do_not_require_tension_columns() -> None:
    mod = _load_module()
    theta_true = np.zeros((3, 30), dtype=float)
    theta_pred = np.deg2rad(np.ones((3, 30), dtype=float))
    ee_err = np.array([1.0, 2.0, 10.0], dtype=float)

    metrics = mod.theta_metrics(theta_true, theta_pred, ee_err)

    assert metrics["theta_mae_deg"] == 1.0
    assert metrics["theta_rmse_deg"] == 1.0
    assert metrics["ee_pos_p95_mm"] > 9.0
    assert "tension_mae_n" not in metrics


def test_lgbm_model_builder_is_available_for_theta_only_baseline() -> None:
    mod = _load_module()

    model = mod._model(
        "lgbm",
        seed=123,
        max_iter=10,
        alpha=1e-5,
        batch_size=32,
        learning_rate_init=1e-3,
        n_jobs=1,
        lgbm_n_estimators=5,
        lgbm_learning_rate=0.1,
        lgbm_num_leaves=7,
    )

    assert model.__class__.__name__ == "MultiOutputRegressor"


def test_run_saves_theta_only_model_package_without_tension_columns(tmp_path: Path) -> None:
    mod = _load_module()
    rng = np.random.default_rng(7)
    n = 24
    data: dict[str, np.ndarray] = {
        "x_m": np.linspace(1.0, 1.1, n),
        "y_m": rng.normal(0.0, 0.01, n),
        "z_m": rng.normal(0.0, 0.01, n),
    }
    for i in range(1, 7):
        data[f"beta{i}_rad"] = rng.normal(0.0, 0.01, n)
    for i in range(1, 31):
        data[f"theta_{i}_rad"] = 0.01 * data["x_m"] + rng.normal(0.0, 0.001, n)
    dataset = tmp_path / "theta_only.parquet"
    pd.DataFrame(data).to_parquet(dataset, index=False)

    args = type(
        "Args",
        (),
        {
            "dataset": dataset,
            "robot_config": REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml",
            "out_dir": tmp_path / "out",
            "splits": "iid",
            "models": "lgbm",
            "seed": 123,
            "val_size": 0.1,
            "test_size": 0.1,
            "max_rows": 0,
            "max_iter": 10,
            "alpha": 1e-5,
            "batch_size": 16,
            "learning_rate_init": 1e-3,
            "n_jobs": 1,
            "lgbm_n_estimators": 5,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 7,
            "skip_fk": True,
            "save_model": True,
        },
    )()

    payload = mod.run(args)

    assert payload["rows"] == n
    package = mod.load_theta_model_package(tmp_path / "out" / "iid" / "lgbm" / "model.joblib")
    assert package["kind"] == "theta_only"
    assert package["model_name"] == "lgbm"
    assert package["split"] == "iid"
    assert package["input_cols"] == ["x_m", "y_m", "z_m"]
    assert len(package["theta_cols"]) == 30
    assert "tension_cols" not in package
