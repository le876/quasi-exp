from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "baselines" / "run_u_pose_baselines.py"
    spec = importlib.util.spec_from_file_location("run_u_pose_baselines", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_saves_u_pose_model_package(tmp_path: Path) -> None:
    mod = _load_module()
    rng = np.random.default_rng(11)
    n = 36
    u_a = np.linspace(-2.0, 2.0, n)
    u_b = np.linspace(1.0, -1.0, n)
    u_eta = np.linspace(-0.2, 0.2, n)
    data: dict[str, np.ndarray] = {
        "x_m": 1.05 + 0.001 * u_a,
        "y_m": 0.02 + 0.001 * u_b,
        "z_m": -0.03 + 0.001 * u_eta,
        "u_a_deg": u_a,
        "u_b_deg": u_b,
        "u_eta": u_eta,
        "s1": np.full(n, 0.125),
        "s2": np.full(n, 0.25),
        "q_distal": np.ones(n),
    }
    for i in range(1, 7):
        data[f"beta{i}_rad"] = rng.normal(0.0, 0.001, n)
    for i in range(1, 31):
        data[f"theta_{i}_rad"] = rng.normal(0.0, 0.001, n)
    dataset = tmp_path / "u_dataset.parquet"
    pd.DataFrame(data).to_parquet(dataset, index=False)

    args = type(
        "Args",
        (),
        {
            "dataset": dataset,
            "robot_config": REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml",
            "out_dir": tmp_path / "out",
            "models": "mlp_u",
            "splits": "iid",
            "feature_set": "raw",
            "max_iter": 5,
            "batch_size": 8,
            "val_size": 0.1,
            "test_size": 0.1,
            "max_rows": 0,
            "seed": 123,
        },
    )()

    payload = mod.run(args)

    assert payload["metric_rows"] == 2
    package = mod.load_u_pose_model_package(tmp_path / "out" / "model_checkpoints" / "iid_mlp_u" / "model.joblib")
    assert package["kind"] == "u_pose"
    assert package["target_cols"] == ["u_a_deg", "u_b_deg", "u_eta"]
    assert package["u_mapping"] == {"s10": 0.125, "s20": 0.25, "ds1": 0.0, "ds2": 0.0, "dq": 0.0}

