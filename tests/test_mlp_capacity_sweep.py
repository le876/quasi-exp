from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "pipelines" / "run_fixed_layer_mlp_capacity_sweep.py"
    spec = importlib.util.spec_from_file_location("run_fixed_layer_mlp_capacity_sweep", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capacity_command_uses_fixed_split_and_params_file() -> None:
    mod = _load_module()

    cmd = mod.build_training_command(
        dataset=Path("data/relabel/dataset.parquet"),
        split_file=Path("runs/relabel/iid/split.npz"),
        out_dir=Path("runs/capacity/relabel/iid/L2_alpha1e-6"),
        params_file=Path("runs/capacity/params/L2_alpha1e-6.yaml"),
        robot_config=Path("configs/robot.yaml"),
        seed=123,
    )

    assert cmd[:2] == [mod.DANTE_PYTHON, "scripts/baselines/run_baselines.py"]
    assert "--models" in cmd
    assert "mlp_large" in cmd
    assert "--split-file" in cmd
    assert "runs/relabel/iid/split.npz" in cmd
    assert "--params-file" in cmd
    assert "runs/capacity/params/L2_alpha1e-6.yaml" in cmd
    assert "--eval-splits" in cmd
    assert "train,val,test" in cmd
    assert "--skip-fk" not in cmd


def test_overfit_detector_requires_train_gain_and_val_or_test_regression() -> None:
    mod = _load_module()

    best = {
        "train_tension_mae_n": 20.0,
        "val_tension_mae_n": 30.0,
        "test_tension_mae_n": 40.0,
    }
    overfit = {
        "train_tension_mae_n": 17.0,
        "val_tension_mae_n": 36.0,
        "test_tension_mae_n": 46.0,
    }
    improving = {
        "train_tension_mae_n": 17.0,
        "val_tension_mae_n": 28.0,
        "test_tension_mae_n": 37.0,
    }

    assert mod.is_overfit_regression(best, overfit)
    assert not mod.is_overfit_regression(best, improving)


def test_recommend_capacity_prefers_smaller_when_scores_within_five_percent() -> None:
    mod = _load_module()

    rows = [
        {"capacity": "L1", "split": "radius", "test_tension_mae_n": 40.0, "fit_time_s": 100.0, "overfit": False},
        {"capacity": "L2", "split": "radius", "test_tension_mae_n": 39.0, "fit_time_s": 200.0, "overfit": False},
        {"capacity": "L3", "split": "radius", "test_tension_mae_n": 38.5, "fit_time_s": 500.0, "overfit": True},
    ]

    recommended = mod.recommend_capacity(rows)

    assert recommended["capacity"] == "L1"
    assert recommended["reason"] == "within_5_percent_choose_smaller"


def test_read_trial_metrics_extracts_train_val_test(tmp_path: Path) -> None:
    mod = _load_module()
    metrics_path = tmp_path / "mlp_large" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "fit_time_s": 12.0,
                "metrics": {
                    "train": {"theta_mae_deg": 0.1, "tension_mae_n": 4.0},
                    "val": {"theta_mae_deg": 0.2, "tension_mae_n": 6.0},
                    "test": {"theta_mae_deg": 0.3, "tension_mae_n": 8.0, "ee_pos_p95_mm": 9.0},
                },
            }
        ),
        encoding="utf-8",
    )

    row = mod.read_trial_metrics(tmp_path, dataset="relabel", split="iid", capacity="L2", alpha=1e-6)

    assert row["dataset"] == "relabel"
    assert row["capacity"] == "L2"
    assert row["train_tension_mae_n"] == 4.0
    assert row["val_tension_mae_n"] == 6.0
    assert row["test_tension_mae_n"] == 8.0
    assert row["test_ee_pos_p95_mm"] == 9.0
