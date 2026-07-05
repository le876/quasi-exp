from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "pipelines" / "run_priority_grid_layer_experiments.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_priority_grid_layer_experiments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_fixed_layers_writes_one_dataset_per_s1_s2_pair(tmp_path: Path) -> None:
    mod = load_module()
    dataset = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "x_m": [0.0, 0.1, 0.2],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            **{f"theta_{i}_rad": [0.0, 0.0, 0.0] for i in range(1, 31)},
            **{f"tension_{i}_n": [10.0, 20.0, 30.0] for i in range(1, 13)},
        }
    )
    meta = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "s1": [0.0, 0.0, 0.125],
            "s2": [0.0, 0.25, 0.25],
            "source_component": ["priority_grid", "priority_grid", "priority_grid"],
        }
    )

    summary = mod.export_fixed_layers(dataset, meta, tmp_path)

    assert summary["layer_count"] == 3
    layer_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert layer_dirs == ["layer_s1_0000_s2_0000", "layer_s1_0000_s2_0250", "layer_s1_0125_s2_0250"]
    exported_meta = pd.read_parquet(tmp_path / "layer_s1_0000_s2_0250" / "dataset_meta.parquet")
    assert exported_meta["sample_id"].tolist() == [0]
    assert exported_meta["source_sample_id"].tolist() == [1]
    assert exported_meta["layer_label"].tolist() == ["s1_0000_s2_0250"]
    assert exported_meta["source_component"].tolist() == ["s1_0000_s2_0250"]


def test_build_training_commands_uses_fixed_split_paths() -> None:
    mod = load_module()

    commands = mod.build_training_commands(
        dataset_path=Path("data/example/dataset.parquet"),
        out_root=Path("runs/example"),
        robot_config=Path("configs/robot.yaml"),
        splits=("iid", "radius"),
        models=("mlp", "tf_mlp"),
        backend="both",
        seed=123,
        dry_run=True,
    )

    assert len(commands) == 2
    assert commands[0][0].endswith("/python")
    assert "scripts/baselines/run_baselines.py" in commands[0]
    assert "--save-split-file" in commands[0]
    assert "runs/example/iid/split.npz" in commands[0]
    assert "--models" in commands[0]
    assert "mlp,tf_mlp" in commands[0]


def test_write_summary_markdown_handles_missing_metrics(tmp_path: Path) -> None:
    mod = load_module()
    out = tmp_path / "summary.md"

    mod.write_summary_markdown(
        {
            "diagnostics": {
                "small_layer": {
                    "rows": 3,
                    "metrics": {
                        "all10_theta_p95_deg": None,
                        "all10_tension_p95_n": None,
                        "multi_branch_ball_ratio": 0.0,
                        "xyz_nn_tension_mae_n": None,
                    },
                }
            },
            "training": [],
        },
        out,
    )

    text = out.read_text(encoding="utf-8")
    assert "small_layer" in text
    assert "nan" in text
