from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "pipelines" / "run_fixed_layer_100k_model_comparison.py"
    spec = importlib.util.spec_from_file_location("run_fixed_layer_100k_model_comparison", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_direct_command_uses_existing_split_file_and_comparison_out_dir() -> None:
    mod = _load_module()

    cmd = mod.build_direct_command(
        dataset=Path("data/relabel/dataset.parquet"),
        split_file=Path("runs/original/iid/split.npz"),
        out_dir=Path("runs/comparison/direct/relabel/iid"),
        robot_config=Path("configs/robot.yaml"),
        models="knn,rf,lgbm",
        seed=123,
    )

    assert cmd[:2] == [mod.DANTE_PYTHON, "scripts/baselines/run_baselines.py"]
    assert "--dataset" in cmd
    assert "data/relabel/dataset.parquet" in cmd
    assert "--split-file" in cmd
    assert "runs/original/iid/split.npz" in cmd
    assert "--out-dir" in cmd
    assert "runs/comparison/direct/relabel/iid" in cmd
    assert "--models" in cmd
    assert "knn,rf,lgbm" in cmd
    assert "--backend" in cmd
    assert "classic" in cmd
    assert "--eval-splits" in cmd
    assert "val,test" in cmd


def test_beta_first_command_uses_dataset_meta_and_expected_prefix() -> None:
    mod = _load_module()

    cmd = mod.build_beta_first_command(
        dataset=Path("data/raw/dataset.parquet"),
        meta=Path("data/raw/dataset_meta.parquet"),
        out_prefix=Path("runs/comparison/beta_first/raw"),
        robot_config=Path("configs/robot.yaml"),
        splits=("iid", "radius"),
        models="mlp,rf,knn",
        tension_modes="direct,beta_knn",
        seed=456,
    )

    assert cmd[:2] == [mod.DANTE_PYTHON, "scripts/baselines/run_beta_first_baselines.py"]
    assert "--dataset" in cmd
    assert "data/raw/dataset.parquet" in cmd
    assert "--meta" in cmd
    assert "data/raw/dataset_meta.parquet" in cmd
    assert "--out-prefix" in cmd
    assert "runs/comparison/beta_first/raw" in cmd
    assert "--splits" in cmd
    assert "iid,radius" in cmd
    assert "--tension-modes" in cmd
    assert "direct,beta_knn" in cmd


def test_collect_summary_merges_existing_direct_and_comparison_metrics(tmp_path: Path) -> None:
    mod = _load_module()
    existing = tmp_path / "existing" / "iid"
    comparison = tmp_path / "comparison" / "direct" / "raw" / "iid"
    existing.mkdir(parents=True)
    comparison.mkdir(parents=True)
    (existing / "all_metrics.json").write_text(
        json.dumps(
            {
                "mlp_large": {
                    "theta_mae_deg": 0.1,
                    "tension_mae_n": 10.0,
                    "tension_rmse_n": 12.0,
                    "ee_pos_p95_mm": 5.0,
                    "fit_time_s": 3.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (comparison / "all_metrics.json").write_text(
        json.dumps(
            {
                "knn": {
                    "theta_mae_deg": 0.2,
                    "tension_mae_n": 8.0,
                    "tension_rmse_n": 11.0,
                    "ee_pos_p95_mm": 6.0,
                    "fit_time_s": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    rows = mod.collect_direct_rows(
        dataset_key="raw",
        split="iid",
        existing_metrics=existing / "all_metrics.json",
        comparison_metrics=comparison / "all_metrics.json",
    )

    assert [row["model"] for row in rows] == ["mlp_large", "knn"]
    assert rows[0]["dataset"] == "raw"
    assert rows[0]["split"] == "iid"
    assert rows[1]["tension_mae_n"] == 8.0
