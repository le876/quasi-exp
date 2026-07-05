from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "pipelines" / "run_fixed_layer_scale_experiment.py"
    spec = importlib.util.spec_from_file_location("run_fixed_layer_scale_experiment", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scale_paths_are_stable_for_raw_and_relabel_outputs() -> None:
    mod = _load_module()

    paths = mod.scale_paths(data_root=Path("data"), s1=0.125, s2=0.25, samples=20000)

    assert paths.label == "s1_0125_s2_0250"
    assert paths.raw_dir == Path("data/priority_grid_fixed_layer_s1_0125_s2_0250_20k")
    assert paths.relabel_dir == Path(
        "data/priority_grid_fixed_layer_s1_0125_s2_0250_20k_relabel_t1_k32_w40_huber_mean_iter1"
    )


def test_quality_gate_promotes_only_when_raw_and_relabel_pass() -> None:
    mod = _load_module()
    raw = {
        "rows": 20000,
        "gate": {
            "hard_gate_passed": True,
            "rms_rnorm_q95": 0.04,
            "max_tension_n": 900.0,
            "all10_theta_p95_deg": 0.12,
            "multi_branch_ball_ratio": 0.0,
            "all10_tension_p95_n": 45.0,
            "beta_close_tension_p95_n": 45.0,
            "xyz_nn_tension_mae_n": 12.0,
            "beta_nn_tension_mae_n": 12.0,
        },
    }
    relabel = {
        "rows": 20000,
        "gate": {
            **raw["gate"],
            "all10_tension_p95_n": 18.0,
            "beta_close_tension_p95_n": 18.0,
            "same_beta_tension_p95_n": 70.0,
            "xyz_nn_tension_mae_n": 9.0,
            "beta_nn_tension_mae_n": 9.0,
        },
    }

    assert mod.quality_promotes_to_100k(raw, relabel)

    relabel["gate"]["same_beta_tension_p95_n"] = 95.0
    assert not mod.quality_promotes_to_100k(raw, relabel)


def test_build_training_commands_use_fixed_split_dirs_and_tf_gpu() -> None:
    mod = _load_module()

    commands = mod.build_training_commands(
        dataset_path=Path("data/example/dataset.parquet"),
        out_root=Path("runs/example"),
        robot_config=Path("configs/robot.yaml"),
        splits=("iid", "radius"),
        models=("mlp", "mlp_large", "tf_mlp", "tf_mlp_large"),
        seed=20260207,
    )

    assert len(commands) == 2
    first = commands[0]
    assert first[:2] == [mod.DANTE_PYTHON, "scripts/baselines/run_baselines.py"]
    assert "--dataset" in first
    assert "data/example/dataset.parquet" in first
    assert "--save-split-file" in first
    assert "runs/example/iid/split.npz" in first
    assert "--models" in first
    assert "mlp,mlp_large,tf_mlp,tf_mlp_large" in first
    assert "--backend" in first
    assert "both" in first
    assert "--tf-device" in first
    assert "gpu" in first
    assert "--require-gpu" in first
    assert "--feature-set" in first
    assert "poly_heavy" in first


def test_build_relabel_command_accepts_infeasible_when_requested() -> None:
    mod = _load_module()

    strict_cmd = mod.build_relabel_command(
        config=Path("configs/robot.yaml"),
        raw_dir=Path("data/raw"),
        out_dir=Path("data/relabel"),
        anchor_k=32,
        w_anchor=40.0,
        anchor_stat="huber_mean",
        workers=8,
    )
    relaxed_cmd = mod.build_relabel_command(
        config=Path("configs/robot.yaml"),
        raw_dir=Path("data/raw"),
        out_dir=Path("data/relabel"),
        anchor_k=32,
        w_anchor=40.0,
        anchor_stat="huber_mean",
        workers=8,
        accept_infeasible=True,
    )

    assert "--accept-infeasible" not in strict_cmd
    assert "--accept-infeasible" in relaxed_cmd


def test_parser_exposes_accept_infeasible_relabel_flag(monkeypatch) -> None:
    mod = _load_module()

    monkeypatch.setattr("sys.argv", ["run_fixed_layer_scale_experiment.py", "--accept-infeasible-relabel"])

    args = mod.parse_args()

    assert args.accept_infeasible_relabel is True


def test_existing_incomplete_dataset_dir_blocks_generation(tmp_path: Path) -> None:
    mod = _load_module()
    target = tmp_path / "dataset_dir"
    target.mkdir()
    (target / "dataset.parquet").write_text("partial", encoding="utf-8")

    try:
        mod.assert_ready_for_generation(target, report_name="fixed_layer_generation_report.json", force=False)
    except RuntimeError as exc:
        assert "incomplete output directory" in str(exc)
    else:
        raise AssertionError("expected incomplete output directory to fail")


def test_summary_markdown_handles_missing_training_metrics(tmp_path: Path) -> None:
    mod = _load_module()
    summary = {
        "datasets": {
            "20k_raw": {
                "path": "data/raw",
                "quality": {
                    "rows": 20000,
                    "gate": {
                        "hard_gate_passed": True,
                        "all10_theta_p95_deg": 0.1,
                        "all10_tension_p95_n": 40.0,
                        "multi_branch_ball_ratio": 0.0,
                    },
                },
            }
        },
        "training": [
            {
                "dataset_key": "20k_raw",
                "split": "iid",
                "out_dir": "runs/missing",
                "returncode": None,
                "status": "missing",
            }
        ],
        "promotion": {"promote_to_100k": True},
    }

    text = mod.summary_markdown(summary)
    out = tmp_path / "summary.md"
    out.write_text(text, encoding="utf-8")

    assert "20k_raw" in text
    assert "missing" in text
    assert "promote_to_100k" in text
