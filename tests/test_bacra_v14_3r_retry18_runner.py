from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quasi_exp.teacher.trajectory_evaluation import path_metrics
from scripts.analysis.run_bacra_v14_3r_retry18_zero_tip_parity import (
    EXPERIMENT_ID,
    STAGE_DIRS,
    _prepare_model_output,
    load_config,
    retry17,
)


ROOT = Path(__file__).resolve().parents[1]


def test_retry18_config_locks_axis_aware_tip_and_data_driven_parity_scale() -> None:
    config = load_config(ROOT / "configs/bacra_v14_3r_retry18_zero_tip_parity_smooth.yaml")
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["root_tip"]["axis_core_radius_mm"] >= 2 * config["root_tip"]["radial_step_mm"]
    assert config["root_tip"]["angular_arc_mm"] == 5.0
    assert config["root_tip"]["proposal_power_per_radius"] == 17
    assert config["root_tip"]["minimum_axial_overlap_mm"] == 5.0
    assert config["root_tip"]["minimum_radial_overlap_p50_mm"] == 10.0
    assert config["data_gate"]["unified_graph_maximum_distance_mm"] == 35.0
    assert config["runtime"]["maximum_concurrent_workers"] == 12
    assert config["runtime"]["trajectory_candidate_workers"] == 1
    assert "radial_scale_mm" not in config["student"]
    assert config["student"]["radial_scale_quantile"] == 0.99
    assert config["student"]["parity_label_tolerance_rad"] == 1.0e-8
    assert config["claims"]["diagnostic_only"]
    assert not any(value for key, value in config["claims"].items() if key != "diagnostic_only")


def test_retry18_stage_order_places_teacher_seam_and_lock_correctly() -> None:
    names = list(STAGE_DIRS)
    assert names.index("seam_causal_audit") < names.index("parity_regularity") < names.index("student_ablation")
    assert names.index("dataset_model_lock") < names.index("heldout_trajectories") < names.index("trajectory_evaluation")
    assert list(STAGE_DIRS.values())[-1] == "13_summary"


def test_retry18_protocol_launcher_and_binding_are_single_experiment() -> None:
    config = load_config(ROOT / "configs/bacra_v14_3r_retry18_zero_tip_parity_smooth.yaml")
    assert (ROOT / config["sources"]["governing_protocol"]).is_file()
    launcher = (ROOT / "scripts/pipelines/run_bacra_v14_3r_retry18_zero_tip_parity.sh").read_text(encoding="utf-8")
    assert "BACRA_RETRY18_ZERO_TIP_PARITY_BINDING_SHA" in launcher
    assert "run_bacra_v14_3r_retry18_zero_tip_parity.py" in launcher
    assert "CUDA_VISIBLE_DEVICES=-1" in launcher


def test_path_metric_blocks_one_sparse_spike_even_when_p99_is_small() -> None:
    class FakeEnvironment:
        def fk(self, beta: np.ndarray) -> np.ndarray:
            return np.asarray(beta)[:, :3]

    xyz = np.column_stack([np.linspace(0, 1, 200), np.zeros(200), np.zeros(200)])
    beta = np.column_stack([xyz, np.zeros((200, 3))])
    beta[100, 1] = 0.02
    metrics = path_metrics(xyz, beta, FakeEnvironment(), closed=False)
    assert metrics["path_step_excess_maximum_mm"] > 5.0
    assert metrics["path_step_excess_p99_mm"] < metrics["path_step_excess_maximum_mm"]


def test_model_output_directory_ownership_matches_trainer_contract(tmp_path: Path) -> None:
    standard = tmp_path / "models" / "S0"
    _prepare_model_output(standard, "standard")
    assert standard.parent.is_dir()
    assert not standard.exists()

    parity = tmp_path / "models" / "S2"
    _prepare_model_output(parity, "parity")
    assert parity.is_dir()


def test_trajectory_candidate_worker_cap_limits_resident_processes(tmp_path: Path, monkeypatch) -> None:
    config = load_config(ROOT / "configs/bacra_v14_3r_retry18_zero_tip_parity_smooth.yaml")
    targets = pd.DataFrame({"target_id": [f"t{i}" for i in range(12)], "x": np.arange(12), "y": 0.0, "z": 0.0})
    launched: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **_kwargs):
            self.command = command
            launched.append(command)

        def communicate(self):
            target_path = Path(self.command[self.command.index("--targets") + 1])
            output_path = Path(self.command[self.command.index("--output") + 1])
            pd.read_parquet(target_path).to_parquet(output_path, index=False)
            return "", ""

    monkeypatch.setattr(retry17.subprocess, "Popen", FakeProcess)
    result = retry17._solve_candidates(
        config,
        targets,
        tmp_path / "stage",
        seed_budget=16,
        smoke=False,
        maximum_workers=config["runtime"]["trajectory_candidate_workers"],
        work_namespace="trajectory_k16",
    )
    assert len(launched) == 1
    assert len(result) == len(targets)
    assert (tmp_path / "stage" / "_work" / "trajectory_k16").is_dir()
