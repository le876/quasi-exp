from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry9.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_3r_retry9", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry9_contract_freezes_stage_scope_claims_and_sampling() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_3r_retry9_stage0_3.yaml")
    assert module.STAGE_ORDER == (
        "inventory", "chart_salvage", "pairwise_overlap",
        "exploratory_atlas", "student", "summary",
    )
    assert config["beta_coordinate_weights"] == [4.0, 4.0, 2.0, 2.0, 1.0, 1.0]
    assert config["fresh_audit"]["logical_shard_count"] == 48
    assert config["fresh_audit"]["maximum_concurrent_workers"] == 12
    assert config["student"]["beta_loss_only"] is True
    assert config["student"]["device"] == "cpu"
    assert config["claims"]["formal_authorized"] is False
    assert config["claims"]["deployment_authorized"] is False
    assert "frontier" not in module.STAGE_RUNNERS


def test_consultation_input_is_tracked_at_the_registered_digest() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_3r_retry9_stage0_3.yaml")
    path = ROOT / config["sources"]["consultation_input"]
    assert path.is_file()
    assert module.sha256_file(path) == config["consultation_sha256"]


def test_lpt_assignment_is_deterministic_and_cost_balanced() -> None:
    module = _module()
    schedules = pd.DataFrame(
        {"schedule_id": ["a", "b", "c", "d"], "cost": [9, 7, 5, 3]}
    )
    first = module._assign_lpt(schedules, 2)
    second = module._assign_lpt(schedules, 2)
    assert first.equals(second)
    loads = first.groupby("shard_id")["cost"].sum().to_dict()
    assert max(loads.values()) - min(loads.values()) <= 2


def test_summary_authorization_is_fail_closed_in_source() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"frontier_relay_executed": False' in source
    assert '"formal_claim_authorized": False' in source
    assert '"recovered_labels_only": True' in source
    assert "row_padding" not in source


def test_retry9_launcher_forces_cpu_before_python_start() -> None:
    launcher = (
        ROOT / "scripts/pipelines/run_bacra_v14_3r_retry9_stage0_3.sh"
    ).read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=-1" in launcher
    assert launcher.index("export CUDA_VISIBLE_DEVICES=-1") < launcher.index('exec "$PYTHON"')
