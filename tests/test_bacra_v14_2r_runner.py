from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_2r_stitched_atlas.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_2r_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v14_2r_config_freezes_twelve_single_threaded_workers_and_gates() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_2r_stitched_atlas.yaml")

    assert config["parallel"] == {"patch_workers": 12, "numerical_threads_per_worker": 1}
    assert config["diagnostic_patch_ids"] == ["patch_00", "patch_03", "patch_07", "patch_09"]
    assert config["diagnostic_only_patch_ids"] == ["patch_09"]
    assert config["confirmation_patch_ids"] == ["patch_08", "patch_10", "patch_11", "patch_12"]
    assert config["mechanism_gate"]["minimum_passing_patches"] == 3
    assert config["audit_v2"]["repeats_per_direction"] == 3
    assert [tier["tier_id"] for tier in config["audit_v2"]["retry_tiers"]] == ["R0", "R1", "R2"]
    assert config["chart_repair"]["minimum_chart_fraction"] > 0
    assert config["meso_bridge"]["parent_cell_count"] == 512


def test_v14_2r_stage_order_separates_repair_confirmation_and_reach() -> None:
    module = _module()

    assert module.STAGE_ORDER == (
        "inventory",
        "replacement_confirmation",
        "artifact_diagnostics",
        "rooted_baseline",
        "registered_retry",
        "patch07_local_audit",
        "search_stability",
        "mechanism_gate",
        "reach_round7",
        "confirmation",
        "meso_bridge",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_v14_2r_scientific_gate_failure_is_not_an_operational_error(tmp_path) -> None:
    module = _module()
    skipped = module.write_scientific_skip(tmp_path, "upstream_mechanism_gate_failed")

    assert skipped["gate_pass"] is False
    assert skipped["operational_completion"] is True
    assert (tmp_path / "gate.json").is_file()
