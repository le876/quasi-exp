from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3_repaired_5k_student.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_3_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v14_3_freezes_unique_supervision_budget_without_padding() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_3_repaired_5k_student.yaml")

    assert config["parallel"]["patch_workers"] == 12
    assert config["pilot"]["parent_cell_count"] == 5000
    assert config["pilot"]["task_probe_count"] == 25000
    assert config["pilot"]["root_count"] == 32
    assert config["dataset"]["minimum_unique_rows"] == 20000
    assert config["dataset"]["maximum_unique_rows"] == 50000
    assert config["dataset"]["row_padding"] is False
    assert config["dataset"]["multiparent_count"] == 2
    assert config["pilot"]["bootstrap_replicates"] == 2000
    assert config["formal_gate"]["labelable_measure_min"] == 0.80
    assert config["student"]["random_set_max_is_diagnostic_only"] is True


def test_v14_3_stage_order_keeps_student_before_formal_admission() -> None:
    module = _module()

    assert module.STAGE_ORDER == (
        "inventory",
        "pilot_registry",
        "root_charts",
        "stitched_atlas",
        "fixed_budget_dataset",
        "students",
        "representation_decision",
        "formal_admission",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_router_feature_contract_is_xyz_only() -> None:
    module = _module()
    assert module.ROUTER_FEATURE_COLUMNS == ("x_m", "y_m", "z_m")
    assert "primary_chart_id" not in module.ROUTER_FEATURE_COLUMNS


def test_v14_3_tracked_plan_resolves_inside_the_fixed_point_worktree() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_3_repaired_5k_student.yaml")
    paths = module._sources(config, Path("/mnt/ML_projects/quasi_exp"))

    assert paths["plan"] == ROOT / "docs/20-BACRA-V14.2R修订执行协议.md"
    assert paths["plan"].is_file()
