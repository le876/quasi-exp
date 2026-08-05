from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_1_cross_cell_repair.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_1_repair_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v14_1_config_freezes_nested_ablation_and_no_padding_contract() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml")

    assert tuple(config["repair"]["ablations"]) == tuple(
        item.value for item in module.RepairAblation
    )
    assert config["exploratory_dataset"]["row_padding"] is False
    assert config["formal_gate"]["row_padding"] is False
    assert config["formal_gate"]["deployment_claim"] is False
    assert [row["seed_a"] for row in config["reach_extension"]["rounds"]] == [
        20260847,
        20260853,
    ]


def test_v14_1_runner_wires_every_registered_stage() -> None:
    module = _module()

    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)
    assert module.STAGE_RUNNERS["ablations"] is module.stage_ablations
    assert module.STAGE_RUNNERS["stability"] is module.stage_stability
    assert module.STAGE_RUNNERS["reach_extension"] is module.stage_reach_extension
    assert module.STAGE_RUNNERS["repaired_pilot"] is module.stage_repaired_pilot
    assert module.STAGE_RUNNERS["exploratory_student"] is module.stage_exploratory_student
    assert module.STAGE_RUNNERS["formal_gate"] is module.stage_formal_gate


def test_formal_gate_never_passes_by_relaxing_failed_reach_or_representation() -> None:
    module = _module()
    report = dict(
        module.evaluate_formal_admission_gate(
            reach_convergence_gate=False,
            selection_stability_gate=True,
            labelable_measure_ratio=1.0,
            minimum_x_bin_coverage=1.0,
            unresolved_ratio=0.0,
            abstention_ratio=0.0,
            single_cell_chart_measure_ratio=0.0,
            largest_region_x_bin_count=10,
            atlas_audit_gate=True,
            representation_frozen=True,
            n_min=1,
            student_gate=True,
        )
    )

    assert report["gate_pass"] is False
    assert report["checks"]["reach_convergence"] is False
    assert report["padding_authorized"] is False
    assert report["deployment_claim"] is False
