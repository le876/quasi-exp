from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    assert config["parallel"]["patch_workers"] == config["patches"]["count"] == 12
    assert config["parallel"]["heavy_ablation_workers"] == 12
    assert [row["seed_a"] for row in config["reach_extension"]["rounds"]] == [
        20260847,
        20260853,
    ]


def test_v14_1_config_rejects_invalid_heavy_worker_reservation(tmp_path: Path) -> None:
    module = _module()
    config = (ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(config.replace("heavy_ablation_workers: 12", "heavy_ablation_workers: 13"))

    try:
        module.load_config(path)
    except ValueError as error:
        assert "heavy_ablation_workers" in str(error)
    else:
        raise AssertionError("invalid heavy worker reservation must fail closed")


def test_v14_1_config_defaults_heavy_workers_to_patch_workers(tmp_path: Path) -> None:
    module = _module()
    config = (ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "legacy-parallel.yaml"
    path.write_text(config.replace("  heavy_ablation_workers: 12\n", ""), encoding="utf-8")

    loaded = module.load_config(path)

    assert loaded["parallel"]["heavy_ablation_workers"] == loaded["parallel"][
        "patch_workers"
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


def test_ablation_scheduler_reserves_light_capacity_then_uses_all_heavy_slots() -> None:
    module = _module()
    e0 = module.RepairAblation.E0_LEGACY
    e3 = module.RepairAblation.E3_DYNAMIC_INSERTION
    e4 = module.RepairAblation.E4_FRONTIER_ENRICHMENT
    ready = [
        *((f"patch_{index:02d}", e3) for index in range(12)),
        ("patch_00", e0),
        ("patch_03", e4),
    ]

    # E4 is the first heavy job because its E3-derived request is already ready.
    assert module._select_next_ablation_job(
        ready,
        (),
        worker_count=12,
        heavy_soft_limit=8,
    ) == ("patch_03", e4)
    # Eight running heavy jobs reserve the remaining slots for short E0--E2 work.
    assert module._select_next_ablation_job(
        ready,
        (e3,) * 8,
        worker_count=12,
        heavy_soft_limit=8,
    ) == ("patch_00", e0)
    # Once the short queue drains, all twelve slots become eligible for E3/E4.
    assert module._select_next_ablation_job(
        [job for job in ready if job[1] is not e0],
        (e3,) * 8,
        worker_count=12,
        heavy_soft_limit=8,
    ) == ("patch_03", e4)


def test_e3_freezes_neighbor_seed_order_for_independent_e4_worker(monkeypatch) -> None:
    module = _module()
    tasks = pd.DataFrame(
        {
            "task_node_id": [10],
            "condition_number": [1.0],
        }
    )
    candidate = module.AtlasCandidate(
        node_id=10,
        candidate_id="seed",
        beta_rad=[0.0] * 6,
        residual_mm=0.0,
        min_margin_deg=2.0,
        normalized_min_margin=0.5,
        condition_number=1.0,
    )
    graph = type(
        "Graph",
        (),
        {
            "node_by_id": {
                10: module.AtlasTaskNode(10, [0.0, 0.0, 0.0], (12, 11)),
            }
        },
    )()
    monkeypatch.setattr(
        module,
        "select_frontier_enrichment_nodes",
        lambda *_args, **_kwargs: pd.DataFrame.from_records(
            [
                {
                    "task_node_id": 10,
                    "flags": "no_cross_edge",
                    "severe": False,
                    "start_budget": 16,
                    "condition_number": 1.0,
                }
            ]
        ),
    )
    config = module.load_config(ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml")

    requests = module._frontier_enrichment_requests(
        tasks,
        (candidate,),
        graph,
        config,
    )

    assert requests.to_dict("records")[0]["seed_node_ids"] == "10|11|12"


def test_v14_1_endpoint_adapter_is_bit_exact_and_keeps_common_fk_source_untouched() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml")
    project_root = module.project_root_from(ROOT)
    base = module.load_environment(
        project_root,
        module._source_paths(config, project_root)["robot_config"],
    )
    optimized = module._EndpointOnlyForwardAdapter(base)
    rng = np.random.default_rng(20260805)
    beta = rng.uniform(base.bounds[:, 0], base.bounds[:, 1], size=(20, 6))

    np.testing.assert_array_equal(optimized.fk(beta), base.fk(beta))
    np.testing.assert_array_equal(
        optimized.jacobian(beta[0]),
        base.jacobian(beta[0]),
    )


def test_e4_request_checkpoint_binds_request_config_and_inventory_bytes(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml")
    inventory = tmp_path / module.STAGE_DIRS["inventory"]
    stage = tmp_path / module.STAGE_DIRS["ablations"]
    inventory.mkdir(parents=True)
    (inventory / "gate.json").write_text('{"gate_pass": true}\n', encoding="utf-8")
    requests = pd.DataFrame.from_records(
        [
            {
                "task_node_id": 10,
                "flags": "no_cross_edge",
                "severe": False,
                "start_budget": 16,
                "condition_number": 1.0,
                "seed_node_ids": "10|11|12",
            }
        ]
    )

    module._seal_e4_requests(stage, "patch_00", requests, config)
    request_path = module._verified_e4_request_path(
        stage, "patch_00", config
    )
    assert request_path is not None
    module._write_parquet(requests.assign(start_budget=32), request_path)

    assert module._verified_e4_request_path(stage, "patch_00", config) is None


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


def test_formal_stage_writes_failed_gate_when_upstream_scientific_gates_fail(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml")
    for stage_name in ("reach_extension", "repaired_pilot", "exploratory_student"):
        stage = tmp_path / module.STAGE_DIRS[stage_name]
        stage.mkdir(parents=True)
        (stage / "gate.json").write_text(
            json.dumps({"gate_pass": False, "scientific_gate_pass": False}),
            encoding="utf-8",
        )

    gate = module.stage_formal_gate(config, ROOT, tmp_path)

    assert gate["gate_pass"] is False
    assert gate["formal_generation_authorized"] is False
    assert gate["checks"]["operational_completion"] is True
