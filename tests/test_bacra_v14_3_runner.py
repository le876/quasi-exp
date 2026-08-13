from __future__ import annotations

import importlib.util
from pathlib import Path
import json

import pandas as pd


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
    config = module.load_config(
        ROOT / "configs/bacra_v14_3_repaired_5k_student_v2.yaml"
    )

    assert config["parallel"]["patch_workers"] == 12
    assert config["sources"]["v14_2r_root"].endswith("retry6")
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
    assert config["audit_execution"]["logical_shard_count"] == 48
    assert config["audit_execution"]["maximum_concurrent_workers"] == 12
    assert config["audit_execution"]["assignment_strategy"] == "cost_balanced_lpt"
    assert config["dataset"]["wave_size"] == 4096
    assert config["dataset"]["logical_shard_count"] == 48
    assert config["dataset"]["maximum_attempt_rows"] == 100000
    assert config["reach_round8"]["seed_a"] != config["reach_round8"]["seed_b"]


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
        "reach_update",
        "formal_admission",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_router_feature_contract_is_xyz_only() -> None:
    module = _module()
    assert module.ROUTER_FEATURE_COLUMNS == ("x_m", "y_m", "z_m")
    assert "primary_chart_id" not in module.ROUTER_FEATURE_COLUMNS


def test_student_variants_exclude_low_value_default_baselines() -> None:
    module = _module()
    config = module.load_config(
        ROOT / "configs/bacra_v14_3_repaired_5k_student_v2.yaml"
    )
    variants = set(config["student"]["variants"])
    assert "sklearn_mlp" not in variants
    assert "lgbm" not in variants
    assert "knn_diagnostic" in variants


def test_reach_round8_is_conditional_and_round9_is_absent() -> None:
    module = _module()
    assert "reach_update" in module.STAGE_ORDER
    assert all("round9" not in stage for stage in module.STAGE_ORDER)


def test_local_refinement_only_adds_edges_in_registered_region() -> None:
    module = _module()
    tasks = pd.DataFrame(
        {
            "task_node_id": [0, 1, 2, 3],
            "x_m": [0.0, 0.001, 0.002, 0.003],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
        }
    )
    edges = pd.DataFrame(
        {
            "left_node_id": [0],
            "right_node_id": [1],
            "adjacency": ["registered"],
        }
    )
    refined = module._add_local_refinement_edges(
        tasks, edges, [2], neighbor_count=2
    )
    observed = {
        tuple(sorted((int(row.left_node_id), int(row.right_node_id))))
        for row in refined.itertuples(index=False)
    }
    assert (0, 1) in observed
    assert len(observed) > 1
    assert len(refined) >= len(edges)


def test_v14_3_tracked_plan_resolves_inside_the_fixed_point_worktree() -> None:
    module = _module()
    config = module.load_config(
        ROOT / "configs/bacra_v14_3_repaired_5k_student_v2.yaml"
    )
    paths = module._sources(config, Path("/mnt/ML_projects/quasi_exp"))

    assert paths["plan"] == ROOT / "docs/20-BACRA-V14.2R修订执行协议.md"
    assert paths["plan"].is_file()


def test_stage_dependency_digest_changes_when_upstream_completion_changes(tmp_path: Path) -> None:
    module = _module()
    inventory = tmp_path / module.STAGE_DIRS["inventory"]
    inventory.mkdir(parents=True)
    completion = inventory / "completion_manifest.json"
    completion.write_text(json.dumps({"version": 1}), encoding="utf-8")

    first = module._upstream_completion_sha256(tmp_path, "pilot_registry")
    completion.write_text(json.dumps({"version": 2}), encoding="utf-8")
    second = module._upstream_completion_sha256(tmp_path, "pilot_registry")

    assert first != second
