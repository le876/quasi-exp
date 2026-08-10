from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


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
    assert config["audit_execution"]["total_shard_workers"] == 12
    assert config["audit_execution"]["checkpoint_resume"] is True
    assert config["diagnostic_patch_ids"] == ["patch_00", "patch_03", "patch_07", "patch_09"]
    assert config["diagnostic_only_patch_ids"] == ["patch_09"]
    assert config["confirmation_patch_ids"] == ["patch_08", "patch_10", "patch_11", "patch_12"]
    assert config["mechanism_gate"]["minimum_passing_patches"] == 3
    assert config["audit_v2"]["repeats_per_direction"] == 3
    assert [tier["tier_id"] for tier in config["audit_v2"]["retry_tiers"]] == ["R0", "R1", "R2"]
    assert config["chart_repair"]["minimum_chart_fraction"] > 0
    assert config["meso_bridge"]["parent_cell_count"] == 512


def test_v14_2r_distributes_a_fixed_twelve_audit_workers_across_patch_jobs() -> None:
    module = _module()

    assert module._audit_shards_per_patch(1, total_workers=12) == 12
    assert module._audit_shards_per_patch(4, total_workers=12) == 3
    assert module._audit_shards_per_patch(5, total_workers=12) == 2
    assert module._audit_shards_per_patch(12, total_workers=12) == 1
    assert module._audit_shard_allocations(1, total_workers=12) == (12,)
    assert module._audit_shard_allocations(4, total_workers=12) == (3, 3, 3, 3)
    assert module._audit_shard_allocations(5, total_workers=12) == (3, 3, 2, 2, 2)
    assert module._audit_shard_allocations(9, total_workers=12) == (
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
    )


def test_real_schedule_subprocess_shards_are_exact_and_resumable(tmp_path) -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_2r_stitched_atlas.yaml")
    project_root = module.project_root_from(ROOT)
    paths = module._source_paths(config, project_root)
    legacy_config = module.legacy.load_config(paths["legacy_config"])
    tasks, _legacy_edges, _candidates, _assignments = module.legacy._patch_inputs(
        legacy_config, project_root, "patch_07"
    )
    task_edges = pd.read_parquet(
        paths["retry4"]
        / "01_patch_ablations/patch_07/E3_dynamic_insertion_task_edges.parquet"
    )
    historical = paths["v14_2"] / "02_mechanism_experiment/patch_07/S4"
    hypotheses = pd.read_parquet(historical / "section_hypotheses.parquet")
    selected_edges = pd.read_parquet(historical / "selected_edges.parquet")
    nodes = module.atlas_nodes_from_frames(tasks, task_edges)
    growth = module.section_growth_from_frames(nodes, hypotheses, selected_edges)
    edge = selected_edges.iloc[0]
    schedules = pd.DataFrame.from_records(
        [
            {
                "patch_id": "patch_07",
                "method": "subprocess_smoke",
                "schedule_id": "schedule_subprocess_smoke",
                "unique_entity_id": "chart_smoke:edge",
                "chart_id": str(edge.chart_id),
                "audit_kind": "edge",
                "path_node_ids": [int(edge.left_node_id), int(edge.right_node_id)],
                "unordered_signature": [
                    min(int(edge.left_node_id), int(edge.right_node_id)),
                    max(int(edge.left_node_id), int(edge.right_node_id)),
                ],
                "task_cell_ids": [int(edge.left_node_id), int(edge.right_node_id)],
                "measure_weight": 2.0,
                "primary_usage": True,
            }
        ]
    )
    executor = module._SubprocessAuditExecutor(
        config=config,
        project_root=project_root,
        tasks=tasks,
        task_edges=task_edges,
        patch_directory=tmp_path,
        shard_count=2,
    )
    policy = module._audit_policy(config)
    first = executor(growth, schedules, None, policy, object(), "chart_initial")
    report_paths = sorted(
        (tmp_path / "_audit_checkpoints/chart_initial").glob("shard_*/report.json")
    )
    mtimes = {path: path.stat().st_mtime_ns for path in report_paths}
    second = executor(growth, schedules, None, policy, object(), "chart_initial")

    assert len(first) == 2 * policy.repeats_per_direction
    assert first.equals(second)
    assert len(report_paths) == 2
    assert mtimes == {path: path.stat().st_mtime_ns for path in report_paths}
    aggregate = module._read_json(
        tmp_path / "_audit_checkpoints/chart_initial/gate.json"
    )
    assert aggregate["gate_pass"] is True
    assert aggregate["resumed_shard_count"] == 2


def test_growth_checkpoint_rehydrates_only_under_the_same_input_closure(tmp_path) -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_2r_stitched_atlas.yaml")
    project_root = module.project_root_from(ROOT)
    paths = module._source_paths(config, project_root)
    legacy_config = module.legacy.load_config(paths["legacy_config"])
    tasks, _legacy_edges, _candidates, _assignments = module.legacy._patch_inputs(
        legacy_config, project_root, "patch_07"
    )
    task_edges = pd.read_parquet(
        paths["retry4"]
        / "01_patch_ablations/patch_07/E3_dynamic_insertion_task_edges.parquet"
    )
    historical = paths["v14_2"] / "02_mechanism_experiment/patch_07/S4"
    frames = {
        "section_hypotheses": pd.read_parquet(
            historical / "section_hypotheses.parquet"
        ),
        "selected_edges": pd.read_parquet(historical / "selected_edges.parquet"),
    }
    nodes = module.atlas_nodes_from_frames(tasks, task_edges)
    growth = module.section_growth_from_frames(
        nodes, frames["section_hypotheses"], frames["selected_edges"]
    )
    module._write_growth_checkpoint(
        tmp_path,
        growth=growth,
        root_report={"root_budget_saturated": True},
        input_sha256="input-a",
        config=config,
        patch_id="patch_07",
        variant="baseline",
        beam_width=4,
        consensus=False,
    )

    loaded = module._load_growth_checkpoint(
        tmp_path,
        tasks=tasks,
        task_edges=task_edges,
        input_sha256="input-a",
        config=config,
        patch_id="patch_07",
        variant="baseline",
    )
    assert loaded is not None
    restored, root_report, restored_frames = loaded
    assert len(restored.charts) == len(growth.charts)
    assert root_report["root_budget_saturated"] is True
    assert set(restored_frames) >= {"section_hypotheses", "selected_edges"}

    assert (
        module._load_growth_checkpoint(
            tmp_path,
            tasks=tasks,
            task_edges=task_edges,
            input_sha256="input-b",
            config=config,
            patch_id="patch_07",
            variant="baseline",
        )
        is None
    )


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


def test_v14_2r_tracked_plan_resolves_inside_the_fixed_point_worktree() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_2r_stitched_atlas.yaml")
    paths = module._source_paths(config, Path("/mnt/ML_projects/quasi_exp"))

    assert paths["plan"] == ROOT / "docs/20-BACRA-V14.2R修订执行协议.md"
    assert paths["plan"].is_file()


def test_v14_2r_scientific_gate_failure_is_not_an_operational_error(tmp_path) -> None:
    module = _module()
    skipped = module.write_scientific_skip(tmp_path, "upstream_mechanism_gate_failed")

    assert skipped["gate_pass"] is False
    assert skipped["operational_completion"] is True
    assert (tmp_path / "gate.json").is_file()
