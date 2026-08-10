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
    assert config["audit_execution"]["progress_every_schedules"] > 0
    assert config["diagnostic_patch_ids"] == ["patch_00", "patch_03", "patch_07", "patch_09"]
    assert config["diagnostic_only_patch_ids"] == ["patch_09"]
    assert config["confirmation_patch_ids"] == ["patch_08", "patch_10", "patch_11", "patch_12"]
    assert config["mechanism_gate"]["minimum_passing_patches"] == 3
    assert config["audit_v2"]["repeats_per_direction"] == 3
    assert [tier["tier_id"] for tier in config["audit_v2"]["retry_tiers"]] == ["R0", "R1", "R2"]
    assert config["chart_repair"]["minimum_chart_fraction"] > 0
    assert config["meso_bridge"]["parent_cell_count"] == 512


def test_frame_stability_selects_beta_columns_without_pandas_tuple_indexing(
    monkeypatch,
    tmp_path,
) -> None:
    module = _module()
    beta = {
        name: [0.1 * index, 0.2 * index]
        for index, name in enumerate(module.BETA_COLUMNS, start=1)
    }
    left = pd.DataFrame(
        {
            "task_node_id": [10, 11],
            "abstained": [False, False],
            **beta,
        }
    )
    right = left.copy()
    monkeypatch.setattr(
        module,
        "_verified_edge_entities",
        lambda _directory, _config: {"edge:10:11"},
    )
    config = {
        "audit_v2": {"geometry_max_deg": 1.0},
        "search_stability": {
            "coverage_jaccard_min": 0.95,
            "beta_p95_max_deg": 1.0,
            "beta_max_deg": 2.0,
            "assignment_change_max": 0.05,
            "verified_edge_change_max": 0.05,
        },
    }

    report = module._frame_stability(
        left,
        right,
        config,
        left_directory=tmp_path,
        right_directory=tmp_path,
    )

    assert report["gate_pass"] is True
    assert report["coverage_jaccard"] == 1.0
    assert report["beta_p95_deg"] == 0.0
    assert report["beta_max_deg"] == 0.0
    assert report["assignment_change_ratio"] == 0.0
    assert report["verified_edge_change_ratio"] == 0.0


def test_patch_jobs_register_full_shard_sets_against_one_global_token_pool(
    monkeypatch,
    tmp_path,
) -> None:
    module = _module()
    commands: list[list[str]] = []

    class CompletedProcess:
        returncode = 0
        pid = 12345

        def __init__(self, command, **_kwargs) -> None:
            commands.append(list(command))

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", CompletedProcess)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    config = {
        "config_path": tmp_path / "config.yaml",
        "parallel": {"patch_workers": 12},
        "audit_execution": {"total_shard_workers": 12},
    }
    jobs = [(f"patch_{index:02d}", "baseline") for index in range(5)]

    module._run_patch_jobs(config, tmp_path, "search_stability", jobs)

    assert len(commands) == len(jobs)
    shard_counts = {
        int(command[command.index("--audit-shards-per-patch") + 1])
        for command in commands
    }
    token_pools = {
        command[command.index("--audit-token-pool") + 1]
        for command in commands
    }
    assert shard_counts == {12}
    assert token_pools == {
        str(tmp_path / "06_search_stability" / "_audit_worker_tokens")
    }


def test_global_audit_token_pool_refills_a_released_slot(tmp_path) -> None:
    module = _module()
    first = module._AuditWorkerTokenPool(tmp_path, capacity=2)
    second = module._AuditWorkerTokenPool(tmp_path, capacity=2)

    token_0 = first.try_acquire()
    token_1 = first.try_acquire()
    assert token_0 is not None
    assert token_1 is not None
    assert second.try_acquire() is None

    token_0.release()
    replacement = second.try_acquire()
    assert replacement is not None

    replacement.release()
    token_1.release()


def test_real_schedule_subprocess_shards_are_exact_and_resumable(tmp_path) -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_2r_stitched_atlas.yaml")
    config["audit_execution"]["progress_every_schedules"] = 1
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
        token_pool=module._AuditWorkerTokenPool(tmp_path / "tokens", capacity=1),
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
    progress_paths = sorted(
        (tmp_path / "_audit_checkpoints/chart_initial").glob("shard_*/progress.json")
    )
    assert len(progress_paths) == 2
    for path in progress_paths:
        progress = module._read_json(path)
        assert progress["status"] == "complete"
        assert progress["completed_schedule_count"] == progress["total_schedule_count"]


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
    assert paths["performance_protocol"] == (
        ROOT / "docs/22-BACRA-V14.2R聚合与work-conserving调度修复协议.md"
    )
    assert paths["performance_protocol"].is_file()


def test_v14_2r_retry3_launchers_keep_benchmark_and_formal_roots_distinct() -> None:
    benchmark = (
        ROOT
        / "scripts/pipelines/run_bacra_v14_2r_patch07_shard_benchmark_retry2.sh"
    )
    retry3 = ROOT / "scripts/pipelines/run_bacra_v14_2r_stitched_atlas_retry3.sh"

    assert benchmark.is_file()
    assert retry3.is_file()
    assert benchmark.stat().st_mode & 0o111
    assert retry3.stat().st_mode & 0o111
    benchmark_text = benchmark.read_text(encoding="utf-8")
    retry3_text = retry3.read_text(encoding="utf-8")
    assert "runs/bacra_v14_2r_patch07_shard_benchmark_retry2" in benchmark_text
    assert "runs/bacra_v14_2r_stitched_atlas_retry3" in retry3_text
    assert "--verify-only" in retry3_text
    assert "--require-pass" in retry3_text
    assert "bacra_v14_2r_stitched_atlas_retry2" not in retry3_text


def test_v14_2r_scientific_gate_failure_is_not_an_operational_error(tmp_path) -> None:
    module = _module()
    skipped = module.write_scientific_skip(tmp_path, "upstream_mechanism_gate_failed")

    assert skipped["gate_pass"] is False
    assert skipped["operational_completion"] is True
    assert (tmp_path / "gate.json").is_file()
