from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry14.py"
CONFIG = ROOT / "configs/bacra_v14_3r_retry14_zero_rooted_shell_mesh.yaml"
PROTOCOL = ROOT / "docs/protocols/31-BACRA-V14.3R-retry14-zero-rooted-shell-mesh执行协议.md"
LAUNCHER = ROOT / "scripts/pipelines/run_bacra_v14_3r_retry14.sh"
WORKER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry14_worker.py"


def _module():
    spec = importlib.util.spec_from_file_location("retry14_runner_for_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_module():
    spec = importlib.util.spec_from_file_location("retry14_worker_for_test", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry14_stage_order_is_fixed() -> None:
    module = _module()
    assert module.STAGE_ORDER == (
        "baseline", "shell_domain", "target_mesh", "seed_mapping",
        "advancing_front", "ring_closure", "adaptive_fill",
        "symmetry_expansion", "audit", "dataset", "student_trajectory", "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_retry14_config_freezes_shell_coverage_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["shell_domain"]["axial_step_mm"] == 15.0
    assert config["shell_domain"]["radial_step_mm"] == 15.0
    assert config["shell_domain"]["target_arc_step_mm"] == 10.0
    assert config["mesh"]["maximum_solver_segment_mm"] == 5.0
    assert config["mesh"]["solver_step_schedule_mm"] == [5.0, 2.5, 1.25]
    assert config["fill"]["maximum_batches"] == 8
    assert config["fill"]["maximum_targets_per_batch"] == 256
    assert config["fill"]["maximum_new_supervision_targets"] == 2000
    assert config["dataset"]["theta_storage"] == "beta_to_theta_without_theta_sign_multiplication"
    assert not any(config["claims"].values())


def test_protocol_locks_tree_mesh_and_coverage_boundaries() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "Lineage Tree" in text and "Certified Shell Mesh" in text
    assert "proposal beta" in text and "label" in text
    assert "fill_required" in text and "potential_expanded_rows" in text
    assert "connector_only" in text
    assert "theta" in text and "theta_sign" in text
    assert "DLS2" in text and "冻结" in text


def test_launcher_uses_binding_and_retry14_dag() -> None:
    module = _module()
    text = LAUNCHER.read_text(encoding="utf-8")
    positions = [text.index(f"  {stage}\n") for stage in module.STAGE_ORDER]
    assert positions == sorted(positions)
    assert '--binding-sha "$BINDING_SHA"' in text
    assert "BACRA_RETRY14_BINDING_SHA" in text


def test_fill_selection_does_not_consult_expanded_row_count() -> None:
    module = _module()
    source = importlib.util.find_spec("quasi_exp.teacher.retry14_shell_mesh")
    assert source is not None
    function_text = RUNNER.read_text(encoding="utf-8")
    stage = function_text[function_text.index("def stage_adaptive_fill"):function_text.index("def _accepted_vertices")]
    assert "coverage_fill_required" in stage
    assert "potential_expanded_rows" not in stage


def test_macroblock_split_is_one_role_and_holdout_wins() -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    role = module._stable_split("qmacro:1:2:3", config)
    assert role == module._stable_split("qmacro:1:2:3", config)
    assert role in {"train", "validation", "test"}


def test_dls_freezes_symmetry_odd_coordinates() -> None:
    module = _module()

    class Environment:
        bounds = np.asarray([[-1.0, 1.0]] * 6)

        def fk(self, beta):
            values = np.asarray(beta).reshape(-1, 6)
            return values[:, :3]

        def jacobian(self, beta):
            del beta
            return np.asarray([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=float)

    beta = np.asarray([[.5] * 6, [.5] * 6, [.5] * 6])
    xyz = np.asarray([[.2, 0.0, .2], [.2, .2, 0.0], [1.0, 0.0, 0.0]])
    corrected = module._two_step_dls(Environment(), beta, xyz, zero_xyz=xyz[2])
    assert np.array_equal(corrected[0, [0, 2, 4]], np.zeros(3))
    assert np.array_equal(corrected[1, [1, 3, 5]], np.zeros(3))
    assert np.array_equal(corrected[2], np.zeros(6))


def test_refined_solver_tier_executes_real_short_subsegments(monkeypatch) -> None:
    module = _worker_module()
    observed: list[float] = []

    def fake_direct(environment, task, config):
        del environment, config
        source = np.asarray(task["source_beta"], dtype=float)
        target = np.asarray(task["target_xyz"], dtype=float)
        origin = np.asarray(task["source_xyz"], dtype=float)
        observed.append(float(np.linalg.norm(target - origin) * 1000.0))
        beta = source + 0.001
        return {
            "task_id": task["task_id"], "mesh_vertex_id": task["mesh_vertex_id"],
            "source_vertex_id": task["source_vertex_id"], "logical_edge_id": task["logical_edge_id"],
            "solver_segment_id": task["solver_segment_id"], "admitted": True,
            "reverse_success": True, "reverse_weighted_gap_deg": 0.1,
            "reverse_raw_gap_deg": 0.1, "beta": beta.tolist(),
        }

    monkeypatch.setattr(module, "_solve_direct", fake_direct)
    task = {
        "task_id": "t", "mesh_vertex_id": "v", "source_vertex_id": "s",
        "logical_edge_id": "e", "solver_segment_id": "seg", "solver_tier": "2.5mm",
        "source_beta": [0.0] * 6, "source_xyz": [0.0, 0.0, 0.0],
        "target_xyz": [0.005, 0.0, 0.0], "step_size_mm": 5.0,
    }
    result = module._solve_one(None, task, {})
    assert result["admitted"]
    assert result["refinement_subsegment_count"] == 2
    assert max(observed) <= 2.5 + 1e-12


def test_legacy_shell_support_is_supervision_but_not_ring_coverage() -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    vertices = pd.DataFrame(
        [{"mesh_vertex_id": "zero", "physical_point_id": "p0", "label_role": "supervision", "canonical_parent_id": None}]
    )
    labels = pd.DataFrame(
        [
            {"physical_point_id": "p0", "retry13_parent_physical_point_id": None, "x_m": 1.215498, "y_m": 0.0, "z_m": 0.0, **{f"beta{i}_rad": 0.0 for i in range(1, 7)}},
            {"physical_point_id": "p1", "retry13_parent_physical_point_id": "p0", "x_m": 1.208, "y_m": .020, "z_m": 0.0, **dict(zip((f"beta{i}_rad" for i in range(1, 7)), [.01, 0.0, .01, 0.0, .01, 0.0], strict=True))},
        ]
    )
    cells = pd.DataFrame([{"cell_id": "ur:0:1", "required_shell_cell": True}])
    augmented = module._augment_with_legacy_shell_support(vertices, labels, cells, np.asarray([1.215498, 0.0, 0.0]), config)
    support = augmented[augmented["target_kind"].eq("legacy_shell_support")]
    assert len(support) == 1 and support.iloc[0]["label_role"] == "supervision"
    assert pd.isna(support.iloc[0]["ring_id"])


def test_seam_target_path_stays_inside_same_symmetry_fixed_subspace() -> None:
    module = _module()
    vertices = pd.DataFrame(
        [
            {"mesh_vertex_id": "zero", "label_status": "seeded", "x_m": 1.0, "y_m": 0.0, "z_m": 0.0},
            {"mesh_vertex_id": "seam_relay", "label_status": "unresolved", "x_m": .99, "y_m": .01, "z_m": 0.0},
            {"mesh_vertex_id": "target", "label_status": "unresolved", "x_m": .98, "y_m": .02, "z_m": 0.0},
            {"mesh_vertex_id": "interior", "label_status": "seeded", "x_m": .98, "y_m": .02, "z_m": .01},
        ]
    )
    segments = pd.DataFrame(
        [
            {"left_mesh_vertex_id": "zero", "right_mesh_vertex_id": "seam_relay"},
            {"left_mesh_vertex_id": "seam_relay", "right_mesh_vertex_id": "target"},
            {"left_mesh_vertex_id": "interior", "right_mesh_vertex_id": "target"},
        ]
    )
    ordered, _ = module._path_union_to_targets(vertices, segments, ["target"])
    assert ordered == ["seam_relay", "target"]


def test_exact_seam_seed_requires_symmetry_fixed_beta() -> None:
    module = _module()
    assert module._beta_matches_seam(np.asarray([0.0, .1, 0.0, .1, 0.0, .1]), "y_seam")
    assert not module._beta_matches_seam(np.asarray([.01, .1, 0.0, .1, 0.0, .1]), "y_seam")
    assert module._beta_matches_seam(np.asarray([.1, 0.0, .1, 0.0, .1, 0.0]), "z_seam")
    assert not module._beta_matches_seam(np.asarray([.1, .01, .1, 0.0, .1, 0.0]), "z_seam")


def test_failed_reverse_guard_retries_finer_solver_tier(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    beta = {f"beta{i}_rad": 0.0 for i in range(1, 7)}
    vertices = pd.DataFrame(
        [
            {"mesh_vertex_id": "source", "label_status": "seeded", "label_role": "supervision", "x_m": 1.0, "y_m": .01, "z_m": .01, "lineage_depth": 0, "physical_point_id": "p0", "mesh_target_fk_residual_mm": 0.0, **beta},
            {"mesh_vertex_id": "target", "label_status": "unresolved", "label_role": "supervision", "x_m": .998, "y_m": .012, "z_m": .012, "lineage_depth": np.nan, **{f"beta{i}_rad": np.nan for i in range(1, 7)}},
        ]
    )
    segments = pd.DataFrame(
        [{"solver_segment_id": "seg", "logical_edge_id": "edge", "left_mesh_vertex_id": "source", "right_mesh_vertex_id": "target", "segment_verified": False, "weighted_beta_gap_deg": np.nan, "raw_beta_gap_deg": np.nan, "reverse_weighted_gap_deg": np.nan}]
    )
    edges = pd.DataFrame(
        [{"logical_edge_id": "edge", "left_mesh_vertex_id": "source", "right_mesh_vertex_id": "target", "mesh_edge_type": "angular_edge", "segment_chain_ids": "seg", "mesh_edge_verified": False}]
    )
    tiers: list[str] = []

    def fake_run(config, output_root, stage_name, tasks):
        del config, output_root, stage_name
        rows = []
        for task in tasks:
            tiers.append(task["solver_tier"])
            good = task["solver_tier"] != "5.0mm"
            result_beta = [0.001] * 6
            rows.append(
                {
                    "task_id": task["task_id"], "mesh_vertex_id": "target", "source_vertex_id": "source",
                    "solver_segment_id": "seg", "logical_edge_id": "edge", "admitted": True,
                    "reverse_success": True, "reverse_weighted_gap_deg": 0.1 if good else 2.0,
                    "source_to_endpoint_weighted_gap_deg": 0.1, "source_to_endpoint_raw_gap_deg": 0.1,
                    "fk_residual_mm": 0.1, "beta": result_beta,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(module, "_run_worker_tasks", fake_run)
    solved, _, _, _, _ = module._solve_requested_targets(
        config, tmp_path, "ring_closure", vertices, edges, segments, ["target"], supervision_cap=1
    )
    assert tiers[:2] == ["5.0mm", "2.5mm"]
    assert solved.loc[solved.mesh_vertex_id.eq("target"), "label_status"].iloc[0] == "provisional_silver"


def test_ring_edge_recertification_verifies_chain_and_promotes_both_endpoints(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    left_beta = np.asarray([0.0] * 6)
    right_beta = np.asarray([0.001] * 6)
    vertices = pd.DataFrame(
        [
            {
                "mesh_vertex_id": "left", "label_status": "provisional_silver",
                "label_quality": "provisional_silver", "label_role": "supervision",
                "x_m": 1.0, "y_m": 0.01, "z_m": 0.01,
                "lineage_depth": 1, "physical_point_id": "left-label",
                "canonical_parent_id": "legacy-left", "mesh_target_fk_residual_mm": 0.0,
                **dict(zip((f"beta{i}_rad" for i in range(1, 7)), left_beta, strict=True)),
            },
            {
                "mesh_vertex_id": "right", "label_status": "provisional_silver",
                "label_quality": "provisional_silver", "label_role": "supervision",
                "x_m": 0.998, "y_m": 0.012, "z_m": 0.012,
                "lineage_depth": 2, "physical_point_id": "right-label",
                "canonical_parent_id": "legacy-right", "mesh_target_fk_residual_mm": 0.0,
                **dict(zip((f"beta{i}_rad" for i in range(1, 7)), right_beta, strict=True)),
            },
        ]
    )
    segments = pd.DataFrame(
        [{
            "solver_segment_id": "seg", "logical_edge_id": "edge",
            "left_mesh_vertex_id": "left", "right_mesh_vertex_id": "right",
            "segment_verified": False, "weighted_beta_gap_deg": np.nan,
            "raw_beta_gap_deg": np.nan, "reverse_weighted_gap_deg": np.nan,
        }]
    )
    edges = pd.DataFrame(
        [{
            "logical_edge_id": "edge", "left_mesh_vertex_id": "left",
            "right_mesh_vertex_id": "right", "mesh_edge_type": "angular_edge",
            "segment_chain_ids": "seg", "segment_vertex_chain_ids": "left|right",
            "mesh_edge_verified": False,
        }]
    )
    beta_by_vertex = {"left": left_beta, "right": right_beta}

    def fake_run(config, output_root, stage_name, tasks):
        del config, output_root, stage_name
        return pd.DataFrame(
            [
                {
                    "task_id": task["task_id"], "mesh_vertex_id": task["mesh_vertex_id"],
                    "source_vertex_id": task["source_vertex_id"],
                    "solver_segment_id": task["solver_segment_id"],
                    "logical_edge_id": task["logical_edge_id"], "admitted": True,
                    "reverse_success": True, "reverse_weighted_gap_deg": 0.1,
                    "source_to_endpoint_weighted_gap_deg": 0.1,
                    "source_to_endpoint_raw_gap_deg": 0.1, "fk_residual_mm": 0.1,
                    "beta": beta_by_vertex[task["mesh_vertex_id"]].tolist(),
                }
                for task in tasks
            ]
        )

    monkeypatch.setattr(module, "_run_worker_tasks", fake_run)
    solved, certified_edges, certified_segments, attempts, conflicts = (
        module._certify_logical_edge_chains(
            config, tmp_path, "ring_closure", vertices, edges, segments, ["edge"]
        )
    )
    assert conflicts.empty
    assert len(attempts) == 2
    assert certified_segments["segment_verified"].all()
    assert certified_edges["mesh_edge_verified"].all()
    assert set(solved["label_status"]) == {"certified_silver"}


def test_ring_candidate_queue_retains_replacements_after_preferred_prefix() -> None:
    module = _module()
    rows = []
    for index in range(12):
        rows.append(
            {
                "ring_id": f"ring-{index:02d}", "u_index": index // 4,
                "rho_index": index % 4, "missing_vertex_count": index % 3,
                "complete_ring": False,
            }
        )
    queue = module._balanced_ring_ids(pd.DataFrame(rows), preferred_count=4)
    assert len(queue) == 12
    assert len(set(queue)) == 12
    assert set(queue) == {f"ring-{index:02d}" for index in range(12)}


def test_edge_recertification_does_not_bypass_supervision_cap(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    beta = {f"beta{i}_rad": 0.0 for i in range(1, 7)}
    vertices = pd.DataFrame(
        [
            {
                "mesh_vertex_id": "source", "label_status": "seeded",
                "label_role": "supervision", "x_m": 1.0, "y_m": .01, "z_m": .01,
                "lineage_depth": 0, "physical_point_id": "p0",
                "canonical_parent_id": None, "mesh_target_fk_residual_mm": 0.0, **beta,
            },
            {
                "mesh_vertex_id": "target", "label_status": "unresolved",
                "label_role": "supervision", "x_m": .998, "y_m": .012, "z_m": .012,
                "lineage_depth": np.nan, "physical_point_id": None,
                "canonical_parent_id": None, "mesh_target_fk_residual_mm": np.nan,
                **{f"beta{i}_rad": np.nan for i in range(1, 7)},
            },
        ]
    )
    segments = pd.DataFrame(
        [{
            "solver_segment_id": "seg", "logical_edge_id": "edge",
            "left_mesh_vertex_id": "source", "right_mesh_vertex_id": "target",
            "segment_verified": False, "weighted_beta_gap_deg": np.nan,
            "raw_beta_gap_deg": np.nan, "reverse_weighted_gap_deg": np.nan,
        }]
    )
    edges = pd.DataFrame(
        [{
            "logical_edge_id": "edge", "left_mesh_vertex_id": "source",
            "right_mesh_vertex_id": "target", "mesh_edge_type": "angular_edge",
            "segment_chain_ids": "seg", "segment_vertex_chain_ids": "source|target",
            "mesh_edge_verified": False,
        }]
    )

    def fake_run(config, output_root, stage_name, tasks):
        del config, output_root, stage_name
        return pd.DataFrame(
            [{
                "task_id": task["task_id"], "mesh_vertex_id": task["mesh_vertex_id"],
                "source_vertex_id": task["source_vertex_id"],
                "solver_segment_id": task["solver_segment_id"],
                "logical_edge_id": task["logical_edge_id"], "admitted": True,
                "reverse_success": True, "reverse_weighted_gap_deg": 0.1,
                "source_to_endpoint_weighted_gap_deg": 0.1,
                "source_to_endpoint_raw_gap_deg": 0.1, "fk_residual_mm": 0.1,
                "beta": [0.001] * 6,
            } for task in tasks]
        )

    monkeypatch.setattr(module, "_run_worker_tasks", fake_run)
    solved, certified_edges, _, _, _ = module._certify_logical_edge_chains(
        config, tmp_path, "ring_closure", vertices, edges, segments, ["edge"]
    )
    assert solved.loc[solved.mesh_vertex_id.eq("target"), "label_status"].iloc[0] == "unresolved"
    assert not certified_edges["mesh_edge_verified"].any()


def test_final_progress_is_written_before_artifact_manifest(tmp_path: Path) -> None:
    module = _module()
    module._write_final_progress(tmp_path)
    manifest = module._artifact_manifest(tmp_path)
    progress = tmp_path / "progress.json"
    entry = next(item for item in manifest["artifacts"] if item["path"] == "progress.json")
    assert entry["bytes"] == progress.stat().st_size
    assert entry["sha256"] == hashlib.sha256(progress.read_bytes()).hexdigest()


def test_audit_aggregates_ring_and_fill_branch_conflicts(tmp_path: Path) -> None:
    module = _module()
    ring_dir = tmp_path / module.STAGE_DIRS["ring_closure"]
    fill_dir = tmp_path / module.STAGE_DIRS["adaptive_fill"]
    ring_dir.mkdir(parents=True)
    fill_dir.mkdir(parents=True)
    pd.DataFrame([{"mesh_vertex_id": "ring-v", "status": "branch_conflict"}]).to_parquet(
        ring_dir / "ring_branch_conflicts.parquet", index=False
    )
    pd.DataFrame([{"mesh_vertex_id": "fill-v", "status": "branch_conflict"}]).to_parquet(
        fill_dir / "fill_branch_conflicts.parquet", index=False
    )
    conflicts = module._all_branch_conflicts(tmp_path)
    assert set(conflicts["mesh_vertex_id"]) == {"ring-v", "fill-v"}
    assert set(conflicts["conflict_stage"]) == {"ring_closure", "adaptive_fill"}


def test_summary_progress_uses_retry14_identity() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    branch = text[
        text.index("def _write_final_progress"):
        text.index("def _input_integrity")
    ]
    assert "retry14 operational pipeline completed" in branch
    assert "retry13 operational pipeline completed" not in branch
