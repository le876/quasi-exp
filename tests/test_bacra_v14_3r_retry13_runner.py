from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry13.py"
CONFIG = ROOT / "configs/bacra_v14_3r_retry13_zero_rooted_connected_roadmap.yaml"
PROTOCOL = ROOT / "docs/protocols/30-BACRA-V14.3R-retry13-zero-rooted-connected-roadmap执行协议.md"
LAUNCHER = ROOT / "scripts/pipelines/run_bacra_v14_3r_retry13.sh"


def _module():
    spec = importlib.util.spec_from_file_location("retry13_runner_for_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry13_stage_order_is_fixed() -> None:
    module = _module()
    assert module.STAGE_ORDER == (
        "input_lineage_audit",
        "connected_backbone",
        "core_axis_spokes",
        "connected_fill",
        "orbit_expansion_prune_split",
        "three_graph_audit",
        "freeze_kinematic_dataset",
        "quotient_student",
        "zero_to_target_trajectory_audit",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_retry13_config_freezes_connected_dataset_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["dataset"]["minimum_expanded_rows"] == 18000
    assert config["dataset"]["target_range_min_rows"] == 20000
    assert config["dataset"]["target_range_max_rows"] == 27000
    assert config["backbone"]["maximum_edge_mm"] == 10
    assert config["audit"]["certified_edge_weighted_p95_max_deg"] == 3
    assert config["audit"]["certified_edge_raw_gt7_rate_max"] == 0.02
    assert config["dataset"]["theta_storage"] == "beta_to_theta_without_theta_sign_multiplication"
    assert config["trajectory"]["expected_lineage_trajectory_count"] == 32
    assert not any(config["claims"].values())


def test_protocol_locks_core_petal_and_claim_boundaries() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "零点有根" in text and "不再要求 exact 19,999" in text
    assert "proposal beta 永远不得作为 label" in text
    assert "镜像时节点和 certified parent edges 同时展开" in text
    assert "quotient macroblock > orbit > row" in text
    assert "far" not in text or "远离零点的直接跨象限运动不属于本轮 Hard Claim" in text


def test_launcher_uses_binding_and_retry13_dag() -> None:
    module = _module()
    text = LAUNCHER.read_text(encoding="utf-8")
    positions = [text.index(f"  {stage}\n") for stage in module.STAGE_ORDER]
    assert positions == sorted(positions)
    assert "--binding-sha \"$BINDING_SHA\"" in text
    assert "BACRA_RETRY13_BINDING_SHA" in text


def test_mixed_oriented_and_new_edges_keep_explicit_new_ids() -> None:
    module = _module()
    frame = pd.DataFrame({"physical_point_id": ["zero", "a", "spoke"]})
    edges = pd.DataFrame(
        {
            "source_physical_point_id": ["legacy", "zero"],
            "target_physical_point_id": ["legacy_target", "spoke"],
            "oriented_source_physical_point_id": ["zero", None],
            "oriented_target_physical_point_id": ["a", None],
        }
    )

    normalized = module._normalize_tree_edges(frame, edges)

    assert normalized["source_physical_point_id"].tolist() == ["zero", "zero"]
    assert normalized["target_physical_point_id"].tolist() == ["a", "spoke"]
    assert normalized["left_position"].tolist() == [0, 0]
    assert normalized["right_position"].tolist() == [1, 2]


def test_summary_progress_uses_retry13_identity() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    branch = text[text.rindex('if stage_name == "summary"'):]
    assert "retry13 operational pipeline completed" in branch
    assert "retry12 operational pipeline completed" not in branch


def test_global_artifact_manifest_has_acyclic_summary_closure(tmp_path: Path) -> None:
    module = _module()
    summary = tmp_path / module.STAGE_DIRS["summary"]
    summary.mkdir(parents=True)
    (summary / "gate.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    (summary / "completion_manifest.json").write_text('{"stage":"summary"}\n', encoding="utf-8")
    (summary / "artifact_manifest.json").write_text('{"old":true}\n', encoding="utf-8")
    (tmp_path / "progress.json").write_text('{"status":"complete"}\n', encoding="utf-8")

    paths = {row["path"] for row in module._artifact_manifest(tmp_path)["artifacts"]}

    assert f'{module.STAGE_DIRS["summary"]}/gate.json' in paths
    assert f'{module.STAGE_DIRS["summary"]}/completion_manifest.json' not in paths
    assert f'{module.STAGE_DIRS["summary"]}/artifact_manifest.json' not in paths
    assert "progress.json" not in paths
