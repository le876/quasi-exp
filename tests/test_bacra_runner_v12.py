from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import run_bacra_v12 as runner

from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    ContinuationOutcome,
    build_canonical_atlas,
)


def test_preset_merge_and_parallel_slices_are_stable() -> None:
    smoke = runner.load_protocol_config(ROOT / "configs/bacra_v12.yaml", "smoke")
    pilot = runner.load_protocol_config(ROOT / "configs/bacra_v12.yaml", "pilot")
    formal = runner.load_protocol_config(ROOT / "configs/bacra_v12.yaml", "formal")
    assert smoke["capability"]["pool_rows"] == 2**16
    assert pilot["capability"]["pool_rows"] == 2**20
    assert formal["capability"]["pool_rows"] == 2**22
    assert runner._slice_ranges(10, 4) == [
        (0, 0, 2),
        (1, 2, 5),
        (2, 5, 7),
        (3, 7, 10),
    ]


def test_atlas_checkpoint_payload_round_trip_preserves_graph_and_charts(
    tmp_path: Path,
) -> None:
    nodes = (
        AtlasTaskNode(0, [0.0, 0.0, 0.0], (1,)),
        AtlasTaskNode(1, [0.001, 0.0, 0.0], (0,)),
    )
    candidates = (
        AtlasCandidate(
            0, "n0", np.zeros(6), 0.1, 2.0, 0.5, condition_number=2.0
        ),
        AtlasCandidate(
            1,
            "n1",
            np.full(6, math.radians(0.1)),
            0.1,
            2.0,
            0.5,
            condition_number=2.0,
        ),
    )

    def continuation(source, target):
        selected = candidates[target.node_id]
        return ContinuationOutcome(
            selected.beta_rad, 0.1, True, True, status="test"
        )

    atlas = build_canonical_atlas(
        nodes, candidates, continuation, policy=AtlasPolicy(root_count=1)
    )
    checkpoint = tmp_path / "atlas.pkl"
    runner._atomic_pickle(runner._atlas_payload(atlas), checkpoint)
    restored = runner._atlas_from_payload(joblib.load(checkpoint))
    assert restored.root_node_ids == atlas.root_node_ids
    assert len(restored.product_graph.directed_edges) == len(
        atlas.product_graph.directed_edges
    )
    assert [chart.selections for chart in restored.charts] == [
        chart.selections for chart in atlas.charts
    ]


def test_fresh_path_executor_returns_aligned_prefix_when_corrector_fails() -> None:
    nodes = (
        AtlasTaskNode(0, [0.0, 0.0, 0.0], (1,)),
        AtlasTaskNode(1, [0.001, 0.0, 0.0], (0, 2)),
        AtlasTaskNode(2, [0.002, 0.0, 0.0], (1,)),
    )
    candidates = tuple(
        AtlasCandidate(
            node_id,
            f"n{node_id}",
            np.full(6, math.radians(0.1 * node_id)),
            0.1,
            2.0,
            0.5,
            condition_number=2.0,
        )
        for node_id in range(3)
    )

    def exact(source, target):
        selected = candidates[target.node_id]
        return ContinuationOutcome(
            selected.beta_rad, 0.1, True, True, status="exact"
        )

    atlas = build_canonical_atlas(
        nodes, candidates, exact, policy=AtlasPolicy(root_count=1)
    )

    def fail_at_second_node(source, target):
        selected = candidates[target.node_id]
        return ContinuationOutcome(
            selected.beta_rad,
            4.0,
            False,
            True,
            status="corrector_failed",
        )

    execute, _repeat = runner._fresh_path_executors(
        atlas, fail_at_second_node
    )
    chart = atlas.charts[0]
    trace = execute(chart, (0, 1, 2))

    assert trace.success is False
    assert trace.status == "corrector_failed"
    assert trace.node_ids == (0, 1)
    assert trace.beta_rad_by_node.shape == (2, 6)


def test_capability_failed_diagnostics_are_strict_json_safe() -> None:
    from quasi_exp.teacher.capability_map import CapabilityGateReport

    report = CapabilityGateReport(
        config_bounds_match=True,
        finite_fk_rate=1.0,
        centerline_core_support=0.0,
        connected_component_exists=False,
        task_graph_connected=False,
        enough_task_nodes=False,
        selected_roi_radius_mm=None,
        gate_pass=False,
        checks={"component": False},
        metrics={"median_nearest_neighbor_m": math.inf},
    )
    assert report.to_dict()["metrics"]["median_nearest_neighbor_m"] is None


def test_dataset_roi_radius_supports_materialized_and_bootstrap_gates() -> None:
    materialized, materialized_source = runner._dataset_roi_radius_mm(
        {"topology_task_region": {"roi_mm": 99.0}},
        {"selected_roi_radius_mm": 10.0},
    )
    assert materialized == 10.0
    assert materialized_source == "capability_gate.selected_roi_radius_mm"

    bootstrapped, bootstrap_source = runner._dataset_roi_radius_mm(
        {"topology_task_region": {"roi_mm": 10.0}},
        {"gate_semantics": "frozen_task_region_admission"},
    )
    assert bootstrapped == 10.0
    assert bootstrap_source == "frozen_config.topology_task_region.roi_mm"


def test_dataset_roi_radius_is_fail_closed_when_not_frozen() -> None:
    with np.testing.assert_raises(KeyError):
        runner._dataset_roi_radius_mm({}, {})
    with np.testing.assert_raises(ValueError):
        runner._dataset_roi_radius_mm(
            {"topology_task_region": {"roi_mm": 0.0}}, {}
        )


def test_v12_2_extended_config_freezes_topology_and_parallelism() -> None:
    config = runner.load_protocol_config(
        ROOT / "configs/bacra_v12_2_topology_preserving.yaml", "pilot"
    )
    assert config["protocol_id"] == (
        "branch-aware-canonical-region-atlas-v12.2-"
        "topology-preserving-task-graph"
    )
    assert config["topology_task_region"]["voxel_mm"] == 7.5
    assert config["topology_task_region"]["expected_task_node_count"] == 2881
    assert config["topology_task_region"]["expected_task_edge_count"] == 5546
    assert config["parallel"]["default_workers"] == 12
    assert config["parallel"]["product_workers"] == 12
    assert config["student"]["probe_seed_count"] == 5


def test_v12_3_recursive_config_freezes_closure_without_losing_base() -> None:
    config = runner.load_protocol_config(
        ROOT / "configs/bacra_v12_3_branch_lift_closure.yaml", "pilot"
    )
    assert config["protocol_id"] == (
        "branch-aware-canonical-region-atlas-v12.3-branch-lift-closure"
    )
    assert config["robot_config"] == "configs/robot_rods_only_standard_100k.yaml"
    assert config["atlas"]["minimum_chart_coverage"] == 0.80
    assert config["branch_lift_closure"] == {
        "match_threshold_deg": 0.5,
        "cluster_threshold_deg": 0.5,
        "gold_margin_deg": 1.5,
        "residual_max_mm": 3.0,
        "propagation_rounds": 5,
        "confirmation_rounds": 1,
        "minimum_chart_coverage": 0.80,
        "solver_time_limit_s": 1800,
        "solver_mip_rel_gap": 0.0,
        "require_solver_certificate": True,
    }
    assert config["parallel"]["closure_workers"] == 12
    assert "extends" not in config


def test_v12_5_student_uses_full_d3_without_changing_ablation_set() -> None:
    config = runner.load_protocol_config(
        ROOT / "configs/bacra_v12_5_composable_lift.yaml", "pilot"
    )
    assert config["student"]["use_full_d3_dataset"] is True
    assert config["dense"]["row_count"] == 50000
