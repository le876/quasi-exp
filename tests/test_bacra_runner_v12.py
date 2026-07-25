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
