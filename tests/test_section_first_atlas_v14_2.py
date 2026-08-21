from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    ContinuationOutcome,
)
from quasi_exp.teacher.section_first_atlas import (
    RootedSectionPolicy,
    build_section_first_atlas,
)
from quasi_exp.teacher.section_stability import (
    SectionStabilityPolicy,
    compare_section_growth,
    evaluate_section_patch_gate,
)
from quasi_exp.teacher.selected_section_audit import (
    SelectedSectionAuditPolicy,
    audit_selected_sections,
)
from quasi_exp.teacher.workspace_reach import CellKey, measure_weighted_boundary_change_ratio


def _candidate(node_id: int, beta: np.ndarray, candidate_id: str) -> AtlasCandidate:
    return AtlasCandidate(
        node_id=node_id,
        candidate_id=candidate_id,
        beta_rad=np.asarray(beta, dtype=float),
        residual_mm=0.0,
        min_margin_deg=5.0,
        normalized_min_margin=0.5,
        posture_cost=float(np.linalg.norm(beta)),
        condition_number=1.0,
        quality="Gold",
    )


def _grid_nodes() -> tuple[AtlasTaskNode, ...]:
    xyz = (
        (0.0, 0.0, 0.0),
        (0.001, 0.0, 0.0),
        (0.0, 0.001, 0.0),
        (0.001, 0.001, 0.0),
    )
    neighbors = ((1, 2), (0, 3), (0, 3), (1, 2))
    return tuple(
        AtlasTaskNode(index, np.asarray(point), neighbors[index])
        for index, point in enumerate(xyz)
    )


def _affine_continuation(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
    beta = source.beta_rad.copy()
    beta[:3] = target.xyz_m
    return ContinuationOutcome(
        beta_rad=beta,
        residual_mm=0.0,
        success=True,
        actual_bounds=True,
        corrector_iterations=1,
        status="affine",
        minimum_margin_deg=5.0,
    )


def test_section_first_growth_is_rooted_single_valued_and_beam_bounded() -> None:
    nodes = _grid_nodes()
    root = _candidate(0, np.zeros(6), "root")

    result = build_section_first_atlas(
        nodes,
        (root,),
        _affine_continuation,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(beam_width=2, maximum_growth_waves=8),
    )

    assert result.covered_node_ids == frozenset({0, 1, 2, 3})
    assert result.abstained_node_ids == frozenset()
    assert result.raw_cap_hit is False
    assert len(result.charts) == 1
    chart = result.charts[0]
    assert chart.root_key == root.key
    assert set(chart.selected_by_node) == {0, 1, 2, 3}
    assert all(len(values) <= 2 for values in chart.hypotheses_by_node.values())
    assert len({item.node_id for item in chart.selected_by_node.values()}) == 4


def test_section_first_frames_have_parquet_safe_typed_lineage(tmp_path) -> None:
    nodes = _grid_nodes()
    root = _candidate(0, np.zeros(6), "root")
    result = build_section_first_atlas(
        nodes,
        (root,),
        _affine_continuation,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(beam_width=2, maximum_growth_waves=8),
    )

    frames = result.frames()
    hypotheses = frames["section_hypotheses"]
    assert "parent_keys" not in hypotheses
    assert hypotheses["parent_node_ids"].map(
        lambda values: all(isinstance(value, int) for value in values)
    ).all()
    assert hypotheses["parent_candidate_ids"].map(
        lambda values: all(isinstance(value, str) for value in values)
    ).all()
    assert hypotheses.apply(
        lambda row: len(row.parent_node_ids) == len(row.parent_candidate_ids), axis=1
    ).all()
    for name, frame in frames.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        assert len(frame) == len(pd.read_parquet(path))


def test_section_first_rejects_nonstitchable_multiparent_endpoint() -> None:
    nodes = _grid_nodes()
    root = _candidate(0, np.zeros(6), "root")

    def conflicting(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
        beta = source.beta_rad.copy()
        beta[:3] = target.xyz_m
        if target.node_id == 3:
            beta[5] = math.radians(3.0 if source.node_id == 1 else -3.0)
        else:
            beta[5] = 0.0
        return ContinuationOutcome(beta, 0.0, True, True, 1, "conflicting", minimum_margin_deg=5.0)

    result = build_section_first_atlas(
        nodes,
        (root,),
        conflicting,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(beam_width=4, maximum_growth_waves=8),
    )

    assert 3 in result.abstained_node_ids
    assert 3 not in result.charts[0].selected_by_node
    assert any(event.reason == "nonstitchable_parent_disagreement" for event in result.events)


def test_selected_section_audit_emits_detailed_selected_only_evidence() -> None:
    nodes = _grid_nodes()
    root = _candidate(0, np.zeros(6), "root")
    result = build_section_first_atlas(
        nodes,
        (root,),
        _affine_continuation,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(beam_width=2, maximum_growth_waves=8),
    )

    audit = audit_selected_sections(
        result,
        _affine_continuation,
        policy=SelectedSectionAuditPolicy(),
        patch_id="patch_test",
        method="S2",
    )

    assert audit.gate_pass is True
    assert set(audit.frames) == {
        "fresh_edge_audit",
        "fresh_path_audit",
        "fresh_cycle_audit",
        "fresh_multipath_audit",
        "fresh_repeat_direction_audit",
        "cap_hit_events",
    }
    edge_frame = audit.frames["fresh_edge_audit"]
    assert len(edge_frame) == 2 * len(result.charts[0].selected_edges)
    assert edge_frame["candidate_was_selected"].all()
    assert set(edge_frame["patch_id"]) == {"patch_test"}
    assert set(edge_frame["method"]) == {"S2"}
    assert audit.metrics["cycle"]["sample_count"] >= 1


def test_section_stability_compares_selected_sections_not_candidate_count() -> None:
    nodes = _grid_nodes()
    root = _candidate(0, np.zeros(6), "root")
    low = build_section_first_atlas(
        nodes,
        (root,),
        _affine_continuation,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(beam_width=2, maximum_growth_waves=8),
    )
    high = build_section_first_atlas(
        nodes,
        (root,),
        _affine_continuation,
        root_keys=(root.key,),
        policy=RootedSectionPolicy(beam_width=8, maximum_growth_waves=8),
    )

    report = compare_section_growth(
        low,
        high,
        policy=SectionStabilityPolicy(),
    )

    assert report.gate_pass is True
    assert report.coverage_jaccard == 1.0
    assert report.selected_beta_p95_deg == 0.0
    assert report.selected_section_affected_by_cap is False


def test_patch_gate_allows_registered_unresolved_patches_but_not_one_success() -> None:
    passing = [
        {"patch_id": f"patch_{index:02d}", "patch_split": "development", "gate_pass": index < 6,
         "largest_selected_component_ratio": 0.7, "single_cell_chart_ratio": 0.0}
        for index in range(8)
    ] + [
        {"patch_id": f"patch_{index:02d}", "patch_split": "confirmation", "gate_pass": index < 11,
         "largest_selected_component_ratio": 0.7, "single_cell_chart_ratio": 0.0}
        for index in range(8, 12)
    ]
    assert evaluate_section_patch_gate(passing)["gate_pass"] is True

    for row in passing:
        row["gate_pass"] = row["patch_id"] == "patch_09"
    failed = evaluate_section_patch_gate(passing)
    assert failed["gate_pass"] is False
    assert failed["representation_diagnostic_required"] is True


def test_measure_weighted_boundary_change_is_normalized_by_workspace_measure() -> None:
    previous = frozenset(CellKey(20, index, 0, 0) for index in range(100))
    current = previous | {CellKey(20, 100, 0, 0)}

    ratio = measure_weighted_boundary_change_ratio(current, previous)

    assert 0.0 < ratio < 0.05
