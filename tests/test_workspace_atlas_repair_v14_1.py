from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    CanonicalAtlas,
    CanonicalChart,
    ContinuationOutcome,
    DirectedContinuationEdge,
    assemble_product_graph,
)
from quasi_exp.teacher.workspace_atlas_repair import (
    CanonicalSelectionObservation,
    CrossCellRepairPolicy,
    RepairAblation,
    build_dynamic_product_graph,
    build_shared_face_task_edges,
    evaluate_canonical_selection_stability,
    make_segmented_continuation,
    piecewise_primary_partition,
    run_cross_cell_repair_ablation,
)


def _task_frame() -> pd.DataFrame:
    rows = []
    task_node_id = 0
    for cell_ix in (0, 1):
        for probe_index, (dy, dz) in enumerate(
            ((0.0, 0.0), (0.001, 0.0), (0.0, 0.001), (0.001, 0.001), (0.0005, 0.0005))
        ):
            rows.append(
                {
                    "task_node_id": task_node_id,
                    "x_m": cell_ix * 0.01 + (0.009 if cell_ix == 0 else 0.001),
                    "y_m": dy,
                    "z_m": dz,
                    "cell_level_mm": 10,
                    "cell_ix": cell_ix,
                    "cell_iy": 0,
                    "cell_iz": 0,
                    "is_representative": probe_index == 0,
                    "core_safe": True,
                    "condition_number": float(task_node_id + 1),
                }
            )
            task_node_id += 1
    return pd.DataFrame.from_records(rows)


def _candidate(node_id: int, beta: np.ndarray, name: str = "seed") -> AtlasCandidate:
    return AtlasCandidate(
        node_id=node_id,
        candidate_id=name,
        beta_rad=np.asarray(beta, dtype=float),
        residual_mm=0.0,
        min_margin_deg=10.0,
        normalized_min_margin=0.5,
        quality="Gold",
    )


def test_shared_face_edges_add_four_matches_and_medoid_without_removing_legacy() -> None:
    tasks = _task_frame()
    legacy = []
    for base in (0, 5):
        for left in range(base, base + 5):
            for right in range(left + 1, base + 5):
                legacy.append({"left_node_id": left, "right_node_id": right})
    legacy.append({"left_node_id": 0, "right_node_id": 5})
    legacy_frame = pd.DataFrame.from_records(legacy)

    repaired = build_shared_face_task_edges(tasks, legacy_frame)

    assert {tuple(row) for row in repaired[["left_node_id", "right_node_id"]].to_numpy()} >= {(0, 5)}
    cross = repaired[
        repaired["left_node_id"].lt(5) & repaired["right_node_id"].ge(5)
    ]
    assert len(cross) >= 4
    assert set(cross["adjacency"]) <= {
        "legacy_representative",
        "shared_face",
        "shared_face_medoid",
    }


def test_segmented_continuation_subdivides_cross_cell_only() -> None:
    nodes = {
        0: AtlasTaskNode(0, np.zeros(3), (1, 2)),
        1: AtlasTaskNode(1, np.asarray([0.012, 0.0, 0.0]), (0,)),
        2: AtlasTaskNode(2, np.asarray([0.003, 0.0, 0.0]), (0,)),
    }
    cells = {0: (10, 0, 0, 0), 1: (10, 1, 0, 0), 2: (10, 0, 0, 0)}
    calls: list[np.ndarray] = []

    def continuation(source, target):
        calls.append(target.xyz_m.copy())
        beta = source.beta_rad.copy()
        beta[:3] = target.xyz_m
        return ContinuationOutcome(beta, 0.0, True, True, status="ok")

    segmented = make_segmented_continuation(continuation, nodes, cells, step_max_mm=5.0)
    source = _candidate(0, np.zeros(6))

    cross = segmented(source, nodes[1])
    assert cross.success is True
    assert cross.waypoint_count == 3
    assert len(calls) == 3
    calls.clear()
    within = segmented(source, nodes[2])
    assert within.waypoint_count == 1
    assert len(calls) == 1


def test_dynamic_insertion_accepts_reverse_verified_endpoint_absent_from_static_bank() -> None:
    nodes = (
        AtlasTaskNode(0, np.zeros(3), (1,)),
        AtlasTaskNode(1, np.asarray([0.01, 0.0, 0.0]), (0,)),
    )

    def continuation(source, target):
        beta = source.beta_rad.copy()
        beta[:3] = target.xyz_m
        return ContinuationOutcome(beta, 0.0, True, True, status="affine")

    graph, records, exhausted = build_dynamic_product_graph(
        nodes,
        (_candidate(0, np.zeros(6)),),
        continuation,
        repair_policy=CrossCellRepairPolicy(maximum_waves=1),
    )

    assert exhausted is False
    assert len(graph.candidates) == 2
    assert len(graph.robust_edges) == 1
    accepted = [row for row in records if row.status == "bidirectional_robust"]
    assert len(accepted) == 1
    assert accepted[0].inserted is True
    assert accepted[0].propagated_next_wave is True


def test_dynamic_insertion_matches_an_existing_target_candidate() -> None:
    """Exercise the real E3 path where propagation lands on a static candidate."""

    nodes = (
        AtlasTaskNode(0, np.zeros(3), (1,)),
        AtlasTaskNode(1, np.asarray([0.01, 0.0, 0.0]), (0,)),
    )
    source = _candidate(0, np.zeros(6), "source")
    target_beta = np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = _candidate(1, target_beta, "existing")

    def continuation(candidate, target_node):
        beta = candidate.beta_rad.copy()
        beta[:3] = target_node.xyz_m
        return ContinuationOutcome(beta, 0.0, True, True, status="affine")

    graph, records, exhausted = build_dynamic_product_graph(
        nodes,
        (source, target),
        continuation,
        repair_policy=CrossCellRepairPolicy(maximum_waves=1),
    )

    assert exhausted is False
    assert len(graph.candidates) == 2
    assert len(graph.robust_edges) == 1
    assert any(row.status == "bidirectional_robust" for row in records)


def test_dynamic_insertion_reuses_identical_directed_continuation_solve() -> None:
    """Logical attempts stay auditable while duplicate deterministic solves are cached."""

    nodes = (
        AtlasTaskNode(0, np.zeros(3), (1,)),
        AtlasTaskNode(1, np.asarray([0.01, 0.0, 0.0]), (0,)),
    )
    source = _candidate(0, np.zeros(6), "source")
    target = _candidate(1, np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), "target")
    calls: list[tuple[tuple[int, str], int]] = []

    def continuation(candidate, target_node):
        calls.append((candidate.key, target_node.node_id))
        beta = candidate.beta_rad.copy()
        beta[:3] = target_node.xyz_m
        return ContinuationOutcome(beta, 0.0, True, True, status="affine")

    graph, _records, exhausted = build_dynamic_product_graph(
        nodes,
        (source, target),
        continuation,
        repair_policy=CrossCellRepairPolicy(maximum_waves=1),
    )

    assert exhausted is False
    assert graph.continuation_attempt_count == 4
    assert calls == [
        ((0, "source"), 1),
        ((1, "target"), 0),
    ]


def _edge(source: AtlasCandidate, target: AtlasCandidate) -> DirectedContinuationEdge:
    return DirectedContinuationEdge(
        source.key,
        target.key,
        target.beta_rad,
        0.0,
        0.0,
        1,
        "unit",
    )


def test_piecewise_partition_covers_one_task_component_with_two_stitchable_regions() -> None:
    nodes = tuple(
        AtlasTaskNode(
            node_id=index,
            xyz_m=np.asarray([index * 0.001, 0.0, 0.0]),
            neighbor_node_ids=tuple(
                neighbor for neighbor in (index - 1, index + 1) if 0 <= neighbor < 4
            ),
        )
        for index in range(4)
    )
    candidates = tuple(
        _candidate(index, np.asarray([index * 0.001, 0.0, 0.0, 0.0, 0.0, 0.0]), f"c{index}")
        for index in range(4)
    )
    directed = []
    for left, right in ((0, 1), (1, 2), (2, 3)):
        directed.extend((_edge(candidates[left], candidates[right]), _edge(candidates[right], candidates[left])))
    graph = assemble_product_graph(
        nodes,
        candidates,
        directed,
        continuation_attempt_count=6,
        rejected_continuation_count=0,
    )
    chart0 = CanonicalChart(0, candidates[0].key, ((0, "c0"), (1, "c1")), {}, ())
    chart1 = CanonicalChart(1, candidates[2].key, ((2, "c2"), (3, "c3")), {}, ())
    atlas = CanonicalAtlas(graph, (chart0, chart1), (0, 2), (), (), AtlasPolicy())

    result = piecewise_primary_partition(
        nodes,
        atlas,
        valid_chart_ids=(0, 1),
        section_id_by_chart={0: "chart_000", 1: "chart_001"},
        overlap_kind_by_pair={("chart_000", "chart_001"): "stitchable"},
    )

    assert result.assignment_by_node == {
        0: "chart_000",
        1: "chart_000",
        2: "chart_001",
        3: "chart_001",
    }
    assert result.abstained_node_ids == frozenset()
    assert result.required_transitions == (("chart_000", "chart_001"),)


def test_selection_stability_gate_uses_selected_section_not_raw_family_count() -> None:
    observations = []
    for node_id in range(20):
        for budget in (4, 8, 16, 32):
            beta = np.zeros(6)
            if budget == 32:
                beta[5] = math.radians(0.1)
            observations.append(
                CanonicalSelectionObservation(
                    node_id=node_id,
                    budget=budget,
                    selected_beta_rad=beta,
                    selected_component_id="canonical",
                    outgoing_neighbor_ids=frozenset({node_id + 1}),
                )
            )

    report = evaluate_canonical_selection_stability(observations)

    assert report.audited_node_count == 20
    assert report.gate_pass is True
    assert report.beta_gap_max_deg < 0.5


def test_e0_is_preserved_as_static_candidate_matching_baseline() -> None:
    tasks = _task_frame().iloc[[0, 5]].reset_index(drop=True)
    legacy = pd.DataFrame({"left_node_id": [0], "right_node_id": [5]})
    beta0 = np.zeros(6)
    beta5 = np.zeros(6)
    beta5[3] = math.radians(4.0)

    def continuation(source, target):
        beta = source.beta_rad.copy()
        beta[:3] = target.xyz_m
        return ContinuationOutcome(beta, 0.0, True, True, status="affine")

    result = run_cross_cell_repair_ablation(
        RepairAblation.E0_LEGACY,
        tasks,
        legacy,
        (_candidate(0, beta0, "left"), _candidate(5, beta5, "right")),
        continuation,
    )

    assert len(result.graph.robust_edges) == 0
    assert result.propagation_records == ()
    assert result.metrics.cross_cell_neighbor_pair_connection_rate == 0.0
