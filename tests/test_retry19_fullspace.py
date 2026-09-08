from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import quasi_exp.teacher.retry19_fullspace as retry19_fullspace
from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS
from quasi_exp.teacher.retry19_fullspace import (
    FullspaceCoveragePolicy,
    Retry19TeacherPolicy,
    axial_slice_coverage,
    classify_old_anchors,
    data_gate,
    geometric_zero_attachment,
    deterministic_path_query_registry,
    largest_unserved_component_fraction,
    mixed_resolution_coverage_metrics,
    mixed_resolution_26_edges,
    mixed_resolution_face_edges,
    path_query_metrics,
    select_fullspace_graph_teacher,
    teacher_zero_attachment,
)


def _cell(cell_id: str, lower: tuple[float, float, float], size: float, *, served: bool = False) -> dict:
    upper = np.asarray(lower) + size
    return {
        "cell_id": cell_id,
        "x_min_mm": lower[0],
        "y_min_mm": lower[1],
        "z_min_mm": lower[2],
        "x_max_mm": upper[0],
        "y_max_mm": upper[1],
        "z_max_mm": upper[2],
        "cell_size_mm": size,
        "volume_mm3": size**3,
        "required": True,
        "served": served,
        "service_distance_mm": 0.0 if served else math.inf,
    }


def test_mixed_resolution_cells_remain_real_boxes_and_use_face_adjacency() -> None:
    cells = pd.DataFrame(
        [
            _cell("coarse", (0, 0, 0), 10),
            _cell("fine_face", (10, 0, 0), 5),
            _cell("fine_edge", (10, 10, 0), 5),
            _cell("fine_corner", (10, 10, 10), 5),
        ]
    )
    edges = mixed_resolution_face_edges(cells)
    pairs = {frozenset((row.left_cell_id, row.right_cell_id)) for row in edges.itertuples()}
    assert frozenset(("coarse", "fine_face")) in pairs
    assert frozenset(("coarse", "fine_edge")) not in pairs
    assert frozenset(("coarse", "fine_corner")) not in pairs
    assert len(cells) == 4
    assert cells.loc[cells["cell_id"].eq("coarse"), "volume_mm3"].item() == 1000

    neighbour_edges = mixed_resolution_26_edges(cells)
    neighbour_pairs = {
        frozenset((row.left_cell_id, row.right_cell_id))
        for row in neighbour_edges.itertuples()
    }
    assert frozenset(("coarse", "fine_face")) in neighbour_pairs
    assert frozenset(("coarse", "fine_edge")) in neighbour_pairs
    assert frozenset(("coarse", "fine_corner")) in neighbour_pairs


def test_q31_data_gate_keeps_non_q31_coverage_checks_diagnostic_only() -> None:
    coverage = {
        "volume_coverage": 0.91,
        "service_distance_volume_weighted_p95_mm": 10.000000000000009,
        "fine_5mm_coverage": 0.01,
        "coarse_10mm_coverage": 0.02,
    }
    gate = data_gate(
        coverage,
        maximum_hole_fraction=0.99,
        minimum_slice_coverage=0.01,
        path_metrics={"success_rate": 0.0, "stretch_p95": math.inf},
        geometric_zero_connected=True,
        teacher_zero_connected=True,
        nearest_zero_nonzero_mm=4.9,
        policy=FullspaceCoveragePolicy(),
    )

    assert gate["status"] == "green"
    assert all(gate["checks"].values())
    assert gate["diagnostics"]["fine_5mm_coverage"] == 0.01
    assert gate["diagnostics"]["path_success_rate"] == 0.0

    coverage["service_distance_volume_weighted_p95_mm"] = 10.001
    gate = data_gate(
        coverage,
        maximum_hole_fraction=0.0,
        minimum_slice_coverage=1.0,
        path_metrics={"success_rate": 1.0, "stretch_p95": 1.0},
        geometric_zero_connected=True,
        teacher_zero_connected=True,
        nearest_zero_nonzero_mm=4.9,
        policy=FullspaceCoveragePolicy(),
    )
    assert gate["status"] == "red"
    assert gate["checks"]["service_distance_p95"] is False


def test_cell_registry_deduplicates_integer_keys_before_materializing_pandas_rows(monkeypatch) -> None:
    materialized_row_counts: list[int] = []
    dataframe = retry19_fullspace.pd.DataFrame

    def tracked_dataframe(*args, **kwargs):
        frame = dataframe(*args, **kwargs)
        materialized_row_counts.append(len(frame))
        return frame

    monkeypatch.setattr(retry19_fullspace.pd, "DataFrame", tracked_dataframe)
    repeated = np.repeat(np.asarray([[1.2, 0.0, 0.0]]), 20_000, axis=0)

    cells = retry19_fullspace.build_mixed_resolution_cell_registry(
        repeated,
        repeated,
        zero_xyz_m=(1.215498, 0.0, 0.0),
    )

    assert len(cells) == 1
    assert materialized_row_counts == [1, 1]


def test_path_query_sampling_is_bounded_instead_of_enumerating_all_pairs(monkeypatch) -> None:
    probes = pd.DataFrame(
        {
            "cell_id": [f"cell_{index:06d}" for index in range(100_000)],
            "required": True,
        }
    )
    sha256 = retry19_fullspace.hashlib.sha256
    calls = 0

    def bounded_sha256(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 3_100:
            raise AssertionError("path-query sampler enumerated the pair universe")
        return sha256(*args, **kwargs)

    monkeypatch.setattr(retry19_fullspace.hashlib, "sha256", bounded_sha256)
    first = deterministic_path_query_registry(probes, pair_count=1_000, seed=20260954)
    first_calls = calls
    calls = 0
    second = deterministic_path_query_registry(probes, pair_count=1_000, seed=20260954)

    assert len(first) == 1_000
    assert first_calls <= 3_100
    assert not first[["left_cell_id", "right_cell_id"]].duplicated().any()
    pd.testing.assert_frame_equal(first, second)


def test_path_query_metrics_uses_sparse_query_sources_instead_of_all_pairs(monkeypatch) -> None:
    target_count = 10_000
    target_x = np.linspace(0.0025, 0.0175, target_count)
    targets = pd.DataFrame(
        {
            "target_id": [f"target_{index:06d}" for index in range(target_count)],
            "x_m": target_x,
            "y_m": np.zeros(target_count),
            "z_m": np.zeros(target_count),
        }
    )
    edges = pd.DataFrame(
        {
            "left_target_id": targets["target_id"].iloc[:-1].to_numpy(),
            "right_target_id": targets["target_id"].iloc[1:].to_numpy(),
            "distance_mm": np.diff(target_x) * 1000.0,
        }
    )
    cells = pd.DataFrame(
        [
            _cell("left", (0.0, -2.5, -2.5), 5.0),
            _cell("right", (15.0, -2.5, -2.5), 5.0),
        ]
    )
    queries = pd.DataFrame(
        [
            {
                "query_id": "query",
                "left_cell_id": "left",
                "right_cell_id": "right",
                "selection_digest": "digest",
            }
        ]
    )
    numpy_full = retry19_fullspace.np.full
    scipy_dijkstra = retry19_fullspace.dijkstra
    observed: dict[str, object] = {}

    def reject_dense_target_square(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape == (target_count, target_count):
            raise AssertionError("path query metrics allocated a dense target square")
        return numpy_full(shape, *args, **kwargs)

    def record_sparse_sources(graph, *args, **kwargs):
        observed["shape"] = graph.shape
        observed["nnz"] = graph.nnz
        observed["indices"] = kwargs.get("indices")
        return scipy_dijkstra(graph, *args, **kwargs)

    monkeypatch.setattr(retry19_fullspace.np, "full", reject_dense_target_square)
    monkeypatch.setattr(retry19_fullspace, "dijkstra", record_sparse_sources)
    metrics, audit = path_query_metrics(cells, targets, edges, queries)

    assert observed == {"shape": (target_count, target_count), "nnz": 2 * (target_count - 1), "indices": [0]}
    assert metrics["success_rate"] == 1.0
    assert metrics["stretch_p95"] == pytest.approx(1.0)
    assert audit["success"].tolist() == [True]


def test_volume_coverage_cannot_be_replaced_by_cell_count() -> None:
    cells = pd.DataFrame(
        [
            _cell("coarse", (0, 0, 0), 10, served=False),
            *[_cell(f"fine_{index}", (10, index * 5, 0), 5, served=True) for index in range(8)],
        ]
    )
    metrics = mixed_resolution_coverage_metrics(cells)
    assert metrics["cell_count_coverage"] == pytest.approx(8 / 9)
    assert metrics["volume_coverage"] == pytest.approx(0.5)
    assert metrics["fine_5mm_coverage"] == 1.0
    assert metrics["coarse_10mm_coverage"] == 0.0


def test_zero_attachment_never_jumps_to_arbitrary_nearest_cell() -> None:
    cells = pd.DataFrame(
        [
            _cell("zero", (-2.5, -2.5, -2.5), 5),
            _cell("near", (5.0, -2.5, -2.5), 5),
            _cell("far", (20, -2.5, -2.5), 5),
        ]
    )
    face = mixed_resolution_face_edges(cells)
    edges, component = geometric_zero_attachment(
        cells,
        face,
        zero_cell_id="zero",
        zero_point_mm=(0, 0, 0),
        registration_threshold_mm=5.0,
    )
    assert set(component) == {"zero", "near"}
    assert "far" not in component
    assert edges["adjacency"].eq("root_box_distance").sum() == 1


def test_teacher_zero_connection_is_separate_and_beta_gated() -> None:
    labels = pd.DataFrame(
        [
            {"target_id": "zero", **dict.fromkeys(BETA_COLUMNS, 0.0), "fk_residual_mm": 0.0},
            {"target_id": "good", **dict.fromkeys(BETA_COLUMNS, np.deg2rad(1.0)), "fk_residual_mm": 1.0},
            {"target_id": "jump", **dict.fromkeys(BETA_COLUMNS, np.deg2rad(12.0)), "fk_residual_mm": 1.0},
        ]
    )
    edges = pd.DataFrame(
        [
            {"left_target_id": "zero", "right_target_id": "good"},
            {"left_target_id": "good", "right_target_id": "jump"},
        ]
    )
    connected, component, audit = teacher_zero_attachment(labels, edges, zero_target_id="zero")
    assert connected
    assert component == {"zero", "good"}
    assert audit["teacher_edge_legal"].tolist() == [True, False]


def test_hard_anchor_old_candidate_cannot_be_replaced_during_legality_clustering() -> None:
    targets = pd.DataFrame(
        [{"target_id": "hard", "x_m": 1.0, "y_m": 0.0, "z_m": 0.0}]
    )
    common = {
        "target_id": "hard",
        "solver_success": True,
        "bounds_pass": True,
        "fk_residual_mm": 0.0,
        "min_margin_deg": 1.0,
        **dict.fromkeys(BETA_COLUMNS, 0.0),
    }
    candidates = pd.DataFrame(
        [
            {**common, "candidate_id": "a_new", "old_label_candidate": False},
            {**common, "candidate_id": "z_old", "old_label_candidate": True},
        ]
    )
    anchors = pd.DataFrame([{"target_id": "hard", "anchor_class": "hard"}])

    labels, _audit = select_fullspace_graph_teacher(
        candidates,
        targets,
        pd.DataFrame(columns=["left_target_id", "right_target_id"]),
        anchors,
        policy=Retry19TeacherPolicy(),
    )

    assert labels["candidate_id"].tolist() == ["z_old"]


def test_hole_component_and_axial_slice_use_real_volume() -> None:
    cells = pd.DataFrame(
        [
            _cell("a", (0, 0, 0), 10, served=True),
            _cell("b", (10, 0, 0), 10, served=False),
            _cell("c", (20, 0, 0), 10, served=False),
        ]
    )
    edges = mixed_resolution_face_edges(cells)
    assert largest_unserved_component_fraction(cells, edges) == pytest.approx(2 / 3)
    slices = axial_slice_coverage(cells, zero_x_mm=30, bin_edges_u_mm=[0, 10, 20, 30])
    assert slices["volume_coverage"].tolist() == [0.0, 0.0, 1.0]


def test_old_labels_are_not_all_immutable_hard_anchors() -> None:
    rows = []
    for index, target in enumerate(("zero", "stable_a", "stable_b", "seam")):
        rows.append(
            {
                "target_id": target,
                "x_m": 1.0 + 0.001 * index,
                "y_m": 0.01,
                "z_m": 0.01,
                "fk_residual_mm": 1.0,
                "repeat_pass": target.startswith("stable"),
                "multiparent_consistency_pass": target.startswith("stable"),
                "seam_or_conflict_risk": target == "seam",
            }
        )
    registry = classify_old_anchors(pd.DataFrame(rows), exact_zero_target_id="zero")
    classes = registry.set_index("target_id")["anchor_class"].to_dict()
    assert classes["zero"] == "hard"
    assert sum(registry["anchor_class"].eq("hard")) == 2  # zero plus one per shared macroblock
    assert classes["seam"] == "diagnostic"


def _candidate(target: str, candidate: str, degrees: float, *, old: bool = False) -> dict:
    return {
        "target_id": target,
        "candidate_id": candidate,
        **dict.fromkeys(BETA_COLUMNS, np.deg2rad(degrees)),
        "solver_success": True,
        "bounds_pass": True,
        "fk_residual_mm": 0.0,
        "min_margin_deg": 1.0,
        "old_label_candidate": old,
        "unary_cost": degrees * degrees,
    }


def test_graph_teacher_uses_mst_dp_keeps_hard_and_can_change_soft() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("a", "a_old", 0, old=True),
            _candidate("a", "a_new", 10),
            _candidate("b", "b_old", 10, old=True),
            _candidate("b", "b_new", 0),
            _candidate("c", "c_new", 0),
        ]
    )
    targets = pd.DataFrame(
        [
            {"target_id": "a", "x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            {"target_id": "b", "x_m": 0.005, "y_m": 0.0, "z_m": 0.0},
            {"target_id": "c", "x_m": 0.010, "y_m": 0.0, "z_m": 0.0},
        ]
    )
    edges = pd.DataFrame(
        [
            {"left_target_id": "a", "right_target_id": "b", "distance_mm": 5.0},
            {"left_target_id": "b", "right_target_id": "c", "distance_mm": 5.0},
            {"left_target_id": "a", "right_target_id": "c", "distance_mm": 10.0},
        ]
    )
    anchors = pd.DataFrame(
        [
            {"target_id": "a", "anchor_class": "hard"},
            {"target_id": "b", "anchor_class": "soft"},
        ]
    )
    selected, audit = select_fullspace_graph_teacher(
        candidates,
        targets,
        edges,
        anchors,
        policy=Retry19TeacherPolicy(pairwise_lambda=8.0),
    )
    chosen = selected.set_index("target_id")["candidate_id"].to_dict()
    assert chosen["a"] == "a_old"
    assert chosen["b"] == "b_new"
    assert audit["mst_dp_initialized"]
    assert audit["selected_icm_order"] in {"target_id", "degree", "root_bfs"}
    data_gate,
