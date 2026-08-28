from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry14_shell_mesh import (
    assert_holdout_macroblock_isolation,
    audit_shell_mesh,
    build_shell_target_mesh,
    choose_canonical_parent,
    classify_candidate_endpoints,
    complete_ring_status,
    connector_loss_mask,
    coverage_fill_required,
    is_low_gain,
    map_legacy_labels_to_mesh,
    mirror_nodes_and_mesh_edges,
    quotient_cylindrical_proposals,
    register_required_shell_domain,
    reverse_delete_shell_dataset,
    select_fill_targets,
)


ZERO = np.asarray([1.0, 0.0, 0.0])


def _proposal(u_mm: float, rho_mm: float, phi: float, index: int) -> dict[str, object]:
    return {
        "proposal_id": f"p{index}",
        "x_m": ZERO[0] - u_mm / 1000.0,
        "y_m": rho_mm / 1000.0 * np.cos(phi),
        "z_m": rho_mm / 1000.0 * np.sin(phi),
        **dict(zip(BETA_COLUMNS, np.full(6, 123.0), strict=True)),
    }


def _supported_cell_proposals(u_mm: float, rho_mm: float, start: int = 0) -> list[dict[str, object]]:
    # Eight bins plus endpoints; every analytic point remains within 15 mm in
    # the small test geometry.
    return [_proposal(u_mm, rho_mm, (index + 0.5) * np.pi / 16.0, start + index) for index in range(8)]


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cell_id": "ur:0:1", "u_index": 0, "rho_index": 1, "u_center_mm": 7.5, "rho_center_mm": 22.5, "required_shell_cell": True, "trajectory_ring_candidate": True, "n_phi_intervals": 4},
            {"cell_id": "ur:1:1", "u_index": 1, "rho_index": 1, "u_center_mm": 22.5, "rho_center_mm": 22.5, "required_shell_cell": True, "trajectory_ring_candidate": True, "n_phi_intervals": 4},
            {"cell_id": "ur:0:2", "u_index": 0, "rho_index": 2, "u_center_mm": 7.5, "rho_center_mm": 37.5, "required_shell_cell": True, "trajectory_ring_candidate": False, "n_phi_intervals": 6},
        ]
    )


def _labelled(vertices: pd.DataFrame) -> pd.DataFrame:
    result = vertices.copy()
    result["label_status"] = "Gold"
    return result


def test_xyz_only_cylindrical_registry_and_ring_domain() -> None:
    proposals = pd.DataFrame(_supported_cell_proposals(7.5, 22.5) + _supported_cell_proposals(22.5, 22.5, 10))
    registry = quotient_cylindrical_proposals(proposals, zero_xyz_m=ZERO)

    assert not set(BETA_COLUMNS).intersection(registry.columns)
    assert registry["proposal_beta_label_eligible"].eq(False).all()
    assert registry["angle_bin_id"].nunique() == 8
    cells, rings = register_required_shell_domain(registry, zero_xyz_m=ZERO)
    assert cells["coverage_ring_candidate"].all()
    assert cells["required_shell_cell"].all()
    assert len(rings) == 2


def test_one_cell_hole_is_allowed_but_extended_gap_is_not() -> None:
    # The hole is surrounded by four coverage candidates, while a second empty
    # adjacent cell prevents it from being interpreted as one isolated hole.
    records = []
    for index, (u, rho) in enumerate([(0, 1), (2, 1), (1, 0), (1, 2)]):
        for row in _supported_cell_proposals((u + .5) * 15, (rho + .5) * 15, index * 10): records.append(row)
    # A real proposal can occupy the hole without supplying enough angle bins;
    # it must be promoted in place, never duplicated as a second cell row.
    records.append(_proposal(22.5, 22.5, np.pi / 4, 99))
    registry = quotient_cylindrical_proposals(pd.DataFrame(records), zero_xyz_m=ZERO)
    cells, _ = register_required_shell_domain(registry, zero_xyz_m=ZERO)
    hole = cells[cells["cell_id"].eq("ur:1:1")]
    assert len(hole) == 1 and bool(hole.iloc[0]["one_cell_hole"])


def test_target_mesh_registers_all_edge_types_and_short_connector_chains() -> None:
    cells = _cells()
    proposals = quotient_cylindrical_proposals(pd.DataFrame(_supported_cell_proposals(7.5, 22.5)), zero_xyz_m=ZERO)
    vertices, edges, connectors = build_shell_target_mesh(cells, proposals, zero_xyz_m=ZERO)

    assert {"angular_edge", "radial_edge", "axial_edge"}.issubset(set(edges["mesh_edge_type"]))
    assert {"zero_y_seam_connector_edge", "zero_z_seam_connector_edge"}.issubset(
        set(edges["mesh_edge_type"])
    )
    assert edges["segment_chain_ids"].notna().all()
    assert not connectors.empty
    assert connectors["segment_length_mm"].le(5.0 + 1e-12).all()
    for row in edges.itertuples():
        assert set(str(row.segment_chain_ids).split("|")).issubset(set(connectors["solver_segment_id"]))
    zero_links = connectors[
        connectors["logical_edge_id"].isin(
            edges.loc[edges.mesh_edge_type.str.startswith("zero_"), "logical_edge_id"]
        )
    ]
    for _, group in zero_links.groupby("logical_edge_id"):
        xyz = group[["left_y_m", "left_z_m", "right_y_m", "right_z_m"]].to_numpy(float)
        assert np.all(np.abs(xyz[:, [0, 2]]) <= 1e-12) or np.all(np.abs(xyz[:, [1, 3]]) <= 1e-12)


def test_legacy_mapping_never_marks_coverage_and_fill_ignores_row_count() -> None:
    cells = _cells().iloc[:1]
    proposal = quotient_cylindrical_proposals(pd.DataFrame(_supported_cell_proposals(7.5, 22.5)), zero_xyz_m=ZERO)
    vertices, _, _ = build_shell_target_mesh(cells, proposal, zero_xyz_m=ZERO)
    legacy = pd.DataFrame([{ "physical_point_id": "legacy", "zero_reachable": True, **dict(zip(XYZ_COLUMNS, vertices.iloc[0][list(XYZ_COLUMNS)], strict=True)) }])
    mapping, _, seeded = map_legacy_labels_to_mesh(legacy, vertices)
    assert len(mapping) == 1
    assert seeded["label_status"].eq("seeded").sum() == 2  # mapped label plus exact zero
    assert coverage_fill_required({"shell_cell_coverage": .1, "ring_completion_fraction": 1.0, "proposal_service_p95_mm": 1.0, "maximum_angular_gap_bins": 0})


def test_fill_priority_and_low_gain_plateau_are_coverage_based() -> None:
    vertices = pd.DataFrame(
        [
            {"mesh_vertex_id": "b", "label_status": "unresolved", "incomplete_trajectory_ring": False, "angular_gap_bins": 4, "unserved_required_shell_cell": True, "proposal_service_distance_mm": 30.0, "radial_or_axial_mesh_hole": False},
            {"mesh_vertex_id": "a", "label_status": "unresolved", "incomplete_trajectory_ring": True, "angular_gap_bins": 0, "unserved_required_shell_cell": False, "proposal_service_distance_mm": 1.0, "radial_or_axial_mesh_hole": False},
        ]
    )
    assert select_fill_targets(vertices, maximum_targets=2)["mesh_vertex_id"].tolist() == ["a", "b"]
    assert is_low_gain({"shell_cell_coverage": .5, "ring_completion_fraction": .5, "complete_ring_count": 2}, {"shell_cell_coverage": .504, "ring_completion_fraction": .519, "complete_ring_count": 2})
    assert not is_low_gain({"shell_cell_coverage": .5, "ring_completion_fraction": .5, "complete_ring_count": 2}, {"shell_cell_coverage": .504, "ring_completion_fraction": .519, "complete_ring_count": 3})


def test_complete_ring_served_cell_and_mesh_cycle_rank() -> None:
    cells = _cells().iloc[:1]
    proposal = quotient_cylindrical_proposals(pd.DataFrame(_supported_cell_proposals(7.5, 22.5)), zero_xyz_m=ZERO)
    vertices, edges, _ = build_shell_target_mesh(cells, proposal, zero_xyz_m=ZERO)
    vertices = _labelled(vertices)
    edges["mesh_edge_verified"] = True
    # A fundamental arc is not a complete ring.  Add the seam-closing logical
    # edge which the symmetry expansion stage normally supplies.
    assert not complete_ring_status(vertices, edges, vertices.iloc[0]["ring_id"])
    ring_vertices = vertices[vertices["cell_id"].eq("ur:0:1")].reset_index(drop=True)
    close = pd.DataFrame([{
        "logical_edge_id": "seam-close", "left_mesh_vertex_id": ring_vertices.iloc[0]["mesh_vertex_id"],
        "right_mesh_vertex_id": ring_vertices.iloc[-1]["mesh_vertex_id"], "mesh_edge_type": "angular_edge",
        "mesh_edge_verified": True,
    }])
    edges = pd.concat([edges, close], ignore_index=True, sort=False)
    metrics = audit_shell_mesh(vertices, edges, cells)
    assert metrics["shell_cell_coverage"] == 1.0
    assert metrics["ring_completion_fraction"] == 1.0
    assert metrics["complete_ring_count"] == 1.0
    assert metrics["mesh_cycle_rank"] >= 1.0


def test_branch_conflict_is_not_aggregated_and_parent_order_is_stable() -> None:
    rows = []
    for parent, depth, condition, transition, beta0 in [("deep", 2, .1, .1, .01), ("shallow", 1, .9, .9, .011)]:
        rows.append({"parent_id": parent, "success": True, "lineage_depth": depth, "source_condition": condition, "weighted_transition_deg": transition, "fk_residual_mm": 1.0, **dict(zip(BETA_COLUMNS, [beta0] * 6, strict=True))})
    endpoints = pd.DataFrame(rows)
    assert classify_candidate_endpoints(endpoints) == "Gold"
    assert choose_canonical_parent(endpoints)["parent_id"] == "shallow"
    conflict = endpoints.copy(); conflict.loc[1, BETA_COLUMNS[0]] = .3
    assert classify_candidate_endpoints(conflict) == "branch_conflict"
    with pytest.raises(ValueError, match="branch-conflict"):
        choose_canonical_parent(conflict)


def test_reverse_delete_protects_ring_and_ancestor_nodes() -> None:
    nodes = pd.DataFrame(
        [
            {"mesh_vertex_id": "zero", "mandatory": True, "is_ring_member": False, "is_lineage_ancestor": True, "zero_radius_mm": 0.0},
            {"mesh_vertex_id": "ring", "mandatory": False, "is_ring_member": True, "is_lineage_ancestor": False, "zero_radius_mm": 10.0},
            {"mesh_vertex_id": "leaf", "mandatory": False, "is_ring_member": False, "is_lineage_ancestor": False, "zero_radius_mm": 20.0},
        ]
    )
    edges = pd.DataFrame([{"left_mesh_vertex_id": "zero", "right_mesh_vertex_id": "ring"}, {"left_mesh_vertex_id": "ring", "right_mesh_vertex_id": "leaf"}])
    kept, _, removed = reverse_delete_shell_dataset(nodes, edges, maximum_rows=2)
    assert removed == ["leaf"]
    assert set(kept["mesh_vertex_id"]) == {"zero", "ring"}


def test_symmetry_mirrors_whole_edges_and_connector_loss_holdout_isolation() -> None:
    nodes = pd.DataFrame(
        [
            {"mesh_vertex_id": "zero", "label_role": "supervision", **dict(zip(XYZ_COLUMNS, ZERO, strict=True)), **dict(zip(BETA_COLUMNS, [0.0] * 6, strict=True))},
            {"mesh_vertex_id": "node", "label_role": "connector_only", **dict(zip(XYZ_COLUMNS, [0.99, .01, .02], strict=True)), **dict(zip(BETA_COLUMNS, [.1] * 6, strict=True))},
        ]
    )
    edges = pd.DataFrame([{"logical_edge_id": "axis", "left_mesh_vertex_id": "zero", "right_mesh_vertex_id": "node", "mesh_edge_type": "axial_edge"}])
    full_nodes, full_edges = mirror_nodes_and_mesh_edges(nodes, edges, zero_vertex_id="zero")
    assert len(full_nodes) == 5 and len(full_edges) == 4
    assert full_edges["mesh_edge_type"].eq("axial_edge").all()
    assert connector_loss_mask(nodes).tolist() == [True, False]
    safe = pd.DataFrame({"quotient_macroblock_id": ["m", "m", "n"], "split_role": ["trajectory_holdout", "trajectory_holdout", "train"]})
    assert_holdout_macroblock_isolation(safe)
    leaky = safe.copy(); leaky.loc[1, "split_role"] = "train"
    with pytest.raises(ValueError, match="leakage"):
        assert_holdout_macroblock_isolation(leaky)


def test_mirrored_quadrant_arc_merges_seams_into_full_cycle() -> None:
    ring_id = "ring"
    rows = []
    for index, phi in enumerate(np.linspace(0.0, np.pi / 2.0, 5)):
        beta = np.asarray([.1, 0.0, .1, 0.0, .1, 0.0]) if index == 0 else (
            np.asarray([0.0, .1, 0.0, .1, 0.0, .1]) if index == 4 else np.asarray([.1] * 6)
        )
        rows.append(
            {
                "mesh_vertex_id": f"v{index}", "ring_id": ring_id,
                "label_status": "Gold", "x_m": .9,
                "y_m": .1 * np.cos(phi), "z_m": .1 * np.sin(phi),
                **dict(zip(BETA_COLUMNS, beta, strict=True)),
            }
        )
    nodes = pd.DataFrame(rows)
    edges = pd.DataFrame(
        [
            {
                "logical_edge_id": f"e{index}", "left_mesh_vertex_id": f"v{index}",
                "right_mesh_vertex_id": f"v{index + 1}", "mesh_edge_type": "angular_edge",
                "mesh_edge_verified": True,
            }
            for index in range(4)
        ]
    )
    full_nodes, full_edges = mirror_nodes_and_mesh_edges(nodes, edges, zero_vertex_id="zero")
    assert len(full_nodes) == 16
    assert len(full_edges) == 16
    assert complete_ring_status(full_nodes, full_edges, ring_id)
