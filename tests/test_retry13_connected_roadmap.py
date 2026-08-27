from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry13_connected_roadmap import (
    bridge_candidates,
    component_registry,
    existing_parent_edges,
    expand_connected_orbits,
    minimum_spanning_component_bridges,
    orient_certified_tree,
    zero_compatible_component,
)


def _row(point_id: str, parent_id: str | None, xyz: tuple[float, float, float], beta: tuple[float, ...]):
    return {
        "physical_point_id": point_id,
        "fundamental_representative_id": f"rep:{point_id}",
        "source_physical_point_id": parent_id,
        "reverse_gap_deg": 0.1,
        "zero_radius_mm": np.linalg.norm(np.asarray(xyz) - np.asarray([1.0, 0.0, 0.0])) * 1000.0,
        "label_quality": "Gold",
        **dict(zip(XYZ_COLUMNS, xyz, strict=True)),
        **dict(zip(BETA_COLUMNS, beta, strict=True)),
    }


def test_zero_component_and_component_bridge_preserve_parent_forests() -> None:
    frame = pd.DataFrame(
        [
            _row("zero", None, (1.0, 0.0, 0.0), (0.0,) * 6),
            _row("a", "zero", (1.0, 0.004, 0.0), (0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
            _row("b", None, (1.0, 0.008, 0.0), (0.002, 0.0, 0.0, 0.0, 0.0, 0.0)),
            _row("conflict", None, (1.0, 0.009, 0.0), (0.5, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ]
    )
    keep, _, report = zero_compatible_component(
        frame,
        zero_position=0,
        maximum_edge_mm=5.0,
        weighted_max_deg=3.0,
        raw_max_deg=7.0,
    )

    assert keep.tolist() == [True, True, True, False]
    assert report["component_count"] == 2

    active = frame[keep].reset_index(drop=True)
    inherited = existing_parent_edges(active)
    candidates, labels, sizes = bridge_candidates(
        active,
        inherited,
        maximum_edge_mm=5.0,
        weighted_max_deg=3.0,
        raw_max_deg=7.0,
    )
    assert sorted(sizes.tolist()) == [1, 2]
    certified = candidates.head(1).copy()
    certified["source_physical_point_id"] = "a"
    certified["target_physical_point_id"] = "b"
    certified["fundamental_edge_id"] = "bridge"
    selected = minimum_spanning_component_bridges(certified, component_ids=sorted(set(labels)))
    assert len(selected) == 1

    combined = pd.concat(
        [
            inherited,
            selected.reindex(columns=sorted(set(inherited.columns).union(selected.columns))),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.loc[combined["fundamental_edge_id"].eq("bridge"), "left_position"] = 1
    combined.loc[combined["fundamental_edge_id"].eq("bridge"), "right_position"] = 2
    nodes, edges = orient_certified_tree(active, combined, zero_physical_point_id="zero")
    assert nodes["retry13_lineage_depth"].tolist() == [0, 1, 2]
    assert edges["tree_orientation"].ne("unused").all()


class _SymmetricEnvironment:
    bounds = np.asarray([[-1.0, 1.0]] * 6)

    @staticmethod
    def fk(beta):
        value = np.asarray(beta, dtype=float).reshape(-1, 6)
        x = 1.0 - 0.01 * np.sum(value**2, axis=1)
        y = value[:, [0, 2, 4]].sum(axis=1)
        z = value[:, [1, 3, 5]].sum(axis=1)
        return np.column_stack([x, y, z])


def test_symmetry_expansion_mirrors_certified_edges_and_merges_zero() -> None:
    environment = _SymmetricEnvironment()
    beta = np.asarray([0.01, 0.02, 0.0, 0.0, 0.0, 0.0])
    xyz = environment.fk(beta)[0]
    representatives = pd.DataFrame(
        [
            _row("zero", None, (1.0, 0.0, 0.0), (0.0,) * 6),
            _row("petal", "zero", tuple(xyz), tuple(beta)),
        ]
    )
    representatives["orbit_size"] = [1, 4]
    edges = pd.DataFrame(
        [
            {
                "fundamental_edge_id": "edge",
                "oriented_source_physical_point_id": "zero",
                "oriented_target_physical_point_id": "petal",
                "source_physical_point_id": "zero",
                "target_physical_point_id": "petal",
                "xyz_length_mm": float(np.linalg.norm(xyz - np.asarray([1.0, 0.0, 0.0])) * 1000.0),
                "weighted_beta_gap_deg": 0.1,
                "raw_beta_gap_deg": 1.0,
            }
        ]
    )

    expanded, full_edges, rejected = expand_connected_orbits(
        representatives,
        edges,
        environment,
        zero_physical_point_id="zero",
        fk_max_mm=1.0e-6,
    )

    assert rejected.empty
    assert len(expanded) == 5
    assert len(full_edges) == 4
    edge_positions = pd.DataFrame(
        {
            "left_position": full_edges["source_physical_point_id"].map(
                {value: index for index, value in enumerate(expanded["physical_point_id"])}
            ),
            "right_position": full_edges["target_physical_point_id"].map(
                {value: index for index, value in enumerate(expanded["physical_point_id"])}
            ),
        }
    )
    labels, _ = component_registry(len(expanded), edge_positions)
    assert len(set(labels)) == 1
