from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quasi_exp.teacher.canonical_atlas import AtlasCandidate, AtlasTaskNode
from quasi_exp.teacher.topology_task_region import (
    BETA_COLUMNS,
    CENTERLINE_BETA_COLUMNS,
    TopologyTaskPolicy,
    build_topology_task_region,
    make_strict_gold_witnessed_continuation,
    waypoint_map_from_frame,
)


def _capability_rows() -> pd.DataFrame:
    rows = []
    for sample_id, x_m in enumerate((0.001, 0.011, 0.021, 0.031)):
        row = {
            "sample_id": sample_id,
            "x_m": x_m,
            "y_m": 0.001,
            "z_m": 0.001,
            "minimum_margin_deg": 2.0,
            "kappa": 2.0 + sample_id,
        }
        row.update({name: 0.0 for name in BETA_COLUMNS})
        rows.append(row)
    # This point shares an occupied support voxel but is not strict Gold.  It
    # must not leak into the seed pool through voxel-level labelling.
    leaked = dict(rows[0])
    leaked["sample_id"] = 99
    leaked["minimum_margin_deg"] = 1.49
    rows.append(leaked)
    return pd.DataFrame(rows)


def _centerline_rows() -> pd.DataFrame:
    rows = []
    for phase, x_m in enumerate((0.001, 0.011, 0.021, 0.031)):
        row = {
            "phase_idx": phase,
            "target_x_m": x_m,
            "target_y_m": 0.001,
            "target_z_m": 0.001,
            "fk_residual_mm": 0.0,
        }
        row.update({name: 0.0 for name in CENTERLINE_BETA_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_task_region_preserves_support_topology_without_voxel_gold_leakage() -> None:
    bounds = np.deg2rad(np.asarray([[-10.0, 10.0]] * 6))
    region = build_topology_task_region(
        _capability_rows(),
        _centerline_rows(),
        bounds,
        TopologyTaskPolicy(
            voxel_mm=10.0,
            roi_mm=10.0,
            expected_phase_count=4,
            expected_main_voxel_count=4,
            expected_task_node_count=8,
            expected_task_edge_count=11,
        ),
    )
    assert region.candidate_admission_gate_pass is True
    assert region.strict_gold_seed_pool["sample_id"].tolist() == [0, 1, 2, 3]
    assert region.task_nodes["kappa"].notna().all()
    assert region.task_edges["edge_type"].value_counts().to_dict() == {
        "centerline_attachment": 4,
        "centerline_cycle": 4,
        "support_face": 3,
    }
    assert region.report["max_waypoint_step_mm"] <= 1.0 + 1.0e-9
    directed = waypoint_map_from_frame(
        region.task_edges, region.task_edge_waypoints
    )
    assert len(directed) == 2 * len(region.task_edges)
    for (left, right), points in directed.items():
        start = region.task_nodes.loc[
            region.task_nodes["task_node_id"].eq(left), ["x_m", "y_m", "z_m"]
        ].to_numpy(dtype=float)[0]
        assert np.max(
            np.linalg.norm(np.diff(np.vstack([start, points]), axis=0), axis=1)
        ) <= 0.001 + 1.0e-12


class _AffineEnvironment:
    bounds = np.deg2rad(np.asarray([[-10.0, 10.0]] * 6))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3] + values[:, 3:]

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        return np.hstack([np.eye(3), np.eye(3)])


def test_witnessed_continuation_records_all_waypoint_minimum_margin() -> None:
    environment = _AffineEnvironment()
    waypoints = {(0, 1): np.asarray([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0]])}
    continuation = make_strict_gold_witnessed_continuation(
        environment,
        waypoints,
        damping=0.001,
        beta_weights=(1.0,) * 6,
        max_corrector_iterations=50,
        residual_max_mm=3.0,
        gold_margin_deg=1.5,
    )
    source = AtlasCandidate(
        0,
        "n0",
        np.zeros(6),
        0.0,
        10.0,
        1.0,
        condition_number=1.0,
    )
    outcome = continuation(
        source, AtlasTaskNode(1, [0.002, 0.0, 0.0], (0,))
    )
    assert outcome.success is True
    assert outcome.waypoint_count == 2
    assert outcome.minimum_margin_deg is not None
    assert outcome.minimum_margin_deg >= 1.5
    assert outcome.residual_mm <= 3.0
