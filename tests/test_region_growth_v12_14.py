from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.region_growth import (
    RegionLabelPolicy,
    canonicalize_seed_set,
    consistency_audit,
    pack_voxels,
    reduce_parent_candidates,
    plan_dense_targets,
    select_sparse_targets,
    spatial_train_validation_split,
    unpack_voxels,
)


def _d3_row(
    family_id: str,
    source: str,
    beta0: float,
    *,
    x: float = 1.0,
    phase: int = 0,
) -> dict:
    return {
        "family_id": family_id,
        "group_id": "test",
        "phase_idx": phase,
        "dataset_source": source,
        "quality_class": "Gold",
        "x_m": x,
        "y_m": 0.2,
        "z_m": 0.3,
        **{
            name: beta0 + index * 1.0e-4
            for index, name in enumerate(BETA_COLUMNS)
        },
    }


def test_seed_cleanup_prefers_bridge_and_drops_zero_offset_tube() -> None:
    frame = pd.DataFrame(
        [
            _d3_row("bridge", "D2_bridge", 0.01),
            _d3_row("bridge_u+0_v+0", "D3_tube", 0.03),
            _d3_row("tube_unique", "D3_tube", 0.02, x=1.001),
        ]
    )
    clean, duplicate = canonicalize_seed_set(frame)
    assert clean["family_id"].tolist() == ["bridge", "tube_unique"]
    assert duplicate["family_id"].tolist() == ["bridge_u+0_v+0"]
    assert duplicate.iloc[0]["selected_family_id"] == "bridge"
    assert duplicate.iloc[0]["duplicate_reason"] == "zero_offset_tube_copy"
    assert duplicate.iloc[0]["beta_rms_to_selected_deg"] > 1.0


def test_voxel_pack_round_trip_in_signed_domain() -> None:
    values = np.asarray([[-100, 20, 0], [0, 0, 0], [202, -37, 91]])
    np.testing.assert_array_equal(unpack_voxels(pack_voxels(values)), values)


def test_candidate_medoid_and_registered_tiers() -> None:
    beta = np.zeros((3, 6))
    beta[1, 0] = np.deg2rad(0.30)
    beta[2, 0] = np.deg2rad(-0.30)
    candidates = pd.DataFrame(
        {
            "success": [True, True, True],
            "residual_mm": [0.2, 0.4, 0.6],
            **{
                name: beta[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
        }
    )
    result = reduce_parent_candidates(
        candidates, policy=RegionLabelPolicy()
    )
    assert result["accepted"]
    assert result["quality_class"] == "RegionGold"
    assert result["selected_candidate_row"] == 0

    silver = candidates.copy()
    silver["residual_mm"] = [1.2, 1.4, 1.5]
    result = reduce_parent_candidates(silver, policy=RegionLabelPolicy())
    assert result["accepted"]
    assert result["quality_class"] == "RegionSilver"

    conflict = candidates.copy()
    conflict.loc[2, "beta1_rad"] = np.deg2rad(3.0)
    result = reduce_parent_candidates(conflict, policy=RegionLabelPolicy())
    assert not result["accepted"]
    assert result["quality_class"] == "Reject"


def test_candidate_reducer_keeps_largest_consistent_parent_clique() -> None:
    beta = np.zeros((5, 6))
    beta[1, 0] = np.deg2rad(0.02)
    beta[2, 0] = np.deg2rad(-0.02)
    beta[3:, 0] = np.deg2rad([1.8, 1.85])
    candidates = pd.DataFrame(
        {
            "success": True,
            "residual_mm": 0.2,
            **{
                name: beta[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
        }
    )
    result = reduce_parent_candidates(
        candidates, policy=RegionLabelPolicy()
    )
    assert result["accepted"]
    assert result["quality_class"] == "RegionGold"
    assert result["consensus_parent_count"] == 3
    assert result["excluded_conflicting_parent_count"] == 2
    assert len(result["consensus_candidate_rows"]) == 3
    assert result["candidate_gap_max_deg"] < 0.02
    assert result["raw_candidate_gap_max_deg"] > 0.7


def test_spatial_split_has_disjoint_macro_voxels_and_buffer() -> None:
    xyz = np.asarray(
        [[index * 0.02, 0.0, 0.0] for index in range(20)], dtype=float
    )
    frame = pd.DataFrame(xyz, columns=XYZ_COLUMNS)
    result, report = spatial_train_validation_split(
        frame,
        macro_voxel_mm=20.0,
        validation_fraction=0.20,
        buffer_mm=5.0,
        seed=7,
    )
    assert report["train_rows"] > 0
    assert report["validation_rows"] > 0
    train = result.loc[result["split"].eq("train"), XYZ_COLUMNS].to_numpy()
    validation = result.loc[
        result["split"].eq("validation"), XYZ_COLUMNS
    ].to_numpy()
    distance = np.linalg.norm(
        train[:, None, :] - validation[None, :, :], axis=2
    )
    assert float(distance.min()) >= 0.005 - 1.0e-12


def test_empty_consistency_evidence_is_json_safe_and_fail_closed() -> None:
    nodes = pd.DataFrame(columns=["node_id", *XYZ_COLUMNS, *BETA_COLUMNS])
    edges = pd.DataFrame(
        columns=[
            "left_node_id",
            "right_node_id",
            "distance_mm",
            "beta_gap_deg",
            "accepted",
        ]
    )
    candidates = pd.DataFrame(
        columns=["target_id", "success", *BETA_COLUMNS]
    )
    _triangles, _paths, report = consistency_audit(
        nodes,
        edges,
        candidates,
        triangle_count=8,
        two_path_count=8,
        seed=1,
    )
    assert report["triangle_p95_deg"] is None
    assert report["triangle_max_deg"] is None
    assert report["two_path_p95_deg"] is None


def test_triangle_audit_runs_predictor_corrector_return_loop() -> None:
    class LinearEnvironment:
        bounds = np.asarray([[-1.0, 1.0]] * 6)

        def fk(self, beta_rad: np.ndarray) -> np.ndarray:
            return np.asarray(beta_rad, dtype=float)[:3]

        def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
            del beta_rad
            return np.column_stack([np.eye(3), np.zeros((3, 3))])

    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.0, 0.005, 0.0]]
    )
    nodes = pd.DataFrame(
        {
            "node_id": ["a", "b", "c"],
            **{
                name: xyz[:, index]
                for index, name in enumerate(XYZ_COLUMNS)
            },
            **{
                name: (
                    xyz[:, index]
                    if index < 3
                    else np.zeros(len(xyz))
                )
                for index, name in enumerate(BETA_COLUMNS)
            },
        }
    )
    edges = pd.DataFrame(
        {
            "left_node_id": ["a", "b", "c"],
            "right_node_id": ["b", "c", "a"],
            "beta_gap_deg": [0.2, 0.2, 0.2],
            "accepted": True,
        }
    )
    _triangles, _paths, report = consistency_audit(
        nodes,
        edges,
        pd.DataFrame(columns=["target_id", "success", *BETA_COLUMNS]),
        triangle_count=1,
        two_path_count=0,
        seed=1,
        environment=LinearEnvironment(),
        policy=RegionLabelPolicy(),
    )
    assert report["triangle_successful"] == 1
    assert report["triangle_failed"] == 0
    assert report["triangle_max_deg"] < 1.0e-9


def test_triangle_audit_ignores_edges_outside_retained_component() -> None:
    nodes = pd.DataFrame(
        {
            "node_id": ["a", "b"],
            "x_m": [0.0, 0.005],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            **{name: [0.0, 0.0] for name in BETA_COLUMNS},
        }
    )
    edges = pd.DataFrame(
        {
            "left_node_id": ["a", "a", "b"],
            "right_node_id": ["b", "outside", "outside"],
            "beta_gap_deg": [0.0, 0.0, 0.0],
            "accepted": True,
        }
    )
    triangles, _paths, report = consistency_audit(
        nodes,
        edges,
        pd.DataFrame(columns=["target_id", "success", *BETA_COLUMNS]),
        triangle_count=1,
        two_path_count=0,
        seed=1,
    )
    assert len(triangles) == 0
    assert report["triangle_available"] == 0


def test_sparse_inventory_contains_a_connected_backbone() -> None:
    coordinates = np.asarray(
        [(i, j, 0) for i in range(8) for j in range(8)], dtype=np.int64
    )
    region = pd.DataFrame(coordinates, columns=("voxel_i", "voxel_j", "voxel_k"))
    region["voxel_key"] = pack_voxels(coordinates)
    region["x_m"] = (region["voxel_i"] + 0.5) * 0.005
    region["y_m"] = (region["voxel_j"] + 0.5) * 0.005
    region["z_m"] = 0.0025
    region["region_wave"] = np.maximum(
        region["voxel_i"], region["voxel_j"]
    )
    selected = select_sparse_targets(
        region,
        np.asarray([[0.0025, 0.0025, 0.0025]]),
        count=32,
        seed=3,
        parent_distance_max_mm=25.0,
        required_parent_support_count=3,
    )
    backbone = selected.loc[selected["selection_role"].eq("connected_backbone")]
    assert len(backbone) == 16
    occupied = set(map(int, backbone["voxel_key"]))
    first = next(iter(occupied))
    reached = {first}
    frontier = [first]
    while frontier:
        coordinate = unpack_voxels(np.asarray([frontier.pop()]))[0]
        for offset in (
            np.asarray(
                [
                    (i, j, k)
                    for i in (-1, 0, 1)
                    for j in (-1, 0, 1)
                    for k in (-1, 0, 1)
                    if (i, j, k) != (0, 0, 0)
                ]
            )
        ):
            neighbor = int(pack_voxels(coordinate[None, :] + offset)[0])
            if neighbor in occupied and neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    assert reached == occupied
    assert len(selected) == 32
    frontier = selected.loc[
        selected["selection_role"].eq("parent_reachable_frontier")
    ]
    assert len(frontier) == 16
    for growth_batch in sorted(frontier["growth_batch"].unique()):
        earlier = selected.loc[selected["growth_batch"].lt(growth_batch)]
        current = selected.loc[selected["growth_batch"].eq(growth_batch)]
        distance, _ = cKDTree(
            earlier.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
        ).query(current.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), k=3)
        assert float(np.max(distance[:, -1]) * 1000.0) <= 25.0 + 1.0e-9


def test_dense_plan_can_sample_multiple_continuous_points_per_voxel() -> None:
    coordinates = np.asarray(
        [(i, j, 0) for i in range(3) for j in range(3)], dtype=np.int64
    )
    sparse = pd.DataFrame(
        {
            "node_id": [f"n{index}" for index in range(len(coordinates))],
            "voxel_key": pack_voxels(coordinates),
            "x_m": (coordinates[:, 0] + 0.5) * 0.005,
            "y_m": (coordinates[:, 1] + 0.5) * 0.005,
            "z_m": (coordinates[:, 2] + 0.5) * 0.005,
            **{name: np.zeros(len(coordinates)) for name in BETA_COLUMNS},
        }
    )
    region = sparse.loc[
        :, ["voxel_key", *XYZ_COLUMNS]
    ].copy()
    region.loc[:, ["voxel_i", "voxel_j", "voxel_k"]] = coordinates
    plan = plan_dense_targets(
        region,
        sparse,
        attempt_count=40,
        seed=7,
        voxel_size_mm=5.0,
        parent_distance_max_mm=15.0,
        minimum_anchor_count=2,
    )
    assert len(plan) == 40
    assert plan["voxel_key"].duplicated().any()
    assert float(plan["planned_farthest_anchor_distance_mm"].max()) <= 15.0
    planned_voxels = np.floor(
        plan.loc[:, XYZ_COLUMNS].to_numpy(dtype=float) / 0.005
    ).astype(np.int64)
    np.testing.assert_array_equal(
        planned_voxels,
        plan.loc[:, ["voxel_i", "voxel_j", "voxel_k"]].to_numpy(
            dtype=np.int64
        ),
    )
