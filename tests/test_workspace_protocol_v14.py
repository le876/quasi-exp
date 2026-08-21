from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.workspace_protocol import (
    WorkspaceAtlasFramePolicy,
    correct_static_targets_from_primary_sections,
    prepare_workspace_atlas_frames,
    supervision_records_from_atlas_frames,
)


class _AffineEnvironment:
    def __init__(self) -> None:
        self.bounds = np.tile(np.asarray([[-1.0, 1.0]]), (6, 1))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3]

    def jacobian(self, beta: np.ndarray) -> np.ndarray:
        del beta
        return np.hstack([np.eye(3), np.zeros((3, 3))])


def _probes() -> pd.DataFrame:
    rows = []
    for parent, cell_ix in ((10, 0), (20, 1)):
        for probe_index, dx in enumerate((0.001, 0.002, 0.003)):
            beta = np.asarray([cell_ix * 0.01 + dx, 0.0, 0.0, 0.0, 0.0, 0.0])
            rows.append(
                {
                    "probe_id": f"n{parent}_p{probe_index}",
                    "physical_point_id": f"point_{parent}_{probe_index}",
                    "node_id": parent,
                    "cell_id": f"cell_{parent}",
                    "cell_level_mm": 10,
                    "cell_ix": cell_ix,
                    "cell_iy": 0,
                    "cell_iz": 0,
                    "is_representative": probe_index == 0,
                    "is_measure_probe": probe_index > 0,
                    "x_m": beta[0],
                    "y_m": beta[1],
                    "z_m": beta[2],
                    **{f"beta{i}_rad": beta[i - 1] for i in range(1, 7)},
                }
            )
    return pd.DataFrame(rows)


def _representative_candidates() -> pd.DataFrame:
    rows = []
    for node_id, x in ((10, 0.001), (20, 0.011)):
        beta = np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0])
        rows.append(
            {
                "node_id": node_id,
                "candidate_id": f"c{node_id}",
                "quality_class": "Gold",
                "solver_success": True,
                "minimum_margin_deg": 10.0,
                "residual_mm": 0.0,
                **{f"beta{i}_rad": beta[i - 1] for i in range(1, 7)},
            }
        )
    return pd.DataFrame(rows)


def test_atlas_frames_expand_every_probe_and_connect_cell_measure_graph() -> None:
    frames = prepare_workspace_atlas_frames(
        _probes(),
        _representative_candidates(),
        pd.DataFrame({"left_node_id": [10], "right_node_id": [20]}),
        _AffineEnvironment(),
        policy=WorkspaceAtlasFramePolicy(
            ill_conditioned_normalized_sigma3_m=1.0e-6,
            ill_conditioned_normalized_kappa=1.0e6,
        ),
    )

    assert len(frames.task_nodes) == 6
    assert frames.task_nodes["task_node_id"].is_unique
    assert frames.task_nodes["cell_measure_complete"].all()
    assert len(frames.task_edges) == 7  # two probe triangles plus one cell edge
    assert (
        frames.task_edges.groupby(
            frames.task_edges["left_node_id"].map(
                frames.task_nodes.set_index("task_node_id")["cell_ix"]
            )
        ).size().max()
        >= 3
    )
    assert set(frames.candidates["task_node_id"]) == set(range(6))
    assert frames.candidates["source"].eq("representative_parent_correction").any()


def test_low_margin_exact_solution_is_silver_risk_not_physical_invalidity() -> None:
    probes = _probes().iloc[:3].copy()
    probes.loc[0, "beta1_rad"] = 0.999
    probes.loc[0, "x_m"] = 0.999
    frames = prepare_workspace_atlas_frames(
        probes,
        _representative_candidates().iloc[:1],
        pd.DataFrame(columns=["left_node_id", "right_node_id"]),
        _AffineEnvironment(),
        policy=WorkspaceAtlasFramePolicy(
            low_margin_deg=0.25,
            ill_conditioned_normalized_sigma3_m=1.0e-6,
            ill_conditioned_normalized_kappa=1.0e6,
        ),
    )

    representative = frames.task_nodes.loc[frames.task_nodes["is_representative"]].iloc[0]
    exact = frames.candidates[
        frames.candidates["task_node_id"].eq(representative.task_node_id)
        & frames.candidates["candidate_id"].eq("capability_exact")
    ].iloc[0]
    assert representative.physical_status == "valid"
    assert "low_margin" in representative.risk_flags
    assert exact.quality == "Silver"


def test_supervision_contract_separates_primary_expert_and_stateful_rows() -> None:
    task_probes = pd.DataFrame(
        [
            {
                "task_probe_id": "p0:a", "task_node_id": 0,
                "physical_point_id": "physical0", "chart_id": "chart_a",
                "selected_candidate_id": "a0", "labelable": True,
                "cell_level_mm": 10, "cell_ix": 0, "cell_iy": 0, "cell_iz": 0,
                "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
            },
            {
                "task_probe_id": "p0:b", "task_node_id": 0,
                "physical_point_id": "physical0", "chart_id": "chart_b",
                "selected_candidate_id": "b0", "labelable": True,
                "cell_level_mm": 10, "cell_ix": 0, "cell_iy": 0, "cell_iz": 0,
                "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
            },
            {
                "task_probe_id": "p1:a", "task_node_id": 1,
                "physical_point_id": "physical1", "chart_id": "chart_a",
                "selected_candidate_id": "a1", "labelable": True,
                "cell_level_mm": 10, "cell_ix": 1, "cell_iy": 0, "cell_iz": 0,
                "x_m": 0.01, "y_m": 0.0, "z_m": 0.0,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {"task_node_id": 0, "candidate_id": "a0", "quality": "Gold", "residual_mm": 0.0, "actual_bounds": True, **{f"beta{i}_rad": 0.0 for i in range(1, 7)}},
            {"task_node_id": 0, "candidate_id": "b0", "quality": "Gold", "residual_mm": 0.0, "actual_bounds": True, **{f"beta{i}_rad": 0.1 for i in range(1, 7)}},
            {"task_node_id": 1, "candidate_id": "a1", "quality": "Silver", "residual_mm": 1.0, "actual_bounds": True, **{f"beta{i}_rad": 0.01 for i in range(1, 7)}},
        ]
    )
    partition = pd.DataFrame(
        {
            "task_node_id": [0, 1],
            "assigned_section_id": ["chart_a", "chart_a"],
        }
    )
    product_edges = pd.DataFrame(
        {
            "left_task_node_id": [0], "left_candidate_id": ["a0"],
            "right_task_node_id": [1], "right_candidate_id": ["a1"],
        }
    )

    records = supervision_records_from_atlas_frames(
        task_probes=task_probes,
        candidates=candidates,
        primary_partition=partition,
        product_edges=product_edges,
    )

    static = [row for row in records if row.kind.value == "static"]
    stateful = [row for row in records if row.kind.value == "stateful"]
    assert len(static) == 3
    assert sum(row.is_primary for row in static) == 2
    assert len(stateful) == 2
    assert {row.chart_id for row in stateful} == {"chart_a"}


def test_dense_static_correction_uses_partition_and_two_parents() -> None:
    task_probes = pd.DataFrame(
        [
            {
                "task_probe_id": f"p{index}", "task_node_id": index,
                "physical_point_id": f"anchor{index}", "chart_id": "chart_a",
                "selected_candidate_id": f"a{index}", "labelable": True,
                "cell_level_mm": 10, "cell_ix": 0, "cell_iy": 0, "cell_iz": 0,
                "x_m": value, "y_m": 0.0, "z_m": 0.0,
            }
            for index, value in enumerate((0.001, 0.003))
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "task_node_id": index, "candidate_id": f"a{index}",
                "quality": "Gold", "residual_mm": 0.0, "actual_bounds": True,
                **{
                    f"beta{i}_rad": (value if i == 1 else 0.0)
                    for i in range(1, 7)
                },
            }
            for index, value in enumerate((0.001, 0.003))
        ]
    )
    partition = pd.DataFrame(
        {"task_node_id": [0, 1], "assigned_section_id": ["chart_a", "chart_a"]}
    )
    targets = pd.DataFrame(
        {
            "physical_point_id": ["new"],
            "x_m": [0.002], "y_m": [0.0], "z_m": [0.0],
        }
    )

    result = correct_static_targets_from_primary_sections(
        targets,
        task_probes=task_probes,
        candidates=candidates,
        primary_partition=partition,
        environment=_AffineEnvironment(),
        source_family="unit_dense",
        priority=0,
        maximum_rows=1,
        policy=WorkspaceAtlasFramePolicy(
            ill_conditioned_normalized_sigma3_m=1.0e-6,
            ill_conditioned_normalized_kappa=1.0e6,
        ),
    )

    assert len(result.records) == 1
    assert result.records[0].chart_id == "chart_a"
    assert result.audit.iloc[0].successful_parent_count == 2
    assert result.audit.iloc[0].status == "accepted"
