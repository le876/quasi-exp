from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quasi_exp.teacher.atlas_audit import AtlasAuditPolicy
from quasi_exp.teacher.canonical_atlas import AtlasPolicy, ContinuationOutcome
from quasi_exp.teacher.workspace_atlas import CellResolution, OverlapKind, RepresentationMode
from quasi_exp.teacher.workspace_atlas_integration import (
    WorkspaceAtlasIntegrationPolicy,
    build_workspace_atlas_integration,
)


class _RedundantAffineEnvironment:
    """FK uses the first three coordinates; the other three are redundant."""

    def __init__(self) -> None:
        self.bounds = np.tile(np.asarray([[-1.0, 1.0]], dtype=float), (6, 1))
        self.callback_calls = 0

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3].copy()

    def jacobian(self, beta: np.ndarray) -> np.ndarray:
        del beta
        return np.hstack([np.eye(3), np.zeros((3, 3))])


def _diamond_task_frame(*, cell_measure_complete: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_node_id": [0, 1, 2, 3],
            "task_id": ["task_0", "task_1", "task_2", "task_3"],
            "physical_point_id": ["p0", "p1", "p2", "p3"],
            "x_m": [0.0, 0.001, 0.0, 0.001],
            "y_m": [0.0, 0.0, 0.001, 0.001],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "cell_level_mm": [10, 10, 10, 10],
            "cell_ix": [0, 0, 0, 0],
            "cell_iy": [0, 0, 0, 0],
            "cell_iz": [0, 0, 0, 0],
            "is_representative": [True, False, False, False],
            "is_measure_probe": [True, True, True, True],
            "cell_measure_complete": [
                cell_measure_complete,
                cell_measure_complete,
                cell_measure_complete,
                cell_measure_complete,
            ],
        }
    )


def _diamond_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "left_node_id": [0, 0, 1, 2],
            "right_node_id": [1, 2, 3, 3],
        }
    )


def _candidate_frame(*, branch_offsets_deg: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = {
        0: (0.0, 0.0, 0.0),
        1: (0.001, 0.0, 0.0),
        2: (0.0, 0.001, 0.0),
        3: (0.001, 0.001, 0.0),
    }
    for node_id, xyz in targets.items():
        for branch_index, offset_deg in enumerate(branch_offsets_deg):
            beta = np.asarray([*xyz, math.radians(offset_deg), 0.0, 0.0])
            rows.append(
                {
                    "task_node_id": node_id,
                    "candidate_id": f"n{node_id}_branch{branch_index}",
                    "beta_rad": beta,
                    "residual_mm": 0.01,
                    "min_margin_deg": 2.0,
                    "normalized_min_margin": 0.5,
                    "posture_cost": 0.0,
                    "condition_number": 1.0,
                    "quality": "Gold",
                    "solver_success": True,
                    "actual_bounds": True,
                }
            )
    return pd.DataFrame.from_records(rows)


def _policy(*, edge_match_deg: float = 0.5) -> WorkspaceAtlasIntegrationPolicy:
    return WorkspaceAtlasIntegrationPolicy(
        atlas_policy=AtlasPolicy(
            edge_match_deg=edge_match_deg,
            root_count=1,
            top_section_count=8,
        ),
        audit_policy=AtlasAuditPolicy(
            endpoint_count=1,
            paths_per_endpoint=2,
            loop_count=1,
            path_p95_deg=0.5,
            loop_p95_deg=0.5,
            direction_p95_deg=0.5,
            repeat_p95_deg=0.2,
            common_max_deg=1.0,
        ),
    )


def test_redundant_affine_diamond_derives_nonstitchable_stateful_partition() -> None:
    environment = _RedundantAffineEnvironment()
    result = build_workspace_atlas_integration(
        _diamond_task_frame(cell_measure_complete=True),
        _candidate_frame(branch_offsets_deg=(0.0, 3.0)),
        environment,
        task_edges=_diamond_edges(),
        policy=_policy(),
    )

    assert len(result.canonical_atlas.charts) == 2
    assert result.audit_report.gate_pass is True
    assert result.audit_report.checks["fundamental_cycle_scope"]["scope"] == (
        "all_selected_robust_edges_and_all_fundamental_cycles"
    )
    assert result.audit_report.checks["multipath"]["gate_pass"] is True
    assert result.workspace_result.overlap_assessments[0].kind is OverlapKind.NON_STITCHABLE
    assert result.workspace_result.representation.mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS
    assert result.workspace_result.representation.static_inverse_authorized is False
    assert result.workspace_result.representation.stateful_inverse_authorized is True

    task_ids = result.frames["audit_tasks"]["task_id"].tolist()
    assert len(task_ids) == len(set(task_ids))
    probe_ids = result.frames["task_probes"]["task_probe_id"].tolist()
    assert len(probe_ids) == len(set(probe_ids))
    for section in result.workspace_input.sections:
        assert set(section.task_edges) <= set(section.robust_edges)
        assert section.full_fundamental_closure_checked is True
    assert result.report["fresh_audit"]["execution_count"] > 0


def test_redundant_affine_diamond_multipath_failure_blocks_section() -> None:
    environment = _RedundantAffineEnvironment()

    def path_dependent_continuation(source, target):
        environment.callback_calls += 1
        beta = np.asarray([*target.xyz_m, 0.0, 0.0, 0.0], dtype=float)
        # All individual graph edges still match within the deliberately wider
        # product-edge threshold.  Only the two fresh root-to-node paths around
        # the diamond expose their incompatible redundant-coordinate return.
        if source.node_id == 1 and target.node_id == 3:
            beta[5] = math.radians(2.0)
        return ContinuationOutcome(
            beta_rad=beta,
            residual_mm=0.01,
            success=True,
            actual_bounds=True,
            status="fresh-path-dependent-affine-corrector",
        )

    result = build_workspace_atlas_integration(
        _diamond_task_frame(cell_measure_complete=True),
        _candidate_frame(branch_offsets_deg=(0.0,)),
        environment,
        task_edges=_diamond_edges(),
        continuation=path_dependent_continuation,
        policy=_policy(edge_match_deg=1.0),
    )

    assert len(result.product_graph.robust_edges) == 4
    assert result.audit_report.checks["multipath"]["gate_pass"] is False
    assert result.audit_report.gate_pass is False
    assert result.workspace_input.sections[0].multipath_p95_deg > 0.5
    assert result.workspace_result.section_assessments[0].valid is False
    assert result.workspace_result.representation.mode is RepresentationMode.BLOCKED
    assert environment.callback_calls > result.product_graph.continuation_attempt_count


def test_representative_task_nodes_do_not_imply_full_cell_coverage() -> None:
    result = build_workspace_atlas_integration(
        _diamond_task_frame(cell_measure_complete=False),
        _candidate_frame(branch_offsets_deg=(0.0,)),
        _RedundantAffineEnvironment(),
        task_edges=_diamond_edges(),
        policy=_policy(),
    )

    assessment = result.workspace_result.cell_assessments[0]
    assert assessment.resolution is CellResolution.PARTIAL
    assert result.report["cell_coverage"]["full_cell_status_asserted_from_representative_alone"] is False
    assert result.report["cell_coverage"]["representative_only_or_incomplete_cell_count"] == 1
