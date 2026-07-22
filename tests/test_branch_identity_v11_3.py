from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.branch_identity import (
    BranchIdentityGate,
    BranchRepairPolicy,
    ReferenceBranchTeacher,
    analyze_branch_variants,
    audit_consensus_variants,
    cyclic_traversal_order,
    link_reference_cyclic_candidates,
    select_canonical_root_solution,
    select_canonical_root_phase,
)
from quasi_exp.teacher.canonical import TeacherPolicy, TeacherVariant


class LinearForwardEnvironment:
    bounds = np.deg2rad(np.asarray([[-30.0, 30.0]] * 6))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        return np.asarray(beta, dtype=float).reshape(-1, 6)[:, :3]

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        return np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)

    def theta(self, beta: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(beta, dtype=float).reshape(-1, 6), 5, axis=1)


def _frame(beta_deg: np.ndarray) -> pd.DataFrame:
    beta = np.deg2rad(np.asarray(beta_deg, dtype=float).reshape(-1, 6))
    count = len(beta)
    frame = pd.DataFrame(
        {
            "phase_idx": np.arange(count),
            "teacher_fk_residual_mm": np.full(count, 0.1),
            "joint_margin_min_deg": np.linspace(2.0, 3.0, count),
            "kappa": np.linspace(20.0, 5.0, count),
        }
    )
    for joint in range(6):
        frame[f"teacher_beta{joint + 1}_rad"] = beta[:, joint]
    return frame


def test_cyclic_traversal_order_starts_at_cut_in_both_directions() -> None:
    assert cyclic_traversal_order(6, direction="forward", cut=2).tolist() == [
        2,
        3,
        4,
        5,
        0,
        1,
    ]
    assert cyclic_traversal_order(6, direction="reverse", cut=2).tolist() == [
        2,
        1,
        0,
        5,
        4,
        3,
    ]


def test_branch_forensics_separates_task_and_nullspace_gaps() -> None:
    primary = _frame(np.zeros((4, 6)))
    variant_beta = np.zeros((4, 6))
    variant_beta[:, 5] = 6.0
    variant = _frame(variant_beta)

    def jacobian(_beta: np.ndarray) -> np.ndarray:
        return np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)

    report = analyze_branch_variants(
        primary,
        {"reverse_cut0000": variant},
        jacobian=jacobian,
        cluster_threshold_deg=0.5,
    )

    gaps = report.per_phase_variant_gap
    assert gaps["gap_rms_deg"].min() > 2.0
    assert gaps["nullspace_gap_rms_deg"].min() > 2.0
    assert gaps["task_gap_rms_deg"].max() < 1.0e-10
    assert report.branch_clusters["cluster_count"].eq(2).all()
    assert report.variant_summary.loc[0, "ratio_gap_gt_2deg"] == 1.0


def test_root_selection_prefers_large_margin_low_conditioning_phase() -> None:
    frame = _frame(np.zeros((4, 6)))
    frame["joint_margin_min_deg"] = [2.0, 4.0, 2.5, 2.0]
    frame["kappa"] = [10.0, 3.0, 20.0, 10.0]
    frame["teacher_fk_residual_mm"] = [0.1, 0.01, 0.1, 0.1]
    cluster = pd.DataFrame(
        {
            "phase_idx": np.arange(4),
            "cluster_count": [2, 2, 2, 2],
            "min_cluster_separation_deg": [2.0, 8.0, 4.0, 2.0],
        }
    )

    selected = select_canonical_root_phase(frame, branch_clusters=cluster)

    assert selected["phase_idx"] == 1
    assert selected["root_score"] == max(selected["score_by_phase"])


def test_reference_linker_freezes_root_cluster_and_avoids_cheaper_other_branch() -> None:
    reference = np.deg2rad(np.zeros((4, 6)))
    other = reference.copy()
    other[:, 5] = np.deg2rad(12.0)
    layers = [np.vstack([reference[i], other[i]]) for i in range(4)]
    residuals = [np.asarray([0.2, 0.0]) for _ in range(4)]

    selected, report = link_reference_cyclic_candidates(
        layers,
        residuals,
        reference_beta=reference,
        root_phase_idx=2,
        canonical_root_beta=reference[2],
        beta_weights=np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0]),
        lambda_velocity=1.0,
        lambda_reference=2.0,
        closure_weight=5.0,
        root_cluster_threshold_deg=0.5,
    )

    assert report["success"] is True
    assert report["root_cluster_candidate_count"] == 1
    assert np.allclose(selected, reference)


def test_consensus_audit_and_gate_are_fail_closed_on_identity_and_margin() -> None:
    consensus = np.deg2rad(np.zeros((100, 6)))
    good = consensus.copy()
    bad = consensus.copy()
    bad[:5, 0] = np.deg2rad(6.0)

    audit = audit_consensus_variants(
        consensus,
        {"repeat": good, "reverse_cut0000": bad},
        cluster_threshold_deg=0.5,
    )
    gate = BranchIdentityGate().evaluate(
        numerical_metrics={
            "residual_p95_mm": 0.1,
            "residual_max_mm": 0.2,
            "joint_margin_min_deg": 1.49,
            "delta_beta_rms_p95_deg": 0.1,
            "delta_beta_rms_max_deg": 0.2,
            "acceleration_beta_rms_p95_deg": 0.1,
            "seam_beta_rms_deg": 0.1,
        },
        variant_audit=audit,
    )

    assert audit.summary.set_index("variant").loc[
        "reverse_cut0000", "ratio_gap_gt_1deg"
    ] == 0.05
    assert gate["gate_pass"] is False
    assert gate["checks"]["strict_joint_margin"] is False
    assert gate["checks"]["traversal_cut_ratio"] is False
    assert gate["checks"]["single_branch_cluster"] is False


def test_reference_branch_teacher_is_direction_and_cut_independent() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    targets = np.column_stack(
        [0.1 * np.cos(angle), 0.1 * np.sin(angle), np.full_like(angle, 0.05)]
    )
    reference = np.column_stack(
        [targets, np.zeros((12, 2)), np.full(12, np.deg2rad(8.0))]
    )
    teacher = ReferenceBranchTeacher(LinearForwardEnvironment())
    corrector = TeacherPolicy(
        variant=TeacherVariant.T1,
        candidate_budget=4,
        max_corrector_iterations=20,
        tracking_tolerance_mm=0.01,
        solver_seed=7,
    )
    policy = BranchRepairPolicy(
        trust_radius_deg=0.5,
        candidate_budget=4,
        lambda_reference=2.0,
        lambda_previous=1.0,
    )

    forward = teacher.solve(
        targets,
        reference_beta=reference,
        root_phase_idx=3,
        canonical_root_beta=reference[3],
        direction="forward",
        cut=3,
        corrector_policy=corrector,
        repair_policy=policy,
        solver_seed=11,
    )
    reverse_cut = teacher.solve(
        targets,
        reference_beta=reference,
        root_phase_idx=3,
        canonical_root_beta=reference[3],
        direction="reverse",
        cut=8,
        corrector_policy=corrector,
        repair_policy=policy,
        solver_seed=99,
    )

    assert forward.success is True
    assert reverse_cut.success is True
    gap_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(forward.beta_rad - reverse_cut.beta_rad), axis=1))
    )
    assert np.percentile(gap_deg, 95) < 0.001
    assert np.allclose(forward.beta_rad[:, 5], reference[:, 5], atol=1.0e-12)
    assert np.allclose(reverse_cut.beta_rad[:, 5], reference[:, 5], atol=1.0e-12)
    assert reverse_cut.provenance["canonical_root_phase_idx"] == 3
    assert reverse_cut.metrics["trust_violation_count"] == 0.0


def test_root_solution_selection_freezes_one_cluster_by_canonical_cost() -> None:
    candidates = np.zeros((4, 6))
    candidates[2:, 5] = np.deg2rad(10.0)
    selection = select_canonical_root_solution(
        candidates,
        residual_mm=np.asarray([0.2, 0.1, 0.05, 0.04]),
        environment=LinearForwardEnvironment(),
        safe_margin_deg=1.5,
        cluster_threshold_deg=0.5,
    )

    assert len(selection.cluster_table) == 2
    assert selection.selected_cluster_id == 0
    assert abs(np.rad2deg(selection.selected_beta[5])) < 1.0e-12
    assert selection.candidate_table["cluster_id"].nunique() == 2
