from __future__ import annotations

import numpy as np

from quasi_exp.teacher.bacra_geometry_holdout import (
    LoopReferencePolicy,
    _gap_directed_enrich_layers,
)
from quasi_exp.teacher.capability_path_bridge import (
    analyze_hard_cyclic_reachability,
    independently_certify_hard_cyclic_path,
)
from quasi_exp.teacher.canonical import TeacherPolicy, link_cyclic_candidates


class LinearEnvironment:
    bounds = np.asarray([[-2.0, 2.0]] * 6, dtype=float)

    def fk(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
        return beta[:, :3]

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
        del beta_rad
        return np.column_stack([np.eye(3), np.zeros((3, 3))])


def _fragmented_layers() -> tuple[np.ndarray, list[np.ndarray]]:
    phase = np.arange(8, dtype=float) * (2.0 * np.pi / 8.0)
    targets = np.column_stack(
        [
            0.005 * np.cos(phase),
            0.005 * np.sin(phase),
            np.zeros_like(phase),
        ]
    )
    layers: list[np.ndarray] = []
    for phase_idx, target in enumerate(targets):
        branch_deg = 0.0 if phase_idx < 4 else 5.0
        beta = np.concatenate(
            [target, np.deg2rad([branch_deg, 0.0, 0.0])]
        )
        layers.append(beta.reshape(1, 6))
    return targets, layers


def test_gap_enrichment_repairs_fragmented_but_continuable_cycle() -> None:
    targets, layers = _fragmented_layers()
    initial = analyze_hard_cyclic_reachability(
        layers, max_transition_deg=2.0
    )
    assert initial.closed_cycle_exists is False

    policy = LoopReferencePolicy(
        phase_count=len(targets),
        joint_margin_min_deg=1.5,
        phase_beta_rms_p95_deg=2.0,
        phase_beta_rms_max_deg=2.0,
        acceleration_beta_rms_p95_deg=2.0,
        seam_beta_rms_deg=2.0,
        candidate_budget_per_phase=1,
        candidate_cluster_deg=0.1,
        gap_enrichment_rounds=3,
        gap_enrichment_sources_per_frontier=2,
        gap_enrichment_round_candidate_budget=32,
        gap_enrichment_max_candidates_per_phase=8,
        teacher_policy=TeacherPolicy(
            tracking_tolerance_mm=1.0,
            safe_joint_margin_deg=1.5,
            max_corrector_iterations=20,
        ),
    )
    enriched, residuals, _sources, report = (
        _gap_directed_enrich_layers(
            LinearEnvironment(),
            targets,
            layers,
            [np.zeros(len(layer)) for layer in layers],
            [["initial"] for _layer in layers],
            policy,
        )
    )
    selected, link = link_cyclic_candidates(
        enriched,
        residuals,
        lambda_velocity=1.0,
        closure_weight=1.0,
        max_transition_deg=2.0,
    )

    assert report["closed_cycle_exists"] is True
    assert report["total_added_candidate_count"] > 0
    assert all(
        addition["source_transition_deg"] <= 2.0 + 1.0e-7
        for round_payload in report["rounds"]
        for addition in round_payload["candidate_additions"]
        if addition["accepted"]
    )
    assert link["success"] is True
    assert len(selected) == len(targets)


def test_independent_certificate_recomputes_all_edges_and_seam() -> None:
    targets, layers = _fragmented_layers()
    beta = np.vstack([layer[0] for layer in layers])
    achieved = LinearEnvironment().fk(beta)
    report, detail = independently_certify_hard_cyclic_path(
        beta,
        targets,
        achieved,
        LinearEnvironment.bounds,
        residual_p95_mm=1.0,
        residual_max_mm=3.0,
        gold_margin_deg=1.5,
        max_transition_deg=2.0,
    )

    assert report["checks"]["residual_max"] is True
    assert report["checks"]["gold_margin"] is True
    assert report["checks"]["all_phase_edges_at_most_limit"] is False
    assert report["gate_pass"] is False
    assert len(detail["certified_transition_to_next_deg"]) == len(targets)
