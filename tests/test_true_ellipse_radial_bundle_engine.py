from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import true_ellipse_radial_bundle_engine as engine  # noqa: E402


def test_registered_joint_domains_match_v6_and_standard_sampling_ranges() -> None:
    current = engine.registered_joint_domain("current_v6")
    standard = engine.registered_joint_domain("standard_beta34_10deg_v1")

    np.testing.assert_allclose(
        current.bounds_deg,
        [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-15, 15], [-15, 15]],
    )
    np.testing.assert_allclose(
        standard.bounds_deg,
        [[-5, 5], [-5, 5], [-10, 10], [-10, 10], [-15, 15], [-15, 15]],
    )
    assert current.fingerprint != standard.fingerprint
    assert engine.registered_joint_domain("standard_beta34_10deg_v1").fingerprint == standard.fingerprint


def test_joint_margin_report_is_per_joint_and_fail_closed_for_nonfinite_labels() -> None:
    domain = engine.registered_joint_domain("standard_beta34_10deg_v1")
    beta_deg = np.asarray(
        [
            [0.0, 0.0, -9.94, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 9.80, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 14.50, 0.0],
        ]
    )

    report = engine.joint_margin_report(np.deg2rad(beta_deg), domain=domain)

    assert report["rows"] == 3
    assert report["out_of_bounds_count"] == 0
    assert report["at_bound_count"] == 0
    assert report["min_joint_margin_deg"] == pytest.approx(0.06)
    assert report["joint_min_margin_deg"]["beta3"] == pytest.approx(0.06)
    assert report["joint_min_margin_deg"]["beta4"] == pytest.approx(0.20)

    broken = np.deg2rad(beta_deg)
    broken[1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        engine.joint_margin_report(broken, domain=domain)


def test_balanced_margin_policy_requires_min_p01_p05_and_zero_boundary_counts() -> None:
    policy = engine.balanced_joint_margin_policy()
    passing = {
        "rows": 100,
        "min_joint_margin_deg": 0.05,
        "joint_margin_p01_deg": 0.10,
        "joint_margin_p05_deg": 0.25,
        "out_of_bounds_count": 0,
        "at_bound_count": 0,
    }

    decision = engine.evaluate_joint_margin_gate(passing, policy=policy)

    assert decision["joint_margin_gate_pass"] is True
    assert all(decision["checks"].values())
    for field in (
        "min_joint_margin_deg",
        "joint_margin_p01_deg",
        "joint_margin_p05_deg",
        "out_of_bounds_count",
        "at_bound_count",
    ):
        failed = dict(passing)
        failed[field] = -1 if "count" not in field else 1
        assert engine.evaluate_joint_margin_gate(failed, policy=policy)["joint_margin_gate_pass"] is False


def test_joint_margin_barrier_is_smooth_finite_and_stronger_near_a_limit() -> None:
    domain = engine.registered_joint_domain("standard_beta34_10deg_v1")
    interior = np.zeros((1, 6), dtype=float)
    near_limit = np.deg2rad([[4.99, 0.0, 0.0, 0.0, 0.0, 0.0]])

    interior_barrier = engine.joint_margin_barrier_residual(
        interior,
        domain=domain,
        soft_margin_deg=0.25,
    )
    near_barrier = engine.joint_margin_barrier_residual(
        near_limit,
        domain=domain,
        soft_margin_deg=0.25,
    )

    assert interior_barrier.shape == (6,)
    assert near_barrier.shape == (6,)
    assert np.isfinite(interior_barrier).all()
    assert np.isfinite(near_barrier).all()
    assert np.linalg.norm(near_barrier) > 100.0 * np.linalg.norm(interior_barrier)


def test_radius_frontier_stops_at_first_strict_failure_but_tracks_exploration() -> None:
    status = pd.DataFrame(
        {
            "radius_mm": [100.0, 102.5, 105.0, 107.5, 110.0],
            "strict_gate_pass": [True, True, True, False, True],
            "rescue_admission_pass": [True, True, True, True, True],
        }
    )

    frontier = engine.compute_radius_frontiers(status, anchor_mm=100.0)

    assert frontier["strict_geometry_rmax_mm"] == 105.0
    assert frontier["exploratory_rescue_rmax_mm"] == 110.0
    assert frontier["first_strict_failure_mm"] == 107.5


def test_training_anchor_densification_adds_full_1p25mm_midpoints_only_above_100() -> None:
    assert engine.training_anchor_radii(105.0) == (101.25, 103.75)
    assert engine.training_anchor_radii(120.0) == (
        101.25,
        103.75,
        106.25,
        108.75,
        111.25,
        113.75,
        116.25,
        118.75,
    )


def test_holdout_selection_walks_down_from_frontier_and_locks_validation_7p5mm_lower() -> None:
    support = pd.DataFrame(
        {
            "radius_mm": [102.5, 105.0, 107.5, 110.0, 112.5],
            "strict_support_gate_pass": [True, True, False, True, False],
        }
    )

    selected = engine.select_registered_holdouts(
        strict_geometry_rmax_mm=112.5,
        support_by_radius=support,
        minimum_test_radius_mm=105.0,
    )

    assert selected["test_radius_mm"] == 110.0
    assert selected["validation_radius_mm"] == 102.5
    assert selected["selection_gate_pass"] is True

    with pytest.raises(ValueError, match="no support-backed test radius"):
        engine.select_registered_holdouts(
            strict_geometry_rmax_mm=104.9,
            support_by_radius=support,
            minimum_test_radius_mm=105.0,
        )


def test_v7_protocol_fingerprint_binds_domain_margin_policy_and_radius_ladder() -> None:
    protocol = engine.v7_standard_domain_protocol()
    changed = engine.RadialBundleProtocol(
        **{
            **protocol.as_dict(),
            "formal_checkpoints_mm": tuple(protocol.formal_checkpoints_mm[:-1]),
        }
    )

    assert protocol.joint_domain_id == "standard_beta34_10deg_v1"
    assert protocol.target_radius_mm == 120.0
    assert protocol.fingerprint != changed.fingerprint


def test_identity_output_link_round_trips_without_clipping() -> None:
    domain = engine.registered_joint_domain("standard_beta34_10deg_v1")
    link = engine.registered_output_link("identity")
    beta = np.deg2rad([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]])

    latent, report = link.encode(beta, domain=domain)
    decoded = link.decode(latent, domain=domain)

    np.testing.assert_array_equal(latent, beta)
    np.testing.assert_array_equal(decoded, beta)
    assert report["target_link_clip_count"] == 0


def test_tanh_bounds_output_link_round_trips_interior_targets_and_never_decodes_outside_domain() -> None:
    domain = engine.registered_joint_domain("standard_beta34_10deg_v1")
    link = engine.registered_output_link("tanh_bounds")
    beta = np.deg2rad([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]])

    latent, report = link.encode(beta, domain=domain)
    decoded = link.decode(latent, domain=domain)

    np.testing.assert_allclose(decoded, beta, atol=1.0e-14, rtol=0.0)
    assert report["target_link_clip_count"] == 0

    extreme = np.full((2, 6), 1.0e9)
    bounded = link.decode(extreme, domain=domain)
    assert np.all(bounded < domain.bounds_rad[:, 1][None, :])
    assert np.all(bounded > domain.bounds_rad[:, 0][None, :])


def test_tanh_bounds_output_link_reports_training_target_clips_at_hard_bounds() -> None:
    domain = engine.registered_joint_domain("standard_beta34_10deg_v1")
    link = engine.registered_output_link("tanh_bounds")
    beta = np.zeros((1, 6), dtype=float)
    beta[0, 2] = domain.bounds_rad[2, 1]

    latent, report = link.encode(beta, domain=domain)
    decoded = link.decode(latent, domain=domain)

    assert report["target_link_clip_count"] == 1
    assert np.isfinite(latent).all()
    assert decoded[0, 2] < domain.bounds_rad[2, 1]


def test_output_link_fingerprint_distinguishes_identity_and_tanh_parameterization() -> None:
    identity = engine.registered_output_link("identity")
    bounded = engine.registered_output_link("tanh_bounds")

    assert identity.fingerprint != bounded.fingerprint
    with pytest.raises(ValueError, match="unregistered output link"):
        engine.registered_output_link("clip_after_test")
