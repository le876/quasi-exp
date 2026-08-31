from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS
from quasi_exp.teacher.retry18_root_tip import (
    RootTipPolicy,
    classify_root_connectors,
    common_zero_tip_support,
    fit_sqrt_u_tip_profile,
    profile_overlap_metrics,
    sample_tip_volume,
    weighted_sobol_beta_ball,
    zero_connected_radial_intervals,
)


def test_weighted_sobol_ball_is_deterministic_bounded_and_not_clipped() -> None:
    bounds = np.deg2rad(np.asarray([[-5, 5], [-5, 5], [-10, 10], [-10, 10], [-15, 15], [-15, 15]], dtype=float))
    weights = np.asarray([4, 4, 2, 2, 1, 1], dtype=float)
    a = weighted_sobol_beta_ball(power=8, seed=41, maximum_weighted_radius_deg=3.0, beta_weights=weights, beta_bounds_rad=bounds)
    b = weighted_sobol_beta_ball(power=8, seed=41, maximum_weighted_radius_deg=3.0, beta_weights=weights, beta_bounds_rad=bounds)
    assert np.array_equal(a, b)
    assert np.logical_and(a >= bounds[:, 0], a <= bounds[:, 1]).all()
    rms = np.sqrt(np.sum(weights**2 * np.rad2deg(a) ** 2, axis=1) / np.sum(weights**2))
    assert float(rms.max()) <= 3.0 + 1.0e-12


def _synthetic_support(policy: RootTipPolicy) -> np.ndarray:
    points = []
    zero_x = 1.2
    for u_index in range(1, 8):
        for rho_index in range(u_index, u_index + 4):
            for sector in range(policy.sector_count):
                phi = (sector + 0.5) * (0.5 * np.pi / policy.sector_count)
                u = (u_index + 0.5) * policy.axial_step_mm
                rho = (rho_index + 0.5) * policy.radial_step_mm
                points.append([zero_x - u / 1000.0, rho * np.cos(phi) / 1000.0, rho * np.sin(phi) / 1000.0])
    return np.asarray(points)


def test_common_support_builds_zero_anchored_sqrt_profile_and_target_only_pool() -> None:
    policy = RootTipPolicy(maximum_u_mm=30, maximum_rho_mm=80)
    xyz = _synthetic_support(policy)
    common = common_zero_tip_support(xyz, xyz.copy(), zero_x_m=1.2, policy=policy)
    intervals = zero_connected_radial_intervals(common, policy=policy)
    profile = fit_sqrt_u_tip_profile(intervals, erosion_mm=0.0)
    assert profile.iloc[0]["u_center_mm"] == 0.0
    assert profile.iloc[0]["inner_radius_mm"] == 0.0
    assert profile.iloc[0]["outer_radius_mm"] == 0.0
    assert np.allclose(profile["sqrt_u_mm"], np.sqrt(profile["u_center_mm"]))
    pool = sample_tip_volume(profile, power=8, seed=42, zero_x_m=1.2, pool_id="target_only")
    assert len(pool) == 256
    assert not any(name in pool for name in BETA_COLUMNS)
    assert not pool["proposal_beta_seed_eligible"].any()
    assert not pool["proposal_beta_label_eligible"].any()


def test_axis_core_collapses_undefined_phi() -> None:
    policy = RootTipPolicy(
        axial_step_mm=2,
        radial_step_mm=2,
        sector_count=16,
        axis_core_radius_mm=5,
    )
    a = np.array([[1.214, 0.001, 0.0]])
    b = np.array([[1.214, 0.0, 0.001]])
    common = common_zero_tip_support(a, b, zero_x_m=1.215498, policy=policy)
    assert len(common) == 1
    assert int(common.iloc[0]["required_sector_count"]) == 1


def test_overlap_requires_real_axial_and_radial_intersection() -> None:
    tip = pd.DataFrame({"u_center_mm": [0, 20, 30, 40], "inner_radius_mm": [0, 30, 50, 60], "outer_radius_mm": [0, 100, 130, 150]})
    tip["sqrt_u_mm"] = np.sqrt(tip["u_center_mm"])
    annulus = pd.DataFrame({"u_center_mm": [25, 35, 45], "inner_radius_mm": [60, 65, 70], "outer_radius_mm": [150, 155, 160]})
    metrics = profile_overlap_metrics(tip, annulus)
    assert metrics["axial_overlap_mm"] >= 10
    assert metrics["radial_overlap_minimum_mm"] > 0


def test_root_connectors_are_classified_per_row_not_by_role() -> None:
    targets = pd.DataFrame({"target_id": ["good", "bad"], "target_role": ["root_connector", "root_connector"]})
    labels = pd.DataFrame({"target_id": ["good", "bad"], **{name: [0.0, np.nan] for name in BETA_COLUMNS}, "fk_residual_mm": [0.2, np.inf]})
    bounds = np.tile(np.asarray([-0.2, 0.2]), (6, 1))
    result = classify_root_connectors(labels, targets, residual_maximum_mm=3.0, beta_bounds_rad=bounds)
    assert result.set_index("target_id").loc["good", "root_connector_class"] == "root_bridge_supervision"
    assert result.set_index("target_id").loc["bad", "root_connector_class"] == "rejected"
