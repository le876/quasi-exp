from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "scripts" / "analysis" / "true_ellipse_family_v5_utils.py"
    spec = importlib.util.spec_from_file_location("true_ellipse_family_v5_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_family_targets_keeps_shared_geometry_across_radii() -> None:
    mod = _load_module()
    family = mod.FamilyParameters(
        family_id="main",
        center_x_m=1.1,
        center_y_m=0.2,
        center_z_m=-0.1,
        phase_y_rad=math.radians(120.0),
        phase_z_rad=math.radians(30.0),
    )

    small = mod.generate_family_targets(family, radius_mm=75.0, n_points=4)
    large = mod.generate_family_targets(family, radius_mm=100.0, n_points=4)

    assert small["family_id"].unique().tolist() == ["main"]
    assert small["candidate_id"].unique().tolist() == ["main"]
    assert small["ellipse_id"].unique().tolist() == ["main"]
    assert large["family_id"].unique().tolist() == ["main"]
    assert np.allclose(small[["center_x_m", "center_y_m", "center_z_m"]], large[["center_x_m", "center_y_m", "center_z_m"]])
    assert np.isclose(small.loc[0, "amp_z_mm"], 112.5)
    assert np.isclose(large.loc[0, "amp_z_mm"], 150.0)
    assert np.isclose(small.loc[1, "x_target_m"], 1.175)
    assert np.isclose(large.loc[1, "x_target_m"], 1.2)


def test_sobol_family_perturbations_are_deterministic_and_bounded() -> None:
    mod = _load_module()
    seeds = pd.DataFrame(
        [
            {
                "family_id": "seed",
                "center_x_m": 1.12,
                "center_y_m": 0.08,
                "center_z_m": -0.15,
                "phase_y_rad": math.radians(180.0),
                "phase_z_rad": math.radians(60.0),
            }
        ]
    )

    first = mod.sobol_family_perturbations(seeds, samples_per_seed=8, seed=17)
    second = mod.sobol_family_perturbations(seeds, samples_per_seed=8, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 8
    assert first["center_x_m"].between(1.100, 1.120).all()
    assert first["center_y_m"].between(0.065, 0.095).all()
    assert first["center_z_m"].between(-0.165, -0.135).all()
    assert np.rad2deg(first["phase_y_rad"]).between(165.0, 195.0).all()
    assert np.rad2deg(first["phase_z_rad"]).between(45.0, 75.0).all()
    assert first["candidate_id"].is_unique


def test_support_gate_and_connected_radius_do_not_skip_a_failed_anchor() -> None:
    mod = _load_module()
    assert mod.strict_support_gate({"nn_p95_mm": 5.0, "nn_max_mm": 8.0, "tube_count_p10": 32.0}) is True
    assert mod.strict_support_gate({"nn_p95_mm": 5.01, "nn_max_mm": 8.0, "tube_count_p10": 32.0}) is False

    status = pd.DataFrame(
        {
            "radius_mm": [75.0, 80.0, 82.5, 85.0, 87.5],
            "strict_gate_pass": [True, True, False, True, True],
        }
    )

    assert mod.connected_radius_max(status, gate_col="strict_gate_pass", anchor_mm=75.0) == 80.0


def test_pointwise_recovery_is_limited_to_small_contiguous_failures() -> None:
    mod = _load_module()
    failed_angles = [14, 15, 16, 17, 18, 19, 20]

    assert mod.should_retry_pointwise(
        target_count=72,
        failed_angle_indices=failed_angles,
        residual_max_mm=4.9,
    ) is True
    assert mod.should_retry_pointwise(
        target_count=72,
        failed_angle_indices=list(range(12)),
        residual_max_mm=4.9,
    ) is False
    assert mod.should_retry_pointwise(
        target_count=72,
        failed_angle_indices=[1, 10, 20, 30, 40],
        residual_max_mm=4.9,
    ) is False
    assert mod.should_retry_pointwise(
        target_count=72,
        failed_angle_indices=failed_angles,
        residual_max_mm=5.1,
    ) is False


def test_family_search_recomputes_exact_multi_radius_support() -> None:
    mod = _load_module()
    good = mod.FamilyParameters(
        family_id="good",
        center_x_m=1.0,
        center_y_m=0.0,
        center_z_m=0.0,
        phase_y_rad=math.radians(120.0),
        phase_z_rad=math.radians(30.0),
    )
    pool = np.vstack(
        [
            mod.generate_family_targets(good, radius_mm=75.0, n_points=12)[mod.TARGET_XYZ_COLS].to_numpy(),
            mod.generate_family_targets(good, radius_mm=100.0, n_points=12)[mod.TARGET_XYZ_COLS].to_numpy(),
        ]
    )
    candidates = pd.DataFrame(
        [
            {**mod.family_record(good), "candidate_id": "good", "nn_p95_mm": 999.0},
            {
                **mod.family_record(good),
                "candidate_id": "bad",
                "family_id": "bad",
                "center_x_m": 1.04,
                "nn_p95_mm": -1.0,
            },
        ]
    )

    scored, by_radius = mod.score_family_candidates(
        candidates,
        pool_xyz=pool,
        radii_mm=[75.0, 100.0],
        n_points=12,
    )

    assert scored.iloc[0]["candidate_id"] == "good"
    assert np.isclose(scored.iloc[0]["worst_nn_p95_mm"], 0.0)
    assert scored.iloc[1]["worst_nn_p95_mm"] > 20.0
    assert set(by_radius["radius_mm"]) == {75.0, 100.0}
    assert len(by_radius) == 4
