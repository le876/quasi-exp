from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_standard_domain_v7.py"
    spec = importlib.util.spec_from_file_location("run_true_ellipse_standard_domain_v7", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_register_standard_domain_120mm_protocol() -> None:
    mod = _load_module()
    args = mod.parse_args([])
    report = mod.formal_protocol_report(args)

    assert args.family_id == "c0273_a100_py210_pz330_s0243"
    assert args.joint_domain_id == "standard_beta34_10deg_v1"
    assert args.target_radius_mm == 120.0
    assert args.max_candidates_per_angle == 8
    assert args.rescue_candidates_per_angle == 24
    assert args.rescue_kappa_threshold == 100.0
    assert args.lambda_margin == 0.01
    assert mod.parse_float_csv(args.radius_checkpoints_mm) == list(
        mod.engine.v7_standard_domain_protocol().formal_checkpoints_mm
    )
    assert report["formal_protocol_gate_pass"] is True
    assert report["joint_domain_fingerprint"] == mod.engine.registered_joint_domain(
        "standard_beta34_10deg_v1"
    ).fingerprint


def test_formal_protocol_rejects_domain_candidate_and_radius_ladder_changes() -> None:
    mod = _load_module()

    baseline = mod.formal_protocol_report(mod.parse_args([]))
    wrong_domain = mod.formal_protocol_report(mod.parse_args(["--joint-domain-id", "current_v6"]))
    wrong_candidates = mod.formal_protocol_report(
        mod.parse_args(["--rescue-candidates-per-angle", "8"])
    )
    wrong_ladder = mod.formal_protocol_report(
        mod.parse_args(["--radius-checkpoints-mm", "75,80,100,120"])
    )

    assert wrong_domain["formal_protocol_gate_pass"] is False
    assert wrong_candidates["formal_protocol_gate_pass"] is False
    assert wrong_ladder["formal_protocol_gate_pass"] is False
    assert wrong_domain["protocol_fingerprint"] != baseline["protocol_fingerprint"]
    assert wrong_candidates["protocol_fingerprint"] != baseline["protocol_fingerprint"]
    assert wrong_ladder["protocol_fingerprint"] != baseline["protocol_fingerprint"]


def test_rescue_admission_is_exploratory_and_does_not_replace_strict_gate() -> None:
    mod = _load_module()
    admitted = mod.rescue_admission_gate(
        {
            "finite": True,
            "in_domain": True,
            "success_ratio_le2mm": 0.80,
            "residual_max_mm": 20.0,
        }
    )
    rejected = mod.rescue_admission_gate(
        {
            "finite": True,
            "in_domain": True,
            "success_ratio_le2mm": 0.79,
            "residual_max_mm": 20.0,
        }
    )

    assert admitted is True
    assert rejected is False
    frontier = mod.engine.compute_radius_frontiers(
        pd.DataFrame(
            {
                "radius_mm": [100.0, 102.5, 105.0, 107.5],
                "strict_gate_pass": [True, True, False, False],
                "rescue_admission_pass": [True, True, True, True],
            }
        ),
        anchor_mm=100.0,
    )
    assert frontier["strict_geometry_rmax_mm"] == 102.5
    assert frontier["exploratory_rescue_rmax_mm"] == 107.5


def test_materialized_dataset_radii_include_checkpoints_and_1p25mm_training_anchors() -> None:
    mod = _load_module()

    assert mod.materialized_dataset_radii(105.0) == (
        75.0,
        80.0,
        82.5,
        85.0,
        87.5,
        90.0,
        92.5,
        95.0,
        97.5,
        100.0,
        101.25,
        102.5,
        103.75,
        105.0,
    )


def test_half_phase_targets_are_exactly_shifted_and_keep_registered_family() -> None:
    mod = _load_module()
    family = mod.v6_utils.FamilySpec(
        family_id="fixed-family",
        center_x_m=1.1,
        center_y_m=0.2,
        center_z_m=-0.3,
        phase_y_rad=0.4,
        phase_z_rad=1.2,
    )

    shifted = mod.generate_half_phase_targets(family, radius_mm=105.0, n_points=360)

    assert shifted["family_id"].eq("fixed-family").all()
    assert shifted["radius_mm"].eq(105.0).all()
    assert np.rad2deg(shifted["angle_rad"].iloc[0]) == pytest.approx(0.5)
    assert np.rad2deg(shifted["angle_rad"].iloc[-1]) == pytest.approx(359.5)
    expected_x0 = 1.1 + 0.105 * np.sin(np.deg2rad(0.5))
    assert shifted["x_target_m"].iloc[0] == pytest.approx(expected_x0)


def test_dynamic_split_holds_out_complete_radii_and_excludes_all_centerlines_from_training() -> None:
    mod = _load_module()
    rows = []
    for radius in (75.0, 97.5, 101.25, 105.0):
        for angle in range(2):
            for centerline in (False, True):
                rows.append(
                    {
                        "sample_id": f"{radius}:{angle}:{centerline}",
                        "family_id": "fixed-family",
                        "trajectory_id": f"fixed-family@{radius:g}",
                        "radius_mm": radius,
                        "angle_idx": angle,
                        "is_centerline": centerline,
                    }
                )
    dataset = pd.DataFrame(rows)

    assigned, report = mod.assign_dynamic_radius_splits(
        dataset,
        validation_radius_mm=97.5,
        test_radius_mm=105.0,
    )

    assert report["split_gate_pass"] is True
    assert set(assigned.loc[assigned["used_for_training"], "radius_mm"]) == {75.0, 101.25}
    assert not assigned.loc[assigned["used_for_training"], "is_centerline"].any()
    assert set(assigned.loc[assigned["split"].eq("validation"), "radius_mm"]) == {97.5}
    assert set(assigned.loc[assigned["split"].eq("test"), "radius_mm"]) == {105.0}


def test_formal_label_gate_requires_every_tube_row_success_and_balanced_margin() -> None:
    mod = _load_module()
    passing = {
        "rows": 9000,
        "expected_rows": 9000,
        "tube_success_ratio": 1.0,
        "surface_gate_pass": True,
        "joint_margin_gate_pass": True,
        "tube_gate_pass": True,
    }

    assert mod.formal_tube_label_gate(passing) is True
    for field, value in (
        ("tube_success_ratio", 0.999),
        ("surface_gate_pass", False),
        ("joint_margin_gate_pass", False),
        ("tube_gate_pass", False),
    ):
        failed = dict(passing)
        failed[field] = value
        assert mod.formal_tube_label_gate(failed) is False


def test_nonformal_presets_apply_registered_resolution_unless_explicitly_overridden() -> None:
    mod = _load_module()

    smoke = mod.parse_args(["--preset", "smoke"])
    pilot = mod.parse_args(["--preset", "pilot"])
    explicit = mod.parse_args(
        [
            "--preset",
            "smoke",
            "--final-points",
            "18",
            "--cut-indices",
            "0,9",
            "--max-opt-nfev",
            "3",
        ]
    )

    assert smoke.final_points == 36
    assert mod.parse_int_csv(smoke.cut_indices) == [0]
    assert smoke.max_opt_nfev == 8
    assert pilot.final_points == 72
    assert mod.parse_int_csv(pilot.cut_indices) == [0, 18, 36, 54]
    assert pilot.max_opt_nfev == 20
    assert explicit.final_points == 18
    assert mod.parse_int_csv(explicit.cut_indices) == [0, 9]
    assert explicit.max_opt_nfev == 3


def test_half_phase_predictor_uses_cyclic_neighbor_average() -> None:
    mod = _load_module()
    family = mod.v6_utils.FamilySpec(
        family_id="fixed-family",
        center_x_m=1.1,
        center_y_m=0.2,
        center_z_m=-0.3,
        phase_y_rad=0.4,
        phase_z_rad=1.2,
    )
    centerline = mod.generate_half_phase_targets(family, radius_mm=105.0, n_points=4)
    centerline["angle_rad"] = np.arange(4, dtype=float) * np.pi / 2.0
    for index, column in enumerate(mod.v6_utils.atlas.BETA_COLS):
        centerline[column] = np.asarray([0.0, 2.0, 4.0, 6.0]) + index
    targets = mod.generate_half_phase_targets(family, radius_mm=105.0, n_points=4)

    predictor = mod.build_half_phase_predictor(centerline, targets)

    assert predictor[mod.v6_utils.atlas.BETA_COLS[0]].tolist() == [1.0, 3.0, 5.0, 3.0]
    assert predictor["angle_idx"].tolist() == [0, 1, 2, 3]
    assert predictor["angle_rad"].tolist() == pytest.approx(targets["angle_rad"].tolist())
    assert predictor["radial_predictor_type"].eq("cyclic_half_phase_average").all()


def test_support_candidate_table_uses_only_strict_training_rows() -> None:
    mod = _load_module()
    rows = []
    for radius in (75.0, 97.5, 105.0):
        for angle in range(4):
            rows.append(
                {
                    "sample_id": f"{radius}:{angle}",
                    "family_id": "fixed-family",
                    "trajectory_id": f"fixed-family@{radius:g}",
                    "radius_mm": radius,
                    "angle_idx": angle,
                    "is_centerline": False,
                    "x_target_m": radius / 1000.0 + angle * 1.0e-4,
                    "y_target_m": 0.0,
                    "z_target_m": 0.0,
                }
            )
    dataset = pd.DataFrame(rows)

    table = mod.compute_holdout_support_candidates(
        dataset,
        candidate_test_radii_mm=[105.0],
        validation_gap_mm=7.5,
        support_radius_mm=15.0,
    )

    assert len(table) == 2
    assert set(table["role"]) == {"validation", "test"}
    assert table["test_radius_mm"].eq(105.0).all()
    assert table["validation_radius_mm"].eq(97.5).all()
    assert table["training_rows"].eq(4).all()
    assert table["held_out_radius_excluded"].all()


def test_dataset_gate_requires_tube_split_support_margin_and_challenge() -> None:
    mod = _load_module()
    passing = {
        "formal_protocol_gate_pass": True,
        "formal_tube_gate_pass": True,
        "completeness_gate_pass": True,
        "split_gate_pass": True,
        "branch_conflict_gate_pass": True,
        "joint_margin_gate_pass": True,
        "holdout_selection_gate_pass": True,
        "validation_support_gate_pass": True,
        "test_support_gate_pass": True,
        "challenge_gate_pass": True,
    }

    assert mod.formal_dataset_gate(passing) is True
    for field in passing:
        failed = dict(passing)
        failed[field] = False
        assert mod.formal_dataset_gate(failed) is False


def test_nonformal_parent_resampling_keeps_physical_phase_and_reindexes_grid() -> None:
    mod = _load_module()
    parent = pd.DataFrame(
        {
            "angle_idx": np.arange(360),
            "angle_rad": np.deg2rad(np.arange(360, dtype=float)),
            "marker": np.arange(360),
        }
    )

    sampled = mod.resample_parent_curve(parent, n_points=36)

    assert sampled["angle_idx"].tolist() == list(range(36))
    assert sampled["marker"].tolist() == list(range(0, 360, 10))
    assert np.rad2deg(sampled["angle_rad"].to_numpy()) == pytest.approx(
        np.arange(0, 360, 10)
    )


def test_tube_surface_initializer_is_vectorized_predictor_not_pointwise_ik(monkeypatch) -> None:
    mod = _load_module()
    lengths_m = np.full(31, 0.04, dtype=float)
    p_end_local_m = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    angle = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    beta = np.zeros((4, 6), dtype=float)
    beta[:, 4] = np.deg2rad(1.0 + 0.1 * np.sin(angle))
    beta[:, 5] = np.deg2rad(-1.0 + 0.1 * np.cos(angle))
    xyz = mod.v6_utils.atlas.fk_from_beta_batch(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
    )
    centerline = pd.DataFrame(
        {
            "angle_idx": np.arange(4),
            "angle_rad": angle,
            "x_target_m": xyz[:, 0],
            "y_target_m": xyz[:, 1],
            "z_target_m": xyz[:, 2],
            "x_m": xyz[:, 0],
            "y_m": xyz[:, 1],
            "z_m": xyz[:, 2],
        }
    )
    for index, column in enumerate(mod.v6_utils.atlas.BETA_COLS):
        centerline[column] = beta[:, index]
    targets = mod.v6_utils.atlas.make_normal_tube_targets(
        centerline, offsets_mm=(-1.0, 0.0, 1.0)
    )
    monkeypatch.setattr(
        mod.v5_expansion,
        "solve_tube_curve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pointwise IK must not be used by the surface initializer")
        ),
    )

    initial = mod._initial_tube_surface(
        centerline=centerline,
        tube_targets=targets,
        offsets_mm=(-1.0, 0.0, 1.0),
        bounds=mod._domain().bounds_rad,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
        max_nfev=5,
    )

    assert len(initial) == 4 * 9
    assert np.isfinite(initial[mod.v6_utils.atlas.BETA_COLS]).all().all()
    assert np.isfinite(initial["xyz_residual_mm"]).all()
    center = initial[
        np.isclose(initial["delta_n1_mm"], 0.0)
        & np.isclose(initial["delta_n2_mm"], 0.0)
    ].sort_values("angle_idx")
    np.testing.assert_allclose(center[mod.v6_utils.atlas.BETA_COLS], beta)


def test_challenge_centerline_is_extracted_from_final_tube_labels() -> None:
    mod = _load_module()
    rows = []
    for angle in range(3):
        for offset, is_centerline in (("center", True), ("outer", False)):
            row = {
                "angle_idx": angle,
                "tube_offset_id": offset,
                "is_centerline": is_centerline,
                "sample_id": f"{angle}:{offset}",
            }
            row.update(
                {
                    column: float(angle + index + (0.0 if is_centerline else 10.0))
                    for index, column in enumerate(mod.v6_utils.atlas.BETA_COLS)
                }
            )
            rows.append(row)
    tube = pd.DataFrame(rows)

    centerline = mod.extract_tube_centerline(tube, expected_points=3)

    assert centerline["angle_idx"].tolist() == [0, 1, 2]
    assert centerline["is_centerline"].all()
    assert centerline[mod.v6_utils.atlas.BETA_COLS[0]].tolist() == [0.0, 1.0, 2.0]


def test_surface_stage_schedule_stops_at_first_complete_gate_pass() -> None:
    mod = _load_module()
    stages = ({"name": "one"}, {"name": "two"}, {"name": "three"})
    calls = []

    def solve_stage(surface, stage):
        calls.append(stage["name"])
        return surface + 1, {"stage": stage["name"]}

    def assess_stage(surface, report, stage):
        return {
            **report,
            "surface_value": surface,
            "formal_tube_label_gate_pass": stage["name"] == "two",
        }

    surface, report, attempts = mod.run_surface_stage_schedule(
        initial_surface=0,
        stage_specs=stages,
        solve_stage=solve_stage,
        assess_stage=assess_stage,
    )

    assert calls == ["one", "two"]
    assert surface == 2
    assert report["stage"] == "two"
    assert [attempt["stage"] for attempt in attempts] == ["one", "two"]
