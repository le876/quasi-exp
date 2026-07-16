from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_radial_bundle_v6.py"
    spec = importlib.util.spec_from_file_location("run_true_ellipse_radial_bundle_v6", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_register_the_fixed_100mm_radial_bundle_protocol() -> None:
    mod = _load_module()
    args = mod.parse_args([])
    protocol = mod.formal_protocol_report(args)

    assert args.family_id == "c0273_a100_py210_pz330_s0243"
    assert args.previous_radius_mm == 75.0
    assert args.start_radius_mm == 80.0
    assert args.target_radius_mm == 100.0
    assert mod.parse_float_csv(args.radius_checkpoints_mm) == [82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0]
    assert mod.parse_int_csv(args.cut_indices) == [0, 90, 180, 270]
    assert mod.parse_name_csv(args.anchor_schedules) == ["conservative", "balanced", "loose"]
    assert args.family_search_sobol_samples == 8192
    assert args.family_search_top_support == 64
    assert args.family_search_top_pointwise == 16
    assert args.family_search_top_formal == 8
    assert protocol["formal_protocol_gate_pass"] is True
    assert mod.parse_phases("all") == ["audit", "radial", "tube", "dataset", "summary"]


def test_formal_protocol_rejects_cut_anchor_radius_and_candidate_budget_relaxation() -> None:
    mod = _load_module()

    wrong_cut = mod.formal_protocol_report(mod.parse_args(["--cut-indices", "0,90"]))
    wrong_anchor = mod.formal_protocol_report(mod.parse_args(["--anchor-schedules", "balanced"]))
    wrong_radius = mod.formal_protocol_report(mod.parse_args(["--target-radius-mm", "87.5"]))
    wrong_candidates = mod.formal_protocol_report(mod.parse_args(["--max-candidates-per-angle", "4"]))

    assert wrong_cut["formal_protocol_gate_pass"] is False
    assert wrong_anchor["formal_protocol_gate_pass"] is False
    assert wrong_radius["formal_protocol_gate_pass"] is False
    assert wrong_candidates["formal_protocol_gate_pass"] is False


def test_radius_bundle_gate_requires_all_cut_predictor_runs_cut_invariance_and_exact_repeat() -> None:
    mod = _load_module()
    jobs = [
        {
            "cut_idx": cut,
            "radial_predictor_type": predictor,
            "centerline_gate_pass": True,
            "selected": True,
        }
        for cut in (0, 90, 180, 270)
        for predictor in ("parent_copy", "radial_secant")
    ]
    complete = mod.aggregate_radius_bundle_gate(
        geometry_report={"target_geometry_gate_pass": True},
        job_reports=jobs,
        required_cuts=(0, 90, 180, 270),
        required_predictors=("parent_copy", "radial_secant"),
        cut_report={"cut_invariance_gate_pass": True},
        repeatability_report={"deterministic_exact_gate_pass": True},
    )
    missing = mod.aggregate_radius_bundle_gate(
        geometry_report={"target_geometry_gate_pass": True},
        job_reports=jobs[:-1],
        required_cuts=(0, 90, 180, 270),
        required_predictors=("parent_copy", "radial_secant"),
        cut_report={"cut_invariance_gate_pass": True},
        repeatability_report={"deterministic_exact_gate_pass": True},
    )

    assert complete["radial_bundle_gate_pass"] is True
    assert complete["required_job_count"] == 8
    assert missing["radial_bundle_gate_pass"] is False
    assert missing["missing_jobs"] == ["270:radial_secant"]


def test_runner_defaults_to_v6_domain_and_can_resolve_the_registered_v7_domain() -> None:
    mod = _load_module()

    current = mod._resolved_joint_domain(mod.parse_args([]))
    standard = mod._resolved_joint_domain(
        SimpleNamespace(joint_domain_id="standard_beta34_10deg_v1")
    )

    assert current.domain_id == "current_v6"
    assert standard.domain_id == "standard_beta34_10deg_v1"
    assert np.rad2deg(current.bounds_rad[2, 1]) == 5.0
    assert np.rad2deg(standard.bounds_rad[2, 1]) == 10.0


def test_radius_bundle_gate_honors_stricter_job_gate_when_present() -> None:
    mod = _load_module()
    jobs = [
        {
            "cut_idx": 0,
            "radial_predictor_type": "parent_copy",
            "selected": True,
            "centerline_gate_pass": True,
            "job_gate_pass": False,
        }
    ]

    decision = mod.aggregate_radius_bundle_gate(
        geometry_report={"target_geometry_gate_pass": True},
        job_reports=jobs,
        required_cuts=(0,),
        required_predictors=("parent_copy",),
        cut_report={"cut_invariance_gate_pass": True},
        repeatability_report={"deterministic_exact_gate_pass": True},
    )

    assert decision["all_job_centerline_gates_pass"] is False
    assert decision["radial_bundle_gate_pass"] is False


def test_rescue_cache_fingerprint_binds_predictor_but_is_cut_independent() -> None:
    mod = _load_module()
    targets = pd.DataFrame(
        {
            "angle_idx": [0, 1],
            "x_target_m": [1.0, 2.0],
            "y_target_m": [0.0, 0.0],
            "z_target_m": [0.0, 0.0],
        }
    )
    predictor = targets[["angle_idx"]].copy()
    for column in mod.v6.atlas.BETA_COLS:
        predictor[column] = 0.0
    domain = mod.registered_joint_domain("standard_beta34_10deg_v1")
    lengths_m = np.linspace(0.01, 0.31, 31)
    p_end_local_m = np.asarray([0.0, 0.0, 0.1, 1.0])

    baseline = mod._rescue_cache_fingerprint(
        targets=targets,
        predictor=predictor,
        domain=domain,
        rescue_budget=24,
        max_ik_nfev=200,
        residual_limit_mm=2.0,
        cluster_threshold_deg=0.25,
        seed=20260716,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
    )
    repeat = mod._rescue_cache_fingerprint(
        targets=targets,
        predictor=predictor.copy(),
        domain=domain,
        rescue_budget=24,
        max_ik_nfev=200,
        residual_limit_mm=2.0,
        cluster_threshold_deg=0.25,
        seed=20260716,
        lengths_m=lengths_m.copy(),
        p_end_local_m=p_end_local_m.copy(),
        theta_sign=-1.0,
    )
    changed = predictor.copy()
    changed.loc[0, mod.v6.atlas.BETA_COLS[0]] = 1.0e-3
    changed_fingerprint = mod._rescue_cache_fingerprint(
        targets=targets,
        predictor=changed,
        domain=domain,
        rescue_budget=24,
        max_ik_nfev=200,
        residual_limit_mm=2.0,
        cluster_threshold_deg=0.25,
        seed=20260716,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
    )
    changed_kinematics_fingerprint = mod._rescue_cache_fingerprint(
        targets=targets,
        predictor=predictor,
        domain=domain,
        rescue_budget=24,
        max_ik_nfev=200,
        residual_limit_mm=2.0,
        cluster_threshold_deg=0.25,
        seed=20260716,
        lengths_m=lengths_m + 1.0e-6,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
    )

    assert baseline == repeat
    assert baseline != changed_fingerprint
    assert baseline != changed_kinematics_fingerprint


def test_v6_defaults_preserve_full_failed_job_diagnostics_and_disable_shared_rescue_cache() -> None:
    mod = _load_module()

    args = mod.parse_args([])

    assert args.stop_after_first_failed_job is False
    assert args.share_rescue_cache_across_cuts is False


def test_required_job_failure_short_circuits_an_unrecoverable_radius_bundle() -> None:
    mod = _load_module()

    assert mod._required_job_failed({"selected": False, "job_gate_pass": False}) is True
    assert mod._required_job_failed({"selected": True, "job_gate_pass": False}) is True
    assert mod._required_job_failed({"selected": True, "job_gate_pass": True}) is False


def test_path_margin_decision_is_optional_for_v6_and_required_for_v7() -> None:
    mod = _load_module()
    frame = pd.DataFrame(
        {
            column: np.zeros(8, dtype=float)
            for column in mod.v6.atlas.BETA_COLS
        }
    )
    frame["beta3_rad"] = np.deg2rad(9.99)

    optional = mod._joint_margin_decision(
        frame,
        SimpleNamespace(joint_domain_id="standard_beta34_10deg_v1", require_joint_margin_gate=False),
    )
    required = mod._joint_margin_decision(
        frame,
        SimpleNamespace(joint_domain_id="standard_beta34_10deg_v1", require_joint_margin_gate=True),
    )

    assert optional["raw_joint_margin_gate_pass"] is False
    assert optional["job_margin_gate_pass"] is True
    assert required["raw_joint_margin_gate_pass"] is False
    assert required["job_margin_gate_pass"] is False


def test_tube_annotation_materializes_v6_provenance_and_unique_sample_ids() -> None:
    mod = _load_module()
    rows = []
    for angle_idx in range(2):
        for offset_id in ("n1_0_n2_0", "n1_5_n2_5"):
            row = {
                "angle_idx": angle_idx,
                "tube_offset_id": offset_id,
                "is_centerline": offset_id == "n1_0_n2_0",
                "x_target_m": float(angle_idx),
                "y_target_m": 0.0,
                "z_target_m": 0.0,
            }
            for column in mod.v6.atlas.BETA_COLS:
                row[column] = 0.0
            rows.append(row)
    tube = pd.DataFrame(rows)

    annotated = mod.annotate_tube_rows(
        tube,
        family_id="fixed-family",
        radius_mm=100.0,
        parent_radius_mm=99.0,
        radial_predictor_type="radial_secant",
        branch_hash="abc123",
    )

    required = {
        "family_id",
        "branch_id",
        "trajectory_id",
        "radius_mm",
        "parent_radius_mm",
        "angle_idx",
        "tube_offset_id",
        "solver_strategy_version",
        "radial_predictor_type",
        "branch_hash",
        "sample_id",
    }
    assert required.issubset(annotated.columns)
    assert annotated["sample_id"].is_unique
    assert annotated["family_id"].eq("fixed-family").all()
    assert np.isclose(annotated["radius_mm"], 100.0).all()


def test_v5_pointwise_certificate_keeps_its_72_point_resolution_separate_from_v6_360() -> None:
    mod = _load_module()
    family = mod.v6.FamilySpec(
        family_id="fixed-family",
        center_x_m=1.1,
        center_y_m=0.1,
        center_z_m=-0.1,
        phase_y_rad=2.0,
        phase_z_rad=0.5,
    )
    source_targets = mod.v6.generate_radius_targets(family, radius_mm=100.0, n_points=72)
    candidates = source_targets[["angle_idx"]].copy()
    candidates["xyz_residual_mm"] = 0.1

    report = mod.recompute_pointwise_certificate(
        source_targets,
        candidates,
        family=family,
        radius_mm=100.0,
    )

    assert report["certificate_resolution_points"] == 72
    assert report["pointwise_gate_pass"] is True
    assert report["registered_target_match_gate_pass"] is True


def test_formal_360_point_tube_uses_process_workers_instead_of_thread_serialization() -> None:
    mod = _load_module()

    assert mod.tube_worker_mode(n_points=360, workers=4) == "subprocess"
    assert mod.tube_worker_mode(n_points=72, workers=4) == "thread"
    assert mod.tube_worker_mode(n_points=360, workers=1) == "serial"


def test_tube_outer_shell_joint_refinement_uses_same_n1_inner_n2_parent() -> None:
    mod = _load_module()

    jobs = mod.tube_outer_shell_refinement_jobs((-5.0, -2.5, 0.0, 2.5, 5.0))

    assert len(jobs) == 10
    assert ((0.0, -5.0), (0.0, -2.5)) in jobs
    assert ((0.0, 5.0), (0.0, 2.5)) in jobs
    assert ((5.0, -5.0), (5.0, -2.5)) in jobs
    assert ((-5.0, 5.0), (-5.0, 2.5)) in jobs
    assert all(pair[0] == parent[0] for pair, parent in jobs)
