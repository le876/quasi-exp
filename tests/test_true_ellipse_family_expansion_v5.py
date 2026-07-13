from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    utils_path = REPO_ROOT / "scripts" / "analysis" / "true_ellipse_family_v5_utils.py"
    utils_spec = importlib.util.spec_from_file_location("true_ellipse_family_v5_utils", utils_path)
    assert utils_spec is not None and utils_spec.loader is not None
    utils = importlib.util.module_from_spec(utils_spec)
    sys.modules[utils_spec.name] = utils
    utils_spec.loader.exec_module(utils)

    path = REPO_ROOT / "scripts" / "analysis" / "run_true_ellipse_family_expansion_v5.py"
    spec = importlib.util.spec_from_file_location("run_true_ellipse_family_expansion_v5", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pointwise_summary_uses_best_candidate_per_angle_and_requires_every_target() -> None:
    mod = _load_module()
    candidates = pd.DataFrame(
        {
            "angle_idx": [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "xyz_residual_mm": [3.0, 0.1, 2.1, 1.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.2],
        }
    )

    report = mod.summarize_pointwise_candidates(candidates, target_count=10)

    assert report["covered_target_count"] == 10
    assert report["success_target_count"] == 9
    assert report["failed_angle_indices"] == [9]
    assert report["pointwise_gate_pass"] is False
    assert report["retry_allowed"] is True

    missing = mod.summarize_pointwise_candidates(candidates[candidates["angle_idx"] != 9], target_count=10)
    assert missing["covered_target_count"] == 9
    assert missing["failed_angle_indices"] == [9]
    assert missing["pointwise_gate_pass"] is False


def test_goal_report_keeps_primary_and_stretch_results_separate() -> None:
    mod = _load_module()
    rows = pd.DataFrame(
        {
            "radius_mm": [75.0, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0],
            "strict_gate_pass": [True, True, True, True, True, True, True, True, True, False],
            "trajectory_generalization_gate_pass": [True, True, True, True, True, True, True, False, False, False],
            "limiting_factor": ["none"] * 9 + ["pointwise_ik"],
        }
    )

    report = mod.build_goal_report(rows, primary_radius_mm=87.5, stretch_radius_mm=100.0)

    assert report["primary_goal_pass"] is True
    assert report["stretch_radius_pass"] is False
    assert report["strict_supported_rmax_mm"] == 97.5
    assert report["trajectory_generalized_rmax_mm"] == 92.5
    assert report["stretch_limiting_factor"] == "pointwise_ik"


def test_dataframe_to_markdown_does_not_require_pandas_tabulate(monkeypatch) -> None:
    mod = _load_module()

    def unexpected_optional_dependency(*_args, **_kwargs):
        raise AssertionError("pandas.to_markdown must not be called")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", unexpected_optional_dependency)
    rendered = mod.dataframe_to_markdown(pd.DataFrame({"radius_mm": [85.0], "gate_pass": [True]}))

    assert "| radius_mm | gate_pass |" in rendered
    assert "| 85.0 | True |" in rendered


def test_cli_defaults_cover_full_v5_pipeline() -> None:
    mod = _load_module()
    args = mod.parse_args([])

    assert args.primary_radius_mm == 87.5
    assert args.stretch_radius_mm == 100.0
    assert args.radius_anchors_mm == "75,80,82.5,85,87.5,90,92.5,95,97.5,100"
    assert args.max_branch_families == 5
    assert mod.parse_phases("all") == ["audit", "search", "pointwise", "branch", "tube", "dataset", "summary"]


def test_family_seeds_include_selected_v3_and_deduplicated_e100_geometry() -> None:
    mod = _load_module()
    v3 = pd.DataFrame(
        [
            {
                "candidate_id": "selected_e75",
                "center_x_m": 1.10,
                "center_y_m": 0.10,
                "center_z_m": -0.10,
                "phase_y_rad": math.radians(120.0),
                "phase_z_rad": math.radians(30.0),
            }
        ]
    )
    v2 = pd.DataFrame(
        [
            {
                "candidate_id": "e75_ignore",
                "amp_xy_mm": 75.0,
                "center_x_m": 1.0,
                "center_y_m": 0.0,
                "center_z_m": 0.0,
                "phase_y_rad": 0.0,
                "phase_z_rad": 1.0,
            },
            {
                "candidate_id": "e100_a",
                "amp_xy_mm": 100.0,
                "center_x_m": 1.12,
                "center_y_m": 0.08,
                "center_z_m": -0.15,
                "phase_y_rad": math.radians(180.0),
                "phase_z_rad": math.radians(60.0),
            },
            {
                "candidate_id": "e100_duplicate_geometry",
                "amp_xy_mm": 100.0,
                "center_x_m": 1.12,
                "center_y_m": 0.08,
                "center_z_m": -0.15,
                "phase_y_rad": math.radians(180.0),
                "phase_z_rad": math.radians(60.0),
            },
        ]
    )

    seeds = mod.family_seeds_from_sources(v3, v2, stretch_radius_mm=100.0)

    assert len(seeds) == 2
    assert seeds["source_kind"].tolist() == ["v3_selected", "v2_stretch"]
    assert seeds["family_id"].tolist() == ["v3_selected", "e100_a"]


def test_phase_search_writes_primary_and_stretch_rankings_from_exact_pool(tmp_path: Path) -> None:
    mod = _load_module()
    v2 = tmp_path / "v2"
    v3 = tmp_path / "v3"
    out = tmp_path / "v5"
    (v2 / "01_reachability_pool").mkdir(parents=True)
    (v2 / "02_ellipse_family_search").mkdir(parents=True)
    (v3 / "05_robustness").mkdir(parents=True)
    (v3 / "06_local_tube").mkdir(parents=True)

    family = {
        "candidate_id": "good",
        "family_id": "good",
        "center_x_m": 1.0,
        "center_y_m": 0.0,
        "center_z_m": 0.0,
        "phase_y_rad": math.radians(120.0),
        "phase_z_rad": math.radians(30.0),
    }
    selected = pd.DataFrame([family])
    selected.to_parquet(v3 / "05_robustness" / "selected_centerline_360.parquet", index=False)
    selected.to_parquet(v3 / "06_local_tube" / "tube_small.parquet", index=False)
    pd.DataFrame([{**family, "candidate_id": "e100", "amp_xy_mm": 100.0}]).to_csv(
        v2 / "02_ellipse_family_search" / "top_candidates_by_radius.csv", index=False
    )

    utils = sys.modules["true_ellipse_family_v5_utils"]
    params = utils.family_from_mapping(family)
    pool_xyz = np.vstack(
        [
            utils.generate_family_targets(params, radius_mm=75.0, n_points=12)[utils.TARGET_XYZ_COLS].to_numpy(),
            utils.generate_family_targets(params, radius_mm=100.0, n_points=12)[utils.TARGET_XYZ_COLS].to_numpy(),
        ]
    )
    pd.DataFrame(pool_xyz, columns=["x_m", "y_m", "z_m"]).to_parquet(
        v2 / "01_reachability_pool" / "reachability_pool_merged.parquet", index=False
    )
    args = SimpleNamespace(
        v2_dir=v2,
        v3_dir=v3,
        out_dir=out,
        primary_radius_mm=75.0,
        stretch_radius_mm=100.0,
        radius_anchors_mm="75,100",
        family_sobol_samples=8,
        coarse_points=12,
        seed=17,
        skip_existing=False,
        phases="audit,search",
    )

    payload = mod.run(args)
    report = payload["results"]["search"]

    assert report["primary_selected_candidate_id"] == "v3_selected"
    assert report["stretch_selected_candidate_id"] == "v3_selected"
    assert payload["phases"] == ["audit", "search"]
    assert (out / "01_family_search" / "primary_family_ranking.csv").exists()
    assert (out / "01_family_search" / "stretch_family_ranking.csv").exists()


def test_pointwise_solver_recovers_exact_pool_targets() -> None:
    mod = _load_module()
    atlas = mod.atlas
    lengths_m = np.full(31, 0.04, dtype=float)
    p_end_local_m = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    beta = np.deg2rad(
        np.asarray(
            [
                [0.2, -0.2, 0.4, -0.4, 2.0, -2.0],
                [0.3, -0.1, 0.5, -0.2, 2.5, -1.5],
                [0.1, -0.3, 0.2, -0.5, 1.5, -2.5],
            ]
        )
    )
    xyz = atlas.fk_from_beta_batch(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
    )
    targets = pd.DataFrame(
        {
            "angle_idx": [0, 1, 2],
            "angle_rad": [0.0, 1.0, 2.0],
            "x_target_m": xyz[:, 0],
            "y_target_m": xyz[:, 1],
            "z_target_m": xyz[:, 2],
        }
    )
    pool = pd.DataFrame(xyz, columns=atlas.XYZ_COLS)
    for idx, column in enumerate(atlas.BETA_COLS):
        pool[column] = beta[:, idx]

    candidates, best, report = mod.solve_pointwise_targets(
        targets,
        pool=pool,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
        seed_budget=1,
        max_nfev=5,
        workers=1,
    )

    assert len(candidates) == 3
    assert len(best) == 3
    assert report["pointwise_gate_pass"] is True
    assert report["residual_max_mm"] < 1.0e-8


def test_pointwise_solver_stops_seed_search_after_reachability_certificate(monkeypatch) -> None:
    mod = _load_module()
    targets = pd.DataFrame(
        [{"angle_idx": 0, "angle_rad": 0.0, "x_target_m": 1.0, "y_target_m": 0.0, "z_target_m": 0.0}]
    )
    pool = pd.DataFrame(
        {
            "x_m": [1.0, 1.001, 1.002],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            **{column: [0.0, 0.1, 0.2] for column in mod.atlas.BETA_COLS},
        }
    )
    calls = []

    def fake_solve(target_xyz, *, init_betas, **_kwargs):
        calls.append(len(init_betas))
        return [
            SimpleNamespace(
                seed_rank=0,
                nfev=1,
                residual_mm=0.1,
                beta_rad=np.asarray(init_betas[0], dtype=float),
                xyz_m=np.asarray(target_xyz, dtype=float),
            )
        ]

    monkeypatch.setattr(mod.atlas, "solve_beta_ik_many", fake_solve)

    candidates, _best, report = mod.solve_pointwise_targets(
        targets,
        pool=pool,
        lengths_m=np.ones(31),
        p_end_local_m=np.ones(4),
        theta_sign=-1.0,
        seed_budget=3,
        max_nfev=5,
        workers=1,
    )

    assert calls == [1]
    assert len(candidates) == 1
    assert report["pointwise_gate_pass"] is True


def test_pointwise_subprocess_pool_matches_exact_robot_targets(tmp_path: Path) -> None:
    mod = _load_module()
    args = mod.parse_args([])
    lengths_m, p_end_local_m, theta_sign = mod._load_robot(args)
    beta = np.deg2rad(
        np.asarray(
            [
                [0.2, -0.2, 0.4, -0.4, 2.0, -2.0],
                [0.3, -0.1, 0.5, -0.2, 2.5, -1.5],
                [0.1, -0.3, 0.2, -0.5, 1.5, -2.5],
            ]
        )
    )
    xyz = mod.atlas.fk_from_beta_batch(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    targets = pd.DataFrame(
        {
            "angle_idx": np.arange(3),
            "angle_rad": [0.0, 1.0, 2.0],
            "x_target_m": xyz[:, 0],
            "y_target_m": xyz[:, 1],
            "z_target_m": xyz[:, 2],
        }
    )
    pool = pd.DataFrame(xyz, columns=mod.atlas.XYZ_COLS)
    for idx, column in enumerate(mod.atlas.BETA_COLS):
        pool[column] = beta[:, idx]

    candidates, best, report = mod.solve_pointwise_targets_subprocess(
        targets,
        pool=pool,
        robot_config=args.robot_config,
        seed_budget=1,
        max_nfev=5,
        workers=2,
        work_dir=tmp_path / "pointwise_workers",
        skip_existing=False,
    )

    assert len(candidates) == 3
    assert len(best) == 3
    assert report["pointwise_gate_pass"] is True
    assert report["execution_backend"] == "subprocess"


def test_pointwise_family_selection_keeps_v3_baseline_and_both_rankings() -> None:
    mod = _load_module()
    columns = [
        "candidate_id",
        "family_id",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "phase_y_rad",
        "phase_z_rad",
    ]

    def row(name: str, x: float) -> dict[str, float | str]:
        return {
            "candidate_id": name,
            "family_id": name,
            "center_x_m": x,
            "center_y_m": 0.0,
            "center_z_m": 0.0,
            "phase_y_rad": 1.0,
            "phase_z_rad": 2.0,
        }

    primary = pd.DataFrame([row("p1", 1.0), row("p2", 1.1), row("p3", 1.2)], columns=columns)
    stretch = pd.DataFrame([row("s1", 1.3), row("s2", 1.4), row("s3", 1.5)], columns=columns)
    seeds = pd.DataFrame([row("v3_selected", 0.9)], columns=columns)

    selected = mod.select_pointwise_families(primary, stretch, seeds, max_families=5)

    assert selected["candidate_id"].tolist() == ["v3_selected", "p1", "p2", "s1", "s2"]


def test_periodic_beta_resampling_preserves_samples_and_wraps_smoothly() -> None:
    mod = _load_module()
    angle = np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)
    values = np.column_stack(
        [
            np.sin(angle),
            np.cos(angle),
            np.sin(angle),
            np.cos(angle),
            np.sin(angle),
            np.cos(angle),
        ]
    )
    source = pd.DataFrame({"angle_idx": np.arange(4), "angle_rad": angle})
    for idx, column in enumerate(mod.atlas.BETA_COLS):
        source[column] = values[:, idx]

    resampled = mod.periodic_resample_beta(source, target_count=8)

    assert resampled.shape == (8, 6)
    assert np.allclose(resampled[::2], values)
    assert np.allclose(resampled[-1], 0.5 * (values[-1] + values[0]))


def test_branch_robustness_gate_requires_centerline_forward_reverse_and_repeatability() -> None:
    mod = _load_module()
    centerline = {
        "residual_p95_mm": 2.0,
        "residual_max_mm": 5.0,
        "delta_beta_p95_deg": 1.0,
        "delta_beta_max_deg": 2.0,
        "delta2_beta_p95_deg": 0.25,
        "seam_beta_rms_deg": 0.75,
        "sigma3_p05_m": 0.0015,
        "kappa_p95": 150.0,
    }
    forward_reverse = {"branch_diff_p95_deg": 1.0, "reproducible": True}
    repeatability = {"branch_diff_p95_deg": 0.0, "reproducible": True}

    report = mod.evaluate_branch_robustness(
        centerline_report=centerline,
        forward_reverse=forward_reverse,
        repeatability=repeatability,
    )

    assert report["centerline_gate_pass"] is True
    assert report["branch_robustness_gate_pass"] is True

    failed = mod.evaluate_branch_robustness(
        centerline_report=centerline,
        forward_reverse={"branch_diff_p95_deg": 1.01, "reproducible": False},
        repeatability=repeatability,
    )
    assert failed["branch_robustness_gate_pass"] is False


def test_tube_gate_uses_v3_hard_thresholds_without_relaxation() -> None:
    mod = _load_module()
    metrics = {
        "normal_grid_size": 25,
        "target_success_ratio": 0.99,
        "normal_grid_coverage_ratio": 0.95,
        "residual_p95_mm": 1.5,
        "residual_max_mm": 3.0,
        "tube10_beta_rms_p95_deg": 1.0,
        "multi_branch_ratio": 0.0,
    }

    assert mod.tube_gate_pass(metrics) is True
    assert mod.tube_gate_pass({**metrics, "residual_p95_mm": 1.5001}) is False
    assert mod.tube_gate_pass({**metrics, "multi_branch_ratio": 0.001}) is False
    assert mod.tube_gate_pass({**metrics, "normal_grid_size": 24}) is False


def test_branch_family_selection_reserves_primary_and_stretch_contenders() -> None:
    mod = _load_module()
    summary = pd.DataFrame(
        [
            {"candidate_id": "p1", "radius_mm": 87.5, "pointwise_gate_pass": True, "residual_p95_mm": 0.10},
            {"candidate_id": "p2", "radius_mm": 87.5, "pointwise_gate_pass": True, "residual_p95_mm": 0.20},
            {"candidate_id": "s1", "radius_mm": 87.5, "pointwise_gate_pass": True, "residual_p95_mm": 0.30},
            {"candidate_id": "s1", "radius_mm": 100.0, "pointwise_gate_pass": True, "residual_p95_mm": 0.40},
            {"candidate_id": "bad", "radius_mm": 100.0, "pointwise_gate_pass": False, "residual_p95_mm": 0.01},
        ]
    )

    selected = mod.select_branch_families(
        summary,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        max_families=3,
    )

    assert selected == ["s1", "p1", "p2"]
    preferred = mod.select_branch_families(
        summary,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        max_families=3,
        primary_preference=["p2", "p1"],
        stretch_preference=["s1"],
    )
    assert preferred == ["s1", "p2", "p1"]
    assert mod.family_covers_both_goals([75.0, 87.5, 100.0], primary_radius_mm=87.5, stretch_radius_mm=100.0)
    assert not mod.family_covers_both_goals([75.0, 87.5], primary_radius_mm=87.5, stretch_radius_mm=100.0)


def test_branch_family_selection_keeps_second_stretch_contender_with_three_slots() -> None:
    mod = _load_module()
    summary = pd.DataFrame(
        [
            {"candidate_id": "primary_1", "radius_mm": 87.5, "pointwise_gate_pass": True, "residual_p95_mm": 0.10},
            {"candidate_id": "primary_2", "radius_mm": 87.5, "pointwise_gate_pass": True, "residual_p95_mm": 0.20},
            {"candidate_id": "stretch_1", "radius_mm": 87.5, "pointwise_gate_pass": True, "residual_p95_mm": 0.30},
            {"candidate_id": "stretch_1", "radius_mm": 100.0, "pointwise_gate_pass": True, "residual_p95_mm": 0.40},
            {"candidate_id": "stretch_2", "radius_mm": 87.5, "pointwise_gate_pass": True, "residual_p95_mm": 0.50},
            {"candidate_id": "stretch_2", "radius_mm": 100.0, "pointwise_gate_pass": True, "residual_p95_mm": 0.60},
        ]
    )

    selected = mod.select_branch_families(
        summary,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        max_families=3,
        primary_preference=["primary_1", "primary_2"],
        stretch_preference=["stretch_1", "stretch_2"],
    )

    assert selected == ["stretch_1", "primary_1", "stretch_2"]


def test_phase_branch_writes_only_a_robust_centerline(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    out = tmp_path / "v5"
    pointwise = out / "02_pointwise"
    radius_dir = pointwise / "c1" / "r075p00"
    radius_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "family_id": "c1",
                "center_x_m": 1.0,
                "center_y_m": 0.0,
                "center_z_m": 0.0,
                "phase_y_rad": 1.0,
                "phase_z_rad": 2.0,
            }
        ]
    ).to_csv(pointwise / "selected_families.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "family_id": "c1",
                "radius_mm": 75.0,
                "executed": True,
                "pointwise_gate_pass": True,
                "residual_p95_mm": 0.1,
            }
        ]
    ).to_csv(pointwise / "pointwise_radius_summary.csv", index=False)
    path = pd.DataFrame({"angle_idx": np.arange(4), "angle_rad": np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)})
    for column in mod.atlas.BETA_COLS:
        path[column] = 0.0
    path.to_parquet(radius_dir / "best_pointwise_path.parquet", index=False)

    passing = {
        "residual_p95_mm": 0.1,
        "residual_max_mm": 0.2,
        "delta_beta_p95_deg": 0.1,
        "delta_beta_max_deg": 0.2,
        "delta2_beta_p95_deg": 0.01,
        "seam_beta_rms_deg": 0.1,
        "sigma3_p05_m": 0.01,
        "kappa_p95": 10.0,
    }

    def solved_frame(targets, initial_beta=None, **_kwargs):
        frame = targets.copy().reset_index(drop=True)
        beta = np.zeros((len(frame), 6), dtype=float) if initial_beta is None else np.asarray(initial_beta, dtype=float)
        for idx, column in enumerate(mod.atlas.BETA_COLS):
            frame[column] = beta[:, idx]
        for target_col, xyz_col in zip(mod.atlas.TARGET_XYZ_COLS, mod.atlas.XYZ_COLS):
            frame[xyz_col] = frame[target_col]
        return frame

    def fake_optimize(*, targets, initial_beta, **_kwargs):
        return solved_frame(targets, initial_beta), dict(passing)

    def fake_continuation(*, targets, **_kwargs):
        return solved_frame(targets), dict(passing)

    monkeypatch.setattr(mod, "_load_robot", lambda _args: (np.ones(31), np.ones(4), -1.0))
    monkeypatch.setattr(mod.atlas, "optimize_cyclic_trajectory", fake_optimize)
    monkeypatch.setattr(mod.atlas, "continuation_lift", fake_continuation)
    monkeypatch.setattr(
        mod.atlas,
        "branch_reproducibility_report",
        lambda *_args, **_kwargs: {"branch_diff_p95_deg": 0.0, "reproducible": True},
    )
    args = SimpleNamespace(
        out_dir=out,
        radius_anchors_mm="75",
        primary_radius_mm=75.0,
        stretch_radius_mm=100.0,
        max_branch_families=1,
        coarse_points=4,
        final_points=8,
        max_opt_nfev=2,
        max_ik_nfev=2,
        skip_existing=False,
    )

    report = mod.phase_branch(args)

    assert report["primary_branch_gate_pass"] is True
    assert report["stretch_branch_gate_pass"] is False
    assert (out / "03_branch" / "c1" / "r075p00" / "selected_centerline_360.parquet").exists()


def test_cyclic_optimizer_can_stop_after_first_fully_passing_stage() -> None:
    mod = _load_module()
    args = mod.parse_args([])
    lengths_m, p_end_local_m, theta_sign = mod._load_robot(args)
    beta = np.deg2rad(np.asarray([0.2, -0.2, 0.4, -0.4, 2.0, -2.0]))
    xyz = mod.atlas.fk_from_beta_batch(
        beta.reshape(1, 6),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )[0]
    targets = pd.DataFrame(
        {
            "angle_idx": np.arange(4),
            "angle_rad": np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False),
            "x_target_m": xyz[0],
            "y_target_m": xyz[1],
            "z_target_m": xyz[2],
        }
    )

    _path, report = mod.atlas.optimize_cyclic_trajectory(
        targets=targets,
        initial_beta=np.tile(beta, (4, 1)),
        bounds=mod.atlas.beta_bounds_rad("current"),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        max_nfev=2,
        stop_on_centerline_gate=True,
    )

    assert report["centerline_gate_pass"] is True
    assert len(report["stage_history"]) == 1


def test_tube_curve_solver_reuses_exact_centerline_branch() -> None:
    mod = _load_module()
    lengths_m = np.full(31, 0.04, dtype=float)
    p_end_local_m = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    beta = np.deg2rad(
        np.asarray(
            [
                [0.2, -0.2, 0.4, -0.4, 2.0, -2.0],
                [0.3, -0.1, 0.5, -0.2, 2.5, -1.5],
                [0.1, -0.3, 0.2, -0.5, 1.5, -2.5],
            ]
        )
    )
    xyz = mod.atlas.fk_from_beta_batch(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
    )
    centerline = pd.DataFrame({"angle_idx": np.arange(3), "angle_rad": [0.0, 1.0, 2.0]})
    for idx, column in enumerate(mod.atlas.BETA_COLS):
        centerline[column] = beta[:, idx]
    for idx, column in enumerate(mod.atlas.XYZ_COLS):
        centerline[column] = xyz[:, idx]
    for idx, column in enumerate(mod.atlas.TARGET_XYZ_COLS):
        centerline[column] = xyz[:, idx]
    centerline["delta_n1_mm"] = 0.0
    centerline["delta_n2_mm"] = 0.0
    centerline["tube_offset_id"] = "n1_0_n2_0"
    centerline["is_centerline"] = True

    curve = mod.solve_tube_curve(
        centerline,
        centerline=centerline,
        parent_curve=centerline,
        weighted_pinv=np.zeros((3, 6, 3), dtype=float),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
        max_nfev=5,
        parent_offset_id="centerline",
    )

    assert len(curve) == 3
    assert curve["tube_success"].all()
    assert curve["xyz_residual_mm"].max() < 0.01


def test_phase_tube_materializes_formal_5x5_grid_only_after_branch_gate(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    out = tmp_path / "v5"
    branch_dir = out / "03_branch"
    radius_dir = branch_dir / "c1" / "r075p00"
    radius_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "family_id": "c1",
                "radius_mm": 75.0,
                "branch_robustness_gate_pass": True,
                "executed": True,
            }
        ]
    ).to_csv(branch_dir / "branch_radius_summary.csv", index=False)
    angle = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    centerline = pd.DataFrame(
        {
            "candidate_id": "c1",
            "family_id": "c1",
            "ellipse_id": "c1",
            "angle_idx": np.arange(8),
            "angle_rad": angle,
            "x_target_m": 1.0 + 0.075 * np.sin(angle),
            "y_target_m": 0.075 * np.cos(angle),
            "z_target_m": 0.1125 * np.sin(angle + 0.5),
        }
    )
    for column in mod.atlas.BETA_COLS:
        centerline[column] = 0.0
    for target_col, xyz_col in zip(mod.atlas.TARGET_XYZ_COLS, mod.atlas.XYZ_COLS):
        centerline[xyz_col] = centerline[target_col]
    centerline.to_parquet(radius_dir / "selected_centerline_360.parquet", index=False)

    def fake_solve(curve_targets, **_kwargs):
        curve = curve_targets.copy()
        for column in mod.atlas.BETA_COLS:
            curve[column] = 0.0
        for target_col, xyz_col in zip(mod.atlas.TARGET_XYZ_COLS, mod.atlas.XYZ_COLS):
            curve[xyz_col] = curve[target_col]
        curve["xyz_residual_mm"] = 0.0
        curve["tube_success"] = True
        curve["inverse_nfev"] = 1
        curve["parent_offset_id"] = str(_kwargs["parent_offset_id"])
        return curve

    monkeypatch.setattr(mod, "_load_robot", lambda _args: (np.ones(31), np.ones(4), -1.0))
    monkeypatch.setattr(mod, "solve_tube_curve", fake_solve)
    monkeypatch.setattr(mod.atlas, "numerical_jacobian_beta", lambda *_args, **_kwargs: np.zeros((3, 6)))
    monkeypatch.setattr(
        mod.atlas,
        "local_beta_consistency_report",
        lambda *_args, **_kwargs: {
            "tube10_beta_rms_p95_deg": 0.0,
            "multi_branch_ratio": 0.0,
            "neighbor_pair_count": 1,
            "neighbor_coverage_ratio": 1.0,
        },
    )
    args = SimpleNamespace(
        out_dir=out,
        primary_radius_mm=75.0,
        stretch_radius_mm=100.0,
        radius_anchors_mm="75",
        tube_offsets_mm="-5,-2.5,0,2.5,5",
        workers=2,
        max_ik_nfev=2,
        skip_existing=False,
    )

    report = mod.phase_tube(args)

    assert report["primary_tube_gate_pass"] is True
    tube = pd.read_parquet(out / "04_tube" / "c1" / "r075p00" / "tube_small.parquet")
    assert len(tube) == 8 * 25
    assert tube["tube_offset_id"].nunique() == 25


def test_branch_conflict_report_detects_one_to_many_beta_mapping_in_2mm_voxel() -> None:
    mod = _load_module()
    frame = pd.DataFrame(
        {
            "x_target_m": [1.0000, 1.0005, 1.0100],
            "y_target_m": [0.0, 0.0005, 0.0],
            "z_target_m": [0.0, 0.0005, 0.0],
        }
    )
    beta = np.deg2rad(np.asarray([[0.0] * 6, [4.0] * 6, [20.0] * 6]))
    for idx, column in enumerate(mod.atlas.BETA_COLS):
        frame[column] = beta[:, idx]

    report = mod.branch_conflict_report(frame, voxel_mm=2.0, threshold_deg=3.0)

    assert report["occupied_voxels"] == 2
    assert report["conflict_voxels"] == 1
    assert report["branch_conflict_gate_pass"] is False

    resolved = frame.copy()
    resolved.loc[1, mod.atlas.BETA_COLS] = np.deg2rad(2.0)
    assert mod.branch_conflict_report(resolved, voxel_mm=2.0, threshold_deg=3.0)["branch_conflict_gate_pass"] is True


def test_phase_dataset_selects_one_conflict_free_family_from_cross_family_diagnostic_union(tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "v5"
    tube_dir = out / "04_tube"
    rows = []
    offsets = [-5.0, -2.5, 0.0, 2.5, 5.0]
    for radius_mm in (75.0, 80.0, 87.5):
        radius_dir = tube_dir / "c1" / mod._radius_slug(radius_mm)
        radius_dir.mkdir(parents=True)
        tube_rows = []
        for angle_idx in range(4):
            angle = 2.0 * math.pi * angle_idx / 4.0
            for dn1 in offsets:
                for dn2 in offsets:
                    record = {
                        "candidate_id": "c1",
                        "family_id": "c1",
                        "ellipse_id": "c1",
                        "angle_idx": angle_idx,
                        "angle_rad": angle,
                        "amp_xy_mm": radius_mm,
                        "amp_z_mm": 1.5 * radius_mm,
                        "delta_n1_mm": dn1,
                        "delta_n2_mm": dn2,
                        "tube_offset_id": f"n1_{dn1:g}_n2_{dn2:g}",
                        "is_centerline": dn1 == 0.0 and dn2 == 0.0,
                        "x_target_m": 1.0 + radius_mm / 1000.0 * math.sin(angle) + dn1 / 1000.0,
                        "y_target_m": radius_mm / 1000.0 * math.cos(angle) + dn2 / 1000.0,
                        "z_target_m": 1.5 * radius_mm / 1000.0 * math.sin(angle + 0.5),
                        "xyz_residual_mm": 0.0,
                        "tube_success": True,
                    }
                    for column in mod.atlas.BETA_COLS:
                        record[column] = 0.0
                    tube_rows.append(record)
        path = radius_dir / "tube_small.parquet"
        pd.DataFrame(tube_rows).to_parquet(path, index=False)
        rows.append(
            {
                "candidate_id": "c1",
                "family_id": "c1",
                "radius_mm": radius_mm,
                "tube_gate_pass": True,
                "rows": len(tube_rows),
            }
        )
    conflicting = pd.read_parquet(tube_dir / "c1" / mod._radius_slug(75.0) / "tube_small.parquet")
    conflicting["candidate_id"] = "c2"
    conflicting["family_id"] = "c2"
    conflicting["ellipse_id"] = "c2"
    for column in mod.atlas.BETA_COLS:
        conflicting[column] = math.radians(10.0)
    conflicting_dir = tube_dir / "c2" / mod._radius_slug(75.0)
    conflicting_dir.mkdir(parents=True)
    conflicting.to_parquet(conflicting_dir / "tube_small.parquet", index=False)
    rows.append(
        {
            "candidate_id": "c2",
            "family_id": "c2",
            "radius_mm": 75.0,
            "tube_gate_pass": True,
            "rows": len(conflicting),
        }
    )
    pd.DataFrame(rows).to_csv(tube_dir / "tube_radius_summary.csv", index=False)
    args = SimpleNamespace(
        out_dir=out,
        final_points=4,
        primary_radius_mm=87.5,
        stretch_radius_mm=100.0,
        skip_existing=False,
    )

    report = mod.phase_dataset(args)

    assert report["dataset_gate_pass"] is True
    assert report["trajectory_count"] == 3
    assert report["all_tube_trajectory_count"] == 4
    assert report["selected_family_id"] == "c1"
    assert report["global_branch_conflict"]["branch_conflict_gate_pass"] is False
    dataset = pd.read_parquet(out / "05_dataset" / "true_ellipse_family_tubes_v5.parquet")
    assert len(dataset) == 3 * 4 * 25
    assert dataset["trajectory_id"].nunique() == 3
    assert dataset["family_id"].unique().tolist() == ["c1"]


def test_expansion_radius_status_requires_same_family_to_pass_all_stages() -> None:
    mod = _load_module()
    pointwise = pd.DataFrame(
        [
            {"candidate_id": "p1", "radius_mm": 75.0, "pointwise_gate_pass": True},
            {"candidate_id": "p1", "radius_mm": 87.5, "pointwise_gate_pass": True},
            {"candidate_id": "p2", "radius_mm": 87.5, "pointwise_gate_pass": True},
        ]
    )
    branch = pd.DataFrame(
        [
            {"candidate_id": "p1", "radius_mm": 75.0, "branch_robustness_gate_pass": True},
            {"candidate_id": "p2", "radius_mm": 87.5, "branch_robustness_gate_pass": True},
        ]
    )
    tube = pd.DataFrame(
        [
            {"candidate_id": "p1", "radius_mm": 75.0, "tube_gate_pass": True},
            {"candidate_id": "p2", "radius_mm": 87.5, "tube_gate_pass": True},
        ]
    )

    status = mod.build_expansion_radius_status(
        pointwise,
        branch,
        tube,
        radii_mm=[75.0, 87.5],
    )

    assert bool(status.iloc[0]["trajectory_materialization_gate_pass"]) is True
    assert bool(status.iloc[1]["trajectory_materialization_gate_pass"]) is True
    assert status.iloc[1]["passing_families"] == "p2"

    tube.loc[tube["candidate_id"] == "p2", "candidate_id"] = "other"
    failed = mod.build_expansion_radius_status(pointwise, branch, tube, radii_mm=[87.5])
    assert bool(failed.iloc[0]["trajectory_materialization_gate_pass"]) is False
    assert failed.iloc[0]["limiting_factor"] == "tube"
