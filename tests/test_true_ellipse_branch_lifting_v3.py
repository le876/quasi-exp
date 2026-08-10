from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    mod_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_utils():
    return _load_module(
        "true_ellipse_atlas_utils",
        "scripts/analysis/true_ellipse_atlas_utils.py",
    )


def _load_runner():
    _load_utils()
    return _load_module(
        "run_true_ellipse_branch_lifting_v3",
        "scripts/analysis/run_true_ellipse_branch_lifting_v3.py",
    )


def _robot_inputs() -> tuple[np.ndarray, np.ndarray, float]:
    lengths_m = np.full(31, 0.04, dtype=float)
    p_end_local_m = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    return lengths_m, p_end_local_m, -1.0


def _smooth_targets(mod, n: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    lengths_m, p_end_local_m, theta_sign = _robot_inputs()
    phase = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    beta_deg = np.column_stack(
        [
            0.6 * np.sin(phase),
            0.5 * np.cos(phase),
            1.0 * np.sin(phase + 0.2),
            0.8 * np.cos(phase - 0.1),
            5.0 * np.sin(phase + 0.4),
            4.0 * np.cos(phase - 0.3),
        ]
    )
    beta = np.deg2rad(beta_deg)
    xyz = mod.fk_from_beta_batch(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )
    targets = pd.DataFrame(
        {
            "candidate_id": ["synthetic"] * n,
            "angle_idx": np.arange(n, dtype=np.int64),
            "angle_rad": phase,
            "x_target_m": xyz[:, 0],
            "y_target_m": xyz[:, 1],
            "z_target_m": xyz[:, 2],
        }
    )
    return targets, beta


def test_continuation_lift_supports_forward_and_reverse_closed_traversal() -> None:
    mod = _load_utils()
    targets, true_beta = _smooth_targets(mod)
    lengths_m, p_end_local_m, theta_sign = _robot_inputs()
    common = dict(
        targets=targets,
        start_beta=true_beta[0],
        bounds=mod.beta_bounds_rad("current"),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        method="predictive",
        lambda_center=1.0e-3,
        max_nfev=80,
    )

    forward, f_report = mod.continuation_lift(direction="forward", **common)
    reverse, r_report = mod.continuation_lift(direction="reverse", **common)
    agreement = mod.branch_reproducibility_report(forward, reverse)

    assert forward["angle_idx"].tolist() == list(range(len(targets)))
    assert reverse["angle_idx"].tolist() == list(range(len(targets)))
    assert f_report["residual_p95_mm"] < 0.1
    assert r_report["residual_p95_mm"] < 0.1
    assert f_report["seam_beta_rms_deg"] <= 1.5
    assert r_report["seam_beta_rms_deg"] <= 1.5
    assert agreement["branch_diff_p95_deg"] <= 1.5


def test_nullspace_seeds_are_diverse_bounded_and_first_order_task_preserving() -> None:
    mod = _load_utils()
    lengths_m, p_end_local_m, theta_sign = _robot_inputs()
    beta = np.deg2rad(np.asarray([0.5, -0.5, 1.0, -1.0, 4.0, -3.0]))
    bounds = mod.beta_bounds_rad("current")
    jac = mod.numerical_jacobian_beta(
        beta,
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
    )

    seeds = mod.generate_nullspace_seeds(
        beta,
        jac,
        bounds=bounds,
        scales_deg=[0.25, 0.5],
        seeds_per_scale=6,
        seed=17,
    )
    delta = seeds - beta[None, :]

    assert seeds.shape == (12, 6)
    assert np.all(seeds >= bounds[:, 0] - 1.0e-12)
    assert np.all(seeds <= bounds[:, 1] + 1.0e-12)
    assert np.linalg.matrix_rank(delta, tol=1.0e-8) >= 3
    assert float(np.max(np.linalg.norm(delta @ jac.T, axis=1))) < 1.0e-8


def test_soft_cyclic_linker_prefers_smooth_closed_path_without_hard_edge_gate() -> None:
    mod = _load_utils()
    rows: list[dict[str, float | int]] = []
    smooth_deg = np.asarray([0.0, 3.5, 7.0, 3.5])
    rough_deg = np.asarray([-10.0, 10.0, -10.0, 10.0])
    for angle_idx in range(4):
        for branch_id, value in enumerate((smooth_deg[angle_idx], rough_deg[angle_idx])):
            row: dict[str, float | int] = {
                "angle_idx": angle_idx,
                "branch_id": branch_id,
                "xyz_residual_mm": 0.01,
                "kappa": 10.0,
            }
            beta = np.deg2rad(np.full(6, value, dtype=float))
            for j, col in enumerate(mod.BETA_COLS):
                row[col] = float(beta[j])
            rows.append(row)
    candidates = pd.DataFrame(rows)

    selected, report = mod.link_cyclic_branch_soft(
        candidates,
        lambda_velocity=1.0,
        closure_weight=5.0,
    )

    assert report["success"] is True
    assert selected["branch_id"].tolist() == [0, 0, 0, 0]
    assert report["seam_beta_rms_deg"] == 3.5
    assert report["delta_beta_rms_p95_deg"] > 3.0


def test_cyclic_residual_sparsity_wraps_first_and_last_waypoints() -> None:
    mod = _load_utils()
    targets, beta = _smooth_targets(mod, n=5)
    sparsity = mod.cyclic_trajectory_jac_sparsity(
        n_points=5,
        include_acceleration=True,
        include_anchor=True,
        include_posture=True,
    ).tocsr()
    residual = mod.cyclic_trajectory_residual(
        beta.reshape(-1),
        targets_xyz=targets[mod.TARGET_XYZ_COLS].to_numpy(dtype=float),
        lengths_m=_robot_inputs()[0],
        p_end_local_m=_robot_inputs()[1],
        theta_sign=-1.0,
        anchor_beta=beta,
        lambda_velocity=1.0,
        lambda_acceleration=1.0,
        lambda_anchor=1.0,
        lambda_posture=1.0,
    )

    assert sparsity.shape == (len(residual), 5 * 6)
    velocity_start = 5 * 3
    last_velocity_rows = slice(velocity_start + 4 * 6, velocity_start + 5 * 6)
    touched = np.unique(sparsity[last_velocity_rows].nonzero()[1])
    assert set(range(0, 6)).issubset(touched)
    assert set(range(24, 30)).issubset(touched)


def test_cyclic_residual_and_sparsity_include_registered_joint_margin_barrier() -> None:
    mod = _load_utils()
    targets, beta = _smooth_targets(mod, n=5)
    bounds = np.deg2rad(
        np.asarray([[-5, 5], [-5, 5], [-10, 10], [-10, 10], [-15, 15], [-15, 15]], dtype=float)
    )
    beta[:, 0] = np.deg2rad(4.99)
    base = mod.cyclic_trajectory_residual(
        beta.reshape(-1),
        targets_xyz=targets[mod.TARGET_XYZ_COLS].to_numpy(dtype=float),
        lengths_m=_robot_inputs()[0],
        p_end_local_m=_robot_inputs()[1],
        theta_sign=-1.0,
        bounds=bounds,
        lambda_margin=0.0,
    )
    with_margin = mod.cyclic_trajectory_residual(
        beta.reshape(-1),
        targets_xyz=targets[mod.TARGET_XYZ_COLS].to_numpy(dtype=float),
        lengths_m=_robot_inputs()[0],
        p_end_local_m=_robot_inputs()[1],
        theta_sign=-1.0,
        bounds=bounds,
        lambda_margin=1.0,
        soft_margin_deg=0.25,
    )
    sparsity = mod.cyclic_trajectory_jac_sparsity(
        n_points=5,
        include_acceleration=False,
        include_anchor=False,
        include_posture=False,
        include_margin=True,
    )

    assert len(with_margin) == len(base) + 5 * 6
    assert sparsity.shape == (len(with_margin), 5 * 6)
    assert np.linalg.norm(with_margin[-30:]) > 0.0


def test_trajectory_optimization_reduces_periodic_roughness_without_losing_tracking() -> None:
    mod = _load_utils()
    targets, true_beta = _smooth_targets(mod, n=12)
    lengths_m, p_end_local_m, theta_sign = _robot_inputs()
    initial = true_beta.copy()
    initial[1::2] += np.deg2rad(np.asarray([0.15, -0.15, 0.12, -0.12, 0.2, -0.2]))
    initial_df = targets.copy()
    for j, col in enumerate(mod.BETA_COLS):
        initial_df[col] = initial[:, j]
    initial_report = mod.smoothness_report(initial_df)

    optimized, report = mod.optimize_cyclic_trajectory(
        targets=targets,
        initial_beta=initial,
        bounds=mod.beta_bounds_rad("current"),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=theta_sign,
        stages=[
            {
                "name": "test",
                "lambda_velocity": 1.0,
                "lambda_acceleration": 0.25,
                "lambda_anchor": 0.01,
                "lambda_posture": 0.0,
            }
        ],
        max_nfev=20,
    )

    assert len(optimized) == len(targets)
    assert report["residual_p95_mm"] <= 2.0
    assert report["delta2_beta_p95_deg"] <= initial_report["delta2_beta_p95_deg"]
    assert report["selected_stage"] == "test"


def test_local_beta_consistency_uses_all_neighbors_inside_radius() -> None:
    mod = _load_utils()
    frame = pd.DataFrame(
        {
            "x_m": [0.0, 0.001, 0.002],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )
    beta = np.deg2rad(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            ]
        )
    )
    for j, col in enumerate(mod.BETA_COLS):
        frame[col] = beta[:, j]

    report = mod.local_beta_consistency_report(
        frame,
        radius_mm=10.0,
        max_neighbors=None,
        multi_branch_threshold_deg=3.0,
    )

    assert report["neighbor_pair_count"] == 6
    assert report["multi_branch_ratio"] > 0.0
    assert report["tube10_beta_rms_p95_deg"] > 3.0


def test_runner_cli_and_tube_gate_stop_without_passing_centerline(tmp_path: Path) -> None:
    mod = _load_runner()
    out_dir = tmp_path / "v3"
    args = mod.parse_args(
        [
            "--preset",
            "smoke",
            "--v2-dir",
            str(tmp_path / "v2"),
            "--out-dir",
            str(out_dir),
            "--seed-budgets",
            "16,32",
            "--center-lambdas",
            "0,0.0001,0.001,0.01",
        ]
    )
    mod.apply_preset_defaults(args)

    assert args.coarse_points <= 24
    assert args.final_points <= 24
    assert mod._parse_int_csv(args.seed_budgets) == [16, 32]
    assert mod._parse_float_csv(args.center_lambdas) == [0.0, 0.0001, 0.001, 0.01]

    result = mod.phase_tube(args)
    report_path = out_dir / "06_local_tube" / "tube_quality_report.json"

    assert result.empty
    assert report_path.exists()
    assert "no_passing_360_centerline" in report_path.read_text(encoding="utf-8")
    assert not list((out_dir / "06_local_tube").glob("*.parquet"))


def test_cached_centerline_report_is_regated_with_current_threshold(tmp_path: Path) -> None:
    mod = _load_runner()
    report_path = tmp_path / "legacy_report.json"
    report_path.write_text(
        json.dumps(
            {
                "residual_p95_mm": 0.1,
                "residual_max_mm": 0.2,
                "delta_beta_p95_deg": 0.1,
                "delta_beta_max_deg": 0.2,
                "delta2_beta_p95_deg": 0.01,
                "seam_beta_rms_deg": 0.76,
                "sigma3_p05_m": 0.2,
                "kappa_p95": 20.0,
                "branch_gate_pass": True,
                "canonical_gate_pass": True,
                "conditioning_gate_pass": True,
                "centerline_gate_pass": True,
            }
        ),
        encoding="utf-8",
    )

    report = mod._load_augmented_report(report_path)

    assert report["branch_gate_pass"] is False
    assert report["canonical_gate_pass"] is False
    assert report["centerline_gate_pass"] is False


def test_robustness_gate_uses_selected_reproducible_pair_not_unrelated_run_ratio() -> None:
    mod = _load_runner()

    passed = mod._selected_branch_robustness_gate(
        selected_pair={"reproducible": True},
        repeatability={"reproducible": True},
        selected_report={"centerline_gate_pass": True},
    )

    assert passed is True


def test_cached_robustness_report_recomputes_gate_from_selected_branch(tmp_path: Path) -> None:
    mod = _load_runner()
    report_path = tmp_path / "legacy_robustness.json"
    report_path.write_text(
        json.dumps(
            {
                "robustness_gate_pass": True,
                "selected_run": {
                    "residual_p95_mm": 0.1,
                    "residual_max_mm": 0.2,
                    "delta_beta_p95_deg": 0.1,
                    "delta_beta_max_deg": 0.2,
                    "delta2_beta_p95_deg": 0.01,
                    "seam_beta_rms_deg": 0.76,
                    "sigma3_p05_m": 0.2,
                    "kappa_p95": 20.0,
                    "centerline_gate_pass": True,
                },
                "selected_forward_reverse": {"reproducible": True},
                "deterministic_repeatability": {"reproducible": True},
            }
        ),
        encoding="utf-8",
    )

    report = mod._load_robustness_report(report_path)

    assert report["selected_run"]["centerline_gate_pass"] is False
    assert report["robustness_gate_pass"] is False
