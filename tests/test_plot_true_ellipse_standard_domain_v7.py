from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "plot_true_ellipse_standard_domain_v7.py"
    spec = importlib.util.spec_from_file_location(
        "plot_true_ellipse_standard_domain_v7",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _curve(radius_mm: float, *, rows: int = 16, include_achieved: bool = True) -> pd.DataFrame:
    angle = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)
    radius_m = float(radius_mm) / 1000.0
    target = np.column_stack(
        [
            1.1 + radius_m * np.sin(angle),
            0.1 + radius_m * np.sin(angle + 0.4),
            -0.2 + 1.5 * radius_m * np.sin(angle + 1.1),
        ]
    )
    beta = np.column_stack(
        [0.01 * np.sin(angle + 0.2 * index) for index in range(6)]
    )
    frame = pd.DataFrame(
        {
            "angle_idx": np.arange(rows),
            "angle_rad": angle,
            "radius_mm": float(radius_mm),
            "x_target_m": target[:, 0],
            "y_target_m": target[:, 1],
            "z_target_m": target[:, 2],
            **{f"beta{index + 1}_rad": beta[:, index] for index in range(6)},
        }
    )
    if include_achieved:
        achieved = target + np.column_stack(
            [
                1.0e-5 * np.cos(angle),
                2.0e-5 * np.sin(angle),
                1.5e-5 * np.cos(angle + 0.3),
            ]
        )
        frame[["x_m", "y_m", "z_m"]] = achieved
        frame["xyz_residual_mm"] = np.linalg.norm(achieved - target, axis=1) * 1000.0
        frame["kappa"] = 40.0 + 20.0 * (1.0 + np.sin(angle))
        frame["sigma3_m"] = 0.05 - 0.01 * np.sin(angle)
    return frame


def _write_run_fixture(root: Path) -> Path:
    radial = root / "01_radial"
    attempts = []
    for radius, passed in ((100.0, True), (102.5, True), (104.0, True), (104.25, False), (105.0, False)):
        attempts.append(
            {
                "parent_radius_mm": radius - 0.5,
                "target_radius_mm": radius,
                "step_mm": 0.5,
                "step_kind": "fixture",
                "radial_bundle_gate_pass": passed,
                "report": {"radius_mm": radius, "radial_bundle_gate_pass": passed},
            }
        )
    _write_json(
        radial / "radial_report.json",
        {
            "protocol_fingerprint": "protocol-fixture",
            "task_fingerprint": "task-fixture",
            "protocol": {
                "formal_checkpoints_mm": [100.0, 102.5, 105.0, 107.5, 110.0, 120.0],
            },
            "joint_domain_id": "standard_beta34_10deg_v1",
            "family_id": "fixed-family",
            "attempts": attempts,
            "strict_geometry_rmax_mm": 102.5,
            "exploratory_rescue_rmax_mm": 120.0,
            "first_strict_failure_mm": 105.0,
            "formal_radial_gate_pass": False,
        },
    )
    _write_json(
        root / "02_tube" / "tube_report.json",
        {
            "radial_strict_rmax_mm": 102.5,
            "strict_geometry_rmax_mm": None,
            "formal_tube_gate_pass": False,
            "reason": "formal_radial_gate_failed_below_105mm",
        },
    )

    for radius in (100.0, 101.0, 102.5, 104.0):
        out = radial / f"r{radius:06.2f}".replace(".", "p") / "selected_centerline_360.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        _curve(radius).to_parquet(out, index=False)

    for radius, schedules in (
        (104.25, (("balanced", 205.0), ("loose", 184.0))),
        (105.0, (("conservative", 362.0), ("balanced", 429.0))),
    ):
        job_dir = (
            radial
            / f"r{radius:06.2f}".replace(".", "p")
            / "jobs"
            / "parent_copy"
            / "cut_000"
        )
        job_dir.mkdir(parents=True, exist_ok=True)
        attempt_rows = []
        for schedule, kappa in schedules:
            curve_path = job_dir / f"{schedule}.parquet"
            report_path = job_dir / f"{schedule}.json"
            frame = _curve(radius)
            frame["kappa"] = kappa + 10.0 * np.sin(frame["angle_rad"])
            frame["sigma3_m"] = 6.0 / frame["kappa"]
            frame.to_parquet(curve_path, index=False)
            _write_json(
                report_path,
                {
                    "anchor_schedule": schedule,
                    "kappa_p95": kappa,
                    "sigma3_p05_m": float(frame["sigma3_m"].quantile(0.05)),
                    "residual_p95_mm": float(frame["xyz_residual_mm"].quantile(0.95)),
                    "residual_max_mm": float(frame["xyz_residual_mm"].max()),
                    "centerline_gate_pass": False,
                    "job_gate_pass": False,
                },
            )
            attempt_rows.append(
                {
                    "strategy": "direct_joint_corrector",
                    "anchor_schedule": schedule,
                    "centerline_gate_pass": False,
                    "job_gate_pass": False,
                    "report_path": str(report_path),
                    "path": str(curve_path),
                    "kappa_p95": kappa,
                }
            )
        _write_json(
            job_dir / "job_report.json",
            {
                "selected": False,
                "centerline_gate_pass": False,
                "job_gate_pass": False,
                "attempts": attempt_rows,
            },
        )

    for radius in (105.0, 120.0):
        path = radial / "exploratory_pointwise" / f"r{radius:06.2f}".replace(".", "p")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = _curve(radius, include_achieved=False)
        frame["xyz_residual_mm"] = np.linspace(0.1, 4.0, len(frame))
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
        _write_json(
            path.with_suffix(".json"),
            {
                "radius_mm": radius,
                "rows": len(frame),
                "success_ratio_le2mm": float(np.mean(frame["xyz_residual_mm"] <= 2.0)),
                "residual_p95_mm": float(frame["xyz_residual_mm"].quantile(0.95)),
                "residual_max_mm": float(frame["xyz_residual_mm"].max()),
                "rescue_admission_pass": True,
            },
        )
    return root


def test_discover_evidence_separates_registered_continuous_failed_and_exploratory(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    run_dir = _write_run_fixture(tmp_path / "run")

    inventory = mod.discover_trajectory_evidence(
        run_dir,
        strict_radii=(100.0, 102.5, 104.0),
        failed_radii=(104.25, 105.0),
        exploratory_radii=(105.0, 120.0),
    )

    assert [item.evidence_class for item in inventory.strict] == [
        "registered_strict_checkpoint",
        "registered_strict_checkpoint",
        "continuous_pass",
    ]
    assert [item.evidence_class for item in inventory.failed] == [
        "failed_continuation",
        "failed_continuation",
    ]
    assert [item.evidence_class for item in inventory.exploratory] == [
        "exploratory_pointwise",
        "exploratory_pointwise",
    ]
    assert inventory.failed[0].schedule == "loose"
    assert inventory.failed[0].kappa_p95 == pytest.approx(184.0)
    assert inventory.failed[1].schedule == "conservative"
    assert inventory.last_continuous_pass_mm == pytest.approx(104.0)
    assert inventory.radial_strict_rmax_mm == pytest.approx(102.5)
    assert inventory.strict_geometry_rmax_mm is None
    assert inventory.formal_tube_gate_pass is False


def test_discover_only_labels_protocol_checkpoints_as_registered(tmp_path: Path) -> None:
    mod = _load_module()
    run_dir = _write_run_fixture(tmp_path / "run")

    inventory = mod.discover_trajectory_evidence(
        run_dir,
        strict_radii=(100.0, 101.0, 102.5, 104.0),
        failed_radii=(104.25,),
        exploratory_radii=(105.0,),
    )

    assert [item.evidence_class for item in inventory.strict] == [
        "registered_strict_checkpoint",
        "continuous_pass",
        "registered_strict_checkpoint",
        "continuous_pass",
    ]


def test_conditioning_selects_largest_passing_radius_regardless_of_input_order() -> None:
    mod = _load_module()
    curve = mod.prepare_curve(_curve(104.0))
    continuous = mod.TrajectoryEvidence(
        radius_mm=104.0,
        evidence_class="continuous_pass",
        path=Path("r104.parquet"),
        report_path=None,
    )
    checkpoint = mod.TrajectoryEvidence(
        radius_mm=102.5,
        evidence_class="registered_strict_checkpoint",
        path=Path("r102p5.parquet"),
        report_path=None,
    )

    selected = mod.select_conditioning_cases(
        [(continuous, curve), (checkpoint, curve)],
        [],
    )

    assert selected[0][0].radius_mm == pytest.approx(104.0)
    assert checkpoint.is_continuous_path_pass is True


def test_discover_rejects_strict_label_above_continuous_frontier(tmp_path: Path) -> None:
    mod = _load_module()
    run_dir = _write_run_fixture(tmp_path / "run")

    with pytest.raises(ValueError, match="strict display radius exceeds continuous pass frontier"):
        mod.discover_trajectory_evidence(
            run_dir,
            strict_radii=(105.0,),
            failed_radii=(104.25,),
            exploratory_radii=(120.0,),
        )


def test_prepare_curve_recomputes_exploratory_fk_and_axis_errors() -> None:
    mod = _load_module()
    frame = _curve(120.0, include_achieved=False)
    target = frame[["x_target_m", "y_target_m", "z_target_m"]].to_numpy(dtype=float)
    offset = np.asarray([0.001, -0.002, 0.003], dtype=float)

    prepared = mod.prepare_curve(frame, fk_from_beta=lambda _beta: target + offset)

    assert prepared[["x_m", "y_m", "z_m"]].to_numpy() == pytest.approx(target + offset)
    assert prepared["x_error_mm"].to_numpy() == pytest.approx(1.0)
    assert prepared["y_error_mm"].to_numpy() == pytest.approx(-2.0)
    assert prepared["z_error_mm"].to_numpy() == pytest.approx(3.0)
    assert prepared["xyz_residual_recomputed_mm"].to_numpy() == pytest.approx(
        np.sqrt(14.0)
    )


def test_visual_style_matches_existing_true_ellipse_figures() -> None:
    mod = _load_module()
    curve = _curve(104.0)
    xyz = curve[["x_target_m", "y_target_m", "z_target_m"]].to_numpy(dtype=float)
    centered = xyz - np.mean(xyz, axis=0, keepdims=True)
    normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
    view = mod.trajectory_view_from_xyz(xyz)
    camera = mod.camera_direction_from_view(view)

    assert abs(float(np.dot(normal, camera))) >= np.cos(np.deg2rad(15.0))
    assert mod.STRICT_OVERVIEW_PANELS == ("trajectory_3d", "xy", "yz", "axis_errors")
    assert mod.CONDITIONING_LOG_SCALE is True


def test_conditioning_summary_uses_registered_percentiles_not_pointwise_extrema() -> None:
    mod = _load_module()
    curve = _curve(104.0)
    curve["kappa"] = np.asarray([10.0] * 15 + [1000.0])
    curve["sigma3_m"] = np.asarray([0.1] * 15 + [0.0001])
    prepared = mod.prepare_curve(curve)

    summary = mod.conditioning_summary(prepared)

    assert summary["kappa_p95"] == pytest.approx(np.percentile(curve["kappa"], 95))
    assert summary["kappa_max"] == pytest.approx(1000.0)
    assert summary["sigma3_p05_m"] == pytest.approx(np.percentile(curve["sigma3_m"], 5))
    assert summary["sigma3_min_m"] == pytest.approx(0.0001)


def test_render_visualizations_writes_four_existing_style_png_and_non_gating_manifest(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    run_dir = _write_run_fixture(tmp_path / "run")
    out_dir = tmp_path / "visualization"

    def exploratory_fk(beta: np.ndarray) -> np.ndarray:
        rows = len(beta)
        angle = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)
        radius_m = 0.105 if rows == 16 else 0.120
        return np.column_stack(
            [
                1.1 + radius_m * np.sin(angle),
                0.1 + radius_m * np.sin(angle + 0.4),
                -0.2 + 1.5 * radius_m * np.sin(angle + 1.1),
            ]
        )

    report = mod.render_v7_trajectory_visualizations(
        run_dir=run_dir,
        out_dir=out_dir,
        strict_radii=(100.0, 102.5, 104.0),
        failed_radii=(104.25, 105.0),
        exploratory_radii=(105.0, 120.0),
        exploratory_fk=exploratory_fk,
    )

    assert report["visualization_only"] is True
    assert report["changes_formal_gate"] is False
    assert report["strict_model_claim_made"] is False
    assert report["radial_strict_rmax_mm"] == pytest.approx(102.5)
    assert report["strict_geometry_rmax_mm"] is None
    assert report["formal_tube_gate_pass"] is False
    assert report["checks"]["evidence_classes_disjoint"] is True
    assert report["checks"]["all_figures_written"] is True
    assert mod.FIGURE_DPI == 180
    assert {figure["figure_id"] for figure in report["figures"]} == {
        "strict_tracking_overview",
        "strict_joint_profiles",
        "conditioning_frontier",
        "exploratory_pointwise_overview",
    }
    for figure in report["figures"]:
        assert "svg_path" not in figure
        path = Path(figure["png_path"])
        assert path.exists()
        assert path.stat().st_size > 1000
    persisted = json.loads((out_dir / "visualization_report.json").read_text(encoding="utf-8"))
    assert persisted == report


def test_cli_defaults_are_explicitly_visualization_only() -> None:
    mod = _load_module()

    args = mod.parse_args([])

    assert args.strict_radii_mm == "100,102.5,104"
    assert args.failed_radii_mm == "104.25,105"
    assert args.exploratory_radii_mm == "105,110,115,120"
    assert args.robot_config.name == "robot_rods_only_standard_100k.yaml"
    assert args.out_dir == args.run_dir / "05_visualization"
