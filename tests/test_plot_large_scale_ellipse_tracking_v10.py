from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "scripts" / "analysis" / "plot_large_scale_ellipse_tracking_v10.py"
    spec = importlib.util.spec_from_file_location(
        "plot_large_scale_ellipse_tracking_v10",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_to_ellipse_plane_uses_pose_axes_and_center() -> None:
    plot = _load_module()
    center = np.array([1.0, 2.0, 3.0])
    major = np.array([0.0, 2.0, 0.0])
    minor = np.array([0.0, 0.0, -4.0])
    points = np.array(
        [
            [1.0, 4.0, -9.0],
            [1.0, 0.0, 7.0],
        ]
    )

    projected = plot.project_to_ellipse_plane(
        points,
        center_m=center,
        major_direction=major,
        minor_direction=minor,
    )

    np.testing.assert_allclose(projected, [[2.0, 12.0], [-2.0, -4.0]])


def _write_scale_fixture(
    root: Path,
    *,
    slug: str,
    major_m: float,
    evidence_tier: str,
    variants: tuple[str, ...],
    rows: int,
) -> None:
    scale_dir = root / slug
    scale_dir.mkdir(parents=True, exist_ok=True)
    pose = {
        "major_semiaxis_m": major_m,
        "minor_semiaxis_m": major_m / 3.0,
        "center_m": [0.7, -0.1, 0.2],
        "major_direction": [0.0, 1.0, 0.0],
        "minor_direction": [0.0, 0.0, 1.0],
    }
    (scale_dir / "pose_report.json").write_text(json.dumps(pose), encoding="utf-8")
    phase = np.arange(rows, dtype=float) * (2.0 * np.pi / rows)
    target = np.column_stack(
        [
            np.full(rows, 0.7),
            -0.1 + major_m * np.cos(phase),
            0.2 + major_m / 3.0 * np.sin(phase),
        ]
    )
    for variant_index, variant in enumerate(variants, start=1):
        residual_m = variant_index * 1.0e-5 * (1.2 + np.cos(phase))
        achieved = target + np.column_stack(
            [residual_m, np.zeros(rows), np.zeros(rows)]
        )
        frame = pd.DataFrame(
            {
                "target_x_m": target[:, 0],
                "target_y_m": target[:, 1],
                "target_z_m": target[:, 2],
                "achieved_x_m": achieved[:, 0],
                "achieved_y_m": achieved[:, 1],
                "achieved_z_m": achieved[:, 2],
                "teacher_fk_residual_mm": np.linalg.norm(achieved - target, axis=1)
                * 1000.0,
                "phase_idx": np.arange(rows),
                "phase_rad": phase,
            }
        )
        variant_dir = scale_dir / evidence_tier / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(variant_dir / "centerline.parquet", index=False)
        (variant_dir / "centerline_report.json").write_text(
            json.dumps({"variant": variant, "centerline_gate_pass": variant == "T3"}),
            encoding="utf-8",
        )


def test_render_tracking_evidence_writes_scale_plots_overview_and_manifest(
    tmp_path: Path,
) -> None:
    plot = _load_module()
    input_root = tmp_path / "challenge"
    _write_scale_fixture(
        input_root,
        slug="a0p500m",
        major_m=0.5,
        evidence_tier="formal",
        variants=("T3", "T4"),
        rows=180,
    )
    _write_scale_fixture(
        input_root,
        slug="a0p750m",
        major_m=0.75,
        evidence_tier="screen",
        variants=("T1", "T3", "T4"),
        rows=24,
    )
    _write_scale_fixture(
        input_root,
        slug="a1p000m",
        major_m=1.0,
        evidence_tier="screen",
        variants=("T1", "T3", "T4"),
        rows=24,
    )

    output = tmp_path / "plots"
    manifest = plot.render_tracking_evidence(input_root=input_root, output_dir=output)

    expected = {
        "a0p500m_teacher_tracking.png",
        "a0p750m_teacher_tracking.png",
        "a1p000m_teacher_tracking.png",
        "large_scale_teacher_tracking_overview.png",
    }
    assert {Path(item["path"]).name for item in manifest["figures"]} == expected
    assert manifest["tracking_source"] == "teacher_solver_fk_not_neural_student"
    assert manifest["visualization_only"] is True
    assert manifest["scales"]["a0p500m"]["phase_count"] == 180
    assert manifest["scales"]["a0p750m"]["evidence_tier"] == "screen"
    assert (output / "plot_manifest.json").is_file()
    assert all((output / name).stat().st_size > 1000 for name in expected)
