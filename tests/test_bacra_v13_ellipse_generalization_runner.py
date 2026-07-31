from pathlib import Path

import numpy as np

import pandas as pd

from scripts.analysis.run_bacra_v13_ellipse_generalization import _cycle_points, _select_catalog


def test_cycle_points_are_complete_analytical_ellipses() -> None:
    row = type(
        "Section",
        (),
        {
            "center_x_m": 1.0, "center_y_m": 2.0, "center_z_m": 3.0,
            "axis0_x": 1.0, "axis0_y": 0.0, "axis0_z": 0.0,
            "axis1_x": 0.0, "axis1_y": 1.0, "axis1_z": 0.0,
            "semimajor_m": 0.3, "semiminor_m": 0.1,
        },
    )()
    points = _cycle_points(row, 720)
    assert points.shape == (720, 3)
    normalized = np.square((points[:, 0] - 1.0) / 0.3) + np.square(
        (points[:, 1] - 2.0) / 0.1
    )
    np.testing.assert_allclose(normalized, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(points[:, 2], 3.0)


def test_ellipse_runner_is_locked_evaluation_only() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/analysis/run_bacra_v13_ellipse_generalization.py").read_text()
    assert "model bytes changed after lock" in text
    assert ".fit(" not in text
    assert "plane_section" in text
    assert "relative_error_p95_max" in text
    assert 'catalog["cycle_id"] < int(ellipse_config["sealed_cycles"])' in text
    assert 'sealed_catalog["orientation_bin"].nunique()' in text
    assert 'return 0 if bool(report.get("gate_pass")) else 2' in text


def test_catalog_selection_keeps_scale_extremes_and_marginal_coverage() -> None:
    rows = []
    for index in range(48):
        rows.append(
            {
                "semimajor_m": 0.01 + 0.01 * index,
                "axis_ratio": 0.30 + 0.01 * (index % 4),
                "normal_x": float(index % 2),
                "normal_y": float((index // 2) % 2),
                "normal_z": float((index // 4) % 2),
                "offset_fraction": -0.97 + 1.94 * index / 47,
                "size_bin": index % 5,
                "axis_ratio_bin": index % 4,
                "orientation_bin": index % 12,
                "offset_bin": index % 4,
            }
        )
    candidates = pd.DataFrame(rows)
    catalog = _select_catalog(candidates, 24)
    assert catalog["semimajor_m"].min() == candidates["semimajor_m"].min()
    assert catalog["semimajor_m"].max() == candidates["semimajor_m"].max()
    assert catalog["size_bin"].nunique() == 5
    assert catalog["axis_ratio_bin"].nunique() == 4
    assert catalog["orientation_bin"].nunique() == 12
    assert catalog["offset_bin"].nunique() == 4
