from __future__ import annotations

from pathlib import Path

import numpy as np

from quasi_exp.teacher.ellipsoidal_shell import (
    EllipsoidSpec,
    build_shell_mesh,
    sample_shell_cells,
)
from scripts.analysis.run_bacra_v13_ellipsoidal_shell_atlas import (
    _parent_prediction_metrics,
    load_config,
)


class _LinearEnvironment:
    bounds = np.asarray([[-1.0, 1.0]] * 6)

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3]

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        return np.hstack((np.eye(3), np.zeros((3, 3))))


def test_v13_config_locks_registered_meshes_thickness_and_known_chart_student() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/bacra_v13_ellipsoidal_shell_atlas.yaml"
    smoke = load_config(config_path, "smoke")
    pilot = load_config(config_path, "pilot")
    formal = load_config(config_path, "formal")
    assert smoke["mesh"]["subdivisions"] == 1
    assert pilot["mesh"]["subdivisions"] == 3
    assert formal["mesh"]["subdivisions"] == 4
    assert pilot["mesh"]["radial_levels_mm"] == [-10.0, -5.0, 0.0, 5.0, 10.0]
    assert formal["mesh"]["radial_levels_mm"] == [
        -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0
    ]
    assert formal["dense"]["row_count"] == 200000
    assert formal["gates"]["thickness_p50_min_mm"] == 15.0
    assert formal["student"]["hidden_units"] == [128, 128, 64]
    assert formal["student"]["automatic_chart_classifier"] is False


def test_six_parent_prediction_agreement_is_exact_for_linear_canonical_field() -> None:
    spec = EllipsoidSpec(np.zeros(3), np.asarray((0.05, 0.04, 0.03)), np.eye(3))
    mesh = build_shell_mesh(spec, subdivisions=0)
    levels = np.asarray((-0.005, 0.0, 0.005))
    mask = np.ones((2, len(mesh.faces)), dtype=bool)
    samples = sample_shell_cells(mesh, levels, mask, 50, seed=17)
    chart_ids = np.full(50, "chart_00", dtype=object)
    beta_by_key = {}
    for level_index, rho in enumerate(levels):
        xyz = mesh.offset_vertices(float(rho))
        for vertex_id, point in enumerate(xyz):
            beta_by_key[(level_index, vertex_id, "chart_00")] = np.r_[point, np.zeros(3)]
    gap_p95, gap_max, parent_count = _parent_prediction_metrics(
        _LinearEnvironment(), mesh, levels, samples, chart_ids, beta_by_key
    )
    np.testing.assert_allclose(gap_p95, 0.0, atol=1.0e-5)
    np.testing.assert_allclose(gap_max, 0.0, atol=1.0e-5)
    np.testing.assert_array_equal(parent_count, 6)


def test_runner_declares_shell_gate_before_any_student_claim() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (
        root / "scripts/analysis/run_bacra_v13_ellipsoidal_shell_atlas.py"
    ).read_text(encoding="utf-8")
    assert "same_chart_multi_parent_physical_radial_shell" in runner
    assert "uniform_accepted_prism_volume_with_consistent_known_chart_labels" in runner
    assert "student_not_used_to_upgrade_shell" in runner
    assert 'student_training_status="not_started_by_primary_shell_pipeline"' in runner
    assert "automatic_chart_classifier_disabled" in runner
