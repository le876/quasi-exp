from pathlib import Path

import numpy as np

from quasi_exp.teacher.ellipsoidal_shell import EllipsoidSpec
from scripts.analysis.run_bacra_v13_3d_path_generalization import (
    PATH_SPECS,
    _registered_catalog,
    build_path,
)
from scripts.analysis.run_bacra_v13_ellipsoidal_shell_atlas import load_config


def _spec() -> EllipsoidSpec:
    return EllipsoidSpec(
        center_m=np.asarray((1.0, -0.1, 0.2)),
        semiaxes_m=np.asarray((0.12, 0.08, 0.055)),
        rotation=np.eye(3),
    )


def test_registered_paths_are_closed_nonplanar_and_inside_shell() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/bacra_v13_ellipsoidal_shell_atlas.yaml",
        "formal",
    )["trajectory_3d"]
    catalog, points = _registered_catalog(_spec(), int(config["phase_count"]))
    assert len(PATH_SPECS) == config["path_count"] == len(catalog) == 16
    assert catalog["shape"].nunique() == config["shape_count"] == 4
    assert sorted(catalog["rho_mm"].unique()) == config["radial_amplitudes_mm"]
    assert (catalog["nonplanarity"] >= config["nonplanarity_min"]).all()
    assert points["rho_mm"].abs().max() <= config["shell_half_thickness_mm"]
    for path_spec in PATH_SPECS:
        path = build_path(_spec(), path_spec, 720)
        unit = np.asarray(path["unit"])
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1.0e-12)
        assert np.linalg.norm(np.asarray(path["target"])[0] - np.asarray(path["target"])[-1]) < 0.01


def test_3d_path_runner_is_locked_evaluation_only() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/analysis/run_bacra_v13_3d_path_generalization.py").read_text()
    assert "model bytes changed after lock" in text
    assert ".fit(" not in text
    assert text.index('"path_catalog.parquet"') < text.index("import tensorflow as tf")
    assert "all_paths_nonplanar" in text
    assert "targets_within_accepted_half_thickness" in text
    assert "return 0 if bool(report.get(\"gate_pass\")) else 2" in text
