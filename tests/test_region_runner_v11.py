from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/run_generalized_ellipse_region_v11.py"
    spec = importlib.util.spec_from_file_location("region_runner_v11", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_stage_freezes_catalog_splits_and_manifest(tmp_path: Path) -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    project_root = runner.project_root_from(source_root)
    config = runner.load_protocol_config(
        source_root / "configs/generalized_ellipse_region_v11.yaml",
        preset="smoke",
    )

    report = runner.run_protocol_stage(
        config=config,
        source_root=source_root,
        project_root=project_root,
        output=tmp_path,
    )

    assert report["gate_pass"] is True
    assert all(type(value) is bool for value in report["checks"].values())
    catalog = pd.read_csv(tmp_path / "00_protocol/family_catalog.csv")
    splits = pd.read_csv(tmp_path / "00_protocol/split_manifest.csv")
    manifest = json.loads((tmp_path / "00_protocol/artifact_manifest.json").read_text())
    assert len(catalog) == 72
    assert len(splits) == 24
    assert catalog["family_id"].is_unique
    assert len(manifest["protocol_sha256"]) == 64
    assert manifest["source_fixed_point"] == config["source_fixed_point"]


def test_stage_cache_recomputes_boolean_gate_instead_of_trusting_cached_flag(tmp_path: Path) -> None:
    runner = _runner_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "gate_pass": True,
                "checks": {"residual": False, "margin": True},
            }
        ),
        encoding="utf-8",
    )

    assert runner.read_valid_gate(gate) is None


def test_stage_cache_is_invalidated_when_a_hashed_artifact_changes(tmp_path: Path) -> None:
    runner = _runner_module()
    artifact = tmp_path / "surface.parquet"
    artifact.write_bytes(b"frozen")
    gate_path = tmp_path / "gate.json"
    runner._write_gate(gate_path, checks={"surface_complete": True})

    assert runner.read_valid_gate(gate_path) is not None
    artifact.write_bytes(b"mutated")
    assert runner.read_valid_gate(gate_path) is None


def test_stage_cache_is_invalidated_when_execution_fingerprint_changes(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    gate_path = tmp_path / "gate.json"
    runner._write_gate(
        gate_path,
        checks={"surface_complete": True},
        cache_fingerprint="configuration-a",
    )

    assert (
        runner.read_valid_gate(
            gate_path, expected_fingerprint="configuration-a"
        )
        is not None
    )
    assert (
        runner.read_valid_gate(
            gate_path, expected_fingerprint="configuration-b"
        )
        is None
    )


def test_relaxed_2x_protocol_doubles_upper_gates_and_halves_lower_gates() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    strict = runner.load_protocol_config(
        source_root / "configs/generalized_ellipse_region_v11.yaml",
        preset="formal",
    )
    relaxed = runner.load_protocol_config(
        source_root / "configs/generalized_ellipse_region_v11_relaxed2x.yaml",
        preset="formal",
    )

    assert relaxed["protocol_id"] == "generalized-ellipse-region-v11.2-relaxed2x"
    assert relaxed["output_root"] == "runs/generalized_ellipse_region_v11_relaxed2x"
    assert relaxed["gates"]["teacher_surface"]["residual_max_mm"] == 2.0 * strict["gates"]["teacher_surface"]["residual_max_mm"]
    assert relaxed["gates"]["teacher_surface"]["success_rate"] == strict["gates"]["teacher_surface"]["success_rate"] / 2.0
    assert relaxed["gates"]["teacher_surface"]["joint_margin_min_deg"] == strict["gates"]["teacher_surface"]["joint_margin_min_deg"] / 2.0
    assert relaxed["gates"]["conditioning"]["sigma_min_p05_min"] == strict["gates"]["conditioning"]["sigma_min_p05_min"] / 2.0
    assert relaxed["gates"]["conditioning"]["kappa_p95_max"] == 2.0 * strict["gates"]["conditioning"]["kappa_p95_max"]
    assert relaxed["gates"]["conflicts"]["xyz_radius_mm"] == strict["gates"]["conflicts"]["xyz_radius_mm"] / 2.0
    assert relaxed["gates"]["conflicts"]["beta_gap_threshold_deg"] == 2.0 * strict["gates"]["conflicts"]["beta_gap_threshold_deg"]
    assert relaxed["gates"]["student"]["required_seed_passes"] == 2
    assert relaxed["gates"]["student"]["seed_count"] == strict["gates"]["student"]["seed_count"]
    assert relaxed["anchor"] == strict["anchor"]
    assert relaxed["formal"] == strict["formal"]
    assert runner._centerline_thresholds(relaxed)["delta_beta_rms_max_deg"] == 4.0


def test_anchor_selection_margin_uses_protocol_threshold_not_a_literal() -> None:
    runner = _runner_module()

    assert runner._anchor_margin_passes(0.75, {"joint_margin_min_deg": 0.75}) is True
    assert runner._anchor_margin_passes(0.749, {"joint_margin_min_deg": 0.75}) is False


def test_interpolation_families_are_midpoints_of_training_families() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    project_root = runner.project_root_from(source_root)
    config = runner.load_protocol_config(
        source_root / "configs/generalized_ellipse_region_v11.yaml",
        preset="smoke",
    )
    baseline = runner._baseline_family(project_root, config)
    catalog = runner.generate_family_catalog(
        baseline, seed=int(config["seeds"]["family"])
    )
    train_ids = catalog.primary_ids("train")

    interpolated = runner._interpolation_families(catalog, train_ids, count=5)

    assert len(interpolated) == 5
    assert all(family.metadata["construction"] == "train_family_midpoint_slerp" for family in interpolated)
    assert all(family.family_id not in set(catalog.frame["family_id"]) for family in interpolated)


def test_near_ood_families_are_outside_frozen_center_and_plane_boxes() -> None:
    runner = _runner_module()
    anchor = runner.EllipseFamilySpec(
        family_id="anchor",
        center_m=np.zeros(3),
        major_direction=np.asarray([1.0, 0.0, 0.0]),
        minor_direction=np.asarray([0.0, 1.0, 0.0]),
        major_semiaxis_m=0.5,
        minor_semiaxis_m=0.17,
    )
    adversarial_inside = runner.EllipseFamilySpec(
        family_id="inside",
        center_m=np.asarray([-0.01, 0.01, 0.005]),
        major_direction=np.asarray([1.0, 0.0, 0.0]),
        minor_direction=np.asarray([0.0, 1.0, 0.0]),
        major_semiaxis_m=0.48,
        minor_semiaxis_m=0.16,
    )

    center_ood = runner._category_families(
        adversarial_inside, category="center_ood", domain_anchor=anchor
    )
    plane_ood = runner._category_families(
        adversarial_inside, category="plane_ood", domain_anchor=anchor
    )

    assert np.dot(center_ood.center_m - anchor.center_m, anchor.major_direction) > 0.01
    plane_angle_deg = np.rad2deg(
        np.arccos(np.clip(np.dot(plane_ood.plane_normal, anchor.plane_normal), -1.0, 1.0))
    )
    assert plane_angle_deg > 3.0
