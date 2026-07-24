from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
    assert config["tube"]["parallel_workers"] == 2
    assert config["core"]["parallel_workers"] == 2
    assert config["pilot"]["parallel_workers"] == 2
    assert config["formal"]["parallel_workers"] == 2
    assert config["training"]["parallel_workers"] == 2
    assert config["evaluate"]["parallel_workers"] == 2
    formal_config = runner.load_protocol_config(
        source_root / "configs/generalized_ellipse_region_v11.yaml",
        preset="formal",
    )
    assert formal_config["tube"]["parallel_workers"] == 8
    assert formal_config["pilot"]["parallel_workers"] == 8
    assert formal_config["formal"]["parallel_workers"] == 8
    assert formal_config["training"]["parallel_workers"] == 5
    assert formal_config["evaluate"]["parallel_workers"] == 8

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


def test_subprocess_scheduler_runs_two_workers_and_preserves_task_order(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner_module()
    task_dir = tmp_path / "_parallel/tasks"
    task_dir.mkdir(parents=True)
    task_files = [task_dir / f"task_{index}.json" for index in range(3)]
    for task_file in task_files:
        task_file.write_text("{}", encoding="utf-8")

    state = {"alive": set(), "max_alive": 0, "environments": []}

    class FakeProcess:
        next_pid = 1000

        def __init__(self, command, **kwargs) -> None:
            del command
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = None
            self.poll_count = 0
            self.is_first = self.pid == 1000
            state["alive"].add(self.pid)
            state["max_alive"] = max(
                state["max_alive"], len(state["alive"])
            )
            state["environments"].append(kwargs["env"])

        def poll(self):
            self.poll_count += 1
            if self.is_first and self.poll_count == 1:
                return None
            self.returncode = 0
            state["alive"].discard(self.pid)
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        runner, "_read_process_usage", lambda _pid: (0.01, 1024)
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    report = runner._run_subprocess_tasks(
        task_files, worker_name="tube-surface", max_workers=2
    )

    assert state["max_alive"] == 2
    assert report["effective_workers"] == 2
    assert report["task_count"] == 3
    assert [row["task_id"] for row in report["tasks"]] == [
        "task_0",
        "task_1",
        "task_2",
    ]
    assert all(
        environment[variable] == "1"
        for environment in state["environments"]
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    )
    expected_src = str(Path(__file__).resolve().parents[1] / "src")
    assert all(
        environment["PYTHONPATH"].split(":")[0] == expected_src
        for environment in state["environments"]
    )
    assert all(
        environment["MPLCONFIGDIR"] == "/tmp/matplotlib-v11"
        for environment in state["environments"]
    )


def test_subprocess_scheduler_preserves_failed_batch_resource_report(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner_module()
    task_file = tmp_path / "_parallel/tasks/failing_task.json"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("{}", encoding="utf-8")

    class FailedProcess:
        pid = 2000
        returncode = 7

        def __init__(self, _command, **_kwargs) -> None:
            pass

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FailedProcess)
    monkeypatch.setattr(
        runner, "_read_process_usage", lambda _pid: (0.25, 4096)
    )

    with pytest.raises(runner.ParallelWorkerError) as captured:
        runner._run_subprocess_tasks(
            [task_file], worker_name="tube-surface", max_workers=2
        )

    report = captured.value.report
    assert report["status"] == "failed"
    assert report["task_count"] == 1
    assert report["tasks"][0]["exit_code"] == 7
    assert report["tasks"][0]["cpu_seconds"] == 0.25
    assert report["peak_concurrent_worker_rss_bytes"] == 4096
    assert report["failures"][0]["task_id"] == "failing_task"


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
    def value(config: dict, path: str) -> float:
        current: object = config
        for part in path.split("."):
            current = current[int(part)] if isinstance(current, list) else current[part]
        return float(current)

    upper_limit_paths = (
        "gates.teacher_surface.residual_p95_mm",
        "gates.teacher_surface.residual_max_mm",
        "gates.teacher_surface.phase_beta_rms_p95_deg",
        "gates.teacher_surface.delta_beta_rms_max_deg",
        "gates.teacher_surface.acceleration_beta_rms_p95_deg",
        "gates.teacher_surface.seam_beta_rms_max_deg",
        "gates.teacher_surface.surface_edge_beta_rms_p95_deg",
        "gates.teacher_surface.surface_laplacian_beta_rms_p95_deg",
        "gates.teacher_surface.surface_block_update_rms_max_deg",
        "gates.repeatability.repeat_beta_rms_p95_deg",
        "gates.repeatability.reverse_cut_beta_rms_p95_deg",
        "gates.local_consistency.gap_5mm_p95_deg",
        "gates.local_consistency.gap_10mm_p95_deg",
        "gates.conditioning.kappa_p95_max",
        "gates.conflicts.beta_gap_threshold_deg",
        "gates.coverage.nearest_p95_mm",
        "gates.coverage.nearest_max_mm",
        "gates.student.interpolation_p95_mm",
        "gates.student.near_ood_p95_mm",
        "gates.student.absolute_max_mm",
        "gates.student.relative_max_fraction",
        "gates.student.per_axis_mean_mm",
        "representation.chart_count_range.1",
        "representation.ambiguous_voxel_max",
    )
    lower_limit_paths = (
        "gates.teacher_surface.success_rate",
        "gates.teacher_surface.joint_margin_min_deg",
        "gates.conditioning.sigma_min_p05_min",
        "gates.conflicts.xyz_radius_mm",
        "representation.chart_count_range.0",
        "representation.chart_repeat_ari_min",
        "representation.chart_xyz_macro_f1_min",
    )
    for path in upper_limit_paths:
        assert value(relaxed, path) == 2.0 * value(strict, path), path
    for path in lower_limit_paths:
        assert value(relaxed, path) == value(strict, path) / 2.0, path
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
