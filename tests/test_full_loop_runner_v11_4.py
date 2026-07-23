from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def _runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/analysis/run_generalized_ellipse_full_loop_v11_4.py"
    )
    spec = importlib.util.spec_from_file_location("full_loop_runner_v11_4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_freezes_strict_gate_and_gate_driven_matrix() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    config = runner.load_config(
        source_root / "configs/generalized_ellipse_region_v11_full_loop.yaml",
        preset="formal",
    )

    assert config["protocol_id"] == (
        "generalized-ellipse-region-v11.4-full-loop-feasible-branch"
    )
    assert config["source_protocol_id"] == (
        "generalized-ellipse-region-v11.3-branch-identity"
    )
    assert config["root_fiber"]["nested_seed_counts"] == [64, 256, 1024]
    assert config["pilot_matrix"]["root_seed_counts"] == [256, 1024]
    assert config["pilot_matrix"]["candidate_caps"] == [16, 32, 64]
    assert config["pilot_matrix"]["parallel_workers"] == 4
    assert config["pilot_matrix"]["max_parallel_workers"] == 8
    assert config["pilot_matrix"]["per_worker_blas_threads"] == 1
    assert config["formal_audit"]["parallel_workers"] == 2
    assert config["formal_audit"]["per_worker_blas_threads"] == 1
    assert config["hard_feasibility"] == {
        "residual_max_mm": 3.0,
        "joint_margin_min_deg": 1.5,
        "distinct_threshold_deg": 0.25,
    }
    assert config["downstream"]["static_mapping_conflict_policy"] == "fail_closed"


def test_smoke_deep_merge_preserves_artifact_contract() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    config = runner.load_config(
        source_root / "configs/generalized_ellipse_region_v11_full_loop.yaml",
        preset="smoke",
    )

    assert config["phase_counts"] == {"pilot": 12, "formal": 24}
    assert config["root_fiber"]["nested_seed_counts"] == [8, 16]
    assert config["pilot_matrix"]["root_seed_counts"] == [16]
    assert config["pilot_matrix"]["candidate_caps"] == [4]
    assert config["pilot_matrix"]["parallel_workers"] == 2
    assert set(runner.STAGE_DIRS) == {
        "protocol",
        "root_fiber",
        "viability",
        "pilot",
        "formal",
        "bridge",
    }
    required = set(runner.CANDIDATE_EVIDENCE_COLUMNS)
    assert {
        "candidate_source",
        "candidate_beta",
        "corrector_success",
        "residual_mm",
        "joint_margin_deg",
        "within_bounds",
        "reference_gap_deg",
        "previous_gap_deg",
        "trust_pass",
        "kappa",
    }.issubset(required)


def test_empty_root_representative_inventory_keeps_csv_schema(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    path = tmp_path / "root_representatives.csv"

    runner.representative_inventory_table([]).to_csv(path, index=False)
    restored = pd.read_csv(path)

    assert restored.empty
    assert restored.columns.tolist() == list(
        runner.ROOT_REPRESENTATIVE_COLUMNS
    )


def test_pilot_parallel_tasks_are_stable_candidate_root_slices() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    config = runner.load_config(
        source_root / "configs/generalized_ellipse_region_v11_full_loop.yaml",
        preset="formal",
    )

    tasks = runner._pilot_task_specs(config)  # noqa: SLF001

    assert [
        (task["candidate_id"], task["root_seed_count"], task["include_e0"])
        for task in tasks
    ] == [
        ("A4_A2_178_r0_reverse_c0045", 256, True),
        ("A4_A2_178_r0_reverse_c0045", 1024, False),
        ("A4_A2_143_r1_reverse_c0045", 256, True),
        ("A4_A2_143_r1_reverse_c0045", 1024, False),
    ]
    assert len({task["task_id"] for task in tasks}) == 4


def test_formal_parallel_tasks_are_stable_candidate_slices() -> None:
    runner = _runner_module()
    source_root = Path(__file__).resolve().parents[1]
    config = runner.load_config(
        source_root / "configs/generalized_ellipse_region_v11_full_loop.yaml",
        preset="formal",
    )

    tasks = runner._formal_task_specs(config)  # noqa: SLF001

    assert [
        (task["candidate_id"], task["candidate_offset"]) for task in tasks
    ] == [
        ("A4_A2_178_r0_reverse_c0045", 0),
        ("A4_A2_143_r1_reverse_c0045", 1),
    ]
    assert len({task["task_id"] for task in tasks}) == 2


def test_pilot_ranking_merge_is_independent_of_worker_completion_order() -> None:
    runner = _runner_module()
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "B",
                "experiment": "E2",
                "root_seed_count": 1024,
                "candidate_cap": 64,
                "edge_limit_deg": 3.0,
            },
            {
                "candidate_id": "A",
                "experiment": "E1",
                "root_seed_count": 256,
                "candidate_cap": 16,
                "edge_limit_deg": 1.0,
            },
            {
                "candidate_id": "A",
                "experiment": "E0",
                "root_seed_count": 64,
                "candidate_cap": 0,
                "edge_limit_deg": np.nan,
            },
        ]
    )
    config = {
        "candidates": ["A", "B"],
        "pilot_matrix": {
            "root_seed_counts": [256, 1024],
            "candidate_caps": [16, 32, 64],
            "edge_limits_deg": [1.0, 2.0, 3.0],
        },
    }

    first = runner._sort_pilot_ranking(rows, config)  # noqa: SLF001
    second = runner._sort_pilot_ranking(  # noqa: SLF001
        rows.iloc[::-1].reset_index(drop=True), config
    )

    pd.testing.assert_frame_equal(first, second)
    assert first[["candidate_id", "experiment"]].values.tolist() == [
        ["A", "E0"],
        ["A", "E1"],
        ["B", "E2"],
    ]


def test_full_loop_viable_root_restriction_stays_empty_without_survivor() -> None:
    runner = _runner_module()
    layers = [np.zeros((2, 6)), np.zeros((1, 6)), np.zeros((1, 6))]
    residuals = [np.zeros(2), np.zeros(1), np.zeros(1)]
    root_nodes = pd.DataFrame(
        {
            "origin_root_seed_idx": [3.0, 4.0],
        }
    )
    root_candidates = pd.DataFrame(
        {
            f"beta{joint}_rad": np.zeros(5)
            for joint in range(1, 7)
        }
    )

    restricted, restricted_residuals = runner._restricted_root_layers(  # noqa: SLF001
        layers,
        residuals,
        root_nodes=root_nodes,
        mode="roots",
        root_candidates=root_candidates,
        allowed_root_indices=set(),
        beta_weights=np.ones(6),
    )

    assert restricted[0].shape == (0, 6)
    assert restricted_residuals[0].shape == (0,)


def test_formal_checkpoint_rejects_tampered_candidate_artifact(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    artifact = tmp_path / "cycle.txt"
    artifact.write_text("frozen", encoding="utf-8")
    manifest = runner._directory_artifact_manifest(tmp_path)  # noqa: SLF001

    assert runner._artifact_manifest_is_valid(  # noqa: SLF001
        tmp_path, manifest
    )
    artifact.write_text("tampered", encoding="utf-8")
    assert not runner._artifact_manifest_is_valid(  # noqa: SLF001
        tmp_path, manifest
    )


def test_resealed_tube_gate_validates_after_audited_surface_selection(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner_module()
    output = tmp_path / "downstream"
    tube = output / runner.v11.STAGE_DIRS["tube"]
    tube.mkdir(parents=True)
    (tube / "selected_surface.parquet").write_bytes(b"new-audited-surface")
    original = {
        "checks": {
            "minimum_0p5_by_0p5_tube_passes": True,
            "at_least_one_dense_tube_passes": True,
        },
        "gate_pass": True,
        "artifact_sha256": {"selected_surface.parquet": "stale"},
        "cache_fingerprint": "stale",
        "anchors_screened": 2,
    }
    monkeypatch.setattr(
        runner.v11,
        "_stage_cache_fingerprint",
        lambda **_kwargs: "fresh-fingerprint",
    )

    runner._rewrite_downstream_tube_gate(  # noqa: SLF001
        tube_gate_path=tube / "gate.json",
        original_tube_gate=original,
        downstream_config={},
        source_root=Path(__file__).resolve().parents[1],
        downstream_output=output,
        selected_tube={"anchor_id": "A", "radial_radius_mm": 0.5},
        half_mm_full_audit_pass=True,
    )

    assert (
        runner.v11.read_valid_gate(
            tube / "gate.json",
            expected_fingerprint="fresh-fingerprint",
        )
        is not None
    )


def test_root_evidence_keeps_selected_and_primary_metrics_separate() -> None:
    runner = _runner_module()
    payload = runner.root_evidence_payload(
        candidate_id="A",
        phase_idx=3,
        selected_row={
            "residual_mm": 0.2,
            "joint_margin_deg": 2.5,
            "kappa": 4.0,
        },
        primary_row={
            "residual_mm": 1.2,
            "joint_margin_deg": 1.6,
            "kappa": 14.0,
        },
        selected_beta=[0.0] * 6,
        primary_beta=[1.0] * 6,
    )
    assert payload["selected_root_residual_mm"] == 0.2
    assert payload["selected_root_margin_deg"] == 2.5
    assert payload["selected_root_kappa"] == 4.0
    assert payload["primary_root_residual_mm"] == 1.2
    assert payload["primary_root_margin_deg"] == 1.6
    assert payload["primary_root_kappa"] == 14.0
    assert payload["selected_root_beta"] != payload["primary_root_beta"]


def test_formal_audit_uses_phase_and_degree_cut_union() -> None:
    runner = _runner_module()
    variants = runner._audit_variant_specs(  # noqa: SLF001
        720,
        [0, 45, 90, 135],
        [0, 45, 90, 135],
    )
    cuts = {int(row[2]) for row in variants}
    assert cuts == {0, 45, 90, 135, 180, 270}


def test_downstream_bridge_preserves_every_passing_anchor(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner_module()
    project_root = tmp_path / "project"
    branch_output = tmp_path / "branch"
    downstream_output = tmp_path / "downstream"
    source_root = Path(__file__).resolve().parents[1]
    candidate_ids = ["A", "B"]
    for offset, candidate_id in enumerate(candidate_ids):
        task_path = (
            project_root
            / "source_anchor"
            / "verify_tasks"
            / f"{candidate_id}.json"
        )
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(
            json.dumps(
                {
                    "family": {
                        "family_id": candidate_id,
                        "center_m": [0.0, 0.0, 0.0],
                        "major_axis_m": [1.0, 0.0, 0.0],
                        "minor_axis_m": [0.0, 1.0, 0.0],
                        "major_radius_m": 0.05,
                        "minor_radius_m": 0.025,
                    }
                }
            ),
            encoding="utf-8",
        )
        cycle_path = (
            branch_output
            / runner.STAGE_DIRS["formal"]
            / candidate_id
            / "selected_cycle"
            / "cycle.parquet"
        )
        cycle_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"phase_idx": np.arange(4)})
        for joint, name in enumerate(runner.BETA_COLUMNS):
            frame[name] = float(offset + joint)
        frame.to_parquet(cycle_path, index=False)
    formal_gate = branch_output / runner.STAGE_DIRS["formal"] / "gate.json"
    formal_gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner.v11,
        "_stage_cache_fingerprint",
        lambda **_kwargs: "test-cache",
    )

    gate = runner._bootstrap_downstream_anchor(  # noqa: SLF001
        config={
            "source_anchor_root": "source_anchor",
            "protocol_id": "test-v11.4",
            "phase_counts": {"formal": 4},
        },
        downstream_config={},
        source_root=source_root,
        project_root=project_root,
        branch_output=branch_output,
        downstream_output=downstream_output,
        candidate_ids=candidate_ids,
    )

    payload = json.loads(
        (
            downstream_output
            / runner.v11.STAGE_DIRS["anchor"]
            / "passing_anchors.json"
        ).read_text(encoding="utf-8")
    )
    assert gate["gate_pass"] is True
    assert [row["candidate_id"] for row in payload] == candidate_ids


def test_whole_curve_correction_jointly_preserves_hard_bounds() -> None:
    runner = _runner_module()

    class LinearEnvironment:
        bounds = np.deg2rad(np.asarray([[-30.0, 30.0]] * 6))

        def fk(self, beta):
            return np.asarray(beta, dtype=float).reshape(-1, 6)[:, :3]

        def jacobian(self, _beta):
            return np.concatenate(
                [np.eye(3), np.zeros((3, 3))], axis=1
            )

        def theta(self, beta):
            return np.repeat(
                np.asarray(beta, dtype=float).reshape(-1, 6), 5, axis=1
            )

    source_root = Path(__file__).resolve().parents[1]
    config = runner.load_config(
        source_root / "configs/generalized_ellipse_region_v11_full_loop.yaml",
        preset="smoke",
    )
    phase = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    target = np.column_stack(
        [
            0.01 * np.sin(phase),
            0.01 * np.cos(phase),
            0.005 * np.sin(2.0 * phase),
        ]
    )
    corrected, evidence = runner._correct_whole_cycle(  # noqa: SLF001
        config=config,
        environment=LinearEnvironment(),
        target=target,
        initial_beta=np.zeros((len(target), 6)),
        solver_seed=123,
    )
    metrics = runner._cycle_metrics(  # noqa: SLF001
        LinearEnvironment(),
        target,
        corrected,
        beta_weights=config["cycle"]["beta_weights"],
    )

    assert corrected.shape == (12, 6)
    assert metrics["residual_max_mm"] <= 3.0
    assert metrics["joint_margin_min_deg"] >= 1.5
    assert evidence["optimizer"].nunique() == 1
    assert (
        evidence["optimizer"].iloc[0]
        == "bounded_whole_trajectory_penalty_least_squares"
    )


def test_real_cli_smoke_writes_complete_branch_decision(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "v11_4_smoke"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                source_root
                / "scripts/analysis/run_generalized_ellipse_full_loop_v11_4.py"
            ),
            "--preset",
            "smoke",
            "--stage",
            "branch",
            "--project-root",
            "/mnt/ML_projects/quasi_exp",
            "--output",
            str(output),
            "--workers",
            "2",
        ],
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout[-4000:]
    marker = json.loads(
        (output / "FULL_LOOP_EXPERIMENT_COMPLETED.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["full_loop_experiment_complete"] is True
    assert set(marker["decision_by_candidate"]) == {
        "A4_A2_178_r0_reverse_c0045",
        "A4_A2_143_r1_reverse_c0045",
    }
    parallel_manifest = json.loads(
        (
            output
            / "03_pilot_matrix"
            / "pilot_parallel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert parallel_manifest["status"] == "completed"
    assert parallel_manifest["requested_workers"] == 2
    assert parallel_manifest["effective_workers"] == 2
    assert parallel_manifest["independent_task_count"] == 2
    assert parallel_manifest["per_worker_blas_threads"] == 1
    assert {task["status"] for task in parallel_manifest["tasks"]} == {
        "completed"
    }
    assert all(
        task["peak_rss_bytes"] > 0 for task in parallel_manifest["tasks"]
    )
    formal_manifest = json.loads(
        (
            output
            / "04_formal"
            / "formal_parallel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert formal_manifest["status"] == "completed"
    assert formal_manifest["effective_workers"] == 2
    assert formal_manifest["independent_task_count"] == 2
    assert {task["status"] for task in formal_manifest["tasks"]} == {
        "completed"
    }
    assert not (output / "05_downstream_bridge").exists()


def test_parallel_pilot_ranking_matches_single_worker(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    runner_path = (
        source_root
        / "scripts/analysis/run_generalized_ellipse_full_loop_v11_4.py"
    )
    outputs = {}
    for workers in (1, 2):
        output = tmp_path / f"workers_{workers}"
        completed = subprocess.run(
            [
                sys.executable,
                str(runner_path),
                "--preset",
                "smoke",
                "--stage",
                "branch",
                "--project-root",
                "/mnt/ML_projects/quasi_exp",
                "--output",
                str(output),
                "--workers",
                str(workers),
            ],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=240,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout[-4000:]
        outputs[workers] = output

    single = pd.read_csv(
        outputs[1] / "03_pilot_matrix" / "pilot_matrix.csv"
    )
    parallel = pd.read_csv(
        outputs[2] / "03_pilot_matrix" / "pilot_matrix.csv"
    )
    pd.testing.assert_frame_equal(single, parallel)
    assert json.loads(
        (
            outputs[1]
            / "03_pilot_matrix"
            / "selected_formal_method.json"
        ).read_text(encoding="utf-8")
    ) == json.loads(
        (
            outputs[2]
            / "03_pilot_matrix"
            / "selected_formal_method.json"
        ).read_text(encoding="utf-8")
    )
    for candidate_id in (
        "A4_A2_178_r0_reverse_c0045",
        "A4_A2_143_r1_reverse_c0045",
    ):
        single_report = json.loads(
            (
                outputs[1]
                / "04_formal"
                / candidate_id
                / "formal_report.json"
            ).read_text(encoding="utf-8")
        )
        parallel_report = json.loads(
            (
                outputs[2]
                / "04_formal"
                / candidate_id
                / "formal_report.json"
            ).read_text(encoding="utf-8")
        )
        for report in (single_report, parallel_report):
            report.pop("candidate_fingerprint")
            report.pop("input_sha256")
            report.pop("candidate_artifact_sha256")
        assert single_report == parallel_report
    assert json.loads(
        (outputs[1] / "FULL_LOOP_EXPERIMENT_COMPLETED.json").read_text(
            encoding="utf-8"
        )
    ) == json.loads(
        (outputs[2] / "FULL_LOOP_EXPERIMENT_COMPLETED.json").read_text(
            encoding="utf-8"
        )
    )
