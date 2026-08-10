from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    analysis_dir = REPO_ROOT / "scripts" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    path = analysis_dir / "run_true_ellipse_standard_domain_gate_sweep_v7.py"
    spec = importlib.util.spec_from_file_location(
        "run_true_ellipse_standard_domain_gate_sweep_v7", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complete_radius_report(
    *, kappa_p95: float = 190.0, radius_mm: float = 105.0
) -> dict:
    jobs = []
    for predictor in ("parent_copy", "radial_secant"):
        for cut in (0, 90, 180, 270):
            jobs.append(
                {
                    "cut_idx": cut,
                    "radial_predictor_type": predictor,
                    "selected": True,
                    "branch_gate_pass": True,
                    "canonical_gate_pass": True,
                    "downstream_admission_gate_pass": True,
                    "job_margin_gate_pass": True,
                    "sigma3_p05_m": 0.02,
                    "kappa_p95": kappa_p95,
                }
            )
    return {
        "task_fingerprint": "radius-task",
        "radius_mm": radius_mm,
        "geometry": {"target_geometry_gate_pass": True},
        "required_cuts": [0, 90, 180, 270],
        "required_predictors": ["parent_copy", "radial_secant"],
        "jobs": jobs,
        "cut_invariance": {"cut_invariance_gate_pass": True},
        "deterministic_repeatability": {"deterministic_exact_gate_pass": True},
    }


def test_cli_registers_exact_candidate_thresholds_and_full_downstream_job_sweep() -> None:
    mod = _load_module()
    args = mod.parse_args([])

    assert mod.parse_registered_thresholds(args.kappa_thresholds) == (
        150.0,
        200.0,
        250.0,
        300.0,
        400.0,
    )
    downstream = mod.build_downstream_radial_args(args)
    assert downstream.radial_job_gate_mode == "downstream_admission"
    assert downstream.stop_after_first_failed_job is False
    assert downstream.share_rescue_cache_across_cuts is True
    assert Path(downstream.out_dir) != Path(mod.v7.DEFAULT_OUT_DIR)

    with pytest.raises(ValueError, match="registered kappa thresholds"):
        mod.parse_registered_thresholds("150,250,400")


def test_candidate_radius_policy_requires_all_eight_jobs_cut_and_repeatability() -> None:
    mod = _load_module()
    report = _complete_radius_report(kappa_p95=190.0)

    passed = mod.evaluate_candidate_radius_policy(report, kappa_threshold=200.0)
    strict_rejected = mod.evaluate_candidate_radius_policy(report, kappa_threshold=150.0)
    missing_report = dict(report, jobs=report["jobs"][:-1])
    missing = mod.evaluate_candidate_radius_policy(
        missing_report, kappa_threshold=200.0
    )

    assert passed["required_job_count"] == 8
    assert passed["selected_job_count"] == 8
    assert passed["candidate_radius_policy_gate_pass"] is True
    assert strict_rejected["candidate_radius_policy_gate_pass"] is False
    assert len(strict_rejected["failed_jobs"]) == 8
    assert missing["candidate_radius_policy_gate_pass"] is False
    assert missing["missing_jobs"] == ["270:radial_secant"]


def test_candidate_policy_replays_materializable_stage_instead_of_legacy_selected_stage() -> None:
    mod = _load_module()
    report = _complete_radius_report(kappa_p95=415.0)
    for job in report["jobs"]:
        job["stage_history"] = [
            {
                "stage": "conservative_1",
                "branch_gate_pass": True,
                "canonical_gate_pass": True,
                "downstream_admission_gate_pass": True,
                "job_margin_gate_pass": True,
                "sigma3_p05_m": 0.018,
                "kappa_p95": 338.0,
                "stage_acceptance_gate_pass": True,
            },
            {
                "stage": "conservative_3",
                "branch_gate_pass": True,
                "canonical_gate_pass": True,
                "downstream_admission_gate_pass": True,
                "job_margin_gate_pass": True,
                "sigma3_p05_m": 0.015,
                "kappa_p95": 415.0,
                "stage_acceptance_gate_pass": False,
            },
        ]

    decision = mod.evaluate_candidate_radius_policy(
        report, kappa_threshold=400.0
    )

    assert decision["candidate_radius_policy_gate_pass"] is True
    assert decision["candidate_paths_materialized"] is False
    assert {job["selected_stage"] for job in decision["jobs"]} == {
        "conservative_1"
    }
    assert all(job["kappa_p95"] == 338.0 for job in decision["jobs"])


def test_policy_registration_fails_closed_without_any_downstream_evidence_layer() -> None:
    mod = _load_module()
    complete = {
        "candidate_radius_policy_gate_pass": True,
        "required_job_count": 8,
        "selected_job_count": 8,
        "cut_invariance_gate_pass": True,
        "deterministic_exact_gate_pass": True,
        "candidate_paths_materialized": True,
        "radial_artifacts_current": True,
        "frontier_consistency_gate_pass": True,
        "tube_evidence_gate_pass": True,
        "tube_artifacts_current": True,
        "support_evidence_gate_pass": True,
        "dataset_evidence_gate_pass": True,
        "dataset_artifacts_current": True,
        "model_seed_count": 5,
        "model_unique_seed_count": 5,
        "model_gate_pass": True,
        "model_artifacts_current": True,
        "provenance_chain_current": True,
    }

    accepted = mod.policy_registration_decision(complete)
    assert accepted["policy_registration_allowed"] is True

    for field in (
        "candidate_radius_policy_gate_pass",
        "tube_evidence_gate_pass",
        "support_evidence_gate_pass",
        "dataset_evidence_gate_pass",
        "model_gate_pass",
        "candidate_paths_materialized",
        "radial_artifacts_current",
        "frontier_consistency_gate_pass",
        "tube_artifacts_current",
        "dataset_artifacts_current",
        "model_artifacts_current",
        "provenance_chain_current",
    ):
        failed = dict(complete, **{field: False})
        assert mod.policy_registration_decision(failed)[
            "policy_registration_allowed"
        ] is False

    four_seeds = dict(complete, model_seed_count=4, model_unique_seed_count=4)
    assert mod.policy_registration_decision(four_seeds)[
        "policy_registration_allowed"
    ] is False


def test_candidate_policy_fingerprint_is_threshold_and_source_bound() -> None:
    mod = _load_module()

    p150 = mod.candidate_policy_fingerprint(
        kappa_threshold=150.0,
        radial_source_fingerprint="source-a",
    )
    p200 = mod.candidate_policy_fingerprint(
        kappa_threshold=200.0,
        radial_source_fingerprint="source-a",
    )
    other_source = mod.candidate_policy_fingerprint(
        kappa_threshold=150.0,
        radial_source_fingerprint="source-b",
    )

    assert len(p150) == 64
    assert p150 != p200
    assert p150 != other_source


def test_candidate_frontiers_stop_at_first_failed_registered_checkpoint() -> None:
    mod = _load_module()
    reports = {
        100.0: _complete_radius_report(kappa_p95=140.0, radius_mm=100.0),
        102.5: _complete_radius_report(kappa_p95=149.0, radius_mm=102.5),
        105.0: _complete_radius_report(kappa_p95=190.0, radius_mm=105.0),
        107.5: _complete_radius_report(kappa_p95=260.0, radius_mm=107.5),
        110.0: _complete_radius_report(kappa_p95=410.0, radius_mm=110.0),
    }

    frontiers = mod.compute_candidate_policy_frontiers(
        reports,
        thresholds=(150.0, 200.0, 250.0, 300.0, 400.0),
        registered_checkpoints=(100.0, 102.5, 105.0, 107.5, 110.0),
    )

    assert frontiers[150.0]["registered_frontier_mm"] == 102.5
    assert frontiers[200.0]["registered_frontier_mm"] == 105.0
    assert frontiers[250.0]["registered_frontier_mm"] == 105.0
    assert frontiers[300.0]["registered_frontier_mm"] == 107.5
    assert frontiers[400.0]["registered_frontier_mm"] == 107.5
    assert frontiers[150.0]["first_failed_checkpoint_mm"] == 105.0
    assert len(frontiers[400.0]["checkpoint_evidence"]) == 5


def test_policy_frontier_groups_reuse_only_identical_registered_frontiers() -> None:
    mod = _load_module()
    frontiers = {
        150.0: {"registered_frontier_mm": 102.5},
        200.0: {"registered_frontier_mm": 105.0},
        250.0: {"registered_frontier_mm": 105.0},
        300.0: {"registered_frontier_mm": 107.5},
        400.0: {"registered_frontier_mm": 107.5},
    }

    groups = mod.group_thresholds_by_frontier(frontiers)

    assert groups == {
        102.5: (150.0,),
        105.0: (200.0, 250.0),
        107.5: (300.0, 400.0),
    }


def test_candidate_radial_report_filters_manifest_and_never_grants_formal_claims(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path)])
    policy_args = mod.build_policy_v7_args(
        args,
        frontier_mm=105.0,
        representative_threshold=200.0,
    )
    downstream = {
        "task_fingerprint": "downstream-task",
        "family_id": "fixed-family",
        "path_manifest": {
            "100": {"path": "/tmp/r100.parquet", "sha256": "a"},
            "105": {"path": "/tmp/r105.parquet", "sha256": "b"},
            "107.5": {"path": "/tmp/r107p5.parquet", "sha256": "c"},
        },
    }
    frontier = {
        "registered_frontier_mm": 105.0,
        "first_failed_checkpoint_mm": 107.5,
        "checkpoint_evidence": [
            {
                "radius_mm": 105.0,
                "candidate_radius_policy_gate_pass": True,
                "required_job_count": 8,
                "selected_job_count": 8,
                "candidate_paths_materialized": True,
            }
        ],
    }

    report = mod.build_candidate_radial_report(
        downstream,
        frontier_report=frontier,
        representative_threshold=200.0,
        policy_args=policy_args,
    )

    assert report["strict_geometry_rmax_mm"] == 105.0
    assert set(report["path_manifest"]) == {"100", "105"}
    assert report["downstream_radial_admission_gate_pass"] is True
    assert report["formal_radial_gate_pass"] is False
    assert report["formal_claims_allowed"] is False


def test_policy_v7_args_use_strictest_threshold_in_shared_frontier_group(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path), "--skip-existing"])

    policy_args = mod.build_policy_v7_args(
        args,
        frontier_mm=107.5,
        representative_threshold=min((300.0, 400.0)),
    )

    assert policy_args.radial_job_gate_mode == "candidate_policy"
    assert policy_args.conditioning_kappa_threshold == 300.0
    assert policy_args.stop_after_first_failed_job is False
    assert policy_args.skip_existing is True
    assert "r107p50_k300" in str(policy_args.out_dir)
    assert mod.formal_protocol_report(policy_args)["formal_protocol_gate_pass"] is False


def test_sweep_phase_parser_registers_ordered_execution_stages() -> None:
    mod = _load_module()

    assert mod.parse_phases("all") == [
        "radial",
        "policies",
        "downstream",
        "models",
        "summary",
    ]
    assert mod.parse_phases("radial,policies") == ["radial", "policies"]
    with pytest.raises(ValueError, match="unsupported V7 gate sweep phases"):
        mod.parse_phases("radial,unknown")


def test_radial_phase_forces_full_downstream_evidence_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path)])
    calls = []

    def fake_audit(v7_args):
        calls.append(("audit", v7_args.radial_job_gate_mode))
        return {"audit_gate_pass": True, "formal_audit_gate_pass": False}

    def fake_radial(v7_args):
        calls.append(("radial", v7_args.stop_after_first_failed_job))
        return {
            "task_fingerprint": "downstream-radial",
            "path_manifest": {"105": {"path": "/tmp/r105", "sha256": "x"}},
            "formal_radial_gate_pass": False,
        }

    monkeypatch.setattr(mod.v7, "phase_audit", fake_audit)
    monkeypatch.setattr(mod.v7, "phase_radial", fake_radial)

    report = mod.phase_radial(args)

    assert calls == [("audit", "downstream_admission"), ("radial", False)]
    assert report["formal_claims_allowed"] is False
    assert report["full_job_sweep_requested"] is True


def test_evidence_training_args_bind_candidate_dataset_and_claim_scope(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path), "--skip-existing"])
    policy_args = mod.build_policy_v7_args(
        args,
        frontier_mm=105.0,
        representative_threshold=200.0,
    )
    dataset = tmp_path / "candidate.parquet"
    dataset.write_bytes(b"candidate")
    dataset_report = {
        "evidence_dataset_path": str(dataset),
        "evidence_dataset_sha256": "hash",
    }

    training_args = mod.build_evidence_training_args(
        args,
        policy_args=policy_args,
        dataset_report=dataset_report,
    )

    assert training_args.evidence_only is True
    assert training_args.preset == "formal"
    assert Path(training_args.tube_dataset) == dataset
    assert training_args.skip_existing is True
    assert "03_models" in str(training_args.out_dir)


def test_policy_phase_computes_frontiers_from_registered_checkpoint_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path)])
    downstream = {
        "task_fingerprint": "downstream-task",
        "family_id": "fixed-family",
        "path_manifest": {},
    }
    reports = {
        100.0: _complete_radius_report(kappa_p95=140.0, radius_mm=100.0),
        102.5: _complete_radius_report(kappa_p95=149.0, radius_mm=102.5),
        105.0: _complete_radius_report(kappa_p95=190.0, radius_mm=105.0),
    }
    monkeypatch.setattr(
        mod,
        "ensure_downstream_radial_source",
        lambda _args: downstream,
    )
    monkeypatch.setattr(
        mod,
        "load_registered_checkpoint_reports",
        lambda _args, _source: reports,
    )

    result = mod.phase_policies(args)

    assert result["frontiers"]["150"]["registered_frontier_mm"] == 102.5
    assert result["frontiers"]["200"]["registered_frontier_mm"] == 105.0
    assert result["frontier_groups"]["105"] == [200.0, 250.0, 300.0, 400.0]
    assert Path(result["report_path"]).is_file()


def test_downstream_phase_materializes_once_per_unique_frontier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path)])
    source = {
        "task_fingerprint": "downstream-task",
        "family_id": "fixed-family",
        "path_manifest": {
            "100": {"path": "/tmp/r100", "sha256": "a"},
            "102.5": {"path": "/tmp/r102p5", "sha256": "b"},
            "105": {"path": "/tmp/r105", "sha256": "c"},
        },
    }
    policies = {
        "source_downstream_report": source,
        "frontiers": {
            "150": {"registered_frontier_mm": 102.5, "checkpoint_evidence": []},
            "200": {"registered_frontier_mm": 105.0, "checkpoint_evidence": []},
            "250": {"registered_frontier_mm": 105.0, "checkpoint_evidence": []},
        },
        "frontier_groups": {"102.5": [150.0], "105": [200.0, 250.0]},
    }
    calls = []
    monkeypatch.setattr(mod, "ensure_policies_report", lambda _args: policies)

    def fake_tube(policy_args, *, radial_report):
        calls.append(("tube", radial_report["strict_geometry_rmax_mm"]))
        return {
            "task_fingerprint": f"tube-{radial_report['strict_geometry_rmax_mm']}",
            "tube_evidence_gate_pass": True,
            "formal_tube_gate_pass": False,
        }

    def fake_dataset(policy_args, *, tube_report):
        calls.append(("dataset", tube_report["task_fingerprint"]))
        return {
            "task_fingerprint": f"dataset-{tube_report['task_fingerprint']}",
            "dataset_evidence_gate_pass": True,
            "formal_dataset_gate_pass": False,
            "evidence_dataset_path": str(tmp_path / "candidate.parquet"),
        }

    monkeypatch.setattr(mod.v7, "phase_tube", fake_tube)
    monkeypatch.setattr(mod.v7, "phase_dataset", fake_dataset)

    result = mod.phase_downstream(args)

    assert calls == [
        ("tube", 102.5),
        ("dataset", "tube-102.5"),
        ("tube", 105.0),
        ("dataset", "tube-105.0"),
    ]
    assert len(result["frontier_runs"]) == 2
    assert result["frontier_runs"]["105"]["representative_threshold"] == 200.0


def test_downstream_phase_reuses_hash_identical_gate_v2_artifacts_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    sweep_dir = tmp_path / "sweep"
    gate_v2_dir = tmp_path / "gate-v2"
    args = mod.parse_args(
        [
            "--out-dir",
            str(sweep_dir),
            "--gate-v2-dir",
            str(gate_v2_dir),
        ]
    )

    radial_artifact = tmp_path / "r102p5.parquet"
    radial_artifact.write_bytes(b"radial-102p5")
    radial_manifest = {
        "102.5": {
            "path": str(radial_artifact),
            "sha256": mod.v7.file_sha256(radial_artifact),
            "branch_hash": "branch-102p5",
        }
    }
    gate_radial = {
        "task_fingerprint": "gate-radial",
        "strict_geometry_rmax_mm": 102.5,
        "downstream_radial_admission_gate_pass": True,
        "formal_radial_gate_pass": True,
        "path_manifest": radial_manifest,
    }
    gate_radial_path = gate_v2_dir / "01_radial" / "radial_report.json"
    gate_radial_path.parent.mkdir(parents=True)
    mod.v7.write_json(gate_radial_path, gate_radial)

    tube_artifact = tmp_path / "tube.parquet"
    tube_artifact.write_bytes(b"tube")
    tube_summary = gate_v2_dir / "02_tube" / "tube_radius_summary.csv"
    tube_summary.parent.mkdir(parents=True)
    pd.DataFrame(
        [
                {
                    "radius_mm": 102.5,
                    "formal_tube_label_gate_pass": True,
                "tube_artifact_path": str(tube_artifact),
                "tube_artifact_sha256": mod.v7.file_sha256(tube_artifact),
            }
        ]
    ).to_csv(tube_summary, index=False)
    gate_tube = {
        "task_fingerprint": "gate-tube",
        "radial_task_fingerprint": "gate-radial",
        "strict_geometry_rmax_mm": 102.5,
        "tube_evidence_gate_pass": True,
            "formal_tube_gate_pass": True,
            "all_dataset_radii_pass": True,
            "materialized_radii_mm": [102.5],
        "summary_path": str(tube_summary),
        "summary_sha256": mod.v7.file_sha256(tube_summary),
        "tube_paths": {"102.5": str(tube_artifact)},
    }
    mod.v7.write_json(gate_v2_dir / "02_tube" / "tube_report.json", gate_tube)

    dataset = tmp_path / "dataset.parquet"
    nonformal = tmp_path / "dataset-nonformal.parquet"
    validation = tmp_path / "validation.parquet"
    test = tmp_path / "test.parquet"
    for path in (dataset, nonformal, validation, test):
        path.write_bytes(path.name.encode())
    gate_dataset = {
        "task_fingerprint": "gate-dataset",
        "tube_task_fingerprint": "gate-tube",
        "strict_geometry_rmax_mm": 102.5,
            "formal_dataset_gate_pass": True,
            "dataset_evidence_gate_pass": True,
            "attempt_path": str(dataset),
            "attempt_sha256": mod.v7.file_sha256(dataset),
            "manifest_path": str(dataset),
            "manifest_sha256": mod.v7.file_sha256(dataset),
            "support_candidates_path": str(dataset),
            "support_candidates_sha256": mod.v7.file_sha256(dataset),
        "dataset_path": str(dataset),
        "dataset_sha256": mod.v7.file_sha256(dataset),
        "evidence_dataset_path": str(dataset),
        "evidence_dataset_sha256": mod.v7.file_sha256(dataset),
        "nonformal_dataset_path": str(nonformal),
        "nonformal_dataset_sha256": mod.v7.file_sha256(nonformal),
        "holdout": {
            "selection_gate_pass": True,
            "validation_radius_mm": 95.0,
            "test_radius_mm": 102.5,
        },
        "validation_support_gate_pass": True,
        "test_support_gate_pass": True,
        "challenge_paths": {
            "validation": str(validation),
            "test": str(test),
        },
        "challenge_reports": {
            "validation": {
                "challenge_artifact_sha256": mod.v7.file_sha256(validation)
            },
            "test": {"challenge_artifact_sha256": mod.v7.file_sha256(test)},
        },
    }
    dataset_report_path = gate_v2_dir / "03_dataset" / "dataset_report.json"
    dataset_report_path.parent.mkdir(parents=True)
    mod.v7.write_json(dataset_report_path, gate_dataset)

    policies = {
        "task_fingerprint": "policies",
        "source_downstream_report": {
            "task_fingerprint": "sweep-radial",
            "family_id": "fixed-family",
            "path_manifest": radial_manifest,
        },
        "frontiers": {
            "150": {
                "registered_frontier_mm": 102.5,
                "checkpoint_evidence": [],
            }
        },
        "frontier_groups": {"102.5": [150.0]},
    }
    monkeypatch.setattr(mod, "ensure_policies_report", lambda _args: policies)
    monkeypatch.setattr(
        mod.v7,
        "phase_tube",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hash-identical gate-v2 tube must be reused")
        ),
    )
    monkeypatch.setattr(
        mod.v7,
        "phase_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hash-identical gate-v2 dataset must be reused")
        ),
    )

    result = mod.phase_downstream(args)

    reused = result["frontier_runs"]["102.5"]
    assert reused["gate_v2_artifact_reuse"] is True
    assert reused["tube"]["formal_tube_gate_pass"] is False
    assert reused["tube"]["tube_evidence_gate_pass"] is True
    assert reused["dataset"]["formal_dataset_gate_pass"] is False
    assert reused["dataset"]["dataset_evidence_gate_pass"] is True
    assert reused["dataset"]["evidence_dataset_sha256"] == mod.v7.file_sha256(
        dataset
    )

    radial_artifact.write_bytes(b"tampered-with-the-same-registered-path")
    policy_args = mod.build_policy_v7_args(
        args,
        frontier_mm=102.5,
        representative_threshold=150.0,
    )
    assert (
        mod.gate_v2_read_only_reuse_views(
            args,
            radial_report=reused["radial"],
            policy_args=policy_args,
        )
        is None
    )


def test_candidate_artifact_audit_accepts_selected_prefix_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    selected_tube = tmp_path / "tube100.parquet"
    failed_outer = tmp_path / "tube101p25.parquet"
    selected_tube.write_bytes(b"selected")
    failed_outer.write_bytes(b"diagnostic")
    summary = tmp_path / "tube_summary.csv"
    pd.DataFrame(
        [
            {
                "radius_mm": 100.0,
                "formal_tube_label_gate_pass": True,
                "tube_artifact_path": str(selected_tube),
                "tube_artifact_sha256": mod.v7.file_sha256(selected_tube),
            },
            {
                "radius_mm": 101.25,
                "formal_tube_label_gate_pass": False,
                "tube_artifact_path": str(failed_outer),
                "tube_artifact_sha256": mod.v7.file_sha256(failed_outer),
            },
        ]
    ).to_csv(summary, index=False)
    tube_report = {
        "formal_tube_gate_pass": False,
        "tube_evidence_gate_pass": True,
        "all_dataset_radii_pass": True,
        "materialized_radii_mm": [100.0],
        "summary_path": str(summary),
        "summary_sha256": mod.v7.file_sha256(summary),
    }

    assert mod._tube_evidence_artifacts_current(tube_report) is True
    failed_outer.write_bytes(b"diagnostic-mutated")
    assert mod._tube_evidence_artifacts_current(tube_report) is True
    selected_tube.write_bytes(b"selected-mutated")
    assert mod._tube_evidence_artifacts_current(tube_report) is False


def test_candidate_dataset_artifact_audit_does_not_require_formal_dataset(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    paths = {
        name: tmp_path / f"{name}.bin"
        for name in (
            "attempt",
            "manifest",
            "support",
            "evidence",
            "nonformal",
            "validation",
            "test",
        )
    }
    for name, path in paths.items():
        path.write_bytes(name.encode())
    report = {
        "formal_dataset_gate_pass": False,
        "dataset_evidence_gate_pass": True,
        "attempt_path": str(paths["attempt"]),
        "attempt_sha256": mod.v7.file_sha256(paths["attempt"]),
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": mod.v7.file_sha256(paths["manifest"]),
        "support_candidates_path": str(paths["support"]),
        "support_candidates_sha256": mod.v7.file_sha256(paths["support"]),
        "evidence_dataset_path": str(paths["evidence"]),
        "evidence_dataset_sha256": mod.v7.file_sha256(paths["evidence"]),
        "nonformal_dataset_path": str(paths["nonformal"]),
        "nonformal_dataset_sha256": mod.v7.file_sha256(paths["nonformal"]),
        "challenge_paths": {
            "validation": str(paths["validation"]),
            "test": str(paths["test"]),
        },
        "challenge_reports": {
            "validation": {
                "challenge_artifact_sha256": mod.v7.file_sha256(paths["validation"])
            },
            "test": {
                "challenge_artifact_sha256": mod.v7.file_sha256(paths["test"])
            },
        },
    }

    assert mod._dataset_evidence_artifacts_current(report) is True
    paths["evidence"].write_bytes(b"mutated")
    assert mod._dataset_evidence_artifacts_current(report) is False


def test_candidate_model_artifact_audit_requires_all_five_seed_centerlines(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    seeds = [20260711, 20260712, 20260713, 20260714, 20260715]
    labels = (
        "validation_integer_centerline",
        "validation_half_phase",
        "test_integer_centerline",
        "test_half_phase",
    )
    model_artifacts = {}
    prediction_artifacts = {}
    for seed in seeds:
        model_path = tmp_path / f"model-{seed}.joblib"
        model_path.write_bytes(f"model-{seed}".encode())
        model_artifacts[str(seed)] = {
            "path": str(model_path),
            "sha256": mod.training_v7.file_sha256(model_path),
        }
        prediction_artifacts[str(seed)] = {}
        for label in labels:
            prediction_path = tmp_path / f"{seed}-{label}.parquet"
            prediction_path.write_bytes(f"{seed}-{label}".encode())
            prediction_artifacts[str(seed)][label] = {
                "path": str(prediction_path),
                "sha256": mod.training_v7.file_sha256(prediction_path),
            }
    report = {
        "seed_count": 5,
        "seeds": seeds,
        "model_artifacts": model_artifacts,
        "prediction_artifacts": prediction_artifacts,
    }

    assert mod._model_evidence_artifacts_current(report) is True
    del prediction_artifacts[str(seeds[-1])]["test_half_phase"]
    assert mod._model_evidence_artifacts_current(report) is False


def test_model_phase_trains_once_per_materialized_frontier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path)])
    policy_dir = tmp_path / "02_downstream" / "r105p00_k200"
    dataset = tmp_path / "candidate.parquet"
    dataset.write_bytes(b"candidate")
    downstream = {
        "frontier_runs": {
            "105": {
                "policy_out_dir": str(policy_dir),
                "dataset": {
                    "dataset_evidence_gate_pass": True,
                    "evidence_dataset_path": str(dataset),
                    "evidence_dataset_sha256": "hash",
                },
            }
        }
    }
    calls = []
    monkeypatch.setattr(mod, "ensure_downstream_report", lambda _args: downstream)

    def fake_run(training_args):
        calls.append(training_args)
        return {
            "results": {
                "train": {
                    "seed_count": 5,
                    "seeds": [1, 2, 3, 4, 5],
                    "model_evidence_gate_pass": True,
                    "formal_model_gate_pass": False,
                }
            }
        }

    monkeypatch.setattr(mod.training_v7, "run", fake_run)

    result = mod.phase_models(args)

    assert len(calls) == 1
    assert calls[0].evidence_only is True
    assert result["frontier_models"]["105"]["model_evidence_gate_pass"] is True
    assert result["frontier_models"]["105"]["formal_model_gate_pass"] is False


def test_summary_recommends_smallest_evidence_backed_threshold_at_max_frontier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    args = mod.parse_args(["--out-dir", str(tmp_path)])

    def radial_evidence(radius):
        return {
            "radius_mm": radius,
            "candidate_radius_policy_gate_pass": True,
            "required_job_count": 8,
            "selected_job_count": 8,
            "cut_invariance_gate_pass": True,
            "deterministic_exact_gate_pass": True,
            "candidate_paths_materialized": True,
        }

    policies = {
        "task_fingerprint": "policies",
        "frontiers": {
            "150": {
                "registered_frontier_mm": 102.5,
                "checkpoint_evidence": [radial_evidence(102.5)],
            },
            "200": {
                "registered_frontier_mm": 105.0,
                "checkpoint_evidence": [radial_evidence(105.0)],
            },
            "250": {
                "registered_frontier_mm": 105.0,
                "checkpoint_evidence": [radial_evidence(105.0)],
            },
        },
    }
    def complete_dataset(frontier):
        return {
            "strict_geometry_rmax_mm": frontier,
            "dataset_evidence_gate_pass": True,
            "validation_support_gate_pass": True,
            "test_support_gate_pass": True,
            "holdout": {
                "selection_gate_pass": True,
                "test_radius_mm": frontier,
            },
        }

    downstream = {
        "task_fingerprint": "downstream",
        "frontier_runs": {
            "102.5": {
                "tube": {
                    "strict_geometry_rmax_mm": 102.5,
                    "tube_evidence_gate_pass": True,
                },
                "dataset": complete_dataset(102.5),
            },
            "105": {
                "tube": {
                    "strict_geometry_rmax_mm": 105.0,
                    "tube_evidence_gate_pass": True,
                },
                "dataset": complete_dataset(105.0),
            },
        },
    }
    models = {
        "task_fingerprint": "models",
        "frontier_models": {
            "102.5": {
                "seed_count": 5,
                "seeds": [1, 2, 3, 4, 5],
                "model_evidence_gate_pass": True,
            },
            "105": {
                "seed_count": 5,
                "seeds": [1, 2, 3, 4, 5],
                "model_evidence_gate_pass": True,
            },
        },
    }
    monkeypatch.setattr(mod, "ensure_policies_report", lambda _args: policies)
    monkeypatch.setattr(mod, "ensure_downstream_report", lambda _args: downstream)
    monkeypatch.setattr(mod, "ensure_models_report", lambda _args: models)
    monkeypatch.setattr(
        mod, "_radial_evidence_artifacts_current", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(mod, "_tube_evidence_artifacts_current", lambda _report: True)
    monkeypatch.setattr(mod, "_dataset_evidence_artifacts_current", lambda _report: True)
    monkeypatch.setattr(mod, "_model_evidence_artifacts_current", lambda _report: True)
    monkeypatch.setattr(
        mod, "_provenance_chain_current", lambda **_kwargs: True
    )

    report = mod.phase_summary(args)

    assert report["policies"]["150"]["policy_registration_allowed"] is True
    assert report["policies"]["200"]["policy_registration_allowed"] is True
    assert report["policies"]["250"]["policy_registration_allowed"] is True
    assert report["recommendation"]["kappa_threshold"] == 200.0
    assert report["recommendation"]["registered_frontier_mm"] == 105.0
    assert report["recommendation"]["applied_to_registered_gate"] is False
    assert report["formal_claims_allowed"] is False
