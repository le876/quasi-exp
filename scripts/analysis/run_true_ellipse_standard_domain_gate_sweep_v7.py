#!/usr/bin/env python3
"""Run the non-formal V7 conditioning-kappa evidence sweep.

The sweep never changes the registered legacy threshold.  It first asks the
V7 radial engine for complete downstream-admission evidence, then replays the
same audited jobs under candidate kappa policies.  Policy registration stays
fail-closed until radial, tube, support, dataset, and five-seed model evidence
are all present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT

import sys

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import run_true_ellipse_standard_domain_v7 as v7  # noqa: E402
import run_true_ellipse_standard_domain_training_v7 as training_v7  # noqa: E402
from true_ellipse_family_v5_utils import stable_fingerprint  # noqa: E402


REGISTERED_KAPPA_THRESHOLDS = (150.0, 200.0, 250.0, 300.0, 400.0)
ALL_PHASES = ("radial", "policies", "downstream", "models", "summary")
SWEEP_STRATEGY_VERSION = 2
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_gate_sweep_v7"
DEFAULT_GATE_V2_DIR = PROJECT_ROOT / "runs" / "true_ellipse_standard_domain_v7_gate_v2"
formal_protocol_report = v7.formal_protocol_report


def _parse_float_csv(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    return tuple(float(item) for item in value)


def parse_registered_thresholds(value: str | Iterable[float]) -> tuple[float, ...]:
    thresholds = _parse_float_csv(value)
    if thresholds != REGISTERED_KAPPA_THRESHOLDS:
        raise ValueError(
            "registered kappa thresholds must be exactly "
            + ",".join(f"{item:g}" for item in REGISTERED_KAPPA_THRESHOLDS)
        )
    return thresholds


def parse_phases(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        phases = [part.strip() for part in value.split(",") if part.strip()]
    else:
        phases = [str(part).strip() for part in value if str(part).strip()]
    if phases == ["all"]:
        return list(ALL_PHASES)
    unknown = sorted(set(phases) - set(ALL_PHASES))
    if unknown:
        raise ValueError(f"unsupported V7 gate sweep phases: {unknown}")
    return phases


def candidate_policy_fingerprint(
    *,
    kappa_threshold: float,
    radial_source_fingerprint: str,
) -> str:
    return stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "policy_kind": "v7_conditioning_kappa_candidate_nonformal",
            "kappa_threshold": float(kappa_threshold),
            "legacy_registered_kappa_threshold": 150.0,
            "radial_source_fingerprint": str(radial_source_fingerprint),
            "formal_claims_allowed": False,
        }
    )


def build_downstream_radial_args(args: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--preset",
        "formal",
        "--out-dir",
        str(Path(args.out_dir) / "00_downstream_radial"),
        "--v5-dir",
        str(args.v5_dir),
        "--v6-dir",
        str(args.v6_dir),
        "--robot-config",
        str(args.robot_config),
        "--workers",
        str(int(args.workers)),
        "--radial-job-gate-mode",
        "downstream_admission",
        "--conditioning-kappa-threshold",
        str(max(parse_registered_thresholds(args.kappa_thresholds))),
    ]
    if bool(args.skip_existing):
        argv.append("--skip-existing")
    downstream = v7.parse_args(argv)
    downstream.stop_after_first_failed_job = False
    downstream.share_rescue_cache_across_cuts = True
    return downstream


def build_policy_v7_args(
    args: argparse.Namespace,
    *,
    frontier_mm: float,
    representative_threshold: float,
) -> argparse.Namespace:
    slug = v7.v6_runner.radius_slug(float(frontier_mm))
    policy_dir = (
        Path(args.out_dir)
        / "02_downstream"
        / f"{slug}_k{float(representative_threshold):g}"
    )
    argv = [
        "--preset",
        "formal",
        "--out-dir",
        str(policy_dir),
        "--v5-dir",
        str(args.v5_dir),
        "--v6-dir",
        str(args.v6_dir),
        "--robot-config",
        str(args.robot_config),
        "--workers",
        str(int(args.workers)),
        "--radial-job-gate-mode",
        "candidate_policy",
        "--conditioning-kappa-threshold",
        f"{float(representative_threshold):g}",
    ]
    if bool(args.skip_existing):
        argv.append("--skip-existing")
    policy_args = v7.parse_args(argv)
    policy_args.stop_after_first_failed_job = False
    policy_args.share_rescue_cache_across_cuts = True
    return policy_args


def build_candidate_radial_report(
    downstream_report: Mapping[str, Any],
    *,
    frontier_report: Mapping[str, Any],
    representative_threshold: float,
    policy_args: argparse.Namespace,
) -> dict[str, Any]:
    frontier = frontier_report.get("registered_frontier_mm")
    if frontier is None:
        raise ValueError("candidate radial report requires a registered frontier")
    frontier_mm = float(frontier)
    matching_frontier_evidence = [
        evidence
        for evidence in frontier_report.get("checkpoint_evidence", ())
        if evidence.get("radius_mm") is not None
        and np.isclose(float(evidence["radius_mm"]), frontier_mm, atol=1.0e-8)
    ]
    if matching_frontier_evidence and not bool(
        matching_frontier_evidence[-1].get("candidate_paths_materialized", False)
    ):
        raise ValueError(
            "candidate frontier requires a policy-specific radial materialization run"
        )
    filtered_manifest = {
        str(key): dict(record)
        for key, record in (downstream_report.get("path_manifest") or {}).items()
        if float(key) <= frontier_mm + 1.0e-9
    }
    protocol = v7.formal_protocol_report(policy_args)
    source_fingerprint = str(downstream_report.get("task_fingerprint", ""))
    policy_fingerprint = candidate_policy_fingerprint(
        kappa_threshold=float(representative_threshold),
        radial_source_fingerprint=source_fingerprint,
    )
    report = {
        "strategy_version": SWEEP_STRATEGY_VERSION,
        **protocol,
        "family_id": downstream_report.get("family_id"),
        "source_downstream_radial_task_fingerprint": source_fingerprint,
        "candidate_policy_fingerprint": policy_fingerprint,
        "conditioning_kappa_threshold": float(representative_threshold),
        "strict_geometry_rmax_mm": frontier_mm,
        "target_105_achieved": bool(frontier_mm >= 105.0 - 1.0e-9),
        "downstream_radial_admission_gate_pass": bool(filtered_manifest),
        "formal_radial_gate_pass": False,
        "formal_claims_allowed": False,
        "path_manifest": filtered_manifest,
        "candidate_frontier_evidence": dict(frontier_report),
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "policy": policy_fingerprint,
            "frontier_mm": frontier_mm,
            "paths": filtered_manifest,
            "source": source_fingerprint,
        }
    )
    return report


def build_evidence_training_args(
    args: argparse.Namespace,
    *,
    policy_args: argparse.Namespace,
    dataset_report: Mapping[str, Any],
) -> argparse.Namespace:
    dataset_path = dataset_report.get("evidence_dataset_path")
    if not dataset_path:
        raise ValueError("candidate model training requires an evidence dataset")
    training_dir = Path(args.out_dir) / "03_models" / Path(policy_args.out_dir).name
    argv = [
        "--preset",
        "formal",
        "--evidence-only",
        "--upstream-dir",
        str(policy_args.out_dir),
        "--tube-dataset",
        str(dataset_path),
        "--out-dir",
        str(training_dir),
        "--robot-config",
        str(args.robot_config),
        "--workers",
        str(max(1, min(int(args.workers), 2))),
    ]
    if bool(args.skip_existing):
        argv.append("--skip-existing")
    return training_v7.parse_args(argv)


def phase_radial(args: argparse.Namespace) -> dict[str, Any]:
    downstream_args = build_downstream_radial_args(args)
    audit = v7.phase_audit(downstream_args)
    radial = v7.phase_radial(downstream_args)
    report = {
        "strategy_version": SWEEP_STRATEGY_VERSION,
        "formal_claims_allowed": False,
        "full_job_sweep_requested": bool(
            not downstream_args.stop_after_first_failed_job
        ),
        "downstream_gate_mode": downstream_args.radial_job_gate_mode,
        "audit_gate_pass": bool(audit.get("audit_gate_pass", False)),
        "downstream_radial_task_fingerprint": str(
            radial.get("task_fingerprint", "")
        ),
        "path_manifest": radial.get("path_manifest", {}),
        "source_report": radial,
    }
    path = Path(downstream_args.out_dir) / "sweep_radial_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    v7.write_json(path, report)
    return report


def ensure_downstream_radial_source(args: argparse.Namespace) -> dict[str, Any]:
    downstream_args = build_downstream_radial_args(args)
    path = Path(downstream_args.out_dir) / "01_radial" / "radial_report.json"
    if path.is_file():
        return v7.read_json(path)
    return dict(phase_radial(args)["source_report"])


def load_registered_checkpoint_reports(
    args: argparse.Namespace,
    downstream_report: Mapping[str, Any],
) -> dict[float, dict[str, Any]]:
    del downstream_report  # the task fingerprint is checked by the phase report
    downstream_args = build_downstream_radial_args(args)
    reports: dict[float, dict[str, Any]] = {}
    protocol = v7.engine.v7_standard_domain_protocol()
    for radius in protocol.formal_checkpoints_mm:
        if float(radius) <= float(protocol.start_radius_mm) + 1.0e-9:
            continue
        path = (
            Path(downstream_args.out_dir)
            / "01_radial"
            / v7.v6_runner.radius_slug(float(radius))
            / "radius_bundle_report.json"
        )
        if path.is_file():
            reports[float(radius)] = v7.read_json(path)
    if not reports:
        raise FileNotFoundError("downstream radial run has no registered checkpoint reports")
    return reports


def _float_key(value: float) -> str:
    return f"{float(value):g}"


def phase_policies(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "01_policies"
    out.mkdir(parents=True, exist_ok=True)
    source = ensure_downstream_radial_source(args)
    reports = load_registered_checkpoint_reports(args, source)
    thresholds = parse_registered_thresholds(args.kappa_thresholds)
    frontiers = compute_candidate_policy_frontiers(
        reports,
        thresholds=thresholds,
        registered_checkpoints=sorted(reports),
    )
    groups = group_thresholds_by_frontier(frontiers)
    serialized_frontiers = {
        _float_key(threshold): report
        for threshold, report in sorted(frontiers.items())
    }
    serialized_groups = {
        _float_key(frontier): list(group)
        for frontier, group in sorted(groups.items())
    }
    report = {
        "strategy_version": SWEEP_STRATEGY_VERSION,
        "formal_claims_allowed": False,
        "source_downstream_task_fingerprint": str(source.get("task_fingerprint", "")),
        "source_downstream_report": source,
        "thresholds": list(thresholds),
        "frontiers": serialized_frontiers,
        "frontier_groups": serialized_groups,
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "source": report["source_downstream_task_fingerprint"],
            "frontiers": serialized_frontiers,
            "groups": serialized_groups,
        }
    )
    report_path = out / "policy_frontiers.json"
    report["report_path"] = str(report_path.resolve())
    v7.write_json(report_path, report)
    return report


def ensure_policies_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "01_policies" / "policy_frontiers.json"
    if path.is_file():
        return v7.read_json(path)
    return phase_policies(args)


def _path_hash_pair_current(
    report: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
) -> bool:
    raw_path = report.get(path_key)
    expected = str(report.get(hash_key, ""))
    if not raw_path or not expected:
        return False
    path = Path(str(raw_path))
    return bool(path.is_file() and v7.file_sha256(path) == expected)


def _frontier_manifest_hashes(
    manifest: Mapping[str, Any],
    *,
    frontier_mm: float,
) -> dict[str, str] | None:
    hashes: dict[str, str] = {}
    for raw_radius, raw_record in manifest.items():
        try:
            radius = float(raw_radius)
            record = dict(raw_record)
        except (TypeError, ValueError):
            return None
        if radius > float(frontier_mm) + 1.0e-9:
            continue
        path = Path(str(record.get("path", "")))
        expected = str(record.get("sha256", ""))
        if not path.is_file() or not expected or v7.file_sha256(path) != expected:
            return None
        hashes[_float_key(radius)] = expected
    return hashes or None


def _radial_evidence_artifacts_current(
    report: Mapping[str, Any],
    *,
    frontier_mm: float,
) -> bool:
    hashes = _frontier_manifest_hashes(
        report.get("path_manifest") or {},
        frontier_mm=float(frontier_mm),
    )
    return bool(hashes and _float_key(float(frontier_mm)) in hashes)


def _tube_evidence_artifacts_current(report: Mapping[str, Any]) -> bool:
    if not bool(
        report.get("tube_evidence_gate_pass", False)
        and report.get("all_dataset_radii_pass", True)
    ):
        return False
    if not _path_hash_pair_current(
        report,
        path_key="summary_path",
        hash_key="summary_sha256",
    ):
        return False
    try:
        summary = pd.read_csv(Path(str(report["summary_path"])))
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required = {
        "radius_mm",
        "formal_tube_label_gate_pass",
        "tube_artifact_path",
        "tube_artifact_sha256",
    }
    if summary.empty or not required.issubset(summary.columns):
        return False
    try:
        materialized = {
            _float_key(float(radius))
            for radius in report.get("materialized_radii_mm", ())
        }
        summary_radius_keys = summary["radius_mm"].astype(float).map(_float_key)
    except (TypeError, ValueError):
        return False
    if not materialized:
        return False
    selected = summary.loc[summary_radius_keys.isin(materialized)].copy()
    selected_keys = {
        _float_key(float(radius)) for radius in selected["radius_mm"].tolist()
    }
    if selected_keys != materialized or len(selected) != len(materialized):
        return False
    passed = selected["formal_tube_label_gate_pass"].fillna(False).astype(bool)
    if not bool(passed.all()):
        return False
    for row in selected.to_dict(orient="records"):
        path = Path(str(row.get("tube_artifact_path", "")))
        expected = str(row.get("tube_artifact_sha256", ""))
        if not path.is_file() or not expected or v7.file_sha256(path) != expected:
            return False
    return True


def _dataset_evidence_artifacts_current(report: Mapping[str, Any]) -> bool:
    if not bool(report.get("dataset_evidence_gate_pass", False)):
        return False
    for path_key, hash_key in (
        ("attempt_path", "attempt_sha256"),
        ("manifest_path", "manifest_sha256"),
        ("support_candidates_path", "support_candidates_sha256"),
        ("evidence_dataset_path", "evidence_dataset_sha256"),
        ("nonformal_dataset_path", "nonformal_dataset_sha256"),
    ):
        if not _path_hash_pair_current(
            report,
            path_key=path_key,
            hash_key=hash_key,
        ):
            return False
    if bool(report.get("formal_dataset_gate_pass", False)) and not _path_hash_pair_current(
        report,
        path_key="dataset_path",
        hash_key="dataset_sha256",
    ):
        return False
    paths = dict(report.get("challenge_paths") or {})
    challenge_reports = dict(report.get("challenge_reports") or {})
    for split in ("validation", "test"):
        path = Path(str(paths.get(split, "")))
        expected = str(
            (challenge_reports.get(split) or {}).get(
                "challenge_artifact_sha256", ""
            )
        )
        if not path.is_file() or not expected or v7.file_sha256(path) != expected:
            return False
    return True


def _model_evidence_artifacts_current(report: Mapping[str, Any]) -> bool:
    seeds = {str(seed) for seed in report.get("seeds", ())}
    if int(report.get("seed_count", 0)) != 5 or len(seeds) != 5:
        return False
    models = dict(report.get("model_artifacts") or {})
    predictions = dict(report.get("prediction_artifacts") or {})
    if set(models) != seeds or set(predictions) != seeds:
        return False
    if not all(training_v7.artifact_record_is_current(record) for record in models.values()):
        return False
    required_centerlines = {
        "validation_integer_centerline",
        "validation_half_phase",
        "test_integer_centerline",
        "test_half_phase",
    }
    label_sets = [set(records) for records in predictions.values()]
    if not label_sets or any(labels != label_sets[0] for labels in label_sets[1:]):
        return False
    if not required_centerlines.issubset(label_sets[0]):
        return False
    return all(
        training_v7.artifact_record_is_current(record)
        for records in predictions.values()
        for record in records.values()
    )


def gate_v2_read_only_reuse_views(
    args: argparse.Namespace,
    *,
    radial_report: Mapping[str, Any],
    policy_args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return non-formal evidence views only for byte-identical gate-v2 inputs."""

    frontier = radial_report.get("strict_geometry_rmax_mm")
    if frontier is None:
        return None
    frontier_mm = float(frontier)
    root = Path(args.gate_v2_dir)
    report_paths = {
        "radial": root / "01_radial" / "radial_report.json",
        "tube": root / "02_tube" / "tube_report.json",
        "dataset": root / "03_dataset" / "dataset_report.json",
    }
    if not all(path.is_file() for path in report_paths.values()):
        return None
    try:
        source_radial = v7.read_json(report_paths["radial"])
        source_tube = v7.read_json(report_paths["tube"])
        source_dataset = v7.read_json(report_paths["dataset"])
    except (OSError, TypeError, ValueError):
        return None
    source_frontiers = (
        source_radial.get("strict_geometry_rmax_mm"),
        source_tube.get("strict_geometry_rmax_mm"),
        source_dataset.get("strict_geometry_rmax_mm"),
    )
    try:
        frontiers_match = all(
            value is not None
            and np.isclose(float(value), frontier_mm, atol=1.0e-8, rtol=0.0)
            for value in source_frontiers
        )
    except (TypeError, ValueError):
        frontiers_match = False
    if not frontiers_match:
        return None
    candidate_hashes = _frontier_manifest_hashes(
        radial_report.get("path_manifest") or {},
        frontier_mm=frontier_mm,
    )
    source_hashes = _frontier_manifest_hashes(
        source_radial.get("path_manifest") or {},
        frontier_mm=frontier_mm,
    )
    if candidate_hashes is None or candidate_hashes != source_hashes:
        return None
    if not bool(
        source_radial.get("formal_radial_gate_pass", False)
        and source_radial.get("downstream_radial_admission_gate_pass", False)
        and source_tube.get("formal_tube_gate_pass", False)
        and source_dataset.get("formal_dataset_gate_pass", False)
        and str(source_tube.get("radial_task_fingerprint", ""))
        == str(source_radial.get("task_fingerprint", ""))
        and str(source_dataset.get("tube_task_fingerprint", ""))
        == str(source_tube.get("task_fingerprint", ""))
    ):
        return None
    if not _tube_evidence_artifacts_current(source_tube):
        return None
    if not _dataset_evidence_artifacts_current(source_dataset):
        return None

    source_report_hashes = {
        key: v7.file_sha256(path) for key, path in report_paths.items()
    }
    reuse_fingerprint = stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "reuse_mode": "gate_v2_hash_identical_read_only_v1",
            "policy_radial": radial_report.get("task_fingerprint", ""),
            "frontier_mm": frontier_mm,
            "manifest_hashes": candidate_hashes,
            "source_report_hashes": source_report_hashes,
        }
    )
    policy_protocol = v7.formal_protocol_report(policy_args)
    tube = {
        **source_tube,
        "protocol_fingerprint": policy_protocol["protocol_fingerprint"],
        "radial_task_fingerprint": str(radial_report.get("task_fingerprint", "")),
        "formal_protocol_gate_pass": False,
        "formal_tube_gate_pass": False,
        "formal_claims_allowed": False,
        "gate_v2_artifact_reuse": True,
        "reuse_mode": "hash_identical_read_only",
        "reuse_fingerprint": reuse_fingerprint,
        "source_gate_v2_task_fingerprint": str(
            source_tube.get("task_fingerprint", "")
        ),
        "source_gate_v2_report_hashes": source_report_hashes,
    }
    tube["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "kind": "candidate_tube_evidence_view",
            "reuse": reuse_fingerprint,
            "radial": tube["radial_task_fingerprint"],
        }
    )
    checks = dict(source_dataset.get("checks_recomputed") or {})
    checks.update(
        {
            "formal_protocol_gate_pass": False,
            "formal_tube_gate_pass": False,
            "tube_evidence_gate_pass": True,
        }
    )
    dataset = {
        **source_dataset,
        "protocol_fingerprint": policy_protocol["protocol_fingerprint"],
        "tube_task_fingerprint": tube["task_fingerprint"],
        "formal_protocol_gate_pass": False,
        "formal_dataset_gate_pass": False,
        "dataset_gate_pass": False,
        "dataset_evidence_gate_pass": True,
        "formal_claims_allowed": False,
        "dataset_path": None,
        "dataset_sha256": None,
        "checks_recomputed": checks,
        "gate_v2_artifact_reuse": True,
        "reuse_mode": "hash_identical_read_only",
        "reuse_fingerprint": reuse_fingerprint,
        "source_gate_v2_task_fingerprint": str(
            source_dataset.get("task_fingerprint", "")
        ),
        "source_gate_v2_report_hashes": source_report_hashes,
    }
    dataset["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "kind": "candidate_dataset_evidence_view",
            "reuse": reuse_fingerprint,
            "tube": tube["task_fingerprint"],
            "evidence_dataset_sha256": dataset.get("evidence_dataset_sha256", ""),
        }
    )
    return tube, dataset


def phase_downstream(args: argparse.Namespace) -> dict[str, Any]:
    policies = ensure_policies_report(args)
    source = dict(policies["source_downstream_report"])
    frontiers = dict(policies["frontiers"])
    runs: dict[str, Any] = {}
    for frontier_key, raw_thresholds in sorted(
        policies["frontier_groups"].items(), key=lambda item: float(item[0])
    ):
        frontier = float(frontier_key)
        thresholds = tuple(sorted(float(value) for value in raw_thresholds))
        representative = min(thresholds)
        frontier_report = dict(frontiers[_float_key(representative)])
        policy_args = build_policy_v7_args(
            args,
            frontier_mm=frontier,
            representative_threshold=representative,
        )
        radial = build_candidate_radial_report(
            source,
            frontier_report=frontier_report,
            representative_threshold=representative,
            policy_args=policy_args,
        )
        radial_path = Path(policy_args.out_dir) / "01_radial" / "radial_report.json"
        radial_path.parent.mkdir(parents=True, exist_ok=True)
        radial["path_manifest_path"] = str(
            (Path(policy_args.out_dir) / "01_radial" / "strict_path_manifest.json").resolve()
        )
        v7.write_json(radial_path, radial)
        v7.write_json(Path(radial["path_manifest_path"]), radial["path_manifest"])
        reused = gate_v2_read_only_reuse_views(
            args,
            radial_report=radial,
            policy_args=policy_args,
        )
        if reused is None:
            tube = v7.phase_tube(policy_args, radial_report=radial)
            dataset = v7.phase_dataset(policy_args, tube_report=tube)
            gate_v2_artifact_reuse = False
        else:
            tube, dataset = reused
            tube_path = Path(policy_args.out_dir) / "02_tube" / "tube_report.json"
            dataset_path = Path(policy_args.out_dir) / "03_dataset" / "dataset_report.json"
            tube_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            v7.write_json(tube_path, tube)
            v7.write_json(dataset_path, dataset)
            gate_v2_artifact_reuse = True
        runs[_float_key(frontier)] = {
            "frontier_mm": frontier,
            "thresholds": list(thresholds),
            "representative_threshold": representative,
            "policy_out_dir": str(Path(policy_args.out_dir).resolve()),
            "gate_v2_artifact_reuse": gate_v2_artifact_reuse,
            "radial": radial,
            "tube": tube,
            "dataset": dataset,
        }
    report = {
        "strategy_version": SWEEP_STRATEGY_VERSION,
        "formal_claims_allowed": False,
        "source_policies_task_fingerprint": str(policies.get("task_fingerprint", "")),
        "frontier_runs": runs,
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "policies": report["source_policies_task_fingerprint"],
            "runs": {
                key: {
                    "radial": value["radial"].get("task_fingerprint", ""),
                    "tube": value["tube"].get("task_fingerprint", ""),
                    "dataset": value["dataset"].get("task_fingerprint", ""),
                }
                for key, value in runs.items()
            },
        }
    )
    path = Path(args.out_dir) / "02_downstream" / "downstream_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(path.resolve())
    v7.write_json(path, report)
    return report


def ensure_downstream_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "02_downstream" / "downstream_report.json"
    if path.is_file():
        return v7.read_json(path)
    return phase_downstream(args)


def phase_models(args: argparse.Namespace) -> dict[str, Any]:
    downstream = ensure_downstream_report(args)
    models: dict[str, Any] = {}
    for frontier_key, record in sorted(
        downstream["frontier_runs"].items(), key=lambda item: float(item[0])
    ):
        dataset = dict(record["dataset"])
        if not bool(dataset.get("dataset_evidence_gate_pass", False)):
            models[str(frontier_key)] = {
                "model_evidence_gate_pass": False,
                "formal_model_gate_pass": False,
                "reason": "candidate_dataset_evidence_failed",
            }
            continue
        representative = record.get("representative_threshold")
        if representative is None:
            directory_name = Path(str(record["policy_out_dir"])).name
            if "_k" not in directory_name:
                raise ValueError("frontier model record is missing its representative threshold")
            representative = float(directory_name.rsplit("_k", 1)[1])
        policy_args = v7.parse_args(
            [
                "--out-dir",
                str(record["policy_out_dir"]),
                "--radial-job-gate-mode",
                "candidate_policy",
                    "--conditioning-kappa-threshold",
                    str(representative),
            ]
        )
        training_args = build_evidence_training_args(
            args,
            policy_args=policy_args,
            dataset_report=dataset,
        )
        run_report = training_v7.run(training_args)
        training = dict(run_report.get("results", {}).get("train", {}))
        models[str(frontier_key)] = {
            **training,
            "training_out_dir": str(Path(training_args.out_dir).resolve()),
            "formal_claims_allowed": False,
        }
    report = {
        "strategy_version": SWEEP_STRATEGY_VERSION,
        "formal_claims_allowed": False,
        "source_downstream_task_fingerprint": str(
            downstream.get("task_fingerprint", "")
        ),
        "frontier_models": models,
    }
    report["task_fingerprint"] = stable_fingerprint(
        {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "downstream": report["source_downstream_task_fingerprint"],
            "models": models,
        }
    )
    path = Path(args.out_dir) / "03_models" / "models_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(path.resolve())
    v7.write_json(path, report)
    return report


def ensure_models_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "03_models" / "models_report.json"
    if path.is_file():
        return v7.read_json(path)
    return phase_models(args)


def _provenance_chain_current(
    *,
    policies: Mapping[str, Any],
    downstream: Mapping[str, Any],
    models: Mapping[str, Any],
    radial: Mapping[str, Any],
    tube: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> bool:
    policies_fingerprint = str(policies.get("task_fingerprint", ""))
    downstream_fingerprint = str(downstream.get("task_fingerprint", ""))
    radial_fingerprint = str(radial.get("task_fingerprint", ""))
    tube_fingerprint = str(tube.get("task_fingerprint", ""))
    return bool(
        policies_fingerprint
        and downstream_fingerprint
        and radial_fingerprint
        and tube_fingerprint
        and str(downstream.get("source_policies_task_fingerprint", ""))
        == policies_fingerprint
        and str(models.get("source_downstream_task_fingerprint", ""))
        == downstream_fingerprint
        and str(tube.get("radial_task_fingerprint", "")) == radial_fingerprint
        and str(dataset.get("tube_task_fingerprint", "")) == tube_fingerprint
    )


def phase_summary(args: argparse.Namespace) -> dict[str, Any]:
    policies = ensure_policies_report(args)
    downstream = ensure_downstream_report(args)
    models = ensure_models_report(args)
    decisions: dict[str, Any] = {}
    eligible: list[tuple[float, float]] = []
    for threshold_key, frontier_report in sorted(
        policies["frontiers"].items(), key=lambda item: float(item[0])
    ):
        threshold = float(threshold_key)
        frontier = frontier_report.get("registered_frontier_mm")
        if frontier is None:
            radial = {
                "candidate_radius_policy_gate_pass": False,
                "required_job_count": 0,
                "selected_job_count": 0,
                "cut_invariance_gate_pass": False,
                "deterministic_exact_gate_pass": False,
            }
            frontier_key = None
            run_record: Mapping[str, Any] = {}
            model: Mapping[str, Any] = {}
        else:
            frontier_key = _float_key(float(frontier))
            checkpoint_evidence = list(frontier_report.get("checkpoint_evidence", ()))
            radial = checkpoint_evidence[-1] if checkpoint_evidence else {}
            run_record = downstream.get("frontier_runs", {}).get(frontier_key, {})
            model = models.get("frontier_models", {}).get(frontier_key, {})
        tube = dict(run_record.get("tube", {}))
        dataset = dict(run_record.get("dataset", {}))
        downstream_radial = dict(run_record.get("radial", {}))
        holdout = dict(dataset.get("holdout") or {})
        seeds = [int(seed) for seed in model.get("seeds", ())]
        frontier_consistent = bool(
            frontier is not None
            and tube.get("strict_geometry_rmax_mm") is not None
            and dataset.get("strict_geometry_rmax_mm") is not None
            and holdout.get("test_radius_mm") is not None
            and np.isclose(
                float(tube["strict_geometry_rmax_mm"]),
                float(frontier),
                atol=1.0e-8,
                rtol=0.0,
            )
            and np.isclose(
                float(dataset["strict_geometry_rmax_mm"]),
                float(frontier),
                atol=1.0e-8,
                rtol=0.0,
            )
            and np.isclose(
                float(holdout["test_radius_mm"]),
                float(frontier),
                atol=1.0e-8,
                rtol=0.0,
            )
        )
        evidence = {
            **dict(radial),
            "radial_artifacts_current": bool(
                frontier is not None
                and _radial_evidence_artifacts_current(
                    downstream_radial,
                    frontier_mm=float(frontier),
                )
            ),
            "frontier_consistency_gate_pass": frontier_consistent,
            "tube_evidence_gate_pass": bool(
                tube.get("tube_evidence_gate_pass", False)
            ),
            "tube_artifacts_current": _tube_evidence_artifacts_current(tube),
            "support_evidence_gate_pass": bool(
                dataset.get("validation_support_gate_pass", False)
                and dataset.get("test_support_gate_pass", False)
                and holdout.get("selection_gate_pass", False)
            ),
            "dataset_evidence_gate_pass": bool(
                dataset.get("dataset_evidence_gate_pass", False)
            ),
            "dataset_artifacts_current": _dataset_evidence_artifacts_current(dataset),
            "model_seed_count": int(model.get("seed_count", 0)),
            "model_unique_seed_count": len(set(seeds)),
            "model_gate_pass": bool(model.get("model_evidence_gate_pass", False)),
            "model_artifacts_current": _model_evidence_artifacts_current(model),
            "provenance_chain_current": _provenance_chain_current(
                policies=policies,
                downstream=downstream,
                models=models,
                radial=downstream_radial,
                tube=tube,
                dataset=dataset,
            ),
        }
        registration = policy_registration_decision(evidence)
        decisions[_float_key(threshold)] = {
            "kappa_threshold": threshold,
            "registered_frontier_mm": frontier,
            "first_failed_checkpoint_mm": frontier_report.get(
                "first_failed_checkpoint_mm"
            ),
            "frontier_group": frontier_key,
            "evidence": evidence,
            **registration,
        }
        if registration["policy_registration_allowed"] and frontier is not None:
            eligible.append((float(frontier), threshold))
    if eligible:
        best_frontier = max(frontier for frontier, _threshold in eligible)
        best_threshold = min(
            threshold
            for frontier, threshold in eligible
            if np.isclose(frontier, best_frontier, atol=1.0e-8)
        )
        recommendation: dict[str, Any] = {
            "available": True,
            "kappa_threshold": best_threshold,
            "registered_frontier_mm": best_frontier,
            "applied_to_registered_gate": False,
            "recommendation_only": True,
        }
    else:
        recommendation = {
            "available": False,
            "kappa_threshold": None,
            "registered_frontier_mm": None,
            "applied_to_registered_gate": False,
            "recommendation_only": True,
            "reason": "no_candidate_policy_has_complete_downstream_evidence",
        }
    report = {
        "strategy_version": SWEEP_STRATEGY_VERSION,
        "formal_claims_allowed": False,
        "registered_legacy_kappa_threshold": 150.0,
        "policies": decisions,
        "recommendation": recommendation,
        "source_task_fingerprints": {
            "policies": str(policies.get("task_fingerprint", "")),
            "downstream": str(downstream.get("task_fingerprint", "")),
            "models": str(models.get("task_fingerprint", "")),
        },
    }
    report["task_fingerprint"] = stable_fingerprint(report)
    out = Path(args.out_dir) / "04_summary"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gate_sweep_report.json"
    report["report_path"] = str(path.resolve())
    v7.write_json(path, report)
    lines = [
        "# True Ellipse V7 conditioning gate sweep",
        "",
        "- Diagnostic/evidence-only sweep: `true`.",
        "- Formal claims allowed: `false`.",
        f"- Recommendation: `{recommendation}`.",
        "",
        "The recommendation is not applied to the registered legacy gate.",
    ]
    (out / "gate_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    phases = parse_phases(args.phases)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    handlers = {
        "radial": phase_radial,
        "policies": phase_policies,
        "downstream": phase_downstream,
        "models": phase_models,
        "summary": phase_summary,
    }
    results = {phase: handlers[phase](args) for phase in phases}
    report = {
        "mode": "true_ellipse_standard_domain_gate_sweep_v7",
        "strategy_version": SWEEP_STRATEGY_VERSION,
        "formal_claims_allowed": False,
        "phases": phases,
        "out_dir": str(Path(args.out_dir).resolve()),
        "results": results,
    }
    v7.write_json(Path(args.out_dir) / "run_report.json", report)
    return report


def _candidate_job_gate(
    report: Mapping[str, Any],
    *,
    kappa_threshold: float,
) -> dict[str, Any]:
    def evaluate(
        candidate: Mapping[str, Any],
        *,
        source: str,
        selected_stage: str | None,
    ) -> dict[str, Any]:
        conditioning = v7.v6_utils.atlas.evaluate_conditioning_policy(
            candidate,
            kappa_threshold=float(kappa_threshold),
            sigma3_min_m=0.0015,
        )
        branch_pass = bool(candidate.get("branch_gate_pass", False))
        canonical_pass = bool(candidate.get("canonical_gate_pass", False))
        downstream_pass = bool(
            candidate.get("downstream_admission_gate_pass", False)
        )
        margin_available = "job_margin_gate_pass" in candidate
        margin_pass = bool(candidate.get("job_margin_gate_pass", False))
        gate = bool(
            branch_pass
            and canonical_pass
            and downstream_pass
            and margin_available
            and margin_pass
            and conditioning["conditioning_policy_gate_pass"]
        )
        return {
            **conditioning,
            "branch_gate_pass": branch_pass,
            "canonical_gate_pass": canonical_pass,
            "downstream_admission_gate_pass": downstream_pass,
            "joint_margin_evidence_available": margin_available,
            "joint_margin_gate_pass": margin_pass,
            "candidate_job_gate_pass": gate,
            "candidate_source": source,
            "selected_stage": selected_stage,
            "kappa_p95": candidate.get("kappa_p95"),
            "sigma3_p05_m": candidate.get("sigma3_p05_m"),
        }

    candidates = [
        evaluate(
            report,
            source="materialized_job_output",
            selected_stage=(
                None
                if report.get("selected_stage") is None
                else str(report.get("selected_stage"))
            ),
        )
    ]
    for stage in report.get("stage_history", ()):
        if not isinstance(stage, Mapping):
            continue
        candidates.append(
            evaluate(
                stage,
                source="optimizer_stage_history",
                selected_stage=(
                    None if stage.get("stage") is None else str(stage.get("stage"))
                ),
            )
        )
    passing = [candidate for candidate in candidates if candidate["candidate_job_gate_pass"]]
    if passing:
        chosen = passing[0]
    else:
        chosen = min(
            candidates,
            key=lambda candidate: (
                float(candidate.get("kappa_p95") or np.inf),
                float(candidate.get("sigma3_p05_m") or -np.inf) * -1.0,
            ),
        )
    original_stage = (
        None if report.get("selected_stage") is None else str(report.get("selected_stage"))
    )
    return {
        **chosen,
        "candidate_stage_reselection_required": bool(
            chosen["candidate_source"] == "optimizer_stage_history"
            and chosen["selected_stage"] != original_stage
        ),
        "candidate_count": int(len(candidates)),
    }


def evaluate_candidate_radius_policy(
    report: Mapping[str, Any],
    *,
    kappa_threshold: float,
) -> dict[str, Any]:
    cuts = tuple(int(value) for value in report.get("required_cuts", ()))
    predictors = tuple(str(value) for value in report.get("required_predictors", ()))
    required = [(cut, predictor) for predictor in predictors for cut in cuts]
    jobs_by_key = {
        (int(job.get("cut_idx", -1)), str(job.get("radial_predictor_type", ""))): job
        for job in report.get("jobs", ())
        if bool(job.get("selected", False))
    }
    missing = [
        f"{cut}:{predictor}"
        for cut, predictor in required
        if (cut, predictor) not in jobs_by_key
    ]
    evaluated_jobs: list[dict[str, Any]] = []
    failed: list[str] = []
    for cut, predictor in required:
        job = jobs_by_key.get((cut, predictor))
        if job is None:
            continue
        decision = _candidate_job_gate(job, kappa_threshold=float(kappa_threshold))
        evaluated_jobs.append(
            {
                "cut_idx": cut,
                "radial_predictor_type": predictor,
                **decision,
            }
        )
        if not decision["candidate_job_gate_pass"]:
            failed.append(f"{cut}:{predictor}")
    geometry_pass = bool(
        (report.get("geometry") or {}).get("target_geometry_gate_pass", False)
    )
    cut_pass = bool(
        (report.get("cut_invariance") or {}).get("cut_invariance_gate_pass", False)
    )
    repeat_pass = bool(
        (report.get("deterministic_repeatability") or {}).get(
            "deterministic_exact_gate_pass", False
        )
    )
    complete_eight = bool(len(required) == 8 and not missing)
    stage_reselection_required = bool(
        any(
            job.get("candidate_job_gate_pass", False)
            and job.get("candidate_stage_reselection_required", False)
            for job in evaluated_jobs
        )
    )
    candidate_paths_materialized = bool(not stage_reselection_required)
    gate = bool(
        geometry_pass
        and complete_eight
        and not failed
        and cut_pass
        and repeat_pass
    )
    source_fingerprint = str(report.get("task_fingerprint", ""))
    return {
        "strategy_version": SWEEP_STRATEGY_VERSION,
        "radius_mm": report.get("radius_mm"),
        "kappa_threshold": float(kappa_threshold),
        "policy_fingerprint": candidate_policy_fingerprint(
            kappa_threshold=float(kappa_threshold),
            radial_source_fingerprint=source_fingerprint,
        ),
        "radial_source_fingerprint": source_fingerprint,
        "required_job_count": len(required),
        "selected_job_count": len(jobs_by_key),
        "complete_eight_job_evidence": complete_eight,
        "missing_jobs": missing,
        "failed_jobs": failed,
        "target_geometry_gate_pass": geometry_pass,
        "cut_invariance_gate_pass": cut_pass,
        "deterministic_exact_gate_pass": repeat_pass,
        "jobs": evaluated_jobs,
        "candidate_stage_reselection_required": stage_reselection_required,
        "candidate_paths_materialized": candidate_paths_materialized,
        "candidate_radius_policy_gate_pass": gate,
        "formal_claims_allowed": False,
    }


def policy_registration_decision(evidence: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "candidate_radius_policy_gate_pass": bool(
            evidence.get("candidate_radius_policy_gate_pass", False)
        ),
        "complete_eight_job_evidence": bool(
            int(evidence.get("required_job_count", 0)) == 8
            and int(evidence.get("selected_job_count", 0)) == 8
        ),
        "cut_invariance_gate_pass": bool(
            evidence.get("cut_invariance_gate_pass", False)
        ),
        "deterministic_exact_gate_pass": bool(
            evidence.get("deterministic_exact_gate_pass", False)
        ),
        "candidate_paths_materialized": bool(
            evidence.get("candidate_paths_materialized", False)
        ),
        "radial_artifacts_current": bool(
            evidence.get("radial_artifacts_current", False)
        ),
        "frontier_consistency_gate_pass": bool(
            evidence.get("frontier_consistency_gate_pass", False)
        ),
        "tube_evidence_gate_pass": bool(evidence.get("tube_evidence_gate_pass", False)),
        "tube_artifacts_current": bool(
            evidence.get("tube_artifacts_current", False)
        ),
        "support_evidence_gate_pass": bool(
            evidence.get("support_evidence_gate_pass", False)
        ),
        "dataset_evidence_gate_pass": bool(
            evidence.get("dataset_evidence_gate_pass", False)
        ),
        "dataset_artifacts_current": bool(
            evidence.get("dataset_artifacts_current", False)
        ),
        "five_seed_evidence_complete": bool(
            int(evidence.get("model_seed_count", 0)) == 5
            and int(evidence.get("model_unique_seed_count", 0)) == 5
        ),
        "model_gate_pass": bool(evidence.get("model_gate_pass", False)),
        "model_artifacts_current": bool(
            evidence.get("model_artifacts_current", False)
        ),
        "provenance_chain_current": bool(
            evidence.get("provenance_chain_current", False)
        ),
    }
    return {
        "checks": checks,
        "policy_registration_allowed": bool(all(checks.values())),
        "formal_claims_allowed": False,
    }


def compute_candidate_policy_frontiers(
    reports_by_radius: Mapping[float, Mapping[str, Any]],
    *,
    thresholds: Iterable[float],
    registered_checkpoints: Iterable[float],
) -> dict[float, dict[str, Any]]:
    normalized_reports = {float(radius): report for radius, report in reports_by_radius.items()}
    checkpoints = tuple(sorted(set(float(value) for value in registered_checkpoints)))
    output: dict[float, dict[str, Any]] = {}
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        frontier: float | None = None
        first_failed: float | None = None
        evidence: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            report = normalized_reports.get(checkpoint)
            if report is None:
                decision = {
                    "radius_mm": checkpoint,
                    "kappa_threshold": threshold,
                    "candidate_radius_policy_gate_pass": False,
                    "reason": "registered_checkpoint_report_missing",
                    "formal_claims_allowed": False,
                }
            else:
                decision = evaluate_candidate_radius_policy(
                    report,
                    kappa_threshold=threshold,
                )
            evidence.append(decision)
            if not bool(decision.get("candidate_radius_policy_gate_pass", False)):
                first_failed = checkpoint
                break
            frontier = checkpoint
        output[threshold] = {
            "strategy_version": SWEEP_STRATEGY_VERSION,
            "kappa_threshold": threshold,
            "registered_frontier_mm": frontier,
            "first_failed_checkpoint_mm": first_failed,
            "checkpoint_evidence": evidence,
            "formal_claims_allowed": False,
        }
    return output


def group_thresholds_by_frontier(
    frontiers: Mapping[float, Mapping[str, Any]],
) -> dict[float, tuple[float, ...]]:
    grouped: dict[float, list[float]] = {}
    for raw_threshold, report in frontiers.items():
        frontier = report.get("registered_frontier_mm")
        if frontier is None:
            continue
        grouped.setdefault(float(frontier), []).append(float(raw_threshold))
    return {
        frontier: tuple(sorted(thresholds))
        for frontier, thresholds in sorted(grouped.items())
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-formal V7 conditioning-kappa evidence sweep."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gate-v2-dir", type=Path, default=DEFAULT_GATE_V2_DIR)
    parser.add_argument("--v5-dir", type=Path, default=v7.DEFAULT_V5_DIR)
    parser.add_argument("--v6-dir", type=Path, default=v7.DEFAULT_V6_DIR)
    parser.add_argument("--robot-config", type=Path, default=v7.DEFAULT_CONFIG)
    parser.add_argument("--phases", default="all")
    parser.add_argument(
        "--kappa-thresholds",
        default=",".join(f"{value:g}" for value in REGISTERED_KAPPA_THRESHOLDS),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    parse_registered_thresholds(args.kappa_thresholds)
    parse_phases(args.phases)
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, default=v7._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
