#!/usr/bin/env python3
"""Validate frozen V12.11 Gold-set Students on newly generated loop geometries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import run_bacra_v12 as v12
import run_bacra_v12_7_canonical_student as canonical_runner
import run_bacra_v12_8_geometry_holdout as v128
import run_bacra_v12_11_gold_set_student as v1211
from quasi_exp.teacher.bacra_canonical_student import (
    evaluate_canonical_predictions,
)
from quasi_exp.teacher.bacra_geometry_holdout import (
    _margin_deg,
    cyclic_beta_metrics,
    family_from_row,
    generate_geometry_holdout,
    weighted_beta_gap_deg,
)
from quasi_exp.teacher.capability_path_bridge import (
    independently_certify_hard_cyclic_path,
)
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.gold_set_student import assign_cyclic_gold_section
from run_trajectory_canonical_teacher_v10 import (
    load_environment,
    runtime_fingerprint,
)


PROTOCOL_ID = (
    "branch-aware-canonical-region-atlas-v12.12-"
    "independent-gold-set-student-geometry-holdout"
)
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)
CORE_GROUP = "independent_core"
STRESS_GROUP = "independent_stress"
PREDICTED_BETA_COLUMNS = tuple(
    f"predicted_{name}" for name in BETA_COLUMNS
)


def _source_path(project_root: Path, value: str) -> Path:
    return v12._source_path(project_root, str(value))


def _gate(
    path: Path,
    checks: Mapping[str, bool],
    *,
    semantics: str,
    **evidence: Any,
) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "gate_semantics": str(semantics),
        "claim_scope": (
            "simulation_decision_independent_task_geometry_"
            "closed_loop_validation"
        ),
        "deployment_claim_gate_pass": False,
        "historical_sealed_artifacts_read": False,
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, payload)
    return payload


def _roots(
    config: Mapping[str, Any], project_root: Path
) -> tuple[Path, Path]:
    policy = config["independent_holdout"]
    return (
        _source_path(project_root, str(policy["source_v12_11_root"])),
        _source_path(project_root, str(policy["historical_v12_9_root"])),
    )


def _source_files(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    source_v12_11, historical_v12_9 = _roots(config, project_root)
    files = {
        "v12_11_training_gate": source_v12_11 / "01_training/gate.json",
        "v12_11_summary_gate": source_v12_11 / "02_summary/gate.json",
        "v12_11_completion_marker": (
            source_v12_11 / "V12_11_GOLD_SET_AWARE_STUDENT_TRAINED.json"
        ),
        "v12_11_metrics": (
            source_v12_11 / "01_training/metrics_per_family_seed.parquet"
        ),
        "historical_geometry_catalog": (
            historical_v12_9
            / "00_protocol/selected_geometry_holdout.parquet"
        ),
        "historical_v12_9_summary_gate": (
            historical_v12_9 / "02_summary/gate.json"
        ),
    }
    retry = config["independent_holdout"].get("teacher_retry")
    if retry:
        reuse_root = _source_path(
            project_root, str(retry["reuse_teacher_root"])
        )
        coarse_root = _source_path(
            project_root, str(retry["coarse_teacher_root"])
        )
        family_id = str(retry["multiresolution_family_id"])
        files.update(
            {
                "retry1_teacher_gate": (
                    reuse_root / "01_teacher_reference/gate.json"
                ),
                "coarse_teacher_gate": (
                    coarse_root / "01_teacher_reference/gate.json"
                ),
                "coarse_multiresolution_reference": (
                    coarse_root
                    / "01_teacher_reference/families"
                    / family_id
                    / "reference.parquet"
                ),
                "retry1_failed_family_report": (
                    reuse_root
                    / "01_teacher_reference/families"
                    / family_id
                    / "report.json"
                ),
            }
        )
    return files


def _model_path(
    config: Mapping[str, Any], project_root: Path, seed: int
) -> Path:
    source_v12_11, _historical = _roots(config, project_root)
    return source_v12_11 / f"01_training/seed_{int(seed)}/model.keras"


def _seed_report_path(
    config: Mapping[str, Any], project_root: Path, seed: int
) -> Path:
    source_v12_11, _historical = _roots(config, project_root)
    return source_v12_11 / f"01_training/seed_{int(seed)}/report.json"


def _rank_accepted_models(
    reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank only with frozen V12.11 evidence, before new geometry exists."""

    rows: list[dict[str, Any]] = []
    for report in reports:
        if not bool(report["seed_admission_gate_pass"]):
            continue
        families = list(report["families"])
        joint_p95 = [
            float(value)
            for family in families
            for value in family["validation_beta_abs_p95_by_joint_deg"]
        ]
        rows.append(
            {
                "seed": int(report["seed"]),
                "v12_11_worst_fk_p95_mm": max(
                    float(row["validation_fk_p95_mm"]) for row in families
                ),
                "v12_11_worst_fk_max_mm": max(
                    float(row["validation_fk_max_mm"]) for row in families
                ),
                "v12_11_minimum_margin_deg": min(
                    float(row["predicted_minimum_joint_margin_deg"])
                    for row in families
                ),
                "v12_11_worst_joint_abs_p95_deg": max(joint_p95),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["v12_11_worst_fk_p95_mm"],
            row["v12_11_worst_fk_max_mm"],
            -row["v12_11_minimum_margin_deg"],
            row["v12_11_worst_joint_abs_p95_deg"],
            row["seed"],
        ),
    )


def stage_protocol(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / "00_protocol"
    stage.mkdir(parents=True, exist_ok=True)
    policy = config["independent_holdout"]
    source_files = _source_files(config, project_root)
    actual_hashes = {
        name: sha256_file(path) if path.is_file() else None
        for name, path in source_files.items()
    }
    expected_hashes = {
        str(key): str(value)
        for key, value in policy["expected_source_sha256"].items()
    }

    training_gate = json.loads(
        source_files["v12_11_training_gate"].read_text(encoding="utf-8")
    )
    summary_gate = json.loads(
        source_files["v12_11_summary_gate"].read_text(encoding="utf-8")
    )
    seed_reports = [
        json.loads(
            _seed_report_path(config, project_root, seed).read_text(
                encoding="utf-8"
            )
        )
        for seed in training_gate["device_plan"]["seed_order"]
    ]
    ranking = _rank_accepted_models(seed_reports)
    accepted_seeds = [int(row["seed"]) for row in ranking]
    primary_seed = accepted_seeds[0] if accepted_seeds else None
    expected_model_hashes = {
        int(seed): str(digest)
        for seed, digest in policy["expected_model_sha256"].items()
    }
    model_rows = []
    for rank, row in enumerate(ranking):
        seed = int(row["seed"])
        model = _model_path(config, project_root, seed)
        model_rows.append(
            {
                **row,
                "selection_rank": rank,
                "role": "primary" if rank == 0 else "replication",
                "path": str(model),
                "sha256": sha256_file(model) if model.is_file() else None,
                "expected_sha256": expected_model_hashes.get(seed),
            }
        )
    model_lock = {
        "selection_timing": (
            "frozen_from_v12_11_metrics_before_v12_12_geometry_generation"
        ),
        "selection_rule": (
            "accepted_only_then_worst_fk_p95_asc,worst_fk_max_asc,"
            "minimum_margin_desc,worst_joint_p95_asc,seed_asc"
        ),
        "primary_seed": primary_seed,
        "accepted_seeds": accepted_seeds,
        "models": model_rows,
        "training_or_refit_allowed": False,
        "new_geometry_result_may_change_selection": False,
    }
    atomic_write_json(stage / "model_lock.json", model_lock)

    anchor_path = v128._source_files(config, project_root)[
        "anchor_task_json"
    ]
    anchor = v128._anchor_family(anchor_path)
    catalog = generate_geometry_holdout(
        anchor, config["geometry_holdout"]["groups"]
    )
    historical = pd.read_parquet(
        source_files["historical_geometry_catalog"]
    )
    overlap = sorted(
        set(catalog["family_fingerprint"])
        & set(historical["family_fingerprint"])
    )
    catalog_path = stage / "new_geometry_catalog.parquet"
    v12._atomic_parquet(catalog, catalog_path)
    catalog.to_csv(stage / "new_geometry_catalog.csv", index=False)
    atomic_write_json(stage / "frozen_config.json", dict(config))
    atomic_write_json(
        stage / "runtime.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "requested_python": str(python),
            "runtime_fingerprint": runtime_fingerprint(),
        },
    )
    implementation = {
        "runner": sha256_file(Path(__file__).resolve()),
        "geometry_solver": sha256_file(
            SOURCE_ROOT / "src/quasi_exp/teacher/bacra_geometry_holdout.py"
        ),
        "gold_set_assignment": sha256_file(
            SOURCE_ROOT / "src/quasi_exp/teacher/gold_set_student.py"
        ),
        "config": sha256_file(Path(str(config["config_path"]))),
    }
    atomic_write_json(
        stage / "source_manifest.json",
        {
            "source_paths": {
                name: str(path) for name, path in source_files.items()
            },
            "source_sha256": actual_hashes,
            "implementation_sha256": implementation,
            "new_geometry_catalog_sha256": sha256_file(catalog_path),
            "historical_geometry_overlap": overlap,
            "historical_sealed_artifacts_read": False,
        },
    )
    group_counts = catalog["group_id"].value_counts().to_dict()
    checks = {
        "v12_11_training_and_admission_passed": bool(
            training_gate["gate_pass"]
            and training_gate["student_admission_gate_pass"]
            and summary_gate["gate_pass"]
        ),
        "source_hashes_match": actual_hashes == expected_hashes,
        "exactly_four_preaccepted_models": (
            len(accepted_seeds) == int(policy["accepted_model_count"])
        ),
        "primary_seed_matches_preregistered_expectation": (
            primary_seed == int(policy["expected_primary_seed"])
        ),
        "all_model_hashes_match": bool(model_rows)
        and all(
            row["sha256"] == row["expected_sha256"] for row in model_rows
        ),
        "new_geometry_catalog_has_eight_families": len(catalog) == 8,
        "core_and_stress_groups_complete": (
            int(group_counts.get(CORE_GROUP, 0)) == 4
            and int(group_counts.get(STRESS_GROUP, 0)) == 4
        ),
        "no_historical_geometry_fingerprint_reused": not overlap,
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="model_lock_then_decision_independent_geometry_freeze",
        primary_seed=primary_seed,
        accepted_seeds=accepted_seeds,
        source_sha256=actual_hashes,
        geometry_catalog_sha256=sha256_file(catalog_path),
        historical_geometry_overlap=overlap,
        implementation_sha256=implementation,
    )


def teacher_worker(args: argparse.Namespace) -> int:
    return v128.teacher_worker(args)


def _upsample_cyclic_beta(
    coarse_beta: np.ndarray, fine_count: int
) -> np.ndarray:
    coarse = np.asarray(coarse_beta, dtype=float).reshape(-1, 6)
    coarse_axis = np.arange(len(coarse) + 1, dtype=float)
    fine_axis = (
        np.arange(int(fine_count), dtype=float)
        * float(len(coarse))
        / float(fine_count)
    )
    extended = np.vstack([coarse, coarse[0]])
    return np.column_stack(
        [
            np.interp(fine_axis, coarse_axis, extended[:, joint])
            for joint in range(6)
        ]
    )


def _correct_path_inside_gold_shell(
    environment: Any,
    targets: np.ndarray,
    seed_path: np.ndarray,
    *,
    gold_margin_deg: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    # Keep a tiny numerical interior so an exact 1.5° bound remains >= 1.5°.
    shell = np.deg2rad(float(gold_margin_deg) + 1.0e-5)
    lower = bounds[:, 0] + shell
    upper = bounds[:, 1] - shell
    corrected: list[np.ndarray] = []
    residuals: list[float] = []
    successes: list[bool] = []
    total_nfev = 0
    for target, seed in zip(
        np.asarray(targets, dtype=float),
        np.asarray(seed_path, dtype=float),
    ):
        def residual(beta: np.ndarray) -> np.ndarray:
            achieved = np.asarray(
                environment.fk(np.asarray(beta).reshape(1, 6)),
                dtype=float,
            ).reshape(3)
            return (achieved - target) / 0.001

        solved = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=int(max_nfev),
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
        )
        beta = np.asarray(solved.x, dtype=float).reshape(6)
        error_mm = float(np.linalg.norm(residual(beta)))
        corrected.append(beta)
        residuals.append(error_mm)
        # The 6->3 IK is underdetermined; SciPy may exhaust its iteration
        # budget after reaching an excellent finite residual. Scientific
        # validity is decided below by the independent path certificate.
        successes.append(bool(np.isfinite(beta).all()))
        total_nfev += int(solved.nfev)
    return (
        np.vstack(corrected),
        np.asarray(residuals, dtype=float),
        np.asarray(successes, dtype=bool),
        total_nfev,
    )


def multiresolution_teacher_worker(args: argparse.Namespace) -> int:
    """Lift an independent coarse Teacher cycle to the formal phase grid."""

    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    catalog = pd.read_parquet(args.catalog)
    selected = catalog.loc[catalog["family_id"].eq(args.family_id)]
    if len(selected) != 1:
        raise ValueError(f"unknown holdout family {args.family_id}")
    family = family_from_row(selected.iloc[0])
    environment = load_environment(
        project_root,
        v128._source_files(config, project_root)["robot_config"],
    )
    retry = config["independent_holdout"]["teacher_retry"]
    reuse_root = _source_path(
        project_root, str(retry["reuse_teacher_root"])
    )
    coarse_root = _source_path(
        project_root, str(retry["coarse_teacher_root"])
    )
    base_root = (
        reuse_root
        / "01_teacher_reference/families"
        / str(args.family_id)
    )
    coarse_root = (
        coarse_root
        / "01_teacher_reference/families"
        / str(args.family_id)
    )
    base_frame = pd.read_parquet(
        base_root / "reference.parquet"
    ).sort_values("phase_idx", kind="stable").reset_index(drop=True)
    base_report = json.loads(
        (base_root / "report.json").read_text(encoding="utf-8")
    )
    coarse_frame = pd.read_parquet(
        coarse_root / "reference.parquet"
    ).sort_values("phase_idx", kind="stable").reset_index(drop=True)
    fine_count = int(config["geometry_holdout"]["phase_count"])
    targets = family.centerline(phase_count=fine_count)
    forward_seed = _upsample_cyclic_beta(
        coarse_frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float),
        fine_count,
    )
    reverse_columns = [
        f"canonical_reverse_{name}" for name in BETA_COLUMNS
    ]
    reverse_seed = _upsample_cyclic_beta(
        coarse_frame.loc[:, reverse_columns].to_numpy(dtype=float),
        fine_count,
    )
    policy = v128._loop_policy(config)
    forward, forward_residual, forward_success, forward_nfev = (
        _correct_path_inside_gold_shell(
            environment,
            targets,
            forward_seed,
            gold_margin_deg=float(policy.joint_margin_min_deg),
            max_nfev=int(policy.teacher_policy.max_corrector_iterations),
        )
    )
    reverse, reverse_residual, reverse_success, reverse_nfev = (
        _correct_path_inside_gold_shell(
            environment,
            targets,
            reverse_seed,
            gold_margin_deg=float(policy.joint_margin_min_deg),
            max_nfev=int(policy.teacher_policy.max_corrector_iterations),
        )
    )
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    forward_achieved = np.asarray(
        environment.fk(forward), dtype=float
    ).reshape(fine_count, 3)
    reverse_achieved = np.asarray(
        environment.fk(reverse), dtype=float
    ).reshape(fine_count, 3)
    forward_certificate, certificate_detail = (
        independently_certify_hard_cyclic_path(
            forward,
            targets,
            forward_achieved,
            bounds,
            residual_p95_mm=float(policy.residual_p95_mm),
            residual_max_mm=float(policy.residual_max_mm),
            gold_margin_deg=float(policy.joint_margin_min_deg),
            max_transition_deg=float(policy.phase_beta_rms_max_deg),
        )
    )
    reverse_certificate, _reverse_detail = (
        independently_certify_hard_cyclic_path(
            reverse,
            targets,
            reverse_achieved,
            bounds,
            residual_p95_mm=float(policy.residual_p95_mm),
            residual_max_mm=float(policy.residual_max_mm),
            gold_margin_deg=float(policy.joint_margin_min_deg),
            max_transition_deg=float(policy.phase_beta_rms_max_deg),
        )
    )
    forward_trajectory = cyclic_beta_metrics(forward)
    reverse_trajectory = cyclic_beta_metrics(reverse)
    direction_gap = weighted_beta_gap_deg(
        forward,
        reverse,
        weights=policy.teacher_policy.beta_weights,
    )
    direction_gap_p95 = float(np.percentile(direction_gap, 95))
    direction_gap_max = float(np.max(direction_gap))
    combined_residual = np.concatenate(
        [forward_residual, reverse_residual]
    )
    combined_margin = np.concatenate(
        [_margin_deg(forward, bounds), _margin_deg(reverse, bounds)]
    )
    residual_p95 = float(np.percentile(combined_residual, 95))
    residual_max = float(np.max(combined_residual))
    margin_min = float(np.min(combined_margin))
    checks = {
        "phase_count_complete": len(base_frame) == fine_count,
        "root_chart_connection": bool(
            base_report["root_connection_success"]
        ),
        "canonical_candidate_layers_complete": bool(
            np.all(forward_success) and np.all(reverse_success)
        ),
        "actual_bounds_and_margin": bool(
            margin_min >= float(policy.joint_margin_min_deg)
        ),
        "fk_residual": bool(
            residual_p95 <= float(policy.residual_p95_mm)
            and residual_max <= float(policy.residual_max_mm)
        ),
        "cyclic_label_continuity": bool(
            forward_trajectory["phase_beta_rms_p95_deg"]
            <= float(policy.phase_beta_rms_p95_deg)
            and forward_trajectory["phase_beta_rms_max_deg"]
            <= float(policy.phase_beta_rms_max_deg)
            and forward_trajectory["acceleration_beta_rms_p95_deg"]
            <= float(policy.acceleration_beta_rms_p95_deg)
            and forward_trajectory["seam_beta_rms_deg"]
            <= float(policy.seam_beta_rms_deg)
            and reverse_trajectory["phase_beta_rms_p95_deg"]
            <= float(policy.phase_beta_rms_p95_deg)
            and reverse_trajectory["phase_beta_rms_max_deg"]
            <= float(policy.phase_beta_rms_max_deg)
            and reverse_trajectory["acceleration_beta_rms_p95_deg"]
            <= float(policy.acceleration_beta_rms_p95_deg)
            and reverse_trajectory["seam_beta_rms_deg"]
            <= float(policy.seam_beta_rms_deg)
        ),
        "canonical_order_audit_complete": bool(
            forward_certificate["gate_pass"]
            and reverse_certificate["gate_pass"]
        ),
        "direction_and_reference_invariance": bool(
            direction_gap_p95 <= float(policy.direction_gap_p95_deg)
            and direction_gap_max <= float(policy.direction_gap_max_deg)
        ),
        "independently_certified_hard_cyclic_path": bool(
            forward_certificate["gate_pass"]
        ),
    }
    frame = base_frame.copy()
    frame["reference_seed_source"] = (
        "independent_coarse_gold_cycle_multiresolution_lift"
    )
    frame["reference_corrector_success"] = forward_success
    frame["reference_residual_mm"] = forward_residual
    frame["reference_corrector_iterations"] = 0
    frame["reference_joint_margin_deg"] = _margin_deg(forward, bounds)
    for index, name in enumerate(BETA_COLUMNS):
        frame[name] = forward[:, index]
        frame[f"forward_{name}"] = forward[:, index]
        frame[f"reverse_{name}"] = reverse[:, index]
        frame[f"canonical_reverse_{name}"] = reverse[:, index]
    frame["forward_residual_mm"] = forward_residual
    frame["reverse_residual_mm"] = reverse_residual
    frame["forward_success"] = forward_success
    frame["reverse_success"] = reverse_success
    for name, values in certificate_detail.items():
        frame[name] = values

    candidate_rows: list[dict[str, Any]] = []
    for phase_idx in range(fine_count):
        for candidate_idx, (source, beta, residual_mm) in enumerate(
            (
                (
                    "multiresolution_forward",
                    forward[phase_idx],
                    forward_residual[phase_idx],
                ),
                (
                    "multiresolution_reverse",
                    reverse[phase_idx],
                    reverse_residual[phase_idx],
                ),
            )
        ):
            candidate_rows.append(
                {
                    "family_id": family.family_id,
                    "group_id": str(selected.iloc[0]["group_id"]),
                    "phase_idx": phase_idx,
                    "candidate_idx": candidate_idx,
                    "source": source,
                    "residual_mm": float(residual_mm),
                    "minimum_joint_margin_deg": float(
                        _margin_deg(beta.reshape(1, 6), bounds)[0]
                    ),
                    **{
                        name: float(beta[index])
                        for index, name in enumerate(BETA_COLUMNS)
                    },
                }
            )
    report = {
        "family_id": family.family_id,
        "group_id": str(selected.iloc[0]["group_id"]),
        "family_fingerprint": family.fingerprint,
        "phase_count": fine_count,
        "supported_phase_count": int(
            base_report["supported_phase_count"]
        ),
        "chart_support_is_diagnostic_only": True,
        "root_phase_idx": int(base_report["root_phase_idx"]),
        "root_seed_source": (
            "coarse_teacher_cycle_with_retry1_chart_connection"
        ),
        "root_anchor_index": int(base_report["root_anchor_index"]),
        "root_anchor_distance_mm": float(
            base_report["root_anchor_distance_mm"]
        ),
        "root_waypoint_count": int(base_report["root_waypoint_count"]),
        "root_corrector_iterations": int(
            base_report["root_corrector_iterations"]
        ),
        "root_residual_mm": float(base_report["root_residual_mm"]),
        "root_joint_margin_deg": float(
            base_report["root_joint_margin_deg"]
        ),
        "root_max_substep_residual_mm": float(
            base_report["root_max_substep_residual_mm"]
        ),
        "root_connection_success": bool(
            base_report["root_connection_success"]
        ),
        "reference_method": (
            "independent_coarse_gold_cycle_multiresolution_lift"
        ),
        "coarse_reference_sha256": sha256_file(
            coarse_root / "reference.parquet"
        ),
        "retry1_report_sha256": sha256_file(
            base_root / "report.json"
        ),
        "candidate_attempt_count": int(forward_nfev + reverse_nfev),
        "gap_enrichment_enabled": False,
        "gap_enrichment": None,
        "continuation_warm_start_phase_count": fine_count,
        "gold_candidate_layer_count": fine_count,
        "gold_candidate_layer_min_count": 2,
        "gold_candidate_layer_max_count": 2,
        "primary_link_report": {
            "success": bool(forward_certificate["gate_pass"]),
            "method": "multiresolution_forward_path",
        },
        "reverse_link_report": {
            "success": bool(reverse_certificate["gate_pass"]),
            "method": "multiresolution_reverse_path",
        },
        "independent_hard_cyclic_certificate": forward_certificate,
        "raw_forward_complete": bool(np.all(forward_success)),
        "raw_reverse_complete": bool(np.all(reverse_success)),
        "raw_direction_gap_p95_deg": direction_gap_p95,
        "raw_direction_gap_max_deg": direction_gap_max,
        "raw_minimum_joint_margin_deg": margin_min,
        "forward_chart_recovery_count": 0,
        "reverse_chart_recovery_count": 0,
        "corrected_phase_count": fine_count,
        "teacher_success_rate": float(np.mean(forward_success)),
        "residual_p95_mm": residual_p95,
        "residual_max_mm": residual_max,
        "minimum_joint_margin_deg": margin_min,
        "trajectory": forward_trajectory,
        "forward_trajectory": forward_trajectory,
        "reverse_trajectory": reverse_trajectory,
        "direction_gap_p95_deg": direction_gap_p95,
        "direction_gap_max_deg": direction_gap_max,
        "canonical_forward_closure_gap_deg": float(
            forward_trajectory["seam_beta_rms_deg"]
        ),
        "canonical_reverse_closure_gap_deg": float(
            reverse_trajectory["seam_beta_rms_deg"]
        ),
        "forward_closure_gap_deg": float(
            forward_trajectory["seam_beta_rms_deg"]
        ),
        "reverse_closure_gap_deg": float(
            reverse_trajectory["seam_beta_rms_deg"]
        ),
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
    }
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    v12._atomic_parquet(frame, destination / "reference.parquet")
    v12._atomic_parquet(
        pd.DataFrame(candidate_rows),
        destination / "gold_candidates.parquet",
    )
    atomic_write_json(destination / "report.json", report)
    print(
        json.dumps(
            {
                "family_id": family.family_id,
                "gate_pass": report["gate_pass"],
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


def _flatten_teacher_report(report: Mapping[str, Any]) -> dict[str, Any]:
    row = v128._flatten_teacher_report(report)
    certificate = report.get("independent_hard_cyclic_certificate") or {}
    row.update(
        {
            f"certificate_{key}": value
            for key, value in certificate.items()
            if key != "checks"
        }
    )
    row.update(
        {
            f"certificate_check_{key}": bool(value)
            for key, value in (certificate.get("checks") or {}).items()
        }
    )
    return row


def stage_teacher(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / "01_teacher_reference"
    stage.mkdir(parents=True, exist_ok=True)
    catalog_path = output_root / "00_protocol/new_geometry_catalog.parquet"
    catalog = pd.read_parquet(catalog_path).sort_values(
        "catalog_order", kind="stable"
    )
    commands: list[tuple[str, Sequence[str], Path]] = []
    retry = config["independent_holdout"].get("teacher_retry")
    reused_families: list[dict[str, Any]] = []
    for row in catalog.itertuples(index=False):
        family_id = str(row.family_id)
        if retry and family_id != str(
            retry["multiresolution_family_id"]
        ):
            reuse_root = _source_path(
                project_root, str(retry["reuse_teacher_root"])
            )
            source = (
                reuse_root
                / "01_teacher_reference/families"
                / family_id
            )
            destination = stage / "families" / family_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            reused_families.append(
                {
                    "family_id": family_id,
                    "source": str(source),
                    "report_sha256": sha256_file(source / "report.json"),
                    "reference_sha256": sha256_file(
                        source / "reference.parquet"
                    ),
                    "gold_candidates_sha256": sha256_file(
                        source / "gold_candidates.parquet"
                    ),
                }
            )
            continue
        worker = "multires_teacher" if retry else "teacher"
        commands.append(
            (
                family_id,
                [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    worker,
                    "--config",
                    str(config["config_path"]),
                    "--preset",
                    str(config["preset"]),
                    "--project-root",
                    str(project_root),
                    "--output-root",
                    str(output_root),
                    "--catalog",
                    str(catalog_path),
                    "--family-id",
                    family_id,
                    "--worker-output",
                    str(stage / "families" / family_id),
                ],
                stage / "logs" / f"{family_id}.log",
            )
        )
    started = time.time()
    parallel = v12._run_subprocess_tasks(
        commands,
        requested_workers=min(
            int(config["parallel"]["teacher_workers"]), len(commands)
        ),
        manifest_path=stage / "parallel_manifest.json",
    )
    atomic_write_json(
        stage / "teacher_artifact_reuse.json",
        {
            "reuse_enabled": bool(retry),
            "reused_families": reused_families,
            "recomputed_family_ids": [
                str(task_id) for task_id, _command, _log in commands
            ],
        },
    )
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for family_id in catalog["family_id"].astype(str):
        family_root = stage / "families" / family_id
        report = json.loads(
            (family_root / "report.json").read_text(encoding="utf-8")
        )
        rows.append(_flatten_teacher_report(report))
        frames.append(pd.read_parquet(family_root / "reference.parquet"))
    metrics = pd.DataFrame(rows).sort_values(
        ["group_id", "family_id"], kind="stable"
    )
    reference = pd.concat(frames, ignore_index=True).sort_values(
        ["group_id", "family_id", "phase_idx"], kind="stable"
    )
    v12._atomic_parquet(
        metrics, stage / "teacher_metrics_per_family.parquet"
    )
    v12._atomic_parquet(
        reference, stage / "reference_dataset.parquet"
    )
    core = metrics.loc[metrics["group_id"].eq(CORE_GROUP)]
    stress = metrics.loc[metrics["group_id"].eq(STRESS_GROUP)]
    core_pass = bool(
        len(core) == 4
        and core["gate_pass"].all()
        and core["certificate_gate_pass"].all()
    )
    stress_pass = bool(
        len(stress) == 4
        and stress["gate_pass"].all()
        and stress["certificate_gate_pass"].all()
    )
    return _gate(
        stage / "gate.json",
        {
            "protocol_gate_pass": bool(
                json.loads(
                    (output_root / "00_protocol/gate.json").read_text(
                        encoding="utf-8"
                    )
                )["gate_pass"]
            ),
            "all_eight_teacher_tasks_complete": len(metrics) == 8,
            "all_four_core_have_independent_hard_cyclic_gold_paths": core_pass,
        },
        semantics="new_geometry_independent_hard_cyclic_teacher_certificate",
        core_teacher_gate_pass=core_pass,
        stress_teacher_gate_pass=stress_pass,
        core_pass_count=int(
            (core["gate_pass"] & core["certificate_gate_pass"]).sum()
        ),
        stress_pass_count=int(
            (stress["gate_pass"] & stress["certificate_gate_pass"]).sum()
        ),
        parallel_evidence=parallel,
        reused_teacher_artifacts=reused_families,
        wall_time_s=float(time.time() - started),
        reference_dataset_sha256=sha256_file(
            stage / "reference_dataset.parquet"
        ),
    )


def prediction_worker(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    output_root = Path(args.output_root).resolve()
    project_root = Path(args.project_root).resolve()
    reference = pd.read_parquet(args.reference_dataset).sort_values(
        ["group_id", "family_id", "phase_idx"], kind="stable"
    )
    model_lock = json.loads(
        (output_root / "00_protocol/model_lock.json").read_text(
            encoding="utf-8"
        )
    )
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(
        int(config["parallel"]["student_intraop_threads"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(config["parallel"]["student_interop_threads"])
    )
    gpus = tf.config.list_physical_devices("GPU")
    xyz = reference.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    frames: list[pd.DataFrame] = []
    model_reports: list[dict[str, Any]] = []
    for model_row in model_lock["models"]:
        seed = int(model_row["seed"])
        model_path = _model_path(config, project_root, seed)
        model = tf.keras.models.load_model(model_path, compile=False)
        prediction = np.asarray(
            model.predict(xyz, batch_size=256, verbose=0), dtype=float
        )
        frame = reference.loc[
            :, ["family_id", "group_id", "phase_idx", *XYZ_COLUMNS]
        ].copy()
        frame.insert(0, "seed", seed)
        for index, name in enumerate(PREDICTED_BETA_COLUMNS):
            frame[name] = prediction[:, index]
        frames.append(frame)
        model_reports.append(
            {
                "seed": seed,
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "prediction_finite": bool(np.isfinite(prediction).all()),
            }
        )
        del model
        tf.keras.backend.clear_session()
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["seed", "group_id", "family_id", "phase_idx"], kind="stable"
    )
    v12._atomic_parquet(predictions, destination / "predictions.parquet")
    report = {
        "device": "gpu:0" if gpus else "cpu",
        "gpu_names": [device.name for device in gpus],
        "model_count": len(model_reports),
        "models": model_reports,
        "predictions_sha256": sha256_file(
            destination / "predictions.parquet"
        ),
    }
    atomic_write_json(destination / "report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


def _point_margins_deg(
    prediction: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    per_joint = np.rad2deg(
        np.minimum(
            prediction - bounds[:, 0][None, :],
            bounds[:, 1][None, :] - prediction,
        )
    )
    return per_joint, np.min(per_joint, axis=1)


def evaluation_worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    seed = int(args.seed)
    reference = pd.read_parquet(args.reference_dataset).sort_values(
        ["group_id", "family_id", "phase_idx"], kind="stable"
    )
    all_predictions = pd.read_parquet(args.predictions)
    predictions = all_predictions.loc[
        all_predictions["seed"].eq(seed)
    ].sort_values(["group_id", "family_id", "phase_idx"], kind="stable")
    environment = load_environment(
        project_root,
        v128._source_files(config, project_root)["robot_config"],
    )
    geometry = canonical_runner._geometry(config, project_root)
    teacher_metrics = pd.read_parquet(args.teacher_metrics).set_index(
        "family_id"
    )
    admission = config["exploratory_admission"]
    trajectory_limits = config["student_trajectory_gate"]
    hard_transition = float(
        config["independent_holdout"]["hard_transition_deg"]
    )
    reports: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    for family_id, family_reference in reference.groupby(
        "family_id", sort=True
    ):
        family_reference = family_reference.sort_values(
            "phase_idx", kind="stable"
        ).reset_index(drop=True)
        family_prediction_frame = predictions.loc[
            predictions["family_id"].eq(family_id)
        ].sort_values("phase_idx", kind="stable")
        prediction = family_prediction_frame.loc[
            :, PREDICTED_BETA_COLUMNS
        ].to_numpy(dtype=float)
        candidates = pd.read_parquet(
            output_root
            / "01_teacher_reference/families"
            / str(family_id)
            / "gold_candidates.parquet"
        )
        assignment, assignment_report = assign_cyclic_gold_section(
            prediction,
            candidates,
            max_transition_deg=hard_transition,
        )
        assignment_succeeded = bool(
            assignment_report.get("success", False)
        )
        if assignment_succeeded:
            assigned_reference = v1211._make_assignment_training_frame(
                family_reference, assignment
            )
            assigned_beta = assignment.loc[
                :, BETA_COLUMNS
            ].to_numpy(dtype=float)
            assigned_candidate_idx = assignment[
                "candidate_idx"
            ].to_numpy(dtype=int)
        else:
            # Preserve raw Student diagnostics when the scientific Gate fails.
            # This fallback is never accepted as a Gold-set assignment.
            assigned_reference = family_reference.copy()
            assigned_beta = family_reference.loc[
                :, BETA_COLUMNS
            ].to_numpy(dtype=float)
            assigned_candidate_idx = np.full(
                len(family_reference), -1, dtype=int
            )
        metrics = evaluate_canonical_predictions(
            assigned_reference, prediction, geometry=geometry
        )
        trajectory = v1211._trajectory_metrics(prediction)
        joint_p95 = list(
            map(
                float,
                metrics["validation_beta_abs_p95_by_joint_deg"],
            )
        )
        teacher_gate_pass = bool(
            teacher_metrics.loc[family_id, "gate_pass"]
            and teacher_metrics.loc[family_id, "certificate_gate_pass"]
        )
        checks = {
            "teacher_independent_hard_cycle_available": teacher_gate_pass,
            "prediction_to_hard_gold_assignment_succeeded": (
                assignment_succeeded
            ),
            "all_joint_abs_p95_below_limit": bool(
                max(joint_p95)
                < float(admission["joint_abs_p95_max_deg"])
            ),
            "fk_p95_below_limit": bool(
                float(metrics["validation_fk_p95_mm"])
                < float(admission["fk_p95_max_mm"])
            ),
            "fk_max_below_limit": bool(
                float(metrics["validation_fk_max_mm"])
                < float(admission["fk_max_mm"])
            ),
            "minimum_joint_margin_above_limit": bool(
                float(metrics["predicted_minimum_joint_margin_deg"])
                > float(admission["minimum_joint_margin_min_deg"])
            ),
            "raw_prediction_cyclic_continuity": bool(
                trajectory["phase_beta_rms_p95_deg"]
                <= float(trajectory_limits["phase_beta_rms_p95_deg"])
                and trajectory["phase_beta_rms_max_deg"]
                <= float(trajectory_limits["phase_beta_rms_max_deg"])
                and trajectory["acceleration_beta_rms_p95_deg"]
                <= float(
                    trajectory_limits[
                        "acceleration_beta_rms_p95_deg"
                    ]
                )
                and trajectory["seam_beta_rms_deg"]
                <= float(trajectory_limits["seam_beta_rms_deg"])
            ),
        }
        achieved = np.asarray(environment.fk(prediction), dtype=float)
        target = family_reference.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
        fk_error_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
        per_joint_margin, point_margin = _point_margins_deg(
            prediction, np.asarray(environment.bounds, dtype=float)
        )
        gold_rms = np.rad2deg(
            np.sqrt(np.mean(np.square(prediction - assigned_beta), axis=1))
        )
        report = {
            "seed": seed,
            "family_id": str(family_id),
            "group_id": str(family_reference.iloc[0]["group_id"]),
            "student_family_gate_pass": bool(all(checks.values())),
            "checks": checks,
            "gold_assignment": assignment_report,
            "worst_joint_abs_p95_deg": max(joint_p95),
            **metrics,
            **trajectory,
        }
        reports.append(report)
        detail = family_reference.loc[
            :, ["family_id", "group_id", "phase_idx", *XYZ_COLUMNS]
        ].copy()
        detail.insert(0, "seed", seed)
        for index, name in enumerate(BETA_COLUMNS):
            detail[f"predicted_{name}"] = prediction[:, index]
            detail[f"gold_{name}"] = assigned_beta[:, index]
            detail[f"margin_{name}_deg"] = per_joint_margin[:, index]
        detail["achieved_x_m"] = achieved[:, 0]
        detail["achieved_y_m"] = achieved[:, 1]
        detail["achieved_z_m"] = achieved[:, 2]
        detail["fk_error_mm"] = fk_error_mm
        detail["predicted_minimum_joint_margin_deg"] = point_margin
        detail["prediction_to_gold_rms_deg"] = gold_rms
        detail["gold_candidate_idx"] = assigned_candidate_idx
        detail_frames.append(detail)
        assigned = assigned_reference.copy()
        assigned.insert(0, "seed", seed)
        assignment_frames.append(assigned)
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    v12._atomic_parquet(
        pd.concat(detail_frames, ignore_index=True),
        destination / "tracking_details.parquet",
    )
    v12._atomic_parquet(
        pd.concat(assignment_frames, ignore_index=True),
        destination / "assigned_gold_sections.parquet",
    )
    atomic_write_json(
        destination / "report.json",
        {
            "seed": seed,
            "families": reports,
            "all_assignments_succeeded": all(
                row["gold_assignment"]["success"] for row in reports
            ),
        },
    )
    print(
        json.dumps(
            {"seed": seed, "family_count": len(reports)}, sort_keys=True
        )
    )
    return 0


def _flatten_student_report(report: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "checks",
        "gold_assignment",
        "validation_beta_abs_p50_by_joint_deg",
        "validation_beta_abs_p95_by_joint_deg",
        "validation_beta_abs_max_by_joint_deg",
        "validation_beta_signed_bias_by_joint_deg",
    }
    row = {key: value for key, value in report.items() if key not in excluded}
    row.update(
        {f"check_{key}": bool(value) for key, value in report["checks"].items()}
    )
    row.update(
        {
            f"gold_assignment_{key}": value
            for key, value in report["gold_assignment"].items()
        }
    )
    row.update(
        {
            f"gold_abs_p95_{joint}_deg": float(value)
            for joint, value in zip(
                BETA_COLUMNS,
                report["validation_beta_abs_p95_by_joint_deg"],
            )
        }
    )
    return row


def stage_student(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / "02_student_evaluation"
    stage.mkdir(parents=True, exist_ok=True)
    model_lock = json.loads(
        (output_root / "00_protocol/model_lock.json").read_text(
            encoding="utf-8"
        )
    )
    seeds = tuple(map(int, model_lock["accepted_seeds"]))
    reference = output_root / "01_teacher_reference/reference_dataset.parquet"
    teacher_metrics = (
        output_root
        / "01_teacher_reference/teacher_metrics_per_family.parquet"
    )
    prediction_command = [
        str(python),
        str(Path(__file__).resolve()),
        "--worker",
        "predict",
        "--config",
        str(config["config_path"]),
        "--preset",
        str(config["preset"]),
        "--project-root",
        str(project_root),
        "--output-root",
        str(output_root),
        "--reference-dataset",
        str(reference),
        "--worker-output",
        str(stage / "prediction"),
    ]
    prediction_parallel = v12._run_subprocess_tasks(
        [
            (
                "gpu_batch_prediction",
                prediction_command,
                stage / "logs/gpu_batch_prediction.log",
            )
        ],
        requested_workers=1,
        manifest_path=stage / "prediction_parallel_manifest.json",
    )
    predictions = stage / "prediction/predictions.parquet"
    commands: list[tuple[str, Sequence[str], Path]] = []
    for seed in seeds:
        task_id = f"seed_{seed}"
        commands.append(
            (
                task_id,
                [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "evaluate",
                    "--config",
                    str(config["config_path"]),
                    "--preset",
                    str(config["preset"]),
                    "--project-root",
                    str(project_root),
                    "--output-root",
                    str(output_root),
                    "--seed",
                    str(seed),
                    "--reference-dataset",
                    str(reference),
                    "--teacher-metrics",
                    str(teacher_metrics),
                    "--predictions",
                    str(predictions),
                    "--worker-output",
                    str(stage / task_id),
                ],
                stage / "logs" / f"{task_id}.log",
            )
        )
    evaluation_parallel = v12._run_subprocess_tasks(
        commands,
        requested_workers=int(
            config["parallel"]["evaluation_cpu_workers"]
        ),
        manifest_path=stage / "evaluation_parallel_manifest.json",
    )
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        report = json.loads(
            (stage / f"seed_{seed}/report.json").read_text(
                encoding="utf-8"
            )
        )
        flattened = [
            _flatten_student_report(family) for family in report["families"]
        ]
        rows.extend(flattened)
        family_frame = pd.DataFrame(flattened)
        core = family_frame.loc[family_frame["group_id"].eq(CORE_GROUP)]
        stress = family_frame.loc[family_frame["group_id"].eq(STRESS_GROUP)]
        seed_rows.append(
            {
                "seed": seed,
                "core_family_pass_count": int(
                    core["student_family_gate_pass"].sum()
                ),
                "core_seed_gate_pass": bool(
                    len(core) == 4 and core["student_family_gate_pass"].all()
                ),
                "stress_family_pass_count": int(
                    stress["student_family_gate_pass"].sum()
                ),
                "stress_seed_gate_pass": bool(
                    len(stress) == 4
                    and stress["student_family_gate_pass"].all()
                ),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["group_id", "family_id", "seed"], kind="stable"
    )
    seed_metrics = pd.DataFrame(seed_rows).sort_values("seed", kind="stable")
    v12._atomic_parquet(
        metrics, stage / "metrics_per_family_seed.parquet"
    )
    v12._atomic_parquet(seed_metrics, stage / "metrics_per_seed.parquet")
    core_pass_count = int(seed_metrics["core_seed_gate_pass"].sum())
    stress_pass_count = int(seed_metrics["stress_seed_gate_pass"].sum())
    primary_seed = int(model_lock["primary_seed"])
    primary_core_pass = bool(
        seed_metrics.loc[
            seed_metrics["seed"].eq(primary_seed), "core_seed_gate_pass"
        ].iloc[0]
    )
    required = int(
        config["independent_holdout"]["required_core_seed_passes"]
    )
    prediction_report = json.loads(
        (stage / "prediction/report.json").read_text(encoding="utf-8")
    )
    return _gate(
        stage / "gate.json",
        {
            "all_four_frozen_models_evaluated": len(seed_metrics) == 4,
            "all_prediction_to_gold_assignments_succeeded": bool(
                metrics["gold_assignment_success"].all()
            ),
            "preregistered_primary_passes_all_core_geometries": (
                primary_core_pass
            ),
            "required_replication_models_pass_all_core_geometries": (
                core_pass_count >= required
            ),
        },
        semantics="frozen_student_on_decision_independent_geometry",
        primary_seed=primary_seed,
        primary_core_gate_pass=primary_core_pass,
        core_seed_pass_count=core_pass_count,
        stress_seed_pass_count=stress_pass_count,
        required_core_seed_passes=required,
        prediction_device=prediction_report["device"],
        gpu_prediction_evidence=prediction_report,
        prediction_parallel_evidence=prediction_parallel,
        evaluation_parallel_evidence=evaluation_parallel,
    )


def _family_axes(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = row.loc[["center_x_m", "center_y_m", "center_z_m"]].to_numpy(
        dtype=float
    )
    major = row.loc[["major_x", "major_y", "major_z"]].to_numpy(dtype=float)
    minor = row.loc[["minor_x", "minor_y", "minor_z"]].to_numpy(dtype=float)
    return center, major, minor


def _plot_tracking_panel(
    detail: pd.DataFrame,
    family_row: pd.Series,
    destination: Path,
    *,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quasi-exp-mpl")
    import matplotlib.pyplot as plt

    target = detail.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    achieved = detail.loc[
        :, ["achieved_x_m", "achieved_y_m", "achieved_z_m"]
    ].to_numpy(dtype=float)
    center, major, minor = _family_axes(family_row)
    target_uv = np.column_stack(
        [(target - center) @ major, (target - center) @ minor]
    )
    achieved_uv = np.column_stack(
        [(achieved - center) @ major, (achieved - center) @ minor]
    )
    phase = detail["phase_idx"].to_numpy(dtype=int)
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax3d.plot(
        target[:, 0] * 1000,
        target[:, 1] * 1000,
        target[:, 2] * 1000,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="target",
    )
    ax3d.plot(
        achieved[:, 0] * 1000,
        achieved[:, 1] * 1000,
        achieved[:, 2] * 1000,
        color="#1f77b4",
        linewidth=1.2,
        label="Student FK",
    )
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.legend(loc="best")

    ax_plane = fig.add_subplot(grid[0, 1])
    ax_plane.plot(
        target_uv[:, 0] * 1000,
        target_uv[:, 1] * 1000,
        "k--",
        linewidth=1.6,
        label="target",
    )
    ax_plane.plot(
        achieved_uv[:, 0] * 1000,
        achieved_uv[:, 1] * 1000,
        color="#1f77b4",
        linewidth=1.2,
        label="Student FK",
    )
    ax_plane.set_aspect("equal", adjustable="box")
    ax_plane.set_xlabel("major-axis coordinate (mm)")
    ax_plane.set_ylabel("minor-axis coordinate (mm)")
    ax_plane.set_title("Task-plane tracking")
    ax_plane.legend(loc="best")

    ax_error = fig.add_subplot(grid[1, 0])
    ax_error.plot(phase, detail["fk_error_mm"], color="#d62728")
    ax_error.axhline(5.0, color="black", linestyle="--", label="P95 gate 5 mm")
    ax_error.axhline(10.0, color="gray", linestyle=":", label="max gate 10 mm")
    ax_error.set_xlabel("phase")
    ax_error.set_ylabel("FK error (mm)")
    ax_error.set_title("Raw Student tracking error")
    ax_error.legend(loc="best")

    ax_margin = fig.add_subplot(grid[1, 1])
    ax_margin.plot(
        phase,
        detail["predicted_minimum_joint_margin_deg"],
        color="#2ca02c",
        label="minimum joint margin",
    )
    ax_margin.axhline(
        1.5, color="black", linestyle="--", label="Gold margin 1.5°"
    )
    ax_margin.set_xlabel("phase")
    ax_margin.set_ylabel("margin (deg)")
    ax_margin.set_title("Raw Student mechanical margin")
    ax_margin.legend(loc="best")
    fig.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _plot_primary_overview(
    details: pd.DataFrame,
    catalog: pd.DataFrame,
    primary_seed: int,
    destination: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quasi-exp-mpl")
    import matplotlib.pyplot as plt

    core = catalog.loc[catalog["group_id"].eq(CORE_GROUP)].sort_values(
        "catalog_order", kind="stable"
    )
    fig, axes = plt.subplots(
        len(core), 2, figsize=(13, 3.3 * len(core)), constrained_layout=True
    )
    for row_index, family_row in enumerate(core.to_dict("records")):
        family_id = str(family_row["family_id"])
        frame = details.loc[
            details["seed"].eq(primary_seed)
            & details["family_id"].eq(family_id)
        ].sort_values("phase_idx", kind="stable")
        target = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
        achieved = frame.loc[
            :, ["achieved_x_m", "achieved_y_m", "achieved_z_m"]
        ].to_numpy(dtype=float)
        center = np.asarray(
            [
                family_row["center_x_m"],
                family_row["center_y_m"],
                family_row["center_z_m"],
            ],
            dtype=float,
        )
        major = np.asarray(
            [family_row["major_x"], family_row["major_y"], family_row["major_z"]],
            dtype=float,
        )
        minor = np.asarray(
            [family_row["minor_x"], family_row["minor_y"], family_row["minor_z"]],
            dtype=float,
        )
        target_uv = np.column_stack(
            [(target - center) @ major, (target - center) @ minor]
        )
        achieved_uv = np.column_stack(
            [(achieved - center) @ major, (achieved - center) @ minor]
        )
        axes[row_index, 0].plot(
            target_uv[:, 0] * 1000,
            target_uv[:, 1] * 1000,
            "k--",
            linewidth=1.4,
            label="target",
        )
        axes[row_index, 0].plot(
            achieved_uv[:, 0] * 1000,
            achieved_uv[:, 1] * 1000,
            color="#1f77b4",
            linewidth=1.0,
            label="Student FK",
        )
        axes[row_index, 0].set_aspect("equal", adjustable="box")
        axes[row_index, 0].set_title(f"{family_id}: task-plane trajectory")
        axes[row_index, 0].set_xlabel("major (mm)")
        axes[row_index, 0].set_ylabel("minor (mm)")
        axes[row_index, 1].plot(
            frame["phase_idx"], frame["fk_error_mm"], color="#d62728"
        )
        axes[row_index, 1].axhline(5.0, color="black", linestyle="--")
        axes[row_index, 1].set_title(
            f"FK error: P95={np.percentile(frame['fk_error_mm'], 95):.3f} mm, "
            f"max={frame['fk_error_mm'].max():.3f} mm"
        )
        axes[row_index, 1].set_xlabel("phase")
        axes[row_index, 1].set_ylabel("error (mm)")
    axes[0, 0].legend(loc="best")
    fig.suptitle(
        f"Preregistered primary Gold-set-aware Student: seed {primary_seed}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def stage_visualize(
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    stage = output_root / "03_visualization"
    stage.mkdir(parents=True, exist_ok=True)
    model_lock = json.loads(
        (output_root / "00_protocol/model_lock.json").read_text(
            encoding="utf-8"
        )
    )
    primary_seed = int(model_lock["primary_seed"])
    seed_metrics = pd.read_parquet(
        output_root / "02_student_evaluation/metrics_per_seed.parquet"
    )
    good_seeds = seed_metrics.loc[
        seed_metrics["core_seed_gate_pass"], "seed"
    ].astype(int).tolist()
    plotted_seeds = good_seeds or [primary_seed]
    details = pd.concat(
        [
            pd.read_parquet(
                output_root
                / f"02_student_evaluation/seed_{seed}/tracking_details.parquet"
            )
            for seed in model_lock["accepted_seeds"]
        ],
        ignore_index=True,
    )
    catalog = pd.read_parquet(
        output_root / "00_protocol/new_geometry_catalog.parquet"
    )
    family_lookup = catalog.set_index("family_id")
    outputs: list[str] = []
    for seed in plotted_seeds:
        family_ids = catalog.loc[
            catalog["group_id"].eq(CORE_GROUP), "family_id"
        ].astype(str).tolist()
        if seed == primary_seed:
            family_ids = catalog["family_id"].astype(str).tolist()
        for family_id in family_ids:
            frame = details.loc[
                details["seed"].eq(seed)
                & details["family_id"].eq(family_id)
            ].sort_values("phase_idx", kind="stable")
            path = stage / f"seed_{seed}/{family_id}_tracking.png"
            _plot_tracking_panel(
                frame,
                family_lookup.loc[family_id],
                path,
                title=(
                    f"V12.12 new geometry | seed {seed} | {family_id} | "
                    "raw Student FK"
                ),
            )
            outputs.append(str(path.relative_to(output_root)))
    overview = stage / "best_model_tracking_overview.png"
    _plot_primary_overview(details, catalog, primary_seed, overview)
    outputs.append(str(overview.relative_to(output_root)))
    manifest = {
        "selection_semantics": (
            "primary was frozen before holdout; additional plotted models are "
            "post-evaluation visualization only"
        ),
        "primary_seed": primary_seed,
        "holdout_core_passing_seeds": good_seeds,
        "plotted_seeds": plotted_seeds,
        "image_count": len(outputs),
        "images": outputs,
    }
    atomic_write_json(stage / "manifest.json", manifest)
    return _gate(
        stage / "gate.json",
        {
            "primary_overview_created": overview.is_file(),
            "every_plotted_seed_has_all_core_tracking_panels": all(
                all(
                    (
                        stage
                        / f"seed_{seed}/{family_id}_tracking.png"
                    ).is_file()
                    for family_id in catalog.loc[
                        catalog["group_id"].eq(CORE_GROUP), "family_id"
                    ].astype(str)
                )
                for seed in plotted_seeds
            ),
        },
        semantics="tracking_visualization_artifact_completion",
        **manifest,
    )


def stage_summary(
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    stage = output_root / "04_summary"
    stage.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(
        (output_root / "00_protocol/gate.json").read_text(encoding="utf-8")
    )
    teacher = json.loads(
        (output_root / "01_teacher_reference/gate.json").read_text(
            encoding="utf-8"
        )
    )
    student = json.loads(
        (output_root / "02_student_evaluation/gate.json").read_text(
            encoding="utf-8"
        )
    )
    visual = json.loads(
        (output_root / "03_visualization/gate.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = pd.read_parquet(
        output_root / "02_student_evaluation/metrics_per_family_seed.parquet"
    )
    family_summary = (
        metrics.groupby(["group_id", "family_id"], sort=True)
        .agg(
            seed_pass_count=("student_family_gate_pass", "sum"),
            worst_fk_p95_mm=("validation_fk_p95_mm", "max"),
            worst_fk_max_mm=("validation_fk_max_mm", "max"),
            minimum_predicted_margin_deg=(
                "predicted_minimum_joint_margin_deg", "min"
            ),
            worst_joint_abs_p95_deg=("worst_joint_abs_p95_deg", "max"),
            worst_prediction_to_gold_p95_deg=(
                "validation_beta_rms_p95_deg", "max"
            ),
            worst_raw_transition_max_deg=(
                "phase_beta_rms_max_deg", "max"
            ),
        )
        .reset_index()
    )
    family_summary.to_csv(stage / "family_summary.csv", index=False)
    gate = _gate(
        stage / "gate.json",
        {
            "model_and_geometry_lock_pass": bool(protocol["gate_pass"]),
            "independent_core_teacher_gate_pass": bool(teacher["gate_pass"]),
            "independent_core_student_gate_pass": bool(student["gate_pass"]),
            "tracking_visualizations_complete": bool(visual["gate_pass"]),
        },
        semantics="decision_independent_new_geometry_closed_loop_validation",
        independent_new_geometry_evidence_pass=bool(
            protocol["gate_pass"]
            and teacher["gate_pass"]
            and student["gate_pass"]
        ),
        primary_seed=int(student["primary_seed"]),
        primary_core_gate_pass=bool(student["primary_core_gate_pass"]),
        core_seed_pass_count=int(student["core_seed_pass_count"]),
        stress_seed_pass_count=int(student["stress_seed_pass_count"]),
        family_summary_sha256=sha256_file(stage / "family_summary.csv"),
        visualization_manifest=str(
            output_root / "03_visualization/manifest.json"
        ),
    )
    lines = [
        "# BACRA V12.12 独立新几何闭环验证",
        "",
        f"- Preregistered primary seed: `{student['primary_seed']}`",
        f"- Core Teacher hard-cycle Gate: `{teacher['gate_pass']}`",
        (
            "- Frozen Student core seed passes: "
            f"`{student['core_seed_pass_count']}/4`"
        ),
        (
            "- Frozen Student stress seed passes: "
            f"`{student['stress_seed_pass_count']}/4`"
        ),
        f"- Independent new-geometry Gate: `{gate['gate_pass']}`",
        "- historical_sealed_artifacts_read: `false`",
        "- deployment_claim_gate_pass: `false`",
        "",
        v128._markdown_table(family_summary),
        "",
    ]
    (stage / "result_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    marker = {
        "protocol_id": PROTOCOL_ID,
        "gate_pass": bool(gate["gate_pass"]),
        "independent_new_geometry_evidence_pass": bool(
            gate["independent_new_geometry_evidence_pass"]
        ),
        "primary_seed": int(student["primary_seed"]),
        "core_seed_pass_count": int(student["core_seed_pass_count"]),
        "stress_seed_pass_count": int(student["stress_seed_pass_count"]),
        "deployment_claim_gate_pass": False,
        "historical_sealed_artifacts_read": False,
    }
    atomic_write_json(
        output_root
        / str(config["independent_holdout"]["completion_marker"]),
        marker,
    )
    if gate["gate_pass"]:
        atomic_write_json(
            output_root / str(config["independent_holdout"]["pass_marker"]),
            gate,
        )
    return gate


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, args.preset)
    if str(config["protocol_id"]) != PROTOCOL_ID:
        raise ValueError("unexpected V12.12 protocol_id")
    project_root = Path(args.project_root).resolve()
    output_root = (
        Path(args.output).resolve()
        if args.output
        else _source_path(project_root, str(config["output_root"]))
    )
    python = Path(args.python).resolve()
    stage_paths = {
        "protocol": "00_protocol/gate.json",
        "teacher": "01_teacher_reference/gate.json",
        "student": "02_student_evaluation/gate.json",
        "visualize": "03_visualization/gate.json",
        "summary": "04_summary/gate.json",
    }
    stages = {
        "protocol": lambda: stage_protocol(
            config, project_root, output_root, python=python
        ),
        "teacher": lambda: stage_teacher(
            config, project_root, output_root, python=python
        ),
        "student": lambda: stage_student(
            config, project_root, output_root, python=python
        ),
        "visualize": lambda: stage_visualize(config, output_root),
        "summary": lambda: stage_summary(config, output_root),
    }
    requested = tuple(stages) if args.stage == "all" else (str(args.stage),)
    for name in requested:
        gate_path = output_root / stage_paths[name]
        gate = (
            json.loads(gate_path.read_text(encoding="utf-8"))
            if gate_path.is_file()
            else stages[name]()
        )
        if name in {"protocol", "teacher"} and not bool(gate["gate_pass"]):
            break
    stage_gates = {
        name: bool(
            json.loads(
                (output_root / relative).read_text(encoding="utf-8")
            )["gate_pass"]
        )
        for name, relative in stage_paths.items()
        if (output_root / relative).is_file()
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "preset": str(config["preset"]),
        "output_root": str(output_root),
        "stage_gate_pass": stage_gates,
        "deployment_claim_gate_pass": False,
        "historical_sealed_artifacts_read": False,
    }
    atomic_write_json(output_root / "run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            SOURCE_ROOT
            / "configs/bacra_v12_12_independent_gold_set_holdout.yaml"
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(v12.project_root_from(SOURCE_ROOT)),
    )
    parser.add_argument("--output")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument(
        "--preset", choices=("smoke", "formal"), default="formal"
    )
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "protocol",
            "teacher",
            "student",
            "visualize",
            "summary",
        ),
        default="all",
    )
    parser.add_argument(
        "--worker",
        choices=("teacher", "multires_teacher", "predict", "evaluate"),
    )
    parser.add_argument("--catalog")
    parser.add_argument("--family-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reference-dataset")
    parser.add_argument("--teacher-metrics")
    parser.add_argument("--predictions")
    parser.add_argument("--output-root")
    parser.add_argument("--worker-output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker == "teacher":
        return teacher_worker(args)
    if args.worker == "multires_teacher":
        return multiresolution_teacher_worker(args)
    if args.worker == "predict":
        return prediction_worker(args)
    if args.worker == "evaluate":
        return evaluation_worker(args)
    started = time.time()
    report = run(args)
    report["wall_time_s"] = float(time.time() - started)
    atomic_write_json(Path(report["output_root"]) / "run_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
