#!/usr/bin/env python3
"""Evaluate frozen V12.7.5 Students on new full-loop task geometries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

import run_bacra_v12 as v12
import run_bacra_v12_7_canonical_student as canonical_runner
from quasi_exp.teacher.bacra_canonical_student import (
    project_toward_canonical_beta,
)
from quasi_exp.teacher.bacra_geometry_holdout import (
    LoopReferencePolicy,
    evaluate_student_loop,
    family_from_row,
    generate_geometry_holdout,
    solve_loop_reference,
)
from quasi_exp.teacher.canonical import TeacherPolicy
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.region import EllipseFamilySpec
from run_trajectory_canonical_teacher_v10 import (
    load_environment,
    runtime_fingerprint,
)


PROTOCOL_ID = (
    "branch-aware-canonical-region-atlas-v12.8.2-enriched-constrained-dp-"
    "independent-geometry-loop-evaluation"
)
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)
CORE_GROUP = "core_interpolation"
STRESS_GROUP = "historical_range_stress"


def _source_path(project_root: Path, value: str) -> Path:
    return v12._source_path(project_root, str(value))


def _source_roots(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    source = config["source_lock"]
    return {
        "v12_7_5": _source_path(
            project_root, str(source["v12_7_5_root"])
        ),
        "v12_7": _source_path(
            project_root, str(source["v12_7_root"])
        ),
        "v12_5": _source_path(
            project_root, str(source["v12_5_root"])
        ),
    }


def _source_files(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    roots = _source_roots(config, project_root)
    source = config["source_lock"]
    return {
        "admission_gate": roots["v12_7_5"]
        / str(source["admission_gate"]),
        "selected_projection": roots["v12_7_5"]
        / str(source["selected_projection"]),
        "completion_marker": roots["v12_7_5"]
        / str(source["completion_marker"]),
        "canonical_charts": roots["v12_5"]
        / str(source["canonical_charts"]),
        "anchor_task_json": _source_path(
            project_root,
            str(config["geometry_holdout"]["anchor_task_json"]),
        ),
        "robot_config": _source_path(
            project_root, str(config["robot_config"])
        ),
    }


def _model_paths(
    config: Mapping[str, Any], project_root: Path, seed: int
) -> tuple[Path, Path]:
    roots = _source_roots(config, project_root)
    source = config["source_lock"]
    mlp = roots["v12_7"] / str(source["mlp_model_template"]).format(
        seed=int(seed)
    )
    tree = roots["v12_7_5"] / str(
        source["tree_model_template"]
    ).format(seed=int(seed))
    return mlp, tree


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
            "simulation_independent_task_geometry_closed_loop_evaluation"
        ),
        "deployment_claim_gate_pass": False,
        "historical_sealed_artifacts_read": False,
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, payload)
    return payload


def _anchor_family(path: Path) -> EllipseFamilySpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    family = payload["family"]
    return EllipseFamilySpec(
        family_id=str(family["family_id"]),
        center_m=np.asarray(family["center_m"], dtype=float),
        major_direction=np.asarray(
            family["major_direction"], dtype=float
        ),
        minor_direction=np.asarray(
            family["minor_direction"], dtype=float
        ),
        major_semiaxis_m=float(family["major_semiaxis_m"]),
        minor_semiaxis_m=float(family["minor_semiaxis_m"]),
        metadata={"source": str(path)},
    )


def _model_lock(
    config: Mapping[str, Any], project_root: Path
) -> tuple[dict[str, Any], bool]:
    rows: list[dict[str, Any]] = []
    all_exist = True
    for seed in map(int, config["canonical_student"]["formal_seeds"]):
        mlp, tree = _model_paths(config, project_root, seed)
        for kind, path in (("mlp", mlp), ("canonical_tree", tree)):
            exists = path.is_file()
            all_exist = all_exist and exists
            rows.append(
                {
                    "seed": seed,
                    "kind": kind,
                    "path": str(path),
                    "sha256": sha256_file(path) if exists else None,
                }
            )
    return {"artifacts": rows}, all_exist


def stage_protocol(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    stage = output_root / "00_protocol"
    stage.mkdir(parents=True, exist_ok=True)
    sources = _source_files(config, project_root)
    expected = {
        **{
            key: str(value)
            for key, value in config["source_lock"][
                "expected_sha256"
            ].items()
        },
        "anchor_task_json": str(
            config["geometry_holdout"]["expected_anchor_sha256"]
        ),
    }
    actual = {
        key: sha256_file(path) if path.is_file() else None
        for key, path in sources.items()
        if key != "robot_config"
    }
    source_hash_match = all(
        actual.get(name) == digest for name, digest in expected.items()
    )
    admission = (
        json.loads(sources["admission_gate"].read_text(encoding="utf-8"))
        if sources["admission_gate"].is_file()
        else {}
    )
    completion = (
        json.loads(
            sources["completion_marker"].read_text(encoding="utf-8")
        )
        if sources["completion_marker"].is_file()
        else {}
    )
    selected = (
        json.loads(
            sources["selected_projection"].read_text(encoding="utf-8")
        )
        if sources["selected_projection"].is_file()
        else {}
    )
    model_lock, models_exist = _model_lock(config, project_root)
    model_lock.update(
        {
            "selected_alpha": selected.get("alpha"),
            "formal_seeds": list(
                map(int, config["canonical_student"]["formal_seeds"])
            ),
            "projection_formula": (
                "beta_mlp + alpha * (I - pinv(J)J) * "
                "(beta_tree - beta_mlp)"
            ),
            "feature_order": list(XYZ_COLUMNS),
            "training_or_refit_allowed": False,
        }
    )
    atomic_write_json(stage / "model_lock.json", model_lock)

    anchor = _anchor_family(sources["anchor_task_json"])
    catalog = generate_geometry_holdout(
        anchor, config["geometry_holdout"]["groups"]
    )
    catalog_path = stage / "selected_geometry_holdout.parquet"
    v12._atomic_parquet(catalog, catalog_path)
    catalog.to_csv(
        stage / "selected_geometry_holdout.csv", index=False
    )
    atomic_write_json(stage / "frozen_config.json", dict(config))
    atomic_write_json(
        stage / "runtime.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "runtime_fingerprint": runtime_fingerprint(),
            "device_plan": dict(config["device"]),
        },
    )
    implementation = {
        "runner": sha256_file(Path(__file__).resolve()),
        "geometry_module": sha256_file(
            SOURCE_ROOT
            / "src/quasi_exp/teacher/bacra_geometry_holdout.py"
        ),
        "config": sha256_file(Path(str(config["config_path"]))),
    }
    atomic_write_json(
        stage / "source_manifest.json",
        {
            "source_paths": {
                name: str(path) for name, path in sources.items()
            },
            "source_sha256": actual,
            "geometry_catalog_sha256": sha256_file(catalog_path),
            "implementation_sha256": implementation,
            "historical_sealed_artifacts_read": False,
        },
    )
    group_counts = catalog["group_id"].value_counts().to_dict()
    checks = {
        "source_artifacts_exist": all(
            path.is_file() for path in sources.values()
        ),
        "source_hashes_match": source_hash_match,
        "source_exploratory_admission_passed": bool(
            admission.get("gate_pass", False)
        ),
        "source_completion_marker_passed": bool(
            completion.get("gate_pass", True)
        ),
        "projection_alpha_frozen_at_0p25": bool(
            float(selected.get("alpha", -1.0))
            == float(config["source_lock"]["expected_alpha"])
        ),
        "all_ten_model_artifacts_exist": models_exist
        and len(model_lock["artifacts"]) == 10,
        "eight_new_families_frozen": len(catalog) == 8,
        "core_and_stress_inventory_complete": (
            int(group_counts.get(CORE_GROUP, 0)) == 4
            and int(group_counts.get(STRESS_GROUP, 0)) == 4
        ),
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="independent_geometry_and_model_lock",
        source_sha256=actual,
        geometry_catalog_sha256=sha256_file(catalog_path),
        implementation_sha256=implementation,
        family_group_counts=group_counts,
        selected_alpha=selected.get("alpha"),
        device_plan=dict(config["device"]),
    )


def _teacher_policy(config: Mapping[str, Any]) -> TeacherPolicy:
    return TeacherPolicy(
        damping=float(config["candidates"]["damping"]),
        beta_weights=tuple(
            map(float, config["candidates"]["beta_weights"])
        ),
        max_corrector_iterations=int(
            config["candidates"]["max_corrector_iterations"]
        ),
        tracking_tolerance_mm=float(
            config["candidates"]["residual_p95_mm"]
        ),
        safe_joint_margin_deg=float(
            config["candidates"]["gold_margin_deg"]
        ),
        safe_margin_repulsion_step_deg=0.0,
        solver_seed=int(config["seeds"]["dense"]),
        max_step_deg=float(config["candidates"]["max_step_deg"]),
    )


def _loop_policy(config: Mapping[str, Any]) -> LoopReferencePolicy:
    values = config["teacher_reference"]
    return LoopReferencePolicy(
        phase_count=int(config["geometry_holdout"]["phase_count"]),
        max_tetrahedron_edge_mm=float(
            values["max_tetrahedron_edge_mm"]
        ),
        target_support_max_mm=float(values["target_support_max_mm"]),
        continuation_step_mm=float(values["continuation_step_mm"]),
        candidate_nearest_count=int(values["candidate_nearest_count"]),
        candidate_representative_count=int(
            values["candidate_representative_count"]
        ),
        candidate_budget_per_phase=int(
            values["candidate_budget_per_phase"]
        ),
        candidate_cluster_deg=float(values["candidate_cluster_deg"]),
        gap_enrichment_rounds=int(
            values.get("gap_enrichment_rounds", 0)
        ),
        gap_enrichment_sources_per_frontier=int(
            values.get("gap_enrichment_sources_per_frontier", 2)
        ),
        gap_enrichment_round_candidate_budget=int(
            values.get("gap_enrichment_round_candidate_budget", 96)
        ),
        gap_enrichment_max_candidates_per_phase=int(
            values.get(
                "gap_enrichment_max_candidates_per_phase",
                max(64, int(values["candidate_budget_per_phase"])),
            )
        ),
        gap_enrichment_topology_dedup_deg=float(
            values.get("gap_enrichment_topology_dedup_deg", 1.0e-6)
        ),
        residual_p95_mm=float(values["residual_p95_mm"]),
        residual_max_mm=float(values["residual_max_mm"]),
        joint_margin_min_deg=float(values["joint_margin_min_deg"]),
        phase_beta_rms_p95_deg=float(
            values["phase_beta_rms_p95_deg"]
        ),
        phase_beta_rms_max_deg=float(
            values["phase_beta_rms_max_deg"]
        ),
        acceleration_beta_rms_p95_deg=float(
            values["acceleration_beta_rms_p95_deg"]
        ),
        seam_beta_rms_deg=float(values["seam_beta_rms_deg"]),
        direction_gap_p95_deg=float(
            values["direction_gap_p95_deg"]
        ),
        direction_gap_max_deg=float(
            values["direction_gap_max_deg"]
        ),
        teacher_policy=_teacher_policy(config),
    )


def teacher_worker(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    catalog = pd.read_parquet(args.catalog)
    selected = catalog.loc[catalog["family_id"].eq(args.family_id)]
    if len(selected) != 1:
        raise ValueError(f"unknown holdout family {args.family_id}")
    family = family_from_row(selected.iloc[0])
    environment = load_environment(
        project_root,
        _source_files(config, project_root)["robot_config"],
    )
    chart = pd.read_parquet(
        _source_files(config, project_root)["canonical_charts"]
    )
    frame, report = solve_loop_reference(
        environment,
        chart,
        family,
        group_id=str(selected.iloc[0]["group_id"]),
        policy=_loop_policy(config),
    )
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    gold_candidate_artifact = report.pop(
        "_gold_candidate_artifact", None
    )
    v12._atomic_parquet(frame, destination / "reference.parquet")
    if gold_candidate_artifact:
        v12._atomic_parquet(
            pd.DataFrame(gold_candidate_artifact),
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
    trajectory = report.get("trajectory") or {}
    forward_trajectory = report.get("forward_trajectory") or {}
    reverse_trajectory = report.get("reverse_trajectory") or {}
    primary_link = report.get("primary_link_report") or {}
    reverse_link = report.get("reverse_link_report") or {}
    checks = report.get("checks") or {}
    return {
        key: value
        for key, value in report.items()
        if key
        not in {
            "trajectory",
            "forward_trajectory",
            "reverse_trajectory",
            "primary_link_report",
            "reverse_link_report",
            "gap_enrichment",
            "independent_hard_cyclic_certificate",
            "checks",
        }
    } | {
        f"trajectory_{key}": value for key, value in trajectory.items()
    } | {
        f"forward_trajectory_{key}": value
        for key, value in forward_trajectory.items()
    } | {
        f"reverse_trajectory_{key}": value
        for key, value in reverse_trajectory.items()
    } | {
        f"primary_link_{key}": value
        for key, value in primary_link.items()
    } | {
        f"reverse_link_{key}": value
        for key, value in reverse_link.items()
    } | {f"check_{key}": bool(value) for key, value in checks.items()}


def stage_teacher(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / "01_teacher_reference"
    stage.mkdir(parents=True, exist_ok=True)
    catalog_path = (
        output_root / "00_protocol/selected_geometry_holdout.parquet"
    )
    catalog = pd.read_parquet(catalog_path).sort_values(
        "catalog_order", kind="stable"
    )
    commands: list[tuple[str, Sequence[str], Path]] = []
    for row in catalog.itertuples(index=False):
        family_id = str(row.family_id)
        commands.append(
            (
                family_id,
                [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "teacher",
                    "--config",
                    str(config["config_path"]),
                    "--preset",
                    str(config["preset"]),
                    "--project-root",
                    str(project_root),
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
    parallel = v12._run_subprocess_tasks(
        commands,
        requested_workers=int(config["parallel"]["teacher_workers"]),
        manifest_path=stage / "parallel_manifest.json",
    )
    reports: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for row in catalog.itertuples(index=False):
        family_id = str(row.family_id)
        family_dir = stage / "families" / family_id
        report = json.loads(
            (family_dir / "report.json").read_text(encoding="utf-8")
        )
        reports.append(_flatten_teacher_report(report))
        frames.append(pd.read_parquet(family_dir / "reference.parquet"))
    metrics = pd.DataFrame(reports).sort_values(
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
    core_pass = bool(len(core) == 4 and core["gate_pass"].all())
    stress_pass = bool(len(stress) == 4 and stress["gate_pass"].all())
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
            "all_four_core_teacher_references_valid": core_pass,
        },
        semantics="independent_geometry_teacher_reference",
        core_geometry_teacher_gate_pass=core_pass,
        stress_teacher_gate_pass=stress_pass,
        core_pass_count=int(core["gate_pass"].sum()),
        stress_pass_count=int(stress["gate_pass"].sum()),
        parallel_evidence=parallel,
        reference_dataset_sha256=sha256_file(
            stage / "reference_dataset.parquet"
        ),
    )


def student_worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    seed = int(args.seed)
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(
        int(config["parallel"]["student_intraop_threads"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(config["parallel"]["student_interop_threads"])
    )
    reference = pd.read_parquet(args.reference_dataset).sort_values(
        ["group_id", "family_id", "phase_idx"], kind="stable"
    )
    xyz = reference.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    mlp_path, tree_path = _model_paths(config, project_root, seed)
    model = tf.keras.models.load_model(mlp_path, compile=False)
    base = np.asarray(
        model.predict(xyz, batch_size=256, verbose=0), dtype=float
    )
    tree = joblib.load(tree_path)
    if hasattr(tree, "n_jobs"):
        tree.n_jobs = 1
    canonical = np.asarray(tree.predict(xyz), dtype=float)
    selected = json.loads(
        _source_files(config, project_root)[
            "selected_projection"
        ].read_text(encoding="utf-8")
    )
    alpha = float(selected["alpha"])
    geometry = canonical_runner._geometry(config, project_root)
    prediction = project_toward_canonical_beta(
        base,
        canonical,
        geometry=geometry,
        alpha=alpha,
    )
    environment = load_environment(
        project_root,
        _source_files(config, project_root)["robot_config"],
    )
    teacher_metrics = pd.read_parquet(args.teacher_metrics).set_index(
        "family_id"
    )
    reports: list[dict[str, Any]] = []
    diagnostics: list[pd.DataFrame] = []
    for family_id, family_frame in reference.groupby(
        "family_id", sort=True
    ):
        positions = family_frame.index.to_numpy(dtype=np.int64)
        report, detail = evaluate_student_loop(
            environment,
            family_frame.reset_index(drop=True),
            prediction[positions],
            teacher_reference_gate_pass=bool(
                teacher_metrics.loc[family_id, "gate_pass"]
            ),
            admission=config["exploratory_admission"],
            trajectory_limits=config["student_trajectory_gate"],
        )
        report["seed"] = seed
        report["alpha"] = alpha
        reports.append(report)
        detail.insert(0, "family_id", str(family_id))
        detail.insert(1, "group_id", report["group_id"])
        detail.insert(2, "seed", seed)
        detail.insert(3, "alpha", alpha)
        for index, name in enumerate(BETA_COLUMNS):
            detail[f"base_{name}"] = base[positions, index]
            detail[f"tree_{name}"] = canonical[positions, index]
        diagnostics.append(detail)
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        destination / "metrics.json",
        {"seed": seed, "alpha": alpha, "families": reports},
    )
    v12._atomic_parquet(
        pd.concat(diagnostics, ignore_index=True),
        destination / "predictions.parquet",
    )
    print(
        json.dumps(
            {
                "seed": seed,
                "family_count": len(reports),
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


def _flatten_student_report(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("checks") or {}
    row = {
        key: value for key, value in report.items() if key != "checks"
    }
    row.update({f"check_{key}": bool(value) for key, value in checks.items()})
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
    reference = output_root / "01_teacher_reference/reference_dataset.parquet"
    teacher_metrics = (
        output_root
        / "01_teacher_reference/teacher_metrics_per_family.parquet"
    )
    commands: list[tuple[str, Sequence[str], Path]] = []
    seeds = tuple(map(int, config["canonical_student"]["formal_seeds"]))
    for seed in seeds:
        task_id = f"seed_{seed}"
        commands.append(
            (
                task_id,
                [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "student",
                    "--config",
                    str(config["config_path"]),
                    "--preset",
                    str(config["preset"]),
                    "--project-root",
                    str(project_root),
                    "--seed",
                    str(seed),
                    "--reference-dataset",
                    str(reference),
                    "--teacher-metrics",
                    str(teacher_metrics),
                    "--worker-output",
                    str(stage / task_id),
                ],
                stage / "logs" / f"{task_id}.log",
            )
        )
    parallel = v12._run_subprocess_tasks(
        commands,
        requested_workers=int(
            config["parallel"]["student_cpu_workers"]
        ),
        manifest_path=stage / "parallel_manifest.json",
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        payload = json.loads(
            (stage / f"seed_{seed}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        rows.extend(
            _flatten_student_report(report)
            for report in payload["families"]
        )
    metrics = pd.DataFrame(rows).sort_values(
        ["group_id", "family_id", "seed"], kind="stable"
    )
    v12._atomic_parquet(
        metrics, stage / "student_metrics_per_family_seed.parquet"
    )
    seed_summary: list[dict[str, Any]] = []
    for seed, seed_rows in metrics.groupby("seed", sort=True):
        core = seed_rows.loc[seed_rows["group_id"].eq(CORE_GROUP)]
        stress = seed_rows.loc[
            seed_rows["group_id"].eq(STRESS_GROUP)
        ]
        seed_summary.append(
            {
                "seed": int(seed),
                "core_family_pass_count": int(
                    core["student_family_gate_pass"].sum()
                ),
                "core_seed_gate_pass": bool(
                    len(core) == 4
                    and core["student_family_gate_pass"].all()
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
    seeds_frame = pd.DataFrame(seed_summary)
    v12._atomic_parquet(
        seeds_frame, stage / "student_metrics_per_seed.parquet"
    )
    core_seed_pass_count = int(
        seeds_frame["core_seed_gate_pass"].sum()
    )
    stress_seed_pass_count = int(
        seeds_frame["stress_seed_gate_pass"].sum()
    )
    required = int(config["exploratory_admission"]["required_seed_passes"])
    core_pass = bool(
        len(seeds_frame) == 5 and core_seed_pass_count >= required
    )
    stress_pass = bool(
        len(seeds_frame) == 5 and stress_seed_pass_count >= required
    )
    return _gate(
        stage / "gate.json",
        {
            "all_five_frozen_student_tasks_complete": len(seeds_frame)
            == 5,
            "all_required_core_seed_family_gates_pass": core_pass,
        },
        semantics="frozen_student_independent_geometry_loops",
        core_geometry_student_gate_pass=core_pass,
        stress_student_gate_pass=stress_pass,
        core_seed_pass_count=core_seed_pass_count,
        stress_seed_pass_count=stress_seed_pass_count,
        required_seed_passes=required,
        device_plan=dict(config["device"]),
        parallel_evidence=parallel,
    )


def _plot_summary(metrics: pd.DataFrame, destination: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quasi-exp-mpl")
    import matplotlib.pyplot as plt

    families = list(
        metrics.sort_values(["group_id", "family_id"], kind="stable")[
            "family_id"
        ].drop_duplicates()
    )
    seeds = sorted(map(int, metrics["seed"].unique()))
    fk = (
        metrics.pivot(
            index="family_id",
            columns="seed",
            values="validation_fk_p95_mm",
        )
        .reindex(index=families, columns=seeds)
        .to_numpy(dtype=float)
    )
    beta = (
        metrics.pivot(
            index="family_id",
            columns="seed",
            values="worst_joint_abs_p95_deg",
        )
        .reindex(index=families, columns=seeds)
        .to_numpy(dtype=float)
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, values, title, label in (
        (axes[0], fk, "FK P95 by family and seed", "mm"),
        (axes[1], beta, "Worst-joint absolute P95", "deg"),
    ):
        image = axis.imshow(values, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(seeds)), labels=seeds, rotation=45)
        axis.set_yticks(range(len(families)), labels=families)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label=label)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' tabulate extra."""

    values = frame.copy()
    headers = [str(column) for column in values.columns]
    rows = [
        [
            "" if pd.isna(value) else str(value).replace("|", "\\|")
            for value in row
        ]
        for row in values.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def stage_summary(
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    stage = output_root / "03_summary"
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
    teacher_metrics = pd.read_parquet(
        output_root
        / "01_teacher_reference/teacher_metrics_per_family.parquet"
    )
    student_metrics = pd.read_parquet(
        output_root
        / "02_student_evaluation/student_metrics_per_family_seed.parquet"
    )
    family_outcomes = teacher_metrics[
        [
            "family_id",
            "group_id",
            "gate_pass",
            "supported_phase_count",
            "minimum_joint_margin_deg",
            "residual_p95_mm",
            "residual_max_mm",
        ]
    ].rename(columns={"gate_pass": "teacher_gate_pass"})
    student_family = (
        student_metrics.groupby(
            ["family_id", "group_id"], sort=True
        )
        .agg(
            student_seed_pass_count=("student_family_gate_pass", "sum"),
            worst_fk_p95_mm=("validation_fk_p95_mm", "max"),
            worst_fk_max_mm=("validation_fk_max_mm", "max"),
            minimum_predicted_margin_deg=(
                "predicted_minimum_joint_margin_deg",
                "min",
            ),
            worst_joint_abs_p95_deg=(
                "worst_joint_abs_p95_deg",
                "max",
            ),
        )
        .reset_index()
    )
    outcomes = family_outcomes.merge(
        student_family,
        on=["family_id", "group_id"],
        how="left",
        validate="one_to_one",
    )
    outcomes.to_csv(stage / "family_outcome_matrix.csv", index=False)
    _plot_summary(
        student_metrics, stage / "closed_loop_metrics_heatmap.png"
    )
    core_teacher = bool(
        teacher.get("core_geometry_teacher_gate_pass", False)
    )
    core_student = bool(
        student.get("core_geometry_student_gate_pass", False)
    )
    stress_teacher = bool(teacher.get("stress_teacher_gate_pass", False))
    stress_student = bool(student.get("stress_student_gate_pass", False))
    gate = _gate(
        stage / "gate.json",
        {
            "protocol_and_model_lock_pass": bool(protocol["gate_pass"]),
            "core_geometry_teacher_gate_pass": core_teacher,
            "core_geometry_student_gate_pass": core_student,
        },
        semantics="independent_core_geometry_closed_loop_admission",
        independent_core_geometry_evidence_pass=bool(
            protocol["gate_pass"] and core_teacher and core_student
        ),
        stress_teacher_gate_pass=stress_teacher,
        stress_student_gate_pass=stress_student,
        next_experiment=(
            "new_geometry_tube_surface_holdout"
            if core_teacher and core_student
            else "classify_teacher_support_vs_student_generalization_failure"
        ),
    )
    lines = [
        "# BACRA V12.8 独立 task-geometry 全闭环结果",
        "",
        f"- Core Teacher Gate: `{core_teacher}`",
        f"- Core Student Gate: `{core_student}`",
        f"- Stress Teacher Gate: `{stress_teacher}`",
        f"- Stress Student Gate: `{stress_student}`",
        f"- Top-level Gate: `{gate['gate_pass']}`",
        "- deployment_claim_gate_pass: `false`",
        "- historical_sealed_artifacts_read: `false`",
        "",
        _markdown_table(outcomes),
        "",
    ]
    (stage / "result_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    atomic_write_json(
        output_root / str(config["completion_marker"]),
        {
            "protocol_id": PROTOCOL_ID,
            "gate_pass": bool(gate["gate_pass"]),
            "core_geometry_teacher_gate_pass": core_teacher,
            "core_geometry_student_gate_pass": core_student,
            "stress_teacher_gate_pass": stress_teacher,
            "stress_student_gate_pass": stress_student,
            "deployment_claim_gate_pass": False,
            "historical_sealed_artifacts_read": False,
        },
    )
    if gate["gate_pass"]:
        atomic_write_json(
            output_root / str(config["pass_marker"]), gate
        )
    return gate


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, args.preset)
    if str(config["protocol_id"]) != PROTOCOL_ID:
        raise ValueError("unexpected V12.8 protocol_id")
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
        "summary": "03_summary/gate.json",
    }
    stages = {
        "protocol": lambda: stage_protocol(
            config, project_root, output_root
        ),
        "teacher": lambda: stage_teacher(
            config,
            project_root,
            output_root,
            python=python,
        ),
        "student": lambda: stage_student(
            config,
            project_root,
            output_root,
            python=python,
        ),
        "summary": lambda: stage_summary(config, output_root),
    }
    requested = (
        tuple(stages) if args.stage == "all" else (str(args.stage),)
    )
    for name in requested:
        gate_path = output_root / stage_paths[name]
        if gate_path.is_file():
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        else:
            gate = stages[name]()
        if name == "protocol" and not bool(gate["gate_pass"]):
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
            / "configs/bacra_v12_8_independent_geometry_loops.yaml"
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(v12.project_root_from(SOURCE_ROOT)),
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--python", default=str(DEFAULT_PYTHON)
    )
    parser.add_argument(
        "--preset", choices=("smoke", "formal"), default="formal"
    )
    parser.add_argument(
        "--stage",
        choices=("all", "protocol", "teacher", "student", "summary"),
        default="all",
    )
    parser.add_argument(
        "--worker", choices=("teacher", "student")
    )
    parser.add_argument("--catalog")
    parser.add_argument("--family-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reference-dataset")
    parser.add_argument("--teacher-metrics")
    parser.add_argument("--worker-output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker == "teacher":
        return teacher_worker(args)
    if args.worker == "student":
        return student_worker(args)
    started = time.time()
    report = run(args)
    report["wall_time_s"] = float(time.time() - started)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
