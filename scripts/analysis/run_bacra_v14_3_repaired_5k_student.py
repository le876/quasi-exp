#!/usr/bin/env python3
"""Run BACRA V14.3 repaired 5k Pilot, exploratory Students, and Formal Gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.stats import qmc

from quasi_exp.teacher.canonical import beta_rms_deg, weighted_damped_pinv
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    deterministic_root_nodes,
)
from quasi_exp.teacher.optimized_continuation import (
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.section_atlas_repair import (
    AtlasRepairPolicy,
    AuditV2Policy,
    RetryTier,
    execute_audit_schedules,
    section_growth_from_frames,
    repair_rooted_section_atlas,
)
from quasi_exp.teacher.section_first_atlas import RootedSectionPolicy, build_section_first_atlas
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_atlas_repair import (
    atlas_nodes_from_frames,
    make_segmented_continuation,
)
from quasi_exp.teacher.workspace_inverse import InverseQuery
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    save_workspace_student_models,
    train_workspace_student,
)
from quasi_exp.teacher.workspace_trajectories import build_workspace_trajectory_suite
from quasi_exp.teacher.workspace_experiment import (
    ReachSamplingRoundSpec,
    generate_independent_reach_samples,
)
from quasi_exp.teacher.workspace_reach import (
    CellKey,
    FrontierProbeEvidence,
    ReachProxyBuilder,
    ReachProxyConfig,
    ReachReplica,
    ReachReplicaRound,
    measure_weighted_boundary_change_ratio,
)

import run_bacra_v14_2_section_first_atlas as legacy
import run_bacra_v14_2r_stitched_atlas as v142r
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("x_m", "y_m", "z_m")
ROUTER_FEATURE_COLUMNS = XYZ_COLUMNS


def _environment(project_root: Path, config: Mapping[str, Any]):
    return optimized_forward(
        load_environment(project_root, _sources(config, project_root)["robot_config"])
    )


def _segmented_continuation(
    environment: Any, tasks: pd.DataFrame, task_edges: pd.DataFrame
):
    nodes = atlas_nodes_from_frames(tasks, task_edges)
    base = make_optimized_predictor_corrector_continuation(
        environment, residual_tolerance_mm=3.0
    )
    cell_by_node = {
        int(row.task_node_id): (
            int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)
        )
        for row in tasks.itertuples(index=False)
    }
    return nodes, make_segmented_continuation(
        base,
        {node.node_id: node for node in nodes},
        cell_by_node,
        step_max_mm=5.0,
    )
STAGE_DIRS = {
    "inventory": "00_inventory",
    "pilot_registry": "01_pilot_registry",
    "root_charts": "02_root_charts",
    "stitched_atlas": "03_stitched_atlas",
    "fixed_budget_dataset": "04_fixed_budget_dataset",
    "students": "05_students",
    "representation_decision": "06_representation_decision",
    "reach_update": "07_reach_update",
    "formal_admission": "08_formal_admission",
    "summary": "09_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def project_root_from(source_root: Path) -> Path:
    return v142r.project_root_from(source_root)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported V14.3 config")
    config = dict(value)
    config["config_path"] = str(config_path)
    if config["parallel"] != {"patch_workers": 12, "numerical_threads_per_worker": 1}:
        raise ValueError("V14.3 requires exactly twelve single-threaded root workers")
    audit = config.get("audit_execution", {})
    if int(audit.get("logical_shard_count", 0)) < 12:
        raise ValueError("V14.3 requires at least twelve logical audit shards")
    if int(audit.get("maximum_concurrent_workers", 0)) != 12:
        raise ValueError("V14.3 audit concurrency must remain exactly twelve workers")
    if str(audit.get("assignment_strategy", "")) != "cost_balanced_lpt":
        raise ValueError("V14.3 requires deterministic cost-balanced LPT sharding")
    if not bool(audit.get("screening_first", False)):
        raise ValueError("V14.3 requires the registered screening-first protocol")
    pilot = config["pilot"]
    if (int(pilot["parent_cell_count"]), int(pilot["task_probe_count"]), int(pilot["root_count"])) != (5000, 25000, 32):
        raise ValueError("V14.3 registers 5000 parent cells, 25000 probes, and 32 roots")
    dataset = config["dataset"]
    if bool(dataset["row_padding"]):
        raise ValueError("V14.3 forbids supervision row padding")
    if not (20000 <= int(dataset["minimum_unique_rows"]) <= int(dataset["maximum_unique_rows"]) <= 50000):
        raise ValueError("V14.3 unique supervision budget must be within 20k-50k")
    if tuple(ROUTER_FEATURE_COLUMNS) != ("x_m", "y_m", "z_m"):
        raise AssertionError("router inference contract must remain xyz-only")
    if (
        int(config["student"]["trajectory_count"]),
        int(config["student"]["trajectory_waypoints"]),
    ) != (36, 100):
        raise ValueError("V14.3 fixed trajectory protocol requires 36 families x 100 waypoints")
    if int(dataset["multiparent_count"]) < 2:
        raise ValueError("new canonical labels require at least two distinct local parents")
    if float(config["formal_gate"]["labelable_measure_min"]) != 0.80:
        raise ValueError("V14.3 Formal measure semantics are frozen at 80% labelable")
    return config


def _sources(config: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    row = config["sources"]
    return {
        "plan": SOURCE_ROOT / str(row["reviewed_plan"]),
        "v14_2r": project_root / str(row["v14_2r_root"]),
        "v14_2r_config": SOURCE_ROOT / str(row["v14_2r_config"]),
        "legacy_config": SOURCE_ROOT / str(row["v14_2_legacy_config"]),
        "original": project_root / str(row["original_pilot_root"]),
        "retry4": project_root / str(row["retry4_root"]),
        "historical_final8_teacher": project_root / str(row["historical_final8_teacher"]),
        "historical_final8_catalog": project_root / str(row["historical_final8_catalog"]),
        "robot_config": SOURCE_ROOT / str(config["robot_config"]),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    v142r._write_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    return v142r._read_json(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    v142r._write_parquet(frame, path)


def _gate(path: Path, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    return v142r._gate(path, checks, **evidence)


def _require(
    output_root: Path, stage_name: str, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    directory = output_root / STAGE_DIRS[stage_name]
    if config is not None:
        result = _load_validated_stage_result(
            output_root, config=config, stage_name=stage_name
        )
        if result is None:
            raise FileNotFoundError(
                f"required V14.3 stage lacks a valid completion closure: {directory}"
            )
        return result
    path = directory / "gate.json"
    if not path.is_file():
        raise FileNotFoundError(f"required V14.3 stage is not sealed: {path}")
    return _read_json(path)


def _upstream_completion_sha256(output_root: Path, stage_name: str) -> str:
    """Bind a stage completion to every earlier V14.3 completion manifest."""

    stage_index = STAGE_ORDER.index(stage_name)
    digest = hashlib.sha256()
    for upstream_name in STAGE_ORDER[:stage_index]:
        path = output_root / STAGE_DIRS[upstream_name] / "completion_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing upstream completion manifest for {stage_name}: {path}"
            )
        digest.update(upstream_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_stage_completion_manifest(
    output_root: Path, *, config: Mapping[str, Any], stage_name: str
) -> dict[str, Any]:
    directory = output_root / STAGE_DIRS[stage_name]
    manifest = v142r._write_stage_completion_manifest(
        directory, config=config, stage_name=stage_name
    )
    manifest["upstream_completion_sha256"] = _upstream_completion_sha256(
        output_root, stage_name
    )
    _write_json(directory / "completion_manifest.json", manifest)
    return manifest


def _load_validated_stage_result(
    output_root: Path, *, config: Mapping[str, Any], stage_name: str
) -> dict[str, Any] | None:
    directory = output_root / STAGE_DIRS[stage_name]
    result = v142r._load_validated_stage_result(
        directory, config=config, stage_name=stage_name
    )
    if result is None:
        return None
    try:
        manifest = _read_json(directory / "completion_manifest.json")
        expected = _upstream_completion_sha256(output_root, stage_name)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if str(manifest.get("upstream_completion_sha256", "")) != expected:
        return None
    return result


def _audit_policy_from_manifest(manifest: Mapping[str, Any]) -> AuditV2Policy:
    row = manifest["audit_policy"]
    return AuditV2Policy(
        geometry_p95_max_deg=float(row["geometry_p95_max_deg"]),
        geometry_max_deg=float(row["geometry_max_deg"]),
        repeat_p95_max_deg=float(row["repeat_p95_max_deg"]),
        continuation_residual_max_mm=float(row["continuation_residual_max_mm"]),
        repeats_per_direction=int(row["repeats_per_direction"]),
        repeat_perturbation_rad=float(row["repeat_perturbation_rad"]),
        retry_tiers=tuple(
            RetryTier(
                str(tier["tier_id"]),
                float(tier["maximum_step_mm"]),
                int(tier["maximum_iterations"]),
                tuple(map(str, tier["solver_chain"])),
            )
            for tier in row["retry_tiers"]
        ),
    )


def _run_v143_audit_shard_worker(
    config: Mapping[str, Any], project_root: Path, phase_directory: Path, shard_id: int
) -> dict[str, Any]:
    """Execute one immutable V14.3 audit micro-shard."""

    started = time.time()
    cpu_started = time.process_time()
    manifest = _read_json(phase_directory / "input_manifest.json")
    v142r._validate_audit_bundle(manifest, phase_directory, config)
    registry = pd.read_parquet(phase_directory / "schedule_registry.parquet")
    shard_registry = registry[registry["shard_id"].astype(int).eq(int(shard_id))].copy()
    if not 0 <= int(shard_id) < int(manifest["shard_count"]):
        raise ValueError(f"audit shard id outside registered range: {shard_id}")
    tasks = pd.read_parquet(phase_directory / "task_nodes.parquet")
    task_edges = pd.read_parquet(phase_directory / "task_edges.parquet")
    hypotheses = pd.read_parquet(phase_directory / "section_hypotheses.parquet")
    selected_edges = pd.read_parquet(phase_directory / "selected_edges.parquet")
    nodes = legacy.atlas_nodes_from_frames(tasks, task_edges)
    growth = section_growth_from_frames(nodes, hypotheses, selected_edges)
    environment = _environment(project_root, config)
    _nodes, continuation = _segmented_continuation(environment, tasks, task_edges)
    retry = v142r._registered_retry_adapter(environment, tasks, task_edges)
    policy = _audit_policy_from_manifest(manifest)
    shard_directory = phase_directory / f"shard_{int(shard_id):02d}"
    shard_directory.mkdir(parents=True, exist_ok=True)
    progress_path = shard_directory / "progress.json"
    progress_every = int(config["audit_execution"]["progress_every_schedules"])
    last_reported = -progress_every

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_reported
        if completed != total and completed - last_reported < progress_every:
            return
        _write_json(
            progress_path,
            {
                "schema_version": 1,
                "status": "running",
                "phase_id": str(manifest["phase_id"]),
                "shard_id": int(shard_id),
                "completed_schedule_count": int(completed),
                "total_schedule_count": int(total),
                "started_at_unix_s": started,
                "updated_at_unix_s": time.time(),
            },
        )
        last_reported = int(completed)

    report_progress(0, len(shard_registry))
    executions = (
        execute_audit_schedules(
            growth,
            shard_registry,
            continuation,
            policy,
            retry_continuation=retry,
            progress_callback=report_progress,
        )
        if len(shard_registry)
        else pd.DataFrame(columns=v142r.AUDIT_EXECUTION_COLUMNS)
    )
    execution_path = shard_directory / "executions.parquet"
    _write_parquet(executions, execution_path)
    report = v142r.build_shard_completion_report(
        shard_registry,
        executions,
        source_sha=v142r._git_sha(),
        config_sha256=sha256_file(Path(str(config["config_path"]))),
        input_sha256=str(manifest["input_sha256"]),
        phase_id=str(manifest["phase_id"]),
        shard_id=int(shard_id),
        repeats_per_direction=policy.repeats_per_direction,
    )
    report.update(
        {
            "pid": os.getpid(),
            "executions_file_sha256": sha256_file(execution_path),
            "started_at_unix_s": started,
            "finished_at_unix_s": time.time(),
            "wall_time_s": time.time() - started,
            "cpu_time_s": time.process_time() - cpu_started,
        }
    )
    _write_json(shard_directory / "report.json", report)
    _write_json(
        progress_path,
        {
            "schema_version": 1,
            "status": "complete",
            "phase_id": str(manifest["phase_id"]),
            "shard_id": int(shard_id),
            "completed_schedule_count": len(shard_registry),
            "total_schedule_count": len(shard_registry),
            "execution_count": len(executions),
            "started_at_unix_s": started,
            "finished_at_unix_s": time.time(),
            "wall_time_s": time.time() - started,
            "cpu_time_s": time.process_time() - cpu_started,
            "updated_at_unix_s": time.time(),
        },
    )
    return report


def stage_inventory(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["inventory"]
    sources = _sources(config, project_root)
    confirmation = sources["v14_2r"] / "09_twelve_patch_confirmation/gate.json"
    reach = sources["v14_2r"] / "08_reach_round7/gate.json"
    meso = sources["v14_2r"] / "10_meso_bridge/gate.json"
    upstream_manifest = sources["v14_2r"] / "11_summary/artifact_manifest.json"
    required = [
        sources["plan"], sources["v14_2r_config"], sources["legacy_config"],
        sources["historical_final8_teacher"], sources["historical_final8_catalog"],
        confirmation, reach, meso, upstream_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V14.3 source inventory incomplete: {missing}")
    confirmation_gate = _read_json(confirmation)
    meso_gate = _read_json(meso)
    upstream_closure = v142r._verify_upstream_artifact_manifest(upstream_manifest)
    records = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in required]
    _write_parquet(pd.DataFrame.from_records(records), stage / "source_inventory.parquet")
    report = {
        "source_sha": v142r._git_sha(),
        "working_tree_clean": v142r._tree_clean(),
        "runtime": runtime_fingerprint(),
        "v14_2r_confirmation_gate": bool(confirmation_gate.get("gate_pass", False)),
        "v14_2r_meso_bridge_gate": bool(meso_gate.get("gate_pass", False)),
        "v14_2r_upstream_closure": upstream_closure,
        "source_records": records,
        "patch_workers": 12,
    }
    _write_json(stage / "source_fixed_point.json", report)
    return _gate(stage / "gate.json", {
        "clean_fixed_point": report["working_tree_clean"],
        "v14_2r_confirmation": report["v14_2r_confirmation_gate"],
        "v14_2r_meso_bridge": report["v14_2r_meso_bridge_gate"],
        "v14_2r_recursive_artifact_closure": bool(
            upstream_closure.get("artifact_count", 0)
        ),
        "source_inventory_complete": not missing,
    }, **report)


def _legacy_pilot_inputs(config: Mapping[str, Any], project_root: Path):
    legacy_config = legacy.load_config(_sources(config, project_root)["legacy_config"])
    return legacy._pilot_inputs(legacy_config, project_root)


def stage_pilot_registry(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    inventory = _require(output_root, "inventory", config)
    stage = output_root / STAGE_DIRS["pilot_registry"]
    if not inventory.get("gate_pass", False):
        return v142r.write_scientific_skip(stage, "v14_2r_confirmation_not_admitted")
    tasks, edges, candidates, parents = _legacy_pilot_inputs(config, project_root)
    pilot = config["pilot"]
    if len(parents) != int(pilot["parent_cell_count"]) or len(tasks) != int(pilot["task_probe_count"]):
        raise RuntimeError(
            f"frozen lower proxy registry drift: parents={len(parents)}, probes={len(tasks)}"
        )
    representatives = tasks[tasks["is_representative"].astype(bool)]
    root_nodes = tuple(
        AtlasTaskNode(int(row.task_node_id), np.asarray([row.x_m, row.y_m, row.z_m]), ())
        for row in representatives.itertuples(index=False)
    )
    selected_nodes = deterministic_root_nodes(root_nodes, count=int(pilot["root_count"]))
    candidate_objects = legacy._candidates_from_frame(candidates)
    by_node: dict[int, list[Any]] = {}
    for candidate in candidate_objects:
        by_node.setdefault(candidate.node_id, []).append(candidate)
    registry_rows = []
    for root_index, node_id in enumerate(selected_nodes):
        ranked = sorted(
            by_node.get(node_id, ()),
            key=lambda item: (0 if item.is_gold else 1, item.posture_cost, -item.min_margin_deg, item.candidate_id),
        )
        if not ranked:
            raise RuntimeError(f"registered pilot root {node_id} lacks a feasible candidate")
        registry_rows.append({"root_index": root_index, "task_node_id": node_id, "candidate_id": ranked[0].candidate_id})
    registry = pd.DataFrame.from_records(registry_rows)
    _write_parquet(tasks, stage / "pilot_task_nodes.parquet")
    _write_parquet(edges, stage / "pilot_task_edges.parquet")
    _write_parquet(candidates, stage / "pilot_candidate_clusters.parquet")
    _write_parquet(parents, stage / "pilot_parent_cells.parquet")
    _write_parquet(registry, stage / "root_registry.parquet")
    return _gate(stage / "gate.json", {
        "exact_parent_cells": len(parents) == 5000,
        "exact_task_probes": len(tasks) == 25000,
        "maximin_roots": len(registry) == 32,
    }, parent_cell_count=len(parents), task_probe_count=len(tasks), root_count=len(registry))


def _root_policy(config: Mapping[str, Any]) -> RootedSectionPolicy:
    pilot = config["pilot"]
    return RootedSectionPolicy(
        beam_width=int(pilot["beam_width"]),
        root_count=1,
        maximum_growth_waves=int(pilot["maximum_growth_waves"]),
        parent_consensus_gold_deg=0.5,
        parent_consensus_silver_deg=1.0,
        continuation_residual_max_mm=3.0,
        reverse_return_max_deg=0.5,
        minimum_alternative_chart_cells=max(
            int(pilot["minimum_chart_cells"]),
            int(math.ceil(float(pilot["minimum_chart_fraction"]) * int(pilot["task_probe_count"]))),
        ),
        beta_weights=(4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
    )


def _execute_root(config: Mapping[str, Any], project_root: Path, output_root: Path, root_index: int) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["root_charts"]
    directory = stage / f"root_{root_index:03d}"
    registry_stage = output_root / STAGE_DIRS["pilot_registry"]
    input_sha = v142r._payload_sha256(
        {
            "root_index": int(root_index),
            "registry": sha256_file(registry_stage / "root_registry.parquet"),
            "tasks": sha256_file(registry_stage / "pilot_task_nodes.parquet"),
            "edges": sha256_file(registry_stage / "pilot_task_edges.parquet"),
            "candidates": sha256_file(registry_stage / "pilot_candidate_clusters.parquet"),
        }
    )
    existing = v142r._load_validated_patch_result(
        directory,
        config=config,
        patch_id=f"root_{root_index:03d}",
        variant="root_growth",
        input_sha256=input_sha,
    )
    if existing is not None:
        return existing
    registry = pd.read_parquet(registry_stage / "root_registry.parquet")
    root = registry[registry["root_index"].astype(int).eq(root_index)].iloc[0]
    tasks = pd.read_parquet(registry_stage / "pilot_task_nodes.parquet")
    edges = pd.read_parquet(registry_stage / "pilot_task_edges.parquet")
    candidate_frame = pd.read_parquet(registry_stage / "pilot_candidate_clusters.parquet")
    candidates = legacy._candidates_from_frame(candidate_frame)
    environment = _environment(project_root, config)
    nodes, continuation = _segmented_continuation(environment, tasks, edges)
    started = time.time()
    growth = build_section_first_atlas(
        nodes,
        candidates,
        continuation,
        root_keys=((int(root.task_node_id), str(root.candidate_id)),),
        policy=_root_policy(config),
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in growth.frames().items():
        frame = frame.copy()
        if "chart_id" in frame:
            frame["chart_id"] = f"root_{root_index:03d}"
        _write_parquet(frame, directory / f"{name}.parquet")
    report = {
        "root_index": root_index,
        "root_task_node_id": int(root.task_node_id),
        "root_candidate_id": str(root.candidate_id),
        "covered_node_count": len(growth.covered_node_ids),
        "coverage_ratio": len(growth.covered_node_ids) / max(1, len(growth.task_nodes)),
        "raw_cap_hit": growth.raw_cap_hit,
        "alternative_hypothesis_ratio": float(
            (growth.frames()["section_hypotheses"].groupby(["chart_id", "task_node_id"]).size() > 1).mean()
        ),
        "runtime_s": time.time() - started,
        "gate_pass": bool(len(growth.covered_node_ids) >= int(config["pilot"]["minimum_chart_cells"])),
    }
    _write_json(directory / "report.json", report)
    v142r._write_patch_completion_manifest(
        directory,
        config=config,
        patch_id=f"root_{root_index:03d}",
        variant="root_growth",
        input_sha256=input_sha,
    )
    return report


def _run_root_jobs(config: Mapping[str, Any], project_root: Path, output_root: Path) -> None:
    pending = list(range(int(config["pilot"]["root_count"])))
    running: list[tuple[int, subprocess.Popen[str], Any]] = []
    failures = []
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/mpl-bacra-v14-3",
    })
    while pending or running:
        survivors = []
        for root_index, process, handle in running:
            status = process.poll()
            if status is None:
                survivors.append((root_index, process, handle))
            else:
                handle.close()
                if status != 0:
                    failures.append({"root_index": root_index, "returncode": status})
        running = survivors
        if failures:
            for _root, process, handle in running:
                process.terminate(); process.wait(timeout=30); handle.close()
            raise RuntimeError(f"V14.3 root worker failures: {failures}")
        while pending and len(running) < 12:
            root_index = pending.pop(0)
            directory = output_root / STAGE_DIRS["root_charts"] / f"root_{root_index:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / "worker.log").open("w", encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config["config_path"]), "--output-root", str(output_root), "--stage", "root_charts", "--root-index", str(root_index)]
            process = subprocess.Popen(command, cwd=SOURCE_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
            running.append((root_index, process, handle))
        if pending or running:
            time.sleep(1.0)


def stage_root_charts(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    registry = _require(output_root, "pilot_registry", config)
    stage = output_root / STAGE_DIRS["root_charts"]
    if not registry.get("gate_pass", False):
        return v142r.write_scientific_skip(stage, "pilot_registry_gate_failed")
    if config.get("_root_index") is not None:
        report = _execute_root(config, project_root, output_root, int(config["_root_index"]))
        return {"gate_pass": True, "worker_report": report}
    _run_root_jobs(config, project_root, output_root)
    reports = [_read_json(stage / f"root_{index:03d}/report.json") for index in range(32)]
    _write_parquet(pd.DataFrame.from_records(reports), stage / "root_chart_reports.parquet")
    return _gate(stage / "gate.json", {
        "all_roots_completed": len(reports) == 32,
        "at_least_one_qualifiable_root": any(bool(row["gate_pass"]) for row in reports),
    }, root_reports=reports, qualifiable_root_count=sum(bool(row["gate_pass"]) for row in reports),
       bad_roots_are_removed_by_chart_qualification=True)


def _combined_growth(config: Mapping[str, Any], output_root: Path):
    registry_stage = output_root / STAGE_DIRS["pilot_registry"]
    tasks = pd.read_parquet(registry_stage / "pilot_task_nodes.parquet")
    edges = pd.read_parquet(registry_stage / "pilot_task_edges.parquet")
    nodes = legacy.atlas_nodes_from_frames(tasks, edges)
    hypothesis_parts, edge_parts = [], []
    for index in range(32):
        directory = output_root / STAGE_DIRS["root_charts"] / f"root_{index:03d}"
        hypothesis_parts.append(pd.read_parquet(directory / "section_hypotheses.parquet"))
        edge_parts.append(pd.read_parquet(directory / "selected_edges.parquet"))
    hypotheses = pd.concat(hypothesis_parts, ignore_index=True)
    selected_edges = pd.concat(edge_parts, ignore_index=True)
    return section_growth_from_frames(nodes, hypotheses, selected_edges, policy=_root_policy(config)), tasks, edges, hypotheses


def _pilot_root_removal_stability(growth: Any, repaired: Any) -> tuple[bool, pd.DataFrame]:
    chart_by_id = {chart.chart_id: chart for chart in growth.charts}
    primary_nodes = set(repaired.primary_beta_by_node)
    rows = []
    for removed in repaired.selected_stitch_component:
        alternatives = [
            chart_id for chart_id in repaired.qualified_chart_ids
            if chart_id != removed
        ]
        best = None
        for chart_id in alternatives:
            chart = chart_by_id[chart_id]
            common = sorted(primary_nodes & set(chart.selected_by_node))
            coverage_jaccard = len(common) / max(1, len(primary_nodes | set(chart.selected_by_node)))
            gaps = np.asarray([
                beta_rms_deg(
                    repaired.primary_beta_by_node[node],
                    chart.selected_by_node[node].candidate.beta_rad,
                )
                for node in common
            ])
            row = {
                "removed_chart_id": removed,
                "alternative_chart_id": chart_id,
                "coverage_jaccard": coverage_jaccard,
                "beta_p95_deg": float(np.percentile(gaps, 95)) if len(gaps) else math.inf,
                "beta_max_deg": float(np.max(gaps)) if len(gaps) else math.inf,
            }
            row["gate_pass"] = bool(
                row["coverage_jaccard"] >= 0.95
                and row["beta_p95_deg"] <= 1.0
                and row["beta_max_deg"] <= 2.0
            )
            if best is None or (
                -row["coverage_jaccard"], row["beta_p95_deg"], row["beta_max_deg"], chart_id
            ) < (
                -best["coverage_jaccard"], best["beta_p95_deg"], best["beta_max_deg"], best["alternative_chart_id"]
            ):
                best = row
        if best is None:
            best = {
                "removed_chart_id": removed,
                "alternative_chart_id": None,
                "coverage_jaccard": 0.0,
                "beta_p95_deg": math.inf,
                "beta_max_deg": math.inf,
                "gate_pass": False,
            }
        rows.append(best)
    frame = pd.DataFrame.from_records(rows)
    # A single uniquely certified root is not itself proof of instability;
    # root-removal is required only when another qualified large chart exists.
    eligible_alternative = bool(
        len(repaired.qualified_chart_ids) > len(repaired.selected_stitch_component)
    )
    gate = bool(not eligible_alternative or (len(frame) and frame["gate_pass"].all()))
    return gate, frame


def _add_local_refinement_edges(
    tasks: pd.DataFrame,
    task_edges: pd.DataFrame,
    node_ids: Sequence[int],
    *,
    neighbor_count: int,
) -> pd.DataFrame:
    """Add deterministic local KNN edges without deleting any task edge."""

    if not node_ids:
        return task_edges.copy()
    xyz = tasks.set_index("task_node_id").loc[:, XYZ_COLUMNS]
    existing = {
        tuple(sorted((int(row.left_node_id), int(row.right_node_id))))
        for row in task_edges.itertuples(index=False)
    }
    additions: list[dict[str, Any]] = []
    for node_id in sorted(set(map(int, node_ids))):
        if node_id not in xyz.index:
            continue
        distances = np.linalg.norm(
            xyz.to_numpy(float) - xyz.loc[node_id].to_numpy(float), axis=1
        )
        candidates = sorted(
            (
                (float(distance), int(other))
                for distance, other in zip(distances, xyz.index, strict=True)
                if int(other) != node_id
            )
        )
        for _distance, other in candidates[: int(neighbor_count)]:
            edge = tuple(sorted((node_id, other)))
            if edge in existing:
                continue
            existing.add(edge)
            additions.append(
                {
                    "left_node_id": edge[0],
                    "right_node_id": edge[1],
                    "adjacency": "registered_local_refinement",
                }
            )
    if not additions:
        return task_edges.copy()
    columns = list(task_edges.columns)
    added = pd.DataFrame.from_records(additions)
    for column in columns:
        if column not in added:
            added[column] = None
    return (
        pd.concat([task_edges, added.loc[:, columns]], ignore_index=True)
        .sort_values(["left_node_id", "right_node_id"], kind="stable")
        .reset_index(drop=True)
    )


def stage_stitched_atlas(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    roots = _require(output_root, "root_charts", config)
    stage = output_root / STAGE_DIRS["stitched_atlas"]
    if not roots.get("gate_pass", False):
        return v142r.write_scientific_skip(stage, "root_chart_gate_failed")
    growth, tasks, edges, hypotheses = _combined_growth(config, output_root)
    environment = _environment(project_root, config)
    _nodes, continuation = _segmented_continuation(environment, tasks, edges)
    pilot = config["pilot"]
    registered_roots = pd.read_parquet(
        output_root / STAGE_DIRS["pilot_registry"] / "root_registry.parquet"
    ).sort_values("root_index", kind="stable")
    canonical_root_priority = tuple(
        (int(row.task_node_id), str(row.candidate_id))
        for row in registered_roots.itertuples(index=False)
    )
    policy = AtlasRepairPolicy(
        audit=v142r._audit_policy(v142r.load_config(_sources(config, project_root)["v14_2r_config"])),
        minimum_chart_cells=int(pilot["minimum_chart_cells"]),
        minimum_chart_fraction=float(pilot["minimum_chart_fraction"]),
        minimum_chart_spread_mm=float(pilot["minimum_chart_spread_mm"]),
        minimum_stitch_overlap_cells=int(pilot["minimum_stitch_overlap_cells"]),
        stitch_p95_max_deg=float(pilot["stitch_p95_max_deg"]),
        stitch_max_deg=float(pilot["stitch_max_deg"]),
        minimum_boundary_transition_edges=int(pilot["minimum_boundary_transition_edges"]),
        minimum_overlap_fraction=float(pilot["minimum_overlap_fraction"]),
        minimum_overlap_spread_mm=float(pilot["minimum_overlap_spread_mm"]),
        abstention_hops=int(pilot["abstention_hops"]),
        abstention_radius_mm=float(pilot["abstention_radius_mm"]),
    )
    audit_executor = v142r._SubprocessAuditExecutor(
        config=config,
        project_root=project_root,
        tasks=tasks,
        task_edges=edges,
        patch_directory=stage,
        shard_count=int(config["audit_execution"]["logical_shard_count"]),
        maximum_concurrent_workers=int(config["parallel"]["patch_workers"]),
        assignment_strategy=str(config["audit_execution"]["assignment_strategy"]),
        worker_runner_path=Path(__file__).resolve(),
    )
    repair_kwargs = {
        "patch_id": "repaired_5k",
        "method": "rooted_S4_R8",
        "policy": policy,
        "retry_continuation": v142r._registered_retry_adapter(environment, tasks, edges),
        "schedule_executor": audit_executor,
        "canonical_root_priority": canonical_root_priority,
    }
    repaired = repair_rooted_section_atlas(
        growth,
        continuation,
        **repair_kwargs,
        screening_first=bool(config["audit_execution"]["screening_first"]),
    )
    screening_fallback_used = False
    if (
        bool(config["audit_execution"]["screening_first"])
        and bool(config["audit_execution"]["screening_fallback_to_full_audit"])
        and not repaired.certificate_gate
    ):
        screening_fallback_used = True
        repaired = repair_rooted_section_atlas(
            growth,
            continuation,
            **{**repair_kwargs, "method": "rooted_S4_R8_full_fallback"},
            screening_first=False,
        )
    local_refinement_used = False
    refinement = pilot["local_refinement"]
    if (
        bool(refinement["enabled"])
        and int(refinement["maximum_rounds"]) == 1
        and repaired.certificate_gate
        and repaired.coverage_ratio < float(pilot["labelable_measure_min"])
        and repaired.abstained_node_ids
    ):
        local_refinement_used = True
        refined_edges = _add_local_refinement_edges(
            tasks,
            edges,
            sorted(repaired.abstained_node_ids),
            neighbor_count=int(refinement["neighbor_count"]),
        )
        _write_parquet(refined_edges, stage / "registered_local_refinement_edges.parquet")
        refined_nodes = legacy.atlas_nodes_from_frames(tasks, refined_edges)
        candidate_frame = pd.read_parquet(
            output_root / STAGE_DIRS["pilot_registry"] / "pilot_candidate_clusters.parquet"
        )
        candidates = legacy._candidates_from_frame(candidate_frame)
        root_keys = tuple(
            chart.root_key
            for chart in growth.charts
            if chart.chart_id in repaired.selected_stitch_component
        )
        _refined_nodes, refined_continuation = _segmented_continuation(
            environment, tasks, refined_edges
        )
        refined_growth = build_section_first_atlas(
            refined_nodes,
            candidates,
            refined_continuation,
            root_keys=root_keys,
            policy=replace(_root_policy(config), root_count=max(1, len(root_keys))),
        )
        refined_executor = v142r._SubprocessAuditExecutor(
            config=config,
            project_root=project_root,
            tasks=tasks,
            task_edges=refined_edges,
            patch_directory=stage / "local_refinement",
            shard_count=int(config["audit_execution"]["logical_shard_count"]),
            maximum_concurrent_workers=int(config["parallel"]["patch_workers"]),
            assignment_strategy=str(config["audit_execution"]["assignment_strategy"]),
            worker_runner_path=Path(__file__).resolve(),
        )
        refined = repair_rooted_section_atlas(
            refined_growth,
            refined_continuation,
            patch_id="repaired_5k_local_refinement",
            method="rooted_S4_R8_local_refinement",
            policy=policy,
            retry_continuation=v142r._registered_retry_adapter(environment, tasks, refined_edges),
            schedule_executor=refined_executor,
            screening_first=False,
            canonical_root_priority=canonical_root_priority,
        )
        if refined.certificate_gate and refined.coverage_ratio >= repaired.coverage_ratio:
            repaired = refined
            growth = refined_growth
            edges = refined_edges
    stage.mkdir(parents=True, exist_ok=True)
    for name, frame in repaired.frames.items():
        filename = {
            "chart_qualification": "qualified_charts.parquet",
            "chart_stitchability": "chart_overlap_audit.parquet",
            "primary_atlas": "global_primary_assignment.parquet",
            "abstention_sensitivity": "abstention_sensitivity.parquet",
            "chart_audit_v2_schedules": "qualified_chart_audit_schedules.parquet",
            "chart_audit_v2_executions": "qualified_chart_audit_executions.parquet",
            "audit_v2_schedules": "primary_certificate_edges.parquet",
            "audit_v2_executions": "primary_certificate_executions.parquet",
            "primary_optimization": "primary_optimization_stability.parquet",
        }[name]
        _write_parquet(frame, stage / filename)
    stitch_graph = repaired.frames["chart_stitchability"]
    _write_parquet(stitch_graph[stitch_graph.get("stitchable", False).astype(bool)].copy() if not stitch_graph.empty else stitch_graph, stage / "chart_stitch_graph.parquet")
    assignment = repaired.frames["primary_atlas"]
    abstention = assignment[assignment["abstained"].astype(bool)].copy()
    _write_parquet(abstention, stage / "abstention_cells.parquet")
    task_columns = ["task_node_id", *XYZ_COLUMNS, "source_parent_node_id", "cell_level_mm", "cell_ix", "cell_iy", "cell_iz", "physical_point_id"]
    labels = assignment[~assignment["abstained"].astype(bool)].merge(tasks.loc[:, task_columns], on="task_node_id", how="left", validate="one_to_one")
    labels = labels.rename(columns={"primary_chart_id": "chart_id"})
    _write_parquet(labels, stage / "primary_canonical_labels.parquet")
    alternatives = hypotheses[hypotheses["selected"].astype(bool)].merge(tasks.loc[:, ["task_node_id", *XYZ_COLUMNS]], on="task_node_id", how="left")
    _write_parquet(alternatives, stage / "alternative_chart_labels.parquet")
    qualified_support = set().union(
        *(
            set(chart.selected_by_node)
            for chart in growth.charts
            if chart.chart_id in repaired.qualified_chart_ids
        )
    ) if repaired.qualified_chart_ids else set()
    primary_nodes = set(repaired.primary_beta_by_node)
    probe_classification = tasks[["task_node_id", "source_parent_node_id"]].copy()
    probe_classification["classification"] = probe_classification["task_node_id"].map(
        lambda node: (
            "labelable" if int(node) in primary_nodes
            else ("abstain" if int(node) in qualified_support else "unresolved")
        )
    )
    _write_parquet(probe_classification, stage / "probe_classification.parquet")
    counts = (
        probe_classification.groupby(["source_parent_node_id", "classification"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["labelable", "abstain", "unresolved"], fill_value=0)
    )
    parent_meta = tasks.groupby("source_parent_node_id", as_index=True).agg(
        probe_count=("task_node_id", "size"),
        cell_level_mm=("cell_level_mm", "first"),
        cell_ix=("cell_ix", "first"),
        cell_iy=("cell_iy", "first"),
        cell_iz=("cell_iz", "first"),
    )
    fractions = parent_meta.join(counts, how="left").fillna(0).reset_index()
    for classification in ("labelable", "abstain", "unresolved"):
        fractions[f"cell_{classification}_fraction"] = (
            fractions[classification] / fractions["probe_count"]
        )
    fractions["cell_volume_m3"] = (fractions["cell_level_mm"] / 1000.0) ** 3
    primary_chart_count = labels.groupby("source_parent_node_id")["chart_id"].nunique()
    fractions["primary_chart_count"] = (
        fractions["source_parent_node_id"].map(primary_chart_count).fillna(0).astype(int)
    )
    fractions["domain_class"] = np.where(
        fractions["cell_labelable_fraction"].le(0),
        "teacher_unresolved",
        np.where(
            fractions["primary_chart_count"].gt(1),
            "resolved_multichart",
            "resolved_single_under_budget",
        ),
    )
    _write_parquet(fractions, stage / "cell_labelable_fraction.parquet")
    total_volume = float(fractions["cell_volume_m3"].sum())
    measures = {
        classification: float(
            np.sum(
                fractions["cell_volume_m3"]
                * fractions[f"cell_{classification}_fraction"]
            )
            / max(total_volume, np.finfo(float).tiny)
        )
        for classification in ("labelable", "abstain", "unresolved")
    }
    labelable_measure = measures["labelable"]
    unresolved_abstention = measures["abstain"] + measures["unresolved"]
    strict_measure = float(
        np.sum(
            fractions["cell_volume_m3"]
            * fractions["cell_labelable_fraction"].eq(1.0).astype(float)
        )
        / max(total_volume, np.finfo(float).tiny)
    )
    bootstrap_rng = np.random.default_rng(int(pilot["bootstrap_seed"]))
    bootstrap_values = np.empty(int(pilot["bootstrap_replicates"]), dtype=float)
    fractions_array = fractions["cell_labelable_fraction"].to_numpy(float)
    volume_array = fractions["cell_volume_m3"].to_numpy(float)
    for index in range(len(bootstrap_values)):
        sample = bootstrap_rng.integers(0, len(fractions), size=len(fractions))
        bootstrap_values[index] = float(
            np.sum(fractions_array[sample] * volume_array[sample])
            / np.sum(volume_array[sample])
        )
    labelable_lcb = float(np.percentile(bootstrap_values, 2.5))
    labelable_ucb = float(np.percentile(bootstrap_values, 97.5))
    retained_nodes = set(labels["task_node_id"].astype(int))
    retained_adjacency: dict[int, set[int]] = {node: set() for node in retained_nodes}
    for row in edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left in retained_nodes and right in retained_nodes:
            retained_adjacency[left].add(right); retained_adjacency[right].add(left)
    components: list[set[int]] = []
    remaining = set(retained_nodes)
    while remaining:
        queue = [min(remaining)]
        component: set[int] = set()
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            queue.extend(retained_adjacency[node] - component)
        remaining -= component
        components.append(component)
    parent_by_task = tasks.set_index("task_node_id")["source_parent_node_id"].astype(int)
    probe_count_by_parent = fractions.set_index("source_parent_node_id")["probe_count"]
    volume_by_parent = fractions.set_index("source_parent_node_id")["cell_volume_m3"]
    coherent_measures = []
    for component in components:
        component_parent_counts = parent_by_task.loc[sorted(component)].value_counts()
        coherent_measures.append(
            float(
                sum(
                    float(volume_by_parent.loc[parent_id])
                    * int(count)
                    / int(probe_count_by_parent.loc[parent_id])
                    for parent_id, count in component_parent_counts.items()
                )
                / max(total_volume, np.finfo(float).tiny)
            )
        )
    largest_coherent_measure = max(coherent_measures, default=0.0)
    x_bins = pd.qcut(tasks["x_m"], q=3, labels=False, duplicates="drop")
    x_bin_by_node = dict(zip(tasks["task_node_id"].astype(int), x_bins.astype(int), strict=True))
    label_bins = {x_bin_by_node[int(node)] for node in labels["task_node_id"]}
    root_stability, root_stability_frame = _pilot_root_removal_stability(growth, repaired)
    _write_parquet(root_stability_frame, stage / "root_removal_stability.parquet")
    optimization = repaired.frames["primary_optimization"]
    checks = {
        "labelable_measure": labelable_measure >= float(pilot["labelable_measure_min"]),
        "labelable_measure_lcb": labelable_lcb >= float(pilot["labelable_measure_lcb_min"]),
        "largest_coherent_region": largest_coherent_measure >= float(pilot["largest_coherent_region_min"]),
        "unresolved_abstention": unresolved_abstention <= float(pilot["unresolved_abstention_max"]),
        "mutually_exclusive_measure_decomposition": abs(sum(measures.values()) - 1.0) <= 1e-12,
        "all_x_tertiles": len(label_bins) == 3,
        "retained_geometry_certificate": repaired.certificate_gate,
        "root_search_stability": root_stability,
        "primary_optimization_finite": bool(
            len(optimization)
            and np.isfinite(pd.to_numeric(optimization["objective"], errors="coerce")).all()
            and int(optimization["selected"].astype(bool).sum()) == 1
        ),
    }
    return _gate(stage / "gate.json", checks,
        scientific_gate_pass=bool(all(checks.values())),
        labelable_measure_ratio=labelable_measure,
        labelable_measure_bootstrap_lcb95=labelable_lcb,
        labelable_measure_bootstrap_ucb95=labelable_ucb,
        strict_all_probes_labelable_measure_ratio=strict_measure,
        largest_coherent_region_ratio=largest_coherent_measure,
        largest_coherent_probe_ratio_diagnostic=repaired.largest_coherent_region_ratio,
        coherent_component_count=len(components),
        abstention_measure_ratio=measures["abstain"],
        unresolved_measure_ratio=measures["unresolved"],
        unresolved_abstention_ratio=unresolved_abstention,
        measure_decomposition_sum=sum(measures.values()),
        x_tertiles_with_labels=sorted(label_bins),
        root_search_stability=root_stability,
        primary_row_count=len(labels),
        selected_stitch_component=list(repaired.selected_stitch_component),
        induced_edge_count=repaired.induced_edge_count,
        audited_edge_count=repaired.audited_edge_count,
        edge_completeness_ratio=repaired.edge_completeness_ratio,
        pre_abstention_cycle_rank=repaired.pre_abstention_cycle_rank,
        retained_cycle_rank=repaired.retained_cycle_rank,
        audited_fundamental_cycle_count=repaired.audited_fundamental_cycle_count,
        cycle_coverage_ratio=repaired.cycle_coverage_ratio,
        xyz_only_primary_hypothesis_retained=True,
        chart_screening_enabled=bool(config["audit_execution"]["screening_first"]),
        full_audit_fallback_used=screening_fallback_used,
        local_refinement_used=local_refinement_used,
        local_refinement_round_count=int(local_refinement_used),
        primary_optimization_objective_spread=(
            float(optimization["objective"].max() - optimization["objective"].min())
            if len(optimization) else math.inf
        ),
        primary_optimization_assignment_spread=(
            float(optimization["assignment_change_ratio_to_selected"].max())
            if len(optimization) else 1.0
        ),
    )


def _split_role(row: Any, config: Mapping[str, Any]) -> tuple[str, str]:
    macro = int(config["dataset"]["macroblock_mm"]) / 1000.0
    block = tuple(np.floor(np.asarray([row.x_m, row.y_m, row.z_m]) / macro).astype(int))
    digest = int.from_bytes(hashlib.sha256(f"{config['dataset']['split_seed']}:{block}".encode()).digest()[:8], "big") % 10000
    fractions = list(map(float, config["dataset"]["split_fractions"]))
    train_cut = int(10000 * fractions[0]); validation_cut = int(10000 * (fractions[0] + fractions[1]))
    role = "train_core" if digest < train_cut else ("validation" if digest < validation_cut else "test")
    return role, str(block)


def _execute_dataset_shard(
    config: Mapping[str, Any], project_root: Path, output_root: Path, shard_index: int,
    wave_index: int,
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["fixed_budget_dataset"]
    wave_directory = stage / "multiparent_waves" / f"wave_{wave_index:03d}"
    directory = wave_directory / f"shard_{shard_index:03d}"
    report_path = directory / "report.json"
    targets_path = wave_directory / "multiparent_targets.parquet"
    sources_path = output_root / STAGE_DIRS["stitched_atlas"] / "primary_canonical_labels.parquet"
    input_sha = v142r._payload_sha256(
        {
            "wave_index": int(wave_index),
            "shard_index": int(shard_index),
            "targets_sha256": sha256_file(targets_path),
            "sources_sha256": sha256_file(sources_path),
        }
    )
    existing = v142r._load_validated_patch_result(
        directory,
        config=config,
        patch_id=f"dataset_wave_{wave_index:03d}_shard_{shard_index:03d}",
        variant="multiparent",
        input_sha256=input_sha,
    )
    if existing is not None:
        return existing
    targets = pd.read_parquet(targets_path)
    targets = targets[targets["shard_index"].astype(int).eq(shard_index)].copy()
    sources = pd.read_parquet(
        output_root / STAGE_DIRS["stitched_atlas"] / "primary_canonical_labels.parquet"
    ).sort_values("task_node_id", kind="stable")
    source_xyz = sources.loc[:, XYZ_COLUMNS].to_numpy(float)
    tree = cKDTree(source_xyz)
    environment = _environment(project_root, config)
    continuation = make_optimized_predictor_corrector_continuation(
        environment,
        damping=2.0e-3,
        max_corrector_iterations=400,
        residual_tolerance_mm=float(config["dataset"]["continuation_residual_max_mm"]),
    )
    multiparent_count = int(config["dataset"]["multiparent_count"])
    output_rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        distances, indices = tree.query(
            np.asarray([target.x_m, target.y_m, target.z_m], dtype=float),
            k=min(len(sources), max(16, multiparent_count * 8)),
        )
        indices = np.atleast_1d(indices).astype(int)
        selected_source_indices: list[int] = []
        selected_parent_ids: set[int] = set()
        for source_index in indices:
            source_row = sources.iloc[int(source_index)]
            parent_id = int(source_row.source_parent_node_id)
            if parent_id in selected_parent_ids:
                continue
            selected_source_indices.append(int(source_index))
            selected_parent_ids.add(parent_id)
            if len(selected_source_indices) == multiparent_count:
                break
        outcomes: list[tuple[Any, Any]] = []
        target_node = AtlasTaskNode(
            int(target.task_node_id),
            np.asarray([target.x_m, target.y_m, target.z_m], dtype=float),
            (),
        )
        for source_index in selected_source_indices:
            source = sources.iloc[source_index]
            candidate = AtlasCandidate(
                node_id=int(source.task_node_id),
                candidate_id=f"primary_source_{int(source.task_node_id)}",
                beta_rad=source.loc[list(BETA_COLUMNS)].to_numpy(float),
                residual_mm=0.0,
                min_margin_deg=1.0,
                normalized_min_margin=1.0,
                posture_cost=0.0,
                condition_number=0.0,
                quality="Gold",
                solver_success=True,
                actual_bounds=True,
            )
            outcome = continuation(candidate, target_node)
            if (
                outcome.success
                and outcome.actual_bounds
                and outcome.residual_mm
                <= float(config["dataset"]["continuation_residual_max_mm"]) + 1e-12
            ):
                outcomes.append((source, outcome))
        gaps = [
            beta_rms_deg(left[1].beta_rad, right[1].beta_rad)
            for left_index, left in enumerate(outcomes)
            for right in outcomes[left_index + 1 :]
        ]
        maximum_gap = max(gaps, default=math.inf)
        accepted = bool(
            len(outcomes) == multiparent_count
            and maximum_gap <= float(config["dataset"]["multiparent_beta_gap_max_deg"])
        )
        chosen = min(outcomes, key=lambda row: (row[1].residual_mm, int(row[0].task_node_id))) if accepted else None
        output_rows.append(
            {
                "task_node_id": int(target.task_node_id),
                "physical_point_id": str(target.physical_point_id),
                "source_parent_node_id": int(target.source_parent_node_id),
                "cell_level_mm": int(target.cell_level_mm),
                "cell_ix": int(target.cell_ix),
                "cell_iy": int(target.cell_iy),
                "cell_iz": int(target.cell_iz),
                "x_m": float(target.x_m),
                "y_m": float(target.y_m),
                "z_m": float(target.z_m),
                "accepted": accepted,
                "successful_parent_count": len(outcomes),
                "multiparent_beta_gap_max_deg": maximum_gap,
                "source_task_node_ids": [int(row[0].task_node_id) for row in outcomes],
                "source_parent_node_ids": [int(row[0].source_parent_node_id) for row in outcomes],
                "chart_id": str(chosen[0].chart_id) if chosen is not None else None,
                "abstained": not accepted,
                "residual_mm": float(chosen[1].residual_mm) if chosen is not None else math.inf,
                **{
                    name: float(chosen[1].beta_rad[index]) if chosen is not None else math.nan
                    for index, name in enumerate(BETA_COLUMNS)
                },
            }
        )
    output = pd.DataFrame.from_records(output_rows)
    directory.mkdir(parents=True, exist_ok=True)
    _write_parquet(output, directory / "multiparent_results.parquet")
    report = {
        "shard_index": shard_index,
        "target_count": len(output),
        "accepted_count": int(output["accepted"].sum()) if len(output) else 0,
        "gate_pass": True,
    }
    _write_json(report_path, report)
    v142r._write_patch_completion_manifest(
        directory,
        config=config,
        patch_id=f"dataset_wave_{wave_index:03d}_shard_{shard_index:03d}",
        variant="multiparent",
        input_sha256=input_sha,
    )
    return report


def _run_dataset_shards(
    config: Mapping[str, Any], output_root: Path, wave_index: int
) -> None:
    logical_count = int(config["dataset"]["logical_shard_count"])
    worker_limit = int(config["parallel"]["patch_workers"])
    pending = list(range(logical_count))
    running: list[tuple[int, subprocess.Popen[str], Any]] = []
    failures: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/mpl-bacra-v14-3-dataset",
        }
    )
    while pending or running:
        survivors = []
        for shard_index, process, handle in running:
            status = process.poll()
            if status is None:
                survivors.append((shard_index, process, handle))
            else:
                handle.close()
                if status != 0:
                    failures.append({"shard_index": shard_index, "returncode": status})
        running = survivors
        if failures:
            for _shard, process, handle in running:
                process.terminate(); process.wait(timeout=30); handle.close()
            raise RuntimeError(f"V14.3 multiparent shard failures: {failures}")
        while pending and len(running) < worker_limit:
            shard_index = pending.pop(0)
            directory = output_root / STAGE_DIRS["fixed_budget_dataset"] / "multiparent_waves" / f"wave_{wave_index:03d}" / f"shard_{shard_index:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / "worker.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config", str(config["config_path"]),
                "--output-root", str(output_root),
                "--stage", "fixed_budget_dataset",
                "--dataset-shard", str(shard_index),
                "--dataset-wave", str(wave_index),
            ]
            process = subprocess.Popen(
                command, cwd=SOURCE_ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            running.append((shard_index, process, handle))
        if pending or running:
            time.sleep(1.0)


def stage_fixed_budget_dataset(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    pilot = _require(output_root, "stitched_atlas", config)
    stage = output_root / STAGE_DIRS["fixed_budget_dataset"]
    if not pilot.get("gate_pass", False):
        return v142r.write_scientific_skip(stage, "repaired_5k_pilot_gate_failed")
    if config.get("_dataset_shard") is not None:
        report = _execute_dataset_shard(
            config, project_root, output_root, int(config["_dataset_shard"]),
            int(config["_dataset_wave"]),
        )
        return {"gate_pass": True, "worker_report": report}
    labels = pd.read_parquet(output_root / STAGE_DIRS["stitched_atlas"] / "primary_canonical_labels.parquet").sort_values("task_node_id", kind="stable")
    budget = config["dataset"]
    labels = labels.iloc[: int(budget["maximum_unique_rows"])].copy()
    existing_certified_row_count = int(len(labels))
    if labels["task_node_id"].duplicated().any():
        raise RuntimeError("primary canonical dataset contains duplicate physical task points")
    desired_new = max(0, int(budget["target_unique_rows"]) - len(labels))
    if desired_new:
        cells = pd.read_parquet(
            output_root / STAGE_DIRS["stitched_atlas"] / "cell_labelable_fraction.parquet"
        )
        cells = cells[cells["cell_labelable_fraction"].gt(0)].sort_values(
            "source_parent_node_id", kind="stable"
        )
        task_meta = pd.read_parquet(
            output_root / STAGE_DIRS["pilot_registry"] / "pilot_task_nodes.parquet"
        )
        representative = task_meta[task_meta["is_representative"].astype(bool)].set_index(
            "source_parent_node_id"
        )
        maximum_attempts = int(budget["maximum_attempt_rows"])
        cell_cycle = np.resize(cells["source_parent_node_id"].to_numpy(dtype=int), maximum_attempts)
        sobol_power = int(math.ceil(math.log2(max(1, maximum_attempts))))
        unit = qmc.Sobol(
            d=3, scramble=True, seed=int(budget["enrichment_seed"])
        ).random_base2(sobol_power)[:maximum_attempts]
        first_id = int(task_meta["task_node_id"].max()) + 1
        accepted_parts: list[pd.DataFrame] = []
        result_parts: list[pd.DataFrame] = []
        attempted = 0
        wave_index = 0
        while len(labels) + sum(len(part) for part in accepted_parts) < int(budget["target_unique_rows"]) and attempted < maximum_attempts:
            wave_size = min(int(budget["wave_size"]), maximum_attempts - attempted)
            targets = []
            for local_index in range(wave_size):
                index = attempted + local_index
                parent_id, offset = int(cell_cycle[index]), unit[index]
                parent = representative.loc[parent_id]
                level_m = float(parent.cell_level_mm) / 1000.0
                xyz = (
                    np.asarray([parent.cell_ix, parent.cell_iy, parent.cell_iz], dtype=float)
                    + np.asarray(offset, dtype=float)
                ) * level_m
                digest = hashlib.sha256(np.asarray(xyz, dtype="<f8").tobytes()).hexdigest()[:24]
                targets.append({
                    "task_node_id": first_id + index,
                    "physical_point_id": f"v14_3_sobol_{digest}",
                    "source_parent_node_id": parent_id,
                    "cell_level_mm": int(parent.cell_level_mm),
                    "cell_ix": int(parent.cell_ix), "cell_iy": int(parent.cell_iy), "cell_iz": int(parent.cell_iz),
                    "x_m": float(xyz[0]), "y_m": float(xyz[1]), "z_m": float(xyz[2]),
                    "shard_index": index % int(budget["logical_shard_count"]),
                })
            wave_directory = stage / "multiparent_waves" / f"wave_{wave_index:03d}"
            _write_parquet(pd.DataFrame.from_records(targets), wave_directory / "multiparent_targets.parquet")
            _run_dataset_shards(config, output_root, wave_index)
            wave_results = pd.concat([
                pd.read_parquet(wave_directory / f"shard_{index:03d}" / "multiparent_results.parquet")
                for index in range(int(budget["logical_shard_count"]))
            ], ignore_index=True)
            result_parts.append(wave_results)
            accepted_parts.append(wave_results[wave_results["accepted"].astype(bool)].copy())
            attempted += wave_size
            wave_index += 1
        results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
        _write_parquet(results, stage / "multiparent_audit.parquet")
        accepted = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
        labels = pd.concat(
            [labels, accepted[[*labels.columns]]], ignore_index=True
        ).iloc[: int(budget["maximum_unique_rows"])]
    else:
        results = pd.DataFrame()
    environment = _environment(project_root, config)
    beta_matrix = labels.loc[:, BETA_COLUMNS].to_numpy(float)
    jacobian_matrix = environment.fk_and_jacobian(beta_matrix)[1]
    rows = []
    for row_index, row in enumerate(labels.itertuples(index=False)):
        jacobian = jacobian_matrix[row_index]
        split_role, macroblock = _split_role(row, config)
        rows.append({
            "record_id": f"v14_3_{int(row.task_node_id):07d}",
            "task_node_id": int(row.task_node_id),
            "physical_point_id": str(row.physical_point_id),
            "kind": "static", "split_role": split_role,
            "chart_id": str(row.chart_id), "is_primary": True, "sample_weight": 1.0,
            **{name: float(getattr(row, name)) for name in XYZ_COLUMNS},
            **{name: float(getattr(row, name)) for name in BETA_COLUMNS},
            **{name: float(jacobian.reshape(-1)[index]) for index, name in enumerate(JACOBIAN_COLUMNS)},
            "macroblock": macroblock,
        })
    frame = pd.DataFrame.from_records(rows)
    _write_parquet(frame, stage / "primary_student_supervision.parquet")
    _write_parquet(
        pd.read_parquet(output_root / STAGE_DIRS["stitched_atlas"] / "alternative_chart_labels.parquet"),
        stage / "diagnostic_alternative_branches_not_supervision.parquet",
    )
    counts = frame["split_role"].value_counts().to_dict()
    checks = {
        "minimum_unique_rows": len(frame) >= int(budget["minimum_unique_rows"]),
        "maximum_budget": len(frame) <= int(budget["maximum_unique_rows"]),
        "no_duplicate_records": not frame["record_id"].duplicated().any(),
        "no_duplicate_physical_points": bool(
            not frame["physical_point_id"].duplicated().any()
            and not frame.loc[:, XYZ_COLUMNS].duplicated().any()
        ),
        "no_padding": not bool(budget["row_padding"]),
        "macroblock_split_complete": all(counts.get(role, 0) > 0 for role in ("train_core", "validation", "test")),
        "multiparent_required_for_new_rows": bool(
            not desired_new
            or (
                len(results)
                and results.loc[results["accepted"].astype(bool), "successful_parent_count"]
                .ge(int(budget["multiparent_count"]))
                .all()
            )
        ),
    }
    return _gate(stage / "gate.json", checks,
        supervision_row_count=len(frame), split_counts=counts, padding_count=0,
        all_supervised_rows_count_toward_budget=True,
        existing_certified_row_count=existing_certified_row_count,
        multiparent_target_count=int(len(results)),
        multiparent_accepted_count=int(results["accepted"].sum()) if len(results) else 0,
        alternative_branches_used_for_xyz_only_supervision=False,
    )


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )


def _fk_metrics(environment: Any, beta: np.ndarray, xyz: np.ndarray) -> dict[str, float]:
    residual = np.linalg.norm(np.asarray(environment.fk(beta)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    return {
        "fk_p95_mm": float(np.percentile(residual, 95)),
        "fk_p99_mm": float(np.percentile(residual, 99)),
        "fk_p999_mm": float(np.percentile(residual, 99.9)),
        "fk_max_mm": float(np.max(residual)),
        "fk_residual_mm": residual,
    }


def _cpu_batch1_latency_ms(
    predictor: Any, xyz: np.ndarray, *, warmup: int, repeats: int
) -> float:
    point = np.asarray(xyz[:1], dtype=float)
    for _ in range(warmup):
        predictor(point)
    started = time.perf_counter()
    for _ in range(repeats):
        predictor(point)
    return (time.perf_counter() - started) * 1000.0 / max(1, repeats)


def _trajectory_metrics(
    environment: Any,
    beta: np.ndarray,
    targets: pd.DataFrame,
    *,
    accepted: np.ndarray | None,
) -> dict[str, float]:
    xyz = targets.loc[:, XYZ_COLUMNS].to_numpy(float)
    metrics = _fk_metrics(environment, beta, xyz)
    residual = np.asarray(metrics.pop("fk_residual_mm"), dtype=float)
    closed_loop_gaps: list[float] = []
    boundary_jumps: list[float] = []
    for _family_id, indices in targets.groupby("family_id", sort=True).groups.items():
        ordered = list(indices)
        if len(ordered) >= 2:
            closed_loop_gaps.append(beta_rms_deg(beta[ordered[0]], beta[ordered[-1]]))
        if str(targets.loc[ordered[0], "family_type"]) == "cross_chart":
            boundary_jumps.extend(
                beta_rms_deg(beta[left], beta[right])
                for left, right in zip(ordered, ordered[1:])
            )
    return {
        "trajectory_fk_p95_mm": float(np.percentile(residual, 95)),
        "trajectory_fk_max_mm": float(np.max(residual)),
        "closed_loop_beta_return_max_deg": max(closed_loop_gaps, default=0.0),
        "chart_boundary_beta_jump_max_deg": max(boundary_jumps, default=0.0),
        "trajectory_suite_coverage": (
            float(np.mean(np.asarray(accepted, dtype=bool))) if accepted is not None else 1.0
        ),
    }


def _dls_metrics(environment: Any, beta: np.ndarray, xyz: np.ndarray) -> dict[str, float]:
    corrected = beta.copy(); output = {}
    bounds = np.asarray(environment.bounds, dtype=float)
    for step in (1, 2):
        for index in range(len(corrected)):
            error = xyz[index] - np.asarray(environment.fk(corrected[index])).reshape(-1, 3)[0]
            proposal = corrected[index] + weighted_damped_pinv(
                np.asarray(environment.jacobian(corrected[index])), damping=1e-3,
                weights=np.asarray([4, 4, 2, 2, 1, 1], dtype=float),
            ) @ error
            if np.all(proposal >= bounds[:, 0]) and np.all(proposal <= bounds[:, 1]):
                corrected[index] = proposal
        metrics = _fk_metrics(environment, corrected, xyz)
        output[f"dls_{step}_fk_p95_mm"] = metrics["fk_p95_mm"]
        output[f"dls_{step}_fk_max_mm"] = metrics["fk_max_mm"]
    return output


def _dls_candidates(
    environment: Any, beta: np.ndarray, xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    corrected = np.asarray(beta, dtype=float).copy()
    outputs: list[np.ndarray] = []
    bounds = np.asarray(environment.bounds, dtype=float)
    for _step in (1, 2):
        current_xyz, jacobians = environment.fk_and_jacobian(corrected)
        error = np.asarray(xyz, dtype=float) - current_xyz
        for index in range(len(corrected)):
            proposal = corrected[index] + weighted_damped_pinv(
                jacobians[index],
                damping=1e-3,
                weights=np.asarray([4, 4, 2, 2, 1, 1], dtype=float),
            ) @ error[index]
            if np.all(proposal >= bounds[:, 0]) and np.all(proposal <= bounds[:, 1]):
                corrected[index] = proposal
        outputs.append(corrected.copy())
    return outputs[0], outputs[1]


def _student_pass_mask(metrics: pd.DataFrame, student: Mapping[str, Any]) -> pd.Series:
    return (
        metrics["training_error"].isna()
        & metrics["fk_p95_mm"].le(float(student["fk_p95_max_mm"]))
        & metrics["fk_p99_mm"].le(float(student["fk_p99_max_mm"]))
        & metrics["fk_p999_mm"].le(float(student["fk_p999_max_mm"]))
        & metrics["trajectory_fk_max_mm"].le(float(student["trajectory_fk_max_mm"]))
        & metrics["closed_loop_beta_return_max_deg"].le(
            float(student["closed_loop_beta_return_max_deg"])
        )
        & metrics["chart_boundary_beta_jump_max_deg"].le(
            float(student["chart_boundary_beta_jump_max_deg"])
        )
        & metrics["trajectory_suite_coverage"].ge(1.0)
    )


def stage_students(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    dataset_gate = _require(output_root, "fixed_budget_dataset", config)
    stage = output_root / STAGE_DIRS["students"]
    if not dataset_gate.get("gate_pass", False):
        return v142r.write_scientific_skip(stage, "unique_supervision_budget_gate_failed")
    frame = pd.read_parquet(output_root / STAGE_DIRS["fixed_budget_dataset"] / "primary_student_supervision.parquet")
    train = frame[frame["split_role"].eq("train_core")].copy()
    validation = frame[frame["split_role"].eq("validation")].copy()
    test = frame[frame["split_role"].eq("test")].copy()
    environment = _environment(project_root, config)
    geometry = _student_geometry(environment)
    student = config["student"]
    sources = _sources(config, project_root)
    historical = pd.read_parquet(sources["historical_final8_teacher"])
    historical_catalog = pd.read_parquet(sources["historical_final8_catalog"])[
        ["family_id", "major_semiaxis_m"]
    ]
    historical = historical.merge(
        historical_catalog, on="family_id", how="left", validate="many_to_one"
    )
    # Analytic trajectories are projected only onto sealed test macroblocks;
    # historical final8 targets are locked external evaluation paths.
    capability = test.loc[:, [*XYZ_COLUMNS]].copy()
    classification = pd.read_parquet(
        output_root / STAGE_DIRS["stitched_atlas"] / "cell_labelable_fraction.parquet"
    )
    trajectory_suite = build_workspace_trajectory_suite(
        capability,
        classification,
        historical,
        phase_count=int(student["trajectory_waypoints"]),
        projection_max_mm=float(student["capability_projection_max_mm"]),
    )
    _write_parquet(trajectory_suite.catalog, stage / "fixed_trajectory_catalog.parquet")
    _write_parquet(trajectory_suite.targets, stage / "fixed_trajectory_targets.parquet")
    trajectory_targets = trajectory_suite.targets[
        trajectory_suite.targets["teacher_feasible"].astype(bool)
    ].reset_index(drop=True)
    _write_json(
        stage / "fixed_trajectory_registration.json",
        {
            "catalog_sha256": sha256_file(stage / "fixed_trajectory_catalog.parquet"),
            "targets_sha256": sha256_file(stage / "fixed_trajectory_targets.parquet"),
            "registered_before_model_open": True,
            "analytic_projection_split": "test_macroblocks_only",
            "historical_paths_are_external_locked_evaluation": True,
            "evaluation_rows_count_toward_training_budget": False,
            "family_count": int(trajectory_suite.catalog["family_id"].nunique()),
            "feasible_family_count": int(
                trajectory_targets["family_id"].nunique()
            ),
            "family_type_counts": {
                str(key): int(value)
                for key, value in trajectory_suite.catalog["family_type"].value_counts().items()
            },
        },
    )
    reports = []
    predictions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    trained_modes: list[RepresentationMode] = []
    for mode in (RepresentationMode.XYZ_GLOBAL, RepresentationMode.XYZ_ROUTER_EXPERTS):
        if mode is RepresentationMode.XYZ_ROUTER_EXPERTS:
            global_frame = pd.DataFrame.from_records(reports)
            if len(global_frame) and _student_pass_mask(global_frame, student).any():
                break
        if mode is RepresentationMode.XYZ_ROUTER_EXPERTS:
            common = set(train["chart_id"].astype(str)) & set(validation["chart_id"].astype(str)) & set(test["chart_id"].astype(str))
            mode_train = train[train["chart_id"].astype(str).isin(common)]
            mode_validation = validation[validation["chart_id"].astype(str).isin(common)]
            mode_test = test[test["chart_id"].astype(str).isin(common)]
            if len(common) < 2:
                continue
        else:
            mode_train, mode_validation, mode_test = train, validation, test
        trained_modes.append(mode)
        for seed in map(int, student["seeds"]):
            try:
                training = train_workspace_student(
                    mode_train, mode_validation, mode=mode, geometry=geometry,
                    config=WorkspaceStudentTrainingConfig(
                        hidden_units=tuple(map(int, student["hidden_units"])),
                        router_hidden_units=tuple(map(int, student["router_hidden_units"])),
                        learning_rate=float(student["learning_rate"]), max_steps=int(student["max_steps"]),
                        validation_interval=int(student["validation_interval"]),
                        patience_intervals=int(student["patience_intervals"]), seed=seed,
                    ),
                )
                model_dir = stage / f"{mode.value}_seed_{seed}"
                save_workspace_student_models(training.models, model_dir)
                _write_parquet(training.history, model_dir / "training_history.parquet")
                xyz = mode_test.loc[:, XYZ_COLUMNS].to_numpy(float)
                prediction = training.inverse.predict(InverseQuery(xyz))
                beta = np.asarray(prediction.beta_rad, dtype=float)
                metrics = _fk_metrics(environment, beta, xyz)
                metrics.pop("fk_residual_mm")
                trajectory_prediction = training.inverse.predict(
                    InverseQuery(trajectory_targets.loc[:, XYZ_COLUMNS].to_numpy(float))
                )
                trajectory_metrics = _trajectory_metrics(
                    environment,
                    np.asarray(trajectory_prediction.beta_rad, dtype=float),
                    trajectory_targets,
                    accepted=np.asarray(trajectory_prediction.accepted, dtype=bool),
                )
                predictions[(mode.value, seed)] = (
                    beta.copy(),
                    np.asarray(trajectory_prediction.beta_rad, dtype=float).copy(),
                    np.asarray(trajectory_prediction.accepted, dtype=bool).copy(),
                )

                def tf_predict(point: np.ndarray) -> np.ndarray:
                    try:
                        import tensorflow as tf
                        with tf.device("/CPU:0"):
                            value = training.inverse.predict(InverseQuery(point))
                    except ImportError:
                        value = training.inverse.predict(InverseQuery(point))
                    return np.asarray(value.beta_rad, dtype=float)

                reports.append({
                    "model": mode.value, "seed": seed, "train_rows": len(mode_train),
                    "test_rows": len(mode_test), "accepted_fraction": float(np.mean(prediction.accepted)),
                    **metrics,
                    **trajectory_metrics,
                    "cpu_batch1_latency_ms": _cpu_batch1_latency_ms(
                        tf_predict,
                        xyz,
                        warmup=int(student["cpu_batch1_warmup"]),
                        repeats=int(student["cpu_batch1_repeats"]),
                    ),
                    "training_error": None,
                })
            except Exception as error:
                reports.append({
                    "model": mode.value, "seed": seed,
                    "training_error": f"{type(error).__name__}:{error}",
                    "fk_p95_mm": math.inf, "fk_p99_mm": math.inf,
                    "fk_p999_mm": math.inf, "fk_max_mm": math.inf,
                    "trajectory_fk_max_mm": math.inf,
                    "closed_loop_beta_return_max_deg": math.inf,
                    "chart_boundary_beta_jump_max_deg": math.inf,
                    "trajectory_suite_coverage": 0.0,
                })
    x_train = train.loc[:, XYZ_COLUMNS].to_numpy(float); y_train = train.loc[:, BETA_COLUMNS].to_numpy(float)
    x_test = test.loc[:, XYZ_COLUMNS].to_numpy(float); y_test_xyz = test.loc[:, XYZ_COLUMNS].to_numpy(float)
    baselines = []
    from sklearn.neighbors import KNeighborsRegressor
    baselines.append(("knn_diagnostic", KNeighborsRegressor(n_neighbors=8, weights="distance", n_jobs=12)))
    for name, model in baselines:
        try:
            model.fit(x_train, y_train)
            beta = np.asarray(model.predict(x_test), dtype=float)
            metrics = _fk_metrics(environment, beta, y_test_xyz)
            metrics.pop("fk_residual_mm")
            trajectory_xyz = trajectory_targets.loc[:, XYZ_COLUMNS].to_numpy(float)
            trajectory_beta = np.asarray(model.predict(trajectory_xyz), dtype=float)
            predictions[(name, 20260881)] = (beta.copy(), trajectory_beta.copy(), None)
            reports.append({
                "model": name, "seed": 20260881, "train_rows": len(train), "test_rows": len(test),
                **metrics,
                **_trajectory_metrics(
                    environment, trajectory_beta, trajectory_targets, accepted=None
                ),
                "cpu_batch1_latency_ms": _cpu_batch1_latency_ms(
                    lambda point, fitted=model: np.asarray(fitted.predict(point), dtype=float),
                    x_test,
                    warmup=int(student["cpu_batch1_warmup"]),
                    repeats=int(student["cpu_batch1_repeats"]),
                ),
                "training_error": None,
                "diagnostic_only": True,
            })
        except Exception as error:
            reports.append({
                "model": name, "training_error": f"{type(error).__name__}:{error}",
                "fk_p95_mm": math.inf, "fk_p99_mm": math.inf,
                "fk_p999_mm": math.inf, "fk_max_mm": math.inf,
                "trajectory_fk_max_mm": math.inf,
                "closed_loop_beta_return_max_deg": math.inf,
                "chart_boundary_beta_jump_max_deg": math.inf,
                "trajectory_suite_coverage": 0.0,
            })
    metrics = pd.DataFrame.from_records(reports)
    if "diagnostic_only" not in metrics:
        metrics["diagnostic_only"] = False
    else:
        metrics["diagnostic_only"] = metrics["diagnostic_only"].fillna(False).astype(bool)
    eligible = metrics[~metrics["diagnostic_only"]].copy()
    eligible_pass = _student_pass_mask(eligible, student)
    if len(eligible) and not eligible_pass.any():
        selected_index = eligible.sort_values(
            ["fk_p95_mm", "trajectory_fk_max_mm", "model", "seed"], kind="stable"
        ).index[0]
        selected = metrics.loc[selected_index]
        key = (str(selected["model"]), int(selected["seed"]))
        if key in predictions:
            test_beta, trajectory_beta, trajectory_accepted = predictions[key]
            test_xyz = test.loc[:, XYZ_COLUMNS].to_numpy(float)
            trajectory_xyz = trajectory_targets.loc[:, XYZ_COLUMNS].to_numpy(float)
            test_steps = _dls_candidates(environment, test_beta, test_xyz)
            trajectory_steps = _dls_candidates(environment, trajectory_beta, trajectory_xyz)
            correction_rows = []
            for step, (corrected_test, corrected_trajectory) in enumerate(
                zip(test_steps, trajectory_steps, strict=True), start=1
            ):
                corrected_metrics = _fk_metrics(environment, corrected_test, test_xyz)
                corrected_metrics.pop("fk_residual_mm")
                correction_rows.append({
                    "model": f"selected_mlp_dls_{step}",
                    "base_model": key[0],
                    "seed": key[1],
                    "train_rows": int(selected.get("train_rows", 0)),
                    "test_rows": len(test),
                    "accepted_fraction": float(selected.get("accepted_fraction", 1.0)),
                    **corrected_metrics,
                    **_trajectory_metrics(
                        environment,
                        corrected_trajectory,
                        trajectory_targets,
                        accepted=trajectory_accepted,
                    ),
                    "cpu_batch1_latency_ms": math.nan,
                    "training_error": None,
                    "diagnostic_only": False,
                    "dls_steps": step,
                })
                candidate_frame = pd.DataFrame.from_records([correction_rows[-1]])
                if _student_pass_mask(candidate_frame, student).iloc[0]:
                    break
            if correction_rows:
                metrics = pd.concat(
                    [metrics, pd.DataFrame.from_records(correction_rows)], ignore_index=True
                )
    _write_parquet(metrics, stage / "student_metrics.parquet")
    successful = metrics[metrics["training_error"].isna() & ~metrics["diagnostic_only"]]
    passing = (
        successful["fk_p95_mm"].le(float(student["fk_p95_max_mm"]))
        & successful["fk_p99_mm"].le(float(student["fk_p99_max_mm"]))
        & successful["fk_p999_mm"].le(float(student["fk_p999_max_mm"]))
        & successful["trajectory_fk_max_mm"].le(float(student["trajectory_fk_max_mm"]))
        & successful["closed_loop_beta_return_max_deg"].le(
            float(student["closed_loop_beta_return_max_deg"])
        )
        & successful["chart_boundary_beta_jump_max_deg"].le(
            float(student["chart_boundary_beta_jump_max_deg"])
        )
        & successful["trajectory_suite_coverage"].ge(1.0)
    )
    physical_pass = _student_pass_mask(successful, student)
    base_models = successful.get(
        "base_model", pd.Series(index=successful.index, dtype=object)
    ).fillna("")
    global_family = successful["model"].eq(
        RepresentationMode.XYZ_GLOBAL.value
    ) | base_models.eq(RepresentationMode.XYZ_GLOBAL.value)
    global_raw_pass = bool(
        (physical_pass & successful["model"].eq(RepresentationMode.XYZ_GLOBAL.value)).any()
    )
    global_dls_pass = bool(
        (
            physical_pass
            & global_family
            & successful["model"].astype(str).str.startswith("selected_mlp_dls_")
        ).any()
    )
    return _gate(stage / "gate.json", {"at_least_one_student_passes": bool(len(passing) and passing.any())},
        scientific_gate_pass=bool(len(passing) and passing.any()), model_count=len(metrics),
        router_features=list(ROUTER_FEATURE_COLUMNS), router_oracle_chart_input=False,
        random_test_max_is_diagnostic_only=True,
        fixed_trajectory_count=int(student["trajectory_count"]),
        fixed_trajectory_waypoints=int(student["trajectory_waypoints"]),
        batch1_latency_device="CPU",
        staged_training_order=[mode.value for mode in trained_modes],
        router_skipped_after_global_pass=(
            RepresentationMode.XYZ_ROUTER_EXPERTS not in trained_modes
        ),
        knn_is_diagnostic_only=True,
        sklearn_mlp_and_lightgbm_removed=True,
        global_xyz_only_pass=bool(global_raw_pass or global_dls_pass),
        global_xyz_only_raw_pass=global_raw_pass,
        global_xyz_only_dls_pass=global_dls_pass,
    )


def stage_representation_decision(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    student = _require(output_root, "students", config)
    pilot = _require(output_root, "stitched_atlas", config)
    stage = output_root / STAGE_DIRS["representation_decision"]
    overlaps = pd.read_parquet(output_root / STAGE_DIRS["stitched_atlas"] / "chart_overlap_audit.parquet")
    nonstitchable = overlaps[(~overlaps.get("stitchable", False).astype(bool)) & pd.to_numeric(overlaps.get("beta_max_deg"), errors="coerce").gt(1.0)] if not overlaps.empty else overlaps
    global_pass = bool(student.get("global_xyz_only_pass", False))
    primary_abstention_required = float(pilot.get("abstention_measure_ratio", 0.0)) > 0.0
    if global_pass:
        representation = (
            "global_xyz_only_with_dls"
            if bool(student.get("global_xyz_only_dls_pass", False))
            and not bool(student.get("global_xyz_only_raw_pass", False))
            else "global_xyz_only"
        )
    elif student.get("scientific_gate_pass", False):
        representation = "xyz_only_router_experts"
    elif primary_abstention_required and float(pilot.get("unresolved_abstention_ratio", 1.0)) <= 0.70:
        representation = "xyz_only_with_abstention"
    else:
        representation = "blocked_pending_paired_history_experiment"
    stateful_triggered = False
    report = {
        "selected_representation": representation,
        "xyz_only_retained": representation.startswith("global_xyz_only") or representation.startswith("xyz_only"),
        "stateful_triggered": stateful_triggered,
        "stateful_trigger_evidence_complete": False,
        "paired_same_xyz_required_before_stateful": True,
        "branch_token_required": False,
        "multiple_candidates_required": False,
        "abstention_required": representation == "xyz_only_with_abstention",
        "nonstitchable_overlap_count": len(nonstitchable),
        "nonstitchable_alternatives_are_diagnostic_only": True,
        "primary_abstention_measure_ratio": float(pilot.get("abstention_measure_ratio", 0.0)),
    }
    return _gate(stage / "gate.json", {"representation_decided": representation != "blocked_pending_paired_history_experiment"}, **report)


def stage_reach_update(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    """Conditionally run the one registered Reach Round 8 and never a Round 9."""

    representation = _require(output_root, "representation_decision", config)
    pilot = _require(output_root, "stitched_atlas", config)
    dataset = _require(output_root, "fixed_budget_dataset", config)
    students = _require(output_root, "students", config)
    stage = output_root / STAGE_DIRS["reach_update"]
    round7_path = _sources(config, project_root)["v14_2r"] / "08_reach_round7/gate.json"
    round7 = _read_json(round7_path)
    formal = config["formal_gate"]
    nonreach_checks = {
        "labelable_measure": float(pilot.get("labelable_measure_ratio", 0.0))
        >= float(formal["labelable_measure_min"]),
        "labelable_measure_lcb": float(
            pilot.get("labelable_measure_bootstrap_lcb95", 0.0)
        )
        >= float(formal["labelable_measure_lcb_min"]),
        "unresolved_plus_abstention": float(
            pilot.get("unresolved_abstention_ratio", 1.0)
        )
        <= float(formal["unresolved_max"]),
        "student": bool(students.get("gate_pass", False)),
        "representation": bool(representation.get("gate_pass", False)),
        "budget": int(dataset.get("supervision_row_count", 10**12))
        <= int(formal["maximum_n_min"]),
    }
    if bool(round7.get("gate_pass", False)):
        return _gate(
            stage / "gate.json",
            {"reach_converged": True},
            reach_source="round7",
            round8_executed=False,
            round9_forbidden=True,
            nonreach_checks=nonreach_checks,
        )
    if not all(nonreach_checks.values()):
        return _gate(
            stage / "gate.json",
            {"reach_converged": False},
            reach_source="round7_failed_round8_skipped_nonreach_impossible",
            round8_executed=False,
            round9_forbidden=True,
            nonreach_checks=nonreach_checks,
        )

    reach = config["reach_round8"]
    v142 = _sources(config, project_root)["v14_2r"]
    round7_a = pd.read_parquet(v142 / "08_reach_round7/replica_a_slab_round7.parquet")
    round7_b = pd.read_parquet(v142 / "08_reach_round7/replica_b_slab_round7.parquet")
    round7_report = _read_json(v142 / "08_reach_round7/reach_round7_report.json")
    chunk = 2 ** int(reach["sobol_power"])
    round6_a, round6_b = round7_a.iloc[:-chunk], round7_b.iloc[:-chunk]
    round5_a, round5_b = round6_a.iloc[:-chunk], round6_b.iloc[:-chunk]
    seeds_a = tuple(map(int, round7_report["replica_a_seed_lineage"]))
    seeds_b = tuple(map(int, round7_report["replica_b_seed_lineage"]))
    legacy_config = legacy.load_config(_sources(config, project_root)["legacy_config"])
    legacy_paths = legacy._paths(legacy_config, project_root)
    grid = legacy._workspace_grid(legacy_paths)
    environment = _environment(project_root, config)
    spec = ReachSamplingRoundSpec(
        int(reach["sobol_power"]), int(reach["seed_a"]), int(reach["seed_b"])
    )
    generated = generate_independent_reach_samples(
        environment,
        np.asarray(environment.bounds, dtype=float),
        grid=grid,
        rounds=(spec,),
        chunk_rows=65536,
    )

    def new_frame(xyz: np.ndarray, beta: np.ndarray, source: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x_m": xyz[:, 0], "y_m": xyz[:, 1], "z_m": xyz[:, 2],
                **{name: beta[:, index] for index, name in enumerate(BETA_COLUMNS)},
                "source": source,
            }
        )

    final_a = pd.concat(
        [round7_a, new_frame(generated.xyz_a, generated.beta_a, "replica_a_round_8")],
        ignore_index=True,
    )
    final_b = pd.concat(
        [round7_b, new_frame(generated.xyz_b, generated.beta_b, "replica_b_round_8")],
        ignore_index=True,
    )
    xyz_columns = ["x_m", "y_m", "z_m"]
    rounds = (
        ReachReplicaRound(6, ReachReplica("A", seeds_a[:-1], round6_a[xyz_columns].to_numpy(float)), ReachReplica("B", seeds_b[:-1], round6_b[xyz_columns].to_numpy(float))),
        ReachReplicaRound(7, ReachReplica("A", seeds_a, round7_a[xyz_columns].to_numpy(float)), ReachReplica("B", seeds_b, round7_b[xyz_columns].to_numpy(float))),
        ReachReplicaRound(8, ReachReplica("A", (*seeds_a, spec.seed_a), final_a[xyz_columns].to_numpy(float)), ReachReplica("B", (*seeds_b, spec.seed_b), final_b[xyz_columns].to_numpy(float))),
    )
    builder = ReachProxyBuilder(ReachProxyConfig(
        grid=grid,
        minimum_weighted_jaccard=float(reach["weighted_jaccard_min"]),
        maximum_new_volume_ratio=float(reach["new_volume_ratio_max"]),
        maximum_boundary_change_ratio=1.0,
        maximum_frontier_new_volume_ratio=float(reach["frontier_new_volume_ratio_max"]),
        required_consecutive_rounds=int(reach["required_consecutive_rounds"]),
    ))
    frontier_evidence: list[FrontierProbeEvidence] = []
    for frontier_path in (
        legacy_paths["original_frontier"],
        legacy_paths["retry4_frontier"],
        v142 / "08_reach_round7/frontier_inverse_probes_round7.parquet",
    ):
        if frontier_path.is_file():
            frame = pd.read_parquet(frontier_path)
            frontier_evidence.extend(
                FrontierProbeEvidence(
                    CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)),
                    bool(row.found_valid_inverse),
                    round_id=int(row.round_id),
                )
                for row in frame.itertuples(index=False)
            )
    domain = pd.read_parquet(legacy_paths["original_domain_cells"])
    supplemental = tuple(
        CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in domain[
            domain["cell_level_mm"].eq(grid.convergence_level_mm)
            & (domain["registered_pool_support"].astype(bool) | domain["tip_pool_support"].astype(bool))
        ].itertuples(index=False)
    )
    preliminary = builder.build(
        rounds,
        frontier_evidence=frontier_evidence,
        supplemental_supported_cells=supplemental,
    )
    frontier_cells = legacy.frontier_candidates(
        preliminary.proxy_upper_cells,
        grid=grid,
        maximum_count=int(reach["frontier_cell_budget"]),
    )
    seed_xyz = np.vstack([final_a[xyz_columns].to_numpy(float), final_b[xyz_columns].to_numpy(float)])
    seed_beta = np.vstack([final_a.loc[:, BETA_COLUMNS].to_numpy(float), final_b.loc[:, BETA_COLUMNS].to_numpy(float)])
    new_frontier, frontier_frame = legacy._frontier_probe_round(
        environment,
        frontier_cells,
        grid=grid,
        seed_xyz=seed_xyz,
        seed_beta=seed_beta,
        starts=int(reach["frontier_starts_per_cell"]),
        round_id=8,
    )
    frontier_evidence.extend(new_frontier)
    result = builder.build(rounds, frontier_evidence=frontier_evidence, supplemental_supported_cells=supplemental)
    level = grid.convergence_level_mm
    unions = [grid.cells_for_points(pair.replica_a.xyz_m, level_mm=level) | grid.cells_for_points(pair.replica_b.xyz_m, level_mm=level) for pair in rounds]
    boundary = {
        7: measure_weighted_boundary_change_ratio(unions[1], unions[0]),
        8: measure_weighted_boundary_change_ratio(unions[2], unions[1]),
    }
    metrics = [asdict(item) for item in result.replica_metrics]
    recent = metrics[-2:]
    checks = [
        bool(
            row["volume_weighted_jaccard"] >= float(reach["weighted_jaccard_min"])
            and row["new_volume_ratio"] <= float(reach["new_volume_ratio_max"])
            and row["frontier_new_volume_ratio"] <= float(reach["frontier_new_volume_ratio_max"])
            and boundary[int(row["round_id"])] <= float(reach["boundary_change_ratio_max"])
        )
        for row in recent
    ]
    gap = (len(result.proxy_upper_cells) - len(result.proxy_lower_cells)) / max(1, len(result.proxy_upper_cells))
    convergence = bool(all(checks) and gap <= float(reach["lower_upper_measure_gap_max"]))
    _write_parquet(final_a, stage / "replica_a_slab_round8.parquet")
    _write_parquet(final_b, stage / "replica_b_slab_round8.parquet")
    _write_parquet(frontier_frame, stage / "frontier_inverse_probes_round8.parquet")
    report = {
        "reach_convergence_gate": convergence,
        "metrics_round6_8": metrics,
        "measure_weighted_boundary_change_ratio": boundary,
        "recent_round_checks": checks,
        "proxy_lower_cell_count": len(result.proxy_lower_cells),
        "proxy_upper_cell_count": len(result.proxy_upper_cells),
        "lower_upper_measure_gap_ratio": gap,
        "replica_a_seed_lineage": [*seeds_a, spec.seed_a],
        "replica_b_seed_lineage": [*seeds_b, spec.seed_b],
        "frontier_round8_probe_count": len(frontier_frame),
        "frontier_round8_found_count": int(frontier_frame.get("found_valid_inverse", pd.Series(dtype=bool)).sum()),
    }
    _write_json(stage / "reach_round8_report.json", report)
    return _gate(
        stage / "gate.json",
        {"reach_converged": convergence},
        reach_source="round8",
        round8_executed=True,
        round9_forbidden=True,
        nonreach_checks=nonreach_checks,
        **report,
    )


def stage_formal_admission(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    pilot = _require(output_root, "stitched_atlas", config)
    dataset = _require(output_root, "fixed_budget_dataset", config)
    students = _require(output_root, "students", config)
    representation = _require(output_root, "representation_decision", config)
    reach = _require(output_root, "reach_update", config)
    formal = config["formal_gate"]
    checks = {
        "reach": bool(reach.get("gate_pass", False)),
        "labelable_measure": float(pilot.get("labelable_measure_ratio", 0.0)) >= float(formal["labelable_measure_min"]),
        "labelable_measure_lcb": float(
            pilot.get("labelable_measure_bootstrap_lcb95", 0.0)
        ) >= float(formal["labelable_measure_lcb_min"]),
        "unresolved_plus_abstention": float(pilot.get("unresolved_abstention_ratio", 1.0)) <= float(formal["unresolved_max"]),
        "measure_partition_closed": abs(float(pilot.get("measure_decomposition_sum", 0.0)) - 1.0) <= 1e-12,
        "student": bool(students.get("gate_pass", False)),
        "n_min_budget": int(dataset.get("supervision_row_count", 10**12)) <= int(formal["maximum_n_min"]),
        "no_padding": int(dataset.get("padding_count", 1)) == 0 and not bool(formal["row_padding"]),
        "representation_frozen": bool(representation.get("gate_pass", False)),
    }
    stage = output_root / STAGE_DIRS["formal_admission"]
    return _gate(stage / "gate.json", checks,
        formal_generation_authorized=bool(all(checks.values())),
        exact_formal_row_budget=int(formal["exact_rows"]),
        deployment_claim=False,
        formal_dataset_generated=False,
        direct_gate_relaxation_authorized=False,
        formal_measure_semantics="labelable>=0.80_equivalently_abstain_plus_unresolved<=0.20",
    )


def stage_summary(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    stage.mkdir(parents=True, exist_ok=True)
    gates = {name: _read_json(output_root / STAGE_DIRS[name] / "gate.json") for name in STAGE_ORDER[:-1] if (output_root / STAGE_DIRS[name] / "gate.json").is_file()}
    artifacts = [{"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(output_root.rglob("*")) if path.is_file() and not path.is_relative_to(stage)]
    _write_json(stage / "artifact_manifest.json", {"schema_version": 1, "artifact_count": len(artifacts), "artifacts": artifacts})
    report = {
        "stage_gates": gates,
        "pilot_gate": bool(gates.get("stitched_atlas", {}).get("gate_pass", False)),
        "student_gate": bool(gates.get("students", {}).get("gate_pass", False)),
        "selected_representation": gates.get("representation_decision", {}).get("selected_representation"),
        "formal_generation_authorized": bool(gates.get("formal_admission", {}).get("gate_pass", False)),
    }
    _write_json(stage / "summary_report.json", report)
    return _gate(stage / "gate.json", {"manifest_nonempty": bool(artifacts)}, **report)


STAGE_RUNNERS = {
    "inventory": stage_inventory,
    "pilot_registry": stage_pilot_registry,
    "root_charts": stage_root_charts,
    "stitched_atlas": stage_stitched_atlas,
    "fixed_budget_dataset": stage_fixed_budget_dataset,
    "students": stage_students,
    "representation_decision": stage_representation_decision,
    "reach_update": stage_reach_update,
    "formal_admission": stage_formal_admission,
    "summary": stage_summary,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SOURCE_ROOT / "configs/bacra_v14_3_repaired_5k_student.yaml"))
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--validate-stage", choices=STAGE_ORDER)
    parser.add_argument("--root-index", type=int)
    parser.add_argument("--dataset-shard", type=int)
    parser.add_argument("--dataset-wave", type=int)
    parser.add_argument("--audit-shard-bundle")
    parser.add_argument("--audit-shard-id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    project_root = project_root_from(SOURCE_ROOT)
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / str(config["output_root"])
    if args.validate_stage:
        result = _load_validated_stage_result(
            output_root, config=config, stage_name=args.validate_stage
        )
        return 0 if result is not None else 1
    if args.stage is None and not args.audit_shard_bundle:
        raise SystemExit("--stage is required unless --validate-stage or --audit-shard-bundle is used")
    if args.audit_shard_bundle:
        if args.audit_shard_id is None:
            raise SystemExit("--audit-shard-id is required with --audit-shard-bundle")
        result = _run_v143_audit_shard_worker(
            config,
            project_root,
            Path(args.audit_shard_bundle).resolve(),
            int(args.audit_shard_id),
        )
        print(json.dumps(v142r._strict(result), ensure_ascii=False, sort_keys=True))
        return 0
    config["_root_index"] = args.root_index
    config["_dataset_shard"] = args.dataset_shard
    config["_dataset_wave"] = args.dataset_wave
    output_root.mkdir(parents=True, exist_ok=True)
    assert args.stage is not None
    result = STAGE_RUNNERS[args.stage](config, project_root, output_root)
    if args.root_index is None and args.dataset_shard is None:
        _write_stage_completion_manifest(
            output_root, config=config, stage_name=args.stage
        )
    print(json.dumps(v142r._strict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
