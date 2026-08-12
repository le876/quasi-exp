#!/usr/bin/env python3
"""Run the BACRA V14.2R stitched-atlas repair and confirmation protocol."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, replace
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
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
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

from quasi_exp.teacher.canonical import beta_rms_deg
from quasi_exp.teacher.audit_shards import (
    add_schedule_waypoint_estimates,
    build_audit_shard_registry,
    build_shard_completion_report,
    merge_validated_audit_shards,
    validate_shard_completion,
)
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    ContinuationOutcome,
    make_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_continuation import (
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.section_atlas_repair import (
    AtlasRepairPolicy,
    AuditV2Policy,
    RetryTier,
    compare_stitched_primary_atlases,
    execute_audit_schedules,
    repair_rooted_section_atlas,
    section_growth_from_frames,
)
from quasi_exp.teacher.section_first_atlas import RootedSectionPolicy, build_section_first_atlas
from quasi_exp.teacher.workspace_atlas_repair import atlas_nodes_from_frames, make_segmented_continuation
from quasi_exp.teacher.workspace_candidate_bank import (
    CandidatePolicy,
    CandidateQuality,
    CandidateSearchMode,
    solve_candidate_bank,
)
from quasi_exp.teacher.workspace_experiment import ReachSamplingRoundSpec, generate_independent_reach_samples
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
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
REGISTERED_RETRY_CHAINS = {
    "R0": ("predictor", "bounded_ls"),
    "R1": ("weighted_dls", "bounded_ls"),
    "R2": ("weighted_dls", "bounded_ls", "slsqp"),
}
STAGE_DIRS = {
    "inventory": "00_inventory",
    "replacement_confirmation": "01_replacement_confirmation",
    "artifact_diagnostics": "02_artifact_diagnostics",
    "rooted_baseline": "03_rooted_baseline",
    "registered_retry": "04_registered_retry",
    "patch07_local_audit": "05_patch07_local_audit",
    "search_stability": "06_search_stability",
    "mechanism_gate": "07_mechanism_gate",
    "reach_round7": "08_reach_round7",
    "confirmation": "09_twelve_patch_confirmation",
    "meso_bridge": "10_meso_bridge",
    "summary": "11_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if ".worktrees" in resolved.parts:
        return Path(*resolved.parts[: resolved.parts.index(".worktrees")])
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported V14.2R config")
    config = dict(value)
    config["config_path"] = str(config_path)
    if config["parallel"] != {"patch_workers": 12, "numerical_threads_per_worker": 1}:
        raise ValueError("V14.2R requires exactly twelve single-threaded patch workers")
    audit_execution = config.get("audit_execution")
    if not isinstance(audit_execution, Mapping):
        raise ValueError("V14.2R requires an audit_execution contract")
    if int(audit_execution.get("total_shard_workers", 0)) != 12:
        raise ValueError("V14.2R requires exactly twelve total audit shard workers")
    if audit_execution.get("checkpoint_resume") is not True:
        raise ValueError("V14.2R audit shards must be checkpoint-resumable")
    if int(audit_execution.get("heartbeat_every_completed_shards", 0)) != 1:
        raise ValueError("V14.2R audit shards require one-shard heartbeat cadence")
    if int(audit_execution.get("progress_every_schedules", 0)) < 1:
        raise ValueError("V14.2R audit shard progress cadence must be positive")
    retry_chains = {
        str(tier["tier_id"]): tuple(map(str, tier["solver_chain"]))
        for tier in config["audit_v2"]["retry_tiers"]
    }
    if retry_chains != REGISTERED_RETRY_CHAINS:
        raise ValueError(
            "V14.2R retry solver chains must match the registered R0/R1/R2 contract"
        )
    if list(config["diagnostic_patch_ids"]) != ["patch_00", "patch_03", "patch_07", "patch_09"]:
        raise ValueError("V14.2R diagnostic patches are registered")
    if list(config["confirmation_patch_ids"]) != ["patch_08", "patch_10", "patch_11", "patch_12"]:
        raise ValueError("V14.2R fresh confirmation set must replace leaked patch_09 with patch_12")
    tiers = config["audit_v2"]["retry_tiers"]
    if [str(row["tier_id"]) for row in tiers] != ["R0", "R1", "R2"]:
        raise ValueError("V14.2R retry tier order is registered")
    if int(config["audit_v2"]["repeats_per_direction"]) != 3:
        raise ValueError("V14.2R requires three source-only repeats per direction")
    if int(config["mechanism_gate"]["minimum_passing_patches"]) != 3:
        raise ValueError("V14.2R four-patch Gate requires 3/4")
    if int(config["reach_round7"]["seed_a"]) == int(config["reach_round7"]["seed_b"]):
        raise ValueError("Reach Round 7 replicas require independent scramble seeds")
    return config


def _strict(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _strict(dict(value)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _gate(path: Path, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    payload = {
        "gate_pass": bool(checks and all(bool(value) for value in checks.values())),
        "checks": {str(key): bool(value) for key, value in checks.items()},
        **evidence,
    }
    _write_json(path, payload)
    return payload


def write_scientific_skip(stage: Path, reason: str) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_pass": False,
        "operational_completion": True,
        "scientific_skip": True,
        "skip_reason": str(reason),
    }
    _write_json(stage / "gate.json", payload)
    return payload


def _require(
    config: Mapping[str, Any], output_root: Path, stage_name: str
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    result = _load_validated_stage_result(
        stage, config=config, stage_name=stage_name
    )
    if result is None:
        raise FileNotFoundError(
            f"required upstream stage lacks a valid completion closure: {stage}"
        )
    return result


def _validate_resume_fixed_point(
    config: Mapping[str, Any], output_root: Path
) -> None:
    fixed_path = output_root / STAGE_DIRS["inventory"] / "source_fixed_point.json"
    if not fixed_path.is_file():
        raise FileNotFoundError(
            f"non-inventory stage requires a sealed fixed point: {fixed_path}"
        )
    fixed = _read_json(fixed_path)
    checks = {
        "source_sha": str(fixed.get("source_sha", "")) == _git_sha(),
        "config_sha256": str(fixed.get("config_sha256", ""))
        == sha256_file(Path(str(config["config_path"]))),
        "working_tree_clean": _tree_clean(),
        "runtime_sha256": str(fixed.get("runtime_sha256", ""))
        == _runtime_sha256(),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"V14.2R resume fixed-point validation failed: {checks}"
        )


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()


def _tree_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE_ROOT, text=True).strip()


def _runtime_closure() -> dict[str, Any]:
    runtime = dict(runtime_fingerprint())
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.__config__.show()
    return {
        "python": runtime.get("python"),
        "executable": runtime.get("executable"),
        "platform": runtime.get("platform"),
        "packages": runtime.get("packages", {}),
        "numpy_build_configuration": buffer.getvalue(),
    }


def _runtime_sha256() -> str:
    return _payload_sha256(_runtime_closure())


def _verify_upstream_artifact_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    payload = _read_json(manifest_path)
    records = tuple(payload.get("artifacts", ()))
    if int(payload.get("artifact_count", -1)) != len(records):
        raise RuntimeError(f"upstream artifact count mismatch: {manifest_path}")
    root = manifest_path.parent.parent
    verified: list[dict[str, Any]] = []
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe upstream artifact path: {relative}")
        path = root / relative
        expected_size = int(record.get("bytes", record.get("size_bytes", -1)))
        expected_sha = str(record.get("sha256", ""))
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise RuntimeError(f"upstream artifact closure mismatch: {path}")
        verified.append(
            {"path": relative.as_posix(), "bytes": expected_size, "sha256": expected_sha}
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_count": len(verified),
        "artifact_closure_sha256": _payload_sha256({"artifacts": verified}),
    }


def _source_paths(config: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    sources = config["sources"]
    return {
        "plan": SOURCE_ROOT / str(sources["reviewed_plan"]),
        "performance_protocol": SOURCE_ROOT / str(sources["performance_protocol"]),
        "scientific_closure_protocol": SOURCE_ROOT
        / str(sources["scientific_closure_protocol"]),
        "legacy_config": SOURCE_ROOT / str(sources["legacy_config"]),
        "retry4": project_root / str(sources["retry4_root"]),
        "v14_2": project_root / str(sources["v14_2_root"]),
        "original": project_root / str(sources["original_pilot_root"]),
        "robot_config": SOURCE_ROOT / str(config["robot_config"]),
    }


def _audit_policy(config: Mapping[str, Any]) -> AuditV2Policy:
    row = config["audit_v2"]
    return AuditV2Policy(
        geometry_p95_max_deg=float(row["geometry_p95_max_deg"]),
        geometry_max_deg=float(row["geometry_max_deg"]),
        repeat_p95_max_deg=float(row["repeat_p95_max_deg"]),
        continuation_residual_max_mm=float(row["continuation_residual_max_mm"]),
        repeats_per_direction=int(row["repeats_per_direction"]),
        repeat_perturbation_rad=float(row.get("repeat_perturbation_rad", 1.0e-8)),
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


def _repair_policy(config: Mapping[str, Any]) -> AtlasRepairPolicy:
    row = config["chart_repair"]
    return AtlasRepairPolicy(
        audit=_audit_policy(config),
        minimum_chart_cells=int(row["minimum_chart_cells"]),
        minimum_chart_fraction=float(row["minimum_chart_fraction"]),
        minimum_chart_spread_mm=float(row["minimum_chart_spread_mm"]),
        minimum_stitch_overlap_cells=int(row["minimum_stitch_overlap_cells"]),
        stitch_p95_max_deg=float(row["stitch_p95_max_deg"]),
        stitch_max_deg=float(row["stitch_max_deg"]),
        minimum_boundary_transition_edges=int(row["minimum_boundary_transition_edges"]),
        minimum_overlap_fraction=float(row["minimum_overlap_fraction"]),
        minimum_overlap_spread_mm=float(row["minimum_overlap_spread_mm"]),
        abstention_hops=int(row["abstention_hops"]),
        abstention_radius_mm=float(row["abstention_radius_mm"]),
    )


def _section_policy(
    config: Mapping[str, Any], *, beam_width: int, root_count: int, consensus: bool
) -> RootedSectionPolicy:
    row = config["rooted_section"]
    return RootedSectionPolicy(
        beam_width=int(beam_width),
        root_count=int(root_count),
        maximum_growth_waves=int(row["maximum_growth_waves"]),
        parent_consensus_gold_deg=float(row["parent_consensus_gold_deg"]),
        parent_consensus_silver_deg=float(row["parent_consensus_silver_deg"]),
        continuation_residual_max_mm=float(row["continuation_residual_max_mm"]),
        reverse_return_max_deg=float(row["reverse_return_max_deg"]),
        minimum_alternative_chart_cells=int(row["minimum_alternative_chart_cells"]),
        beta_weights=tuple(map(float, row["beta_weights"])),
        online_cycle_repair=bool(consensus),
    )


def stage_inventory(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["inventory"]
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2R inventory already sealed: {stage}")
    paths = _source_paths(config, project_root)
    required = [
        paths["plan"], paths["performance_protocol"],
        paths["scientific_closure_protocol"], paths["legacy_config"],
        paths["robot_config"],
        paths["retry4"] / "07_summary/artifact_manifest.json",
        paths["v14_2"] / "08_summary/artifact_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V14.2R source inventory incomplete: {missing}")
    upstream_closures = [
        _verify_upstream_artifact_manifest(
            paths["retry4"] / "07_summary/artifact_manifest.json"
        ),
        _verify_upstream_artifact_manifest(
            paths["v14_2"] / "08_summary/artifact_manifest.json"
        ),
    ]
    upstream_closure_sha256 = _payload_sha256(
        {"upstream_closures": upstream_closures}
    )
    _write_json(
        stage / "upstream_artifact_closure.json",
        {
            "schema_version": 1,
            "upstream_closure_sha256": upstream_closure_sha256,
            "manifests": upstream_closures,
        },
    )
    reviewed_files = list(_audit_source_records())
    reviewed_files.extend(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (
            Path(str(config["config_path"])),
            paths["plan"],
            paths["performance_protocol"],
            paths["scientific_closure_protocol"],
        )
    )
    reviewed_files = sorted(reviewed_files, key=lambda record: str(record["path"]))
    source_tree_mapping = {
        "schema_version": 1,
        "scientific_source_sha": _git_sha(),
        "release_sha": None,
        "release_mapping_status": "pending_filtered_publication",
        "reviewed_files": reviewed_files,
        "combined_tree_sha256": _payload_sha256(
            {"reviewed_files": reviewed_files}
        ),
    }
    _write_json(stage / "source_tree_mapping.json", source_tree_mapping)
    records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in required
    ]
    _write_parquet(pd.DataFrame.from_records(records), stage / "source_inventory.parquet")
    report = {
        "source_sha": _git_sha(),
        "working_tree_clean": _tree_clean(),
        "runtime": runtime_fingerprint(),
        "runtime_closure": _runtime_closure(),
        "runtime_sha256": _runtime_sha256(),
        "upstream_closure_sha256": upstream_closure_sha256,
        "reviewed_source_tree_sha256": source_tree_mapping[
            "combined_tree_sha256"
        ],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "patch_workers": int(config["parallel"]["patch_workers"]),
        "numerical_threads_per_worker": int(
            config["parallel"]["numerical_threads_per_worker"]
        ),
        "audit_shard_workers_total": int(
            config["audit_execution"]["total_shard_workers"]
        ),
        "audit_checkpoint_resume": bool(
            config["audit_execution"]["checkpoint_resume"]
        ),
        "source_records": records,
    }
    _write_json(stage / "source_fixed_point.json", report)
    return _gate(
        stage / "gate.json",
        {
            "clean_fixed_point": report["working_tree_clean"],
            "source_inventory_complete": not missing,
            "upstream_artifact_closure_complete": bool(upstream_closures),
            "source_tree_mapping_complete": bool(reviewed_files),
            "twelve_workers_registered": report["patch_workers"] == 12,
            "twelve_audit_shards_registered": report["audit_shard_workers_total"]
            == 12,
            "audit_checkpoint_resume_registered": report[
                "audit_checkpoint_resume"
            ],
        },
        **report,
    )


def stage_replacement_confirmation(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    """Seal an unused patch_12 before any mechanism result is produced."""

    _require(config, output_root, "inventory")
    stage = output_root / STAGE_DIRS["replacement_confirmation"]
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"replacement confirmation already sealed: {stage}")
    paths = _source_paths(config, project_root)
    legacy_config = legacy.load_config(paths["legacy_config"])
    legacy_paths = legacy._paths(legacy_config, project_root)
    parents = pd.read_parquet(legacy_paths["original_parent_nodes"])
    parent_edges = pd.read_parquet(paths["original"] / "02_domain_registry/pilot_task_edges.parquet")
    tasks = pd.read_parquet(legacy_paths["original_atlas_tasks"])
    task_edges = pd.read_parquet(legacy_paths["original_atlas_edges"])
    candidates = pd.read_parquet(legacy_paths["original_candidate_clusters"])
    atlas_candidates = pd.read_parquet(paths["original"] / "04_workspace_atlas/atlas_candidates.parquet")
    product_edges = pd.read_parquet(paths["original"] / "04_workspace_atlas/product_edges.parquet")
    historical = pd.read_parquet(legacy_paths["assignments"])
    used_parent_ids = set(historical["parent_node_id"].astype(int))

    representative = tasks[tasks["is_representative"].astype(bool)][
        ["task_node_id", "source_parent_node_id"]
    ]
    exact = atlas_candidates[atlas_candidates["candidate_id"].eq("capability_exact")][
        ["task_node_id", "condition_number"]
    ]
    condition = representative.merge(exact, on="task_node_id", validate="one_to_one").set_index(
        "source_parent_node_id"
    )["condition_number"]
    parent_by_task = tasks.set_index("task_node_id")["source_parent_node_id"].astype(int).to_dict()
    existing_cross: set[int] = set()
    for row in product_edges.itertuples(index=False):
        left = parent_by_task[int(row.left_task_node_id)]
        right = parent_by_task[int(row.right_task_node_id)]
        if left != right:
            existing_cross.update((left, right))

    parent_index = parents.set_index("node_id")
    x_tertile = pd.qcut(parents["x_m"], q=3, labels=False, duplicates="drop")
    x_tertile_by_node = dict(zip(parents["node_id"].astype(int), x_tertile.astype(int), strict=True))
    condition_rows = condition.reindex(parents["node_id"].astype(int))
    if condition_rows.isna().any():
        raise RuntimeError("replacement confirmation condition evidence is incomplete")
    condition_tertile = pd.qcut(condition_rows, q=3, labels=False, duplicates="drop")
    condition_tertile_by_node = dict(
        zip(parents["node_id"].astype(int), condition_tertile.astype(int), strict=True)
    )
    adjacency: dict[int, set[int]] = {int(node): set() for node in parents["node_id"]}
    for row in parent_edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        adjacency[left].add(right); adjacency[right].add(left)
    unused = set(adjacency) - used_parent_ids
    seed_candidates = sorted(
        node for node in unused
        if x_tertile_by_node[node] == 1
        and condition_tertile_by_node[node] == 0
        and node not in existing_cross
    )
    selected: list[int] | None = None
    seed_node: int | None = None
    for candidate_seed in seed_candidates:
        queue = [candidate_seed]
        visited: list[int] = []
        seen = {candidate_seed}
        while queue and len(visited) < 64:
            current = queue.pop(0)
            if current not in unused:
                continue
            visited.append(current)
            center = parent_index.loc[candidate_seed, ["x_m", "y_m", "z_m"]].to_numpy(float)
            neighbours = sorted(
                adjacency[current] - seen,
                key=lambda node: (
                    float(np.linalg.norm(parent_index.loc[node, ["x_m", "y_m", "z_m"]].to_numpy(float) - center)),
                    node,
                ),
            )
            seen.update(neighbours)
            queue.extend(neighbours)
        if len(visited) == 64:
            selected, seed_node = visited, candidate_seed
            break
    if selected is None or seed_node is None:
        raise RuntimeError("unable to construct an unused connected 64-cell patch_12 in the registered stratum")
    selected_set = set(selected)
    assignment_rows = []
    for node in selected:
        parent = parent_index.loc[node]
        assignment_rows.append(
            {
                "patch_id": "patch_12", "patch_index": 12, "patch_split": "confirmation",
                "parent_node_id": node, "seed_node_id": seed_node, "is_seed": node == seed_node,
                "seed_role": "replacement_no_existing_cross_edge",
                "x_tertile": x_tertile_by_node[node],
                "condition_tertile": condition_tertile_by_node[node],
                "boundary": str(parent.strata) == "boundary",
                "retention": str(parent.strata) == "retention",
                "tip": str(parent.strata) == "tip",
                "existing_cross_edge": node in existing_cross,
                "no_existing_cross_edge": node not in existing_cross,
            }
        )
    assignments = pd.DataFrame.from_records(assignment_rows)
    selected_tasks = tasks[tasks["source_parent_node_id"].astype(int).isin(selected_set)].copy()
    selected_tasks["patch_id"] = "patch_12"
    selected_task_ids = set(selected_tasks["task_node_id"].astype(int))
    selected_edges = task_edges[
        task_edges["left_node_id"].astype(int).isin(selected_task_ids)
        & task_edges["right_node_id"].astype(int).isin(selected_task_ids)
    ].copy()
    selected_edges["patch_id"] = "patch_12"
    selected_candidates = candidates[candidates["task_node_id"].astype(int).isin(selected_task_ids)].copy()
    selected_candidates["patch_id"] = "patch_12"
    outputs = {
        "patch12_assignments.parquet": assignments,
        "patch12_task_nodes.parquet": selected_tasks,
        "patch12_task_edges.parquet": selected_edges,
        "patch12_candidate_clusters.parquet": selected_candidates,
    }
    for name, frame in outputs.items():
        _write_parquet(frame, stage / name)
    artifacts = [
        {"path": name, "bytes": (stage / name).stat().st_size, "sha256": sha256_file(stage / name)}
        for name in outputs
    ]
    _write_json(stage / "patch12_manifest.json", {
        "schema_version": 1,
        "source_sha": _git_sha(),
        "selection_before_scientific_results": True,
        "historical_parent_overlap_count": len(selected_set & used_parent_ids),
        "seed_stratum": {"x_tertile": 1, "condition_tertile": 0, "no_existing_cross_edge": True},
        "artifacts": artifacts,
    })
    return _gate(stage / "gate.json", {
        "exact_64_cells": len(assignments) == 64,
        "exact_320_task_probes": len(selected_tasks) == 320,
        "no_historical_overlap": not (selected_set & used_parent_ids),
        "seed_stratum_match": x_tertile_by_node[seed_node] == 1 and condition_tertile_by_node[seed_node] == 0 and seed_node not in existing_cross,
    }, patch_id="patch_12", seed_node_id=seed_node, artifacts=artifacts,
       assignment_sha256=sha256_file(stage / "patch12_assignments.parquet"))


def stage_artifact_diagnostics(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require(config, output_root, "replacement_confirmation")
    stage = output_root / STAGE_DIRS["artifact_diagnostics"]
    paths = _source_paths(config, project_root)
    rows: list[dict[str, Any]] = []
    chart_support_rows: list[dict[str, Any]] = []
    primary_usage_rows: list[dict[str, Any]] = []
    cross_edge_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    pair_gap_rows: list[dict[str, Any]] = []
    for patch_id in config["diagnostic_patch_ids"]:
        directory = paths["v14_2"] / "02_mechanism_experiment" / str(patch_id) / "S4"
        charts = pd.read_parquet(directory / "section_charts.parquet")
        hypotheses = pd.read_parquet(directory / "section_hypotheses.parquet")
        selected = hypotheses[hypotheses["selected"].astype(bool)].copy()
        primary = pd.read_parquet(directory / "primary_section.parquet")
        domain_nodes = set(primary["task_node_id"].astype(int))
        primary_map = primary.loc[~primary["abstained"].astype(bool)].set_index(
            "task_node_id"
        )["primary_chart_id"].astype(str).to_dict()
        task_edges = pd.read_parquet(
            paths["retry4"]
            / "01_patch_ablations"
            / str(patch_id)
            / "E3_dynamic_insertion_task_edges.parquet"
        )
        for row in charts.itertuples(index=False):
            support = selected[selected["chart_id"].astype(str).eq(str(row.chart_id))]
            chart_support_rows.append(
                {
                    "patch_id": str(patch_id),
                    "chart_id": str(row.chart_id),
                    "selected_node_count": int(support["task_node_id"].nunique()),
                    "domain_node_count": len(domain_nodes),
                    "coverage_ratio": int(support["task_node_id"].nunique()) / max(1, len(domain_nodes)),
                    "full_domain_chart": set(support["task_node_id"].astype(int)) == domain_nodes,
                }
            )
        for chart_id, count in pd.Series(primary_map).value_counts().items():
            primary_usage_rows.append(
                {
                    "patch_id": str(patch_id),
                    "chart_id": str(chart_id),
                    "primary_node_count": int(count),
                    "primary_measure_ratio": int(count) / max(1, len(domain_nodes)),
                }
            )
        for row in task_edges.itertuples(index=False):
            left, right = int(row.left_node_id), int(row.right_node_id)
            left_chart, right_chart = primary_map.get(left), primary_map.get(right)
            if left_chart is None or right_chart is None:
                continue
            cross_edge_rows.append(
                {
                    "patch_id": str(patch_id),
                    "left_node_id": left,
                    "right_node_id": right,
                    "left_chart_id": left_chart,
                    "right_chart_id": right_chart,
                    "cross_chart": left_chart != right_chart,
                }
            )
        multiplicity = hypotheses.groupby(["chart_id", "task_node_id"], as_index=False).size()
        for row in multiplicity.itertuples(index=False):
            hypothesis_rows.append(
                {
                    "patch_id": str(patch_id),
                    "chart_id": str(row.chart_id),
                    "task_node_id": int(row.task_node_id),
                    "hypothesis_count": int(row.size),
                }
            )
        selected_by_chart = {
            str(chart_id): group.set_index("task_node_id")
            for chart_id, group in selected.groupby("chart_id", sort=True)
        }
        chart_ids = sorted(selected_by_chart)
        for left_index, left_chart in enumerate(chart_ids):
            for right_chart in chart_ids[left_index + 1 :]:
                left = selected_by_chart[left_chart]
                right = selected_by_chart[right_chart]
                overlap = sorted(set(left.index.astype(int)) & set(right.index.astype(int)))
                for node_id in overlap:
                    pair_gap_rows.append(
                        {
                            "patch_id": str(patch_id),
                            "left_chart_id": left_chart,
                            "right_chart_id": right_chart,
                            "task_node_id": int(node_id),
                            "beta_gap_deg": beta_rms_deg(
                                left.loc[node_id, list(BETA_COLUMNS)].to_numpy(float),
                                right.loc[node_id, list(BETA_COLUMNS)].to_numpy(float),
                            ),
                        }
                    )
        for filename, kind in (
            ("fresh_edge_audit.parquet", "edge"),
            ("fresh_path_audit.parquet", "path"),
            ("fresh_cycle_audit.parquet", "cycle"),
            ("fresh_multipath_audit.parquet", "multipath"),
            ("fresh_repeat_direction_audit.parquet", "repeat"),
        ):
            frame = pd.read_parquet(directory / filename)
            for row in frame.itertuples(index=False):
                schedule_id = str(row.schedule_id)
                source, target = int(row.source_node), int(row.target_node)
                if kind in {"edge", "repeat"}:
                    # selected_edge and edge_repeat are executions of the same
                    # unordered physical edge, not independent entities.
                    unique = f"{patch_id}:{row.chart_id}:edge:{min(source, target)}:{max(source, target)}"
                elif kind in {"path", "multipath"}:
                    unique = (
                        f"{patch_id}:{row.chart_id}:{kind}:"
                        f"{min(source, target)}:{max(source, target)}:{int(row.edge_or_cycle_length)}"
                    )
                else:
                    # V14.2 did not persist full cycle paths.  Keep its stable
                    # ordinal instead of fabricating an unordered signature.
                    unique = f"{patch_id}:{row.chart_id}:cycle:{schedule_id}"
                rows.append(
                    {
                        "patch_id": str(patch_id),
                        "chart_id": str(row.chart_id),
                        "audit_kind": kind,
                        "legacy_schedule_id": schedule_id,
                        "unique_entity_id": unique,
                        "passed": bool(row.passed),
                        "failure_reason": str(row.failure_reason),
                        "geometry_observed": bool(math.isfinite(float(row.path_endpoint_gap))) if row.path_endpoint_gap is not None else False,
                    }
                )
    frame = pd.DataFrame.from_records(rows)
    grouped = frame.groupby("unique_entity_id", as_index=False).agg(
        execution_count=("passed", "size"),
        any_pass=("passed", "any"),
        all_pass=("passed", "all"),
        geometry_observed=("geometry_observed", "any"),
    )
    _write_parquet(frame, stage / "legacy_audit_executions.parquet")
    _write_parquet(grouped, stage / "deduplicated_schedule_inventory.parquet")
    diagnostic_frames = {
        "chart_support.parquet": pd.DataFrame.from_records(chart_support_rows),
        "primary_chart_usage.parquet": pd.DataFrame.from_records(primary_usage_rows),
        "primary_cross_chart_edges.parquet": pd.DataFrame.from_records(cross_edge_rows),
        "hypothesis_multiplicity.parquet": pd.DataFrame.from_records(hypothesis_rows),
        "root_chart_pair_gaps.parquet": pd.DataFrame.from_records(pair_gap_rows),
    }
    for filename, diagnostic_frame in diagnostic_frames.items():
        _write_parquet(diagnostic_frame, stage / filename)
    cross = diagnostic_frames["primary_cross_chart_edges.parquet"]
    support = diagnostic_frames["chart_support.parquet"]
    multiplicity = diagnostic_frames["hypothesis_multiplicity.parquet"]
    pair_gaps = diagnostic_frames["root_chart_pair_gaps.parquet"]
    summary = {
        "schema_version": 1,
        "source_sha": _git_sha(),
        "cross_chart_edge_ratio_by_patch": {
            str(key): float(value)
            for key, value in cross.groupby("patch_id")["cross_chart"].mean().items()
        },
        "full_domain_chart_count_by_patch": {
            str(key): int(value)
            for key, value in support.groupby("patch_id")["full_domain_chart"].sum().items()
        },
        "multi_hypothesis_entity_ratio": float((multiplicity["hypothesis_count"] > 1).mean()),
        "root_chart_pair_gap_p95_deg": float(np.percentile(pair_gaps["beta_gap_deg"], 95)),
        "root_chart_pair_gap_max_deg": float(pair_gaps["beta_gap_deg"].max()),
        "interpretation_status": "sealed_provisional_facts_before_method_freeze",
    }
    _write_json(stage / "artifact_diagnostic_summary.json", summary)
    generated = [
        stage / "legacy_audit_executions.parquet",
        stage / "deduplicated_schedule_inventory.parquet",
        *(stage / filename for filename in diagnostic_frames),
        stage / "artifact_diagnostic_summary.json",
    ]
    _write_json(
        stage / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifacts": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in generated
            ],
        },
    )
    return _gate(
        stage / "gate.json",
        {
            "four_patch_artifacts_read": frame["patch_id"].nunique() == 4,
            "deduplication_complete": len(grouped) > 0,
            "entity_diagnostics_sealed": all(len(item) > 0 for item in diagnostic_frames.values()),
            "at_least_one_full_domain_chart_observed": bool(support["full_domain_chart"].any()),
        },
        raw_execution_count=len(frame),
        unique_entity_count=len(grouped),
        legacy_missing_is_not_geometry_failure=True,
        diagnostic_summary=summary,
    )


def _root_candidates_with_enrichment(
    environment: Any,
    tasks: pd.DataFrame,
    candidates: tuple[AtlasCandidate, ...],
    root_keys: Sequence[tuple[int, str]],
    desired: int,
    seed: int,
) -> tuple[tuple[AtlasCandidate, ...], tuple[tuple[int, str], ...], dict[str, Any]]:
    root_node = int(root_keys[0][0])
    roots = [candidate for candidate in candidates if candidate.node_id == root_node and candidate.is_strict_feasible]
    roots.sort(key=lambda item: (0 if item.is_gold else 1, item.posture_cost, -item.min_margin_deg, item.candidate_id))
    before = len(roots)
    if len(roots) < desired:
        target_row = tasks.loc[tasks["task_node_id"].astype(int).eq(root_node)].iloc[0]
        task_xyz = tasks.set_index("task_node_id")[["x_m", "y_m", "z_m"]]
        capability = [candidate for candidate in candidates if candidate.is_strict_feasible]
        capability_beta = np.vstack([item.beta_rad for item in capability])
        capability_xyz = np.vstack([task_xyz.loc[item.node_id].to_numpy(float) for item in capability])
        bank = solve_candidate_bank(
            environment,
            np.asarray([[target_row.x_m, target_row.y_m, target_row.z_m]], dtype=float),
            CandidatePolicy(
                capability_nearest_count=min(128, len(capability)),
                capability_representative_count=32,
                candidate_budget_per_node=64,
                difficult_candidate_budget_per_node=64,
                candidate_seed_budget_per_node=24,
                difficult_seed_budget_per_node=24,
                nullspace_seed_budget_per_node=16,
                search_mode=CandidateSearchMode.DIVERSITY,
                solver_names=("weighted_dls", "bounded_least_squares", "slsqp"),
                gold_candidate_target=max(8, desired),
                max_corrector_iterations=400,
                solver_seed=int(seed),
            ),
            capability_beta_rad=capability_beta,
            capability_xyz_m=capability_xyz,
            node_seed_beta_rad={0: roots[0].beta_rad},
            difficult_node_ids=(0,),
        )
        for record in bank.records:
            if record.quality is CandidateQuality.REJECT:
                continue
            candidate = AtlasCandidate(
                node_id=root_node,
                candidate_id=f"enriched_{seed}_{record.candidate_id}",
                beta_rad=record.beta_rad,
                residual_mm=record.residual_mm,
                min_margin_deg=record.min_margin_deg,
                normalized_min_margin=record.normalized_min_margin,
                posture_cost=float(np.linalg.norm(record.beta_rad)),
                condition_number=0.0,
                quality=record.quality.value,
                solver_success=record.solver_success,
                actual_bounds=True,
                diagnostics={"root_diversity_enrichment": True, "solver": record.solver},
            )
            if all(beta_rms_deg(candidate.beta_rad, existing.beta_rad) > 0.5 for existing in roots):
                roots.append(candidate)
    ordered = list(roots)
    random.Random(int(seed)).shuffle(ordered)
    selected = tuple(ordered[:desired])
    merged = tuple(candidates) + tuple(item for item in roots if item.key not in {row.key for row in candidates})
    return merged, tuple(item.key for item in selected), {
        "root_candidate_count_before_enrichment": before,
        "root_candidate_count_after_enrichment": len(roots),
        "requested_root_count": int(desired),
        "selected_root_count": len(selected),
        "root_budget_saturated": len(selected) >= desired,
    }


def _registered_retry_adapter(environment: Any, tasks: pd.DataFrame, task_edges: pd.DataFrame):
    nodes = atlas_nodes_from_frames(tasks, task_edges)
    node_by_id = {node.node_id: node for node in nodes}
    cell_by_node = {
        int(row.task_node_id): (int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in tasks.itertuples(index=False)
    }
    cache: dict[str, Any] = {}

    def adapter(source: AtlasCandidate, target: AtlasTaskNode, tier: RetryTier) -> ContinuationOutcome:
        expected_chain = REGISTERED_RETRY_CHAINS.get(str(tier.tier_id))
        if expected_chain is None or tuple(tier.solver_chain) != expected_chain:
            raise ValueError(
                f"unregistered retry solver chain: {tier.tier_id}={tier.solver_chain}"
            )
        if tier.tier_id not in cache:
            continuation_factory = (
                make_optimized_predictor_corrector_continuation
                if hasattr(environment, "fk_and_jacobian")
                else make_predictor_corrector_continuation
            )
            raw_base = continuation_factory(
                environment,
                damping={"R0": 1e-3, "R1": 2e-3, "R2": 5e-3}[tier.tier_id],
                max_corrector_iterations=tier.maximum_iterations,
                residual_tolerance_mm=3.0,
            )
            bounded_chain = expected_chain[:2]

            def base(
                local_source: AtlasCandidate, local_target: AtlasTaskNode
            ) -> ContinuationOutcome:
                outcome = raw_base(local_source, local_target)
                return replace(
                    outcome,
                    status=(
                        f"audit_solver_chain[{'>'.join(bounded_chain)}]::"
                        f"{outcome.status}"
                    ),
                )
            if "slsqp" in tier.solver_chain:
                bounded = base

                def robust(local_source: AtlasCandidate, local_target: AtlasTaskNode) -> ContinuationOutcome:
                    first = bounded(local_source, local_target)
                    if first.success:
                        return first
                    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
                    target_xyz = local_target.xyz_m.copy()

                    def objective(beta: np.ndarray) -> float:
                        delta = (np.asarray(environment.fk(np.asarray(beta).reshape(1, 6))).reshape(3) - target_xyz) / 0.001
                        return float(delta @ delta)

                    result = minimize(
                        objective,
                        local_source.beta_rad,
                        method="SLSQP",
                        bounds=[tuple(map(float, row)) for row in bounds],
                        options={"maxiter": tier.maximum_iterations, "ftol": 1e-12, "disp": False},
                    )
                    beta = np.asarray(result.x, dtype=float).reshape(6)
                    residual = math.sqrt(max(0.0, float(result.fun)))
                    inside = bool(np.all(beta >= bounds[:, 0]) and np.all(beta <= bounds[:, 1]))
                    return ContinuationOutcome(
                        beta, residual, bool(result.success and residual <= 3.0 and inside), inside,
                        int(getattr(result, "nit", 0)),
                        (
                            f"audit_solver_chain[{'>'.join(expected_chain)}]::"
                            f"slsqp:{result.status}"
                        ),
                    )

                base = robust
            cache[tier.tier_id] = make_segmented_continuation(
                base, node_by_id, cell_by_node, step_max_mm=tier.maximum_step_mm
            )
        return cache[tier.tier_id](source, target)

    return adapter


AUDIT_EXECUTION_COLUMNS = (
    "schedule_id",
    "chart_id",
    "direction",
    "repeat_index",
    "repeat_perturbation_l2_rad",
    "solver_success",
    "geometry_gap_deg",
    "repeat_gap_deg",
    "residual_mm",
    "retry_tier",
    "registered_solver_chain",
    "executed_solver_chain",
    "solver_chain_sha256",
    "classification",
    "failure_source_node",
    "failure_target_node",
    "oracle_used_for_pass",
)


class _AuditWorkerToken:
    """One process slot held through an advisory file lock."""

    def __init__(self, handle: Any, slot_id: int) -> None:
        self._handle = handle
        self.slot_id = int(slot_id)
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._released = True


class _AuditWorkerTokenPool:
    """Cross-process, work-conserving limit for numerical audit workers."""

    def __init__(self, directory: Path, *, capacity: int) -> None:
        if int(capacity) < 1:
            raise ValueError("audit token capacity must be positive")
        self.directory = Path(directory)
        self.capacity = int(capacity)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._paths = tuple(
            self.directory / f"slot_{slot_id:02d}.lock"
            for slot_id in range(self.capacity)
        )
        for path in self._paths:
            path.open("a", encoding="utf-8").close()

    def try_acquire(self) -> _AuditWorkerToken | None:
        for slot_id, path in enumerate(self._paths):
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            return _AuditWorkerToken(handle, slot_id)
        return None


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _strict(dict(value)), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_payload_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].map(
                lambda value: json.dumps(
                    _strict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
    row_hashes = pd.util.hash_pandas_object(normalized, index=True).to_numpy(
        dtype=np.uint64
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(map(str, normalized.columns)), separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _thread_limited_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/mpl-bacra-v14-2r",
        }
    )
    return environment


def _audit_bundle_artifacts(phase_directory: Path) -> tuple[dict[str, Any], ...]:
    names = (
        "task_nodes.parquet",
        "task_edges.parquet",
        "section_hypotheses.parquet",
        "selected_edges.parquet",
        "schedule_registry.parquet",
    )
    return tuple(
        {
            "path": name,
            "bytes": (phase_directory / name).stat().st_size,
            "sha256": sha256_file(phase_directory / name),
        }
        for name in names
    )


def _audit_source_records(
    extra_paths: Sequence[Path] = (),
) -> tuple[dict[str, Any], ...]:
    paths = (
        Path(__file__).resolve(),
        SOURCE_ROOT / "src/quasi_exp/model/endpoint_kinematics.py",
        SOURCE_ROOT / "src/quasi_exp/teacher/audit_shards.py",
        SOURCE_ROOT / "src/quasi_exp/teacher/optimized_continuation.py",
        SOURCE_ROOT / "src/quasi_exp/teacher/optimized_forward.py",
        SOURCE_ROOT / "src/quasi_exp/teacher/section_atlas_repair.py",
        SOURCE_ROOT / "src/quasi_exp/teacher/section_first_atlas.py",
        *tuple(Path(path).resolve() for path in extra_paths),
    )
    return tuple(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    )


def _validate_audit_bundle(
    manifest: Mapping[str, Any], phase_directory: Path, config: Mapping[str, Any]
) -> None:
    if str(manifest.get("source_sha", "")) != _git_sha():
        raise RuntimeError("audit bundle source SHA does not match the executing fixed point")
    config_sha = sha256_file(Path(str(config["config_path"])))
    if str(manifest.get("config_sha256", "")) != config_sha:
        raise RuntimeError("audit bundle config hash does not match the executing config")
    if str(manifest.get("runtime_sha256", "")) != _runtime_sha256():
        raise RuntimeError("audit bundle runtime hash does not match the executing runtime")
    artifacts = tuple(manifest.get("artifacts", ()))
    for record in artifacts:
        path = phase_directory / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"audit bundle artifact is absent or truncated: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"audit bundle artifact hash mismatch: {path}")
    source_records = tuple(manifest.get("source_records", ()))
    for record in source_records:
        path = Path(str(record["path"]))
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"audit source artifact is absent or truncated: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"audit source artifact hash mismatch: {path}")
    expected_input_sha = _payload_sha256(
        {
            "artifacts": artifacts,
            "source_records": source_records,
            "runtime_sha256": str(manifest.get("runtime_sha256", "")),
        }
    )
    if str(manifest.get("input_sha256", "")) != expected_input_sha:
        raise RuntimeError("audit bundle input closure hash mismatch")


def _run_audit_shard_worker(
    config: Mapping[str, Any], project_root: Path, phase_directory: Path, shard_id: int
) -> dict[str, Any]:
    worker_started_at = time.time()
    worker_cpu_started = time.process_time()
    manifest = _read_json(phase_directory / "input_manifest.json")
    _validate_audit_bundle(manifest, phase_directory, config)
    registry = pd.read_parquet(phase_directory / "schedule_registry.parquet")
    shard_registry = registry[registry["shard_id"].astype(int).eq(int(shard_id))].copy()
    registered_count = int(manifest["shard_count"])
    if not 0 <= int(shard_id) < registered_count:
        raise ValueError(f"audit shard id outside registered range: {shard_id}")
    tasks = pd.read_parquet(phase_directory / "task_nodes.parquet")
    task_edges = pd.read_parquet(phase_directory / "task_edges.parquet")
    hypotheses = pd.read_parquet(phase_directory / "section_hypotheses.parquet")
    selected_edges = pd.read_parquet(phase_directory / "selected_edges.parquet")
    nodes = atlas_nodes_from_frames(tasks, task_edges)
    growth = section_growth_from_frames(nodes, hypotheses, selected_edges)
    environment = legacy._EndpointOnlyForwardAdapter(
        load_environment(project_root, _source_paths(config, project_root)["robot_config"])
    )
    _nodes, continuation = legacy._segmented_continuation(
        environment, tasks, task_edges
    )
    retry = _registered_retry_adapter(environment, tasks, task_edges)
    shard_directory = phase_directory / f"shard_{int(shard_id):02d}"
    shard_directory.mkdir(parents=True, exist_ok=True)
    progress_path = shard_directory / "progress.json"
    progress_every = int(config["audit_execution"]["progress_every_schedules"])
    last_reported = -progress_every

    def report_progress(completed_schedule_count: int, total_schedule_count: int) -> None:
        nonlocal last_reported
        completed_schedule_count = int(completed_schedule_count)
        total_schedule_count = int(total_schedule_count)
        if (
            completed_schedule_count != total_schedule_count
            and completed_schedule_count - last_reported < progress_every
        ):
            return
        _write_json(
            progress_path,
            {
                "schema_version": 1,
                "status": "running",
                "phase_id": str(manifest["phase_id"]),
                "shard_id": int(shard_id),
                "completed_schedule_count": completed_schedule_count,
                "total_schedule_count": total_schedule_count,
                "started_at_unix_s": worker_started_at,
                "updated_at_unix_s": time.time(),
            },
        )
        last_reported = completed_schedule_count

    report_progress(0, len(shard_registry))
    if len(shard_registry):
        executions = execute_audit_schedules(
            growth,
            shard_registry,
            continuation,
            _audit_policy(config),
            retry_continuation=retry,
            progress_callback=report_progress,
        )
    else:
        executions = pd.DataFrame(columns=AUDIT_EXECUTION_COLUMNS)
    _write_parquet(executions, shard_directory / "executions.parquet")
    report = build_shard_completion_report(
        shard_registry,
        executions,
        source_sha=_git_sha(),
        config_sha256=sha256_file(Path(str(config["config_path"]))),
        input_sha256=str(manifest["input_sha256"]),
        phase_id=str(manifest["phase_id"]),
        shard_id=int(shard_id),
        repeats_per_direction=int(config["audit_v2"]["repeats_per_direction"]),
    )
    report["pid"] = os.getpid()
    report["executions_file_sha256"] = sha256_file(
        shard_directory / "executions.parquet"
    )
    report["started_at_unix_s"] = worker_started_at
    report["finished_at_unix_s"] = time.time()
    report["wall_time_s"] = report["finished_at_unix_s"] - worker_started_at
    report["cpu_time_s"] = time.process_time() - worker_cpu_started
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
            "started_at_unix_s": worker_started_at,
            "finished_at_unix_s": time.time(),
            "wall_time_s": time.time() - worker_started_at,
            "cpu_time_s": time.process_time() - worker_cpu_started,
            "updated_at_unix_s": time.time(),
        },
    )
    return report


class _SubprocessAuditExecutor:
    """Run exact audit schedules in independent, resumable subprocess shards."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        project_root: Path,
        tasks: pd.DataFrame,
        task_edges: pd.DataFrame,
        patch_directory: Path,
        shard_count: int,
        token_pool: _AuditWorkerTokenPool | None = None,
        maximum_concurrent_workers: int | None = None,
        assignment_strategy: str = "hash",
        worker_runner_path: Path | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.tasks = tasks.copy()
        self.task_edges = task_edges.copy()
        self.patch_directory = patch_directory
        self.shard_count = max(1, int(shard_count))
        self.token_pool = token_pool
        self.maximum_concurrent_workers = max(
            1,
            min(
                self.shard_count,
                int(maximum_concurrent_workers or self.shard_count),
            ),
        )
        self.assignment_strategy = str(assignment_strategy)
        self.worker_runner_path = (
            Path(__file__).resolve()
            if worker_runner_path is None
            else Path(worker_runner_path).resolve()
        )

    def __call__(
        self,
        growth: Any,
        schedules: pd.DataFrame,
        continuation: Any,
        policy: AuditV2Policy,
        retry_continuation: Any,
        phase_id: str,
    ) -> pd.DataFrame:
        del continuation
        if retry_continuation is None:
            raise ValueError("subprocess audit execution requires the registered retry adapter")
        phase_directory = self.patch_directory / "_audit_checkpoints" / str(phase_id)
        phase_directory.mkdir(parents=True, exist_ok=True)
        frames = growth.frames()
        registered_schedules = schedules
        if self.assignment_strategy == "cost_balanced_lpt":
            registered_schedules = add_schedule_waypoint_estimates(
                schedules, self.tasks, maximum_step_mm=5.0
            )
        registry = build_audit_shard_registry(
            registered_schedules,
            shard_count=self.shard_count,
            repeats_per_direction=policy.repeats_per_direction,
            assignment_strategy=self.assignment_strategy,
        )
        _write_parquet(self.tasks, phase_directory / "task_nodes.parquet")
        _write_parquet(self.task_edges, phase_directory / "task_edges.parquet")
        _write_parquet(
            frames["section_hypotheses"], phase_directory / "section_hypotheses.parquet"
        )
        _write_parquet(frames["selected_edges"], phase_directory / "selected_edges.parquet")
        _write_parquet(registry, phase_directory / "schedule_registry.parquet")
        artifacts = _audit_bundle_artifacts(phase_directory)
        source_records = _audit_source_records((self.worker_runner_path,))
        input_sha = _payload_sha256(
            {
                "artifacts": artifacts,
                "source_records": source_records,
                "runtime_sha256": _runtime_sha256(),
            }
        )
        manifest = {
            "schema_version": 1,
            "source_sha": _git_sha(),
            "config_sha256": sha256_file(Path(str(self.config["config_path"]))),
            "runtime_sha256": _runtime_sha256(),
            "input_sha256": input_sha,
            "phase_id": str(phase_id),
            "shard_count": self.shard_count,
            "schedule_count": len(registry),
            "repeats_per_direction": policy.repeats_per_direction,
            "audit_policy": {
                "geometry_p95_max_deg": policy.geometry_p95_max_deg,
                "geometry_max_deg": policy.geometry_max_deg,
                "repeat_p95_max_deg": policy.repeat_p95_max_deg,
                "continuation_residual_max_mm": policy.continuation_residual_max_mm,
                "repeats_per_direction": policy.repeats_per_direction,
                "repeat_perturbation_rad": policy.repeat_perturbation_rad,
                "retry_tiers": [
                    {
                        "tier_id": tier.tier_id,
                        "maximum_step_mm": tier.maximum_step_mm,
                        "maximum_iterations": tier.maximum_iterations,
                        "solver_chain": list(tier.solver_chain),
                    }
                    for tier in policy.retry_tiers
                ],
            },
            "assignment_strategy": self.assignment_strategy,
            "maximum_concurrent_workers": self.maximum_concurrent_workers,
            "artifacts": artifacts,
            "source_records": source_records,
        }
        _write_json(phase_directory / "input_manifest.json", manifest)

        completed: dict[int, pd.DataFrame] = {}
        missing: list[int] = []
        for shard_id in range(self.shard_count):
            shard_directory = phase_directory / f"shard_{shard_id:02d}"
            report_path = shard_directory / "report.json"
            execution_path = shard_directory / "executions.parquet"
            if report_path.is_file() and execution_path.is_file():
                try:
                    executions = pd.read_parquet(execution_path)
                    report = _read_json(report_path)
                    shard_registry = registry[registry["shard_id"].eq(shard_id)]
                    valid = validate_shard_completion(
                        report,
                        shard_registry,
                        executions,
                        source_sha=_git_sha(),
                        config_sha256=str(manifest["config_sha256"]),
                        input_sha256=input_sha,
                        phase_id=str(phase_id),
                        shard_id=shard_id,
                        repeats_per_direction=policy.repeats_per_direction,
                    )
                    file_hash_ok = str(report.get("executions_file_sha256", "")) == sha256_file(
                        execution_path
                    )
                    if valid["gate_pass"] and file_hash_ok:
                        completed[shard_id] = executions
                        continue
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    pass
            missing.append(shard_id)
        reused_shard_count = len(completed)

        started = time.time()
        running: list[
            tuple[int, subprocess.Popen[str], Any, _AuditWorkerToken | None]
        ] = []
        environment = _thread_limited_environment()

        def heartbeat() -> None:
            payload = {
                "phase_id": str(phase_id),
                "completed_shards": len(completed),
                "total_shards": self.shard_count,
                "running_shards": [
                    shard for shard, _process, _handle, _token in running
                ],
                "pending_shards": list(missing),
                "elapsed_s": time.time() - started,
                "updated_at_unix_s": time.time(),
            }
            _write_json(phase_directory / "heartbeat.json", payload)
            print(json.dumps(_strict({"audit_shard_progress": payload}), sort_keys=True), flush=True)

        heartbeat()
        while missing or running:
            while missing and len(running) < self.maximum_concurrent_workers:
                token = self.token_pool.try_acquire() if self.token_pool is not None else None
                if self.token_pool is not None and token is None:
                    break
                shard_id = missing.pop(0)
                shard_directory = phase_directory / f"shard_{shard_id:02d}"
                shard_directory.mkdir(parents=True, exist_ok=True)
                handle = (shard_directory / "worker.log").open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    str(self.worker_runner_path),
                    "--config",
                    str(self.config["config_path"]),
                    "--audit-shard-bundle",
                    str(phase_directory),
                    "--audit-shard-id",
                    str(shard_id),
                ]
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=SOURCE_ROOT,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=handle,
                        text=True,
                    )
                except Exception:
                    handle.close()
                    if token is not None:
                        token.release()
                    raise
                running.append((shard_id, process, handle, token))
            progressed = False
            survivors: list[
                tuple[int, subprocess.Popen[str], Any, _AuditWorkerToken | None]
            ] = []
            for shard_id, process, handle, token in running:
                if process.poll() is None:
                    survivors.append((shard_id, process, handle, token))
                    continue
                stdout, _stderr = process.communicate()
                handle.close()
                if token is not None:
                    token.release()
                if process.returncode != 0:
                    for _other_id, other, other_handle, other_token in running:
                        if other.pid == process.pid:
                            continue
                        if other.poll() is None:
                            other.terminate()
                            other.wait(timeout=30)
                        other_handle.close()
                        if other_token is not None:
                            other_token.release()
                    raise RuntimeError(
                        f"audit shard failed phase={phase_id} shard={shard_id} "
                        f"returncode={process.returncode} stdout={stdout[-1000:]}"
                    )
                shard_directory = phase_directory / f"shard_{shard_id:02d}"
                executions = pd.read_parquet(shard_directory / "executions.parquet")
                report = _read_json(shard_directory / "report.json")
                shard_registry = registry[registry["shard_id"].eq(shard_id)]
                valid = validate_shard_completion(
                    report,
                    shard_registry,
                    executions,
                    source_sha=_git_sha(),
                    config_sha256=str(manifest["config_sha256"]),
                    input_sha256=input_sha,
                    phase_id=str(phase_id),
                    shard_id=shard_id,
                    repeats_per_direction=policy.repeats_per_direction,
                )
                if not valid["gate_pass"]:
                    raise RuntimeError(
                        f"audit shard exact-set validation failed: {valid}"
                    )
                if str(report.get("executions_file_sha256", "")) != sha256_file(
                    shard_directory / "executions.parquet"
                ):
                    raise RuntimeError(
                        f"audit shard file hash failed phase={phase_id} shard={shard_id}"
                    )
                completed[shard_id] = executions
                progressed = True
            running = survivors
            if progressed:
                heartbeat()
            elif running or missing:
                time.sleep(0.25)
        merged = merge_validated_audit_shards(
            registry,
            completed,
            repeats_per_direction=policy.repeats_per_direction,
        )
        _write_parquet(merged, phase_directory / "executions.parquet")
        aggregate = _gate(
            phase_directory / "gate.json",
            {
                "all_registered_shards_complete": len(completed) == self.shard_count,
                "exact_execution_set": len(merged)
                == len(registry) * 2 * policy.repeats_per_direction,
            },
            phase_id=str(phase_id),
            schedule_count=len(registry),
            execution_count=len(merged),
            shard_count=self.shard_count,
            maximum_concurrent_workers=self.maximum_concurrent_workers,
            assignment_strategy=self.assignment_strategy,
            input_sha256=input_sha,
            executions_sha256=sha256_file(phase_directory / "executions.parquet"),
            resumed_shard_count=reused_shard_count,
        )
        if not aggregate["gate_pass"]:
            raise RuntimeError(f"audit shard aggregate failed: {aggregate}")
        return merged


def _growth_checkpoint_input_sha256(
    tasks: pd.DataFrame,
    task_edges: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> str:
    return _payload_sha256(
        {
            "tasks": _frame_payload_sha256(tasks),
            "task_edges": _frame_payload_sha256(task_edges),
            "candidates": _frame_payload_sha256(candidate_frame),
            "assignments": _frame_payload_sha256(assignments),
        }
    )


def _load_growth_checkpoint(
    directory: Path,
    *,
    tasks: pd.DataFrame,
    task_edges: pd.DataFrame,
    input_sha256: str,
    config: Mapping[str, Any],
    patch_id: str,
    variant: str,
) -> tuple[Any, dict[str, Any], dict[str, pd.DataFrame]] | None:
    checkpoint = directory / "_growth_checkpoint"
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
        checks = (
            str(manifest.get("source_sha", "")) == _git_sha(),
            str(manifest.get("config_sha256", ""))
            == sha256_file(Path(str(config["config_path"]))),
            str(manifest.get("runtime_sha256", "")) == _runtime_sha256(),
            str(manifest.get("input_sha256", "")) == str(input_sha256),
            str(manifest.get("patch_id", "")) == str(patch_id),
            str(manifest.get("variant", "")) == str(variant),
        )
        if not all(checks):
            return None
        frames: dict[str, pd.DataFrame] = {}
        for record in manifest.get("artifacts", ()):
            path = checkpoint / str(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
                or sha256_file(path) != str(record["sha256"])
            ):
                return None
            if path.suffix == ".parquet":
                frames[path.stem] = pd.read_parquet(path)
        required = {"section_hypotheses", "selected_edges"}
        if not required <= set(frames):
            return None
        root_report = _read_json(checkpoint / "root_report.json")
        nodes = atlas_nodes_from_frames(tasks, task_edges)
        growth = section_growth_from_frames(
            nodes,
            frames["section_hypotheses"],
            frames["selected_edges"],
            policy=_section_policy(
                config,
                beam_width=int(manifest["beam_width"]),
                root_count=int(manifest["selected_root_count"]),
                consensus=bool(manifest["consensus"]),
            ),
        )
        return growth, root_report, frames
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_growth_checkpoint(
    directory: Path,
    *,
    growth: Any,
    root_report: Mapping[str, Any],
    input_sha256: str,
    config: Mapping[str, Any],
    patch_id: str,
    variant: str,
    beam_width: int,
    consensus: bool,
) -> dict[str, pd.DataFrame]:
    checkpoint = directory / "_growth_checkpoint"
    frames = {str(name): frame.copy() for name, frame in growth.frames().items()}
    artifacts = []
    for name, frame in frames.items():
        path = checkpoint / f"{name}.parquet"
        _write_parquet(frame, path)
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    _write_json(checkpoint / "root_report.json", root_report)
    root_path = checkpoint / "root_report.json"
    artifacts.append(
        {
            "path": root_path.name,
            "bytes": root_path.stat().st_size,
            "sha256": sha256_file(root_path),
        }
    )
    _write_json(
        checkpoint / "manifest.json",
        {
            "schema_version": 1,
            "source_sha": _git_sha(),
            "config_sha256": sha256_file(Path(str(config["config_path"]))),
            "runtime_sha256": _runtime_sha256(),
            "input_sha256": str(input_sha256),
            "patch_id": str(patch_id),
            "variant": str(variant),
            "beam_width": int(beam_width),
            "selected_root_count": int(len(growth.charts)),
            "consensus": bool(consensus),
            "artifacts": artifacts,
        },
    )
    return frames


def _variant_spec(name: str) -> tuple[int, int, int, bool, bool, bool]:
    specs = {
        "baseline": (4, 8, 20260871, False, False, False),
        "main_K1_R5": (1, 5, 20260871, False, False, False),
        "root_dropout": (1, 5, 20260871, False, True, False),
        "root_order2": (1, 5, 20260872, False, False, False),
        "consensus_patch07": (1, 5, 20260871, True, False, False),
        "task_graph_refined": (1, 5, 20260871, False, False, True),
        "meso_root_dropout": (4, 8, 20260871, False, True, False),
    }
    if name not in specs:
        raise ValueError(f"unknown V14.2R variant: {name}")
    return specs[name]


def _refine_task_graph(tasks: pd.DataFrame, task_edges: pd.DataFrame) -> pd.DataFrame:
    """Add only local physical neighbours; never delete a registered task edge."""

    ordered = tasks.sort_values("task_node_id", kind="stable")
    node_ids = ordered["task_node_id"].to_numpy(dtype=int)
    xyz = ordered[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    pairs = cKDTree(xyz).query_pairs(r=0.0100001, output_type="set")
    existing = {
        tuple(sorted((int(row.left_node_id), int(row.right_node_id))))
        for row in task_edges.itertuples(index=False)
    }
    additions = []
    for left_index, right_index in sorted(pairs):
        edge = tuple(sorted((int(node_ids[left_index]), int(node_ids[right_index]))))
        if edge in existing:
            continue
        additions.append(
            {"left_node_id": edge[0], "right_node_id": edge[1], "adjacency": "registered_10mm_refinement"}
        )
    if not additions:
        return task_edges.copy()
    return pd.concat([task_edges, pd.DataFrame.from_records(additions)], ignore_index=True)


def _strong_reference_primary(
    repaired: Any,
    tasks: pd.DataFrame,
    task_edges: pd.DataFrame,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """MILP reference over every probe and induced edge of a 64-cell patch."""

    component = tuple(repaired.selected_stitch_component)
    reference_nodes = set(repaired.primary_chart_by_node)
    if not component or not reference_nodes:
        return {"gate_pass": False, "reason": "empty_primary_reference_domain"}
    if len(component) == 1:
        return {
            "gate_pass": True,
            "reason": "qualified_singleton_is_the_exact_reference",
            "reference_node_count": len(reference_nodes),
            "reference_edge_count": int(repaired.induced_edge_count),
            "objective_gap": 0.0,
            "beta_p95_deg": 0.0,
            "beta_max_deg": 0.0,
        }
    charts = repaired.growth.chart_by_id
    stitch_pairs = {tuple(sorted(edge)) for edge in repaired.stitch_edges}
    available = {
        node: tuple(chart for chart in component if node in charts[chart].selected_by_node)
        for node in sorted(reference_nodes)
    }
    available = {node: labels for node, labels in available.items() if labels}
    node_ids = set(available)
    reference_edge_set: set[tuple[int, int]] = set()
    for row in task_edges.itertuples(index=False):
        left = int(row.left_node_id)
        right = int(row.right_node_id)
        if left in node_ids and right in node_ids:
            reference_edge_set.add(tuple(sorted((left, right))))
    reference_edges = sorted(reference_edge_set)
    y_index: dict[tuple[int, str], int] = {}
    costs: list[float] = []
    for node, labels in available.items():
        for chart_id in labels:
            candidate = charts[chart_id].selected_by_node[node].candidate
            y_index[(node, chart_id)] = len(costs)
            costs.append(
                float(candidate.residual_mm) / 3.0
                - min(float(candidate.min_margin_deg), 10.0) / 100.0
                + min(float(candidate.condition_number), 1000.0) / 100000.0
            )
    z_index: dict[tuple[int, int, str, str], int] = {}
    allowed_pairs_by_edge: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for left, right in reference_edges:
        pairs = []
        for left_chart in available[left]:
            for right_chart in available[right]:
                if left_chart != right_chart and tuple(sorted((left_chart, right_chart))) not in stitch_pairs:
                    continue
                pairs.append((left_chart, right_chart))
                left_beta = charts[left_chart].selected_by_node[left].candidate.beta_rad
                right_beta = charts[right_chart].selected_by_node[right].candidate.beta_rad
                pair_cost = beta_rms_deg(left_beta, right_beta) / 10.0
                if left_chart != right_chart:
                    pair_cost += 1.0
                z_index[(left, right, left_chart, right_chart)] = len(costs)
                costs.append(pair_cost)
        if not pairs:
            return {"gate_pass": False, "reason": f"no_compatible_pair_for_edge:{left}:{right}"}
        allowed_pairs_by_edge[(left, right)] = pairs
    constraints: list[tuple[dict[int, float], float]] = []
    for node, labels in available.items():
        constraints.append(({y_index[(node, chart)]: 1.0 for chart in labels}, 1.0))
    for (left, right), pairs in allowed_pairs_by_edge.items():
        for chart in available[left]:
            row = {y_index[(left, chart)]: -1.0}
            for left_chart, right_chart in pairs:
                if left_chart == chart:
                    row[z_index[(left, right, left_chart, right_chart)]] = 1.0
            constraints.append((row, 0.0))
        for chart in available[right]:
            row = {y_index[(right, chart)]: -1.0}
            for left_chart, right_chart in pairs:
                if right_chart == chart:
                    row[z_index[(left, right, left_chart, right_chart)]] = 1.0
            constraints.append((row, 0.0))
    matrix = lil_matrix((len(constraints), len(costs)), dtype=float)
    rhs = np.empty(len(constraints), dtype=float)
    for row_index, (coefficients, value) in enumerate(constraints):
        for column, coefficient in coefficients.items():
            matrix[row_index, column] = coefficient
        rhs[row_index] = value
    result = milp(
        np.asarray(costs),
        integrality=np.ones(len(costs), dtype=int),
        bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
        constraints=LinearConstraint(matrix.tocsr(), rhs, rhs),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return {"gate_pass": False, "reason": f"milp_failed:{result.message}"}
    reference = {
        node: min(available[node], key=lambda chart: (-result.x[y_index[(node, chart)]], chart))
        for node in available
    }
    gaps = np.asarray([
        beta_rms_deg(
            repaired.primary_beta_by_node[node],
            charts[reference[node]].selected_by_node[node].candidate.beta_rad,
        )
        for node in sorted(reference)
    ])
    # Evaluate the registered assignment in the same objective by fixing each
    # node and compatible edge pair.
    registered_objective = 0.0
    for node, chart in repaired.primary_chart_by_node.items():
        if (node, chart) in y_index:
            registered_objective += costs[y_index[(node, chart)]]
    for left, right in reference_edges:
        left_chart = repaired.primary_chart_by_node[left]
        right_chart = repaired.primary_chart_by_node[right]
        key = (left, right, left_chart, right_chart)
        registered_objective += costs[z_index[key]] if key in z_index else 1.0e9
    objective_gap = float(registered_objective - result.fun)
    relative_objective_gap = objective_gap / max(1.0, abs(float(result.fun)))
    p95 = float(np.percentile(gaps, 95)) if len(gaps) else math.inf
    maximum = float(np.max(gaps)) if len(gaps) else math.inf
    return {
        "gate_pass": bool(
            relative_objective_gap
            <= float(policy["strong_reference_relative_objective_gap_max"])
            and p95 <= float(policy["strong_reference_beta_p95_max_deg"])
            and maximum <= float(policy["strong_reference_beta_max_deg"])
        ),
        "reason": "milp_reference_comparison",
        "reference_node_count": len(reference),
        "reference_edge_count": len(reference_edges),
        "milp_objective": float(result.fun),
        "registered_objective": registered_objective,
        "objective_gap": objective_gap,
        "relative_objective_gap": relative_objective_gap,
        "beta_p95_deg": p95,
        "beta_max_deg": maximum,
        "solver_message": str(result.message),
    }


def _patch_completion_artifacts(directory: Path) -> tuple[dict[str, Any], ...]:
    operational_names = {
        "completion_manifest.json",
        "heartbeat.json",
        "progress.json",
        "worker.log",
    }
    paths = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in operational_names
    ]
    return tuple(
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    )


def _write_patch_completion_manifest(
    directory: Path,
    *,
    config: Mapping[str, Any],
    patch_id: str,
    variant: str,
    input_sha256: str,
) -> dict[str, Any]:
    artifacts = _patch_completion_artifacts(directory)
    if "report.json" not in {str(record["path"]) for record in artifacts}:
        raise RuntimeError("patch completion requires report.json")
    manifest = {
        "schema_version": 1,
        "source_sha": _git_sha(),
        "config_sha256": sha256_file(Path(str(config["config_path"]))),
        "runtime_sha256": _runtime_sha256(),
        "input_sha256": str(input_sha256),
        "patch_id": str(patch_id),
        "variant": str(variant),
        "artifacts": artifacts,
    }
    _write_json(directory / "completion_manifest.json", manifest)
    return manifest


def _load_validated_patch_result(
    directory: Path,
    *,
    config: Mapping[str, Any],
    patch_id: str,
    variant: str,
    input_sha256: str,
) -> dict[str, Any] | None:
    manifest_path = directory / "completion_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
        checks = (
            int(manifest.get("schema_version", 0)) == 1,
            str(manifest.get("source_sha", "")) == _git_sha(),
            str(manifest.get("config_sha256", ""))
            == sha256_file(Path(str(config["config_path"]))),
            str(manifest.get("runtime_sha256", "")) == _runtime_sha256(),
            str(manifest.get("patch_id", "")) == str(patch_id),
            str(manifest.get("variant", "")) == str(variant),
            str(manifest.get("input_sha256", "")) == str(input_sha256),
        )
        if not all(checks):
            return None
        artifacts = tuple(manifest.get("artifacts", ()))
        if "report.json" not in {str(record.get("path", "")) for record in artifacts}:
            return None
        for record in artifacts:
            relative = Path(str(record["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                return None
            path = directory / relative
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
                or sha256_file(path) != str(record["sha256"])
            ):
                return None
        return _read_json(directory / "report.json")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _stage_completion_artifacts(directory: Path) -> tuple[dict[str, Any], ...]:
    completion = directory / "completion_manifest.json"
    paths = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != completion
    ]
    return tuple(
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    )


def _write_stage_completion_manifest(
    directory: Path,
    *,
    config: Mapping[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    artifacts = _stage_completion_artifacts(directory)
    if "gate.json" not in {str(record["path"]) for record in artifacts}:
        raise RuntimeError("stage completion requires gate.json")
    manifest = {
        "schema_version": 1,
        "source_sha": _git_sha(),
        "config_sha256": sha256_file(Path(str(config["config_path"]))),
        "runtime_sha256": _runtime_sha256(),
        "stage_name": str(stage_name),
        "artifacts": artifacts,
    }
    _write_json(directory / "completion_manifest.json", manifest)
    return manifest


def _load_validated_stage_result(
    directory: Path,
    *,
    config: Mapping[str, Any],
    stage_name: str,
) -> dict[str, Any] | None:
    manifest_path = directory / "completion_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
        checks = (
            int(manifest.get("schema_version", 0)) == 1,
            str(manifest.get("source_sha", "")) == _git_sha(),
            str(manifest.get("config_sha256", ""))
            == sha256_file(Path(str(config["config_path"]))),
            str(manifest.get("runtime_sha256", "")) == _runtime_sha256(),
            str(manifest.get("stage_name", "")) == str(stage_name),
        )
        if not all(checks):
            return None
        artifacts = tuple(manifest.get("artifacts", ()))
        if "gate.json" not in {str(record.get("path", "")) for record in artifacts}:
            return None
        for record in artifacts:
            relative = Path(str(record["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                return None
            path = directory / relative
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
                or sha256_file(path) != str(record["sha256"])
            ):
                return None
        return _read_json(directory / "gate.json")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _execute_patch(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    patch_id: str,
    variant: str,
    directory: Path,
) -> dict[str, Any]:
    patch_started = time.time()
    directory.mkdir(parents=True, exist_ok=True)
    legacy_config = legacy.load_config(_source_paths(config, project_root)["legacy_config"])
    if patch_id == "patch_12":
        replacement = output_root / STAGE_DIRS["replacement_confirmation"]
        tasks = pd.read_parquet(replacement / "patch12_task_nodes.parquet").drop(columns=["patch_id"])
        task_edges = pd.read_parquet(replacement / "patch12_task_edges.parquet").drop(columns=["patch_id"])
        candidate_frame = pd.read_parquet(replacement / "patch12_candidate_clusters.parquet").drop(columns=["patch_id"])
        assignments = pd.read_parquet(replacement / "patch12_assignments.parquet")
    elif patch_id == "meso_512":
        meso_input = output_root / STAGE_DIRS["meso_bridge"] / "input"
        tasks = pd.read_parquet(meso_input / "meso_task_nodes.parquet").drop(columns=["patch_id"])
        task_edges = pd.read_parquet(meso_input / "meso_task_edges.parquet").drop(columns=["patch_id"])
        candidate_frame = pd.read_parquet(meso_input / "meso_candidate_clusters.parquet").drop(columns=["patch_id"])
        assignments = pd.read_parquet(meso_input / "meso_assignments.parquet")
    else:
        tasks, _legacy_edges, candidate_frame, assignments = legacy._patch_inputs(legacy_config, project_root, patch_id)
    if patch_id not in {"patch_12", "meso_512"}:
        retry_patch = _source_paths(config, project_root)["retry4"] / "01_patch_ablations" / patch_id
        task_edges = pd.read_parquet(retry_patch / "E3_dynamic_insertion_task_edges.parquet")
    beam, root_count, seed, consensus, leave_one_out, refine_graph = _variant_spec(variant)
    if refine_graph:
        task_edges = _refine_task_graph(tasks, task_edges)
    growth_input_sha = _growth_checkpoint_input_sha256(
        tasks, task_edges, candidate_frame, assignments
    )
    existing = _load_validated_patch_result(
        directory,
        config=config,
        patch_id=patch_id,
        variant=variant,
        input_sha256=growth_input_sha,
    )
    if existing is not None:
        return existing
    candidates = legacy._candidates_from_frame(candidate_frame)
    environment = legacy._EndpointOnlyForwardAdapter(
        load_environment(project_root, _source_paths(config, project_root)["robot_config"])
    )
    nodes, continuation = legacy._segmented_continuation(
        environment, tasks, task_edges
    )
    input_ready = time.time()
    growth_started = time.time()
    checkpoint = _load_growth_checkpoint(
        directory,
        tasks=tasks,
        task_edges=task_edges,
        input_sha256=growth_input_sha,
        config=config,
        patch_id=patch_id,
        variant=variant,
    )
    if checkpoint is None:
        initial_roots = legacy._root_keys(tasks, assignments, candidates, max(1, root_count))
        candidates, root_keys, root_report = _root_candidates_with_enrichment(
            environment, tasks, candidates, initial_roots, root_count, seed
        )
        if leave_one_out and root_keys:
            root_keys = root_keys[1:]
            root_report["leave_one_root_out"] = True
        if not root_keys:
            raise RuntimeError(f"no usable roots for {patch_id}/{variant}")
        growth = build_section_first_atlas(
            nodes,
            candidates,
            continuation,
            root_keys=root_keys,
            policy=_section_policy(
                config,
                beam_width=beam,
                root_count=max(1, len(root_keys)),
                consensus=consensus,
            ),
        )
        growth_frames = _write_growth_checkpoint(
            directory,
            growth=growth,
            root_report=root_report,
            input_sha256=growth_input_sha,
            config=config,
            patch_id=patch_id,
            variant=variant,
            beam_width=beam,
            consensus=consensus,
        )
        growth_checkpoint_resumed = False
    else:
        growth, root_report, growth_frames = checkpoint
        growth_checkpoint_resumed = True
    growth_finished = time.time()
    token_pool = (
        _AuditWorkerTokenPool(
            Path(str(config["_audit_token_pool"])),
            capacity=int(config["_audit_token_count"]),
        )
        if config.get("_audit_token_pool")
        else None
    )
    repair_started = time.time()
    repaired = repair_rooted_section_atlas(
        growth,
        continuation,
        patch_id=patch_id,
        method=variant,
        policy=_repair_policy(config),
        retry_continuation=_registered_retry_adapter(environment, tasks, task_edges),
        schedule_executor=_SubprocessAuditExecutor(
            config=config,
            project_root=project_root,
            tasks=tasks,
            task_edges=task_edges,
            patch_directory=directory,
            shard_count=int(config.get("_audit_shards_per_patch", 1)),
            token_pool=token_pool,
        ),
    )
    repair_finished = time.time()
    strong_reference_required = bool(
        int(tasks["source_parent_node_id"].nunique()) <= 64
        and patch_id in set(map(str, config["diagnostic_patch_ids"]))
        and variant in {"baseline", "main_K1_R5"}
    )
    strong_reference_started = time.time()
    strong_reference = (
        _strong_reference_primary(repaired, tasks, task_edges, config["search_stability"])
        if strong_reference_required
        else {
            "gate_pass": True,
            "required": False,
            "reason": "strong_reference_registered_only_for_small_patches",
        }
    )
    strong_reference_finished = time.time()
    artifact_write_started = time.time()
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in growth_frames.items():
        _write_parquet(frame, directory / f"{name}.parquet")
    for name, frame in repaired.frames.items():
        _write_parquet(frame, directory / f"{name}.parquet")
    _write_json(directory / "strong_reference.json", strong_reference)
    artifact_write_finished = time.time()
    hypothesis = growth_frames["section_hypotheses"]
    multiplicity = hypothesis.groupby(["chart_id", "task_node_id"]).size()
    alternative_ratio = float((multiplicity > 1).mean()) if len(multiplicity) else 0.0
    executions = repaired.frames["audit_v2_executions"]
    optimization = repaired.frames["primary_optimization"]
    schedule_classification = executions.groupby("schedule_id")["classification"].agg(
        lambda values: tuple(sorted(set(map(str, values))))
    )
    first_pass_failure_count = sum(
        any(value in {"recoverable_numerical", "persistent_numerical"} for value in values)
        for values in schedule_classification
    )
    parent_by_task = tasks.set_index("task_node_id")["source_parent_node_id"].astype(int).to_dict()
    primary_nodes_by_chart: dict[str, list[int]] = {}
    for node_id, chart_id in repaired.primary_chart_by_node.items():
        primary_nodes_by_chart.setdefault(str(chart_id), []).append(int(node_id))
    parent_count_by_chart = {
        chart_id: len({parent_by_task[node_id] for node_id in node_ids})
        for chart_id, node_ids in primary_nodes_by_chart.items()
    }
    tiny_policy = config["confirmation_gate"]
    total_parent_count = int(tasks["source_parent_node_id"].nunique())
    tiny_parent_threshold = max(
        int(tiny_policy["tiny_chart_min_cells_abs"]),
        int(math.ceil(float(tiny_policy["tiny_chart_min_fraction"]) * total_parent_count)),
    )
    tiny_nodes = sum(
        len(primary_nodes_by_chart[chart_id])
        for chart_id, parent_count in parent_count_by_chart.items()
        if parent_count < tiny_parent_threshold
    )
    report = {
        "patch_id": patch_id,
        "patch_split": str(assignments["patch_split"].iloc[0]),
        "variant": variant,
        "beam_width": beam,
        "task_graph_refined": refine_graph,
        "task_edge_count": len(task_edges),
        "coverage_ratio": repaired.coverage_ratio,
        "largest_stitched_component_ratio": repaired.largest_coherent_region_ratio,
        "abstention_ratio": len(repaired.abstained_node_ids) / max(1, len(growth.task_nodes)),
        "qualified_chart_count": len(repaired.qualified_chart_ids),
        "selected_stitch_component": list(repaired.selected_stitch_component),
        "geometry_gate": repaired.geometry_gate,
        "solver_gate": repaired.solver_gate,
        "repeat_gate": repaired.repeat_gate,
        "certificate_gate": repaired.certificate_gate,
        "induced_edge_count": repaired.induced_edge_count,
        "audited_edge_count": repaired.audited_edge_count,
        "edge_completeness_ratio": repaired.edge_completeness_ratio,
        "pre_abstention_cycle_rank": repaired.pre_abstention_cycle_rank,
        "retained_cycle_rank": repaired.retained_cycle_rank,
        "audited_fundamental_cycle_count": repaired.audited_fundamental_cycle_count,
        "cycle_coverage_ratio": repaired.cycle_coverage_ratio,
        "strong_reference": strong_reference,
        "strong_reference_required": strong_reference_required,
        "strong_reference_gate": bool(strong_reference.get("gate_pass", False)),
        "geometry_p95_deg": repaired.diagnostic.geometry_metrics["p95_deg"],
        "geometry_max_deg": repaired.diagnostic.geometry_metrics["max_deg"],
        "repeat_p95_deg": repaired.diagnostic.repeat_metrics["p95_deg"],
        "repeat_missing_count": repaired.diagnostic.repeat_metrics["missing_count"],
        "persistent_numerical_count": repaired.diagnostic.solver_metrics["persistent_numerical_count"],
        "first_pass_solver_failure_count": int(first_pass_failure_count),
        "first_pass_solver_failure_is_diagnostic_only": True,
        "geometric_branch_disagreement_count": repaired.diagnostic.solver_metrics["geometric_branch_disagreement_count"],
        "alternative_hypothesis_ratio": alternative_ratio,
        "tiny_chart_parent_threshold": tiny_parent_threshold,
        "tiny_primary_measure_ratio": tiny_nodes / max(1, len(repaired.primary_chart_by_node)),
        "primary_parent_count_by_chart": parent_count_by_chart,
        "primary_optimization_objective_spread": (
            float(optimization["objective"].max() - optimization["objective"].min())
            if len(optimization) else math.inf
        ),
        "primary_optimization_assignment_spread": (
            float(optimization["assignment_change_ratio_to_selected"].max())
            if len(optimization) else 1.0
        ),
        "primary_optimization_beta_p95_spread_deg": (
            float(optimization.get("beta_p95_deg_to_selected", pd.Series([0.0])).max())
            if len(optimization) else math.inf
        ),
        "runtime_s": time.time() - patch_started,
        "phase_runtime_s": {
            "input_load": input_ready - patch_started,
            "root_growth_or_resume": growth_finished - growth_started,
            "atlas_repair_and_audits": repair_finished - repair_started,
            "strong_reference": strong_reference_finished
            - strong_reference_started,
            "artifact_write": artifact_write_finished - artifact_write_started,
        },
        "growth_checkpoint_resumed": growth_checkpoint_resumed,
        **root_report,
    }
    mechanism = config["mechanism_gate"]
    report["gate_pass"] = bool(
        report["coverage_ratio"] >= float(mechanism["primary_label_coverage_min"])
        and report["largest_stitched_component_ratio"] >= float(mechanism["largest_stitched_component_min"])
        and report["geometry_gate"]
        and report["solver_gate"]
        and report["repeat_gate"]
        and report["certificate_gate"]
        and abs(report["edge_completeness_ratio"] - 1.0) <= 1e-12
        and abs(report["cycle_coverage_ratio"] - 1.0) <= 1e-12
        and report["strong_reference_gate"]
        and (report["repeat_p95_deg"] is not None and report["repeat_p95_deg"] <= float(mechanism["repeat_p95_max_deg"]))
        and report["persistent_numerical_count"] <= int(mechanism["unresolved_critical_edges_max"])
        and report["root_budget_saturated"]
    )
    _write_json(directory / "report.json", report)
    _write_patch_completion_manifest(
        directory,
        config=config,
        patch_id=patch_id,
        variant=variant,
        input_sha256=growth_input_sha,
    )
    return report


def _run_patch_jobs(
    config: Mapping[str, Any], output_root: Path, stage_name: str, jobs: Sequence[tuple[str, str]]
) -> None:
    # A coordinator must recompute the physical input digest before accepting
    # a completion manifest.  Launching the lightweight coordinator for every
    # registered job avoids trusting report/manifest presence in the parent.
    pending = list(jobs)
    if not pending:
        return
    environment = _thread_limited_environment()
    patch_worker_limit = int(config["parallel"]["patch_workers"])
    total_audit_workers = int(config["audit_execution"]["total_shard_workers"])
    token_pool_directory = (
        output_root / STAGE_DIRS[stage_name] / "_audit_worker_tokens"
    )
    while pending:
        batch = pending[:patch_worker_limit]
        del pending[: len(batch)]
        running: list[tuple[str, str, subprocess.Popen[str], Any]] = []
        for patch_id, variant in batch:
            directory = output_root / STAGE_DIRS[stage_name] / patch_id / variant
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / "worker.log").open("a", encoding="utf-8")
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--config", str(config["config_path"]),
                "--output-root", str(output_root),
                "--stage", stage_name,
                "--patch-id", patch_id,
                "--variant", variant,
                "--audit-shards-per-patch", str(total_audit_workers),
                "--audit-token-pool", str(token_pool_directory),
                "--audit-token-count", str(total_audit_workers),
            ]
            process = subprocess.Popen(
                command, cwd=SOURCE_ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            running.append((patch_id, variant, process, handle))
        failures: list[dict[str, Any]] = []
        while running:
            survivors = []
            for patch_id, variant, process, handle in running:
                status = process.poll()
                if status is None:
                    survivors.append((patch_id, variant, process, handle))
                    continue
                handle.close()
                if status != 0:
                    failures.append(
                        {
                            "patch_id": patch_id,
                            "variant": variant,
                            "returncode": status,
                        }
                    )
            running = survivors
            if failures:
                for _patch, _variant, process, handle in running:
                    process.terminate()
                    process.wait(timeout=30)
                    handle.close()
                raise RuntimeError(f"V14.2R patch worker failures: {failures}")
            if running:
                time.sleep(1.0)
        if pending:
            time.sleep(1.0)


def stage_rooted_baseline(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require(config, output_root, "artifact_diagnostics")
    stage = output_root / STAGE_DIRS["rooted_baseline"]
    if config.get("_patch_id"):
        report = _execute_patch(config, project_root, output_root, str(config["_patch_id"]), str(config["_variant"]), stage / str(config["_patch_id"]) / str(config["_variant"]))
        return {"gate_pass": True, "worker_report": report}
    patches = tuple(map(str, config["diagnostic_patch_ids"]))
    _run_patch_jobs(config, output_root, "rooted_baseline", [(patch, "baseline") for patch in patches])
    reports = [_read_json(stage / patch / "baseline/report.json") for patch in patches]
    _write_parquet(pd.DataFrame.from_records(reports), stage / "baseline_patch_reports.parquet")
    return _gate(stage / "gate.json", {"four_patch_baseline_complete": len(reports) == 4}, patch_reports=reports)


def stage_registered_retry(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    baseline = _require(config, output_root, "rooted_baseline")
    stage = output_root / STAGE_DIRS["registered_retry"]
    rows = baseline["patch_reports"]
    classifications = []
    for row in rows:
        path = output_root / STAGE_DIRS["rooted_baseline"] / row["patch_id"] / "baseline/audit_v2_executions.parquet"
        frame = pd.read_parquet(path)
        counts = frame["classification"].value_counts().to_dict()
        classifications.append({"patch_id": row["patch_id"], **{str(key): int(value) for key, value in counts.items()}})
    _write_parquet(pd.DataFrame.from_records(classifications).fillna(0), stage / "retry_classifications.parquet")
    return _gate(
        stage / "gate.json",
        {"all_retry_schedules_classified": len(classifications) == 4},
        patch_reports=rows,
        retry_classifications=classifications,
        oracle_used_for_pass=False,
    )


def stage_patch07_local_audit(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require(config, output_root, "registered_retry")
    stage = output_root / STAGE_DIRS["patch07_local_audit"]
    base = output_root / STAGE_DIRS["rooted_baseline"] / "patch_07/baseline"
    schedules = pd.read_parquet(base / "audit_v2_schedules.parquet")
    executions = pd.read_parquet(base / "audit_v2_executions.parquet")
    failures = executions[
        (~executions["solver_success"].astype(bool))
        | pd.to_numeric(executions["geometry_gap_deg"], errors="coerce").gt(float(config["audit_v2"]["geometry_max_deg"]))
        | pd.to_numeric(executions["repeat_gap_deg"], errors="coerce").gt(float(config["audit_v2"]["repeat_p95_max_deg"]))
    ].copy()
    localized = failures.merge(
        schedules[["schedule_id", "audit_kind", "path_node_ids", "unique_entity_id"]],
        on="schedule_id",
        how="left",
    )
    _write_parquet(localized, stage / "patch07_failed_schedule_localization.parquet")
    entities = (
        localized.groupby("unique_entity_id", as_index=False)
        .agg(
            execution_count=("schedule_id", "size"),
            audit_kind=("audit_kind", "first"),
            any_solver_success=("solver_success", "any"),
            maximum_geometry_gap_deg=("geometry_gap_deg", "max"),
            maximum_repeat_gap_deg=("repeat_gap_deg", "max"),
            path_node_ids=("path_node_ids", "first"),
        )
        .sort_values("unique_entity_id", kind="stable")
    )
    _write_parquet(entities, stage / "patch07_failed_unique_entities.parquet")
    affected: set[int] = set()
    for values in localized.get("path_node_ids", []):
        affected.update(map(int, values))
    primary = pd.read_parquet(base / "primary_atlas.parquet")
    retained_domain = set(primary.loc[~primary["abstained"].astype(bool), "task_node_id"].astype(int))
    paths = _source_paths(config, project_root)
    task_edges = pd.read_parquet(
        paths["retry4"] / "01_patch_ablations/patch_07/E3_dynamic_insertion_task_edges.parquet"
    )
    adjacency: dict[int, set[int]] = {node: set() for node in retained_domain}
    all_edges: set[tuple[int, int]] = set()
    for row in task_edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left in retained_domain and right in retained_domain:
            edge = tuple(sorted((left, right)))
            all_edges.add(edge)
            adjacency[left].add(right)
            adjacency[right].add(left)

    def expanded(seed: set[int], hops: int) -> set[int]:
        output = set(seed) & retained_domain
        frontier = set(output)
        for _ in range(hops):
            frontier = set().union(*(adjacency.get(node, set()) for node in frontier)) - output
            output.update(frontier)
        return output

    endpoint = affected & retained_domain
    one_ring = expanded(endpoint, 1)
    two_ring = expanded(endpoint, 2)

    def induced_stats(removed: set[int]) -> dict[str, Any]:
        nodes = retained_domain - removed
        edges = {edge for edge in all_edges if edge[0] in nodes and edge[1] in nodes}
        local_adjacency = {node: set() for node in nodes}
        for left, right in edges:
            local_adjacency[left].add(right); local_adjacency[right].add(left)
        components = 0
        remaining = set(nodes)
        while remaining:
            components += 1
            queue = [min(remaining)]
            seen: set[int] = set()
            while queue:
                node = queue.pop()
                if node in seen:
                    continue
                seen.add(node)
                queue.extend(local_adjacency[node] - seen)
            remaining -= seen
        return {
            "retained_node_count": len(nodes),
            "retained_edge_count": len(edges),
            "component_count": components,
            "cycle_rank": max(0, len(edges) - len(nodes) + components),
            "retained_measure_ratio": len(nodes) / max(1, len(retained_domain)),
        }

    sensitivity = pd.DataFrame.from_records(
        [
            {"abstention_policy": "endpoint_only", **induced_stats(endpoint)},
            {"abstention_policy": "endpoint_plus_1hop", **induced_stats(one_ring)},
            {"abstention_policy": "endpoint_plus_2hop", **induced_stats(two_ring)},
        ]
    )
    _write_parquet(sensitivity, stage / "patch07_abstention_sensitivity.parquet")
    repeats = pd.to_numeric(executions["repeat_gap_deg"], errors="coerce")
    repeats = repeats[np.isfinite(repeats)]
    stitch = pd.read_parquet(base / "chart_stitchability.parquet")
    between = pd.to_numeric(stitch["beta_max_deg"], errors="coerce")
    between = between[np.isfinite(between)]
    return _gate(
        stage / "gate.json",
        {
            "local_audit_complete": True,
            "unique_entities_deduplicated": len(entities) <= len(localized),
            "induced_graph_reported": bool(len(sensitivity) == 3),
            "nonvacuous_cycle_coverage_reported": bool(sensitivity["cycle_rank"].max() > 0),
        },
        failed_execution_count=len(localized),
        failed_unique_entity_count=len(entities),
        affected_node_ids=sorted(affected),
        targeted_abstention_eligible=bool(len(affected) <= 8),
        endpoint_measure_ratio=len(endpoint) / max(1, len(retained_domain)),
        one_ring_measure_ratio=len(one_ring) / max(1, len(retained_domain)),
        two_ring_measure_ratio=len(two_ring) / max(1, len(retained_domain)),
        within_history_repeat_p95_deg=float(np.percentile(repeats, 95)) if len(repeats) else None,
        between_root_chart_gap_p95_deg=float(np.percentile(between, 95)) if len(between) else None,
        paired_history_test_status="diagnostic_only_not_a_stateful_trigger",
        s4_freeze_requires_mechanism_and_stability_gates=True,
    )


def _verified_edge_entities(directory: Path, config: Mapping[str, Any]) -> set[str]:
    schedules = pd.read_parquet(directory / "audit_v2_schedules.parquet")
    executions = pd.read_parquet(directory / "audit_v2_executions.parquet")
    edge_ids = set(
        schedules.loc[schedules["audit_kind"].eq("edge"), "schedule_id"].astype(str)
    )
    if not edge_ids:
        return set()
    subset = executions[executions["schedule_id"].astype(str).isin(edge_ids)].copy()
    subset["verified"] = (
        subset["solver_success"].astype(bool)
        & pd.to_numeric(subset["geometry_gap_deg"], errors="coerce").le(
            float(config["audit_v2"]["geometry_max_deg"])
        )
    )
    complete = subset.groupby("schedule_id")["verified"].all()
    verified_ids = set(complete[complete].index.astype(str))
    return set(
        schedules.loc[schedules["schedule_id"].astype(str).isin(verified_ids), "unique_entity_id"].astype(str)
    )


def _frame_stability(
    left: pd.DataFrame,
    right: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    left_directory: Path | None = None,
    right_directory: Path | None = None,
) -> dict[str, Any]:
    left = left[~left["abstained"].astype(bool)].set_index("task_node_id")
    right = right[~right["abstained"].astype(bool)].set_index("task_node_id")
    left_nodes, right_nodes = set(map(int, left.index)), set(map(int, right.index))
    common = sorted(left_nodes & right_nodes)
    union = left_nodes | right_nodes
    gaps = np.asarray([
        beta_rms_deg(
            left.loc[node, list(BETA_COLUMNS)].to_numpy(float),
            right.loc[node, list(BETA_COLUMNS)].to_numpy(float),
        )
        for node in common
    ])
    policy = config["search_stability"]
    report = {
        "coverage_jaccard": len(common) / max(1, len(union)),
        "beta_p95_deg": float(np.percentile(gaps, 95)) if len(gaps) else math.inf,
        "beta_max_deg": float(np.max(gaps)) if len(gaps) else math.inf,
        "assignment_change_ratio": float(np.mean(gaps > 1.0)) if len(gaps) else 1.0,
        "compared_node_count": len(common),
    }
    if left_directory is not None and right_directory is not None:
        left_edges = _verified_edge_entities(left_directory, config)
        right_edges = _verified_edge_entities(right_directory, config)
        edge_union = left_edges | right_edges
        report["verified_edge_change_ratio"] = len(left_edges ^ right_edges) / max(1, len(edge_union))
    else:
        report["verified_edge_change_ratio"] = math.inf
    report["gate_pass"] = bool(
        report["coverage_jaccard"] >= float(policy["coverage_jaccard_min"])
        and report["beta_p95_deg"] <= float(policy["beta_p95_max_deg"])
        and report["beta_max_deg"] <= float(policy["beta_max_deg"])
        and report["assignment_change_ratio"] <= float(policy["assignment_change_max"])
        and report["verified_edge_change_ratio"] <= float(policy["verified_edge_change_max"])
    )
    return report


def stage_search_stability(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require(config, output_root, "patch07_local_audit")
    stage = output_root / STAGE_DIRS["search_stability"]
    if config.get("_patch_id"):
        report = _execute_patch(config, project_root, output_root, str(config["_patch_id"]), str(config["_variant"]), stage / str(config["_patch_id"]) / str(config["_variant"]))
        return {"gate_pass": True, "worker_report": report}
    variants = ("main_K1_R5", "root_dropout", "root_order2", "task_graph_refined")
    patches = tuple(map(str, config["diagnostic_patch_ids"]))
    jobs = [(patch, variant) for patch in patches for variant in variants]
    jobs.append(("patch_07", "consensus_patch07"))
    _run_patch_jobs(config, output_root, "search_stability", jobs)
    rows = []
    alternative_ratios = []
    for patch in patches:
        baseline_directory = output_root / STAGE_DIRS["rooted_baseline"] / patch / "baseline"
        baseline = pd.read_parquet(baseline_directory / "primary_atlas.parquet")
        patch_variants = (*variants, "consensus_patch07") if patch == "patch_07" else variants
        for variant in patch_variants:
            directory = stage / patch / variant
            comparison = _frame_stability(
                baseline,
                pd.read_parquet(directory / "primary_atlas.parquet"),
                config,
                left_directory=baseline_directory,
                right_directory=directory,
            )
            report = _read_json(directory / "report.json")
            rows.append({"patch_id": patch, "variant": variant, **comparison})
            alternative_ratios.append(float(report["alternative_hypothesis_ratio"]))
    frame = pd.DataFrame.from_records(rows)
    _write_parquet(frame, stage / "primary_search_stability.parquet")
    trigger = bool(max(alternative_ratios, default=0.0) > float(config["search_stability"]["true_beam_alternative_ratio_trigger"]))
    required = frame[~frame["variant"].eq("consensus_patch07")]
    patch_pass = required.groupby("patch_id")["gate_pass"].all()
    selected_variant = "baseline"
    candidate = "main_K1_R5"
    stable = bool(frame.loc[frame["variant"].eq(candidate), "gate_pass"].all())
    scientific = all(
        bool(_read_json(stage / patch / candidate / "report.json")["gate_pass"])
        for patch in patches
    )
    if stable and scientific:
        selected_variant = candidate
    refinement_gate = bool(
        frame.loc[frame["variant"].eq("task_graph_refined"), "gate_pass"].all()
    )
    consensus_report = _read_json(stage / "patch_07/consensus_patch07/report.json")
    return _gate(
        stage / "gate.json",
        {
            "all_diagnostic_patches_stable": bool(patch_pass.all()),
            "task_graph_refinement_stable": refinement_gate,
            "true_beam_not_required": not trigger,
        },
        comparison_count=len(frame),
        patch_gate_by_id={str(key): bool(value) for key, value in patch_pass.items()},
        true_beam_triggered=trigger,
        maximum_alternative_hypothesis_ratio=max(alternative_ratios, default=0.0),
        nested_contrasts=[
            "main_K1_R5_vs_budget_K4_R8",
            "root_dropout",
            "root_order2",
            "task_graph_refinement",
            "patch07_consensus_only",
        ],
        full_factorial_search_executed=False,
        patch07_consensus_gate=bool(consensus_report["gate_pass"]),
        selected_variant=selected_variant,
        selection_rule="least_complexity_among_scientifically_passing_and_stable_variants",
    )


def stage_mechanism_gate(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    retry = _require(config, output_root, "registered_retry")
    stability = _require(config, output_root, "search_stability")
    stage = output_root / STAGE_DIRS["mechanism_gate"]
    selected_variant = str(stability.get("selected_variant", "baseline"))
    if selected_variant == "baseline":
        reports = list(retry["patch_reports"])
    else:
        reports = [
            _read_json(output_root / STAGE_DIRS["search_stability"] / patch / selected_variant / "report.json")
            for patch in map(str, config["diagnostic_patch_ids"])
        ]
    stable_by_patch = stability.get("patch_gate_by_id", {})
    for row in reports:
        row["search_stability_gate"] = bool(stable_by_patch.get(row["patch_id"], False))
        row["final_gate_pass"] = bool(row["gate_pass"] and row["search_stability_gate"])
    passing = sum(bool(row["final_gate_pass"]) for row in reports)
    _write_parquet(pd.DataFrame.from_records(reports), stage / "four_patch_mechanism_reports.parquet")
    return _gate(
        stage / "gate.json",
        {
            "minimum_three_of_four": passing >= int(config["mechanism_gate"]["minimum_passing_patches"]),
            "search_stability": bool(stability.get("gate_pass", False)),
            "true_beam_not_triggered": not bool(stability.get("true_beam_triggered", False)),
        },
        passing_patch_count=passing,
        patch_reports=reports,
        representation_hypothesis="stateless_xyz_only",
        direct_threshold_relaxation_authorized=False,
        selected_variant=selected_variant,
    )


def stage_reach_round7(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require(config, output_root, "inventory")
    stage = output_root / STAGE_DIRS["reach_round7"]
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"Reach Round 7 already sealed: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    paths = _source_paths(config, project_root)
    v142 = paths["v14_2"]
    round6_a = pd.read_parquet(v142 / "04_reach_round6/replica_a_slab_round6.parquet")
    round6_b = pd.read_parquet(v142 / "04_reach_round6/replica_b_slab_round6.parquet")
    previous_report = _read_json(v142 / "04_reach_round6/reach_round6_report.json")
    chunk = 2 ** int(config["reach_round7"]["sobol_power"])
    round5_a, round5_b = round6_a.iloc[:-chunk], round6_b.iloc[:-chunk]
    seeds_a = tuple(map(int, previous_report["replica_a_seed_lineage"]))
    seeds_b = tuple(map(int, previous_report["replica_b_seed_lineage"]))
    legacy_config = legacy.load_config(paths["legacy_config"])
    legacy_paths = legacy._paths(legacy_config, project_root)
    grid = legacy._workspace_grid(legacy_paths)
    environment = legacy._EndpointOnlyForwardAdapter(load_environment(project_root, paths["robot_config"]))
    spec = ReachSamplingRoundSpec(
        int(config["reach_round7"]["sobol_power"]),
        int(config["reach_round7"]["seed_a"]),
        int(config["reach_round7"]["seed_b"]),
    )
    generated = generate_independent_reach_samples(
        environment, np.asarray(environment.bounds, dtype=float), grid=grid, rounds=(spec,), chunk_rows=65536
    )
    def new_frame(xyz: np.ndarray, beta: np.ndarray, source: str) -> pd.DataFrame:
        return pd.DataFrame({
            "x_m": xyz[:, 0], "y_m": xyz[:, 1], "z_m": xyz[:, 2],
            **{name: beta[:, index] for index, name in enumerate(BETA_COLUMNS)},
            "source": source,
        })
    final_a = pd.concat([round6_a, new_frame(generated.xyz_a, generated.beta_a, "replica_a_round_7")], ignore_index=True)
    final_b = pd.concat([round6_b, new_frame(generated.xyz_b, generated.beta_b, "replica_b_round_7")], ignore_index=True)
    rounds = (
        ReachReplicaRound(5, ReachReplica("A", seeds_a[:-1], round5_a[["x_m", "y_m", "z_m"]].to_numpy(float)), ReachReplica("B", seeds_b[:-1], round5_b[["x_m", "y_m", "z_m"]].to_numpy(float))),
        ReachReplicaRound(6, ReachReplica("A", seeds_a, round6_a[["x_m", "y_m", "z_m"]].to_numpy(float)), ReachReplica("B", seeds_b, round6_b[["x_m", "y_m", "z_m"]].to_numpy(float))),
        ReachReplicaRound(7, ReachReplica("A", (*seeds_a, spec.seed_a), final_a[["x_m", "y_m", "z_m"]].to_numpy(float)), ReachReplica("B", (*seeds_b, spec.seed_b), final_b[["x_m", "y_m", "z_m"]].to_numpy(float))),
    )
    reach = config["reach_round7"]
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
        v142 / "04_reach_round6/frontier_inverse_probes_round6.parquet",
    ):
        if not frontier_path.is_file():
            continue
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
    seed_xyz = np.vstack(
        [
            final_a[["x_m", "y_m", "z_m"]].to_numpy(float),
            final_b[["x_m", "y_m", "z_m"]].to_numpy(float),
        ]
    )
    seed_beta = np.vstack(
        [final_a.loc[:, BETA_COLUMNS].to_numpy(float), final_b.loc[:, BETA_COLUMNS].to_numpy(float)]
    )
    new_frontier, frontier_frame = legacy._frontier_probe_round(
        environment,
        frontier_cells,
        grid=grid,
        seed_xyz=seed_xyz,
        seed_beta=seed_beta,
        starts=int(reach["frontier_starts_per_cell"]),
        round_id=7,
    )
    frontier_evidence.extend(new_frontier)
    result = builder.build(
        rounds,
        frontier_evidence=frontier_evidence,
        supplemental_supported_cells=supplemental,
    )
    level = grid.convergence_level_mm
    unions = [grid.cells_for_points(pair.replica_a.xyz_m, level_mm=level) | grid.cells_for_points(pair.replica_b.xyz_m, level_mm=level) for pair in rounds]
    boundary = {
        6: measure_weighted_boundary_change_ratio(unions[1], unions[0]),
        7: measure_weighted_boundary_change_ratio(unions[2], unions[1]),
    }
    metrics = [asdict(item) for item in result.replica_metrics]
    recent = metrics[-2:]
    checks = [
        bool(row["volume_weighted_jaccard"] >= float(reach["weighted_jaccard_min"])
             and row["new_volume_ratio"] <= float(reach["new_volume_ratio_max"])
             and row["frontier_new_volume_ratio"] <= float(reach["frontier_new_volume_ratio_max"])
             and boundary[int(row["round_id"])] <= float(reach["boundary_change_ratio_max"]))
        for row in recent
    ]
    gap = (len(result.proxy_upper_cells) - len(result.proxy_lower_cells)) / max(1, len(result.proxy_upper_cells))
    convergence = bool(all(checks) and gap <= float(reach["lower_upper_measure_gap_max"]))
    _write_parquet(final_a, stage / "replica_a_slab_round7.parquet")
    _write_parquet(final_b, stage / "replica_b_slab_round7.parquet")
    _write_parquet(frontier_frame, stage / "frontier_inverse_probes_round7.parquet")
    report = {
        "reach_convergence_gate": convergence,
        "metrics_round5_7": metrics,
        "measure_weighted_boundary_change_ratio": boundary,
        "recent_round_checks": checks,
        "proxy_lower_cell_count": len(result.proxy_lower_cells),
        "proxy_upper_cell_count": len(result.proxy_upper_cells),
        "lower_upper_measure_gap_ratio": gap,
        "replica_a_seed_lineage": [*seeds_a, spec.seed_a],
        "replica_b_seed_lineage": [*seeds_b, spec.seed_b],
        "frontier_round7_probe_count": len(frontier_frame),
        "frontier_round7_found_count": int(frontier_frame.get("found_valid_inverse", pd.Series(dtype=bool)).sum()),
    }
    _write_json(stage / "reach_round7_report.json", report)
    return _gate(stage / "gate.json", {"two_consecutive_rounds": convergence}, **report)


def stage_confirmation(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    mechanism = _require(config, output_root, "mechanism_gate")
    stage = output_root / STAGE_DIRS["confirmation"]
    if not mechanism.get("gate_pass", False):
        return write_scientific_skip(stage, "four_patch_mechanism_gate_failed")
    if config.get("_patch_id"):
        selected_variant = str(mechanism.get("selected_variant", "baseline"))
        report = _execute_patch(config, project_root, output_root, str(config["_patch_id"]), selected_variant, stage / str(config["_patch_id"]) / selected_variant)
        return {"gate_pass": True, "worker_report": report}
    patches = tuple(map(str, [*config["development_patch_ids"], *config["confirmation_patch_ids"]]))
    diagnostic = set(map(str, config["diagnostic_patch_ids"]))
    selected_variant = str(mechanism.get("selected_variant", "baseline"))
    _run_patch_jobs(config, output_root, "confirmation", [(patch, selected_variant) for patch in patches if patch not in diagnostic])
    reports = []
    for patch in patches:
        if patch in diagnostic:
            if selected_variant == "baseline":
                report = _read_json(output_root / STAGE_DIRS["rooted_baseline"] / patch / "baseline/report.json")
            else:
                report = _read_json(output_root / STAGE_DIRS["search_stability"] / patch / selected_variant / "report.json")
        else:
            report = _read_json(stage / patch / selected_variant / "report.json")
        reports.append(report)
    frame = pd.DataFrame.from_records(reports)
    _write_parquet(frame, stage / "confirmation_patch_reports.parquet")
    development = frame[frame["patch_split"].eq("development")]
    confirmation = frame[frame["patch_split"].eq("confirmation")]
    gate = config["confirmation_gate"]
    checks = {
        "development_pass": int(development["gate_pass"].sum()) >= int(gate["development_pass_min"]),
        "confirmation_pass": int(confirmation["gate_pass"].sum()) >= int(gate["confirmation_pass_min"]),
        "median_largest_component": float(frame.loc[frame["gate_pass"], "largest_stitched_component_ratio"].median()) >= float(gate["largest_component_median_min"]),
        "retained_certificates": bool(frame.loc[frame["gate_pass"], "certificate_gate"].all()),
        "root_budget_saturated": bool(frame["root_budget_saturated"].all()),
        "strong_reference_validated": bool(frame["strong_reference_gate"].all()),
        "tiny_charts_do_not_dominate": bool(
            frame["tiny_primary_measure_ratio"].le(float(gate["tiny_chart_ratio_max"])).all()
        ),
    }
    return _gate(
        stage / "gate.json", checks,
        development_pass_count=int(development["gate_pass"].sum()),
        confirmation_pass_count=int(confirmation["gate_pass"].sum()),
        patch_reports=reports,
        method_frozen_before_confirmation=True,
        patch09_role="diagnostic_only_excluded_from_confirmation_counts",
        patch12_role="sealed_replacement_confirmation",
        threshold_changes_after_confirmation=False,
        selected_variant=selected_variant,
    )


def stage_meso_bridge(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    """Bridge 64-cell evidence to 5k with a sealed connected 512-cell domain."""

    confirmation = _require(config, output_root, "confirmation")
    stage = output_root / STAGE_DIRS["meso_bridge"]
    if not confirmation.get("gate_pass", False):
        return write_scientific_skip(stage, "fresh_confirmation_gate_failed")
    if config.get("_patch_id"):
        report = _execute_patch(
            config,
            project_root,
            output_root,
            str(config["_patch_id"]),
            str(config["_variant"]),
            stage / str(config["_patch_id"]) / str(config["_variant"]),
        )
        return {"gate_pass": True, "worker_report": report}

    input_stage = stage / "input"
    if not (input_stage / "input_manifest.json").is_file():
        paths = _source_paths(config, project_root)
        legacy_config = legacy.load_config(paths["legacy_config"])
        legacy_paths = legacy._paths(legacy_config, project_root)
        parents = pd.read_parquet(legacy_paths["original_parent_nodes"])
        parent_edges = pd.read_parquet(
            paths["original"] / "02_domain_registry/pilot_task_edges.parquet"
        )
        tasks = pd.read_parquet(legacy_paths["original_atlas_tasks"])
        task_edges = pd.read_parquet(legacy_paths["original_atlas_edges"])
        candidates = pd.read_parquet(legacy_paths["original_candidate_clusters"])
        historical = pd.read_parquet(legacy_paths["assignments"])
        replacement = pd.read_parquet(
            output_root / STAGE_DIRS["replacement_confirmation"] / "patch12_assignments.parquet"
        )
        excluded = set(historical["parent_node_id"].astype(int)) | set(
            replacement["parent_node_id"].astype(int)
        )
        adjacency: dict[int, set[int]] = {int(node): set() for node in parents["node_id"]}
        for row in parent_edges.itertuples(index=False):
            left, right = int(row.left_node_id), int(row.right_node_id)
            adjacency[left].add(right); adjacency[right].add(left)
        representative = tasks[tasks["is_representative"].astype(bool)][
            ["task_node_id", "source_parent_node_id"]
        ]
        atlas_candidates = pd.read_parquet(
            paths["original"] / "04_workspace_atlas/atlas_candidates.parquet"
        )
        condition = (
            representative.merge(
                atlas_candidates[atlas_candidates["candidate_id"].eq("capability_exact")][
                    ["task_node_id", "condition_number"]
                ],
                on="task_node_id",
                validate="one_to_one",
            )
            .set_index("source_parent_node_id")["condition_number"]
            .to_dict()
        )
        parent_index = parents.set_index("node_id")
        available = set(adjacency) - excluded
        seed_candidates = sorted(
            available,
            key=lambda node: (
                str(parent_index.loc[node, "strata"]) != "boundary",
                -float(condition.get(node, -math.inf)),
                node,
            ),
        )
        target_count = int(config["meso_bridge"]["parent_cell_count"])
        selected: list[int] | None = None
        seed_node: int | None = None
        for candidate_seed in seed_candidates:
            queue = [candidate_seed]
            seen = {candidate_seed}
            visited: list[int] = []
            while queue and len(visited) < target_count:
                current = queue.pop(0)
                if current not in available:
                    continue
                visited.append(current)
                neighbours = sorted(adjacency[current] - seen)
                seen.update(neighbours)
                queue.extend(neighbours)
            if len(visited) == target_count:
                selected, seed_node = visited, candidate_seed
                break
        if selected is None or seed_node is None:
            raise RuntimeError("unable to seal a connected unused 512-cell meso domain")
        selected_set = set(selected)
        induced_parent_edges = {
            tuple(sorted((node, neighbour)))
            for node in selected_set
            for neighbour in adjacency[node]
            if neighbour in selected_set
        }
        connected_seen: set[int] = set()
        connected_queue = [seed_node]
        while connected_queue:
            current = connected_queue.pop()
            if current in connected_seen:
                continue
            connected_seen.add(current)
            connected_queue.extend((adjacency[current] & selected_set) - connected_seen)
        assignments = pd.DataFrame.from_records(
            [
                {
                    "patch_id": "meso_512",
                    "patch_index": 512,
                    "patch_split": "meso",
                    "parent_node_id": node,
                    "seed_node_id": seed_node,
                    "is_seed": node == seed_node,
                    "seed_role": "boundary_high_condition_meso",
                    "x_tertile": -1,
                    "condition_tertile": -1,
                    "boundary": str(parent_index.loc[node, "strata"]) == "boundary",
                    "retention": str(parent_index.loc[node, "strata"]) == "retention",
                    "tip": str(parent_index.loc[node, "strata"]) == "tip",
                    "existing_cross_edge": False,
                    "no_existing_cross_edge": True,
                }
                for node in selected
            ]
        )
        selected_tasks = tasks[
            tasks["source_parent_node_id"].astype(int).isin(selected_set)
        ].copy()
        selected_tasks["patch_id"] = "meso_512"
        task_ids = set(selected_tasks["task_node_id"].astype(int))
        selected_task_edges = task_edges[
            task_edges["left_node_id"].astype(int).isin(task_ids)
            & task_edges["right_node_id"].astype(int).isin(task_ids)
        ].copy()
        selected_task_edges["patch_id"] = "meso_512"
        selected_candidates = candidates[
            candidates["task_node_id"].astype(int).isin(task_ids)
        ].copy()
        selected_candidates["patch_id"] = "meso_512"
        outputs = {
            "meso_assignments.parquet": assignments,
            "meso_task_nodes.parquet": selected_tasks,
            "meso_task_edges.parquet": selected_task_edges,
            "meso_candidate_clusters.parquet": selected_candidates,
        }
        for filename, frame in outputs.items():
            _write_parquet(frame, input_stage / filename)
        _write_json(
            input_stage / "input_manifest.json",
            {
                "schema_version": 1,
                "source_sha": _git_sha(),
                "selection_before_meso_results": True,
                "seed_node_id": seed_node,
                "parent_cell_count": len(assignments),
                "task_probe_count": len(selected_tasks),
                "historical_overlap_count": len(selected_set & excluded),
                "induced_parent_edge_count": len(induced_parent_edges),
                "connected_parent_count": len(connected_seen),
                "artifacts": [
                    {
                        "path": filename,
                        "bytes": (input_stage / filename).stat().st_size,
                        "sha256": sha256_file(input_stage / filename),
                    }
                    for filename in outputs
                ],
            },
        )

    _run_patch_jobs(
        config,
        output_root,
        "meso_bridge",
        [("meso_512", "baseline"), ("meso_512", "meso_root_dropout")],
    )
    baseline_dir = stage / "meso_512/baseline"
    dropout_dir = stage / "meso_512/meso_root_dropout"
    baseline_report = _read_json(baseline_dir / "report.json")
    dropout_report = _read_json(dropout_dir / "report.json")
    stability = _frame_stability(
        pd.read_parquet(baseline_dir / "primary_atlas.parquet"),
        pd.read_parquet(dropout_dir / "primary_atlas.parquet"),
        config,
        left_directory=baseline_dir,
        right_directory=dropout_dir,
    )
    schedules = pd.read_parquet(baseline_dir / "audit_v2_schedules.parquet").copy()
    schedules["path_length"] = schedules["path_node_ids"].map(len)
    path_schedules = schedules[schedules["audit_kind"].eq("root_path")]
    long_cutoff = max(8, int(path_schedules["path_length"].quantile(0.75))) if len(path_schedules) else 8
    long_ids = set(
        path_schedules.loc[path_schedules["path_length"].ge(long_cutoff), "schedule_id"].astype(str)
    )
    executions = pd.read_parquet(baseline_dir / "audit_v2_executions.parquet")
    long_executions = executions[executions["schedule_id"].astype(str).isin(long_ids)]
    long_gaps = pd.to_numeric(long_executions["geometry_gap_deg"], errors="coerce")
    long_gaps = long_gaps[np.isfinite(long_gaps)]
    nonlocal_cycles = schedules[
        schedules["audit_kind"].eq("fundamental_cycle")
        & schedules["path_length"].ge(long_cutoff)
    ]
    input_manifest = _read_json(input_stage / "input_manifest.json")
    parent_count = int(input_manifest["parent_cell_count"])
    policy = config["meso_bridge"]
    report = {
        "baseline": baseline_report,
        "root_dropout": dropout_report,
        "root_dropout_stability": stability,
        "long_path_cutoff_edges": long_cutoff - 1,
        "long_path_schedule_count": len(long_ids),
        "long_path_geometry_p95_deg": float(np.percentile(long_gaps, 95)) if len(long_gaps) else math.inf,
        "long_path_geometry_max_deg": float(np.max(long_gaps)) if len(long_gaps) else math.inf,
        "nonlocal_cycle_count": len(nonlocal_cycles),
        "runtime_per_parent_s": float(baseline_report["runtime_s"]) / max(1, parent_count),
        "artifact_bytes_per_parent": sum(
            path.stat().st_size for path in baseline_dir.iterdir() if path.is_file()
        ) / max(1, parent_count),
        "parent_cell_count": parent_count,
        "task_probe_count": int(input_manifest["task_probe_count"]),
    }
    _write_json(stage / "meso_bridge_report.json", report)
    return _gate(
        stage / "gate.json",
        {
            "exact_connected_scale": bool(
                parent_count == int(policy["parent_cell_count"])
                and int(input_manifest["connected_parent_count"]) == parent_count
            ),
            "primary_coverage": baseline_report["coverage_ratio"]
            >= float(policy["minimum_primary_coverage"]),
            "largest_coherent_region": baseline_report["largest_stitched_component_ratio"]
            >= float(policy["minimum_largest_coherent_region"]),
            "node_induced_certificate": bool(baseline_report["certificate_gate"]),
            "long_path_geometry": bool(
                len(long_gaps)
                and report["long_path_geometry_p95_deg"]
                <= float(policy["maximum_long_path_p95_deg"])
                and report["long_path_geometry_max_deg"] <= float(policy["maximum_long_path_deg"])
            ),
            "nonlocal_cycle_nonvacuous": len(nonlocal_cycles) > 0,
            "root_dropout_stable": bool(stability["gate_pass"])
            and stability["beta_p95_deg"]
            <= float(policy["maximum_root_dropout_beta_p95_deg"]),
            "resource_scaling_bounded": report["runtime_per_parent_s"]
            <= float(policy["maximum_runtime_per_parent_s"]),
        },
        **report,
    )


def stage_summary(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    stage.mkdir(parents=True, exist_ok=True)
    gates = {
        name: _read_json(output_root / STAGE_DIRS[name] / "gate.json")
        for name in STAGE_ORDER[:-1]
        if (output_root / STAGE_DIRS[name] / "gate.json").is_file()
    }
    artifacts = [
        {"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and not path.is_relative_to(stage)
    ]
    _write_json(stage / "artifact_manifest.json", {"schema_version": 1, "artifact_count": len(artifacts), "artifacts": artifacts})
    scientific_gate_pass = bool(
        gates.get("mechanism_gate", {}).get("gate_pass", False)
        and gates.get("reach_round7", {}).get("gate_pass", False)
        and gates.get("confirmation", {}).get("gate_pass", False)
        and gates.get("meso_bridge", {}).get("gate_pass", False)
    )
    deployment_authorized = bool(
        gates.get("confirmation", {}).get("gate_pass", False)
        and gates.get("meso_bridge", {}).get("gate_pass", False)
    )
    report = {
        "stage_gates": gates,
        "mechanism_gate": bool(gates.get("mechanism_gate", {}).get("gate_pass", False)),
        "reach_round7_gate": bool(gates.get("reach_round7", {}).get("gate_pass", False)),
        "confirmation_gate": bool(gates.get("confirmation", {}).get("gate_pass", False)),
        "meso_bridge_gate": bool(gates.get("meso_bridge", {}).get("gate_pass", False)),
        "operational_completion": True,
        "artifact_manifest_complete": bool(artifacts),
        "scientific_gate_pass": scientific_gate_pass,
        "deployment_authorized": deployment_authorized,
        "v14_3_authorized": deployment_authorized,
        "formal_200k_generation_authorized": False,
    }
    _write_json(stage / "summary_report.json", report)
    payload = {
        "gate_pass": scientific_gate_pass,
        "gate_semantics": "scientific",
        **report,
    }
    _write_json(stage / "gate.json", payload)
    return payload


STAGE_RUNNERS = {
    "inventory": stage_inventory,
    "replacement_confirmation": stage_replacement_confirmation,
    "artifact_diagnostics": stage_artifact_diagnostics,
    "rooted_baseline": stage_rooted_baseline,
    "registered_retry": stage_registered_retry,
    "patch07_local_audit": stage_patch07_local_audit,
    "search_stability": stage_search_stability,
    "mechanism_gate": stage_mechanism_gate,
    "reach_round7": stage_reach_round7,
    "confirmation": stage_confirmation,
    "meso_bridge": stage_meso_bridge,
    "summary": stage_summary,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SOURCE_ROOT / "configs/bacra_v14_2r_stitched_atlas.yaml"))
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--patch-id")
    parser.add_argument("--variant")
    parser.add_argument("--audit-shards-per-patch", type=int)
    parser.add_argument("--audit-token-pool")
    parser.add_argument("--audit-token-count", type=int)
    parser.add_argument("--audit-shard-bundle")
    parser.add_argument("--audit-shard-id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    project_root = project_root_from(SOURCE_ROOT)
    if args.audit_shard_bundle is not None:
        if args.audit_shard_id is None:
            raise SystemExit("--audit-shard-id is required with --audit-shard-bundle")
        report = _run_audit_shard_worker(
            config,
            project_root,
            Path(args.audit_shard_bundle).resolve(),
            int(args.audit_shard_id),
        )
        print(json.dumps(_strict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage is None:
        raise SystemExit("--stage is required outside audit shard worker mode")
    config["_patch_id"] = args.patch_id
    config["_variant"] = args.variant
    config["_audit_shards_per_patch"] = (
        int(args.audit_shards_per_patch)
        if args.audit_shards_per_patch is not None
        else int(config["audit_execution"]["total_shard_workers"])
    )
    if int(config["_audit_shards_per_patch"]) < 1:
        raise SystemExit("--audit-shards-per-patch must be positive")
    config["_audit_token_pool"] = args.audit_token_pool
    config["_audit_token_count"] = (
        int(args.audit_token_count)
        if args.audit_token_count is not None
        else int(config["audit_execution"]["total_shard_workers"])
    )
    if int(config["_audit_token_count"]) < 1:
        raise SystemExit("--audit-token-count must be positive")
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / str(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stage != "inventory":
        _validate_resume_fixed_point(config, output_root)
    top_level_stage = args.patch_id is None
    stage_directory = output_root / STAGE_DIRS[args.stage]
    if top_level_stage:
        completed = _load_validated_stage_result(
            stage_directory, config=config, stage_name=args.stage
        )
        if completed is not None:
            print(json.dumps(_strict(completed), ensure_ascii=False, sort_keys=True))
            return 0
        if (stage_directory / "completion_manifest.json").exists() or (
            stage_directory / "gate.json"
        ).exists():
            raise RuntimeError(
                f"stage has an invalid or incomplete completion closure: {stage_directory}"
            )
    result = STAGE_RUNNERS[args.stage](config, project_root, output_root)
    if top_level_stage:
        _write_stage_completion_manifest(
            stage_directory, config=config, stage_name=args.stage
        )
    print(json.dumps(_strict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
