#!/usr/bin/env python3
"""Branch-Aware Canonical Region Atlas (BACRA) V12 pipeline.

The runner is intentionally thin: numerical and Gate semantics live in deep
teacher modules.  This file owns immutable protocol bootstrap, deterministic
subprocess slicing, checkpointing and artifact persistence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(SOURCE_ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "scripts" / "analysis"))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml
import joblib

from quasi_exp.teacher.bacra_student import (
    StudentGeometry,
    StudentStrategy,
    choose_student_strategy,
    plan_seed_execution,
    train_one_seed,
)
from quasi_exp.teacher.bacra_dataset import (
    SpatialSplitPolicy,
    assign_spatial_splits,
    detect_cross_chart_conflicts,
    local_fill_distance_by_role,
    make_nested_datasets,
)
from quasi_exp.teacher.capability_map import (
    CapabilityPolicy,
    batch_fk,
    batch_jacobian_metrics,
    centerline_component_diagnostics,
    materialize_capability_region_from_samples,
    nested_sobol_beta,
)
from quasi_exp.teacher.atlas_audit import (
    AtlasAuditPolicy,
    PathTrace,
    audit_atlas,
)
from quasi_exp.teacher.canonical import TeacherPolicy, beta_rms_deg
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    CanonicalAtlas,
    CanonicalChart,
    ChartMergeDecision,
    ChartOverlap,
    DirectedContinuationEdge,
    ProductGraph,
    RobustBidirectionalEdge,
    build_canonical_atlas,
    make_predictor_corrector_continuation,
)
from quasi_exp.teacher.topology_task_region import (
    make_strict_gold_witnessed_continuation,
    waypoint_map_from_frame,
)
from quasi_exp.teacher.dense_chart_sampling import (
    DenseSamplingPolicy,
    chart_fill_distance_metrics,
    densify_chart,
    local_label_consistency,
)
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.multi_ik_candidates import (
    CandidatePolicy,
    CandidateQuality,
    solve_candidate_bank,
    stable_cluster_representatives,
)
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


STAGE_DIRS = {
    "protocol": "00_protocol",
    "capability": "01_capability",
    "candidates": "02_candidates",
    "atlas": "03_atlas",
    "audit": "04_audit",
    "dense": "05_dense",
    "dataset": "06_dataset",
    "student": "07_student",
    "evaluation": "08_sealed_evaluation",
    "summary": "13_summary",
}
PROTOCOL_STAGES = tuple(STAGE_DIRS)
TARGET_COLUMNS = ("target_x_m", "target_y_m", "target_z_m")
WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if ".worktrees" in resolved.parts:
        index = resolved.parts.index(".worktrees")
        return Path(*resolved.parts[:index])
    return resolved


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _load_config_with_extends(
    path: Path, *, chain: tuple[Path, ...] = ()
) -> dict[str, Any]:
    """Load an arbitrarily deep config inheritance chain deterministically."""

    config_path = path.resolve()
    if config_path in chain:
        cycle = " -> ".join(str(item) for item in (*chain, config_path))
        raise ValueError(f"BACRA config extends cycle: {cycle}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("BACRA config must contain a mapping")
    if "extends" not in payload:
        return copy.deepcopy(dict(payload))
    parent_path = Path(str(payload["extends"]))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = _load_config_with_extends(
        parent_path, chain=(*chain, config_path)
    )
    override = {key: value for key, value in payload.items() if key != "extends"}
    return _deep_merge(parent, override)


def load_protocol_config(path: str | Path, preset: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = _load_config_with_extends(config_path)
    mode = str(preset).lower()
    if mode not in {"smoke", "pilot", "formal"}:
        raise ValueError("preset must be smoke, pilot or formal")
    merged = dict(payload)
    if mode == "smoke":
        merged = _deep_merge(merged, merged.get("smoke", {}))
    merged.pop("smoke", None)
    if mode == "formal":
        merged["capability"]["pool_rows"] = int(
            merged["capability"]["formal_pool_rows"]
        )
        merged["capability"]["voxel_mm"] = float(
            merged["capability"]["formal_voxel_mm"]
        )
        merged["task_region"]["node_count"] = int(
            merged["task_region"]["formal_node_count"]
        )
        merged["atlas"]["root_count"] = int(merged["atlas"]["formal_root_count"])
        merged["audit"]["endpoint_count"] = int(
            merged["audit"]["formal_endpoint_count"]
        )
        merged["audit"]["paths_per_endpoint"] = int(
            merged["audit"]["formal_paths_per_endpoint"]
        )
        merged["audit"]["loop_count"] = int(merged["audit"]["formal_loop_count"])
        merged["candidates"]["difficult_seed_budget"] = int(
            merged["candidates"]["formal_difficult_seed_budget"]
        )
    merged["preset"] = mode
    merged["config_path"] = str(config_path)
    return merged


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atlas_payload(atlas: CanonicalAtlas) -> dict[str, Any]:
    directed = [
        {
            "source_key": edge.source_key,
            "target_key": edge.target_key,
            "continuation_beta_rad": edge.continuation_beta_rad,
            "match_gap_deg": edge.match_gap_deg,
            "residual_mm": edge.residual_mm,
            "corrector_iterations": edge.corrector_iterations,
            "status": edge.status,
            "minimum_margin_deg": edge.minimum_margin_deg,
            "waypoint_count": edge.waypoint_count,
        }
        for edge in atlas.product_graph.directed_edges
    ]
    robust = [
        {
            "left_key": edge.left_key,
            "right_key": edge.right_key,
            "forward_keys": (edge.forward.source_key, edge.forward.target_key),
            "reverse_keys": (edge.reverse.source_key, edge.reverse.target_key),
            "cost": edge.cost,
            "transition_deg": edge.transition_deg,
            "posture_term": edge.posture_term,
            "condition_term": edge.condition_term,
            "margin_term": edge.margin_term,
        }
        for edge in atlas.product_graph.robust_edges
    ]
    return {
        "task_nodes": [
            {
                "node_id": node.node_id,
                "xyz_m": node.xyz_m,
                "neighbor_node_ids": node.neighbor_node_ids,
                "core_safe": node.core_safe,
            }
            for node in atlas.product_graph.task_nodes
        ],
        "candidates": [
            {
                "node_id": candidate.node_id,
                "candidate_id": candidate.candidate_id,
                "beta_rad": candidate.beta_rad,
                "residual_mm": candidate.residual_mm,
                "min_margin_deg": candidate.min_margin_deg,
                "normalized_min_margin": candidate.normalized_min_margin,
                "posture_cost": candidate.posture_cost,
                "condition_number": candidate.condition_number,
                "quality": candidate.quality,
                "solver_success": candidate.solver_success,
                "actual_bounds": candidate.actual_bounds,
                "cluster_id": candidate.cluster_id,
                "diagnostics": dict(candidate.diagnostics),
            }
            for candidate in atlas.product_graph.candidates
        ],
        "directed": directed,
        "robust": robust,
        "continuation_attempt_count": atlas.product_graph.continuation_attempt_count,
        "rejected_continuation_count": atlas.product_graph.rejected_continuation_count,
        "charts": [
            {
                "chart_id": chart.chart_id,
                "root_key": chart.root_key,
                "selections": chart.selections,
                "metrics": dict(chart.metrics),
                "merged_root_keys": chart.merged_root_keys,
            }
            for chart in atlas.charts
        ],
        "root_node_ids": atlas.root_node_ids,
        "merge_decisions": [row.__dict__ for row in atlas.merge_decisions],
        "overlap_reports": [row.__dict__ for row in atlas.overlap_reports],
        "policy": {
            "edge_match_deg": atlas.policy.edge_match_deg,
            "continuation_residual_max_mm": atlas.policy.continuation_residual_max_mm,
            "root_count": atlas.policy.root_count,
            "icm_max_sweeps": atlas.policy.icm_max_sweeps,
            "top_section_count": atlas.policy.top_section_count,
            "split_gap_deg": atlas.policy.split_gap_deg,
            "merge_overlap_p95_deg": atlas.policy.merge_overlap_p95_deg,
            "merge_overlap_max_deg": atlas.policy.merge_overlap_max_deg,
            "edge_cost_weights": dict(atlas.policy.edge_cost_weights),
        },
    }


def _atlas_from_payload(payload: Mapping[str, Any]) -> CanonicalAtlas:
    task_nodes = tuple(AtlasTaskNode(**row) for row in payload["task_nodes"])
    candidates = tuple(AtlasCandidate(**row) for row in payload["candidates"])
    directed = tuple(
        DirectedContinuationEdge(**row) for row in payload["directed"]
    )
    edge_lookup = {
        (edge.source_key, edge.target_key): edge for edge in directed
    }
    robust = tuple(
        RobustBidirectionalEdge(
            left_key=tuple(row["left_key"]),
            right_key=tuple(row["right_key"]),
            forward=edge_lookup[
                (tuple(row["forward_keys"][0]), tuple(row["forward_keys"][1]))
            ],
            reverse=edge_lookup[
                (tuple(row["reverse_keys"][0]), tuple(row["reverse_keys"][1]))
            ],
            cost=float(row["cost"]),
            transition_deg=float(row["transition_deg"]),
            posture_term=float(row["posture_term"]),
            condition_term=float(row["condition_term"]),
            margin_term=float(row["margin_term"]),
        )
        for row in payload["robust"]
    )
    graph = ProductGraph(
        task_nodes=task_nodes,
        candidates=candidates,
        directed_edges=directed,
        robust_edges=robust,
        continuation_attempt_count=int(payload["continuation_attempt_count"]),
        rejected_continuation_count=int(payload["rejected_continuation_count"]),
    )
    charts = tuple(CanonicalChart(**row) for row in payload["charts"])
    return CanonicalAtlas(
        product_graph=graph,
        charts=charts,
        root_node_ids=tuple(payload["root_node_ids"]),
        merge_decisions=tuple(
            ChartMergeDecision(**row) for row in payload["merge_decisions"]
        ),
        overlap_reports=tuple(
            ChartOverlap(**row) for row in payload["overlap_reports"]
        ),
        policy=AtlasPolicy(**payload["policy"]),
    )


def _gate(
    path: Path,
    checks: Mapping[str, bool],
    *,
    semantics: str,
    **evidence: Any,
) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    output_root = path.parent.parent
    protocol_id = "branch-aware-canonical-region-atlas-v12"
    frozen_config = output_root / STAGE_DIRS["protocol"] / "frozen_config.json"
    if frozen_config.is_file():
        protocol_id = str(
            json.loads(frozen_config.read_text(encoding="utf-8")).get(
                "protocol_id", protocol_id
            )
        )
    capability_lineage: dict[str, Any] = {}
    capability_gate = output_root / STAGE_DIRS["capability"] / "gate.json"
    if capability_gate.is_file() and capability_gate.resolve() != path.resolve():
        source_gate = json.loads(capability_gate.read_text(encoding="utf-8"))
        capability_lineage = {
            "source_capability_gate_sha256": sha256_file(capability_gate),
            "source_capability_bridge_pc_conflict": bool(
                source_gate.get("capability_bridge_pc_conflict", False)
            ),
            "source_predictor_corrector_bridge_pass": source_gate.get(
                "predictor_corrector_bridge_pass"
            ),
            "source_capability_cause_classification": source_gate.get(
                "cause_classification"
            ),
        }
    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "gate_semantics": semantics,
        "claim_scope": "simulation_canonical_atlas_and_student_diagnostics",
        "deployment_claim_gate_pass": False,
        **capability_lineage,
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, payload)
    return payload


def _source_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_centerline(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    candidates = (
        ("target_x_m", "target_y_m", "target_z_m"),
        ("x_target_m", "y_target_m", "z_target_m"),
        ("x_m", "y_m", "z_m"),
    )
    for columns in candidates:
        if set(columns).issubset(frame.columns):
            values = frame[list(columns)].to_numpy(dtype=float)
            if len(values) >= 2 and np.isfinite(values).all():
                return values
    raise ValueError(f"centerline lacks finite xyz columns: {path}")


def _preflight_sources(
    config: Mapping[str, Any], project_root: Path
) -> tuple[dict[str, Path], dict[str, str]]:
    sources = {
        "robot_config": _source_path(project_root, str(config["robot_config"])),
        "formal_gate": _source_path(project_root, str(config["source_formal_gate"])),
        "primary_centerline": _source_path(project_root, str(config["primary_centerline"])),
        "comparison_centerline": _source_path(
            project_root, str(config["comparison_centerline"])
        ),
    }
    expected = dict(config["expected_source_sha256"])
    actual: dict[str, str] = {}
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"required source missing: {name}={path}")
        actual[name] = sha256_file(path)
        if actual[name] != str(expected[name]):
            raise RuntimeError(
                f"source hash mismatch for {name}: {actual[name]} != {expected[name]}"
            )
    return sources, actual


def stage_protocol(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["protocol"]
    stage.mkdir(parents=True, exist_ok=True)
    sources, hashes = _preflight_sources(config, project_root)
    incompatible = _source_path(project_root, str(config["incompatible_legacy_pool"]))
    checks = {
        "source_hashes_match": True,
        "formal_gate_exists": sources["formal_gate"].is_file(),
        "primary_centerline_exists": sources["primary_centerline"].is_file(),
        "legacy_pool_not_reused": not str(incompatible.resolve()).startswith(
            str(output_root.resolve())
        ),
        "deployment_claim_disabled": config["deployment_claim_gate_pass"] is False,
    }
    snapshot = {
        key: value
        for key, value in config.items()
        if key not in {"config_path"}
    }
    atomic_write_json(stage / "frozen_config.json", snapshot)
    runtime = runtime_fingerprint()
    runtime.update(
        {
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "source_worktree": str(SOURCE_ROOT),
            "project_root": str(project_root),
        }
    )
    atomic_write_json(stage / "runtime.json", runtime)
    source_manifest = {
        name: {"path": str(sources[name]), "sha256": hashes[name]}
        for name in sources
    }
    source_manifest["incompatible_legacy_pool"] = {
        "path": str(incompatible),
        "reused": False,
        "reason": "priority-grid/mismatched-bounds pool is not BACRA-V12 standard-domain evidence",
    }
    atomic_write_json(stage / "source_manifest.json", source_manifest)
    return _gate(
        stage / "gate.json",
        checks,
        semantics="frozen_protocol_bootstrap",
        config_sha256=sha256_file(Path(str(config["config_path"]))),
        source_sha256=hashes,
        primary_anchor="A2_178",
        comparison_anchor="A2_143",
    )


def _slice_ranges(count: int, workers: int) -> list[tuple[int, int, int]]:
    effective = max(1, min(int(workers), int(count)))
    boundaries = np.linspace(0, int(count), effective + 1, dtype=np.int64)
    return [
        (index, int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(effective)
        if int(boundaries[index]) < int(boundaries[index + 1])
    ]


def _run_subprocess_tasks(
    commands: Sequence[tuple[str, Sequence[str], Path]],
    *,
    requested_workers: int,
    manifest_path: Path,
) -> dict[str, Any]:
    if not commands:
        raise ValueError("subprocess task list must not be empty")
    effective = min(max(1, int(requested_workers)), len(commands))
    env = os.environ.copy()
    env.update(WORKER_ENV)
    pending = list(commands)
    running: list[tuple[str, subprocess.Popen[str], Path, float, float]] = []
    completed: list[dict[str, Any]] = []
    peak_concurrent_rss_mib = 0.0
    started = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    while pending or running:
        while pending and len(running) < effective:
            task_id, command, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                list(command),
                cwd=SOURCE_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            handle.close()
            running.append((task_id, process, log_path, time.time(), 0.0))
        time.sleep(0.1)
        survivors: list[
            tuple[str, subprocess.Popen[str], Path, float, float]
        ] = []
        concurrent_rss_mib = 0.0
        for task_id, process, log_path, task_started, task_peak_rss_mib in running:
            rss_mib = 0.0
            try:
                status_text = Path(f"/proc/{process.pid}/status").read_text(
                    encoding="utf-8"
                )
                for line in status_text.splitlines():
                    if line.startswith("VmRSS:"):
                        rss_mib = float(line.split()[1]) / 1024.0
                        break
            except (FileNotFoundError, PermissionError, ValueError):
                pass
            task_peak_rss_mib = max(task_peak_rss_mib, rss_mib)
            concurrent_rss_mib += rss_mib
            return_code = process.poll()
            if return_code is None:
                survivors.append(
                    (
                        task_id,
                        process,
                        log_path,
                        task_started,
                        task_peak_rss_mib,
                    )
                )
                continue
            completed.append(
                {
                    "task_id": task_id,
                    "return_code": int(return_code),
                    "wall_time_s": float(time.time() - task_started),
                    "log_path": str(log_path),
                    "peak_rss_mib": float(task_peak_rss_mib),
                }
            )
            if return_code != 0:
                for (
                    _other_id,
                    other,
                    _other_log,
                    _other_started,
                    _other_peak,
                ) in survivors:
                    other.terminate()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(f"subprocess task {task_id} failed:\n{tail}")
        peak_concurrent_rss_mib = max(
            peak_concurrent_rss_mib, concurrent_rss_mib
        )
        running = survivors
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = float(time.time() - started)
    child_cpu = float(
        (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    )
    report = {
        "requested_workers": int(requested_workers),
        "effective_workers": int(effective),
        "task_count": int(len(commands)),
        "wall_time_s": wall,
        "aggregate_child_cpu_time_s": child_cpu,
        "aggregate_cpu_utilization": child_cpu / wall if wall > 0.0 else 0.0,
        "peak_concurrent_rss_mib": float(peak_concurrent_rss_mib),
        "worker_thread_environment": WORKER_ENV,
        "tasks": sorted(completed, key=lambda row: row["task_id"]),
    }
    atomic_write_json(manifest_path, report)
    return report


def capability_worker(args: argparse.Namespace) -> int:
    config = load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    environment = load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )
    beta = np.load(args.beta_npy, mmap_mode="r")[int(args.start) : int(args.stop)]
    xyz = batch_fk(environment, beta, chunk_rows=int(config["capability"]["chunk_rows"]))
    metrics = batch_jacobian_metrics(environment, beta)
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(int(args.start), int(args.stop), dtype=np.int64),
            "x_m": xyz[:, 0],
            "y_m": xyz[:, 1],
            "z_m": xyz[:, 2],
            **metrics,
        }
    )
    _atomic_parquet(frame, Path(args.output))
    return 0


def stage_capability(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["capability"]
    stage.mkdir(parents=True, exist_ok=True)
    policy = CapabilityPolicy.from_mapping(config, project_root=project_root)
    environment = load_environment(project_root, Path(policy.robot_config_path))
    beta = nested_sobol_beta(
        environment.bounds, power=policy.sobol_power, seed=policy.seed
    )
    beta_path = stage / "sobol_beta.npy"
    np.save(beta_path, beta, allow_pickle=False)
    workers = int(config["parallel"]["default_workers"])
    ranges = _slice_ranges(len(beta), workers)
    shard_dir = stage / "shards"
    commands: list[tuple[str, Sequence[str], Path]] = []
    for shard_id, start, stop in ranges:
        output = shard_dir / f"capability_{shard_id:03d}.parquet"
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "_capability-worker",
            "--config",
            str(config["config_path"]),
            "--preset",
            str(config["preset"]),
            "--project-root",
            str(project_root),
            "--beta-npy",
            str(beta_path),
            "--start",
            str(start),
            "--stop",
            str(stop),
            "--output",
            str(output),
        ]
        commands.append(
            (f"capability_{shard_id:03d}", command, shard_dir / f"capability_{shard_id:03d}.log")
        )
    parallel = _run_subprocess_tasks(
        commands,
        requested_workers=workers,
        manifest_path=stage / "parallel_manifest.json",
    )
    fragments = [
        pd.read_parquet(shard_dir / f"capability_{shard_id:03d}.parquet")
        for shard_id, _start, _stop in ranges
    ]
    computed = pd.concat(fragments, ignore_index=True).sort_values(
        "sample_id", kind="stable"
    )
    if computed["sample_id"].tolist() != list(range(len(beta))):
        raise RuntimeError("capability shards are incomplete or out of order")
    sources, _hashes = _preflight_sources(config, project_root)
    centerline = _load_centerline(sources["primary_centerline"])
    manifest = materialize_capability_region_from_samples(
        environment,
        centerline,
        policy,
        beta_rad=beta,
        xyz_m=computed[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        jacobian_metrics={
            name: computed[name].to_numpy(dtype=float)
            for name in ("sigma1_m", "sigma2_m", "sigma3_m", "kappa")
        },
        output_dir=stage,
    )
    anchor_diagnostics = {
        "primary_A2_178": centerline_component_diagnostics(
            manifest.capability_rows,
            centerline,
            roi_radii_mm_desc=policy.roi_radii_mm_desc,
        ),
        "comparison_A2_143": centerline_component_diagnostics(
            manifest.capability_rows,
            _load_centerline(sources["comparison_centerline"]),
            roi_radii_mm_desc=policy.roi_radii_mm_desc,
        ),
    }
    atomic_write_json(stage / "anchor_component_diagnostics.json", anchor_diagnostics)
    gate = manifest.gate.to_dict()
    gate["parallel_manifest_sha256"] = sha256_file(stage / "parallel_manifest.json")
    gate["parallel_evidence"] = parallel
    atomic_write_json(stage / "gate.json", gate)
    return gate


def _candidate_policy(config: Mapping[str, Any]) -> CandidatePolicy:
    values = config["candidates"]
    return CandidatePolicy(
        capability_nearest_count=int(values["nearest_pool_count"]),
        capability_representative_count=int(values["representative_seed_count"]),
        candidate_budget_per_node=int(values["ordinary_seed_budget"]),
        difficult_candidate_budget_per_node=int(values["difficult_seed_budget"]),
        gold_candidate_target=max(1, int(values["ordinary_gold_candidate_median_min"])),
        beta_weights=tuple(map(float, values["beta_weights"])),
        damping=float(values["damping"]),
        max_rms_step_deg=float(values["max_step_deg"]),
        max_corrector_iterations=int(values["max_corrector_iterations"]),
        tracking_tolerance_mm=float(values["residual_p95_mm"]),
        max_residual_mm=float(values["residual_max_mm"]),
        gold_margin_deg=float(values["gold_margin_deg"]),
        candidate_cluster_deg=float(values["cluster_threshold_deg"]),
        cluster_sensitivity_deg=tuple(map(float, values["cluster_sensitivity_deg"])),
        solver_seed=int(config["seeds"]["candidates"]),
    )


def _candidate_record_rows(bank: Any, *, node_offset: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in bank.records:
        global_node = int(record.node_id) + int(node_offset)
        row: dict[str, Any] = {
            "node_id": global_node,
            "candidate_id": f"n{global_node:06d}_{record.candidate_id.split('_', 1)[1]}",
            "source": record.source,
            "solver": record.solver,
            "seed_rank": int(record.seed_rank),
            "residual_mm": float(record.residual_mm),
            "min_margin_deg": float(record.min_margin_deg),
            "normalized_min_margin": float(record.normalized_min_margin),
            "quality": record.quality.value,
            "solver_success": bool(record.solver_success),
            "diagnostics_json": json.dumps(
                dict(record.diagnostics),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ),
        }
        for index, name in enumerate(BETA_COLUMNS):
            row[name] = float(record.beta_rad[index])
        for index, name in enumerate(("achieved_x_m", "achieved_y_m", "achieved_z_m")):
            row[name] = float(record.achieved_xyz_m[index])
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_worker(args: argparse.Namespace) -> int:
    config = load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    environment = load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )
    task = pd.read_parquet(args.task_nodes)
    capability = pd.read_parquet(
        args.capability,
        columns=[*XYZ_COLUMNS, *BETA_COLUMNS],
    )
    start, stop = int(args.start), int(args.stop)
    selected = task.iloc[start:stop]
    targets = selected[list(XYZ_COLUMNS)].to_numpy(dtype=float)
    all_task_beta = task[list(BETA_COLUMNS)].to_numpy(dtype=float)
    edges = pd.read_parquet(args.task_edges)
    neighbour: dict[int, list[np.ndarray]] = {}
    for local, global_node in enumerate(range(start, stop)):
        linked = edges[
            edges["left_node_id"].eq(global_node)
            | edges["right_node_id"].eq(global_node)
        ]
        ids = sorted(
            {
                int(row.right_node_id)
                if int(row.left_node_id) == global_node
                else int(row.left_node_id)
                for row in linked.itertuples(index=False)
            }
        )
        neighbour[local] = [all_task_beta[index] for index in ids]
    difficult = [
        local
        for local, value in enumerate(selected["task_category"].astype(str))
        if value == "difficult"
    ] if "task_category" in selected else []
    bank = solve_candidate_bank(
        environment,
        targets,
        _candidate_policy(config),
        capability_beta_rad=capability[list(BETA_COLUMNS)].to_numpy(dtype=float),
        capability_xyz_m=capability[list(XYZ_COLUMNS)].to_numpy(dtype=float),
        neighbor_beta_rad={
            key: np.asarray(value, dtype=float).reshape(-1, 6)
            for key, value in neighbour.items()
        },
        difficult_node_ids=difficult,
        node_seed_beta_rad={
            local: selected.iloc[local][list(BETA_COLUMNS)].to_numpy(dtype=float)
            for local in range(len(selected))
        },
    )
    frame = _candidate_record_rows(bank, node_offset=start)
    _atomic_parquet(frame, Path(args.output))
    reports = {
        str(int(node_id) + start): {
            **dict(report),
            "node_id": int(node_id) + start,
        }
        for node_id, report in bank.node_reports.items()
    }
    atomic_write_json(Path(args.report), reports)
    return 0


def stage_candidates(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    capability_stage = output_root / STAGE_DIRS["capability"]
    prerequisite = json.loads((capability_stage / "gate.json").read_text(encoding="utf-8"))
    stage = output_root / STAGE_DIRS["candidates"]
    stage.mkdir(parents=True, exist_ok=True)
    candidate_admission = prerequisite.get(
        "candidate_admission_gate_pass", prerequisite.get("gate_pass", False)
    )
    if not bool(candidate_admission):
        return _gate(
            stage / "gate.json",
            {"capability_gate_pass": False},
            semantics="strict_multi_ik_candidate_admission",
            stopped_before_compute=True,
        )
    task_path = capability_stage / "task_nodes.parquet"
    edge_path = capability_stage / "task_edges.parquet"
    strict_seed_path = capability_stage / "strict_gold_seed_pool.parquet"
    capability_path = (
        strict_seed_path
        if strict_seed_path.is_file()
        else capability_stage / "capability_map.parquet"
    )
    task_count = len(pd.read_parquet(task_path, columns=["task_node_id"]))
    workers = int(config["parallel"]["default_workers"])
    ranges = _slice_ranges(task_count, workers)
    shard_dir = stage / "shards"
    commands: list[tuple[str, Sequence[str], Path]] = []
    for shard_id, start, stop in ranges:
        output = shard_dir / f"candidates_{shard_id:03d}.parquet"
        report = shard_dir / f"candidate_nodes_{shard_id:03d}.json"
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "_candidate-worker",
            "--config",
            str(config["config_path"]),
            "--preset",
            str(config["preset"]),
            "--project-root",
            str(project_root),
            "--task-nodes",
            str(task_path),
            "--task-edges",
            str(edge_path),
            "--capability",
            str(capability_path),
            "--start",
            str(start),
            "--stop",
            str(stop),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
        commands.append(
            (f"candidates_{shard_id:03d}", command, shard_dir / f"candidates_{shard_id:03d}.log")
        )
    parallel = _run_subprocess_tasks(
        commands,
        requested_workers=workers,
        manifest_path=stage / "parallel_manifest.json",
    )
    records = pd.concat(
        [
            pd.read_parquet(shard_dir / f"candidates_{shard_id:03d}.parquet")
            for shard_id, _start, _stop in ranges
        ],
        ignore_index=True,
    ).sort_values(["node_id", "candidate_id"], kind="stable")
    reports: dict[str, Any] = {}
    for shard_id, _start, _stop in ranges:
        reports.update(
            json.loads(
                (shard_dir / f"candidate_nodes_{shard_id:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    _atomic_parquet(records, stage / "candidate_bank.parquet")
    node_rows = []
    for node_id in range(task_count):
        group = records[records["node_id"].eq(node_id)]
        node_rows.append(
            {
                "node_id": node_id,
                "candidate_count": int(len(group)),
                "gold_candidate_count": int(group["quality"].eq("Gold").sum()),
                "silver_candidate_count": int(group["quality"].eq("Silver").sum()),
                "reject_candidate_count": int(group["quality"].eq("Reject").sum()),
            }
        )
    nodes = pd.DataFrame(node_rows)
    _atomic_parquet(nodes, stage / "candidate_node_summary.parquet")
    sensitivity: dict[str, Any] = {
        key: value.get("cluster_threshold_sensitivity", {})
        for key, value in sorted(reports.items(), key=lambda item: int(item[0]))
    }
    atomic_write_json(stage / "cluster_sensitivity.json", sensitivity)
    gold_node_ratio = float(nodes["gold_candidate_count"].gt(0).mean())
    ordinary_median = float(nodes["gold_candidate_count"].median())
    checks = {
        "candidate_bank_complete": nodes["candidate_count"].gt(0).all(),
        "gold_node_ratio": gold_node_ratio
        >= float(config["candidates"]["gold_node_ratio_min"]),
        "ordinary_gold_candidate_median": ordinary_median
        >= float(config["candidates"]["ordinary_gold_candidate_median_min"]),
        "no_padded_records": not records["source"].astype(str).str.contains("padding").any(),
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="strict_multi_ik_candidate_admission",
        task_node_count=task_count,
        gold_node_ratio=gold_node_ratio,
        gold_candidate_median=ordinary_median,
        quality_counts=records["quality"].value_counts().sort_index().to_dict(),
        parallel_evidence=parallel,
    )


def _atlas_task_nodes(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[AtlasTaskNode, ...]:
    neighbours: dict[int, set[int]] = {
        int(value): set() for value in nodes["task_node_id"]
    }
    for row in edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        neighbours[left].add(right)
        neighbours[right].add(left)
    return tuple(
        AtlasTaskNode(
            node_id=int(row.task_node_id),
            xyz_m=np.asarray([row.x_m, row.y_m, row.z_m], dtype=float),
            neighbor_node_ids=tuple(sorted(neighbours[int(row.task_node_id)])),
            core_safe=str(getattr(row, "capability_tier", "Core-safe"))
            == "Core-safe",
        )
        for row in nodes.sort_values("task_node_id", kind="stable").itertuples(index=False)
    )


def _atlas_candidates(
    records: pd.DataFrame, bounds: np.ndarray, *, cluster_deg: float
) -> tuple[tuple[AtlasCandidate, ...], pd.DataFrame]:
    strict = records[
        records["quality"].isin(["Gold", "Silver"]) & records["solver_success"].eq(True)
    ].copy()
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    span = np.maximum(bounds[:, 1] - bounds[:, 0], np.finfo(float).eps)
    candidates: list[AtlasCandidate] = []
    inventory: list[dict[str, Any]] = []
    for node_id, group in strict.groupby("node_id", sort=True):
        group = group.sort_values(
            ["quality", "min_margin_deg", "residual_mm", "candidate_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        beta = group[list(BETA_COLUMNS)].to_numpy(dtype=float)
        score = (
            group["residual_mm"].to_numpy(dtype=float)
            - 0.001 * group["min_margin_deg"].to_numpy(dtype=float)
        )
        _representatives, relative = stable_cluster_representatives(
            beta, scores=score, threshold_deg=float(cluster_deg)
        )
        keep = set(map(int, relative))
        for relative_index, row in enumerate(group.itertuples(index=False)):
            selected = relative_index in keep
            reason = "cluster_representative" if selected else "same_cluster"
            diagnostics = json.loads(row.diagnostics_json)
            condition = float(diagnostics.get("kappa", math.inf))
            if not math.isfinite(condition):
                selected = False
                reason = "nonfinite_condition"
            inventory.append(
                {
                    "node_id": int(node_id),
                    "candidate_id": str(row.candidate_id),
                    "selected_for_product_graph": bool(selected),
                    "selection_reason": reason,
                }
            )
            if not selected:
                continue
            values = np.asarray(
                [getattr(row, name) for name in BETA_COLUMNS], dtype=float
            )
            posture = float(np.mean(np.square((values - midpoint) / span)))
            candidates.append(
                AtlasCandidate(
                    node_id=int(node_id),
                    candidate_id=str(row.candidate_id),
                    beta_rad=values,
                    residual_mm=float(row.residual_mm),
                    min_margin_deg=float(row.min_margin_deg),
                    normalized_min_margin=float(row.normalized_min_margin),
                    posture_cost=posture,
                    condition_number=condition,
                    quality=str(row.quality),
                    solver_success=True,
                    actual_bounds=bool(
                        np.all(values >= bounds[:, 0])
                        and np.all(values <= bounds[:, 1])
                    ),
                    diagnostics=diagnostics,
                )
            )
    candidates.sort(key=lambda value: value.key)
    return tuple(candidates), pd.DataFrame(inventory)


def _atlas_policy(config: Mapping[str, Any]) -> AtlasPolicy:
    atlas = config["atlas"]
    return AtlasPolicy(
        edge_match_deg=float(atlas["edge_match_deg"]),
        continuation_residual_max_mm=float(config["candidates"]["residual_max_mm"]),
        root_count=int(atlas["root_count"]),
        icm_max_sweeps=int(atlas["icm_max_sweeps"]),
        top_section_count=int(atlas["top_section_count"]),
        split_gap_deg=float(atlas["split_gap_deg"]),
        merge_overlap_p95_deg=float(atlas["merge_overlap_p95_deg"]),
        merge_overlap_max_deg=float(config["audit"]["common_max_deg"]),
        edge_cost_weights=dict(atlas["edge_weights"]),
    )


def _continuation_for_output(
    config: Mapping[str, Any],
    output_root: Path,
    environment: Any,
) -> Callable[..., Any]:
    capability_stage = output_root / STAGE_DIRS["capability"]
    waypoint_path = capability_stage / "task_edge_waypoints.parquet"
    edge_path = capability_stage / "task_edges.parquet"
    if waypoint_path.is_file() and edge_path.is_file():
        waypoints = waypoint_map_from_frame(
            pd.read_parquet(edge_path),
            pd.read_parquet(waypoint_path),
        )
        return make_strict_gold_witnessed_continuation(
            environment,
            waypoints,
            damping=float(config["candidates"]["damping"]),
            beta_weights=tuple(map(float, config["candidates"]["beta_weights"])),
            max_corrector_iterations=int(
                config["candidates"]["max_corrector_iterations"]
            ),
            residual_max_mm=float(config["candidates"]["residual_max_mm"]),
            gold_margin_deg=float(config["candidates"]["gold_margin_deg"]),
        )
    return make_predictor_corrector_continuation(
        environment,
        damping=float(config["candidates"]["damping"]),
        beta_weights=tuple(map(float, config["candidates"]["beta_weights"])),
        max_corrector_iterations=int(config["candidates"]["max_corrector_iterations"]),
        residual_tolerance_mm=float(config["candidates"]["residual_max_mm"]),
    )


def _persist_atlas(atlas: CanonicalAtlas, stage: Path) -> None:
    directed_rows = [
        {
            "source_node_id": edge.source_key[0],
            "source_candidate_id": edge.source_key[1],
            "target_node_id": edge.target_key[0],
            "target_candidate_id": edge.target_key[1],
            "match_gap_deg": edge.match_gap_deg,
            "residual_mm": edge.residual_mm,
            "corrector_iterations": edge.corrector_iterations,
            "status": edge.status,
            "minimum_margin_deg": edge.minimum_margin_deg,
            "waypoint_count": edge.waypoint_count,
        }
        for edge in atlas.product_graph.directed_edges
    ]
    robust_rows = [
        {
            "left_node_id": edge.left_key[0],
            "left_candidate_id": edge.left_key[1],
            "right_node_id": edge.right_key[0],
            "right_candidate_id": edge.right_key[1],
            "cost": edge.cost,
            "transition_deg": edge.transition_deg,
            "posture_term": edge.posture_term,
            "condition_term": edge.condition_term,
            "margin_term": edge.margin_term,
        }
        for edge in atlas.product_graph.robust_edges
    ]
    chart_rows: list[dict[str, Any]] = []
    candidate_by_key = atlas.product_graph.candidate_by_key
    node_by_id = atlas.product_graph.node_by_id
    for chart in atlas.charts:
        for node_id, candidate_id in chart.selections:
            candidate = candidate_by_key[(node_id, candidate_id)]
            node = node_by_id[node_id]
            chart_rows.append(
                {
                    "chart_id": chart.chart_id,
                    "root_node_id": chart.root_key[0],
                    "root_candidate_id": chart.root_key[1],
                    "node_id": node_id,
                    "candidate_id": candidate_id,
                    "x_m": float(node.xyz_m[0]),
                    "y_m": float(node.xyz_m[1]),
                    "z_m": float(node.xyz_m[2]),
                    **{
                        name: float(candidate.beta_rad[index])
                        for index, name in enumerate(BETA_COLUMNS)
                    },
                    "quality": candidate.quality,
                    "residual_mm": candidate.residual_mm,
                    "min_margin_deg": candidate.min_margin_deg,
                }
            )
    overlap_rows = [
        {
            "chart_a_id": row.chart_a_id,
            "chart_b_id": row.chart_b_id,
            "shared_node_count": len(row.shared_node_ids),
            "shared_node_ids_json": json.dumps(row.shared_node_ids),
            "gap_p95_deg": row.gap_p95_deg,
            "gap_max_deg": row.gap_max_deg,
            "resolution": row.resolution,
        }
        for row in atlas.overlap_reports
    ]
    coverage_rows = [
        {
            "chart_id": chart.chart_id,
            "node_count": len(chart.selections),
            **dict(chart.metrics),
        }
        for chart in atlas.charts
    ]
    _atomic_parquet(pd.DataFrame(directed_rows), stage / "directed_edges.parquet")
    _atomic_parquet(pd.DataFrame(robust_rows), stage / "robust_edges.parquet")
    _atomic_parquet(pd.DataFrame(chart_rows), stage / "canonical_charts.parquet")
    pd.DataFrame(overlap_rows).to_csv(stage / "chart_overlap_report.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(stage / "chart_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "kept_root_key": json.dumps(row.kept_root_key),
                "merged_root_key": json.dumps(row.merged_root_key),
                "shared_node_count": row.shared_node_count,
                "gap_p95_deg": row.gap_p95_deg,
                "gap_max_deg": row.gap_max_deg,
                "resolution": row.resolution,
            }
            for row in atlas.merge_decisions
        ]
    ).to_csv(stage / "chart_merge_decisions.csv", index=False)


def build_atlas_in_memory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> CanonicalAtlas:
    environment = load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )
    capability_stage = output_root / STAGE_DIRS["capability"]
    nodes = pd.read_parquet(capability_stage / "task_nodes.parquet")
    edges = pd.read_parquet(capability_stage / "task_edges.parquet")
    records = pd.read_parquet(
        output_root / STAGE_DIRS["candidates"] / "candidate_bank.parquet"
    )
    task_nodes = _atlas_task_nodes(nodes, edges)
    candidates, _inventory = _atlas_candidates(
        records,
        environment.bounds,
        cluster_deg=float(config["candidates"]["cluster_threshold_deg"]),
    )
    continuation = _continuation_for_output(config, output_root, environment)
    return build_canonical_atlas(
        task_nodes, candidates, continuation, policy=_atlas_policy(config)
    )


def stage_atlas(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["atlas"]
    stage.mkdir(parents=True, exist_ok=True)
    candidate_gate = json.loads(
        (output_root / STAGE_DIRS["candidates"] / "gate.json").read_text(
            encoding="utf-8"
        )
    )
    if not bool(candidate_gate["gate_pass"]):
        return _gate(
            stage / "gate.json",
            {"candidate_gate_pass": False},
            semantics="canonical_product_graph_admission",
            stopped_before_compute=True,
        )
    environment = load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )
    records = pd.read_parquet(
        output_root / STAGE_DIRS["candidates"] / "candidate_bank.parquet"
    )
    _candidates, inventory = _atlas_candidates(
        records,
        environment.bounds,
        cluster_deg=float(config["candidates"]["cluster_threshold_deg"]),
    )
    _atomic_parquet(inventory, stage / "product_graph_candidate_inventory.parquet")
    atlas = build_atlas_in_memory(config, project_root, output_root)
    _atomic_pickle(_atlas_payload(atlas), stage / "atlas.pkl")
    _persist_atlas(atlas, stage)
    task_count = len(atlas.product_graph.task_nodes)
    coverage = [
        len(chart.selections) / task_count for chart in atlas.charts
    ] if task_count else []
    checks = {
        "product_graph_has_directed_edges": len(atlas.product_graph.directed_edges) > 0,
        "product_graph_has_robust_edges": len(atlas.product_graph.robust_edges) > 0,
        "canonical_chart_exists": len(atlas.charts) > 0,
        "chart_coverage": bool(coverage)
        and max(coverage) >= float(config["atlas"]["minimum_chart_coverage"]),
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="canonical_product_graph_admission",
        task_node_count=task_count,
        candidate_count=len(atlas.product_graph.candidates),
        continuation_attempt_count=atlas.product_graph.continuation_attempt_count,
        rejected_continuation_count=atlas.product_graph.rejected_continuation_count,
        directed_edge_count=len(atlas.product_graph.directed_edges),
        robust_edge_count=len(atlas.product_graph.robust_edges),
        chart_count=len(atlas.charts),
        chart_coverage=coverage,
        root_node_ids=list(atlas.root_node_ids),
    )


def _fresh_path_executors(
    atlas: CanonicalAtlas, continuation: Callable[..., Any]
) -> tuple[Callable[..., PathTrace], Callable[..., Sequence[PathTrace]]]:
    nodes = atlas.product_graph.node_by_id
    candidates = atlas.product_graph.candidate_by_key

    def execute(chart: CanonicalChart, path: tuple[int, ...]) -> PathTrace:
        if not path or any(node_id not in chart.selection_by_node for node_id in path):
            return PathTrace(path, np.zeros((len(path), 6)), False, "unknown_chart_node")
        first = candidates[chart.selection_by_node[path[0]]]
        values = [first.beta_rad.copy()]
        previous = first
        for node_id in path[1:]:
            outcome = continuation(previous, nodes[node_id])
            if not outcome.success or not outcome.actual_bounds:
                failed_prefix = path[: len(values) + 1]
                return PathTrace(
                    failed_prefix,
                    np.vstack(values + [outcome.beta_rad]),
                    False,
                    outcome.status,
                )
            values.append(outcome.beta_rad.copy())
            reference = candidates[chart.selection_by_node[node_id]]
            previous = AtlasCandidate(
                node_id=node_id,
                candidate_id=f"fresh:{reference.candidate_id}",
                beta_rad=outcome.beta_rad,
                residual_mm=outcome.residual_mm,
                min_margin_deg=reference.min_margin_deg,
                normalized_min_margin=reference.normalized_min_margin,
                posture_cost=reference.posture_cost,
                condition_number=reference.condition_number,
                quality=reference.quality,
                solver_success=True,
                actual_bounds=True,
            )
        return PathTrace(path, np.vstack(values), True, "fresh_predictor_corrector")

    def repeat(
        chart: CanonicalChart, path: tuple[int, ...]
    ) -> Sequence[PathTrace]:
        return (execute(chart, path), execute(chart, path))

    return execute, repeat


def _audit_policy(config: Mapping[str, Any]) -> AtlasAuditPolicy:
    audit = config["audit"]
    return AtlasAuditPolicy(
        endpoint_count=int(audit["endpoint_count"]),
        paths_per_endpoint=int(audit["paths_per_endpoint"]),
        loop_count=int(audit["loop_count"]),
        path_p95_deg=float(audit["path_p95_deg"]),
        loop_p95_deg=float(audit["loop_p95_deg"]),
        direction_p95_deg=float(audit["direction_p95_deg"]),
        overlap_p95_deg=float(audit["overlap_p95_deg"]),
        repeat_p95_deg=float(audit["repeat_p95_deg"]),
        common_max_deg=float(audit["common_max_deg"]),
    )


def stage_audit(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["audit"]
    stage.mkdir(parents=True, exist_ok=True)
    atlas_gate = json.loads(
        (output_root / STAGE_DIRS["atlas"] / "gate.json").read_text(encoding="utf-8")
    )
    if not bool(atlas_gate["gate_pass"]):
        return _gate(
            stage / "gate.json",
            {"atlas_gate_pass": False},
            semantics="atlas_path_independence_and_representation",
            stopped_before_compute=True,
            static_inverse_authorized=False,
            chart_conditioned_inverse_authorized=False,
        )
    atlas = _atlas_from_payload(
        joblib.load(output_root / STAGE_DIRS["atlas"] / "atlas.pkl")
    )
    environment = load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )
    continuation = _continuation_for_output(config, output_root, environment)
    path_executor, repeat_executor = _fresh_path_executors(atlas, continuation)
    report = audit_atlas(
        atlas,
        policy=_audit_policy(config),
        path_executor=path_executor,
        repeat_executor=repeat_executor,
    )
    representation = report.representation_decision
    payload = {
        "schema_version": 1,
        "protocol_id": str(config["protocol_id"]),
        "gate_semantics": "atlas_path_independence_and_representation",
        "claim_scope": str(config["claim_scope"]),
        "deployment_claim_gate_pass": False,
        "path": report.path.as_dict(),
        "loop": report.loop.as_dict(),
        "direction": report.direction.as_dict(),
        "repeat": report.repeat.as_dict(),
        "overlap": report.overlap.as_dict(),
        "multi_chart_overlap_valid": report.multi_chart_overlap_valid,
        "representation_decision": representation,
        "static_inverse_authorized": bool(
            report.gate_pass and representation == "static_xyz_to_beta6"
        ),
        "chart_conditioned_inverse_authorized": bool(
            report.gate_pass
            and representation == "multi_chart_xyz_chart_id_to_beta6"
        ),
        "stateful_inverse_recommended": representation
        == "stateful_xyz_beta_prev_to_delta_beta",
        "endpoint_count": report.endpoint_count,
        "loop_count": report.loop_count,
        "evidence_limitations": list(report.evidence_limitations),
        "checks": dict(report.checks),
        "gate_pass": bool(report.gate_pass),
    }
    atomic_write_json(stage / "gate.json", payload)
    pd.DataFrame(
        [
            {"audit": name, **getattr(report, name).as_dict()}
            for name in ("path", "loop", "direction", "repeat", "overlap")
        ]
    ).to_csv(stage / "audit_metrics.csv", index=False)
    return payload


def stage_dense(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["dense"]
    stage.mkdir(parents=True, exist_ok=True)
    audit_gate = json.loads(
        (output_root / STAGE_DIRS["audit"] / "gate.json").read_text(encoding="utf-8")
    )
    if not bool(audit_gate["gate_pass"]):
        return _gate(
            stage / "gate.json",
            {"atlas_audit_gate_pass": False},
            semantics="dense_chart_dataset_admission",
            stopped_before_compute=True,
        )
    chart_frame = pd.read_parquet(
        output_root / STAGE_DIRS["atlas"] / "canonical_charts.parquet"
    )
    environment = load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )
    chart_counts = (
        chart_frame.groupby("chart_id", sort=True).size().sort_values(ascending=False)
    )
    requested = int(config["dense"]["row_count"])
    allocations: dict[Any, int] = {}
    remaining = requested
    for index, (chart_id, count) in enumerate(chart_counts.items()):
        allocation = (
            remaining
            if index == len(chart_counts) - 1
            else max(1, int(round(requested * int(count) / int(chart_counts.sum()))))
        )
        allocation = min(remaining, allocation)
        allocations[chart_id] = allocation
        remaining -= allocation
    rows: list[pd.DataFrame] = []
    attempts: list[pd.DataFrame] = []
    reports: list[Mapping[str, Any]] = []
    for offset, (chart_id, allocation) in enumerate(allocations.items()):
        if allocation <= 0:
            continue
        chart = chart_frame[chart_frame["chart_id"].eq(chart_id)]
        candidate_policy = TeacherPolicy(
            damping=float(config["candidates"]["damping"]),
            beta_weights=tuple(map(float, config["candidates"]["beta_weights"])),
            max_corrector_iterations=int(
                config["candidates"]["max_corrector_iterations"]
            ),
            tracking_tolerance_mm=float(config["candidates"]["residual_p95_mm"]),
            safe_joint_margin_deg=float(config["candidates"]["gold_margin_deg"]),
            safe_margin_repulsion_step_deg=0.0,
            solver_seed=int(config["seeds"]["dense"]) + offset,
            max_step_deg=float(config["candidates"]["max_step_deg"]),
        )
        result = densify_chart(
            environment,
            chart[list(XYZ_COLUMNS)].to_numpy(dtype=float),
            chart[list(BETA_COLUMNS)].to_numpy(dtype=float),
            chart_id=chart_id,
            policy=DenseSamplingPolicy(
                row_count=allocation,
                attempt_multiplier=int(config["dense"]["attempt_multiplier"]),
                dual_anchor_gap_deg=float(config["dense"]["dual_anchor_gap_deg"]),
                residual_max_mm=float(config["candidates"]["residual_max_mm"]),
                gold_margin_deg=float(config["candidates"]["gold_margin_deg"]),
                seed=int(config["seeds"]["dense"]) + offset,
                candidate_policy=candidate_policy,
            ),
        )
        rows.append(result.rows)
        attempts.append(result.attempts)
        reports.append({"chart_id": chart_id, **dict(result.report)})
    dense = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    attempt_frame = (
        pd.concat(attempts, ignore_index=True) if attempts else pd.DataFrame()
    )
    _atomic_parquet(dense, stage / "dense_dataset.parquet")
    _atomic_parquet(attempt_frame, stage / "dense_attempts.parquet")
    task = pd.read_parquet(
        output_root / STAGE_DIRS["capability"] / "task_nodes.parquet"
    )
    if len(dense):
        fill = chart_fill_distance_metrics(
            dense[list(XYZ_COLUMNS)].to_numpy(dtype=float),
            task[list(XYZ_COLUMNS)].to_numpy(dtype=float),
        )
        local = local_label_consistency(
            dense[list(XYZ_COLUMNS)].to_numpy(dtype=float),
            dense[list(BETA_COLUMNS)].to_numpy(dtype=float),
        )
        conflicts, conflict_report = detect_cross_chart_conflicts(
            dense,
            voxel_mm=float(config["dense"]["conflict_voxel_mm"]),
            beta_gap_deg=float(config["dense"]["conflict_beta_deg"]),
        )
    else:
        fill = {"nn_p95_mm": math.inf, "nn_max_mm": math.inf}
        local = {
            "local_5mm_beta_gap_p95_deg": math.inf,
            "local_10mm_beta_gap_p95_deg": math.inf,
        }
        conflicts = pd.DataFrame()
        conflict_report = {"gate_pass": False, "conflict_voxel_ratio": math.inf}
    _atomic_parquet(conflicts, stage / "cross_chart_conflicts.parquet")
    total_attempts = sum(int(report["attempt_count"]) for report in reports)
    acceptance = len(dense) / total_attempts if total_attempts else 0.0
    checks = {
        "requested_rows_complete": len(dense) == requested,
        "dense_acceptance_ratio": acceptance
        >= float(config["dense"]["acceptance_ratio_min"]),
        "fill_distance_p95": fill["nn_p95_mm"]
        <= float(config["dense"]["nn_p95_mm"]),
        "fill_distance_max": fill["nn_max_mm"]
        <= float(config["dense"]["nn_max_mm"]),
        "local_5mm_consistency": local["local_5mm_beta_gap_p95_deg"]
        <= float(config["dense"]["local_5mm_beta_p95_deg"]),
        "local_10mm_consistency": local["local_10mm_beta_gap_p95_deg"]
        <= float(config["dense"]["local_10mm_beta_p95_deg"]),
        "cross_chart_conflict_free": bool(conflict_report["gate_pass"]),
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="dense_chart_dataset_admission",
        requested_rows=requested,
        accepted_rows=len(dense),
        attempt_count=total_attempts,
        acceptance_ratio=acceptance,
        chart_reports=reports,
        fill_distance=fill,
        local_consistency=local,
        conflict_report=conflict_report,
    )


def _standardize_v11_surface(frame: pd.DataFrame, source_id: str) -> pd.DataFrame:
    required = {
        *(f"teacher_beta{index}_rad" for index in range(1, 7)),
        "target_x_m",
        "target_y_m",
        "target_z_m",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V11 tube surface missing columns: {missing}")
    output = frame.rename(
        columns={
            "target_x_m": "x_m",
            "target_y_m": "y_m",
            "target_z_m": "z_m",
            **{
                f"teacher_beta{index}_rad": f"beta{index}_rad"
                for index in range(1, 7)
            },
        }
    ).copy()
    output["source_id"] = str(source_id)
    if "chart_id" not in output:
        output["chart_id"] = 0
    return output


def _dataset_roi_radius_mm(
    config: Mapping[str, Any], capability_gate: Mapping[str, Any]
) -> tuple[float, str]:
    """Resolve the frozen task-region radius without depending on Gate shape.

    V12/V12.1 materialized this value in the capability Gate.  V12.2 and later
    bootstrap an already frozen topology task region, so their adapter Gates do
    not own the value; the protocol's ``topology_task_region`` block does.
    """

    gate_value = capability_gate.get("selected_roi_radius_mm")
    if gate_value is not None:
        value = float(gate_value)
        source = "capability_gate.selected_roi_radius_mm"
    else:
        topology = config.get("topology_task_region")
        if not isinstance(topology, Mapping) or topology.get("roi_mm") is None:
            raise KeyError(
                "dataset ROI radius is absent from both capability Gate "
                "and topology_task_region.roi_mm"
            )
        value = float(topology["roi_mm"])
        source = "frozen_config.topology_task_region.roi_mm"
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"dataset ROI radius must be finite and positive: {value}")
    return value, source


def _ablation_frames(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    capability_stage = output_root / STAGE_DIRS["capability"]
    candidate_stage = output_root / STAGE_DIRS["candidates"]
    dense_stage = output_root / STAGE_DIRS["dense"]
    capability = pd.read_parquet(capability_stage / "capability_map.parquet")
    task = pd.read_parquet(capability_stage / "task_nodes.parquet")
    records = pd.read_parquet(candidate_stage / "candidate_bank.parquet")
    d3 = pd.read_parquet(dense_stage / "dense_dataset.parquet")

    sources, _hashes = _preflight_sources(config, project_root)
    centerline = _load_centerline(sources["primary_centerline"])
    capability_gate = json.loads(
        (capability_stage / "gate.json").read_text(encoding="utf-8")
    )
    radius_mm, radius_source = _dataset_roi_radius_mm(
        config, capability_gate
    )
    distance_to_centerline = cKDTree(centerline).query(
        capability[list(XYZ_COLUMNS)].to_numpy(dtype=float), k=1
    )[0]
    d0 = capability[
        capability["capability_tier"].eq("Core-safe")
        & (distance_to_centerline <= radius_mm / 1000.0)
    ][[*XYZ_COLUMNS, *BETA_COLUMNS]].copy()
    d0["chart_id"] = "mixed"

    strict = records[records["quality"].eq("Gold")].sort_values(
        ["node_id", "min_margin_deg", "residual_mm", "candidate_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    best = strict.drop_duplicates("node_id", keep="first")
    d1 = task[["task_node_id", *XYZ_COLUMNS]].merge(
        best[["node_id", *BETA_COLUMNS]],
        left_on="task_node_id",
        right_on="node_id",
        how="inner",
        validate="one_to_one",
    )
    d1["chart_id"] = "pointwise"

    tube_root = (
        project_root
        / "runs/generalized_ellipse_region_v11_full_loop_downstream/02_tube/anchors"
    )
    tube_paths = sorted(tube_root.glob("*/screen/*/surface.parquet"))
    if not tube_paths:
        raise FileNotFoundError("registered V11.4 trajectory/tube surfaces are missing")
    d2_parts = [
        _standardize_v11_surface(pd.read_parquet(path), path.parent.as_posix())
        for path in tube_paths
    ]
    d2 = pd.concat(d2_parts, ignore_index=True)
    tube_distance = cKDTree(centerline).query(
        d2[list(XYZ_COLUMNS)].to_numpy(dtype=float), k=1
    )[0]
    d2 = d2[tube_distance <= radius_mm / 1000.0].copy()
    d2["chart_id"] = "trajectory_tube"

    methods = {"D0": d0, "D1": d1, "D2": d2, "D3": d3}
    common_count = min(len(value) for value in methods.values())
    if common_count < 4:
        raise RuntimeError(f"ablation common sample count is too small: {common_count}")
    selected: dict[str, pd.DataFrame] = {}
    seed = int(config["seeds"]["split"])
    for offset, name in enumerate(sorted(methods)):
        frame = methods[name].reset_index(drop=True)
        order = np.random.default_rng(seed + offset).permutation(len(frame))
        selected[name] = frame.iloc[order[:common_count]].copy().reset_index(drop=True)
        selected[name]["data_method"] = name
    evidence = {
        "available_row_counts": {name: int(len(value)) for name, value in methods.items()},
        "equal_ablation_row_count": int(common_count),
        "dataset_roi_radius_mm": float(radius_mm),
        "dataset_roi_radius_source": radius_source,
        "tube_sources": [str(path) for path in tube_paths],
        "tube_source_sha256": {str(path): sha256_file(path) for path in tube_paths},
    }
    return selected, evidence


def stage_dataset(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["dataset"]
    stage.mkdir(parents=True, exist_ok=True)
    audit_gate = json.loads(
        (output_root / STAGE_DIRS["audit"] / "gate.json").read_text(encoding="utf-8")
    )
    dense_gate = json.loads(
        (output_root / STAGE_DIRS["dense"] / "gate.json").read_text(encoding="utf-8")
    )
    if not (bool(audit_gate["gate_pass"]) and bool(dense_gate["gate_pass"])):
        return _gate(
            stage / "gate.json",
            {
                "atlas_audit_gate_pass": bool(audit_gate["gate_pass"]),
                "dense_gate_pass": bool(dense_gate["gate_pass"]),
            },
            semantics="bacra_dataset_admission",
            stopped_before_compute=True,
        )
    methods, ablation_evidence = _ablation_frames(config, project_root, output_root)
    policy = SpatialSplitPolicy(
        macro_voxel_mm=float(config["split"]["macro_voxel_mm"]),
        train_fraction=float(config["split"]["train_fraction"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
        sealed_fraction=float(config["split"]["sealed_fraction"]),
        buffer_mm=float(config["split"]["buffer_mm"]),
        seed=int(config["seeds"]["split"]),
    )
    split_rows: list[pd.DataFrame] = []
    split_manifests: list[pd.DataFrame] = []
    seal_tokens: dict[str, str] = {}
    method_reports: list[dict[str, Any]] = []
    for method, frame in sorted(methods.items()):
        result = assign_spatial_splits(frame, policy)
        method_dir = stage / method
        method_dir.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(result.public_rows, method_dir / "public_dataset.parquet")
        _atomic_parquet(result._sealed_rows, method_dir / "sealed_dataset.parquet")
        manifest = result.manifest.copy()
        manifest["data_method"] = method
        _atomic_parquet(manifest, method_dir / "spatial_split_manifest.parquet")
        seal_tokens[method] = result.seal_token
        split_rows.append(result.public_rows.assign(data_method=method))
        split_manifests.append(manifest)
        role_counts = (
            result.public_rows["split_role"].value_counts().sort_index().to_dict()
        )
        method_reports.append(
            {
                "data_method": method,
                "public_row_count": int(len(result.public_rows)),
                "sealed_row_count": int(len(result._sealed_rows)),
                "role_counts": role_counts,
            }
        )
    full_d3_report: dict[str, Any] | None = None
    if bool(config["student"].get("use_full_d3_dataset", False)):
        full_d3 = pd.read_parquet(
            output_root
            / STAGE_DIRS["dense"]
            / "dense_dataset.parquet"
        )
        full_result = assign_spatial_splits(full_d3, policy)
        full_dir = stage / "D3_full"
        full_dir.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(
            full_result.public_rows,
            full_dir / "public_dataset.parquet",
        )
        _atomic_parquet(
            full_result._sealed_rows,
            full_dir / "sealed_dataset.parquet",
        )
        full_manifest = full_result.manifest.copy()
        full_manifest["data_method"] = "D3_full"
        _atomic_parquet(
            full_manifest,
            full_dir / "spatial_split_manifest.parquet",
        )
        seal_tokens["D3_full"] = full_result.seal_token
        full_role_counts = (
            full_result.public_rows["split_role"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        full_d3_report = {
            "source_row_count": int(len(full_d3)),
            "public_row_count": int(len(full_result.public_rows)),
            "sealed_row_count": int(len(full_result._sealed_rows)),
            "role_counts": full_role_counts,
            "sealed_not_in_public": bool(
                not full_result.public_rows["split_base_role"]
                .eq("sealed_test")
                .any()
            ),
        }
    d3_public = split_rows[sorted(methods).index("D3")]
    conflicts, conflict_report = detect_cross_chart_conflicts(
        d3_public,
        voxel_mm=float(config["dense"]["conflict_voxel_mm"]),
        beta_gap_deg=float(config["dense"]["conflict_beta_deg"]),
    )
    _atomic_parquet(conflicts, stage / "cross_chart_conflicts.parquet")
    _atomic_parquet(
        pd.concat(split_manifests, ignore_index=True),
        stage / "spatial_split_manifest.parquet",
    )
    fill_report = local_fill_distance_by_role(d3_public)
    _atomic_parquet(fill_report, stage / "fill_distance_by_role.parquet")
    atomic_write_json(stage / "sealed_tokens.json", seal_tokens)
    atomic_write_json(stage / "ablation_manifest.json", ablation_evidence)

    nested_sizes = [
        int(value)
        for value in config["dense"]["nested_rows"]
        if int(value) <= len(d3_public)
    ]
    if nested_sizes:
        nested = make_nested_datasets(
            d3_public, nested_sizes, seed=int(config["seeds"]["dense"])
        )
        for size, frame in nested.items():
            _atomic_parquet(frame, stage / "D3" / f"nested_{size:06d}.parquet")
    checks = {
        "equal_sample_count": len({len(value) for value in methods.values()}) == 1,
        "all_methods_have_train": all(
            report["role_counts"].get("train", 0) > 0 for report in method_reports
        ),
        "all_methods_have_validation": all(
            report["role_counts"].get("validation", 0) > 0
            for report in method_reports
        ),
        "all_methods_have_sealed": all(
            report["sealed_row_count"] > 0 for report in method_reports
        ),
        "cross_chart_conflict_free": bool(conflict_report["gate_pass"]),
        "sealed_not_in_public": all(
            not frame["split_base_role"].eq("sealed_test").any()
            for frame in split_rows
        ),
        "full_d3_student_dataset_valid": bool(
            full_d3_report is None
            or (
                full_d3_report["source_row_count"]
                == full_d3_report["public_row_count"]
                + full_d3_report["sealed_row_count"]
                and full_d3_report["role_counts"].get("train", 0) > 0
                and full_d3_report["role_counts"].get(
                    "validation", 0
                )
                > 0
                and full_d3_report["sealed_row_count"] > 0
                and full_d3_report["sealed_not_in_public"]
            )
        ),
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="bacra_dataset_admission",
        method_reports=method_reports,
        conflict_report=conflict_report,
        nested_sizes=nested_sizes,
        full_d3_student_report=full_d3_report,
        **ablation_evidence,
    )


def _tensorflow_gpu_available(python: Path) -> bool:
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import tensorflow as tf; print(int(bool(tf.config.list_physical_devices('GPU'))))",
        ],
        cwd=SOURCE_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **WORKER_ENV},
        timeout=120,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip().endswith("1")


def student_worker(args: argparse.Namespace) -> int:
    config = load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    environment = load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )
    geometry = StudentGeometry(
        lengths_m=environment.lengths_m,
        p_end_local_m=environment.p_end_local_m,
        theta_sign=environment.theta_sign,
        beta_bounds_rad=environment.bounds,
    )
    frame = pd.read_parquet(args.dataset)
    strategy = StudentStrategy(args.strategy)
    beta_loss_scale_deg: float | tuple[float, ...] | None = None
    if args.beta_loss_scale_deg is not None:
        values = tuple(map(float, args.beta_loss_scale_deg))
        if len(values) == 1:
            beta_loss_scale_deg = values[0]
        elif len(values) == 6:
            beta_loss_scale_deg = values
        else:
            raise ValueError(
                "--beta-loss-scale-deg requires one value or six joint values"
            )
    report = train_one_seed(
        frame,
        strategy=strategy,
        geometry=geometry,
        seed=int(args.seed),
        hidden_units=tuple(map(int, config["student"]["hidden_units"])),
        lambda_fk=float(args.lambda_fk),
        beta_loss_scale_deg=beta_loss_scale_deg,
        beta5_head_units=tuple(
            map(int, config["student"].get("beta5_head_units", ()))
        ),
        learning_rate=float(config["student"]["learning_rate"]),
        batch_size=int(config["student"]["batch_size"]),
        max_epochs=int(config["student"]["max_epochs"]),
        patience=int(config["student"]["patience"]),
        output_dir=args.output,
    )
    atomic_write_json(Path(args.output) / "metrics.json", report)
    return 0


def stage_student(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["student"]
    stage.mkdir(parents=True, exist_ok=True)
    dataset_gate = json.loads(
        (output_root / STAGE_DIRS["dataset"] / "gate.json").read_text(encoding="utf-8")
    )
    audit_gate = json.loads(
        (output_root / STAGE_DIRS["audit"] / "gate.json").read_text(encoding="utf-8")
    )
    if not bool(dataset_gate["gate_pass"]):
        return _gate(
            stage / "gate.json",
            {"dataset_gate_pass": False},
            semantics="bacra_student_diagnostic",
            stopped_before_compute=True,
        )
    strategy = choose_student_strategy(audit_gate)
    if strategy is StudentStrategy.STATEFUL:
        return _gate(
            stage / "gate.json",
            {"sequential_training_artifact_available": False},
            semantics="bacra_student_diagnostic",
            selected_strategy=strategy.value,
            stopped_before_compute=True,
            reason="unordered regional dataset cannot be relabelled as a stateful trajectory dataset",
        )
    seeds = tuple(map(int, config["seeds"]["models"]))
    if str(config["preset"]) != "formal":
        seeds = seeds[: int(config["student"]["probe_seed_count"])]
    gpu = _tensorflow_gpu_available(python)
    execution = plan_seed_execution(
        seeds,
        gpu_available=gpu,
        cpu_workers=int(config["parallel"]["student_cpu_workers"]),
        intraop_threads=int(config["parallel"]["student_intraop_threads"]),
        interop_threads=int(config["parallel"]["student_interop_threads"]),
    )
    atomic_write_json(stage / "device_plan.json", execution.__dict__)
    methods = ("D0", "D1", "D2", "D3")
    lambda_fk = float(config["student"]["lambda_fk"][-1])
    commands: list[tuple[str, Sequence[str], Path]] = []
    for method in methods:
        dataset_method = (
            "D3_full"
            if method == "D3"
            and bool(
                config["student"].get(
                    "use_full_d3_dataset", False
                )
            )
            else method
        )
        dataset = (
            output_root
            / STAGE_DIRS["dataset"]
            / dataset_method
            / "public_dataset.parquet"
        )
        for seed in execution.seed_order:
            task_id = f"{method}_seed{seed}"
            destination = stage / method / f"seed_{seed}"
            command = [
                str(python),
                str(Path(__file__).resolve()),
                "_student-worker",
                "--config",
                str(config["config_path"]),
                "--preset",
                str(config["preset"]),
                "--project-root",
                str(project_root),
                "--dataset",
                str(dataset),
                "--strategy",
                strategy.value,
                "--seed",
                str(seed),
                "--lambda-fk",
                str(lambda_fk),
                "--output",
                str(destination),
            ]
            commands.append((task_id, command, stage / "logs" / f"{task_id}.log"))
    requested_workers = 1 if gpu else execution.effective_workers
    parallel = _run_subprocess_tasks(
        commands,
        requested_workers=requested_workers,
        manifest_path=stage / "parallel_manifest.json",
    )
    rows = []
    for method in methods:
        for seed in execution.seed_order:
            report = json.loads(
                (stage / method / f"seed_{seed}" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            report["data_method"] = method
            rows.append(report)
    metrics = pd.DataFrame(rows)
    _atomic_parquet(metrics, stage / "student_metrics_per_seed.parquet")
    threshold = float(config["student"]["interpolation_p95_mm"])
    d3 = metrics[metrics["data_method"].eq("D3")]
    pass_count = int(d3["validation_fk_p95_mm"].le(threshold).sum())
    required = min(int(config["student"]["required_seed_passes"]), len(d3))
    median_by_method = (
        metrics.groupby("data_method", sort=True)["validation_fk_p95_mm"]
        .median()
        .to_dict()
    )
    d3_better = all(
        float(median_by_method["D3"]) < float(median_by_method[name])
        for name in ("D0", "D1")
    )
    checks = {
        "all_seed_tasks_complete": len(metrics) == len(methods) * len(seeds),
        "d3_required_seed_passes": pass_count >= required,
        "d3_better_than_d0_d1": d3_better,
        "bounded_strategy": strategy
        in {StudentStrategy.STATIC, StudentStrategy.CHART_CONDITIONED},
        "virgin_sealed_data_unopened": True,
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="bacra_student_diagnostic",
        selected_strategy=strategy.value,
        device_plan=execution.__dict__,
        parallel_evidence=parallel,
        median_validation_fk_p95_mm=median_by_method,
        d3_training_dataset=(
            "D3_full"
            if bool(
                config["student"].get(
                    "use_full_d3_dataset", False
                )
            )
            else "D3"
        ),
        d3_seed_pass_count=pass_count,
        required_seed_passes=required,
    )


def stage_summary(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    stopped_after: str | None,
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    stage.mkdir(parents=True, exist_ok=True)
    gate_rows: list[dict[str, Any]] = []
    for name, relative in STAGE_DIRS.items():
        if name in {"summary", "evaluation"}:
            continue
        path = output_root / relative / "gate.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            gate_rows.append(
                {
                    "stage": name,
                    "gate_pass": bool(payload.get("gate_pass", False)),
                    "gate_semantics": payload.get("gate_semantics"),
                    "gate_path": str(path),
                    "gate_sha256": sha256_file(path),
                }
            )
    pd.DataFrame(gate_rows).to_csv(stage / "gate_checkpoint_table.csv", index=False)

    copies = {
        output_root / STAGE_DIRS["candidates"] / "candidate_node_summary.parquet":
            "sparse_candidate_summary.csv",
        output_root / STAGE_DIRS["atlas"] / "chart_coverage.csv":
            "chart_coverage.csv",
        output_root / STAGE_DIRS["atlas"] / "chart_overlap_report.csv":
            "chart_overlap_report.csv",
        output_root / STAGE_DIRS["dataset"] / "spatial_split_manifest.parquet":
            "spatial_split_manifest.csv",
        output_root / STAGE_DIRS["student"] / "student_metrics_per_seed.parquet":
            "student_metrics_per_seed.csv",
    }
    for source, destination in copies.items():
        if not source.is_file():
            continue
        if source.suffix == ".parquet":
            pd.read_parquet(source).to_csv(stage / destination, index=False)
        else:
            (stage / destination).write_bytes(source.read_bytes())
    audit_metrics = output_root / STAGE_DIRS["audit"] / "audit_metrics.csv"
    if audit_metrics.is_file():
        audit = pd.read_csv(audit_metrics)
        audit[audit["audit"].eq("path")].to_csv(
            stage / "path_consistency_report.csv", index=False
        )
        audit[audit["audit"].eq("loop")].to_csv(
            stage / "cycle_consistency_report.csv", index=False
        )
    dense_gate_path = output_root / STAGE_DIRS["dense"] / "gate.json"
    if dense_gate_path.is_file():
        dense_payload = json.loads(dense_gate_path.read_text(encoding="utf-8"))
        atomic_write_json(stage / "dense_dataset_report.json", dense_payload)
    capability_manifest = (
        output_root / STAGE_DIRS["capability"] / "capability_manifest.json"
    )
    if capability_manifest.is_file():
        payload = json.loads(capability_manifest.read_text(encoding="utf-8"))
        atomic_write_json(stage / "capability_map_summary.json", payload)
        capability_rows = pd.read_parquet(
            output_root / STAGE_DIRS["capability"] / "capability_map.parquet",
            columns=[
                "x_m",
                "y_m",
                "z_m",
                "voxel_x",
                "voxel_y",
                "voxel_z",
                "capability_tier",
            ],
        )
        sources, _hashes = _preflight_sources(config, project_root)
        radii = tuple(map(float, config["capability"]["roi_radii_mm_desc"]))
        atomic_write_json(
            stage / "anchor_component_diagnostics.json",
            {
                "primary_A2_178": centerline_component_diagnostics(
                    capability_rows,
                    _load_centerline(sources["primary_centerline"]),
                    roi_radii_mm_desc=radii,
                ),
                "comparison_A2_143": centerline_component_diagnostics(
                    capability_rows,
                    _load_centerline(sources["comparison_centerline"]),
                    roi_radii_mm_desc=radii,
                ),
            },
        )
    conflict_source = (
        output_root / STAGE_DIRS["dataset"] / "cross_chart_conflicts.parquet"
    )
    if conflict_source.is_file():
        (stage / "cross_chart_conflicts.parquet").write_bytes(
            conflict_source.read_bytes()
        )
    student_metrics = output_root / STAGE_DIRS["student"] / "student_metrics_per_seed.parquet"
    if student_metrics.is_file():
        metrics = pd.read_parquet(student_metrics)
        metrics.groupby("data_method", sort=True).agg(
            seed_count=("seed", "size"),
            validation_fk_p95_mm=("validation_fk_p95_mm", "median"),
            validation_fk_max_mm=("validation_fk_max_mm", "median"),
        ).reset_index().to_csv(stage / "data_ablation_table.csv", index=False)

    timing_rows: list[dict[str, Any]] = []
    for name, relative in STAGE_DIRS.items():
        manifest = output_root / relative / "parallel_manifest.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            timing_rows.append(
                {
                    "stage": name,
                    "requested_workers": payload.get("requested_workers"),
                    "effective_workers": payload.get("effective_workers"),
                    "wall_time_s": payload.get("wall_time_s"),
                    "aggregate_child_cpu_time_s": payload.get(
                        "aggregate_child_cpu_time_s"
                    ),
                    "aggregate_cpu_utilization": payload.get(
                        "aggregate_cpu_utilization"
                    ),
                }
            )
    pd.DataFrame(timing_rows).to_csv(stage / "teacher_student_timing.csv", index=False)
    if stopped_after == "capability" and (
        stage / "anchor_component_diagnostics.json"
    ).is_file():
        diagnostics = json.loads(
            (stage / "anchor_component_diagnostics.json").read_text(
                encoding="utf-8"
            )
        )
        primary = diagnostics["primary_A2_178"]["radii"][0]
        comparison = diagnostics["comparison_A2_143"]["radii"][0]
        recommendation = (
            "BACRA-V12 stopped fail-closed at capability connectivity. "
            f"At the registered {primary['radius_mm']:g} mm ROI, A2_178 has "
            f"{100.0 * primary['centerline_core_support']:.2f}% pointwise Core-safe "
            "support, but its best single connected component covers only "
            f"{100.0 * primary['best_component_centerline_support']:.2f}% of the "
            "closed centerline; A2_143's best component covers "
            f"{100.0 * comparison['best_component_centerline_support']:.2f}%. "
            "The next protocol should redesign or split the task ROI / atlas topology; "
            "Student tuning and post-hoc Gate relaxation are not authorized by this evidence."
        )
    else:
        recommendation = (
            f"BACRA-V12 stopped fail-closed after `{stopped_after}`; inspect that stage's "
            "Gate evidence before changing region, chart structure, or Student class."
            if stopped_after
            else "BACRA-V12 completed all unsealed diagnostics; deployment remains unauthorized."
        )
    (stage / "final_recommendation.md").write_text(
        "# BACRA-V12 final recommendation\n\n"
        + recommendation
        + "\n\n`deployment_claim_gate_pass=false`; sealed/deployment claims require a separate calibrated protocol.\n",
        encoding="utf-8",
    )
    (stage / "protocol_summary.md").write_text(
        "# BACRA-V12 protocol summary\n\n"
        f"- Protocol: `{config['protocol_id']}`\n"
        f"- Preset: `{config['preset']}`\n"
        "- Primary ROI anchor: `A2_178`\n"
        "- Comparator: `A2_143`\n"
        "- Gold margin: `1.5°`\n"
        "- Claim scope: `simulation_canonical_atlas_and_student_diagnostics`\n"
        f"- Stopped after: `{stopped_after or 'none'}`\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "protocol_id": str(config["protocol_id"]),
        "preset": str(config["preset"]),
        "stopped_after": stopped_after,
        "completed_stage_count": len(gate_rows),
        "all_completed_gates_pass": bool(
            gate_rows and all(row["gate_pass"] for row in gate_rows)
        ),
        "deployment_claim_gate_pass": False,
    }
    atomic_write_json(stage / "summary.json", report)
    return report


def _output_root(
    config: Mapping[str, Any], project_root: Path, explicit: str | None
) -> Path:
    if explicit:
        return Path(explicit).resolve()
    root = _source_path(project_root, str(config["output_root"]))
    if str(config["preset"]) == "pilot":
        return root.with_name(root.name + "_pilot")
    return root


def _stage_gate(output_root: Path, stage: str) -> dict[str, Any] | None:
    path = output_root / STAGE_DIRS[stage] / "gate.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config = load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = _output_root(config, project_root, args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    functions: Mapping[str, Callable[[], dict[str, Any]]] = {
        "protocol": lambda: stage_protocol(config, project_root, output_root),
        "capability": lambda: stage_capability(
            config, project_root, output_root, python=python
        ),
        "candidates": lambda: stage_candidates(
            config, project_root, output_root, python=python
        ),
        "atlas": lambda: stage_atlas(config, project_root, output_root),
        "audit": lambda: stage_audit(config, project_root, output_root),
        "dense": lambda: stage_dense(config, project_root, output_root),
        "dataset": lambda: stage_dataset(config, project_root, output_root),
        "student": lambda: stage_student(
            config, project_root, output_root, python=python
        ),
    }
    requested = (
        tuple(functions)
        if args.stage == "all"
        else (str(args.stage),)
    )
    stopped_after: str | None = None
    executed: list[str] = []
    for stage_name in requested:
        existing = _stage_gate(output_root, stage_name)
        if existing is not None:
            report = existing
        else:
            # An individual checkpoint may run only after every earlier Gate
            # exists and passed.  This prevents partial artifacts being treated
            # as a valid resume.
            index = tuple(functions).index(stage_name)
            for prerequisite in tuple(functions)[:index]:
                gate = _stage_gate(output_root, prerequisite)
                if gate is None or not bool(gate.get("gate_pass", False)):
                    raise RuntimeError(
                        f"{stage_name} requires passed checkpoint {prerequisite}"
                    )
            report = functions[stage_name]()
            executed.append(stage_name)
        if not bool(report.get("gate_pass", False)):
            stopped_after = stage_name
            break
    summary = stage_summary(
        config,
        project_root,
        output_root,
        stopped_after=stopped_after,
    )
    run_report = {
        "protocol_id": str(config["protocol_id"]),
        "preset": str(config["preset"]),
        "output_root": str(output_root),
        "requested_stage": str(args.stage),
        "executed_stages": executed,
        "stopped_after": stopped_after,
        "summary": summary,
    }
    atomic_write_json(output_root / "run_report.json", run_report)
    return run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    capability = subparsers.add_parser("_capability-worker")
    capability.add_argument("--config", required=True)
    capability.add_argument("--preset", required=True)
    capability.add_argument("--project-root", required=True)
    capability.add_argument("--beta-npy", required=True)
    capability.add_argument("--start", required=True, type=int)
    capability.add_argument("--stop", required=True, type=int)
    capability.add_argument("--output", required=True)

    candidate = subparsers.add_parser("_candidate-worker")
    candidate.add_argument("--config", required=True)
    candidate.add_argument("--preset", required=True)
    candidate.add_argument("--project-root", required=True)
    candidate.add_argument("--task-nodes", required=True)
    candidate.add_argument("--task-edges", required=True)
    candidate.add_argument("--capability", required=True)
    candidate.add_argument("--start", required=True, type=int)
    candidate.add_argument("--stop", required=True, type=int)
    candidate.add_argument("--output", required=True)
    candidate.add_argument("--report", required=True)

    student = subparsers.add_parser("_student-worker")
    student.add_argument("--config", required=True)
    student.add_argument("--preset", required=True)
    student.add_argument("--project-root", required=True)
    student.add_argument("--dataset", required=True)
    student.add_argument("--strategy", required=True)
    student.add_argument("--seed", required=True, type=int)
    student.add_argument("--lambda-fk", required=True, type=float)
    student.add_argument("--beta-loss-scale-deg", type=float, nargs="+")
    student.add_argument("--output", required=True)

    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v12.yaml"),
    )
    parser.add_argument("--preset", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument(
        "--project-root",
        default=str(project_root_from(SOURCE_ROOT)),
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--stage",
        choices=("all", "protocol", "capability", "candidates", "atlas", "audit", "dense", "dataset", "student"),
        default="all",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "_capability-worker":
        return capability_worker(args)
    if args.command == "_candidate-worker":
        return candidate_worker(args)
    if args.command == "_student-worker":
        return student_worker(args)
    report = run_pipeline(args)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
