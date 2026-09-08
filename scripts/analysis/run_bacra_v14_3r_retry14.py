#!/usr/bin/env python3
"""Run retry14 zero-rooted shell-mesh and ring-coverage experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import percentile, weighted_beta_rms_deg
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.retry10 import normalized_weighted_beta_deg, raw_beta_max_deg
from quasi_exp.teacher.retry12_symmetry import (
    BETA_COLUMNS,
    THETA_COLUMNS,
    XYZ_COLUMNS,
    corrected_trajectory_metrics,
    quotient_macroblock_id,
    stable_id,
    transform_beta,
    transform_xyz,
)
from quasi_exp.teacher.retry13_connected_roadmap import (
    component_registry,
    radius_edges,
)
from quasi_exp.teacher.retry14_shell_mesh import (
    assert_holdout_macroblock_isolation,
    audit_shell_mesh,
    build_shell_target_mesh,
    choose_canonical_parent,
    classify_candidate_endpoints,
    complete_ring_status,
    connector_loss_mask,
    coverage_fill_required,
    is_low_gain,
    map_legacy_labels_to_mesh,
    mesh_cycle_rank,
    mirror_nodes_and_mesh_edges,
    quotient_cylindrical_proposals,
    register_required_shell_domain,
    reverse_delete_shell_dataset,
    select_fill_targets,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    load_workspace_student_models,
    save_workspace_student_models,
    train_workspace_student,
)


EXPERIMENT_ID = "bacra_v14_3r_retry14_zero_rooted_shell_mesh_ring_coverage"
STAGE_DIRS = {
    "baseline": "00_baseline",
    "shell_domain": "01_shell_domain",
    "target_mesh": "02_target_mesh",
    "seed_mapping": "03_seed_mapping",
    "advancing_front": "04_advancing_front",
    "ring_closure": "05_ring_closure",
    "adaptive_fill": "06_adaptive_fill",
    "symmetry_expansion": "07_symmetry_expansion",
    "audit": "08_audit",
    "dataset": "09_dataset",
    "student_trajectory": "10_student_trajectory",
    "summary": "11_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)
WORKER = SOURCE_ROOT / "scripts" / "analysis" / "run_bacra_v14_3r_retry14_worker.py"


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    return resolved.parent.parent if resolved.parent.name == ".worktrees" else resolved


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()


def _config_sha(config: Mapping[str, Any]) -> str:
    return sha256_file(Path(str(config["config_path"])))


def _environment(config: Mapping[str, Any]) -> Any:
    reference = load_environment(
        project_root_from(SOURCE_ROOT), SOURCE_ROOT / str(config["sources"]["robot_config"])
    )
    return optimized_forward(reference)


def _thread_limited_environment() -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "PYTHONHASHSEED": str(20260894),
        }
    )
    return result


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    spec = config["upstream"][key]
    if "absolute_path" in spec:
        return Path(str(spec["absolute_path"]))
    root_key = "retry13_root" if key.startswith("retry13_") else "retry12_root"
    return Path(str(config["upstream"][root_key])) / str(spec["path"])


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("retry14 experiment_id mismatch")
    if int(config["runtime"]["maximum_concurrent_workers"]) != 12:
        raise ValueError("retry14 maximum_concurrent_workers must remain 12")
    if int(config["runtime"]["numerical_threads_per_worker"]) != 1:
        raise ValueError("retry14 numerical workers must remain single-threaded")
    if tuple(map(float, config["mesh"]["solver_step_schedule_mm"])) != (5.0, 2.5, 1.25):
        raise ValueError("retry14 solver step schedule mismatch")
    if int(config["fill"]["maximum_batches"]) != 8:
        raise ValueError("retry14 fill budget mismatch")
    if config["dataset"]["theta_storage"] != "beta_to_theta_without_theta_sign_multiplication":
        raise ValueError("retry14 theta storage convention mismatch")
    if any(bool(value) for value in config["claims"].values()):
        raise ValueError("retry14 cannot grant formal/deployment claims")
    consultation = SOURCE_ROOT / str(config["sources"]["consultation_input"])
    if sha256_file(consultation) != str(config["consultation_sha256"]):
        raise ValueError("retry14 consultation hash mismatch")
    robot = SOURCE_ROOT / str(config["sources"]["robot_config"])
    if sha256_file(robot) != str(config["sources"]["robot_config_sha256"]):
        raise ValueError("retry14 robot config hash mismatch")
    return config


def _binding_definition(binding_sha: str) -> Mapping[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{binding_sha}:spec/registry.yaml"], cwd=SOURCE_ROOT, text=True
    )
    return yaml.safe_load(raw)["experiments"][EXPERIMENT_ID]


def _ensure_identity(config: Mapping[str, Any], output_root: Path, binding_sha: str) -> dict[str, Any]:
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_source_fixed_point": _git_sha(),
        "binding_fixed_point": str(binding_sha),
        "config_sha256": _config_sha(config),
    }
    definition = _binding_definition(str(binding_sha))
    expected = {
        "scientific_source_fixed_point": _git_sha(),
        "config": str(Path(str(config["config_path"])).relative_to(SOURCE_ROOT)),
        "runner": str(Path(__file__).resolve().relative_to(SOURCE_ROOT)),
    }
    for key, value in expected.items():
        if str(definition.get(key)) != str(value):
            raise RuntimeError(f"retry14 binding mismatch for {key}: {definition.get(key)!r} != {value!r}")
    path = output_root / "run_identity.json"
    if path.exists() and _read_json(path) != identity:
        raise RuntimeError("retry14 output root identity mismatch")
    if not path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(path, identity)
    return identity


def _stage_manifest(output_root: Path, stage_name: str) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = []
    for path in sorted(stage.glob("*")):
        if path.is_file() and path.name != "completion_manifest.json":
            artifacts.append(
                {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            )
    upstream = []
    position = STAGE_ORDER.index(stage_name)
    if position:
        previous = output_root / STAGE_DIRS[STAGE_ORDER[position - 1]] / "completion_manifest.json"
        upstream.append({"stage": STAGE_ORDER[position - 1], "sha256": sha256_file(previous)})
    return {
        "schema_version": 1,
        "stage": stage_name,
        "scientific_source_fixed_point": _git_sha(),
        "artifacts": artifacts,
        "upstream": upstream,
    }


def _seal_stage(output_root: Path, stage_name: str) -> None:
    _write_json(
        output_root / STAGE_DIRS[stage_name] / "completion_manifest.json",
        _stage_manifest(output_root, stage_name),
    )


def _seal_gate(
    output_root: Path,
    config: Mapping[str, Any],
    stage_name: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        **dict(gate),
        "stage": stage_name,
        "experiment_id": EXPERIMENT_ID,
        "scientific_source_fixed_point": _git_sha(),
        "config_sha256": _config_sha(config),
        "formal_authorized": False,
        "deployment_authorized": False,
    }
    _write_json(output_root / STAGE_DIRS[stage_name] / "gate.json", value)
    _seal_stage(output_root, stage_name)
    return value


def _gate(output_root: Path, stage_name: str) -> dict[str, Any]:
    return _read_json(output_root / STAGE_DIRS[stage_name] / "gate.json")


def _stage_is_complete(output_root: Path, stage_name: str) -> bool:
    path = output_root / STAGE_DIRS[stage_name] / "completion_manifest.json"
    return path.exists() and _read_json(path) == _stage_manifest(output_root, stage_name)


def _progress(
    output_root: Path,
    stage_name: str,
    *,
    completed: int | None = None,
    total: int | None = None,
    message: str = "",
) -> None:
    _write_json(
        output_root / "progress.json",
        {
            "status": "running",
            "phase": stage_name,
            "completed": completed,
            "total": total,
            "message": message,
            "observed_at_unix": time.time(),
        },
    )


def _write_final_progress(output_root: Path) -> None:
    _write_json(
        output_root / "progress.json",
        {
            "status": "complete",
            "phase": "summary",
            "completed": len(STAGE_ORDER),
            "total": len(STAGE_ORDER),
            "message": "retry14 operational pipeline completed",
            "observed_at_unix": time.time(),
        },
    )


def _input_integrity(config: Mapping[str, Any]) -> tuple[pd.DataFrame, bool]:
    records: list[dict[str, Any]] = []
    for key, spec in config["upstream"].items():
        if not isinstance(spec, Mapping) or "sha256" not in spec:
            continue
        path = _upstream_path(config, key)
        observed = sha256_file(path) if path.exists() else None
        records.append(
            {
                "source_key": key,
                "path": str(path),
                "expected_sha256": str(spec["sha256"]),
                "observed_sha256": observed,
                "exists": path.exists(),
                "match": bool(path.exists() and observed == str(spec["sha256"])),
            }
        )
    frame = pd.DataFrame.from_records(records)
    return frame, bool(len(frame) and frame["match"].all())


def _graph_sensitivity(frame: pd.DataFrame, zero_position: int, radii: Sequence[float]) -> pd.DataFrame:
    rows = []
    for radius in radii:
        edges = radius_edges(frame, radius_mm=float(radius))
        labels, sizes = component_registry(len(frame), edges)
        zero_component = int(labels[int(zero_position)])
        rows.append(
            {
                "radius_mm": float(radius),
                "edge_count": len(edges),
                "component_count": len(sizes),
                "zero_component_size": int(np.sum(labels == zero_component)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _lineage_dag(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = labels.copy().reset_index(drop=True)
    ids = frame["physical_point_id"].astype(str).tolist()
    id_set = set(ids)
    zero_position = int(frame["zero_radius_mm"].astype(float).argmin())
    zero_id = ids[zero_position]
    parent = frame["retry13_parent_physical_point_id"].where(
        frame["retry13_parent_physical_point_id"].notna(), None
    )
    children: dict[str, list[str]] = {value: [] for value in ids}
    for child, parent_id in zip(ids, parent, strict=True):
        if parent_id is not None and str(parent_id) in id_set:
            children[str(parent_id)].append(child)
    depth = {zero_id: 0}
    queue = [zero_id]
    for current in queue:
        for child in sorted(children[current]):
            if child not in depth:
                depth[child] = depth[current] + 1
                queue.append(child)
    frame["zero_reachable"] = frame["physical_point_id"].astype(str).isin(depth)
    frame["retry14_baseline_lineage_depth"] = frame["physical_point_id"].astype(str).map(depth)
    return frame, {
        "zero_physical_point_id": zero_id,
        "zero_reachable_label_count": len(depth),
        "orphan_label_count": len(frame) - len(depth),
        "maximum_lineage_depth": int(max(depth.values())) if depth else 0,
    }


def _cache_key(task: Mapping[str, Any]) -> str:
    payload = {
        key: task[key]
        for key in (
            "source_beta_hash",
            "target_xyz",
            "solver_tier",
            "mesh_vertex_id",
            "mesh_edge_type",
            "lineage_id",
            "gauge_version",
            "step_size_mm",
        )
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _run_worker_tasks(
    config: Mapping[str, Any],
    output_root: Path,
    stage_name: str,
    tasks: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if not tasks:
        return pd.DataFrame()
    work = output_root / STAGE_DIRS[stage_name] / "_work"
    task_dir = work / "tasks"
    log_dir = work / "logs"
    cache_dir = work / "cache"
    for path in (task_dir, log_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    cached: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for raw_task in tasks:
        task = dict(raw_task)
        key = _cache_key(task)
        task["cache_key"] = key
        path = cache_dir / f"{key}.json"
        if path.exists():
            cached.append(_read_json(path))
        else:
            pending.append(task)
    if not pending:
        return pd.DataFrame.from_records(sorted(cached, key=lambda row: str(row["task_id"])))
    worker_count = min(int(config["runtime"]["maximum_concurrent_workers"]), len(pending))
    chunks = [pending[index::worker_count] for index in range(worker_count)]
    processes: list[tuple[subprocess.Popen[str], Any, list[dict[str, Any]]]] = []
    for index, chunk in enumerate(chunks):
        batch_id = stable_id("retry14_worker_batch", stage_name, *(row["task_id"] for row in chunk))
        task_path = task_dir / f"{batch_id}.json"
        _write_json(task_path, {"batch_id": batch_id, "tasks": chunk})
        log_handle = (log_dir / f"{batch_id}.stderr.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(config["runtime"]["python"]),
                str(WORKER),
                "--config",
                str(config["config_path"]),
                "--task-file",
                str(task_path),
            ],
            cwd=SOURCE_ROOT,
            env=_thread_limited_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=log_handle,
        )
        processes.append((process, log_handle, chunk))
    fresh: list[dict[str, Any]] = []
    try:
        for process, log_handle, chunk in processes:
            stdout, _ = process.communicate()
            log_handle.close()
            if process.returncode:
                for other, other_log, _ in processes:
                    if other.poll() is None:
                        other.terminate()
                    if not other_log.closed:
                        other_log.close()
                raise RuntimeError(f"retry14 numerical worker failed rc={process.returncode}")
            lines = [line for line in stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise RuntimeError("retry14 worker must emit exactly one JSON line")
            payload = json.loads(lines[0])
            by_task = {str(row["task_id"]): row for row in chunk}
            for result in payload["results"]:
                task = by_task[str(result["task_id"])]
                record = {**result, "cache_key": task["cache_key"]}
                path = cache_dir / f"{task['cache_key']}.json"
                if path.exists() and _read_json(path) != record:
                    raise RuntimeError(f"conflicting retry14 cache result for {task['cache_key']}")
                _write_json(path, record)
                fresh.append(record)
    finally:
        for _, log_handle, _ in processes:
            if not log_handle.closed:
                log_handle.close()
    records = sorted([*cached, *fresh], key=lambda row: str(row["task_id"]))
    return pd.DataFrame.from_records(records)


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )


def _two_step_dls(
    environment: Any,
    beta: np.ndarray,
    xyz: np.ndarray,
    *,
    zero_xyz: np.ndarray,
) -> np.ndarray:
    corrected = np.asarray(beta, dtype=float).copy()
    points = np.asarray(xyz, dtype=float)
    bounds = np.asarray(environment.bounds, dtype=float)
    weights = np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0], dtype=float)
    for _ in range(2):
        for index, point in enumerate(points):
            if np.linalg.norm(point - zero_xyz) <= 1.0e-12:
                corrected[index] = 0.0
                continue
            if abs(point[1]) <= 1.0e-12:
                freeze = {0, 2, 4}
            elif abs(point[2]) <= 1.0e-12:
                freeze = {1, 3, 5}
            else:
                freeze = set()
            free = np.asarray([axis for axis in range(6) if axis not in freeze], dtype=int)
            current = np.asarray(environment.fk(corrected[index]), dtype=float).reshape(-1, 3)[0]
            jacobian = np.asarray(environment.jacobian(corrected[index]), dtype=float).reshape(3, 6)[:, free]
            weight_inverse = np.diag(1.0 / weights[free])
            task = jacobian @ weight_inverse @ jacobian.T + 1.0e-6 * np.eye(3)
            pseudoinverse = weight_inverse @ jacobian.T @ np.linalg.pinv(task, rcond=1.0e-12)
            corrected[index, free] = np.clip(
                corrected[index, free] + pseudoinverse @ (point - current),
                bounds[free, 0],
                bounds[free, 1],
            )
            if freeze:
                corrected[index, sorted(freeze)] = 0.0
    return corrected


def _symmetry_prediction(model: Any, xyz: np.ndarray, zero_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    canonical = points.copy()
    canonical[:, 1:] = np.abs(canonical[:, 1:])
    beta = np.asarray(model(canonical.astype(np.float32), training=False), dtype=float)
    for index, point in enumerate(points):
        if point[1] < 0.0:
            beta[index] = transform_beta(beta[index], "mirror_y")
        if point[2] < 0.0:
            beta[index] = transform_beta(beta[index], "mirror_z")
        if abs(point[1]) <= 1.0e-12:
            beta[index, [0, 2, 4]] = 0.0
        if abs(point[2]) <= 1.0e-12:
            beta[index, [1, 3, 5]] = 0.0
        if np.linalg.norm(point - zero_xyz) <= 1.0e-12:
            beta[index] = 0.0
    return beta


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root)
        if relative.name in {"artifact_manifest.json", "completion_manifest.json"}:
            continue
        if relative.parts and relative.parts[-1].endswith(".tmp"):
            continue
        if "_work" in relative.parts:
            continue
        artifacts.append(
            {
                "path": str(relative),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific_source_fixed_point": _git_sha(),
        "artifacts": artifacts,
    }


def stage_baseline(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["baseline"]
    inventory, source_pass = _input_integrity(config)
    run_identity_path = Path(str(config["upstream"]["retry13_root"])) / "run_identity.json"
    run_identity = _read_json(run_identity_path) if run_identity_path.exists() else {}
    identity_pass = bool(
        sha256_file(run_identity_path) == str(config["upstream"]["retry13_run_identity_sha256"])
        and run_identity.get("scientific_source_fixed_point")
        == config["upstream"]["retry13_scientific_source_fixed_point"]
        and run_identity.get("binding_fixed_point")
        == config["upstream"]["retry13_binding_fixed_point"]
    )
    _write_parquet(inventory, stage / "source_inventory.parquet")
    if not source_pass or not identity_pass:
        return _seal_gate(
            output_root,
            config,
            "baseline",
            {
                "status": "source_integrity_red",
                "source_integrity_pass": source_pass,
                "retry13_identity_pass": identity_pass,
                "shell_domain_authorized": False,
            },
        )
    labels = pd.read_parquet(_upstream_path(config, "retry13_fundamental_representatives"))
    edges = pd.read_parquet(_upstream_path(config, "retry13_fundamental_edges"))
    proposals = pd.read_parquet(_upstream_path(config, "retry12_target_registry"))
    zero = np.asarray(_read_json(_upstream_path(config, "retry12_zero_anchor"))["xyz0_m"], dtype=float)
    lineage, lineage_report = _lineage_dag(labels)
    zero_position = int(lineage["zero_radius_mm"].astype(float).argmin())
    sensitivity = _graph_sensitivity(
        lineage,
        zero_position,
        tuple(map(float, config["audit"]["diagnostic_radii_mm"])),
    )
    quotient = quotient_cylindrical_proposals(
        proposals,
        zero_xyz_m=zero,
        axial_step_mm=float(config["shell_domain"]["axial_step_mm"]),
        radial_step_mm=float(config["shell_domain"]["radial_step_mm"]),
        angle_bins=int(config["shell_domain"]["coarse_angle_bin_count"]),
    )
    cells, rings = register_required_shell_domain(
        quotient,
        zero_xyz_m=zero,
        axial_step_mm=float(config["shell_domain"]["axial_step_mm"]),
        radial_step_mm=float(config["shell_domain"]["radial_step_mm"]),
        target_arc_step_mm=float(config["shell_domain"]["target_arc_step_mm"]),
        support_distance_mm=float(config["shell_domain"]["trajectory_support_maximum_mm"]),
    )
    label_tree = cKDTree(lineage.loc[:, XYZ_COLUMNS].to_numpy(float))
    distance, _ = label_tree.query(quotient.loc[:, XYZ_COLUMNS].to_numpy(float), k=1)
    service = distance * 1000.0
    edge_lengths = edges.get("xyz_length_mm", pd.Series(dtype=float)).dropna().to_numpy(float)
    lineage_edge_count = int(len(edges))
    lineage_cycle_rank = max(0, lineage_edge_count - len(lineage) + 1)
    geometric_edges = radius_edges(lineage, radius_mm=10.0)
    geometric_labels, geometric_sizes = component_registry(len(lineage), geometric_edges)
    geometric_cycle_rank = int(len(geometric_edges) - len(lineage) + len(geometric_sizes))
    ring_baseline = rings.copy()
    if len(ring_baseline):
        ring_baseline["current_complete"] = False
        ring_baseline["maximum_angular_gap_bins"] = int(
            config["shell_domain"]["coarse_angle_bin_count"]
        )
    _write_parquet(lineage, stage / "lineage_dag.parquet")
    _write_json(stage / "zero_reachability.json", lineage_report)
    _write_parquet(cells, stage / "shell_cell_registry.parquet")
    _write_parquet(ring_baseline, stage / "ring_baseline.parquet")
    _write_json(
        stage / "proposal_service_baseline.json",
        {
            "proposal_count": len(quotient),
            "proposal_to_label_p50_mm": percentile(service, 50),
            "proposal_to_label_p90_mm": percentile(service, 90),
            "proposal_to_label_p95_mm": percentile(service, 95),
            "proposal_to_label_max_mm": float(np.max(service)),
        },
    )
    _write_parquet(sensitivity, stage / "graph_sensitivity.parquet")
    return _seal_gate(
        output_root,
        config,
        "baseline",
        {
            "status": "complete",
            "source_integrity_pass": True,
            "retry13_identity_pass": True,
            **lineage_report,
            "lineage_edge_length_p50_mm": percentile(edge_lengths, 50),
            "lineage_edge_length_p95_mm": percentile(edge_lengths, 95),
            "lineage_edge_length_max_mm": float(np.max(edge_lengths)) if len(edge_lengths) else math.inf,
            "required_shell_cell_count_target_only_baseline": int(cells["required_shell_cell"].sum()),
            "registered_trajectory_ring_count_target_only_baseline": len(rings),
            "current_complete_ring_count": 0,
            "lineage_tree_cycle_rank": lineage_cycle_rank,
            "geometric_graph_cycle_rank_10mm": geometric_cycle_rank,
            "proposal_service_p95_mm": percentile(service, 95),
            "shell_domain_authorized": True,
        },
    )


def stage_shell_domain(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["shell_domain"]
    if not _gate(output_root, "baseline").get("shell_domain_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "shell_domain",
            {"status": "not_authorized", "target_mesh_authorized": False},
        )
    proposals = pd.read_parquet(_upstream_path(config, "retry12_target_registry"))
    labels = pd.read_parquet(output_root / STAGE_DIRS["baseline"] / "lineage_dag.parquet")
    zero = np.asarray(_read_json(_upstream_path(config, "retry12_zero_anchor"))["xyz0_m"], dtype=float)
    domain = config["shell_domain"]
    quotient = quotient_cylindrical_proposals(
        proposals,
        zero_xyz_m=zero,
        axial_step_mm=float(domain["axial_step_mm"]),
        radial_step_mm=float(domain["radial_step_mm"]),
        angle_bins=int(domain["coarse_angle_bin_count"]),
    )
    legacy_as_proposals = labels.loc[:, ["physical_point_id", *XYZ_COLUMNS]].rename(
        columns={"physical_point_id": "proposal_id"}
    )
    legacy_cells = quotient_cylindrical_proposals(
        legacy_as_proposals,
        zero_xyz_m=zero,
        axial_step_mm=float(domain["axial_step_mm"]),
        radial_step_mm=float(domain["radial_step_mm"]),
        angle_bins=int(domain["coarse_angle_bin_count"]),
    )
    core_mask = labels["zero_reachable"].astype(bool) & labels["zero_radius_mm"].astype(float).le(30.0)
    core_label_ids = set(labels.loc[core_mask, "physical_point_id"].astype(str))
    core_ids = sorted(
        set(legacy_cells.loc[legacy_cells["proposal_id"].astype(str).isin(core_label_ids), "cell_id"].astype(str))
    )
    cells, rings = register_required_shell_domain(
        quotient,
        zero_xyz_m=zero,
        axial_step_mm=float(domain["axial_step_mm"]),
        radial_step_mm=float(domain["radial_step_mm"]),
        target_arc_step_mm=float(domain["target_arc_step_mm"]),
        support_distance_mm=float(domain["trajectory_support_maximum_mm"]),
        zero_core_cell_ids=core_ids,
    )
    required = cells[cells["required_shell_cell"]].copy()
    _write_parquet(quotient, stage / "quotient_proposal_registry.parquet")
    _write_parquet(cells, stage / "ur_cell_support.parquet")
    _write_parquet(required, stage / "required_shell_domain.parquet")
    _write_parquet(rings, stage / "registered_ring_families.parquet")
    valid = bool(len(required) and len(core_ids))
    return _seal_gate(
        output_root,
        config,
        "shell_domain",
        {
            "status": "registered" if valid else "domain_registration_red",
            "proposal_count": len(quotient),
            "zero_core_cell_count": len(core_ids),
            "coverage_ring_candidate_count": int(cells["coverage_ring_candidate"].sum()),
            "required_shell_cell_count": len(required),
            "trajectory_ring_candidate_count": len(rings),
            "one_cell_hole_count": int(required.get("one_cell_hole", pd.Series(dtype=bool)).sum()),
            "target_mesh_authorized": valid,
        },
    )


def _connector_vertices(solver_segments: pd.DataFrame, vertices: pd.DataFrame) -> pd.DataFrame:
    known = set(vertices["mesh_vertex_id"].astype(str))
    rows: dict[str, dict[str, Any]] = {}
    for segment in solver_segments.to_dict("records"):
        for side in ("left", "right"):
            vertex_id = str(segment[f"{side}_mesh_vertex_id"])
            if vertex_id in known or vertex_id in rows:
                continue
            rows[vertex_id] = {
                "mesh_vertex_id": vertex_id,
                "cell_id": None,
                "ring_id": None,
                "phi_index": -1,
                "phi_rad": math.nan,
                "x_m": float(segment[f"{side}_x_m"]),
                "y_m": float(segment[f"{side}_y_m"]),
                "z_m": float(segment[f"{side}_z_m"]),
                "target_kind": "connector_relay",
                "proposal_support_distance_mm": math.nan,
                "label_role": "connector_only",
                "label_status": "unresolved",
                "required_shell_cell": False,
                "trajectory_ring_candidate": False,
                "seam_class": "interior",
            }
    return pd.DataFrame.from_records(list(rows.values()))


def stage_target_mesh(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["target_mesh"]
    if not _gate(output_root, "shell_domain").get("target_mesh_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "target_mesh",
            {"status": "not_authorized", "seed_mapping_authorized": False},
        )
    cells = pd.read_parquet(output_root / STAGE_DIRS["shell_domain"] / "ur_cell_support.parquet")
    quotient = pd.read_parquet(
        output_root / STAGE_DIRS["shell_domain"] / "quotient_proposal_registry.parquet"
    )
    zero = np.asarray(_read_json(_upstream_path(config, "retry12_zero_anchor"))["xyz0_m"], dtype=float)
    vertices, logical_edges, solver_segments = build_shell_target_mesh(
        cells,
        quotient,
        zero_xyz_m=zero,
        maximum_solver_segment_mm=float(config["mesh"]["maximum_solver_segment_mm"]),
    )
    connectors = _connector_vertices(solver_segments, vertices)
    all_vertices = pd.concat([vertices, connectors], ignore_index=True, sort=False)
    rings = pd.read_parquet(
        output_root / STAGE_DIRS["shell_domain"] / "registered_ring_families.parquet"
    )
    _write_parquet(all_vertices, stage / "shell_mesh_vertices.parquet")
    _write_parquet(logical_edges, stage / "shell_mesh_edges.parquet")
    _write_parquet(connectors, stage / "connector_target_registry.parquet")
    _write_parquet(solver_segments, stage / "solver_segment_registry.parquet")
    _write_parquet(rings, stage / "trajectory_ring_registry.parquet")
    length_pass = bool(
        len(solver_segments)
        and solver_segments["segment_length_mm"].max()
        <= float(config["mesh"]["maximum_solver_segment_mm"]) + 1.0e-9
    )
    valid = bool(len(vertices) and len(logical_edges) and length_pass)
    return _seal_gate(
        output_root,
        config,
        "target_mesh",
        {
            "status": "registered" if valid else "mesh_registration_red",
            "mesh_vertex_count": len(vertices),
            "connector_vertex_count": len(connectors),
            "logical_edge_count": len(logical_edges),
            "solver_segment_count": len(solver_segments),
            "maximum_solver_segment_mm": float(solver_segments["segment_length_mm"].max()),
            "solver_segment_length_pass": length_pass,
            "seed_mapping_authorized": valid,
        },
    )


def stage_seed_mapping(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["seed_mapping"]
    if not _gate(output_root, "target_mesh").get("seed_mapping_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "seed_mapping",
            {"status": "not_authorized", "advancing_front_authorized": False},
        )
    labels = pd.read_parquet(output_root / STAGE_DIRS["baseline"] / "lineage_dag.parquet")
    vertices = pd.read_parquet(output_root / STAGE_DIRS["target_mesh"] / "shell_mesh_vertices.parquet")
    mapping, unused, seeded_vertices = map_legacy_labels_to_mesh(
        labels,
        vertices,
        maximum_snap_mm=float(config["mesh"]["legacy_snap_maximum_mm"]),
    )
    accepted_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    label_by_id = labels.set_index(labels["physical_point_id"].astype(str), drop=False)
    environment = _environment(config)
    if len(mapping):
        for vertex_id, group in mapping.groupby("mesh_vertex_id", sort=True):
            candidates = labels[
                labels["physical_point_id"].astype(str).isin(group["legacy_label_id"].astype(str))
            ].copy()
            vertex = seeded_vertices[
                seeded_vertices["mesh_vertex_id"].astype(str).eq(str(vertex_id))
            ].iloc[0]
            target_xyz = vertex.loc[list(XYZ_COLUMNS)].to_numpy(float)
            target_seam = _seam_class(target_xyz, 1.0e-9)
            candidates = candidates[
                candidates.loc[:, BETA_COLUMNS].apply(
                    lambda row: _beta_matches_seam(row.to_numpy(float), target_seam),
                    axis=1,
                )
            ].copy()
            if candidates.empty:
                continue
            beta = candidates.loc[:, BETA_COLUMNS].to_numpy(float)
            raw_max = (
                float(np.max(np.abs(np.degrees(beta[:, None, :] - beta[None, :, :]))))
                if len(beta) > 1
                else 0.0
            )
            if raw_max > float(config["solver"]["branch_conflict_raw_minimum_deg"]):
                conflicts.append(
                    {
                        "mesh_vertex_id": str(vertex_id),
                        "legacy_candidate_count": len(candidates),
                        "same_target_raw_gap_deg": raw_max,
                        "status": "branch_conflict",
                    }
                )
                continue
            distances = group.set_index("legacy_label_id")["snap_distance_mm"]
            candidates["snap_distance_mm"] = candidates["physical_point_id"].astype(str).map(distances)
            chosen = candidates.sort_values(
                ["snap_distance_mm", "retry14_baseline_lineage_depth", "physical_point_id"],
                kind="stable",
            ).iloc[0]
            candidate_beta = chosen.loc[list(BETA_COLUMNS)].to_numpy(float)
            fk_residual = float(
                np.linalg.norm(
                    np.asarray(environment.fk(candidate_beta), dtype=float).reshape(-1, 3)[0]
                    - target_xyz
                )
                * 1000.0
            )
            if fk_residual > float(config["solver"]["fk_residual_maximum_mm"]):
                continue
            record = {**vertex.to_dict(), **chosen.to_dict()}
            record.update(
                {
                    "mesh_vertex_id": str(vertex_id),
                    **{name: float(vertex[name]) for name in XYZ_COLUMNS},
                    "label_status": "seeded",
                    "label_quality": str(chosen.get("label_quality", "certified_silver")),
                    "label_role": str(vertex.get("label_role", "supervision")),
                    "mesh_target_fk_residual_mm": fk_residual,
                    "legacy_snap_distance_mm": float(chosen["snap_distance_mm"]),
                    "canonical_parent_id": chosen.get("retry13_parent_physical_point_id"),
                    "lineage_depth": int(chosen["retry14_baseline_lineage_depth"]),
                    "zero_reachable": True,
                }
            )
            accepted_rows.append(record)
    exact_zero = labels.iloc[int(labels["zero_radius_mm"].astype(float).argmin())]
    zero_vertex = vertices[vertices["target_kind"].eq("exact_zero")].iloc[0]
    accepted_rows = [row for row in accepted_rows if row["mesh_vertex_id"] != zero_vertex["mesh_vertex_id"]]
    accepted_rows.append(
        {
            **zero_vertex.to_dict(),
            **exact_zero.to_dict(),
            "mesh_vertex_id": str(zero_vertex["mesh_vertex_id"]),
            **{name: float(zero_vertex[name]) for name in XYZ_COLUMNS},
            "label_status": "seeded",
            "label_quality": "ExactAnchor",
            "label_role": "supervision",
            "mesh_target_fk_residual_mm": 0.0,
            "legacy_snap_distance_mm": 0.0,
            "canonical_parent_id": None,
            "lineage_depth": 0,
            "zero_reachable": True,
        }
    )
    accepted = pd.DataFrame.from_records(accepted_rows)
    accepted_legacy_ids = set(accepted.get("physical_point_id", pd.Series(dtype=str)).astype(str))
    parent_of = {
        str(row.physical_point_id): None
        if pd.isna(row.retry13_parent_physical_point_id)
        else str(row.retry13_parent_physical_point_id)
        for row in labels.itertuples()
    }
    ancestor_ids: set[str] = set()
    for label_id in accepted_legacy_ids:
        current = label_id
        while current in parent_of and parent_of[current] is not None:
            current = str(parent_of[current])
            if current in ancestor_ids:
                break
            ancestor_ids.add(current)
    connectors = labels[
        labels["physical_point_id"].astype(str).isin(ancestor_ids - accepted_legacy_ids)
    ].copy()
    connectors["label_role"] = "connector_only"
    connectors["label_status"] = "seeded"
    unused = unused[
        ~unused["physical_point_id"].astype(str).isin(ancestor_ids)
    ].reset_index(drop=True)
    _write_parquet(mapping, stage / "legacy_to_mesh_mapping.parquet")
    _write_parquet(accepted, stage / "accepted_mesh_seeds.parquet")
    _write_parquet(connectors, stage / "connector_only_labels.parquet")
    _write_parquet(unused, stage / "unused_legacy_labels.parquet")
    _write_parquet(pd.DataFrame.from_records(conflicts), stage / "legacy_seed_branch_conflicts.parquet")
    authorized = bool(len(accepted) and accepted["label_quality"].eq("ExactAnchor").any())
    return _seal_gate(
        output_root,
        config,
        "seed_mapping",
        {
            "status": "mapped" if authorized else "exact_zero_red",
            "legacy_label_count": len(labels),
            "mapped_seed_count": len(accepted),
            "connector_only_legacy_count": len(connectors),
            "unused_legacy_label_count": len(unused),
            "legacy_seed_branch_conflict_count": len(conflicts),
            "exact_zero_seeded": authorized,
            "advancing_front_authorized": authorized,
        },
    )


def _initialize_mesh_state(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vertices = pd.read_parquet(
        output_root / STAGE_DIRS["target_mesh"] / "shell_mesh_vertices.parquet"
    ).copy()
    logical_edges = pd.read_parquet(
        output_root / STAGE_DIRS["target_mesh"] / "shell_mesh_edges.parquet"
    ).copy()
    segments = pd.read_parquet(
        output_root / STAGE_DIRS["target_mesh"] / "solver_segment_registry.parquet"
    ).copy()
    for column in BETA_COLUMNS:
        vertices[column] = np.nan
    vertices["physical_point_id"] = None
    vertices["label_quality"] = None
    vertices["canonical_parent_id"] = None
    vertices["canonical_parent_segment_id"] = None
    vertices["lineage_depth"] = np.nan
    vertices["zero_reachable"] = False
    vertices["mesh_target_fk_residual_mm"] = np.nan
    vertices["label_origin"] = None
    seeds = pd.read_parquet(
        output_root / STAGE_DIRS["seed_mapping"] / "accepted_mesh_seeds.parquet"
    )
    position = {value: index for index, value in enumerate(vertices["mesh_vertex_id"].astype(str))}
    for seed in seeds.to_dict("records"):
        vertex_id = str(seed["mesh_vertex_id"])
        if vertex_id not in position:
            continue
        row = position[vertex_id]
        for column in BETA_COLUMNS:
            vertices.at[row, column] = float(seed[column])
        for column, default in (
            ("physical_point_id", stable_id("retry14_seed", vertex_id)),
            ("label_quality", "certified_silver"),
            ("canonical_parent_id", None),
            ("lineage_depth", 0),
            ("zero_reachable", True),
            ("mesh_target_fk_residual_mm", 0.0),
        ):
            value = seed.get(column, default)
            vertices.at[row, column] = default if pd.isna(value) else value
        vertices.at[row, "label_status"] = "seeded"
        vertices.at[row, "label_origin"] = "retry13_legacy_mesh_seed"
    segments["segment_verified"] = False
    segments["weighted_beta_gap_deg"] = np.nan
    segments["raw_beta_gap_deg"] = np.nan
    segments["reverse_weighted_gap_deg"] = np.nan
    logical_edges["mesh_edge_verified"] = False
    return vertices, logical_edges, segments


def _segment_adjacency(segments: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for index, row in enumerate(segments.itertuples(index=False)):
        left = str(row.left_mesh_vertex_id)
        right = str(row.right_mesh_vertex_id)
        adjacency.setdefault(left, []).append((right, index))
        adjacency.setdefault(right, []).append((left, index))
    for node in adjacency:
        adjacency[node].sort(key=lambda item: (item[0], item[1]))
    return adjacency


def _path_union_to_targets(
    vertices: pd.DataFrame,
    segments: pd.DataFrame,
    target_ids: Iterable[str],
) -> tuple[list[str], dict[str, int]]:
    adjacency = _segment_adjacency(segments)
    solved = set(
        vertices.loc[
            vertices["label_status"].isin(["seeded", "Gold", "certified_silver", "provisional_silver"]),
            "mesh_vertex_id",
        ].astype(str)
    )
    positions = vertices.set_index(vertices["mesh_vertex_id"].astype(str), drop=False)

    def bfs(allowed: set[str] | None = None) -> tuple[dict[str, str | None], dict[str, int]]:
        roots = sorted(solved if allowed is None else solved.intersection(allowed))
        predecessor: dict[str, str | None] = {node: None for node in roots}
        depth: dict[str, int] = {node: 0 for node in roots}
        queue = list(roots)
        for current in queue:
            for neighbor, _ in adjacency.get(current, []):
                if allowed is not None and neighbor not in allowed:
                    continue
                if neighbor not in predecessor:
                    predecessor[neighbor] = current
                    depth[neighbor] = depth[current] + 1
                    queue.append(neighbor)
        return predecessor, depth

    regular = bfs()
    seam_nodes: dict[str, set[str]] = {"y_seam": set(), "z_seam": set()}
    for row in vertices.loc[:, ["mesh_vertex_id", *XYZ_COLUMNS]].itertuples(index=False):
        node_id = str(row.mesh_vertex_id)
        seam = _seam_class(np.asarray([row.x_m, row.y_m, row.z_m], dtype=float), 1.0e-9)
        if seam in {"y_seam", "exact_zero"}:
            seam_nodes["y_seam"].add(str(node_id))
        if seam in {"z_seam", "exact_zero"}:
            seam_nodes["z_seam"].add(str(node_id))
    seam_search = {name: bfs(nodes) for name, nodes in seam_nodes.items()}
    required: set[str] = set()
    selected_depth: dict[str, int] = {}
    for target in sorted(set(map(str, target_ids))):
        target_row = positions.loc[target]
        seam = _seam_class(target_row.loc[list(XYZ_COLUMNS)].to_numpy(float), 1.0e-9)
        predecessor, depth = seam_search.get(seam, regular)
        current = target
        if current not in predecessor:
            continue
        while current not in solved:
            required.add(current)
            selected_depth[current] = min(selected_depth.get(current, depth[current]), depth[current])
            parent = predecessor[current]
            if parent is None:
                break
            current = parent
    ordered = sorted(required, key=lambda value: (selected_depth[value], value))
    return ordered, selected_depth


def _seam_class(xyz: np.ndarray, tolerance: float) -> str:
    if abs(float(xyz[1])) <= tolerance and abs(float(xyz[2])) <= tolerance:
        return "exact_zero"
    if abs(float(xyz[1])) <= tolerance:
        return "y_seam"
    if abs(float(xyz[2])) <= tolerance:
        return "z_seam"
    return "interior"


def _beta_matches_seam(beta: np.ndarray, seam_class: str, tolerance: float = 1.0e-8) -> bool:
    values = np.asarray(beta, dtype=float)
    if seam_class == "exact_zero":
        return bool(np.max(np.abs(values)) <= tolerance)
    if seam_class == "y_seam":
        return bool(np.max(np.abs(values[[0, 2, 4]])) <= tolerance)
    if seam_class == "z_seam":
        return bool(np.max(np.abs(values[[1, 3, 5]])) <= tolerance)
    return True


def _make_solver_task(
    source: pd.Series,
    target: pd.Series,
    segment: pd.Series,
    *,
    solver_tier: str,
    target_xyz: np.ndarray | None = None,
    target_vertex_id: str | None = None,
) -> dict[str, Any]:
    source_beta = source.loc[list(BETA_COLUMNS)].to_numpy(float)
    target_point = (
        target.loc[list(XYZ_COLUMNS)].to_numpy(float)
        if target_xyz is None
        else np.asarray(target_xyz, dtype=float)
    )
    vertex_id = str(target["mesh_vertex_id"]) if target_vertex_id is None else str(target_vertex_id)
    source_hash = hashlib.sha256(source_beta.astype("<f8").tobytes()).hexdigest()
    source_xyz = source.loc[list(XYZ_COLUMNS)].to_numpy(float)
    step = float(np.linalg.norm(target_point - source_xyz) * 1000.0)
    task_id = stable_id(
        "retry14_solver_task",
        source["mesh_vertex_id"],
        vertex_id,
        segment["solver_segment_id"],
        solver_tier,
    )
    return {
        "task_id": task_id,
        "mesh_vertex_id": vertex_id,
        "source_vertex_id": str(source["mesh_vertex_id"]),
        "source_node_id": int(source.get("lineage_depth", 0) or 0),
        "target_node_id": int(hashlib.sha256(vertex_id.encode()).hexdigest()[:12], 16),
        "source_beta": source_beta.tolist(),
        "source_beta_hash": source_hash,
        "source_xyz": source_xyz.tolist(),
        "target_xyz": target_point.tolist(),
        "solver_tier": solver_tier,
        "mesh_edge_type": str(segment.get("mesh_edge_type", "solver_segment")),
        "logical_edge_id": str(segment["logical_edge_id"]),
        "solver_segment_id": str(segment["solver_segment_id"]),
        "lineage_id": str(source.get("physical_point_id", source["mesh_vertex_id"])),
        "gauge_version": "retry14_zero_rooted_shell_mesh_v1",
        "step_size_mm": step,
        "seam_class": _seam_class(target_point, 1.0e-9),
        "source_fk_residual_mm": float(source.get("mesh_target_fk_residual_mm", 0.0) or 0.0),
        "source_minimum_margin_deg": float(source.get("minimum_margin_deg", 1.0) or 1.0),
        "source_condition_number": float(source.get("source_condition", 1.0) or 1.0),
    }


def _successful_endpoint_rows(
    results: pd.DataFrame,
    vertices: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    position = vertices.set_index(vertices["mesh_vertex_id"].astype(str), drop=False)
    rows: list[dict[str, Any]] = []
    for result in results.to_dict("records"):
        source_id = str(result["source_vertex_id"])
        if source_id not in position.index or not result.get("admitted", False):
            continue
        source = position.loc[source_id]
        beta = np.asarray(result["beta"], dtype=float)
        success = bool(
            result.get("reverse_success", False)
            and float(result.get("reverse_weighted_gap_deg", math.inf))
            <= float(config["solver"]["reverse_weighted_maximum_deg"])
            and float(result.get("source_to_endpoint_weighted_gap_deg", math.inf))
            <= float(config["solver"]["local_weighted_maximum_deg"])
            and float(result.get("source_to_endpoint_raw_gap_deg", math.inf))
            <= float(config["solver"]["local_raw_maximum_deg"])
        )
        row = {
            **result,
            "success": success,
            "parent_id": source_id,
            "lineage_depth": int(source["lineage_depth"]) + 1,
            "source_condition": float(result.get("source_condition_number", 1.0)),
            "weighted_transition_deg": float(result.get("source_to_endpoint_weighted_gap_deg", math.inf)),
        }
        row.update(dict(zip(BETA_COLUMNS, beta, strict=True)))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _refresh_logical_edge_certificates(
    logical_edges: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    result = logical_edges.copy()
    verified_by_id = segments.set_index("solver_segment_id")["segment_verified"].astype(bool).to_dict()
    result["mesh_edge_verified"] = result["segment_chain_ids"].fillna("").map(
        lambda value: bool(value)
        and all(verified_by_id.get(segment_id, False) for segment_id in str(value).split("|"))
    )
    return result


def _certify_logical_edge_chains(
    config: Mapping[str, Any],
    output_root: Path,
    stage_name: str,
    vertices: pd.DataFrame,
    logical_edges: pd.DataFrame,
    segments: pd.DataFrame,
    logical_edge_ids: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Certify complete logical edges and recertify their existing endpoints.

    Target lifting alone may label both ends of a logical edge through unrelated
    lineage corridors.  It does not certify the edge between them.  This pass
    traverses each registered segment chain in both directions, compares every
    solved endpoint with an existing canonical label, and promotes provisional
    supervision only after a distinct mesh path agrees at the same target.
    """

    accepted_status = {"seeded", "Gold", "certified_silver", "provisional_silver"}
    vertices = vertices.copy().reset_index(drop=True)
    logical_edges = logical_edges.copy().reset_index(drop=True)
    segments = segments.copy().reset_index(drop=True)
    requested = set(map(str, logical_edge_ids))
    selected = logical_edges[
        logical_edges["logical_edge_id"].astype(str).isin(requested)
    ].sort_values("logical_edge_id", kind="stable")
    if selected.empty:
        return vertices, logical_edges, segments, pd.DataFrame(), pd.DataFrame()

    vertex_position = {
        value: index for index, value in enumerate(vertices["mesh_vertex_id"].astype(str))
    }
    segment_position = {
        value: index for index, value in enumerate(segments["solver_segment_id"].astype(str))
    }
    paths: list[dict[str, Any]] = []
    for edge in selected.to_dict("records"):
        chain_segments = str(edge.get("segment_chain_ids", "")).split("|")
        chain_vertices = str(edge.get("segment_vertex_chain_ids", "")).split("|")
        if (
            not chain_segments
            or not chain_vertices
            or len(chain_vertices) != len(chain_segments) + 1
            or any(value not in segment_position for value in chain_segments)
            or any(value not in vertex_position for value in chain_vertices)
        ):
            continue
        paths.append(
            {
                "logical_edge_id": str(edge["logical_edge_id"]),
                "mesh_edge_type": str(edge["mesh_edge_type"]),
                "segments": chain_segments,
                "vertices": chain_vertices,
            }
        )

    attempt_frames: list[pd.DataFrame] = []
    conflicts: list[dict[str, Any]] = []

    def solve_direction(*, reverse: bool) -> None:
        oriented: list[dict[str, Any]] = []
        for path in paths:
            oriented.append(
                {
                    **path,
                    "segments": list(reversed(path["segments"])) if reverse else list(path["segments"]),
                    "vertices": list(reversed(path["vertices"])) if reverse else list(path["vertices"]),
                }
            )
        maximum_steps = max((len(path["segments"]) for path in oriented), default=0)
        for step_index in range(maximum_steps):
            tasks: list[dict[str, Any]] = []
            for path in oriented:
                if step_index >= len(path["segments"]):
                    continue
                source_id = str(path["vertices"][step_index])
                target_id = str(path["vertices"][step_index + 1])
                source = vertices.iloc[vertex_position[source_id]]
                if str(source["label_status"]) not in accepted_status:
                    continue
                target = vertices.iloc[vertex_position[target_id]]
                segment = segments.iloc[segment_position[str(path["segments"][step_index])]].copy()
                segment["mesh_edge_type"] = str(path["mesh_edge_type"])
                tasks.append(_make_solver_task(source, target, segment, solver_tier="5.0mm"))
            if not tasks:
                continue
            combined = _run_worker_tasks(config, output_root, stage_name, tasks)
            if len(combined):
                combined = combined.copy()
                combined["edge_recertification"] = True
                combined["edge_recertification_direction"] = "reverse" if reverse else "forward"
                attempt_frames.append(combined)
            for tier in ("2.5mm", "1.25mm"):
                failed_targets: set[str] = set()
                for task in tasks:
                    target_id = str(task["mesh_vertex_id"])
                    group = combined[
                        combined["mesh_vertex_id"].astype(str).eq(target_id)
                    ] if len(combined) else pd.DataFrame()
                    endpoints = _successful_endpoint_rows(group, vertices, config)
                    if endpoints.empty or not endpoints["success"].astype(bool).any():
                        failed_targets.add(target_id)
                if not failed_targets:
                    break
                retry_tasks: list[dict[str, Any]] = []
                for original in tasks:
                    if str(original["mesh_vertex_id"]) not in failed_targets:
                        continue
                    retry = dict(original)
                    retry["solver_tier"] = tier
                    retry["task_id"] = stable_id(
                        "retry14_solver_task", retry["source_vertex_id"],
                        retry["mesh_vertex_id"], retry["solver_segment_id"], tier,
                    )
                    retry_tasks.append(retry)
                retry_results = _run_worker_tasks(
                    config, output_root, stage_name, retry_tasks
                )
                if len(retry_results):
                    retry_results = retry_results.copy()
                    retry_results["edge_recertification"] = True
                    retry_results["edge_recertification_direction"] = "reverse" if reverse else "forward"
                    attempt_frames.append(retry_results)
                    combined = pd.concat(
                        [combined, retry_results], ignore_index=True, sort=False
                    )

            for task in tasks:
                source_id = str(task["source_vertex_id"])
                target_id = str(task["mesh_vertex_id"])
                group = combined[
                    combined["mesh_vertex_id"].astype(str).eq(target_id)
                    & combined["source_vertex_id"].astype(str).eq(source_id)
                    & combined["solver_segment_id"].astype(str).eq(str(task["solver_segment_id"]))
                ] if len(combined) else pd.DataFrame()
                endpoints = _successful_endpoint_rows(group, vertices, config)
                successful = endpoints[
                    endpoints.get("success", pd.Series(False, index=endpoints.index)).astype(bool)
                ] if len(endpoints) else pd.DataFrame()
                if successful.empty:
                    continue
                if "maximum_refinement_step_mm" not in successful:
                    successful["maximum_refinement_step_mm"] = float(task["step_size_mm"])
                if "fk_residual_mm" not in successful:
                    successful["fk_residual_mm"] = math.inf
                candidate = successful.sort_values(
                    ["maximum_refinement_step_mm", "fk_residual_mm"],
                    ascending=[False, True], kind="stable",
                ).iloc[0]
                target_row = vertex_position[target_id]
                target_status = str(vertices.at[target_row, "label_status"])
                candidate_beta = candidate.loc[list(BETA_COLUMNS)].to_numpy(float)
                if target_status in accepted_status:
                    stored_beta = vertices.loc[target_row, list(BETA_COLUMNS)].to_numpy(float)
                    same_weighted = float(weighted_beta_rms_deg(stored_beta, candidate_beta))
                    same_raw = float(np.max(np.abs(np.degrees(stored_beta - candidate_beta))))
                    if (
                        same_weighted > float(config["solver"]["gold_weighted_maximum_deg"]) + 1.0e-12
                        or same_raw > float(config["solver"]["gold_raw_maximum_deg"]) + 1.0e-12
                    ):
                        conflicts.append(
                            {
                                "mesh_vertex_id": target_id,
                                "source_vertex_id": source_id,
                                "logical_edge_id": str(task["logical_edge_id"]),
                                "solver_segment_id": str(task["solver_segment_id"]),
                                "same_point_weighted_gap_deg": same_weighted,
                                "same_point_raw_gap_deg": same_raw,
                                "status": "branch_conflict",
                                "stored_beta": stored_beta.tolist(),
                                "candidate_beta": candidate_beta.tolist(),
                            }
                        )
                        mask = segments["solver_segment_id"].astype(str).eq(str(task["solver_segment_id"]))
                        segments.loc[mask, "segment_verified"] = False
                        continue
                    canonical_parent = str(vertices.at[target_row, "canonical_parent_id"])
                    if target_status == "provisional_silver" and source_id != canonical_parent:
                        vertices.at[target_row, "label_status"] = "certified_silver"
                        vertices.at[target_row, "label_quality"] = "certified_silver"
                elif str(vertices.at[target_row, "label_role"]) == "connector_only":
                    for column, value in zip(BETA_COLUMNS, candidate_beta, strict=True):
                        vertices.at[target_row, column] = value
                    vertices.at[target_row, "label_status"] = "provisional_silver"
                    vertices.at[target_row, "label_quality"] = "provisional_silver"
                    vertices.at[target_row, "physical_point_id"] = stable_id("retry14_label", target_id)
                    vertices.at[target_row, "canonical_parent_id"] = source_id
                    vertices.at[target_row, "canonical_parent_segment_id"] = str(task["solver_segment_id"])
                    source_depth = vertices.at[vertex_position[source_id], "lineage_depth"]
                    vertices.at[target_row, "lineage_depth"] = int(source_depth) + 1
                    vertices.at[target_row, "zero_reachable"] = True
                    vertices.at[target_row, "mesh_target_fk_residual_mm"] = float(candidate["fk_residual_mm"])
                    vertices.at[target_row, "label_origin"] = f"retry14_{stage_name}_edge_chain"
                else:
                    # Supervision targets are admitted only by the registered
                    # supervision-cap path.  Edge recertification may fill
                    # connector relays, but must not create extra supervision.
                    continue
                mask = segments["solver_segment_id"].astype(str).eq(str(task["solver_segment_id"]))
                segments.loc[mask, "segment_verified"] = True
                segments.loc[mask, "weighted_beta_gap_deg"] = float(candidate["weighted_transition_deg"])
                segments.loc[mask, "raw_beta_gap_deg"] = float(candidate["source_to_endpoint_raw_gap_deg"])
                segments.loc[mask, "reverse_weighted_gap_deg"] = float(candidate["reverse_weighted_gap_deg"])

    solve_direction(reverse=False)
    solve_direction(reverse=True)
    logical_edges = _refresh_logical_edge_certificates(logical_edges, segments)
    return (
        vertices,
        logical_edges,
        segments,
        pd.concat(attempt_frames, ignore_index=True, sort=False) if attempt_frames else pd.DataFrame(),
        pd.DataFrame.from_records(conflicts),
    )


def _solve_requested_targets(
    config: Mapping[str, Any],
    output_root: Path,
    stage_name: str,
    vertices: pd.DataFrame,
    logical_edges: pd.DataFrame,
    segments: pd.DataFrame,
    requested_ids: Iterable[str],
    *,
    supervision_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vertices = vertices.copy().reset_index(drop=True)
    segments = segments.copy().reset_index(drop=True)
    if int(supervision_cap) <= 0:
        return (
            vertices,
            logical_edges.copy(),
            segments,
            pd.DataFrame(),
            pd.DataFrame(),
        )
    ordered, path_depth = _path_union_to_targets(vertices, segments, requested_ids)
    vertex_position = {
        value: index for index, value in enumerate(vertices["mesh_vertex_id"].astype(str))
    }
    adjacency = _segment_adjacency(segments)
    attempts: list[pd.DataFrame] = []
    conflicts: list[dict[str, Any]] = []
    new_supervision = 0
    for depth_value in sorted(set(path_depth.get(node, 0) for node in ordered)):
        wave_nodes = [
            node
            for node in ordered
            if path_depth.get(node) == depth_value
            and vertices.iloc[vertex_position[node]]["label_status"] in {"unresolved", "local_hole"}
        ]
        tasks: list[dict[str, Any]] = []
        scheduled_supervision = 0
        for node in wave_nodes:
            target = vertices.iloc[vertex_position[node]]
            if target["label_role"] == "supervision":
                if new_supervision + scheduled_supervision >= supervision_cap:
                    continue
                scheduled_supervision += 1
            sources: list[tuple[int, str, int]] = []
            for neighbor, segment_position in adjacency.get(node, []):
                source = vertices.iloc[vertex_position[neighbor]]
                if source["label_status"] not in {"seeded", "Gold", "certified_silver", "provisional_silver"}:
                    continue
                target_seam = _seam_class(target.loc[list(XYZ_COLUMNS)].to_numpy(float), 1.0e-9)
                source_seam = _seam_class(source.loc[list(XYZ_COLUMNS)].to_numpy(float), 1.0e-9)
                if target_seam in {"y_seam", "z_seam"} and source_seam != target_seam:
                    continue
                sources.append((int(source["lineage_depth"]), neighbor, segment_position))
            for _, neighbor, segment_position in sorted(sources)[: int(config["mesh"]["maximum_parent_sources"])]:
                source = vertices.iloc[vertex_position[neighbor]]
                segment = segments.iloc[segment_position].copy()
                segment["mesh_edge_type"] = logical_edges.set_index("logical_edge_id").loc[
                    segment["logical_edge_id"], "mesh_edge_type"
                ]
                tasks.append(
                    _make_solver_task(source, target, segment, solver_tier="5.0mm")
                )
        wave_results = _run_worker_tasks(config, output_root, stage_name, tasks)
        if len(wave_results):
            attempts.append(wave_results)
        combined_results = wave_results.copy()
        for tier in ("2.5mm", "1.25mm"):
            failed_nodes: set[str] = set()
            for node in {str(task["mesh_vertex_id"]) for task in tasks}:
                group = combined_results[
                    combined_results["mesh_vertex_id"].astype(str).eq(node)
                ] if len(combined_results) else pd.DataFrame()
                endpoints = _successful_endpoint_rows(group, vertices, config)
                if endpoints.empty or not endpoints["success"].astype(bool).any():
                    failed_nodes.add(node)
            if not failed_nodes:
                break
            retry_tasks: list[dict[str, Any]] = []
            for original in tasks:
                if str(original["mesh_vertex_id"]) not in failed_nodes:
                    continue
                retry = dict(original)
                retry["solver_tier"] = tier
                retry["task_id"] = stable_id(
                    "retry14_solver_task",
                    retry["source_vertex_id"], retry["mesh_vertex_id"],
                    retry["solver_segment_id"], tier,
                )
                retry_tasks.append(retry)
            retry_results = _run_worker_tasks(
                config, output_root, stage_name, retry_tasks
            )
            if len(retry_results):
                attempts.append(retry_results)
                combined_results = pd.concat(
                    [combined_results, retry_results], ignore_index=True, sort=False
                )
        for node in wave_nodes:
            group = combined_results[combined_results["mesh_vertex_id"].astype(str).eq(node)].copy() if len(combined_results) else pd.DataFrame()
            endpoint_rows = _successful_endpoint_rows(group, vertices, config)
            status = classify_candidate_endpoints(endpoint_rows)
            target_row = vertex_position[node]
            if status == "branch_conflict":
                vertices.at[target_row, "label_status"] = "local_hole"
                conflicts.append(
                    {
                        "mesh_vertex_id": node,
                        "candidate_endpoint_count": int(endpoint_rows["success"].sum()),
                        "status": "branch_conflict",
                    }
                )
                continue
            successful = endpoint_rows[endpoint_rows.get("success", False).astype(bool)] if len(endpoint_rows) else pd.DataFrame()
            if successful.empty:
                continue
            canonical = choose_canonical_parent(endpoint_rows)
            beta = canonical.loc[list(BETA_COLUMNS)].to_numpy(float)
            for column, value in zip(BETA_COLUMNS, beta, strict=True):
                vertices.at[target_row, column] = value
            quality = "Gold" if status == "Gold" else "provisional_silver"
            vertices.at[target_row, "label_status"] = quality
            vertices.at[target_row, "label_quality"] = quality
            vertices.at[target_row, "physical_point_id"] = stable_id("retry14_label", node)
            vertices.at[target_row, "canonical_parent_id"] = str(canonical["parent_id"])
            vertices.at[target_row, "canonical_parent_segment_id"] = str(canonical["solver_segment_id"])
            vertices.at[target_row, "lineage_depth"] = int(canonical["lineage_depth"])
            vertices.at[target_row, "zero_reachable"] = True
            vertices.at[target_row, "mesh_target_fk_residual_mm"] = float(canonical["fk_residual_mm"])
            vertices.at[target_row, "label_origin"] = f"retry14_{stage_name}"
            if vertices.at[target_row, "label_role"] == "supervision":
                new_supervision += 1
            for endpoint in successful.to_dict("records"):
                mask = segments["solver_segment_id"].astype(str).eq(str(endpoint["solver_segment_id"]))
                segments.loc[mask, "segment_verified"] = True
                segments.loc[mask, "weighted_beta_gap_deg"] = float(endpoint["weighted_transition_deg"])
                segments.loc[mask, "raw_beta_gap_deg"] = float(endpoint["source_to_endpoint_raw_gap_deg"])
                segments.loc[mask, "reverse_weighted_gap_deg"] = float(endpoint["reverse_weighted_gap_deg"])
        _progress(
            output_root,
            stage_name,
            completed=new_supervision,
            total=supervision_cap,
            message=f"wave_depth={depth_value}",
        )
    logical_edges = _refresh_logical_edge_certificates(logical_edges, segments)
    return (
        vertices,
        logical_edges,
        segments,
        pd.concat(attempts, ignore_index=True, sort=False) if attempts else pd.DataFrame(),
        pd.DataFrame.from_records(conflicts),
    )


def _ring_status_frame(
    vertices: pd.DataFrame,
    logical_edges: pd.DataFrame,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    accepted_status = {"seeded", "Gold", "certified_silver", "provisional_silver"}
    accepted = vertices[vertices["label_status"].isin(accepted_status)].copy()
    accepted_ids = set(accepted["mesh_vertex_id"].astype(str))
    certified = logical_edges[
        logical_edges["mesh_edge_verified"].astype(bool)
        & logical_edges["left_mesh_vertex_id"].astype(str).isin(accepted_ids)
        & logical_edges["right_mesh_vertex_id"].astype(str).isin(accepted_ids)
    ].copy()
    zero_rows = accepted[accepted["target_kind"].eq("exact_zero")]
    if zero_rows.empty:
        return pd.DataFrame()
    zero_id = str(zero_rows.iloc[0]["mesh_vertex_id"])
    full_nodes, full_edges = mirror_nodes_and_mesh_edges(
        accepted, certified, zero_vertex_id=zero_id
    )
    rows: list[dict[str, Any]] = []
    for cell in cells[
        cells["required_shell_cell"] & cells["trajectory_ring_candidate"]
    ].sort_values(["u_index", "rho_index"], kind="stable").itertuples():
        ring_id = stable_id("retry14_ring", cell.cell_id)
        fundamental = vertices[vertices["ring_id"].astype(str).eq(ring_id)].copy()
        expected = len(fundamental)
        labelled = int(fundamental["label_status"].isin(accepted_status).sum())
        complete = complete_ring_status(full_nodes, full_edges, ring_id)
        rows.append(
            {
                "ring_id": ring_id,
                "cell_id": str(cell.cell_id),
                "u_index": int(cell.u_index),
                "rho_index": int(cell.rho_index),
                "expected_fundamental_vertex_count": expected,
                "labelled_fundamental_vertex_count": labelled,
                "missing_vertex_count": expected - labelled,
                "complete_ring": complete,
                "maximum_angular_gap_bins": 0 if complete else min(8, max(1, int(math.ceil(8 * (expected - labelled) / max(1, expected))))),
            }
        )
    return pd.DataFrame.from_records(rows)


def _mesh_metrics(
    vertices: pd.DataFrame,
    logical_edges: pd.DataFrame,
    cells: pd.DataFrame,
    quotient_proposals: pd.DataFrame,
) -> dict[str, float]:
    fundamental = audit_shell_mesh(vertices, logical_edges, cells)
    ring_status = _ring_status_frame(vertices, logical_edges, cells)
    registered = len(ring_status)
    complete_count = int(ring_status.get("complete_ring", pd.Series(dtype=bool)).sum())
    accepted = vertices[
        vertices["label_role"].eq("supervision")
        & vertices["label_status"].isin(["seeded", "Gold", "certified_silver", "provisional_silver"])
    ]
    required_ids = set(cells.loc[cells["required_shell_cell"], "cell_id"].astype(str))
    service_proposals = quotient_proposals[
        quotient_proposals["cell_id"].astype(str).isin(required_ids)
    ]
    if len(accepted) and len(service_proposals):
        distance, _ = cKDTree(accepted.loc[:, XYZ_COLUMNS].to_numpy(float)).query(
            service_proposals.loc[:, XYZ_COLUMNS].to_numpy(float), k=1
        )
        service_p95 = percentile(distance * 1000.0, 95)
        service_p50 = percentile(distance * 1000.0, 50)
        service_p90 = percentile(distance * 1000.0, 90)
        service_max = float(np.max(distance * 1000.0))
    else:
        service_p50 = service_p90 = service_p95 = service_max = math.inf
    accepted_ids = set(accepted["mesh_vertex_id"].astype(str))
    all_accepted = vertices[vertices["mesh_vertex_id"].astype(str).isin(accepted_ids)]
    certified = logical_edges[
        logical_edges["mesh_edge_verified"].astype(bool)
        & logical_edges["left_mesh_vertex_id"].astype(str).isin(accepted_ids)
        & logical_edges["right_mesh_vertex_id"].astype(str).isin(accepted_ids)
    ]
    if len(all_accepted) and all_accepted["target_kind"].eq("exact_zero").any():
        zero_id = str(all_accepted[all_accepted["target_kind"].eq("exact_zero")].iloc[0]["mesh_vertex_id"])
        full_nodes, full_edges = mirror_nodes_and_mesh_edges(
            all_accepted, certified, zero_vertex_id=zero_id
        )
        cycle_rank = mesh_cycle_rank(full_nodes, full_edges)
    else:
        cycle_rank = 0
    return {
        "shell_cell_coverage": float(fundamental["shell_cell_coverage"]),
        "ring_completion_fraction": complete_count / registered if registered else 0.0,
        "complete_ring_count": float(complete_count),
        "registered_trajectory_ring_count": float(registered),
        "proposal_service_p50_mm": float(service_p50),
        "proposal_service_p90_mm": float(service_p90),
        "proposal_service_p95_mm": float(service_p95),
        "proposal_service_max_mm": float(service_max),
        "maximum_angular_gap_bins": float(
            ring_status["maximum_angular_gap_bins"].max() if len(ring_status) else 8
        ),
        "mesh_cycle_rank": float(cycle_rank),
        "served_required_shell_cell_count": float(fundamental["served_required_shell_cell_count"]),
    }


def _initial_target_ids(vertices: pd.DataFrame, zero_xyz: np.ndarray, cap: int) -> list[str]:
    frame = vertices[
        vertices["label_role"].eq("supervision")
        & vertices["label_status"].isin(["unresolved", "local_hole"])
    ].copy()
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float)
    frame["zero_radius_mm"] = np.linalg.norm(xyz - zero_xyz.reshape(1, 3), axis=1) * 1000.0
    frame["seam_priority"] = np.minimum(np.abs(frame["y_m"]), np.abs(frame["z_m"]))
    return frame.sort_values(
        ["zero_radius_mm", "seam_priority", "trajectory_ring_candidate", "mesh_vertex_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).head(int(cap))["mesh_vertex_id"].astype(str).tolist()


def stage_advancing_front(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["advancing_front"]
    if not _gate(output_root, "seed_mapping").get("advancing_front_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "advancing_front",
            {"status": "not_authorized", "ring_closure_authorized": False},
        )
    vertices, logical_edges, segments = _initialize_mesh_state(output_root)
    zero = np.asarray(_read_json(_upstream_path(config, "retry12_zero_anchor"))["xyz0_m"], dtype=float)
    cap = int(config["mesh"]["advancing_front_initial_supervision_cap"])
    requested = _initial_target_ids(vertices, zero, cap)
    vertices, logical_edges, segments, attempts, conflicts = _solve_requested_targets(
        config,
        output_root,
        "advancing_front",
        vertices,
        logical_edges,
        segments,
        requested,
        supervision_cap=cap,
    )
    lineage_rows = vertices[
        vertices["label_status"].isin(["seeded", "Gold", "certified_silver", "provisional_silver"])
        & vertices["canonical_parent_id"].notna()
    ][
        [
            "mesh_vertex_id",
            "canonical_parent_id",
            "canonical_parent_segment_id",
            "lineage_depth",
        ]
    ].copy()
    unresolved = vertices[vertices["label_status"].isin(["unresolved", "local_hole"])].copy()
    _write_parquet(vertices, stage / "mesh_vertex_labels.parquet")
    _write_parquet(lineage_rows, stage / "lineage_parent_edges.parquet")
    _write_parquet(attempts, stage / "candidate_endpoints.parquet")
    _write_parquet(conflicts, stage / "branch_conflicts.parquet")
    _write_parquet(unresolved, stage / "unresolved_vertices.parquet")
    _write_parquet(segments, stage / "solver_segments.parquet")
    _write_parquet(logical_edges, stage / "logical_mesh_edges.parquet")
    accepted = vertices["label_status"].isin(["seeded", "Gold", "certified_silver", "provisional_silver"])
    new_supervision = int(
        (
            accepted
            & vertices["label_role"].eq("supervision")
            & vertices["label_origin"].astype(str).eq("retry14_advancing_front")
        ).sum()
    )
    return _seal_gate(
        output_root,
        config,
        "advancing_front",
        {
            "status": "complete",
            "accepted_vertex_count": int(accepted.sum()),
            "new_supervision_count": new_supervision,
            "verified_solver_segment_count": int(segments["segment_verified"].sum()),
            "verified_logical_edge_count": int(logical_edges["mesh_edge_verified"].sum()),
            "branch_conflict_count": len(conflicts),
            "unresolved_vertex_count": len(unresolved),
            "ring_closure_authorized": bool(accepted.sum()),
        },
    )


def _balanced_ring_ids(
    ring_status: pd.DataFrame, *, preferred_count: int
) -> list[str]:
    """Return a balanced preferred prefix followed by every replacement ring."""

    if ring_status.empty:
        return []
    frame = ring_status.sort_values(
        ["missing_vertex_count", "u_index", "rho_index", "ring_id"], kind="stable"
    ).copy()
    selected: list[str] = []
    u_groups = np.array_split(np.asarray(sorted(frame["u_index"].unique())), min(3, frame["u_index"].nunique()))
    for group in u_groups:
        candidates = frame[frame["u_index"].isin(group)]
        rho_groups = np.array_split(
            np.asarray(sorted(candidates["rho_index"].unique())),
            min(3, candidates["rho_index"].nunique()),
        )
        for rho_group in rho_groups:
            choice = candidates[candidates["rho_index"].isin(rho_group)].head(1)
            if len(choice):
                selected.append(str(choice.iloc[0]["ring_id"]))
    for ring_id in frame["ring_id"].astype(str):
        if ring_id not in selected:
            selected.append(ring_id)
    prefix = selected[: int(preferred_count)]
    return prefix + [ring_id for ring_id in selected if ring_id not in set(prefix)]


def stage_ring_closure(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["ring_closure"]
    if not _gate(output_root, "advancing_front").get("ring_closure_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "ring_closure",
            {"status": "not_authorized", "adaptive_fill_authorized": False},
        )
    source = output_root / STAGE_DIRS["advancing_front"]
    vertices = pd.read_parquet(source / "mesh_vertex_labels.parquet")
    logical_edges = pd.read_parquet(source / "logical_mesh_edges.parquet")
    segments = pd.read_parquet(source / "solver_segments.parquet")
    cells = pd.read_parquet(output_root / STAGE_DIRS["shell_domain"] / "ur_cell_support.parquet")
    ring_status = _ring_status_frame(vertices, logical_edges, cells)
    target_complete_count = int(config["ring"]["green_minimum_complete_rings"])
    candidate_ring_ids = _balanced_ring_ids(
        ring_status, preferred_count=target_complete_count,
    )
    cap = int(config["mesh"]["ring_closure_supervision_cap"])
    remaining_cap = cap
    accepted_status = {"seeded", "Gold", "certified_silver", "provisional_silver"}
    complete_ring_ids = set(
        ring_status.loc[ring_status["complete_ring"].astype(bool), "ring_id"].astype(str)
    )
    attempt_frames: list[pd.DataFrame] = []
    conflict_frames: list[pd.DataFrame] = []
    seam_first_target_count = 0
    attempted_ring_count = 0

    def accepted_supervision_count() -> int:
        return int(
            (
                vertices["label_role"].eq("supervision")
                & vertices["label_status"].isin(accepted_status)
            ).sum()
        )

    for ring_id in candidate_ring_ids:
        if len(complete_ring_ids) >= target_complete_count:
            break
        if ring_id in complete_ring_ids:
            continue
        ring_vertices = vertices[
            vertices["ring_id"].astype(str).eq(ring_id)
            & vertices["label_role"].eq("supervision")
        ]
        if ring_vertices.empty:
            continue
        fully_labelled_before = ring_vertices["label_status"].isin(accepted_status).all()
        if remaining_cap <= 0 and not fully_labelled_before:
            continue
        attempted_ring_count += 1
        requested = ring_vertices["mesh_vertex_id"].astype(str).tolist()
        seam_targets = ring_vertices.loc[
            (ring_vertices["y_m"].abs() <= 1.0e-9)
            | (ring_vertices["z_m"].abs() <= 1.0e-9),
            "mesh_vertex_id",
        ].astype(str).tolist()
        seam_first_target_count += len(seam_targets)

        before = accepted_supervision_count()
        vertices, logical_edges, segments, seam_attempts, seam_conflicts = _solve_requested_targets(
            config, output_root, "ring_closure", vertices, logical_edges,
            segments, seam_targets, supervision_cap=remaining_cap,
        )
        after_seams = accepted_supervision_count()
        remaining_cap = max(0, remaining_cap - max(0, after_seams - before))
        vertices, logical_edges, segments, arc_attempts, arc_conflicts = _solve_requested_targets(
            config, output_root, "ring_closure", vertices, logical_edges,
            segments, requested, supervision_cap=remaining_cap,
        )
        after_arc = accepted_supervision_count()
        remaining_cap = max(0, remaining_cap - max(0, after_arc - after_seams))
        for frame in (seam_attempts, arc_attempts):
            if len(frame):
                frame = frame.copy(); frame["ring_candidate_id"] = ring_id
                attempt_frames.append(frame)
        for frame in (seam_conflicts, arc_conflicts):
            if len(frame):
                frame = frame.copy(); frame["ring_candidate_id"] = ring_id
                conflict_frames.append(frame)

        refreshed_ring = vertices[
            vertices["ring_id"].astype(str).eq(ring_id)
            & vertices["label_role"].eq("supervision")
        ]
        if len(refreshed_ring) and refreshed_ring["label_status"].isin(accepted_status).all():
            ring_vertex_ids = set(refreshed_ring["mesh_vertex_id"].astype(str))
            angular_edges = logical_edges[logical_edges["mesh_edge_type"].eq("angular_edge")]
            ring_edge_ids = angular_edges.loc[
                angular_edges["left_mesh_vertex_id"].astype(str).isin(ring_vertex_ids)
                & angular_edges["right_mesh_vertex_id"].astype(str).isin(ring_vertex_ids),
                "logical_edge_id",
            ].astype(str).tolist()
            vertices, logical_edges, segments, edge_attempts, edge_conflicts = (
                _certify_logical_edge_chains(
                    config, output_root, "ring_closure", vertices,
                    logical_edges, segments, ring_edge_ids,
                )
            )
            if len(edge_attempts):
                edge_attempts = edge_attempts.copy()
                edge_attempts["ring_candidate_id"] = ring_id
                attempt_frames.append(edge_attempts)
            if len(edge_conflicts):
                edge_conflicts = edge_conflicts.copy()
                edge_conflicts["ring_candidate_id"] = ring_id
                conflict_frames.append(edge_conflicts)

        ring_status = _ring_status_frame(vertices, logical_edges, cells)
        complete_ring_ids = set(
            ring_status.loc[ring_status["complete_ring"].astype(bool), "ring_id"].astype(str)
        )
        _progress(
            output_root, "ring_closure", completed=len(complete_ring_ids),
            total=target_complete_count,
            message=f"attempted_rings={attempted_ring_count} remaining_supervision={remaining_cap}",
        )

    attempts = (
        pd.concat(attempt_frames, ignore_index=True, sort=False)
        if attempt_frames else pd.DataFrame()
    )
    conflicts = (
        pd.concat(conflict_frames, ignore_index=True, sort=False)
        if conflict_frames else pd.DataFrame()
    )
    ring_status = _ring_status_frame(vertices, logical_edges, cells)
    complete = ring_status[ring_status.get("complete_ring", False).astype(bool)].copy() if len(ring_status) else pd.DataFrame()
    angular = logical_edges[logical_edges["mesh_edge_type"].eq("angular_edge")].copy()
    cycle_rows: list[dict[str, Any]] = []
    for ring in ring_status.to_dict("records"):
        ring_vertices = vertices[vertices["ring_id"].astype(str).eq(str(ring["ring_id"]))]
        beta = ring_vertices.sort_values("phi_index", kind="stable").loc[:, BETA_COLUMNS].dropna().to_numpy(float)
        weighted = weighted_beta_rms_deg(beta[:-1], beta[1:]) if len(beta) > 1 else np.asarray([])
        raw = np.max(np.abs(np.degrees(np.diff(beta, axis=0))), axis=1) if len(beta) > 1 else np.asarray([])
        cycle_rows.append(
            {
                "ring_id": ring["ring_id"],
                "complete_ring": bool(ring["complete_ring"]),
                "angular_weighted_p95_deg": percentile(weighted, 95),
                "angular_raw_gt7_rate": float(np.mean(raw > 7.0)) if len(raw) else 1.0,
            }
        )
    _write_parquet(vertices, stage / "mesh_vertex_labels.parquet")
    _write_parquet(logical_edges, stage / "logical_mesh_edges.parquet")
    _write_parquet(segments, stage / "solver_segments.parquet")
    _write_parquet(attempts, stage / "ring_solver_attempts.parquet")
    _write_parquet(conflicts, stage / "ring_branch_conflicts.parquet")
    _write_parquet(ring_status, stage / "ring_status.parquet")
    _write_parquet(angular, stage / "angular_edge_certificate.parquet")
    _write_parquet(pd.DataFrame.from_records(cycle_rows), stage / "cycle_return_audit.parquet")
    _write_parquet(complete, stage / "complete_ring_registry.parquet")
    return _seal_gate(
        output_root,
        config,
        "ring_closure",
        {
            "status": "complete",
            "registered_trajectory_ring_count": len(ring_status),
            "complete_ring_count": len(complete),
            "ring_completion_fraction": len(complete) / len(ring_status) if len(ring_status) else 0.0,
            "seam_first_target_count": seam_first_target_count,
            "attempted_ring_count": attempted_ring_count,
            "remaining_ring_closure_supervision_budget": remaining_cap,
            "branch_conflict_count": len(conflicts),
            "adaptive_fill_authorized": True,
        },
    )


def _annotate_fill_priorities(
    vertices: pd.DataFrame,
    logical_edges: pd.DataFrame,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    result = vertices.copy()
    ring_status = _ring_status_frame(result, logical_edges, cells)
    incomplete = set(
        ring_status.loc[~ring_status["complete_ring"].astype(bool), "ring_id"].astype(str)
    ) if len(ring_status) else set()
    gap_by_ring = (
        ring_status.set_index("ring_id")["maximum_angular_gap_bins"].to_dict()
        if len(ring_status)
        else {}
    )
    required_cells = set(cells.loc[cells["required_shell_cell"], "cell_id"].astype(str))
    accepted = {"seeded", "Gold", "certified_silver", "provisional_silver"}
    served_cells: set[str] = set()
    angular = logical_edges[logical_edges["mesh_edge_type"].eq("angular_edge")]
    for cell_id in required_cells:
        subset = result[result["cell_id"].astype(str).eq(cell_id)]
        ids = set(subset["mesh_vertex_id"].astype(str))
        local = angular[
            angular["left_mesh_vertex_id"].astype(str).isin(ids)
            & angular["right_mesh_vertex_id"].astype(str).isin(ids)
        ]
        if len(subset) and subset["label_status"].isin(accepted).all() and len(local) and local["mesh_edge_verified"].astype(bool).all():
            served_cells.add(cell_id)
    incident_holes: set[str] = set()
    bridge = logical_edges[logical_edges["mesh_edge_type"].isin(["radial_edge", "axial_edge"])]
    for edge in bridge[~bridge["mesh_edge_verified"].astype(bool)].itertuples():
        incident_holes.add(str(edge.left_mesh_vertex_id))
        incident_holes.add(str(edge.right_mesh_vertex_id))
    result["incomplete_trajectory_ring"] = result["ring_id"].astype(str).isin(incomplete)
    result["angular_gap_bins"] = result["ring_id"].astype(str).map(gap_by_ring).fillna(0).astype(int)
    result["unserved_required_shell_cell"] = (
        result["cell_id"].astype(str).isin(required_cells - served_cells)
    )
    result["proposal_service_distance_mm"] = result["proposal_support_distance_mm"].fillna(0.0)
    result["radial_or_axial_mesh_hole"] = result["mesh_vertex_id"].astype(str).isin(incident_holes)
    return result


def stage_adaptive_fill(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["adaptive_fill"]
    if not _gate(output_root, "ring_closure").get("adaptive_fill_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "adaptive_fill",
            {"status": "not_authorized", "symmetry_expansion_authorized": False},
        )
    source = output_root / STAGE_DIRS["ring_closure"]
    vertices = pd.read_parquet(source / "mesh_vertex_labels.parquet")
    logical_edges = pd.read_parquet(source / "logical_mesh_edges.parquet")
    segments = pd.read_parquet(source / "solver_segments.parquet")
    cells = pd.read_parquet(output_root / STAGE_DIRS["shell_domain"] / "ur_cell_support.parquet")
    quotient = pd.read_parquet(
        output_root / STAGE_DIRS["shell_domain"] / "quotient_proposal_registry.parquet"
    )
    total_cap = int(config["fill"]["maximum_new_supervision_targets"])
    already_new = int(
        (
            vertices["label_role"].eq("supervision")
            & vertices["label_origin"].astype(str).str.startswith("retry14_")
        ).sum()
    )
    remaining = max(0, total_cap - already_new)
    low_gain_batches = 0
    batch_rows: list[dict[str, Any]] = []
    attempts: list[pd.DataFrame] = []
    conflicts: list[pd.DataFrame] = []
    metrics = _mesh_metrics(vertices, logical_edges, cells, quotient)
    for batch_id in range(int(config["fill"]["maximum_batches"])):
        required = coverage_fill_required(
            metrics,
            target_shell_coverage=float(config["fill"]["target_shell_coverage"]),
            target_ring_fraction=float(config["fill"]["target_ring_fraction"]),
            target_service_radius_mm=float(config["fill"]["target_service_radius_mm"]),
            allowed_gap_bins=int(config["shell_domain"]["allowed_angular_gap_bins"]),
        )
        if not required or remaining <= 0:
            break
        prioritized = _annotate_fill_priorities(vertices, logical_edges, cells)
        batch_limit = min(
            int(config["fill"]["maximum_targets_per_batch"]), remaining
        )
        targets = select_fill_targets(prioritized, maximum_targets=batch_limit)
        if targets.empty:
            break
        before = dict(metrics)
        vertices, logical_edges, segments, batch_attempts, batch_conflicts = _solve_requested_targets(
            config,
            output_root,
            "adaptive_fill",
            vertices,
            logical_edges,
            segments,
            targets["mesh_vertex_id"].astype(str).tolist(),
            supervision_cap=batch_limit,
        )
        target_ring_ids = set(
            targets.loc[targets["ring_id"].notna(), "ring_id"].astype(str)
        )
        if target_ring_ids:
            accepted_status = {"seeded", "Gold", "certified_silver", "provisional_silver"}
            fully_labelled: set[str] = set()
            for ring_id in sorted(target_ring_ids):
                ring_vertices = vertices[vertices["ring_id"].astype(str).eq(ring_id)]
                if len(ring_vertices) and ring_vertices["label_status"].isin(accepted_status).all():
                    fully_labelled.add(ring_id)
            if fully_labelled:
                ring_by_vertex = vertices.set_index(
                    vertices["mesh_vertex_id"].astype(str)
                )["ring_id"].astype(str).to_dict()
                angular_edges = logical_edges[
                    logical_edges["mesh_edge_type"].eq("angular_edge")
                ]
                edge_ids = angular_edges.loc[
                    angular_edges["left_mesh_vertex_id"].astype(str).map(ring_by_vertex).isin(fully_labelled)
                    & angular_edges["right_mesh_vertex_id"].astype(str).map(ring_by_vertex).isin(fully_labelled)
                    & angular_edges["left_mesh_vertex_id"].astype(str).map(ring_by_vertex).eq(
                        angular_edges["right_mesh_vertex_id"].astype(str).map(ring_by_vertex)
                    ),
                    "logical_edge_id",
                ].astype(str).tolist()
                vertices, logical_edges, segments, edge_attempts, edge_conflicts = (
                    _certify_logical_edge_chains(
                        config, output_root, "adaptive_fill", vertices,
                        logical_edges, segments, edge_ids,
                    )
                )
                if len(edge_attempts):
                    edge_attempts["fill_batch_id"] = batch_id
                    batch_attempts = pd.concat(
                        [batch_attempts, edge_attempts], ignore_index=True, sort=False
                    )
                if len(edge_conflicts):
                    edge_conflicts["fill_batch_id"] = batch_id
                    batch_conflicts = pd.concat(
                        [batch_conflicts, edge_conflicts], ignore_index=True, sort=False
                    )
        if len(batch_attempts):
            batch_attempts["fill_batch_id"] = batch_id
            attempts.append(batch_attempts)
        if len(batch_conflicts):
            batch_conflicts["fill_batch_id"] = batch_id
            conflicts.append(batch_conflicts)
        new_total = int(
            (
                vertices["label_role"].eq("supervision")
                & vertices["label_origin"].astype(str).str.startswith("retry14_")
            ).sum()
        )
        gained = max(0, new_total - already_new)
        remaining = max(0, total_cap - new_total)
        already_new = new_total
        metrics = _mesh_metrics(vertices, logical_edges, cells, quotient)
        low_gain = is_low_gain(before, metrics)
        low_gain_batches = low_gain_batches + 1 if low_gain else 0
        batch_rows.append(
            {
                "fill_batch_id": batch_id,
                "selected_target_count": len(targets),
                "new_supervision_count": gained,
                "low_gain": low_gain,
                "consecutive_low_gain_batches": low_gain_batches,
                **metrics,
            }
        )
        _progress(
            output_root,
            "adaptive_fill",
            completed=new_total,
            total=total_cap,
            message=f"batch={batch_id} complete_rings={int(metrics['complete_ring_count'])}",
        )
        if low_gain_batches >= int(config["fill"]["low_gain_batches_to_plateau"]):
            break
    ring_status = _ring_status_frame(vertices, logical_edges, cells)
    _write_parquet(vertices, stage / "mesh_vertex_labels.parquet")
    _write_parquet(logical_edges, stage / "logical_mesh_edges.parquet")
    _write_parquet(segments, stage / "solver_segments.parquet")
    _write_parquet(
        pd.concat(attempts, ignore_index=True, sort=False) if attempts else pd.DataFrame(),
        stage / "fill_solver_attempts.parquet",
    )
    _write_parquet(
        pd.concat(conflicts, ignore_index=True, sort=False) if conflicts else pd.DataFrame(),
        stage / "fill_branch_conflicts.parquet",
    )
    _write_parquet(pd.DataFrame.from_records(batch_rows), stage / "fill_batch_metrics.parquet")
    _write_parquet(ring_status, stage / "ring_status.parquet")
    required_now = coverage_fill_required(
        metrics,
        target_shell_coverage=float(config["fill"]["target_shell_coverage"]),
        target_ring_fraction=float(config["fill"]["target_ring_fraction"]),
        target_service_radius_mm=float(config["fill"]["target_service_radius_mm"]),
        allowed_gap_bins=int(config["shell_domain"]["allowed_angular_gap_bins"]),
    )
    return _seal_gate(
        output_root,
        config,
        "adaptive_fill",
        {
            "status": "complete",
            "fill_required_initially": True,
            "fill_required_at_stop": required_now,
            "fill_batch_count": len(batch_rows),
            "new_supervision_target_count": already_new,
            "remaining_supervision_budget": remaining,
            "plateau": low_gain_batches >= int(config["fill"]["low_gain_batches_to_plateau"]),
            **metrics,
            "symmetry_expansion_authorized": True,
        },
    )


def _accepted_vertices(vertices: pd.DataFrame) -> pd.DataFrame:
    accepted = {"seeded", "Gold", "certified_silver", "provisional_silver"}
    result = vertices[vertices["label_status"].isin(accepted)].copy()
    finite_columns = [*XYZ_COLUMNS, *BETA_COLUMNS]
    return result[np.isfinite(result.loc[:, finite_columns].to_numpy(float)).all(axis=1)].reset_index(drop=True)


def _lineage_edge_frame(vertices: pd.DataFrame) -> pd.DataFrame:
    ids = set(vertices["mesh_vertex_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for row in vertices.to_dict("records"):
        parent = row.get("canonical_parent_id")
        if parent is None or pd.isna(parent) or str(parent) not in ids:
            continue
        left, right = str(parent), str(row["mesh_vertex_id"])
        rows.append(
            {
                "logical_edge_id": stable_id("retry14_lineage_edge", left, right),
                "left_mesh_vertex_id": left,
                "right_mesh_vertex_id": right,
                "mesh_edge_type": "lineage_edge",
                "mesh_edge_verified": True,
                "segment_chain_ids": str(row.get("canonical_parent_segment_id", "")),
            }
        )
    return pd.DataFrame.from_records(rows)


def _augment_with_legacy_shell_support(
    vertices: pd.DataFrame,
    labels: pd.DataFrame,
    cells: pd.DataFrame,
    zero_xyz: np.ndarray,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Retain zero-rooted labels inside the denominator without faking coverage.

    Structured arcs remain the only authority for served cells and complete
    rings.  A legacy label that lies inside a required shell cell is still a
    legitimate supervision observation, while labels needed only by its parent
    chain remain connector-only.
    """

    result = vertices.copy().reset_index(drop=True)
    required = set(cells.loc[cells["required_shell_cell"], "cell_id"].astype(str))
    axial = float(config["shell_domain"]["axial_step_mm"])
    radial = float(config["shell_domain"]["radial_step_mm"])
    physical_to_mesh: dict[str, str] = {}
    for row in result.to_dict("records"):
        physical = row.get("physical_point_id")
        if physical is not None and not pd.isna(physical) and row.get("label_role") == "supervision":
            physical_to_mesh[str(physical)] = str(row["mesh_vertex_id"])
    legacy_rows: list[dict[str, Any]] = []
    for label in labels.to_dict("records"):
        physical = str(label["physical_point_id"])
        if physical in physical_to_mesh:
            continue
        u_index = int(math.floor((float(zero_xyz[0]) - float(label["x_m"])) * 1000.0 / axial))
        rho_index = int(math.floor(math.hypot(float(label["y_m"]), float(label["z_m"])) * 1000.0 / radial))
        cell_id = f"ur:{u_index}:{rho_index}"
        label_xyz = np.asarray([label[name] for name in XYZ_COLUMNS], dtype=float)
        label_beta = np.asarray([label[name] for name in BETA_COLUMNS], dtype=float)
        seam = _seam_class(label_xyz, 1.0e-9)
        supervision = cell_id in required and _beta_matches_seam(label_beta, seam)
        mesh_id = stable_id("retry14_legacy_shell_support", physical)
        physical_to_mesh[physical] = mesh_id
        legacy_rows.append(
            {
                **label,
                "mesh_vertex_id": mesh_id,
                "cell_id": cell_id if supervision else "legacy_lineage_connector",
                "ring_id": None,
                "phi_index": -1,
                "phi_rad": math.atan2(abs(float(label["z_m"])), abs(float(label["y_m"]))) if supervision else np.nan,
                "target_kind": "legacy_shell_support" if supervision else "legacy_lineage_connector",
                "proposal_support_distance_mm": np.nan,
                "label_role": "supervision" if supervision else "connector_only",
                "label_status": "seeded",
                "label_quality": str(label.get("label_quality", "certified_silver")),
                "canonical_parent_id": label.get("retry13_parent_physical_point_id"),
                "lineage_depth": label.get("retry14_baseline_lineage_depth"),
                "zero_reachable": bool(label.get("zero_reachable", True)),
                "mesh_target_fk_residual_mm": float(label.get("fk_residual_mm", 0.0)),
                "label_origin": "retry13_legacy_shell_support" if supervision else "retry13_legacy_lineage_connector",
                "required_shell_cell": supervision,
                "trajectory_ring_candidate": False,
            }
        )
    if legacy_rows:
        result = pd.concat([result, pd.DataFrame.from_records(legacy_rows)], ignore_index=True, sort=False)
    result["canonical_parent_id"] = result["canonical_parent_id"].map(
        lambda value: physical_to_mesh.get(str(value), value) if value is not None and not pd.isna(value) else None
    )
    return result


def stage_symmetry_expansion(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["symmetry_expansion"]
    if not _gate(output_root, "adaptive_fill").get("symmetry_expansion_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "symmetry_expansion",
            {"status": "not_authorized", "audit_authorized": False},
        )
    source = output_root / STAGE_DIRS["adaptive_fill"]
    fundamental = _accepted_vertices(pd.read_parquet(source / "mesh_vertex_labels.parquet"))
    labels = pd.read_parquet(output_root / STAGE_DIRS["baseline"] / "lineage_dag.parquet")
    cells = pd.read_parquet(output_root / STAGE_DIRS["shell_domain"] / "ur_cell_support.parquet")
    zero_xyz = np.asarray(_read_json(_upstream_path(config, "retry12_zero_anchor"))["xyz0_m"], dtype=float)
    fundamental = _augment_with_legacy_shell_support(
        fundamental, labels, cells, zero_xyz, config
    )
    logical = pd.read_parquet(source / "logical_mesh_edges.parquet")
    ids = set(fundamental["mesh_vertex_id"].astype(str))
    certified = logical[
        logical["mesh_edge_verified"].astype(bool)
        & logical["left_mesh_vertex_id"].astype(str).isin(ids)
        & logical["right_mesh_vertex_id"].astype(str).isin(ids)
    ].copy()
    zero_rows = fundamental[fundamental["target_kind"].eq("exact_zero")]
    if len(zero_rows) != 1:
        raise RuntimeError("retry14 symmetry expansion requires exactly one accepted zero")
    zero_id = str(zero_rows.iloc[0]["mesh_vertex_id"])
    full_nodes, full_mesh_edges = mirror_nodes_and_mesh_edges(
        fundamental, certified, zero_vertex_id=zero_id
    )
    lineage = _lineage_edge_frame(fundamental)
    _, full_lineage_edges = mirror_nodes_and_mesh_edges(
        fundamental, lineage, zero_vertex_id=zero_id
    )
    orbit_registry = (
        full_nodes.groupby("fundamental_mesh_vertex_id", sort=True)
        .agg(
            symmetry_orbit_size=("mesh_vertex_id", "nunique"),
            full_member_ids=("mesh_vertex_id", lambda values: "|".join(sorted(map(str, values)))),
        )
        .reset_index()
    )
    ring_status = pd.read_parquet(source / "ring_status.parquet")
    expanded_rings = ring_status.copy()
    expanded_rings["full_cycle_verified"] = expanded_rings["complete_ring"].astype(bool)
    _write_parquet(fundamental, stage / "fundamental_mesh_nodes.parquet")
    _write_parquet(certified, stage / "fundamental_mesh_edges.parquet")
    _write_parquet(full_nodes, stage / "expanded_nodes.parquet")
    _write_parquet(full_lineage_edges, stage / "expanded_lineage_edges.parquet")
    _write_parquet(full_mesh_edges, stage / "expanded_mesh_edges.parquet")
    _write_parquet(expanded_rings, stage / "expanded_ring_registry.parquet")
    _write_parquet(orbit_registry, stage / "orbit_registry.parquet")
    expected = int(orbit_registry["symmetry_orbit_size"].sum())
    whole_orbit = bool(expected == len(full_nodes))
    mirrored_types = sorted(full_mesh_edges["mesh_edge_type"].astype(str).unique()) if len(full_mesh_edges) else []
    required_types = {
        edge_type
        for edge_type in ("angular_edge", "radial_edge", "axial_edge")
        if edge_type in set(certified.get("mesh_edge_type", pd.Series(dtype=str)).astype(str))
    }
    edge_types_pass = required_types.issubset(mirrored_types)
    return _seal_gate(
        output_root,
        config,
        "symmetry_expansion",
        {
            "status": "complete" if whole_orbit and edge_types_pass else "invalid",
            "fundamental_accepted_node_count": len(fundamental),
            "expanded_node_count": len(full_nodes),
            "fundamental_certified_mesh_edge_count": len(certified),
            "expanded_certified_mesh_edge_count": len(full_mesh_edges),
            "expanded_lineage_edge_count": len(full_lineage_edges),
            "whole_orbit_integrity_pass": whole_orbit,
            "mesh_edge_types_mirrored_pass": edge_types_pass,
            "mirrored_mesh_edge_types": mirrored_types,
            "audit_authorized": bool(whole_orbit and edge_types_pass),
        },
    )


def _rooted_lineage_report(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    zero_rows = nodes[nodes["target_kind"].eq("exact_zero")]
    if len(zero_rows) != 1:
        return pd.DataFrame(), {"zero_count": len(zero_rows), "zero_reachable_count": 0, "all_rows_zero_reachable": False}
    zero_id = str(zero_rows.iloc[0]["mesh_vertex_id"])
    adjacency: dict[str, list[str]] = {str(value): [] for value in nodes["mesh_vertex_id"]}
    for edge in edges.to_dict("records"):
        left = str(edge["left_mesh_vertex_id"]); right = str(edge["right_mesh_vertex_id"])
        adjacency.setdefault(left, []).append(right)
    depth = {zero_id: 0}; queue = [zero_id]
    for current in queue:
        for child in sorted(adjacency.get(current, [])):
            if child not in depth:
                depth[child] = depth[current] + 1; queue.append(child)
    report = nodes[["mesh_vertex_id", "fundamental_mesh_vertex_id", "symmetry_element"]].copy()
    report["zero_reachable"] = report["mesh_vertex_id"].astype(str).isin(depth)
    report["lineage_depth"] = report["mesh_vertex_id"].astype(str).map(depth)
    return report, {
        "zero_count": 1,
        "zero_reachable_count": len(depth),
        "orphan_count": len(nodes) - len(depth),
        "all_rows_zero_reachable": len(depth) == len(nodes),
        "maximum_lineage_depth": int(max(depth.values())) if depth else 0,
    }


def _edge_consistency(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexed = nodes.set_index(nodes["mesh_vertex_id"].astype(str), drop=False)
    rows: list[dict[str, Any]] = []
    for edge in edges.to_dict("records"):
        left_id = str(edge["left_mesh_vertex_id"]); right_id = str(edge["right_mesh_vertex_id"])
        if left_id not in indexed.index or right_id not in indexed.index:
            continue
        left = indexed.loc[left_id, list(BETA_COLUMNS)].to_numpy(float)
        right = indexed.loc[right_id, list(BETA_COLUMNS)].to_numpy(float)
        rows.append(
            {
                "full_logical_edge_id": str(edge.get("full_logical_edge_id", edge.get("logical_edge_id"))),
                "mesh_edge_type": str(edge["mesh_edge_type"]),
                "weighted_beta_gap_deg": normalized_weighted_beta_deg(left, right),
                "raw_beta_gap_deg": raw_beta_max_deg(left, right),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame, {"weighted_beta_gap_p95_deg": math.inf, "raw_beta_gap_gt7_rate": 1.0}
    return frame, {
        "weighted_beta_gap_p95_deg": percentile(frame["weighted_beta_gap_deg"].to_numpy(float), 95),
        "raw_beta_gap_gt7_rate": float(np.mean(frame["raw_beta_gap_deg"].to_numpy(float) > 7.0)),
    }


def _all_branch_conflicts(output_root: Path) -> pd.DataFrame:
    sources = (
        ("seed_mapping", STAGE_DIRS["seed_mapping"], "legacy_seed_branch_conflicts.parquet"),
        ("advancing_front", STAGE_DIRS["advancing_front"], "branch_conflicts.parquet"),
        ("ring_closure", STAGE_DIRS["ring_closure"], "ring_branch_conflicts.parquet"),
        ("adaptive_fill", STAGE_DIRS["adaptive_fill"], "fill_branch_conflicts.parquet"),
    )
    frames: list[pd.DataFrame] = []
    for conflict_stage, directory, filename in sources:
        path = output_root / directory / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["conflict_stage"] = conflict_stage
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def stage_audit(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["audit"]
    if not _gate(output_root, "symmetry_expansion").get("audit_authorized", False):
        return _seal_gate(output_root, config, "audit", {"status": "not_authorized", "dataset_authorized": False})
    expanded = output_root / STAGE_DIRS["symmetry_expansion"]
    fundamental = pd.read_parquet(expanded / "fundamental_mesh_nodes.parquet")
    fundamental_edges = pd.read_parquet(expanded / "fundamental_mesh_edges.parquet")
    nodes = pd.read_parquet(expanded / "expanded_nodes.parquet")
    mesh_edges = pd.read_parquet(expanded / "expanded_mesh_edges.parquet")
    lineage_edges = pd.read_parquet(expanded / "expanded_lineage_edges.parquet")
    cells = pd.read_parquet(output_root / STAGE_DIRS["shell_domain"] / "ur_cell_support.parquet")
    quotient = pd.read_parquet(output_root / STAGE_DIRS["shell_domain"] / "quotient_proposal_registry.parquet")
    lineage_frame, lineage_report = _rooted_lineage_report(nodes, lineage_edges)
    # Coverage denominators require the full registered mesh, including every
    # unresolved vertex.  The accepted/augmented fundamental table is valid for
    # dataset rows and lineage only; using it here would make missing arc
    # vertices disappear and falsely shrink angular gaps.
    mesh_state = pd.read_parquet(
        output_root / STAGE_DIRS["adaptive_fill"] / "mesh_vertex_labels.parquet"
    )
    mesh_state_edges = pd.read_parquet(
        output_root / STAGE_DIRS["adaptive_fill"] / "logical_mesh_edges.parquet"
    )
    metrics = _mesh_metrics(mesh_state, mesh_state_edges, cells, quotient)
    consistency_frame, consistency = _edge_consistency(nodes, mesh_edges)
    ring_status = pd.read_parquet(expanded / "expanded_ring_registry.parquet")
    branch_conflicts = _all_branch_conflicts(output_root)
    supervision_count = int((fundamental["label_role"].eq("supervision")).sum())
    expanded_supervision_count = int((nodes["label_role"].eq("supervision")).sum())
    ring_count = int(metrics["complete_ring_count"])
    green = bool(
        lineage_report["all_rows_zero_reachable"]
        and metrics["shell_cell_coverage"] >= float(config["audit"]["green_shell_coverage_minimum"])
        and metrics["ring_completion_fraction"] >= float(config["audit"]["green_ring_fraction_minimum"])
        and metrics["proposal_service_p95_mm"] <= float(config["audit"]["green_service_p95_maximum_mm"])
        and ring_count >= int(config["ring"]["green_minimum_complete_rings"])
        and consistency["weighted_beta_gap_p95_deg"] <= float(config["ring"]["angular_weighted_p95_maximum_deg"])
        and consistency["raw_beta_gap_gt7_rate"] <= float(config["ring"]["angular_raw_gt_threshold_rate_maximum"])
    )
    yellow = bool(
        lineage_report["all_rows_zero_reachable"]
        and metrics["shell_cell_coverage"] >= float(config["audit"]["yellow_shell_coverage_minimum"])
        and metrics["ring_completion_fraction"] >= float(config["audit"]["yellow_ring_fraction_minimum"])
        and metrics["proposal_service_p95_mm"] <= float(config["audit"]["yellow_service_p95_maximum_mm"])
        and ring_count >= int(config["ring"]["yellow_minimum_complete_rings"])
    )
    data_minimum = bool(
        supervision_count >= int(config["dataset"]["minimum_fundamental_supervision_representatives"])
        and expanded_supervision_count >= int(config["dataset"]["minimum_expanded_rows"])
    )
    status = "green" if green else "yellow" if yellow else "red" if not data_minimum else "coverage_red_data_retained"
    _write_parquet(lineage_frame, stage / "rooted_lineage_audit.parquet")
    _write_parquet(ring_status, stage / "ring_completeness_audit.parquet")
    _write_parquet(consistency_frame, stage / "certified_mesh_edge_consistency.parquet")
    _write_parquet(branch_conflicts, stage / "branch_conflicts.parquet")
    _write_json(stage / "proposal_service_radius.json", {key: value for key, value in metrics.items() if key.startswith("proposal_service")})
    return _seal_gate(
        output_root,
        config,
        "audit",
        {
            "status": status,
            "green_shell_result": green,
            "yellow_shell_result": yellow,
            "rooted_lineage": lineage_report,
            "fundamental_supervision_representative_count": supervision_count,
            "expanded_supervision_row_count": expanded_supervision_count,
            "branch_conflict_count": len(branch_conflicts),
            **metrics,
            **consistency,
            "cycle_rank_at_least_complete_rings": metrics["mesh_cycle_rank"] >= metrics["complete_ring_count"],
            "arbitrary_geometric_neighbors_are_diagnostic_only": True,
            "dataset_authorized": data_minimum,
        },
    )


def _stable_split(block_id: str, config: Mapping[str, Any]) -> str:
    token = f"{int(config['runtime']['seed'])}:{block_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64)
    train = float(config["split"]["train_fraction"])
    validation = train + float(config["split"]["validation_fraction"])
    return "train" if value < train else "validation" if value < validation else "test"


def _heldout_ring_ids(rings: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    complete = rings[rings.get("complete_ring", pd.Series(False, index=rings.index)).astype(bool)].copy()
    if complete.empty:
        return []
    selected: list[str] = []
    u_values = np.array_split(
        np.asarray(sorted(complete["u_index"].unique())),
        min(int(config["split"]["trajectory_holdout_axial_levels"]), complete["u_index"].nunique()),
    )
    for u_group in u_values:
        candidates = complete[complete["u_index"].isin(u_group)]
        rho_choices = np.array_split(
            np.asarray(sorted(candidates["rho_index"].unique())),
            min(int(config["split"]["trajectory_holdout_radial_levels_per_axial"]), candidates["rho_index"].nunique()),
        )
        for rho_group in rho_choices:
            choice = candidates[candidates["rho_index"].isin(rho_group)].sort_values("ring_id", kind="stable").head(1)
            if len(choice):
                selected.append(str(choice.iloc[0]["ring_id"]))
    return list(dict.fromkeys(selected))


def stage_dataset(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["dataset"]
    audit_gate = _gate(output_root, "audit")
    expanded = output_root / STAGE_DIRS["symmetry_expansion"]
    fundamental = pd.read_parquet(expanded / "fundamental_mesh_nodes.parquet")
    nodes = pd.read_parquet(expanded / "expanded_nodes.parquet")
    mesh_edges = pd.read_parquet(expanded / "expanded_mesh_edges.parquet")
    lineage_edges = pd.read_parquet(expanded / "expanded_lineage_edges.parquet")
    rings = pd.read_parquet(output_root / STAGE_DIRS["audit"] / "ring_completeness_audit.parquet")
    holdout_rings = _heldout_ring_ids(rings, config)
    dataset = nodes[nodes["label_role"].eq("supervision")].copy().reset_index(drop=True)
    block_size = int(config["split"]["quotient_macroblock_size_mm"])
    dataset["quotient_macroblock_id"] = quotient_macroblock_id(
        dataset, block_size_mm=block_size
    )
    holdout_blocks = set(dataset.loc[dataset["ring_id"].astype(str).isin(holdout_rings), "quotient_macroblock_id"].astype(str))
    dataset["split_role"] = [
        "trajectory_holdout" if str(block) in holdout_blocks else _stable_split(str(block), config)
        for block in dataset["quotient_macroblock_id"]
    ]
    assert_holdout_macroblock_isolation(dataset)
    dataset["symmetry_orbit_id"] = dataset["fundamental_mesh_vertex_id"].astype(str)
    dataset["physical_point_id"] = dataset["mesh_vertex_id"].astype(str)
    theta = np.asarray([beta_to_theta(value) for value in dataset.loc[:, BETA_COLUMNS].to_numpy(float)])
    for index, column in enumerate(THETA_COLUMNS):
        dataset[column] = theta[:, index]
    dataset_class = "zero_rooted_certified_shell_mesh" if audit_gate.get("green_shell_result") else "domain_limited_zero_rooted_shell_mesh"
    dataset["dataset_id"] = str(config["dataset"]["dataset_id"] if audit_gate.get("green_shell_result") else config["dataset"]["domain_limited_dataset_id"])
    dataset["dataset_class"] = dataset_class
    dataset["kinematics_theta_sign"] = float(config["dataset"]["kinematics_theta_sign_metadata"])
    dataset["scientific_source_fixed_point"] = _git_sha()
    dataset["config_sha256"] = _config_sha(config)
    split_macroblock_leakage = int(dataset.groupby("quotient_macroblock_id")["split_role"].nunique().gt(1).sum())
    split_orbit_leakage = int(dataset.groupby("symmetry_orbit_id")["split_role"].nunique().gt(1).sum())
    expected_theta = np.asarray([beta_to_theta(value) for value in dataset.loc[:, BETA_COLUMNS].to_numpy(float)])
    theta_pass = bool(np.array_equal(theta, expected_theta))
    environment = _environment(config)
    bounds = np.asarray(environment.bounds, dtype=float)
    beta = dataset.loc[:, BETA_COLUMNS].to_numpy(float)
    finite = bool(np.isfinite(dataset.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS, *THETA_COLUMNS]].to_numpy(float)).all())
    bounds_pass = bool(np.all((beta >= bounds[:, 0] - 1e-12) & (beta <= bounds[:, 1] + 1e-12)))
    minimum = int(config["dataset"]["minimum_expanded_rows"])
    complete = bool(audit_gate.get("dataset_authorized", False) and len(dataset) >= minimum and finite and bounds_pass and theta_pass and split_macroblock_leakage == 0 and split_orbit_leakage == 0)
    fundamental_selected = fundamental[fundamental["label_role"].eq("supervision")].copy()
    split_by_fundamental = dataset.groupby("fundamental_mesh_vertex_id", sort=False)["split_role"].first()
    block_by_fundamental = dataset.groupby("fundamental_mesh_vertex_id", sort=False)["quotient_macroblock_id"].first()
    fundamental_selected["split_role"] = fundamental_selected["mesh_vertex_id"].astype(str).map(split_by_fundamental)
    fundamental_selected["quotient_macroblock_id"] = fundamental_selected["mesh_vertex_id"].astype(str).map(block_by_fundamental)
    _write_parquet(fundamental_selected, stage / "selected_fundamental_representatives.parquet")
    _write_parquet(dataset, stage / "retry14_zero_rooted_shell_mesh_kinematic_dataset.parquet")
    _write_parquet(mesh_edges, stage / "selected_expanded_mesh_edges.parquet")
    _write_parquet(lineage_edges, stage / "selected_expanded_lineage_edges.parquet")
    _write_parquet(rings[rings["ring_id"].astype(str).isin(holdout_rings)], stage / "heldout_trajectory_rings.parquet")
    _write_parquet(pd.DataFrame({"removed_orbit_id": []}), stage / "pruned_orbits.parquet")
    return _seal_gate(
        output_root,
        config,
        "dataset",
        {
            "status": "complete" if complete else "invalid_or_underfilled",
            "dataset_freeze_complete": complete,
            "dataset_row_count": len(dataset),
            "target_range_complete": int(config["dataset"]["target_range_minimum_rows"]) <= len(dataset) <= int(config["dataset"]["target_range_maximum_rows"]),
            "over_target_preserved_to_protect_mesh": len(dataset) > int(config["dataset"]["target_range_maximum_rows"]),
            "theta_storage_direct_beta_expansion_pass": theta_pass,
            "theta_sign_applied_to_storage": False,
            "finite_pass": finite,
            "bounds_pass": bounds_pass,
            "quotient_macroblock_split_leakage_count": split_macroblock_leakage,
            "orbit_split_leakage_count": split_orbit_leakage,
            "heldout_trajectory_ring_count": len(holdout_rings),
            "dataset_class": dataset_class,
            "student_execution_authorized": complete,
        },
    )


def _trajectory_panel(
    dataset: pd.DataFrame,
    mesh_edges: pd.DataFrame,
    lineage_edges: pd.DataFrame,
    holdout_rings: pd.DataFrame,
) -> list[tuple[str, str, pd.DataFrame]]:
    panels: list[tuple[str, str, pd.DataFrame]] = []
    ring_ids = holdout_rings.get("ring_id", pd.Series(dtype=str)).astype(str).tolist()
    for ring_id in ring_ids:
        ring = dataset[dataset["ring_id"].astype(str).eq(ring_id)].copy()
        if len(ring) < 4:
            continue
        ring["full_phi_rad"] = np.mod(np.arctan2(ring["z_m"], ring["y_m"]), 2.0 * np.pi)
        ring = ring.sort_values(["full_phi_rad", "mesh_vertex_id"], kind="stable").drop_duplicates(
            ["x_m", "y_m", "z_m"], keep="first"
        )
        ring = pd.concat([ring, ring.iloc[[0]]], ignore_index=True)
        panels.append((f"complete_circle:{ring_id}", "complete_circle", ring))
    indexed = dataset.set_index(dataset["mesh_vertex_id"].astype(str), drop=False)
    for edge_type, trajectory_class in (("radial_edge", "radial_transition"), ("axial_edge", "axial_transition")):
        candidates = mesh_edges[mesh_edges["mesh_edge_type"].eq(edge_type)].head(9)
        for edge in candidates.to_dict("records"):
            left, right = str(edge["left_mesh_vertex_id"]), str(edge["right_mesh_vertex_id"])
            if left in indexed.index and right in indexed.index:
                panels.append((f"{trajectory_class}:{edge.get('full_logical_edge_id', edge.get('logical_edge_id'))}", trajectory_class, indexed.loc[[left, right]].reset_index(drop=True)))
    if ring_ids and len(dataset):
        ring = dataset[dataset["ring_id"].astype(str).eq(ring_ids[0])]
        zero = dataset[dataset["target_kind"].eq("exact_zero")]
        if len(ring) and len(zero):
            target = str(ring.sort_values("mesh_vertex_id", kind="stable").iloc[0]["mesh_vertex_id"])
            zero_id = str(zero.iloc[0]["mesh_vertex_id"])
            parent = {
                str(row["right_mesh_vertex_id"]): str(row["left_mesh_vertex_id"])
                for row in lineage_edges.to_dict("records")
            }
            path = [target]; seen = {target}
            while path[-1] != zero_id and path[-1] in parent:
                current = parent[path[-1]]
                if current in seen:
                    break
                seen.add(current); path.append(current)
            if path[-1] == zero_id and all(value in indexed.index for value in path):
                panels.append(("zero_to_ring_anchor", "zero_to_ring", indexed.loc[list(reversed(path))].reset_index(drop=True)))
    return panels


def _evaluate_trajectories(
    model: Any,
    environment: Any,
    panels: Sequence[tuple[str, str, pd.DataFrame]],
    zero_xyz: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    waypoint_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    complete_circle_successes = 0
    complete_circle_count = 0
    pooled_circle_fk: list[float] = []
    pooled_circle_steps: list[float] = []
    threshold = float(config["trajectory"]["corrected_beta_step_raw_threshold_deg"])
    for trajectory_id, trajectory_class, frame in panels:
        xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float)
        raw = _symmetry_prediction(model, xyz, zero_xyz)
        corrected = _two_step_dls(environment, raw, xyz, zero_xyz=zero_xyz)
        raw_fk = np.linalg.norm(np.asarray(environment.fk(raw)).reshape(-1, 3) - xyz, axis=1) * 1000.0
        corrected_fk = np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - xyz, axis=1) * 1000.0
        metrics = corrected_trajectory_metrics(raw, corrected)
        correction = np.max(np.abs(np.degrees(corrected - raw)), axis=1)
        corrected_steps = np.max(np.abs(np.degrees(np.diff(corrected, axis=0))), axis=1) if len(corrected) > 1 else np.asarray([])
        success = bool(
            np.isfinite(corrected).all()
            and percentile(corrected_fk, 95) <= float(config["trajectory"]["fk_p95_maximum_mm"])
            and (float(np.mean(corrected_steps > threshold)) if len(corrected_steps) else 0.0)
            <= float(config["trajectory"]["corrected_beta_step_raw_gt_threshold_rate_maximum"])
        )
        if trajectory_class == "complete_circle":
            complete_circle_count += 1
            complete_circle_successes += int(success)
            pooled_circle_fk.extend(corrected_fk.tolist())
            pooled_circle_steps.extend(corrected_steps.tolist())
        reports.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_class": trajectory_class,
                "waypoint_count": len(frame),
                "success": success,
                "raw_fk_p95_mm": percentile(raw_fk, 95),
                "dls2_fk_p95_mm": percentile(corrected_fk, 95),
                "correction_magnitude_p95_deg": percentile(correction, 95),
                "dls2_loop_return_raw_deg": raw_beta_max_deg(corrected[0], corrected[-1]) if trajectory_class == "complete_circle" else None,
                **metrics,
            }
        )
        for index in range(len(frame)):
            record = {
                "trajectory_id": trajectory_id,
                "trajectory_class": trajectory_class,
                "waypoint_index": index,
                **dict(zip(XYZ_COLUMNS, xyz[index], strict=True)),
                "raw_fk_residual_mm": raw_fk[index],
                "dls2_fk_residual_mm": corrected_fk[index],
                "correction_magnitude_deg": correction[index],
            }
            record.update({f"raw_{name}": raw[index, axis] for axis, name in enumerate(BETA_COLUMNS)})
            record.update({f"dls2_{name}": corrected[index, axis] for axis, name in enumerate(BETA_COLUMNS)})
            waypoint_rows.append(record)
    success_rate = complete_circle_successes / complete_circle_count if complete_circle_count else 0.0
    pooled_p95 = percentile(np.asarray(pooled_circle_fk), 95) if pooled_circle_fk else math.inf
    step_rate = float(np.mean(np.asarray(pooled_circle_steps) > threshold)) if pooled_circle_steps else 1.0
    gate_pass = bool(
        success_rate >= float(config["trajectory"]["complete_circle_dls2_success_minimum"])
        and pooled_p95 <= float(config["trajectory"]["fk_p95_maximum_mm"])
        and step_rate <= float(config["trajectory"]["corrected_beta_step_raw_gt_threshold_rate_maximum"])
    )
    return pd.DataFrame.from_records(waypoint_rows), pd.DataFrame.from_records(reports), {
        "trajectory_gate_pass": gate_pass,
        "complete_circle_count": complete_circle_count,
        "complete_circle_dls2_success_rate": success_rate,
        "complete_circle_pooled_dls2_fk_p95_mm": pooled_p95,
        "complete_circle_pooled_corrected_step_gt7_rate": step_rate,
    }


def stage_student_trajectory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["student_trajectory"]
    dataset_gate = _gate(output_root, "dataset")
    if not dataset_gate.get("student_execution_authorized", False):
        _write_parquet(pd.DataFrame(), stage / "trajectory_waypoints.parquet")
        _write_parquet(pd.DataFrame(), stage / "trajectory_report.parquet")
        return _seal_gate(
            output_root,
            config,
            "student_trajectory",
            {"status": "not_authorized", "student_training_complete": False, "trajectory_gate_pass": False},
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("retry14 Student requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("retry14 CPU Student unexpectedly sees a GPU")
    source = output_root / STAGE_DIRS["dataset"]
    supervision = pd.read_parquet(source / "selected_fundamental_representatives.parquet")
    supervision = supervision[connector_loss_mask(supervision)].copy()
    environment = _environment(config)
    jacobians = np.asarray([
        np.asarray(environment.jacobian(beta), dtype=float).reshape(-1)
        for beta in supervision.loc[:, BETA_COLUMNS].to_numpy(float)
    ])
    for index, column in enumerate(JACOBIAN_COLUMNS):
        supervision[column] = jacobians[:, index]
    supervision["record_id"] = supervision["mesh_vertex_id"].astype(str)
    supervision["kind"] = "static"
    supervision["chart_id"] = "retry14_zero_rooted_shell_quotient"
    supervision["is_primary"] = True
    supervision["sample_weight"] = supervision["label_quality"].astype(str).map(
        config["student"]["quality_weight"]
    ).fillna(0.5).astype(float)
    train = supervision[supervision["split_role"].eq("train")].copy()
    validation = supervision[supervision["split_role"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise RuntimeError("retry14 quotient macroblock split produced empty train or validation")
    student_config = config["student"]
    trained = train_workspace_student(
        train,
        validation,
        mode=RepresentationMode.XYZ_GLOBAL,
        geometry=_student_geometry(environment),
        config=WorkspaceStudentTrainingConfig(
            hidden_units=tuple(map(int, student_config["hidden_units"])),
            learning_rate=float(student_config["learning_rate"]),
            max_steps=int(student_config["maximum_steps"]),
            validation_interval=int(student_config["validation_interval"]),
            patience_intervals=int(student_config["patience_intervals"]),
            seed=int(student_config["seed"]),
            beta_coordinate_weights=(4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
            beta_loss_only=True,
        ),
    )
    save_workspace_student_models(trained.models, stage / "models")
    model = trained.models.global_model
    validation_xyz = validation.loc[:, XYZ_COLUMNS].to_numpy(float)
    zero_xyz = np.asarray(_read_json(_upstream_path(config, "retry12_zero_anchor"))["xyz0_m"], dtype=float)
    prediction = np.asarray(model(validation_xyz.astype(np.float32), training=False), dtype=float)
    corrected = _two_step_dls(environment, prediction, validation_xyz, zero_xyz=zero_xyz)
    raw_fk = np.linalg.norm(np.asarray(environment.fk(prediction)).reshape(-1, 3) - validation_xyz, axis=1) * 1000.0
    dls_fk = np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - validation_xyz, axis=1) * 1000.0
    predictions = validation[["mesh_vertex_id", *XYZ_COLUMNS, *BETA_COLUMNS]].copy()
    for index, name in enumerate(BETA_COLUMNS):
        predictions[f"predicted_{name}"] = prediction[:, index]
        predictions[f"dls2_{name}"] = corrected[:, index]
    predictions["raw_fk_residual_mm"] = raw_fk
    predictions["dls2_fk_residual_mm"] = dls_fk
    bounds = np.asarray(environment.bounds, dtype=float)
    finite = bool(np.isfinite(prediction).all() and np.isfinite(corrected).all())
    bounds_pass = bool(np.all((prediction >= bounds[:, 0] - 1e-12) & (prediction <= bounds[:, 1] + 1e-12)))
    dataset = pd.read_parquet(source / "retry14_zero_rooted_shell_mesh_kinematic_dataset.parquet")
    mesh_edges = pd.read_parquet(source / "selected_expanded_mesh_edges.parquet")
    lineage_edges = pd.read_parquet(source / "selected_expanded_lineage_edges.parquet")
    holdout = pd.read_parquet(source / "heldout_trajectory_rings.parquet")
    panels = _trajectory_panel(dataset, mesh_edges, lineage_edges, holdout)
    waypoints, reports, trajectory_gate = _evaluate_trajectories(
        model, environment, panels, zero_xyz, config
    )
    _write_parquet(supervision, stage / "student_supervision.parquet")
    _write_parquet(trained.history, stage / "training_history.parquet")
    _write_parquet(predictions, stage / "validation_predictions.parquet")
    _write_parquet(waypoints, stage / "trajectory_waypoints.parquet")
    _write_parquet(reports, stage / "trajectory_report.parquet")
    student_complete = bool(finite and bounds_pass)
    return _seal_gate(
        output_root,
        config,
        "student_trajectory",
        {
            "status": "complete" if student_complete else "student_red",
            "student_training_complete": student_complete,
            "train_row_count": len(train),
            "validation_row_count": len(validation),
            "connector_only_rows_in_loss": 0,
            "finite_pass": finite,
            "bounds_pass": bounds_pass,
            "validation_raw_fk_p95_mm": percentile(raw_fk, 95),
            "validation_dls2_fk_p95_mm": percentile(dls_fk, 95),
            "exact_seam_dls_coordinates_frozen": True,
            "corrected_beta_continuity_is_hard_gate": True,
            **trajectory_gate,
        },
    )


def stage_summary(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    _write_final_progress(output_root)
    stage = output_root / STAGE_DIRS["summary"]
    audit = _gate(output_root, "audit")
    dataset = _gate(output_root, "dataset")
    student = _gate(output_root, "student_trajectory")
    operational = all(_stage_is_complete(output_root, name) for name in STAGE_ORDER[:-1])
    gate = {
        "status": "complete" if operational else "incomplete",
        "operational_completion": operational,
        "artifact_completeness": operational,
        "scientific_result": audit.get("status"),
        "coverage": {
            "shell_cell_coverage": audit.get("shell_cell_coverage"),
            "ring_completion_fraction": audit.get("ring_completion_fraction"),
            "proposal_service_p95_mm": audit.get("proposal_service_p95_mm"),
            "complete_ring_count": audit.get("complete_ring_count"),
        },
        "consistency": {
            "certified_edge_weighted_beta_gap_p95_deg": audit.get("weighted_beta_gap_p95_deg"),
            "certified_edge_raw_beta_gap_gt7_rate": audit.get("raw_beta_gap_gt7_rate"),
            "branch_conflict_count": audit.get("branch_conflict_count"),
        },
        "lineage": audit.get("rooted_lineage"),
        "mesh_cycle_rank": audit.get("mesh_cycle_rank"),
        "dataset": {
            "status": dataset.get("status"),
            "row_count": dataset.get("dataset_row_count"),
            "class": dataset.get("dataset_class"),
            "complete": dataset.get("dataset_freeze_complete", False),
        },
        "student": {
            "status": student.get("status"),
            "training_complete": student.get("student_training_complete", False),
            "validation_dls2_fk_p95_mm": student.get("validation_dls2_fk_p95_mm"),
        },
        "trajectory": {
            "gate_pass": student.get("trajectory_gate_pass", False),
            "complete_circle_count": student.get("complete_circle_count", 0),
            "complete_circle_dls2_success_rate": student.get("complete_circle_dls2_success_rate"),
        },
        "row_count_is_not_coverage_authority": True,
        "lineage_tree_is_not_shell_mesh": True,
        "theta_sign_applied_to_storage": False,
        "tension_executed": False,
        "formal_authorization": False,
        "deployment_authorization": False,
        "full_workspace_authorization": False,
        "continuous_workspace_authorization": False,
    }
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "retry14_summary.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>retry14 summary</title></head>"
        f"<body><h1>retry14 零点有根壳网格与环轨迹覆盖实验</h1><pre>{json.dumps(gate, ensure_ascii=False, indent=2)}</pre></body></html>",
        encoding="utf-8",
    )
    sealed = _seal_gate(output_root, config, "summary", gate)
    _write_json(stage / "artifact_manifest.json", _artifact_manifest(output_root))
    _seal_stage(output_root, "summary")
    return sealed


STAGE_RUNNERS: dict[str, Callable[[Mapping[str, Any], Path, Path], dict[str, Any]]] = {
    "baseline": stage_baseline,
    "shell_domain": stage_shell_domain,
    "target_mesh": stage_target_mesh,
    "seed_mapping": stage_seed_mapping,
    "advancing_front": stage_advancing_front,
    "ring_closure": stage_ring_closure,
    "adaptive_fill": stage_adaptive_fill,
    "symmetry_expansion": stage_symmetry_expansion,
    "audit": stage_audit,
    "dataset": stage_dataset,
    "student_trajectory": stage_student_trajectory,
    "summary": stage_summary,
}


def run(
    config_path: str | Path,
    output_root: str | Path,
    binding_sha: str,
    *,
    stage_name: str | None = None,
    validate_stage: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_root).resolve()
    _ensure_identity(config, output, binding_sha)
    if validate_stage is not None:
        if validate_stage not in STAGE_RUNNERS:
            raise ValueError(f"unknown stage {validate_stage}")
        if not _stage_is_complete(output, validate_stage):
            raise RuntimeError(f"retry14 stage is incomplete or mutated: {validate_stage}")
        return _gate(output, validate_stage)
    if stage_name is not None:
        if stage_name not in STAGE_RUNNERS:
            raise ValueError(f"unknown stage {stage_name}")
        position = STAGE_ORDER.index(stage_name)
        if position and not _stage_is_complete(output, STAGE_ORDER[position - 1]):
            raise RuntimeError(f"retry14 predecessor incomplete: {STAGE_ORDER[position - 1]}")
        _progress(output, stage_name, message="stage_started")
        result = STAGE_RUNNERS[stage_name](config, project_root_from(SOURCE_ROOT), output)
        if stage_name != "summary":
            _progress(output, stage_name, completed=1, total=1, message="stage_completed")
        return result
    result: dict[str, Any] = {}
    for name in STAGE_ORDER:
        if _stage_is_complete(output, name):
            result = _gate(output, name); continue
        result = STAGE_RUNNERS[name](config, project_root_from(SOURCE_ROOT), output)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--binding-sha", required=True)
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--validate-stage", choices=STAGE_ORDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.config, args.output_root, args.binding_sha, stage_name=args.stage, validate_stage=args.validate_stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
