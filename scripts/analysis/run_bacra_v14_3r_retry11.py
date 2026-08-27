#!/usr/bin/env python3
"""Run BACRA V14.3R retry11 dual-track outer closure and zero bridge POC."""

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
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import yaml

from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.canonical import weighted_damped_pinv
from quasi_exp.teacher.canonical_atlas import AtlasCandidate, AtlasTaskNode
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import percentile
from quasi_exp.teacher.optimized_continuation import (
    make_iterative_weighted_dls_continuation,
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.retry10 import (
    BETA_COLUMNS,
    METRIC_VERSION,
    XYZ_COLUMNS,
    normalized_weighted_beta_deg,
    raw_beta_max_deg,
)
from quasi_exp.teacher.retry11_bridge import (
    classify_retry11_targets,
    corridor_path_registry,
    discover_graph_components,
    evaluate_bridge_certificate,
    zero_seed_candidate_registry,
    zero_seed_status,
)
from quasi_exp.teacher.retry11_sampling import (
    annotate_spatial_strata,
    apply_macroblock_split_registry,
    build_macroblock_split_registry,
    graph_shell_width_mm,
    hierarchical_replacement_candidates,
    select_parent_first_sparse_wide,
    stable_row_id,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    save_workspace_student_models,
    train_workspace_student,
)


STAGE_DIRS = {
    "inventory": "00_inventory",
    "outer_sparse_wide": "01_outer_sparse_wide",
    "outer_student": "02_outer_student",
    "outer_theta": "03_outer_theta",
    "outer_tension_pilot": "04_outer_tension_pilot",
    "outer_tension_materialization": "05_outer_tension_materialization",
    "zero_seed": "06_zero_seed",
    "zero_frontier_poc": "07_zero_frontier_poc",
    "zero_outer_bridge": "08_zero_outer_bridge",
    "summary": "09_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True
    ).strip()


def _config_sha(config: Mapping[str, Any]) -> str:
    return sha256_file(Path(str(config["config_path"])))


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported retry11 config")
    config = dict(value)
    config["config_path"] = str(config_path)
    if config.get("experiment_id") != "bacra_v14_3r_retry11_dual_track_zero_bridge":
        raise ValueError("retry11 experiment identity mismatch")
    if str(config["metric"]["version"]) != METRIC_VERSION:
        raise ValueError("retry11 requires normalized_weighted_v1")
    if tuple(map(float, config["metric"]["beta_coordinate_weights"])) != (
        4.0,
        4.0,
        2.0,
        2.0,
        1.0,
        1.0,
    ):
        raise ValueError("retry11 beta weights are frozen")
    runtime = config["runtime"]
    if (
        str(runtime["device"]) != "cpu"
        or int(runtime["maximum_concurrent_workers"]) != 12
        or int(runtime["logical_shard_count"]) != 48
    ):
        raise ValueError("retry11 requires CPU, twelve workers and 48 logical shards")
    dataset = config["outer_dataset"]
    if (
        int(dataset["mandatory_rows"]) != 10_455
        or int(dataset["target_rows"]) != 20_000
        or int(dataset["minimum_rows"]) != 18_000
    ):
        raise ValueError("retry11 outer dataset is frozen at 10455 -> 20k/min18k")
    tension = config["tension"]
    if not (
        float(tension["yellow_success_rate_min"]) == 0.80
        and float(tension["green_success_rate_min"]) == 0.95
        and bool(tension["projected_200k_is_diagnostic_only"])
    ):
        raise ValueError("retry11 tension uses the frozen 80/95 diagnostic-runtime contract")
    seed = config["zero_seed"]
    if (
        int(seed["candidate_count"]) != 8
        or int(seed["green_valid_count_min"]) != 4
        or int(seed["yellow_valid_count_min"]) != 2
    ):
        raise ValueError("retry11 zero seed requires 8 candidates and 4/2 thresholds")
    if str(config["track_b_quality"]["scheme"]) != "retry11_gold_wide_silver_v1":
        raise ValueError("retry11 label quality scheme mismatch")
    if any(
        bool(config["claims"][key])
        for key in (
            "formal_authorized",
            "deployment_authorized",
            "full_workspace_authorized",
            "unified_component_student_authorized",
            "automatic_component_merge_authorized",
        )
    ):
        raise ValueError("retry11 cannot authorize formal, deployment or component merge")
    return config


def _upstream_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["upstream"]["retry10_root"]))


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    row = config["upstream"][key]
    if "absolute_path" in row:
        return Path(str(row["absolute_path"]))
    return _upstream_root(config) / str(row["path"])


def _binding_definition(binding_sha: str, experiment_id: str) -> Mapping[str, Any]:
    payload = subprocess.check_output(
        ["git", "show", f"{binding_sha}:spec/registry.yaml"],
        cwd=SOURCE_ROOT,
        text=True,
    )
    registry = yaml.safe_load(payload)
    definitions = registry.get("experiments", registry)
    if experiment_id not in definitions:
        raise ValueError(f"binding {binding_sha} does not register {experiment_id}")
    definition = definitions[experiment_id]
    if not isinstance(definition, Mapping):
        raise ValueError("retry11 registry definition is not a mapping")
    return definition


def _stage_upstream_digest(output_root: Path, stage_name: str) -> str:
    digest = hashlib.sha256()
    for name in STAGE_ORDER[: STAGE_ORDER.index(stage_name)]:
        path = output_root / STAGE_DIRS[name] / "completion_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing upstream completion manifest: {path}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_identity(output_root: Path) -> dict[str, Any]:
    return _read_json(output_root / "run_identity.json")


def _seal_stage(
    output_root: Path, config: Mapping[str, Any], stage_name: str
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = [
        {
            "path": str(path.relative_to(stage)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    ]
    identity = _run_identity(output_root)
    manifest = {
        "schema_version": 1,
        "stage_name": stage_name,
        "scientific_source_fixed_point": _git_sha(),
        "binding_fixed_point": identity["binding_fixed_point"],
        "config_sha256": _config_sha(config),
        "upstream_completion_sha256": _stage_upstream_digest(output_root, stage_name),
        "artifacts": artifacts,
    }
    _write_json(stage / "completion_manifest.json", manifest)
    return manifest


def _seal_gate(
    output_root: Path,
    config: Mapping[str, Any],
    stage_name: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    _write_json(stage / "gate.json", dict(gate))
    _seal_stage(output_root, config, stage_name)
    return dict(gate)


def _gate(output_root: Path, stage_name: str) -> dict[str, Any]:
    return _read_json(output_root / STAGE_DIRS[stage_name] / "gate.json")


def _stage_is_complete(
    output_root: Path, config: Mapping[str, Any], stage_name: str
) -> bool:
    stage = output_root / STAGE_DIRS[stage_name]
    manifest_path = stage / "completion_manifest.json"
    if not manifest_path.is_file() or not (output_root / "run_identity.json").is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        identity = _run_identity(output_root)
        if not (
            manifest["stage_name"] == stage_name
            and manifest["scientific_source_fixed_point"] == _git_sha()
            and manifest["binding_fixed_point"] == identity["binding_fixed_point"]
            and manifest["config_sha256"] == _config_sha(config)
            and manifest["upstream_completion_sha256"]
            == _stage_upstream_digest(output_root, stage_name)
        ):
            return False
        return all(
            (stage / str(row["path"])).is_file()
            and sha256_file(stage / str(row["path"])) == str(row["sha256"])
            for row in manifest["artifacts"]
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _environment(config: Mapping[str, Any]) -> Any:
    reference = load_environment(
        project_root_from(SOURCE_ROOT),
        SOURCE_ROOT / str(config["sources"]["robot_config"]),
    )
    return optimized_forward(reference)


def _thread_limited_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }
    )
    return environment


def _progress(
    output_root: Path,
    *,
    status: str,
    stage_name: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    _write_json(
        output_root / "progress.json",
        {
            "status": status,
            "phase": stage_name,
            "completed": int(completed),
            "total": int(total),
            "message": str(message),
            "updated_at_unix": time.time(),
        },
    )


def _source_frames(config: Mapping[str, Any]) -> tuple[pd.DataFrame, ...]:
    return (
        pd.read_parquet(_upstream_path(config, "frozen_atlas")),
        pd.read_parquet(_upstream_path(config, "accepted_candidate_pool")),
        pd.read_parquet(_upstream_path(config, "task_nodes")),
        pd.read_parquet(_upstream_path(config, "refined_task_edges")),
    )


def stage_inventory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["inventory"]
    checks: dict[str, bool] = {}
    source_rows: list[dict[str, Any]] = []
    consultation_path = Path(str(config["sources"]["consultation_input"]))
    if not consultation_path.is_absolute():
        consultation_path = SOURCE_ROOT / consultation_path
    source_paths = {
        "consultation_input": consultation_path,
        "governing_protocol": SOURCE_ROOT / str(config["sources"]["governing_protocol"]),
        "robot_config": SOURCE_ROOT / str(config["sources"]["robot_config"]),
        "tension_robot_config": SOURCE_ROOT / str(config["sources"]["tension_robot_config"]),
    }
    source_hashes = {
        "consultation_input": str(config["consultation_sha256"]),
        "robot_config": str(config["sources"]["robot_config_sha256"]),
        "tension_robot_config": str(config["sources"]["tension_robot_config_sha256"]),
    }
    for name, path in source_paths.items():
        observed = sha256_file(path) if path.is_file() else None
        expected = source_hashes.get(name)
        passed = path.is_file() and (expected is None or observed == expected)
        checks[f"source_{name}"] = passed
        source_rows.append(
            {
                "source_id": name,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "pass": passed,
            }
        )
    for key, row in config["upstream"].items():
        if not isinstance(row, Mapping) or "sha256" not in row:
            continue
        path = _upstream_path(config, key)
        observed = sha256_file(path) if path.is_file() else None
        passed = observed == str(row["sha256"])
        checks[f"upstream_{key}"] = passed
        source_rows.append(
            {
                "source_id": key,
                "path": str(path),
                "expected_sha256": str(row["sha256"]),
                "observed_sha256": observed,
                "pass": passed,
            }
        )

    identity = _run_identity(output_root)
    try:
        definition = _binding_definition(
            str(identity["binding_fixed_point"]), str(config["experiment_id"])
        )
        checks["binding_scientific_fixed_point"] = (
            str(definition["scientific_source_fixed_point"]) == _git_sha()
        )
        checks["binding_config"] = str(definition["config"]) == str(
            Path(str(config["config_path"])).relative_to(SOURCE_ROOT)
        )
        checks["binding_runner"] = str(definition["runner"]) == str(
            Path(__file__).resolve().relative_to(SOURCE_ROOT)
        )
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError):
        checks["binding_scientific_fixed_point"] = False
        checks["binding_config"] = False
        checks["binding_runner"] = False

    frozen, accepted, tasks, edges = _source_frames(config)
    checks["frozen_row_count"] = len(frozen) == int(config["outer_dataset"]["mandatory_rows"])
    checks["frozen_unique_task_nodes"] = not frozen["task_node_id"].astype(int).duplicated().any()
    checks["accepted_pool_nonempty"] = bool(accepted.get("accepted", False).astype(bool).sum())
    checks["task_nodes_unique"] = not tasks["task_node_id"].astype(int).duplicated().any()
    checks["task_graph_nonempty"] = len(edges) > 0

    environment = _environment(config)
    zero_beta = np.zeros(6, dtype=float)
    zero_xyz = np.asarray(environment.fk(zero_beta), dtype=float).reshape(-1, 3)[0]
    zero_roundtrip = np.asarray(environment.fk(zero_beta.copy()), dtype=float).reshape(-1, 3)[0]
    checks["zero_anchor_finite"] = bool(np.isfinite(zero_xyz).all())
    checks["zero_anchor_bounds"] = bool(
        np.all(zero_beta >= np.asarray(environment.bounds)[:, 0])
        and np.all(zero_beta <= np.asarray(environment.bounds)[:, 1])
    )
    checks["zero_anchor_roundtrip"] = bool(
        np.max(np.abs(zero_roundtrip - zero_xyz))
        <= float(config["zero_seed"]["exact_anchor_roundtrip_tolerance_m"])
    )
    components, component_report = discover_graph_components(
        tasks,
        edges,
        zero_xyz_m=zero_xyz,
        outer_task_node_ids=frozen["task_node_id"].astype(int),
    )
    shell_width, edge_p95 = graph_shell_width_mm(
        tasks,
        edges,
        percentile=float(config["spatial_registry"]["shell_edge_percentile"]),
        multiplier=float(config["spatial_registry"]["shell_edge_multiplier"]),
        minimum_mm=float(config["spatial_registry"]["shell_width_min_mm"]),
        maximum_mm=float(config["spatial_registry"]["shell_width_max_mm"]),
    )
    _write_parquet(pd.DataFrame.from_records(source_rows), stage / "source_inventory.parquet")
    _write_parquet(components, stage / "component_registry.parquet")
    _write_json(
        stage / "zero_anchor.json",
        {
            "zero_configuration_beta_rad": zero_beta.tolist(),
            "zero_configuration_xyz_m": zero_xyz.tolist(),
            "roundtrip_absolute_error_m": float(np.max(np.abs(zero_roundtrip - zero_xyz))),
            "robot_config_sha256": str(config["sources"]["robot_config_sha256"]),
            "fk_implementation_sha256": sha256_file(
                SOURCE_ROOT / "src/quasi_exp/model/endpoint_kinematics.py"
            ),
        },
    )
    passed = bool(all(checks.values()))
    gate = {
        "status": "pass" if passed else "fail",
        "shared_gate_pass": passed,
        "checks": checks,
        "shell_width_mm": shell_width,
        "registered_edge_p95_mm": edge_p95,
        **component_report,
        "formal_authorized": False,
        "deployment_authorized": False,
    }
    return _seal_gate(output_root, config, "inventory", gate)


def _normalize_outer_rows(
    frame: pd.DataFrame,
    *,
    origin: str,
    zero_xyz: Sequence[float],
    shell_width_mm: float,
    component: Mapping[str, Any],
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    if "xyz_key" not in result:
        result["xyz_key"] = result.loc[:, XYZ_COLUMNS].round(12).astype(str).agg("|".join, axis=1)
    if "physical_point_id" not in result:
        result["physical_point_id"] = result["xyz_key"].astype(str).map(
            lambda value: stable_row_id(f"retry11_{origin}", value)
        )
    result["canonical_component_id"] = str(component["canonical_component_id"])
    result["component_role"] = str(component["component_role"])
    result["zero_centered_primary"] = bool(component["zero_centered_primary"])
    result["canonical_lineage_id"] = result["physical_point_id"].astype(str).map(
        lambda value: f"outer:{value}"
    )
    result["sampling_origin"] = result.get("sampling_origin", origin)
    result["weighted_consistency_deg"] = result.get(
        "multiparent_weighted_gap_deg", 0.0
    )
    result["fk_residual_mm"] = result.get(
        "teacher_fk_residual_mm", result.get("fk_residual_mm", 0.0)
    )
    return annotate_spatial_strata(
        result, zero_xyz_m=zero_xyz, shell_width_mm=shell_width_mm
    )


def stage_outer_sparse_wide(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["outer_sparse_wide"]
    inventory = _gate(output_root, "inventory")
    if not inventory.get("shared_gate_pass", False):
        return _seal_gate(
            output_root,
            config,
            "outer_sparse_wide",
            {
                "status": "not_authorized",
                "teacher_data_integrity_pass": False,
                "reason": "shared_inventory_failed",
                "formal_authorized": False,
            },
        )
    frozen, attempts, _tasks, _edges = _source_frames(config)
    zero = _read_json(output_root / STAGE_DIRS["inventory"] / "zero_anchor.json")
    component = config["components"]["outer"]
    mandatory = _normalize_outer_rows(
        frozen,
        origin="mandatory_frozen_atlas",
        zero_xyz=zero["zero_configuration_xyz_m"],
        shell_width_mm=float(inventory["shell_width_mm"]),
        component=component,
    )
    accepted = attempts[attempts["accepted"].astype(bool)].copy()
    accepted = accepted[
        accepted["label_quality"].astype(str).isin(config["outer_dataset"]["allowed_qualities"])
    ]
    numeric = accepted.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy(float)
    accepted = accepted[np.isfinite(numeric).all(axis=1)].copy()
    accepted = accepted[accepted["actual_bounds"].astype(bool)].copy()
    if "solver_success_count" in accepted and "multiparent_raw_gap_deg" in accepted:
        accepted = accepted[
            accepted["solver_success_count"].astype(int).lt(2)
            | accepted["multiparent_raw_gap_deg"].astype(float).le(5.0 + 1.0e-12)
        ].copy()
    accepted = _normalize_outer_rows(
        accepted,
        origin="retry10_accepted_candidate_pool",
        zero_xyz=zero["zero_configuration_xyz_m"],
        shell_width_mm=float(inventory["shell_width_mm"]),
        component=component,
    )
    panels = {
        name: pd.read_parquet(_upstream_path(config, f"panel_{name.lower()}"))
        for name in ("A", "B", "C")
    }
    split_registry = build_macroblock_split_registry(
        [mandatory, accepted],
        panels,
        block_size_mm=int(config["spatial_registry"]["macroblock_size_mm"]),
        seed=int(config["runtime"]["seed"]),
    )
    mandatory = apply_macroblock_split_registry(
        mandatory,
        split_registry,
        block_size_mm=int(config["spatial_registry"]["macroblock_size_mm"]),
    )
    accepted = apply_macroblock_split_registry(
        accepted,
        split_registry,
        block_size_mm=int(config["spatial_registry"]["macroblock_size_mm"]),
    )
    dataset, audit, selection = select_parent_first_sparse_wide(
        mandatory,
        accepted,
        target_rows=int(config["outer_dataset"]["target_rows"]),
        minimum_rows=int(config["outer_dataset"]["minimum_rows"]),
    )
    environment = _environment(config)
    bounds = np.asarray(environment.bounds, dtype=float)
    beta = dataset.loc[:, BETA_COLUMNS].to_numpy(float)
    finite_pass = bool(
        np.isfinite(dataset.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy(float)).all()
    )
    bounds_pass = bool(
        np.all(beta >= bounds[:, 0] - 1.0e-12)
        and np.all(beta <= bounds[:, 1] + 1.0e-12)
    )
    mandatory_ids = set(mandatory["physical_point_id"].astype(str))
    preserved = mandatory_ids.issubset(set(dataset["physical_point_id"].astype(str)))
    round1_pass = math.isclose(
        float(selection["new_parent_service_coverage_round1"]),
        float(config["outer_dataset"]["required_round1_new_parent_service_coverage"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    branch_purity_pass = not bool(
        (
            dataset.get("solver_success_count", pd.Series(0, index=dataset.index)).fillna(0).astype(int).ge(2)
            & dataset.get("multiparent_raw_gap_deg", pd.Series(0.0, index=dataset.index)).fillna(0.0).astype(float).gt(5.0 + 1.0e-12)
        ).any()
    )
    teacher_pass = bool(
        preserved
        and finite_pass
        and bounds_pass
        and branch_purity_pass
        and round1_pass
        and bool(selection["parent_first_scheduler_invariant"])
        and int(selection["duplicate_xyz_count"]) == 0
        and int(selection["duplicate_physical_point_count"]) == 0
        and len(dataset) >= int(config["outer_dataset"]["minimum_rows"])
    )
    dataset["dataset_id"] = str(config["outer_dataset"]["dataset_id"])
    dataset["provenance"] = dataset.apply(
        lambda row: json.dumps(
            {
                "source": str(row["selection_origin"]),
                "physical_point_id": str(row["physical_point_id"]),
                "source_parent_node_id": int(row["source_parent_node_id"]),
            },
            sort_keys=True,
        ),
        axis=1,
    )
    _write_parquet(dataset, stage / "outer_sparse_wide_beta_dataset_v1.parquet")
    _write_parquet(audit, stage / "parent_first_selection_audit.parquet")
    _write_parquet(accepted, stage / "annotated_accepted_candidate_pool.parquet")
    _write_parquet(split_registry, stage / "macroblock_split_registry.parquet")
    _write_json(stage / "selection_report.json", selection)
    gate = {
        "status": "pass" if teacher_pass else "fail",
        "teacher_data_integrity_pass": teacher_pass,
        "dataset_id": str(config["outer_dataset"]["dataset_id"]),
        "actual_rows": len(dataset),
        "mandatory_row_count": len(mandatory),
        "mandatory_seed_preserved": preserved,
        "finite_pass": finite_pass,
        "bounds_pass": bounds_pass,
        "same_point_branch_purity_pass": branch_purity_pass,
        "minimum_rows_pass": len(dataset) >= int(config["outer_dataset"]["minimum_rows"]),
        "exact_target_pass": len(dataset) == int(config["outer_dataset"]["target_rows"]),
        **selection,
        "shell_sector_quota_is_diagnostic_only": True,
        "total_parent_coverage_is_diagnostic_only": True,
        "theta_execution_authorized": teacher_pass,
        "tension_pilot_authorized": teacher_pass,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "outer_sparse_wide", gate)


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )


def _two_step_dls(environment: Any, beta: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    corrected = np.asarray(beta, dtype=float).copy()
    bounds = np.asarray(environment.bounds, dtype=float)
    weights = tuple(map(float, (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)))
    for _ in range(2):
        for index in range(len(corrected)):
            current = np.asarray(environment.fk(corrected[index]), dtype=float).reshape(-1, 3)[0]
            jacobian = np.asarray(environment.jacobian(corrected[index]), dtype=float).reshape(3, 6)
            step = weighted_damped_pinv(jacobian, damping=1.0e-3, weights=weights) @ (
                xyz[index] - current
            )
            corrected[index] = np.clip(corrected[index] + step, bounds[:, 0], bounds[:, 1])
    return corrected


def _evaluate_student_panel(
    model: Any,
    panel: pd.DataFrame,
    environment: Any,
    *,
    panel_id: str,
    dls_success_max_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    xyz = panel.loc[:, XYZ_COLUMNS].to_numpy(np.float32)
    truth = panel.loc[:, BETA_COLUMNS].to_numpy(float)
    started = time.perf_counter()
    prediction = np.asarray(model(xyz, training=False), dtype=float)
    raw_latency = (time.perf_counter() - started) / max(1, len(panel))
    raw_fk = np.linalg.norm(np.asarray(environment.fk(prediction)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    started = time.perf_counter()
    corrected = _two_step_dls(environment, prediction, xyz)
    dls_latency = (time.perf_counter() - started) / max(1, len(panel))
    dls_fk = np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    bounds = np.asarray(environment.bounds, dtype=float)
    violation = ~np.all(
        (prediction >= bounds[:, 0] - 1.0e-12)
        & (prediction <= bounds[:, 1] + 1.0e-12),
        axis=1,
    )
    weighted = np.asarray(
        [normalized_weighted_beta_deg(left, right) for left, right in zip(prediction, truth, strict=True)]
    )
    dls_success = bool(len(panel)) and (
        np.isfinite(corrected).all(axis=1)
        & np.all(
            (corrected >= bounds[:, 0] - 1.0e-12)
            & (corrected <= bounds[:, 1] + 1.0e-12),
            axis=1,
        )
        & (dls_fk <= float(dls_success_max_mm))
    )
    result = panel[[name for name in ("task_node_id", *XYZ_COLUMNS, *BETA_COLUMNS) if name in panel]].copy()
    result["panel_id"] = panel_id
    for index, column in enumerate(BETA_COLUMNS):
        result[f"predicted_{column}"] = prediction[:, index]
        result[f"dls2_{column}"] = corrected[:, index]
    result["weighted_beta_error_deg"] = weighted
    result["raw_fk_residual_mm"] = raw_fk
    result["dls2_fk_residual_mm"] = dls_fk
    result["bounds_violation"] = violation
    result["dls_success"] = dls_success
    report = {
        "panel_id": panel_id,
        "row_count": len(panel),
        "no_nan": bool(
            np.isfinite(prediction).all()
            and np.isfinite(raw_fk).all()
            and np.isfinite(dls_fk).all()
        ),
        "bounds_violation_count": int(np.count_nonzero(violation)),
        "weighted_beta_p95_deg": percentile(weighted, 95),
        "raw_fk_p95_mm": percentile(raw_fk, 95),
        "dls_two_step_fk_p95_mm": percentile(dls_fk, 95),
        "dls_two_step_fk_p99_mm": percentile(dls_fk, 99),
        "dls_success_rate": float(np.mean(dls_success)),
        "raw_latency_ms_per_row": raw_latency * 1000.0,
        "dls_latency_ms_per_row": dls_latency * 1000.0,
    }
    return result, report


def stage_outer_student(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["outer_student"]
    dataset_gate = _gate(output_root, "outer_sparse_wide")
    teacher_pass = bool(dataset_gate.get("teacher_data_integrity_pass", False))
    if not teacher_pass:
        return _seal_gate(
            output_root,
            config,
            "outer_student",
            {
                "status": "red",
                "student_status": "red",
                "reason": "teacher_data_integrity_red",
                "theta_execution_authorized": False,
                "tension_pilot_authorized": False,
                "formal_authorized": False,
            },
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("retry11 Student requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf

    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("retry11 CPU Student unexpectedly sees a GPU")
    dataset = pd.read_parquet(
        output_root / STAGE_DIRS["outer_sparse_wide"] / "outer_sparse_wide_beta_dataset_v1.parquet"
    )
    environment = _environment(config)
    supervision = dataset[dataset["split_role"].astype(str).isin(("train_core", "validation"))].copy()
    jacobians = np.asarray(
        [
            np.asarray(environment.jacobian(beta), dtype=float).reshape(-1)
            for beta in supervision.loc[:, BETA_COLUMNS].to_numpy(float)
        ]
    )
    for index, column in enumerate(JACOBIAN_COLUMNS):
        supervision[column] = jacobians[:, index]
    supervision["record_id"] = supervision["physical_point_id"].astype(str)
    supervision["kind"] = "static"
    supervision["chart_id"] = "outer_component"
    supervision["is_primary"] = True
    weights = config["student"]["quality_weight"]
    supervision["sample_weight"] = supervision["label_quality"].astype(str).map(weights).astype(float)
    train = supervision[supervision["split_role"].eq("train_core")].copy()
    validation = supervision[supervision["split_role"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise RuntimeError("retry11 frozen macroblock split produced an empty train/validation set")
    student = config["student"]
    trained = train_workspace_student(
        train,
        validation,
        mode=RepresentationMode.XYZ_GLOBAL,
        geometry=_student_geometry(environment),
        config=WorkspaceStudentTrainingConfig(
            hidden_units=tuple(map(int, student["hidden_units"])),
            learning_rate=float(student["learning_rate"]),
            max_steps=int(student["max_steps"]),
            validation_interval=int(student["validation_interval"]),
            patience_intervals=int(student["patience_intervals"]),
            seed=int(student["seed"]),
            beta_coordinate_weights=tuple(map(float, config["metric"]["beta_coordinate_weights"])),
            beta_loss_only=True,
        ),
    )
    save_workspace_student_models(trained.models, stage / "models")
    _write_parquet(supervision, stage / "student_supervision.parquet")
    _write_parquet(trained.history, stage / "training_history.parquet")
    panel_reports: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for panel_id in ("A", "B", "C"):
        panel = pd.read_parquet(_upstream_path(config, f"panel_{panel_id.lower()}"))
        predictions, report = _evaluate_student_panel(
            trained.models.global_model,
            panel,
            environment,
            panel_id=panel_id,
            dls_success_max_mm=float(student["dls_success_fk_max_mm"]),
        )
        _write_parquet(predictions, stage / f"panel_{panel_id}_predictions.parquet")
        prediction_frames.append(predictions)
        panel_reports[panel_id] = report
    pooled = pd.concat(prediction_frames, ignore_index=True, sort=False)
    finite_pass = bool(
        np.isfinite(
            pooled[[
                *[f"predicted_{name}" for name in BETA_COLUMNS],
                "raw_fk_residual_mm",
                "dls2_fk_residual_mm",
            ]].to_numpy(float)
        ).all()
    )
    bounds_pass = not bool(pooled["bounds_violation"].astype(bool).any())
    success_rate = float(pooled["dls_success"].astype(bool).mean())
    p95 = percentile(pooled["dls2_fk_residual_mm"], 95)
    p99 = percentile(pooled["dls2_fk_residual_mm"], 99)
    green = bool(
        finite_pass
        and bounds_pass
        and success_rate >= float(student["green_success_rate_min"])
        and p95 <= float(student["green_dls_fk_p95_max_mm"])
        and p99 <= float(student["green_dls_fk_p99_max_mm"])
    )
    yellow = bool(
        not green
        and finite_pass
        and bounds_pass
        and success_rate >= float(student["yellow_success_rate_min"])
    )
    status = "green" if green else "yellow" if yellow else "red"
    gate = {
        "status": status,
        "student_status": status,
        "student_quality_green": green,
        "student_quality_yellow": yellow,
        "dataset_row_count": len(dataset),
        "train_row_count": len(train),
        "validation_row_count": len(validation),
        "panels": panel_reports,
        "pooled_panel_row_count": len(pooled),
        "finite_pass": finite_pass,
        "bounds_pass": bounds_pass,
        "dls2_success_rate": success_rate,
        "dls2_fk_p95_mm": p95,
        "dls2_fk_p99_mm": p99,
        "student_red_is_not_track_a_physics_red": True,
        "theta_execution_authorized": teacher_pass,
        "tension_pilot_authorized": teacher_pass,
        "physics_authorization_source": "teacher_data_integrity",
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "outer_student", gate)


def stage_outer_theta(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["outer_theta"]
    dataset_gate = _gate(output_root, "outer_sparse_wide")
    authorized = bool(dataset_gate.get("teacher_data_integrity_pass", False))
    if not authorized:
        return _seal_gate(
            output_root,
            config,
            "outer_theta",
            {
                "status": "not_authorized",
                "theta_generation_complete": False,
                "reason": "teacher_data_integrity_red",
                "student_status_ignored_for_physics_authorization": True,
                "formal_authorized": False,
            },
        )
    dataset = pd.read_parquet(
        output_root / STAGE_DIRS["outer_sparse_wide"] / "outer_sparse_wide_beta_dataset_v1.parquet"
    )
    environment = _environment(config)
    theta_sign = float(config["theta"]["theta_sign"])
    if not math.isclose(theta_sign, float(environment.theta_sign), abs_tol=0.0):
        raise RuntimeError("retry11 theta sign does not match robot environment")
    beta = dataset.loc[:, BETA_COLUMNS].to_numpy(float)
    theta = np.vstack([beta_to_theta(row) for row in beta]) * theta_sign
    result = dataset.copy()
    for index in range(theta.shape[1]):
        result[f"theta{index + 1}_rad"] = theta[:, index]
    condition: list[float] = []
    for row in beta:
        singular = np.linalg.svd(np.asarray(environment.jacobian(row), dtype=float), compute_uv=False)
        condition.append(float(math.inf if singular[-1] <= 0.0 else singular[0] / singular[-1]))
    result["condition_number"] = condition
    theta_columns = [f"theta{index}_rad" for index in range(1, 31)]
    finite_pass = bool(np.isfinite(result.loc[:, theta_columns].to_numpy(float)).all())
    shape_pass = theta.shape == (len(dataset), int(config["theta"]["output_width"]))
    complete = bool(finite_pass and shape_pass and len(result) == len(dataset))
    _write_parquet(result, stage / "outer_sparse_wide_theta_dataset_v1.parquet")
    gate = {
        "status": "pass" if complete else "fail",
        "theta_generation_complete": complete,
        "row_count": len(result),
        "output_width": theta.shape[1],
        "shape_pass": shape_pass,
        "finite_pass": finite_pass,
        "theta_sign": theta_sign,
        "student_status_ignored_for_physics_authorization": True,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "outer_theta", gate)


def _tension_worker(
    config: Mapping[str, Any], worker_root: Path, shard_id: int
) -> dict[str, Any]:
    from quasi_exp.io import load_config as load_robot_config, load_robot_inputs
    from quasi_exp.model.quasi_static import QuasiStaticModel
    from quasi_exp.opt.tension_labeler import solve_tension_label

    frame = pd.read_parquet(worker_root / "tension_registry.parquet")
    frame = frame[frame["shard_id"].astype(int).eq(int(shard_id))]
    robot_path = SOURCE_ROOT / str(config["sources"]["tension_robot_config"])
    robot = load_robot_config(robot_path)
    inputs = load_robot_inputs(robot)
    model = QuasiStaticModel(robot, inputs)
    residual_max = float(config["tension"]["residual_max"])
    tension_min, tension_max = map(float, config["tension"]["tension_bounds_n"])
    pso_config = robot.get("pso", {})
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        started = time.perf_counter()
        beta = np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float)
        try:
            cache = model.build_cache(beta_to_theta(beta))
            solved = solve_tension_label(
                model=model,
                cache=cache,
                pso_cfg=pso_config,
                pso_seed=int(config["runtime"]["seed"]) + int(row.tension_candidate_id),
                rms_thresh=residual_max,
            )
            tension = np.asarray(solved.T_base_12, dtype=float).reshape(12)
            residual = float(solved.meta.get("rms_rnorm", math.inf))
            success = bool(
                solved.ok
                and np.isfinite(tension).all()
                and residual < residual_max
                and np.all(tension >= tension_min - 1.0e-12)
                and np.all(tension <= tension_max + 1.0e-12)
            )
            meta = dict(solved.meta)
        except Exception as error:
            tension = np.full(12, np.nan)
            residual = math.inf
            success = False
            meta = {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
        record = row._asdict()
        record.update(
            {
                **{f"T{index + 1}_N": float(tension[index]) for index in range(12)},
                "tension_success": success,
                "tension_residual": residual,
                "tension_elapsed_s": time.perf_counter() - started,
                "tension_solver_method": str(
                    robot.get("tension_labeler", {}).get("method", "unknown")
                ),
                "tension_meta_json": json.dumps(meta, sort_keys=True, default=_json_default),
            }
        )
        rows.append(record)
    result = pd.DataFrame.from_records(rows)
    shard_root = worker_root / f"shard_{int(shard_id):02d}"
    _write_parquet(result, shard_root / "tension.parquet")
    report = {
        "shard_id": int(shard_id),
        "target_count": len(frame),
        "success_count": int(
            result.get("tension_success", pd.Series(dtype=bool)).astype(bool).sum()
        ),
    }
    _write_json(shard_root / "report.json", report)
    return report


def _run_tension_registry(
    config: Mapping[str, Any],
    directory: Path,
    registry: pd.DataFrame,
    *,
    output_root: Path,
    progress_phase: str,
) -> pd.DataFrame:
    frame = registry.copy().reset_index(drop=True)
    frame["tension_candidate_id"] = np.arange(len(frame), dtype=np.int64)
    frame["shard_id"] = frame["tension_candidate_id"].map(
        lambda value: int(
            hashlib.sha256(
                f"tension:{int(config['runtime']['seed'])}:{int(value)}".encode()
            ).hexdigest()[:16],
            16,
        )
        % int(config["runtime"]["logical_shard_count"])
    )
    _write_parquet(frame, directory / "tension_registry.parquet")
    shard_ids = list(range(int(config["runtime"]["logical_shard_count"])))
    pending = list(shard_ids)
    running: list[tuple[int, subprocess.Popen[str], Any]] = []
    completed = 0
    while pending or running:
        while pending and len(running) < int(config["tension"]["maximum_concurrent_workers"]):
            shard_id = pending.pop(0)
            shard = directory / f"shard_{shard_id:02d}"
            shard.mkdir(parents=True, exist_ok=True)
            handle = (shard / "worker.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config["config_path"]),
                    "--tension-worker-root",
                    str(directory),
                    "--shard-id",
                    str(shard_id),
                ],
                cwd=SOURCE_ROOT,
                env=_thread_limited_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((shard_id, process, handle))
        survivors: list[tuple[int, subprocess.Popen[str], Any]] = []
        for shard_id, process, handle in running:
            if process.poll() is None:
                survivors.append((shard_id, process, handle))
                continue
            handle.close()
            if process.returncode != 0:
                raise RuntimeError(f"retry11 tension worker failed: shard {shard_id}")
            completed += 1
            _progress(
                output_root,
                status="running",
                stage_name=progress_phase,
                completed=completed,
                total=len(shard_ids),
                message="tension shards complete",
            )
        running = survivors
        if running:
            time.sleep(0.2)
    frames = [
        pd.read_parquet(directory / f"shard_{shard_id:02d}" / "tension.parquet")
        for shard_id in shard_ids
    ]
    nonempty = [frame for frame in frames if len(frame)]
    return pd.concat(nonempty, ignore_index=True, sort=False) if nonempty else pd.DataFrame()


def _pilot_registry(dataset: pd.DataFrame, *, target_rows: int, seed: int) -> pd.DataFrame:
    frame = dataset.copy().reset_index(drop=True)
    frame["condition_tertile"] = pd.qcut(
        frame["condition_number"].rank(method="first"), 3, labels=False
    )
    frame["pilot_stratum"] = (
        frame["shell_sector"].astype(str)
        + ":"
        + frame["label_quality"].astype(str)
        + ":condition_"
        + frame["condition_tertile"].astype(str)
        + ":"
        + frame["boundary_role"].astype(str)
    )
    frame["_rank"] = frame["physical_point_id"].astype(str).map(
        lambda value: hashlib.sha256(f"{int(seed)}:{value}".encode()).digest()
    )
    queues = {
        stratum: group.sort_values("_rank", kind="stable").index.tolist()
        for stratum, group in frame.groupby("pilot_stratum", sort=True)
    }
    selected: list[int] = []
    strata = sorted(queues)
    while len(selected) < int(target_rows):
        progressed = False
        for stratum in strata:
            if queues[stratum] and len(selected) < int(target_rows):
                selected.append(queues[stratum].pop(0))
                progressed = True
        if not progressed:
            break
    return frame.loc[selected].drop(columns="_rank").reset_index(drop=True)


def _runtime_projection_hours(
    elapsed_s: Sequence[float], *, target_rows: int, worker_count: int
) -> float:
    values = np.asarray(elapsed_s, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.inf
    return float(int(target_rows) * float(np.mean(values)) / int(worker_count) / 3600.0)


def stage_outer_tension_pilot(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["outer_tension_pilot"]
    theta_gate = _gate(output_root, "outer_theta")
    if not theta_gate.get("theta_generation_complete", False):
        return _seal_gate(
            output_root,
            config,
            "outer_tension_pilot",
            {
                "status": "red",
                "tension_status": "red",
                "full_materialization_authorized": False,
                "reason": "teacher_beta_theta_integrity_red",
                "formal_authorized": False,
            },
        )
    dataset = pd.read_parquet(
        output_root / STAGE_DIRS["outer_theta"] / "outer_sparse_wide_theta_dataset_v1.parquet"
    )
    registry = _pilot_registry(
        dataset,
        target_rows=int(config["tension"]["pilot_rows"]),
        seed=int(config["runtime"]["seed"]),
    )
    registry["tension_attempt_role"] = "pilot_primary"
    registry["replacement_for_row_id"] = None
    registry["replacement_level"] = "primary"
    solved = _run_tension_registry(
        config,
        stage / "workers",
        registry,
        output_root=output_root,
        progress_phase="outer_tension_pilot",
    )
    successes = solved[solved["tension_success"].astype(bool)].copy()
    success_rate = float(solved["tension_success"].astype(bool).mean()) if len(solved) else 0.0
    elapsed = solved["tension_elapsed_s"].to_numpy(float) if len(solved) else np.asarray([])
    worker_count = int(config["tension"]["maximum_concurrent_workers"])
    projections = {
        int(target): _runtime_projection_hours(
            elapsed, target_rows=int(target), worker_count=worker_count
        )
        for target in config["tension"]["runtime_projection_targets"]
    }
    if success_rate >= float(config["tension"]["green_success_rate_min"]):
        status = "green"
    elif success_rate >= float(config["tension"]["yellow_success_rate_min"]):
        status = "yellow"
    else:
        status = "red"
    full_authorized = status in {"green", "yellow"}
    runtime_attention = projections[20_000] > float(
        config["runtime"]["registered_attention_budget_hours"]
    )
    _write_parquet(solved, stage / "tension_pilot_attempts.parquet")
    _write_parquet(successes, stage / "tension_pilot_successes.parquet")
    gate = {
        "status": status,
        "tension_status": status,
        "tension_success_rate": success_rate,
        "pilot_target_rows": int(config["tension"]["pilot_rows"]),
        "pilot_attempt_rows": len(solved),
        "pilot_success_rows": len(successes),
        "runtime_p50_s": percentile(elapsed, 50),
        "runtime_p95_s": percentile(elapsed, 95),
        "observed_worker_count": worker_count,
        "projected_20k_wall_time_hours": projections[20_000],
        "projected_50k_wall_time_hours": projections[50_000],
        "projected_200k_wall_time_hours": projections[200_000],
        "runtime_projection_formula": "target_rows * observed_mean_row_seconds / observed_worker_count / 3600",
        "runtime_attention": runtime_attention,
        "registered_attention_budget_hours": float(
            config["runtime"]["registered_attention_budget_hours"]
        ),
        "projected_200k_is_diagnostic_only": True,
        "full_materialization_authorized": full_authorized,
        "tension_degraded": status == "yellow",
        "physics_quality_claim_authorized": status == "green",
        "residual_p50": percentile(successes.get("tension_residual", ()), 50),
        "residual_p95": percentile(successes.get("tension_residual", ()), 95),
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "outer_tension_pilot", gate)


def _attach_theta_columns(frame: pd.DataFrame, theta_sign: float) -> pd.DataFrame:
    result = frame.copy()
    beta = result.loc[:, BETA_COLUMNS].to_numpy(float)
    theta = np.vstack([beta_to_theta(row) for row in beta]) * float(theta_sign)
    for index in range(theta.shape[1]):
        result[f"theta{index + 1}_rad"] = theta[:, index]
    return result


def _stage_artifact_manifest(stage: Path, scientific_sha: str) -> dict[str, Any]:
    artifacts = [
        {
            "path": str(path.relative_to(stage)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "completion_manifest.json", "gate.json"}
    ]
    return {
        "schema_version": 1,
        "scientific_source_fixed_point": scientific_sha,
        "artifacts": artifacts,
    }


def stage_outer_tension_materialization(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["outer_tension_materialization"]
    pilot_gate = _gate(output_root, "outer_tension_pilot")
    authorized = bool(pilot_gate.get("full_materialization_authorized", False))
    if not authorized:
        checkpoint = {
            "status": "red",
            "outer_end_to_end_closure_complete": False,
            "outer_exact_target_complete": False,
            "reason": "tension_pilot_below_80_percent_or_teacher_integrity_red",
            "track_b_may_continue": True,
        }
        _write_json(stage / "track_a_checkpoint.json", checkpoint)
        _write_json(
            stage / "artifact_manifest.json", _stage_artifact_manifest(stage, _git_sha())
        )
        return _seal_gate(
            output_root,
            config,
            "outer_tension_materialization",
            {
                **checkpoint,
                "tension_materialization_complete": False,
                "formal_authorized": False,
            },
        )

    primary = pd.read_parquet(
        output_root / STAGE_DIRS["outer_theta"] / "outer_sparse_wide_theta_dataset_v1.parquet"
    )
    pilot_attempts = pd.read_parquet(
        output_root / STAGE_DIRS["outer_tension_pilot"] / "tension_pilot_attempts.parquet"
    )
    pilot_ids = set(pilot_attempts["physical_point_id"].astype(str))
    remaining = primary[~primary["physical_point_id"].astype(str).isin(pilot_ids)].copy()
    remaining["tension_attempt_role"] = "full_primary"
    remaining["replacement_for_row_id"] = None
    remaining["replacement_level"] = "primary"
    solved_remaining = _run_tension_registry(
        config,
        stage / "primary_workers",
        remaining,
        output_root=output_root,
        progress_phase="outer_tension_materialization_primary",
    )
    primary_attempts = pd.concat(
        [pilot_attempts, solved_remaining], ignore_index=True, sort=False
    ).sort_values("physical_point_id", kind="stable")
    if primary_attempts["physical_point_id"].astype(str).duplicated().any():
        raise RuntimeError("retry11 primary tension attempts contain duplicate identities")
    primary_success = primary_attempts[primary_attempts["tension_success"].astype(bool)].copy()
    failed = primary_attempts[~primary_attempts["tension_success"].astype(bool)].copy()
    failed_by_id = {
        str(row.physical_point_id): row._asdict() for row in failed.itertuples(index=False)
    }
    unresolved = list(sorted(failed_by_id))
    candidate_pool = pd.read_parquet(
        output_root
        / STAGE_DIRS["outer_sparse_wide"]
        / "annotated_accepted_candidate_pool.parquet"
    )
    used_ids = set(primary["physical_point_id"].astype(str))
    replacement_successes: list[pd.DataFrame] = []
    replacement_attempts: list[pd.DataFrame] = []
    replacement_failure_ledger: list[dict[str, Any]] = []
    for replacement_round in range(1, int(config["tension"]["replacement_max_rounds"]) + 1):
        registry_rows: list[pd.Series] = []
        for failed_id in unresolved:
            queue = hierarchical_replacement_candidates(
                failed_by_id[failed_id],
                candidate_pool,
                used_physical_point_ids=tuple(used_ids),
            )
            if queue.empty:
                replacement_failure_ledger.append(
                    {
                        "failed_primary_physical_point_id": failed_id,
                        "replacement_round": replacement_round,
                        "reason": "hierarchical_replacement_queue_exhausted",
                    }
                )
                continue
            chosen = queue.iloc[0].copy()
            chosen["replacement_for_row_id"] = failed_id
            chosen["replacement_round"] = replacement_round
            chosen["tension_attempt_role"] = "replacement"
            used_ids.add(str(chosen["physical_point_id"]))
            registry_rows.append(chosen)
        if not registry_rows:
            break
        replacement_registry = pd.DataFrame(registry_rows).reset_index(drop=True)
        replacement_registry = _attach_theta_columns(
            replacement_registry, float(config["theta"]["theta_sign"])
        )
        solved = _run_tension_registry(
            config,
            stage / f"replacement_round_{replacement_round:02d}_workers",
            replacement_registry,
            output_root=output_root,
            progress_phase=f"outer_tension_replacement_round_{replacement_round}",
        )
        replacement_attempts.append(solved)
        successes = solved[solved["tension_success"].astype(bool)].copy()
        if len(successes):
            replacement_successes.append(successes)
        resolved = set(successes["replacement_for_row_id"].astype(str))
        unresolved = [failed_id for failed_id in unresolved if failed_id not in resolved]
        if not unresolved:
            break

    completed_parts = [primary_success]
    completed_parts.extend(replacement_successes)
    completed = pd.concat(completed_parts, ignore_index=True, sort=False)
    if completed["physical_point_id"].astype(str).duplicated().any():
        raise RuntimeError("retry11 complete tension dataset contains duplicate physical points")
    complete_count = len(completed)
    minimum = int(config["outer_dataset"]["minimum_rows"])
    target = int(config["outer_dataset"]["target_rows"])
    closure = complete_count >= minimum
    exact = complete_count == target
    all_attempts = [primary_attempts, *replacement_attempts]
    attempts = pd.concat(all_attempts, ignore_index=True, sort=False)
    full_success_rate = float(attempts["tension_success"].astype(bool).mean())
    degraded = bool(
        str(pilot_gate.get("tension_status")) == "yellow" or full_success_rate < 0.95
    )
    completed["dataset_id"] = str(config["outer_dataset"]["complete_dataset_id"])
    completed["tension_degraded"] = degraded
    completed["provenance"] = completed.apply(
        lambda row: json.dumps(
            {
                "primary_or_replacement": str(row.get("tension_attempt_role", "primary")),
                "replacement_for_row_id": row.get("replacement_for_row_id"),
                "replacement_level": str(row.get("replacement_level", "primary")),
                "tension_candidate_id": int(row["tension_candidate_id"]),
            },
            sort_keys=True,
            default=_json_default,
        ),
        axis=1,
    )
    _write_parquet(completed, stage / "outer_sparse_wide_complete_dataset_v1.parquet")
    _write_parquet(attempts, stage / "tension_attempts.parquet")
    _write_parquet(failed, stage / "primary_failure_ledger.parquet")
    _write_parquet(
        pd.DataFrame.from_records(replacement_failure_ledger),
        stage / "replacement_failure_ledger.parquet",
    )
    checkpoint = {
        "status": "pass" if closure else "underfilled",
        "outer_end_to_end_closure_complete": closure,
        "outer_exact_target_complete": exact,
        "complete_row_count": complete_count,
        "minimum_complete_rows": minimum,
        "target_complete_rows": target,
        "track_b_may_continue": True,
        "track_b_cannot_modify_this_checkpoint": True,
    }
    _write_json(stage / "track_a_checkpoint.json", checkpoint)
    _write_json(
        stage / "artifact_manifest.json", _stage_artifact_manifest(stage, _git_sha())
    )
    gate = {
        **checkpoint,
        "tension_materialization_complete": closure,
        "primary_attempt_count": len(primary_attempts),
        "primary_failure_count": len(failed),
        "replacement_attempt_count": sum(len(frame) for frame in replacement_attempts),
        "replacement_success_count": sum(len(frame) for frame in replacement_successes),
        "unresolved_primary_failure_count": len(unresolved),
        "full_attempt_success_rate": full_success_rate,
        "tension_degraded": degraded,
        "physics_quality_claim_authorized": bool(not degraded and closure),
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "outer_tension_materialization", gate)


def _atlas_candidate_from_row(row: Mapping[str, Any] | pd.Series, *, node_id: int) -> AtlasCandidate:
    values = dict(row)
    beta = np.asarray([values[name] for name in BETA_COLUMNS], dtype=float)
    provenance_quality = str(values.get("label_quality", "Gold"))
    continuation_quality = {
        "Gold": "Gold",
        "Silver": "Silver",
        "ExactAnchor": "Gold",
        "Wide-Silver": "Silver",
    }.get(provenance_quality)
    if continuation_quality is None:
        raise ValueError(
            f"retry11 retained source has unsupported quality {provenance_quality!r}"
        )
    return AtlasCandidate(
        node_id=int(node_id),
        candidate_id=str(values.get("physical_point_id", f"node_{int(node_id)}")),
        beta_rad=beta,
        residual_mm=float(values.get("fk_residual_mm", values.get("teacher_fk_residual_mm", 0.0)) or 0.0),
        min_margin_deg=float(values.get("min_margin_deg", 1.0) or 1.0),
        normalized_min_margin=float(values.get("normalized_min_margin", 1.0) or 1.0),
        posture_cost=float(np.linalg.norm(beta)),
        condition_number=float(values.get("condition_number", 1.0) or 1.0),
        quality=continuation_quality,
        solver_success=True,
        actual_bounds=True,
        diagnostics={"retry11_provenance_label_quality": provenance_quality},
    )


def _solve_target_attempts(
    environment: Any,
    targets: pd.DataFrame,
    sources_by_target: Mapping[object, pd.DataFrame],
    *,
    physical_prefix: str,
) -> pd.DataFrame:
    optimized = make_optimized_predictor_corrector_continuation(
        environment,
        damping=2.0e-3,
        max_corrector_iterations=400,
        residual_tolerance_mm=10.0,
    )
    iterative = make_iterative_weighted_dls_continuation(
        environment,
        damping=2.0e-3,
        max_corrector_iterations=400,
        residual_tolerance_mm=10.0,
    )
    methods = {"optimized": optimized, "iterative": iterative}
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        target_id = getattr(target, "target_id", getattr(target, "task_node_id", None))
        if target_id is None:
            raise ValueError("retry11 solve target requires target_id or task_node_id")
        target_node = AtlasTaskNode(
            int(target_id),
            np.asarray([target.x_m, target.y_m, target.z_m], dtype=float),
            (),
        )
        sources = sources_by_target.get(target_id, pd.DataFrame())
        for source_position, source in sources.reset_index(drop=True).iterrows():
            source_node_id = int(source.get("task_node_id", -1_000_000 - source_position))
            source_candidate = _atlas_candidate_from_row(source, node_id=source_node_id)
            source_target = AtlasTaskNode(
                source_node_id,
                source.loc[list(XYZ_COLUMNS)].to_numpy(float),
                (),
            )
            for method_id, method in methods.items():
                outcome = method(source_candidate, target_node)
                reverse_gap = math.inf
                if outcome.success and outcome.actual_bounds:
                    endpoint = AtlasCandidate(
                        node_id=int(target_id),
                        candidate_id=f"{physical_prefix}_{target_id}_{source_position}_{method_id}",
                        beta_rad=outcome.beta_rad,
                        residual_mm=float(outcome.residual_mm),
                        min_margin_deg=float(outcome.minimum_margin_deg or 1.0),
                        normalized_min_margin=1.0,
                        posture_cost=float(np.linalg.norm(outcome.beta_rad)),
                        condition_number=1.0,
                        quality="Gold",
                        solver_success=True,
                        actual_bounds=True,
                    )
                    reverse = method(endpoint, source_target)
                    if reverse.success and reverse.actual_bounds:
                        reverse_gap = normalized_weighted_beta_deg(
                            reverse.beta_rad, source_candidate.beta_rad
                        )
                record = {
                    "target_id": target_id,
                    "task_node_id": int(target_id),
                    "physical_point_id": stable_row_id(physical_prefix, str(target_id)),
                    "source_task_node_id": source_node_id,
                    "source_physical_point_id": str(source.get("physical_point_id", source_node_id)),
                    "source_path_id": f"{source.get('physical_point_id', source_node_id)}:{method_id}",
                    "solver_method": method_id,
                    "solver_success": bool(outcome.success),
                    "actual_bounds": bool(outcome.actual_bounds),
                    "fk_residual_mm": float(outcome.residual_mm),
                    "reverse_gap_deg": reverse_gap,
                    "solver_status": str(outcome.status),
                    "x_m": float(target.x_m),
                    "y_m": float(target.y_m),
                    "z_m": float(target.z_m),
                    **{
                        name: float(outcome.beta_rad[index])
                        for index, name in enumerate(BETA_COLUMNS)
                    },
                }
                rows.append(record)
    return pd.DataFrame.from_records(rows)


def _exact_zero_anchor_label(config: Mapping[str, Any], output_root: Path) -> pd.DataFrame:
    zero = _read_json(output_root / STAGE_DIRS["inventory"] / "zero_anchor.json")
    component = config["components"]["zero"]
    return pd.DataFrame.from_records(
        [
            {
                "target_id": "exact_zero_anchor",
                "task_node_id": -1,
                "physical_point_id": "retry11_exact_zero_anchor",
                "label_quality": "ExactAnchor",
                "x_m": float(zero["zero_configuration_xyz_m"][0]),
                "y_m": float(zero["zero_configuration_xyz_m"][1]),
                "z_m": float(zero["zero_configuration_xyz_m"][2]),
                **{name: 0.0 for name in BETA_COLUMNS},
                "fk_residual_mm": 0.0,
                "canonical_component_id": str(component["canonical_component_id"]),
                "component_role": str(component["component_role"]),
                "zero_centered_primary": True,
                "canonical_lineage_id": "zero:exact_anchor",
            }
        ]
    )


def stage_zero_seed(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["zero_seed"]
    inventory = _gate(output_root, "inventory")
    anchor = _exact_zero_anchor_label(config, output_root)
    zero_record = _read_json(output_root / STAGE_DIRS["inventory"] / "zero_anchor.json")
    if not inventory.get("shared_gate_pass", False):
        _write_parquet(anchor, stage / "exact_zero_anchor.parquet")
        _write_json(stage / "exact_zero_anchor.json", zero_record)
        return _seal_gate(
            output_root,
            config,
            "zero_seed",
            {
                "status": "red",
                "zero_seed_status": "red",
                "exact_anchor_valid": False,
                "zero_seed_candidate_count": 0,
                "zero_seed_valid_count": 0,
                "zero_seed_selected_root_count": 0,
                "zero_seed_rejected_count": 0,
                "frontier_execution_authorized": False,
                "reason": "shared_inventory_failed",
                "formal_authorized": False,
            },
        )
    exact_valid = bool(
        inventory.get("checks", {}).get("zero_anchor_finite", False)
        and inventory.get("checks", {}).get("zero_anchor_bounds", False)
        and inventory.get("checks", {}).get("zero_anchor_roundtrip", False)
        and np.allclose(anchor.loc[:, BETA_COLUMNS].to_numpy(float), 0.0, atol=0.0)
    )
    components = pd.read_parquet(
        output_root / STAGE_DIRS["inventory"] / "component_registry.parquet"
    )
    candidates = zero_seed_candidate_registry(
        components, candidate_count=int(config["zero_seed"]["candidate_count"])
    )
    candidates = candidates.rename(columns={"task_node_id": "target_id"})
    sources_by_target = {target_id: anchor for target_id in candidates["target_id"]}
    attempts = _solve_target_attempts(
        _environment(config),
        candidates,
        sources_by_target,
        physical_prefix="retry11_zero_seed",
    )
    valid, rejected = classify_retry11_targets(attempts, anchor)
    if not valid.empty:
        valid = valid.merge(
            candidates[["target_id", "zero_seed_candidate_rank"]],
            on="target_id",
            how="left",
        ).sort_values(["zero_seed_candidate_rank", "target_id"], kind="stable")
        valid["canonical_component_id"] = str(
            config["components"]["zero"]["canonical_component_id"]
        )
        valid["component_role"] = str(config["components"]["zero"]["component_role"])
        valid["zero_centered_primary"] = True
        valid["canonical_lineage_id"] = valid["physical_point_id"].astype(str).map(
            lambda value: f"zero:{value}"
        )
    selected = valid.head(int(config["zero_seed"]["green_valid_count_min"])).copy()
    status = zero_seed_status(exact_anchor_valid=exact_valid, valid_count=len(valid))
    _write_parquet(anchor, stage / "exact_zero_anchor.parquet")
    _write_json(stage / "exact_zero_anchor.json", zero_record)
    _write_parquet(candidates, stage / "zero_seed_candidate_registry.parquet")
    _write_parquet(attempts, stage / "zero_seed_attempts.parquet")
    _write_parquet(valid, stage / "zero_seed_valid_labels.parquet")
    _write_parquet(selected, stage / "zero_seed_selected_roots.parquet")
    _write_parquet(rejected, stage / "zero_seed_rejected_labels.parquet")
    gate = {
        "status": status,
        "zero_seed_status": status,
        "exact_anchor_valid": exact_valid,
        "zero_seed_candidate_count": len(candidates),
        "zero_seed_valid_count": len(valid),
        "zero_seed_selected_root_count": len(selected),
        "zero_seed_rejected_count": len(rejected),
        "frontier_execution_authorized": status in {"green", "yellow"},
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "zero_seed", gate)


def _graph_adjacency(edges: pd.DataFrame, allowed: set[int]) -> dict[int, tuple[int, ...]]:
    adjacency = {node: set() for node in allowed}
    for row in edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return {node: tuple(sorted(neighbors)) for node, neighbors in adjacency.items()}


def _zero_retained_artifact(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed exact-anchor/task-node identities for Parquet closure."""

    result = frame.copy()
    if "target_id" in result:
        result["target_id"] = result["target_id"].astype(str)
    return result


def stage_zero_frontier_poc(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["zero_frontier_poc"]
    seed_gate = _gate(output_root, "zero_seed")
    if not seed_gate.get("frontier_execution_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "zero_frontier_poc",
            {
                "status": "not_authorized",
                "frontier_status": "not_authorized",
                "reason": "zero_seed_red",
                "bridge_execution_authorized": False,
                "formal_authorized": False,
            },
        )
    anchor = pd.read_parquet(output_root / STAGE_DIRS["zero_seed"] / "exact_zero_anchor.parquet")
    roots = pd.read_parquet(output_root / STAGE_DIRS["zero_seed"] / "zero_seed_selected_roots.parquet")
    roots["task_node_id"] = roots["target_id"].astype(int)
    retained = pd.concat([anchor, roots], ignore_index=True, sort=False)
    components = pd.read_parquet(
        output_root / STAGE_DIRS["inventory"] / "component_registry.parquet"
    )
    inventory = _gate(output_root, "inventory")
    zero = _read_json(output_root / STAGE_DIRS["inventory"] / "zero_anchor.json")
    components = annotate_spatial_strata(
        components,
        zero_xyz_m=zero["zero_configuration_xyz_m"],
        shell_width_mm=float(inventory["shell_width_mm"]),
    )
    zero_nodes = components[components["is_zero_component"].astype(bool)].copy()
    node_by_id = zero_nodes.set_index("task_node_id")
    edges = pd.read_parquet(_upstream_path(config, "refined_task_edges"))
    adjacency = _graph_adjacency(edges, set(zero_nodes["task_node_id"].astype(int)))
    active = {
        int(row.task_node_id): int(row.task_node_id) for row in roots.itertuples(index=False)
    }
    visited = set(roots["task_node_id"].astype(int))
    attempts_by_wave: list[pd.DataFrame] = []
    labels_by_wave: list[pd.DataFrame] = []
    wave_reports: list[dict[str, Any]] = []
    maximum_waves = int(config["zero_frontier"]["extended_waves"])
    initial_waves = int(config["zero_frontier"]["initial_waves"])
    environment = _environment(config)
    for wave in range(1, maximum_waves + 1):
        target_rows: list[pd.Series] = []
        source_map: dict[int, pd.DataFrame] = {}
        target_owner: dict[int, int] = {}
        sector_counts = (
            annotate_spatial_strata(
                retained,
                zero_xyz_m=zero["zero_configuration_xyz_m"],
                shell_width_mm=float(inventory["shell_width_mm"]),
            )["sector_id"]
            .value_counts()
            .to_dict()
        )
        for root_id, current_node in sorted(active.items()):
            current = node_by_id.loc[current_node]
            candidates = [node for node in adjacency.get(current_node, ()) if node not in visited]
            if not candidates:
                continue
            chosen_node = min(
                candidates,
                key=lambda node: (
                    0
                    if float(node_by_id.loc[node, "distance_to_zero_mm"])
                    >= float(current["distance_to_zero_mm"])
                    else 1,
                    int(sector_counts.get(int(node_by_id.loc[node, "sector_id"]), 0)),
                    -float(node_by_id.loc[node, "distance_to_zero_mm"]),
                    int(node),
                ),
            )
            target = node_by_id.loc[chosen_node].copy()
            target["target_id"] = int(chosen_node)
            target_rows.append(target)
            target_owner[int(chosen_node)] = root_id
            current_source = retained[retained["task_node_id"].astype(int).eq(current_node)]
            target_xyz = target.loc[list(XYZ_COLUMNS)].to_numpy(float)
            others = retained[~retained["task_node_id"].astype(int).eq(current_node)].copy()
            if len(others):
                others["_distance"] = np.linalg.norm(
                    others.loc[:, XYZ_COLUMNS].to_numpy(float) - target_xyz, axis=1
                )
                nearest_other = others.sort_values("_distance", kind="stable").head(1).drop(columns="_distance")
                sources = pd.concat([current_source, nearest_other], ignore_index=True, sort=False)
            else:
                sources = current_source.copy()
            source_map[int(chosen_node)] = sources
            visited.add(int(chosen_node))
        if not target_rows:
            wave_reports.append({"wave": wave, "attempted_targets": 0, "accepted_labels": 0})
            break
        target_frame = pd.DataFrame(target_rows).reset_index(drop=True)
        attempts = _solve_target_attempts(
            environment,
            target_frame,
            source_map,
            physical_prefix=f"retry11_zero_frontier_w{wave:02d}",
        )
        labels, _rejects = classify_retry11_targets(attempts, retained)
        if len(labels):
            labels["task_node_id"] = labels["target_id"].astype(int)
            labels["frontier_wave"] = wave
            labels["canonical_component_id"] = "zero_component"
            labels["component_role"] = "zero_primary_component"
            labels["zero_centered_primary"] = True
            labels["canonical_lineage_id"] = labels["physical_point_id"].astype(str).map(
                lambda value: f"zero:{value}"
            )
            retained = pd.concat([retained, labels], ignore_index=True, sort=False)
            for node in labels["task_node_id"].astype(int):
                active[target_owner[node]] = node
        accepted_nodes = set(labels.get("task_node_id", pd.Series(dtype=int)).astype(int))
        active = {
            root_id: node
            for root_id, node in active.items()
            if node in accepted_nodes or node not in target_owner
        }
        attempts["frontier_wave"] = wave
        attempts_by_wave.append(attempts)
        labels_by_wave.append(labels)
        wave_reports.append(
            {
                "wave": wave,
                "attempted_targets": len(target_frame),
                "accepted_labels": len(labels),
                "active_root_count_after_wave": len(active),
            }
        )
        if not active:
            break
        if wave == initial_waves and len(labels) == 0:
            break
    attempts = pd.concat(attempts_by_wave, ignore_index=True, sort=False) if attempts_by_wave else pd.DataFrame()
    labels = pd.concat(labels_by_wave, ignore_index=True, sort=False) if labels_by_wave else pd.DataFrame()
    _write_parquet(attempts, stage / "zero_frontier_attempts.parquet")
    _write_parquet(labels, stage / "zero_frontier_labels.parquet")
    _write_parquet(
        _zero_retained_artifact(retained),
        stage / "zero_component_labels_after_frontier.parquet",
    )
    _write_parquet(pd.DataFrame.from_records(wave_reports), stage / "frontier_wave_report.parquet")
    status = "verified" if len(labels) else "not_verified"
    gate = {
        "status": status,
        "frontier_status": status,
        "root_count": len(roots),
        "waves_executed": len(wave_reports),
        "new_label_count": len(labels),
        "retained_zero_component_label_count": len(retained),
        "bridge_execution_authorized": True,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "zero_frontier_poc", gate)


def _directional_corridor_solve(
    environment: Any,
    path: pd.DataFrame,
    source_row: pd.Series,
    *,
    reverse: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = path.sort_values("probe_index", ascending=not reverse, kind="stable")
    optimized = make_optimized_predictor_corrector_continuation(
        environment, damping=2.0e-3, max_corrector_iterations=400, residual_tolerance_mm=20.0
    )
    iterative = make_iterative_weighted_dls_continuation(
        environment, damping=2.0e-3, max_corrector_iterations=400, residual_tolerance_mm=20.0
    )
    methods = {"optimized": optimized, "iterative": iterative}
    current = source_row.copy()
    selected_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    blocked = False
    for target in ordered.itertuples(index=False):
        if blocked:
            selected_rows.append(
                {
                    "bridge_path_id": str(target.bridge_path_id),
                    "probe_index": int(target.probe_index),
                    "direction": "reverse" if reverse else "forward",
                    "success": False,
                }
            )
            continue
        source_candidate = _atlas_candidate_from_row(current, node_id=int(current.get("task_node_id", -1)))
        target_node = AtlasTaskNode(
            int(target.probe_index),
            np.asarray([target.x_m, target.y_m, target.z_m], dtype=float),
            (),
        )
        outcomes: list[tuple[str, Any]] = []
        for method_id, method in methods.items():
            outcome = method(source_candidate, target_node)
            attempt_rows.append(
                {
                    "bridge_path_id": str(target.bridge_path_id),
                    "probe_index": int(target.probe_index),
                    "direction": "reverse" if reverse else "forward",
                    "solver_method": method_id,
                    "success": bool(outcome.success),
                    "actual_bounds": bool(outcome.actual_bounds),
                    "fk_residual_mm": float(outcome.residual_mm),
                    "solver_status": str(outcome.status),
                    **{
                        name: float(outcome.beta_rad[index])
                        for index, name in enumerate(BETA_COLUMNS)
                    },
                }
            )
            if outcome.success and outcome.actual_bounds:
                outcomes.append((method_id, outcome))
        if not outcomes:
            blocked = True
            selected_rows.append(
                {
                    "bridge_path_id": str(target.bridge_path_id),
                    "probe_index": int(target.probe_index),
                    "direction": "reverse" if reverse else "forward",
                    "success": False,
                }
            )
            continue
        method_id, chosen = min(outcomes, key=lambda item: (item[1].residual_mm, item[0]))
        record = {
            "bridge_path_id": str(target.bridge_path_id),
            "probe_index": int(target.probe_index),
            "direction": "reverse" if reverse else "forward",
            "success": True,
            "solver_method": method_id,
            "fk_residual_mm": float(chosen.residual_mm),
            "x_m": float(target.x_m),
            "y_m": float(target.y_m),
            "z_m": float(target.z_m),
            **{name: float(chosen.beta_rad[index]) for index, name in enumerate(BETA_COLUMNS)},
        }
        selected_rows.append(record)
        current = pd.Series(
            {
                **record,
                "physical_point_id": f"{target.bridge_path_id}:{target.probe_index}:{record['direction']}",
                "task_node_id": int(target.probe_index),
                "label_quality": "Gold",
            }
        )
    return pd.DataFrame.from_records(selected_rows), pd.DataFrame.from_records(attempt_rows)


def stage_zero_outer_bridge(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["zero_outer_bridge"]
    frontier_gate = _gate(output_root, "zero_frontier_poc")
    if not frontier_gate.get("bridge_execution_authorized", False):
        return _seal_gate(
            output_root,
            config,
            "zero_outer_bridge",
            {
                "status": "not_authorized",
                "bridge_status": "not_authorized",
                "zero_outer_branch_compatible": "unknown",
                "reason": "zero_seed_or_frontier_not_authorized",
                "formal_authorized": False,
            },
        )
    zero_labels = pd.read_parquet(
        output_root / STAGE_DIRS["zero_frontier_poc"] / "zero_component_labels_after_frontier.parquet"
    )
    zero_labels = zero_labels[zero_labels["task_node_id"].astype(int).ge(0)].copy()
    outer = pd.read_parquet(_upstream_path(config, "frozen_atlas"))
    zero_tree = zero_labels.loc[:, XYZ_COLUMNS].to_numpy(float)
    outer_tree = outer.loc[:, XYZ_COLUMNS].to_numpy(float)
    from scipy.spatial import cKDTree

    distances, outer_indices = cKDTree(outer_tree).query(zero_tree, k=1)
    zero_index = int(np.argmin(distances))
    zero_source = zero_labels.iloc[zero_index]
    outer_source = outer.iloc[int(outer_indices[zero_index])].copy()
    if "physical_point_id" not in outer_source:
        outer_source["physical_point_id"] = f"outer_task_{int(outer_source['task_node_id'])}"
    paths = corridor_path_registry(
        zero_source.loc[list(XYZ_COLUMNS)].to_numpy(float),
        outer_source.loc[list(XYZ_COLUMNS)].to_numpy(float),
        maximum_step_mm=float(config["bridge"]["maximum_step_mm"]),
        maximum_probes=int(config["bridge"]["maximum_probes_per_path"]),
        diversity_ray_count=int(config["bridge"]["diversity_ray_count"]),
    )
    environment = _environment(config)
    selected_parts: list[pd.DataFrame] = []
    attempt_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for path_id, path in paths.groupby("bridge_path_id", sort=True):
        forward, forward_attempts = _directional_corridor_solve(
            environment, path, zero_source, reverse=False
        )
        reverse, reverse_attempts = _directional_corridor_solve(
            environment, path, outer_source, reverse=True
        )
        selected_parts.extend([forward, reverse])
        attempt_parts.extend([forward_attempts, reverse_attempts])
        forward_by_index = forward.set_index("probe_index")
        reverse_by_index = reverse.set_index("probe_index")
        prior_forward_beta: np.ndarray | None = None
        prior_reverse_beta: np.ndarray | None = None
        for probe_index in sorted(path["probe_index"].astype(int)):
            frow = forward_by_index.loc[probe_index]
            rrow = reverse_by_index.loc[probe_index]
            forward_success = bool(frow.get("success", False))
            reverse_success = bool(rrow.get("success", False))
            same_raw = math.inf
            local_gaps: list[float] = []
            neighbor_raw: list[float] = []
            fk_values: list[float] = []
            if forward_success:
                fbeta = frow.loc[list(BETA_COLUMNS)].to_numpy(float)
                fk_values.append(float(frow["fk_residual_mm"]))
                if prior_forward_beta is not None:
                    local_gaps.append(normalized_weighted_beta_deg(fbeta, prior_forward_beta))
                    neighbor_raw.append(raw_beta_max_deg(fbeta, prior_forward_beta))
                prior_forward_beta = fbeta
            if reverse_success:
                rbeta = rrow.loc[list(BETA_COLUMNS)].to_numpy(float)
                fk_values.append(float(rrow["fk_residual_mm"]))
                if prior_reverse_beta is not None:
                    local_gaps.append(normalized_weighted_beta_deg(rbeta, prior_reverse_beta))
                    neighbor_raw.append(raw_beta_max_deg(rbeta, prior_reverse_beta))
                prior_reverse_beta = rbeta
            if forward_success and reverse_success:
                same_raw = raw_beta_max_deg(fbeta, rbeta)
            branch_conflict = same_raw > float(config["bridge"]["same_point_raw_max_deg"])
            audit_rows.append(
                {
                    "bridge_path_id": str(path_id),
                    "probe_index": probe_index,
                    "solver_success": forward_success and reverse_success,
                    "forward_success": forward_success,
                    "reverse_success": reverse_success,
                    "same_point_raw_gap_deg": same_raw,
                    "local_weighted_gap_deg": max(local_gaps, default=0.0),
                    "neighbor_raw_gap_deg": max(neighbor_raw, default=0.0),
                    "persistent_edge_failure": not (forward_success and reverse_success),
                    "fk_residual_mm": max(fk_values, default=math.inf),
                    "branch_conflict": branch_conflict,
                    "forward_reverse_conflict": branch_conflict,
                }
            )
    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    attempts = pd.concat(attempt_parts, ignore_index=True, sort=False)
    audit = pd.DataFrame.from_records(audit_rows)
    certificate = evaluate_bridge_certificate(audit)
    _write_parquet(paths, stage / "bridge_path_registry.parquet")
    _write_parquet(attempts, stage / "bridge_solver_attempts.parquet")
    _write_parquet(selected, stage / "bridge_directional_labels.parquet")
    _write_parquet(audit, stage / "bridge_probe_audit.parquet")
    _write_json(stage / "bridge_certificate.json", certificate)
    gate = {
        **certificate,
        "status": certificate["bridge_status"],
        "bridge_batch_count": 1,
        "path_count": paths["bridge_path_id"].nunique(),
        "straight_path_is_single_candidate_path": True,
        "automatic_component_merge_authorized": False,
        "unified_component_student_authorized": False,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "zero_outer_bridge", gate)


def _report_html(gate: Mapping[str, Any]) -> str:
    track_a = gate["track_a"]
    track_b = gate["track_b"]
    student = track_a["student"]
    tension = track_a["tension"]
    bridge = track_b["bridge"]
    closure_label = "已完成" if track_a["outer_end_to_end_closure_complete"] else "未完成"
    exact_label = "20k exact" if track_a["outer_exact_target_complete"] else "18k minimum/未达"
    return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>retry11 双轨探索实验报告</title>
<style>
@page {{ size: A4; margin: 16mm; }}
body {{ font-family: -apple-system,BlinkMacSystemFont,\"Noto Sans CJK SC\",sans-serif; color:#172033; margin:0; line-height:1.55; }}
.page {{ max-width:1120px; margin:0 auto; padding:36px; }}
.hero {{ background:linear-gradient(135deg,#16263f,#274f67); color:white; border-radius:18px; padding:30px; }}
.kicker {{ letter-spacing:.12em; font-size:12px; opacity:.8; }} h1 {{ margin:.2em 0; font-size:30px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:18px; }}
.card {{ border:1px solid #dce4eb; border-radius:14px; padding:18px; background:#fff; break-inside:avoid; }}
.metric {{ font-size:26px; font-weight:700; color:#193b55; }} .muted {{ color:#617181; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; border-bottom:1px solid #e6ebef; padding:8px; }}
.warning {{ border-left:5px solid #d48a19; background:#fff8e9; padding:12px 16px; margin:18px 0; }}
code {{ background:#eef3f6; padding:2px 5px; border-radius:4px; }}
@media(max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} .page {{ padding:18px; }} }}
</style></head><body><main class=\"page\">
<section class=\"hero\"><div class=\"kicker\">BACRA V14.3R · RETRY11 · EXPLORATORY</div>
<h1>Outer 完整闭环与 Zero-centered Bridge POC</h1>
<p>双轨结果保持 component 隔离；Track B 不作为 Track A 的前置，也不能追溯修改 Stage 5 checkpoint。</p></section>
<section class=\"grid\">
<article class=\"card\"><div class=\"muted\">Track A end-to-end closure</div><div class=\"metric\">{closure_label}</div>
<p>{int(track_a['complete_row_count']):,} complete rows · {exact_label}</p></article>
<article class=\"card\"><div class=\"muted\">Outer Student</div><div class=\"metric\">{student['status']}</div>
<p>DLS2 success {100.0*float(student['dls2_success_rate']):.2f}%；该状态不决定 theta/Tension 授权。</p></article>
<article class=\"card\"><div class=\"muted\">Tension pilot / full</div><div class=\"metric\">{tension['pilot_status']}</div>
<p>Pilot {100.0*float(tension['pilot_success_rate']):.2f}% · full complete={str(tension['materialization_complete']).lower()} · degraded={str(tension['degraded']).lower()}</p></article>
<article class=\"card\"><div class=\"muted\">Zero seed / frontier</div><div class=\"metric\">{track_b['zero_seed']['status']}</div>
<p>{int(track_b['zero_seed']['valid_count'])} valid seeds；frontier 新增 {int(track_b['frontier']['new_label_count'])} labels。</p></article>
</section>
<section class=\"card\" style=\"margin-top:16px\"><h2>Bridge POC</h2><table>
<tr><th>Bridge status</th><td>{bridge['status']}</td></tr>
<tr><th>Failure classification</th><td>{bridge['failure_classification']}</td></tr>
<tr><th>Branch compatibility</th><td>{bridge['zero_outer_branch_compatible']}</td></tr>
<tr><th>Path interpretation</th><td>straight corridor is one candidate task-space path</td></tr>
</table></section>
<div class=\"warning\"><strong>Claim boundary：</strong>本轮不授权 Formal、deployment、full-workspace、automatic component merge 或 unified-component Student。corridor 未验证不等于物理不可连接。</div>
<section class=\"card\"><h2>Fixed point</h2><p><code>{gate['scientific_source_fixed_point']}</code></p>
<p class=\"muted\">Binding: <code>{gate['binding_fixed_point']}</code><br>Config SHA-256: <code>{gate['config_sha256']}</code></p></section>
</main></body></html>"""


def _render_pdf_nonblocking(html_path: Path, pdf_path: Path) -> dict[str, Any]:
    try:
        from weasyprint import HTML

        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return {"generated": True, "renderer": "weasyprint", "path": pdf_path.name, "sha256": sha256_file(pdf_path)}
    except Exception as first_error:
        browser = next(
            (shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome") if shutil.which(name)),
            None,
        )
        if browser is None:
            return {
                "generated": False,
                "exception_type": type(first_error).__name__,
                "exception_message": str(first_error),
                "presentation_failure_does_not_change_science": True,
            }
        try:
            with tempfile.TemporaryDirectory(prefix="retry11_pdf_", dir="/tmp") as profile:
                subprocess.run(
                    [browser, "--headless", "--no-sandbox", "--disable-gpu", f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=180,
                )
            return {"generated": True, "renderer": Path(browser).name, "path": pdf_path.name, "sha256": sha256_file(pdf_path)}
        except Exception as second_error:
            return {
                "generated": False,
                "exception_type": type(second_error).__name__,
                "exception_message": str(second_error),
                "fallback_from": type(first_error).__name__,
                "presentation_failure_does_not_change_science": True,
            }


def stage_summary(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["summary"]
    inventory = _gate(output_root, "inventory")
    dataset = _gate(output_root, "outer_sparse_wide")
    student = _gate(output_root, "outer_student")
    theta = _gate(output_root, "outer_theta")
    pilot = _gate(output_root, "outer_tension_pilot")
    materialization = _gate(output_root, "outer_tension_materialization")
    zero_seed = _gate(output_root, "zero_seed")
    frontier = _gate(output_root, "zero_frontier_poc")
    bridge = _gate(output_root, "zero_outer_bridge")
    checkpoint_path = (
        output_root
        / STAGE_DIRS["outer_tension_materialization"]
        / "track_a_checkpoint.json"
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    identity = _run_identity(output_root)
    gate: dict[str, Any] = {
        "governance_structure": "registered_retry11_protocol_config_runner",
        "scientific_contract": "dual_track_outer_closure_then_zero_frontier_bridge_poc",
        "operational_completion": True,
        "artifact_completeness": all(
            _stage_is_complete(output_root, config, name)
            for name in STAGE_ORDER[:-1]
        ),
        "scientific_source_fixed_point": _git_sha(),
        "binding_fixed_point": identity["binding_fixed_point"],
        "config_sha256": _config_sha(config),
        "shared": {
            "status": inventory.get("status"),
            "shared_gate_pass": bool(inventory.get("shared_gate_pass", False)),
        },
        "track_a": {
            "dataset": {
                "status": dataset.get("status"),
                "actual_rows": int(dataset.get("actual_rows", 0)),
                "new_parent_service_coverage_round1": float(dataset.get("new_parent_service_coverage_round1", 0.0)),
            },
            "student": {
                "status": student.get("student_status", student.get("status")),
                "dls2_success_rate": float(student.get("dls2_success_rate", 0.0)),
                "theta_execution_authorized": bool(student.get("theta_execution_authorized", False)),
            },
            "theta": {
                "status": theta.get("status"),
                "theta_generation_complete": bool(theta.get("theta_generation_complete", False)),
            },
            "tension": {
                "pilot_status": pilot.get("tension_status", pilot.get("status")),
                "pilot_success_rate": float(pilot.get("tension_success_rate", 0.0)),
                "materialization_complete": bool(materialization.get("tension_materialization_complete", False)),
                "degraded": bool(materialization.get("tension_degraded", pilot.get("tension_degraded", False))),
            },
            "complete_row_count": int(materialization.get("complete_row_count", 0)),
            "outer_end_to_end_closure_complete": bool(materialization.get("outer_end_to_end_closure_complete", False)),
            "outer_exact_target_complete": bool(materialization.get("outer_exact_target_complete", False)),
            "checkpoint_sha256_before_summary": checkpoint_sha,
            "checkpoint_immutable_after_stage5": True,
        },
        "track_b": {
            "zero_seed": {
                "status": zero_seed.get("zero_seed_status", zero_seed.get("status")),
                "exact_anchor_valid": bool(zero_seed.get("exact_anchor_valid", False)),
                "valid_count": int(zero_seed.get("zero_seed_valid_count", 0)),
            },
            "frontier": {
                "status": frontier.get("frontier_status", frontier.get("status")),
                "new_label_count": int(frontier.get("new_label_count", 0)),
            },
            "bridge": {
                "status": bridge.get("bridge_status", bridge.get("status")),
                "failure_classification": bridge.get("failure_classification", "not_classified"),
                "zero_outer_branch_compatible": bridge.get("zero_outer_branch_compatible", "unknown"),
                "straight_path_is_single_candidate_path": True,
            },
        },
        "formal_authorization": False,
        "deployment_authorization": False,
        "full_workspace_authorization": False,
        "unified_component_student_authorization": False,
        "automatic_component_merge_authorization": False,
    }
    _write_json(stage / "gate.json", gate)
    html_path = stage / "retry11_advisor_report.html"
    html_path.write_text(_report_html(gate), encoding="utf-8")
    render = _render_pdf_nonblocking(html_path, stage / "retry11_advisor_report.pdf")
    _write_json(stage / "report_render.json", render)
    artifacts = [
        {
            "path": str(path.relative_to(output_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and stage not in path.parents
        and path.name != "progress.json"
    ]
    _write_json(
        stage / "artifact_manifest.json",
        {
            "schema_version": 1,
            "scientific_source_fixed_point": _git_sha(),
            "binding_fixed_point": identity["binding_fixed_point"],
            "config_sha256": _config_sha(config),
            "artifacts": artifacts,
        },
    )
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("summary modified the sealed Track A checkpoint")
    completion = _seal_stage(output_root, config, "summary")
    completion.update(
        {
            "operational_completion": True,
            "gate_sha256": sha256_file(stage / "gate.json"),
            "artifact_manifest_sha256": sha256_file(stage / "artifact_manifest.json"),
            "track_a_checkpoint_sha256": checkpoint_sha,
        }
    )
    _write_json(stage / "completion_manifest.json", completion)
    return gate


STAGE_RUNNERS: dict[str, Callable[[Mapping[str, Any], Path, Path], dict[str, Any]]] = {
    "inventory": stage_inventory,
    "outer_sparse_wide": stage_outer_sparse_wide,
    "outer_student": stage_outer_student,
    "outer_theta": stage_outer_theta,
    "outer_tension_pilot": stage_outer_tension_pilot,
    "outer_tension_materialization": stage_outer_tension_materialization,
    "zero_seed": stage_zero_seed,
    "zero_frontier_poc": stage_zero_frontier_poc,
    "zero_outer_bridge": stage_zero_outer_bridge,
    "summary": stage_summary,
}


def _preflight_checkout() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=SOURCE_ROOT).returncode != 0:
        raise RuntimeError("retry11 run requires a clean scientific checkout")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=SOURCE_ROOT).returncode != 0:
        raise RuntimeError("retry11 run requires a clean scientific index")
    if subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=SOURCE_ROOT,
        stdout=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError("retry11 run must execute from a detached scientific fixed point")


def _ensure_identity(
    config: Mapping[str, Any], output_root: Path, binding_sha: str
) -> None:
    definition = _binding_definition(binding_sha, str(config["experiment_id"]))
    scientific_sha = _git_sha()
    if str(definition.get("scientific_source_fixed_point")) != scientific_sha:
        raise RuntimeError("retry11 binding does not point to the executing scientific fixed point")
    identity = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "scientific_source_fixed_point": scientific_sha,
        "binding_fixed_point": binding_sha,
        "config_sha256": _config_sha(config),
    }
    path = output_root / "run_identity.json"
    if path.exists():
        if _read_json(path) != identity:
            raise RuntimeError("output root belongs to a different retry11 identity")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _write_json(path, identity)


def run(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    binding_sha: str,
    only_stage: str | None = None,
) -> dict[str, Any]:
    _preflight_checkout()
    _ensure_identity(config, output_root, binding_sha)
    project_root = project_root_from(SOURCE_ROOT)
    stages = STAGE_ORDER if only_stage is None else (only_stage,)
    for position, stage_name in enumerate(stages, start=1):
        if _stage_is_complete(output_root, config, stage_name):
            continue
        for predecessor in STAGE_ORDER[: STAGE_ORDER.index(stage_name)]:
            if not _stage_is_complete(output_root, config, predecessor):
                raise RuntimeError(f"retry11 predecessor is not sealed: {predecessor}")
        _progress(
            output_root,
            status="running",
            stage_name=stage_name,
            completed=position - 1,
            total=len(stages),
            message=f"running {stage_name}",
        )
        STAGE_RUNNERS[stage_name](config, project_root, output_root)
    terminal = output_root / STAGE_DIRS["summary"] / "gate.json"
    if terminal.is_file():
        _progress(
            output_root,
            status="complete",
            stage_name="summary",
            completed=len(STAGE_ORDER),
            total=len(STAGE_ORDER),
            message="retry11 operational execution complete",
        )
        return _read_json(terminal)
    return _gate(output_root, stages[-1])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--binding-sha")
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--validate-stage", choices=STAGE_ORDER)
    parser.add_argument("--tension-worker-root")
    parser.add_argument("--shard-id", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.tension_worker_root is not None:
        if args.shard_id is None:
            raise ValueError("tension worker requires --shard-id")
        _tension_worker(config, Path(args.tension_worker_root).resolve(), args.shard_id)
        return 0
    if args.output_root is None or args.binding_sha is None:
        raise ValueError("retry11 execution requires --output-root and --binding-sha")
    output_root = Path(args.output_root).resolve()
    if args.validate_stage is not None:
        return 0 if _stage_is_complete(output_root, config, args.validate_stage) else 1
    gate = run(
        config,
        output_root,
        binding_sha=str(args.binding_sha),
        only_stage=args.stage,
    )
    print(json.dumps(gate, sort_keys=True, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
