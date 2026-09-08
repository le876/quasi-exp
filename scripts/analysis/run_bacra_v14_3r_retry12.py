#!/usr/bin/env python3
"""Run retry12 symmetry-quotient zero-centred kinematic closure."""

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
from typing import Any, Callable, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

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
from quasi_exp.teacher.retry10 import normalized_weighted_beta_deg, raw_beta_max_deg
from quasi_exp.teacher.retry11_bridge import classify_retry11_targets
from quasi_exp.teacher.retry11_sampling import graph_shell_width_mm
from quasi_exp.teacher.retry12_symmetry import (
    BETA_COLUMNS,
    THETA_COLUMNS,
    XYZ_COLUMNS,
    SYMMETRY_BETA_SIGNS,
    assign_quotient_macroblock_splits,
    corrected_trajectory_metrics,
    deduplicate_proposals,
    expand_symmetry_orbits,
    orbit_class,
    proposal_registry,
    reduced_seam_beta_samples,
    seam_continuity_metrics,
    select_exact_orbit_budget,
    sobol_beta_samples,
    spatially_balanced_frontier,
    stable_id,
    transform_beta,
    validate_symmetry_group,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    load_workspace_student_models,
    save_workspace_student_models,
    train_workspace_student,
)


EXPERIMENT_ID = "bacra_v14_3r_retry12_symmetry_quotient_zero_centered_kinematic_closure"
STAGE_DIRS = {
    "symmetry_contract": "00_symmetry_contract",
    "quotient_target_registry": "01_quotient_target_registry",
    "zero_rooted_outward_atlas": "02_zero_rooted_outward_atlas",
    "symmetry_fixed_seams": "03_symmetry_fixed_seams",
    "orbit_selection_expansion_split": "04_orbit_selection_expansion_split",
    "full_graph_connectivity_audit": "05_full_graph_connectivity_audit",
    "freeze_kinematic_dataset": "06_freeze_kinematic_dataset",
    "quotient_student": "07_quotient_student",
    "zero_to_target_trajectory_audit": "08_zero_to_target_trajectory_audit",
    "summary": "09_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    return resolved.parent.parent if resolved.parent.name == ".worktrees" else resolved


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()


def _config_sha(config: Mapping[str, Any]) -> str:
    return sha256_file(Path(str(config["config_path"])))


def _upstream_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["upstream"]["retry11_root"]))


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    spec = config["upstream"][key]
    return Path(str(spec["absolute_path"])) if "absolute_path" in spec else _upstream_root(config) / str(spec["path"])


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
        }
    )
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("retry12 experiment_id mismatch")
    if int(config["dataset"]["target_expanded_rows"]) != 19999:
        raise ValueError("retry12 exact expanded target is fixed at 19999")
    if config["dataset"]["theta_storage"] != "beta_to_theta_without_theta_sign_multiplication":
        raise ValueError("retry12 theta storage convention mismatch")
    if any(bool(value) for value in config["claims"].values()):
        raise ValueError("retry12 config cannot grant formal/deployment claims")
    consultation = SOURCE_ROOT / str(config["sources"]["consultation_input"])
    if sha256_file(consultation) != str(config["consultation_sha256"]):
        raise ValueError("retry12 consultation hash mismatch")
    robot = SOURCE_ROOT / str(config["sources"]["robot_config"])
    if sha256_file(robot) != str(config["sources"]["robot_config_sha256"]):
        raise ValueError("retry12 robot config hash mismatch")
    return config


def _binding_definition(binding_sha: str, experiment_id: str) -> Mapping[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{binding_sha}:spec/registry.yaml"], cwd=SOURCE_ROOT, text=True
    )
    return yaml.safe_load(raw)["experiments"][experiment_id]


def _ensure_identity(config: Mapping[str, Any], output_root: Path, binding_sha: str) -> dict[str, Any]:
    path = output_root / "run_identity.json"
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_source_fixed_point": _git_sha(),
        "binding_fixed_point": str(binding_sha),
        "config_sha256": _config_sha(config),
    }
    definition = _binding_definition(str(binding_sha), EXPERIMENT_ID)
    expected = {
        "scientific_source_fixed_point": _git_sha(),
        "config": str(Path(str(config["config_path"])).relative_to(SOURCE_ROOT)),
        "runner": str(Path(__file__).resolve().relative_to(SOURCE_ROOT)),
    }
    for key, value in expected.items():
        if str(definition.get(key)) != str(value):
            raise RuntimeError(f"retry12 binding mismatch for {key}: {definition.get(key)!r} != {value!r}")
    if path.exists():
        if _read_json(path) != identity:
            raise RuntimeError("retry12 output root identity mismatch")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(path, identity)
    return identity


def _stage_manifest(output_root: Path, stage_name: str) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = []
    for path in sorted(stage.glob("*")):
        if path.is_file() and path.name != "completion_manifest.json":
            artifacts.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
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
    _write_json(output_root / STAGE_DIRS[stage_name] / "completion_manifest.json", _stage_manifest(output_root, stage_name))


def _seal_gate(
    output_root: Path, config: Mapping[str, Any], stage_name: str, gate: Mapping[str, Any]
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
    if not path.exists():
        return False
    observed = _read_json(path)
    expected = _stage_manifest(output_root, stage_name)
    return observed == expected


def _progress(output_root: Path, stage_name: str, *, completed: int | None = None, total: int | None = None, message: str = "") -> None:
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


def _source_integrity(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    passed = True
    for key, spec in config["upstream"].items():
        if key in {"retry11_root", "retry11_scientific_source_fixed_point", "retry11_binding_fixed_point"}:
            continue
        path = _upstream_path(config, key)
        observed = sha256_file(path) if path.exists() else None
        ok = observed == str(spec["sha256"])
        rows.append({"source_id": key, "path": str(path), "expected_sha256": str(spec["sha256"]), "observed_sha256": observed, "pass": ok})
        passed = passed and ok
    return rows, passed


def _symmetry_samples(environment: Any, config: Mapping[str, Any]) -> np.ndarray:
    bounds = np.asarray(environment.bounds, dtype=float)
    rows = [np.zeros(6)]
    sweep_count = int(config["symmetry"]["validation_axis_sweep_points"])
    for axis in range(6):
        for value in np.linspace(bounds[axis, 0], bounds[axis, 1], sweep_count):
            beta = np.zeros(6)
            beta[axis] = value
            rows.append(beta)
    upstream = pd.read_parquet(_upstream_path(config, "outer_xyz_targets"))
    existing = upstream.loc[:, BETA_COLUMNS].to_numpy(float)
    take = min(len(existing), int(config["symmetry"]["validation_existing_maximin_rows"]))
    if take:
        rows.extend(existing[np.linspace(0, len(existing) - 1, take).astype(int)])
    rows.extend(
        sobol_beta_samples(
            bounds,
            power=int(config["symmetry"]["validation_sobol_power"]),
            seed=int(config["runtime"]["seed"]),
        )
    )
    near_count = int(config["symmetry"]["validation_near_boundary_rows"])
    unit = sobol_beta_samples(bounds, power=int(math.ceil(math.log2(near_count))), seed=int(config["runtime"]["seed"]) + 1)[:near_count]
    for index in range(len(unit)):
        axis = index % 6
        unit[index, axis] = bounds[axis, index % 2]
    rows.extend(unit)
    return np.asarray(rows, dtype=float)


def stage_symmetry_contract(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["symmetry_contract"]
    source_rows, source_pass = _source_integrity(config)
    environment = _environment(config)
    samples = _symmetry_samples(environment, config)
    evidence, registry = validate_symmetry_group(
        environment,
        samples,
        p99_max_mm=float(config["symmetry"]["equivariance_p99_max_mm"]),
        individual_max_mm=float(config["symmetry"]["equivariance_individual_max_mm"]),
    )
    zero_xyz = np.asarray(environment.fk(np.zeros(6)), dtype=float).reshape(-1, 3)[0]
    task_nodes = pd.read_parquet(_upstream_path(config, "task_nodes"))
    edges = pd.read_parquet(_upstream_path(config, "refined_task_edges"))
    q = config["quotient_registry"]
    shell_width_mm, edge_p95_mm = graph_shell_width_mm(
        task_nodes,
        edges,
        percentile=float(q["shell_edge_percentile"]),
        multiplier=float(q["shell_edge_multiplier"]),
        minimum_mm=float(q["shell_width_min_mm"]),
        maximum_mm=float(q["shell_width_max_mm"]),
    )
    seam_half_width_mm = float(np.clip(edge_p95_mm, q["seam_width_min_mm"], q["seam_width_max_mm"]))
    authorized = all(bool(registry[name]["authorized"]) for name in config["symmetry"]["group"])
    passed = bool(source_pass and authorized and registry["group_composition_verified"] and registry["exact_zero_fixed"])
    _write_parquet(pd.DataFrame.from_records(source_rows), stage / "source_inventory.parquet")
    _write_parquet(evidence, stage / "symmetry_equivariance_samples.parquet")
    _write_json(stage / "symmetry_group_registry.json", registry)
    _write_json(stage / "zero_anchor.json", {"beta0_rad": [0.0] * 6, "xyz0_m": zero_xyz.tolist(), "kinematics_theta_sign": float(environment.theta_sign)})
    return _seal_gate(
        output_root,
        config,
        "symmetry_contract",
        {
            "status": "pass" if passed else "fail",
            "symmetry_gate_pass": passed,
            "source_integrity_pass": source_pass,
            "authorized_group": list(config["symmetry"]["group"]) if passed else [],
            "sample_count": len(samples),
            "registered_edge_p95_mm": edge_p95_mm,
            "shell_width_mm": shell_width_mm,
            "seam_half_width_mm": seam_half_width_mm,
            "theta_storage_convention": "beta_to_theta_without_theta_sign_multiplication",
        },
    )


def stage_quotient_target_registry(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["quotient_target_registry"]
    symmetry_gate = _gate(output_root, "symmetry_contract")
    if not symmetry_gate.get("symmetry_gate_pass", False):
        return _seal_gate(output_root, config, "quotient_target_registry", {"status": "not_authorized", "target_registry_valid": False, "reason": "symmetry_contract_failed"})
    zero = _read_json(output_root / STAGE_DIRS["symmetry_contract"] / "zero_anchor.json")["xyz0_m"]
    q = config["quotient_registry"]
    kwargs = {
        "zero_xyz_m": zero,
        "shell_width_mm": float(symmetry_gate["shell_width_mm"]),
        "seam_half_width_mm": float(symmetry_gate["seam_half_width_mm"]),
        "angle_bins": int(q["angle_bin_count"]),
        "cell_size_mm": float(q["cell_size_mm"]),
        "x_bin_mm": float(q["x_bin_mm"]),
    }
    task_nodes = pd.read_parquet(_upstream_path(config, "task_nodes"))
    outer = pd.read_parquet(_upstream_path(config, "outer_xyz_targets"))
    environment = _environment(config)
    proposal_beta = sobol_beta_samples(
        np.asarray(environment.bounds), power=int(q["remote_sobol_power"]), seed=int(config["runtime"]["seed"])
    )
    proposal_xyz = np.asarray(environment.fk(proposal_beta), dtype=float).reshape(-1, 3)
    provenance = pd.DataFrame(proposal_beta, columns=[f"proposal_{name}" for name in BETA_COLUMNS])
    provenance.loc[:, XYZ_COLUMNS] = proposal_xyz
    provenance["proposal_id"] = [stable_id("retry12_sobol", index) for index in range(len(provenance))]
    for name in ("label", "seed", "warm_start", "tie_break", "branch_hint"):
        provenance[f"proposal_beta_{name}_eligible"] = False
    frames = [
        proposal_registry(task_nodes.loc[:, XYZ_COLUMNS].to_numpy(float), source="registered_task", **kwargs),
        proposal_registry(outer.loc[:, XYZ_COLUMNS].to_numpy(float), source="outer_xyz", **kwargs),
        proposal_registry(proposal_xyz, source="sobol_target_only", **kwargs),
    ]
    registry = deduplicate_proposals(frames)
    registry["solver_target_node_id"] = np.arange(1, len(registry) + 1, dtype=np.int64)
    outer_canonical = proposal_registry(outer.loc[:, XYZ_COLUMNS].to_numpy(float), source="outer_xyz", **kwargs)
    remote_start = float(outer_canonical["zero_radius_mm"].min())
    remote = registry[registry["zero_radius_mm"].ge(remote_start)]
    serviced_bins = int(remote["angle_bin_id"].nunique())
    if serviced_bins >= int(q["remote_green_angle_bins"]):
        coverage_status = "green"
    elif serviced_bins >= int(q["remote_yellow_angle_bins_min"]):
        coverage_status = "yellow"
    else:
        coverage_status = "red"
    _write_parquet(provenance, stage / "sobol_proposal_beta_provenance.parquet")
    _write_parquet(registry, stage / "quotient_target_registry.parquet")
    source_report = registry.groupby(["proposal_source", "angle_bin_id"], sort=True).size().rename("target_count").reset_index()
    _write_parquet(source_report, stage / "target_source_angle_report.parquet")
    return _seal_gate(
        output_root,
        config,
        "quotient_target_registry",
        {
            "status": "pass",
            "target_registry_valid": True,
            "target_count": len(registry),
            "sobol_proposal_count": len(provenance),
            "proposal_beta_scientifically_discarded": True,
            "remote_start_radius_mm": remote_start,
            "remote_serviced_angle_bins": serviced_bins,
            "remote_uniform_coverage_status": coverage_status,
            "remote_uniform_coverage_pass": coverage_status == "green",
            "dataset_continuation_authorized": True,
        },
    )


def _canonicalize_zero_lineage(frame: pd.DataFrame, config: Mapping[str, Any], output_root: Path) -> pd.DataFrame:
    zero = _read_json(output_root / STAGE_DIRS["symmetry_contract"] / "zero_anchor.json")["xyz0_m"]
    gate = _gate(output_root, "symmetry_contract")
    q = config["quotient_registry"]
    records: list[dict[str, Any]] = []
    for position, row in enumerate(frame.to_dict("records")):
        xyz = np.asarray([row[name] for name in XYZ_COLUMNS], dtype=float)
        beta = np.asarray([row[name] for name in BETA_COLUMNS], dtype=float)
        applied: list[str] = []
        if xyz[1] < 0.0:
            xyz[1] *= -1.0
            beta = transform_beta(beta, "mirror_y")
            applied.append("mirror_y")
        if xyz[2] < 0.0:
            xyz[2] *= -1.0
            beta = transform_beta(beta, "mirror_z")
            applied.append("mirror_z")
        record = {
            **row,
            **dict(zip(XYZ_COLUMNS, xyz, strict=True)),
            **dict(zip(BETA_COLUMNS, beta, strict=True)),
            "physical_point_id": stable_id("retry12_seed", row.get("physical_point_id", position)),
            "fundamental_representative_id": stable_id("retry12_rep", row.get("physical_point_id", position)),
            "lineage_node_id": position,
            "canonical_lineage_id": f"zero:{row.get('canonical_lineage_id', row.get('physical_point_id', position))}",
            "label_origin": "retry11_zero_lineage_seed",
            "source_symmetry_canonicalization": "+".join(applied) if applied else "identity",
        }
        records.append(record)
    result = pd.DataFrame.from_records(records)
    annotation = proposal_registry(
        result.loc[:, XYZ_COLUMNS].to_numpy(float),
        source="zero_lineage_seed",
        zero_xyz_m=zero,
        shell_width_mm=float(gate["shell_width_mm"]),
        seam_half_width_mm=float(gate["seam_half_width_mm"]),
        angle_bins=int(q["angle_bin_count"]),
        cell_size_mm=float(q["cell_size_mm"]),
        x_bin_mm=float(q["x_bin_mm"]),
    )
    for column in ("zero_radius_mm", "rho_m", "phi_rad", "radial_shell_id", "angle_bin_id", "x_bin_id", "quotient_parent_id", "region_role"):
        result[column] = annotation[column].to_numpy()
    result.loc[result["zero_radius_mm"].le(1.0e-6), "label_quality"] = "ExactAnchor"
    return result


def _source_candidate(row: Mapping[str, Any], *, node_id: int) -> AtlasCandidate:
    beta = np.asarray([row[name] for name in BETA_COLUMNS], dtype=float)
    return AtlasCandidate(
        node_id=int(node_id),
        candidate_id=str(row["physical_point_id"]),
        beta_rad=beta,
        residual_mm=float(row.get("fk_residual_mm", 0.0) or 0.0),
        min_margin_deg=float(row.get("min_margin_deg", 1.0) or 1.0),
        normalized_min_margin=float(row.get("normalized_min_margin", 1.0) or 1.0),
        posture_cost=float(np.linalg.norm(beta)),
        condition_number=float(row.get("condition_number", 1.0) or 1.0),
        quality="Gold" if str(row.get("label_quality")) in {"Gold", "ExactAnchor"} else "Silver",
        solver_success=True,
        actual_bounds=True,
    )


def _solve_outward_targets(
    environment: Any,
    targets: pd.DataFrame,
    retained: pd.DataFrame,
    *,
    maximum_step_mm: float,
) -> pd.DataFrame:
    optimized = make_optimized_predictor_corrector_continuation(
        environment, damping=2.0e-3, max_corrector_iterations=400, residual_tolerance_mm=10.0
    )
    iterative = make_iterative_weighted_dls_continuation(
        environment, damping=2.0e-3, max_corrector_iterations=400, residual_tolerance_mm=10.0
    )
    retained_xyz = retained.loc[:, XYZ_COLUMNS].to_numpy(float)
    tree = cKDTree(retained_xyz)
    rows: list[dict[str, Any]] = []
    for target in targets.to_dict("records"):
        target_xyz = np.asarray([target[name] for name in XYZ_COLUMNS], dtype=float)
        k = min(2, len(retained))
        distances, positions = tree.query(target_xyz, k=k)
        positions = np.atleast_1d(positions)
        distances = np.atleast_1d(distances)
        for distance_m, source_position in zip(distances, positions, strict=True):
            if float(distance_m) * 1000.0 > float(maximum_step_mm) + 1.0e-9:
                continue
            source = retained.iloc[int(source_position)].to_dict()
            source_node_id = int(source.get("lineage_node_id", source_position))
            candidate = _source_candidate(source, node_id=source_node_id)
            target_node = AtlasTaskNode(int(target["solver_target_node_id"]), target_xyz, ())
            outcome = optimized(candidate, target_node)
            method_name = "optimized"
            method = optimized
            if not outcome.success or not outcome.actual_bounds:
                fallback = iterative(candidate, target_node)
                if fallback.success or float(fallback.residual_mm) < float(outcome.residual_mm):
                    outcome = fallback
                    method_name = "iterative"
                    method = iterative
            reverse_gap = math.inf
            if outcome.success and outcome.actual_bounds:
                endpoint = AtlasCandidate(
                    node_id=int(target["solver_target_node_id"]),
                    candidate_id=stable_id("retry12_endpoint", target["proposal_id"], source["physical_point_id"]),
                    beta_rad=np.asarray(outcome.beta_rad, dtype=float),
                    residual_mm=float(outcome.residual_mm),
                    min_margin_deg=float(outcome.minimum_margin_deg or 1.0),
                    normalized_min_margin=1.0,
                    posture_cost=float(np.linalg.norm(outcome.beta_rad)),
                    condition_number=1.0,
                    quality="Gold",
                    solver_success=True,
                    actual_bounds=True,
                )
                source_node = AtlasTaskNode(source_node_id, np.asarray([source[name] for name in XYZ_COLUMNS], dtype=float), ())
                reverse = method(endpoint, source_node)
                if reverse.success and reverse.actual_bounds:
                    reverse_gap = normalized_weighted_beta_deg(reverse.beta_rad, candidate.beta_rad)
            rows.append(
                {
                    "target_id": str(target["proposal_id"]),
                    "solver_target_node_id": int(target["solver_target_node_id"]),
                    "physical_point_id": stable_id("retry12_label", target["proposal_id"]),
                    "source_task_node_id": source_node_id,
                    "source_physical_point_id": str(source["physical_point_id"]),
                    "source_path_id": str(source["physical_point_id"]),
                    "solver_method": method_name,
                    "solver_success": bool(outcome.success),
                    "actual_bounds": bool(outcome.actual_bounds),
                    "fk_residual_mm": float(outcome.residual_mm),
                    "reverse_gap_deg": reverse_gap,
                    "source_edge_length_mm": float(distance_m) * 1000.0,
                    "solver_status": str(outcome.status),
                    **{name: target[name] for name in XYZ_COLUMNS},
                    **{name: float(outcome.beta_rad[index]) for index, name in enumerate(BETA_COLUMNS)},
                }
            )
    return pd.DataFrame.from_records(rows)


def _annotate_new_labels(labels: pd.DataFrame, targets: pd.DataFrame, *, batch: int, lineage_offset: int) -> pd.DataFrame:
    if labels.empty:
        return labels
    target_columns = [
        "proposal_id", "proposal_source", "zero_radius_mm", "rho_m", "phi_rad",
        "radial_shell_id", "angle_bin_id", "x_bin_id", "quotient_parent_id", "region_role",
    ]
    result = labels.merge(
        targets.loc[:, target_columns], left_on="target_id", right_on="proposal_id", how="left", validate="one_to_one"
    )
    result["lineage_node_id"] = np.arange(lineage_offset, lineage_offset + len(result), dtype=np.int64)
    result["canonical_lineage_id"] = result["physical_point_id"].astype(str).map(lambda value: f"zero:{value}")
    result["fundamental_representative_id"] = result["physical_point_id"].astype(str).map(lambda value: stable_id("retry12_rep", value))
    result["label_origin"] = "retry12_zero_rooted_outward"
    result["frontier_batch"] = int(batch)
    return result


def _write_atlas_checkpoint(
    work: Path,
    *,
    attempted_count: int,
    retained: pd.DataFrame,
    attempts: pd.DataFrame,
    attempted_ids: set[str],
) -> None:
    stem = f"checkpoint_{attempted_count:06d}"
    _write_parquet(retained, work / f"{stem}_retained.parquet")
    _write_parquet(attempts, work / f"{stem}_attempts.parquet")
    _write_json(work / f"{stem}.json", {"attempted_count": attempted_count, "attempted_ids": sorted(attempted_ids), "retained": f"{stem}_retained.parquet", "attempts": f"{stem}_attempts.parquet"})


def _load_atlas_checkpoint(work: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[str], int] | None:
    checkpoints = sorted(work.glob("checkpoint_*.json"))
    if not checkpoints:
        return None
    meta = _read_json(checkpoints[-1])
    return (
        pd.read_parquet(work / meta["retained"]),
        pd.read_parquet(work / meta["attempts"]),
        set(map(str, meta["attempted_ids"])),
        int(meta["attempted_count"]),
    )


def stage_zero_rooted_outward_atlas(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["zero_rooted_outward_atlas"]
    registry_gate = _gate(output_root, "quotient_target_registry")
    if not registry_gate.get("dataset_continuation_authorized", False):
        return _seal_gate(output_root, config, "zero_rooted_outward_atlas", {"status": "not_authorized", "canonical_row_integrity_pass": False})
    source = pd.read_parquet(_upstream_path(config, "zero_lineage_labels"))
    initial = _canonicalize_zero_lineage(source, config, output_root)
    registry = pd.read_parquet(output_root / STAGE_DIRS["quotient_target_registry"] / "quotient_target_registry.parquet")
    work = stage / "_work"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_atlas_checkpoint(work)
    if checkpoint is None:
        retained = initial.copy()
        attempts = pd.DataFrame()
        attempted_ids: set[str] = set()
        attempted_count = 0
    else:
        retained, attempts, attempted_ids, attempted_count = checkpoint
    environment = _environment(config)
    atlas = config["atlas"]
    accepted_target = int(atlas["target_accepted_labels"])
    maximum_attempted = int(atlas["maximum_attempted_targets"])
    batch_size = int(atlas["roots_per_batch"])
    batch = attempted_count // max(1, batch_size)
    no_frontier_rounds = 0
    while len(retained) - len(initial) < accepted_target and attempted_count < maximum_attempted:
        remaining = registry[~registry["proposal_id"].astype(str).isin(attempted_ids)].copy()
        frontier = spatially_balanced_frontier(
            remaining,
            retained,
            maximum_step_mm=float(atlas["maximum_step_mm"]),
            count=batch_size,
        )
        if frontier.empty:
            no_frontier_rounds += 1
            break
        batch += 1
        batch_attempts = _solve_outward_targets(
            environment, frontier, retained, maximum_step_mm=float(atlas["maximum_step_mm"])
        )
        attempted = set(frontier["proposal_id"].astype(str))
        attempted_ids.update(attempted)
        attempted_count += len(attempted)
        labels, _rejects = classify_retry11_targets(batch_attempts, retained)
        labels = _annotate_new_labels(labels, frontier, batch=batch, lineage_offset=len(retained))
        if not labels.empty:
            retained = pd.concat([retained, labels], ignore_index=True, sort=False)
        attempts = pd.concat([attempts, batch_attempts], ignore_index=True, sort=False)
        if attempted_count % 100 < batch_size:
            _write_atlas_checkpoint(
                work,
                attempted_count=attempted_count,
                retained=retained,
                attempts=attempts,
                attempted_ids=attempted_ids,
            )
        _progress(output_root, "zero_rooted_outward_atlas", completed=len(retained) - len(initial), total=accepted_target, message=f"attempted_targets={attempted_count}")
    new_labels = retained[retained["label_origin"].eq("retry12_zero_rooted_outward")].copy()
    bounds = np.asarray(environment.bounds, dtype=float)
    beta = retained.loc[:, BETA_COLUMNS].to_numpy(float)
    finite = bool(np.isfinite(retained.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy(float)).all())
    bounds_pass = bool(np.all((beta >= bounds[:, 0] - 1.0e-12) & (beta <= bounds[:, 1] + 1.0e-12)))
    fk = np.linalg.norm(np.asarray(environment.fk(beta)).reshape(-1, 3) - retained.loc[:, XYZ_COLUMNS].to_numpy(float), axis=1) * 1000.0
    retained["fk_residual_mm"] = fk
    remote_start = float(registry_gate["remote_start_radius_mm"])
    remote_bins = int(retained[retained["zero_radius_mm"].ge(remote_start)]["angle_bin_id"].nunique())
    q = config["quotient_registry"]
    coverage_status = "green" if remote_bins >= int(q["remote_green_angle_bins"]) else "yellow" if remote_bins >= int(q["remote_yellow_angle_bins_min"]) else "red"
    row_integrity = bool(finite and bounds_pass and percentile(fk, 99) <= float(atlas["fk_p99_max_mm"]))
    _write_parquet(initial, stage / "zero_lineage_seed_labels.parquet")
    _write_parquet(attempts, stage / "outward_solver_attempts.parquet")
    _write_parquet(new_labels, stage / "outward_new_labels.parquet")
    _write_parquet(retained, stage / "zero_rooted_fundamental_atlas.parquet")
    return _seal_gate(
        output_root,
        config,
        "zero_rooted_outward_atlas",
        {
            "status": "verified" if row_integrity else "row_integrity_red",
            "canonical_row_integrity_pass": row_integrity,
            "initial_seed_count": len(initial),
            "attempted_target_count": attempted_count,
            "new_label_count": len(new_labels),
            "retained_label_count": len(retained),
            "fk_p95_mm": percentile(fk, 95),
            "fk_p99_mm": percentile(fk, 99),
            "remote_serviced_angle_bins": remote_bins,
            "remote_uniform_coverage_status": coverage_status,
            "remote_uniform_coverage_pass": coverage_status == "green",
            "no_frontier_rounds": no_frontier_rounds,
            "seam_execution_authorized": row_integrity,
        },
    )


def _constrained_beta_solve(
    environment: Any,
    source_beta: np.ndarray,
    target_xyz: np.ndarray,
    *,
    free_indices: np.ndarray,
) -> tuple[np.ndarray, bool, float, str]:
    bounds = np.asarray(environment.bounds, dtype=float)
    initial = np.asarray(source_beta, dtype=float).copy()
    fixed = np.ones(6, dtype=bool)
    fixed[free_indices] = False
    initial[fixed] = 0.0

    def residual(free_beta: np.ndarray) -> np.ndarray:
        beta = initial.copy()
        beta[free_indices] = free_beta
        return np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)[0] - target_xyz

    result = least_squares(
        residual,
        initial[free_indices],
        bounds=(bounds[free_indices, 0], bounds[free_indices, 1]),
        max_nfev=400,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    beta = initial.copy()
    beta[free_indices] = result.x
    residual_mm = float(np.linalg.norm(residual(result.x)) * 1000.0)
    success = bool(result.success and residual_mm <= 10.0 and np.isfinite(beta).all())
    return beta, success, residual_mm, str(result.status)


def _solve_seam_targets(
    environment: Any,
    targets: pd.DataFrame,
    retained: pd.DataFrame,
    *,
    seam_class: str,
    maximum_step_mm: float,
) -> pd.DataFrame:
    free = np.asarray([1, 3, 5] if seam_class == "y_seam" else [0, 2, 4], dtype=int)
    tree = cKDTree(retained.loc[:, XYZ_COLUMNS].to_numpy(float))
    reverse_solver = make_optimized_predictor_corrector_continuation(
        environment, damping=2.0e-3, max_corrector_iterations=400, residual_tolerance_mm=10.0
    )
    rows: list[dict[str, Any]] = []
    for target in targets.to_dict("records"):
        target_xyz = np.asarray([target[name] for name in XYZ_COLUMNS], dtype=float)
        k = min(2, len(retained))
        distances, positions = tree.query(target_xyz, k=k)
        for distance_m, source_position in zip(np.atleast_1d(distances), np.atleast_1d(positions), strict=True):
            if float(distance_m) * 1000.0 > float(maximum_step_mm) + 1.0e-9:
                continue
            source = retained.iloc[int(source_position)].to_dict()
            source_beta = np.asarray([source[name] for name in BETA_COLUMNS], dtype=float)
            beta, success, residual_mm, status = _constrained_beta_solve(
                environment, source_beta, target_xyz, free_indices=free
            )
            reverse_gap = math.inf
            if success:
                endpoint = AtlasCandidate(
                    node_id=int(target["solver_target_node_id"]), candidate_id="seam_endpoint",
                    beta_rad=beta, residual_mm=residual_mm, min_margin_deg=1.0,
                    normalized_min_margin=1.0, posture_cost=float(np.linalg.norm(beta)),
                    condition_number=1.0, quality="Gold", solver_success=True, actual_bounds=True,
                )
                source_node = AtlasTaskNode(int(source["lineage_node_id"]), np.asarray([source[name] for name in XYZ_COLUMNS], dtype=float), ())
                reverse = reverse_solver(endpoint, source_node)
                if reverse.success and reverse.actual_bounds:
                    reverse_gap = normalized_weighted_beta_deg(reverse.beta_rad, source_beta)
            rows.append(
                {
                    "target_id": str(target["proposal_id"]),
                    "solver_target_node_id": int(target["solver_target_node_id"]),
                    "physical_point_id": stable_id("retry12_seam_label", target["proposal_id"]),
                    "source_task_node_id": int(source["lineage_node_id"]),
                    "source_physical_point_id": str(source["physical_point_id"]),
                    "source_path_id": str(source["physical_point_id"]),
                    "solver_method": f"constrained_{seam_class}",
                    "solver_success": success,
                    "actual_bounds": success,
                    "fk_residual_mm": residual_mm,
                    "reverse_gap_deg": reverse_gap,
                    "source_edge_length_mm": float(distance_m) * 1000.0,
                    "solver_status": status,
                    **dict(zip(XYZ_COLUMNS, target_xyz, strict=True)),
                    **dict(zip(BETA_COLUMNS, beta, strict=True)),
                }
            )
    return pd.DataFrame.from_records(rows)


def stage_symmetry_fixed_seams(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["symmetry_fixed_seams"]
    atlas_gate = _gate(output_root, "zero_rooted_outward_atlas")
    if not atlas_gate.get("seam_execution_authorized", False):
        return _seal_gate(output_root, config, "symmetry_fixed_seams", {"status": "not_authorized", "seam_continuity_pass": False})
    retained = pd.read_parquet(output_root / STAGE_DIRS["zero_rooted_outward_atlas"] / "zero_rooted_fundamental_atlas.parquet")
    registry = pd.read_parquet(output_root / STAGE_DIRS["quotient_target_registry"] / "quotient_target_registry.parquet")
    symmetry_gate = _gate(output_root, "symmetry_contract")
    zero = _read_json(output_root / STAGE_DIRS["symmetry_contract"] / "zero_anchor.json")["xyz0_m"]
    environment = _environment(config)
    q = config["quotient_registry"]
    kwargs = {
        "zero_xyz_m": zero,
        "shell_width_mm": float(symmetry_gate["shell_width_mm"]),
        "seam_half_width_mm": float(symmetry_gate["seam_half_width_mm"]),
        "angle_bins": int(q["angle_bin_count"]),
        "cell_size_mm": float(q["cell_size_mm"]),
        "x_bin_mm": float(q["x_bin_mm"]),
    }
    all_attempts: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    proposal_frames: list[pd.DataFrame] = []
    provenance_frames: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    for seam_index, seam_class in enumerate(("y_seam", "z_seam")):
        proposal_beta = reduced_seam_beta_samples(
            np.asarray(environment.bounds), seam=seam_class,
            power=int(config["seams"]["proposal_power_per_seam"]),
            seed=int(config["runtime"]["seed"]) + 100 + seam_index,
        )
        proposal_xyz = np.asarray(environment.fk(proposal_beta), dtype=float).reshape(-1, 3)
        plane_axis = 1 if seam_class == "y_seam" else 2
        noise = np.abs(proposal_xyz[:, plane_axis])
        snap = noise <= float(config["symmetry"]["exact_plane_snap_tolerance_m"])
        proposal_xyz[snap, plane_axis] = 0.0
        provenance = pd.DataFrame(proposal_beta, columns=[f"proposal_{name}" for name in BETA_COLUMNS])
        provenance.loc[:, XYZ_COLUMNS] = proposal_xyz
        provenance["seam_class"] = seam_class
        provenance["plane_noise_m"] = noise
        provenance["proposal_beta_label_eligible"] = False
        provenance["proposal_beta_seed_eligible"] = False
        provenance["proposal_beta_warm_start_eligible"] = False
        provenance["proposal_beta_tie_break_eligible"] = False
        provenance["proposal_beta_branch_hint_eligible"] = False
        provenance_frames.append(provenance)
        proposals = proposal_registry(proposal_xyz[snap], source=f"{seam_class}_reduced_sobol", **kwargs)
        proposals["seam_class"] = seam_class
        proposals["region_role"] = seam_class
        proposals = proposals.sort_values(["zero_radius_mm", "proposal_id"], kind="stable").drop_duplicates("quotient_parent_id")
        envelope = registry.loc[:, XYZ_COLUMNS]
        proposals = proposals[
            proposals["x_m"].between(float(envelope["x_m"].min()), float(envelope["x_m"].max()))
            & proposals["zero_radius_mm"].le(float(registry["zero_radius_mm"].max()) + 1.0e-9)
        ].copy().reset_index(drop=True)
        proposals["solver_target_node_id"] = np.arange(
            10_000_000 + seam_index * 1_000_000,
            10_000_000 + seam_index * 1_000_000 + len(proposals), dtype=np.int64,
        )
        proposal_frames.append(proposals)
        # Every certified zero-lineage row may be an adjacent-interior source;
        # accepted exact-seam rows then provide the dedicated seam-chart path.
        seam_retained = retained.copy()
        pending = proposals.copy()
        accepted_ids: set[str] = set()
        attempts_for_seam: list[pd.DataFrame] = []
        labels_for_seam: list[pd.DataFrame] = []
        batch = 0
        while not pending.empty:
            frontier = spatially_balanced_frontier(
                pending, seam_retained,
                maximum_step_mm=float(config["seams"]["maximum_verified_edge_mm"]),
                count=int(config["atlas"]["roots_per_batch"]),
            )
            if frontier.empty:
                break
            batch += 1
            attempts = _solve_seam_targets(
                environment, frontier, seam_retained, seam_class=seam_class,
                maximum_step_mm=float(config["seams"]["maximum_verified_edge_mm"]),
            )
            labels, _rejects = classify_retry11_targets(attempts, seam_retained)
            labels = _annotate_new_labels(labels, frontier, batch=batch, lineage_offset=len(retained) + len(seam_retained))
            if not labels.empty:
                labels["label_origin"] = "retry12_symmetry_fixed_seam"
                labels["seam_class"] = seam_class
                labels["region_role"] = seam_class
                labels["mandatory_seam_representative"] = True
                edge_by_target = attempts.groupby("target_id")["source_edge_length_mm"].min()
                labels["source_edge_length_mm"] = labels["target_id"].map(edge_by_target)
                seam_retained = pd.concat([seam_retained, labels], ignore_index=True, sort=False)
                labels_for_seam.append(labels)
                accepted_ids.update(labels["target_id"].astype(str))
            attempted_ids = set(frontier["proposal_id"].astype(str))
            pending = pending[~pending["proposal_id"].astype(str).isin(attempted_ids)].copy()
            attempts_for_seam.append(attempts)
            _progress(output_root, "symmetry_fixed_seams", completed=len(accepted_ids), total=len(proposals), message=seam_class)
        seam_labels = pd.concat(labels_for_seam, ignore_index=True, sort=False) if labels_for_seam else pd.DataFrame()
        seam_attempts = pd.concat(attempts_for_seam, ignore_index=True, sort=False) if attempts_for_seam else pd.DataFrame()
        if not seam_labels.empty:
            reports[seam_class] = seam_continuity_metrics(
                proposals, seam_labels, seam_class=seam_class,
                maximum_edge_mm=float(config["seams"]["maximum_verified_edge_mm"]),
            )
        else:
            reports[seam_class] = {
                "seam_class": seam_class, "eligible_parent_count": len(proposals),
                "served_parent_count": 0, "served_parent_fraction": 0.0,
                "served_shell_fraction": 0.0, "maximum_consecutive_empty_shells": int(proposals["radial_shell_id"].nunique()),
                "maximum_verified_edge_length_mm": None, "verified_edge_length_pass": False,
            }
        all_attempts.append(seam_attempts)
        all_labels.append(seam_labels)
    attempts = pd.concat(all_attempts, ignore_index=True, sort=False) if all_attempts else pd.DataFrame()
    labels = pd.concat(all_labels, ignore_index=True, sort=False) if all_labels else pd.DataFrame()
    continuity_pass = all(
        float(report["served_parent_fraction"]) >= float(config["seams"]["served_parent_fraction_min"])
        and int(report["maximum_consecutive_empty_shells"]) <= int(config["seams"]["maximum_consecutive_empty_shells"])
        and bool(report["verified_edge_length_pass"])
        for report in reports.values()
    )
    _write_parquet(pd.concat(provenance_frames, ignore_index=True, sort=False), stage / "seam_proposal_beta_provenance.parquet")
    _write_parquet(pd.concat(proposal_frames, ignore_index=True, sort=False), stage / "seam_target_registry.parquet")
    _write_parquet(attempts, stage / "seam_solver_attempts.parquet")
    _write_parquet(labels, stage / "symmetry_fixed_seam_labels.parquet")
    _write_json(stage / "seam_continuity_report.json", reports)
    return _seal_gate(
        output_root,
        config,
        "symmetry_fixed_seams",
        {
            "status": "pass" if continuity_pass else "coverage_red",
            "seam_continuity_pass": continuity_pass,
            "seam_label_count": len(labels),
            "seams": reports,
            "dataset_continuation_authorized": True,
            "coverage_failure_does_not_invalidate_clean_labels": True,
        },
    )


def _representatives_for_selection(config: Mapping[str, Any], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    atlas = pd.read_parquet(
        output_root / STAGE_DIRS["zero_rooted_outward_atlas"] / "zero_rooted_fundamental_atlas.parquet"
    )
    seams = pd.read_parquet(
        output_root / STAGE_DIRS["symmetry_fixed_seams"] / "symmetry_fixed_seam_labels.parquet"
    )
    if "mandatory_seam_representative" not in atlas:
        atlas["mandatory_seam_representative"] = False
    else:
        atlas["mandatory_seam_representative"] = atlas["mandatory_seam_representative"].fillna(False)
    combined = pd.concat([atlas, seams], ignore_index=True, sort=False)
    combined["mandatory_seam_representative"] = combined["mandatory_seam_representative"].fillna(False).astype(bool)
    zero = _read_json(output_root / STAGE_DIRS["symmetry_contract"] / "zero_anchor.json")["xyz0_m"]
    tolerance = float(config["symmetry"]["exact_plane_snap_tolerance_m"])
    kinds = [orbit_class(row, exact_tolerance_m=tolerance, zero_xyz_m=zero) for row in combined.loc[:, XYZ_COLUMNS].to_numpy(float)]
    combined["seam_class"] = kinds
    invalid_nonzero_axis = combined["seam_class"].eq("double_seam_nonzero_abstain")
    inherited_exact_seam = combined["seam_class"].isin(["y_seam", "z_seam"]) & ~combined["label_origin"].eq("retry12_symmetry_fixed_seam")
    abstained = combined[invalid_nonzero_axis | inherited_exact_seam].copy()
    abstained["selection_rejection_reason"] = np.where(
        invalid_nonzero_axis[invalid_nonzero_axis | inherited_exact_seam],
        "double_seam_nonzero_abstain",
        "exact_seam_without_symmetry_fixed_label",
    )
    combined = combined[~(invalid_nonzero_axis | inherited_exact_seam)].copy()
    combined["orbit_size"] = combined["seam_class"].map({"exact_zero": 1, "y_seam": 2, "z_seam": 2, "interior": 4}).astype(int)
    combined = combined.sort_values(["fk_residual_mm", "fundamental_representative_id"], kind="stable").drop_duplicates("fundamental_representative_id")
    combined["selection_priority"] = combined["fundamental_representative_id"].astype(str).map(
        lambda value: int(hashlib.sha256(f"{config['runtime']['seed']}:{value}".encode()).hexdigest()[:16], 16)
    )
    return combined.reset_index(drop=True), abstained.reset_index(drop=True)


def _select_under_orbit_budget(representatives: pd.DataFrame, target: int) -> pd.DataFrame:
    zero = representatives[representatives["orbit_size"].eq(1)].head(1)
    seams = representatives[representatives["orbit_size"].eq(2)].sort_values(
        ["mandatory_seam_representative", "selection_priority"], ascending=[False, True], kind="stable"
    )
    interiors = representatives[representatives["orbit_size"].eq(4)].sort_values("selection_priority", kind="stable")
    seam_count = min(len(seams), max(1, (int(target) - 1) // 2))
    if seam_count % 2 == 0:
        seam_count -= 1
    remaining = max(0, int(target) - 1 - 2 * seam_count)
    interior_count = min(len(interiors), remaining // 4)
    return pd.concat([zero, seams.head(seam_count), interiors.head(interior_count)], ignore_index=True)


def stage_orbit_selection_expansion_split(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["orbit_selection_expansion_split"]
    representatives, abstained = _representatives_for_selection(config, output_root)
    target = int(config["dataset"]["target_expanded_rows"])
    exact_selection = True
    try:
        selected = select_exact_orbit_budget(representatives, target_expanded_rows=target)
    except ValueError:
        exact_selection = False
        selected = _select_under_orbit_budget(representatives, target)
    environment = _environment(config)
    zero = _read_json(output_root / STAGE_DIRS["symmetry_contract"] / "zero_anchor.json")["xyz0_m"]
    expanded, orbit_rejects = expand_symmetry_orbits(
        selected,
        environment,
        zero_xyz_m=zero,
        exact_tolerance_m=float(config["symmetry"]["exact_plane_snap_tolerance_m"]),
        fk_max_mm=float(config["atlas"]["fk_wide_silver_max_mm"]),
    )
    expanded = assign_quotient_macroblock_splits(
        expanded,
        block_size_mm=int(config["quotient_registry"]["macroblock_size_mm"]),
        seed=int(config["runtime"]["seed"]),
    )
    selected = assign_quotient_macroblock_splits(
        selected,
        block_size_mm=int(config["quotient_registry"]["macroblock_size_mm"]),
        seed=int(config["runtime"]["seed"]),
    )
    split_leakage = int(expanded.groupby("symmetry_orbit_id")["split_role"].nunique().gt(1).sum()) if len(expanded) else 0
    macroblock_leakage = int(expanded.groupby("quotient_macroblock_id")["split_role"].nunique().gt(1).sum()) if len(expanded) else 0
    expected_rows = int(selected["orbit_size"].sum())
    whole_orbit_pass = bool(len(expanded) == expected_rows and orbit_rejects.empty)
    _write_parquet(representatives, stage / "fundamental_candidate_representatives.parquet")
    _write_parquet(abstained, stage / "abstained_representatives.parquet")
    _write_parquet(selected, stage / "selected_fundamental_representatives.parquet")
    _write_parquet(orbit_rejects, stage / "rejected_orbits.parquet")
    _write_parquet(expanded, stage / "expanded_kinematic_candidates.parquet")
    return _seal_gate(
        output_root,
        config,
        "orbit_selection_expansion_split",
        {
            "status": "pass" if whole_orbit_pass else "underfilled_or_invalid",
            "available_representative_count": len(representatives),
            "selected_representative_count": len(selected),
            "selected_size1_orbits": int(selected["orbit_size"].eq(1).sum()),
            "selected_size2_orbits": int(selected["orbit_size"].eq(2).sum()),
            "selected_size4_orbits": int(selected["orbit_size"].eq(4).sum()),
            "required_size2_orbit_parity": "odd",
            "size2_orbit_parity_pass": bool(int(selected["orbit_size"].eq(2).sum()) % 2 == 1),
            "exact_target_selection_available": exact_selection,
            "expected_expanded_rows": expected_rows,
            "actual_expanded_rows": len(expanded),
            "whole_orbit_integrity_pass": whole_orbit_pass,
            "orbit_split_leakage_count": split_leakage,
            "quotient_macroblock_split_leakage_count": macroblock_leakage,
            "split_integrity_pass": split_leakage == 0 and macroblock_leakage == 0,
            "dataset_freeze_authorized": bool(whole_orbit_pass and split_leakage == 0 and macroblock_leakage == 0),
        },
    )


def _connected_components(node_count: int, edge_pairs: np.ndarray) -> tuple[np.ndarray, list[int]]:
    adjacency = [[] for _ in range(node_count)]
    for left, right in edge_pairs:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    labels = np.full(node_count, -1, dtype=np.int64)
    sizes: list[int] = []
    component = 0
    for start in range(node_count):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = component
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    stack.append(neighbor)
        sizes.append(size)
        component += 1
    return labels, sizes


def stage_full_graph_connectivity_audit(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["full_graph_connectivity_audit"]
    expanded = pd.read_parquet(
        output_root / STAGE_DIRS["orbit_selection_expansion_split"] / "expanded_kinematic_candidates.parquet"
    )
    if expanded.empty:
        _write_parquet(pd.DataFrame(), stage / "verified_task_edges.parquet")
        return _seal_gate(output_root, config, "full_graph_connectivity_audit", {"status": "fail", "full_graph_connectivity_pass": False, "reason": "empty_expanded_dataset"})
    xyz = expanded.loc[:, XYZ_COLUMNS].to_numpy(float)
    maximum_edge_mm = float(config["seams"]["maximum_verified_edge_mm"])
    pairs = np.asarray(sorted(cKDTree(xyz).query_pairs(maximum_edge_mm / 1000.0 + 1.0e-12)), dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    labels, sizes = _connected_components(len(expanded), pairs)
    zero_positions = np.flatnonzero(expanded["seam_class"].eq("exact_zero").to_numpy())
    zero_component = int(labels[zero_positions[0]]) if len(zero_positions) == 1 else -1
    in_zero_component = labels == zero_component
    quadrant = np.column_stack([np.sign(xyz[:, 1]).astype(int), np.sign(xyz[:, 2]).astype(int)])
    quadrant_connected: dict[str, bool] = {}
    for sy, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        mask = (quadrant[:, 0] == sy) & (quadrant[:, 1] == sz)
        quadrant_connected[f"y{sy:+d}_z{sz:+d}"] = bool(mask.any() and np.all(in_zero_component[mask]))
    beta = expanded.loc[:, BETA_COLUMNS].to_numpy(float)
    edge_rows: list[dict[str, Any]] = []
    seam_weighted: list[float] = []
    seam_raw: list[float] = []
    for left, right in pairs:
        distance_mm = float(np.linalg.norm(xyz[left] - xyz[right]) * 1000.0)
        cross_y = bool(xyz[left, 1] * xyz[right, 1] <= 0.0 and (abs(xyz[left, 1]) + abs(xyz[right, 1]) > 0.0))
        cross_z = bool(xyz[left, 2] * xyz[right, 2] <= 0.0 and (abs(xyz[left, 2]) + abs(xyz[right, 2]) > 0.0))
        weighted = normalized_weighted_beta_deg(beta[left], beta[right])
        raw = raw_beta_max_deg(beta[left], beta[right])
        if cross_y or cross_z:
            seam_weighted.append(weighted)
            seam_raw.append(raw)
        edge_rows.append({"left_row": int(left), "right_row": int(right), "distance_mm": distance_mm, "cross_y_seam": cross_y, "cross_z_seam": cross_z, "weighted_beta_gap_deg": weighted, "raw_beta_gap_deg": raw})
    orbit_expected = expanded.groupby("symmetry_orbit_id")["orbit_size"].first().sum()
    orbit_completeness = float(len(expanded) / orbit_expected) if orbit_expected else 0.0
    seam_weighted_p95 = percentile(np.asarray(seam_weighted), 95) if seam_weighted else math.inf
    seam_raw_rate = float(np.mean(np.asarray(seam_raw) > float(config["atlas"]["neighbor_raw_max_deg"]))) if seam_raw else 1.0
    seam_pair_pass = bool(seam_weighted_p95 <= float(config["atlas"]["neighbor_weighted_max_deg"]) and seam_raw_rate <= float(config["atlas"]["neighbor_raw_catastrophic_rate_max"]))
    full_pass = bool(all(quadrant_connected.values()) and orbit_completeness >= 0.99 and seam_pair_pass)
    expanded_with_components = expanded.copy()
    expanded_with_components["expanded_graph_component_id"] = labels
    _write_parquet(pd.DataFrame.from_records(edge_rows), stage / "verified_task_edges.parquet")
    _write_parquet(expanded_with_components, stage / "expanded_component_registry.parquet")
    return _seal_gate(
        output_root,
        config,
        "full_graph_connectivity_audit",
        {
            "status": "pass" if full_pass else "not_verified",
            "full_graph_connectivity_pass": full_pass,
            "component_count": len(sizes),
            "zero_component_row_count": int(np.count_nonzero(in_zero_component)),
            "quadrant_connected_to_zero": quadrant_connected,
            "all_quadrants_connected_to_zero": all(quadrant_connected.values()),
            "verified_edge_count": len(pairs),
            "maximum_verified_edge_mm": float(np.max([row["distance_mm"] for row in edge_rows])) if edge_rows else None,
            "orbit_completeness": orbit_completeness,
            "seam_weighted_p95_deg": seam_weighted_p95,
            "seam_raw_gt7_rate": seam_raw_rate,
            "seam_beta_continuity_pass": seam_pair_pass,
        },
    )


def stage_freeze_kinematic_dataset(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["freeze_kinematic_dataset"]
    selection_gate = _gate(output_root, "orbit_selection_expansion_split")
    dataset = pd.read_parquet(
        output_root / STAGE_DIRS["orbit_selection_expansion_split"] / "expanded_kinematic_candidates.parquet"
    )
    minimum = int(config["dataset"]["minimum_expanded_rows"])
    target = int(config["dataset"]["target_expanded_rows"])
    environment = _environment(config)
    theta = dataset.loc[:, THETA_COLUMNS].to_numpy(float) if len(dataset) else np.empty((0, 30))
    beta = dataset.loc[:, BETA_COLUMNS].to_numpy(float) if len(dataset) else np.empty((0, 6))
    expected_theta = np.asarray([beta_to_theta(row) for row in beta]) if len(beta) else np.empty((0, 30))
    theta_storage_pass = bool(theta.shape == (len(dataset), 30) and np.allclose(theta, expected_theta, atol=0.0, rtol=0.0))
    finite = bool(len(dataset) and np.isfinite(dataset.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS, *THETA_COLUMNS]].to_numpy(float)).all())
    bounds = np.asarray(environment.bounds, dtype=float)
    bounds_pass = bool(len(beta) and np.all((beta >= bounds[:, 0] - 1.0e-12) & (beta <= bounds[:, 1] + 1.0e-12)))
    split_pass = bool(selection_gate.get("split_integrity_pass", False))
    row_integrity = bool(selection_gate.get("whole_orbit_integrity_pass", False) and theta_storage_pass and finite and bounds_pass and split_pass)
    freeze_pass = bool(row_integrity and len(dataset) >= minimum)
    seam_gate = _gate(output_root, "symmetry_fixed_seams")
    graph_gate = _gate(output_root, "full_graph_connectivity_audit")
    atlas_gate = _gate(output_root, "zero_rooted_outward_atlas")
    wide_claim = bool(atlas_gate.get("remote_uniform_coverage_pass", False) and seam_gate.get("seam_continuity_pass", False) and graph_gate.get("full_graph_connectivity_pass", False))
    dataset_class = "wide_zero_centered_symmetry_dataset" if wide_claim else "domain_limited_zero_centered_dataset"
    dataset["dataset_id"] = str(config["dataset"]["dataset_id"] if wide_claim else config["dataset"]["domain_limited_dataset_id"])
    dataset["dataset_class"] = dataset_class
    dataset["canonical_component_id"] = "retry12_zero_centered_symmetry_component"
    dataset["kinematics_theta_sign"] = float(config["dataset"]["kinematics_theta_sign_metadata"])
    dataset["scientific_source_fixed_point"] = _git_sha()
    dataset["config_sha256"] = _config_sha(config)
    dataset_path = stage / "retry12_zero_centered_kinematic_dataset.parquet"
    _write_parquet(dataset, dataset_path)
    return _seal_gate(
        output_root,
        config,
        "freeze_kinematic_dataset",
        {
            "status": "pass" if freeze_pass else "underfilled_or_invalid",
            "dataset_freeze_complete": freeze_pass,
            "dataset_row_count": len(dataset),
            "minimum_row_count": minimum,
            "exact_target_row_count": target,
            "exact_target_complete": len(dataset) == target,
            "canonical_row_integrity_pass": row_integrity,
            "theta_storage_direct_beta_expansion_pass": theta_storage_pass,
            "theta_sign_applied_to_storage": False,
            "split_integrity_pass": split_pass,
            "dataset_class": dataset_class,
            "wide_zero_centered_dataset_claim": wide_claim,
            "student_execution_authorized": freeze_pass,
        },
    )


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
    frozen_indices: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    corrected = np.asarray(beta, dtype=float).copy()
    bounds = np.asarray(environment.bounds, dtype=float)
    weights = np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0], dtype=float)
    frozen = [tuple() for _ in range(len(corrected))] if frozen_indices is None else list(frozen_indices)
    for _ in range(2):
        for index in range(len(corrected)):
            freeze = np.asarray(frozen[index], dtype=int)
            free = np.asarray([axis for axis in range(6) if axis not in set(freeze)], dtype=int)
            if not len(free):
                corrected[index] = 0.0
                continue
            current = np.asarray(environment.fk(corrected[index]), dtype=float).reshape(-1, 3)[0]
            jacobian = np.asarray(environment.jacobian(corrected[index]), dtype=float).reshape(3, 6)[:, free]
            free_weights = weights[free]
            weight_inverse = np.diag(1.0 / free_weights)
            task = jacobian @ weight_inverse @ jacobian.T + 1.0e-6 * np.eye(3)
            pseudoinverse = weight_inverse @ jacobian.T @ np.linalg.pinv(task, rcond=1.0e-12)
            step = pseudoinverse @ (xyz[index] - current)
            corrected[index, free] = np.clip(corrected[index, free] + step, bounds[free, 0], bounds[free, 1])
            if len(freeze):
                corrected[index, freeze] = 0.0
    return corrected


def stage_quotient_student(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["quotient_student"]
    freeze_gate = _gate(output_root, "freeze_kinematic_dataset")
    if not freeze_gate.get("student_execution_authorized", False):
        return _seal_gate(output_root, config, "quotient_student", {"status": "not_authorized", "student_training_complete": False, "trajectory_execution_authorized": False})
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("retry12 Student requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("retry12 CPU Student unexpectedly sees a GPU")
    representatives = pd.read_parquet(
        output_root / STAGE_DIRS["orbit_selection_expansion_split"] / "selected_fundamental_representatives.parquet"
    )
    environment = _environment(config)
    supervision = representatives.copy()
    jacobians = np.asarray([
        np.asarray(environment.jacobian(beta), dtype=float).reshape(-1)
        for beta in supervision.loc[:, BETA_COLUMNS].to_numpy(float)
    ])
    for index, column in enumerate(JACOBIAN_COLUMNS):
        supervision[column] = jacobians[:, index]
    supervision["record_id"] = supervision["fundamental_representative_id"].astype(str)
    supervision["kind"] = "static"
    supervision["chart_id"] = "retry12_quotient"
    supervision["is_primary"] = True
    weights = config["student"]["quality_weight"]
    supervision["sample_weight"] = supervision["label_quality"].astype(str).map(weights).fillna(0.5).astype(float)
    train = supervision[supervision["split_role"].eq("train")].copy()
    validation = supervision[supervision["split_role"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise RuntimeError("retry12 quotient macroblock split produced empty train or validation")
    student = config["student"]
    trained = train_workspace_student(
        train,
        validation,
        mode=RepresentationMode.XYZ_GLOBAL,
        geometry=_student_geometry(environment),
        config=WorkspaceStudentTrainingConfig(
            hidden_units=tuple(map(int, student["hidden_units"])),
            learning_rate=float(student["learning_rate"]),
            max_steps=int(student["maximum_steps"]),
            validation_interval=int(student["validation_interval"]),
            patience_intervals=int(student["patience_intervals"]),
            seed=int(student["seed"]),
            beta_coordinate_weights=(4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
            beta_loss_only=True,
        ),
    )
    save_workspace_student_models(trained.models, stage / "models")
    model = trained.models.global_model
    xyz = validation.loc[:, XYZ_COLUMNS].to_numpy(np.float32)
    truth = validation.loc[:, BETA_COLUMNS].to_numpy(float)
    prediction = np.asarray(model(xyz, training=False), dtype=float)
    raw_fk = np.linalg.norm(np.asarray(environment.fk(prediction)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    corrected = _two_step_dls(environment, prediction, xyz)
    dls_fk = np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    bounds = np.asarray(environment.bounds, dtype=float)
    bounds_pass = bool(np.all((prediction >= bounds[:, 0] - 1.0e-12) & (prediction <= bounds[:, 1] + 1.0e-12)))
    finite = bool(np.isfinite(prediction).all() and np.isfinite(raw_fk).all() and np.isfinite(dls_fk).all())
    dls_success = np.isfinite(corrected).all(axis=1) & (dls_fk <= float(student["dls_success_fk_max_mm"]))
    predictions = validation.loc[:, ["fundamental_representative_id", *XYZ_COLUMNS, *BETA_COLUMNS]].copy()
    for index, name in enumerate(BETA_COLUMNS):
        predictions[f"predicted_{name}"] = prediction[:, index]
        predictions[f"dls2_{name}"] = corrected[:, index]
    predictions["raw_fk_residual_mm"] = raw_fk
    predictions["dls2_fk_residual_mm"] = dls_fk
    predictions["weighted_beta_error_deg"] = [normalized_weighted_beta_deg(left, right) for left, right in zip(prediction, truth, strict=True)]
    predictions["dls2_success"] = dls_success
    _write_parquet(supervision, stage / "student_supervision.parquet")
    _write_parquet(trained.history, stage / "training_history.parquet")
    _write_parquet(predictions, stage / "validation_predictions.parquet")
    complete = bool(finite and bounds_pass)
    return _seal_gate(
        output_root,
        config,
        "quotient_student",
        {
            "status": "complete" if complete else "red",
            "student_training_complete": complete,
            "train_row_count": len(train),
            "validation_row_count": len(validation),
            "finite_pass": finite,
            "bounds_pass": bounds_pass,
            "validation_raw_fk_p95_mm": percentile(raw_fk, 95),
            "validation_dls2_fk_p95_mm": percentile(dls_fk, 95),
            "validation_dls2_success_rate": float(np.mean(dls_success)),
            "trajectory_execution_authorized": complete,
            "student_failure_does_not_invalidate_dataset": True,
        },
    )


def _symmetry_wrapper_prediction(model: Any, xyz: np.ndarray) -> np.ndarray:
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
    exact_zero = np.linalg.norm(points - points[0], axis=1) <= 1.0e-12
    if exact_zero.any():
        beta[exact_zero] = 0.0
    return beta


def _trajectory_endpoints(representatives: pd.DataFrame, zero_xyz: np.ndarray) -> pd.DataFrame:
    interior = representatives[representatives["orbit_size"].eq(4)].copy()
    if interior.empty:
        return pd.DataFrame()
    radii = np.quantile(interior["zero_radius_mm"], [0.35, 0.65, 0.90])
    angle_targets = [np.pi / 6.0, np.pi / 3.0]
    records: list[dict[str, Any]] = []
    for radial_index, radius in enumerate(radii):
        for angle_index, angle in enumerate(angle_targets):
            score = np.abs(interior["zero_radius_mm"].to_numpy(float) - radius) / max(1.0, radius)
            score += np.abs(interior["phi_rad"].to_numpy(float) - angle)
            representative = interior.iloc[int(np.argmin(score))]
            base = representative.loc[list(XYZ_COLUMNS)].to_numpy(float)
            for sy, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                point = base.copy()
                point[1] *= sy
                point[2] *= sz
                records.append({"trajectory_id": f"interior_r{radial_index}_a{angle_index}_y{sy:+d}_z{sz:+d}", "trajectory_class": "interior", **dict(zip(XYZ_COLUMNS, point, strict=True))})
    seams = representatives[representatives["orbit_size"].eq(2)].copy()
    for seam_class in ("y_seam", "z_seam"):
        candidate = seams[seams["seam_class"].eq(seam_class)].sort_values("zero_radius_mm", ascending=False, kind="stable").head(1)
        if candidate.empty:
            continue
        base = candidate.iloc[0].loc[list(XYZ_COLUMNS)].to_numpy(float)
        if seam_class == "y_seam":
            signs = ((1, 1), (1, -1))
        else:
            signs = ((1, 1), (-1, 1))
        for ordinal, (sy, sz) in enumerate(signs):
            point = base.copy()
            point[1] *= sy
            point[2] *= sz
            records.append({"trajectory_id": f"{seam_class}_{ordinal}", "trajectory_class": seam_class, **dict(zip(XYZ_COLUMNS, point, strict=True))})
    return pd.DataFrame.from_records(records).head(28)


def stage_zero_to_target_trajectory_audit(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["zero_to_target_trajectory_audit"]
    student_gate = _gate(output_root, "quotient_student")
    if not student_gate.get("trajectory_execution_authorized", False):
        return _seal_gate(output_root, config, "zero_to_target_trajectory_audit", {"status": "not_authorized", "trajectory_gate_pass": False})
    models = load_workspace_student_models(output_root / STAGE_DIRS["quotient_student"] / "models")
    model = models.global_model
    representatives = pd.read_parquet(
        output_root / STAGE_DIRS["orbit_selection_expansion_split"] / "selected_fundamental_representatives.parquet"
    )
    zero = np.asarray(_read_json(output_root / STAGE_DIRS["symmetry_contract"] / "zero_anchor.json")["xyz0_m"], dtype=float)
    endpoints = _trajectory_endpoints(representatives, zero)
    environment = _environment(config)
    waypoint_count = int(config["trajectory"]["waypoints_per_trajectory"])
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_reports: list[dict[str, Any]] = []
    all_dls_fk: list[float] = []
    all_corrected_steps: list[float] = []
    successful_trajectories = 0
    for endpoint in endpoints.to_dict("records"):
        target = np.asarray([endpoint[name] for name in XYZ_COLUMNS], dtype=float)
        fraction = np.linspace(0.0, 1.0, waypoint_count)
        xyz = zero.reshape(1, 3) + fraction.reshape(-1, 1) * (target - zero).reshape(1, 3)
        raw_beta = _symmetry_wrapper_prediction(model, xyz)
        frozen: list[Sequence[int]] = []
        for point in xyz:
            if np.linalg.norm(point - zero) <= 1.0e-12:
                frozen.append(tuple(range(6)))
            elif abs(point[1]) <= 1.0e-12:
                frozen.append((0, 2, 4))
            elif abs(point[2]) <= 1.0e-12:
                frozen.append((1, 3, 5))
            else:
                frozen.append(())
        corrected = _two_step_dls(environment, raw_beta, xyz, frozen_indices=frozen)
        raw_fk = np.linalg.norm(np.asarray(environment.fk(raw_beta)).reshape(-1, 3) - xyz, axis=1) * 1000.0
        dls_fk = np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - xyz, axis=1) * 1000.0
        metrics = corrected_trajectory_metrics(raw_beta, corrected)
        corrected_steps = np.max(np.abs(np.diff(corrected, axis=0)), axis=1) * 180.0 / np.pi
        success = bool(
            np.isfinite(corrected).all()
            and percentile(dls_fk, 95) <= float(config["trajectory"]["fk_p95_max_mm"])
            and metrics["dls2_corrected_step_gt7_rate"] <= float(config["trajectory"]["corrected_beta_step_catastrophic_rate_max"])
        )
        successful_trajectories += int(success)
        all_dls_fk.extend(dls_fk.tolist())
        all_corrected_steps.extend(corrected_steps.tolist())
        trajectory_reports.append({"trajectory_id": endpoint["trajectory_id"], "trajectory_class": endpoint["trajectory_class"], "success": success, "dls2_fk_p95_mm": percentile(dls_fk, 95), **metrics})
        for index in range(waypoint_count):
            row = {"trajectory_id": endpoint["trajectory_id"], "trajectory_class": endpoint["trajectory_class"], "waypoint_index": index, **dict(zip(XYZ_COLUMNS, xyz[index], strict=True)), "raw_fk_residual_mm": raw_fk[index], "dls2_fk_residual_mm": dls_fk[index]}
            row.update({f"raw_{name}": raw_beta[index, position] for position, name in enumerate(BETA_COLUMNS)})
            row.update({f"dls2_{name}": corrected[index, position] for position, name in enumerate(BETA_COLUMNS)})
            trajectory_rows.append(row)
    success_rate = successful_trajectories / len(endpoints) if len(endpoints) else 0.0
    pooled_dls_p95 = percentile(np.asarray(all_dls_fk), 95) if all_dls_fk else math.inf
    corrected_rate = float(np.mean(np.asarray(all_corrected_steps) > float(config["trajectory"]["corrected_beta_step_raw_gap_deg"]))) if all_corrected_steps else 1.0
    gate_pass = bool(
        len(endpoints) == int(config["trajectory"]["trajectory_count"])
        and success_rate >= float(config["trajectory"]["success_rate_min"])
        and pooled_dls_p95 <= float(config["trajectory"]["fk_p95_max_mm"])
        and corrected_rate <= float(config["trajectory"]["corrected_beta_step_catastrophic_rate_max"])
    )
    _write_parquet(endpoints, stage / "trajectory_panel.parquet")
    _write_parquet(pd.DataFrame.from_records(trajectory_rows), stage / "trajectory_waypoints.parquet")
    _write_parquet(pd.DataFrame.from_records(trajectory_reports), stage / "trajectory_report.parquet")
    return _seal_gate(
        output_root,
        config,
        "zero_to_target_trajectory_audit",
        {
            "status": "pass" if gate_pass else "fail",
            "trajectory_gate_pass": gate_pass,
            "trajectory_count": len(endpoints),
            "trajectory_success_rate": success_rate,
            "pooled_dls2_fk_p95_mm": pooled_dls_p95,
            "pooled_dls2_corrected_step_gt7_rate": corrected_rate,
            "corrected_beta_continuity_is_hard_gate": True,
            "raw_student_continuity_is_diagnostic_only": True,
            "exact_seam_dls_coordinates_frozen": True,
        },
    )


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if (
            not path.is_file()
            or "_work" in path.parts
            or path.name in {"artifact_manifest.json", "progress.json"}
        ):
            continue
        artifacts.append({"path": str(path.relative_to(output_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"schema_version": 1, "scientific_source_fixed_point": _git_sha(), "artifacts": artifacts}


def stage_summary(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["summary"]
    symmetry = _gate(output_root, "symmetry_contract")
    registry = _gate(output_root, "quotient_target_registry")
    atlas = _gate(output_root, "zero_rooted_outward_atlas")
    seams = _gate(output_root, "symmetry_fixed_seams")
    selection = _gate(output_root, "orbit_selection_expansion_split")
    graph = _gate(output_root, "full_graph_connectivity_audit")
    freeze = _gate(output_root, "freeze_kinematic_dataset")
    student = _gate(output_root, "quotient_student")
    trajectory = _gate(output_root, "zero_to_target_trajectory_audit")
    operational = all(_stage_is_complete(output_root, name) for name in STAGE_ORDER[:-1])
    gate = {
        "status": "complete" if operational else "incomplete",
        "operational_completion": operational,
        "artifact_completeness": operational,
        "symmetry_contract": {"status": symmetry["status"], "pass": symmetry.get("symmetry_gate_pass", False)},
        "canonical_row_integrity": {"atlas_pass": atlas.get("canonical_row_integrity_pass", False), "dataset_pass": freeze.get("canonical_row_integrity_pass", False)},
        "remote_coverage": {"proposal_status": registry.get("remote_uniform_coverage_status"), "retained_status": atlas.get("remote_uniform_coverage_status"), "serviced_bins": atlas.get("remote_serviced_angle_bins")},
        "seam_continuity": {"status": seams["status"], "pass": seams.get("seam_continuity_pass", False)},
        "full_graph_connectivity": {"status": graph["status"], "pass": graph.get("full_graph_connectivity_pass", False)},
        "dataset": {"status": freeze["status"], "class": freeze.get("dataset_class"), "row_count": freeze.get("dataset_row_count"), "minimum_complete": freeze.get("dataset_freeze_complete", False), "exact_target_complete": freeze.get("exact_target_complete", False)},
        "student": {"status": student["status"], "training_complete": student.get("student_training_complete", False)},
        "trajectory": {"status": trajectory["status"], "pass": trajectory.get("trajectory_gate_pass", False)},
        "theta_storage": {"direct_beta_expansion": freeze.get("theta_storage_direct_beta_expansion_pass", False), "theta_sign_applied": False},
        "tension_executed": False,
        "formal_authorization": False,
        "deployment_authorization": False,
        "full_workspace_authorization": False,
        "continuous_workspace_authorization": False,
    }
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>retry12 summary</title></head><body><h1>retry12 对称商空间零点中心运动学闭环</h1><pre>{json.dumps(gate, ensure_ascii=False, indent=2)}</pre></body></html>"""
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "retry12_summary.html").write_text(html, encoding="utf-8")
    _write_json(stage / "artifact_manifest.json", _artifact_manifest(output_root))
    return _seal_gate(output_root, config, "summary", gate)


STAGE_RUNNERS: dict[str, Callable[[Mapping[str, Any], Path, Path], dict[str, Any]]] = {
    "symmetry_contract": stage_symmetry_contract,
    "quotient_target_registry": stage_quotient_target_registry,
    "zero_rooted_outward_atlas": stage_zero_rooted_outward_atlas,
    "symmetry_fixed_seams": stage_symmetry_fixed_seams,
    "orbit_selection_expansion_split": stage_orbit_selection_expansion_split,
    "full_graph_connectivity_audit": stage_full_graph_connectivity_audit,
    "freeze_kinematic_dataset": stage_freeze_kinematic_dataset,
    "quotient_student": stage_quotient_student,
    "zero_to_target_trajectory_audit": stage_zero_to_target_trajectory_audit,
    "summary": stage_summary,
}


def run(config_path: str | Path, output_root: str | Path, binding_sha: str, *, stage_name: str | None = None, validate_stage: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_root).resolve()
    _ensure_identity(config, output, binding_sha)
    if validate_stage is not None:
        if validate_stage not in STAGE_RUNNERS:
            raise ValueError(f"unknown stage {validate_stage}")
        if not _stage_is_complete(output, validate_stage):
            raise RuntimeError(f"retry12 stage is incomplete or mutated: {validate_stage}")
        return _gate(output, validate_stage)
    if stage_name is not None:
        if stage_name not in STAGE_RUNNERS:
            raise ValueError(f"unknown stage {stage_name}")
        position = STAGE_ORDER.index(stage_name)
        if position and not _stage_is_complete(output, STAGE_ORDER[position - 1]):
            raise RuntimeError(f"retry12 predecessor incomplete: {STAGE_ORDER[position - 1]}")
        _progress(output, stage_name, message="stage_started")
        result = STAGE_RUNNERS[stage_name](config, project_root_from(SOURCE_ROOT), output)
        if stage_name == "summary":
            _write_json(
                output / "progress.json",
                {
                    "status": "complete",
                    "phase": "summary",
                    "completed": len(STAGE_ORDER),
                    "total": len(STAGE_ORDER),
                    "message": "retry12 operational pipeline completed",
                    "observed_at_unix": time.time(),
                },
            )
        else:
            _progress(output, stage_name, completed=1, total=1, message="stage_completed")
        return result
    result: dict[str, Any] = {}
    for name in STAGE_ORDER:
        if _stage_is_complete(output, name):
            result = _gate(output, name)
            continue
        result = STAGE_RUNNERS[name](config, project_root_from(SOURCE_ROOT), output)
    _write_json(output / "progress.json", {"status": "complete", "phase": "summary", "completed": len(STAGE_ORDER), "total": len(STAGE_ORDER), "message": "retry12 operational pipeline completed"})
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
