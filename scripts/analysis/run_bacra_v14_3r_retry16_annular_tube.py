#!/usr/bin/env python3
"""Run the retry16 Omega600 exact-zero-rooted annular Teacher preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import percentile, weighted_beta_rms_deg
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry15_candidate_solver import seed_bank_xyz, teacher_seed_bank
from quasi_exp.teacher.retry15_canonical_graph import (
    Omega600Contract,
    ProbePolicy,
    legal_candidate_clusters,
    mutual_knn_edges,
    select_t1,
    select_t2,
    workspace_probe,
)
from quasi_exp.teacher.retry16_annular_tube import (
    AnnularPolicy,
    annular_volume_mm3,
    build_annular_target_registry,
    build_nested_annular_profiles,
    deterministic_target_sample,
    expand_annular_labels,
    root_connector_voxels,
)


EXPERIMENT_ID = "bacra_v14_3r_retry16_omega600_annular_tube_preflight"
STAGE_DIRS = {
    "inventory": "00_inventory",
    "annular_domain": "01_annular_domain",
    "objective_feasibility": "02_objective_feasibility",
    "seed_banks": "03_teacher_seed_banks",
    "candidate_preflight": "04_candidate_preflight",
    "dataset": "05_diagnostic_dataset",
    "summary": "06_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)
WORKER = SOURCE_ROOT / "scripts" / "analysis" / "run_bacra_v14_3r_retry15_worker.py"


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
    return optimized_forward(
        load_environment(
            project_root_from(SOURCE_ROOT),
            SOURCE_ROOT / str(config["sources"]["robot_config"]),
        )
    )


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
            "PYTHONHASHSEED": "20260901",
        }
    )
    return result


def _annular_policy(config: Mapping[str, Any]) -> AnnularPolicy:
    values = config["annular_domain"]
    return AnnularPolicy(
        axial_step_mm=float(values["axial_step_mm"]),
        radial_step_mm=float(values["radial_step_mm"]),
        full_sector_count=int(values["full_sector_count"]),
        inner_margin_mm=float(values["inner_margin_mm"]),
        outer_margin_mm=float(values["outer_margin_mm"]),
        slope_limit_mm=float(values["slope_limit_mm"]),
        minimum_thickness_mm=float(values["minimum_thickness_mm"]),
        root_link_maximum_mm=float(values["root_link_maximum_mm"]),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("retry16 annular experiment_id mismatch")
    omega = config["omega600"]
    Omega600Contract(
        x0_m=float(omega["zero_x_m"]),
        x_min_m=float(omega["minimum_x_m"]),
        length_mm=float(omega["axial_length_mm"]),
    )
    _annular_policy(config)
    if int(config["runtime"]["maximum_concurrent_workers"]) != 12:
        raise ValueError("retry16 must register 12 numerical workers")
    if int(config["runtime"]["numerical_threads_per_worker"]) != 1:
        raise ValueError("retry16 numerical workers must be single-threaded")
    domain = config["annular_domain"]
    if (int(domain["comparison_power"]), int(domain["final_power"])) != (18, 19):
        raise ValueError("retry16 must compare nested proposal powers 18 and 19")
    if int(domain["required_circle_count"]) != 9 or float(domain["required_large_circle_radius_mm"]) != 100.0:
        raise ValueError("retry16 must preserve the nine-circle 100 mm objective")
    if tuple(map(int, config["candidate_solver"]["saturation_budgets"])) != (8, 16, 32):
        raise ValueError("retry16 saturation budgets must be 8/16/32")
    seeds = [int(value["seed"]) for value in config["teacher_seed_banks"].values()]
    if seeds != [20260901, 20260902, 20260903] or len(set(seeds)) != 3:
        raise ValueError("retry16 Teacher seed identity mismatch")
    claims = config["claims"]
    if not claims["diagnostic_only"] or claims["claim_bearing_run_authorized"]:
        raise ValueError("retry16 preflight must remain diagnostic-only")
    if any(
        bool(claims[key])
        for key in (
            "downstream_scale_authorized",
            "formal_authorized",
            "deployment_authorized",
            "full_workspace_authorized",
            "continuous_workspace_authorized",
            "tension_authorized",
        )
    ):
        raise ValueError("retry16 cannot pre-authorize downstream claims")
    robot = SOURCE_ROOT / str(config["sources"]["robot_config"])
    if sha256_file(robot) != str(config["sources"]["robot_config_sha256"]):
        raise ValueError("retry16 robot config SHA mismatch")
    return config


def _binding_definition(binding_sha: str) -> Mapping[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{binding_sha}:spec/registry.yaml"], cwd=SOURCE_ROOT, text=True
    )
    return yaml.safe_load(raw)["experiments"][EXPERIMENT_ID]


def _ensure_identity(
    config: Mapping[str, Any], output_root: Path, binding_sha: str, *, smoke: bool
) -> dict[str, Any]:
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_source_fixed_point": _git_sha(),
        "binding_fixed_point": str(binding_sha),
        "config_sha256": _config_sha(config),
        "diagnostic_smoke": bool(smoke),
    }
    definition = _binding_definition(binding_sha)
    expected = {
        "scientific_source_fixed_point": _git_sha(),
        "config": str(Path(str(config["config_path"])).relative_to(SOURCE_ROOT)),
        "runner": str(Path(__file__).resolve().relative_to(SOURCE_ROOT)),
    }
    for key, value in expected.items():
        if str(definition.get(key)) != str(value):
            raise RuntimeError(f"retry16 binding mismatch for {key}: {definition.get(key)!r} != {value!r}")
    path = output_root / "run_identity.json"
    if path.exists() and _read_json(path) != identity:
        raise RuntimeError("retry16 output root identity mismatch")
    if not path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(path, identity)
    return identity


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    return Path(str(config["upstream"]["retry15_root"])) / str(config["upstream"][key]["path"])


def _stage_manifest(output_root: Path, stage_name: str) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "completion_manifest.json" and "_work" not in path.parts:
            artifacts.append(
                {
                    "path": str(path.relative_to(stage)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
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
        "diagnostic_only": True,
        "claim_bearing_run_authorized": False,
        "downstream_scale_authorized": False,
        "formal_authorized": False,
        "deployment_authorized": False,
        "full_workspace_authorized": False,
        "continuous_workspace_authorized": False,
        "tension_authorized": False,
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


def _input_integrity(config: Mapping[str, Any]) -> tuple[pd.DataFrame, bool]:
    rows = []
    for key, spec in config["upstream"].items():
        if not isinstance(spec, Mapping) or "sha256" not in spec:
            continue
        path = _upstream_path(config, key)
        observed = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "source_key": key,
                "path": str(path),
                "expected_sha256": str(spec["sha256"]),
                "observed_sha256": observed,
                "exists": path.exists(),
                "match": bool(path.exists() and observed == str(spec["sha256"])),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, bool(len(frame) and frame["match"].all())


def stage_inventory(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["inventory"]
    integrity, passed = _input_integrity(config)
    _write_parquet(integrity, stage / "upstream_verification.parquet")
    if not passed:
        raise RuntimeError("retry16 upstream integrity failed")
    summary = _read_json(_upstream_path(config, "retry15_summary_gate"))
    environment = _environment(config)
    zero_xyz = np.asarray(environment.fk(np.zeros((1, 6))), dtype=float).reshape(3)
    expected = np.asarray([float(config["omega600"]["zero_x_m"]), 0.0, 0.0])
    zero_pass = bool(np.allclose(zero_xyz, expected, atol=1.0e-12, rtol=0.0))
    if not zero_pass:
        raise RuntimeError("retry16 exact-zero anchor mismatch")
    _write_json(
        stage / "zero_anchor.json",
        {"beta0_rad": np.zeros(6), "xyz0_m": zero_xyz, "exact_zero_pass": zero_pass},
    )
    return _seal_gate(
        output_root,
        config,
        "inventory",
        {
            "status": "complete",
            "upstream_integrity_pass": passed,
            "retry15_operational_completion": bool(summary.get("operational_completion", False)),
            "retry15_proposal_stability_pass": bool(
                summary.get("empirical_outer_denominator", {}).get("proposal_stability_pass", False)
            ),
            "exact_zero_pass": zero_pass,
            "proposal_beta_used_as_label_or_hint": False,
        },
    )


def stage_annular_domain(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["annular_domain"]
    a = pd.read_parquet(_upstream_path(config, "proposal_pool_a_xyz"))
    b = pd.read_parquet(_upstream_path(config, "proposal_pool_b_xyz"))
    final_support = pd.read_parquet(_upstream_path(config, "final_voxel_support"))
    values = config["annular_domain"]
    comparison_rows = 2 ** int(values["comparison_power"])
    comparison_support, _solid_profile_unused = workspace_probe(
        a.loc[:, XYZ_COLUMNS].to_numpy(float)[:comparison_rows],
        b.loc[:, XYZ_COLUMNS].to_numpy(float)[:comparison_rows],
        policy=ProbePolicy(
            axial_step_mm=float(values["axial_step_mm"]),
            radial_step_mm=float(values["radial_step_mm"]),
            full_sector_count=int(values["full_sector_count"]),
            radial_margin_mm=float(values["outer_margin_mm"]),
            slope_limit_mm=float(values["slope_limit_mm"]),
            allow_single_voxel_closing=True,
        ),
    )
    policy = _annular_policy(config)
    raw_expanded, raw_core, expanded, core = build_nested_annular_profiles(
        final_support, comparison_support, policy=policy
    )
    connector = root_connector_voxels(final_support, expanded, policy=policy)
    _write_parquet(comparison_support, stage / "comparison_power18_voxel_support.parquet")
    _write_parquet(raw_expanded, stage / "raw_expanded_annular_profile.parquet")
    _write_parquet(raw_core, stage / "raw_stable_core_annular_profile.parquet")
    _write_parquet(expanded, stage / "expanded_annular_profile.parquet")
    _write_parquet(core, stage / "stable_core_annular_profile.parquet")
    _write_parquet(connector, stage / "root_connector_voxels.parquet")
    proposals = pd.concat([a.loc[:, XYZ_COLUMNS], b.loc[:, XYZ_COLUMNS]], ignore_index=True).to_numpy(float)
    selected = None
    errors: dict[str, str] = {}
    for budget in (int(values["pilot_target_budget"]), int(values["expanded_target_budget"])):
        try:
            selected = build_annular_target_registry(
                expanded,
                core,
                connector,
                proposals,
                target_budget=budget,
                maximum_circle_step_mm=float(values["maximum_circle_step_mm"]),
                spacing_minimum_mm=float(values["target_spacing_minimum_mm"]),
                spacing_maximum_mm=float(values["target_spacing_maximum_mm"]),
                policy=policy,
            )
        except ValueError as exc:
            errors[str(budget)] = str(exc)
            continue
        break
    if selected is None:
        _write_json(stage / "target_selection_errors.json", errors)
        raise RuntimeError("retry16 has no budget-feasible annular target registry")
    targets, edges, circles, summary = selected
    _write_parquet(targets, stage / "selected_target_registry.parquet")
    _write_parquet(edges, stage / "registered_explicit_edges.parquet")
    _write_parquet(circles, stage / "registered_circle_waypoints.parquet")
    circle_registry = (
        circles.sort_values(["circle_id", "phase_index"], kind="stable")
        .drop_duplicates("circle_id")
        .loc[:, ["circle_id", "level_rank", "radial_rank", "u_index", "u_mm", "radius_mm", "diameter_mm", "held_out"]]
        .reset_index(drop=True)
    )
    _write_parquet(circle_registry, stage / "registered_circle_registry.parquet")
    _write_json(
        stage / "domain_summary.json",
        {
            **dict(summary),
            "expanded_annular_volume_mm3": annular_volume_mm3(expanded),
            "stable_core_annular_volume_mm3": annular_volume_mm3(core),
            "proposal_stability_pass": False,
            "expanded_is_diagnostic_only": True,
        },
    )
    pass_gate = bool(
        summary["budget_feasible"]
        and summary["registered_circle_count"] == int(values["required_circle_count"])
        and summary["large_circle_count"] >= 3
        and summary["outside_expanded_annulus_count"] == 0
        and summary["exact_zero_count"] == 1
        and summary["axis_core_target_count"] == 0
        and summary["connector_target_count"] > 0
    )
    if not pass_gate:
        raise RuntimeError("retry16 annular domain integrity failed")
    return _seal_gate(
        output_root,
        config,
        "annular_domain",
        {
            "status": "diagnostic_domain_frozen",
            "annular_domain_pass": True,
            **dict(summary),
            "expanded_annular_volume_mm3": annular_volume_mm3(expanded),
            "stable_core_annular_volume_mm3": annular_volume_mm3(core),
            "proposal_stability_pass": False,
        },
    )


def _registry_ref(path: Path, *, root: Path, id_column: str, row_count: int) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "id_column": id_column,
        "row_count": int(row_count),
    }


def stage_objective_feasibility(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["objective_feasibility"]
    domain_root = output_root / STAGE_DIRS["annular_domain"]
    targets = pd.read_parquet(domain_root / "selected_target_registry.parquet")
    circles = pd.read_parquet(domain_root / "registered_circle_registry.parquet")
    coverage = targets[targets["annular_coverage_eligible"].astype(bool)].copy()
    connector = targets[targets["target_role"].isin(["exact_zero", "root_connector"])].copy()
    _write_parquet(coverage, stage / "annular_coverage_registry.parquet")
    _write_parquet(connector, stage / "root_connector_registry.parquet")
    target_ref = _registry_ref(
        domain_root / "selected_target_registry.parquet",
        root=output_root,
        id_column="target_id",
        row_count=len(targets),
    )
    coverage_ref = _registry_ref(
        stage / "annular_coverage_registry.parquet",
        root=output_root,
        id_column="target_id",
        row_count=len(coverage),
    )
    circle_ref = _registry_ref(
        domain_root / "registered_circle_registry.parquet",
        root=output_root,
        id_column="circle_id",
        row_count=len(circles),
    )
    connector_ref = _registry_ref(
        stage / "root_connector_registry.parquet",
        root=output_root,
        id_column="target_id",
        row_count=len(connector),
    )
    seed_budget = max(map(int, config["candidate_solver"]["saturation_budgets"]))
    solver_attempts = int(2 * len(targets) * seed_budget + 4 * len(targets))
    registered_target_budget = int(_gate(output_root, "annular_domain")["target_budget"])
    resources_feasible = bool(len(targets) <= registered_target_budget)
    source_artifacts = [
        {
            "path": str(_upstream_path(config, key)),
            "sha256": str(config["upstream"][key]["sha256"]),
        }
        for key in ("retry15_artifact_manifest", "proposal_pool_a_xyz", "proposal_pool_b_xyz", "final_voxel_support")
    ]
    documents: dict[str, Mapping[str, Any]] = {
        "objective_contract.json": {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "scope": "coverage_trajectory",
            "scientific_source_sha": _git_sha(),
            "config_sha256": _config_sha(config),
            "diagnostic_pilot_allowed": True,
            "primary_objectives": [
                {"id": "annular_coverage", "kind": "coverage", "metric": "registered_annular_target_acceptance", "required_for_claim": True, "denominator_id": "retry16_annular_targets", "target": {"operator": ">=", "value": 0.8, "unit": "fraction"}},
                {"id": "nine_large_circles", "kind": "complete_trajectory", "metric": "complete_circle_count", "required_for_claim": True, "denominator_id": "retry16_registered_circles", "target": {"operator": ">=", "value": 9, "unit": "count"}},
                {"id": "exact_zero_root_connection", "kind": "complete_trajectory", "metric": "complete_root_connector", "required_for_claim": True, "denominator_id": "retry16_root_connector", "target": {"operator": ">=", "value": 1, "unit": "count"}},
            ],
        },
        "denominator_size.json": {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "frozen_before_launch": True,
            "denominators": [
                {"id": "retry16_annular_targets", "kind": "coverage", "unit": "target", "required_count": len(coverage), "registry": coverage_ref},
                {"id": "retry16_registered_circles", "kind": "complete_trajectory", "unit": "circle", "required_count": len(circles), "registry": circle_ref},
                {"id": "retry16_root_connector", "kind": "complete_trajectory", "unit": "connector_target", "required_count": len(connector), "registry": connector_ref},
            ],
            "resource_denominators": {"required_supervision_vertex_count": len(targets), "required_logical_edge_count": int(len(pd.read_parquet(domain_root / "registered_explicit_edges.parquet"))), "required_second_parent_certification_count": 0},
        },
        "reusable_evidence.json": {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "source_artifacts": source_artifacts,
            "eligible_counts": {"supervision_vertices": 0, "connector_only_vertices": 0, "served_coverage_units": 0, "complete_trajectories": 0, "verified_edges": 0, "second_parent_certifications": 0},
            "ineligible_counts": {"proposal_only": len(coverage), "branch_conflicts": 0, "unused": 0},
            "credit_registry": target_ref,
            "proposal_beta_used_as_label_or_hint": False,
        },
        "budget_lower_bound.json": {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "objective_lower_bounds": [
                {"objective_id": "annular_coverage", "method": "count_credit", "required_units": len(coverage), "target_units": int(math.ceil(0.8 * len(coverage))), "reusable_eligible_units": 0, "maximum_credit_per_new_supervision_vertex": 1, "minimum_resources": {"new_supervision_vertices": int(math.ceil(0.8 * len(coverage)))}, "basis_registry": coverage_ref},
                {"objective_id": "nine_large_circles", "method": "registered_atomic_set_union", "minimum_resources": {"new_supervision_vertices": int(targets["target_role"].eq("heldout_circle").sum())}, "basis_registry": circle_ref},
                {"objective_id": "exact_zero_root_connection", "method": "registered_atomic_set_union", "minimum_resources": {"new_supervision_vertices": len(connector)}, "basis_registry": connector_ref},
            ],
            "resources": {
                "new_supervision_vertices": {"optimistic_minimum": len(targets), "registered_budget": registered_target_budget, "basis_registry": target_ref},
                "solver_attempts": {"optimistic_minimum": solver_attempts, "registered_budget": solver_attempts, "basis_registry": target_ref},
            },
            "all_required_objectives_bounded": True,
            "all_required_resources_feasible": resources_feasible,
        },
        "atomic_objective_schedule.json": {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "row_count_is_stop_condition": False,
            "scheduled_objective_ids": ["annular_coverage", "nine_large_circles", "exact_zero_root_connection"],
            "entries": [
                {"id": "retry16_annular_registry", "objective_id": "annular_coverage", "kind": "coverage", "priority": 0, "required_for_claim": True, "requirement_registry": coverage_ref, "reserved_resources": {"new_supervision_vertices": len(coverage), "solver_attempts": solver_attempts}},
                {"id": "retry16_nine_atomic_circles", "objective_id": "nine_large_circles", "kind": "trajectory", "priority": 0, "required_for_claim": True, "minimum_diameter_mm": 200.0, "full_cycle_required": True, "held_out": True, "requirement_registry": circle_ref, "reserved_resources": {"new_supervision_vertices": int(targets["target_role"].eq("heldout_circle").sum()), "solver_attempts": 0}},
                {"id": "retry16_exact_zero_root_connector", "objective_id": "exact_zero_root_connection", "kind": "trajectory", "priority": 0, "required_for_claim": True, "full_cycle_required": False, "held_out": False, "requirement_registry": connector_ref, "reserved_resources": {"new_supervision_vertices": len(connector), "solver_attempts": 0}},
            ],
        },
        "gate.json": {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "inputs_valid": True,
            "required_objectives_budget_feasible": resources_feasible,
            "status": "feasible" if resources_feasible else "diagnostic_only",
            "claim_bearing_run_authorized": resources_feasible,
            "diagnostic_pilot_authorized": resources_feasible,
            "failed_objective_ids": [] if resources_feasible else ["annular_coverage", "nine_large_circles", "exact_zero_root_connection"],
            "reason_codes": [] if resources_feasible else ["REGISTERED_RESOURCE_BUDGET_BELOW_OPTIMISTIC_MINIMUM"],
        },
    }
    for name, document in documents.items():
        _write_json(stage / name, document)
    # This gate owns budget feasibility only.  A feasible result authorizes the
    # registered computation, while the experiment-level claim remains blocked
    # independently by retry15 proposal-profile instability.
    gate = {
        **dict(documents["gate.json"]),
        "stage": "objective_feasibility",
        "scientific_source_fixed_point": _git_sha(),
        "config_sha256": _config_sha(config),
        "experiment_claim_bearing_run_authorized": False,
        "proposal_stability_blocks_experiment_claim": True,
        "target_count": len(targets),
        "coverage_target_count": len(coverage),
        "registered_circle_count": len(circles),
        "root_connector_target_count": len(connector),
        "solver_attempt_upper_bound": solver_attempts,
        "formal_authorized": False,
        "deployment_authorized": False,
        "full_workspace_authorized": False,
        "continuous_workspace_authorized": False,
        "tension_authorized": False,
    }
    _write_json(stage / "gate.json", gate)
    _seal_stage(output_root, "objective_feasibility")
    return gate


def stage_seed_banks(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["seed_banks"]
    if not _gate(output_root, "objective_feasibility").get("diagnostic_pilot_authorized", False):
        raise RuntimeError("retry16 seed banks are not authorized")
    environment = _environment(config)
    specs = config["teacher_seed_banks"]
    free = {"full": None, "y_seam": (1, 3, 5), "z_seam": (0, 2, 4)}
    manifest: dict[str, Any] = {"schema_version": 1, "proposal_identity_shared": False, "banks": {}}
    for key in ("full", "y_seam", "z_seam"):
        size = 128 if smoke else int(specs[key]["size"])
        frame = teacher_seed_bank(
            np.asarray(environment.bounds),
            size=size,
            seed=int(specs[key]["seed"]),
            bank_id=str(specs[key]["bank_id"]),
            free_indices=free[key],
        )
        with_xyz = seed_bank_xyz(environment, frame)
        path = stage / f"{key}_seed_bank_with_xyz.parquet"
        _write_parquet(with_xyz, path)
        manifest["banks"][key] = {
            "bank_id": str(specs[key]["bank_id"]),
            "seed": int(specs[key]["seed"]),
            "row_count": len(with_xyz),
            "path": path.name,
            "sha256": sha256_file(path),
        }
    _write_json(stage / "seed_bank_manifest.json", manifest)
    return _seal_gate(
        output_root,
        config,
        "seed_banks",
        {
            "status": "complete",
            "seed_banks_frozen": True,
            "proposal_identity_shared": False,
            "full_seed_count": manifest["banks"]["full"]["row_count"],
            "y_seam_seed_count": manifest["banks"]["y_seam"]["row_count"],
            "z_seam_seed_count": manifest["banks"]["z_seam"]["row_count"],
        },
    )


def _run_candidate_workers(
    config: Mapping[str, Any],
    targets: pd.DataFrame,
    *,
    output_root: Path,
    work_root: Path,
    seed_budget: int,
    smoke: bool,
    progress_offset: int,
    progress_total: int,
) -> pd.DataFrame:
    worker_count = min(2 if smoke else int(config["runtime"]["maximum_concurrent_workers"]), len(targets))
    shards = np.array_split(np.arange(len(targets)), worker_count)
    seed_root = output_root / STAGE_DIRS["seed_banks"]
    processes: list[tuple[subprocess.Popen[str], Path]] = []
    environment = _thread_limited_environment()
    for shard_id, indices in enumerate(shards):
        target_path = work_root / f"target_shard_{shard_id:03d}.parquet"
        output_path = work_root / f"candidate_shard_{shard_id:03d}.parquet"
        _write_parquet(targets.iloc[np.asarray(indices, dtype=int)].copy(), target_path)
        command = [
            str(config["runtime"]["python"]),
            str(WORKER),
            "--config",
            str(config["config_path"]),
            "--targets",
            str(target_path),
            "--seed-bank",
            str(seed_root / "full_seed_bank_with_xyz.parquet"),
            "--y-seed-bank",
            str(seed_root / "y_seam_seed_bank_with_xyz.parquet"),
            "--z-seed-bank",
            str(seed_root / "z_seam_seed_bank_with_xyz.parquet"),
            "--output",
            str(output_path),
            "--seed-budget",
            str(int(seed_budget)),
        ]
        processes.append(
            (
                subprocess.Popen(
                    command,
                    cwd=SOURCE_ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ),
                output_path,
            )
        )
    outputs = []
    for completed, (process, output_path) in enumerate(processes, start=1):
        stdout, stderr = process.communicate()
        if process.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"retry16 candidate worker failed ({process.returncode}): {stderr[-4000:]} {stdout[-1000:]}")
        outputs.append(pd.read_parquet(output_path))
        _progress(
            output_root,
            "candidate_preflight",
            completed=progress_offset + completed,
            total=progress_total,
            message=f"candidate shard {completed}/{len(processes)} complete",
        )
    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()


def _prefix_candidates(candidates: pd.DataFrame, budget: int) -> pd.DataFrame:
    return candidates[candidates["seed_rank"].astype(int).lt(int(budget))].copy()


def _saturation_metrics(candidates: pd.DataFrame, targets: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    selections: dict[int, pd.DataFrame] = {}
    weights = config["candidate_solver"]["beta_weights"]
    for budget in map(int, config["candidate_solver"]["saturation_budgets"]):
        prefix = _prefix_candidates(candidates, budget)
        selected = select_t1(prefix, weights=weights)
        selections[budget] = selected
        rows.append(
            {
                "seed_budget": budget,
                "target_count": len(targets),
                "accepted_target_count": int(selected["target_id"].nunique()),
                "acceptance": float(selected["target_id"].nunique() / len(targets)),
                "candidate_attempt_count": len(prefix),
                "legal_cluster_count": len(
                    legal_candidate_clusters(
                        prefix,
                        residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]),
                        cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]),
                        weights=weights,
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    for lower, upper in ((8, 16), (16, 32)):
        left = selections[lower].set_index("target_id")
        right = selections[upper].set_index("target_id")
        common = sorted(set(left.index.astype(str)) & set(right.index.astype(str)))
        if common:
            difference = weighted_beta_rms_deg(
                left.loc[common, BETA_COLUMNS].to_numpy(float),
                right.loc[common, BETA_COLUMNS].to_numpy(float),
                weights,
            )
            p95 = percentile(np.asarray(difference, dtype=float), 95)
        else:
            p95 = math.inf
        frame.loc[frame["seed_budget"].eq(upper), "previous_budget"] = lower
        frame.loc[frame["seed_budget"].eq(upper), "t1_common_difference_p95_deg"] = p95
        lower_acceptance = float(frame.loc[frame["seed_budget"].eq(lower), "acceptance"].iloc[0])
        upper_acceptance = float(frame.loc[frame["seed_budget"].eq(upper), "acceptance"].iloc[0])
        frame.loc[frame["seed_budget"].eq(upper), "acceptance_gain"] = upper_acceptance - lower_acceptance
    return frame


def _choose_seed_budget(metrics: pd.DataFrame, config: Mapping[str, Any], *, smoke: bool) -> int:
    if smoke:
        return 8
    row = metrics.loc[metrics["seed_budget"].eq(32)].iloc[0]
    if (
        float(row["t1_common_difference_p95_deg"])
        <= float(config["candidate_solver"]["saturation_t1_difference_p95_maximum_deg"])
        and float(row["acceptance_gain"])
        < float(config["candidate_solver"]["saturation_acceptance_gain_maximum"])
    ):
        return 16
    return 32


def _accepted_lcc_fraction(labels: pd.DataFrame, edges: pd.DataFrame) -> float:
    ids = sorted(labels["target_id"].astype(str).unique())
    if not ids:
        return 0.0
    if len(ids) == 1:
        return 1.0
    index = {target_id: position for position, target_id in enumerate(ids)}
    rows, columns = [], []
    for edge in edges.to_dict("records"):
        left, right = str(edge["left_target_id"]), str(edge["right_target_id"])
        if left not in index or right not in index:
            continue
        rows.extend([index[left], index[right]])
        columns.extend([index[right], index[left]])
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=(len(ids), len(ids)))
    _count, components = connected_components(graph, directed=False, return_labels=True)
    return float(np.max(np.bincount(components)) / len(ids))


def _teacher_metrics(
    labels: pd.DataFrame,
    targets: pd.DataFrame,
    edges: pd.DataFrame,
    explicit_edges: pd.DataFrame,
    circles: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = set(labels["target_id"].astype(str))
    annular = targets[targets["annular_coverage_eligible"].astype(bool)]
    core = annular[annular["domain_class"].eq("stable_core")]
    expanded_acceptance = float(annular["target_id"].astype(str).isin(accepted).mean()) if len(annular) else 0.0
    core_acceptance = float(core["target_id"].astype(str).isin(accepted).mean()) if len(core) else 0.0
    beta_by_id = {
        str(row["target_id"]): np.asarray([row[name] for name in BETA_COLUMNS], dtype=float)
        for row in labels.to_dict("records")
    }
    weighted, raw = [], []
    for edge in edges.to_dict("records"):
        left, right = str(edge["left_target_id"]), str(edge["right_target_id"])
        if left not in beta_by_id or right not in beta_by_id:
            continue
        delta = beta_by_id[left] - beta_by_id[right]
        weighted.append(float(weighted_beta_rms_deg(beta_by_id[left], beta_by_id[right], config["candidate_solver"]["beta_weights"])))
        raw.append(float(np.max(np.abs(np.rad2deg(delta)))))
    edge_p95 = percentile(np.asarray(weighted), 95) if weighted else math.inf
    raw_rate = float(np.mean(np.asarray(raw) > 7.0)) if raw else 1.0
    complete_circles = 0
    complete_large_circles = 0
    for circle_id, frame in circles.groupby("circle_id", sort=True):
        ids = set(frame["target_id"].astype(str))
        edge_ids = explicit_edges[explicit_edges["circle_id"].astype(str).eq(str(circle_id))]
        edge_complete = all(
            str(row.left_target_id) in accepted and str(row.right_target_id) in accepted
            for row in edge_ids.itertuples(index=False)
        )
        if ids <= accepted and edge_complete:
            complete_circles += 1
            if float(frame["radius_mm"].iloc[0]) >= 100.0:
                complete_large_circles += 1
    root_targets = set(
        targets.loc[targets["target_role"].isin(["exact_zero", "root_connector"]), "target_id"].astype(str)
    )
    root_edges = explicit_edges[explicit_edges["edge_type"].eq("root_connector")]
    root_complete = bool(
        root_targets <= accepted
        and all(
            str(row.left_target_id) in accepted and str(row.right_target_id) in accepted
            for row in root_edges.itertuples(index=False)
        )
    )
    lcc = _accepted_lcc_fraction(labels, edges)
    values = {
        "accepted_target_count": len(accepted),
        "target_count": len(targets),
        "expanded_acceptance": expanded_acceptance,
        "stable_core_acceptance": core_acceptance,
        "accepted_lcc_fraction": lcc,
        "edge_weighted_p95_deg": edge_p95,
        "edge_raw_gt7_rate": raw_rate,
        "complete_circle_count": complete_circles,
        "complete_large_circle_count": complete_large_circles,
        "root_connector_complete": root_complete,
    }
    green = config["preflight_gates"]["green"]
    yellow = config["preflight_gates"]["yellow"]
    values["green"] = bool(
        core_acceptance >= float(green["stable_core_acceptance_minimum"])
        and expanded_acceptance >= float(green["expanded_acceptance_minimum"])
        and lcc >= float(green["accepted_lcc_minimum"])
        and edge_p95 <= float(green["edge_weighted_p95_maximum_deg"])
        and raw_rate <= float(green["edge_raw_gt7_rate_maximum"])
        and complete_circles >= int(green["complete_circle_count_minimum"])
        and (root_complete or not bool(green["root_connector_complete_required"]))
    )
    values["yellow"] = bool(
        core_acceptance >= float(yellow["stable_core_acceptance_minimum"])
        and expanded_acceptance >= float(yellow["expanded_acceptance_minimum"])
        and lcc >= float(yellow["accepted_lcc_minimum"])
        and edge_p95 <= float(yellow["edge_weighted_p95_maximum_deg"])
        and raw_rate <= float(yellow["edge_raw_gt7_rate_maximum"])
        and complete_large_circles >= int(yellow["complete_large_circle_count_minimum"])
        and (root_complete or not bool(yellow["root_connector_complete_required"]))
    )
    values["gate_class"] = "GREEN" if values["green"] else "YELLOW" if values["yellow"] else "RED"
    return values


def stage_candidate_preflight(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["candidate_preflight"]
    domain_root = output_root / STAGE_DIRS["annular_domain"]
    targets = pd.read_parquet(domain_root / "selected_target_registry.parquet")
    explicit_edges = pd.read_parquet(domain_root / "registered_explicit_edges.parquet")
    circles = pd.read_parquet(domain_root / "registered_circle_waypoints.parquet")
    sample_source = targets
    if smoke:
        sample_source = targets.copy()
        # Smoke checks execution plumbing, not the nine-circle scientific
        # panel.  Only the exact root path remains mandatory in this bounded
        # diagnostic sample; the formal run still keeps every circle target.
        sample_source["mandatory"] = sample_source["target_role"].isin(["exact_zero", "root_connector"])
    saturation_limit = min(32 if smoke else int(config["candidate_solver"]["saturation_target_count"]), len(sample_source))
    preflight_limit = min(64 if smoke else int(config["candidate_solver"]["preflight_target_count"]), len(sample_source))
    saturation_targets = deterministic_target_sample(sample_source, limit=saturation_limit, salt="retry16-saturation")
    preflight_targets = deterministic_target_sample(sample_source, limit=preflight_limit, salt="retry16-preflight")
    _write_parquet(saturation_targets, stage / "saturation_targets.parquet")
    _write_parquet(preflight_targets, stage / "preflight_targets.parquet")
    worker_total = (2 if smoke else int(config["runtime"]["maximum_concurrent_workers"])) * 2
    saturation_candidates = _run_candidate_workers(
        config,
        saturation_targets,
        output_root=output_root,
        work_root=stage / "_work" / "saturation",
        seed_budget=8 if smoke else 32,
        smoke=smoke,
        progress_offset=0,
        progress_total=worker_total,
    )
    _write_parquet(saturation_candidates, stage / "saturation_candidate_bank_k32.parquet")
    saturation_metrics = _saturation_metrics(saturation_candidates, saturation_targets, config)
    selected_budget = _choose_seed_budget(saturation_metrics, config, smoke=smoke)
    saturation_metrics["selected_for_preflight"] = saturation_metrics["seed_budget"].eq(selected_budget)
    _write_parquet(saturation_metrics, stage / "candidate_saturation_comparison.parquet")
    saturation_ids = set(saturation_targets["target_id"].astype(str))
    remaining = preflight_targets[~preflight_targets["target_id"].astype(str).isin(saturation_ids)].copy()
    if len(remaining):
        remaining_candidates = _run_candidate_workers(
            config,
            remaining,
            output_root=output_root,
            work_root=stage / "_work" / "remaining",
            seed_budget=selected_budget,
            smoke=smoke,
            progress_offset=2 if smoke else int(config["runtime"]["maximum_concurrent_workers"]),
            progress_total=worker_total,
        )
    else:
        remaining_candidates = pd.DataFrame()
    candidates = pd.concat(
        [_prefix_candidates(saturation_candidates, selected_budget), remaining_candidates],
        ignore_index=True,
        sort=False,
    )
    candidate_ids = set(preflight_targets["target_id"].astype(str))
    candidates = candidates[candidates["target_id"].astype(str).isin(candidate_ids)].copy()
    _write_parquet(candidates, stage / "preflight_candidate_bank.parquet")
    legal = legal_candidate_clusters(
        candidates,
        residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]),
        cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]),
        weights=config["candidate_solver"]["beta_weights"],
    )
    _write_parquet(legal, stage / "preflight_legal_candidate_clusters.parquet")
    preflight_ids = set(preflight_targets["target_id"].astype(str))
    explicit = explicit_edges[
        explicit_edges["left_target_id"].astype(str).isin(preflight_ids)
        & explicit_edges["right_target_id"].astype(str).isin(preflight_ids)
    ].copy()
    spacing = float(_gate(output_root, "annular_domain")["target_spacing_mm"])
    teachers: list[tuple[str, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]] = []
    t1_graph = mutual_knn_edges(
        preflight_targets,
        explicit,
        k=min(map(int, config["graph_teacher"]["k_values"])),
        maximum_distance_mm=float(config["graph_teacher"]["maximum_edge_spacing_factor"]) * spacing,
    )
    t1 = select_t1(candidates, weights=config["candidate_solver"]["beta_weights"]).assign(
        teacher="T1", graph_k=0, pairwise_lambda=0.0
    )
    teachers.append(("T1", t1, t1_graph, _teacher_metrics(t1, preflight_targets, t1_graph, explicit, circles, config)))
    for k in map(int, config["graph_teacher"]["k_values"]):
        graph = mutual_knn_edges(
            preflight_targets,
            explicit,
            k=k,
            maximum_distance_mm=float(config["graph_teacher"]["maximum_edge_spacing_factor"]) * spacing,
        )
        for pairwise_lambda in map(float, config["graph_teacher"]["pairwise_lambdas"]):
            if pairwise_lambda == 0.0:
                continue
            labels = select_t2(
                candidates,
                graph,
                pairwise_lambda=pairwise_lambda,
                tau_deg=float(config["graph_teacher"]["robust_pairwise_cap_deg"]),
                weights=config["candidate_solver"]["beta_weights"],
                maximum_sweeps=int(config["graph_teacher"]["maximum_icm_sweeps"]),
            ).assign(graph_k=k)
            name = f"T2_k{k}_lambda{pairwise_lambda:g}"
            teachers.append((name, labels, graph, _teacher_metrics(labels, preflight_targets, graph, explicit, circles, config)))
    selected_name, selected_labels, selected_graph, selected_metrics = max(
        teachers,
        key=lambda item: (
            {"GREEN": 2, "YELLOW": 1, "RED": 0}[str(item[3]["gate_class"])],
            float(item[3]["stable_core_acceptance"]),
            float(item[3]["expanded_acceptance"]),
            float(item[3]["accepted_lcc_fraction"]),
            -float(item[3]["edge_raw_gt7_rate"]),
            -float(item[3]["edge_weighted_p95_deg"]),
            item[0],
        ),
    )
    _write_parquet(selected_labels, stage / "selected_teacher_labels.parquet")
    _write_parquet(selected_graph, stage / "selected_target_graph.parquet")
    _write_parquet(
        pd.DataFrame([{"teacher_name": name, **dict(metrics)} for name, _labels, _graph, metrics in teachers]),
        stage / "teacher_comparison.parquet",
    )
    _write_json(stage / "selected_teacher.json", {"selected_teacher": selected_name, **dict(selected_metrics)})
    return _seal_gate(
        output_root,
        config,
        "candidate_preflight",
        {
            "status": f"diagnostic_{str(selected_metrics['gate_class']).lower()}",
            "candidate_preflight_complete": True,
            "selected_seed_budget": selected_budget,
            "saturation_target_count": len(saturation_targets),
            "preflight_target_count": len(preflight_targets),
            "candidate_attempt_count": len(candidates),
            "legal_candidate_cluster_count": len(legal),
            "selected_teacher": selected_name,
            "selected_metrics": dict(selected_metrics),
            "proposal_beta_used_as_label_or_hint": False,
        },
    )


def _audit_expanded_full_circles(
    expanded: pd.DataFrame,
    circle_waypoints: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit the actual G4-expanded full cycles, including the closing edge."""

    weights = config["candidate_solver"]["beta_weights"]
    residual_maximum_mm = float(config["candidate_solver"]["residual_maximum_mm"])
    workspace_step_maximum_mm = float(config["annular_domain"]["maximum_circle_step_mm"])
    required_circle_count = int(config["annular_domain"]["required_circle_count"])
    large_radius_mm = float(config["annular_domain"]["required_large_circle_radius_mm"])
    edge_weighted_p95_maximum_deg = float(
        config["preflight_gates"]["green"]["edge_weighted_p95_maximum_deg"]
    )
    edge_raw_gt7_rate_maximum = float(
        config["preflight_gates"]["green"]["edge_raw_gt7_rate_maximum"]
    )
    audit_rows: list[dict[str, Any]] = []
    all_weighted_edges: list[float] = []
    all_raw_edges: list[float] = []

    for circle_id, registered in circle_waypoints.groupby("circle_id", sort=True):
        selected = expanded.loc[expanded["circle_id"].fillna("").astype(str).eq(str(circle_id))].copy()
        expected_rows = int(registered["full_segment_count"].iloc[0])
        radius_mm = float(registered["radius_mm"].iloc[0])
        actual_rows = len(selected)
        unique_xyz_rows = len(selected.drop_duplicates(list(XYZ_COLUMNS)))

        if actual_rows:
            phase = np.mod(
                np.arctan2(selected["z_m"].to_numpy(float), selected["y_m"].to_numpy(float)),
                2.0 * np.pi,
            )
            selected = selected.assign(_full_cycle_phase=phase).sort_values(
                "_full_cycle_phase", kind="stable"
            )
            beta = selected.loc[:, list(BETA_COLUMNS)].to_numpy(float)
            xyz_mm = 1000.0 * selected.loc[:, list(XYZ_COLUMNS)].to_numpy(float)
            next_beta = np.roll(beta, -1, axis=0)
            next_xyz_mm = np.roll(xyz_mm, -1, axis=0)
            weighted_edges = np.asarray(
                [weighted_beta_rms_deg(left, right, weights) for left, right in zip(beta, next_beta, strict=True)],
                dtype=float,
            )
            raw_edges = np.max(np.abs(np.rad2deg(next_beta - beta)), axis=1)
            workspace_edges_mm = np.linalg.norm(next_xyz_mm - xyz_mm, axis=1)
            fk_residual_maximum_mm = float(selected["fk_residual_mm"].max())
            weighted_p95_deg = percentile(weighted_edges, 95)
            weighted_maximum_deg = float(np.max(weighted_edges))
            raw_gt7_rate = float(np.mean(raw_edges > 7.0))
            raw_maximum_deg = float(np.max(raw_edges))
            workspace_step_maximum_observed_mm = float(np.max(workspace_edges_mm))
            closing_workspace_step_mm = float(workspace_edges_mm[-1])
            closing_weighted_deg = float(weighted_edges[-1])
            closing_raw_deg = float(raw_edges[-1])
            all_weighted_edges.extend(weighted_edges.tolist())
            all_raw_edges.extend(raw_edges.tolist())
        else:
            fk_residual_maximum_mm = math.inf
            weighted_p95_deg = math.inf
            weighted_maximum_deg = math.inf
            raw_gt7_rate = 1.0
            raw_maximum_deg = math.inf
            workspace_step_maximum_observed_mm = math.inf
            closing_workspace_step_mm = math.inf
            closing_weighted_deg = math.inf
            closing_raw_deg = math.inf

        complete = bool(
            actual_rows == expected_rows
            and unique_xyz_rows == expected_rows
            and fk_residual_maximum_mm <= residual_maximum_mm
            and workspace_step_maximum_observed_mm <= workspace_step_maximum_mm + 1e-6
            and weighted_p95_deg <= edge_weighted_p95_maximum_deg
            and raw_gt7_rate <= edge_raw_gt7_rate_maximum
        )
        audit_rows.append(
            {
                "circle_id": str(circle_id),
                "radius_mm": radius_mm,
                "expected_full_cycle_rows": expected_rows,
                "actual_full_cycle_rows": actual_rows,
                "unique_xyz_rows": unique_xyz_rows,
                "maximum_fk_residual_mm": fk_residual_maximum_mm,
                "maximum_workspace_step_mm": workspace_step_maximum_observed_mm,
                "closing_workspace_step_mm": closing_workspace_step_mm,
                "edge_weighted_p95_deg": weighted_p95_deg,
                "edge_weighted_maximum_deg": weighted_maximum_deg,
                "closing_edge_weighted_deg": closing_weighted_deg,
                "edge_raw_gt7_rate": raw_gt7_rate,
                "edge_raw_maximum_deg": raw_maximum_deg,
                "closing_edge_raw_deg": closing_raw_deg,
                "full_cycle_complete": complete,
            }
        )

    audit = pd.DataFrame(audit_rows)
    complete_count = int(audit["full_cycle_complete"].sum()) if len(audit) else 0
    complete_large_count = int(
        (audit["full_cycle_complete"] & audit["radius_mm"].ge(large_radius_mm)).sum()
    ) if len(audit) else 0
    expected_row_count = int(audit["expected_full_cycle_rows"].sum()) if len(audit) else 0
    actual_row_count = int(audit["actual_full_cycle_rows"].sum()) if len(audit) else 0
    summary = {
        "full_cycle_pass": bool(complete_count == required_circle_count == len(audit)),
        "complete_full_circle_count": complete_count,
        "complete_large_full_circle_count": complete_large_count,
        "full_circle_expected_row_count": expected_row_count,
        "full_circle_actual_row_count": actual_row_count,
        "full_circle_edge_weighted_p95_deg": percentile(np.asarray(all_weighted_edges), 95)
        if all_weighted_edges
        else math.inf,
        "full_circle_edge_raw_gt7_rate": float(np.mean(np.asarray(all_raw_edges) > 7.0))
        if all_raw_edges
        else 1.0,
        "full_circle_edge_raw_max_deg": float(np.max(all_raw_edges)) if all_raw_edges else math.inf,
        "full_circle_maximum_fk_residual_mm": float(audit["maximum_fk_residual_mm"].max())
        if len(audit)
        else math.inf,
        "full_circle_maximum_workspace_step_mm": float(audit["maximum_workspace_step_mm"].max())
        if len(audit)
        else math.inf,
    }
    return audit, summary


def stage_dataset(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["dataset"]
    domain_root = output_root / STAGE_DIRS["annular_domain"]
    candidate_root = output_root / STAGE_DIRS["candidate_preflight"]
    targets = pd.read_parquet(domain_root / "selected_target_registry.parquet")
    circles = pd.read_parquet(domain_root / "registered_circle_waypoints.parquet")
    labels = pd.read_parquet(candidate_root / "selected_teacher_labels.parquet")
    environment = _environment(config)
    expanded, rejected = expand_annular_labels(
        labels,
        targets,
        environment,
        residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]),
    )
    _write_parquet(labels, stage / "quotient_teacher_labels.parquet")
    _write_parquet(expanded, stage / "retry16_annular_diagnostic_kinematic_dataset.parquet")
    _write_parquet(rejected, stage / "symmetry_expansion_rejections.parquet")
    circle_audit, circle_summary = _audit_expanded_full_circles(expanded, circles, config)
    _write_parquet(circle_audit, stage / "expanded_full_circle_audit.parquet")
    full_circle_required = not smoke
    hard_pass = bool(
        len(expanded)
        and rejected.empty
        and not expanded["proposal_beta_used"].astype(bool).any()
        and not expanded.duplicated(list(XYZ_COLUMNS)).any()
        and np.isfinite(expanded.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy(float)).all()
        and int(expanded["target_role"].eq("exact_zero").sum()) == 1
        and (not full_circle_required or circle_summary["full_cycle_pass"])
    )
    if not hard_pass:
        raise RuntimeError("retry16 diagnostic dataset hard integrity failed")
    metrics = _gate(output_root, "candidate_preflight")["selected_metrics"]
    return _seal_gate(
        output_root,
        config,
        "dataset",
        {
            "status": "complete",
            "dataset_integrity_pass": hard_pass,
            "quotient_label_count": len(labels),
            "expanded_dataset_row_count": len(expanded),
            "symmetry_rejection_count": len(rejected),
            "teacher_gate_class": metrics["gate_class"],
            "full_circle_audit_required": full_circle_required,
            **circle_summary,
            "proposal_beta_used_as_label_or_hint": False,
            "theta_mapping": "beta_to_theta_without_extra_sign",
        },
    )


def _artifact_manifest(output_root: Path) -> Mapping[str, Any]:
    artifacts = []
    for stage_name in STAGE_ORDER[:-1]:
        stage = output_root / STAGE_DIRS[stage_name]
        for path in sorted(stage.rglob("*")):
            if path.is_file() and "_work" not in path.parts:
                artifacts.append(
                    {"path": str(path.relative_to(output_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                )
    return {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "scientific_source_fixed_point": _git_sha(), "artifacts": artifacts}


def stage_summary(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    inventory = _gate(output_root, "inventory")
    domain = _gate(output_root, "annular_domain")
    feasibility = _gate(output_root, "objective_feasibility")
    candidate = _gate(output_root, "candidate_preflight")
    dataset = _gate(output_root, "dataset")
    operational = all(_stage_is_complete(output_root, name) for name in STAGE_ORDER[:-1])
    metrics = candidate["selected_metrics"]
    gate = {
        "status": "complete" if operational else "incomplete",
        "operational_completion": operational,
        "artifact_completeness": operational,
        "diagnostic_smoke": bool(smoke),
        "scientific_gate": f"diagnostic_{str(metrics['gate_class']).lower()}",
        "upstream_proposal_stability_pass": inventory["retry15_proposal_stability_pass"],
        "annular_domain": {
            "target_count": domain["target_count"],
            "coverage_target_count": domain["coverage_target_count"],
            "stable_core_target_count": domain["stable_core_target_count"],
            "target_spacing_mm": domain["target_spacing_mm"],
            "expanded_u_range_mm": [domain["expanded_first_u_mm"], domain["expanded_last_u_mm"]],
            "stable_core_u_range_mm": [domain["stable_core_first_u_mm"], domain["stable_core_last_u_mm"]],
            "registered_circle_count": domain["registered_circle_count"],
            "large_circle_count": domain["large_circle_count"],
            "circle_radius_range_mm": [domain["minimum_circle_radius_mm"], domain["maximum_circle_radius_mm"]],
            "connector_target_count": domain["connector_target_count"],
            "axis_core_target_count": domain["axis_core_target_count"],
        },
        "objective_feasibility": {
            "diagnostic_pilot_authorized": feasibility["diagnostic_pilot_authorized"],
            "proposal_stability_blocks_claim_bearing": feasibility["proposal_stability_blocks_experiment_claim"],
        },
        "teacher_preflight": {
            "selected_seed_budget": candidate["selected_seed_budget"],
            "selected_teacher": candidate["selected_teacher"],
            **dict(metrics),
        },
        "dataset": {
            "quotient_label_count": dataset["quotient_label_count"],
            "expanded_dataset_row_count": dataset["expanded_dataset_row_count"],
            "integrity_pass": dataset["dataset_integrity_pass"],
            "full_circle_audit_required": dataset["full_circle_audit_required"],
            "full_cycle_pass": dataset["full_cycle_pass"],
            "complete_full_circle_count": dataset["complete_full_circle_count"],
            "complete_large_full_circle_count": dataset["complete_large_full_circle_count"],
            "full_circle_expected_row_count": dataset["full_circle_expected_row_count"],
            "full_circle_actual_row_count": dataset["full_circle_actual_row_count"],
            "full_circle_edge_weighted_p95_deg": dataset["full_circle_edge_weighted_p95_deg"],
            "full_circle_edge_raw_gt7_rate": dataset["full_circle_edge_raw_gt7_rate"],
            "full_circle_edge_raw_max_deg": dataset["full_circle_edge_raw_max_deg"],
            "full_circle_maximum_fk_residual_mm": dataset["full_circle_maximum_fk_residual_mm"],
            "full_circle_maximum_workspace_step_mm": dataset["full_circle_maximum_workspace_step_mm"],
        },
        "proposal_beta_used_as_label_or_hint": False,
        "downstream_authorization": False,
        "formal_authorization": False,
        "deployment_authorization": False,
        "full_workspace_authorization": False,
        "continuous_workspace_authorization": False,
        "tension_executed": False,
    }
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "retry16_annular_preflight_summary.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>retry16 annular preflight</title></head>"
        f"<body><h1>retry16 Omega600 annular/tube preflight</h1><pre>{json.dumps(gate, ensure_ascii=False, indent=2)}</pre></body></html>",
        encoding="utf-8",
    )
    sealed = _seal_gate(output_root, config, "summary", gate)
    _write_json(stage / "artifact_manifest.json", _artifact_manifest(output_root))
    _seal_stage(output_root, "summary")
    _write_json(
        output_root / "progress.json",
        {"status": "complete", "phase": "summary", "completed": len(STAGE_ORDER), "total": len(STAGE_ORDER), "message": "retry16 annular preflight completed", "observed_at_unix": time.time()},
    )
    return sealed


STAGE_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "inventory": stage_inventory,
    "annular_domain": stage_annular_domain,
    "objective_feasibility": stage_objective_feasibility,
    "seed_banks": stage_seed_banks,
    "candidate_preflight": stage_candidate_preflight,
    "dataset": stage_dataset,
    "summary": stage_summary,
}


def run(
    config_path: str | Path,
    output_root: str | Path,
    binding_sha: str,
    *,
    stage_name: str | None = None,
    validate_stage: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_root).resolve()
    _ensure_identity(config, output, binding_sha, smoke=smoke)
    if validate_stage is not None:
        if validate_stage not in STAGE_RUNNERS:
            raise ValueError(f"unknown retry16 stage {validate_stage}")
        if not _stage_is_complete(output, validate_stage):
            raise RuntimeError(f"retry16 stage is incomplete or mutated: {validate_stage}")
        return _gate(output, validate_stage)
    if stage_name is not None:
        if stage_name not in STAGE_RUNNERS:
            raise ValueError(f"unknown retry16 stage {stage_name}")
        position = STAGE_ORDER.index(stage_name)
        if position and not _stage_is_complete(output, STAGE_ORDER[position - 1]):
            raise RuntimeError("retry16 previous stage is incomplete")
        if _stage_is_complete(output, stage_name):
            return _gate(output, stage_name)
        return STAGE_RUNNERS[stage_name](config, output, smoke=smoke)
    result: dict[str, Any] = {}
    for index, name in enumerate(STAGE_ORDER):
        _progress(output, name, completed=index, total=len(STAGE_ORDER))
        result = _gate(output, name) if _stage_is_complete(output, name) else STAGE_RUNNERS[name](config, output, smoke=smoke)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--binding-sha", required=True)
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--validate-stage", choices=STAGE_ORDER)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.config,
        args.output_root,
        args.binding_sha,
        stage_name=args.stage,
        validate_stage=args.validate_stage,
        smoke=bool(args.smoke),
    )
    print(json.dumps(result, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
