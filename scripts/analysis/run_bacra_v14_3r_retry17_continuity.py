#!/usr/bin/env python3
"""Run retry17 continuous annular filling and post-lock trajectory diagnostics."""

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
from typing import Any, Callable, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import percentile, weighted_beta_rms_deg
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, THETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry15_canonical_graph import legal_candidate_clusters
from quasi_exp.teacher.retry16_annular_tube import expand_annular_labels
from quasi_exp.teacher.retry17_continuity_fill import (
    ContinuityPolicy,
    continuous_annular_sobol,
    coverage_distance,
    density_normalized_select_t2,
    geometric_graph,
    graph_lcc_fraction,
    graph_stretch,
    local_directional_coverage,
    points_inside_profile,
    progressive_farthest_fill,
    teacher_edge_metrics,
)
from quasi_exp.teacher.retry17_trajectories import (
    frozen_shape_objective_registry,
    generate_maximal_shape_registry,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher import trajectory_evaluation
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    save_workspace_student_models,
    train_workspace_student,
)


EXPERIMENT_ID = "bacra_v14_3r_retry17_omega600_annular_continuity"
STAGE_DIRS = {
    "objective_feasibility": "00_objective_feasibility",
    "retry16_baseline": "00_retry16_baseline",
    "target_pool": "01_target_pool",
    "phase_a_targets": "02_phase_a_targets",
    "phase_a_candidate_bank": "03_phase_a_candidate_bank",
    "phase_a_graph_teacher": "04_phase_a_graph_teacher",
    "phase_a_adaptive_fill": "05_phase_a_adaptive_fill",
    "phase_a_freeze_audit": "06_phase_a_freeze_audit",
    "phase_b_continuation": "07_phase_b_continuation",
    "student": "08_student",
    "dataset_student_lock": "09_dataset_student_lock",
    "heldout_shapes": "10_heldout_shapes",
    "trajectory_evaluation": "11_trajectory_evaluation",
    "summary": "12_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)
WORKER = SOURCE_ROOT / "scripts" / "analysis" / "run_bacra_v14_3r_retry15_worker.py"


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    return resolved.parent.parent if resolved.parent.name == ".worktrees" else resolved


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False); temporary.replace(path)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()


def _config_sha(config: Mapping[str, Any]) -> str:
    return sha256_file(Path(str(config["config_path"])))


def _environment(config: Mapping[str, Any]) -> Any:
    return optimized_forward(load_environment(project_root_from(SOURCE_ROOT), SOURCE_ROOT / str(config["sources"]["robot_config"])))


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    spec = config["upstream"][key]
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else Path(str(config["upstream"]["retry16_root"])) / path


def _continuity_policy(config: Mapping[str, Any]) -> ContinuityPolicy:
    sampling, gates, graph = config["sampling"], config["continuity"], config["graph_teacher"]
    return ContinuityPolicy(
        phase_a_fill_mm=float(gates["phase_a_fill_p95_maximum_mm"]),
        phase_b_fill_mm=float(gates["phase_b_fill_p95_maximum_mm"]),
        minimum_separation_mm=float(sampling["minimum_separation_mm"]),
        batch_size=int(sampling["batch_size"]), geometric_k=max(map(int, graph["k_values"])),
        maximum_edge_spacing_factor=float(graph["maximum_edge_spacing_factor"]),
        direction_projection_mm=float(gates["direction_projection_minimum_mm"]),
        direction_cosine_minimum=float(gates["direction_cosine_minimum"]),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); config["config_path"] = str(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID: raise ValueError("retry17 experiment_id mismatch")
    if int(config["runtime"]["maximum_concurrent_workers"]) != 12 or int(config["runtime"]["numerical_threads_per_worker"]) != 1:
        raise ValueError("retry17 requires twelve single-threaded numerical workers")
    if tuple(map(int, config["graph_teacher"]["k_values"])) != (12, 16) or tuple(map(float, config["graph_teacher"]["pairwise_lambdas"])) != (2.0, 4.0, 8.0):
        raise ValueError("retry17 Teacher sweep identity mismatch")
    if tuple(int(config["sampling"][key]) for key in ("evaluation_probe_seed", "target_pool_seed", "trajectory_center_seed", "graph_stretch_seed", "spatial_split_seed", "student_seed")) != tuple(range(20260910, 20260916)):
        raise ValueError("retry17 independent seed registry mismatch")
    _continuity_policy(config)
    claims = config["claims"]
    if not claims["diagnostic_only"] or any(bool(claims[key]) for key in claims if key != "diagnostic_only"):
        raise ValueError("retry17 must remain diagnostic-only and claim-bearing false")
    if sha256_file(SOURCE_ROOT / str(config["sources"]["robot_config"])) != str(config["sources"]["robot_config_sha256"]):
        raise ValueError("retry17 robot config SHA mismatch")
    return config


def _binding_definition(binding_sha: str) -> Mapping[str, Any]:
    raw = subprocess.check_output(["git", "show", f"{binding_sha}:spec/registry.yaml"], cwd=SOURCE_ROOT, text=True)
    return yaml.safe_load(raw)["experiments"][EXPERIMENT_ID]


def _ensure_identity(config: Mapping[str, Any], output_root: Path, binding_sha: str, *, smoke: bool) -> None:
    identity = {"experiment_id": EXPERIMENT_ID, "scientific_source_fixed_point": _git_sha(), "binding_fixed_point": str(binding_sha), "config_sha256": _config_sha(config), "diagnostic_smoke": bool(smoke)}
    definition = _binding_definition(binding_sha)
    expected = {"scientific_source_fixed_point": _git_sha(), "config": str(Path(config["config_path"]).relative_to(SOURCE_ROOT)), "runner": str(Path(__file__).resolve().relative_to(SOURCE_ROOT))}
    for key, value in expected.items():
        if str(definition.get(key)) != str(value): raise RuntimeError(f"retry17 binding mismatch for {key}: {definition.get(key)!r} != {value!r}")
    path = output_root / "run_identity.json"
    if path.exists() and _read_json(path) != identity: raise RuntimeError("retry17 output root identity mismatch")
    if not path.exists(): output_root.mkdir(parents=True, exist_ok=False); _write_json(path, identity)


def _stage_manifest(output_root: Path, stage_name: str) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = [{"path": str(path.relative_to(stage)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "completion_manifest.json" and "_work" not in path.parts]
    upstream = []
    position = STAGE_ORDER.index(stage_name)
    if position:
        previous = output_root / STAGE_DIRS[STAGE_ORDER[position - 1]] / "completion_manifest.json"
        upstream.append({"stage": STAGE_ORDER[position - 1], "sha256": sha256_file(previous)})
    return {"schema_version": 1, "stage": stage_name, "scientific_source_fixed_point": _git_sha(), "artifacts": artifacts, "upstream": upstream}


def _seal_stage(output_root: Path, stage_name: str) -> None:
    _write_json(output_root / STAGE_DIRS[stage_name] / "completion_manifest.json", _stage_manifest(output_root, stage_name))


def _seal_gate(output_root: Path, config: Mapping[str, Any], stage_name: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    value = {**dict(gate), "stage": stage_name, "experiment_id": EXPERIMENT_ID, "scientific_source_fixed_point": _git_sha(), "config_sha256": _config_sha(config), **config["claims"]}
    _write_json(output_root / STAGE_DIRS[stage_name] / "gate.json", value); _seal_stage(output_root, stage_name); return value


def _gate(output_root: Path, stage_name: str) -> dict[str, Any]: return _read_json(output_root / STAGE_DIRS[stage_name] / "gate.json")


def _stage_is_complete(output_root: Path, stage_name: str) -> bool:
    path = output_root / STAGE_DIRS[stage_name] / "completion_manifest.json"
    return path.exists() and _read_json(path) == _stage_manifest(output_root, stage_name)


def _progress(output_root: Path, stage_name: str, *, completed: int | None = None, total: int | None = None, message: str = "") -> None:
    _write_json(output_root / "progress.json", {"status": "running", "phase": stage_name, "completed": completed, "total": total, "message": message, "observed_at_unix": time.time()})


def _input_integrity(config: Mapping[str, Any]) -> tuple[pd.DataFrame, bool]:
    rows = []
    for key, spec in config["upstream"].items():
        if not isinstance(spec, Mapping) or "sha256" not in spec: continue
        path = _upstream_path(config, key); observed = sha256_file(path) if path.exists() else None
        rows.append({"source_key": key, "path": str(path), "expected_sha256": str(spec["sha256"]), "observed_sha256": observed, "exists": path.exists(), "match": bool(path.exists() and observed == str(spec["sha256"]))})
    frame = pd.DataFrame(rows); return frame, bool(len(frame) and frame["match"].all())


def _old_quotient(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.read_parquet(_upstream_path(config, "target_registry")); labels = pd.read_parquet(_upstream_path(config, "quotient_labels"))
    merged = labels.drop(columns=[name for name in XYZ_COLUMNS if name in labels], errors="ignore").merge(targets, on="target_id", how="inner", validate="one_to_one")
    merged["legacy_in_sample_circle"] = merged["target_role"].eq("heldout_circle")
    merged["retry17_reused_locked"] = True
    return targets, merged


def _objective_documents(config: Mapping[str, Any], output_root: Path, probe: pd.DataFrame, slots: pd.DataFrame) -> None:
    stage = output_root / STAGE_DIRS["objective_feasibility"]
    probe_ref = {"path": str((stage / "evaluation_probe_registry.parquet").relative_to(output_root)), "sha256": sha256_file(stage / "evaluation_probe_registry.parquet"), "id_column": "target_id", "row_count": len(probe)}
    slot_ref = {"path": str((stage / "shape_objective_registry.parquet").relative_to(output_root)), "sha256": sha256_file(stage / "shape_objective_registry.parquet"), "id_column": "slot_id", "row_count": len(slots)}
    cap = int(config["sampling"]["phase_b_quotient_cap"]); attempts = cap * (2 * int(config["candidate_solver"]["difficult_seed_budget"]) + 4)
    docs = {
        "objective_contract.json": {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "scope": "coverage_trajectory", "scientific_source_sha": _git_sha(), "config_sha256": _config_sha(config), "diagnostic_pilot_allowed": True, "primary_objectives": [{"id": "continuous_annular_fill", "kind": "coverage", "metric": "independent_probe_fill_p95_mm", "required_for_claim": True, "denominator_id": "retry17_evaluation_probes", "target": {"operator": "<=", "value": 25.0, "unit": "mm"}}, {"id": "heldout_maximal_shapes", "kind": "complete_trajectory", "metric": "green_shape_classes", "required_for_claim": True, "denominator_id": "retry17_shape_slots", "target": {"operator": ">=", "value": 3, "unit": "class"}}]},
        "denominator_size.json": {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "frozen_before_launch": True, "denominators": [{"id": "retry17_evaluation_probes", "kind": "coverage", "unit": "probe", "required_count": len(probe), "registry": probe_ref}, {"id": "retry17_shape_slots", "kind": "complete_trajectory", "unit": "slot", "required_count": len(slots), "registry": slot_ref}], "resource_denominators": {"required_supervision_vertex_count": cap, "required_logical_edge_count": 0, "required_second_parent_certification_count": 0}},
        "reusable_evidence.json": {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "source_artifacts": [{"path": str(_upstream_path(config, key)), "sha256": str(config["upstream"][key]["sha256"])} for key in ("artifact_manifest", "stable_core_profile", "quotient_labels")], "eligible_counts": {"supervision_vertices": 1999, "connector_only_vertices": 0, "served_coverage_units": 0, "complete_trajectories": 0, "verified_edges": 0, "second_parent_certifications": 0}, "ineligible_counts": {"proposal_only": 0, "branch_conflicts": 0, "unused": 0}, "credit_registry": probe_ref, "proposal_beta_used_as_label_or_hint": False},
        "budget_lower_bound.json": {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "objective_lower_bounds": [{"objective_id": "continuous_annular_fill", "method": "not_bounded", "reason": "labelability of progressive volume targets is empirical", "basis_registry": probe_ref}, {"objective_id": "heldout_maximal_shapes", "method": "not_bounded", "reason": "post-lock maximal feasible scale is empirical", "basis_registry": slot_ref}], "resources": {"new_supervision_vertices": {"optimistic_minimum": 0, "registered_budget": cap, "basis_registry": probe_ref}, "solver_attempts": {"optimistic_minimum": 0, "registered_budget": attempts, "basis_registry": probe_ref}}, "all_required_objectives_bounded": False, "all_required_resources_feasible": True},
        "budget_schedule.json": {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "row_count_is_stop_condition": False, "scheduled_objective_ids": ["continuous_annular_fill", "heldout_maximal_shapes"], "entries": [{"id": "phase_a", "priority": 0, "registered_supervision_cap": int(config["sampling"]["phase_a_quotient_cap"]), "stop_metric": "fill_p95_mm"}, {"id": "phase_b", "priority": 1, "registered_supervision_cap": cap, "conditional": True}]},
        "feasibility_gate.json": {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "diagnostic_only", "claim_bearing_run_authorized": False, "diagnostic_pilot_authorized": True, "all_required_objectives_bounded": False, "all_required_resources_feasible": True, "service_radius_lower_bound_status": "not_bounded", "trajectory_lower_bound_status": "not_bounded"},
    }
    for name, value in docs.items(): _write_json(stage / name, value)


def stage_objective_feasibility(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["objective_feasibility"]
    integrity, passed = _input_integrity(config); _write_parquet(integrity, stage / "upstream_verification.parquet")
    if not passed: raise RuntimeError("retry17 upstream integrity failed")
    environment = _environment(config); zero = np.asarray(environment.fk(np.zeros(6))).reshape(3)
    expected = np.asarray([float(config["omega600"]["zero_x_m"]), 0.0, 0.0])
    if not np.allclose(zero, expected, atol=1e-12, rtol=0): raise RuntimeError("retry17 exact-zero anchor mismatch")
    profile = pd.read_parquet(_upstream_path(config, "stable_core_profile")); power = 10 if smoke else int(config["sampling"]["evaluation_probe_power"])
    probe = continuous_annular_sobol(profile, power=power, seed=int(config["sampling"]["evaluation_probe_seed"]), zero_x_m=expected[0], pool_id="retry17_independent_evaluation")
    slots = frozen_shape_objective_registry(); _write_parquet(probe, stage / "evaluation_probe_registry.parquet"); _write_parquet(slots, stage / "shape_objective_registry.parquet")
    _objective_documents(config, output_root, probe, slots)
    return _seal_gate(output_root, config, "objective_feasibility", {"status": "diagnostic_only", "upstream_integrity_pass": True, "exact_zero_pass": True, "evaluation_probe_count": len(probe), "shape_objective_slot_count": len(slots), "diagnostic_pilot_authorized": True, "objective_lower_bounds_complete": False, "proposal_beta_used_as_label_or_hint": False})


def _continuity_audit(targets: pd.DataFrame, labels: pd.DataFrame, profile: pd.DataFrame, probes: pd.DataFrame, config: Mapping[str, Any], *, h_mm: float, k: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = targets.get("annular_coverage_eligible", pd.Series(True, index=targets.index)).fillna(False).astype(bool)
    accepted = targets[eligible & targets["target_id"].isin(labels["target_id"])].copy().drop_duplicates("target_id")
    edges = geometric_graph(accepted, k=k, maximum_distance_mm=float(config["graph_teacher"]["maximum_edge_spacing_factor"]) * h_mm)
    fill = coverage_distance(probes, accepted); lcc = graph_lcc_fraction(accepted, edges)
    direction, direction_audit = local_directional_coverage(accepted, profile, zero_x_m=float(config["omega600"]["zero_x_m"]), neighbourhood_radius_mm=2.0 * h_mm, projection_minimum_mm=float(config["continuity"]["direction_projection_minimum_mm"]), cosine_minimum=float(config["continuity"]["direction_cosine_minimum"]))
    stretch, stretch_audit = graph_stretch(accepted, edges, pair_count=int(config["continuity"]["graph_stretch_pair_count"]), seed=int(config["sampling"]["graph_stretch_seed"]))
    beta = teacher_edge_metrics(labels, edges, weights=config["candidate_solver"]["beta_weights"])
    metrics = {"accepted_target_count": len(accepted), "fill": fill, "lcc_fraction": lcc, "local_directional_coverage": direction, "graph_stretch": stretch, "label_continuity": beta, "graph_k": k, "h_mm": h_mm}
    return metrics, edges, direction_audit, stretch_audit


def _green(metrics: Mapping[str, Any], config: Mapping[str, Any], *, phase: str) -> bool:
    gate = config["continuity"]; fill_limit = float(gate[f"phase_{phase.lower()}_fill_p95_maximum_mm"])
    return bool(metrics["fill"]["p95_mm"] <= fill_limit and metrics["lcc_fraction"] >= float(gate["lcc_minimum"]) and metrics["local_directional_coverage"] >= float(gate["local_directional_coverage_minimum"]) and metrics["graph_stretch"]["p95"] <= float(gate["graph_stretch_p95_maximum"]) and metrics["label_continuity"]["weighted_p95_deg"] <= float(gate["edge_weighted_p95_maximum_deg"]) and metrics["label_continuity"]["raw_gt7_rate"] <= float(gate["edge_raw_gt7_rate_maximum"]))


def stage_retry16_baseline(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["retry16_baseline"]; profile = pd.read_parquet(_upstream_path(config, "stable_core_profile")); probes = pd.read_parquet(output_root / STAGE_DIRS["objective_feasibility"] / "evaluation_probe_registry.parquet")
    targets, merged = _old_quotient(config); metrics, edges, direction, stretch = _continuity_audit(targets, merged, profile, probes, config, h_mm=38.0, k=12)
    ordinary = targets[targets["target_id"].isin(merged["target_id"]) & targets["annular_coverage_eligible"].astype(bool)]
    u_values = np.sort(np.unique(np.round(ordinary["u_mm"].to_numpy(float), 6)))
    metrics["axial_layer_count"] = len(u_values); metrics["axial_gap_p95_mm"] = float(np.percentile(np.diff(u_values), 95)) if len(u_values) > 1 else math.inf
    _write_json(stage / "baseline_metrics.json", metrics); _write_parquet(edges, stage / "geometric_graph.parquet"); _write_parquet(direction, stage / "directional_coverage.parquet"); _write_parquet(stretch, stage / "graph_stretch_pairs.parquet")
    return _seal_gate(output_root, config, "retry16_baseline", {"status": "complete", **metrics, "legacy_circles_are_in_sample": True})


def stage_target_pool(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["target_pool"]; profile = pd.read_parquet(_upstream_path(config, "stable_core_profile")); power = 12 if smoke else int(config["sampling"]["target_pool_power"])
    pool = continuous_annular_sobol(profile, power=power, seed=int(config["sampling"]["target_pool_seed"]), zero_x_m=float(config["omega600"]["zero_x_m"]), pool_id="retry17_continuous_stable_core")
    pool["domain_class"] = "stable_core"; pool["proposal_service_mm"] = 0.0; pool["proposal_service_pass"] = True
    expanded_profile = pd.read_parquet(_upstream_path(config, "expanded_profile")); expanded = continuous_annular_sobol(expanded_profile, power=max(8, power - 1), seed=int(config["sampling"]["target_pool_seed"]) + 100, zero_x_m=float(config["omega600"]["zero_x_m"]), pool_id="retry17_continuous_expanded_exploration")
    outside_stable = ~points_inside_profile(expanded.loc[:, XYZ_COLUMNS].to_numpy(float), profile, zero_x_m=float(config["omega600"]["zero_x_m"])); expanded = expanded.loc[outside_stable].reset_index(drop=True)
    proposal = pd.read_parquet(_upstream_path(config, "retry15_proposal_xyz")); service = cKDTree(proposal.loc[:, XYZ_COLUMNS].to_numpy(float)).query(expanded.loc[:, XYZ_COLUMNS].to_numpy(float), k=1)[0] * 1000.0; expanded["proposal_service_mm"] = service; expanded["proposal_service_pass"] = service <= float(config["sampling"]["expanded_proposal_service_maximum_mm"]); expanded["domain_class"] = "expanded_exploratory"
    _write_parquet(pool, stage / "continuous_target_only_pool.parquet"); _write_parquet(expanded, stage / "expanded_exploratory_target_only_pool.parquet")
    return _seal_gate(output_root, config, "target_pool", {"status": "complete", "target_only_pool_count": len(pool), "expanded_exploratory_pool_count": len(expanded), "expanded_proposal_service_pass_count": int(expanded["proposal_service_pass"].sum()), "continuous_u_unique_count": int(pool["u_mm"].nunique()), "volume_correct_radius_sampling": True, "fixed_u_layers_used": False, "proposal_beta_used_as_label_or_hint": False})


def stage_phase_a_targets(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["phase_a_targets"]; pool = pd.read_parquet(output_root / STAGE_DIRS["target_pool"] / "continuous_target_only_pool.parquet"); old_targets, old = _old_quotient(config)
    cap = min(len(old) + 24, int(config["sampling"]["phase_a_quotient_cap"])) if smoke else int(config["sampling"]["phase_a_quotient_cap"])
    maximum_new = max(0, cap - len(old)); selected, batches = progressive_farthest_fill(pool, old.loc[:, XYZ_COLUMNS].to_numpy(float), maximum_new_points=maximum_new, minimum_separation_mm=float(config["sampling"]["minimum_separation_mm"]), batch_size=min(25, int(config["sampling"]["batch_size"])) if smoke else int(config["sampling"]["batch_size"]), target_fill_mm=float(config["continuity"]["phase_a_fill_p95_maximum_mm"]))
    selected["target_role"] = "retry17_continuous_fill"; selected["mandatory"] = False; selected["annular_coverage_eligible"] = True; selected["retry17_reused_locked"] = False; selected["circle_id"] = ""
    selected["legacy_in_sample_circle"] = False; selected["proposal_beta_label_eligible"] = False; selected["proposal_beta_seed_eligible"] = False; selected["proposal_beta_warm_start_eligible"] = False; selected["proposal_beta_branch_hint_eligible"] = False
    _write_parquet(selected, stage / "phase_a_new_target_registry.parquet"); _write_parquet(batches, stage / "progressive_fill_batches.parquet"); _write_parquet(old, stage / "reused_locked_supervision.parquet")
    all_targets = pd.concat([old, selected], ignore_index=True, sort=False); _write_parquet(all_targets, stage / "phase_a_all_targets.parquet")
    return _seal_gate(output_root, config, "phase_a_targets", {"status": "complete", "quotient_cap": cap, "reused_locked_count": len(old), "new_target_count": len(selected), "registered_total_count": len(all_targets), "minimum_separation_mm": float(config["sampling"]["minimum_separation_mm"]), "row_count_is_stop_condition": False})


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy(); environment.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "PYTHONHASHSEED": "20260915"}); return environment


def _effective_seed_budget(seed_budget: int, *, smoke: bool) -> int:
    value = 8 if smoke else int(seed_budget)
    if value not in (8, 16, 32): raise ValueError("retry17 candidate seed budget must reuse the registered 8/16/32 solver contract")
    return value


def _solve_candidates(
    config: Mapping[str, Any],
    targets: pd.DataFrame,
    stage: Path,
    *,
    seed_budget: int,
    smoke: bool,
    maximum_workers: int | None = None,
    work_namespace: str | None = None,
) -> pd.DataFrame:
    if targets.empty: return pd.DataFrame()
    work = stage / "_work" / (work_namespace or f"k{seed_budget}"); work.mkdir(parents=True, exist_ok=True)
    configured_workers = int(config["runtime"]["maximum_concurrent_workers"])
    worker_limit = configured_workers if maximum_workers is None else min(configured_workers, int(maximum_workers))
    if worker_limit < 1: raise ValueError("candidate worker limit must be positive")
    worker_count = min(worker_limit, len(targets)); shards = [frame for frame in np.array_split(targets, worker_count) if len(frame)]
    processes = []
    for index, frame in enumerate(shards):
        target_path = work / f"targets_{index:03d}.parquet"; output_path = work / f"candidates_{index:03d}.parquet"; _write_parquet(frame, target_path)
        command = [str(config["runtime"]["python"]), str(WORKER), "--config", str(config["config_path"]), "--targets", str(target_path), "--seed-bank", str(_upstream_path(config, "full_seed_bank")), "--y-seed-bank", str(_upstream_path(config, "y_seed_bank")), "--z-seed-bank", str(_upstream_path(config, "z_seed_bank")), "--output", str(output_path), "--seed-budget", str(_effective_seed_budget(seed_budget, smoke=smoke))]
        processes.append((subprocess.Popen(command, cwd=SOURCE_ROOT, env=_worker_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), output_path))
    frames = []
    for process, output_path in processes:
        stdout, stderr = process.communicate()
        if process.returncode: raise RuntimeError(f"retry17 candidate worker failed ({process.returncode}): {stderr[-4000:]} {stdout[-1000:]}")
        frames.append(pd.read_parquet(output_path))
    return pd.concat(frames, ignore_index=True, sort=False)


def stage_phase_a_candidate_bank(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["phase_a_candidate_bank"]; targets = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "phase_a_new_target_registry.parquet")
    candidates = _solve_candidates(config, targets, stage, seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]), smoke=smoke); legal = legal_candidate_clusters(candidates, residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]), cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]), weights=config["candidate_solver"]["beta_weights"])
    missing = targets[~targets["target_id"].isin(legal["target_id"] if len(legal) else [])]
    difficult = _solve_candidates(config, missing, stage, seed_budget=int(config["candidate_solver"]["difficult_seed_budget"]), smoke=smoke)
    if len(difficult): candidates = pd.concat([candidates, difficult], ignore_index=True, sort=False).drop_duplicates(["target_id", "candidate_id"], keep="last")
    legal = legal_candidate_clusters(candidates, residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]), cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]), weights=config["candidate_solver"]["beta_weights"])
    _write_parquet(candidates, stage / "phase_a_candidate_bank.parquet"); _write_parquet(legal, stage / "phase_a_legal_candidate_clusters.parquet"); _write_parquet(missing, stage / "difficult_target_registry.parquet")
    return _seal_gate(output_root, config, "phase_a_candidate_bank", {"status": "complete", "target_count": len(targets), "candidate_attempt_count": len(candidates), "accepted_target_count": int(legal["target_id"].nunique()) if len(legal) else 0, "difficult_target_count": len(missing), "proposal_beta_used_as_label_or_hint": False})


def _fixed_candidate_rows(old: pd.DataFrame) -> pd.DataFrame:
    frame = old.copy()
    frame["candidate_id"] = frame["target_id"].map(lambda value: f"retry17_locked:{value}")
    frame["seed_id"] = "retry16_locked"; frame["seed_rank"] = -1; frame["solver"] = "retry16_locked"
    frame["solver_success"] = True; frame["bounds_pass"] = True; frame["fk_residual_mm"] = frame.get("fk_residual_mm", 0.0)
    frame["min_margin_deg"] = frame.get("min_margin_deg", 0.0); frame["seam_class"] = frame.get("seam_class", "interior")
    frame["proposal_beta_used"] = False; frame["diagnostic_status"] = "locked_reuse"
    return frame


def _teacher_trial_metrics(labels: pd.DataFrame, targets: pd.DataFrame, edges: pd.DataFrame, old_ids: set[str], config: Mapping[str, Any]) -> dict[str, Any]:
    metric = dict(teacher_edge_metrics(labels, edges, weights=config["candidate_solver"]["beta_weights"]))
    lookup = labels.set_index("target_id"); conflict = []
    for edge in edges.to_dict("records"):
        left, right = str(edge["left_target_id"]), str(edge["right_target_id"])
        if (left in old_ids) == (right in old_ids) or left not in lookup.index or right not in lookup.index: continue
        a = lookup.loc[left, list(BETA_COLUMNS)].to_numpy(float); b = lookup.loc[right, list(BETA_COLUMNS)].to_numpy(float)
        conflict.append(float(np.max(np.abs(np.rad2deg(a - b)))) > 7.0)
    metric["old_new_conflict_count"] = int(sum(conflict)); metric["old_new_edge_count"] = len(conflict)
    metric["old_new_conflict_rate"] = float(np.mean(conflict)) if conflict else 0.0
    metric["accepted_target_count"] = int(labels["target_id"].nunique()); metric["target_count"] = len(targets)
    return metric


def _select_teacher(candidates: pd.DataFrame, targets: pd.DataFrame, old_ids: set[str], config: Mapping[str, Any], *, h_mm: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, solutions = [], {}
    for k in map(int, config["graph_teacher"]["k_values"]):
        edges = geometric_graph(targets, k=k, maximum_distance_mm=float(config["graph_teacher"]["maximum_edge_spacing_factor"]) * h_mm)
        t1 = density_normalized_select_t2(candidates, edges, pairwise_lambda=0.0, sigma_mm=h_mm, locked_target_ids=old_ids, weights=config["candidate_solver"]["beta_weights"])
        t1_signature = tuple(t1.sort_values("target_id")["candidate_id"].astype(str))
        rows.append({"teacher_id": f"T1_k{k}_lambda0", "graph_k": k, "pairwise_lambda": 0.0, "lambda0_equals_t1": True, **_teacher_trial_metrics(t1, targets, edges, old_ids, config)})
        for lam in map(float, config["graph_teacher"]["pairwise_lambdas"]):
            labels = density_normalized_select_t2(candidates, edges, pairwise_lambda=lam, sigma_mm=h_mm, locked_target_ids=old_ids, tau_deg=float(config["graph_teacher"]["robust_pairwise_cap_deg"]), weights=config["candidate_solver"]["beta_weights"], maximum_sweeps=int(config["graph_teacher"]["maximum_icm_sweeps"]))
            labels["graph_k"] = k; labels["teacher_id"] = f"T2_k{k}_lambda{lam:g}"
            metric = _teacher_trial_metrics(labels, targets, edges, old_ids, config)
            rows.append({"teacher_id": f"T2_k{k}_lambda{lam:g}", "graph_k": k, "pairwise_lambda": lam, "lambda0_equals_t1": tuple(t1.sort_values("target_id")["candidate_id"].astype(str)) == t1_signature, **metric})
            solutions[f"T2_k{k}_lambda{lam:g}"] = (labels, edges)
    comparison = pd.DataFrame(rows)
    eligible = comparison[comparison["pairwise_lambda"].gt(0)].copy()
    eligible["lambda4_distance"] = np.abs(eligible["pairwise_lambda"] - 4.0)
    chosen = eligible.sort_values(["old_new_conflict_count", "raw_gt7_rate", "weighted_p95_deg", "lambda4_distance", "graph_k", "teacher_id"], kind="stable").iloc[0]
    labels, edges = solutions[str(chosen["teacher_id"])]
    return labels.reset_index(drop=True), edges.reset_index(drop=True), comparison


def stage_phase_a_graph_teacher(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["phase_a_graph_teacher"]; targets = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "phase_a_all_targets.parquet"); old = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "reused_locked_supervision.parquet"); candidates = pd.read_parquet(output_root / STAGE_DIRS["phase_a_candidate_bank"] / "phase_a_candidate_bank.parquet")
    combined = pd.concat([_fixed_candidate_rows(old), candidates], ignore_index=True, sort=False); accepted_ids = set(legal_candidate_clusters(combined, weights=config["candidate_solver"]["beta_weights"])["target_id"].astype(str)); graph_targets = targets[targets["target_id"].astype(str).isin(accepted_ids)].drop_duplicates("target_id")
    labels, edges, comparison = _select_teacher(combined, graph_targets, set(old["target_id"].astype(str)), config, h_mm=float(config["continuity"]["phase_a_fill_p95_maximum_mm"]))
    _write_parquet(labels, stage / "phase_a_selected_teacher_labels.parquet"); _write_parquet(edges, stage / "phase_a_teacher_graph.parquet"); _write_parquet(comparison, stage / "phase_a_teacher_sweep.parquet")
    selected_id = str(labels["teacher_id"].iloc[0]) if len(labels) else ""
    return _seal_gate(output_root, config, "phase_a_graph_teacher", {"status": "complete", "selected_teacher": selected_id, "accepted_target_count": len(labels), "locked_retry16_label_count": len(old), "old_labels_overwritten": False, "density_normalized_pairwise_energy": True, "lambda0_control_equals_t1": bool(comparison["lambda0_equals_t1"].all())})


def stage_phase_a_adaptive_fill(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["phase_a_adaptive_fill"]; all_targets = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "phase_a_all_targets.parquet"); new = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "phase_a_new_target_registry.parquet"); old = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "reused_locked_supervision.parquet"); labels = pd.read_parquet(output_root / STAGE_DIRS["phase_a_graph_teacher"] / "phase_a_selected_teacher_labels.parquet")
    profile = pd.read_parquet(_upstream_path(config, "stable_core_profile")); probes = pd.read_parquet(output_root / STAGE_DIRS["objective_feasibility"] / "evaluation_probe_registry.parquet"); batch = min(25, int(config["sampling"]["batch_size"])) if smoke else int(config["sampling"]["batch_size"])
    records, audits = [], {}
    cutoffs = list(range(batch, len(new) + 1, batch));
    if not cutoffs or cutoffs[-1] != len(new): cutoffs.append(len(new))
    for cutoff in cutoffs:
        ids = set(old["target_id"].astype(str)) | set(new.head(cutoff)["target_id"].astype(str)); prefix_targets = all_targets[all_targets["target_id"].astype(str).isin(ids)].drop_duplicates("target_id"); prefix_labels = labels[labels["target_id"].astype(str).isin(ids)]
        metrics, edges, direction, stretch = _continuity_audit(prefix_targets, prefix_labels, profile, probes, config, h_mm=float(config["continuity"]["phase_a_fill_p95_maximum_mm"]), k=int(labels["graph_k"].iloc[0]))
        green = _green(metrics, config, phase="a"); records.append({"new_target_cutoff": cutoff, "green": green, **metrics["fill"], "lcc_fraction": metrics["lcc_fraction"], "local_directional_coverage": metrics["local_directional_coverage"], "stretch_p95": metrics["graph_stretch"]["p95"], "edge_weighted_p95_deg": metrics["label_continuity"]["weighted_p95_deg"], "edge_raw_gt7_rate": metrics["label_continuity"]["raw_gt7_rate"]}); audits[cutoff] = (metrics, edges, direction, stretch)
    audit = pd.DataFrame(records); eligible = audit[audit["green"]]; low_gain_cutoff = None; low_streak = 0
    for index in range(1, len(audit)):
        previous, current = audit.iloc[index - 1], audit.iloc[index]
        low = bool((previous["p95_mm"] - current["p95_mm"]) < float(config["continuity"]["low_gain_fill_mm"]) and (current["local_directional_coverage"] - previous["local_directional_coverage"]) < float(config["continuity"]["low_gain_direction_fraction"]) and current["lcc_fraction"] == previous["lcc_fraction"])
        low_streak = low_streak + 1 if low else 0
        if low_streak >= int(config["continuity"]["low_gain_consecutive_batches"]): low_gain_cutoff = int(current["new_target_cutoff"]); break
    if len(eligible): selected_cutoff, stop_reason = int(eligible.iloc[0]["new_target_cutoff"]), "phase_a_green"
    elif low_gain_cutoff is not None: selected_cutoff, stop_reason = low_gain_cutoff, "two_consecutive_low_gain_batches"
    else: selected_cutoff, stop_reason = int(audit.iloc[-1]["new_target_cutoff"]), "registered_phase_a_cap"
    metrics, edges, direction, stretch = audits[selected_cutoff]; _write_parquet(audit, stage / "adaptive_prefix_audit.parquet"); _write_parquet(edges, stage / "selected_geometric_graph.parquet"); _write_parquet(direction, stage / "selected_directional_coverage.parquet"); _write_parquet(stretch, stage / "selected_graph_stretch_pairs.parquet"); _write_json(stage / "selected_metrics.json", metrics)
    return _seal_gate(output_root, config, "phase_a_adaptive_fill", {"status": "green" if _green(metrics, config, phase="a") else "red", "phase_a_green": _green(metrics, config, phase="a"), "selected_new_target_cutoff": selected_cutoff, "stop_reason": stop_reason, **metrics})


def _density_weights(frame: pd.DataFrame, clip: Sequence[float]) -> np.ndarray:
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float); k = min(17, len(xyz)); distance = cKDTree(xyz).query(xyz, k=k)[0]
    radius = distance[:, -1] * 1000.0 if np.ndim(distance) == 2 else np.asarray(distance) * 1000.0
    weight = np.maximum(radius, 1e-6) ** 3; weight /= np.median(weight); return np.clip(weight, float(clip[0]), float(clip[1]))


def _split_roles(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    size = float(config["student"]["macroblock_mm"]); zero_x = float(config["omega600"]["zero_x_m"]); xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float); block = np.floor(np.column_stack([1000.0 * (zero_x - xyz[:, 0]), 1000.0 * xyz[:, 1], 1000.0 * xyz[:, 2]]) / size).astype(int)
    seed = int(config["sampling"]["spatial_split_seed"]); values = []
    train, validation, _test = map(float, config["student"]["split_fraction"])
    for row in block:
        digest = hashlib.sha256(f"{seed}:{row[0]}:{row[1]}:{row[2]}".encode()).digest(); value = int.from_bytes(digest[:8], "big") / 2**64
        values.append("train" if value < train else "validation" if value < train + validation else "test")
    return pd.Series(values, index=frame.index, dtype=str)


def _merge_label_metadata(labels: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [name for name in ("target_id", *XYZ_COLUMNS, "u_mm", "rho_mm", "target_role", "domain_class", "circle_id", "annular_coverage_eligible", "legacy_in_sample_circle", "retry17_reused_locked") if name in targets]
    metadata = targets.loc[:, metadata_columns].drop_duplicates("target_id")
    duplicate_metadata = [name for name in metadata_columns if name != "target_id" and name in labels]
    canonical_labels = labels.drop(columns=duplicate_metadata, errors="ignore")
    return canonical_labels.merge(metadata, on="target_id", how="inner", validate="one_to_one")


def _strip_target_metadata_from_labels(labels: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    metadata = set(targets.columns) - {"target_id"} - set(BETA_COLUMNS)
    return labels.drop(columns=sorted(metadata & set(labels.columns)), errors="ignore")


def _materialize_dataset(labels: pd.DataFrame, targets: pd.DataFrame, environment: Any, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = _merge_label_metadata(labels, targets)
    quotient = merged[~merged["target_role"].astype(str).eq("root_connector")].copy()
    quotient["split_role"] = _split_roles(quotient, config); quotient["sample_weight"] = _density_weights(quotient, config["student"]["density_weight_clip"])
    supervision_ids = set(quotient["target_id"].astype(str)); supervision_labels = labels[labels["target_id"].astype(str).isin(supervision_ids)]; supervision_targets = targets[targets["target_id"].astype(str).isin(supervision_ids)].drop_duplicates("target_id"); supervision_labels = _strip_target_metadata_from_labels(supervision_labels, supervision_targets)
    expanded, rejected = expand_annular_labels(supervision_labels, supervision_targets, environment, residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]))
    split = quotient.set_index("target_id")["split_role"]; weight = quotient.set_index("target_id")["sample_weight"]; expanded["split_role"] = expanded["target_id"].map(split); expanded["sample_weight"] = expanded["target_id"].map(weight)
    return quotient, expanded, rejected


def stage_phase_a_freeze_audit(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["phase_a_freeze_audit"]; cutoff = int(_gate(output_root, "phase_a_adaptive_fill")["selected_new_target_cutoff"]); old = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "reused_locked_supervision.parquet"); new = pd.read_parquet(output_root / STAGE_DIRS["phase_a_targets"] / "phase_a_new_target_registry.parquet").head(cutoff); targets = pd.concat([old, new], ignore_index=True, sort=False).drop_duplicates("target_id"); selected_ids = set(targets["target_id"].astype(str)); labels = pd.read_parquet(output_root / STAGE_DIRS["phase_a_graph_teacher"] / "phase_a_selected_teacher_labels.parquet"); labels = labels[labels["target_id"].astype(str).isin(selected_ids)]
    quotient, expanded, rejected = _materialize_dataset(labels, targets, _environment(config), config); hard = bool(len(quotient) and rejected.empty and not labels.get("proposal_beta_used", pd.Series(False, index=labels.index)).astype(bool).any() and np.isfinite(quotient.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy(float)).all() and not quotient.duplicated(list(XYZ_COLUMNS)).any() and set(quotient["split_role"]) >= {"train", "validation"})
    _write_parquet(targets, stage / "phase_a_frozen_targets.parquet"); _write_parquet(labels, stage / "phase_a_frozen_quotient_labels.parquet"); _write_parquet(quotient, stage / "phase_a_quotient_supervision.parquet"); _write_parquet(expanded, stage / "phase_a_full_g4_dataset.parquet"); _write_parquet(rejected, stage / "phase_a_symmetry_rejections.parquet")
    admission = bool(_gate(output_root, "phase_a_adaptive_fill")["fill"]["p95_mm"] <= float(config["continuity"]["phase_a_fill_p95_maximum_mm"]) and _gate(output_root, "phase_a_adaptive_fill")["lcc_fraction"] >= float(config["continuity"]["lcc_minimum"]) and _gate(output_root, "phase_a_adaptive_fill")["label_continuity"]["raw_gt7_rate"] <= float(config["continuity"]["edge_raw_gt7_rate_maximum_deg"] if "edge_raw_gt7_rate_maximum_deg" in config["continuity"] else config["continuity"]["edge_raw_gt7_rate_maximum"]))
    return _seal_gate(output_root, config, "phase_a_freeze_audit", {"status": "complete" if hard else "data_hard_red", "data_hard_gates_pass": hard, "phase_b_admitted": bool(hard and admission), "quotient_row_count": len(quotient), "expanded_row_count": len(expanded), "symmetry_rejection_count": len(rejected), "split_counts": quotient["split_role"].value_counts().to_dict(), "legacy_circle_rows_marked_in_sample": int(quotient.get("legacy_in_sample_circle", pd.Series(False)).fillna(False).sum()), "theta_mapping": "beta_to_theta_without_extra_sign"})


def stage_phase_b_continuation(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["phase_b_continuation"]; phase_a_gate = _gate(output_root, "phase_a_freeze_audit")
    source = output_root / STAGE_DIRS["phase_a_freeze_audit"]
    a_targets = pd.read_parquet(source / "phase_a_frozen_targets.parquet"); a_labels = pd.read_parquet(source / "phase_a_frozen_quotient_labels.parquet")
    if not phase_a_gate["phase_b_admitted"] or smoke:
        for name, frame in (("final_targets.parquet", a_targets), ("final_quotient_labels.parquet", a_labels), ("final_quotient_supervision.parquet", pd.read_parquet(source / "phase_a_quotient_supervision.parquet")), ("final_full_g4_dataset.parquet", pd.read_parquet(source / "phase_a_full_g4_dataset.parquet"))): _write_parquet(frame, stage / name)
        return _seal_gate(output_root, config, "phase_b_continuation", {"status": "not_admitted" if not phase_a_gate["phase_b_admitted"] else "smoke_skipped", "phase_b_executed": False, "final_dataset_source": "phase_a", "final_quotient_row_count": len(a_labels)})
    pool = pd.read_parquet(output_root / STAGE_DIRS["target_pool"] / "continuous_target_only_pool.parquet"); cap = int(config["sampling"]["phase_b_quotient_cap"]); maximum_new = max(0, cap - len(a_targets))
    pool = pool[~pool["target_id"].isin(a_targets["target_id"])].reset_index(drop=True); selected, batches = progressive_farthest_fill(pool, a_targets.loc[:, XYZ_COLUMNS].to_numpy(float), maximum_new_points=maximum_new, minimum_separation_mm=float(config["sampling"]["minimum_separation_mm"]), batch_size=int(config["sampling"]["batch_size"]), target_fill_mm=float(config["continuity"]["phase_b_fill_p95_maximum_mm"]))
    selected["target_role"] = "retry17_phase_b_fill"; selected["mandatory"] = False; selected["annular_coverage_eligible"] = True; selected["domain_class"] = "stable_core"; selected["legacy_in_sample_circle"] = False; selected["retry17_reused_locked"] = False; selected["circle_id"] = ""
    for column in ("proposal_beta_label_eligible", "proposal_beta_seed_eligible", "proposal_beta_warm_start_eligible", "proposal_beta_branch_hint_eligible"): selected[column] = False
    candidates = _solve_candidates(config, selected, stage, seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]), smoke=False); legal = legal_candidate_clusters(candidates, weights=config["candidate_solver"]["beta_weights"]); missing = selected[~selected["target_id"].isin(legal["target_id"] if len(legal) else [])]
    difficult = _solve_candidates(config, missing, stage, seed_budget=int(config["candidate_solver"]["difficult_seed_budget"]), smoke=False)
    if len(difficult): candidates = pd.concat([candidates, difficult], ignore_index=True, sort=False).drop_duplicates(["target_id", "candidate_id"], keep="last")
    all_targets = pd.concat([a_targets, selected], ignore_index=True, sort=False).drop_duplicates("target_id"); combined = pd.concat([_fixed_candidate_rows(a_labels), candidates], ignore_index=True, sort=False); accepted = set(legal_candidate_clusters(combined, weights=config["candidate_solver"]["beta_weights"])["target_id"].astype(str)); graph_targets = all_targets[all_targets["target_id"].astype(str).isin(accepted)]
    labels, teacher_edges, comparison = _select_teacher(combined, graph_targets, set(a_labels["target_id"].astype(str)), config, h_mm=float(config["continuity"]["phase_b_fill_p95_maximum_mm"]))
    profile = pd.read_parquet(_upstream_path(config, "stable_core_profile")); probes = pd.read_parquet(output_root / STAGE_DIRS["objective_feasibility"] / "evaluation_probe_registry.parquet"); metrics, edges, direction, stretch = _continuity_audit(all_targets, labels, profile, probes, config, h_mm=float(config["continuity"]["phase_b_fill_p95_maximum_mm"]), k=int(labels["graph_k"].iloc[0])); eligible = bool(metrics["label_continuity"]["weighted_p95_deg"] <= float(config["continuity"]["edge_weighted_p95_maximum_deg"]) and metrics["label_continuity"]["raw_gt7_rate"] <= float(config["continuity"]["edge_raw_gt7_rate_maximum"]))
    if eligible:
        quotient, expanded, rejected = _materialize_dataset(labels, all_targets, _environment(config), config); hard = bool(rejected.empty and len(quotient) and not quotient.duplicated(list(XYZ_COLUMNS)).any())
    else:
        hard = False
    if hard:
        final_targets, final_labels, source_name = all_targets, labels, "phase_b"; final_quotient, final_expanded = quotient, expanded
    else:
        final_targets, final_labels, source_name = a_targets, a_labels, "phase_a"; final_quotient = pd.read_parquet(source / "phase_a_quotient_supervision.parquet"); final_expanded = pd.read_parquet(source / "phase_a_full_g4_dataset.parquet")
    _write_parquet(selected, stage / "phase_b_new_target_registry.parquet"); _write_parquet(batches, stage / "phase_b_fill_batches.parquet"); _write_parquet(candidates, stage / "phase_b_candidate_bank.parquet"); _write_parquet(comparison, stage / "phase_b_teacher_sweep.parquet"); _write_parquet(edges, stage / "phase_b_geometric_graph.parquet"); _write_parquet(direction, stage / "phase_b_directional_coverage.parquet"); _write_parquet(stretch, stage / "phase_b_graph_stretch_pairs.parquet"); _write_json(stage / "phase_b_metrics.json", metrics)
    _write_parquet(final_targets, stage / "final_targets.parquet"); _write_parquet(final_labels, stage / "final_quotient_labels.parquet"); _write_parquet(final_quotient, stage / "final_quotient_supervision.parquet"); _write_parquet(final_expanded, stage / "final_full_g4_dataset.parquet")
    return _seal_gate(output_root, config, "phase_b_continuation", {"status": "green" if _green(metrics, config, phase="b") and hard else "diagnostic_red", "phase_b_executed": True, "phase_b_green": bool(_green(metrics, config, phase="b") and hard), "phase_b_label_continuity_eligible": eligible, "final_dataset_source": source_name, "final_quotient_row_count": len(final_labels), "phase_b_metrics": metrics})


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(lengths_m=np.asarray(environment.lengths_m, dtype=np.float32), p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32), theta_sign=float(environment.theta_sign), beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32))


def stage_student(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["student"]; if_hard = bool(_gate(output_root, "phase_a_freeze_audit")["data_hard_gates_pass"])
    if not if_hard:
        _write_parquet(pd.DataFrame(), stage / "training_history.parquet"); return _seal_gate(output_root, config, "student", {"status": "not_authorized_data_hard_red", "student_training_complete": False})
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1": raise RuntimeError("retry17 Student requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    if tf.config.get_visible_devices("GPU"): raise RuntimeError("retry17 CPU Student unexpectedly sees a GPU")
    supervision = pd.read_parquet(output_root / STAGE_DIRS["phase_b_continuation"] / "final_quotient_supervision.parquet").copy(); environment = _environment(config)
    jacobians = np.asarray([np.asarray(environment.jacobian(beta)).reshape(-1) for beta in supervision.loc[:, BETA_COLUMNS].to_numpy(float)])
    for index, column in enumerate(JACOBIAN_COLUMNS): supervision[column] = jacobians[:, index]
    supervision["record_id"] = supervision["target_id"].astype(str); supervision["kind"] = "static"; supervision["chart_id"] = "retry17_continuous_annular_quotient"; supervision["is_primary"] = True
    train = supervision[supervision["split_role"].eq("train")].copy(); validation = supervision[supervision["split_role"].eq("validation")].copy()
    if train.empty or validation.empty: raise RuntimeError("retry17 macroblock split produced empty train or validation")
    student = config["student"]; result = train_workspace_student(train, validation, mode=RepresentationMode.XYZ_GLOBAL, geometry=_student_geometry(environment), config=WorkspaceStudentTrainingConfig(hidden_units=tuple(map(int, student["hidden_units"])), learning_rate=float(student["learning_rate"]), max_steps=50 if smoke else int(student["maximum_steps"]), validation_interval=10 if smoke else int(student["validation_interval"]), patience_intervals=2 if smoke else int(student["patience_intervals"]), seed=int(config["sampling"]["student_seed"]), beta_coordinate_weights=(4, 4, 2, 2, 1, 1), beta_loss_only=True))
    save_workspace_student_models(result.models, stage / "models"); model = result.models.global_model; xyz = validation.loc[:, XYZ_COLUMNS].to_numpy(float); raw = np.asarray(model(xyz.astype(np.float32), training=False), dtype=float); zero = np.asarray(environment.fk(np.zeros(6))).reshape(3); corrected = trajectory_evaluation.two_step_dls(environment, raw, xyz, zero_xyz=zero); raw_fk = np.linalg.norm(np.asarray(environment.fk(raw)).reshape(-1, 3) - xyz, axis=1) * 1000; corrected_fk = np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - xyz, axis=1) * 1000
    predictions = validation[["target_id", *XYZ_COLUMNS, *BETA_COLUMNS]].copy()
    for axis, name in enumerate(BETA_COLUMNS): predictions[f"raw_{name}"] = raw[:, axis]; predictions[f"dls2_{name}"] = corrected[:, axis]
    predictions["raw_fk_residual_mm"] = raw_fk; predictions["dls2_fk_residual_mm"] = corrected_fk
    _write_parquet(supervision, stage / "student_supervision.parquet"); _write_parquet(result.history, stage / "training_history.parquet"); _write_parquet(predictions, stage / "validation_predictions.parquet")
    complete = bool(np.isfinite(raw).all() and np.isfinite(corrected).all()); return _seal_gate(output_root, config, "student", {"status": "complete" if complete else "red", "student_training_complete": complete, "diagnostic_red_dataset_allowed": not _gate(output_root, "phase_a_adaptive_fill")["phase_a_green"], "train_row_count": len(train), "validation_row_count": len(validation), "validation_raw_fk_p95_mm": percentile(raw_fk, 95), "validation_dls2_fk_p95_mm": percentile(corrected_fk, 95)})


def _tree_manifest(root: Path) -> dict[str, Any]:
    return {"schema_version": 1, "artifacts": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(root.rglob("*")) if path.is_file()]}


def stage_dataset_student_lock(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["dataset_student_lock"]; dataset = output_root / STAGE_DIRS["phase_b_continuation"] / "final_full_g4_dataset.parquet"; quotient = output_root / STAGE_DIRS["phase_b_continuation"] / "final_quotient_supervision.parquet"; model_root = output_root / STAGE_DIRS["student"] / "models"
    model_manifest = _tree_manifest(model_root) if model_root.exists() else {"schema_version": 1, "artifacts": []}; lock = {"schema_version": 1, "locked_before_shape_generation": True, "dataset": {"path": str(dataset.relative_to(output_root)), "sha256": sha256_file(dataset)}, "quotient": {"path": str(quotient.relative_to(output_root)), "sha256": sha256_file(quotient)}, "models": model_manifest}
    _write_json(stage / "dataset_student_lock.json", lock); return _seal_gate(output_root, config, "dataset_student_lock", {"status": "locked", "dataset_student_locked": True, "dataset_sha256": lock["dataset"]["sha256"], "student_model_artifact_count": len(model_manifest["artifacts"])})


def _verify_lock(output_root: Path) -> bool:
    lock = _read_json(output_root / STAGE_DIRS["dataset_student_lock"] / "dataset_student_lock.json"); dataset = output_root / lock["dataset"]["path"]
    if sha256_file(dataset) != lock["dataset"]["sha256"]: return False
    model_root = output_root / STAGE_DIRS["student"] / "models"; return (not lock["models"]["artifacts"]) or _tree_manifest(model_root) == lock["models"]


def stage_heldout_shapes(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["heldout_shapes"]
    if not _verify_lock(output_root): raise RuntimeError("retry17 dataset/model lock mutated before trajectory generation")
    profile = pd.read_parquet(_upstream_path(config, "stable_core_profile")); dataset = pd.read_parquet(output_root / STAGE_DIRS["phase_b_continuation"] / "final_full_g4_dataset.parquet"); trajectory = config["trajectories"]
    selected, waypoints, candidates = generate_maximal_shape_registry(profile, dataset, zero_x_m=float(config["omega600"]["zero_x_m"]), center_seed=int(config["sampling"]["trajectory_center_seed"]), center_power=9 if smoke else int(config["sampling"]["trajectory_center_power"]), center_count=8 if smoke else int(trajectory["center_count"]), waypoint_count=48 if smoke else int(trajectory["waypoint_count"]), erosion_mm=float(trajectory["erosion_mm"]), support_maximum_mm=float(trajectory["support_maximum_mm"]), minimum_axial_span_mm=20.0 if smoke else float(trajectory["minimum_axial_span_mm"]))
    _write_parquet(selected, stage / "heldout_shape_registry.parquet"); _write_parquet(waypoints, stage / "heldout_shape_waypoints.parquet"); _write_parquet(candidates, stage / "shape_search_candidates.parquet")
    lock_after = _verify_lock(output_root); selected_count = int(selected["selection_status"].eq("selected").sum()) if len(selected) else 0
    return _seal_gate(output_root, config, "heldout_shapes", {"status": "complete" if selected_count == 9 else "slot_failure", "dataset_student_lock_verified_before_and_after": lock_after, "required_slot_count": 9, "selected_slot_count": selected_count, "shape_selection_used_teacher_or_student_error": False, "largest_found_not_global_optimum": True, "sharp_rectangle_diagnostic_registered": bool(len(waypoints) and waypoints["trajectory_id"].eq("retry17_sharp_rectangle_diagnostic").any())})


def _trajectory_metrics(beta: np.ndarray, xyz: np.ndarray, environment: Any, *, threshold_deg: float = 7.0) -> dict[str, float]:
    achieved = np.asarray(environment.fk(beta)).reshape(-1, 3); residual = np.linalg.norm(achieved - xyz, axis=1) * 1000.0; closed = np.vstack([beta, beta[:1]]); delta = np.abs(np.rad2deg(np.diff(closed, axis=0))); raw_step = np.max(delta, axis=1); weighted_step = np.asarray(weighted_beta_rms_deg(closed[1:], closed[:-1], (4, 4, 2, 2, 1, 1)), dtype=float)
    return {"success_rate": float(np.mean(residual <= 10.0)), "fk_p95_mm": percentile(residual, 95), "fk_maximum_mm": float(np.max(residual)), "raw_step_gt7_rate": float(np.mean(raw_step > threshold_deg)), "raw_step_maximum_deg": float(np.max(raw_step)), "weighted_step_p95_deg": percentile(weighted_step, 95), "closing_unique_edge_raw_deg": float(raw_step[-1]), "closing_unique_edge_weighted_deg": float(weighted_step[-1]), "repeated_endpoint_return_deg": 0.0}


def stage_trajectory_evaluation(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["trajectory_evaluation"]
    if not _verify_lock(output_root): raise RuntimeError("retry17 dataset/model lock mutated before trajectory evaluation")
    registry = pd.read_parquet(output_root / STAGE_DIRS["heldout_shapes"] / "heldout_shape_registry.parquet"); waypoints = pd.read_parquet(output_root / STAGE_DIRS["heldout_shapes"] / "heldout_shape_waypoints.parquet"); gating = waypoints[~waypoints["shape_class"].eq("sharp_rectangle_diagnostic")].copy()
    if gating.empty:
        _write_parquet(pd.DataFrame(), stage / "trajectory_report.parquet"); _write_parquet(pd.DataFrame(), stage / "trajectory_waypoint_evaluation.parquet"); return _seal_gate(output_root, config, "trajectory_evaluation", {"status": "not_available", "trajectory_green": False})
    gating["target_id"] = [
        f"{trajectory}:{int(index):05d}"
        for trajectory, index in zip(gating["trajectory_id"], gating["waypoint_index"], strict=True)
    ]
    candidates = _solve_candidates(
        config,
        gating,
        stage,
        seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]),
        smoke=smoke,
    )
    # The solver owns candidate fields; the frozen waypoint registry owns trajectory metadata.
    candidates = candidates.merge(
        gating[["target_id", "trajectory_id", "waypoint_index"]],
        on="target_id",
        how="left",
        validate="many_to_one",
    )
    environment = _environment(config); zero = np.asarray(environment.fk(np.zeros(6))).reshape(3); model = None
    if _gate(output_root, "student").get("student_training_complete", False):
        import tensorflow as tf
        model = tf.keras.models.load_model(output_root / STAGE_DIRS["student"] / "models" / "global.keras", compile=False)
    report_rows, waypoint_rows = [], []
    selected_lambda = float(pd.read_parquet(output_root / STAGE_DIRS["phase_a_graph_teacher"] / "phase_a_selected_teacher_labels.parquet")["pairwise_lambda"].iloc[0])
    for trajectory_id, points in gating.groupby("trajectory_id", sort=True):
        ordered = points.sort_values("waypoint_index", kind="stable"); local = candidates[candidates["trajectory_id"].astype(str).eq(str(trajectory_id))]; teacher = trajectory_evaluation.cycle_teacher(local, pairwise_lambda=selected_lambda, weights=config["candidate_solver"]["beta_weights"], tau_deg=float(config["graph_teacher"]["robust_pairwise_cap_deg"])); complete = len(teacher) == len(ordered)
        xyz = ordered.loc[:, XYZ_COLUMNS].to_numpy(float); teacher_beta = teacher.sort_values("waypoint_index").loc[:, BETA_COLUMNS].to_numpy(float) if complete else np.full((len(xyz), 6), np.nan)
        teacher_metric = _trajectory_metrics(teacher_beta, xyz, environment) if complete else {"success_rate": 0.0, "fk_p95_mm": math.inf, "raw_step_gt7_rate": 1.0, "repeated_endpoint_return_deg": math.inf}
        if model is not None:
            raw = trajectory_evaluation.symmetry_prediction(model, xyz, zero); corrected = trajectory_evaluation.two_step_dls(environment, raw, xyz, zero_xyz=zero); raw_metric = _trajectory_metrics(raw, xyz, environment); dls_metric = _trajectory_metrics(corrected, xyz, environment)
        else:
            raw = corrected = np.full((len(xyz), 6), np.nan); raw_metric = dls_metric = {"success_rate": 0.0, "fk_p95_mm": math.inf, "raw_step_gt7_rate": 1.0, "repeated_endpoint_return_deg": math.inf}
        registry_row = registry[registry["trajectory_id"].astype(str).eq(str(trajectory_id))].iloc[0]; green = bool(complete and dls_metric["success_rate"] >= float(config["trajectories"]["dls2_success_minimum"]) and dls_metric["fk_p95_mm"] <= float(config["trajectories"]["dls2_fk_p95_maximum_mm"]) and dls_metric["raw_step_gt7_rate"] <= float(config["trajectories"]["corrected_raw_step_gt7_rate_maximum"]) and dls_metric["repeated_endpoint_return_deg"] <= float(config["trajectories"]["loop_return_maximum_deg"]) and float(registry_row["support_maximum_mm"]) <= float(config["trajectories"]["support_maximum_mm"]) and float(registry_row["axial_span_mm"]) >= (20.0 if smoke else float(config["trajectories"]["minimum_axial_span_mm"])))
        record = {"trajectory_id": trajectory_id, "shape_class": str(ordered["shape_class"].iloc[0]), "waypoint_count": len(ordered), "reference_teacher_complete": complete, "support_maximum_mm": float(registry_row["support_maximum_mm"]), "axial_span_mm": float(registry_row["axial_span_mm"]), "trajectory_green": green}
        for prefix, metric in (("teacher", teacher_metric), ("raw_student", raw_metric), ("dls2", dls_metric)):
            for key, value in metric.items(): record[f"{prefix}_{key}"] = value
        report_rows.append(record)
        evaluated = ordered.copy()
        for axis, name in enumerate(BETA_COLUMNS): evaluated[f"teacher_{name}"] = teacher_beta[:, axis]; evaluated[f"raw_student_{name}"] = raw[:, axis]; evaluated[f"dls2_{name}"] = corrected[:, axis]
        waypoint_rows.append(evaluated)
    reports = pd.DataFrame(report_rows); evaluated = pd.concat(waypoint_rows, ignore_index=True, sort=False); _write_parquet(candidates, stage / "trajectory_candidate_bank.parquet"); _write_parquet(reports, stage / "trajectory_report.parquet"); _write_parquet(evaluated, stage / "trajectory_waypoint_evaluation.parquet")
    by_class = reports.groupby("shape_class")["trajectory_green"].any(); overall = bool(all(by_class.get(name, False) for name in ("ellipse", "rounded_rectangle", "rounded_star")))
    return _seal_gate(output_root, config, "trajectory_evaluation", {"status": "green" if overall else "red", "trajectory_green": overall, "green_by_class": by_class.to_dict(), "trajectory_count": len(reports), "teacher_selection": "independent_cycle_dynamic_programming", "student_warm_started_teacher": False, "trajectory_labels_added_back_to_dataset": False, "dls2_is_real_closed_loop": False})


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    artifacts = []
    for stage_name in STAGE_ORDER[:-1]:
        stage = output_root / STAGE_DIRS[stage_name]
        for path in sorted(stage.rglob("*")):
            if path.is_file() and "_work" not in path.parts: artifacts.append({"path": str(path.relative_to(output_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "scientific_source_fixed_point": _git_sha(), "artifacts": artifacts}


def stage_summary(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]; operational = all(_stage_is_complete(output_root, name) for name in STAGE_ORDER[:-1]); baseline = _gate(output_root, "retry16_baseline"); phase_a = _gate(output_root, "phase_a_adaptive_fill"); phase_b = _gate(output_root, "phase_b_continuation"); student = _gate(output_root, "student"); trajectory = _gate(output_root, "trajectory_evaluation")
    gate = {"status": "complete" if operational else "incomplete", "operational_completion": operational, "artifact_completeness": operational, "diagnostic_smoke": bool(smoke), "diagnostic_continuity_result": "green" if phase_a["phase_a_green"] and trajectory.get("trajectory_green", False) else "red", "retry16_baseline": {"fill_p95_mm": baseline["fill"]["p95_mm"], "lcc_fraction": baseline["lcc_fraction"], "local_directional_coverage": baseline["local_directional_coverage"], "axial_layer_count": baseline["axial_layer_count"], "axial_gap_p95_mm": baseline["axial_gap_p95_mm"]}, "phase_a": {"green": phase_a["phase_a_green"], "fill_p95_mm": phase_a["fill"]["p95_mm"], "lcc_fraction": phase_a["lcc_fraction"], "local_directional_coverage": phase_a["local_directional_coverage"], "stretch_p95": phase_a["graph_stretch"]["p95"], "weighted_edge_p95_deg": phase_a["label_continuity"]["weighted_p95_deg"], "raw_gt7_rate": phase_a["label_continuity"]["raw_gt7_rate"]}, "phase_b": {"executed": phase_b["phase_b_executed"], "green": phase_b.get("phase_b_green", False), "final_dataset_source": phase_b["final_dataset_source"], "final_quotient_row_count": phase_b["final_quotient_row_count"]}, "student": {"training_complete": student.get("student_training_complete", False), "validation_raw_fk_p95_mm": student.get("validation_raw_fk_p95_mm"), "validation_dls2_fk_p95_mm": student.get("validation_dls2_fk_p95_mm")}, "trajectory": {"green": trajectory.get("trajectory_green", False), "green_by_class": trajectory.get("green_by_class", {})}, "upstream_proposal_stability_pass": False, "proposal_beta_used_as_label_or_hint": False, "downstream_authorization": False, "formal_authorization": False, "deployment_authorization": False, "full_workspace_authorization": False, "continuous_workspace_authorization": False, "tension_executed": False}
    stage.mkdir(parents=True, exist_ok=True); (stage / "retry17_continuity_summary.html").write_text("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>retry17 continuity</title></head><body><h1>retry17 连续环状域诊断</h1><pre>" + json.dumps(gate, ensure_ascii=False, indent=2, default=_json_default) + "</pre></body></html>", encoding="utf-8")
    sealed = _seal_gate(output_root, config, "summary", gate); _write_json(stage / "artifact_manifest.json", _artifact_manifest(output_root)); _seal_stage(output_root, "summary"); _write_json(output_root / "progress.json", {"status": "complete", "phase": "summary", "completed": len(STAGE_ORDER), "total": len(STAGE_ORDER), "message": "retry17 continuity experiment completed", "observed_at_unix": time.time()}); return sealed


STAGE_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "objective_feasibility": stage_objective_feasibility, "retry16_baseline": stage_retry16_baseline,
    "target_pool": stage_target_pool, "phase_a_targets": stage_phase_a_targets,
    "phase_a_candidate_bank": stage_phase_a_candidate_bank, "phase_a_graph_teacher": stage_phase_a_graph_teacher,
    "phase_a_adaptive_fill": stage_phase_a_adaptive_fill, "phase_a_freeze_audit": stage_phase_a_freeze_audit,
    "phase_b_continuation": stage_phase_b_continuation, "student": stage_student,
    "dataset_student_lock": stage_dataset_student_lock, "heldout_shapes": stage_heldout_shapes,
    "trajectory_evaluation": stage_trajectory_evaluation, "summary": stage_summary,
}


def run(config_path: str | Path, output_root: str | Path, binding_sha: str, *, smoke: bool = False, stage: str | None = None, validate_stage: str | None = None) -> dict[str, Any]:
    config = load_config(config_path); output = Path(output_root).resolve(); _ensure_identity(config, output, binding_sha, smoke=smoke)
    if validate_stage:
        if validate_stage not in STAGE_RUNNERS: raise ValueError(f"unknown retry17 stage {validate_stage}")
        if not _stage_is_complete(output, validate_stage): raise RuntimeError(f"retry17 stage is incomplete or mutated: {validate_stage}")
        return _gate(output, validate_stage)
    if stage:
        position = STAGE_ORDER.index(stage)
        if position and not _stage_is_complete(output, STAGE_ORDER[position - 1]): raise RuntimeError("retry17 previous stage is incomplete")
        return _gate(output, stage) if _stage_is_complete(output, stage) else STAGE_RUNNERS[stage](config, output, smoke=smoke)
    result = {}
    for index, name in enumerate(STAGE_ORDER):
        _progress(output, name, completed=index, total=len(STAGE_ORDER)); result = _gate(output, name) if _stage_is_complete(output, name) else STAGE_RUNNERS[name](config, output, smoke=smoke)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--binding-sha", required=True); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--stage", choices=STAGE_ORDER); parser.add_argument("--validate-stage", choices=STAGE_ORDER); return parser.parse_args()


def main() -> int:
    args = parse_args(); gate = run(args.config, args.output_root, args.binding_sha, smoke=args.smoke, stage=args.stage, validate_stage=args.validate_stage); print(json.dumps(gate, sort_keys=True, default=_json_default)); return 0


if __name__ == "__main__": raise SystemExit(main())
