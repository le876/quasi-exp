#!/usr/bin/env python3
"""Run retry18 zero-tip completion and parity-smooth Student diagnostics."""

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

# retry17 still owns environment, Candidate Bank, Graph Teacher, and StudentGeometry composition.
import run_bacra_v14_3r_retry17_continuity as retry17
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher import trajectory_evaluation
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import percentile, weighted_beta_rms_deg
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, THETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry15_canonical_graph import legal_candidate_clusters
from quasi_exp.teacher.retry16_annular_tube import expand_annular_labels
from quasi_exp.teacher.retry17_continuity_fill import geometric_graph, graph_lcc_fraction, progressive_farthest_fill, teacher_edge_metrics
from quasi_exp.teacher.retry17_trajectories import generate_maximal_shape_registry
from quasi_exp.teacher.retry18_parity_student import (
    ParityStudentConfig,
    data_driven_radial_scale_mm,
    load_parity_student,
    save_parity_student,
    train_parity_smooth_student,
)
from quasi_exp.teacher.retry18_root_tip import (
    RootTipPolicy,
    add_tip_coordinates,
    classify_root_connectors,
    common_zero_tip_support,
    coverage_distance,
    fit_sqrt_u_tip_profile,
    interpolate_tip_profile,
    profile_overlap_metrics,
    sample_tip_volume,
    weighted_sobol_beta_ball,
    zero_connected_radial_intervals,
)
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_student import WorkspaceStudentTrainingConfig, save_workspace_student_models, train_workspace_student


EXPERIMENT_ID = "bacra_v14_3r_retry18_zero_tip_parity_smooth"
STAGE_DIRS = {
    "objective_baseline": "00_objective_baseline",
    "zero_tip_discovery": "01_zero_tip_discovery",
    "profile_preflight": "02_profile_preflight",
    "geometric_budget": "03_geometric_budget",
    "root_teacher": "04_root_teacher",
    "unified_dataset": "05_unified_dataset",
    "seam_causal_audit": "06_seam_causal_audit",
    "parity_regularity": "07_parity_regularity",
    "student_ablation": "08_student_ablation",
    "student_selection": "09_student_selection",
    "dataset_model_lock": "10_dataset_model_lock",
    "heldout_trajectories": "11_heldout_trajectories",
    "trajectory_evaluation": "12_trajectory_evaluation",
    "summary": "13_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False); temporary.replace(path)


def _git_sha() -> str: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()


def _config_sha(config: Mapping[str, Any]) -> str: return sha256_file(Path(str(config["config_path"])))


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    spec = config["upstream"][key]
    path = Path(str(spec["path"]))
    if path.is_absolute(): return path
    root = config["upstream"]["retry16_root" if spec.get("root") == "retry16" else "retry17_root"]
    return Path(str(root)) / path


def _environment(config: Mapping[str, Any]) -> Any: return retry17._environment(config)


def _tip_policy(config: Mapping[str, Any]) -> RootTipPolicy:
    tip = config["root_tip"]
    return RootTipPolicy(**{key: tip[key] for key in (
        "axial_step_mm", "radial_step_mm", "sector_count", "maximum_u_mm", "maximum_rho_mm",
        "continuity_gap_mm", "axis_core_radius_mm", "angular_arc_mm",
    )})


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); config["config_path"] = str(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID: raise ValueError("retry18 experiment_id mismatch")
    if int(config["runtime"]["maximum_concurrent_workers"]) != 12 or int(config["runtime"]["numerical_threads_per_worker"]) != 1: raise ValueError("retry18 requires twelve single-threaded workers")
    if not 1 <= int(config["runtime"]["trajectory_candidate_workers"]) <= int(config["runtime"]["maximum_concurrent_workers"]): raise ValueError("retry18 trajectory worker cap mismatch")
    if tuple(map(int, config["graph_teacher"]["k_values"])) != (12, 16): raise ValueError("retry18 Teacher k registry mismatch")
    if tuple(map(float, config["graph_teacher"]["pairwise_lambdas"])) != (2.0, 4.0, 8.0): raise ValueError("retry18 Teacher lambda registry mismatch")
    if float(config["root_tip"]["axis_core_radius_mm"]) < 2 * float(config["root_tip"]["radial_step_mm"]): raise ValueError("axis core must cover at least two radial bins")
    _tip_policy(config)
    claims = config["claims"]
    if not claims["diagnostic_only"] or any(bool(value) for key, value in claims.items() if key != "diagnostic_only"): raise ValueError("retry18 claim boundary mismatch")
    if sha256_file(SOURCE_ROOT / str(config["sources"]["robot_config"])) != str(config["sources"]["robot_config_sha256"]): raise ValueError("retry18 robot config SHA mismatch")
    return config


def _binding_definition(binding_sha: str) -> Mapping[str, Any]:
    raw = subprocess.check_output(["git", "show", f"{binding_sha}:spec/registry.yaml"], cwd=SOURCE_ROOT, text=True)
    return yaml.safe_load(raw)["experiments"][EXPERIMENT_ID]


def _ensure_identity(config: Mapping[str, Any], output_root: Path, binding_sha: str, *, smoke: bool) -> None:
    identity = {"experiment_id": EXPERIMENT_ID, "scientific_source_fixed_point": _git_sha(), "binding_fixed_point": str(binding_sha), "config_sha256": _config_sha(config), "diagnostic_smoke": bool(smoke)}
    definition = _binding_definition(binding_sha)
    expected = {"scientific_source_fixed_point": _git_sha(), "config": str(Path(config["config_path"]).relative_to(SOURCE_ROOT)), "runner": str(Path(__file__).relative_to(SOURCE_ROOT))}
    for key, value in expected.items():
        if str(definition.get(key)) != str(value): raise RuntimeError(f"retry18 binding mismatch for {key}")
    path = output_root / "run_identity.json"
    if path.exists() and _read_json(path) != identity: raise RuntimeError("retry18 output identity mismatch")
    if not path.exists(): output_root.mkdir(parents=True, exist_ok=False); _write_json(path, identity)


def _stage_manifest(output_root: Path, stage_name: str) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = [{"path": str(p.relative_to(stage)), "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(stage.rglob("*")) if p.is_file() and p.name != "completion_manifest.json" and "_work" not in p.parts]
    upstream = []
    index = STAGE_ORDER.index(stage_name)
    if index:
        previous = output_root / STAGE_DIRS[STAGE_ORDER[index - 1]] / "completion_manifest.json"
        upstream = [{"stage": STAGE_ORDER[index - 1], "sha256": sha256_file(previous)}]
    return {"schema_version": 1, "stage": stage_name, "scientific_source_fixed_point": _git_sha(), "artifacts": artifacts, "upstream": upstream}


def _seal_stage(output_root: Path, stage_name: str) -> None: _write_json(output_root / STAGE_DIRS[stage_name] / "completion_manifest.json", _stage_manifest(output_root, stage_name))


def _seal_gate(output_root: Path, config: Mapping[str, Any], stage_name: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    value = {**dict(gate), "stage": stage_name, "experiment_id": EXPERIMENT_ID, "scientific_source_fixed_point": _git_sha(), "config_sha256": _config_sha(config), **config["claims"]}
    _write_json(output_root / STAGE_DIRS[stage_name] / "gate.json", value); _seal_stage(output_root, stage_name); return value


def _gate(output_root: Path, stage_name: str) -> dict[str, Any]: return _read_json(output_root / STAGE_DIRS[stage_name] / "gate.json")


def _complete(output_root: Path, stage_name: str) -> bool:
    path = output_root / STAGE_DIRS[stage_name] / "completion_manifest.json"
    return path.exists() and _read_json(path) == _stage_manifest(output_root, stage_name)


def _progress(output_root: Path, stage_name: str, index: int) -> None:
    _write_json(output_root / "progress.json", {"status": "running", "phase": stage_name, "completed": index, "total": len(STAGE_ORDER), "observed_at_unix": time.time()})


def _input_integrity(config: Mapping[str, Any]) -> tuple[pd.DataFrame, bool]:
    rows = []
    for key, spec in config["upstream"].items():
        if not isinstance(spec, Mapping) or "sha256" not in spec: continue
        path = _upstream_path(config, key); observed = sha256_file(path) if path.exists() else None
        rows.append({"source_key": key, "path": str(path), "expected_sha256": str(spec["sha256"]), "observed_sha256": observed, "match": observed == str(spec["sha256"])})
    frame = pd.DataFrame(rows); return frame, bool(len(frame) and frame["match"].all())


def stage_objective_baseline(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["objective_baseline"]
    integrity, passed = _input_integrity(config); _write_parquet(integrity, stage / "upstream_verification.parquet")
    if not passed: raise RuntimeError("retry18 upstream integrity failed")
    environment = _environment(config); zero = np.asarray(environment.fk(np.zeros(6))).reshape(3); expected = np.asarray([config["omega600"]["zero_x_m"], 0, 0], float)
    if not np.allclose(zero, expected, atol=1e-12, rtol=0): raise RuntimeError("retry18 exact-zero mismatch")
    old = pd.read_parquet(_upstream_path(config, "quotient_supervision")); nonzero = old[np.linalg.norm(old.loc[:, XYZ_COLUMNS].to_numpy(float) - zero, axis=1) > 1e-12]
    nearest = float(cKDTree(nonzero.loc[:, XYZ_COLUMNS]).query(zero)[0] * 1000.0)
    old_eval = pd.read_parquet(_upstream_path(config, "old_trajectory_evaluation")); rows = []
    for trajectory_id, frame in old_eval[old_eval["shape_class"].eq("rounded_rectangle")].groupby("trajectory_id", sort=True):
        frame = frame.sort_values("waypoint_index"); beta = frame[[f"raw_student_{name}" for name in BETA_COLUMNS]].to_numpy(float)
        rows.append({"trajectory_id": trajectory_id, **trajectory_evaluation.path_metrics(frame.loc[:, XYZ_COLUMNS].to_numpy(float), beta, environment)})
    rectangle = pd.DataFrame(rows); _write_parquet(rectangle, stage / "rectangle_raw_spike_report.parquet")
    objective = {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "diagnostic_pilot_allowed": True, "objectives": ["zero_tip_geometric_and_teacher_connection", "teacher_student_seam_causality", "raw_student_spike_removal"], "data_student_dls_axes_separate": True}
    denominator = {"schema_version": 1, "tip_target_cap": int(config["root_tip"]["target_cap"]), "seam_anchor_count_per_axis": int(config["seam_audit"]["anchor_count_per_seam"]), "old_rectangle_count": len(rectangle), "resource_feasibility_refrozen_after_profile": True}
    reusable = {"schema_version": 1, "retry17_quotient_rows": len(old), "retry17_labels_locked_by_default": True, "proposal_beta_used_as_label_or_hint": False}
    lower = {"schema_version": 1, "zero_tip_target_requirement": "not_bounded_until_profile", "all_required_objectives_bounded": False, "registered_cap": int(config["root_tip"]["target_cap"])}
    schedule = {"schema_version": 1, "atomic_order": list(STAGE_ORDER), "row_count_is_stop_condition": False}
    feasibility = {"schema_version": 1, "status": "diagnostic_only", "claim_bearing_run_authorized": False, "diagnostic_pilot_authorized": True}
    for name, value in (("objective_contract.json", objective), ("denominator_size.json", denominator), ("reusable_evidence.json", reusable), ("budget_lower_bound.json", lower), ("atomic_objective_schedule.json", schedule), ("feasibility_gate.json", feasibility)): _write_json(stage / name, value)
    return _seal_gate(output_root, config, "objective_baseline", {"status": "diagnostic_only", "upstream_integrity_pass": True, "retry17_nearest_zero_nonzero_mm": nearest, "rectangle_count": len(rectangle), "rectangle_any_max_step_excess_gt5": bool(len(rectangle) and rectangle["path_step_excess_maximum_mm"].gt(5).any()), "objective_lower_bounds_complete": False})


def stage_zero_tip_discovery(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["zero_tip_discovery"]; environment = _environment(config); tip = config["root_tip"]; policy = _tip_policy(config)
    zero_beta = np.zeros(6); zero = np.asarray(environment.fk(zero_beta)).reshape(3); j0 = np.asarray(environment.jacobian(zero_beta)).reshape(3, 6); singular = np.linalg.svd(j0, compute_uv=False)
    _write_json(stage / "zero_jacobian.json", {"jacobian_m_per_rad": j0, "singular_values_m_per_rad": singular, "rank": int(np.linalg.matrix_rank(j0)), "zero_xyz_m": zero})
    second = []
    for axis in range(6):
        for radius_deg in (0.1, 0.25, 0.5):
            delta = np.zeros(6); delta[axis] = np.deg2rad(radius_deg)
            plus = np.asarray(environment.fk(delta)).reshape(3); minus = np.asarray(environment.fk(-delta)).reshape(3)
            second.append({"beta_axis": axis + 1, "radius_deg": radius_deg, "symmetric_second_response_mm": float(np.linalg.norm(plus + minus - 2 * zero) * 1000.0), "linear_response_mm": float(np.linalg.norm(j0 @ delta) * 1000.0)})
    _write_parquet(pd.DataFrame(second), stage / "zero_second_order_response.parquet")
    pools = []
    power = min(10, int(tip["proposal_power_per_radius"])) if smoke else int(tip["proposal_power_per_radius"])
    for pool_id, seed in zip(("a", "b"), tip["proposal_seeds"], strict=True):
        parts = []
        for radius in tip["weighted_radii_deg"]:
            beta = weighted_sobol_beta_ball(power=power, seed=int(seed) + int(round(float(radius) * 100)), maximum_weighted_radius_deg=float(radius), beta_weights=config["candidate_solver"]["beta_weights"], beta_bounds_rad=np.asarray(environment.bounds))
            xyz = np.asarray(environment.fk(beta)).reshape(-1, 3); frame = add_tip_coordinates(xyz, zero_x_m=zero[0]); frame["pool_id"] = pool_id; frame["weighted_radius_limit_deg"] = float(radius); frame["proposal_ordinal"] = np.arange(len(frame)); parts.append(frame)
        pool = pd.concat(parts, ignore_index=True); pools.append(pool); _write_parquet(pool, stage / f"local_fk_proposal_{pool_id}.parquet")
    common = common_zero_tip_support(pools[0].loc[:, XYZ_COLUMNS].to_numpy(float), pools[1].loc[:, XYZ_COLUMNS].to_numpy(float), zero_x_m=zero[0], policy=policy)
    intervals = zero_connected_radial_intervals(common, policy=policy); _write_parquet(common, stage / "common_axis_aware_support.parquet"); _write_parquet(intervals, stage / "zero_connected_radial_intervals.parquet")
    profiles = []
    for erosion in tip["profile_erosions_mm"]:
        profile = fit_sqrt_u_tip_profile(intervals, erosion_mm=float(erosion)) if len(intervals) else pd.DataFrame(columns=["u_center_mm", "inner_radius_mm", "outer_radius_mm", "sqrt_u_mm", "erosion_mm", "profile_id"])
        profiles.append(profile); _write_parquet(profile, stage / f"tip_profile_erode_{float(erosion):g}mm.parquet")
    return _seal_gate(output_root, config, "zero_tip_discovery", {"status": "complete" if len(intervals) else "no_common_support", "proposal_power_per_radius": power, "proposal_row_count_per_pool": [len(p) for p in pools], "common_voxel_count": len(common), "connected_u_bin_count": len(intervals), "axis_core_radius_mm": policy.axis_core_radius_mm, "angular_arc_mm": policy.angular_arc_mm, "proposal_beta_used_as_label_or_hint": False})


def _old_supervision(config: Mapping[str, Any]) -> pd.DataFrame: return pd.read_parquet(_upstream_path(config, "quotient_supervision"))


def _teacher_for_targets(config: Mapping[str, Any], targets: pd.DataFrame, stage: Path, *, smoke: bool, anchors: pd.DataFrame | None = None, h_mm: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if targets.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    candidates = retry17._solve_candidates(config, targets, stage, seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]), smoke=smoke)
    legal = legal_candidate_clusters(candidates, residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]), cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]), weights=config["candidate_solver"]["beta_weights"])
    missing = targets[~targets["target_id"].isin(legal["target_id"] if len(legal) else [])]
    if len(missing):
        difficult = retry17._solve_candidates(config, missing, stage, seed_budget=int(config["candidate_solver"]["difficult_seed_budget"]), smoke=smoke)
        candidates = pd.concat([candidates, difficult], ignore_index=True, sort=False).drop_duplicates(["target_id", "candidate_id"], keep="last")
    legal = legal_candidate_clusters(candidates, residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]), cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]), weights=config["candidate_solver"]["beta_weights"])
    anchors = pd.DataFrame() if anchors is None else anchors.copy(); combined = candidates
    graph_targets = targets[targets["target_id"].isin(legal["target_id"] if len(legal) else [])].copy(); old_ids: set[str] = set()
    if len(anchors):
        combined = pd.concat([retry17._fixed_candidate_rows(anchors), candidates], ignore_index=True, sort=False)
        graph_targets = pd.concat([anchors, graph_targets], ignore_index=True, sort=False).drop_duplicates("target_id")
        old_ids = set(anchors["target_id"].astype(str))
    if graph_targets.empty: return candidates, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    labels, edges, comparison = retry17._select_teacher(combined, graph_targets, old_ids, config, h_mm=h_mm)
    return candidates, labels, edges, comparison


def _nearest_anchors(old: pd.DataFrame, targets: pd.DataFrame, *, limit: int = 256, maximum_mm: float = 30.0) -> pd.DataFrame:
    if targets.empty: return old.head(0)
    distance = cKDTree(targets.loc[:, XYZ_COLUMNS]).query(old.loc[:, XYZ_COLUMNS], k=1)[0] * 1000.0
    result = old.loc[distance <= maximum_mm].copy()
    if len(result) > limit:
        result["_hash"] = result["target_id"].astype(str).map(lambda v: hashlib.sha256(v.encode()).hexdigest()); result = result.sort_values("_hash").head(limit).drop(columns="_hash")
    return result


def stage_profile_preflight(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["profile_preflight"]; tip = config["root_tip"]; annulus = pd.read_parquet(_upstream_path(config, "stable_core_profile")); old = _old_supervision(config); zero_x = float(config["omega600"]["zero_x_m"]); trials = []
    candidates_by_size = []
    for erosion in tip["profile_erosions_mm"]:
        path = output_root / STAGE_DIRS["zero_tip_discovery"] / f"tip_profile_erode_{float(erosion):g}mm.parquet"; profile = pd.read_parquet(path)
        if not len(profile): continue
        overlap = profile_overlap_metrics(profile, annulus); candidates_by_size.append((float(profile["u_center_mm"].max()), -float(erosion), profile, overlap))
    selected_profile = pd.DataFrame(); selected_id = ""
    for _size, _neg_erosion, profile, overlap in sorted(candidates_by_size, reverse=True, key=lambda item: (item[0], item[1])):
        overlap_pass = overlap["axial_overlap_mm"] >= float(tip["minimum_axial_overlap_mm"]) and overlap["radial_overlap_p50_mm"] >= float(tip["minimum_radial_overlap_p50_mm"])
        count = min(64, int(tip["preflight_target_count"])) if smoke else int(tip["preflight_target_count"])
        power = max(6, int(math.ceil(math.log2(count)))); probes = sample_tip_volume(profile, power=power, seed=int(tip["preflight_seed"]), zero_x_m=zero_x, pool_id=f"preflight_{profile['profile_id'].iloc[0]}").head(count).copy()
        probes["target_role"] = "tip_preflight"; probes["domain_class"] = "zero_tip"; probes["circle_id"] = ""; probes["annular_coverage_eligible"] = False; probes["mandatory"] = False
        anchors = _nearest_anchors(old, probes); bank, labels, edges, comparison = _teacher_for_targets(config, probes, stage / str(profile["profile_id"].iloc[0]), smoke=smoke, anchors=anchors)
        new_ids = set(probes["target_id"]); accepted = labels[labels["target_id"].isin(new_ids)] if len(labels) else labels; acceptance = accepted["target_id"].nunique() / len(probes) if len(probes) else 0.0
        edge_metric = teacher_edge_metrics(labels, edges, weights=config["candidate_solver"]["beta_weights"]) if len(labels) and len(edges) else {"weighted_p95_deg": math.inf, "raw_gt7_rate": 1.0}
        trial = {"profile_id": str(profile["profile_id"].iloc[0]), **overlap, "overlap_pass": overlap_pass, "target_count": len(probes), "accepted_target_count": int(accepted["target_id"].nunique()) if len(accepted) else 0, "acceptance_rate": acceptance, **{f"edge_{k}": v for k, v in edge_metric.items()}}
        trial["preflight_pass"] = bool(overlap_pass and acceptance >= 0.95 and edge_metric["weighted_p95_deg"] <= 3.0 and edge_metric["raw_gt7_rate"] <= 0.02); trials.append(trial)
        _write_parquet(probes, stage / f"{trial['profile_id']}_targets.parquet"); _write_parquet(bank, stage / f"{trial['profile_id']}_candidate_bank.parquet"); _write_parquet(labels, stage / f"{trial['profile_id']}_labels.parquet"); _write_parquet(comparison, stage / f"{trial['profile_id']}_teacher_sweep.parquet")
        if trial["preflight_pass"]: selected_profile = profile; selected_id = trial["profile_id"]; break
    comparison = pd.DataFrame(trials); _write_parquet(comparison, stage / "profile_preflight_comparison.parquet"); _write_parquet(selected_profile, stage / "frozen_tip_profile.parquet")
    return _seal_gate(output_root, config, "profile_preflight", {"status": "green" if len(selected_profile) else "data_red_no_teacher_feasible_profile", "profile_selected": bool(len(selected_profile)), "selected_profile_id": selected_id, "trial_count": len(comparison), "data_track_may_continue": bool(len(selected_profile)), "student_track_continues_regardless": True})


def stage_geometric_budget(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["geometric_budget"]; profile = pd.read_parquet(output_root / STAGE_DIRS["profile_preflight"] / "frozen_tip_profile.parquet"); tip = config["root_tip"]
    if profile.empty:
        _write_parquet(pd.DataFrame(), stage / "geometric_budget_curve.parquet"); _write_parquet(pd.DataFrame(), stage / "selected_tip_targets.parquet"); _write_parquet(pd.DataFrame(), stage / "tip_evaluation_probes.parquet")
        return _seal_gate(output_root, config, "geometric_budget", {"status": "profile_not_available", "geometric_resource_feasible": False, "scientific_data_red": False})
    pool_power = min(11, int(tip["target_pool_power"])) if smoke else int(tip["target_pool_power"]); eval_power = min(11, int(tip["evaluation_probe_power"])) if smoke else int(tip["evaluation_probe_power"])
    pool = sample_tip_volume(profile, power=pool_power, seed=int(tip["target_pool_seed"]), zero_x_m=float(config["omega600"]["zero_x_m"]), pool_id="retry18_tip_target_only")
    probes = sample_tip_volume(profile, power=eval_power, seed=int(tip["evaluation_probe_seed"]), zero_x_m=float(config["omega600"]["zero_x_m"]), pool_id="retry18_tip_evaluation")
    old = _old_supervision(config); u = 1000 * (float(config["omega600"]["zero_x_m"]) - old["x_m"].to_numpy(float)); inner, outer = interpolate_tip_profile(profile, np.clip(u, 0, None)); rho = 1000 * np.hypot(old["y_m"], old["z_m"]); inside = (u >= 0) & (u <= profile["u_center_mm"].max()) & (rho >= inner) & (rho <= outer); existing = old.loc[inside, XYZ_COLUMNS].to_numpy(float)
    cap = min(500, int(tip["target_cap"])) if smoke else int(tip["target_cap"]); selected, batches = progressive_farthest_fill(pool, existing, maximum_new_points=cap, minimum_separation_mm=float(tip["minimum_separation_mm"]), batch_size=100 if smoke else int(tip["batch_size"]), target_fill_mm=0.0)
    curve = []
    steps = sorted(set([min(len(selected), n) for n in range(500, cap + 1, 500)] + [len(selected)]))
    required = None
    for count in steps:
        labels = pd.concat([old.loc[inside, XYZ_COLUMNS], selected.head(count).loc[:, XYZ_COLUMNS]], ignore_index=True)
        metric = coverage_distance(probes, labels); curve.append({"new_target_count": count, **metric})
        if required is None and metric["p95_mm"] <= float(tip["fill_p95_maximum_mm"]): required = count
    curve_frame = pd.DataFrame(curve); feasible = required is not None and required <= int(tip["target_cap"])
    chosen_count = int(required) if feasible else min(len(selected), int(tip["target_cap"])); chosen = selected.head(chosen_count).copy(); chosen["target_role"] = "zero_tip_fill"; chosen["domain_class"] = "zero_tip"; chosen["circle_id"] = ""; chosen["annular_coverage_eligible"] = False; chosen["mandatory"] = False; chosen["tip_coverage_eligible"] = True
    _write_parquet(pool, stage / "tip_target_only_pool.parquet"); _write_parquet(probes, stage / "tip_evaluation_probes.parquet"); _write_parquet(curve_frame, stage / "geometric_budget_curve.parquet"); _write_parquet(batches, stage / "farthest_fill_batches.parquet"); _write_parquet(chosen, stage / "selected_tip_targets.parquet")
    return _seal_gate(output_root, config, "geometric_budget", {"status": "green" if feasible else "objective_resource_infeasible", "geometric_resource_feasible": feasible, "required_target_count": required, "registered_cap": int(tip["target_cap"]), "selected_target_count": len(chosen), "scientific_data_red": False})


def stage_root_teacher(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["root_teacher"]; targets = pd.read_parquet(output_root / STAGE_DIRS["geometric_budget"] / "selected_tip_targets.parquet"); old = _old_supervision(config)
    if targets.empty:
        for name in ("root_candidate_bank", "root_canonical_labels", "tip_annulus_overlap_edges", "axis_branch_ambiguity", "unresolved_root_targets"): _write_parquet(pd.DataFrame(), stage / f"{name}.parquet")
        return _seal_gate(output_root, config, "root_teacher", {"status": "not_run_resource_or_profile_unavailable", "accepted_tip_count": 0})
    profile = pd.read_parquet(output_root / STAGE_DIRS["profile_preflight"] / "frozen_tip_profile.parquet"); u = 1000 * (float(config["omega600"]["zero_x_m"]) - old["x_m"].to_numpy(float)); inner, outer = interpolate_tip_profile(profile, np.clip(u, 0, None)); rho = 1000 * np.hypot(old["y_m"], old["z_m"]); inside = (u >= 0) & (u <= profile["u_center_mm"].max()) & (rho >= inner - 10) & (rho <= outer + 10); anchors = old.loc[inside].copy()
    bank, labels, edges, comparison = _teacher_for_targets(config, targets, stage, smoke=smoke, anchors=anchors, h_mm=float(config["root_tip"]["fill_p95_maximum_mm"])); new_ids = set(targets["target_id"]); selected = labels[labels["target_id"].isin(new_ids)].copy(); unresolved = targets[~targets["target_id"].isin(selected["target_id"] if len(selected) else [])]
    overlap_edges = edges[(edges["left_target_id"].isin(new_ids)) ^ (edges["right_target_id"].isin(new_ids))].copy() if len(edges) else pd.DataFrame()
    ambiguity = comparison[[c for c in comparison.columns if c in ("teacher_id", "old_new_conflict_count", "old_new_conflict_rate", "raw_gt7_rate", "weighted_p95_deg")]].copy() if len(comparison) else pd.DataFrame()
    _write_parquet(bank, stage / "root_candidate_bank.parquet"); _write_parquet(selected, stage / "root_canonical_labels.parquet"); _write_parquet(overlap_edges, stage / "tip_annulus_overlap_edges.parquet"); _write_parquet(ambiguity, stage / "axis_branch_ambiguity.parquet"); _write_parquet(unresolved, stage / "unresolved_root_targets.parquet"); _write_parquet(comparison, stage / "root_teacher_sweep.parquet")
    acceptance = len(selected) / len(targets); return _seal_gate(output_root, config, "root_teacher", {"status": "complete" if acceptance >= .95 else "data_red", "target_count": len(targets), "accepted_tip_count": len(selected), "acceptance_rate": acceptance, "unresolved_count": len(unresolved), "old_annular_interior_overwritten": False, "boundary_reselection_registered_only_on_raw_gt5": True})


def _split_role(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    zero_x = float(config["omega600"]["zero_x_m"]); size = float(config["data_gate"]["macroblock_mm"]); seed = int(config["data_gate"]["split_seed"]); xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float); block = np.floor(np.column_stack([1000 * (zero_x - xyz[:, 0]), 1000 * xyz[:, 1], 1000 * xyz[:, 2]]) / size).astype(int); result=[]
    for row in block:
        value = int.from_bytes(hashlib.sha256(f"{seed}:{row[0]}:{row[1]}:{row[2]}".encode()).digest()[:8], "big") / 2**64; result.append("train" if value < .8 else "validation" if value < .9 else "test")
    return pd.Series(result, index=frame.index)


def _density_weight(frame: pd.DataFrame, clip: Sequence[float]) -> np.ndarray:
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(float); k=min(17,len(xyz)); distance=cKDTree(xyz).query(xyz,k=k)[0]; radius=(distance[:,-1] if np.ndim(distance)==2 else distance)*1000; weight=np.maximum(radius,1e-6)**3; weight/=np.median(weight); return np.clip(weight,float(clip[0]),float(clip[1]))


def _parity_compatibility(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows=[]
    for seam, coord, odd in (("y", "y_m", ("beta1_rad","beta3_rad","beta5_rad")), ("z", "z_m", ("beta2_rad","beta4_rad","beta6_rad"))):
        exact=frame[np.abs(frame[coord].to_numpy(float)) <= 1e-12]
        for row in exact.itertuples(index=False):
            maximum=max(abs(float(getattr(row,name))) for name in odd); rows.append({"target_id": str(row.target_id), "seam": seam, "maximum_forbidden_beta_rad": maximum, "tolerance_rad": tolerance, "compatible": maximum <= tolerance})
    return pd.DataFrame(rows)


def stage_unified_dataset(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["unified_dataset"]; environment = _environment(config); old = _old_supervision(config).copy(); new_targets = pd.read_parquet(output_root / STAGE_DIRS["geometric_budget"] / "selected_tip_targets.parquet"); new_labels = pd.read_parquet(output_root / STAGE_DIRS["root_teacher"] / "root_canonical_labels.parquet")
    if len(new_labels):
        metadata = new_targets[["target_id", *XYZ_COLUMNS, "u_mm", "rho_mm", "target_role", "domain_class", "circle_id", "annular_coverage_eligible", "tip_coverage_eligible"]]; new = new_labels.drop(columns=[c for c in metadata if c != "target_id" and c in new_labels], errors="ignore").merge(metadata,on="target_id",how="inner",validate="one_to_one")
    else: new = old.head(0).copy()
    old["tip_coverage_eligible"] = old.get("tip_coverage_eligible", False); old["supervision_eligible"] = True; new["supervision_eligible"] = True
    # Historical connectors are judged per row and promoted only when their
    # own numerical evidence is valid; no role-wide exclusion remains.
    old_labels_all = pd.read_parquet(_upstream_path(config, "quotient_labels")); old_targets_all = pd.read_parquet(_upstream_path(config, "target_registry")); connector_targets = old_targets_all[old_targets_all["target_role"].eq("root_connector")].drop_duplicates("target_id"); connector_labels = old_labels_all[old_labels_all["target_id"].isin(connector_targets["target_id"])]
    classified = classify_root_connectors(connector_labels, connector_targets, residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]), beta_bounds_rad=np.asarray(environment.bounds)); _write_parquet(classified, stage / "root_connector_classification.parquet")
    promoted = classified[classified["supervision_eligible"]].copy()
    if len(promoted): promoted["tip_coverage_eligible"] = True; promoted["annular_coverage_eligible"] = False
    quotient = pd.concat([old, new, promoted], ignore_index=True, sort=False).drop_duplicates("target_id", keep="first")
    quotient["split_role"] = quotient.get("split_role", pd.Series(index=quotient.index,dtype=str)); missing=quotient["split_role"].isna(); quotient.loc[missing,"split_role"]=_split_role(quotient.loc[missing],config)
    quotient["density_weight"] = _density_weight(quotient, config["data_gate"]["density_weight_clip"]); quotient["sample_weight"] = quotient["density_weight"]
    jacobians=np.asarray([np.asarray(environment.jacobian(beta)).reshape(-1) for beta in quotient.loc[:,BETA_COLUMNS].to_numpy(float)])
    for index, column in enumerate(JACOBIAN_COLUMNS):
        quotient.loc[:, column] = jacobians[:, index]
    quotient["record_id"]=quotient["target_id"].astype(str); quotient["kind"]="static"; quotient["chart_id"]="retry18_unified_quotient"; quotient["is_primary"]=True
    new_for_expand = quotient[quotient["target_id"].isin(set(new["target_id"]) | set(promoted["target_id"]))].copy()
    if len(new_for_expand):
        labels_expand=new_for_expand[["target_id",*BETA_COLUMNS]]; targets_expand=new_for_expand[["target_id",*XYZ_COLUMNS,"target_role","domain_class","circle_id","annular_coverage_eligible"]]; expanded_new,rejected=expand_annular_labels(labels_expand,targets_expand,environment,residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]))
        split=quotient.set_index("target_id")["split_role"]; weight=quotient.set_index("target_id")["sample_weight"]; expanded_new["split_role"]=expanded_new["target_id"].map(split); expanded_new["sample_weight"]=expanded_new["target_id"].map(weight)
    else: expanded_new=pd.DataFrame(); rejected=pd.DataFrame()
    full_old=pd.read_parquet(_upstream_path(config,"full_g4_dataset")); full=pd.concat([full_old,expanded_new],ignore_index=True,sort=False).drop_duplicates(list(XYZ_COLUMNS),keep="first")
    compatibility=_parity_compatibility(quotient,float(config["student"]["parity_label_tolerance_rad"])); _write_parquet(compatibility,stage/"parity_label_compatibility.parquet"); _write_parquet(quotient,stage/"unified_quotient_supervision.parquet"); _write_parquet(full,stage/"unified_full_g4_dataset.parquet"); _write_parquet(rejected,stage/"symmetry_expansion_rejected.parquet")
    zero=np.asarray(environment.fk(np.zeros(6))).reshape(3); nonzero=quotient[np.linalg.norm(quotient.loc[:,XYZ_COLUMNS].to_numpy(float)-zero,axis=1)>1e-12]; nearest=float(cKDTree(nonzero.loc[:,XYZ_COLUMNS]).query(zero)[0]*1000)
    tip_probes=pd.read_parquet(output_root/STAGE_DIRS["geometric_budget"] / "tip_evaluation_probes.parquet"); tip_eligible=quotient.get("tip_coverage_eligible",pd.Series(False,index=quotient.index,dtype=bool)).astype("boolean").fillna(False).to_numpy(bool); tip_labels=quotient.loc[tip_eligible]; fill=coverage_distance(tip_probes,tip_labels)
    graph_cutoff=float(config["data_gate"]["unified_graph_maximum_distance_mm"]); edges=geometric_graph(quotient[["target_id",*XYZ_COLUMNS]],k=12,maximum_distance_mm=graph_cutoff); lcc=graph_lcc_fraction(quotient[["target_id",*XYZ_COLUMNS]],edges); continuity=teacher_edge_metrics(quotient,edges,weights=config["candidate_solver"]["beta_weights"])
    parity_ok=bool(compatibility.empty or compatibility["compatible"].all()); green=bool(nearest<=10 and fill["p95_mm"]<=10 and lcc==1 and continuity["weighted_p95_deg"]<=3 and continuity["raw_gt7_rate"]<=.02 and rejected.empty)
    return _seal_gate(output_root,config,"unified_dataset",{"status":"green" if green else "data_red","data_axis_green":green,"nearest_zero_nonzero_mm":nearest,"tip_fill":fill,"root_to_annulus_lcc":lcc,"unified_graph_maximum_distance_mm":graph_cutoff,"label_continuity":continuity,"quotient_row_count":len(quotient),"full_g4_row_count":len(full),"parity_labels_compatible":parity_ok,"incompatible_exact_seam_count":int((~compatibility["compatible"]).sum()) if len(compatibility) else 0,"representation_axis_abstention_not_coverage_failure":True})


def _seam_panel(config: Mapping[str, Any], *, smoke: bool) -> pd.DataFrame:
    old=_old_supervision(config); anchors=[]; count=2 if smoke else int(config["seam_audit"]["anchor_count_per_seam"]); offsets=np.arange(-6,7,2) if smoke else np.arange(float(config["seam_audit"]["normal_minimum_mm"]),float(config["seam_audit"]["normal_maximum_mm"])+1e-9,float(config["seam_audit"]["normal_step_mm"]))
    for seam,coord in (("y","y_m"),("z","z_m")):
        candidates=old[np.abs(old[coord].to_numpy(float))<=1e-12].copy(); candidates["_hash"]=candidates["target_id"].astype(str).map(lambda v:hashlib.sha256(f"{config['seam_audit']['anchor_seed']}:{seam}:{v}".encode()).hexdigest())
        for anchor_index,row in candidates.sort_values("_hash").head(count).reset_index(drop=True).iterrows():
            for waypoint_index,offset in enumerate(offsets):
                record={"sweep_id":f"{seam}_{anchor_index:03d}","seam":seam,"anchor_target_id":row["target_id"],"normal_offset_mm":float(offset),"waypoint_index":waypoint_index,"target_id":f"retry18_seam_{seam}_{anchor_index:03d}_{waypoint_index:03d}","x_m":row["x_m"],"y_m":row["y_m"],"z_m":row["z_m"],"target_role":"evaluation_only_seam_sweep","domain_class":"stable_core","circle_id":"","annular_coverage_eligible":False,"mandatory":True}
                record[coord]=float(offset)/1000; anchors.append(record)
    return pd.DataFrame(anchors)


def _ordered_teacher(candidates: pd.DataFrame, order: Sequence[str], config: Mapping[str,Any]) -> pd.DataFrame:
    legal=legal_candidate_clusters(candidates,weights=config["candidate_solver"]["beta_weights"]); groups={t:legal[legal["target_id"].astype(str).eq(str(t))].sort_values("candidate_id").reset_index(drop=True) for t in order}
    if any(groups[t].empty for t in order): return pd.DataFrame()
    weights=np.asarray(config["candidate_solver"]["beta_weights"],float); lam=4.0; tau=float(config["graph_teacher"]["robust_pairwise_cap_deg"]); values=np.square(np.asarray(weighted_beta_rms_deg(groups[order[0]][list(BETA_COLUMNS)].to_numpy(float),np.zeros((len(groups[order[0]]),6)),weights))); backs=[]
    for left,right in zip(order[:-1],order[1:]):
        a=groups[left][list(BETA_COLUMNS)].to_numpy(float); b=groups[right][list(BETA_COLUMNS)].to_numpy(float); matrix=np.empty((len(a),len(b)))
        for i in range(len(a)):
            gap=np.asarray(weighted_beta_rms_deg(np.broadcast_to(a[i],b.shape),b,weights)); matrix[i]=np.minimum(gap**2,tau**2)
        total=values[:,None]+lam*matrix; back=np.argmin(total,axis=0); unary=np.square(np.asarray(weighted_beta_rms_deg(b,np.zeros_like(b),weights))); values=unary+total[back,np.arange(len(b))]; backs.append(back)
    state=int(np.argmin(values)); states=[state]
    for back in reversed(backs): states.append(int(back[states[-1]]))
    states.reverse(); return pd.concat([groups[t].iloc[[s]] for t,s in zip(order,states,strict=True)],ignore_index=True)


def stage_seam_causal_audit(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage=output_root/STAGE_DIRS["seam_causal_audit"]; panel=_seam_panel(config,smoke=smoke); _write_parquet(panel,stage/"seam_sweep_registry.parquet"); bank=retry17._solve_candidates(config,panel,stage,seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]),smoke=smoke); teacher_parts=[]
    for sweep_id,part in panel.groupby("sweep_id",sort=True):
        order=list(part.sort_values("waypoint_index")["target_id"].astype(str)); selected=_ordered_teacher(bank[bank["target_id"].isin(order)],order,config)
        if len(selected): selected=selected.merge(part[["target_id","sweep_id","seam","normal_offset_mm","waypoint_index",*XYZ_COLUMNS]],on="target_id",how="left"); teacher_parts.append(selected)
    teacher=pd.concat(teacher_parts,ignore_index=True,sort=False) if teacher_parts else pd.DataFrame(); _write_parquet(bank,stage/"teacher_seam_candidate_bank.parquet"); _write_parquet(teacher,stage/"teacher_seam_sweeps.parquet")
    import tensorflow as tf
    old_model=tf.keras.models.load_model(_upstream_path(config,"frozen_student"),compile=False); environment=_environment(config); zero=np.asarray(environment.fk(np.zeros(6))).reshape(3); xyz=panel.loc[:,XYZ_COLUMNS].to_numpy(float); raw=trajectory_evaluation.symmetry_prediction(old_model,xyz,zero); achieved=np.asarray(environment.fk(raw)).reshape(-1,3); audit=panel.copy(); audit["student_fk_residual_mm"]=np.linalg.norm(achieved-xyz,axis=1)*1000
    for i,name in enumerate(BETA_COLUMNS): audit[f"student_{name}"]=raw[:,i]
    teacher_lookup=teacher.set_index("target_id") if len(teacher) else pd.DataFrame(); teacher_beta=np.full_like(raw,np.nan)
    if len(teacher):
        for i,target_id in enumerate(panel["target_id"].astype(str)):
            if target_id in teacher_lookup.index: teacher_beta[i]=teacher_lookup.loc[target_id,list(BETA_COLUMNS)].to_numpy(float)
    for i,name in enumerate(BETA_COLUMNS): audit[f"teacher_{name}"]=teacher_beta[:,i]
    finite_teacher=np.isfinite(teacher_beta).all(axis=1); student_teacher=np.full(len(raw),np.nan)
    if finite_teacher.any(): student_teacher[finite_teacher]=np.asarray(weighted_beta_rms_deg(raw[finite_teacher],teacher_beta[finite_teacher],config["candidate_solver"]["beta_weights"]))
    audit["student_teacher_weighted_deg"]=student_teacher
    train=pd.read_parquet(_upstream_path(config,"quotient_supervision")); for_roles={role:cKDTree(train[train["split_role"].eq(role)].loc[:,XYZ_COLUMNS]) for role in ("train","validation")}
    for role,tree in for_roles.items(): audit[f"nearest_{role}_row_mm"]=tree.query(xyz,k=1)[0]*1000
    jac=np.asarray([np.asarray(environment.jacobian(beta)).reshape(3,6) for beta in np.nan_to_num(teacher_beta)]); audit["teacher_jacobian_spectral_norm_m_per_rad"]=[np.linalg.svd(j,compute_uv=False)[0] for j in jac]
    soft=[]
    for delta in config["seam_audit"]["soft_gate_delta_mm"]:
        gated=raw.copy(); gated[:,[0,2,4]]*=np.tanh(np.abs(xyz[:,1:2])*1000/float(delta)); gated[:,[1,3,5]]*=np.tanh(np.abs(xyz[:,2:3])*1000/float(delta)); residual=np.linalg.norm(np.asarray(environment.fk(gated)).reshape(-1,3)-xyz,axis=1)*1000; soft.append({"delta_mm":float(delta),"fk_p95_mm":percentile(residual,95),"fk_maximum_mm":float(np.max(residual))})
    _write_parquet(audit,stage/"student_teacher_seam_causal_panel.parquet"); _write_parquet(pd.DataFrame(soft),stage/"frozen_student_soft_gate_ablation.parquet")
    teacher_complete=len(teacher)==len(panel); return _seal_gate(output_root,config,"seam_causal_audit",{"status":"complete" if teacher_complete else "teacher_incomplete","panel_target_count":len(panel),"teacher_complete_count":len(teacher),"teacher_reference_complete":teacher_complete,"soft_gate_is_diagnostic_only":True})


def stage_parity_regularity(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage=output_root/STAGE_DIRS["parity_regularity"]; teacher=pd.read_parquet(output_root/STAGE_DIRS["seam_causal_audit"]/"teacher_seam_sweeps.parquet"); rows=[]
    if len(teacher):
        for (sweep,seam),part in teacher.groupby(["sweep_id","seam"],sort=True):
            odd=("beta1_rad","beta3_rad","beta5_rad") if seam=="y" else ("beta2_rad","beta4_rad","beta6_rad"); distance=np.abs(part["normal_offset_mm"].to_numpy(float)); mask=(distance>0)&(distance<=10)
            for component in odd:
                value=np.abs(part[component].to_numpy(float)); valid=mask&(value>1e-10)
                alpha=float(np.polyfit(np.log(distance[valid]),np.log(value[valid]),1)[0]) if valid.sum()>=3 else math.nan; rows.append({"sweep_id":sweep,"seam":seam,"component":component,"alpha":alpha,"point_count":int(valid.sum())})
    exponent=pd.DataFrame(rows); _write_parquet(exponent,stage/"teacher_odd_vanish_exponent.parquet"); finite=exponent[np.isfinite(exponent.get("alpha",pd.Series(dtype=float)))] if len(exponent) else exponent; median=float(finite["alpha"].median()) if len(finite) else math.nan; student_cfg=config["student"]; compatible=_gate(output_root,"unified_dataset")["parity_labels_compatible"]; authorized=bool(len(finite) and compatible and float(student_cfg["parity_exponent_minimum"])<=median<=float(student_cfg["parity_exponent_maximum"]))
    return _seal_gate(output_root,config,"parity_regularity",{"status":"authorized" if authorized else "not_authorized","parity_student_authorized":authorized,"teacher_alpha_median":median,"teacher_alpha_component_count":len(finite),"parity_labels_compatible":compatible})


def _student_frame(config: Mapping[str,Any], output_root:Path) -> pd.DataFrame:
    frame=pd.read_parquet(output_root/STAGE_DIRS["unified_dataset"]/"unified_quotient_supervision.parquet").copy(); frame["density_weight"]=frame.get("density_weight",frame["sample_weight"]); return frame


def _region_weights(frame:pd.DataFrame,config:Mapping[str,Any],enhanced:bool)->pd.DataFrame:
    result=frame.copy(); region=np.ones(len(result)); seam=np.minimum(np.abs(result["y_m"]),np.abs(result["z_m"]))*1000<=float(config["seam_audit"]["seam_band_mm"]); exact=(np.abs(result["y_m"])<=1e-12)|(np.abs(result["z_m"])<=1e-12); root=result.get("domain_class",pd.Series("",index=result.index)).astype(str).str.contains("tip")
    if enhanced: region[seam]*=float(config["student"]["seam_multiplier"]); region[root]*=float(config["student"]["root_tip_multiplier"]); region[exact]*=float(config["student"]["exact_seam_multiplier"])
    result["region_weight"]=region; lo,hi=map(float,config["data_gate"]["density_weight_clip"]); final=np.clip(result["density_weight"].to_numpy(float)*region,lo,hi); final/=np.mean(final); result["final_weight"]=final; result["sample_weight"]=final; return result


def _predict_model(kind:str,model:Any,xyz:np.ndarray,zero:np.ndarray)->np.ndarray: return np.asarray(model(np.asarray(xyz,np.float32),training=False),float) if kind=="parity" else trajectory_evaluation.symmetry_prediction(model,xyz,zero)


def _prepare_model_output(model_dir: Path, kind: str) -> None:
    """Respect each trainer's directory ownership contract."""
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if kind == "parity":
        model_dir.mkdir(exist_ok=False)


def _evaluate_model(model_id:str,kind:str,model:Any,validation:pd.DataFrame,old_shapes:pd.DataFrame,seam_panel:pd.DataFrame,environment:Any,zero:np.ndarray)->dict[str,Any]:
    record={"model_id":model_id,"model_kind":kind}; xyz=validation.loc[:,XYZ_COLUMNS].to_numpy(float); beta=_predict_model(kind,model,xyz,zero); residual=np.linalg.norm(np.asarray(environment.fk(beta)).reshape(-1,3)-xyz,axis=1)*1000; record.update({"validation_fk_p95_mm":percentile(residual,95),"validation_fk_maximum_mm":float(np.max(residual))})
    rect=[]
    for _,part in old_shapes[old_shapes["shape_class"].eq("rounded_rectangle")].groupby("trajectory_id",sort=True):
        pxyz=part.sort_values("waypoint_index").loc[:,XYZ_COLUMNS].to_numpy(float); pred=_predict_model(kind,model,pxyz,zero); rect.append(trajectory_evaluation.path_metrics(pxyz,pred,environment))
    for key in ("fk_p95_mm","fk_maximum_mm","path_step_excess_p99_mm","path_step_excess_maximum_mm","raw_step_gt7_rate"): record[f"rectangle_{key}"]=max(item[key] for item in rect) if rect else math.inf
    sxyz=seam_panel.loc[:,XYZ_COLUMNS].to_numpy(float); sbeta=_predict_model(kind,model,sxyz,zero); sres=np.linalg.norm(np.asarray(environment.fk(sbeta)).reshape(-1,3)-sxyz,axis=1)*1000; record["seam_fk_p95_mm"]=percentile(sres,95); record["seam_fk_maximum_mm"]=float(np.max(sres)); return record


def stage_student_ablation(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage=output_root/STAGE_DIRS["student_ablation"]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1": raise RuntimeError("retry18 Student requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    if tf.config.get_visible_devices("GPU"): raise RuntimeError("retry18 CPU Student sees GPU")
    environment=_environment(config); geometry=retry17._student_geometry(environment); zero=np.asarray(environment.fk(np.zeros(6))).reshape(3); base=_student_frame(config,output_root); old_shapes=pd.read_parquet(_upstream_path(config,"old_shape_waypoints")); seam=pd.read_parquet(output_root/STAGE_DIRS["seam_causal_audit"]/"seam_sweep_registry.parquet"); parity_authorized=_gate(output_root,"parity_regularity")["parity_student_authorized"]; student=config["student"]; steps=50 if smoke else int(student["maximum_steps"]); interval=10 if smoke else int(student["validation_interval"]); patience=2 if smoke else int(student["patience_intervals"]); radial_scale=data_driven_radial_scale_mm(base,quantile=float(student["radial_scale_quantile"]),safety_factor=float(student["radial_scale_safety_factor"])); _write_json(stage/"parity_model_contract.json",{"radial_scale_mm":radial_scale,"source":"1.05_times_q99_unified_supervision_rho","linear_parity_authorized":parity_authorized})
    spectral=np.asarray([np.linalg.svd(row.reshape(3,6),compute_uv=False)[0] for row in base.loc[:,JACOBIAN_COLUMNS].to_numpy(float)]); clip=float(np.quantile(spectral,float(student["jacobian_spectral_clip_quantile"]))); _write_json(stage/"jacobian_sensitivity.json",{"spectral_p95":percentile(spectral,95),"spectral_maximum":float(np.max(spectral)),"spectral_clip_q99":clip,"jacobian_units":"m_per_rad","task_scale_m":0.003})
    compatibility=pd.read_parquet(output_root/STAGE_DIRS["unified_dataset"]/"parity_label_compatibility.parquet"); incompatible=set(compatibility.loc[~compatibility["compatible"],"target_id"].astype(str)) if len(compatibility) else set(); parity_base=base[~base["target_id"].astype(str).isin(incompatible)].copy()
    model_specs=[("S0_baseline","standard",False,tuple(student["hidden_units"]),0.0),("S1_seam_root","standard",True,tuple(student["hidden_units"]),0.0),("S4_capacity_control","standard",False,tuple(student["capacity_control_hidden_units"]),0.0)]
    if parity_authorized:
        model_specs.append(("S2_parity","parity",True,tuple(student["hidden_units"]),0.0)); model_specs.extend((f"S3_parity_j{lam:g}","parity",True,tuple(student["hidden_units"]),float(lam)) for lam in student["jacobian_lambdas"])
    metrics=[]
    for model_id,kind,enhanced,hidden,jlam in model_specs:
        data=_region_weights(parity_base if kind=="parity" else base,config,enhanced); train=data[data["split_role"].eq("train")].copy(); valid=data[data["split_role"].eq("validation")].copy(); model_dir=stage/"models"/model_id; _prepare_model_output(model_dir,kind)
        if kind=="standard":
            result=train_workspace_student(train,valid,mode=RepresentationMode.XYZ_GLOBAL,geometry=geometry,config=WorkspaceStudentTrainingConfig(hidden_units=tuple(map(int,hidden)),learning_rate=float(student["learning_rate"]),max_steps=steps,validation_interval=interval,patience_intervals=patience,seed=int(student["primary_seed"]),beta_coordinate_weights=(4,4,2,2,1,1),beta_loss_only=True)); save_workspace_student_models(result.models,model_dir); model=result.models.global_model; history=result.history
        else:
            training=train.copy(); validation=valid.copy()
            if jlam>0:
                for frame in (training,validation):
                    norms=np.asarray([np.linalg.svd(row.reshape(3,6),compute_uv=False)[0] for row in frame.loc[:,JACOBIAN_COLUMNS].to_numpy(float)]); scale=np.minimum(1.0,clip/np.maximum(norms,1e-12)); frame.loc[:,JACOBIAN_COLUMNS]=frame.loc[:,JACOBIAN_COLUMNS].to_numpy(float)*scale[:,None]
            model,history=train_parity_smooth_student(training,validation,zero_x_m=zero[0],geometry=geometry,config=ParityStudentConfig(hidden_units=tuple(map(int,hidden)),learning_rate=float(student["learning_rate"]),maximum_steps=steps,validation_interval=interval,patience_intervals=patience,seed=int(student["primary_seed"]),axial_scale_mm=float(student["axial_scale_mm"]),radial_scale_mm=radial_scale,jacobian_lambda=jlam)); save_parity_student(model,model_dir/"parity.keras"); _write_json(model_dir/"model_manifest.json",{"schema_version":1,"model_kind":"parity","artifact":"parity.keras","parity_student_radial_scale_mm":radial_scale,"jacobian_lambda":jlam})
        _write_parquet(history,model_dir/"training_history.parquet"); _write_parquet(data[["target_id","density_weight","region_weight","final_weight","sample_weight"]],model_dir/"supervision_weights.parquet"); metrics.append(_evaluate_model(model_id,kind,model,valid,old_shapes,seam,environment,zero))
    result=pd.DataFrame(metrics); _write_parquet(result,stage/"primary_ablation_metrics.parquet"); return _seal_gate(output_root,config,"student_ablation",{"status":"complete","model_count":len(result),"parity_models_authorized":parity_authorized,"parity_student_radial_scale_mm":radial_scale,"s0_s4_complete":set(result["model_id"])>={"S0_baseline","S1_seam_root","S4_capacity_control"}})


def stage_student_selection(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage=output_root/STAGE_DIRS["student_selection"]; metrics=pd.read_parquet(output_root/STAGE_DIRS["student_ablation"]/"primary_ablation_metrics.parquet"); s0=metrics[metrics["model_id"].eq("S0_baseline")].iloc[0]; cfg=config["student"]
    metrics["seam_relative_reduction"] = 1 - metrics["seam_fk_p95_mm"] / float(s0["seam_fk_p95_mm"])
    metrics["raw_green"]=(metrics["seam_relative_reduction"]>=float(cfg["seam_p95_relative_reduction_minimum"]))&(metrics["rectangle_fk_p95_mm"]<=float(cfg["rectangle_fk_p95_maximum_mm"]))&(metrics["rectangle_fk_maximum_mm"]<=float(cfg["rectangle_fk_maximum_mm"]))&(metrics["rectangle_path_step_excess_p99_mm"]<=float(cfg["rectangle_step_excess_maximum_mm"]))&(metrics["rectangle_path_step_excess_maximum_mm"]<=float(cfg["rectangle_step_excess_maximum_mm"]))&(metrics["rectangle_raw_step_gt7_rate"]<=float(cfg["raw_step_gt7_rate_maximum"]))&(metrics["validation_fk_p95_mm"]<=float(s0["validation_fk_p95_mm"])*float(cfg["global_validation_s0_ratio_maximum"]))
    metrics["selection_score"]=metrics["seam_fk_p95_mm"]+metrics["rectangle_fk_p95_mm"]+2*metrics["rectangle_path_step_excess_maximum_mm"]+.25*metrics["validation_fk_p95_mm"]
    eligible=metrics[metrics["raw_green"]]; selected=(eligible if len(eligible) else metrics).sort_values(["selection_score","model_id"]).iloc[0]; _write_parquet(metrics,stage/"model_selection_metrics.parquet"); _write_json(stage/"selected_model.json",{"model_id":selected["model_id"],"model_kind":selected["model_kind"],"raw_green":bool(selected["raw_green"]),"selection_score":float(selected["selection_score"]),"selection_priority":"raw_seam_then_rectangles_then_global_then_dls2","dls2_used_as_primary_metric":False})
    return _seal_gate(output_root,config,"student_selection",{"status":"green" if bool(selected["raw_green"]) else "yellow_best_available","raw_student_axis_green":bool(selected["raw_green"]),"selected_model_id":str(selected["model_id"]),"selected_model_kind":str(selected["model_kind"]),"maximum_path_step_excess_is_blocking":True})


def _tree_manifest(root:Path)->dict[str,Any]: return {"schema_version":1,"artifacts":[{"path":str(p.relative_to(root)),"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file()]}


def stage_dataset_model_lock(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage=output_root/STAGE_DIRS["dataset_model_lock"]; selected=_read_json(output_root/STAGE_DIRS["student_selection"]/"selected_model.json"); quotient=output_root/STAGE_DIRS["unified_dataset"]/"unified_quotient_supervision.parquet"; full=output_root/STAGE_DIRS["unified_dataset"]/"unified_full_g4_dataset.parquet"; model_root=output_root/STAGE_DIRS["student_ablation"]/"models"/selected["model_id"]
    lock={"schema_version":1,"locked_before_heldout_generation":True,"quotient":{"path":str(quotient.relative_to(output_root)),"sha256":sha256_file(quotient)},"full_g4":{"path":str(full.relative_to(output_root)),"sha256":sha256_file(full)},"selected_model":selected,"model_root":str(model_root.relative_to(output_root)),"model_manifest":_tree_manifest(model_root)}; _write_json(stage/"dataset_model_lock.json",lock); return _seal_gate(output_root,config,"dataset_model_lock",{"status":"locked","dataset_model_locked":True,"selected_model_id":selected["model_id"],"dataset_sha256":lock["full_g4"]["sha256"]})


def _verify_lock(output_root:Path)->bool:
    lock=_read_json(output_root/STAGE_DIRS["dataset_model_lock"]/"dataset_model_lock.json"); return sha256_file(output_root/lock["quotient"]["path"])==lock["quotient"]["sha256"] and sha256_file(output_root/lock["full_g4"]["path"])==lock["full_g4"]["sha256"] and _tree_manifest(output_root/lock["model_root"])==lock["model_manifest"]


def _profile_trajectories(profile:pd.DataFrame,config:Mapping[str,Any],*,smoke:bool)->pd.DataFrame:
    if profile.empty:return pd.DataFrame()
    count=48 if smoke else int(config["trajectories"]["waypoint_count"]); u_max=float(profile["u_center_mm"].max()); u=np.linspace(max(1e-6,u_max/count),u_max,count); inner,outer=interpolate_tip_profile(profile,u); rows=[]; specs=[]
    for index,alpha in enumerate((.25,.5,.75,.5)):
        phi=(index+.5)*.5*np.pi/4; specs.append((f"retry18_profile_ray_{index}","profile_ray",alpha,lambda q,p=phi:np.full_like(q,p)))
    specs.append(("retry18_profile_spiral","profile_spiral",.5,lambda q: .25*np.pi+2*np.pi*q/u_max))
    for trajectory_id,shape,alpha,phi_fn in specs:
        rho=inner+alpha*(outer-inner); phi=phi_fn(u)
        for waypoint,(uu,rr,pp) in enumerate(zip(u,rho,phi,strict=True)): rows.append({"trajectory_id":trajectory_id,"shape_class":shape,"waypoint_index":waypoint,"x_m":float(config["omega600"]["zero_x_m"])-uu/1000,"y_m":rr*np.cos(pp)/1000,"z_m":rr*np.sin(pp)/1000,"profile_alpha":alpha})
    return pd.DataFrame(rows)


def stage_heldout_trajectories(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage=output_root/STAGE_DIRS["heldout_trajectories"]
    if not _verify_lock(output_root): raise RuntimeError("retry18 lock mutated before heldout generation")
    profile=pd.read_parquet(output_root/STAGE_DIRS["profile_preflight"]/"frozen_tip_profile.parquet"); root=_profile_trajectories(profile,config,smoke=smoke); annulus=pd.read_parquet(_upstream_path(config,"stable_core_profile")); dataset=pd.read_parquet(output_root/STAGE_DIRS["unified_dataset"]/"unified_full_g4_dataset.parquet"); selected,shapes,search=generate_maximal_shape_registry(annulus,dataset,zero_x_m=float(config["omega600"]["zero_x_m"]),center_seed=int(config["trajectories"]["heldout_seed"]),center_power=9 if smoke else 14,center_count=8 if smoke else 100,waypoint_count=48 if smoke else int(config["trajectories"]["waypoint_count"]),erosion_mm=15,support_maximum_mm=float(config["trajectories"]["support_maximum_mm"]),minimum_axial_span_mm=20 if smoke else 150)
    shapes["heldout_kind"]="postlock_maximal_shape"; root["heldout_kind"]="postlock_profile_curve"; all_points=pd.concat([root,shapes],ignore_index=True,sort=False); all_points["target_id"]=[f"{t}:{int(i):05d}" for t,i in zip(all_points["trajectory_id"],all_points["waypoint_index"],strict=True)]; _write_parquet(root,stage/"profile_trajectory_waypoints.parquet"); _write_parquet(selected,stage/"maximal_shape_registry.parquet"); _write_parquet(search,stage/"shape_search_candidates.parquet"); _write_parquet(all_points,stage/"heldout_waypoints.parquet")
    return _seal_gate(output_root,config,"heldout_trajectories",{"status":"complete","lock_verified_before_and_after":_verify_lock(output_root),"profile_curve_count":root["trajectory_id"].nunique() if len(root) else 0,"maximal_shape_count":shapes["trajectory_id"].nunique() if len(shapes) else 0,"generated_after_lock":True,"trajectory_labels_added_back_to_dataset":False,"profile_curves_constructed_inside_profile":True})


def _load_selected_model(output_root:Path)->tuple[str,Any]:
    selected=_read_json(output_root/STAGE_DIRS["student_selection"]/"selected_model.json"); root=output_root/STAGE_DIRS["student_ablation"]/"models"/selected["model_id"]
    if selected["model_kind"]=="parity": return "parity",load_parity_student(root/"parity.keras")
    import tensorflow as tf
    return "standard",tf.keras.models.load_model(root/"global.keras",compile=False)


def stage_trajectory_evaluation(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage=output_root/STAGE_DIRS["trajectory_evaluation"]
    if not _verify_lock(output_root): raise RuntimeError("retry18 lock mutated before evaluation")
    points=pd.read_parquet(output_root/STAGE_DIRS["heldout_trajectories"]/"heldout_waypoints.parquet"); bank=retry17._solve_candidates(config,points,stage,seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]),smoke=smoke,maximum_workers=int(config["runtime"]["trajectory_candidate_workers"]),work_namespace="trajectory_k16"); kind,model=_load_selected_model(output_root); environment=_environment(config); zero=np.asarray(environment.fk(np.zeros(6))).reshape(3); reports=[]; rows=[]
    for trajectory_id,part in points.groupby("trajectory_id",sort=True):
        ordered=part.sort_values("waypoint_index"); ids=list(ordered["target_id"].astype(str)); teacher=_ordered_teacher(bank[bank["target_id"].isin(ids)],ids,config); xyz=ordered.loc[:,XYZ_COLUMNS].to_numpy(float); complete=len(teacher)==len(ordered); teacher_beta=teacher.loc[:,BETA_COLUMNS].to_numpy(float) if complete else np.full((len(xyz),6),np.nan); raw=_predict_model(kind,model,xyz,zero); dls=trajectory_evaluation.two_step_dls(environment,raw,xyz,zero_xyz=zero)
        record={"trajectory_id":trajectory_id,"shape_class":ordered["shape_class"].iloc[0],"waypoint_count":len(ordered),"teacher_complete":complete}
        for prefix,beta in (("teacher",teacher_beta),("raw_student",raw),("dls2",dls)):
            metric=trajectory_evaluation.path_metrics(xyz,beta,environment,closed=ordered["shape_class"].iloc[0] not in ("profile_ray",)); record.update({f"{prefix}_{k}":v for k,v in metric.items()})
        raw_green=bool(record["raw_student_fk_p95_mm"]<=10 and record["raw_student_fk_maximum_mm"]<=15 and record["raw_student_path_step_excess_p99_mm"]<=5 and record["raw_student_path_step_excess_maximum_mm"]<=5 and record["raw_student_raw_step_gt7_rate"]<=.02); dls_green=bool(record["dls2_fk_p95_mm"]<=float(config["trajectories"]["dls2_fk_p95_maximum_mm"])); record["raw_student_green"]=raw_green; record["dls2_green"]=dls_green; reports.append(record)
        evaluated=ordered.copy()
        for i,name in enumerate(BETA_COLUMNS): evaluated[f"teacher_{name}"]=teacher_beta[:,i]; evaluated[f"raw_student_{name}"]=raw[:,i]; evaluated[f"dls2_{name}"]=dls[:,i]
        rows.append(evaluated)
    report=pd.DataFrame(reports); evaluated=pd.concat(rows,ignore_index=True,sort=False) if rows else pd.DataFrame(); _write_parquet(bank,stage/"trajectory_candidate_bank.parquet"); _write_parquet(report,stage/"trajectory_report.parquet"); _write_parquet(evaluated,stage/"trajectory_waypoint_evaluation.parquet")
    return _seal_gate(output_root,config,"trajectory_evaluation",{"status":"complete","trajectory_count":len(report),"teacher_complete_rate":float(report["teacher_complete"].mean()) if len(report) else 0,"raw_student_axis_green":bool(len(report) and report["raw_student_green"].all()),"dls2_axis_green":bool(len(report) and report["dls2_green"].all()),"dls2_is_real_closed_loop":False,"raw_red_not_overridden_by_dls2":True})


def _artifact_manifest(output_root:Path)->dict[str,Any]:
    return {"schema_version":1,"experiment_id":EXPERIMENT_ID,"scientific_source_fixed_point":_git_sha(),"artifacts":[{"path":str(p.relative_to(output_root)),"sha256":sha256_file(p),"bytes":p.stat().st_size} for name in STAGE_ORDER[:-1] for p in sorted((output_root/STAGE_DIRS[name]).rglob("*")) if p.is_file() and "_work" not in p.parts]}


def stage_summary(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage=output_root/STAGE_DIRS["summary"]; operational=all(_complete(output_root,name) for name in STAGE_ORDER[:-1]); data=_gate(output_root,"unified_dataset"); student=_gate(output_root,"student_selection"); dls=_gate(output_root,"trajectory_evaluation"); causal=_gate(output_root,"seam_causal_audit"); regularity=_gate(output_root,"parity_regularity"); resource=_gate(output_root,"geometric_budget")
    gate={"status":"complete" if operational else "incomplete","operational_completion":operational,"artifact_completeness":operational,"diagnostic_smoke":bool(smoke),"data_axis":{"green":data.get("data_axis_green",False),"nearest_zero_nonzero_mm":data.get("nearest_zero_nonzero_mm"),"tip_fill":data.get("tip_fill"),"lcc":data.get("root_to_annulus_lcc")},"raw_student_axis":{"green":student.get("raw_student_axis_green",False),"selected_model_id":student.get("selected_model_id")},"dls2_axis":{"green":dls.get("dls2_axis_green",False),"real_closed_loop":False},"resource_feasibility":{"geometric_resource_feasible":resource.get("geometric_resource_feasible",False),"status":resource.get("status")},"teacher_seam_reference_complete":causal.get("teacher_reference_complete",False),"parity_regularity_authorized":regularity.get("parity_student_authorized",False),"causal_chain_claimed":bool(causal.get("teacher_reference_complete",False) and regularity.get("parity_student_authorized",False) and student.get("raw_student_axis_green",False)),"proposal_beta_used_as_label_or_hint":False,"downstream_authorization":False,"formal_authorization":False,"deployment_authorization":False,"full_workspace_authorization":False,"continuous_workspace_authorization":False,"tension_executed":False}
    (stage/"retry18_summary.html").parent.mkdir(parents=True,exist_ok=True); (stage/"retry18_summary.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>retry18</title><h1>retry18 分轴摘要</h1><pre>"+json.dumps(gate,ensure_ascii=False,indent=2,default=_json_default)+"</pre></html>",encoding="utf-8"); sealed=_seal_gate(output_root,config,"summary",gate); _write_json(stage/"artifact_manifest.json",_artifact_manifest(output_root)); _seal_stage(output_root,"summary"); _write_json(output_root/"progress.json",{"status":"complete","phase":"summary","completed":len(STAGE_ORDER),"total":len(STAGE_ORDER),"observed_at_unix":time.time()}); return sealed


STAGE_RUNNERS:dict[str,Callable[...,dict[str,Any]]]={
    "objective_baseline":stage_objective_baseline,"zero_tip_discovery":stage_zero_tip_discovery,"profile_preflight":stage_profile_preflight,"geometric_budget":stage_geometric_budget,"root_teacher":stage_root_teacher,"unified_dataset":stage_unified_dataset,"seam_causal_audit":stage_seam_causal_audit,"parity_regularity":stage_parity_regularity,"student_ablation":stage_student_ablation,"student_selection":stage_student_selection,"dataset_model_lock":stage_dataset_model_lock,"heldout_trajectories":stage_heldout_trajectories,"trajectory_evaluation":stage_trajectory_evaluation,"summary":stage_summary,
}


def run(config_path:str|Path,output_root:str|Path,binding_sha:str,*,smoke:bool=False,stage:str|None=None,validate_stage:str|None=None)->dict[str,Any]:
    config=load_config(config_path); output=Path(output_root).resolve(); _ensure_identity(config,output,binding_sha,smoke=smoke)
    if validate_stage:
        if not _complete(output,validate_stage): raise RuntimeError(f"retry18 stage incomplete or mutated: {validate_stage}")
        return _gate(output,validate_stage)
    if stage:
        position=STAGE_ORDER.index(stage)
        if position and not _complete(output,STAGE_ORDER[position-1]): raise RuntimeError("retry18 previous stage incomplete")
        return _gate(output,stage) if _complete(output,stage) else STAGE_RUNNERS[stage](config,output,smoke=smoke)
    result={}
    for index,name in enumerate(STAGE_ORDER): _progress(output,name,index); result=_gate(output,name) if _complete(output,name) else STAGE_RUNNERS[name](config,output,smoke=smoke)
    return result


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-root",required=True); parser.add_argument("--binding-sha",required=True); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--stage",choices=STAGE_ORDER); parser.add_argument("--validate-stage",choices=STAGE_ORDER); return parser.parse_args()


def main()->int:
    args=parse_args(); gate=run(args.config,args.output_root,args.binding_sha,smoke=args.smoke,stage=args.stage,validate_stage=args.validate_stage); print(json.dumps(gate,sort_keys=True,default=_json_default)); return 0


if __name__=="__main__": raise SystemExit(main())
