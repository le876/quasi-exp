#!/usr/bin/env python3
"""Execute retry19 full signed-workspace and Direct Student diagnostics.

The runner is intentionally stage-explicit.  It does not start an experiment
when imported, and every later stage consumes hash-sealed outputs from its
immediate predecessor.
"""

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
from scipy.spatial import cKDTree
import yaml

import run_bacra_v14_3r_retry17_continuity as retry17
import run_bacra_v14_3r_retry18_zero_tip_parity as retry18
from quasi_exp.teacher.exploration_qualification import weighted_beta_rms_deg
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.retry12_symmetry import (
    BETA_COLUMNS,
    XYZ_COLUMNS,
    sobol_beta_samples,
    stable_id,
)
from quasi_exp.teacher.retry17_continuity_fill import teacher_edge_metrics
from quasi_exp.teacher.retry17_trajectories import plane_basis, unit_shape
from quasi_exp.teacher.retry18_root_tip import weighted_sobol_beta_ball
from quasi_exp.teacher.retry15_canonical_graph import legal_candidate_clusters
from quasi_exp.teacher.retry19_direct_student import (
    DirectStudentConfig,
    aligned_q0_augmented_and_f0_rows,
    classify_causal_evidence,
    load_direct_student,
    train_direct_student,
    unified_split_registry,
)
from quasi_exp.teacher.retry19_fullspace import (
    FullspaceCoveragePolicy,
    Retry19TeacherPolicy,
    axial_slice_coverage,
    build_mixed_resolution_cell_registry,
    classify_old_anchors,
    data_gate,
    deterministic_path_query_registry,
    geometric_zero_attachment,
    largest_unserved_component_fraction,
    mark_served_cells,
    mixed_resolution_coverage_metrics,
    mixed_resolution_26_edges,
    mixed_resolution_face_edges,
    overlap_compatibility,
    path_query_metrics,
    select_fullspace_graph_teacher,
    target_knn_edges,
    teacher_zero_attachment,
)


EXPERIMENT_ID = "bacra_v14_3r_retry19_fullspace_direct_student"
STAGE_DIRS = {
    "fullspace_discovery": "00_fullspace_discovery",
    "causal_controls": "01_causal_controls",
    "candidate_pilot": "02_candidate_pilot",
    "continuation_authorization": "03_continuation_authorization",
    "adaptive_fill": "04_adaptive_fullspace_fill",
    "graph_teacher": "05_fullspace_graph_teacher",
    "unified_dataset": "06_unified_dataset",
    "student_ablations": "07_direct_student_ablations",
    "student_lock": "08_three_seed_selection_and_lock",
    "postlock_trajectories": "09_postlock_trajectories",
    "trajectory_teacher": "10_trajectory_teacher",
    "trajectory_evaluation": "11_raw_and_dls2_evaluation",
    "summary": "12_four_axis_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)
_ACTIVE_SCIENTIFIC_SOURCE_SHA: str | None = None


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    return resolved.parent.parent if resolved.parent.name == ".worktrees" else resolved


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _checkout_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()


def _git_sha() -> str:
    """Return the scientific source, not the later registry-binding commit."""

    return _ACTIVE_SCIENTIFIC_SOURCE_SHA or _checkout_sha()


def _config_sha(config: Mapping[str, Any]) -> str:
    return sha256_file(Path(str(config["config_path"])))


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    spec = config["upstream"][key]
    path = Path(str(spec["path"]))
    if path.is_absolute():
        return path
    root_key = "retry16_root" if spec.get("root") == "retry16" else "retry18_root"
    return Path(str(config["upstream"][root_key])) / path


def _environment(config: Mapping[str, Any]) -> Any:
    return retry17._environment(config)


def _coverage_policy(config: Mapping[str, Any]) -> FullspaceCoveragePolicy:
    gate, workspace = config["coverage_gate"], config["workspace"]
    return FullspaceCoveragePolicy(
        service_radius_mm=float(workspace["service_radius_mm"]),
        root_registration_mm=float(workspace["root_registration_mm"]),
        volume_coverage_minimum=float(gate["volume_coverage_minimum"]),
        fine_coverage_minimum=float(gate["fine_5mm_coverage_minimum"]),
        coarse_coverage_minimum=float(gate["coarse_10mm_coverage_minimum"]),
        maximum_hole_fraction=float(gate["maximum_hole_fraction"]),
        minimum_axial_slice_coverage=float(gate["minimum_axial_slice_coverage"]),
        path_success_minimum=float(gate["path_success_minimum"]),
        stretch_p95_maximum=float(gate["stretch_p95_maximum"]),
    )


def _teacher_policy(config: Mapping[str, Any], pairwise_lambda: float | None = None) -> Retry19TeacherPolicy:
    graph = config["graph_teacher"]
    return Retry19TeacherPolicy(
        k=int(graph["k"]),
        sigma_mm=float(graph["sigma_mm"]),
        pairwise_lambda=float(graph["selected_pairwise_lambda"] if pairwise_lambda is None else pairwise_lambda),
        huber_delta_deg=float(graph["huber_delta_deg"]),
        soft_anchor_bonus=float(graph["soft_anchor_bonus"]),
        maximum_icm_sweeps=int(graph["maximum_icm_sweeps"]),
        beta_weights=tuple(map(float, config["candidate_solver"]["beta_weights"])),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("retry19 experiment_id mismatch")
    workspace = config["workspace"]
    if not 20_000 <= int(workspace["primary_target_cap"]) <= 30_000:
        raise ValueError("retry19 primary target cap must stay within the Q31 20k-30k range")
    if not 0 < int(workspace["audit_target_cap"]) <= 2_000:
        raise ValueError("retry19 audit-only target cap must remain a small positive diagnostic budget")
    if int(workspace["pilot_target_count"]) != 5000 or int(workspace["immutable_batch_size"]) <= 0:
        raise ValueError("retry19 requires the Q31 5k pilot and positive immutable batches")
    if int(workspace["global_proposal_power"]) != 19 or int(workspace["shell_proposal_power"]) != 16:
        raise ValueError("retry19 proposal powers changed")
    if tuple(workspace["pool_a_seeds"]) != (20260940, 20260942, 20260944) or tuple(workspace["pool_b_seeds"]) != (20260941, 20260943, 20260945):
        raise ValueError("retry19 independent proposal seeds changed")
    if not tuple(map(float, config["graph_teacher"]["pairwise_lambdas"])):
        raise ValueError("retry19 Teacher lambda registry must not be empty")
    if int(config["graph_teacher"]["k"]) != 16 or float(config["graph_teacher"]["selected_pairwise_lambda"]) != 4.0:
        raise ValueError("retry19 primary Graph Teacher changed")
    if float(config["student"]["signed_power_epsilon_mm"]) != 3.0 or float(config["student"]["edge_minimum_distance_mm"]) != 5.0:
        raise ValueError("retry19 Student smooth/edge contract changed")
    inventory = config["trajectories"]["inventory"]
    if sum(map(int, inventory.values())) != 15:
        raise ValueError("retry19 trajectory inventory must contain exactly 15 trajectories")
    claims = config["claims"]
    if not claims["diagnostic_only"] or not claims["exploratory_e2e_authorized"]:
        raise ValueError("retry19 exploratory diagnostic execution must be authorized")
    forbidden_claims = {key: value for key, value in claims.items() if key not in {"diagnostic_only", "exploratory_e2e_authorized"}}
    if any(bool(value) for value in forbidden_claims.values()):
        raise ValueError("retry19 claim boundary changed")
    if sha256_file(SOURCE_ROOT / str(config["sources"]["robot_config"])) != str(config["sources"]["robot_config_sha256"]):
        raise ValueError("retry19 robot config SHA mismatch")
    _coverage_policy(config)
    _teacher_policy(config)
    return config


def _binding_definition(binding_sha: str) -> Mapping[str, Any]:
    raw = subprocess.check_output(["git", "show", f"{binding_sha}:spec/registry.yaml"], cwd=SOURCE_ROOT, text=True)
    return yaml.safe_load(raw)["experiments"][EXPERIMENT_ID]


def _ensure_identity(config: Mapping[str, Any], output_root: Path, binding_sha: str, *, smoke: bool) -> None:
    global _ACTIVE_SCIENTIFIC_SOURCE_SHA
    definition = _binding_definition(binding_sha)
    checkout_sha = _checkout_sha()
    if checkout_sha != str(binding_sha):
        raise RuntimeError("retry19 must run from the exact binding checkout")
    scientific_source_sha = str(definition.get("scientific_source_fixed_point", ""))
    if not scientific_source_sha:
        raise RuntimeError("retry19 binding has no scientific source fixed point")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", scientific_source_sha, checkout_sha],
        cwd=SOURCE_ROOT,
        check=True,
    )
    closure_paths = [
        *map(str, definition.get("protocol_sources", [])),
        str(definition.get("config", "")),
        str(definition.get("runner", "")),
        str(definition.get("launcher", "")),
        *map(str, definition.get("tests", {}).get("self_contained", [])),
    ]
    closure_paths = [path for path in closure_paths if path]
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", scientific_source_sha, checkout_sha, "--", *closure_paths],
        cwd=SOURCE_ROOT,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("retry19 binding checkout differs from its scientific source closure")
    _ACTIVE_SCIENTIFIC_SOURCE_SHA = scientific_source_sha
    expected = {
        "scientific_source_fixed_point": scientific_source_sha,
        "config": str(Path(config["config_path"]).relative_to(SOURCE_ROOT)),
        "runner": str(Path(__file__).relative_to(SOURCE_ROOT)),
    }
    for key, value in expected.items():
        if str(definition.get(key)) != str(value):
            raise RuntimeError(f"retry19 binding mismatch for {key}")
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_source_fixed_point": scientific_source_sha,
        "binding_fixed_point": str(binding_sha),
        "config_sha256": _config_sha(config),
        "diagnostic_smoke": bool(smoke),
    }
    path = output_root / "run_identity.json"
    if path.exists() and _read_json(path) != identity:
        raise RuntimeError("retry19 output identity mismatch")
    if not path.exists():
        output_root.mkdir(parents=True, exist_ok=False)
        _write_json(path, identity)


def _stage_manifest(output_root: Path, stage_name: str) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = [
        {"path": str(path.relative_to(stage)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json" and "_work" not in path.parts
    ]
    index = STAGE_ORDER.index(stage_name)
    upstream = []
    if index:
        previous = output_root / STAGE_DIRS[STAGE_ORDER[index - 1]] / "completion_manifest.json"
        if previous.exists():
            upstream.append({"stage": STAGE_ORDER[index - 1], "sha256": sha256_file(previous)})
    return {"schema_version": 1, "stage": stage_name, "scientific_source_fixed_point": _git_sha(), "artifacts": artifacts, "upstream": upstream}


def _seal_stage(output_root: Path, stage_name: str) -> None:
    _write_json(output_root / STAGE_DIRS[stage_name] / "completion_manifest.json", _stage_manifest(output_root, stage_name))


def _seal_gate(output_root: Path, config: Mapping[str, Any], stage_name: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        **dict(gate),
        "stage": stage_name,
        "experiment_id": EXPERIMENT_ID,
        "scientific_source_fixed_point": _git_sha(),
        "config_sha256": _config_sha(config),
        **config["claims"],
    }
    _write_json(output_root / STAGE_DIRS[stage_name] / "gate.json", value)
    _seal_stage(output_root, stage_name)
    return value


def _gate(output_root: Path, stage_name: str) -> dict[str, Any]:
    return _read_json(output_root / STAGE_DIRS[stage_name] / "gate.json")


def _complete(output_root: Path, stage_name: str) -> bool:
    path = output_root / STAGE_DIRS[stage_name] / "completion_manifest.json"
    return path.exists() and _read_json(path) == _stage_manifest(output_root, stage_name)


def _verify_upstream(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for key, spec in config["upstream"].items():
        if not isinstance(spec, Mapping) or "path" not in spec or "sha256" not in spec:
            continue
        path = _upstream_path(config, key)
        observed = sha256_file(path) if path.is_file() else "missing"
        rows.append({"source": key, "path": str(path), "expected_sha256": spec["sha256"], "observed_sha256": observed, "verified": observed == spec["sha256"]})
    return pd.DataFrame(rows)


def _proposal_pool(config: Mapping[str, Any], environment: Any, *, pool: str, smoke: bool) -> pd.DataFrame:
    workspace = config["workspace"]
    seeds = workspace[f"pool_{pool}_seeds"]
    global_power = min(8, int(workspace["global_proposal_power"])) if smoke else int(workspace["global_proposal_power"])
    shell_power = min(6, int(workspace["shell_proposal_power"])) if smoke else int(workspace["shell_proposal_power"])
    parts: list[pd.DataFrame] = []
    beta = sobol_beta_samples(np.asarray(environment.bounds), power=global_power, seed=int(seeds[0]))
    parts.append(pd.DataFrame(np.asarray(environment.fk(beta)).reshape(-1, 3), columns=XYZ_COLUMNS).assign(proposal_family="global_sobol"))
    for family, radii, seed in (
        ("zero_biased", workspace["zero_biased_radii_deg"], int(seeds[1])),
        ("medium_bend", workspace["medium_bend_radii_deg"], int(seeds[2])),
    ):
        for radius_index, radius in enumerate(radii):
            beta = weighted_sobol_beta_ball(
                power=shell_power,
                seed=seed + radius_index,
                maximum_weighted_radius_deg=float(radius),
                beta_weights=config["candidate_solver"]["beta_weights"],
                beta_bounds_rad=np.asarray(environment.bounds),
            )
            part = pd.DataFrame(np.asarray(environment.fk(beta)).reshape(-1, 3), columns=XYZ_COLUMNS)
            part["proposal_family"] = family
            part["radius_deg"] = float(radius)
            parts.append(part)
    result = pd.concat(parts, ignore_index=True, sort=False)
    result["proposal_pool"] = pool
    result["proposal_id"] = [stable_id("retry19_proposal", pool, index) for index in range(len(result))]
    result["proposal_beta_label_eligible"] = False
    result["proposal_beta_seed_eligible"] = False
    result["proposal_beta_warm_start_eligible"] = False
    return result


def _historical_signed_rows(config: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(_upstream_path(config, "retry18_full_g4_dataset")).copy()
    frame["historical_target_id"] = frame["target_id"].astype(str)
    frame["target_id"] = [stable_id("retry19_old_signed", target, element) for target, element in zip(frame["historical_target_id"], frame["symmetry_element"], strict=True)]
    frame["old_label_candidate"] = True
    frame["source"] = "retry18"
    return frame


def _zero_cell(cells: pd.DataFrame, zero_xyz_m: np.ndarray) -> tuple[pd.DataFrame, str]:
    zero_mm = zero_xyz_m * 1000.0
    contains = np.logical_and(
        cells.loc[:, ("x_min_mm", "y_min_mm", "z_min_mm")].to_numpy(float) <= zero_mm,
        zero_mm <= cells.loc[:, ("x_max_mm", "y_max_mm", "z_max_mm")].to_numpy(float),
    ).all(axis=1)
    if contains.any():
        index = int(np.flatnonzero(contains)[0])
        result = cells.copy()
        zero_id = str(result.iloc[index]["cell_id"])
        result["exact_zero_cell"] = result["cell_id"].astype(str).eq(zero_id)
        result.loc[result["exact_zero_cell"], "required"] = False
        return result, zero_id
    lower = np.floor(zero_mm / 5.0) * 5.0
    row = {
        "cell_id": stable_id("retry19_exact_zero_cell", *zero_mm),
        "x_min_mm": lower[0], "y_min_mm": lower[1], "z_min_mm": lower[2],
        "x_max_mm": lower[0] + 5, "y_max_mm": lower[1] + 5, "z_max_mm": lower[2] + 5,
        "cell_size_mm": 5.0, "volume_mm3": 125.0,
        "pool_a_observed": False, "pool_b_observed": False,
        "u_center_mm": 0.0, "domain_class": "primary",
        "probe_x_m": zero_xyz_m[0], "probe_y_m": zero_xyz_m[1], "probe_z_m": zero_xyz_m[2],
        "required": False, "served": False, "exact_zero_cell": True,
        "proposal_beta_used_as_label_or_hint": False,
    }
    result = pd.concat([cells.assign(exact_zero_cell=False), pd.DataFrame([row])], ignore_index=True, sort=False)
    return result, str(row["cell_id"])


def _packing_lower_bound(points_m: np.ndarray, *, radius_mm: float) -> int:
    if len(points_m) == 0:
        return 0
    tree = cKDTree(points_m)
    available = np.ones(len(points_m), dtype=bool)
    count = 0
    for index in range(len(points_m)):
        if not available[index]:
            continue
        count += 1
        neighbours = tree.query_ball_point(points_m[index], 2.0 * radius_mm / 1000.0 + 1.0e-12)
        available[np.asarray(neighbours, dtype=int)] = False
    return count


def _target_registry(cells: pd.DataFrame, history: pd.DataFrame, config: Mapping[str, Any], *, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    workspace = config["workspace"]
    audit_cap = min(16, int(workspace["audit_target_cap"])) if smoke else int(workspace["audit_target_cap"])
    old = history.copy()
    old["u_mm"] = 1000.0 * (float(workspace["zero_x_m"]) - old["x_m"])
    old = old[old["u_mm"].between(0.0, float(workspace["audit_u_maximum_mm"]), inclusive="both")].copy()
    old["macro10"] = pd.DataFrame(
        np.floor(old.loc[:, XYZ_COLUMNS].to_numpy(float) * 100.0).astype(int), index=old.index
    ).astype(str).agg(":".join, axis=1)
    representatives = old.sort_values(["macro10", "target_id"], kind="stable").drop_duplicates("macro10")
    primary_old = representatives[representatives["u_mm"].le(float(workspace["primary_u_maximum_mm"]))].copy()
    primary_old["target_role"] = "retry18_recertification"
    primary_old["domain_class"] = "primary"
    cell_targets = cells[cells["required"].astype(bool) & cells["domain_class"].eq("primary")].copy()
    cell_targets = cell_targets.rename(columns={"probe_x_m": "x_m", "probe_y_m": "y_m", "probe_z_m": "z_m"})
    cell_targets["target_id"] = cell_targets["cell_id"].map(lambda value: stable_id("retry19_cell_target", value))
    cell_targets["target_role"] = "primary_cell_service"
    cell_targets["old_label_candidate"] = False
    combined = pd.concat([primary_old, cell_targets], ignore_index=True, sort=False)
    combined["xyz_key"] = combined.loc[:, XYZ_COLUMNS].round(9).astype(str).agg(":".join, axis=1)
    combined = combined.sort_values(["old_label_candidate", "target_id"], ascending=[False, True], kind="stable").drop_duplicates("xyz_key")
    zero = {"target_id": "retry19_exact_zero", "x_m": float(workspace["zero_x_m"]), "y_m": 0.0, "z_m": 0.0, "target_role": "exact_zero", "domain_class": "primary", "old_label_candidate": True, **dict.fromkeys(BETA_COLUMNS, 0.0)}
    combined = pd.concat([pd.DataFrame([zero]), combined], ignore_index=True, sort=False)
    combined["xyz_key"] = combined.loc[:, XYZ_COLUMNS].round(9).astype(str).agg(":".join, axis=1)
    combined = combined.drop_duplicates("xyz_key", keep="first")
    combined["priority"] = np.where(combined["target_role"].eq("exact_zero"), 0, 1)
    combined["digest"] = combined["target_id"].astype(str).map(lambda value: hashlib.sha256(f"20260952:{value}".encode()).hexdigest())
    combined = combined.sort_values(["priority", "digest", "target_id"], kind="stable").drop(columns=["priority", "digest"])
    combined["selection_ordinal"] = np.arange(len(combined), dtype=np.int64)
    audit = representatives[representatives["u_mm"].gt(float(workspace["primary_u_maximum_mm"]))].copy()
    audit["digest"] = audit["target_id"].astype(str).map(lambda value: hashlib.sha256(f"20260953:{value}".encode()).hexdigest())
    audit = audit.sort_values(["digest", "target_id"], kind="stable").head(audit_cap).drop(columns="digest")
    audit["target_role"] = "audit_only_exact_retry18_target"
    audit["domain_class"] = "audit_only"
    audit["selection_ordinal"] = np.arange(len(audit), dtype=np.int64)
    return combined.reset_index(drop=True), audit.reset_index(drop=True)


def _write_objective_feasibility_contract(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    required_primary: pd.DataFrame,
    primary_targets: pd.DataFrame,
    verification: pd.DataFrame,
    proposal_only_count: int,
) -> dict[str, Any]:
    """Freeze and validate the quantitative launch denominator before later stages."""

    root = output_root / "00_objective_feasibility"
    root.mkdir(parents=True, exist_ok=False)
    unit_volume_mm3 = float(config["workspace"]["fine_cell_mm"]) ** 3
    unit_counts = np.rint(required_primary["volume_mm3"].to_numpy(float) / unit_volume_mm3).astype(np.int64)
    if len(required_primary) == 0 or np.any(unit_counts <= 0):
        raise RuntimeError("retry19 objective denominator is empty or invalid")

    repeated = required_primary.index.repeat(unit_counts)
    volume_units = required_primary.loc[repeated, ["cell_id"]].reset_index(drop=True)
    volume_units["unit_offset"] = volume_units.groupby("cell_id", sort=False).cumcount()
    volume_units["coverage_unit_id"] = (
        volume_units["cell_id"].astype(str) + ":volume_unit:" + volume_units["unit_offset"].astype(str)
    )
    volume_units["unit_volume_mm3"] = unit_volume_mm3
    volume_units = volume_units[["coverage_unit_id", "cell_id", "unit_offset", "unit_volume_mm3"]]
    _write_parquet(volume_units, root / "required_volume_unit_registry.parquet")

    probes = required_primary.loc[:, ["probe_x_m", "probe_y_m", "probe_z_m"]].to_numpy(float)
    target_xyz = primary_targets.loc[:, XYZ_COLUMNS].to_numpy(float)
    tree = cKDTree(probes)
    radius_m = float(config["workspace"]["service_radius_mm"]) / 1000.0
    credits = []
    for target_id, indices in zip(
        primary_targets["target_id"].astype(str),
        tree.query_ball_point(target_xyz, radius_m),
        strict=True,
    ):
        credit = int(unit_counts[np.asarray(indices, dtype=np.int64)].sum()) if indices else 0
        credits.append({"target_id": target_id, "maximum_volume_unit_credit": credit})
    credit_basis = pd.DataFrame(credits)
    _write_parquet(credit_basis, root / "coverage_credit_basis.parquet")
    maximum_credit = max(1, int(credit_basis["maximum_volume_unit_credit"].max()))

    reusable_credit = primary_targets[
        primary_targets["target_role"].astype(str).eq("retry18_recertification")
    ][["target_id", "target_role"]].copy()
    reusable_credit["credit_class"] = "connector_only_pending_fresh_recertification"
    reusable_credit["eligible_coverage_units"] = 0
    _write_parquet(reusable_credit, root / "reusable_credit_registry.parquet")

    required_units = int(len(volume_units))
    target_fraction = float(config["coverage_gate"]["volume_coverage_minimum"])
    target_units = int(math.ceil(required_units * target_fraction))
    minimum_new = int(math.ceil(target_units / maximum_credit))
    registered_budget = int(config["workspace"]["primary_target_cap"])
    feasible = minimum_new <= registered_budget

    registry = {
        "path": "00_objective_feasibility/required_volume_unit_registry.parquet",
        "sha256": sha256_file(root / "required_volume_unit_registry.parquet"),
        "id_column": "coverage_unit_id",
        "row_count": required_units,
    }
    basis_registry = {
        "path": "00_objective_feasibility/coverage_credit_basis.parquet",
        "sha256": sha256_file(root / "coverage_credit_basis.parquet"),
    }
    objective_id = "primary_volume_coverage"
    objective = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scope": "coverage_trajectory",
        "scientific_source_sha": _git_sha(),
        "config_sha256": _config_sha(config),
        "diagnostic_pilot_allowed": False,
        "primary_objectives": [
            {
                "id": objective_id,
                "kind": "coverage",
                "metric": "volume_weighted_served_coverage",
                "required_for_claim": True,
                "denominator_id": "required_primary_volume_units",
                "target": {"operator": ">=", "value": target_fraction, "unit": "fraction"},
            }
        ],
    }
    denominator = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "frozen_before_launch": True,
        "denominators": [
            {
                "id": "required_primary_volume_units",
                "kind": "coverage",
                "unit": f"{unit_volume_mm3:g}_mm3_volume_unit",
                "required_count": required_units,
                "registry": registry,
            }
        ],
        "resource_denominators": {
            "required_supervision_vertex_count": int(len(primary_targets)),
            "required_logical_edge_count": 0,
            "required_second_parent_certification_count": 0,
        },
    }
    source_artifacts = [
        {"path": str(row.path), "sha256": str(row.expected_sha256)}
        for row in verification.itertuples(index=False)
    ]
    reusable = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_artifacts": source_artifacts,
        "eligible_counts": {
            "supervision_vertices": 0,
            "connector_only_vertices": int(len(reusable_credit)),
            "served_coverage_units": 0,
            "complete_trajectories": 0,
            "verified_edges": 0,
            "second_parent_certifications": 0,
        },
        "ineligible_counts": {
            "proposal_only": int(proposal_only_count),
            "branch_conflicts": 0,
            "unused": 0,
        },
        "credit_registry": {
            "path": "00_objective_feasibility/reusable_credit_registry.parquet",
            "sha256": sha256_file(root / "reusable_credit_registry.parquet"),
        },
        "proposal_beta_used_as_label_or_hint": False,
    }
    lower = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "objective_lower_bounds": [
            {
                "objective_id": objective_id,
                "method": "count_credit",
                "required_units": required_units,
                "target_units": target_units,
                "reusable_eligible_units": 0,
                "maximum_credit_per_new_supervision_vertex": maximum_credit,
                "minimum_resources": {"new_supervision_vertices": minimum_new},
                "basis_registry": basis_registry,
            }
        ],
        "resources": {
            "new_supervision_vertices": {
                "optimistic_minimum": minimum_new,
                "registered_budget": registered_budget,
                "basis_registry": basis_registry,
            }
        },
        "all_required_objectives_bounded": True,
        "all_required_resources_feasible": feasible,
    }
    schedule = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "row_count_is_stop_condition": False,
        "scheduled_objective_ids": [objective_id],
        "entries": [
            {
                "id": "primary_volume_fill",
                "objective_id": objective_id,
                "kind": "coverage",
                "priority": 0,
                "required_for_claim": True,
                "requirement_registry": registry,
                "reserved_resources": {"new_supervision_vertices": minimum_new},
            }
        ],
    }
    gate = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "inputs_valid": True,
        "required_objectives_budget_feasible": feasible,
        "status": "feasible" if feasible else "diagnostic_only",
        "claim_bearing_run_authorized": feasible,
        "diagnostic_pilot_authorized": False,
        "failed_objective_ids": [] if feasible else [objective_id],
        "reason_codes": [] if feasible else ["NEW_SUPERVISION_BUDGET_BELOW_OPTIMISTIC_MINIMUM"],
        "authorization_scope": "retry19 bounded empirical diagnostic execution; final config claim boundaries remain unchanged",
        "optimistic_minimum_new_supervision_vertices": minimum_new,
        "required_volume_units": required_units,
        "maximum_volume_units_per_new_supervision_vertex": maximum_credit,
    }
    for name, value in (
        ("objective_contract.json", objective),
        ("denominator_size.json", denominator),
        ("reusable_evidence.json", reusable),
        ("budget_lower_bound.json", lower),
        ("atomic_objective_schedule.json", schedule),
        ("gate.json", gate),
    ):
        _write_json(root / name, value)
    return gate


def stage_fullspace_discovery(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["fullspace_discovery"]
    verification = _verify_upstream(config)
    _write_parquet(verification, stage / "upstream_verification.parquet")
    if not verification["verified"].all():
        raise RuntimeError("retry19 upstream integrity failed")
    environment = _environment(config)
    zero = np.asarray(environment.fk(np.zeros(6))).reshape(3)
    expected = np.asarray([config["workspace"]["zero_x_m"], 0.0, 0.0])
    if not np.allclose(zero, expected, atol=1.0e-12, rtol=0):
        raise RuntimeError("retry19 exact-zero mismatch")
    pool_a = _proposal_pool(config, environment, pool="a", smoke=smoke)
    pool_b = _proposal_pool(config, environment, pool="b", smoke=smoke)
    _write_parquet(pool_a, stage / "proposal_pool_a_xyz_only.parquet")
    _write_parquet(pool_b, stage / "proposal_pool_b_xyz_only.parquet")
    workspace = config["workspace"]
    cells = build_mixed_resolution_cell_registry(
        pool_a.loc[:, XYZ_COLUMNS].to_numpy(float), pool_b.loc[:, XYZ_COLUMNS].to_numpy(float),
        zero_xyz_m=zero, primary_u_maximum_mm=float(workspace["primary_u_maximum_mm"]),
        audit_u_maximum_mm=float(workspace["audit_u_maximum_mm"]), fine_root_radius_mm=float(workspace["fine_root_radius_mm"]),
    )
    cells, zero_cell_id = _zero_cell(cells, zero)
    component_edges = mixed_resolution_26_edges(cells)
    face_edges = mixed_resolution_face_edges(cells)
    root_edges, component = geometric_zero_attachment(
        cells, component_edges, zero_cell_id=zero_cell_id, zero_point_mm=zero * 1000.0,
        registration_threshold_mm=float(workspace["root_registration_mm"]),
    )
    cells["geometric_zero_connected"] = cells["cell_id"].astype(str).isin(component)
    history = _historical_signed_rows(config)
    primary_targets, audit_targets = _target_registry(cells, history, config, smoke=smoke)
    required_primary = cells[cells["required"] & cells["domain_class"].eq("primary")]
    queries = deterministic_path_query_registry(
        required_primary,
        pair_count=min(int(workspace["path_query_count"]), max(1, len(required_primary) * (len(required_primary) - 1) // 2)),
        seed=int(workspace["path_query_seed"]),
    ) if len(required_primary) >= 2 else pd.DataFrame(columns=["query_id", "left_cell_id", "right_cell_id"])
    _write_parquet(cells, stage / "mixed_resolution_cell_registry.parquet")
    _write_parquet(component_edges, stage / "primary_26_adjacency.parquet")
    _write_parquet(face_edges, stage / "primary_face_adjacency.parquet")
    _write_parquet(root_edges, stage / "geometric_root_graph.parquet")
    _write_parquet(primary_targets, stage / "primary_target_registry.parquet")
    _write_parquet(audit_targets, stage / "audit_only_target_registry.parquet")
    _write_parquet(queries, stage / "path_query_registry.parquet")
    feasibility_gate = _write_objective_feasibility_contract(
        config,
        output_root,
        required_primary=required_primary,
        primary_targets=primary_targets,
        verification=verification,
        proposal_only_count=len(pool_a) + len(pool_b),
    )
    return _seal_gate(output_root, config, "fullspace_discovery", {
        "status": "complete",
        "objective_resource_feasible": bool(feasibility_gate["claim_bearing_run_authorized"]),
        "objective_resource_feasibility_is_diagnostic": True,
        "geometric_zero_connected": bool(len(required_primary) and required_primary["geometric_zero_connected"].all()),
        "required_primary_cell_count": len(required_primary),
        "primary_target_registry_count": len(primary_targets),
        "audit_target_registry_count": len(audit_targets),
        "optimistic_minimum_primary_targets": int(feasibility_gate["optimistic_minimum_new_supervision_vertices"]),
        "proposal_beta_used_as_label_or_hint": False,
    })


def _model_metrics(model: Any, waypoints: pd.DataFrame, environment: Any, *, direct: bool, zero: np.ndarray) -> Mapping[str, float]:
    records = []
    for _trajectory, part in waypoints[waypoints["shape_class"].eq("rounded_rectangle")].groupby("trajectory_id", sort=True):
        ordered = part.sort_values("waypoint_index", kind="stable")
        xyz = ordered.loc[:, XYZ_COLUMNS].to_numpy(float)
        beta = np.asarray(model(np.asarray(xyz, np.float32), training=False), float) if direct else retry17._symmetry_prediction(model, xyz, zero)
        records.append(retry18._path_metrics(xyz, beta, environment, closed=True))
    if not records:
        return {"rectangle_fk_p95_mm": math.inf, "rectangle_fk_maximum_mm": math.inf, "rectangle_path_step_excess_maximum_mm": math.inf, "raw_max_spike_mm": math.inf}
    return {
        "rectangle_fk_p95_mm": max(row["fk_p95_mm"] for row in records),
        "rectangle_fk_maximum_mm": max(row["fk_maximum_mm"] for row in records),
        "rectangle_path_step_excess_maximum_mm": max(row["path_step_excess_maximum_mm"] for row in records),
        "raw_max_spike_mm": max(row["path_step_excess_maximum_mm"] for row in records),
    }


def _direct_config(
    config: Mapping[str, Any],
    *,
    seed: int,
    alpha: float,
    edge_lambda: float = 0.0,
    jacobian_lambda: float = 0.0,
    hidden_units: Sequence[int] | None = None,
    smoke: bool,
) -> DirectStudentConfig:
    student = config["student"]
    full = _historical_signed_rows(config)
    radius = np.hypot(full["y_m"], full["z_m"]) * 1000.0
    radial_scale = float(student["radial_scale_safety_factor"]) * float(np.quantile(radius, float(student["radial_scale_quantile"])))
    return DirectStudentConfig(
        hidden_units=tuple(map(int, hidden_units or student["hidden_units"])), learning_rate=float(student["learning_rate"]),
        maximum_steps=min(20, int(student["maximum_steps"])) if smoke else int(student["maximum_steps"]),
        validation_interval=min(5, int(student["validation_interval"])) if smoke else int(student["validation_interval"]),
        patience_intervals=2 if smoke else int(student["patience_intervals"]), batch_size=int(student["batch_size"]),
        seed=int(seed), axial_scale_mm=float(student["axial_scale_mm"]), radial_scale_mm=radial_scale,
        signed_power_alpha=float(alpha), signed_power_epsilon_mm=float(student["signed_power_epsilon_mm"]),
        edge_lambda=float(edge_lambda), jacobian_lambda=float(jacobian_lambda),
    )


def stage_causal_controls(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["causal_controls"]
    full = _historical_signed_rows(config)
    q0, f0 = aligned_q0_augmented_and_f0_rows(full)
    q0["sample_weight"] = q0["orbit_normalized_weight"]
    f0["sample_weight"] = f0["orbit_normalized_weight"]
    _write_parquet(q0, stage / "q0_augmented_rows.parquet")
    _write_parquet(f0, stage / "f0_direct_rows.parquet")
    row_alignment = bool(q0[["control_row_id", "control_row_ordinal", "orbit_normalized_weight"]].equals(f0[["control_row_id", "control_row_ordinal", "orbit_normalized_weight"]]))
    contract = {
        "q0_exact": "frozen retry18 S1 rows/order/normalization/wrapper",
        "q0_augmented": "same full-G4 rows/order/weights/batches as F0, canonicalized to quotient",
        "f0_direct": "same rows/order/weights/batches, signed xyz retained",
        "row_alignment_verified": row_alignment,
        "causal_claim": "direct signed full-space representation; not unique mirror-wrapper cause",
    }
    _write_json(stage / "control_contract.json", contract)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("retry19 causal controls require CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("retry19 CPU causal controls see GPU")
    environment = _environment(config)
    zero = np.asarray(environment.fk(np.zeros(6))).reshape(3)
    old_waypoints = pd.read_parquet(_upstream_path(config, "retry18_heldout_waypoints"))
    rows: list[dict[str, Any]] = []

    q0_exact = tf.keras.models.load_model(_upstream_path(config, "retry18_s1_model"), compile=False)
    exact_metrics = _model_metrics(q0_exact, old_waypoints, environment, direct=False, zero=zero)
    expected = pd.read_parquet(_upstream_path(config, "retry18_selection_metrics"))
    expected = expected[expected["model_id"].eq("S1_seam_root")].iloc[0]
    expected_spike = float(expected["rectangle_path_step_excess_maximum_mm"])
    reproduction_error = abs(float(exact_metrics["raw_max_spike_mm"]) - expected_spike) / max(abs(expected_spike), 1.0e-12)
    rows.append({"model_id": "Q0_exact", "seed": -1, **exact_metrics, "q0_reproduction_relative_error": reproduction_error})

    seed = int(config["student"]["primary_seed"])
    control_specs = (
        ("Q0_augmented", q0, 1.0, False, False),
        ("F0_direct", f0, 1.0, False, True),
        ("F1_smooth_alpha1", f0, 1.0, True, True),
        ("F2_smooth_alpha05", f0, 0.5, True, True),
    )
    for model_id, control_rows, alpha, include_power, direct_prediction in control_specs:
        train = control_rows[control_rows["split_role"].eq("train")].copy()
        valid = control_rows[control_rows["split_role"].eq("validation")].copy()
        model_cfg = _direct_config(config, seed=seed, alpha=alpha, smoke=smoke)
        model, history = train_direct_student(
            train, valid, pd.DataFrame(), zero_x_m=zero[0], beta_bounds_rad=np.asarray(environment.bounds),
            config=model_cfg, include_signed_power=include_power,
            beta_coordinate_weights=config["candidate_solver"]["beta_weights"],
        )
        model_dir = stage / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=False)
        model.save(model_dir / "control.keras")
        _write_parquet(history, model_dir / "training_history.parquet")
        _write_json(model_dir / "model_manifest.json", {
            **_model_manifest(model_id, model_cfg, seed, sha256_file(stage / ("q0_augmented_rows.parquet" if model_id == "Q0_augmented" else "f0_direct_rows.parquet"))),
            "control_row_order_sha256": sha256_file(stage / ("q0_augmented_rows.parquet" if model_id == "Q0_augmented" else "f0_direct_rows.parquet")),
            "batch_sequence_seed": seed,
        })
        metrics = _model_metrics(model, old_waypoints, environment, direct=direct_prediction, zero=zero)
        xyz = valid.loc[:, XYZ_COLUMNS].to_numpy(float)
        prediction = np.asarray(model(np.asarray(xyz, np.float32), training=False), float)
        if not direct_prediction:
            prediction = retry17._symmetry_prediction(model, xyz, zero)
        residual = np.linalg.norm(np.asarray(environment.fk(prediction)).reshape(-1, 3) - xyz, axis=1) * 1000.0
        rows.append({"model_id": model_id, "seed": seed, **metrics, "validation_fk_p95_mm": float(np.percentile(residual, 95)), "validation_fk_maximum_mm": float(np.max(residual))})
    metrics = pd.DataFrame(rows)
    _write_parquet(metrics, stage / "control_metrics.parquet")
    indexed = metrics.set_index("model_id")
    preliminary = classify_causal_evidence({
        "q0_augmented_raw_max_spike_mm": float(indexed.loc["Q0_augmented", "raw_max_spike_mm"]),
        "f0_direct_raw_max_spike_mm": float(indexed.loc["F0_direct", "raw_max_spike_mm"]),
        "f0_nonregression": float(indexed.loc["F0_direct", "rectangle_fk_p95_mm"]) <= 1.10 * float(indexed.loc["Q0_augmented", "rectangle_fk_p95_mm"]),
        "replicated_seed_pass_count": 0,
    })
    _write_json(stage / "preliminary_causal_evidence.json", preliminary)
    reproduction_pass = reproduction_error <= float(config["student"]["q0_reproduction_relative_tolerance"])
    return _seal_gate(output_root, config, "causal_controls", {
        "status": "complete",
        "row_alignment_verified": row_alignment,
        "q0_exact_reproduction_pass": reproduction_pass,
        "q0_exact_reproduction_is_diagnostic": True,
        "q0_exact_reproduction_relative_error": reproduction_error,
        "controls_materialized": True, "controls_trained": True,
        "q0_augmented_and_f0_share_batch_sequence": True,
        **preliminary,
    })


def _candidate_rows_for_old(targets: pd.DataFrame, anchor_registry: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    classes = anchor_registry.set_index("target_id")["anchor_class"].to_dict() if len(anchor_registry) else {}
    rows = targets[targets.get("old_label_candidate", pd.Series(False, index=targets.index)).fillna(False).astype(bool)].copy()
    rows = rows[~rows["target_id"].astype(str).eq("retry19_exact_zero")]
    rows = rows[rows["target_id"].astype(str).map(classes).ne("diagnostic")]
    if rows.empty:
        return rows
    rows["candidate_id"] = rows["target_id"].map(lambda value: f"retry19_old:{value}")
    rows["seed_id"] = "retry18_old_label"
    rows["solver_success"] = True
    rows["bounds_pass"] = True
    rows["fk_residual_mm"] = rows.get("fk_residual_mm", 0.0)
    rows["min_margin_deg"] = rows.get("min_margin_deg", 0.0)
    rows["old_label_candidate"] = True
    rows["proposal_beta_used"] = False
    return rows


def _anchor_registry(targets: pd.DataFrame) -> pd.DataFrame:
    old = targets[targets.get("old_label_candidate", pd.Series(False, index=targets.index)).fillna(False).astype(bool)].copy()
    if old.empty:
        return pd.DataFrame(columns=["target_id", "anchor_class"])
    old["fresh_fk_residual_mm"] = old.get("fk_residual_mm", 0.0)
    old["repeat_pass"] = old.get("repeat_pass", False)
    old["multiparent_consistency_pass"] = old.get("multiparent_consistency_pass", False)
    old["seam_or_conflict_risk"] = (old["y_m"].abs() <= 0.020) | (old["z_m"].abs() <= 0.020)
    registry = classify_old_anchors(old, exact_zero_target_id="retry19_exact_zero")
    registry.loc[registry["target_id"].eq("retry19_exact_zero"), "anchor_class"] = "hard"
    if "domain_class" in old and old["domain_class"].astype(str).eq("audit_only").any():
        audit = registry.merge(old[["target_id", "domain_class"]], on="target_id", how="left")
        eligible = audit[
            audit["domain_class"].eq("audit_only")
            & audit["fresh_fk_residual_mm"].le(3.0)
            & ~audit["seam_or_conflict_risk"]
        ].sort_values(["macroblock_id", "fresh_fk_residual_mm", "target_id"], kind="stable").drop_duplicates("macroblock_id")
        registry.loc[registry["target_id"].isin(eligible["target_id"]), ["anchor_class", "classification_reasons"]] = [
            "hard", "audit_outer_reference_macroblock_representative"
        ]
    return registry


def _solve_target_batch(config: Mapping[str, Any], targets: pd.DataFrame, stage: Path, *, smoke: bool, namespace: str) -> pd.DataFrame:
    exact = targets[targets["target_id"].eq("retry19_exact_zero")]
    solve = targets[~targets["target_id"].eq("retry19_exact_zero")]
    fresh = retry17._solve_candidates(
        config, solve, stage, seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]),
        smoke=smoke, work_namespace=namespace,
    ) if len(solve) else pd.DataFrame()
    legal = legal_candidate_clusters(
        fresh,
        residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]),
        cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]),
        weights=config["candidate_solver"]["beta_weights"],
    ) if len(fresh) else pd.DataFrame()
    legal_ids = set(legal["target_id"].astype(str)) if len(legal) else set()
    difficult = solve[~solve["target_id"].astype(str).isin(legal_ids)]
    if len(difficult):
        retry = retry17._solve_candidates(
            config, difficult, stage, seed_budget=int(config["candidate_solver"]["difficult_seed_budget"]),
            smoke=smoke, work_namespace=f"{namespace}_difficult",
        )
        fresh = pd.concat([fresh, retry], ignore_index=True, sort=False)
    if len(exact):
        zero = exact.iloc[[0]].copy()
        zero["candidate_id"] = "retry19_exact_zero_candidate"
        zero["solver_success"] = True; zero["bounds_pass"] = True; zero["fk_residual_mm"] = 0.0; zero["min_margin_deg"] = 0.0
        zero["old_label_candidate"] = True; zero["proposal_beta_used"] = False
        fresh = pd.concat([fresh, zero], ignore_index=True, sort=False)
    return fresh.drop_duplicates(["target_id", "candidate_id"], keep="last").reset_index(drop=True)


def _select_teacher(config: Mapping[str, Any], candidates: pd.DataFrame, targets: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = target_knn_edges(targets, k=int(config["graph_teacher"]["k"]))
    trials = []
    solutions = {}
    for pairwise_lambda in config["graph_teacher"]["pairwise_lambdas"]:
        labels, audit = select_fullspace_graph_teacher(candidates, targets, edges, anchors, policy=_teacher_policy(config, float(pairwise_lambda)))
        metric = teacher_edge_metrics(labels, edges, weights=config["candidate_solver"]["beta_weights"]) if len(labels) and len(edges) else {"weighted_p95_deg": math.inf, "raw_gt7_rate": 1.0}
        trials.append({"pairwise_lambda": float(pairwise_lambda), **audit, **metric})
        solutions[float(pairwise_lambda)] = labels
    selected_lambda = float(config["graph_teacher"]["selected_pairwise_lambda"])
    return solutions[selected_lambda], edges, pd.DataFrame(trials)


def stage_candidate_pilot(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["candidate_pilot"]
    targets = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "primary_target_registry.parquet")
    count = min(64, int(config["workspace"]["pilot_target_count"])) if smoke else int(config["workspace"]["pilot_target_count"])
    pilot = targets.sort_values("selection_ordinal", kind="stable").head(count).copy()
    anchors = _anchor_registry(pilot)
    fresh = _solve_target_batch(config, pilot, stage, smoke=smoke, namespace="pilot")
    candidates = pd.concat([fresh, _candidate_rows_for_old(pilot, anchors)], ignore_index=True, sort=False).drop_duplicates(["target_id", "candidate_id"], keep="last")
    labels, edges, trials = _select_teacher(config, candidates, pilot, anchors)
    _write_parquet(pilot, stage / "pilot_targets.parquet")
    _write_parquet(anchors, stage / "old_anchor_registry.parquet")
    _write_parquet(candidates, stage / "pilot_candidate_bank.parquet")
    _write_parquet(labels, stage / "pilot_teacher_labels.parquet")
    _write_parquet(edges, stage / "pilot_teacher_graph.parquet")
    _write_parquet(trials, stage / "pilot_teacher_sweep.parquet")
    accepted = labels["target_id"].nunique() if len(labels) else 0
    return _seal_gate(output_root, config, "candidate_pilot", {
        "status": "complete",
        "target_count": len(pilot), "accepted_target_count": accepted,
        "acceptance_rate": accepted / max(len(pilot), 1),
        "q31_suggested_acceptance_met": accepted / max(len(pilot), 1) >= 0.95,
        "acceptance_is_exploratory_diagnostic": True,
        "proposal_beta_used_as_label_or_hint": False,
    })


def stage_continuation_authorization(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["continuation_authorization"]
    discovery = _gate(output_root, "fullspace_discovery")
    pilot = _gate(output_root, "candidate_pilot")
    authorized = bool(pilot["accepted_target_count"] > 0 and discovery["geometric_zero_connected"])
    reasons = []
    if not discovery["objective_resource_feasible"]: reasons.append("OBJECTIVE_RESOURCE_INFEASIBLE")
    if pilot["acceptance_rate"] < 0.95: reasons.append("PILOT_CANDIDATE_ACCEPTANCE_RED")
    if not discovery["geometric_zero_connected"]: reasons.append("GEOMETRIC_ZERO_NOT_ATTACHED")
    if pilot["accepted_target_count"] <= 0: reasons.append("NO_PILOT_TEACHER_LABELS")
    _write_json(stage / "continuation_decision.json", {"authorized": authorized, "reason_codes": reasons})
    return _seal_gate(output_root, config, "continuation_authorization", {
        "status": "authorized" if authorized else "blocked",
        "continuation_authorized": authorized, "reason_codes": reasons,
        "exploratory_e2e_authorization": True,
    })


def stage_adaptive_fill(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["adaptive_fill"]
    if not _gate(output_root, "continuation_authorization")["continuation_authorized"]:
        _write_parquet(pd.DataFrame(), stage / "full_candidate_bank.parquet")
        return _seal_gate(output_root, config, "adaptive_fill", {"status": "not_authorized", "target_count": 0})
    primary_pool = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "primary_target_registry.parquet")
    audit = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "audit_only_target_registry.parquet")
    pilot_targets = pd.read_parquet(output_root / STAGE_DIRS["candidate_pilot"] / "pilot_targets.parquet")
    pilot_bank = pd.read_parquet(output_root / STAGE_DIRS["candidate_pilot"] / "pilot_candidate_bank.parquet")
    batch_size = min(16, int(config["workspace"]["immutable_batch_size"])) if smoke else int(config["workspace"]["immutable_batch_size"])
    primary_cap = min(64, int(config["workspace"]["primary_target_cap"])) if smoke else int(config["workspace"]["primary_target_cap"])
    remaining = primary_pool[~primary_pool["target_id"].isin(pilot_targets["target_id"])].copy()
    audit = audit.head(8) if smoke else audit
    banks = [pilot_bank]
    batch_rows = []
    selected_targets = pilot_targets.copy()
    fill_rows: list[dict[str, Any]] = []
    cells = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "mixed_resolution_cell_registry.parquet")

    def geometric_fill_metrics(targets: pd.DataFrame) -> tuple[Mapping[str, float | int], float, float]:
        pseudo = targets[["target_id", *XYZ_COLUMNS]].copy()
        marked = mark_served_cells(
            cells, pseudo, teacher_zero_connected_ids=set(pseudo["target_id"].astype(str)),
            service_radius_mm=float(config["workspace"]["service_radius_mm"]),
        )
        primary_cells = marked[marked["domain_class"].eq("primary")]
        metric = mixed_resolution_coverage_metrics(primary_cells)
        face = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "primary_face_adjacency.parquet")
        hole = largest_unserved_component_fraction(primary_cells, face)
        slices = axial_slice_coverage(
            primary_cells, zero_x_mm=float(config["workspace"]["zero_x_m"]) * 1000.0,
            bin_edges_u_mm=np.arange(0, 210, 10),
        )
        return metric, hole, float(slices["volume_coverage"].min())

    batch_index = 0
    while len(selected_targets) < primary_cap and len(remaining):
        selected_xyz = selected_targets.loc[:, XYZ_COLUMNS].to_numpy(float)
        distance = cKDTree(selected_xyz).query(remaining.loc[:, XYZ_COLUMNS].to_numpy(float), k=1)[0] * 1000.0
        remaining = remaining.assign(_fill_distance_mm=distance)
        take = min(batch_size, primary_cap - len(selected_targets), len(remaining))
        batch = remaining.sort_values(["_fill_distance_mm", "target_id"], ascending=[False, True], kind="stable").head(take).drop(columns="_fill_distance_mm")
        remaining = remaining[~remaining["target_id"].isin(batch["target_id"])].drop(columns="_fill_distance_mm")
        batch_dir = stage / f"batch_{batch_index:04d}"
        solved = _solve_target_batch(config, batch, batch_dir, smoke=smoke, namespace=f"batch_{batch_index:04d}")
        _write_parquet(batch, batch_dir / "targets.parquet")
        _write_parquet(solved, batch_dir / "candidate_bank.parquet")
        manifest = {"batch_index": batch_index, "target_count": len(batch), "candidate_count": len(solved), "target_sha256": sha256_file(batch_dir / "targets.parquet"), "candidate_sha256": sha256_file(batch_dir / "candidate_bank.parquet"), "immutable": True}
        _write_json(batch_dir / "batch_manifest.json", manifest)
        batch_rows.append(manifest)
        banks.append(solved)
        selected_targets = pd.concat([selected_targets, batch], ignore_index=True, sort=False)
        metric, hole, minimum_slice = geometric_fill_metrics(selected_targets)
        fill_rows.append({"batch_index": batch_index, "selected_primary_target_count": len(selected_targets), **metric, "maximum_hole_fraction": hole, "minimum_axial_slice_coverage": minimum_slice})
        batch_index += 1
        gate = config["coverage_gate"]
        service_distance_p95 = float(metric["service_distance_volume_weighted_p95_mm"])
        service_distance_limit = float(gate["service_distance_p95_maximum_mm"])
        if (
            float(metric["volume_coverage"]) >= float(gate["volume_coverage_minimum"])
            and (
                service_distance_p95 <= service_distance_limit
                or math.isclose(service_distance_p95, service_distance_limit, rel_tol=0.0, abs_tol=1e-9)
            )
        ):
            break
    # Audit-only labels are a separate fixed exact-target budget and never
    # affect the primary adaptive stopping curve.
    if len(audit):
        batch_dir = stage / "audit_only_batch"
        solved = _solve_target_batch(config, audit, batch_dir, smoke=smoke, namespace="audit_only")
        _write_parquet(audit, batch_dir / "targets.parquet")
        _write_parquet(solved, batch_dir / "candidate_bank.parquet")
        manifest = {"batch_index": -1, "batch_kind": "audit_only", "target_count": len(audit), "candidate_count": len(solved), "target_sha256": sha256_file(batch_dir / "targets.parquet"), "candidate_sha256": sha256_file(batch_dir / "candidate_bank.parquet"), "immutable": True}
        _write_json(batch_dir / "batch_manifest.json", manifest)
        batch_rows.append(manifest)
        banks.append(solved)
    full = pd.concat(banks, ignore_index=True, sort=False).drop_duplicates(["target_id", "candidate_id"], keep="last")
    _write_parquet(full, stage / "full_candidate_bank.parquet")
    _write_parquet(pd.DataFrame(batch_rows), stage / "immutable_batch_registry.parquet")
    _write_parquet(selected_targets, stage / "final_primary_target_registry.parquet")
    _write_parquet(pd.DataFrame(fill_rows), stage / "adaptive_fill_curve.parquet")
    return _seal_gate(output_root, config, "adaptive_fill", {
        "status": "complete", "primary_target_count": len(selected_targets), "primary_pool_count": len(primary_pool), "audit_target_count": len(audit),
        "candidate_attempt_count": len(full), "immutable_batch_count": len(batch_rows),
        "proposal_beta_used_as_label_or_hint": False,
    })


def stage_graph_teacher(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["graph_teacher"]
    if _gate(output_root, "adaptive_fill")["status"] != "complete":
        for name in ("old_anchor_registry", "primary_direct_teacher_labels", "primary_teacher_graph", "teacher_lambda_sweep", "teacher_root_attachment", "served_cell_registry", "axial_slice_coverage", "path_query_audit"):
            _write_parquet(pd.DataFrame(), stage / f"{name}.parquet")
        return _seal_gate(output_root, config, "graph_teacher", {"status": "not_authorized", "data_axis_green": False, "teacher_zero_connected": False})
    primary = pd.read_parquet(output_root / STAGE_DIRS["adaptive_fill"] / "final_primary_target_registry.parquet")
    bank = pd.read_parquet(output_root / STAGE_DIRS["adaptive_fill"] / "full_candidate_bank.parquet")
    bank = bank[bank["target_id"].isin(primary["target_id"])].copy()
    anchors = _anchor_registry(primary)
    candidates = pd.concat([bank, _candidate_rows_for_old(primary, anchors)], ignore_index=True, sort=False).drop_duplicates(["target_id", "candidate_id"], keep="last")
    labels, edges, trials = _select_teacher(config, candidates, primary, anchors)
    labels = labels.drop(columns=list(XYZ_COLUMNS), errors="ignore").merge(
        primary[["target_id", *XYZ_COLUMNS]],
        on="target_id",
        how="left",
        validate="one_to_one",
    )
    if not np.isfinite(labels.loc[:, XYZ_COLUMNS].to_numpy(float)).all():
        raise ValueError("graph teacher labels must map to finite frozen target coordinates")
    teacher_connected, component, root_audit = teacher_zero_attachment(
        labels, edges, zero_target_id="retry19_exact_zero",
        residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]),
        beta_gap_maximum_deg=float(config["graph_teacher"]["teacher_root_beta_gap_maximum_deg"]),
        weights=config["candidate_solver"]["beta_weights"],
    )
    cells = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "mixed_resolution_cell_registry.parquet")
    served = mark_served_cells(cells, labels, teacher_zero_connected_ids=component, service_radius_mm=float(config["workspace"]["service_radius_mm"]))
    face_edges = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "primary_face_adjacency.parquet")
    coverage = mixed_resolution_coverage_metrics(served[served["domain_class"].eq("primary")])
    hole = largest_unserved_component_fraction(served[served["domain_class"].eq("primary")], face_edges)
    slices = axial_slice_coverage(served[served["domain_class"].eq("primary")], zero_x_mm=float(config["workspace"]["zero_x_m"]) * 1000.0, bin_edges_u_mm=np.arange(0, 200 + float(config["coverage_gate"]["axial_bin_mm"]), float(config["coverage_gate"]["axial_bin_mm"])))
    queries = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "path_query_registry.parquet")
    path_metric, path_audit = path_query_metrics(served, labels[labels["target_id"].isin(component)], edges, queries, service_radius_mm=float(config["workspace"]["service_radius_mm"])) if len(queries) else ({"success_rate": 0.0, "stretch_p95": math.inf}, pd.DataFrame())
    zero_xyz = labels.loc[labels["target_id"].eq("retry19_exact_zero"), XYZ_COLUMNS].to_numpy(float)
    nonzero_xyz = labels.loc[~labels["target_id"].eq("retry19_exact_zero"), XYZ_COLUMNS].to_numpy(float)
    nearest_zero_nonzero_mm = (
        float(np.linalg.norm(nonzero_xyz - zero_xyz[0], axis=1).min() * 1000.0)
        if len(zero_xyz) == 1 and len(nonzero_xyz)
        else math.inf
    )
    gate = data_gate(
        coverage, maximum_hole_fraction=hole, minimum_slice_coverage=float(slices["volume_coverage"].min()),
        path_metrics=path_metric,
        geometric_zero_connected=bool(
            len(served[served["required"] & served["domain_class"].eq("primary")])
            and served.loc[served["required"] & served["domain_class"].eq("primary"), "geometric_zero_connected"].all()
        ),
        teacher_zero_connected=teacher_connected, nearest_zero_nonzero_mm=nearest_zero_nonzero_mm,
        policy=_coverage_policy(config),
    )
    edge_metric = teacher_edge_metrics(labels, edges, weights=config["candidate_solver"]["beta_weights"])
    gate["checks"]["teacher_edge_p95"] = edge_metric["weighted_p95_deg"] <= float(config["coverage_gate"]["edge_weighted_p95_maximum_deg"])
    gate["checks"]["teacher_edge_gt7_rate"] = edge_metric["raw_gt7_rate"] <= float(config["coverage_gate"]["edge_raw_gt7_rate_maximum"])
    gate["status"] = "green" if all(gate["checks"].values()) else "red"
    _write_parquet(anchors, stage / "old_anchor_registry.parquet")
    _write_parquet(labels, stage / "primary_direct_teacher_labels.parquet")
    _write_parquet(edges, stage / "primary_teacher_graph.parquet")
    _write_parquet(trials, stage / "teacher_lambda_sweep.parquet")
    _write_parquet(root_audit, stage / "teacher_root_attachment.parquet")
    _write_parquet(served, stage / "served_cell_registry.parquet")
    _write_parquet(slices, stage / "axial_slice_coverage.parquet")
    _write_parquet(path_audit, stage / "path_query_audit.parquet")
    return _seal_gate(output_root, config, "graph_teacher", {
        **gate, "data_axis_green": gate["status"] == "green", "coverage": coverage,
        "nearest_zero_nonzero_mm": nearest_zero_nonzero_mm,
        "maximum_hole_fraction": hole, "minimum_axial_slice_coverage": float(slices["volume_coverage"].min()),
        "path_query": path_metric, "label_continuity": edge_metric,
    })


def _materialize_jacobians(frame: pd.DataFrame, environment: Any) -> pd.DataFrame:
    result = frame.copy()
    beta = result.loc[:, BETA_COLUMNS].to_numpy(float)
    _xyz, jacobian = environment.fk_and_jacobian(beta)
    jacobian = np.asarray(jacobian, dtype=float).reshape(-1, 18)
    result.loc[:, JACOBIAN_COLUMNS] = jacobian
    return result


def stage_unified_dataset(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["unified_dataset"]
    primary = pd.read_parquet(output_root / STAGE_DIRS["graph_teacher"] / "primary_direct_teacher_labels.parquet")
    if primary.empty:
        for name in ("audit_only_direct_labels", "audit_only_teacher_graph", "audit_only_teacher_sweep", "overlap_compatibility", "unified_signed_supervision", "unified_split_registry"):
            _write_parquet(pd.DataFrame(), stage / f"{name}.parquet")
        _write_json(stage / "overlap_gate.json", {"compatible": False, "reason": "primary_teacher_not_available"})
        return _seal_gate(output_root, config, "unified_dataset", {"status": "not_available", "overlap_compatible": False, "static_single_section_failure": True, "unified_global_student_authorized": False, "student_training_authorized": False})
    targets = pd.read_parquet(output_root / STAGE_DIRS["adaptive_fill"] / "final_primary_target_registry.parquet")
    primary = primary.merge(targets[["target_id", *XYZ_COLUMNS, "domain_class"]], on="target_id", how="left", suffixes=("", "_target"))
    for column in XYZ_COLUMNS:
        if f"{column}_target" in primary:
            primary[column] = primary[f"{column}_target"]
    history = _historical_signed_rows(config)
    history["u_mm"] = 1000.0 * (float(config["workspace"]["zero_x_m"]) - history["x_m"])
    outer = history[history["u_mm"].gt(float(config["workspace"]["primary_u_maximum_mm"]))].copy()
    audit_targets = pd.read_parquet(output_root / STAGE_DIRS["fullspace_discovery"] / "audit_only_target_registry.parquet")
    bank = pd.read_parquet(output_root / STAGE_DIRS["adaptive_fill"] / "full_candidate_bank.parquet")
    audit_bank = bank[bank["target_id"].isin(audit_targets["target_id"])].copy()
    audit_anchors = _anchor_registry(audit_targets)
    audit_candidates = pd.concat([audit_bank, _candidate_rows_for_old(audit_targets, audit_anchors)], ignore_index=True, sort=False).drop_duplicates(["target_id", "candidate_id"], keep="last")
    if len(audit_targets) and len(audit_candidates):
        audit_labels, audit_edges, audit_trials = _select_teacher(config, audit_candidates, audit_targets, audit_anchors)
    else:
        audit_labels, audit_edges, audit_trials = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    direct_overlap = pd.concat([
        primary[1000.0 * (float(config["workspace"]["zero_x_m"]) - primary["x_m"]) >= float(config["overlap_gate"]["u_minimum_mm"])],
        audit_labels,
    ], ignore_index=True, sort=False).drop_duplicates("target_id")
    retry18_overlap = history[history["target_id"].isin(direct_overlap.get("target_id", pd.Series(dtype=str)))].copy()
    compatibility, comparison = overlap_compatibility(
        direct_overlap, retry18_overlap, weights=config["candidate_solver"]["beta_weights"],
        p95_maximum_deg=float(config["overlap_gate"]["weighted_gap_p95_maximum_deg"]),
        gt7_rate_maximum=float(config["overlap_gate"]["weighted_gap_gt7_rate_maximum"]),
    )
    training_scope = "primary_plus_retry18_outer" if compatibility["compatible"] else "primary_core_only"
    unified = pd.concat([primary, outer], ignore_index=True, sort=False) if compatibility["compatible"] else primary.copy()
    unified["domain_class"] = np.where(1000.0 * (float(config["workspace"]["zero_x_m"]) - unified["x_m"]) <= 200.0, "primary", "retry18_outer")
    unified, split_registry = unified_split_registry(unified, history, block_size_mm=float(config["split"]["macroblock_mm"]), seed=int(config["split"]["seed"]))
    orbit_split_conflict_count = (
        int(unified.groupby("symmetry_orbit_id")["split_role"].nunique().gt(1).sum())
        if "symmetry_orbit_id" in unified
        else 0
    )
    unified = _materialize_jacobians(unified, _environment(config))
    _write_parquet(audit_labels, stage / "audit_only_direct_labels.parquet")
    _write_parquet(audit_edges, stage / "audit_only_teacher_graph.parquet")
    _write_parquet(audit_trials, stage / "audit_only_teacher_sweep.parquet")
    _write_parquet(comparison, stage / "overlap_compatibility.parquet")
    _write_json(stage / "overlap_gate.json", compatibility)
    _write_parquet(unified, stage / "unified_signed_supervision.parquet")
    _write_parquet(split_registry, stage / "unified_split_registry.parquet")
    return _seal_gate(output_root, config, "unified_dataset", {
        "status": "complete",
        "overlap_compatible": compatibility["compatible"],
        "static_single_section_failure": not compatibility["compatible"],
        "unified_global_student_authorized": compatibility["compatible"],
        "student_training_authorized": len(primary) > 0,
        "training_scope": training_scope,
        "symmetry_orbit_cross_split_count": orbit_split_conflict_count,
        "symmetry_orbit_split_is_diagnostic": True,
        "primary_direct_row_count": len(primary), "outer_retry18_row_count": len(outer),
        "audit_only_rows_enter_training": False,
    })


def _model_manifest(model_id: str, cfg: DirectStudentConfig, seed: int, dataset_sha: str) -> Mapping[str, Any]:
    return {
        "schema_version": 1, "model_id": model_id, "seed": int(seed),
        "radial_scale_mm": cfg.radial_scale_mm, "signed_power_alpha": cfg.signed_power_alpha,
        "signed_power_epsilon_mm": cfg.signed_power_epsilon_mm,
        "signed_power_epsilon_normalized": cfg.signed_power_epsilon_mm / cfg.radial_scale_mm,
        "hidden_units": list(cfg.hidden_units),
        "edge_lambda": cfg.edge_lambda, "jacobian_lambda": cfg.jacobian_lambda,
        "dataset_sha256": dataset_sha,
    }


def stage_student_ablations(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["student_ablations"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("retry19 Student requires CUDA_VISIBLE_DEVICES=-1")
    if not _gate(output_root, "unified_dataset")["student_training_authorized"]:
        _write_parquet(pd.DataFrame(), stage / "three_seed_model_metrics.parquet")
        return _seal_gate(output_root, config, "student_ablations", {"status": "not_authorized_no_primary_teacher", "model_count": 0})
    import tensorflow as tf
    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("retry19 CPU Student sees GPU")
    dataset_path = output_root / STAGE_DIRS["unified_dataset"] / "unified_signed_supervision.parquet"
    data = pd.read_parquet(dataset_path)
    train = data[data["split_role"].eq("train")].copy()
    valid = data[data["split_role"].eq("validation")].copy()
    teacher_edges = pd.read_parquet(output_root / STAGE_DIRS["graph_teacher"] / "primary_teacher_graph.parquet")
    if len(teacher_edges) > int(config["student"]["edge_registry_cap"]):
        teacher_edges["digest"] = teacher_edges["edge_id"].astype(str).map(lambda value: hashlib.sha256(value.encode()).hexdigest())
        teacher_edges = teacher_edges.sort_values("digest").head(int(config["student"]["edge_registry_cap"])).drop(columns="digest")
    environment = _environment(config)
    zero = np.asarray(environment.fk(np.zeros(6))).reshape(3)
    old_waypoints = pd.read_parquet(_upstream_path(config, "retry18_heldout_waypoints"))
    specs = (
        ("D0_signed_xyz", 1.0, 0.0, 0.0, False, config["student"]["hidden_units"]),
        ("D1_signed_power", 0.5, 0.0, 0.0, True, config["student"]["hidden_units"]),
        ("D2_jacobian", 0.5, 0.0, float(config["student"]["jacobian_lambda"]), True, config["student"]["hidden_units"]),
        ("D3_graph_edge", 0.5, float(config["student"]["edge_lambda"]), float(config["student"]["jacobian_lambda"]), True, config["student"]["hidden_units"]),
        ("D4_capacity", 0.5, float(config["student"]["edge_lambda"]), float(config["student"]["jacobian_lambda"]), True, config["student"]["capacity_hidden_units"]),
    )
    rows: list[dict[str, Any]] = []
    primary_seed = int(config["student"]["primary_seed"])

    def train_one(spec: tuple[Any, ...], seed: int) -> None:
        model_id, alpha, edge_lambda, jacobian_lambda, include_power, hidden_units = spec
        model_cfg = _direct_config(
            config, seed=seed, alpha=alpha, edge_lambda=edge_lambda,
            jacobian_lambda=jacobian_lambda, hidden_units=hidden_units, smoke=smoke,
        )
        model, history = train_direct_student(
            train, valid, teacher_edges, zero_x_m=zero[0], beta_bounds_rad=np.asarray(environment.bounds),
            config=model_cfg, include_signed_power=include_power, beta_coordinate_weights=config["candidate_solver"]["beta_weights"],
        )
        model_dir = stage / "models" / model_id / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=False)
        model.save(model_dir / "direct.keras")
        _write_parquet(history, model_dir / "training_history.parquet")
        _write_json(model_dir / "model_manifest.json", _model_manifest(model_id, model_cfg, seed, sha256_file(dataset_path)))
        model_metrics = _model_metrics(model, old_waypoints, environment, direct=True, zero=zero)
        xyz = valid.loc[:, XYZ_COLUMNS].to_numpy(float)
        prediction = np.asarray(model(np.asarray(xyz, np.float32), training=False), float)
        residual = np.linalg.norm(np.asarray(environment.fk(prediction)).reshape(-1, 3) - xyz, axis=1) * 1000.0
        rows.append({"model_id": model_id, "seed": seed, **model_metrics, "validation_fk_p95_mm": float(np.percentile(residual, 95)), "validation_fk_maximum_mm": float(np.max(residual))})

    for spec in specs:
        train_one(spec, primary_seed)
    screen = pd.DataFrame(rows).sort_values(
        ["raw_max_spike_mm", "rectangle_fk_p95_mm", "validation_fk_p95_mm", "model_id"],
        kind="stable",
    )
    finalist_ids = screen.head(2)["model_id"].astype(str).tolist()
    spec_by_id = {str(spec[0]): spec for spec in specs}
    for model_id in finalist_ids:
        for seed in map(int, config["student"]["robustness_seeds"]):
            train_one(spec_by_id[model_id], seed)
    metrics = pd.DataFrame(rows)
    _write_parquet(metrics, stage / "three_seed_model_metrics.parquet")
    _write_json(stage / "primary_seed_screen.json", {"primary_seed": primary_seed, "finalist_model_ids": finalist_ids})
    return _seal_gate(output_root, config, "student_ablations", {
        "status": "complete", "model_count": metrics["model_id"].nunique(),
        "fit_count": len(metrics), "finalist_model_ids": finalist_ids,
        "primary_seed_screen_count": len(specs), "robustness_seed_rerun_model_count": len(finalist_ids),
        "raw_first_selection": True,
    })


def _tree_manifest(root: Path) -> Mapping[str, Any]:
    return {"artifacts": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(root.rglob("*")) if path.is_file()]}


def stage_student_lock(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["student_lock"]
    metrics = pd.read_parquet(output_root / STAGE_DIRS["student_ablations"] / "three_seed_model_metrics.parquet")
    if metrics.empty:
        return _seal_gate(output_root, config, "student_lock", {"status": "not_available", "model_locked": False})
    robust = metrics.groupby("model_id", as_index=False).agg(
        raw_max_spike_mm=("raw_max_spike_mm", "max"), rectangle_fk_p95_mm=("rectangle_fk_p95_mm", "max"),
        rectangle_fk_maximum_mm=("rectangle_fk_maximum_mm", "max"), rectangle_path_step_excess_maximum_mm=("rectangle_path_step_excess_maximum_mm", "max"),
        validation_fk_p95_mm=("validation_fk_p95_mm", "max"), seed_count=("seed", "nunique"),
    )
    primary = metrics[metrics["seed"].eq(int(config["student"]["primary_seed"]))].copy()
    primary = primary.merge(robust[["model_id", "seed_count"]], on="model_id", how="left", validate="one_to_one")
    student = config["student"]
    primary["prelock_raw_green"] = (
        primary["rectangle_fk_p95_mm"].le(float(student["old_rectangle_fk_p95_maximum_mm"]))
        & primary["rectangle_fk_maximum_mm"].le(float(student["old_rectangle_fk_maximum_mm"]))
    )
    primary["old_rectangle_path_excess_diagnostic_green"] = primary[
        "rectangle_path_step_excess_maximum_mm"
    ].le(float(student["old_rectangle_path_excess_maximum_mm"]))
    primary["selection_score"] = primary["raw_max_spike_mm"] + primary["rectangle_fk_p95_mm"] + 0.25 * primary["validation_fk_p95_mm"]
    direct_candidates = primary[primary["seed_count"].eq(3)].copy()
    eligible = direct_candidates[direct_candidates["prelock_raw_green"]]
    selected = (eligible if len(eligible) else direct_candidates).sort_values(["selection_score", "model_id"], kind="stable").iloc[0]
    selected_id = str(selected["model_id"])
    dataset = output_root / STAGE_DIRS["unified_dataset"] / "unified_signed_supervision.parquet"
    teacher = output_root / STAGE_DIRS["graph_teacher"] / "primary_direct_teacher_labels.parquet"
    split = output_root / STAGE_DIRS["unified_dataset"] / "unified_split_registry.parquet"
    model_root = output_root / STAGE_DIRS["student_ablations"] / "models" / selected_id
    lock = {
        "schema_version": 1, "locked_before_new_trajectory_generation": True,
        "dataset": {"path": str(dataset.relative_to(output_root)), "sha256": sha256_file(dataset)},
        "teacher": {"path": str(teacher.relative_to(output_root)), "sha256": sha256_file(teacher)},
        "split": {"path": str(split.relative_to(output_root)), "sha256": sha256_file(split)},
        "selected_model_id": selected_id, "model_root": str(model_root.relative_to(output_root)),
        "model_manifest": _tree_manifest(model_root), "selection_uses_dls2": False,
    }
    _write_parquet(primary, stage / "raw_first_model_selection.parquet")
    _write_parquet(robust, stage / "seed_robustness_diagnostics.parquet")
    _write_json(stage / "dataset_teacher_split_model_lock.json", lock)
    return _seal_gate(output_root, config, "student_lock", {
        "status": "locked", "model_locked": True, "selected_model_id": selected_id,
        "prelock_raw_green": bool(selected["prelock_raw_green"]),
        "old_rectangle_path_excess_is_diagnostic": True,
        "selected_model_seed_count": int(selected["seed_count"]),
        "seed_robustness_is_diagnostic": True,
    })


def _verify_lock(output_root: Path) -> bool:
    lock = _read_json(output_root / STAGE_DIRS["student_lock"] / "dataset_teacher_split_model_lock.json")
    return all(sha256_file(output_root / lock[key]["path"]) == lock[key]["sha256"] for key in ("dataset", "teacher", "split")) and _tree_manifest(output_root / lock["model_root"]) == lock["model_manifest"]


def _curve_frame(xyz: np.ndarray, trajectory_id: str, shape_class: str, *, role: str) -> pd.DataFrame:
    frame = pd.DataFrame(xyz, columns=XYZ_COLUMNS)
    frame["trajectory_id"] = trajectory_id; frame["shape_class"] = shape_class; frame["trajectory_role"] = role
    frame["waypoint_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _new_trajectory_inventory(config: Mapping[str, Any], dataset: pd.DataFrame) -> pd.DataFrame:
    count = int(config["trajectories"]["waypoint_count"])
    zero = np.asarray([config["workspace"]["zero_x_m"], 0.0, 0.0])
    xyz = dataset.loc[:, XYZ_COLUMNS].to_numpy(float)
    centre = np.median(xyz, axis=0)
    scale = min(50.0, 0.2 * float(np.ptp(xyz[:, 0]) * 1000.0))
    frames = []
    rectangle = unit_shape("rounded_rectangle", count)
    for index in range(3):
        _normal, e1, e2 = plane_basis(45.0, index * 45.0)
        local = centre + scale / 1000.0 * (rectangle[:, :1] * e1 + rectangle[:, 1:] * e2)
        local[:, 1] += (index - 1) * 0.003
        frames.append(_curve_frame(local, f"retry19_seam_between_rectangle_{index}", "new_rectangle", role="seam_between_waypoints"))
    for index in range(3):
        _normal, e1, e2 = plane_basis(60.0, index * 45.0)
        local = centre + scale / 1000.0 * (rectangle[:, :1] * e1 + rectangle[:, 1:] * e2)
        local[np.argmin(np.abs(local[:, 1])), 1] = 0.0
        frames.append(_curve_frame(local, f"retry19_exact_seam_rectangle_{index}", "new_rectangle", role="exact_seam_waypoint"))
    _normal, e1, e2 = plane_basis(45.0, 30.0)
    frames.append(_curve_frame(centre + scale / 1000.0 * (unit_shape("ellipse", count)[:, :1] * e1 + unit_shape("ellipse", count)[:, 1:] * e2), "retry19_slanted_ellipse", "slanted_ellipse", role="new_postlock"))
    frames.append(_curve_frame(centre + scale / 1000.0 * (unit_shape("rounded_star", count)[:, :1] * e1 + unit_shape("rounded_star", count)[:, 1:] * e2), "retry19_star", "star", role="new_postlock"))
    phase = np.linspace(0, 4 * np.pi, count)
    spiral = np.column_stack([centre[0] + np.linspace(-scale, scale, count) / 1000.0, centre[1] + 0.5 * scale * np.cos(phase) / 1000.0, centre[2] + 0.5 * scale * np.sin(phase) / 1000.0])
    frames.append(_curve_frame(spiral, "retry19_spiral_3d", "spiral_3d", role="new_postlock"))
    boundary_indices = np.linspace(0, len(xyz) - 1, 3, dtype=int)
    for index, endpoint in enumerate(xyz[boundary_indices]):
        line = zero + np.linspace(0, 1, count)[:, None] * (endpoint - zero)
        frames.append(_curve_frame(line, f"retry19_zero_to_boundary_{index}", "zero_to_boundary", role="new_postlock"))
    return pd.concat(frames, ignore_index=True, sort=False)


def stage_postlock_trajectories(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["postlock_trajectories"]
    lock_path = output_root / STAGE_DIRS["student_lock"] / "dataset_teacher_split_model_lock.json"
    if not lock_path.exists():
        _write_parquet(pd.DataFrame(), stage / "postlock_15_trajectory_waypoints.parquet")
        _write_parquet(pd.DataFrame(), stage / "trajectory_inventory.parquet")
        return _seal_gate(output_root, config, "postlock_trajectories", {"status": "not_authorized_no_model_lock", "trajectory_count": 0, "generated_after_lock": False})
    if not _verify_lock(output_root):
        raise RuntimeError("retry19 dataset/Teacher/split/model lock mutated")
    old = pd.read_parquet(_upstream_path(config, "retry18_heldout_waypoints"))
    old = old[old["shape_class"].eq("rounded_rectangle")].copy()
    old_ids = sorted(old["trajectory_id"].astype(str).unique())[:3]
    old = old[old["trajectory_id"].astype(str).isin(old_ids)].copy()
    old["trajectory_role"] = "retry18_regression_not_new_heldout"
    data = pd.read_parquet(output_root / STAGE_DIRS["unified_dataset"] / "unified_signed_supervision.parquet")
    primary_data = data[data["domain_class"].eq("primary")].copy()
    new = _new_trajectory_inventory(config, primary_data)
    if smoke:
        old = old.groupby("trajectory_id", sort=False).head(32)
        new = new.groupby("trajectory_id", sort=False).head(32)
    waypoints = pd.concat([old, new], ignore_index=True, sort=False)
    inventory = waypoints.groupby(["trajectory_id", "shape_class", "trajectory_role"], as_index=False).size().rename(columns={"size": "waypoint_count"})
    _write_parquet(waypoints, stage / "postlock_15_trajectory_waypoints.parquet")
    _write_parquet(inventory, stage / "trajectory_inventory.parquet")
    return _seal_gate(output_root, config, "postlock_trajectories", {"status": "complete", "trajectory_count": inventory["trajectory_id"].nunique(), "old_rectangle_regression_count": len(old_ids), "generated_after_lock": True})


def stage_trajectory_teacher(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["trajectory_teacher"]
    if _gate(output_root, "postlock_trajectories")["status"] != "complete":
        _write_parquet(pd.DataFrame(), stage / "trajectory_candidate_bank.parquet")
        _write_parquet(pd.DataFrame(), stage / "trajectory_teacher_labels.parquet")
        return _seal_gate(output_root, config, "trajectory_teacher", {"status": "not_authorized", "teacher_complete": False})
    if not _verify_lock(output_root):
        raise RuntimeError("retry19 lock mutated before trajectory Teacher")
    waypoints = pd.read_parquet(output_root / STAGE_DIRS["postlock_trajectories"] / "postlock_15_trajectory_waypoints.parquet")
    waypoints["target_id"] = [f"{trajectory}:{int(index):05d}" for trajectory, index in zip(waypoints["trajectory_id"], waypoints["waypoint_index"], strict=True)]
    candidates = retry17._solve_candidates(
        config, waypoints, stage, seed_budget=int(config["candidate_solver"]["ordinary_seed_budget"]), smoke=smoke,
        maximum_workers=int(config["runtime"]["trajectory_candidate_workers"]), work_namespace="trajectory_candidate_bank",
    )
    candidates = candidates.merge(
        waypoints[["target_id", "trajectory_id", "waypoint_index"]],
        on="target_id",
        how="left",
        validate="many_to_one",
    )
    labels = []
    for trajectory_id, part in waypoints.groupby("trajectory_id", sort=True):
        local = candidates[candidates["target_id"].isin(part["target_id"])]
        selected = retry17._cycle_teacher(local, pairwise_lambda=float(config["graph_teacher"]["selected_pairwise_lambda"]), weights=config["candidate_solver"]["beta_weights"], tau_deg=float(config["graph_teacher"]["huber_delta_deg"]))
        selected["trajectory_id"] = trajectory_id
        labels.append(selected)
    teacher = pd.concat(labels, ignore_index=True, sort=False) if labels else pd.DataFrame()
    _write_parquet(candidates, stage / "trajectory_candidate_bank.parquet")
    _write_parquet(teacher, stage / "trajectory_teacher_labels.parquet")
    complete = teacher["target_id"].nunique() == waypoints["target_id"].nunique() if len(teacher) else False
    return _seal_gate(output_root, config, "trajectory_teacher", {"status": "complete" if complete else "teacher_red", "teacher_complete": complete, "proposal_beta_used_as_label_or_hint": False, "trajectory_labels_added_back_to_training": False})


def _load_selected_models(output_root: Path) -> Mapping[int, Any]:
    lock = _read_json(output_root / STAGE_DIRS["student_lock"] / "dataset_teacher_split_model_lock.json")
    root = output_root / lock["model_root"]
    models = {}
    for seed_dir in sorted(root.glob("seed_*")):
        models[int(seed_dir.name.split("_")[-1])] = load_direct_student(seed_dir / "direct.keras")
    return models


def stage_trajectory_evaluation(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["trajectory_evaluation"]
    trajectory_teacher_gate = _gate(output_root, "trajectory_teacher")
    trajectory_teacher_complete = trajectory_teacher_gate["status"] == "complete"
    lock_path = output_root / STAGE_DIRS["student_lock"] / "dataset_teacher_split_model_lock.json"
    if not lock_path.is_file():
        for name in ("trajectory_report", "trajectory_waypoint_evaluation", "three_seed_summary"):
            _write_parquet(pd.DataFrame(), stage / f"{name}.parquet")
        return _seal_gate(output_root, config, "trajectory_evaluation", {
            "status": "not_authorized_no_model_lock",
            "raw_student_axis_green": False,
            "dls2_axis_green": False,
            "dls2_does_not_override_raw": True,
            "trajectory_teacher_complete": trajectory_teacher_complete,
            "trajectory_teacher_completeness_is_diagnostic": True,
        })
    if not _verify_lock(output_root):
        raise RuntimeError("retry19 lock mutated before trajectory evaluation")
    waypoints = pd.read_parquet(output_root / STAGE_DIRS["postlock_trajectories"] / "postlock_15_trajectory_waypoints.parquet")
    environment = _environment(config)
    zero = np.asarray(environment.fk(np.zeros(6))).reshape(3)
    models = _load_selected_models(output_root)
    reports = []
    evaluated = []
    for seed, model in models.items():
        for trajectory_id, part in waypoints.groupby("trajectory_id", sort=True):
            ordered = part.sort_values("waypoint_index", kind="stable")
            xyz = ordered.loc[:, XYZ_COLUMNS].to_numpy(float)
            raw = np.asarray(model(np.asarray(xyz, np.float32), training=False), float)
            dls2 = retry17._two_step_dls(environment, raw, xyz, zero_xyz=zero)
            raw_metric = retry18._path_metrics(xyz, raw, environment, closed=not ordered["shape_class"].eq("zero_to_boundary").all())
            dls_metric = retry18._path_metrics(xyz, dls2, environment, closed=not ordered["shape_class"].eq("zero_to_boundary").all())
            dls_metric["success_rate"] = (
                float(
                    np.mean(
                        np.linalg.norm(np.asarray(environment.fk(dls2)).reshape(-1, 3) - xyz, axis=1) * 1000.0
                        <= float(config["trajectories"]["dls2_fk_p95_maximum_mm"])
                    )
                )
                if np.isfinite(dls2).all()
                else 0.0
            )
            shape = str(ordered["shape_class"].iloc[0])
            old = str(ordered["trajectory_role"].iloc[0]).startswith("retry18_regression")
            raw_green = bool(
                raw_metric["fk_p95_mm"] <= float(config["student"]["old_rectangle_fk_p95_maximum_mm"])
                and raw_metric["fk_maximum_mm"] <= float(config["student"]["old_rectangle_fk_maximum_mm"])
                if old else raw_metric["path_step_excess_maximum_mm"] <= float(config["student"]["new_rectangle_raw_spike_maximum_mm"]) if shape == "new_rectangle" else raw_metric["fk_p95_mm"] <= 10.0
            )
            dls_green = bool(dls_metric["success_rate"] >= float(config["trajectories"]["dls2_success_minimum"]) and dls_metric["fk_p95_mm"] <= float(config["trajectories"]["dls2_fk_p95_maximum_mm"]))
            row = {
                "seed": seed, "trajectory_id": trajectory_id, "shape_class": shape,
                "trajectory_role": str(ordered["trajectory_role"].iloc[0]),
                "raw_green": raw_green, "dls2_green": dls_green,
                "raw_step_gt7_diagnostic_green": raw_metric["raw_step_gt7_rate"] <= float(config["student"]["raw_step_gt7_rate_maximum"]),
                "old_rectangle_path_excess_diagnostic_green": (not old) or raw_metric["path_step_excess_maximum_mm"] <= float(config["student"]["old_rectangle_path_excess_maximum_mm"]),
            }
            for prefix, metric in (("raw", raw_metric), ("dls2", dls_metric)):
                row.update({f"{prefix}_{key}": value for key, value in metric.items()})
            reports.append(row)
            frame = ordered.copy(); frame["seed"] = seed
            for axis, name in enumerate(BETA_COLUMNS): frame[f"raw_{name}"] = raw[:, axis]; frame[f"dls2_{name}"] = dls2[:, axis]
            evaluated.append(frame)
    report = pd.DataFrame(reports)
    points = pd.concat(evaluated, ignore_index=True, sort=False) if evaluated else pd.DataFrame()
    seed_summary = report.groupby("seed", as_index=False).agg(raw_green_count=("raw_green", "sum"), dls2_green_count=("dls2_green", "sum"), trajectory_count=("trajectory_id", "nunique"))
    primary_seed = int(config["student"]["primary_seed"])
    primary_summary = seed_summary[seed_summary["seed"].eq(primary_seed)]
    primary_old = report[
        report["seed"].eq(primary_seed)
        & report["trajectory_role"].str.startswith("retry18_regression")
    ]
    raw_axis = bool(
        len(primary_summary) == 1
        and int(primary_summary.iloc[0]["raw_green_count"]) >= int(config["student"]["raw_green_trajectory_minimum"])
        and len(primary_old) == 3
        and primary_old["raw_green"].all()
    )
    dls_axis = bool(
        len(primary_summary) == 1
        and int(primary_summary.iloc[0]["dls2_green_count"]) == int(primary_summary.iloc[0]["trajectory_count"])
    )
    _write_parquet(report, stage / "trajectory_report.parquet")
    _write_parquet(points, stage / "trajectory_waypoint_evaluation.parquet")
    _write_parquet(seed_summary, stage / "three_seed_summary.parquet")
    return _seal_gate(output_root, config, "trajectory_evaluation", {
        "status": "complete", "raw_student_axis_green": raw_axis,
        "dls2_axis_green": dls_axis, "dls2_does_not_override_raw": True,
        "primary_seed": primary_seed, "seed_robustness_is_diagnostic": True,
        "trajectory_teacher_complete": trajectory_teacher_complete,
        "trajectory_teacher_completeness_is_diagnostic": True,
        "raw_step_gt7_rate_is_diagnostic": True,
        "old_rectangle_path_excess_is_diagnostic": True,
    })


def _artifact_manifest(output_root: Path) -> Mapping[str, Any]:
    artifacts = []
    roots = [output_root / "00_objective_feasibility"] + [
        output_root / STAGE_DIRS[stage_name] for stage_name in STAGE_ORDER[:-1]
    ]
    for artifact_root in roots:
        for path in sorted(artifact_root.rglob("*")):
            if path.is_file() and "_work" not in path.parts:
                artifacts.append({"path": str(path.relative_to(output_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "scientific_source_fixed_point": _git_sha(), "artifacts": artifacts}


def stage_summary(config: Mapping[str, Any], output_root: Path, *, smoke: bool) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    operational = all(_complete(output_root, name) for name in STAGE_ORDER[:-1])
    graph = _gate(output_root, "graph_teacher")
    unified = _gate(output_root, "unified_dataset")
    trajectory = _gate(output_root, "trajectory_evaluation")
    metrics_path = output_root / STAGE_DIRS["student_ablations"] / "three_seed_model_metrics.parquet"
    metrics = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
    control_path = output_root / STAGE_DIRS["causal_controls"] / "control_metrics.parquet"
    controls = pd.read_parquet(control_path) if control_path.exists() else pd.DataFrame()
    contrasts: dict[str, Any] = {
        "teacher_static_section_failed": bool(unified.get("static_single_section_failure", False)),
        "replicated_seed_pass_count": 3 if trajectory.get("raw_student_axis_green", False) else 0,
    }
    if len(metrics):
        grouped = metrics.groupby("model_id")["raw_max_spike_mm"].max()
        control_grouped = controls.groupby("model_id")["raw_max_spike_mm"].max() if len(controls) else pd.Series(dtype=float)
        contrasts.update({
            "q0_augmented_raw_max_spike_mm": float(control_grouped.get("Q0_augmented", math.inf)),
            "f0_direct_raw_max_spike_mm": float(control_grouped.get("F0_direct", grouped.get("F0_direct", math.inf))),
            "f0_nonregression": bool(control_grouped.get("F0_direct", math.inf) <= control_grouped.get("Q0_augmented", math.inf) * 1.10),
            "best_without_jacobian_raw_max_spike_mm": float(grouped.drop(labels=["D2_jacobian", "D3_graph_edge", "D4_capacity"], errors="ignore").min()),
            "jacobian_raw_max_spike_mm": float(grouped.get("D2_jacobian", math.inf)),
            "jacobian_crosses_raw_gate": bool(trajectory.get("raw_student_axis_green", False)),
        })
    cause = classify_causal_evidence(contrasts)
    gate = {
        "status": "complete" if operational else "incomplete",
        "operational_completion": operational, "artifact_completeness": operational,
        "diagnostic_smoke": bool(smoke),
        "data_axis": "GREEN" if graph.get("data_axis_green", False) else "RED",
        "teacher_axis": "GREEN" if graph.get("teacher_zero_connected", False) and graph.get("checks", {}).get("teacher_edge_p95", False) and graph.get("checks", {}).get("teacher_edge_gt7_rate", False) else "RED",
        "overlap_axis": "GREEN" if unified.get("overlap_compatible", False) else "RED",
        "raw_student_axis": "GREEN" if trajectory.get("raw_student_axis_green", False) else "RED",
        "dls2_axis": "GREEN" if trajectory.get("dls2_axis_green", False) else "RED",
        "static_single_section_failure": bool(unified.get("static_single_section_failure", False)),
        **cause,
        "resource_infeasible": not _gate(output_root, "fullspace_discovery").get("objective_resource_feasible", False),
        "scientific_failure": not all((graph.get("data_axis_green", False), trajectory.get("raw_student_axis_green", False))),
        "dls2_does_not_override_raw": True,
        "claim_boundary": "diagnostic empirical robust proxy for 0<=u<=200mm; not mathematical continuous reachability or deployment",
    }
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "retry19_summary.html").write_text("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>retry19 summary</title></head><body><h1>retry19 四轴诊断</h1><pre>" + json.dumps(gate, ensure_ascii=False, indent=2, default=_json_default) + "</pre></body></html>", encoding="utf-8")
    sealed = _seal_gate(output_root, config, "summary", gate)
    _write_json(stage / "artifact_manifest.json", _artifact_manifest(output_root))
    _seal_stage(output_root, "summary")
    _write_json(output_root / "progress.json", {"status": "complete", "phase": "summary", "completed": len(STAGE_ORDER), "total": len(STAGE_ORDER), "observed_at_unix": time.time()})
    return sealed


STAGE_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "fullspace_discovery": stage_fullspace_discovery,
    "causal_controls": stage_causal_controls,
    "candidate_pilot": stage_candidate_pilot,
    "continuation_authorization": stage_continuation_authorization,
    "adaptive_fill": stage_adaptive_fill,
    "graph_teacher": stage_graph_teacher,
    "unified_dataset": stage_unified_dataset,
    "student_ablations": stage_student_ablations,
    "student_lock": stage_student_lock,
    "postlock_trajectories": stage_postlock_trajectories,
    "trajectory_teacher": stage_trajectory_teacher,
    "trajectory_evaluation": stage_trajectory_evaluation,
    "summary": stage_summary,
}


def run(
    config_path: str | Path,
    output_root: str | Path,
    binding_sha: str,
    *,
    smoke: bool = False,
    stage: str | None = None,
    validate_stage: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_root).resolve()
    _ensure_identity(config, output, binding_sha, smoke=smoke)
    if validate_stage:
        if validate_stage not in STAGE_RUNNERS:
            raise ValueError(f"unknown retry19 stage {validate_stage}")
        if not _complete(output, validate_stage):
            raise RuntimeError(f"retry19 stage is incomplete or mutated: {validate_stage}")
        return _gate(output, validate_stage)
    if stage:
        position = STAGE_ORDER.index(stage)
        if position and not _complete(output, STAGE_ORDER[position - 1]):
            raise RuntimeError("retry19 previous stage is incomplete")
        return _gate(output, stage) if _complete(output, stage) else STAGE_RUNNERS[stage](config, output, smoke=smoke)
    result: dict[str, Any] = {}
    for index, name in enumerate(STAGE_ORDER):
        _write_json(output / "progress.json", {"status": "running", "phase": name, "completed": index, "total": len(STAGE_ORDER), "observed_at_unix": time.time()})
        try:
            result = _gate(output, name) if _complete(output, name) else STAGE_RUNNERS[name](config, output, smoke=smoke)
        except Exception as error:
            failure = {
                "status": "operational_failure",
                "operational_completion": False,
                "artifact_completeness": False,
                "failed_stage": name,
                "completed_stage_count": index,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "data_axis": "NOT_EVALUATED",
                "teacher_axis": "NOT_EVALUATED",
                "overlap_axis": "NOT_EVALUATED",
                "raw_student_axis": "NOT_EVALUATED",
                "dls2_axis": "NOT_EVALUATED",
                "claim_boundary": "operational failure; no scientific outcome",
            }
            summary_dir = output / STAGE_DIRS["summary"]
            _seal_gate(output, config, "summary", failure)
            _write_json(summary_dir / "artifact_manifest.json", _artifact_manifest(output))
            _seal_stage(output, "summary")
            _write_json(output / "progress.json", {"status": "failed", "phase": name, "completed": index, "total": len(STAGE_ORDER), "observed_at_unix": time.time()})
            raise
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--binding-sha", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--validate-stage", choices=STAGE_ORDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate = run(args.config, args.output_root, args.binding_sha, smoke=args.smoke, stage=args.stage, validate_stage=args.validate_stage)
    print(json.dumps(gate, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
