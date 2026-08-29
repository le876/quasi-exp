#!/usr/bin/env python3
"""Run retry15 Omega600 diagnostic domain and Teacher preflight discovery."""

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
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import percentile, weighted_beta_rms_deg
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.retry12_symmetry import (
    BETA_COLUMNS,
    XYZ_COLUMNS,
    sobol_beta_samples,
    stable_id,
    transform_beta,
    validate_symmetry_group,
)
from quasi_exp.teacher.retry15_candidate_solver import seed_bank_xyz, teacher_seed_bank
from quasi_exp.teacher.retry15_canonical_graph import (
    Omega600Contract,
    ProbePolicy,
    build_target_registry,
    capped_profile,
    legal_candidate_clusters,
    mutual_knn_edges,
    quotient_volume_mm3,
    select_budget_profile,
    select_t1,
    select_t2,
    workspace_probe,
)


EXPERIMENT_ID = "bacra_v14_3r_retry15_omega600_domain_discovery"
STAGE_DIRS = {
    "inventory": "00_inventory",
    "workspace_probe": "01_workspace_probe",
    "domain_selection": "02_domain_selection",
    "seed_banks": "03_teacher_seed_banks",
    "teacher_preflight": "04_teacher_preflight",
    "summary": "05_summary",
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
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True
    ).strip()


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
            "PYTHONHASHSEED": "20260895",
        }
    )
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("retry15 discovery experiment_id mismatch")
    omega = config["omega600"]
    Omega600Contract(
        x0_m=float(omega["zero_x_m"]),
        x_min_m=float(omega["minimum_x_m"]),
        length_mm=float(omega["axial_length_mm"]),
    )
    if int(config["runtime"]["maximum_concurrent_workers"]) != 12:
        raise ValueError("retry15 discovery must register 12 workers")
    if int(config["runtime"]["numerical_threads_per_worker"]) != 1:
        raise ValueError("retry15 numerical workers must be single-threaded")
    proposal = config["proposal"]
    if tuple(int(proposal[key]) for key in ("comparison_power", "initial_power", "maximum_power")) != (17, 18, 19):
        raise ValueError("retry15 proposal powers must remain 17/18/19")
    if any(bool(proposal[key]) for key in ("beta_label_eligible", "beta_seed_eligible", "beta_warm_start_eligible", "beta_branch_hint_eligible")):
        raise ValueError("retry15 proposal beta must remain ineligible")
    banks = config["teacher_seed_banks"]
    expected = {
        "full": (32768, 20260897),
        "y_seam": (4096, 20260898),
        "z_seam": (4096, 20260899),
    }
    for key, values in expected.items():
        if (int(banks[key]["size"]), int(banks[key]["seed"])) != values:
            raise ValueError(f"retry15 {key} Teacher seed bank mismatch")
    if not config["claims"]["diagnostic_only"] or config["claims"]["claim_bearing_run_authorized"]:
        raise ValueError("retry15 discovery must remain diagnostic-only")
    if any(
        bool(config["claims"][key])
        for key in (
            "formal_authorized",
            "deployment_authorized",
            "full_workspace_authorized",
            "continuous_workspace_authorized",
            "tension_authorized",
        )
    ):
        raise ValueError("retry15 discovery cannot grant scientific claims")
    robot = SOURCE_ROOT / str(config["sources"]["robot_config"])
    if sha256_file(robot) != str(config["sources"]["robot_config_sha256"]):
        raise ValueError("retry15 robot config SHA mismatch")
    return config


def _binding_definition(binding_sha: str) -> Mapping[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{binding_sha}:spec/registry.yaml"],
        cwd=SOURCE_ROOT,
        text=True,
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
            raise RuntimeError(
                f"retry15 discovery binding mismatch for {key}: {definition.get(key)!r} != {value!r}"
            )
    path = output_root / "run_identity.json"
    if path.exists() and _read_json(path) != identity:
        raise RuntimeError("retry15 discovery output root identity mismatch")
    if not path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(path, identity)
    return identity


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    spec = config["upstream"][key]
    if "absolute_path" in spec:
        return Path(str(spec["absolute_path"]))
    return Path(str(config["upstream"]["retry14_root"])) / str(spec["path"])


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
        previous = (
            output_root
            / STAGE_DIRS[STAGE_ORDER[position - 1]]
            / "completion_manifest.json"
        )
        upstream.append(
            {"stage": STAGE_ORDER[position - 1], "sha256": sha256_file(previous)}
        )
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


def stage_inventory(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["inventory"]
    integrity, integrity_pass = _input_integrity(config)
    _write_parquet(integrity, stage / "upstream_verification.parquet")
    if not integrity_pass:
        raise RuntimeError("retry15 discovery upstream integrity failed")
    zero = _read_json(_upstream_path(config, "retry12_zero_anchor"))
    expected_x = float(config["omega600"]["zero_x_m"])
    zero_pass = bool(
        np.array_equal(np.asarray(zero["beta0_rad"], dtype=float), np.zeros(6))
        and np.allclose(np.asarray(zero["xyz0_m"], dtype=float), [expected_x, 0.0, 0.0], atol=1.0e-12, rtol=0.0)
    )
    environment = _environment(config)
    beta = sobol_beta_samples(np.asarray(environment.bounds), power=10, seed=20260901)
    symmetry_samples, symmetry = validate_symmetry_group(
        environment,
        beta,
        p99_max_mm=1.0e-6,
        individual_max_mm=1.0e-5,
    )
    symmetry_pass = bool(
        symmetry["group_composition_verified"]
        and symmetry["exact_zero_fixed"]
        and all(
            bool(value["authorized"])
            for key, value in symmetry.items()
            if isinstance(value, Mapping) and key not in {"group_composition_verified", "exact_zero_fixed"}
        )
    )
    _write_parquet(symmetry_samples, stage / "symmetry_revalidation.parquet")
    _write_json(stage / "symmetry_registry.json", symmetry)
    _write_json(
        stage / "omega600_contract.json",
        {
            "schema_version": 1,
            "x0_m": expected_x,
            "x_min_m": float(config["omega600"]["minimum_x_m"]),
            "axial_length_mm": float(config["omega600"]["axial_length_mm"]),
            "theoretical_reach_claimed": False,
            "empirical_denominator_required": True,
        },
    )
    return _seal_gate(
        output_root,
        config,
        "inventory",
        {
            "status": "complete" if zero_pass and symmetry_pass else "invalid",
            "upstream_integrity_pass": integrity_pass,
            "exact_zero_pass": zero_pass,
            "symmetry_revalidation_pass": symmetry_pass,
        },
    )


def _fk_chunks(environment: Any, beta: np.ndarray, *, chunk_rows: int) -> np.ndarray:
    rows = []
    for start in range(0, len(beta), int(chunk_rows)):
        rows.append(
            np.asarray(environment.fk(beta[start : start + int(chunk_rows)]), dtype=float).reshape(-1, 3)
        )
    result = np.vstack(rows)
    if result.shape != (len(beta), 3) or not np.isfinite(result).all():
        raise RuntimeError("retry15 proposal FK returned invalid values")
    return result


def _proposal_frames(beta: np.ndarray, xyz: np.ndarray, *, pool_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [stable_id("retry15_proposal", pool_id, index) for index in range(len(beta))]
    beta_frame = pd.DataFrame(beta, columns=BETA_COLUMNS)
    beta_frame.insert(0, "proposal_id", ids)
    beta_frame.insert(0, "proposal_pool_id", pool_id)
    beta_frame["label_eligible"] = False
    beta_frame["teacher_seed_eligible"] = False
    beta_frame["warm_start_eligible"] = False
    beta_frame["branch_hint_eligible"] = False
    xyz_frame = pd.DataFrame(xyz, columns=XYZ_COLUMNS)
    xyz_frame.insert(0, "proposal_id", ids)
    xyz_frame.insert(0, "proposal_pool_id", pool_id)
    xyz_frame["proposal_beta_columns_present"] = False
    return beta_frame, xyz_frame


def stage_workspace_probe(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["workspace_probe"]
    if not _gate(output_root, "inventory").get("symmetry_revalidation_pass", False):
        raise RuntimeError("retry15 discovery inventory did not authorize workspace probe")
    environment = _environment(config)
    proposal = config["proposal"]
    initial_power = 12 if smoke else int(proposal["initial_power"])
    comparison_power = initial_power - 1 if smoke else int(proposal["comparison_power"])
    maximum_power = initial_power if smoke else int(proposal["maximum_power"])
    policy = ProbePolicy(
        axial_step_mm=float(config["domain"]["axial_step_mm"]),
        radial_step_mm=float(config["domain"]["radial_step_mm"]),
        full_sector_count=int(config["domain"]["full_sector_count"]),
        radial_margin_mm=float(config["domain"]["radial_margin_mm"]),
        slope_limit_mm=float(config["domain"]["slope_limit_mm"]),
        allow_single_voxel_closing=bool(config["domain"]["isolated_single_voxel_closing"]),
    )
    beta_a = sobol_beta_samples(np.asarray(environment.bounds), power=initial_power, seed=int(proposal["pool_a_seed"]))
    beta_b = sobol_beta_samples(np.asarray(environment.bounds), power=initial_power, seed=int(proposal["pool_b_seed"]))
    xyz_a = _fk_chunks(environment, beta_a, chunk_rows=int(proposal["fk_chunk_rows"]))
    xyz_b = _fk_chunks(environment, beta_b, chunk_rows=int(proposal["fk_chunk_rows"]))
    _support_low, profile_low = workspace_probe(
        xyz_a[: 2**comparison_power], xyz_b[: 2**comparison_power], policy=policy
    )
    support, profile = workspace_probe(xyz_a, xyz_b, policy=policy)
    change = np.abs(
        profile["radius_mm"].to_numpy(float) - profile_low["radius_mm"].to_numpy(float)
    )
    change_p95 = percentile(change, 95)
    incomplete = bool((~profile.loc[profile["u_index"].gt(0), "all_sector_supported"]).any())
    expanded = bool(
        not smoke
        and initial_power < maximum_power
        and (
            change_p95 > float(proposal["profile_change_p95_expand_mm"])
            or incomplete
        )
    )
    final_power = initial_power
    if expanded:
        final_power = maximum_power
        beta_a = sobol_beta_samples(np.asarray(environment.bounds), power=final_power, seed=int(proposal["pool_a_seed"]))
        beta_b = sobol_beta_samples(np.asarray(environment.bounds), power=final_power, seed=int(proposal["pool_b_seed"]))
        xyz_a = _fk_chunks(environment, beta_a, chunk_rows=int(proposal["fk_chunk_rows"]))
        xyz_b = _fk_chunks(environment, beta_b, chunk_rows=int(proposal["fk_chunk_rows"]))
        previous_profile = profile
        support, profile = workspace_probe(xyz_a, xyz_b, policy=policy)
        change = np.abs(
            profile["radius_mm"].to_numpy(float)
            - previous_profile["radius_mm"].to_numpy(float)
        )
        change_p95 = percentile(change, 95)
        incomplete = bool((~profile.loc[profile["u_index"].gt(0), "all_sector_supported"]).any())
    beta_a_frame, xyz_a_frame = _proposal_frames(beta_a, xyz_a, pool_id="proposal_A")
    beta_b_frame, xyz_b_frame = _proposal_frames(beta_b, xyz_b, pool_id="proposal_B")
    _write_parquet(beta_a_frame, stage / "proposal_pool_A_beta_provenance.parquet")
    _write_parquet(beta_b_frame, stage / "proposal_pool_B_beta_provenance.parquet")
    _write_parquet(xyz_a_frame, stage / "proposal_pool_A_xyz.parquet")
    _write_parquet(xyz_b_frame, stage / "proposal_pool_B_xyz.parquet")
    _write_parquet(support, stage / "empirical_voxel_support.parquet")
    _write_parquet(support[support["robust_supported"]].copy(), stage / "empirical_outer_denominator.parquet")
    _write_parquet(profile, stage / "raw_radius_profile.parquet")
    _write_json(
        stage / "proposal_only_provenance.json",
        {
            "proposal_beta_target_generation": True,
            "proposal_beta_label_eligible": False,
            "proposal_beta_teacher_seed_eligible": False,
            "proposal_beta_branch_hint_eligible": False,
            "proposal_beta_warm_start_eligible": False,
            "pool_a_seed": int(proposal["pool_a_seed"]),
            "pool_b_seed": int(proposal["pool_b_seed"]),
            "final_power": int(final_power),
        },
    )
    stable = bool(
        change_p95 <= float(proposal["profile_change_p95_expand_mm"])
        and not incomplete
    )
    return _seal_gate(
        output_root,
        config,
        "workspace_probe",
        {
            "status": "diagnostic_complete" if stable else "diagnostic_unstable",
            "proposal_pool_rows_each": int(len(beta_a)),
            "final_power": int(final_power),
            "expanded_to_maximum_power": expanded,
            "profile_change_p95_mm": change_p95,
            "all_nonzero_axial_bins_sector_supported": not incomplete,
            "empirical_outer_voxel_count": int(support["robust_supported"].sum()),
            "raw_profile_maximum_radius_mm": float(profile["radius_mm"].max()),
            "proposal_stability_pass": stable,
        },
    )


def _outer_volume_mm3(support: pd.DataFrame, config: Mapping[str, Any]) -> float:
    robust = support[support["robust_supported"]]
    dr = float(config["domain"]["radial_step_mm"])
    du = float(config["domain"]["axial_step_mm"])
    sectors = int(config["domain"]["full_sector_count"])
    inner = robust["rho_index"].to_numpy(float) * dr
    outer = inner + dr
    return float(np.sum((np.pi / sectors) * (np.square(outer) - np.square(inner)) * du))


def stage_domain_selection(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    del smoke
    stage = output_root / STAGE_DIRS["domain_selection"]
    probe = output_root / STAGE_DIRS["workspace_probe"]
    raw = pd.read_parquet(probe / "raw_radius_profile.parquet")
    a = pd.read_parquet(probe / "proposal_pool_A_xyz.parquet")
    b = pd.read_parquet(probe / "proposal_pool_B_xyz.parquet")
    proposals = pd.concat([a.loc[:, XYZ_COLUMNS], b.loc[:, XYZ_COLUMNS]], ignore_index=True)
    selected = None
    selected_budget = None
    errors: dict[str, str] = {}
    for budget in (
        int(config["domain"]["pilot_target_budget"]),
        int(config["domain"]["expanded_target_budget"]),
    ):
        try:
            selected = select_budget_profile(
                raw,
                proposals.to_numpy(float),
                target_budget=budget,
                axial_step_mm=float(config["domain"]["axial_step_mm"]),
                ceiling_step_mm=float(config["domain"]["ceiling_step_mm"]),
                slope_limit_mm=float(config["domain"]["slope_limit_mm"]),
            )
        except ValueError as exc:
            errors[str(budget)] = str(exc)
            continue
        selected_budget = budget
        break
    if selected is None or selected_budget is None:
        _write_json(stage / "profile_selection_errors.json", errors)
        _write_parquet(pd.DataFrame(), stage / "selected_radius_profile.parquet")
        _write_parquet(pd.DataFrame(), stage / "selected_target_registry.parquet")
        _write_parquet(pd.DataFrame(), stage / "registered_target_edges.parquet")
        _write_parquet(pd.DataFrame(), stage / "registered_circle_waypoints.parquet")
        return _seal_gate(
            output_root,
            config,
            "domain_selection",
            {
                "status": "objective_infeasible",
                "domain_selection_pass": False,
                "profile_selection_errors": errors,
            },
        )
    profile, targets, edges, circles, summary = selected
    _write_parquet(profile, stage / "selected_radius_profile.parquet")
    _write_parquet(targets, stage / "selected_target_registry.parquet")
    _write_parquet(edges, stage / "registered_target_edges.parquet")
    _write_parquet(circles, stage / "registered_circle_waypoints.parquet")
    support = pd.read_parquet(probe / "empirical_voxel_support.parquet")
    outer_volume = _outer_volume_mm3(support, config)
    inner_volume = 4.0 * quotient_volume_mm3(
        profile, axial_step_mm=float(config["domain"]["axial_step_mm"])
    )
    _write_json(
        stage / "domain_fraction_report.json",
        {
            "inner_full_volume_mm3": inner_volume,
            "empirical_outer_voxel_volume_mm3": outer_volume,
            "inner_to_empirical_outer_volume_fraction": None if outer_volume <= 0 else inner_volume / outer_volume,
            "empirical_outer_is_not_exact_reach": True,
        },
    )
    _write_json(stage / "selected_domain_summary.json", {**dict(summary), "selected_target_budget": selected_budget})
    pass_gate = bool(
        summary["budget_feasible"]
        and summary["registered_circle_count"] == int(config["domain"]["required_circle_count"])
        and summary["large_circle_count"] >= 3
        and summary["complete_600mm_axis_core"]
    )
    return _seal_gate(
        output_root,
        config,
        "domain_selection",
        {
            "status": "diagnostic_domain_frozen" if pass_gate else "objective_infeasible",
            "domain_selection_pass": pass_gate,
            "selected_target_budget": selected_budget,
            **dict(summary),
            "inner_full_volume_mm3": inner_volume,
            "empirical_outer_voxel_volume_mm3": outer_volume,
            "inner_to_empirical_outer_volume_fraction": None if outer_volume <= 0 else inner_volume / outer_volume,
        },
    )


def stage_seed_banks(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["seed_banks"]
    if not _gate(output_root, "domain_selection").get("domain_selection_pass", False):
        _write_json(stage / "seed_bank_manifest.json", {"status": "not_authorized"})
        return _seal_gate(
            output_root,
            config,
            "seed_banks",
            {"status": "not_authorized", "seed_banks_frozen": False},
        )
    environment = _environment(config)
    specs = config["teacher_seed_banks"]
    free = {"full": None, "y_seam": (1, 3, 5), "z_seam": (0, 2, 4)}
    manifest = {"schema_version": 1, "proposal_identity_shared": False, "banks": {}}
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


def _preflight_targets(targets: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    mandatory = targets[targets["mandatory"].astype(bool)].copy()
    if len(mandatory) > int(limit):
        raise RuntimeError("mandatory retry15 targets exceed the registered preflight budget")
    remaining = targets[~targets["target_id"].isin(mandatory["target_id"])].copy()
    remaining["selection_hash"] = remaining["target_id"].astype(str).map(
        lambda value: hashlib.sha256(f"retry15-preflight:{value}".encode()).hexdigest()
    )
    selected = remaining.sort_values(
        ["selection_hash", "u_index", "rho_mm", "target_id"], kind="stable"
    ).head(max(0, int(limit) - len(mandatory)))
    result = pd.concat([mandatory, selected.drop(columns=["selection_hash"])], ignore_index=True, sort=False)
    return result.sort_values(["u_index", "rho_mm", "target_id"], kind="stable").reset_index(drop=True)


def _run_candidate_workers(
    config: Mapping[str, Any],
    targets: pd.DataFrame,
    *,
    output_root: Path,
    work_root: Path,
    seed_budget: int,
    smoke: bool,
) -> pd.DataFrame:
    workers = min(2 if smoke else int(config["runtime"]["maximum_concurrent_workers"]), len(targets))
    shards = np.array_split(np.arange(len(targets)), workers)
    seed_root = output_root / STAGE_DIRS["seed_banks"]
    processes: list[tuple[subprocess.Popen[str], Path]] = []
    env = _thread_limited_environment()
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
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ),
                output_path,
            )
        )
    outputs = []
    for process, output_path in processes:
        stdout, stderr = process.communicate()
        if process.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"retry15 candidate worker failed ({process.returncode}): {stderr[-4000:]} {stdout[-1000:]}"
            )
        outputs.append(pd.read_parquet(output_path))
    return pd.concat(outputs, ignore_index=True, sort=False)


def _accepted_lcc_fraction(labels: pd.DataFrame, edges: pd.DataFrame) -> float:
    ids = sorted(set(labels["target_id"].astype(str)))
    if not ids:
        return 0.0
    index = {target_id: ordinal for ordinal, target_id in enumerate(ids)}
    rows, cols = [], []
    for edge in edges.to_dict("records"):
        left, right = str(edge["left_target_id"]), str(edge["right_target_id"])
        if left in index and right in index:
            rows.extend([index[left], index[right]])
            cols.extend([index[right], index[left]])
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(ids), len(ids)))
    _count, components = connected_components(graph, directed=False, return_labels=True)
    sizes = np.bincount(components)
    return float(np.max(sizes) / len(ids))


def _teacher_metrics(
    labels: pd.DataFrame,
    targets: pd.DataFrame,
    edges: pd.DataFrame,
    circles: pd.DataFrame,
    profile: pd.DataFrame,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    accepted = set(labels["target_id"].astype(str))
    acceptance = float(len(accepted) / len(targets)) if len(targets) else 0.0
    radius = dict(zip(profile["u_index"].astype(int), profile["radius_mm"].astype(float), strict=True))
    core_mask = targets.apply(
        lambda row: float(row["rho_mm"]) <= max(0.0, radius.get(int(row["u_index"]), 0.0) - float(config["preflight_gates"]["core_shrink_mm"])) + 1.0e-9,
        axis=1,
    )
    core = targets[core_mask]
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
        weighted.append(float(weighted_beta_rms_deg(beta_by_id[left], beta_by_id[right], (4, 4, 2, 2, 1, 1))))
        raw.append(float(np.max(np.abs(np.rad2deg(delta)))))
    edge_p95 = percentile(np.asarray(weighted), 95) if weighted else math.inf
    raw_rate = float(np.mean(np.asarray(raw) > 7.0)) if raw else 1.0
    complete_circles = 0
    if len(circles):
        for _circle_id, frame in circles.groupby("circle_id", sort=True):
            if set(frame["target_id"].astype(str)) <= accepted:
                complete_circles += 1
    lcc = _accepted_lcc_fraction(labels, edges)
    target_roles = targets.set_index("target_id")["target_role"].astype(str).to_dict()
    axis_labels = labels[
        labels["target_id"].astype(str).map(target_roles).eq("axis_core")
    ]
    axis_invariant = 0
    for row in axis_labels.to_dict("records"):
        beta = np.asarray([row[name] for name in BETA_COLUMNS], dtype=float)
        if all(
            np.max(np.abs(transform_beta(beta, element) - beta)) <= 1.0e-10
            for element in ("identity", "mirror_y", "mirror_z", "mirror_yz")
        ):
            axis_invariant += 1
    gates = config["preflight_gates"]
    passed = bool(
        core_acceptance >= float(gates["core_acceptance_minimum"])
        and acceptance >= float(gates["expanded_acceptance_minimum"])
        and lcc >= float(gates["accepted_lcc_minimum"])
        and edge_p95 <= float(gates["edge_weighted_p95_maximum_deg"])
        and raw_rate <= float(gates["edge_raw_gt7_rate_maximum"])
        and complete_circles >= int(gates["required_complete_circle_count"])
    )
    return {
        "accepted_target_count": len(accepted),
        "target_count": len(targets),
        "expanded_acceptance": acceptance,
        "core_acceptance": core_acceptance,
        "accepted_lcc_fraction": lcc,
        "edge_weighted_p95_deg": edge_p95,
        "edge_raw_gt7_rate": raw_rate,
        "complete_circle_count": complete_circles,
        "axis_label_count": int(len(axis_labels)),
        "axis_stabilizer_invariant_count": int(axis_invariant),
        "axis_stabilizer_noninvariant_count": int(len(axis_labels) - axis_invariant),
        "axis_gauge_fixed_not_g4_equivariant": bool(len(axis_labels) > axis_invariant),
        "preflight_green": passed,
    }


def stage_teacher_preflight(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["teacher_preflight"]
    if not _gate(output_root, "seed_banks").get("seed_banks_frozen", False):
        _write_json(stage / "preflight_comparison.json", {"status": "not_authorized"})
        return _seal_gate(
            output_root,
            config,
            "teacher_preflight",
            {"status": "not_authorized", "teacher_preflight_complete": False},
        )
    domain_root = output_root / STAGE_DIRS["domain_selection"]
    probe_root = output_root / STAGE_DIRS["workspace_probe"]
    selected = pd.read_parquet(domain_root / "selected_radius_profile.parquet")
    selected_ceiling = float(selected["ceiling_mm"].iloc[0])
    raw = pd.read_parquet(probe_root / "raw_radius_profile.parquet")
    a = pd.read_parquet(probe_root / "proposal_pool_A_xyz.parquet")
    b = pd.read_parquet(probe_root / "proposal_pool_B_xyz.parquet")
    proposals = pd.concat([a.loc[:, XYZ_COLUMNS], b.loc[:, XYZ_COLUMNS]], ignore_index=True).to_numpy(float)
    target_budget = int(_gate(output_root, "domain_selection")["selected_target_budget"])
    limit = 32 if smoke else int(config["candidate_solver"]["preflight_target_count_per_profile"])
    seed_budget = 8 if smoke else int(config["candidate_solver"]["default_preflight_seed_budget"])
    comparison = []
    selected_outputs: list[tuple[float, Mapping[str, Any], pd.DataFrame]] = []
    for profile_index, shrink in enumerate(map(float, config["domain"]["shrink_profiles_mm"])):
        _progress(output_root, "teacher_preflight", completed=profile_index, total=len(config["domain"]["shrink_profiles_mm"]), message=f"shrink={shrink:g}")
        profile = capped_profile(
            raw,
            ceiling_mm=selected_ceiling,
            shrink_mm=shrink,
            slope_limit_mm=float(config["domain"]["slope_limit_mm"]),
        )
        targets, explicit_edges, circles, target_summary = build_target_registry(
            profile,
            proposals,
            target_budget=target_budget,
            axial_step_mm=float(config["domain"]["axial_step_mm"]),
            maximum_circle_step_mm=float(config["domain"]["maximum_circle_step_mm"]),
        )
        profile_root = stage / f"profile_{profile_index:02d}_delta_{int(shrink):02d}"
        _write_parquet(profile, profile_root / "radius_profile.parquet")
        _write_parquet(targets, profile_root / "target_registry.parquet")
        _write_parquet(circles, profile_root / "circle_waypoints.parquet")
        if not target_summary["budget_feasible"] or target_summary["registered_circle_count"] != 9:
            metrics = {
                "shrink_mm": shrink,
                "profile_target_budget_feasible": bool(target_summary["budget_feasible"]),
                "registered_circle_count": int(target_summary["registered_circle_count"]),
                "preflight_green": False,
                "reason": "profile_does_not_preserve_registered_atomic_objectives",
            }
            _write_json(profile_root / "gate.json", metrics)
            comparison.append(metrics)
            continue
        sampled = _preflight_targets(targets, limit=min(limit, len(targets)))
        _write_parquet(sampled, profile_root / "preflight_targets.parquet")
        sampled_ids = set(sampled["target_id"].astype(str))
        explicit = explicit_edges[
            explicit_edges["left_target_id"].astype(str).isin(sampled_ids)
            & explicit_edges["right_target_id"].astype(str).isin(sampled_ids)
        ].copy()
        graph_k = min(map(int, config["graph_teacher"]["k_values"]))
        graph = mutual_knn_edges(
            sampled,
            explicit,
            k=graph_k,
            maximum_distance_mm=float(config["graph_teacher"]["maximum_edge_spacing_factor"])
            * float(target_summary["target_spacing_mm"]),
        )
        _write_parquet(graph, profile_root / "target_graph_edges.parquet")
        candidates = _run_candidate_workers(
            config,
            sampled,
            output_root=output_root,
            work_root=profile_root / "_work",
            seed_budget=seed_budget,
            smoke=smoke,
        )
        _write_parquet(candidates, profile_root / "candidate_bank.parquet")
        legal = legal_candidate_clusters(
            candidates,
            residual_maximum_mm=float(config["candidate_solver"]["residual_maximum_mm"]),
            cluster_threshold_deg=float(config["candidate_solver"]["cluster_threshold_deg"]),
            weights=config["candidate_solver"]["beta_weights"],
        )
        _write_parquet(legal, profile_root / "legal_candidate_clusters.parquet")
        teachers: list[tuple[str, pd.DataFrame, Mapping[str, Any]]] = []
        t1 = select_t1(candidates, weights=config["candidate_solver"]["beta_weights"]).assign(
            teacher="T1", graph_k=0, pairwise_lambda=0.0
        )
        teachers.append(("T1", t1, _teacher_metrics(t1, sampled, graph, circles, profile, config)))
        for k in map(int, config["graph_teacher"]["k_values"]):
            graph_k_frame = mutual_knn_edges(
                sampled,
                explicit,
                k=k,
                maximum_distance_mm=float(config["graph_teacher"]["maximum_edge_spacing_factor"])
                * float(target_summary["target_spacing_mm"]),
            )
            for pairwise_lambda in map(float, config["graph_teacher"]["pairwise_lambdas"]):
                if pairwise_lambda == 0.0:
                    continue
                labels = select_t2(
                    candidates,
                    graph_k_frame,
                    pairwise_lambda=pairwise_lambda,
                    tau_deg=float(config["graph_teacher"]["robust_pairwise_cap_deg"]),
                    weights=config["candidate_solver"]["beta_weights"],
                    maximum_sweeps=int(config["graph_teacher"]["maximum_icm_sweeps"]),
                ).assign(graph_k=k)
                name = f"T2_k{k}_lambda{pairwise_lambda:g}"
                teachers.append((name, labels, _teacher_metrics(labels, sampled, graph_k_frame, circles, profile, config)))
        teacher_name, labels, metrics = max(
            teachers,
            key=lambda item: (
                bool(item[2]["preflight_green"]),
                float(item[2]["expanded_acceptance"]),
                float(item[2]["accepted_lcc_fraction"]),
                -float(item[2]["edge_raw_gt7_rate"]),
                -float(item[2]["edge_weighted_p95_deg"]),
                item[0],
            ),
        )
        _write_parquet(labels, profile_root / "selected_teacher_labels.parquet")
        teacher_table = pd.DataFrame(
            [{"teacher_name": name, **dict(values)} for name, _labels, values in teachers]
        )
        _write_parquet(teacher_table, profile_root / "teacher_comparison.parquet")
        profile_metrics = {
            "shrink_mm": shrink,
            "selected_teacher": teacher_name,
            "profile_target_count": len(targets),
            "preflight_target_count": len(sampled),
            "candidate_attempt_count": len(candidates),
            **dict(metrics),
        }
        _write_json(profile_root / "gate.json", profile_metrics)
        comparison.append(profile_metrics)
        selected_outputs.append((shrink, profile_metrics, labels))
    if selected_outputs:
        passing = [item for item in selected_outputs if item[1]["preflight_green"]]
        selected_result = min(passing, key=lambda item: item[0]) if passing else max(
            selected_outputs,
            key=lambda item: (
                float(item[1]["expanded_acceptance"]),
                float(item[1]["accepted_lcc_fraction"]),
                -float(item[1]["edge_weighted_p95_deg"]),
                -item[0],
            ),
        )
        selected_shrink, selected_metrics, selected_labels = selected_result
        _write_parquet(selected_labels, stage / "selected_preflight_teacher_labels.parquet")
        _write_json(stage / "selected_preflight_profile.json", {"selected_shrink_mm": selected_shrink, **dict(selected_metrics)})
    else:
        selected_shrink, selected_metrics = math.nan, {"preflight_green": False}
        _write_parquet(pd.DataFrame(), stage / "selected_preflight_teacher_labels.parquet")
        _write_json(stage / "selected_preflight_profile.json", {"status": "no_executable_profile"})
    _write_parquet(pd.DataFrame(comparison), stage / "preflight_comparison.parquet")
    green = bool(selected_metrics.get("preflight_green", False))
    return _seal_gate(
        output_root,
        config,
        "teacher_preflight",
        {
            "status": "diagnostic_green" if green else "diagnostic_yellow_or_red",
            "teacher_preflight_complete": True,
            "selected_profile_shrink_mm": selected_shrink,
            "selected_profile_green": green,
            "profile_count": len(comparison),
            "preflight_target_limit_per_profile": limit,
            "preflight_seed_budget": seed_budget,
            "selected_metrics": dict(selected_metrics),
        },
    )


def _artifact_manifest(output_root: Path) -> Mapping[str, Any]:
    artifacts = []
    for stage_name in STAGE_ORDER[:-1]:
        stage = output_root / STAGE_DIRS[stage_name]
        for path in sorted(stage.rglob("*")):
            if path.is_file() and "_work" not in path.parts:
                artifacts.append(
                    {
                        "path": str(path.relative_to(output_root)),
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


def stage_summary(
    config: Mapping[str, Any], output_root: Path, *, smoke: bool
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    inventory = _gate(output_root, "inventory")
    probe = _gate(output_root, "workspace_probe")
    domain = _gate(output_root, "domain_selection")
    seeds = _gate(output_root, "seed_banks")
    preflight = _gate(output_root, "teacher_preflight")
    operational = all(_stage_is_complete(output_root, name) for name in STAGE_ORDER[:-1])
    gate = {
        "status": "complete" if operational else "incomplete",
        "operational_completion": operational,
        "artifact_completeness": operational,
        "diagnostic_smoke": bool(smoke),
        "scientific_gate": "not_evaluated_discovery_only",
        "downstream_authorization": False,
        "inventory": {
            "exact_zero_pass": inventory.get("exact_zero_pass", False),
            "symmetry_revalidation_pass": inventory.get("symmetry_revalidation_pass", False),
        },
        "empirical_outer_denominator": {
            "voxel_count": probe.get("empirical_outer_voxel_count"),
            "proposal_stability_pass": probe.get("proposal_stability_pass", False),
            "profile_change_p95_mm": probe.get("profile_change_p95_mm"),
        },
        "selected_domain": {
            "status": domain.get("status"),
            "target_count": domain.get("target_count"),
            "target_budget": domain.get("selected_target_budget"),
            "selected_ceiling_mm": domain.get("selected_ceiling_mm"),
            "registered_circle_count": domain.get("registered_circle_count"),
            "large_circle_count": domain.get("large_circle_count"),
            "inner_to_empirical_outer_volume_fraction": domain.get("inner_to_empirical_outer_volume_fraction"),
        },
        "teacher_seed_banks_frozen": seeds.get("seed_banks_frozen", False),
        "teacher_preflight": {
            "status": preflight.get("status"),
            "selected_profile_shrink_mm": preflight.get("selected_profile_shrink_mm"),
            "selected_profile_green": preflight.get("selected_profile_green", False),
            "selected_metrics": preflight.get("selected_metrics"),
        },
        "proposal_beta_used_as_label_or_hint": False,
        "formal_authorization": False,
        "deployment_authorization": False,
        "full_workspace_authorization": False,
        "continuous_workspace_authorization": False,
        "tension_executed": False,
    }
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "retry15_discovery_summary.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>retry15 discovery</title></head>"
        f"<body><h1>retry15 Omega600 diagnostic discovery</h1><pre>{json.dumps(gate, ensure_ascii=False, indent=2)}</pre></body></html>",
        encoding="utf-8",
    )
    sealed = _seal_gate(output_root, config, "summary", gate)
    _write_json(stage / "artifact_manifest.json", _artifact_manifest(output_root))
    _seal_stage(output_root, "summary")
    _write_json(
        output_root / "progress.json",
        {
            "status": "complete",
            "phase": "summary",
            "completed": len(STAGE_ORDER),
            "total": len(STAGE_ORDER),
            "message": "retry15 diagnostic discovery completed",
            "observed_at_unix": time.time(),
        },
    )
    return sealed


STAGE_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "inventory": stage_inventory,
    "workspace_probe": stage_workspace_probe,
    "domain_selection": stage_domain_selection,
    "seed_banks": stage_seed_banks,
    "teacher_preflight": stage_teacher_preflight,
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
            raise ValueError(f"unknown retry15 discovery stage {validate_stage}")
        if not _stage_is_complete(output, validate_stage):
            raise RuntimeError(f"retry15 discovery stage is incomplete or mutated: {validate_stage}")
        return _gate(output, validate_stage)
    if stage_name is not None:
        if stage_name not in STAGE_RUNNERS:
            raise ValueError(f"unknown retry15 discovery stage {stage_name}")
        position = STAGE_ORDER.index(stage_name)
        if position and not _stage_is_complete(output, STAGE_ORDER[position - 1]):
            raise RuntimeError("retry15 discovery previous stage is incomplete")
        if _stage_is_complete(output, stage_name):
            return _gate(output, stage_name)
        return STAGE_RUNNERS[stage_name](config, output, smoke=smoke)
    result: dict[str, Any] = {}
    for index, name in enumerate(STAGE_ORDER):
        _progress(output, name, completed=index, total=len(STAGE_ORDER))
        if _stage_is_complete(output, name):
            result = _gate(output, name)
        else:
            result = STAGE_RUNNERS[name](config, output, smoke=smoke)
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
