#!/usr/bin/env python3
"""Run BACRA V14 Omega200 fixed-budget workspace-atlas experiments.

Smoke and Pilot are scientific discovery runs.  Formal is deliberately
fail-closed on the independently persisted Pilot reach, branch-saturation and
representation gates; selecting ``--preset formal`` cannot bypass them.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
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
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(SOURCE_ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "scripts" / "analysis"))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from quasi_exp.teacher.capability_map import batch_fk
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.workspace_candidate_bank import (
    CandidatePolicy,
    CandidateQuality,
    CandidateSearchMode,
    solve_candidate_bank,
    stable_cluster_representatives,
)
from quasi_exp.teacher.canonical import beta_rms_deg
from quasi_exp.teacher.atlas_audit import AtlasAuditPolicy
from quasi_exp.teacher.canonical_atlas import AtlasPolicy
from quasi_exp.teacher.workspace_atlas import (
    BranchAuditObservation,
    RepresentationMode,
    WorkspaceAtlasBuilder,
    WorkspaceAtlasInput,
    WorkspaceAtlasPolicy,
)
from quasi_exp.teacher.workspace_atlas_integration import (
    WorkspaceAtlasIntegrationPolicy,
    build_workspace_atlas_integration,
)
from quasi_exp.teacher.workspace_experiment import (
    ReachSamplingRoundSpec,
    cell_centers,
    frontier_candidates,
    generate_independent_reach_samples,
    tip_focused_beta,
)
from quasi_exp.teacher.workspace_reach import (
    CellKey,
    FrontierProbeEvidence,
    ReachProxyBuilder,
    ReachProxyConfig,
    WorkspaceGridSpec,
)
from quasi_exp.teacher.workspace_registry import (
    CapabilityColumns,
    WorkspaceRegistryBuilder,
    WorkspaceRegistryPolicy,
)
from quasi_exp.teacher.workspace_protocol import (
    WorkspaceAtlasFramePolicy,
    correct_static_targets_from_primary_sections,
    prepare_workspace_atlas_frames,
    supervision_records_frame,
    supervision_records_from_atlas_frames,
)
from quasi_exp.teacher.workspace_dataset import (
    BudgetFeasibilityStatus,
    FixedBudgetDemand,
    MacroblockSplitPolicy,
    SplitRole,
    SupervisionBudgetPolicy,
    SupervisionKind,
    SupervisionMaterializer,
    SupervisionPriority,
    evaluate_fixed_budget_feasibility,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.workspace_inverse import (
    InverseQuery,
    WorkspaceRegistry,
    refine_inverse_with_bounded_dls,
)
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentLossWeights,
    WorkspaceStudentTrainingConfig,
    build_workspace_student_inverse,
    load_workspace_student_models,
    records_to_workspace_student_frame,
    save_workspace_student_models,
    student_feature_columns,
    train_workspace_student,
)
from quasi_exp.teacher.workspace_trajectories import build_workspace_trajectory_suite
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("x_m", "y_m", "z_m")
STAGE_DIRS = {
    "protocol": "00_protocol",
    "workspace_proxy": "01_workspace_proxy",
    "domain_registry": "02_domain_registry",
    "candidate_bank": "03_candidate_bank",
    "workspace_atlas": "04_workspace_atlas",
    "budget_allocation": "05_budget_allocation",
    "dataset": "06_dataset",
    "student": "07_student",
    "spatial_evaluation": "08_spatial_evaluation",
    "trajectory_evaluation": "10_trajectory_evaluation",
    "summary": "11_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)
V14_SCRIPT_SOURCES = (
    "scripts/analysis/run_bacra_v14_omega200_atlas.py",
    "scripts/analysis/train_bacra_v14_student.py",
    "scripts/analysis/evaluate_bacra_v14_workspace.py",
    # The V14 runner imports the V10 environment adapter directly.  Library
    # code is inventoried as a complete src/quasi_exp tree below.
    "scripts/analysis/run_trajectory_canonical_teacher_v10.py",
)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            # Lists are protocol objects (not append operations).  A preset
            # therefore replaces reach rounds rather than silently extending
            # the base seed registry.
            result[key] = copy.deepcopy(value)
    return result


def load_protocol_config(path: str | Path, preset: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("BACRA V14 config root must be a mapping")
    presets = raw.get("presets", {})
    if preset not in presets:
        raise KeyError(f"unknown BACRA V14 preset {preset!r}")
    base = {key: value for key, value in raw.items() if key != "presets"}
    merged = _deep_merge(base, presets[preset])
    merged["preset"] = str(preset)
    merged["config_path"] = str(config_path)
    _validate_config(merged)
    return merged


def _validate_config(config: Mapping[str, Any]) -> None:
    domain = config["domain"]
    if not float(domain["x_min_m"]) < float(domain["x_max_m"]):
        raise ValueError("Omega200 x slab must be ordered")
    levels = tuple(map(int, domain["levels_mm"]))
    if levels != (20, 10, 5):
        raise ValueError("BACRA V14 protocol requires registered 20/10/5 mm levels")
    rounds = tuple(config["reach"]["rounds"])
    seed_a = [int(row["seed_a"]) for row in rounds]
    seed_b = [int(row["seed_b"]) for row in rounds]
    if (
        not rounds
        or len(set(seed_a)) != len(seed_a)
        or len(set(seed_b)) != len(seed_b)
        or set(seed_a) & set(seed_b)
    ):
        raise ValueError("reach A/B scramble seeds must be non-empty and disjoint")
    budget = config["budget"]
    total = int(budget["total_rows"])
    parts = (
        int(budget["base_rows_max"]),
        int(budget["refine_rows_max"]),
        int(budget["router_boundary_rows_max"]),
        int(budget["retention_rows_max"]),
        int(budget["active_rows"]),
    )
    if sum(parts) != total:
        raise ValueError("fixed-budget allocation categories must sum exactly to total_rows")
    if int(budget["hard_max_rows"]) < total:
        raise ValueError("hard_max_rows must not be below total_rows")


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if ".worktrees" in resolved.parts:
        index = resolved.parts.index(".worktrees")
        return Path(*resolved.parts[:index])
    return resolved


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _source_paths(config: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    sources = config["sources"]
    return {
        "robot_config": SOURCE_ROOT / str(config["robot_config"]),
        "capability_pool": _resolve_project_path(project_root, sources["capability_pool"]),
        "capability_beta": _resolve_project_path(project_root, sources["capability_beta"]),
        "v13_retention": _resolve_project_path(project_root, sources["v13_retention"]),
        "historical_final8_teacher": _resolve_project_path(
            project_root, sources["historical_final8_teacher"]
        ),
        "historical_final8_catalog": _resolve_project_path(
            project_root, sources["historical_final8_catalog"]
        ),
        "v13_ellipse_points": _resolve_project_path(
            project_root, sources["v13_ellipse_points"]
        ),
        "v13_path_targets": _resolve_project_path(
            project_root, sources["v13_path_targets"]
        ),
        "reviewed_plan": _resolve_project_path(project_root, sources["reviewed_plan"]),
    }


def implementation_source_inventory(
    source_root: Path = SOURCE_ROOT,
) -> dict[str, dict[str, Any]]:
    """Hash the exact local Python implementation available to a V14 run.

    Git SHA alone is insufficient for Smoke/Pilot development runs because the
    V14 files may still be untracked or modified.  The complete reusable
    ``src/quasi_exp`` tree plus every V14-facing script is therefore captured
    byte-for-byte.  A clean Formal fixed point remains an additional gate; this
    inventory does not weaken it.
    """

    root = Path(source_root).resolve()
    library_root = root / "src" / "quasi_exp"
    if not library_root.is_dir():
        raise FileNotFoundError(f"missing quasi_exp source tree: {library_root}")
    paths = set(library_root.rglob("*.py"))
    paths.update(root / relative for relative in V14_SCRIPT_SOURCES)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing implementation sources: {missing}")
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(paths)
    }


def implementation_source_root_sha256(
    inventory: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return an order-independent digest over relative path, hash and size."""

    canonical = [
        {
            "path": str(path),
            "sha256": str(payload["sha256"]),
            "size_bytes": int(payload["size_bytes"]),
        }
        for path, payload in sorted(inventory.items())
    ]
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def check_formal_prerequisites(
    config: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    requirements = config.get("formal_requirements", {})
    pilot_path = _resolve_project_path(project_root, requirements["pilot_summary"])
    reasons: list[str] = []
    exists = pilot_path.is_file()
    payload: dict[str, Any] = {}
    if not exists:
        reasons.append("missing_pilot_gate")
    else:
        try:
            payload = _read_json(pilot_path)
        except Exception as error:
            reasons.append(f"invalid_pilot_gate:{type(error).__name__}")
    checks = {
        "pilot_gate_exists": exists,
        "pilot_gate_pass": bool(payload.get("gate_pass", False)),
        "reach_convergence_gate": bool(payload.get("reach_convergence_gate", False)),
        "branch_saturation_gate": bool(payload.get("branch_saturation_gate", False)),
        "representation_gate_pass": bool(payload.get("representation_gate_pass", False)),
    }
    if exists and not checks["pilot_gate_pass"]:
        reasons.append("pilot_gate_failed")
    if requirements.get("require_reach_convergence", True) and not checks["reach_convergence_gate"]:
        reasons.append("reach_convergence_not_proven")
    if requirements.get("require_branch_saturation", True) and not checks["branch_saturation_gate"]:
        reasons.append("branch_discovery_not_saturated")
    if requirements.get("require_representation_gate", True) and not checks["representation_gate_pass"]:
        reasons.append("deployable_representation_not_authorized")
    return {
        **checks,
        "pilot_gate_path": str(pilot_path),
        "pilot_gate_sha256": sha256_file(pilot_path) if exists else None,
        "reasons": sorted(set(reasons)),
        "gate_pass": not reasons,
    }


def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _gate(path: Path, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    payload = {
        "gate_pass": bool(checks and all(bool(value) for value in checks.values())),
        "checks": {str(key): bool(value) for key, value in checks.items()},
        **evidence,
    }
    atomic_write_json(path, _strict_json(payload))
    return payload


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _require_stage(output_root: Path, stage_name: str) -> dict[str, Any]:
    gate_path = output_root / STAGE_DIRS[stage_name] / "gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"missing prerequisite gate: {gate_path}")
    payload = _read_json(gate_path)
    if not bool(payload.get("gate_pass", False)):
        raise RuntimeError(f"prerequisite stage failed: {stage_name}")
    return payload


def stage_protocol(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["protocol"]
    stage.mkdir(parents=True, exist_ok=False)
    paths = _source_paths(config, project_root)
    expected = {str(key): str(value) for key, value in config["sources"]["expected_sha256"].items()}
    actual = {
        key: sha256_file(path) if path.is_file() else "missing"
        for key, path in paths.items()
    }
    implementation_sources = implementation_source_inventory(SOURCE_ROOT)
    implementation_root = implementation_source_root_sha256(
        implementation_sources
    )
    git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    formal = (
        check_formal_prerequisites(config, project_root=project_root)
        if config["preset"] == "formal"
        else {"gate_pass": True, "reasons": [], "not_applicable": True}
    )
    frozen_config = {key: value for key, value in config.items() if key != "config_path"}
    atomic_write_json(stage / "frozen_config.json", _strict_json(frozen_config))
    manifest = {
        "source_manifest_schema_version": 2,
        "protocol_id": config["protocol_id"],
        "preset": config["preset"],
        "source_root": str(SOURCE_ROOT),
        "project_data_root": str(project_root),
        "source_git_sha": git_sha,
        "working_tree_clean": not dirty,
        "working_tree_status": dirty,
        "config_path": str(config["config_path"]),
        "config_sha256": sha256_file(config["config_path"]),
        "sources": {
            key: {
                "path": str(path),
                "sha256": actual[key],
                "expected_sha256": expected[key],
            }
            for key, path in paths.items()
        },
        "implementation_sources": implementation_sources,
        "implementation_source_count": len(implementation_sources),
        "implementation_source_root_sha256": implementation_root,
        "runtime": runtime_fingerprint(),
        "formal_prerequisites": formal,
    }
    atomic_write_json(stage / "source_manifest.json", _strict_json(manifest))
    checks = {
        "all_sources_exist": all(path.is_file() for path in paths.values()),
        "all_source_hashes_match": actual == expected,
        "formal_prerequisites": bool(formal["gate_pass"]),
        "formal_clean_fixed_point": config["preset"] != "formal" or not dirty,
        "implementation_source_inventory_nonempty": bool(implementation_sources),
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="fixed_sources_and_formal_pilot_authorization",
        source_git_sha=git_sha,
        working_tree_clean=not dirty,
        formal_prerequisites=formal,
        source_sha256=actual,
        implementation_source_count=len(implementation_sources),
        implementation_source_root_sha256=implementation_root,
    )


def _grid(config: Mapping[str, Any]) -> WorkspaceGridSpec:
    domain = config["domain"]
    return WorkspaceGridSpec(
        levels_mm=tuple(map(int, domain["levels_mm"])),
        x_slab_m=(float(domain["x_min_m"]), float(domain["x_max_m"])),
        origin_m=tuple(map(float, domain["origin_m"])),
    )


def _reach_proxy_config(config: Mapping[str, Any]) -> ReachProxyConfig:
    reach = config["reach"]
    return ReachProxyConfig(
        grid=_grid(config),
        minimum_weighted_jaccard=float(reach["minimum_weighted_jaccard"]),
        maximum_new_volume_ratio=float(reach["maximum_new_volume_ratio"]),
        maximum_boundary_change_ratio=float(reach["maximum_boundary_change_ratio"]),
        maximum_frontier_new_volume_ratio=float(reach["maximum_frontier_new_volume_ratio"]),
        required_consecutive_rounds=int(reach["required_consecutive_rounds"]),
    )


def _load_capability_slab(
    path: Path,
    grid: WorkspaceGridSpec,
    *,
    row_limit: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[*XYZ_COLUMNS, *BETA_COLUMNS, "minimum_margin_deg", "sigma3_m", "kappa"],
    )
    if row_limit is not None:
        if int(row_limit) < 1:
            raise ValueError("capability_row_limit must be positive when set")
        frame = frame.iloc[: int(row_limit)].copy()
    low, high = grid.x_slab_m
    return frame[frame["x_m"].between(low, high, inclusive="both")].reset_index(drop=True)


def _beta_xyz_frame(beta: np.ndarray, xyz: np.ndarray, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{name: xyz[:, index] for index, name in enumerate(XYZ_COLUMNS)},
            **{name: beta[:, index] for index, name in enumerate(BETA_COLUMNS)},
            "source": source,
        }
    )


def _frontier_policy(config: Mapping[str, Any]) -> CandidatePolicy:
    reach = config["reach"]
    candidate = config["candidates"]
    starts = int(reach["frontier_starts"])
    return CandidatePolicy(
        candidate_budget_per_node=starts,
        difficult_candidate_budget_per_node=starts,
        candidate_seed_budget_per_node=starts,
        difficult_seed_budget_per_node=starts,
        nullspace_seed_budget_per_node=0,
        search_mode=CandidateSearchMode.CORRECTION,
        solver_names=tuple(map(str, candidate["solvers"])),
        max_corrector_iterations=int(candidate["max_corrector_iterations"]),
        tracking_tolerance_mm=float(candidate["tracking_tolerance_mm"]),
        max_residual_mm=float(candidate["max_residual_mm"]),
        gold_margin_deg=float(candidate["gold_margin_deg"]),
        silver_margin_deg=float(candidate["silver_margin_deg"]),
        candidate_cluster_deg=float(candidate["cluster_deg"]),
    )


def _frontier_probe_rows(
    environment: Any,
    cells: Sequence[CellKey],
    *,
    grid: WorkspaceGridSpec,
    capability: pd.DataFrame,
    policy: CandidatePolicy,
    round_id: int,
) -> tuple[tuple[FrontierProbeEvidence, ...], pd.DataFrame]:
    cells = tuple(cells)
    if not cells:
        return (), pd.DataFrame()
    targets = cell_centers(cells, grid=grid)
    bank = solve_candidate_bank(
        environment,
        targets,
        policy,
        capability_beta_rad=capability.loc[:, BETA_COLUMNS].to_numpy(dtype=float),
        capability_xyz_m=capability.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
    )
    evidence: list[FrontierProbeEvidence] = []
    rows: list[dict[str, Any]] = []
    for node_id, cell in enumerate(cells):
        accepted = [
            row
            for row in bank.for_node(node_id)
            if row.quality is not CandidateQuality.REJECT
        ]
        evidence.append(
            FrontierProbeEvidence(
                cell=cell,
                found_valid_inverse=bool(accepted),
                # Numerical failure is deliberately unresolved, never a
                # certificate of physical unreachability.
                unreachable_certificate=None,
                round_id=round_id,
            )
        )
        report = bank.node_reports[node_id]
        best = min(
            accepted,
            key=lambda row: (
                0 if row.quality is CandidateQuality.GOLD else 1,
                row.residual_mm,
                -row.min_margin_deg,
                row.candidate_id,
            ),
        ) if accepted else None
        rows.append(
            {
                "round_id": round_id,
                "cell_level_mm": cell.level_mm,
                "cell_ix": cell.ix,
                "cell_iy": cell.iy,
                "cell_iz": cell.iz,
                "target_x_m": targets[node_id, 0],
                "target_y_m": targets[node_id, 1],
                "target_z_m": targets[node_id, 2],
                "found_valid_inverse": bool(accepted),
                "status": "empirical_supported" if accepted else "teacher_unresolved",
                "candidate_attempt_count": int(report["candidate_attempt_count"]),
                "gold_candidate_count": int(report["gold_candidate_count"]),
                "silver_candidate_count": int(report["silver_candidate_count"]),
                "selected_candidate_id": None if best is None else best.candidate_id,
                "selected_residual_mm": None if best is None else best.residual_mm,
                **{
                    name: None if best is None else float(best.beta_rad[index])
                    for index, name in enumerate(BETA_COLUMNS)
                },
            }
        )
    return tuple(evidence), pd.DataFrame(rows)


def _cell_frame(
    result: Any,
    *,
    a_cells: frozenset[CellKey],
    b_cells: frozenset[CellKey],
    existing_cells: frozenset[CellKey],
    tip_cells: frozenset[CellKey],
    frontier_cells: frozenset[CellKey],
) -> pd.DataFrame:
    rows = []
    for cell, status in sorted(result.status_by_cell.items()):
        rows.append(
            {
                "cell_id": f"l{cell.level_mm}_x{cell.ix}_y{cell.iy}_z{cell.iz}",
                "cell_level_mm": cell.level_mm,
                "cell_ix": cell.ix,
                "cell_iy": cell.iy,
                "cell_iz": cell.iz,
                "cell_volume_m3": cell.volume_m3,
                "reach_status": status.value,
                "in_proxy_lower": cell in result.proxy_lower_cells,
                "in_proxy_upper": cell in result.proxy_upper_cells,
                "replica_a_support": cell in a_cells,
                "replica_b_support": cell in b_cells,
                "registered_pool_support": cell in existing_cells,
                "tip_pool_support": cell in tip_cells,
                "frontier_ik_support": cell in frontier_cells,
            }
        )
    return pd.DataFrame(rows)


def stage_workspace_proxy(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "protocol")
    stage = output_root / STAGE_DIRS["workspace_proxy"]
    stage.mkdir(parents=True, exist_ok=False)
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    bounds = np.asarray(environment.bounds, dtype=float)
    grid = _grid(config)
    reach = config["reach"]
    round_specs = tuple(
        ReachSamplingRoundSpec(
            power=int(row["power"]),
            seed_a=int(row["seed_a"]),
            seed_b=int(row["seed_b"]),
        )
        for row in reach["rounds"]
    )
    generated = generate_independent_reach_samples(
        environment,
        bounds,
        grid=grid,
        rounds=round_specs,
        chunk_rows=int(reach["chunk_rows"]),
    )
    capability = _load_capability_slab(
        paths["capability_pool"],
        grid,
        row_limit=config["registry"].get("capability_row_limit"),
    )
    existing_cells = grid.cells_for_points(
        capability.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        level_mm=grid.convergence_level_mm,
    )
    tip_beta = tip_focused_beta(
        bounds,
        power=int(reach["tip_power"]),
        seed=int(reach["tip_seed"]),
        scale_max=float(reach["tip_scale_max"]),
    )
    tip_xyz = batch_fk(environment, tip_beta, chunk_rows=int(reach["chunk_rows"]))
    low, high = grid.x_slab_m
    tip_selected = (tip_xyz[:, 0] >= low) & (tip_xyz[:, 0] <= high)
    tip_beta, tip_xyz = tip_beta[tip_selected], tip_xyz[tip_selected]
    tip_cells = grid.cells_for_points(tip_xyz, level_mm=grid.convergence_level_mm)
    supplemental = frozenset(existing_cells | tip_cells)
    builder = ReachProxyBuilder(_reach_proxy_config(config))
    preliminary = builder.build(
        generated.rounds,
        supplemental_supported_cells=supplemental,
    )

    frontier_evidence: list[FrontierProbeEvidence] = []
    frontier_frames: list[pd.DataFrame] = []
    current_supported = set(preliminary.proxy_upper_cells)
    seed_pool = pd.concat(
        [
            capability.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]],
            _beta_xyz_frame(generated.beta_a, generated.xyz_a, "replica_a").drop(columns="source"),
            _beta_xyz_frame(generated.beta_b, generated.xyz_b, "replica_b").drop(columns="source"),
            _beta_xyz_frame(tip_beta, tip_xyz, "tip").drop(columns="source"),
        ],
        ignore_index=True,
    )
    for frontier_round in range(1, int(reach["frontier_rounds"]) + 1):
        cells = frontier_candidates(
            current_supported,
            grid=grid,
            maximum_count=int(reach["frontier_cells_per_round"]),
        )
        evidence, frame = _frontier_probe_rows(
            environment,
            cells,
            grid=grid,
            capability=seed_pool,
            policy=_frontier_policy(config),
            round_id=len(generated.rounds),
        )
        frontier_evidence.extend(evidence)
        if len(frame):
            frontier_frames.append(frame)
        new_supported = {row.cell for row in evidence if row.found_valid_inverse}
        current_supported.update(new_supported)
        if not new_supported:
            break
    result = builder.build(
        generated.rounds,
        frontier_evidence=frontier_evidence,
        supplemental_supported_cells=supplemental,
    )
    level = grid.convergence_level_mm
    latest = generated.rounds[-1]
    a_cells = grid.cells_for_points(latest.replica_a.xyz_m, level_mm=level)
    b_cells = grid.cells_for_points(latest.replica_b.xyz_m, level_mm=level)
    frontier_supported = frozenset(
        row.cell for row in frontier_evidence if row.found_valid_inverse
    )
    cells = _cell_frame(
        result,
        a_cells=a_cells,
        b_cells=b_cells,
        existing_cells=existing_cells,
        tip_cells=tip_cells,
        frontier_cells=frontier_supported,
    )
    _atomic_parquet(cells, stage / "domain_cells.parquet")
    _atomic_parquet(
        _beta_xyz_frame(generated.beta_a, generated.xyz_a, "replica_a"),
        stage / "replica_a_slab.parquet",
    )
    _atomic_parquet(
        _beta_xyz_frame(generated.beta_b, generated.xyz_b, "replica_b"),
        stage / "replica_b_slab.parquet",
    )
    _atomic_parquet(_beta_xyz_frame(tip_beta, tip_xyz, "tip"), stage / "tip_pool.parquet")
    frontier_frame = (
        pd.concat(frontier_frames, ignore_index=True)
        if frontier_frames
        else pd.DataFrame(
            columns=[
                "round_id", "cell_level_mm", "cell_ix", "cell_iy", "cell_iz",
                "target_x_m", "target_y_m", "target_z_m", "found_valid_inverse",
                "status", "candidate_attempt_count", "gold_candidate_count",
                "silver_candidate_count", "selected_candidate_id",
                "selected_residual_mm", *BETA_COLUMNS,
            ]
        )
    )
    _atomic_parquet(frontier_frame, stage / "frontier_inverse_probes.parquet")
    metrics = [vars(row) for row in result.replica_metrics]
    atomic_write_json(stage / "occupancy_convergence.json", _strict_json(metrics))
    report = {
        "semantics": "empirical_reach_lower_upper_not_continuous_reach_proof",
        "replica_a_seed_lineage": list(latest.replica_a.scramble_seeds),
        "replica_b_seed_lineage": list(latest.replica_b.scramble_seeds),
        "replica_a_slab_rows": len(generated.beta_a),
        "replica_b_slab_rows": len(generated.beta_b),
        "registered_capability_slab_rows": len(capability),
        "tip_slab_rows": len(tip_beta),
        "proxy_lower_cell_count": len(result.proxy_lower_cells),
        "proxy_upper_cell_count": len(result.proxy_upper_cells),
        "certified_unreachable_cell_count": len(result.certified_unreachable_cells),
        "frontier_probe_count": len(frontier_evidence),
        "frontier_new_supported_count": len(frontier_supported - supplemental),
        "convergence_gate": bool(result.convergence_gate),
        "metrics": metrics,
    }
    atomic_write_json(stage / "workspace_proxy_report.json", _strict_json(report))
    exploratory = config["preset"] in {"smoke", "pilot"}
    checks = {
        "independent_seed_lineage": not (
            set(latest.replica_a.scramble_seeds)
            & set(latest.replica_b.scramble_seeds)
        ),
        "lower_proxy_nonempty": len(result.proxy_lower_cells) > 0,
        "all_cells_explicitly_classified": len(cells) == len(result.status_by_cell),
        "no_numerical_failure_certified_unreachable": len(result.certified_unreachable_cells) == 0,
        "formal_reach_convergence_or_exploratory_only": exploratory or result.convergence_gate,
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics=report["semantics"],
        exploratory_only=exploratory and not result.convergence_gate,
        reach_convergence_gate=bool(result.convergence_gate),
        proxy_lower_cell_count=len(result.proxy_lower_cells),
        proxy_upper_cell_count=len(result.proxy_upper_cells),
        frontier_probe_count=len(frontier_evidence),
    )


def _registry_policy(
    config: Mapping[str, Any], bounds_rad: np.ndarray, capability: pd.DataFrame
) -> WorkspaceRegistryPolicy:
    values = config["registry"]
    grid = _grid(config)
    xyz = capability.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    indices = np.floor(xyz / 0.01).astype(np.int64)
    _keys, counts = np.unique(indices, axis=0, return_counts=True)
    density_quantile = float(values["low_density_quantile"])
    low_density = max(1, int(np.quantile(counts, density_quantile))) if len(counts) else 1
    slab_width_mm = (grid.x_slab_m[1] - grid.x_slab_m[0]) * 1000.0
    x_bin_count = max(1, int(round(slab_width_mm / float(values["x_bin_width_mm"]))))
    return WorkspaceRegistryPolicy(
        beta_bounds_rad=np.asarray(bounds_rad, dtype=float),
        pilot_level_mm=10,
        pilot_x_bins=x_bin_count,
        measure_probe_count=int(values["probe_count"]),
        max_seed_betas=int(values["seed_count"]),
        seed_cluster_radius_normalized=0.025,
        low_density_max_samples=low_density,
        tip_x_fraction=float(values["tip_x_width_mm"]) / slab_width_mm,
        # This is only a Pilot sampling screen using the stored radian-coordinate
        # capability metric.  Final physical/risk classification recomputes the
        # normalized Jacobian at selected candidates in the atlas stage.
        ill_conditioned_sigma3_m_max=None,
        ill_conditioned_kappa_min=float(values["ill_conditioned_kappa"]),
        selection_seed=int(values["pilot_seed"]),
    )


def _cell_id(cell: CellKey) -> str:
    return f"l{cell.level_mm}_x{cell.ix}_y{cell.iy}_z{cell.iz}"


def pilot_probe_frame(
    pilot_cells: Sequence[Any],
    *,
    sample_by_id: Mapping[str, Any],
    node_id_by_cell: Mapping[CellKey, int],
) -> pd.DataFrame:
    """Render the exact registry-to-atlas probe schema.

    Cell identity is repeated on every probe deliberately: downstream atlas
    construction must not recover it by parsing ``cell_id`` or by joining an
    unsealed, potentially reordered representative table.
    """

    rows: list[dict[str, Any]] = []
    for item in pilot_cells:
        node_id = int(node_id_by_cell[item.cell])
        probe_ids = (item.representative_sample_id, *item.measure_sample_ids)
        for probe_index, sample_id in enumerate(probe_ids):
            sample = sample_by_id[sample_id]
            rows.append(
                {
                    "probe_id": f"n{node_id:06d}_p{probe_index:02d}",
                    "physical_point_id": sample_id,
                    "node_id": node_id,
                    "cell_id": _cell_id(item.cell),
                    "cell_level_mm": item.cell.level_mm,
                    "cell_ix": item.cell.ix,
                    "cell_iy": item.cell.iy,
                    "cell_iz": item.cell.iz,
                    "is_representative": probe_index == 0,
                    "is_measure_probe": probe_index > 0,
                    **{
                        name: sample.xyz_m[index]
                        for index, name in enumerate(XYZ_COLUMNS)
                    },
                    **{
                        name: sample.beta_rad[index]
                        for index, name in enumerate(BETA_COLUMNS)
                    },
                }
            )
    return pd.DataFrame.from_records(rows)


def stage_domain_registry(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "workspace_proxy")
    stage = output_root / STAGE_DIRS["domain_registry"]
    stage.mkdir(parents=True, exist_ok=False)
    paths = _source_paths(config, project_root)
    grid = _grid(config)
    environment = load_environment(project_root, paths["robot_config"])
    capability = _load_capability_slab(
        paths["capability_pool"],
        grid,
        row_limit=config["registry"].get("capability_row_limit"),
    )
    capability["sample_id"] = capability.index.map(lambda value: f"cap_{int(value):09d}")
    capability["reach_status"] = "empirical_supported"
    capability["ill_conditioned_raw_screen"] = capability["kappa"].ge(
        float(config["registry"]["ill_conditioned_kappa"])
    )
    retention = pd.read_parquet(paths["v13_retention"], columns=list(XYZ_COLUMNS))
    retention_xyz = retention.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    if len(retention_xyz):
        distance, _index = cKDTree(retention_xyz).query(
            capability.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), k=1
        )
        capability["retention"] = distance <= (
            float(config["registry"]["retention_radius_mm"]) / 1000.0
        )
    else:
        capability["retention"] = False
    policy = _registry_policy(config, environment.bounds, capability)
    builder = WorkspaceRegistryBuilder(
        grid,
        policy,
        CapabilityColumns(
            reach_status="reach_status",
            retention="retention",
            tip=None,
            ill_conditioned="ill_conditioned_raw_screen",
            sigma3_m=None,
            kappa=None,
        ),
    )
    registry = builder.build(capability)
    requested = int(config["registry"]["pilot_cell_count"])
    if config["preset"] == "formal" and requested == 0:
        requested = len(registry.pilot_cells)
    selection = registry.select_pilot(requested)
    sample_by_id = registry.samples_by_id

    registry_rows: list[dict[str, Any]] = []
    for level, cells_at_level in registry.cells_by_level.items():
        for cell, item in cells_at_level.items():
            registry_rows.append(
                {
                    "cell_id": _cell_id(cell),
                    "cell_level_mm": level,
                    "cell_ix": cell.ix,
                    "cell_iy": cell.iy,
                    "cell_iz": cell.iz,
                    "sample_count": item.sample_count,
                    "occupancy_fraction": item.occupancy_fraction,
                    "cell_measure_m3": (
                        cell.volume_m3
                        if item.occupancy_fraction is None
                        else cell.volume_m3 * item.occupancy_fraction
                    ),
                    "strata": "|".join(sorted(value.value for value in item.strata)),
                    "empirical_reach_status": "empirical_supported",
                }
            )
    _atomic_parquet(pd.DataFrame(registry_rows), stage / "registry_cells.parquet")

    ordered_pilots = tuple(sorted(selection.cells, key=lambda item: item.cell))
    node_id_by_cell = {item.cell: index for index, item in enumerate(ordered_pilots)}
    task_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for item in ordered_pilots:
        node_id = node_id_by_cell[item.cell]
        representative = sample_by_id[item.representative_sample_id]
        task_rows.append(
            {
                "node_id": node_id,
                "cell_id": _cell_id(item.cell),
                "cell_level_mm": item.cell.level_mm,
                "cell_ix": item.cell.ix,
                "cell_iy": item.cell.iy,
                "cell_iz": item.cell.iz,
                "x_bin": item.x_bin,
                "strata": "|".join(sorted(value.value for value in item.strata)),
                "representative_sample_id": item.representative_sample_id,
                **{name: representative.xyz_m[index] for index, name in enumerate(XYZ_COLUMNS)},
                **{name: representative.beta_rad[index] for index, name in enumerate(BETA_COLUMNS)},
            }
        )
        for seed_rank, (sample_id, beta) in enumerate(
            zip(item.seed_sample_ids, item.seed_beta_rad, strict=True)
        ):
            seed_rows.append(
                {
                    "node_id": node_id,
                    "cell_id": _cell_id(item.cell),
                    "seed_rank": seed_rank,
                    "source_sample_id": sample_id,
                    **{name: beta[index] for index, name in enumerate(BETA_COLUMNS)},
                }
            )
    tasks = pd.DataFrame(task_rows)
    probes = pilot_probe_frame(
        ordered_pilots,
        sample_by_id=sample_by_id,
        node_id_by_cell=node_id_by_cell,
    )
    seeds = pd.DataFrame(seed_rows)
    edges = pd.DataFrame.from_records(
        [
            {
                "left_node_id": node_id_by_cell[left],
                "right_node_id": node_id_by_cell[right],
                "left_cell_id": _cell_id(left),
                "right_cell_id": _cell_id(right),
                "adjacency": "face_6",
            }
            for left, right in selection.task_edges
        ],
        columns=[
            "left_node_id", "right_node_id", "left_cell_id", "right_cell_id", "adjacency"
        ],
    )
    _atomic_parquet(tasks, stage / "pilot_task_nodes.parquet")
    _atomic_parquet(probes, stage / "pilot_task_probes.parquet")
    _atomic_parquet(seeds, stage / "pilot_seed_bank.parquet")
    _atomic_parquet(edges, stage / "pilot_task_edges.parquet")

    proxy = pd.read_parquet(
        output_root / STAGE_DIRS["workspace_proxy"] / "domain_cells.parquet"
    )
    lower = {
        CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in proxy.loc[proxy["in_proxy_lower"]].itertuples(index=False)
    }
    registered_20 = set(registry.cells_by_level[20])
    registry_coverage = float(len(lower & registered_20) / len(lower)) if lower else 0.0
    selected_strata = {
        value.value for item in ordered_pilots for value in item.strata
    }
    selected_x_bins = {item.x_bin for item in ordered_pilots}
    slab_low, slab_high = grid.x_slab_m
    def registered_x_bin(x_m: float) -> int:
        relative = (float(x_m) - slab_low) / (slab_high - slab_low)
        return min(
            policy.pilot_x_bins - 1,
            max(0, int(np.floor(relative * policy.pilot_x_bins))),
        )
    eligible_x_bins = {
        registered_x_bin(item.x_center_m)
        for item in registry.pilot_cells.values()
        if item.sample_count >= policy.measure_probe_count + 1
    }
    report = {
        "semantics": "empirical_supported_cells_with_multi_probe_pilot_requests",
        "registered_sample_count": len(registry.samples_by_id),
        "registered_cell_count_by_level": {
            str(level): len(rows) for level, rows in registry.cells_by_level.items()
        },
        "requested_pilot_cell_count": requested,
        "eligible_pilot_cell_count": selection.eligible_cell_count,
        "selected_pilot_cell_count": len(ordered_pilots),
        "pilot_probe_count": len(probes),
        "pilot_seed_count": len(seeds),
        "pilot_edge_count": len(edges),
        "selected_x_bins": sorted(selected_x_bins),
        "eligible_x_bins": sorted(eligible_x_bins),
        "selected_strata": sorted(selected_strata),
        "proxy_lower_registry_20mm_coverage": registry_coverage,
        "occupancy_fraction_semantics": "supported_5mm_children_divided_by_8",
        "ill_conditioned_stratum_semantics": "raw_radian_kappa_screen_only; normalized_J_recomputed_after_IK",
    }
    atomic_write_json(stage / "registry_report.json", _strict_json(report))
    exploratory = config["preset"] in {"smoke", "pilot"}
    checks = {
        "pilot_selection_nonempty": len(ordered_pilots) > 0,
        "pilot_selection_exact_without_padding": len(ordered_pilots)
        == min(requested, selection.eligible_cell_count),
        "multiple_measure_probes_per_cell": len(probes)
        == len(ordered_pilots) * (policy.measure_probe_count + 1),
        "all_eligible_x_bins_represented": selected_x_bins == eligible_x_bins,
        "formal_proxy_registry_coverage_or_exploratory_only": exploratory
        or registry_coverage >= 0.95,
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics=report["semantics"],
        exploratory_only=exploratory,
        selected_pilot_cell_count=len(ordered_pilots),
        eligible_pilot_cell_count=selection.eligible_cell_count,
        proxy_lower_registry_20mm_coverage=registry_coverage,
        selected_strata=sorted(selected_strata),
    )


def select_saturation_node_ids(
    tasks: pd.DataFrame, *, fraction: float, seed: int
) -> tuple[int, ...]:
    """Select an exact deterministic, x-bin/stratum-balanced audit subset."""

    required = {"node_id", "x_bin", "strata"}
    missing = sorted(required - set(tasks.columns))
    if missing:
        raise ValueError(f"saturation tasks missing columns: {missing}")
    if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) <= 1.0:
        raise ValueError("saturation fraction must be in (0, 1]")
    if tasks["node_id"].duplicated().any():
        raise ValueError("saturation task node IDs must be unique")
    target = min(len(tasks), max(1, int(math.ceil(len(tasks) * float(fraction)))))
    buckets: dict[tuple[int, str], list[int]] = {}
    for row in tasks.itertuples(index=False):
        # Use the complete overlapping-strata signature so rare compound cases
        # do not disappear behind a single arbitrarily chosen label.
        key = (int(row.x_bin), str(row.strata))
        buckets.setdefault(key, []).append(int(row.node_id))
    def rank(node_id: int) -> tuple[bytes, int]:
        digest = hashlib.sha256(f"{int(seed)}:{node_id}".encode("utf-8")).digest()
        return digest, node_id
    for key in buckets:
        buckets[key].sort(key=rank)
    groups = tuple(sorted(buckets))
    cursors = {key: 0 for key in groups}
    selected: list[int] = []
    while len(selected) < target:
        progress = False
        for key in groups:
            cursor = cursors[key]
            if cursor >= len(buckets[key]):
                continue
            node_id = buckets[key][cursor]
            cursors[key] += 1
            if node_id in selected:
                continue
            selected.append(node_id)
            progress = True
            if len(selected) == target:
                break
        if not progress:
            break
    return tuple(sorted(selected))


def diverse_capability_seed_rows(
    targets: pd.DataFrame,
    *,
    capability_xyz: np.ndarray,
    capability_beta: np.ndarray,
    bounds_rad: np.ndarray,
    seed_count: int,
    neighbor_count: int,
) -> pd.DataFrame:
    """Choose task-near but beta-maximin starts for branch discovery."""

    required = {"node_id", *XYZ_COLUMNS}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"seed targets missing columns: {missing}")
    xyz = np.asarray(capability_xyz, dtype=float)
    beta = np.asarray(capability_beta, dtype=float)
    bounds = np.asarray(bounds_rad, dtype=float)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or beta.shape != (len(xyz), 6)
        or bounds.shape != (6, 2)
        or not np.isfinite(xyz).all()
        or not np.isfinite(beta).all()
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("capability seed arrays or bounds are invalid")
    if not 1 <= int(seed_count) <= int(neighbor_count) or int(neighbor_count) < 1:
        raise ValueError("seed_count must be positive and not exceed neighbor_count")
    if len(xyz) < int(seed_count):
        raise ValueError("capability pool is smaller than seed_count")
    query_count = min(len(xyz), int(neighbor_count))
    query_xyz = targets.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    _distance, raw_indices = cKDTree(xyz).query(query_xyz, k=query_count)
    raw_indices = np.asarray(raw_indices, dtype=int)
    if raw_indices.ndim == 1:
        raw_indices = raw_indices[:, None]
    span = bounds[:, 1] - bounds[:, 0]
    rows: list[dict[str, Any]] = []
    for target_offset, target in enumerate(targets.itertuples(index=False)):
        indices = list(dict.fromkeys(map(int, raw_indices[target_offset].tolist())))
        normalized = (beta[indices] - bounds[:, 0]) / span
        chosen = [0]
        while len(chosen) < min(int(seed_count), len(indices)):
            remaining = [index for index in range(len(indices)) if index not in chosen]
            next_index = min(
                remaining,
                key=lambda index: (
                    -float(
                        np.min(
                            np.linalg.norm(
                                normalized[index] - normalized[chosen], axis=1
                            )
                        )
                    ),
                    indices[index],
                ),
            )
            chosen.append(next_index)
        for rank_index, relative in enumerate(chosen):
            source_index = indices[relative]
            rows.append(
                {
                    "node_id": int(target.node_id),
                    "seed_rank": rank_index,
                    "capability_row_index": source_index,
                    **{
                        name: float(beta[source_index, beta_index])
                        for beta_index, name in enumerate(BETA_COLUMNS)
                    },
                }
            )
    return pd.DataFrame(rows)


WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _slice_ranges(count: int, workers: int) -> tuple[tuple[int, int, int], ...]:
    if int(count) < 1:
        return ()
    effective = max(1, min(int(workers), int(count)))
    boundaries = np.linspace(0, int(count), effective + 1, dtype=np.int64)
    return tuple(
        (index, int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(effective)
        if int(boundaries[index]) < int(boundaries[index + 1])
    )


def _run_subprocess_tasks(
    commands: Sequence[tuple[str, Sequence[str], Path]],
    *,
    requested_workers: int,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run independent Python worker processes without multiprocessing IPC."""

    if not commands:
        raise ValueError("subprocess task list must not be empty")
    effective = min(max(1, int(requested_workers)), len(commands))
    environment = os.environ.copy()
    environment.update(WORKER_ENV)
    pending = list(commands)
    running: list[tuple[str, subprocess.Popen[str], Path, float, float]] = []
    completed: list[dict[str, Any]] = []
    started = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_concurrent_rss_mib = 0.0
    while pending or running:
        while pending and len(running) < effective:
            task_id, command, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                list(command),
                cwd=SOURCE_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            handle.close()
            running.append((task_id, process, log_path, time.time(), 0.0))
        time.sleep(0.1)
        survivors: list[tuple[str, subprocess.Popen[str], Path, float, float]] = []
        concurrent_rss_mib = 0.0
        for task_id, process, log_path, task_started, task_peak in running:
            rss_mib = 0.0
            try:
                status = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8")
                for line in status.splitlines():
                    if line.startswith("VmRSS:"):
                        rss_mib = float(line.split()[1]) / 1024.0
                        break
            except (FileNotFoundError, PermissionError, ValueError):
                pass
            task_peak = max(task_peak, rss_mib)
            concurrent_rss_mib += rss_mib
            return_code = process.poll()
            if return_code is None:
                survivors.append((task_id, process, log_path, task_started, task_peak))
                continue
            completed.append(
                {
                    "task_id": task_id,
                    "return_code": int(return_code),
                    "wall_time_s": time.time() - task_started,
                    "peak_rss_mib": task_peak,
                    "log_path": str(log_path),
                }
            )
            if return_code != 0:
                for _other_id, other, _path, _start, _peak in survivors:
                    other.terminate()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(f"subprocess task {task_id} failed:\n{tail}")
        peak_concurrent_rss_mib = max(peak_concurrent_rss_mib, concurrent_rss_mib)
        running = survivors
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.time() - started
    child_cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    report = {
        "requested_workers": int(requested_workers),
        "effective_workers": effective,
        "task_count": len(commands),
        "wall_time_s": wall,
        "aggregate_child_cpu_time_s": child_cpu,
        "aggregate_cpu_utilization": child_cpu / wall if wall > 0.0 else 0.0,
        "peak_concurrent_rss_mib": peak_concurrent_rss_mib,
        "worker_thread_environment": WORKER_ENV,
        "tasks": sorted(completed, key=lambda row: row["task_id"]),
    }
    atomic_write_json(manifest_path, _strict_json(report))
    return report


def _candidate_policy(config: Mapping[str, Any], mode: str) -> CandidatePolicy:
    values = config["candidates"]
    if mode == "saturation":
        ordinary = difficult = int(values["saturation_starts"])
        nullspace = max(int(values["nullspace_starts"]), 4)
    elif mode == "base":
        ordinary = int(values["normal_starts"])
        difficult = int(values["difficult_starts"])
        nullspace = int(values["nullspace_starts"])
    else:
        raise ValueError(f"unknown candidate worker mode {mode!r}")
    maximum = max(ordinary, difficult)
    return CandidatePolicy(
        candidate_budget_per_node=maximum,
        difficult_candidate_budget_per_node=maximum,
        candidate_seed_budget_per_node=ordinary,
        difficult_seed_budget_per_node=difficult,
        nullspace_seed_budget_per_node=nullspace,
        search_mode=CandidateSearchMode.DIVERSITY,
        solver_names=tuple(map(str, values["solvers"])),
        max_corrector_iterations=int(values["max_corrector_iterations"]),
        tracking_tolerance_mm=float(values["tracking_tolerance_mm"]),
        max_residual_mm=float(values["max_residual_mm"]),
        gold_margin_deg=float(values["gold_margin_deg"]),
        silver_margin_deg=float(values["silver_margin_deg"]),
        candidate_cluster_deg=float(values["cluster_deg"]),
        cluster_sensitivity_deg=(0.25, float(values["cluster_deg"]), 1.0),
        solver_seed=int(values["saturation_seed"]),
    )


def _candidate_records_frame(
    bank: Any, global_node_ids: Sequence[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in bank.records:
        global_id = int(global_node_ids[int(record.node_id)])
        rows.append(
            {
                "node_id": global_id,
                "candidate_id": record.candidate_id,
                "source": record.source,
                "solver": record.solver,
                "seed_rank": record.seed_rank,
                "residual_mm": record.residual_mm,
                "minimum_margin_deg": record.min_margin_deg,
                "normalized_minimum_margin": record.normalized_min_margin,
                "quality_class": record.quality.value,
                "solver_success": record.solver_success,
                "achieved_x_m": record.achieved_xyz_m[0],
                "achieved_y_m": record.achieved_xyz_m[1],
                "achieved_z_m": record.achieved_xyz_m[2],
                "sigma3_m_raw": record.diagnostics.get("sigma3_m"),
                "kappa_raw": record.diagnostics.get("kappa"),
                "diagnostics_json": json.dumps(
                    _strict_json(dict(record.diagnostics)),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                **{
                    name: float(record.beta_rad[index])
                    for index, name in enumerate(BETA_COLUMNS)
                },
            }
        )
    return pd.DataFrame(rows)


def candidate_worker(args: argparse.Namespace) -> int:
    config = load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    tasks = pd.read_parquet(args.task_nodes).sort_values("node_id", kind="stable")
    selected = tasks.iloc[int(args.start) : int(args.stop)].reset_index(drop=True)
    seeds = pd.read_parquet(args.seed_bank)
    global_ids = selected["node_id"].astype(int).tolist()
    seed_by_local: dict[int, np.ndarray] = {}
    for local_id, global_id in enumerate(global_ids):
        group = seeds[seeds["node_id"].eq(global_id)].sort_values(
            "seed_rank", kind="stable"
        )
        seed_by_local[local_id] = group.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    difficult_words = {"boundary", "tip", "low_density", "ill_conditioned"}
    difficult = tuple(
        local_id
        for local_id, value in enumerate(selected["strata"].astype(str))
        if difficult_words & set(value.split("|"))
    )
    exact = {
        local_id: selected.iloc[local_id].loc[list(BETA_COLUMNS)].to_numpy(dtype=float)
        for local_id in range(len(selected))
    }
    bank = solve_candidate_bank(
        environment,
        selected.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        _candidate_policy(config, args.worker_mode),
        neighbor_beta_rad=seed_by_local,
        difficult_node_ids=difficult,
        node_seed_beta_rad=exact,
    )
    frame = _candidate_records_frame(bank, global_ids)
    _atomic_parquet(frame, Path(args.output))
    reports = {
        str(global_ids[int(local_id)]): {
            **_strict_json(dict(report)),
            "node_id": global_ids[int(local_id)],
        }
        for local_id, report in bank.node_reports.items()
    }
    atomic_write_json(Path(args.report), reports)
    return 0


def _run_candidate_wave(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    tasks_path: Path,
    seeds_path: Path,
    output_dir: Path,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    tasks = pd.read_parquet(tasks_path, columns=["node_id"])
    ranges = _slice_ranges(len(tasks), int(config["parallel"]["candidate_workers"]))
    if not ranges:
        raise ValueError("candidate wave requires at least one task")
    commands: list[tuple[str, Sequence[str], Path]] = []
    for shard_id, start, stop in ranges:
        output = output_dir / f"{mode}_candidates_{shard_id:03d}.parquet"
        report = output_dir / f"{mode}_nodes_{shard_id:03d}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_candidate-worker",
            "--config", str(config["config_path"]),
            "--preset", str(config["preset"]),
            "--project-root", str(project_root),
            "--worker-mode", mode,
            "--task-nodes", str(tasks_path),
            "--seed-bank", str(seeds_path),
            "--start", str(start),
            "--stop", str(stop),
            "--output", str(output),
            "--report", str(report),
        ]
        commands.append(
            (f"{mode}_{shard_id:03d}", command, output_dir / f"{mode}_{shard_id:03d}.log")
        )
    parallel = _run_subprocess_tasks(
        commands,
        requested_workers=int(config["parallel"]["candidate_workers"]),
        manifest_path=output_dir / f"{mode}_parallel_manifest.json",
    )
    frames = [
        pd.read_parquet(output_dir / f"{mode}_candidates_{shard_id:03d}.parquet")
        for shard_id, _start, _stop in ranges
    ]
    records = pd.concat(frames, ignore_index=True).sort_values(
        ["node_id", "candidate_id"], kind="stable"
    )
    reports: dict[str, Any] = {}
    for shard_id, _start, _stop in ranges:
        report = _read_json(output_dir / f"{mode}_nodes_{shard_id:03d}.json")
        overlap = set(reports) & set(report)
        if overlap:
            raise RuntimeError(f"duplicate candidate node reports: {sorted(overlap)}")
        reports.update(report)
    expected = set(tasks["node_id"].astype(int))
    actual_reports = set(map(int, reports))
    actual_records = set(records["node_id"].astype(int))
    if actual_reports != expected or actual_records != expected:
        raise RuntimeError("candidate shard exact-set verification failed")
    return records.reset_index(drop=True), reports, parallel


def _candidate_cluster_betas(frame: pd.DataFrame, threshold_deg: float) -> np.ndarray:
    accepted = frame[
        frame["quality_class"].isin(("Gold", "Silver"))
        & frame["solver_success"].eq(True)
    ].copy()
    if not len(accepted):
        return np.empty((0, 6), dtype=float)
    beta = accepted.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    score = accepted["residual_mm"].to_numpy(dtype=float) - 1.0e-3 * accepted[
        "minimum_margin_deg"
    ].to_numpy(dtype=float)
    representatives, _indices = stable_cluster_representatives(
        beta, scores=score, threshold_deg=float(threshold_deg)
    )
    return representatives


def stage_candidate_bank(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "domain_registry")
    stage = output_root / STAGE_DIRS["candidate_bank"]
    stage.mkdir(parents=True, exist_ok=False)
    shard_dir = stage / "shards"
    shard_dir.mkdir(parents=True, exist_ok=False)
    registry_stage = output_root / STAGE_DIRS["domain_registry"]
    task_path = registry_stage / "pilot_task_nodes.parquet"
    base_seed_path = registry_stage / "pilot_seed_bank.parquet"
    base, base_reports, base_parallel = _run_candidate_wave(
        config,
        project_root=project_root,
        tasks_path=task_path,
        seeds_path=base_seed_path,
        output_dir=shard_dir,
        mode="base",
    )
    _atomic_parquet(base, stage / "cell_candidates.parquet")
    atomic_write_json(stage / "candidate_node_reports.json", base_reports)

    tasks = pd.read_parquet(task_path).sort_values("node_id", kind="stable")
    audit_ids = select_saturation_node_ids(
        tasks,
        fraction=float(config["candidates"]["saturation_fraction"]),
        seed=int(config["candidates"]["saturation_seed"]),
    )
    audit_tasks = tasks[tasks["node_id"].isin(audit_ids)].reset_index(drop=True)
    audit_task_path = stage / "saturation_task_nodes.parquet"
    _atomic_parquet(audit_tasks, audit_task_path)
    paths = _source_paths(config, project_root)
    capability = _load_capability_slab(
        paths["capability_pool"],
        _grid(config),
        row_limit=config["registry"].get("capability_row_limit"),
    )
    saturation_starts = int(config["candidates"]["saturation_starts"])
    audit_seeds = diverse_capability_seed_rows(
        audit_tasks,
        capability_xyz=capability.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        capability_beta=capability.loc[:, BETA_COLUMNS].to_numpy(dtype=float),
        bounds_rad=load_environment(project_root, paths["robot_config"]).bounds,
        seed_count=saturation_starts,
        neighbor_count=min(
            len(capability), max(saturation_starts, saturation_starts * 16)
        ),
    )
    audit_seed_path = stage / "saturation_seed_bank.parquet"
    _atomic_parquet(audit_seeds, audit_seed_path)
    saturation, saturation_reports, saturation_parallel = _run_candidate_wave(
        config,
        project_root=project_root,
        tasks_path=audit_task_path,
        seeds_path=audit_seed_path,
        output_dir=shard_dir,
        mode="saturation",
    )
    _atomic_parquet(saturation, stage / "saturation_candidates.parquet")
    atomic_write_json(stage / "saturation_node_reports.json", saturation_reports)

    cluster_gap = float(config["candidates"]["new_branch_gap_deg"])
    audit_rows: list[dict[str, Any]] = []
    observations: list[BranchAuditObservation] = []
    task_by_id = tasks.set_index("node_id")
    for node_id in audit_ids:
        base_beta = _candidate_cluster_betas(
            base[base["node_id"].eq(node_id)], cluster_gap
        )
        audit_beta = _candidate_cluster_betas(
            saturation[saturation["node_id"].eq(node_id)], cluster_gap
        )
        new_gaps = [
            min(
                (beta_rms_deg(beta, reference) for reference in base_beta),
                default=math.inf,
            )
            for beta in audit_beta
        ]
        discovered = bool(new_gaps and max(new_gaps) > cluster_gap)
        task = task_by_id.loc[node_id]
        cell = CellKey(
            int(task.cell_level_mm), int(task.cell_ix), int(task.cell_iy), int(task.cell_iz)
        )
        stratum = str(task.strata)
        observations.append(BranchAuditObservation(cell, stratum, discovered))
        audit_rows.append(
            {
                "node_id": node_id,
                "cell_id": str(task.cell_id),
                "stratum": stratum,
                "base_cluster_count": len(base_beta),
                "saturation_cluster_count": len(audit_beta),
                "new_stable_branch": discovered,
                "new_branch_max_gap_deg": max(new_gaps, default=None),
                "base_solver_attempt_count": int(len(base[base["node_id"].eq(node_id)])),
                "saturation_solver_attempt_count": int(
                    len(saturation[saturation["node_id"].eq(node_id)])
                ),
            }
        )
    audit_frame = pd.DataFrame(audit_rows)
    _atomic_parquet(audit_frame, stage / "branch_saturation_audit.parquet")
    atlas_policy = config["atlas"]
    saturation_report = WorkspaceAtlasBuilder(
        WorkspaceAtlasPolicy(
            branch_overall_wilson_upper=float(
                atlas_policy["branch_overall_wilson_upper"]
            ),
            branch_stratum_wilson_upper=float(
                atlas_policy["branch_stratum_wilson_upper"]
            ),
        )
    ).build(WorkspaceAtlasInput(branch_audit=tuple(observations))).branch_saturation

    node_summary: list[dict[str, Any]] = []
    for row in tasks.itertuples(index=False):
        group = base[base["node_id"].eq(int(row.node_id))]
        accepted = group["quality_class"].isin(("Gold", "Silver")) & group[
            "solver_success"
        ].eq(True)
        node_summary.append(
            {
                "node_id": int(row.node_id),
                "cell_id": str(row.cell_id),
                "strata": str(row.strata),
                "attempt_count": len(group),
                "accepted_candidate_count": int(accepted.sum()),
                "cluster_count_1deg": len(
                    _candidate_cluster_betas(group, cluster_gap)
                ),
                "representative_status": (
                    "resolved_under_budget" if accepted.any() else "teacher_unresolved"
                ),
            }
        )
    summary = pd.DataFrame(node_summary)
    _atomic_parquet(summary, stage / "candidate_node_summary.parquet")
    candidate_acceptance = float(summary["accepted_candidate_count"].gt(0).mean())
    report = {
        "semantics": "correction_and_diversity_candidate_search_with_saturation_audit",
        "task_node_count": len(tasks),
        "candidate_attempt_count": len(base),
        "resolved_under_budget_count": int(summary["accepted_candidate_count"].gt(0).sum()),
        "resolved_under_budget_ratio": candidate_acceptance,
        "audit_sample_count": saturation_report.sample_count,
        "audit_new_branch_count": saturation_report.new_branch_count,
        "audit_observed_new_branch_rate": saturation_report.observed_rate,
        "audit_overall_wilson_upper": saturation_report.overall_wilson_upper,
        "audit_stratum_wilson_upper": dict(saturation_report.stratum_wilson_upper),
        "branch_saturation_gate": saturation_report.gate_pass,
        "base_parallel": base_parallel,
        "saturation_parallel": saturation_parallel,
    }
    atomic_write_json(stage / "candidate_report.json", _strict_json(report))
    exploratory = config["preset"] in {"smoke", "pilot"}
    checks = {
        "candidate_attempt_exact_set": set(base["node_id"].astype(int))
        == set(tasks["node_id"].astype(int)),
        "saturation_audit_exact_size": len(audit_ids)
        == min(
            len(tasks),
            max(
                1,
                int(
                    math.ceil(
                        len(tasks)
                        * float(config["candidates"]["saturation_fraction"])
                    )
                ),
            ),
        ),
        "no_candidate_padding": not base["source"].astype(str).str.contains("padding").any(),
        "formal_branch_saturation_or_exploratory_only": exploratory
        or saturation_report.gate_pass,
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics=report["semantics"],
        exploratory_only=exploratory and not saturation_report.gate_pass,
        branch_saturation_gate=saturation_report.gate_pass,
        audit_sample_count=saturation_report.sample_count,
        audit_new_branch_count=saturation_report.new_branch_count,
        resolved_under_budget_ratio=candidate_acceptance,
    )


def _atlas_policy(config: Mapping[str, Any]) -> WorkspaceAtlasIntegrationPolicy:
    values = config["atlas"]
    domain = config["domain"]
    return WorkspaceAtlasIntegrationPolicy(
        atlas_policy=AtlasPolicy(
            edge_match_deg=float(values["edge_match_deg"]),
            continuation_residual_max_mm=float(config["candidates"]["max_residual_mm"]),
            root_count=int(values["root_count"]),
            top_section_count=int(values["top_section_count"]),
            split_gap_deg=float(values["split_gap_deg"]),
            merge_overlap_p95_deg=float(values["merge_overlap_p95_deg"]),
            merge_overlap_max_deg=float(values["merge_overlap_max_deg"]),
        ),
        audit_policy=AtlasAuditPolicy(
            endpoint_count=int(values["audit_endpoint_count"]),
            paths_per_endpoint=int(values["audit_paths_per_endpoint"]),
            loop_count=int(values["audit_loop_count"]),
            path_p95_deg=float(values["cycle_p95_deg"]),
            loop_p95_deg=float(values["cycle_p95_deg"]),
            direction_p95_deg=float(values["cycle_p95_deg"]),
            overlap_p95_deg=float(values["merge_overlap_p95_deg"]),
            repeat_p95_deg=min(0.2, float(values["cycle_p95_deg"])),
            common_max_deg=float(values["cycle_max_deg"]),
        ),
        workspace_policy=WorkspaceAtlasPolicy(
            stitchable_p95_deg=float(values["merge_overlap_p95_deg"]),
            stitchable_max_deg=float(values["merge_overlap_max_deg"]),
            nonstitchable_p95_deg=float(values["split_gap_deg"]),
            cycle_p95_deg=float(values["cycle_p95_deg"]),
            cycle_max_deg=float(values["cycle_max_deg"]),
            minimum_primary_measure_coverage=float(
                values["minimum_primary_measure_coverage"]
            ),
            minimum_x_bin_coverage=float(values["minimum_x_bin_coverage"]),
            maximum_abstention_measure_ratio=float(
                values["maximum_abstention_measure_ratio"]
            ),
            x_slab_low_m=float(domain["x_min_m"]),
            x_bin_width_m=float(config["registry"]["x_bin_width_mm"]) / 1000.0,
            branch_overall_wilson_upper=float(
                values["branch_overall_wilson_upper"]
            ),
            branch_stratum_wilson_upper=float(
                values["branch_stratum_wilson_upper"]
            ),
        ),
        candidate_cluster_deg=float(config["candidates"]["cluster_deg"]),
        default_cell_level_mm=10,
    )


def _atlas_frame_policy(config: Mapping[str, Any]) -> WorkspaceAtlasFramePolicy:
    values = config["atlas"]
    return WorkspaceAtlasFramePolicy(
        maximum_residual_mm=float(config["candidates"]["max_residual_mm"]),
        gold_margin_deg=float(config["candidates"]["gold_margin_deg"]),
        low_margin_deg=float(values["low_margin_deg"]),
        ill_conditioned_normalized_sigma3_m=float(
            values["normalized_sigma3_min_m"]
        ),
        ill_conditioned_normalized_kappa=float(values["normalized_kappa_max"]),
        candidate_cluster_deg=float(config["candidates"]["cluster_deg"]),
        maximum_measure_parent_candidates=min(
            4, int(config["candidates"]["difficult_starts"])
        ),
        minimum_measure_probes_per_cell=int(config["registry"]["probe_count"]),
    )


def _branch_observations(
    audit: pd.DataFrame, tasks: pd.DataFrame
) -> tuple[BranchAuditObservation, ...]:
    by_node = tasks.set_index("node_id")
    observations: list[BranchAuditObservation] = []
    for row in audit.itertuples(index=False):
        task = by_node.loc[int(row.node_id)]
        observations.append(
            BranchAuditObservation(
                cell=CellKey(
                    int(task.cell_level_mm),
                    int(task.cell_ix),
                    int(task.cell_iy),
                    int(task.cell_iz),
                ),
                stratum=str(row.stratum),
                new_stable_branch=bool(row.new_stable_branch),
            )
        )
    return tuple(observations)


def _domain_classification(
    *, registry_cells: pd.DataFrame, atlas_cells: pd.DataFrame,
    atlas_task_probes: pd.DataFrame,
) -> pd.DataFrame:
    registry = registry_cells[registry_cells["cell_level_mm"].eq(10)].copy()
    registry["domain_class"] = "teacher_unresolved"
    registry["representative_status"] = "not_selected_or_unresolved"
    registry["empirical_labelable_fraction"] = 0.0
    registry["labelable_measure_m3"] = 0.0
    registry["risk_flags"] = [[] for _ in range(len(registry))]
    registry["chart_count"] = 0

    probe_chart_count: dict[tuple[int, int, int], int] = {}
    if len(atlas_task_probes):
        labelable = atlas_task_probes[atlas_task_probes["labelable"].astype(bool)]
        for key, group in labelable.groupby(
            ["cell_ix", "cell_iy", "cell_iz"], sort=False
        ):
            probe_chart_count[tuple(map(int, key))] = int(
                group["chart_id"].dropna().astype(str).nunique()
            )
    atlas_by_key = {
        (int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)): row
        for row in atlas_cells.itertuples(index=False)
    }
    for index, row in registry.iterrows():
        key = (int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        assessment = atlas_by_key.get(key)
        if assessment is None:
            continue
        fraction = float(assessment.empirical_labelable_fraction)
        chart_count = probe_chart_count.get(key, 0)
        risk_flags = assessment.risk_flags
        if isinstance(risk_flags, np.ndarray):
            risk_flags = risk_flags.tolist()
        if fraction <= 0.0:
            domain_class = "teacher_unresolved"
        elif chart_count > 1:
            domain_class = "resolved_multichart"
        else:
            domain_class = "resolved_single_under_budget"
        registry.at[index, "domain_class"] = domain_class
        registry.at[index, "representative_status"] = str(
            assessment.representative_status
        )
        registry.at[index, "empirical_labelable_fraction"] = fraction
        registry.at[index, "labelable_measure_m3"] = float(row.cell_measure_m3) * fraction
        registry.at[index, "risk_flags"] = risk_flags
        registry.at[index, "chart_count"] = chart_count
    return registry.reset_index(drop=True)


def stage_workspace_atlas(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "candidate_bank")
    stage = output_root / STAGE_DIRS["workspace_atlas"]
    stage.mkdir(parents=True, exist_ok=False)
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    registry_stage = output_root / STAGE_DIRS["domain_registry"]
    candidate_stage = output_root / STAGE_DIRS["candidate_bank"]
    probes = pd.read_parquet(registry_stage / "pilot_task_probes.parquet")
    representative_edges = pd.read_parquet(
        registry_stage / "pilot_task_edges.parquet"
    )
    representative_candidates = pd.read_parquet(
        candidate_stage / "cell_candidates.parquet"
    )
    frames = prepare_workspace_atlas_frames(
        probes,
        representative_candidates,
        representative_edges,
        environment,
        policy=_atlas_frame_policy(config),
    )
    _atomic_parquet(frames.task_nodes, stage / "atlas_task_nodes.parquet")
    _atomic_parquet(frames.candidates, stage / "atlas_candidates.parquet")
    _atomic_parquet(frames.task_edges, stage / "atlas_task_edges.parquet")
    _atomic_parquet(
        frames.candidate_diagnostics, stage / "candidate_diagnostics.parquet"
    )
    audit = pd.read_parquet(candidate_stage / "branch_saturation_audit.parquet")
    representative_tasks = pd.read_parquet(
        registry_stage / "pilot_task_nodes.parquet"
    )
    result = build_workspace_atlas_integration(
        frames.task_nodes,
        frames.candidates,
        environment,
        task_edges=frames.task_edges,
        branch_audit=_branch_observations(audit, representative_tasks),
        policy=_atlas_policy(config),
    )
    artifact_names = {
        "task_nodes": "product_nodes.parquet",
        "candidates": "candidate_clusters.parquet",
        "product_edges": "product_edges.parquet",
        "audit_tasks": "fresh_audit_executions.parquet",
        "sections": "charts.parquet",
        "overlaps": "chart_overlaps.parquet",
        "primary_partition": "primary_chart_partition.parquet",
        "task_probes": "task_probe_labels.parquet",
        "cells": "selected_cell_classification.parquet",
    }
    for key, name in artifact_names.items():
        _atomic_parquet(result.frames[key], stage / name)
    atomic_write_json(stage / "atlas_report.json", _strict_json(dict(result.report)))

    registry_cells = pd.read_parquet(registry_stage / "registry_cells.parquet")
    classification = _domain_classification(
        registry_cells=registry_cells,
        atlas_cells=result.frames["cells"],
        atlas_task_probes=result.frames["task_probes"],
    )
    _atomic_parquet(classification, stage / "domain_classification.parquet")
    total_measure = float(classification["cell_measure_m3"].sum())
    class_measures = {
        str(name): float(group["cell_measure_m3"].sum())
        for name, group in classification.groupby("domain_class", sort=True)
    }
    class_ratios = {
        name: (value / total_measure if total_measure > 0.0 else 0.0)
        for name, value in class_measures.items()
    }
    representation = result.workspace_result.representation
    branch = result.workspace_result.branch_saturation
    report = {
        **dict(result.report),
        "domain_class_measure_m3": class_measures,
        "domain_class_measure_ratio": class_ratios,
        "domain_measure_denominator_m3": total_measure,
        "all_registered_10mm_cells_classified": len(classification)
        == int((registry_cells["cell_level_mm"] == 10).sum()),
        "branch_saturation_gate": bool(branch.gate_pass),
        "representation_gate_pass": representation.mode is not RepresentationMode.BLOCKED,
        "representation_mode": representation.mode.value,
        "representation_reasons": list(representation.reasons),
    }
    atomic_write_json(stage / "workspace_atlas_report.json", _strict_json(report))
    exploratory = config["preset"] in {"smoke", "pilot"}
    measure_closure = bool(
        total_measure > 0.0
        and math.isclose(sum(class_measures.values()), total_measure, rel_tol=0.0, abs_tol=1.0e-15)
    )
    scientific_gate = bool(
        branch.gate_pass
        and representation.mode is not RepresentationMode.BLOCKED
        and result.audit_report.gate_pass
    )
    checks = {
        "all_registered_10mm_cells_classified": report[
            "all_registered_10mm_cells_classified"
        ],
        "domain_measure_closure": measure_closure,
        "atlas_artifacts_nonempty": len(result.frames["task_nodes"]) > 0,
        "formal_scientific_gate_or_exploratory_only": exploratory or scientific_gate,
    }
    return _gate(
        stage / "gate.json",
        checks,
        semantics="multi_probe_sections_fresh_cycle_audit_and_representation_gate",
        exploratory_only=exploratory and not scientific_gate,
        scientific_gate_pass=scientific_gate,
        branch_saturation_gate=bool(branch.gate_pass),
        representation_gate_pass=representation.mode is not RepresentationMode.BLOCKED,
        representation_mode=representation.mode.value,
        atlas_audit_gate_pass=bool(result.audit_report.gate_pass),
        canonical_chart_count=len(result.canonical_atlas.charts),
        domain_class_measure_ratio=class_ratios,
    )


def _coarsened_base_row_count(classification: pd.DataFrame) -> int:
    labelable = classification[
        classification["domain_class"].isin(
            ("resolved_single_under_budget", "resolved_multichart")
        )
    ].copy()
    if not len(labelable):
        return 0

    def has_risk(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return bool(len(value))
        if isinstance(value, (list, tuple, set)):
            return bool(value)
        return bool(str(value).strip()) and str(value).strip() not in {"[]", "nan"}

    difficult_words = {"boundary", "tip", "low_density", "ill_conditioned"}
    low_complexity = []
    for row in labelable.itertuples(index=False):
        strata = set(str(row.strata).split("|"))
        low_complexity.append(
            str(row.domain_class) == "resolved_single_under_budget"
            and float(row.empirical_labelable_fraction) >= 1.0 - 1.0e-12
            and not bool(difficult_words & strata)
            and not has_risk(row.risk_flags)
        )
    labelable["low_complexity_interior"] = low_complexity
    interior = labelable[labelable["low_complexity_interior"]]
    if not len(interior):
        return len(labelable)
    step_m = 0.015
    centers = np.column_stack(
        [
            (interior["cell_ix"].to_numpy(dtype=float) + 0.5) * 0.010,
            (interior["cell_iy"].to_numpy(dtype=float) + 0.5) * 0.010,
            (interior["cell_iz"].to_numpy(dtype=float) + 0.5) * 0.010,
        ]
    )
    coarsened_groups = len(np.unique(np.floor(centers / step_m).astype(np.int64), axis=0))
    return int((~labelable["low_complexity_interior"]).sum() + coarsened_groups)


def stage_budget_allocation(
    config: Mapping[str, Any], _project_root: Path, output_root: Path
) -> dict[str, Any]:
    atlas_gate = _require_stage(output_root, "workspace_atlas")
    stage = output_root / STAGE_DIRS["budget_allocation"]
    stage.mkdir(parents=True, exist_ok=False)
    atlas_stage = output_root / STAGE_DIRS["workspace_atlas"]
    classification = pd.read_parquet(atlas_stage / "domain_classification.parquet")
    task_probes = pd.read_parquet(atlas_stage / "task_probe_labels.parquet")
    labelable = classification[
        classification["domain_class"].isin(
            ("resolved_single_under_budget", "resolved_multichart")
        )
    ]
    n_labelable = len(labelable)
    n_multichart = int(
        classification["domain_class"].eq("resolved_multichart").sum()
    )
    difficult_words = {"boundary", "tip", "low_density", "ill_conditioned"}
    n_high = int(
        sum(
            bool(difficult_words & set(str(row.strata).split("|")))
            or float(row.empirical_labelable_fraction) < 1.0 - 1.0e-12
            for row in labelable.itertuples(index=False)
        )
    )
    labelable_probes = task_probes[task_probes["labelable"].astype(bool)]
    expert_rows = len(
        labelable_probes.drop_duplicates(["physical_point_id", "chart_id"])
    )
    physical_rows = int(labelable_probes["physical_point_id"].nunique())
    chart_expert_extra = max(0, expert_rows - physical_rows)
    budget = config["budget"]
    refinement_extra = 7 * n_high
    demand = FixedBudgetDemand(
        labelable_base_rows=n_labelable,
        refinement_extra_rows=refinement_extra,
        chart_expert_extra_rows=chart_expert_extra,
        retention_rows=int(budget["retention_rows_max"]),
        active_rows=int(budget["active_rows"]),
        coarsened_interior_rows=_coarsened_base_row_count(classification),
    )
    feasibility = evaluate_fixed_budget_feasibility(
        demand, total_budget=int(budget["total_rows"])
    )
    category_rows = pd.DataFrame.from_records(
        [
            {
                "category": "base_coverage",
                "requested_rows": demand.labelable_base_rows,
                "configured_max_rows": int(budget["base_rows_max"]),
            },
            {
                "category": "five_mm_refinement",
                "requested_rows": demand.refinement_extra_rows,
                "configured_max_rows": int(budget["refine_rows_max"]),
            },
            {
                "category": "chart_expert_boundary",
                "requested_rows": demand.chart_expert_extra_rows,
                "configured_max_rows": int(budget["router_boundary_rows_max"]),
            },
            {
                "category": "v13_retention",
                "requested_rows": demand.retention_rows,
                "configured_max_rows": int(budget["retention_rows_max"]),
            },
            {
                "category": "active_refinement",
                "requested_rows": demand.active_rows,
                "configured_max_rows": int(budget["active_rows"]),
            },
        ]
    )
    _atomic_parquet(category_rows, stage / "budget_categories.parquet")
    over_category_caps = category_rows[
        category_rows["requested_rows"] > category_rows["configured_max_rows"]
    ]["category"].astype(str).tolist()
    report = {
        "semantics": "all_supervised_primary_expert_retention_and_active_rows_count_toward_budget",
        "N_L_labelable_base_cells": n_labelable,
        "N_M_multichart_cells": n_multichart,
        "N_H_refinement_cells": n_high,
        "N_refine_extra": refinement_extra,
        "N_chart_expert_extra": chart_expert_extra,
        "N_min": demand.minimum_supervised_rows,
        "N_min_after_15mm_low_complexity_coarsening": demand.coarsened_minimum_rows,
        "fixed_total_budget": int(budget["total_rows"]),
        "pilot_materialization_rows": int(budget["pilot_rows"]),
        "status": feasibility.status.value,
        "shortfall_rows": feasibility.shortfall_rows,
        "budget_feasibility_gate": feasibility.gate_pass,
        "coarsening_required": feasibility.status
        is BudgetFeasibilityStatus.FITS_AFTER_INTERIOR_COARSENING,
        "over_category_caps": over_category_caps,
        "representation_gate_pass": bool(atlas_gate.get("representation_gate_pass", False)),
    }
    atomic_write_json(stage / "budget_report.json", _strict_json(report))
    exploratory = config["preset"] in {"smoke", "pilot"}
    scientific_gate = bool(
        feasibility.gate_pass
        and not over_category_caps
        and atlas_gate.get("representation_gate_pass", False)
    )
    return _gate(
        stage / "gate.json",
        {
            "budget_arithmetic_registered": int(category_rows["configured_max_rows"].sum())
            == int(budget["total_rows"]),
            "formal_fixed_budget_feasible_or_exploratory_only": exploratory
            or feasibility.gate_pass,
            "formal_category_caps_or_exploratory_only": exploratory
            or not over_category_caps,
            "formal_representation_or_exploratory_only": exploratory
            or bool(atlas_gate.get("representation_gate_pass", False)),
        },
        semantics=report["semantics"],
        exploratory_only=exploratory and not scientific_gate,
        scientific_gate_pass=scientific_gate,
        budget_feasibility_gate=feasibility.gate_pass,
        representation_gate_pass=bool(atlas_gate.get("representation_gate_pass", False)),
        N_L=n_labelable,
        N_M=n_multichart,
        N_H=n_high,
        N_min=demand.minimum_supervised_rows,
        N_min_coarsened=demand.coarsened_minimum_rows,
        budget_status=feasibility.status.value,
        over_category_caps=over_category_caps,
    )


def _split_policy(config: Mapping[str, Any]) -> MacroblockSplitPolicy:
    values = config["split"]
    return MacroblockSplitPolicy(
        macroblock_mm=int(values["macroblock_mm"]),
        seed=int(values["seed"]),
        train_core_fraction=float(values["train_core_fraction"]),
        active_probe_fraction=float(values["active_probe_fraction"]),
        validation_fraction=float(values["validation_fraction"]),
        sealed_fraction=float(values["sealed_fraction"]),
    )


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )


def _target_cells(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    xyz = result.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    indices = np.floor(xyz / 0.010).astype(np.int64)
    result["cell_ix"] = indices[:, 0]
    result["cell_iy"] = indices[:, 1]
    result["cell_iz"] = indices[:, 2]
    return result


def _ranked_source_rows(
    frame: pd.DataFrame, *, identity_column: str, seed: int
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_rank"] = ranked[identity_column].astype(str).map(
        lambda value: hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).hexdigest()
    )
    return ranked.sort_values(["_rank", identity_column], kind="stable").drop(
        columns="_rank"
    )


def _dense_target_groups(
    capability: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    excluded_point_ids: set[str],
    seed: int,
) -> tuple[tuple[str, SupervisionPriority, pd.DataFrame], ...]:
    labelable = classification[
        classification["domain_class"].isin(
            ("resolved_single_under_budget", "resolved_multichart")
        )
    ][
        [
            "cell_ix", "cell_iy", "cell_iz", "domain_class", "strata",
            "empirical_labelable_fraction",
        ]
    ].copy()
    targets = capability.copy()
    if "physical_point_id" not in targets:
        targets["physical_point_id"] = targets.index.map(
            lambda value: f"cap_dense_{int(value):09d}"
        )
    targets = _target_cells(targets)
    targets = targets.merge(
        labelable,
        on=["cell_ix", "cell_iy", "cell_iz"],
        how="inner",
        validate="many_to_one",
    )
    targets = targets[
        ~targets["physical_point_id"].astype(str).isin(excluded_point_ids)
    ].drop_duplicates("physical_point_id", keep="first")
    difficult_words = {"boundary", "tip", "low_density", "ill_conditioned"}
    is_router = targets["domain_class"].eq("resolved_multichart")
    is_refine = np.asarray(
        [
            bool(difficult_words & set(str(value).split("|")))
            for value in targets["strata"]
        ]
    ) | targets["empirical_labelable_fraction"].lt(1.0 - 1.0e-12).to_numpy()
    groups = (
        (
            "chart_boundary",
            SupervisionPriority.CHART_EXPERT,
            targets[is_router],
        ),
        (
            "adaptive_refinement",
            SupervisionPriority.REFINEMENT,
            targets[~is_router & is_refine],
        ),
        (
            "workspace_maximin",
            SupervisionPriority.MAXIMIN,
            targets[~is_router & ~is_refine],
        ),
    )
    return tuple(
        (name, priority, _ranked_source_rows(rows, identity_column="physical_point_id", seed=seed))
        for name, priority, rows in groups
    )


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return data
    order = np.argsort(np.argsort(data, kind="stable"), kind="stable")
    return order.astype(float) / max(1, len(data) - 1)


def _probe_active_selection(
    records: Sequence[Any],
    *,
    representation_mode: RepresentationMode,
    environment: Any,
    config: Mapping[str, Any],
    active_reserve: int,
) -> tuple[tuple[str, ...], pd.DataFrame, Mapping[str, Any]]:
    """Train the one registered probe Student and rank validation-only cells."""

    split = _split_policy(config)
    assigned = tuple(
        record
        if record.split_role is not None
        else replace(
            record,
            macroblock_id=split.assignment_for_cell(record.cell).macroblock_id,
            split_role=split.assignment_for_cell(record.cell).split_role,
        )
        for record in records
    )
    active_pool = tuple(
        row for row in assigned if row.split_role is SplitRole.ACTIVE_PROBE
    )
    if not active_pool or active_reserve <= 0:
        return (), pd.DataFrame(), {"status": "no_active_pool", "trained": False}

    if representation_mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        probe_mode = representation_mode
        eligible = tuple(row for row in assigned if row.kind is SupervisionKind.STATEFUL)
    else:
        # The active acquisition model is deliberately a global primary
        # baseline.  It has a stable validation contract even when some local
        # experts have no rows in a small Pilot validation split.
        probe_mode = RepresentationMode.XYZ_GLOBAL
        eligible = tuple(
            row
            for row in assigned
            if row.kind is SupervisionKind.STATIC and row.is_primary
        )
        active_pool = tuple(
            row
            for row in active_pool
            if row.kind is SupervisionKind.STATIC and row.is_primary
        )
    train = tuple(row for row in eligible if row.split_role is SplitRole.TRAIN_CORE)
    validation = tuple(row for row in eligible if row.split_role is SplitRole.VALIDATION)
    if not train or not validation or not active_pool:
        return (), pd.DataFrame(), {
            "status": "insufficient_probe_train_validation_or_active_rows",
            "trained": False,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "active_pool_rows": len(active_pool),
        }
    train_frame = records_to_workspace_student_frame(
        train, jacobian_at_beta=environment.jacobian
    )
    validation_frame = records_to_workspace_student_frame(
        validation, jacobian_at_beta=environment.jacobian
    )
    student = config["student"]
    steps = int(student.get("active_probe_max_steps", min(50, student["max_epochs"])))
    trained = train_workspace_student(
        train_frame,
        validation_frame,
        mode=probe_mode,
        geometry=_student_geometry(environment),
        config=WorkspaceStudentTrainingConfig(
            hidden_units=tuple(map(int, student["hidden_units"])),
            learning_rate=float(student["learning_rate"]),
            max_steps=max(1, steps),
            validation_interval=max(1, min(10, steps)),
            patience_intervals=max(1, int(student["patience"])),
            seed=int(student["seeds"][0]),
            loss_weights=WorkspaceStudentLossWeights(
                beta=float(student["lambda_beta"]),
                fk=float(student["lambda_fk"]),
                row_space=float(student["lambda_row"]),
            ),
        ),
    )
    xyz = np.vstack([row.xyz_m for row in active_pool])
    previous = (
        np.vstack([row.previous_beta_rad for row in active_pool])
        if probe_mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS
        else None
    )
    prediction = trained.inverse.predict(
        InverseQuery(xyz_m=xyz, previous_beta_rad=previous)
    )
    residual = np.asarray(prediction.fk_residual_mm, dtype=float)
    residual[~np.isfinite(residual)] = np.nanmax(
        residual[np.isfinite(residual)], initial=0.0
    ) + 1.0
    train_xyz = np.vstack([row.xyz_m for row in train])
    nearest = cKDTree(train_xyz).query(xyz, k=1)[0] * 1000.0
    beta_true = np.vstack([row.beta_rad for row in active_pool])
    beta_pred = np.asarray(prediction.beta_rad, dtype=float)
    beta_gap = np.sqrt(np.mean(np.square(np.rad2deg(beta_pred - beta_true)), axis=1))
    score = (
        _normalized_rank(residual)
        + _normalized_rank(nearest)
        + _normalized_rank(beta_gap)
    )
    selection = pd.DataFrame(
        {
            "record_id": [row.record_id for row in active_pool],
            "probe_fk_residual_mm": residual,
            "nearest_train_distance_mm": nearest,
            "probe_beta_rms_deg": beta_gap,
            "router_entropy": np.nan,
            "ensemble_disagreement_deg": np.nan,
            "active_score": score,
        }
    ).sort_values(["active_score", "record_id"], ascending=[False, True], kind="stable")
    chosen = tuple(selection.head(min(int(active_reserve), len(selection)))["record_id"])
    selection["selected"] = selection["record_id"].isin(chosen)
    report = {
        "status": "probe_student_scored_once",
        "trained": True,
        "probe_mode": probe_mode.value,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "active_pool_rows": len(active_pool),
        "selected_active_rows": len(chosen),
        "selection_inputs": [
            "validation_fk_error",
            "nearest_train_distance",
            "beta_disagreement",
        ],
        "router_entropy_available": False,
        "ensemble_disagreement_available": False,
    }
    return chosen, selection.reset_index(drop=True), report


def _empty_dataset_frames(stage: Path, *, reason: str) -> None:
    columns = [
        "record_id", "kind", "physical_point_id", "chart_id", "cell_level_mm",
        "cell_ix", "cell_iy", "cell_iz", "is_primary", "required", "priority",
        "source_family", "sample_weight", "quality_class", "residual_mm",
        "actual_bounds", "macroblock_id", "split_role", *XYZ_COLUMNS, *BETA_COLUMNS,
    ]
    empty = pd.DataFrame(columns=columns)
    for name in (
        "supervision_candidates.parquet",
        "primary_canonical.parquet",
        "chart_expert.parquet",
        "stateful_transition.parquet",
        "student_supervision.parquet",
        "active_selection.parquet",
    ):
        _atomic_parquet(empty, stage / name)
    atomic_write_json(stage / "skip.json", {"reason": reason})


def chart_training_support(frame: pd.DataFrame) -> dict[str, Any]:
    """Report whether every supervised expert chart has train-core support."""

    required = {"chart_id", "split_role"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"chart support frame missing columns: {missing}")
    chart_ids = sorted(frame["chart_id"].dropna().astype(str).unique())
    train_chart_ids = sorted(
        frame.loc[
            frame["split_role"].eq(SplitRole.TRAIN_CORE.value), "chart_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    unsupported = sorted(set(chart_ids) - set(train_chart_ids))
    role_counts = (
        frame.groupby(["chart_id", "split_role"], sort=True)
        .size()
        .rename("row_count")
        .reset_index()
        .to_dict("records")
    )
    return {
        "chart_count": len(chart_ids),
        "train_supported_chart_count": len(train_chart_ids),
        "unsupported_chart_ids": unsupported,
        "role_counts": role_counts,
        "gate_pass": bool(chart_ids and not unsupported),
    }


def stage_dataset(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    budget_gate = _require_stage(output_root, "budget_allocation")
    stage = output_root / STAGE_DIRS["dataset"]
    stage.mkdir(parents=True, exist_ok=False)
    atlas_stage = output_root / STAGE_DIRS["workspace_atlas"]
    atlas_gate = _read_json(atlas_stage / "gate.json")
    mode = RepresentationMode(str(atlas_gate.get("representation_mode", "blocked")))
    exploratory = config["preset"] in {"smoke", "pilot"}
    if mode is RepresentationMode.BLOCKED:
        _empty_dataset_frames(stage, reason="representation_gate_blocked")
        report = {
            "semantics": "no_supervision_materialized_when_representation_is_blocked",
            "representation_mode": mode.value,
            "materialization_authorized": False,
            "target_rows": int(config["budget"]["pilot_rows"]),
            "final_rows": 0,
            "target_reached": False,
            "padding_rows": 0,
        }
        atomic_write_json(stage / "dataset_report.json", report)
        return _gate(
            stage / "gate.json",
            {
                "blocked_representation_produces_no_conflicting_rows": True,
                "formal_representation_required": exploratory,
            },
            **report,
            exploratory_only=exploratory,
            scientific_gate_pass=False,
        )

    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    task_probes = pd.read_parquet(atlas_stage / "task_probe_labels.parquet")
    candidates = pd.read_parquet(atlas_stage / "candidate_clusters.parquet")
    partition = pd.read_parquet(atlas_stage / "primary_chart_partition.parquet")
    product_edges = pd.read_parquet(atlas_stage / "product_edges.parquet")
    base_records = supervision_records_from_atlas_frames(
        task_probes=task_probes,
        candidates=candidates,
        primary_partition=partition,
        product_edges=product_edges,
    )
    target_rows = int(config["budget"]["pilot_rows"])
    active_reserve = min(int(config["budget"]["active_rows"]), max(0, target_rows - 1))
    all_records: list[Any] = list(base_records)
    correction_audits: list[pd.DataFrame] = []
    if mode is not RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        classification = pd.read_parquet(atlas_stage / "domain_classification.parquet")
        capability = _load_capability_slab(
            paths["capability_pool"],
            _grid(config),
            row_limit=config["registry"].get("capability_row_limit"),
        )
        capability["physical_point_id"] = capability.index.map(
            lambda value: f"cap_dense_{int(value):09d}"
        )
        excluded = {row.physical_point_id for row in base_records}
        category_caps = {
            "chart_boundary": int(config["budget"]["router_boundary_rows_max"]),
            "adaptive_refinement": int(config["budget"]["refine_rows_max"]),
            "workspace_maximin": max(
                0,
                target_rows
                - int(config["budget"]["retention_rows_max"])
                - active_reserve,
            ),
        }
        for family, priority, targets in _dense_target_groups(
            capability,
            classification,
            excluded_point_ids=excluded,
            seed=int(config["registry"]["pilot_seed"]),
        ):
            corrected = correct_static_targets_from_primary_sections(
                targets,
                task_probes=task_probes,
                candidates=candidates,
                primary_partition=partition,
                environment=environment,
                source_family=family,
                priority=priority,
                maximum_rows=min(len(targets), category_caps[family]),
                parent_gap_max_deg=float(config["atlas"]["split_gap_deg"]),
                policy=_atlas_frame_policy(config),
            )
            all_records.extend(corrected.records)
            if len(corrected.audit):
                correction_audits.append(corrected.audit.assign(source_family=family))
            excluded.update(row.physical_point_id for row in corrected.records)

        retention = pd.read_parquet(
            paths["v13_retention"],
            columns=["sample_id", *XYZ_COLUMNS],
        )
        low, high = _grid(config).x_slab_m
        retention = retention[retention["x_m"].between(low, high, inclusive="both")].copy()
        retention["physical_point_id"] = retention["sample_id"].map(
            lambda value: f"v13_retention_{value}"
        )
        retention = _ranked_source_rows(
            retention,
            identity_column="physical_point_id",
            seed=int(config["split"]["seed"]),
        )
        corrected_retention = correct_static_targets_from_primary_sections(
            retention,
            task_probes=task_probes,
            candidates=candidates,
            primary_partition=partition,
            environment=environment,
            source_family="v13_retention",
            priority=SupervisionPriority.RETENTION,
            maximum_rows=int(config["budget"]["retention_rows_max"]),
            parent_gap_max_deg=float(config["atlas"]["split_gap_deg"]),
            policy=_atlas_frame_policy(config),
        )
        all_records.extend(corrected_retention.records)
        if len(corrected_retention.audit):
            correction_audits.append(
                corrected_retention.audit.assign(source_family="v13_retention")
            )

    _atomic_parquet(
        supervision_records_frame(all_records), stage / "supervision_candidates.parquet"
    )
    correction_audit = (
        pd.concat(correction_audits, ignore_index=True)
        if correction_audits
        else pd.DataFrame(
            columns=[
                "physical_point_id", "status", "successful_parent_count",
                "parent_gap_max_deg", "source_family",
            ]
        )
    )
    _atomic_parquet(correction_audit, stage / "dense_correction_audit.parquet")
    hard_max = max(target_rows, min(int(config["budget"]["hard_max_rows"]), target_rows))
    materializer = SupervisionMaterializer(
        SupervisionBudgetPolicy(
            target_total=target_rows,
            hard_max=hard_max,
            active_reserve=active_reserve,
        ),
        _split_policy(config),
    )
    preactive = materializer.materialize(
        all_records, representation_mode=mode
    )
    active_ids, active_selection, active_report = _probe_active_selection(
        all_records,
        representation_mode=mode,
        environment=environment,
        config=config,
        active_reserve=active_reserve,
    )
    final = materializer.materialize(
        all_records,
        representation_mode=mode,
        active_record_ids=active_ids,
    )
    _atomic_parquet(active_selection, stage / "active_selection.parquet")
    atomic_write_json(stage / "active_refinement_report.json", _strict_json(active_report))
    for name, rows in (
        ("primary_canonical.parquet", final.primary_canonical),
        ("chart_expert.parquet", final.chart_expert),
        ("stateful_transition.parquet", final.stateful_transition),
        ("student_supervision.parquet", final.supervision_records),
    ):
        _atomic_parquet(supervision_records_frame(rows), stage / name)
    selected_contract = (
        final.stateful_transition
        if mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS
        else final.chart_expert
        if mode is RepresentationMode.XYZ_ROUTER_EXPERTS
        else final.primary_canonical
    )
    split_support = chart_training_support(
        supervision_records_frame(selected_contract)
    )
    representation_split_support = bool(
        mode is RepresentationMode.XYZ_GLOBAL or split_support["gate_pass"]
    )
    _atomic_parquet(final.supervision_index, stage / "supervision_index.parquet")
    _atomic_parquet(preactive.supervision_index, stage / "preactive_index.parquet")
    split_counts = {
        str(key): int(value)
        for key, value in final.supervision_index["split_role"].value_counts().items()
    }
    source_counts = {
        str(key): int(value)
        for key, value in final.supervision_index["source_family"].value_counts().items()
    }
    final_rows = final.budget_report.unique_supervision_count
    target_reached = final_rows == target_rows
    report = {
        "semantics": "primary_and_chart_expert_contracts_with_all_supervised_rows_budgeted",
        "representation_mode": mode.value,
        "materialization_authorized": True,
        "candidate_supervision_rows": len(all_records),
        "preactive_rows": preactive.budget_report.unique_supervision_count,
        "active_selected_rows": len(active_ids),
        "final_rows": final_rows,
        "target_rows": target_rows,
        "hard_max_rows": hard_max,
        "target_reached": target_reached,
        "padding_rows": final.budget_report.padding_count,
        "primary_rows": len(final.primary_canonical),
        "chart_expert_rows": len(final.chart_expert),
        "stateful_transition_rows": len(final.stateful_transition),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "active_refinement": active_report,
        "chart_training_support": split_support,
        "representation_split_support_gate": representation_split_support,
        "budget_feasibility_gate": bool(
            budget_gate.get("budget_feasibility_gate", False)
        ),
    }
    atomic_write_json(stage / "dataset_report.json", _strict_json(report))
    scientific_gate = bool(
        target_reached
        and final.budget_report.padding_count == 0
        and representation_split_support
        and budget_gate.get("scientific_gate_pass", False)
    )
    return _gate(
        stage / "gate.json",
        {
            "no_padding": final.budget_report.padding_count == 0,
            "within_hard_cap": final_rows <= hard_max,
            "primary_contract_single_valued": len(
                {row.physical_point_id for row in final.primary_canonical}
            )
            == len(final.primary_canonical),
            "formal_all_expert_charts_have_train_support_or_exploratory_only":
            exploratory or representation_split_support,
            "formal_exact_target_or_exploratory_only": exploratory or target_reached,
        },
        **report,
        exploratory_only=exploratory and not scientific_gate,
        scientific_gate_pass=scientific_gate,
    )


def _student_frame_from_dataset(frame: pd.DataFrame, environment: Any) -> pd.DataFrame:
    if len(frame) == 0:
        return frame.copy()
    result = frame.copy()
    jacobian = np.vstack(
        [
            np.asarray(
                environment.jacobian(
                    row.loc[list(BETA_COLUMNS)].to_numpy(dtype=float)
                ),
                dtype=float,
            ).reshape(1, 18)
            for _index, row in result.iterrows()
        ]
    )
    for index, name in enumerate(JACOBIAN_COLUMNS):
        result[name] = jacobian[:, index]
    return result


def _metric_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _evaluate_trained_inverse(
    trained: Any,
    frame: pd.DataFrame,
    *,
    mode: RepresentationMode,
    environment: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    previous = None
    if mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        previous = frame.loc[
            :, [f"previous_beta{index}_rad" for index in range(1, 7)]
        ].to_numpy(dtype=float)
    prediction = trained.inverse.predict(
        InverseQuery(xyz_m=xyz, previous_beta_rad=previous)
    )
    true_beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    beta_gap = np.full(len(frame), np.nan, dtype=float)
    beta_gap[prediction.accepted] = np.sqrt(
        np.mean(
            np.square(
                np.rad2deg(
                    prediction.beta_rad[prediction.accepted]
                    - true_beta[prediction.accepted]
                )
            ),
            axis=1,
        )
    )
    dls_metrics: dict[str, Any] = {}
    for steps in (1, 2):
        residual = np.full(len(frame), np.nan, dtype=float)
        accepted_indices = np.flatnonzero(prediction.accepted)
        if len(accepted_indices):
            refined = refine_inverse_with_bounded_dls(
                environment,
                xyz[accepted_indices],
                prediction.beta_rad[accepted_indices],
                steps=steps,
                acceptance_residual_mm=float(
                    10.0
                ),
            )
            residual[accepted_indices] = refined.residual_mm
        dls_metrics[f"dls_{steps}_step_fk_mm"] = _metric_summary(residual)
    predicted_charts = np.asarray(prediction.chart_ids, dtype=object)
    target_charts = frame["chart_id"].astype(str).to_numpy()
    accepted = np.asarray(prediction.accepted, dtype=bool)
    router_accuracy = (
        float(np.mean(predicted_charts[accepted] == target_charts[accepted]))
        if np.any(accepted) and mode is not RepresentationMode.XYZ_GLOBAL
        else None
    )
    rows = frame[
        ["record_id", "split_role", "chart_id", *XYZ_COLUMNS, *BETA_COLUMNS]
    ].copy()
    rows["accepted"] = accepted
    rows["reason"] = list(prediction.reasons)
    rows["predicted_chart_id"] = list(prediction.chart_ids)
    rows["fk_residual_mm"] = prediction.fk_residual_mm
    rows["beta_rms_deg"] = beta_gap
    for index, name in enumerate(BETA_COLUMNS, start=1):
        rows[f"predicted_beta{index}_rad"] = prediction.beta_rad[:, index - 1]
    metrics = {
        "row_count": len(frame),
        "accepted_count": int(accepted.sum()),
        "abstention_rate": float(1.0 - accepted.mean()) if len(accepted) else 1.0,
        "fk_mm": _metric_summary(prediction.fk_residual_mm[accepted]),
        "beta_rms_deg": _metric_summary(beta_gap),
        "router_top1_accuracy_on_accepted": router_accuracy,
        "bounds_violation_count": int(
            np.count_nonzero(
                accepted
                & (
                    np.any(
                        prediction.beta_rad
                        < np.asarray(environment.bounds)[:, 0] - 1.0e-12,
                        axis=1,
                    )
                    | np.any(
                        prediction.beta_rad
                        > np.asarray(environment.bounds)[:, 1] + 1.0e-12,
                        axis=1,
                    )
                )
            )
        ),
        **dls_metrics,
    }
    return metrics, rows


def _training_config(config: Mapping[str, Any], seed: int) -> WorkspaceStudentTrainingConfig:
    values = config["student"]
    return WorkspaceStudentTrainingConfig(
        hidden_units=tuple(map(int, values["hidden_units"])),
        router_hidden_units=tuple(map(int, values["router_hidden_units"])),
        learning_rate=float(values["learning_rate"]),
        max_steps=int(values["max_epochs"]),
        validation_interval=max(1, min(20, int(values["max_epochs"]))),
        patience_intervals=max(1, int(values["patience"])),
        seed=int(seed),
        loss_weights=WorkspaceStudentLossWeights(
            beta=float(values["lambda_beta"]),
            fk=float(values["lambda_fk"]),
            row_space=float(values["lambda_row"]),
        ),
    )


def _train_student_variant(
    *,
    variant_id: str,
    mode: RepresentationMode,
    frame: pd.DataFrame,
    environment: Any,
    config: Mapping[str, Any],
    stage: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    train = frame[
        frame["split_role"].isin(
            (SplitRole.TRAIN_CORE.value, SplitRole.ACTIVE_PROBE.value)
        )
    ].reset_index(drop=True)
    validation = frame[
        frame["split_role"].eq(SplitRole.VALIDATION.value)
    ].reset_index(drop=True)
    if not len(train) or not len(validation):
        raise ValueError(f"{variant_id} has no train or validation rows")
    reports: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for seed in map(int, config["student"]["seeds"]):
        result = train_workspace_student(
            train,
            validation,
            mode=mode,
            geometry=_student_geometry(environment),
            config=_training_config(config, seed),
        )
        model_dir = stage / "models" / variant_id / f"seed_{seed}"
        manifest = save_workspace_student_models(result.models, model_dir)
        history_path = stage / f"history_{variant_id}_seed_{seed}.parquet"
        _atomic_parquet(result.history, history_path)
        metrics, prediction = _evaluate_trained_inverse(
            result, validation, mode=mode, environment=environment
        )
        prediction["variant_id"] = variant_id
        prediction["seed"] = seed
        predictions.append(prediction)
        artifact_hashes = {
            str(path.relative_to(model_dir)): sha256_file(path)
            for path in sorted(model_dir.iterdir())
            if path.is_file()
        }
        reports.append(
            {
                "variant_id": variant_id,
                "representation_mode": mode.value,
                "seed": seed,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "metrics": metrics,
                "model_manifest": manifest,
                "artifact_sha256": artifact_hashes,
            }
        )
    return reports, predictions


def stage_student(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    dataset_gate = _require_stage(output_root, "dataset")
    stage = output_root / STAGE_DIRS["student"]
    stage.mkdir(parents=True, exist_ok=False)
    mode = RepresentationMode(str(dataset_gate.get("representation_mode", "blocked")))
    exploratory = config["preset"] in {"smoke", "pilot"}
    if mode is RepresentationMode.BLOCKED or not dataset_gate.get(
        "materialization_authorized", False
    ):
        report = {
            "semantics": "student_not_trained_without_authorized_representation_and_dataset",
            "representation_mode": mode.value,
            "training_authorized": False,
            "variants": [],
        }
        atomic_write_json(stage / "student_report.json", report)
        _atomic_parquet(pd.DataFrame(), stage / "validation_predictions.parquet")
        return _gate(
            stage / "gate.json",
            {"formal_training_authorization_required": exploratory},
            **report,
            exploratory_only=exploratory,
            scientific_gate_pass=False,
        )
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    dataset_stage = output_root / STAGE_DIRS["dataset"]
    variants: list[tuple[str, RepresentationMode, Path]] = []
    failures: list[dict[str, str]] = []
    primary_path = dataset_stage / "primary_canonical.parquet"
    if primary_path.is_file() and len(pd.read_parquet(primary_path, columns=["record_id"])):
        variants.append(("global_baseline", RepresentationMode.XYZ_GLOBAL, primary_path))
    selected_path = (
        dataset_stage / "stateful_transition.parquet"
        if mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS
        else dataset_stage / "chart_expert.parquet"
        if mode is RepresentationMode.XYZ_ROUTER_EXPERTS
        else primary_path
    )
    selected_already_global = (
        mode is RepresentationMode.XYZ_GLOBAL
        and any(name == "global_baseline" for name, _mode, _path in variants)
    )
    split_support = bool(
        dataset_gate.get("representation_split_support_gate", False)
    )
    if not selected_already_global and mode is not RepresentationMode.XYZ_GLOBAL and not split_support:
        failures.append(
            {
                "variant_id": "selected_representation",
                "reason": "dataset_chart_training_support_gate_failed",
            }
        )
    elif not selected_already_global:
        variants.append(("selected_representation", mode, selected_path))
    reports: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for variant_id, variant_mode, path in variants:
        frame = pd.read_parquet(path)
        if not len(frame):
            failures.append({"variant_id": variant_id, "reason": "empty_supervision"})
            continue
        try:
            student_frame = _student_frame_from_dataset(frame, environment)
            variant_reports, variant_predictions = _train_student_variant(
                variant_id=variant_id,
                mode=variant_mode,
                frame=student_frame,
                environment=environment,
                config=config,
                stage=stage,
            )
        except Exception as error:
            if not exploratory:
                raise
            failures.append(
                {
                    "variant_id": variant_id,
                    "reason": f"{type(error).__name__}:{error}",
                }
            )
            continue
        reports.extend(variant_reports)
        predictions.extend(variant_predictions)
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame()
    )
    _atomic_parquet(prediction_frame, stage / "validation_predictions.parquet")
    p95_values = [
        row["metrics"]["fk_mm"]["p95"]
        for row in reports
        if row["metrics"]["fk_mm"]["p95"] is not None
    ]
    maximum_values = [
        row["metrics"]["fk_mm"]["max"]
        for row in reports
        if row["metrics"]["fk_mm"]["max"] is not None
    ]
    validation_gate = bool(
        reports
        and p95_values
        and maximum_values
        and max(p95_values) <= float(config["student"]["validation_fk_p95_max_mm"])
        and max(maximum_values) <= float(config["student"]["validation_fk_max_mm"])
    )
    report = {
        "semantics": "global_baseline_and_gate_selected_internal_router_student",
        "representation_mode": mode.value,
        "training_authorized": True,
        "variants": reports,
        "variant_failures": failures,
        "validation_gate": validation_gate,
        "known_chart_id_inference_input": False,
        "static_fk_branch_tiebreak_allowed_only_within_stitchable_gap": True,
        "one_and_two_step_dls_reported": True,
    }
    atomic_write_json(stage / "student_report.json", _strict_json(report))
    scientific_gate = bool(
        validation_gate
        and not failures
        and dataset_gate.get("scientific_gate_pass", False)
    )
    return _gate(
        stage / "gate.json",
        {
            "at_least_one_variant_trained": bool(reports),
            "no_external_known_chart_input": True,
            "formal_all_registered_variants_trained": exploratory or not failures,
            "formal_validation_thresholds_or_exploratory_only": exploratory
            or validation_gate,
        },
        semantics=report["semantics"],
        exploratory_only=exploratory and not scientific_gate,
        scientific_gate_pass=scientific_gate,
        representation_mode=mode.value,
        validation_gate=validation_gate,
        trained_variant_count=len(reports),
        variant_failures=failures,
    )


def _registered_inverse_domain(classification: pd.DataFrame) -> WorkspaceRegistry:
    allowed = frozenset(
        CellKey(10, int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in classification[
            classification["domain_class"].isin(
                ("resolved_single_under_budget", "resolved_multichart")
            )
        ].itertuples(index=False)
    )
    return WorkspaceRegistry(grid=WorkspaceGridSpec(levels_mm=(20, 10, 5)), allowed_cells=allowed)


def _evaluation_dataset_path(
    dataset_stage: Path, *, variant_id: str, mode: RepresentationMode
) -> Path:
    if variant_id == "global_baseline" or mode is RepresentationMode.XYZ_GLOBAL:
        return dataset_stage / "primary_canonical.parquet"
    if mode is RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        return dataset_stage / "stateful_transition.parquet"
    return dataset_stage / "chart_expert.parquet"


def _per_group_metrics(predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    values = predictions.merge(
        frame[["record_id", "cell_ix", "source_family"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    values["x_bin"] = np.floor(
        (values["x_m"] - 1.015498) / 0.010
    ).astype(int)
    rows: list[dict[str, Any]] = []
    for dimension in ("x_bin", "chart_id", "source_family"):
        for group_id, group in values.groupby(dimension, sort=True):
            accepted = group[group["accepted"].astype(bool)]
            fk = _metric_summary(accepted["fk_residual_mm"].to_numpy(dtype=float))
            rows.append(
                {
                    "dimension": dimension,
                    "group_id": str(group_id),
                    "row_count": len(group),
                    "accepted_count": len(accepted),
                    "abstention_rate": float(1.0 - len(accepted) / len(group)),
                    "fk_p50_mm": fk["p50"],
                    "fk_p95_mm": fk["p95"],
                    "fk_max_mm": fk["max"],
                }
            )
    return pd.DataFrame.from_records(rows)


def stage_spatial_evaluation(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    student_gate = _require_stage(output_root, "student")
    stage = output_root / STAGE_DIRS["spatial_evaluation"]
    stage.mkdir(parents=True, exist_ok=False)
    exploratory = config["preset"] in {"smoke", "pilot"}
    if not student_gate.get("trained_variant_count", 0):
        report = {
            "semantics": "spatial_evaluation_skipped_without_locked_student",
            "evaluation_authorized": False,
            "models": [],
        }
        atomic_write_json(stage / "spatial_report.json", report)
        _atomic_parquet(pd.DataFrame(), stage / "sealed_predictions.parquet")
        _atomic_parquet(pd.DataFrame(), stage / "per_group_metrics.parquet")
        return _gate(
            stage / "gate.json",
            {"formal_locked_student_required": exploratory},
            **report,
            exploratory_only=exploratory,
            scientific_gate_pass=False,
        )
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    geometry = _student_geometry(environment)
    student_stage = output_root / STAGE_DIRS["student"]
    dataset_stage = output_root / STAGE_DIRS["dataset"]
    atlas_stage = output_root / STAGE_DIRS["workspace_atlas"]
    student_report = _read_json(student_stage / "student_report.json")
    classification = pd.read_parquet(atlas_stage / "domain_classification.parquet")
    registry = _registered_inverse_domain(classification)
    all_predictions: list[pd.DataFrame] = []
    all_groups: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for item in student_report["variants"]:
        variant_id = str(item["variant_id"])
        seed = int(item["seed"])
        mode = RepresentationMode(str(item["representation_mode"]))
        model_dir = student_stage / "models" / variant_id / f"seed_{seed}"
        models = load_workspace_student_models(model_dir)
        inverse = build_workspace_student_inverse(
            models, geometry=geometry, registry=registry
        )
        source_frame = pd.read_parquet(
            _evaluation_dataset_path(dataset_stage, variant_id=variant_id, mode=mode)
        )
        sealed = source_frame[
            source_frame["split_role"].eq(SplitRole.SEALED.value)
        ].reset_index(drop=True)
        if not len(sealed):
            reports.append(
                {
                    "variant_id": variant_id,
                    "seed": seed,
                    "mode": mode.value,
                    "status": "missing_sealed_rows",
                }
            )
            continue
        metrics, predictions = _evaluate_trained_inverse(
            SimpleNamespace(inverse=inverse),
            sealed,
            mode=mode,
            environment=environment,
        )
        predictions["variant_id"] = variant_id
        predictions["seed"] = seed
        all_predictions.append(predictions)
        groups = _per_group_metrics(predictions, sealed)
        groups["variant_id"] = variant_id
        groups["seed"] = seed
        all_groups.append(groups)
        train = source_frame[
            source_frame["split_role"].isin(
                (SplitRole.TRAIN_CORE.value, SplitRole.ACTIVE_PROBE.value)
            )
        ]
        distances = (
            cKDTree(train.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)).query(
                sealed.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), k=1
            )[0]
            * 1000.0
        )
        router_metrics: dict[str, Any] = {
            "entropy": None,
            "top1_accuracy": None,
            "top2_accuracy": None,
        }
        if mode is not RepresentationMode.XYZ_GLOBAL:
            features = sealed.loc[:, student_feature_columns(mode)].to_numpy(
                dtype=np.float32
            )
            probability = np.asarray(models.router_model(features, training=False), dtype=float)
            entropy = -np.sum(probability * np.log(np.maximum(probability, 1.0e-12)), axis=1)
            target_index = np.asarray(
                [models.chart_ids.index(str(value)) for value in sealed["chart_id"]],
                dtype=int,
            )
            order = np.argsort(-probability, axis=1)
            router_metrics = {
                "entropy": _metric_summary(entropy),
                "top1_accuracy": float(np.mean(order[:, 0] == target_index)),
                "top2_accuracy": float(
                    np.mean(
                        [target_index[index] in order[index, :2] for index in range(len(order))]
                    )
                ),
            }
        reports.append(
            {
                "variant_id": variant_id,
                "seed": seed,
                "mode": mode.value,
                "status": "evaluated_locked_model",
                "sealed": metrics,
                "fill_distance_mm": _metric_summary(distances),
                "maximum_hole_radius_proxy_mm": float(np.max(distances)),
                "router": router_metrics,
                "model_manifest_sha256": sha256_file(
                    model_dir / "model_manifest.json"
                ),
            }
        )
    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    groups = (
        pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()
    )
    _atomic_parquet(predictions, stage / "sealed_predictions.parquet")
    _atomic_parquet(groups, stage / "per_group_metrics.parquet")
    evaluated = [row for row in reports if row.get("status") == "evaluated_locked_model"]
    p95_values = [row["sealed"]["fk_mm"]["p95"] for row in evaluated]
    max_values = [row["sealed"]["fk_mm"]["max"] for row in evaluated]
    sealed_gate = bool(
        evaluated
        and all(value is not None for value in p95_values + max_values)
        and max(p95_values) <= float(config["student"]["sealed_fk_p95_max_mm"])
        and max(max_values) <= float(config["student"]["sealed_fk_max_mm"])
    )
    coverage_report = _read_json(atlas_stage / "workspace_atlas_report.json")
    report = {
        "semantics": "locked_models_on_whole_40mm_sealed_macroblocks",
        "evaluation_authorized": True,
        "models": reports,
        "sealed_gate": sealed_gate,
        "domain_class_measure_ratio": coverage_report[
            "domain_class_measure_ratio"
        ],
        "evaluation_rows_count_toward_training_budget": False,
    }
    atomic_write_json(stage / "spatial_report.json", _strict_json(report))
    scientific_gate = bool(
        sealed_gate and student_gate.get("scientific_gate_pass", False)
    )
    return _gate(
        stage / "gate.json",
        {
            "at_least_one_locked_model_evaluated": bool(evaluated),
            "evaluation_rows_not_training_rows": True,
            "formal_sealed_thresholds_or_exploratory_only": exploratory
            or sealed_gate,
        },
        semantics=report["semantics"],
        exploratory_only=exploratory and not scientific_gate,
        scientific_gate_pass=scientific_gate,
        sealed_gate=sealed_gate,
        evaluated_model_count=len(evaluated),
    )


def _v13_regression_trajectories(
    paths: Mapping[str, Path], config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = config["evaluation"]
    ellipse = pd.read_parquet(
        paths["v13_ellipse_points"],
        columns=["seed", "cycle_id", "phase_idx", *XYZ_COLUMNS],
    )
    first_seed = int(ellipse["seed"].min())
    ellipse = ellipse[ellipse["seed"].eq(first_seed)].drop_duplicates(
        ["cycle_id", "phase_idx"]
    )
    ellipse_ids = sorted(ellipse["cycle_id"].unique())[
        : int(evaluation["v13_ellipse_regression_count"])
    ]
    ellipse = ellipse[ellipse["cycle_id"].isin(ellipse_ids)].copy()
    ellipse["family_id"] = ellipse["cycle_id"].map(
        lambda value: f"v13_ellipse_{int(value):03d}"
    )
    ellipse["family_type"] = "v13_local_ellipse_regression"
    path = pd.read_parquet(paths["v13_path_targets"])
    path_ids = sorted(path["path_id"].unique())[
        : int(evaluation["v13_path_regression_count"])
    ]
    path = path[path["path_id"].isin(path_ids)].copy()
    path["family_id"] = path["path_id"].map(
        lambda value: f"v13_path_{int(value):03d}"
    )
    path["family_type"] = "v13_local_3d_path_regression"
    targets = pd.concat(
        [
            ellipse[["family_id", "family_type", "phase_idx", *XYZ_COLUMNS]],
            path[["family_id", "family_type", "phase_idx", *XYZ_COLUMNS]],
        ],
        ignore_index=True,
    ).sort_values(["family_id", "phase_idx"], kind="stable")
    targets["teacher_feasible"] = True
    catalog = (
        targets.groupby(["family_id", "family_type"], sort=True)
        .size()
        .rename("phase_count")
        .reset_index()
    )
    catalog["source"] = "locked_v13_regression_targets"
    catalog["teacher_feasible"] = True
    return catalog, targets.reset_index(drop=True)


def _predict_trajectory_family(
    *,
    inverse: Any,
    mode: RepresentationMode,
    xyz: np.ndarray,
    initial_beta: np.ndarray | None,
) -> Any:
    if mode is not RepresentationMode.STATEFUL_ROUTER_EXPERTS:
        return inverse.predict(InverseQuery(xyz_m=xyz))
    if initial_beta is None:
        raise ValueError("stateful trajectory inference requires one registered initial beta")
    beta_rows = np.full((len(xyz), 6), np.nan, dtype=float)
    accepted = np.zeros(len(xyz), dtype=bool)
    reasons: list[str] = []
    charts: list[str] = []
    residual = np.full(len(xyz), np.nan, dtype=float)
    previous = np.asarray(initial_beta, dtype=float).reshape(1, 6)
    for index, target in enumerate(xyz):
        result = inverse.predict(
            InverseQuery(xyz_m=target.reshape(1, 3), previous_beta_rad=previous)
        )
        beta_rows[index] = result.beta_rad[0]
        accepted[index] = result.accepted[0]
        reasons.append(result.reasons[0])
        charts.append(result.chart_ids[0])
        residual[index] = result.fk_residual_mm[0]
        if result.accepted[0]:
            previous = result.beta_rad[0].reshape(1, 6)
    from quasi_exp.teacher.workspace_inverse import PredictionResult

    return PredictionResult(
        beta_rad=beta_rows,
        accepted=accepted,
        reasons=tuple(reasons),
        chart_ids=tuple(charts),
        fk_residual_mm=residual,
        representation_mode=mode,
    )


def stage_trajectory_evaluation(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    spatial_gate = _require_stage(output_root, "spatial_evaluation")
    stage = output_root / STAGE_DIRS["trajectory_evaluation"]
    stage.mkdir(parents=True, exist_ok=False)
    exploratory = config["preset"] in {"smoke", "pilot"}
    if not spatial_gate.get("evaluated_model_count", 0):
        report = {
            "semantics": "trajectory_evaluation_skipped_without_locked_spatial_model",
            "evaluation_authorized": False,
            "trajectory_family_count": 0,
        }
        atomic_write_json(stage / "trajectory_report.json", report)
        _atomic_parquet(pd.DataFrame(), stage / "trajectory_catalog.parquet")
        _atomic_parquet(pd.DataFrame(), stage / "trajectory_targets.parquet")
        _atomic_parquet(pd.DataFrame(), stage / "trajectory_predictions.parquet")
        _atomic_parquet(pd.DataFrame(), stage / "trajectory_metrics.parquet")
        return _gate(
            stage / "gate.json",
            {"formal_locked_spatial_model_required": exploratory},
            **report,
            exploratory_only=exploratory,
            scientific_gate_pass=False,
        )
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    geometry = _student_geometry(environment)
    atlas_stage = output_root / STAGE_DIRS["workspace_atlas"]
    student_stage = output_root / STAGE_DIRS["student"]
    dataset_stage = output_root / STAGE_DIRS["dataset"]
    classification = pd.read_parquet(atlas_stage / "domain_classification.parquet")
    registry = _registered_inverse_domain(classification)
    capability = _load_capability_slab(
        paths["capability_pool"],
        _grid(config),
        row_limit=config["registry"].get("capability_row_limit"),
    )
    historical = pd.read_parquet(paths["historical_final8_teacher"])
    historical_catalog = pd.read_parquet(paths["historical_final8_catalog"])[
        ["family_id", "major_semiaxis_m"]
    ]
    historical = historical.merge(
        historical_catalog, on="family_id", how="left", validate="many_to_one"
    )
    suite = build_workspace_trajectory_suite(
        capability,
        classification,
        historical,
        phase_count=int(config["evaluation"]["trajectory_phase_count"]),
        projection_max_mm=float(
            config["evaluation"]["capability_projection_max_mm"]
        ),
    )
    regression_catalog, regression_targets = _v13_regression_trajectories(
        paths, config
    )
    catalog = pd.concat([suite.catalog, regression_catalog], ignore_index=True, sort=False)
    targets = pd.concat([suite.targets, regression_targets], ignore_index=True, sort=False)
    # The catalog and targets are sealed before any model is opened.
    _atomic_parquet(catalog, stage / "trajectory_catalog.parquet")
    _atomic_parquet(targets, stage / "trajectory_targets.parquet")
    catalog_hash = sha256_file(stage / "trajectory_catalog.parquet")
    target_hash = sha256_file(stage / "trajectory_targets.parquet")
    atomic_write_json(
        stage / "trajectory_registration.json",
        {
            "catalog_sha256": catalog_hash,
            "targets_sha256": target_hash,
            "core_suite_family_count": 36,
            "evaluation_rows_count_toward_training_budget": False,
        },
    )

    student_report = _read_json(student_stage / "student_report.json")
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for item in student_report["variants"]:
        variant_id = str(item["variant_id"])
        seed = int(item["seed"])
        mode = RepresentationMode(str(item["representation_mode"]))
        model_dir = student_stage / "models" / variant_id / f"seed_{seed}"
        models = load_workspace_student_models(model_dir)
        inverse = build_workspace_student_inverse(
            models, geometry=geometry, registry=registry
        )
        supervision = pd.read_parquet(
            _evaluation_dataset_path(dataset_stage, variant_id=variant_id, mode=mode)
        )
        train_rows = supervision[
            supervision["split_role"].isin(
                (SplitRole.TRAIN_CORE.value, SplitRole.ACTIVE_PROBE.value)
            )
        ]
        train_tree = cKDTree(train_rows.loc[:, XYZ_COLUMNS].to_numpy(dtype=float))
        for catalog_row in catalog.itertuples(index=False):
            family = targets[
                targets["family_id"].astype(str).eq(str(catalog_row.family_id))
            ].sort_values("phase_idx", kind="stable")
            xyz = family.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
            _distance, initial_index = train_tree.query(xyz[0], k=1)
            initial_beta = train_rows.iloc[int(initial_index)].loc[
                list(BETA_COLUMNS)
            ].to_numpy(dtype=float)
            prediction = _predict_trajectory_family(
                inverse=inverse,
                mode=mode,
                xyz=xyz,
                initial_beta=initial_beta,
            )
            in_domain = registry.contains(xyz)
            accepted_in_domain = prediction.accepted & in_domain
            denominator = int(in_domain.sum())
            accepted_fraction = (
                float(accepted_in_domain.sum() / denominator) if denominator else 0.0
            )
            residual = prediction.fk_residual_mm[accepted_in_domain]
            fk = _metric_summary(residual)
            beta_jump = np.full(len(family), np.nan, dtype=float)
            if len(family) > 1:
                valid_pair = prediction.accepted & np.roll(prediction.accepted, 1)
                beta_jump[valid_pair] = np.sqrt(
                    np.mean(
                        np.square(
                            np.rad2deg(
                                prediction.beta_rad[valid_pair]
                                - np.roll(prediction.beta_rad, 1, axis=0)[valid_pair]
                            )
                        ),
                        axis=1,
                    )
                )
            teacher_feasible = bool(catalog_row.teacher_feasible)
            path_pass = bool(
                teacher_feasible
                and denominator > 0
                and accepted_fraction
                >= float(config["evaluation"]["path_minimum_accepted_fraction"])
                and fk["p95"] is not None
                and fk["p95"]
                <= float(config["evaluation"]["path_fk_p95_max_mm"])
                and fk["max"]
                <= float(config["evaluation"]["path_fk_max_mm"])
            )
            dls = {"p50": None, "p95": None, "max": None}
            indices = np.flatnonzero(accepted_in_domain)
            if len(indices):
                refined = refine_inverse_with_bounded_dls(
                    environment,
                    xyz[indices],
                    prediction.beta_rad[indices],
                    steps=2,
                    acceptance_residual_mm=float(
                        config["evaluation"]["path_fk_max_mm"]
                    ),
                )
                dls = _metric_summary(refined.residual_mm)
            metric_rows.append(
                {
                    "variant_id": variant_id,
                    "seed": seed,
                    "family_id": str(catalog_row.family_id),
                    "family_type": str(catalog_row.family_type),
                    "teacher_feasible": teacher_feasible,
                    "point_count": len(family),
                    "in_registered_domain_count": denominator,
                    "accepted_count": int(accepted_in_domain.sum()),
                    "accepted_fraction": accepted_fraction,
                    "fk_p50_mm": fk["p50"],
                    "fk_p95_mm": fk["p95"],
                    "fk_max_mm": fk["max"],
                    "beta_jump_p95_deg": _metric_summary(beta_jump)["p95"],
                    "dls2_fk_p95_mm": dls["p95"],
                    "path_gate_pass": path_pass,
                }
            )
            for local_index, source_row in enumerate(family.itertuples(index=False)):
                prediction_rows.append(
                    {
                        "variant_id": variant_id,
                        "seed": seed,
                        "family_id": str(catalog_row.family_id),
                        "family_type": str(catalog_row.family_type),
                        "phase_idx": int(source_row.phase_idx),
                        "x_m": float(source_row.x_m),
                        "y_m": float(source_row.y_m),
                        "z_m": float(source_row.z_m),
                        "in_registered_domain": bool(in_domain[local_index]),
                        "accepted": bool(prediction.accepted[local_index]),
                        "reason": prediction.reasons[local_index],
                        "predicted_chart_id": prediction.chart_ids[local_index],
                        "fk_residual_mm": float(prediction.fk_residual_mm[local_index]),
                        "beta_jump_deg": float(beta_jump[local_index]),
                    }
                )
    predictions = pd.DataFrame.from_records(prediction_rows)
    metrics = pd.DataFrame.from_records(metric_rows)
    _atomic_parquet(predictions, stage / "trajectory_predictions.parquet")
    _atomic_parquet(metrics, stage / "trajectory_metrics.parquet")
    core = metrics[
        metrics["family_type"].isin(
            (
                "new_ellipse_circle", "lissajous", "bspline_3d_closed",
                "high_x_near_zero_pose", "cross_chart",
            )
        )
        & metrics["teacher_feasible"].astype(bool)
    ]
    core_pass_fraction = (
        float(core["path_gate_pass"].mean()) if len(core) else 0.0
    )
    trajectory_gate = bool(
        len(core)
        and core_pass_fraction
        >= float(config["evaluation"]["minimum_feasible_path_pass_fraction"])
    )
    report = {
        "semantics": "locked_models_on_preregistered_36_family_suite_plus_v13_regressions",
        "evaluation_authorized": True,
        "core_suite_family_count": len(suite.catalog),
        "regression_family_count": len(regression_catalog),
        "trajectory_target_count": len(targets),
        "catalog_sha256": catalog_hash,
        "targets_sha256": target_hash,
        "teacher_feasible_core_model_family_count": len(core),
        "teacher_feasible_core_pass_fraction": core_pass_fraction,
        "trajectory_gate": trajectory_gate,
        "evaluation_rows_count_toward_training_budget": False,
    }
    atomic_write_json(stage / "trajectory_report.json", _strict_json(report))
    scientific_gate = bool(
        trajectory_gate and spatial_gate.get("scientific_gate_pass", False)
    )
    return _gate(
        stage / "gate.json",
        {
            "exact_core_family_inventory": len(suite.catalog) == 36,
            "catalog_sealed_before_model_inference": True,
            "evaluation_rows_not_training_rows": True,
            "formal_trajectory_threshold_or_exploratory_only": exploratory
            or trajectory_gate,
        },
        **report,
        exploratory_only=exploratory and not scientific_gate,
        scientific_gate_pass=scientific_gate,
    )


def _prior_artifact_manifest(output_root: Path) -> dict[str, Any]:
    """Hash every persisted input artifact consumed by the final summary.

    Summary files are intentionally excluded: including a manifest's own bytes
    would make exact-set verification recursive.  The scope is recorded in the
    manifest and the summary gate separately hashes the manifest, Pilot gate,
    and summary report.
    """

    records: list[dict[str, Any]] = []
    for stage_name in STAGE_ORDER[:-1]:
        stage_root = output_root / STAGE_DIRS[stage_name]
        for path in sorted(stage_root.rglob("*")):
            if not path.is_file():
                continue
            records.append(
                {
                    "stage": stage_name,
                    "relative_path": str(path.relative_to(output_root)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return {
        "scope": "all_regular_files_from_protocol_through_trajectory_evaluation",
        "artifact_count": len(records),
        "artifacts": records,
    }


def stage_summary(
    config: Mapping[str, Any], _project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "trajectory_evaluation")
    stage = output_root / STAGE_DIRS["summary"]
    stage.mkdir(parents=True, exist_ok=False)
    gates = {
        name: _read_json(output_root / STAGE_DIRS[name] / "gate.json")
        for name in STAGE_ORDER[:-1]
    }
    source_manifest_path = (
        output_root / STAGE_DIRS["protocol"] / "source_manifest.json"
    )
    source_manifest = _read_json(source_manifest_path)
    artifact_manifest = _prior_artifact_manifest(output_root)
    artifact_manifest_path = stage / "artifact_manifest.json"
    atomic_write_json(artifact_manifest_path, _strict_json(artifact_manifest))

    reach_gate = gates["workspace_proxy"]
    candidate_gate = gates["candidate_bank"]
    atlas_gate = gates["workspace_atlas"]
    budget_gate = gates["budget_allocation"]
    dataset_gate = gates["dataset"]
    student_gate = gates["student"]
    spatial_gate = gates["spatial_evaluation"]
    trajectory_gate = gates["trajectory_evaluation"]
    scientific_checks = {
        "reach_convergence_gate": bool(
            reach_gate.get("reach_convergence_gate", False)
        ),
        "branch_saturation_gate": bool(
            candidate_gate.get("branch_saturation_gate", False)
            and atlas_gate.get("branch_saturation_gate", False)
        ),
        "representation_gate_pass": bool(
            atlas_gate.get("representation_gate_pass", False)
        ),
        "atlas_audit_gate_pass": bool(
            atlas_gate.get("atlas_audit_gate_pass", False)
        ),
        "budget_feasibility_gate": bool(
            budget_gate.get("budget_feasibility_gate", False)
        ),
        "exact_dataset_gate": bool(
            dataset_gate.get("scientific_gate_pass", False)
            and dataset_gate.get("target_reached", False)
        ),
        "student_validation_gate": bool(
            student_gate.get("scientific_gate_pass", False)
            and student_gate.get("validation_gate", False)
        ),
        "sealed_spatial_gate": bool(
            spatial_gate.get("scientific_gate_pass", False)
            and spatial_gate.get("sealed_gate", False)
        ),
        "trajectory_gate": bool(
            trajectory_gate.get("scientific_gate_pass", False)
            and trajectory_gate.get("trajectory_gate", False)
        ),
    }
    all_scientific_checks = all(scientific_checks.values())
    # Smoke thresholds are intentionally cheap wiring thresholds.  Even when
    # every Smoke-local check passes, that run is not scientific evidence for
    # Pilot or Formal authorization.
    scientific_gate = bool(
        config["preset"] in {"pilot", "formal"} and all_scientific_checks
    )
    pilot_gate = _gate(
        stage / "pilot_gate.json",
        {
            "pilot_preset": config["preset"] == "pilot",
            **scientific_checks,
        },
        semantics="pilot_authorization_for_fail_closed_formal_generation",
        protocol_id=config["protocol_id"],
        preset=config["preset"],
        scientific_gate_pass=scientific_gate,
        reach_convergence_gate=scientific_checks["reach_convergence_gate"],
        branch_saturation_gate=scientific_checks["branch_saturation_gate"],
        representation_gate_pass=scientific_checks["representation_gate_pass"],
        representation_mode=atlas_gate.get("representation_mode", "blocked"),
        source_git_sha=source_manifest.get("source_git_sha"),
        source_manifest_sha256=sha256_file(source_manifest_path),
        artifact_manifest_sha256=sha256_file(artifact_manifest_path),
    )
    operational_stage_pass = {
        name: bool(payload.get("gate_pass", False))
        for name, payload in gates.items()
    }
    report = {
        "semantics": "operational_completion_separate_from_scientific_and_formal_authorization",
        "protocol_id": config["protocol_id"],
        "preset": config["preset"],
        "source_git_sha": source_manifest.get("source_git_sha"),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "artifact_count": artifact_manifest["artifact_count"],
        "stage_gate_pass": operational_stage_pass,
        "scientific_checks": scientific_checks,
        "all_preset_checks_pass": all_scientific_checks,
        "scientific_gate_pass": scientific_gate,
        "pilot_gate_pass": bool(pilot_gate["gate_pass"]),
        "formal_authorized_by_this_run": bool(
            config["preset"] == "pilot" and pilot_gate["gate_pass"]
        ),
        "deployment_claim_gate_pass": False,
    }
    summary_report_path = stage / "summary_report.json"
    atomic_write_json(summary_report_path, _strict_json(report))
    exploratory = config["preset"] in {"smoke", "pilot"}
    return _gate(
        stage / "gate.json",
        {
            "all_operational_stages_pass": all(operational_stage_pass.values()),
            "prior_artifact_manifest_nonempty": artifact_manifest["artifact_count"] > 0,
            "formal_scientific_gate_or_exploratory_only": exploratory
            or scientific_gate,
        },
        **report,
        pilot_gate_sha256=sha256_file(stage / "pilot_gate.json"),
        summary_report_sha256=sha256_file(summary_report_path),
    )


STAGE_RUNNERS: dict[str, Callable[[Mapping[str, Any], Path, Path], dict[str, Any]]] = {
    "protocol": stage_protocol,
    "workspace_proxy": stage_workspace_proxy,
    "domain_registry": stage_domain_registry,
    "candidate_bank": stage_candidate_bank,
    "workspace_atlas": stage_workspace_atlas,
    "budget_allocation": stage_budget_allocation,
    "dataset": stage_dataset,
    "student": stage_student,
    "spatial_evaluation": stage_spatial_evaluation,
    "trajectory_evaluation": stage_trajectory_evaluation,
    "summary": stage_summary,
}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config = load_protocol_config(args.config, args.preset)
    project_root = project_root_from(SOURCE_ROOT)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _resolve_project_path(project_root, config["output_root"])
    )
    selected = STAGE_ORDER if args.stage == "all" else (args.stage,)
    reports: dict[str, Any] = {}
    for name in selected:
        reports[name] = STAGE_RUNNERS[name](config, project_root, output_root)
        if not bool(reports[name].get("gate_pass", False)):
            break
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v14_omega200_workspace_atlas.yaml"),
    )
    parser.add_argument("--preset", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--stage", choices=("all", *STAGE_ORDER), default="all")
    parser.add_argument("--output-root")
    return parser


def build_candidate_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BACRA V14 candidate shard worker")
    parser.add_argument("_command")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--worker-mode", choices=("base", "saturation"), required=True)
    parser.add_argument("--task-nodes", required=True)
    parser.add_argument("--seed-bank", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "_candidate-worker":
        return candidate_worker(build_candidate_worker_parser().parse_args(values))
    args = build_parser().parse_args(values)
    reports = run_pipeline(args)
    print(json.dumps(_strict_json(reports), sort_keys=True, indent=2))
    return 0 if reports and all(bool(row.get("gate_pass", False)) for row in reports.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
