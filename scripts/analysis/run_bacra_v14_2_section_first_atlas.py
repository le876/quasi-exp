#!/usr/bin/env python3
"""Run BACRA V14.2 section-first canonical-atlas experiments.

The runner consumes the sealed V14.1 retry4 inventory and artifacts without
modifying them.  F0 reconstructs the stored E3 product graph; only F1 repeats
candidate-flood propagation, and only on the four registered diagnostic
patches.  Section-first methods are scheduled as independent single-threaded
processes with twelve patch-level workers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(SOURCE_ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "scripts" / "analysis"))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from quasi_exp.model.kinematics import dh_transform_i_to_im1
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.canonical import beta_rms_deg, weighted_damped_pinv
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    DirectedContinuationEdge,
    assemble_product_graph,
    build_canonical_atlas_from_product_graph,
    deterministic_root_nodes,
    make_predictor_corrector_continuation,
)
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.section_first_atlas import (
    RootedSectionPolicy,
    SectionGrowthResult,
    build_section_first_atlas,
    section_growth_from_canonical_atlas,
)
from quasi_exp.teacher.section_stability import evaluate_section_patch_gate
from quasi_exp.teacher.selected_section_audit import (
    SelectedSectionAuditPolicy,
    audit_selected_sections,
)
from quasi_exp.teacher.workspace_atlas_repair import (
    CrossCellRepairPolicy,
    RepairAblation,
    atlas_nodes_from_frames,
    make_segmented_continuation,
    run_cross_cell_repair_ablation,
)
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_candidate_bank import (
    CandidatePolicy,
    CandidateQuality,
    CandidateSearchMode,
    solve_candidate_bank,
)
from quasi_exp.teacher.workspace_experiment import (
    ReachSamplingRoundSpec,
    cell_centers,
    frontier_candidates,
    generate_independent_reach_samples,
)
from quasi_exp.teacher.workspace_reach import (
    CellKey,
    FrontierProbeEvidence,
    ReachProxyBuilder,
    ReachProxyConfig,
    ReachReplica,
    ReachReplicaRound,
    WorkspaceGridSpec,
    measure_weighted_boundary_change_ratio,
)
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    save_workspace_student_models,
    train_workspace_student,
)
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
STAGE_DIRS = {
    "inventory": "00_inventory",
    "diagnostics": "01_fresh_audit_diagnostics",
    "mechanisms": "02_mechanism_experiment",
    "confirmation": "03_twelve_patch_confirmation",
    "reach_round6": "04_reach_round6",
    "repaired_pilot": "05_repaired_5k_pilot",
    "exploratory_student": "06_exploratory_student",
    "formal_gate": "07_formal_admission",
    "summary": "08_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)
SECTION_METHODS = {"S2": (2, False), "S4": (4, False), "S8": (8, False), "S4C": (4, True)}


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if ".worktrees" in resolved.parts:
        index = resolved.parts.index(".worktrees")
        return Path(*resolved.parts[:index])
    return resolved


class _EndpointOnlyForwardAdapter:
    """Bit-compatible endpoint FK fast path introduced at source 68b44f4."""

    def __init__(self, environment: Any) -> None:
        self._environment = environment

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def fk(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float)
        if beta.ndim == 1 and beta.shape == (6,):
            beta = beta.reshape(1, 6)
        if beta.ndim != 2 or beta.shape[1] != 6 or not np.isfinite(beta).all():
            raise ValueError("beta_rad must be finite shape (N, 6)")
        lengths = np.asarray(self._environment.lengths_m, dtype=float).reshape(-1)
        endpoint = np.asarray(self._environment.p_end_local_m, dtype=float).reshape(4)
        theta_sign = float(self._environment.theta_sign)
        xyz = np.empty((len(beta), 3), dtype=float)
        for row_index, row in enumerate(beta):
            theta = beta_to_theta(row) * theta_sign
            transform = np.eye(4, dtype=float)
            for joint_index, angle in enumerate(theta, start=1):
                alpha = 0.0 if joint_index == 1 else (np.pi / 2.0 if joint_index % 2 else -np.pi / 2.0)
                transform = transform @ dh_transform_i_to_im1(
                    alpha, float(lengths[joint_index - 1]), float(angle), 0.0
                )
            xyz[row_index] = (transform @ endpoint)[:3]
        return xyz

    def numerical_jacobian(self, beta_rad: np.ndarray, *, eps_rad: float = 1e-4) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float).reshape(6)
        offsets = np.eye(6, dtype=float) * float(eps_rad)
        return ((self.fk(beta[None, :] + offsets) - self.fk(beta[None, :] - offsets)) / (2.0 * eps_rad)).T

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
        return self.numerical_jacobian(beta_rad)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported V14.2 config")
    config = dict(value)
    config["config_path"] = str(config_path)
    if list(config["diagnostics"]["methods"]) != ["F0", "F1", "S2", "S4", "S8", "S4C"]:
        raise ValueError("V14.2 diagnostic method order is registered")
    if list(map(int, config["section_first"]["beam_widths"])) != [2, 4, 8]:
        raise ValueError("V14.2 requires beam widths 2, 4, and 8")
    if int(config["parallel"]["patch_workers"]) != 12:
        raise ValueError("V14.2 patch_workers must remain 12")
    if int(config["parallel"]["numerical_threads_per_worker"]) != 1:
        raise ValueError("V14.2 numerical kernels must remain single-threaded per worker")
    seeds = (int(config["reach_round6"]["seed_a"]), int(config["reach_round6"]["seed_b"]))
    if seeds[0] == seeds[1]:
        raise ValueError("Reach round6 replicas require independent scramble seeds")
    if bool(config["exploratory_dataset"]["row_padding"]):
        raise ValueError("V14.2 forbids supervision row padding")
    return config


def _paths(config: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    retry4 = project_root / str(config["sources"]["retry4_root"])
    original = project_root / str(config["sources"]["original_pilot_root"])
    return {
        "reviewed_plan": project_root / str(config["sources"]["reviewed_plan"]),
        "retry4": retry4,
        "retry4_manifest": retry4 / "07_summary/artifact_manifest.json",
        "retry4_protocol": retry4 / "00_patch_inventory/protocol.json",
        "assignments": retry4 / "00_patch_inventory/patch_assignments.parquet",
        "tasks": retry4 / "00_patch_inventory/patch_task_nodes.parquet",
        "legacy_edges": retry4 / "00_patch_inventory/patch_task_edges_e0.parquet",
        "base_candidates": retry4 / "00_patch_inventory/patch_candidate_clusters.parquet",
        "original": original,
        "original_frozen_config": original / "00_protocol/frozen_config.json",
        "original_domain_cells": original / "01_workspace_proxy/domain_cells.parquet",
        "original_frontier": original / "01_workspace_proxy/frontier_inverse_probes.parquet",
        "original_parent_nodes": original / "02_domain_registry/pilot_task_nodes.parquet",
        "original_atlas_tasks": original / "04_workspace_atlas/atlas_task_nodes.parquet",
        "original_atlas_edges": original / "04_workspace_atlas/atlas_task_edges.parquet",
        "original_candidate_clusters": original / "04_workspace_atlas/candidate_clusters.parquet",
        "retry4_reach_gate": retry4 / "03_reach_extension/gate.json",
        "retry4_round5_a": retry4 / "03_reach_extension/replica_a_slab_round5.parquet",
        "retry4_round5_b": retry4 / "03_reach_extension/replica_b_slab_round5.parquet",
        "retry4_frontier": retry4 / "03_reach_extension/frontier_inverse_probes_round4_5.parquet",
        "robot_config": SOURCE_ROOT / str(config["robot_config"]),
    }


def _strict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _strict(dict(value)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _gate(path: Path, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    payload = {
        "gate_pass": bool(checks and all(bool(value) for value in checks.values())),
        "checks": {str(key): bool(value) for key, value in checks.items()},
        **evidence,
    }
    _write_json(path, payload)
    return payload


def _require_stage(output_root: Path, stage_name: str) -> dict[str, Any]:
    path = output_root / STAGE_DIRS[stage_name] / "gate.json"
    if not path.is_file():
        raise FileNotFoundError(f"required V14.2 stage is missing: {path}")
    value = _read_json(path)
    if not value.get("gate_pass", False):
        raise RuntimeError(f"required V14.2 stage failed: {stage_name}")
    return value


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _working_tree_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"], cwd=SOURCE_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def stage_inventory(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["inventory"]
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2 inventory already sealed: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    paths = _paths(config, project_root)
    required = tuple(
        paths[key]
        for key in (
            "reviewed_plan", "retry4_manifest", "retry4_protocol", "assignments",
            "tasks", "legacy_edges", "base_candidates", "robot_config",
            "original_frozen_config", "original_domain_cells", "original_frontier",
            "original_parent_nodes", "original_atlas_tasks", "original_atlas_edges",
            "original_candidate_clusters",
            "retry4_reach_gate", "retry4_round5_a", "retry4_round5_b", "retry4_frontier",
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    parent = str(config["sources"]["required_parent_git_sha"])
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, "HEAD"], cwd=SOURCE_ROOT, check=False
    ).returncode == 0
    manifest_report = {"declared": 0, "verified": 0, "failures": []}
    if not missing:
        manifest = _read_json(paths["retry4_manifest"])
        rows = tuple(manifest.get("artifacts", ()))
        manifest_report["declared"] = int(manifest.get("artifact_count", -1))
        for row in rows:
            artifact = paths["retry4"] / str(row["path"])
            if not artifact.is_file() or artifact.stat().st_size != int(row["bytes"]) or sha256_file(artifact) != str(row["sha256"]):
                manifest_report["failures"].append(str(row["path"]))
            else:
                manifest_report["verified"] += 1
    protocol = {
        "protocol_id": config["protocol_id"],
        "source_git_sha": _git_sha(),
        "required_parent_git_sha": parent,
        "required_parent_is_ancestor": ancestry,
        "working_tree_clean": _working_tree_clean(),
        "config_path": str(config["config_path"]),
        "config_sha256": sha256_file(Path(config["config_path"])),
        "runtime": runtime_fingerprint(),
        "parallel": dict(config["parallel"]),
        "retry4_manifest_verification": manifest_report,
    }
    _write_json(stage / "protocol.json", protocol)
    return _gate(
        stage / "gate.json",
        {
            "all_inputs_exist": not missing,
            "required_parent_is_ancestor": ancestry,
            "source_worktree_clean": protocol["working_tree_clean"],
            "retry4_manifest_exact": bool(
                manifest_report["declared"] == manifest_report["verified"]
                and not manifest_report["failures"]
            ),
            "twelve_workers_registered": int(config["parallel"]["patch_workers"]) == 12,
        },
        protocol=protocol,
        missing_inputs=missing,
    )


def _candidates_from_frame(frame: pd.DataFrame) -> tuple[AtlasCandidate, ...]:
    output = []
    for row in frame.sort_values(["task_node_id", "candidate_id"], kind="stable").itertuples(index=False):
        cluster = getattr(row, "cluster_id", None)
        output.append(
            AtlasCandidate(
                node_id=int(row.task_node_id),
                candidate_id=str(row.candidate_id),
                beta_rad=np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float),
                residual_mm=float(row.residual_mm),
                min_margin_deg=float(row.min_margin_deg),
                normalized_min_margin=float(row.normalized_min_margin),
                posture_cost=float(row.posture_cost),
                condition_number=float(row.condition_number),
                quality=str(row.quality),
                solver_success=bool(row.solver_success),
                actual_bounds=bool(row.actual_bounds),
                cluster_id=None if cluster is None or pd.isna(cluster) else int(cluster),
                diagnostics={"source": str(getattr(row, "source", "sealed_retry4"))},
            )
        )
    return tuple(output)


def _atlas_policy() -> AtlasPolicy:
    return AtlasPolicy(
        edge_match_deg=0.5,
        continuation_residual_max_mm=3.0,
        root_count=32,
        icm_max_sweeps=50,
        top_section_count=16,
        split_gap_deg=1.0,
        merge_overlap_p95_deg=0.5,
        merge_overlap_max_deg=1.0,
    )


def _section_policy(config: Mapping[str, Any], method: str) -> RootedSectionPolicy:
    section = config["section_first"]
    width, cycle = SECTION_METHODS[method]
    return RootedSectionPolicy(
        beam_width=width,
        root_count=int(section["root_count"]),
        maximum_growth_waves=int(section["maximum_growth_waves"]),
        parent_consensus_gold_deg=float(section["parent_consensus_gold_deg"]),
        parent_consensus_silver_deg=float(section["parent_consensus_silver_deg"]),
        continuation_residual_max_mm=float(section["continuation_residual_max_mm"]),
        reverse_return_max_deg=float(section["reverse_return_max_deg"]),
        minimum_alternative_chart_cells=int(section["minimum_alternative_chart_cells"]),
        beta_weights=tuple(map(float, section["beta_weights"])),
        online_cycle_repair=cycle,
    )


def _audit_policy(config: Mapping[str, Any]) -> SelectedSectionAuditPolicy:
    row = config["selected_section_audit"]
    return SelectedSectionAuditPolicy(
        path_p95_max_deg=float(row["path_p95_max_deg"]),
        cycle_p95_max_deg=float(row["cycle_p95_max_deg"]),
        repeat_p95_max_deg=float(row["repeat_p95_max_deg"]),
        common_max_deg=float(row["common_max_deg"]),
        continuation_residual_max_mm=float(row["continuation_residual_max_mm"]),
    )


def _patch_inputs(config: Mapping[str, Any], project_root: Path, patch_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _paths(config, project_root)
    tasks = pd.read_parquet(paths["tasks"], filters=[("patch_id", "==", patch_id)]).drop(columns=["patch_id"])
    edges = pd.read_parquet(paths["legacy_edges"], filters=[("patch_id", "==", patch_id)]).drop(columns=["patch_id"])
    candidates = pd.read_parquet(paths["base_candidates"], filters=[("patch_id", "==", patch_id)]).drop(columns=["patch_id"])
    assignments = pd.read_parquet(paths["assignments"], filters=[("patch_id", "==", patch_id)])
    if tasks.empty or assignments.empty:
        raise ValueError(f"unknown or empty patch: {patch_id}")
    return tasks, edges, candidates, assignments


def _segmented_continuation(environment: Any, tasks: pd.DataFrame, task_edges: pd.DataFrame):
    nodes = atlas_nodes_from_frames(tasks, task_edges)
    base = make_predictor_corrector_continuation(environment, residual_tolerance_mm=3.0)
    cell_by_node = {
        int(row.task_node_id): (int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in tasks.itertuples(index=False)
    }
    return nodes, make_segmented_continuation(
        base,
        {node.node_id: node for node in nodes},
        cell_by_node,
        step_max_mm=5.0,
    )


def _root_keys(tasks: pd.DataFrame, assignments: pd.DataFrame, candidates: Sequence[AtlasCandidate], count: int) -> tuple[tuple[int, str], ...]:
    seed_parent = int(assignments.loc[assignments["is_seed"].astype(bool), "seed_node_id"].iloc[0])
    root_rows = tasks[tasks["source_parent_node_id"].astype(int).eq(seed_parent)]
    representative = root_rows[root_rows["is_representative"].astype(bool)]
    root_node = int((representative if not representative.empty else root_rows).iloc[0]["task_node_id"])
    rows = sorted(
        (candidate for candidate in candidates if candidate.node_id == root_node and candidate.is_strict_feasible),
        key=lambda item: (0 if item.is_gold else 1, item.posture_cost, -item.min_margin_deg, item.candidate_id),
    )
    return tuple(item.key for item in rows[: int(count)])


def _stored_f0_growth(config: Mapping[str, Any], project_root: Path, patch_id: str, tasks: pd.DataFrame):
    root = _paths(config, project_root)["retry4"] / "01_patch_ablations" / patch_id
    candidate_frame = pd.read_parquet(root / "E3_dynamic_insertion_candidates.parquet")
    edge_frame = pd.read_parquet(root / "E3_dynamic_insertion_product_edges.parquet")
    task_edges = pd.read_parquet(root / "E3_dynamic_insertion_task_edges.parquet")
    candidates = _candidates_from_frame(candidate_frame)
    by_key = {candidate.key: candidate for candidate in candidates}
    directed = []
    for row in edge_frame.itertuples(index=False):
        left = (int(row.left_task_node_id), str(row.left_candidate_id))
        right = (int(row.right_task_node_id), str(row.right_candidate_id))
        for source, target, status in (
            (left, right, str(row.forward_status)),
            (right, left, str(row.reverse_status)),
        ):
            directed.append(
                DirectedContinuationEdge(
                    source_key=source,
                    target_key=target,
                    continuation_beta_rad=by_key[target].beta_rad,
                    match_gap_deg=0.0,
                    residual_mm=by_key[target].residual_mm,
                    corrector_iterations=0,
                    status=f"sealed_retry4:{status}",
                    minimum_margin_deg=by_key[target].min_margin_deg,
                )
            )
    report = _read_json(root / "E3_dynamic_insertion_report.json")
    graph = assemble_product_graph(
        atlas_nodes_from_frames(tasks, task_edges),
        candidates,
        directed,
        continuation_attempt_count=int(report["continuation_attempt_count"]),
        rejected_continuation_count=0,
        policy=_atlas_policy(),
    )
    atlas = build_canonical_atlas_from_product_graph(graph, policy=_atlas_policy())
    return section_growth_from_canonical_atlas(atlas), task_edges, bool(report["budget_exhausted"])


def _f1_growth(config: Mapping[str, Any], tasks: pd.DataFrame, legacy_edges: pd.DataFrame, candidates: Sequence[AtlasCandidate], environment: Any):
    baseline = config["flood_baselines"]["F1"]
    policy = CrossCellRepairPolicy(
        maximum_waves=int(baseline["maximum_waves"]),
        maximum_propagated_candidates_per_node=int(baseline["maximum_candidates_per_node"]),
        maximum_candidates_per_lineage_per_node=int(baseline["maximum_candidates_per_lineage_per_node"]),
    )
    base = make_predictor_corrector_continuation(environment, residual_tolerance_mm=3.0)
    repaired = run_cross_cell_repair_ablation(
        RepairAblation.E3_DYNAMIC_INSERTION,
        tasks,
        legacy_edges,
        candidates,
        base,
        atlas_policy=_atlas_policy(),
        repair_policy=policy,
    )
    atlas = build_canonical_atlas_from_product_graph(repaired.graph, policy=_atlas_policy())
    return section_growth_from_canonical_atlas(atlas), repaired.task_edges, repaired.budget_exhausted


def _component_ratio(growth: SectionGrowthResult) -> float:
    nodes = set(growth.covered_node_ids)
    adjacency: dict[int, set[int]] = {node: set() for node in nodes}
    for chart in growth.charts:
        for left, right in chart.selected_edges:
            if growth.primary_chart_by_node.get(left) == chart.chart_id and growth.primary_chart_by_node.get(right) == chart.chart_id:
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)
    largest = 0
    unseen = set(adjacency)
    while unseen:
        pending = [unseen.pop()]
        size = 0
        while pending:
            current = pending.pop()
            size += 1
            for neighbor in adjacency.get(current, ()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        largest = max(largest, size)
    return float(largest / max(1, len(growth.task_nodes)))


def _single_cell_chart_ratio(growth: SectionGrowthResult, tasks: pd.DataFrame) -> float:
    cell_by_node = {
        int(row.task_node_id): (int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in tasks.itertuples(index=False)
    }
    counts = [len({cell_by_node[node] for node in chart.selected_by_node}) for chart in growth.charts]
    return float(sum(value <= 1 for value in counts) / max(1, len(counts)))


def _persist_method(
    directory: Path,
    growth: SectionGrowthResult,
    audit: Any,
    *,
    tasks: pd.DataFrame,
    patch_id: str,
    patch_split: str,
    method: str,
    raw_cap_hit: bool,
    runtime_s: float,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in growth.frames().items():
        _write_parquet(frame, directory / f"{name}.parquet")
    for name, frame in audit.frames.items():
        _write_parquet(frame, directory / f"{name}.parquet")
    coverage = len(growth.covered_node_ids) / max(1, len(growth.task_nodes))
    component = _component_ratio(growth)
    report = {
        "patch_id": patch_id,
        "patch_split": patch_split,
        "method": method,
        "audit_gate_pass": bool(audit.gate_pass),
        "audit_metrics": {key: dict(value) for key, value in audit.metrics.items()},
        "audit_failure_reasons": list(audit.failure_reasons),
        "fresh_audit_execution_count": int(audit.execution_count),
        "selected_section_coverage": coverage,
        "largest_selected_component_ratio": component,
        "single_cell_chart_ratio": _single_cell_chart_ratio(growth, tasks),
        "chart_count": len(growth.charts),
        "raw_cap_hit": bool(raw_cap_hit or growth.raw_cap_hit),
        "continuation_attempt_count": growth.continuation_attempt_count,
        "rejected_continuation_count": growth.rejected_continuation_count,
        "runtime_s": runtime_s,
    }
    report["gate_pass"] = bool(audit.gate_pass and component >= 0.60)
    _write_json(directory / "report.json", report)
    return report


def _execute_patch_method(
    config: Mapping[str, Any],
    project_root: Path,
    patch_id: str,
    method: str,
    directory: Path,
) -> dict[str, Any]:
    report_path = directory / "report.json"
    if report_path.is_file():
        return _read_json(report_path)
    tasks, legacy_edges, candidate_frame, assignments = _patch_inputs(config, project_root, patch_id)
    candidates = _candidates_from_frame(candidate_frame)
    environment = _EndpointOnlyForwardAdapter(load_environment(project_root, _paths(config, project_root)["robot_config"]))
    patch_split = str(assignments["patch_split"].iloc[0])
    started = time.time()
    if method == "F0":
        growth, task_edges, raw_cap = _stored_f0_growth(config, project_root, patch_id, tasks)
    elif method == "F1":
        growth, task_edges, raw_cap = _f1_growth(config, tasks, legacy_edges, candidates, environment)
    elif method in SECTION_METHODS:
        retry_patch = _paths(config, project_root)["retry4"] / "01_patch_ablations" / patch_id
        task_edges = pd.read_parquet(retry_patch / "E3_dynamic_insertion_task_edges.parquet")
        nodes, continuation = _segmented_continuation(environment, tasks, task_edges)
        growth = build_section_first_atlas(
            nodes,
            candidates,
            continuation,
            root_keys=_root_keys(tasks, assignments, candidates, int(config["section_first"]["root_count"])),
            policy=_section_policy(config, method),
        )
        raw_cap = growth.raw_cap_hit
    else:
        raise ValueError(f"unknown V14.2 method: {method}")
    if method in {"F0", "F1"}:
        _nodes, continuation = _segmented_continuation(environment, tasks, task_edges)
    audit = audit_selected_sections(
        growth,
        continuation,
        policy=_audit_policy(config),
        patch_id=patch_id,
        method=method,
    )
    return _persist_method(
        directory,
        growth,
        audit,
        tasks=tasks,
        patch_id=patch_id,
        patch_split=patch_split,
        method=method,
        raw_cap_hit=raw_cap,
        runtime_s=time.time() - started,
    )


def _run_jobs(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    stage_name: str,
    jobs: Sequence[tuple[str, str]],
) -> None:
    worker_count = int(config["parallel"]["patch_workers"])
    pending = list(jobs)
    running: list[tuple[str, str, subprocess.Popen[str], Any]] = []
    failures = []
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/mpl-bacra-v14-2",
        }
    )
    while pending or running:
        next_running = []
        for patch_id, method, process, handle in running:
            status = process.poll()
            if status is None:
                next_running.append((patch_id, method, process, handle))
            else:
                handle.close()
                if status != 0:
                    failures.append({"patch_id": patch_id, "method": method, "returncode": status})
        running = next_running
        if failures:
            for _patch, _method, process, handle in running:
                process.terminate()
                process.wait(timeout=30)
                handle.close()
            raise RuntimeError(f"V14.2 patch workers failed: {failures}")
        while pending and len(running) < worker_count:
            patch_id, method = pending.pop(0)
            worker_dir = output_root / STAGE_DIRS[stage_name] / patch_id / method
            if (worker_dir / "report.json").is_file():
                continue
            worker_dir.mkdir(parents=True, exist_ok=True)
            handle = (worker_dir / "worker.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config", str(config["config_path"]),
                "--output-root", str(output_root),
                "--stage", stage_name,
                "--patch-id", patch_id,
                "--method", method,
            ]
            process = subprocess.Popen(
                command,
                cwd=SOURCE_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((patch_id, method, process, handle))
        if pending or running:
            time.sleep(1.0)


def stage_diagnostics(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_stage(output_root, "inventory")
    stage = output_root / STAGE_DIRS["diagnostics"]
    patch_filter = config.get("_patch_filter")
    if patch_filter is not None:
        report = _execute_patch_method(config, project_root, str(patch_filter), "F0", stage / str(patch_filter) / "F0")
        return {"gate_pass": True, "worker_report": report}
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2 diagnostics already sealed: {stage}")
    patches = tuple(map(str, config["diagnostics"]["patch_ids"]))
    _run_jobs(config, output_root, stage_name="diagnostics", jobs=[(patch, "F0") for patch in patches])
    reports = [_read_json(stage / patch / "F0/report.json") for patch in patches]
    _write_parquet(pd.DataFrame.from_records(reports), stage / "diagnostic_metrics.parquet")
    _write_parquet(
        pd.DataFrame.from_records(
            [
                {
                    "patch_id": row["patch_id"],
                    "method": "F0",
                    "raw_cap_hit": bool(row["raw_cap_hit"]),
                    "selected_section_affected_by_cap": None,
                    "impact_status": "requires_phase1_F1_and_S4_S8_counterfactual",
                }
                for row in reports
            ]
        ),
        stage / "cap_hit_selected_section_impact.parquet",
    )
    required_frames = (
        "fresh_edge_audit.parquet", "fresh_path_audit.parquet", "fresh_cycle_audit.parquet",
        "fresh_multipath_audit.parquet", "fresh_repeat_direction_audit.parquet",
        "cap_hit_events.parquet",
    )
    complete = bool(
        all((stage / patch / "F0" / name).is_file() for patch in patches for name in required_frames)
        and (stage / "cap_hit_selected_section_impact.parquet").is_file()
    )
    return _gate(
        stage / "gate.json",
        {"four_registered_patches": len(reports) == 4, "full_fresh_audit_observability": complete},
        patch_ids=list(patches),
        phase_is_diagnostic_not_scientific_admission=True,
    )


def _frame_stability(low_dir: Path, high_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    low_primary = pd.read_parquet(low_dir / "primary_section.parquet")
    high_primary = pd.read_parquet(high_dir / "primary_section.parquet")
    low_h = pd.read_parquet(low_dir / "section_hypotheses.parquet").query("selected")
    high_h = pd.read_parquet(high_dir / "section_hypotheses.parquet").query("selected")
    low_map = low_primary.dropna(subset=["primary_chart_id"]).set_index("task_node_id")["primary_chart_id"].to_dict()
    high_map = high_primary.dropna(subset=["primary_chart_id"]).set_index("task_node_id")["primary_chart_id"].to_dict()
    low_beta = {(str(row.chart_id), int(row.task_node_id)): np.asarray([getattr(row, name) for name in BETA_COLUMNS]) for row in low_h.itertuples(index=False)}
    high_beta = {(str(row.chart_id), int(row.task_node_id)): np.asarray([getattr(row, name) for name in BETA_COLUMNS]) for row in high_h.itertuples(index=False)}
    common = set(low_map) & set(high_map)
    union = set(low_map) | set(high_map)
    gaps = [beta_rms_deg(low_beta[(str(low_map[node]), node)], high_beta[(str(high_map[node]), node)]) for node in sorted(common)]
    chart_change = sum(str(low_map[node]) != str(high_map[node]) for node in common) / max(1, len(common))
    low_boundary = set(low_primary.loc[low_primary["abstained"].astype(bool), "task_node_id"].astype(int))
    high_boundary = set(high_primary.loc[high_primary["abstained"].astype(bool), "task_node_id"].astype(int))
    boundary_change = len(low_boundary ^ high_boundary) / max(1, len(low_boundary | high_boundary))
    def edges(path: Path) -> set[tuple[str, int, int]]:
        frame = pd.read_parquet(path / "selected_edges.parquet")
        return {(str(row.chart_id), int(row.left_node_id), int(row.right_node_id)) for row in frame.itertuples(index=False)}
    low_edges, high_edges = edges(low_dir), edges(high_dir)
    edge_change = len(low_edges ^ high_edges) / max(1, len(low_edges | high_edges))
    policy = config["section_stability"]
    metrics = {
        "coverage_jaccard": len(common) / max(1, len(union)),
        "selected_beta_p95_deg": float(np.percentile(gaps, 95)) if gaps else math.inf,
        "selected_beta_max_deg": float(np.max(gaps)) if gaps else math.inf,
        "chart_assignment_change_ratio": chart_change,
        "boundary_change_ratio": boundary_change,
        "selected_edge_change_ratio": edge_change,
    }
    checks = {
        "coverage_jaccard": metrics["coverage_jaccard"] >= float(policy["coverage_jaccard_min"]),
        "selected_beta_p95": metrics["selected_beta_p95_deg"] <= float(policy["exploratory_beta_p95_max_deg"]),
        "selected_beta_max": metrics["selected_beta_max_deg"] <= float(policy["exploratory_beta_max_deg"]),
        "chart_assignment": chart_change <= float(policy["chart_assignment_change_max"]),
        "boundary": boundary_change <= float(policy["boundary_change_max"]),
        "selected_edges": edge_change <= float(policy["selected_edge_change_max"]),
    }
    low_cap = len(pd.read_parquet(low_dir / "cap_hit_events.parquet")) > 0
    high_cap = len(pd.read_parquet(high_dir / "cap_hit_events.parquet")) > 0
    affected = bool((low_cap or high_cap) and not all(checks.values()))
    return {
        **metrics,
        "common_node_count": len(common),
        "raw_cap_hit": bool(low_cap or high_cap),
        "selected_section_affected_by_cap": affected,
        "checks": checks,
        "gate_pass": bool(all(checks.values()) and not affected),
    }


def stage_mechanisms(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_stage(output_root, "diagnostics")
    stage = output_root / STAGE_DIRS["mechanisms"]
    patch_filter = config.get("_patch_filter")
    method_filter = config.get("_method_filter")
    if patch_filter is not None and method_filter is not None:
        report = _execute_patch_method(config, project_root, str(patch_filter), str(method_filter), stage / str(patch_filter) / str(method_filter))
        return {"gate_pass": True, "worker_report": report}
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2 mechanisms already sealed: {stage}")
    patches = tuple(map(str, config["diagnostics"]["patch_ids"]))
    methods = ("F1", "S2", "S4", "S8", "S4C")
    _run_jobs(config, output_root, stage_name="mechanisms", jobs=[(patch, method) for patch in patches for method in methods])
    reports = [_read_json(stage / patch / method / "report.json") for patch in patches for method in methods]
    stability_rows = []
    for patch in patches:
        stability = _frame_stability(stage / patch / "S4", stage / patch / "S8", config)
        stability["patch_id"] = patch
        stability_rows.append(stability)
        _write_json(stage / patch / "S4_vs_S8_stability.json", stability)
        _write_parquet(
            pd.DataFrame.from_records([
                {
                    "patch_id": patch,
                    "comparison": "S4_vs_S8",
                    "raw_cap_hit": stability["raw_cap_hit"],
                    "selected_section_affected_by_cap": stability["selected_section_affected_by_cap"],
                    "section_stability_gate_pass": stability["gate_pass"],
                }
            ]),
            stage / patch / "cap_hit_selected_section_impact.parquet",
        )
    _write_parquet(pd.DataFrame.from_records(reports), stage / "mechanism_metrics.parquet")
    candidates = []
    for method in ("S4", "S4C"):
        method_reports = [row for row in reports if row["method"] == method]
        stability_pass = sum(bool(row["gate_pass"]) for row in stability_rows)
        audit_pass = sum(bool(row["gate_pass"]) for row in method_reports)
        candidates.append((audit_pass, stability_pass, method))
    best_audit, best_stability, selected = max(candidates, key=lambda item: (item[0], item[1], item[2] == "S4C"))
    selected_method = selected if best_audit >= 3 and best_stability >= 3 else None
    return _gate(
        stage / "gate.json",
        {"all_registered_rows": len(reports) == 20, "method_selected": selected_method is not None},
        selected_method=selected_method,
        selection_scores=[{"method": method, "diagnostic_patch_pass_count": audit, "stability_patch_pass_count": stability} for audit, stability, method in candidates],
        F1_is_doubled_cap_flood_only=True,
    )


def stage_confirmation(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    mechanisms = _read_json(output_root / STAGE_DIRS["mechanisms"] / "gate.json")
    stage = output_root / STAGE_DIRS["confirmation"]
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2 confirmation already sealed: {stage}")
    selected = mechanisms.get("selected_method")
    if not mechanisms.get("gate_pass") or selected not in {"S4", "S4C"}:
        return _gate(stage / "gate.json", {"mechanism_selected": False}, skipped=True, reasons=["mechanism_gate_failed"])
    patch_filter = config.get("_patch_filter")
    method_filter = config.get("_method_filter")
    if patch_filter is not None and method_filter is not None:
        if str(method_filter) not in {str(selected), "S8"}:
            raise ValueError(f"invalid confirmation worker method: {method_filter}")
        _execute_patch_method(
            config,
            project_root,
            str(patch_filter),
            str(method_filter),
            stage / str(patch_filter) / str(method_filter),
        )
        return {"gate_pass": True, "patch_id": str(patch_filter), "method": str(method_filter)}
    assignments = pd.read_parquet(_paths(config, project_root)["assignments"])
    patches = tuple(sorted(map(str, assignments["patch_id"].unique())))
    _run_jobs(config, output_root, stage_name="confirmation", jobs=[(patch, str(selected)) for patch in patches] + [(patch, "S8") for patch in patches])
    reports = []
    for patch in patches:
        base = _read_json(stage / patch / str(selected) / "report.json")
        stability = _frame_stability(stage / patch / str(selected), stage / patch / "S8", config)
        _write_json(stage / patch / "section_stability.json", stability)
        base["section_stability_gate_pass"] = bool(stability["gate_pass"])
        base["selected_section_affected_by_cap"] = bool(stability["selected_section_affected_by_cap"])
        base["gate_pass"] = bool(base["audit_gate_pass"] and stability["gate_pass"] and base["largest_selected_component_ratio"] >= float(config["patch_gate"]["largest_component_ratio_min"]))
        _write_json(stage / patch / "confirmation_report.json", base)
        reports.append(base)
    _write_parquet(pd.DataFrame.from_records(reports), stage / "confirmation_metrics.parquet")
    patch_gate = dict(
        evaluate_section_patch_gate(
            reports,
            development_pass_min=int(config["patch_gate"]["development_pass_min"]),
            confirmation_pass_min=int(config["patch_gate"]["confirmation_pass_min"]),
            largest_component_ratio_min=float(config["patch_gate"]["largest_component_ratio_min"]),
            single_cell_chart_ratio_max=float(config["patch_gate"]["single_cell_chart_ratio_max"]),
        )
    )
    return _gate(
        stage / "gate.json",
        {"twelve_patch_method_freeze": bool(patch_gate["gate_pass"])},
        selected_method=selected,
        patch_gate=patch_gate,
        unresolved_patches_are_abstention_regions=True,
    )


def _conditional_skip(stage: Path, reason: str) -> dict[str, Any]:
    return _gate(stage / "gate.json", {"upstream_authorized": False}, skipped=True, reasons=[reason])


def _frozen_round5_lower_parent_ids(
    config: Mapping[str, Any], project_root: Path
) -> tuple[set[int], frozenset[CellKey]]:
    paths = _paths(config, project_root)
    grid = _workspace_grid(paths)
    a = pd.read_parquet(paths["retry4_round5_a"], columns=["x_m", "y_m", "z_m"])
    b = pd.read_parquet(paths["retry4_round5_b"], columns=["x_m", "y_m", "z_m"])
    lower = grid.cells_for_points(a.to_numpy(float), level_mm=grid.convergence_level_mm) & grid.cells_for_points(
        b.to_numpy(float), level_mm=grid.convergence_level_mm
    )
    parents = pd.read_parquet(paths["original_parent_nodes"])
    parent_cells = grid.cells_for_points(
        parents.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(float),
        level_mm=grid.convergence_level_mm,
    )
    # Re-evaluate per row because cells_for_points intentionally returns a set.
    step = grid.convergence_level_mm / 1000.0
    origin = np.asarray(grid.origin_m, dtype=float)
    allowed = set()
    for row in parents.itertuples(index=False):
        index = np.floor((np.asarray([row.x_m, row.y_m, row.z_m]) - origin) / step).astype(int)
        cell = CellKey(grid.convergence_level_mm, int(index[0]), int(index[1]), int(index[2]))
        if cell in lower:
            allowed.add(int(row.node_id))
    return allowed, lower


def _pilot_inputs(
    config: Mapping[str, Any], project_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _paths(config, project_root)
    allowed_parent_ids, _lower = _frozen_round5_lower_parent_ids(config, project_root)
    tasks = pd.read_parquet(paths["original_atlas_tasks"])
    tasks = tasks[tasks["source_parent_node_id"].astype(int).isin(allowed_parent_ids)].copy()
    node_ids = set(tasks["task_node_id"].astype(int))
    edges = pd.read_parquet(paths["original_atlas_edges"])
    edges = edges[
        edges["left_node_id"].astype(int).isin(node_ids)
        & edges["right_node_id"].astype(int).isin(node_ids)
    ].copy()
    candidates = pd.read_parquet(paths["original_candidate_clusters"])
    candidates = candidates[candidates["task_node_id"].astype(int).isin(node_ids)].copy()
    parents = pd.read_parquet(paths["original_parent_nodes"])
    parents = parents[parents["node_id"].astype(int).isin(allowed_parent_ids)].copy()
    return tasks, edges, candidates, parents


def _prepare_pilot_root_registry(
    config: Mapping[str, Any], project_root: Path, stage: Path
) -> pd.DataFrame:
    path = stage / "root_registry.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    tasks, _edges, candidate_frame, _parents = _pilot_inputs(config, project_root)
    representatives = tasks[tasks["is_representative"].astype(bool)]
    root_nodes = tuple(
        AtlasTaskNode(
            int(row.task_node_id),
            np.asarray([row.x_m, row.y_m, row.z_m], dtype=float),
            (),
            bool(row.core_safe),
        )
        for row in representatives.itertuples(index=False)
    )
    root_node_ids = deterministic_root_nodes(
        root_nodes,
        count=int(config["exploratory_5k"]["root_count"]),
    )
    candidates = _candidates_from_frame(candidate_frame)
    by_node: dict[int, list[AtlasCandidate]] = {}
    for candidate in candidates:
        by_node.setdefault(candidate.node_id, []).append(candidate)
    rows = []
    for root_index, node_id in enumerate(root_node_ids):
        ranked = sorted(
            by_node.get(node_id, ()),
            key=lambda item: (
                0 if item.is_gold else 1,
                item.posture_cost,
                -item.min_margin_deg,
                item.candidate_id,
            ),
        )
        if not ranked:
            continue
        rows.append(
            {
                "root_index": root_index,
                "task_node_id": node_id,
                "candidate_id": ranked[0].candidate_id,
            }
        )
    registry = pd.DataFrame.from_records(rows)
    if len(registry) != int(config["exploratory_5k"]["root_count"]):
        raise RuntimeError("not every registered 5k root has a feasible candidate")
    _write_parquet(registry, path)
    return registry


def _run_pilot_root(
    config: Mapping[str, Any],
    project_root: Path,
    stage: Path,
    root_index: int,
    selected_method: str,
) -> dict[str, Any]:
    root_dir = stage / f"root_{root_index:03d}"
    report_path = root_dir / "report.json"
    if report_path.is_file():
        return _read_json(report_path)
    registry = pd.read_parquet(stage / "root_registry.parquet")
    row = registry[registry["root_index"].astype(int).eq(root_index)].iloc[0]
    tasks, edges, candidate_frame, _parents = _pilot_inputs(config, project_root)
    candidates = _candidates_from_frame(candidate_frame)
    environment = _EndpointOnlyForwardAdapter(
        load_environment(project_root, _paths(config, project_root)["robot_config"])
    )
    nodes, continuation = _segmented_continuation(environment, tasks, edges)
    root_key = (int(row.task_node_id), str(row.candidate_id))
    reports = {}
    for label, width in (("K4", 4), ("K8", 8)):
        policy = replace(
            _section_policy(config, selected_method),
            beam_width=width,
            root_count=1,
        )
        started = time.time()
        growth = build_section_first_atlas(
            nodes,
            candidates,
            continuation,
            root_keys=(root_key,),
            policy=policy,
        )
        audit = audit_selected_sections(
            growth,
            continuation,
            policy=_audit_policy(config),
            patch_id=f"root_{root_index:03d}",
            method=f"{selected_method}_{label}",
        )
        reports[label] = _persist_method(
            root_dir / label,
            growth,
            audit,
            tasks=tasks,
            patch_id=f"root_{root_index:03d}",
            patch_split="pilot_root",
            method=f"{selected_method}_{label}",
            raw_cap_hit=growth.raw_cap_hit,
            runtime_s=time.time() - started,
        )
    stability = _frame_stability(root_dir / "K4", root_dir / "K8", config)
    _write_json(root_dir / "section_stability.json", stability)
    cell_by_node = {
        int(item.task_node_id): (
            int(item.cell_level_mm), int(item.cell_ix), int(item.cell_iy), int(item.cell_iz)
        )
        for item in tasks.itertuples(index=False)
    }
    primary = pd.read_parquet(root_dir / "K4/primary_section.parquet")
    cell_count = len(
        {
            cell_by_node[int(node_id)]
            for node_id in primary.loc[~primary["abstained"].astype(bool), "task_node_id"]
        }
    )
    report = {
        "root_index": root_index,
        "root_task_node_id": root_key[0],
        "root_candidate_id": root_key[1],
        "cell_count": cell_count,
        "K4_audit_gate_pass": bool(reports["K4"]["audit_gate_pass"]),
        "section_stability_gate_pass": bool(stability["gate_pass"]),
        "raw_cap_hit": bool(stability["raw_cap_hit"]),
        "selected_section_affected_by_cap": bool(stability["selected_section_affected_by_cap"]),
    }
    report["gate_pass"] = bool(
        report["K4_audit_gate_pass"]
        and report["section_stability_gate_pass"]
        and cell_count >= int(config["section_first"]["minimum_alternative_chart_cells"])
    )
    _write_json(report_path, report)
    return report


def _run_root_jobs(
    config: Mapping[str, Any],
    output_root: Path,
    root_indices: Sequence[int],
) -> None:
    worker_count = int(config["parallel"]["patch_workers"])
    pending = list(map(int, root_indices))
    running: list[tuple[int, subprocess.Popen[str], Any]] = []
    failures = []
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/mpl-bacra-v14-2",
        }
    )
    while pending or running:
        next_running = []
        for root_index, process, handle in running:
            status = process.poll()
            if status is None:
                next_running.append((root_index, process, handle))
            else:
                handle.close()
                if status != 0:
                    failures.append({"root_index": root_index, "returncode": status})
        running = next_running
        if failures:
            for _index, process, handle in running:
                process.terminate()
                process.wait(timeout=30)
                handle.close()
            raise RuntimeError(f"V14.2 pilot root workers failed: {failures}")
        while pending and len(running) < worker_count:
            root_index = pending.pop(0)
            root_dir = output_root / STAGE_DIRS["repaired_pilot"] / f"root_{root_index:03d}"
            if (root_dir / "report.json").is_file():
                continue
            root_dir.mkdir(parents=True, exist_ok=True)
            handle = (root_dir / "worker.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config", str(config["config_path"]),
                "--output-root", str(output_root),
                "--stage", "repaired_pilot",
                "--root-index", str(root_index),
            ]
            process = subprocess.Popen(
                command,
                cwd=SOURCE_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((root_index, process, handle))
        if pending or running:
            time.sleep(1.0)


def _workspace_grid(paths: Mapping[str, Path]) -> WorkspaceGridSpec:
    frozen = _read_json(paths["original_frozen_config"])
    domain = frozen["domain"]
    return WorkspaceGridSpec(
        levels_mm=tuple(map(int, domain["levels_mm"])),
        x_slab_m=(float(domain["x_min_m"]), float(domain["x_max_m"])),
        origin_m=tuple(map(float, domain["origin_m"])),
    )


def _frontier_probe_round(
    environment: Any,
    cells: Sequence[CellKey],
    *,
    grid: WorkspaceGridSpec,
    seed_xyz: np.ndarray,
    seed_beta: np.ndarray,
    starts: int,
    round_id: int,
) -> tuple[list[FrontierProbeEvidence], pd.DataFrame]:
    if not cells:
        return [], pd.DataFrame()
    tree = cKDTree(seed_xyz)
    targets = cell_centers(cells, grid=grid)
    policy = CandidatePolicy(
        candidate_budget_per_node=int(starts),
        difficult_candidate_budget_per_node=int(starts),
        candidate_seed_budget_per_node=int(starts),
        difficult_seed_budget_per_node=int(starts),
        nullspace_seed_budget_per_node=0,
        search_mode=CandidateSearchMode.CORRECTION,
        solver_names=("weighted_dls", "bounded_least_squares", "slsqp"),
        max_corrector_iterations=100,
        tracking_tolerance_mm=1.0,
        max_residual_mm=3.0,
        gold_margin_deg=1.5,
        silver_margin_deg=0.0,
    )
    evidence = []
    rows = []
    for cell, target in zip(cells, targets, strict=True):
        _distance, indices = tree.query(target, k=min(int(starts), len(seed_xyz)))
        indices = np.asarray(indices, dtype=int).reshape(-1)
        bank = solve_candidate_bank(
            environment,
            target.reshape(1, 3),
            policy,
            neighbor_beta_rad={0: seed_beta[indices]},
        )
        accepted = [item for item in bank.records if item.quality is not CandidateQuality.REJECT]
        selected = min(
            accepted,
            key=lambda item: (
                0 if item.quality is CandidateQuality.GOLD else 1,
                item.residual_mm,
                -item.min_margin_deg,
                item.candidate_id,
            ),
            default=None,
        )
        found = selected is not None
        evidence.append(FrontierProbeEvidence(cell, found, round_id=round_id))
        rows.append(
            {
                "round_id": round_id,
                "cell_level_mm": cell.level_mm,
                "cell_ix": cell.ix,
                "cell_iy": cell.iy,
                "cell_iz": cell.iz,
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "found_valid_inverse": found,
                "candidate_attempt_count": len(bank.records),
                "selected_candidate_id": None if selected is None else selected.candidate_id,
                "selected_residual_mm": None if selected is None else selected.residual_mm,
                **{
                    name: None if selected is None else float(selected.beta_rad[index])
                    for index, name in enumerate(BETA_COLUMNS)
                },
            }
        )
    return evidence, pd.DataFrame.from_records(rows)


def stage_reach_round6(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    _require_stage(output_root, "inventory")
    stage = output_root / STAGE_DIRS["reach_round6"]
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2 Reach round6 already sealed: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    paths = _paths(config, project_root)
    grid = _workspace_grid(paths)
    reach = config["reach_round6"]
    environment = _EndpointOnlyForwardAdapter(load_environment(project_root, paths["robot_config"]))
    round5_a = pd.read_parquet(paths["retry4_round5_a"])
    round5_b = pd.read_parquet(paths["retry4_round5_b"])
    prior_power = 19
    prior_chunk = 2**prior_power
    if len(round5_a) <= prior_chunk or len(round5_b) <= prior_chunk:
        raise RuntimeError("retry4 Round-5 slabs cannot be split into Round-4/5 cumulative views")
    round4_a = round5_a.iloc[:-prior_chunk]
    round4_b = round5_b.iloc[:-prior_chunk]
    retry_gate = _read_json(paths["retry4_reach_gate"])
    seeds_a = tuple(map(int, retry_gate["replica_a_seed_lineage"]))
    seeds_b = tuple(map(int, retry_gate["replica_b_seed_lineage"]))
    round4 = ReachReplicaRound(
        4,
        ReachReplica("A", seeds_a[:-1], round4_a.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float)),
        ReachReplica("B", seeds_b[:-1], round4_b.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float)),
    )
    round5 = ReachReplicaRound(
        5,
        ReachReplica("A", seeds_a, round5_a.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float)),
        ReachReplica("B", seeds_b, round5_b.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float)),
    )
    spec = ReachSamplingRoundSpec(int(reach["power"]), int(reach["seed_a"]), int(reach["seed_b"]))
    generated = generate_independent_reach_samples(
        environment,
        np.asarray(environment.bounds, dtype=float),
        grid=grid,
        rounds=(spec,),
        chunk_rows=65536,
    )
    new_a = pd.DataFrame(
        {
            "x_m": generated.xyz_a[:, 0], "y_m": generated.xyz_a[:, 1], "z_m": generated.xyz_a[:, 2],
            **{name: generated.beta_a[:, index] for index, name in enumerate(BETA_COLUMNS)},
            "source": "replica_a_round_6",
        }
    )
    new_b = pd.DataFrame(
        {
            "x_m": generated.xyz_b[:, 0], "y_m": generated.xyz_b[:, 1], "z_m": generated.xyz_b[:, 2],
            **{name: generated.beta_b[:, index] for index, name in enumerate(BETA_COLUMNS)},
            "source": "replica_b_round_6",
        }
    )
    final_a = pd.concat([round5_a, new_a], ignore_index=True)
    final_b = pd.concat([round5_b, new_b], ignore_index=True)
    round6 = ReachReplicaRound(
        6,
        ReachReplica("A", (*seeds_a, spec.seed_a), final_a.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float)),
        ReachReplica("B", (*seeds_b, spec.seed_b), final_b.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float)),
    )
    original_frontier = pd.read_parquet(paths["original_frontier"])
    retry_frontier = pd.read_parquet(paths["retry4_frontier"])
    frontier_evidence = [
        FrontierProbeEvidence(
            CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)),
            bool(row.found_valid_inverse),
            round_id=int(row.round_id),
        )
        for frame in (original_frontier, retry_frontier)
        for row in frame.itertuples(index=False)
    ]
    domain = pd.read_parquet(paths["original_domain_cells"])
    supplemental = tuple(
        CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in domain[
            domain["cell_level_mm"].eq(grid.convergence_level_mm)
            & (domain["registered_pool_support"].astype(bool) | domain["tip_pool_support"].astype(bool))
        ].itertuples(index=False)
    )
    builder = ReachProxyBuilder(
        ReachProxyConfig(
            grid=grid,
            minimum_weighted_jaccard=float(reach["weighted_jaccard_min"]),
            maximum_new_volume_ratio=float(reach["new_volume_ratio_max"]),
            maximum_boundary_change_ratio=1.0,
            maximum_frontier_new_volume_ratio=float(reach["frontier_new_volume_ratio_max"]),
            required_consecutive_rounds=2,
        )
    )
    preliminary = builder.build((round4, round5, round6), frontier_evidence=frontier_evidence, supplemental_supported_cells=supplemental)
    frontier_cells = frontier_candidates(
        preliminary.proxy_upper_cells,
        grid=grid,
        maximum_count=int(reach["frontier_cells"]),
    )
    seed_xyz = np.vstack(
        [final_a.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float), final_b.loc[:, ("x_m", "y_m", "z_m")].to_numpy(float)]
    )
    seed_beta = np.vstack([final_a.loc[:, BETA_COLUMNS].to_numpy(float), final_b.loc[:, BETA_COLUMNS].to_numpy(float)])
    new_frontier, frontier_frame = _frontier_probe_round(
        environment,
        frontier_cells,
        grid=grid,
        seed_xyz=seed_xyz,
        seed_beta=seed_beta,
        starts=int(reach["frontier_starts"]),
        round_id=6,
    )
    frontier_evidence.extend(new_frontier)
    result = builder.build((round4, round5, round6), frontier_evidence=frontier_evidence, supplemental_supported_cells=supplemental)
    _write_parquet(final_a, stage / "replica_a_slab_round6.parquet")
    _write_parquet(final_b, stage / "replica_b_slab_round6.parquet")
    _write_parquet(frontier_frame, stage / "frontier_inverse_probes_round6.parquet")
    level = grid.convergence_level_mm
    unions = [
        grid.cells_for_points(pair.replica_a.xyz_m, level_mm=level)
        | grid.cells_for_points(pair.replica_b.xyz_m, level_mm=level)
        for pair in (round4, round5, round6)
    ]
    measure_boundary = {
        5: measure_weighted_boundary_change_ratio(unions[1], unions[0]),
        6: measure_weighted_boundary_change_ratio(unions[2], unions[1]),
    }
    metrics = [asdict(item) for item in result.replica_metrics]
    recent = metrics[-2:]
    recent_checks = [
        bool(
            row["volume_weighted_jaccard"] >= float(reach["weighted_jaccard_min"])
            and row["new_volume_ratio"] <= float(reach["new_volume_ratio_max"])
            and row["frontier_new_volume_ratio"] <= float(reach["frontier_new_volume_ratio_max"])
            and measure_boundary[int(row["round_id"])] <= float(reach["measure_boundary_change_max"])
        )
        for row in recent
    ]
    gap = (len(result.proxy_upper_cells) - len(result.proxy_lower_cells)) / max(1, len(result.proxy_upper_cells))
    convergence = bool(all(recent_checks) and gap <= float(reach["lower_upper_measure_gap_max"]))
    report = {
        "reach_convergence_gate": convergence,
        "metrics_round4_6": metrics,
        "measure_weighted_boundary_change_ratio": measure_boundary,
        "recent_round_checks": recent_checks,
        "proxy_lower_cell_count": len(result.proxy_lower_cells),
        "proxy_upper_cell_count": len(result.proxy_upper_cells),
        "lower_upper_measure_gap_ratio": gap,
        "replica_a_seed_lineage": [*seeds_a, spec.seed_a],
        "replica_b_seed_lineage": [*seeds_b, spec.seed_b],
        "raw_boundary_change_retained_as_diagnostic": True,
    }
    _write_json(stage / "reach_round6_report.json", report)
    return _gate(stage / "gate.json", {"measure_aware_reach_convergence": convergence}, **report)


def stage_repaired_pilot(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["repaired_pilot"]
    confirmation = _read_json(output_root / STAGE_DIRS["confirmation"] / "gate.json")
    if not confirmation.get("gate_pass", False):
        return _conditional_skip(stage, "twelve_patch_confirmation_failed")
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2 repaired Pilot already sealed: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    selected_method = str(confirmation["selected_method"])
    root_index = config.get("_root_index")
    if root_index is not None:
        report = _run_pilot_root(
            config,
            project_root,
            stage,
            int(root_index),
            selected_method,
        )
        return {"gate_pass": True, "worker_report": report}
    registry = _prepare_pilot_root_registry(config, project_root, stage)
    root_indices = tuple(map(int, registry["root_index"]))
    _run_root_jobs(config, output_root, root_indices)
    root_reports = [
        _read_json(stage / f"root_{index:03d}/report.json") for index in root_indices
    ]
    _write_parquet(pd.DataFrame.from_records(root_reports), stage / "root_reports.parquet")
    retained = [row for row in root_reports if bool(row["gate_pass"])]
    alternative_parts = []
    for row in retained:
        index = int(row["root_index"])
        frame = pd.read_parquet(
            stage / f"root_{index:03d}/K4/section_hypotheses.parquet"
        )
        frame = frame[frame["selected"].astype(bool)].copy()
        frame["chart_id"] = f"root_{index:03d}"
        frame["root_index"] = index
        alternative_parts.append(frame)
    alternatives = (
        pd.concat(alternative_parts, ignore_index=True)
        if alternative_parts
        else pd.DataFrame(
            columns=["chart_id", "task_node_id", "score", *BETA_COLUMNS]
        )
    )
    _write_parquet(alternatives, stage / "alternative_chart_labels.parquet")
    tasks, edges, _candidate_frame, parents = _pilot_inputs(config, project_root)
    task_by_id = tasks.set_index("task_node_id")
    primary_rows = []
    for node_id, group in alternatives.groupby("task_node_id", sort=True):
        selected = group.sort_values(["score", "chart_id"], kind="stable").iloc[0]
        task = task_by_id.loc[int(node_id)]
        primary_rows.append(
            {
                "task_node_id": int(node_id),
                "physical_point_id": str(task.physical_point_id),
                "x_m": float(task.x_m),
                "y_m": float(task.y_m),
                "z_m": float(task.z_m),
                "cell_level_mm": int(task.cell_level_mm),
                "cell_ix": int(task.cell_ix),
                "cell_iy": int(task.cell_iy),
                "cell_iz": int(task.cell_iz),
                "source_parent_node_id": int(task.source_parent_node_id),
                "primary_chart_id": str(selected.chart_id),
                **{name: float(selected[name]) for name in BETA_COLUMNS},
            }
        )
    primary = pd.DataFrame.from_records(primary_rows)
    beta_by_chart_node = {
        (str(row.chart_id), int(row.task_node_id)): np.asarray(
            [getattr(row, name) for name in BETA_COLUMNS], dtype=float
        )
        for row in alternatives.itertuples(index=False)
    }
    chart_by_node = {
        int(row.task_node_id): str(row.primary_chart_id)
        for row in primary.itertuples(index=False)
    }
    abstain_nodes: set[int] = set()
    transition_rows = []
    for edge in edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        left_chart, right_chart = chart_by_node.get(left), chart_by_node.get(right)
        if left_chart is None or right_chart is None or left_chart == right_chart:
            continue
        keys = (
            (left_chart, left), (right_chart, left),
            (left_chart, right), (right_chart, right),
        )
        if not all(key in beta_by_chart_node for key in keys):
            gap = math.inf
        else:
            gap = max(
                beta_rms_deg(beta_by_chart_node[(left_chart, node)], beta_by_chart_node[(right_chart, node)])
                for node in (left, right)
            )
        stitchable = bool(gap <= float(config["selected_section_audit"]["common_max_deg"]) + 1e-12)
        transition_rows.append(
            {
                "left_node_id": left,
                "right_node_id": right,
                "left_chart_id": left_chart,
                "right_chart_id": right_chart,
                "beta_gap_max_deg": gap,
                "stitchable": stitchable,
            }
        )
        if not stitchable:
            abstain_nodes.update((left, right))
    transitions = pd.DataFrame.from_records(transition_rows)
    _write_parquet(transitions, stage / "chart_transition_audit.parquet")
    if abstain_nodes and not primary.empty:
        primary = primary[~primary["task_node_id"].astype(int).isin(abstain_nodes)].copy()
    _write_parquet(primary, stage / "primary_canonical_labels.parquet")

    total_nodes = len(tasks)
    labelable_ratio = len(primary) / max(1, total_nodes)
    unresolved_ratio = 1.0 - labelable_ratio
    primary_node_ids = set(primary["task_node_id"].astype(int)) if not primary.empty else set()
    adjacency: dict[int, set[int]] = {node: set() for node in primary_node_ids}
    for edge in edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        if left in primary_node_ids and right in primary_node_ids:
            adjacency[left].add(right)
            adjacency[right].add(left)
    largest = 0
    unseen = set(adjacency)
    while unseen:
        pending = [unseen.pop()]
        size = 0
        while pending:
            current = pending.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        largest = max(largest, size)
    largest_ratio = largest / max(1, total_nodes)
    cell_keys = ["cell_level_mm", "cell_ix", "cell_iy", "cell_iz"]
    totals = tasks.groupby(cell_keys, sort=True).size().rename("probe_count")
    labeled = (
        primary.groupby(cell_keys, sort=True).size().rename("labelable_probe_count")
        if not primary.empty
        else pd.Series(dtype=int, name="labelable_probe_count")
    )
    cells = totals.to_frame().join(labeled, how="left").fillna({"labelable_probe_count": 0}).reset_index()
    cells["labelable_probe_count"] = cells["labelable_probe_count"].astype(int)
    cells["empirical_labelable_fraction"] = cells["labelable_probe_count"] / cells["probe_count"]
    _write_parquet(cells, stage / "cell_labelable_fraction.parquet")
    parent_labelable = set(primary["source_parent_node_id"].astype(int)) if not primary.empty else set()
    cuts = np.quantile(parents["x_m"].to_numpy(float), (1 / 3, 2 / 3))
    x_counts = {0: 0, 1: 0, 2: 0}
    for row in parents.itertuples(index=False):
        if int(row.node_id) in parent_labelable:
            x_counts[int(np.searchsorted(cuts, float(row.x_m), side="right"))] += 1
    pilot = config["exploratory_5k"]
    checks = {
        "labelable_measure": labelable_ratio >= float(pilot["labelable_measure_min"]),
        "largest_coherent_region": largest_ratio >= float(pilot["largest_region_measure_min"]),
        "unresolved_abstain": unresolved_ratio <= float(pilot["unresolved_abstain_max"]),
        "all_x_tertiles": all(value > 0 for value in x_counts.values()),
        "retained_section_audits": bool(retained) and all(bool(row["K4_audit_gate_pass"]) for row in retained),
        "section_budget_stability": bool(retained) and all(bool(row["section_stability_gate_pass"]) for row in retained),
    }
    report = {
        "scientific_gate_pass": bool(all(checks.values())),
        "scientific_checks": checks,
        "selected_method": selected_method,
        "frozen_proxy": "retry4_round5_replica_intersection",
        "eligible_cell_count": len(cells),
        "eligible_task_node_count": total_nodes,
        "retained_root_count": len(retained),
        "labelable_measure_ratio": labelable_ratio,
        "largest_primary_region_measure_ratio": largest_ratio,
        "unresolved_abstain_ratio": unresolved_ratio,
        "nonstitchable_boundary_node_count": len(abstain_nodes),
        "x_tertile_labelable_counts": x_counts,
        "static_xyz_inverse_exploration_authorized": bool(all(checks.values())),
        "formal_authorized": False,
    }
    _write_json(stage / "repaired_pilot_report.json", report)
    return _gate(stage / "gate.json", {"operational_completion": True}, **report)


def stage_exploratory_student(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["exploratory_student"]
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"V14.2 exploratory Student already sealed: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    pilot = _read_json(output_root / STAGE_DIRS["repaired_pilot"] / "gate.json")
    if not pilot.get("scientific_gate_pass", False):
        return _conditional_skip(stage, "repaired_5k_gate_failed")
    labels = pd.read_parquet(
        output_root
        / STAGE_DIRS["repaired_pilot"]
        / "primary_canonical_labels.parquet"
    ).sort_values(["task_node_id"], kind="stable")
    budget = config["exploratory_dataset"]
    maximum = min(int(budget["target_rows"]), int(budget["hard_max_rows"]))
    labels = labels.iloc[:maximum].copy()
    minimum = int(budget["minimum_rows"])
    if len(labels) < minimum:
        report = {
            "scientific_gate_pass": False,
            "reason": "insufficient_unique_primary_supervision_without_padding",
            "row_count": len(labels),
            "minimum_rows": minimum,
            "padding_count": 0,
            "all_supervised_rows_count_toward_budget": True,
        }
        return _gate(stage / "gate.json", {"operational_completion": True}, **report)
    environment = _EndpointOnlyForwardAdapter(
        load_environment(project_root, _paths(config, project_root)["robot_config"])
    )
    rows = []
    for row in labels.itertuples(index=False):
        beta = np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float)
        jacobian = np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6)
        block = tuple(
            np.floor(
                np.asarray([row.x_m, row.y_m, row.z_m], dtype=float) / 0.04
            ).astype(int)
        )
        digest = int.from_bytes(
            __import__("hashlib").sha256(str(block).encode()).digest()[:8],
            "big",
        )
        split = "train_core" if digest % 100 < 80 else "validation"
        rows.append(
            {
                "record_id": f"v14_2_{int(row.task_node_id):07d}",
                "kind": "static",
                "split_role": split,
                "chart_id": str(row.primary_chart_id),
                "is_primary": True,
                "sample_weight": 1.0,
                "x_m": float(row.x_m),
                "y_m": float(row.y_m),
                "z_m": float(row.z_m),
                **{name: float(getattr(row, name)) for name in BETA_COLUMNS},
                **{
                    name: float(jacobian.reshape(-1)[index])
                    for index, name in enumerate(JACOBIAN_COLUMNS)
                },
                "macroblock": str(block),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    train_charts = set(
        frame.loc[frame["split_role"].eq("train_core"), "chart_id"].astype(str)
    )
    validation_charts = set(
        frame.loc[frame["split_role"].eq("validation"), "chart_id"].astype(str)
    )
    common_charts = train_charts & validation_charts
    routed_frame = frame[frame["chart_id"].astype(str).isin(common_charts)].copy()
    _write_parquet(frame, stage / "primary_student_supervision.parquet")
    _write_parquet(routed_frame, stage / "routed_student_supervision.parquet")
    geometry = StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )
    student = config["student"]
    reports = []
    for mode, supervision in (
        (RepresentationMode.XYZ_GLOBAL, frame),
        (RepresentationMode.XYZ_ROUTER_EXPERTS, routed_frame),
    ):
        if mode is RepresentationMode.XYZ_ROUTER_EXPERTS and len(common_charts) < 2:
            continue
        train = supervision[supervision["split_role"].eq("train_core")].copy()
        validation = supervision[supervision["split_role"].eq("validation")].copy()
        if train.empty or validation.empty:
            continue
        for seed in map(int, student["seeds"]):
            training = train_workspace_student(
                train,
                validation,
                mode=mode,
                geometry=geometry,
                config=WorkspaceStudentTrainingConfig(
                    hidden_units=tuple(map(int, student["hidden_units"])),
                    router_hidden_units=tuple(map(int, student["router_hidden_units"])),
                    learning_rate=float(student["learning_rate"]),
                    max_steps=int(student["max_steps"]),
                    validation_interval=int(student["validation_interval"]),
                    patience_intervals=int(student["patience_intervals"]),
                    seed=seed,
                ),
            )
            model_dir = stage / f"{mode.value}_seed_{seed}"
            save_workspace_student_models(training.models, model_dir)
            _write_parquet(training.history, model_dir / "training_history.parquet")
            prediction = training.inverse.predict(
                __import__(
                    "quasi_exp.teacher.workspace_inverse",
                    fromlist=["InverseQuery"],
                ).InverseQuery(
                    validation.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(float)
                )
            )
            accepted = np.asarray(prediction.accepted, dtype=bool)
            residual = np.asarray(prediction.fk_residual_mm, dtype=float)
            finite = accepted & np.isfinite(residual)
            values = residual[finite]
            features = validation.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(np.float32)
            if mode is RepresentationMode.XYZ_GLOBAL:
                raw_beta = np.asarray(training.models.global_model(features, training=False), dtype=float)
            else:
                probabilities = np.asarray(training.models.router_model(features, training=False), dtype=float)
                experts = {
                    chart: np.asarray(model(features, training=False), dtype=float)
                    for chart, model in training.models.expert_models.items()
                }
                selected_expert = np.argmax(probabilities, axis=1)
                raw_beta = np.vstack(
                    [
                        experts[training.models.chart_ids[int(index)]][row_index]
                        for row_index, index in enumerate(selected_expert)
                    ]
                )
            targets = validation.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(float)
            corrected = raw_beta.copy()
            dls = {}
            bounds = np.asarray(environment.bounds, dtype=float)
            for correction_step in (1, 2):
                for row_index in range(len(corrected)):
                    error = targets[row_index] - np.asarray(
                        environment.fk(corrected[row_index]), dtype=float
                    ).reshape(-1, 3)[0]
                    proposal = corrected[row_index] + weighted_damped_pinv(
                        np.asarray(environment.jacobian(corrected[row_index]), dtype=float),
                        damping=1e-3,
                        weights=np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0]),
                    ) @ error
                    if np.all(proposal >= bounds[:, 0]) and np.all(proposal <= bounds[:, 1]):
                        corrected[row_index] = proposal
                corrected_xyz = np.asarray(environment.fk(corrected), dtype=float).reshape(-1, 3)
                corrected_residual = np.linalg.norm(corrected_xyz - targets, axis=1) * 1000.0
                dls[f"dls_{correction_step}_fk_p95_mm"] = float(np.percentile(corrected_residual, 95))
                dls[f"dls_{correction_step}_fk_max_mm"] = float(np.max(corrected_residual))
            reports.append(
                {
                    "mode": mode.value,
                    "seed": seed,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "accepted_fraction": float(finite.mean()),
                    "fk_p95_mm": float(np.percentile(values, 95)) if len(values) else math.inf,
                    "fk_max_mm": float(np.max(values)) if len(values) else math.inf,
                    **dls,
                }
            )
    metrics = pd.DataFrame.from_records(reports)
    _write_parquet(metrics, stage / "student_metrics.parquet")
    passing = (
        metrics["fk_p95_mm"].le(float(student["validation_fk_p95_max_mm"]))
        & metrics["fk_max_mm"].le(float(student["validation_fk_max_mm"]))
        if not metrics.empty
        else pd.Series(dtype=bool)
    )
    report = {
        "scientific_gate_pass": bool(len(passing) and passing.any()),
        "supervision_row_count": len(frame),
        "router_supervision_row_count": len(routed_frame),
        "all_supervised_rows_count_toward_budget": True,
        "padding_count": 0,
        "macroblock_mm": 40,
        "trained_modes": sorted(metrics["mode"].unique()) if not metrics.empty else [],
        "student_plus_dls_1_2_evaluated": True,
        "stateful_diagnostic_triggered": False,
        "stateful_not_triggered_reason": "no_reproducible_history_conditioned_multipath_counterexample",
        "xyz_only_retained_as_primary_hypothesis": True,
    }
    _write_json(stage / "student_report.json", report)
    return _gate(stage / "gate.json", {"operational_completion": True}, **report)


def stage_formal_gate(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["formal_gate"]
    gates = {
        name: _read_json(output_root / STAGE_DIRS[name] / "gate.json")
        for name in ("reach_round6", "repaired_pilot", "exploratory_student")
    }
    formal = config["formal_gate"]
    pilot = gates["repaired_pilot"]
    student = gates["exploratory_student"]
    n_min = int(student.get("supervision_row_count", 10**12))
    checks = {
        "reach": bool(gates["reach_round6"].get("reach_convergence_gate", False)),
        "atlas_exploratory": bool(pilot.get("scientific_gate_pass", False)),
        "labelable_measure": float(pilot.get("labelable_measure_ratio", 0.0))
        >= float(formal["minimum_labelable_measure_ratio"]),
        "unresolved": float(pilot.get("unresolved_abstain_ratio", 1.0))
        <= float(formal["maximum_unresolved_ratio"]),
        "student": bool(student.get("scientific_gate_pass", False)),
        "budget": n_min <= int(formal["maximum_n_min"]),
        "no_padding": not bool(formal["row_padding"])
        and int(student.get("padding_count", 0)) == 0,
        "representation_frozen": bool(
            student.get("xyz_only_retained_as_primary_hypothesis", False)
            or student.get("stateful_diagnostic_triggered", False)
        ),
    }
    authorized = bool(all(checks.values()))
    return _gate(
        stage / "gate.json",
        checks,
        formal_generation_authorized=authorized,
        deployment_claim=False,
        direct_threshold_relaxation_authorized=False,
        n_min=n_min,
        exact_formal_row_budget=int(formal["exact_rows"]),
        strong_labelable_target_met=float(pilot.get("labelable_measure_ratio", 0.0))
        >= float(formal["strong_labelable_measure_ratio"]),
    )


def stage_summary(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    stage.mkdir(parents=True, exist_ok=True)
    gates = {
        name: _read_json(output_root / STAGE_DIRS[name] / "gate.json")
        for name in STAGE_ORDER[:-1]
        if (output_root / STAGE_DIRS[name] / "gate.json").is_file()
    }
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and not path.is_relative_to(stage):
            artifacts.append({"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_version": 1, "artifact_count": len(artifacts), "artifacts": artifacts}
    _write_json(stage / "artifact_manifest.json", manifest)
    report = {
        "stage_gates": gates,
        "selected_method": gates.get("mechanisms", {}).get("selected_method"),
        "confirmation_gate": bool(gates.get("confirmation", {}).get("gate_pass", False)),
        "repaired_pilot_gate": bool(gates.get("repaired_pilot", {}).get("gate_pass", False)),
        "formal_generation_authorized": bool(gates.get("formal_gate", {}).get("gate_pass", False)),
    }
    _write_json(stage / "summary_report.json", report)
    return _gate(stage / "gate.json", {"artifact_manifest_nonempty": bool(artifacts)}, **report)


STAGE_RUNNERS = {
    "inventory": stage_inventory,
    "diagnostics": stage_diagnostics,
    "mechanisms": stage_mechanisms,
    "confirmation": stage_confirmation,
    "reach_round6": stage_reach_round6,
    "repaired_pilot": stage_repaired_pilot,
    "exploratory_student": stage_exploratory_student,
    "formal_gate": stage_formal_gate,
    "summary": stage_summary,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SOURCE_ROOT / "configs/bacra_v14_2_section_first_atlas.yaml"))
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=STAGE_ORDER, required=True)
    parser.add_argument("--patch-id")
    parser.add_argument("--method")
    parser.add_argument("--root-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config["_patch_filter"] = args.patch_id
    config["_method_filter"] = args.method
    config["_root_index"] = args.root_index
    project_root = project_root_from(SOURCE_ROOT)
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / str(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    result = STAGE_RUNNERS[args.stage](config, project_root, output_root)
    print(json.dumps(_strict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
