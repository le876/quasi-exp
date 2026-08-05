#!/usr/bin/env python3
"""Run the BACRA V14.1 cross-cell atlas repair experiment.

The runner consumes the sealed V14 5k Pilot as immutable E0 evidence.  It
creates new artifacts under ``runs/bacra_v14_1_*`` and never modifies the V14
Pilot.  Every scientific stage is create-once and downstream stages stop with
an explicit skip artifact when an upstream Gate is not satisfied.
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

from quasi_exp.teacher.atlas_audit import AtlasAuditPolicy
from quasi_exp.teacher.canonical import beta_rms_deg, weighted_damped_pinv
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    ContinuationOutcome,
    build_canonical_atlas_from_product_graph,
    make_predictor_corrector_continuation,
)
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.workspace_atlas import WorkspaceAtlasPolicy, RepresentationMode
from quasi_exp.teacher.workspace_atlas_integration import (
    WorkspaceAtlasIntegrationPolicy,
    build_workspace_atlas_integration,
)
from quasi_exp.teacher.workspace_atlas_repair import (
    CanonicalSelectionObservation,
    CrossCellRepairPolicy,
    RepairAblation,
    evaluate_canonical_selection_stability,
    evaluate_formal_admission_gate,
    evaluate_patch_ablation_gate,
    evaluate_repaired_pilot_gate,
    make_segmented_continuation,
    run_cross_cell_repair_ablation,
    select_connected_repair_patches,
    select_frontier_enrichment_nodes,
)
from quasi_exp.teacher.workspace_candidate_bank import (
    CandidatePolicy,
    CandidateQuality,
    CandidateSearchMode,
    solve_candidate_bank,
    stable_cluster_representatives,
)
from quasi_exp.teacher.workspace_protocol import (
    WorkspaceAtlasFramePolicy,
    correct_static_targets_from_primary_sections,
    supervision_records_frame,
    supervision_records_from_atlas_frames,
)
from quasi_exp.teacher.workspace_dataset import (
    MacroblockSplitPolicy,
    SplitRole,
    SupervisionBudgetPolicy,
    SupervisionMaterializer,
    SupervisionPriority,
)
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    records_to_workspace_student_frame,
    save_workspace_student_models,
    train_workspace_student,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_inverse import InverseQuery
from quasi_exp.teacher.workspace_experiment import (
    ReachSamplingRoundSpec,
    generate_independent_reach_samples,
    frontier_candidates,
    cell_centers,
)
from quasi_exp.teacher.workspace_reach import (
    FrontierProbeEvidence,
    ReachProxyBuilder,
    ReachProxyConfig,
    ReachReplica,
    ReachReplicaRound,
    WorkspaceGridSpec,
    CellKey,
)
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("x_m", "y_m", "z_m")
STAGE_DIRS = {
    "inventory": "00_patch_inventory",
    "ablations": "01_patch_ablations",
    "stability": "02_selection_stability",
    "reach_extension": "03_reach_extension",
    "repaired_pilot": "04_repaired_5k_pilot",
    "exploratory_student": "05_exploratory_student",
    "formal_gate": "06_formal_admission",
    "summary": "07_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def project_root_from(source_root: Path) -> Path:
    resolved = source_root.resolve()
    if ".worktrees" in resolved.parts:
        index = resolved.parts.index(".worktrees")
        return Path(*resolved.parts[:index])
    return resolved


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
    atomic_write_json(path, _strict(dict(value)))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _gate(path: Path, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    payload = {
        "gate_pass": bool(checks and all(bool(value) for value in checks.values())),
        "checks": {str(key): bool(value) for key, value in checks.items()},
        **evidence,
    }
    _write_json(path, payload)
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("V14.1 config root must be a mapping")
    config = dict(value)
    config["config_path"] = str(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported V14.1 config schema")
    repair = config["repair"]
    if tuple(repair["ablations"]) != tuple(item.value for item in RepairAblation):
        raise ValueError("V14.1 requires the registered nested E0--E4 ablations")
    reach_rounds = tuple(config["reach_extension"]["rounds"])
    seeds = [int(row[key]) for row in reach_rounds for key in ("seed_a", "seed_b")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("V14.1 Reach A/B extension seeds must be disjoint")
    return config


def _source_paths(config: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    original = project_root / str(config["sources"]["original_pilot_root"])
    return {
        "reviewed_plan": project_root / str(config["sources"]["reviewed_plan"]),
        "original_artifact_manifest": original / "11_summary/artifact_manifest.json",
        "original_source_manifest": original / "00_protocol/source_manifest.json",
        "parent_task_nodes": original / "02_domain_registry/pilot_task_nodes.parquet",
        "parent_task_edges": original / "02_domain_registry/pilot_task_edges.parquet",
        "parent_task_probes": original / "02_domain_registry/pilot_task_probes.parquet",
        "atlas_task_nodes": original / "04_workspace_atlas/atlas_task_nodes.parquet",
        "atlas_task_edges": original / "04_workspace_atlas/atlas_task_edges.parquet",
        "atlas_candidates": original / "04_workspace_atlas/atlas_candidates.parquet",
        "candidate_clusters": original / "04_workspace_atlas/candidate_clusters.parquet",
        "product_edges": original / "04_workspace_atlas/product_edges.parquet",
        "saturation_candidates": original / "03_candidate_bank/saturation_candidates.parquet",
        "saturation_task_nodes": original / "03_candidate_bank/saturation_task_nodes.parquet",
        "replica_a_slab": original / "01_workspace_proxy/replica_a_slab.parquet",
        "replica_b_slab": original / "01_workspace_proxy/replica_b_slab.parquet",
        "tip_pool": original / "01_workspace_proxy/tip_pool.parquet",
        "frontier_inverse_probes": original / "01_workspace_proxy/frontier_inverse_probes.parquet",
        "workspace_proxy_report": original / "01_workspace_proxy/workspace_proxy_report.json",
        "domain_cells": original / "01_workspace_proxy/domain_cells.parquet",
        "robot_config": SOURCE_ROOT / str(config["robot_config"]),
    }


def _repair_policy(config: Mapping[str, Any]) -> CrossCellRepairPolicy:
    repair = config["repair"]
    patches = config["patches"]
    return CrossCellRepairPolicy(
        shared_face_probe_count=int(patches["shared_face_probe_count"]),
        cartesian_step_max_mm=float(repair["cartesian_step_max_mm"]),
        propagated_cluster_deg=float(repair["propagated_cluster_deg"]),
        reverse_return_max_deg=float(repair["reverse_return_max_deg"]),
        continuation_residual_max_mm=float(repair["continuation_residual_max_mm"]),
        maximum_waves=int(repair["maximum_waves"]),
        maximum_propagated_candidates_per_node=int(
            repair["maximum_propagated_candidates_per_node"]
        ),
        maximum_candidates_per_lineage_per_node=int(
            repair["maximum_candidates_per_lineage_per_node"]
        ),
        icm_max_sweeps=int(config["atlas"]["icm_max_sweeps"]),
        patch_size=int(patches["cells_per_patch"]),
        patch_count=int(patches["count"]),
        development_patch_count=int(patches["development_count"]),
        severe_enrichment_start_count=int(repair["severe_starts"]),
        ordinary_enrichment_start_count=int(repair["ordinary_starts"]),
        severe_enrichment_fraction_max=float(repair["severe_enrichment_fraction_max"]),
        enrichment_fraction_max=float(repair["enrichment_fraction_max"]),
        enrichment_node_max=int(repair["enrichment_node_max"]),
    )


def _atlas_policy(config: Mapping[str, Any]) -> AtlasPolicy:
    atlas = config["atlas"]
    return AtlasPolicy(
        edge_match_deg=float(atlas["edge_match_deg"]),
        continuation_residual_max_mm=float(config["repair"]["continuation_residual_max_mm"]),
        root_count=int(atlas["root_count"]),
        icm_max_sweeps=int(atlas["icm_max_sweeps"]),
        top_section_count=int(atlas["top_section_count"]),
        split_gap_deg=float(atlas["split_gap_deg"]),
        merge_overlap_p95_deg=float(atlas["merge_overlap_p95_deg"]),
        merge_overlap_max_deg=float(atlas["merge_overlap_max_deg"]),
    )


def _integration_policy(config: Mapping[str, Any]) -> WorkspaceAtlasIntegrationPolicy:
    atlas = config["atlas"]
    return WorkspaceAtlasIntegrationPolicy(
        atlas_policy=_atlas_policy(config),
        audit_policy=AtlasAuditPolicy(
            endpoint_count=int(atlas["audit_endpoint_count"]),
            paths_per_endpoint=int(atlas["audit_paths_per_endpoint"]),
            loop_count=int(atlas["audit_loop_count"]),
            path_p95_deg=float(atlas["cycle_p95_deg"]),
            loop_p95_deg=float(atlas["cycle_p95_deg"]),
            direction_p95_deg=float(atlas["cycle_p95_deg"]),
            repeat_p95_deg=0.2,
            common_max_deg=float(atlas["cycle_max_deg"]),
        ),
        workspace_policy=WorkspaceAtlasPolicy(
            stitchable_p95_deg=float(atlas["merge_overlap_p95_deg"]),
            stitchable_max_deg=float(atlas["merge_overlap_max_deg"]),
            nonstitchable_p95_deg=float(atlas["split_gap_deg"]),
            cycle_p95_deg=float(atlas["cycle_p95_deg"]),
            cycle_max_deg=float(atlas["cycle_max_deg"]),
            minimum_primary_measure_coverage=0.0,
            minimum_x_bin_coverage=0.0,
            maximum_abstention_measure_ratio=1.0,
            branch_overall_wilson_upper=1.0,
            branch_stratum_wilson_upper=1.0,
        ),
        candidate_cluster_deg=float(config["repair"]["propagated_cluster_deg"]),
        piecewise_primary_partition=True,
    )


def _candidates_from_frame(frame: pd.DataFrame) -> tuple[AtlasCandidate, ...]:
    rows: list[AtlasCandidate] = []
    for row in frame.sort_values(["task_node_id", "candidate_id"], kind="stable").itertuples(index=False):
        rows.append(
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
                cluster_id=(None if not hasattr(row, "cluster_id") else int(row.cluster_id)),
                diagnostics={"source": str(getattr(row, "source", "sealed_v14_cluster"))},
            )
        )
    return tuple(rows)


def _candidate_frame(candidates: Sequence[AtlasCandidate]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "task_node_id": row.node_id,
                "candidate_id": row.candidate_id,
                "source": str(row.diagnostics.get("source", "v14_1_propagated")),
                "residual_mm": row.residual_mm,
                "min_margin_deg": row.min_margin_deg,
                "normalized_min_margin": row.normalized_min_margin,
                "posture_cost": row.posture_cost,
                "condition_number": row.condition_number,
                "quality": row.quality,
                "solver_success": row.solver_success,
                "actual_bounds": row.actual_bounds,
                **{
                    name: float(row.beta_rad[index])
                    for index, name in enumerate(BETA_COLUMNS)
                },
            }
            for row in candidates
        ]
    )


def _graph_edge_frame(graph: Any) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "left_task_node_id": edge.left_key[0],
                "left_candidate_id": edge.left_key[1],
                "right_task_node_id": edge.right_key[0],
                "right_candidate_id": edge.right_key[1],
                "cost": edge.cost,
                "transition_deg": edge.transition_deg,
                "forward_status": edge.forward.status,
                "reverse_status": edge.reverse.status,
            }
            for edge in graph.robust_edges
        ]
    )


def _input_closure(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    paths = _source_paths(config, project_root)
    expected = dict(config["sources"]["expected_sha256"])
    actual: dict[str, str] = {}
    for key in expected:
        path = paths[key]
        actual[key] = sha256_file(path) if path.is_file() else "missing"
    return {
        "paths": {key: str(paths[key]) for key in expected},
        "expected_sha256": expected,
        "actual_sha256": actual,
        "all_match": actual == expected,
    }


def stage_inventory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["inventory"]
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "gate.json").exists():
        raise FileExistsError(f"inventory stage is already sealed: {stage}")
    closure = _input_closure(config, project_root)
    if not closure["all_match"]:
        return _gate(stage / "gate.json", {"sealed_inputs": False}, input_closure=closure)
    paths = _source_paths(config, project_root)
    parent_nodes = pd.read_parquet(paths["parent_task_nodes"])
    parent_edges = pd.read_parquet(paths["parent_task_edges"])
    task_nodes = pd.read_parquet(paths["atlas_task_nodes"])
    candidates = pd.read_parquet(paths["candidate_clusters"])
    atlas_candidates = pd.read_parquet(paths["atlas_candidates"])
    product_edges = pd.read_parquet(paths["product_edges"])
    representative = task_nodes[task_nodes["is_representative"].astype(bool)][
        ["task_node_id", "source_parent_node_id"]
    ]
    exact = atlas_candidates[atlas_candidates["candidate_id"].eq("capability_exact")][
        ["task_node_id", "condition_number"]
    ]
    condition = representative.merge(exact, on="task_node_id", validate="one_to_one").set_index(
        "source_parent_node_id"
    )["condition_number"].to_dict()
    parent_by_task = task_nodes.set_index("task_node_id")["source_parent_node_id"].to_dict()
    existing_cross: set[int] = set()
    for row in product_edges.itertuples(index=False):
        left = int(parent_by_task[int(row.left_task_node_id)])
        right = int(parent_by_task[int(row.right_task_node_id)])
        if left != right:
            existing_cross.update((left, right))
    inventory = select_connected_repair_patches(
        parent_nodes,
        parent_edges,
        condition_by_node=condition,
        existing_cross_edge_node_ids=existing_cross,
        policy=_repair_policy(config),
    )
    _write_parquet(inventory.assignments, stage / "patch_assignments.parquet")
    selected_parents = set(inventory.assignments["parent_node_id"].astype(int))
    patch_by_parent = inventory.assignments.set_index("parent_node_id")["patch_id"].to_dict()
    selected_tasks = task_nodes[
        task_nodes["source_parent_node_id"].astype(int).isin(selected_parents)
    ].copy()
    selected_tasks["patch_id"] = selected_tasks["source_parent_node_id"].map(patch_by_parent)
    selected_task_ids = set(selected_tasks["task_node_id"].astype(int))
    atlas_edges = pd.read_parquet(paths["atlas_task_edges"])
    selected_edges = atlas_edges[
        atlas_edges["left_node_id"].astype(int).isin(selected_task_ids)
        & atlas_edges["right_node_id"].astype(int).isin(selected_task_ids)
    ].copy()
    selected_edges["patch_id"] = selected_edges["left_node_id"].map(
        selected_tasks.set_index("task_node_id")["patch_id"]
    )
    right_patch = selected_edges["right_node_id"].map(
        selected_tasks.set_index("task_node_id")["patch_id"]
    )
    excluded_cross_patch_task_edges = int((right_patch != selected_edges["patch_id"]).sum())
    selected_edges = selected_edges[right_patch.eq(selected_edges["patch_id"])].copy()
    selected_candidates = candidates[
        candidates["task_node_id"].astype(int).isin(selected_task_ids)
    ].copy()
    selected_candidates["patch_id"] = selected_candidates["task_node_id"].map(
        selected_tasks.set_index("task_node_id")["patch_id"]
    )
    selected_product = product_edges[
        product_edges["left_task_node_id"].astype(int).isin(selected_task_ids)
        & product_edges["right_task_node_id"].astype(int).isin(selected_task_ids)
    ].copy()
    selected_product["patch_id"] = selected_product["left_task_node_id"].map(
        selected_tasks.set_index("task_node_id")["patch_id"]
    )
    product_right_patch = selected_product["right_task_node_id"].map(
        selected_tasks.set_index("task_node_id")["patch_id"]
    )
    excluded_cross_patch_product_edges = int(
        (product_right_patch != selected_product["patch_id"]).sum()
    )
    selected_product = selected_product[
        product_right_patch.eq(selected_product["patch_id"])
    ].copy()
    _write_parquet(selected_tasks, stage / "patch_task_nodes.parquet")
    _write_parquet(selected_edges, stage / "patch_task_edges_e0.parquet")
    _write_parquet(selected_candidates, stage / "patch_candidate_clusters.parquet")
    _write_parquet(selected_product, stage / "patch_product_edges_e0.parquet")
    git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_paths = tuple(
        sorted(
            {
                Path(__file__).resolve(),
                SOURCE_ROOT / "src/quasi_exp/teacher/workspace_atlas_repair.py",
                SOURCE_ROOT / "src/quasi_exp/teacher/workspace_atlas.py",
                SOURCE_ROOT / "src/quasi_exp/teacher/workspace_atlas_integration.py",
                Path(config["config_path"]).resolve(),
            }
        )
    )
    implementation_sources = {
        path.relative_to(SOURCE_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in implementation_paths
    }
    protocol = {
        "protocol_id": config["protocol_id"],
        "source_git_sha": git_sha,
        "source_root": str(SOURCE_ROOT),
        "project_root": str(project_root),
        "config_path": str(config["config_path"]),
        "config_sha256": sha256_file(config["config_path"]),
        "input_closure": closure,
        "patch_inventory": dict(inventory.audit),
        "runtime": runtime_fingerprint(),
        "implementation_sources": implementation_sources,
    }
    _write_json(stage / "protocol.json", protocol)
    return _gate(
        stage / "gate.json",
        {
            "sealed_inputs": closure["all_match"],
            "patch_inventory": bool(inventory.audit["gate_pass"]),
            "selected_task_exact_set": len(selected_tasks) == int(config["patches"]["count"])
            * int(config["patches"]["cells_per_patch"])
            * 5,
        },
        patch_count=inventory.audit["patch_count"],
        cell_count=inventory.audit["cell_count"],
        task_node_count=len(selected_tasks),
        candidate_count=len(selected_candidates),
        excluded_cross_patch_task_edge_count=excluded_cross_patch_task_edges,
        excluded_cross_patch_product_edge_count=excluded_cross_patch_product_edges,
        source_git_sha=git_sha,
        input_closure=closure,
        patch_inventory=dict(inventory.audit),
    )


def _require_stage(output_root: Path, stage_name: str) -> dict[str, Any]:
    path = output_root / STAGE_DIRS[stage_name] / "gate.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing prerequisite stage: {path}")
    gate = _read_json(path)
    if not gate.get("gate_pass", False):
        raise RuntimeError(f"prerequisite stage failed: {stage_name}")
    return gate


def _enrich_candidates(
    task_frame: pd.DataFrame,
    base_candidates: Sequence[AtlasCandidate],
    graph: Any,
    environment: Any,
    config: Mapping[str, Any],
) -> tuple[AtlasCandidate, ...]:
    task_frame = task_frame.copy()
    condition_by_node: dict[int, list[float]] = {}
    for candidate in base_candidates:
        condition_by_node.setdefault(candidate.node_id, []).append(candidate.condition_number)
    task_frame["condition_number"] = task_frame["task_node_id"].map(
        lambda node_id: float(np.median(condition_by_node.get(int(node_id), (0.0,))))
    )
    requests = select_frontier_enrichment_nodes(
        task_frame,
        graph,
        policy=_repair_policy(config),
    )
    if not len(requests):
        return tuple(base_candidates)
    node_by_id = {int(row.task_node_id): row for row in task_frame.itertuples(index=False)}
    candidates_by_node: dict[int, list[AtlasCandidate]] = {}
    for candidate in base_candidates:
        candidates_by_node.setdefault(candidate.node_id, []).append(candidate)
    additions: list[AtlasCandidate] = []
    for request in requests.itertuples(index=False):
        global_node_id = int(request.task_node_id)
        target = node_by_id[global_node_id]
        graph_node = graph.node_by_id[global_node_id]
        seed_node_ids = (global_node_id, *graph_node.neighbor_node_ids)
        source_candidates = sorted(
            (
                candidate
                for seed_node_id in seed_node_ids
                for candidate in candidates_by_node.get(seed_node_id, ())
            ),
            key=lambda item: (item.node_id != global_node_id, item.node_id, item.candidate_id),
        )
        if not source_candidates:
            continue
        unique_seeds: list[np.ndarray] = []
        for candidate in source_candidates:
            if not any(beta_rms_deg(candidate.beta_rad, seed) <= 1.0e-9 for seed in unique_seeds):
                unique_seeds.append(candidate.beta_rad)
        seed_budget = int(request.start_budget)
        policy = CandidatePolicy(
            candidate_budget_per_node=seed_budget,
            difficult_candidate_budget_per_node=seed_budget,
            candidate_seed_budget_per_node=seed_budget,
            difficult_seed_budget_per_node=seed_budget,
            nullspace_seed_budget_per_node=(
                int(config["repair"]["severe_nullspace_starts"])
                if bool(request.severe)
                else int(config["repair"]["ordinary_nullspace_starts"])
            ),
            search_mode=CandidateSearchMode.DIVERSITY,
            solver_names=tuple(map(str, config["repair"]["solvers"])),
            max_corrector_iterations=100,
            tracking_tolerance_mm=1.0,
            max_residual_mm=float(config["repair"]["continuation_residual_max_mm"]),
            gold_margin_deg=1.5,
            silver_margin_deg=0.0,
            candidate_cluster_deg=float(config["repair"]["propagated_cluster_deg"]),
            nullspace_step_deg=1.0,
        )
        bank = solve_candidate_bank(
            environment,
            np.asarray([[target.x_m, target.y_m, target.z_m]], dtype=float),
            policy,
            neighbor_beta_rad={0: np.vstack(unique_seeds)},
            node_seed_beta_rad={0: unique_seeds[0]},
            difficult_node_ids=(0,) if bool(request.severe) else (),
        )
        for ordinal, record in enumerate(bank.records):
            if record.quality is CandidateQuality.REJECT:
                continue
            additions.append(
                AtlasCandidate(
                    node_id=global_node_id,
                    candidate_id=f"e4_{global_node_id:06d}_{ordinal:04d}",
                    beta_rad=record.beta_rad,
                    residual_mm=record.residual_mm,
                    min_margin_deg=record.min_margin_deg,
                    normalized_min_margin=record.normalized_min_margin,
                    posture_cost=float(np.linalg.norm(record.beta_rad)),
                    condition_number=float(record.diagnostics.get("kappa", 0.0)),
                    quality=record.quality.value,
                    solver_success=record.solver_success,
                    actual_bounds=True,
                    diagnostics={
                        "source": "e4_frontier_diversity",
                        "request_flags": str(request.flags),
                    },
                )
            )
    return tuple(base_candidates) + tuple(additions)


def _e0_edge_key_frame(frame: pd.DataFrame) -> set[tuple[int, str, int, str]]:
    return {
        (
            int(row.left_task_node_id),
            str(row.left_candidate_id),
            int(row.right_task_node_id),
            str(row.right_candidate_id),
        )
        for row in frame.itertuples(index=False)
    }


def stage_ablations(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "inventory")
    stage = output_root / STAGE_DIRS["ablations"]
    stage.mkdir(parents=True, exist_ok=True)
    inventory = output_root / STAGE_DIRS["inventory"]
    tasks = pd.read_parquet(inventory / "patch_task_nodes.parquet")
    edges = pd.read_parquet(inventory / "patch_task_edges_e0.parquet")
    candidate_frame = pd.read_parquet(inventory / "patch_candidate_clusters.parquet")
    sealed_product = pd.read_parquet(inventory / "patch_product_edges_e0.parquet")
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    base_continuation = make_predictor_corrector_continuation(
        environment,
        residual_tolerance_mm=float(config["repair"]["continuation_residual_max_mm"]),
    )
    reports: list[dict[str, Any]] = []
    all_patch_ids = tuple(sorted(tasks["patch_id"].unique()))
    patch_filter = config.get("_patch_filter")
    if patch_filter is None:
        pending = [
            patch_id
            for patch_id in all_patch_ids
            if not (stage / str(patch_id) / "patch_complete.json").is_file()
        ]
        if pending:
            environment = os.environ.copy()
            environment.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "MPLCONFIGDIR": "/tmp/mpl-bacra-v14-1",
                }
            )
            waiting = list(pending)
            running: list[tuple[str, subprocess.Popen[str], Any]] = []
            failures: list[dict[str, Any]] = []
            worker_count = min(int(config["parallel"]["patch_workers"]), len(waiting))
            while waiting or running:
                while waiting and len(running) < worker_count:
                    patch_id = waiting.pop(0)
                    log_path = stage / f"{patch_id}.log"
                    handle = log_path.open("w", encoding="utf-8")
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--config", str(config["config_path"]),
                        "--output-root", str(output_root),
                        "--stage", "ablations",
                        "--patch-id", str(patch_id),
                    ]
                    process = subprocess.Popen(
                        command,
                        cwd=SOURCE_ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    running.append((str(patch_id), process, handle))
                next_running: list[tuple[str, subprocess.Popen[str], Any]] = []
                for patch_id, process, handle in running:
                    status = process.poll()
                    if status is None:
                        next_running.append((patch_id, process, handle))
                        continue
                    handle.close()
                    if status != 0:
                        failures.append({"patch_id": patch_id, "returncode": status})
                running = next_running
                if failures:
                    for _patch_id, process, handle in running:
                        process.terminate()
                        process.wait(timeout=30)
                        handle.close()
                    raise RuntimeError(f"patch ablation workers failed: {failures}")
                if waiting or running:
                    time.sleep(1.0)
        reports = [
            _read_json(stage / str(patch_id) / f"{ablation.value}_report.json")
            for patch_id in all_patch_ids
            for ablation in RepairAblation
        ]
    patch_ids_to_run = all_patch_ids if patch_filter is None else (str(patch_filter),)
    for patch_id in (() if patch_filter is None else patch_ids_to_run):
        if patch_id not in all_patch_ids:
            raise ValueError(f"unknown repair patch ID: {patch_id}")
        patch_dir = stage / str(patch_id)
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_tasks = tasks[tasks["patch_id"].eq(patch_id)].drop(columns=["patch_id"])
        patch_task_ids = set(patch_tasks["task_node_id"].astype(int))
        patch_edges = edges[edges["patch_id"].eq(patch_id)].drop(columns=["patch_id"])
        patch_candidates_frame = candidate_frame[
            candidate_frame["patch_id"].eq(patch_id)
        ].drop(columns=["patch_id"])
        patch_candidates = _candidates_from_frame(patch_candidates_frame)
        patch_split = str(
            pd.read_parquet(inventory / "patch_assignments.parquet")
            .loc[lambda frame: frame["patch_id"].eq(patch_id), "patch_split"]
            .iloc[0]
        )
        previous_graph = None
        for ablation in RepairAblation:
            report_path = patch_dir / f"{ablation.value}_report.json"
            e4_report = patch_dir / f"{RepairAblation.E4_FRONTIER_ENRICHMENT.value}_report.json"
            if report_path.is_file() and not (
                ablation is RepairAblation.E3_DYNAMIC_INSERTION
                and not e4_report.is_file()
            ):
                reports.append(_read_json(report_path))
                continue
            candidates_for_method = patch_candidates
            if ablation is RepairAblation.E4_FRONTIER_ENRICHMENT:
                if previous_graph is None:
                    raise RuntimeError("E4 requires the completed E3 graph")
                candidates_for_method = _enrich_candidates(
                    patch_tasks,
                    patch_candidates,
                    previous_graph,
                    environment,
                    config,
                )
            started = time.time()
            result = run_cross_cell_repair_ablation(
                ablation,
                patch_tasks,
                patch_edges,
                candidates_for_method,
                base_continuation,
                atlas_policy=_atlas_policy(config),
                repair_policy=_repair_policy(config),
            )
            if ablation is RepairAblation.E3_DYNAMIC_INSERTION:
                previous_graph = result.graph
            result_candidates = _candidate_frame(result.graph.candidates)
            _write_parquet(result.task_edges, patch_dir / f"{ablation.value}_task_edges.parquet")
            _write_parquet(result_candidates, patch_dir / f"{ablation.value}_candidates.parquet")
            graph_edges = _graph_edge_frame(result.graph)
            _write_parquet(graph_edges, patch_dir / f"{ablation.value}_product_edges.parquet")
            propagation = pd.DataFrame.from_records(
                [
                    {
                        **{
                            key: value
                            for key, value in asdict(row).items()
                            if key != "source_key"
                        },
                        "source_node_id": row.source_key[0],
                        "source_candidate_id": row.source_key[1],
                    }
                    for row in result.propagation_records
                ],
                columns=[
                    "wave", "target_node_id", "lineage_id", "status",
                    "forward_success", "reverse_success", "inserted",
                    "propagated_next_wave", "residual_mm", "reverse_return_gap_deg",
                    "target_candidate_id", "source_node_id", "source_candidate_id",
                ],
            )
            _write_parquet(
                propagation,
                patch_dir / f"{ablation.value}_propagation.parquet",
            )

            # Fresh audits are performed from the candidate/table contract,
            # never by replaying stored continuation endpoints.
            atlas_nodes = patch_tasks.copy()
            active_continuation = base_continuation
            if ablation in {
                RepairAblation.E2_SEGMENTED,
                RepairAblation.E3_DYNAMIC_INSERTION,
                RepairAblation.E4_FRONTIER_ENRICHMENT,
            }:
                node_by_id = {
                    node.node_id: node for node in result.graph.task_nodes
                }
                cell_by_node = {
                    int(row.task_node_id): (
                        int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)
                    )
                    for row in patch_tasks.itertuples(index=False)
                }
                active_continuation = make_segmented_continuation(
                    base_continuation,
                    node_by_id,
                    cell_by_node,
                    step_max_mm=float(config["repair"]["cartesian_step_max_mm"]),
                )
            integrated = build_workspace_atlas_integration(
                atlas_nodes,
                result_candidates,
                environment,
                task_edges=result.task_edges,
                continuation=active_continuation,
                policy=_integration_policy(config),
            )
            for name, frame in integrated.frames.items():
                if name in {"charts", "chart_overlaps", "primary_partition", "domain_classification"}:
                    _write_parquet(
                        frame,
                        patch_dir / f"{ablation.value}_{name}.parquet",
                    )
            e0_exact = True
            if ablation is RepairAblation.E0_LEGACY:
                expected = sealed_product[
                    sealed_product["patch_id"].eq(patch_id)
                    & sealed_product["left_task_node_id"].astype(int).isin(patch_task_ids)
                    & sealed_product["right_task_node_id"].astype(int).isin(patch_task_ids)
                ]
                e0_exact = _e0_edge_key_frame(expected) == _e0_edge_key_frame(graph_edges)
            metric = asdict(result.metrics)
            report = {
                "patch_id": str(patch_id),
                "patch_split": patch_split,
                "ablation": ablation.value,
                **metric,
                "audit_gate_pass": bool(integrated.audit_report.gate_pass),
                "representation_mode": integrated.workspace_result.representation.mode.value,
                "primary_measure_coverage": integrated.workspace_result.representation.measure_coverage,
                "abstention_measure_ratio": integrated.workspace_result.representation.abstention_measure_ratio,
                "chart_count": len(integrated.canonical_atlas.charts),
                "chart_selection_counts": [
                    len(chart.selections) for chart in integrated.canonical_atlas.charts
                ],
                "propagation_record_count": len(result.propagation_records),
                "propagation_inserted_count": sum(row.inserted for row in result.propagation_records),
                "propagation_bidirectional_count": sum(
                    row.reverse_success for row in result.propagation_records
                ),
                "budget_exhausted": result.budget_exhausted,
                "e0_exact_edge_closure": e0_exact,
                "runtime_s": time.time() - started,
                "candidate_count": len(result.graph.candidates),
                "robust_edge_count": len(result.graph.robust_edges),
                "continuation_attempt_count": result.graph.continuation_attempt_count,
                "fresh_audit_execution_count": integrated.report["fresh_audit"]["execution_count"],
            }
            _write_json(report_path, report)
            reports.append(report)
        _write_json(
            patch_dir / "patch_complete.json",
            {
                "gate_pass": True,
                "patch_id": str(patch_id),
                "ablation_count": len(RepairAblation),
            },
        )
    if patch_filter is not None:
        return {
            "gate_pass": True,
            "patch_id": str(patch_filter),
            "ablation_count": len(RepairAblation),
        }
    metrics = pd.DataFrame.from_records(reports).sort_values(
        ["patch_id", "ablation"], kind="stable"
    )
    _write_parquet(metrics, stage / "ablation_metrics.parquet")
    gate_report = dict(evaluate_patch_ablation_gate(metrics))
    e0_closure = bool(
        metrics.loc[metrics["ablation"].eq(RepairAblation.E0_LEGACY.value), "e0_exact_edge_closure"].all()
    )
    return _gate(
        stage / "gate.json",
        {
            "all_patch_ablation_rows": len(metrics) == 12 * 5,
            "E0_exact_edge_closure": e0_closure,
            "repair_method_frozen": bool(gate_report["gate_pass"]),
        },
        patch_gate=gate_report,
        selected_method=gate_report["selected_method"],
        E0_exact_edge_closure=e0_closure,
    )


def _inside_bounds(beta: np.ndarray, bounds: np.ndarray) -> bool:
    return bool(
        np.isfinite(beta).all()
        and np.all(beta >= bounds[:, 0] - 1.0e-12)
        and np.all(beta <= bounds[:, 1] + 1.0e-12)
    )


def _fk_one(environment: Any, beta: np.ndarray) -> np.ndarray:
    return np.asarray(environment.fk(np.asarray(beta, dtype=float).reshape(1, 6)), dtype=float).reshape(-1, 3)[0]


def _fiber_walk(
    environment: Any,
    target_xyz: np.ndarray,
    start_beta: np.ndarray,
    goal_beta: np.ndarray,
    *,
    step_deg: float,
    maximum_steps: int,
    residual_max_mm: float,
) -> bool:
    """Forced null-space predictor/corrector path on one fixed XYZ fiber."""

    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    current = np.asarray(start_beta, dtype=float).copy()
    goal = np.asarray(goal_beta, dtype=float).copy()
    for _step in range(int(maximum_steps)):
        gap = beta_rms_deg(current, goal)
        if gap <= float(step_deg) + 1.0e-12:
            return True
        jacobian = np.asarray(environment.jacobian(current), dtype=float).reshape(3, 6)
        null_projector = np.eye(6) - np.linalg.pinv(jacobian) @ jacobian
        direction = null_projector @ (goal - current)
        direction_rms_deg = float(np.sqrt(np.mean(np.rad2deg(direction) ** 2)))
        if direction_rms_deg <= 1.0e-10:
            return False
        predictor = current + direction * min(1.0, float(step_deg) / direction_rms_deg)
        if not _inside_bounds(predictor, bounds):
            return False
        corrected = predictor.copy()
        converged = False
        for _iteration in range(12):
            error = np.asarray(target_xyz, dtype=float) - _fk_one(environment, corrected)
            residual_mm = float(np.linalg.norm(error) * 1000.0)
            if residual_mm <= float(residual_max_mm):
                converged = True
                break
            jacobian = np.asarray(environment.jacobian(corrected), dtype=float).reshape(3, 6)
            corrected = corrected + weighted_damped_pinv(
                jacobian,
                damping=1.0e-3,
                weights=np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0]),
            ) @ error
            if not _inside_bounds(corrected, bounds):
                return False
        if not converged:
            return False
        current = corrected
    return beta_rms_deg(current, goal) <= float(step_deg) + 1.0e-12


def _same_fiber_components(
    environment: Any,
    target_xyz: np.ndarray,
    beta: np.ndarray,
    *,
    neighbor_count: int,
    initial_gap_max_deg: float,
    step_deg: float,
    maximum_steps: int,
    residual_max_mm: float,
) -> tuple[tuple[int, ...], ...]:
    count = len(beta)
    if count == 0:
        return ()
    adjacency: dict[int, set[int]] = {index: set() for index in range(count)}
    distances = np.asarray(
        [
            [beta_rms_deg(beta[left], beta[right]) for right in range(count)]
            for left in range(count)
        ],
        dtype=float,
    )
    pairs: set[tuple[int, int]] = set()
    for left in range(count):
        neighbors = np.argsort(distances[left], kind="stable")[1 : 1 + int(neighbor_count)]
        for right_value in neighbors:
            right = int(right_value)
            if distances[left, right] <= float(initial_gap_max_deg) + 1.0e-12:
                pairs.add(tuple(sorted((left, right))))
    for left, right in sorted(pairs):
        forward = _fiber_walk(
            environment,
            target_xyz,
            beta[left],
            beta[right],
            step_deg=step_deg,
            maximum_steps=maximum_steps,
            residual_max_mm=residual_max_mm,
        )
        reverse = forward and _fiber_walk(
            environment,
            target_xyz,
            beta[right],
            beta[left],
            step_deg=step_deg,
            maximum_steps=maximum_steps,
            residual_max_mm=residual_max_mm,
        )
        if forward and reverse:
            adjacency[left].add(right)
            adjacency[right].add(left)
    components: list[tuple[int, ...]] = []
    reached: set[int] = set()
    for root in range(count):
        if root in reached:
            continue
        pending = [root]
        component: list[int] = []
        while pending:
            current = pending.pop(0)
            if current in reached:
                continue
            reached.add(current)
            component.append(current)
            pending.extend(sorted(adjacency[current] - reached))
        components.append(tuple(sorted(component)))
    return tuple(components)


def _cluster_audit_candidates(frame: pd.DataFrame, threshold_deg: float) -> pd.DataFrame:
    accepted = frame[
        frame["quality_class"].isin(("Gold", "Silver"))
        & frame["solver_success"].astype(bool)
    ].copy()
    if not len(accepted):
        return accepted
    accepted["quality_rank"] = accepted["quality_class"].map({"Gold": 0, "Silver": 1})
    accepted = accepted.sort_values(
        ["quality_rank", "minimum_margin_deg", "residual_mm", "seed_rank", "candidate_id"],
        ascending=[True, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    beta = accepted.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    scores = (
        accepted["quality_rank"].to_numpy(dtype=float) * 1.0e6
        - accepted["minimum_margin_deg"].to_numpy(dtype=float) * 1.0e3
        + accepted["residual_mm"].to_numpy(dtype=float)
    )
    _representatives, indices = stable_cluster_representatives(
        beta, scores=scores, threshold_deg=float(threshold_deg)
    )
    result = accepted.iloc[list(indices)].copy().reset_index(drop=True)
    all_beta = accepted.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    first_budget: list[int] = []
    for representative in result.loc[:, BETA_COLUMNS].to_numpy(dtype=float):
        member_seed_ranks = [
            int(accepted.iloc[index].seed_rank)
            for index, candidate in enumerate(all_beta)
            if beta_rms_deg(representative, candidate) <= float(threshold_deg) + 1.0e-12
        ]
        minimum_seed_count = min(member_seed_ranks) + 1
        first_budget.append(next((value for value in (4, 8, 16, 32) if minimum_seed_count <= value), 33))
    result["first_budget"] = first_budget
    return result


def stage_stability(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "inventory")
    stage = output_root / STAGE_DIRS["stability"]
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"selection stability stage is already sealed: {stage}")
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    raw = pd.read_parquet(paths["saturation_candidates"])
    audit_tasks = pd.read_parquet(paths["saturation_task_nodes"])
    inventory = output_root / STAGE_DIRS["inventory"]
    task_nodes = pd.read_parquet(inventory / "patch_task_nodes.parquet").drop(columns=["patch_id"])
    # Stability uses the original 500 audit parent cells, not only the 768-cell patches.
    full_task_nodes = pd.read_parquet(paths["atlas_task_nodes"])
    representative = full_task_nodes[full_task_nodes["is_representative"].astype(bool)].copy()
    representative_by_parent = representative.set_index("source_parent_node_id")
    parent_edges = pd.read_parquet(paths["parent_task_edges"])
    neighbors: dict[int, set[int]] = {int(value): set() for value in audit_tasks["node_id"]}
    all_parent_neighbors: dict[int, set[int]] = {}
    for edge in parent_edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        all_parent_neighbors.setdefault(left, set()).add(right)
        all_parent_neighbors.setdefault(right, set()).add(left)
    stability = config["selection_stability"]
    observations: list[CanonicalSelectionObservation] = []
    component_rows: list[dict[str, Any]] = []
    base_continuation = make_predictor_corrector_continuation(
        environment,
        residual_tolerance_mm=float(config["repair"]["continuation_residual_max_mm"]),
    )
    bounds = np.asarray(environment.bounds, dtype=float)
    for parent_node_id in sorted(audit_tasks["node_id"].astype(int)):
        node_frame = _cluster_audit_candidates(
            raw[raw["node_id"].eq(parent_node_id)],
            float(stability["candidate_cluster_deg"]),
        )
        if not len(node_frame):
            continue
        beta = node_frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
        parent_row = audit_tasks.set_index("node_id").loc[parent_node_id]
        target_xyz = parent_row.loc[list(XYZ_COLUMNS)].to_numpy(dtype=float)
        components = _same_fiber_components(
            environment,
            target_xyz,
            beta,
            neighbor_count=int(stability["same_fiber_neighbor_count"]),
            initial_gap_max_deg=float(stability["initial_gap_max_deg"]),
            step_deg=float(stability["nullspace_step_deg"]),
            maximum_steps=int(stability["maximum_steps"]),
            residual_max_mm=float(stability["residual_max_mm"]),
        )
        component_by_index = {
            index: f"n{parent_node_id:06d}_fiber_{component_id:03d}"
            for component_id, component in enumerate(components)
            for index in component
        }
        for budget in map(int, stability["budgets"]):
            available = node_frame[node_frame["first_budget"].le(budget)].copy()
            if not len(available):
                continue
            selected_index = int(available.index[0])
            selected = node_frame.loc[selected_index]
            beta_selected = selected.loc[list(BETA_COLUMNS)].to_numpy(dtype=float)
            source_task_id = int(representative_by_parent.loc[parent_node_id].task_node_id)
            source = AtlasCandidate(
                node_id=source_task_id,
                candidate_id=f"stability_b{budget}",
                beta_rad=beta_selected,
                residual_mm=float(selected.residual_mm),
                min_margin_deg=float(selected.minimum_margin_deg),
                normalized_min_margin=float(selected.normalized_minimum_margin),
                quality=str(selected.quality_class),
            )
            outgoing: set[int] = set()
            for neighbor_parent in sorted(all_parent_neighbors.get(parent_node_id, ())):
                if neighbor_parent not in representative_by_parent.index:
                    continue
                neighbor_row = representative_by_parent.loc[neighbor_parent]
                target_node = AtlasTaskNode(
                    int(neighbor_row.task_node_id),
                    neighbor_row.loc[list(XYZ_COLUMNS)].to_numpy(dtype=float),
                )
                forward = base_continuation(source, target_node)
                if not forward.success or not forward.actual_bounds:
                    continue
                target_candidate = AtlasCandidate(
                    node_id=target_node.node_id,
                    candidate_id="stability_return",
                    beta_rad=forward.beta_rad,
                    residual_mm=forward.residual_mm,
                    min_margin_deg=source.min_margin_deg,
                    normalized_min_margin=source.normalized_min_margin,
                    quality=source.quality,
                )
                source_node = AtlasTaskNode(source_task_id, target_xyz)
                reverse = base_continuation(target_candidate, source_node)
                if (
                    reverse.success
                    and reverse.actual_bounds
                    and beta_rms_deg(reverse.beta_rad, beta_selected)
                    <= float(config["repair"]["reverse_return_max_deg"]) + 1.0e-12
                ):
                    outgoing.add(neighbor_parent)
            observation = CanonicalSelectionObservation(
                node_id=parent_node_id,
                budget=budget,
                selected_beta_rad=beta_selected,
                selected_component_id=component_by_index[selected_index],
                outgoing_neighbor_ids=frozenset(outgoing),
            )
            observations.append(observation)
            component_rows.append(
                {
                    "node_id": parent_node_id,
                    "budget": budget,
                    "selected_component_id": observation.selected_component_id,
                    "outgoing_neighbor_ids": sorted(outgoing),
                    "available_cluster_count": len(available),
                    "full_budget_cluster_count": len(node_frame),
                    "same_fiber_component_count": len(components),
                    **{
                        name: float(beta_selected[index])
                        for index, name in enumerate(BETA_COLUMNS)
                    },
                }
            )
    report = evaluate_canonical_selection_stability(
        observations,
        beta_p95_max_deg=float(stability["beta_p95_max_deg"]),
        beta_max_deg=float(stability["beta_max_deg"]),
        component_switch_ratio_max=float(stability["component_switch_ratio_max"]),
        outgoing_neighbor_change_ratio_max=float(
            stability["outgoing_neighbor_change_ratio_max"]
        ),
    )
    _write_parquet(pd.DataFrame.from_records(component_rows), stage / "selection_observations.parquet")
    _write_json(stage / "stability_report.json", asdict(report))
    return _gate(
        stage / "gate.json",
        {
            "exact_audit_cell_count": report.audited_node_count
            == int(stability["audit_cell_count"]),
            "canonical_selection_stability": report.gate_pass,
        },
        **asdict(report),
        raw_new_candidate_families_are_not_the_gate=True,
    )


def _grid_from_original(paths: Mapping[str, Path]) -> WorkspaceGridSpec:
    source = _read_json(paths["original_source_manifest"])
    frozen_path = Path(source["source_root"]) / "configs/bacra_v14_omega200_workspace_atlas.yaml"
    # Source roots can be old worktrees; the sealed Pilot frozen config is the
    # direct authority and is always co-located with its source manifest.
    pilot_root = paths["original_source_manifest"].parents[1]
    frozen = _read_json(pilot_root / "00_protocol/frozen_config.json")
    domain = frozen["domain"]
    return WorkspaceGridSpec(
        levels_mm=tuple(map(int, domain["levels_mm"])),
        x_slab_m=(float(domain["x_min_m"]), float(domain["x_max_m"])),
        origin_m=tuple(map(float, domain["origin_m"])),
    )


def _frontier_extension_rows(
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
    rows: list[dict[str, Any]] = []
    evidence: list[FrontierProbeEvidence] = []
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
    for local_id, (cell, target) in enumerate(zip(cells, targets, strict=True)):
        _distance, indices = tree.query(target, k=min(int(starts), len(seed_xyz)))
        indices = np.asarray(indices, dtype=int).reshape(-1)
        bank = solve_candidate_bank(
            environment,
            target.reshape(1, 3),
            policy,
            neighbor_beta_rad={0: seed_beta[indices]},
        )
        accepted = [
            record for record in bank.records if record.quality is not CandidateQuality.REJECT
        ]
        selected = min(
            accepted,
            key=lambda record: (
                0 if record.quality is CandidateQuality.GOLD else 1,
                record.residual_mm,
                -record.min_margin_deg,
                record.candidate_id,
            ),
            default=None,
        )
        found = selected is not None
        evidence.append(
            FrontierProbeEvidence(cell=cell, found_valid_inverse=found, round_id=round_id)
        )
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
                "gold_candidate_count": sum(
                    record.quality is CandidateQuality.GOLD for record in accepted
                ),
                "silver_candidate_count": sum(
                    record.quality is CandidateQuality.SILVER for record in accepted
                ),
                "selected_candidate_id": None if selected is None else selected.candidate_id,
                "selected_residual_mm": None if selected is None else selected.residual_mm,
                **{
                    name: None if selected is None else float(selected.beta_rad[index])
                    for index, name in enumerate(BETA_COLUMNS)
                },
            }
        )
    return evidence, pd.DataFrame.from_records(rows)


def stage_reach_extension(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_stage(output_root, "inventory")
    stage = output_root / STAGE_DIRS["reach_extension"]
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"Reach extension stage is already sealed: {stage}")
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    grid = _grid_from_original(paths)
    original_report = _read_json(paths["workspace_proxy_report"])
    original_a = pd.read_parquet(paths["replica_a_slab"])
    original_b = pd.read_parquet(paths["replica_b_slab"])
    beta_a_parts = [original_a.loc[:, BETA_COLUMNS].to_numpy(dtype=float)]
    beta_b_parts = [original_b.loc[:, BETA_COLUMNS].to_numpy(dtype=float)]
    xyz_a_parts = [original_a.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)]
    xyz_b_parts = [original_b.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)]
    seeds_a = list(map(int, original_report["replica_a_seed_lineage"]))
    seeds_b = list(map(int, original_report["replica_b_seed_lineage"]))
    rounds: list[ReachReplicaRound] = [
        ReachReplicaRound(
            3,
            ReachReplica("A", tuple(seeds_a), np.vstack(xyz_a_parts)),
            ReachReplica("B", tuple(seeds_b), np.vstack(xyz_b_parts)),
        )
    ]
    generated_frames_a: list[pd.DataFrame] = []
    generated_frames_b: list[pd.DataFrame] = []
    for spec_row in config["reach_extension"]["rounds"]:
        spec = ReachSamplingRoundSpec(
            int(spec_row["power"]), int(spec_row["seed_a"]), int(spec_row["seed_b"])
        )
        generated = generate_independent_reach_samples(
            environment,
            np.asarray(environment.bounds, dtype=float),
            grid=grid,
            rounds=(spec,),
            chunk_rows=65536,
        )
        beta_a_parts.append(generated.beta_a)
        beta_b_parts.append(generated.beta_b)
        xyz_a_parts.append(generated.xyz_a)
        xyz_b_parts.append(generated.xyz_b)
        seeds_a.append(spec.seed_a)
        seeds_b.append(spec.seed_b)
        round_id = int(spec_row["round"])
        rounds.append(
            ReachReplicaRound(
                round_id,
                ReachReplica("A", tuple(seeds_a), np.vstack(xyz_a_parts)),
                ReachReplica("B", tuple(seeds_b), np.vstack(xyz_b_parts)),
            )
        )
        generated_frames_a.append(
            pd.DataFrame(
                {
                    **{name: generated.xyz_a[:, index] for index, name in enumerate(XYZ_COLUMNS)},
                    **{name: generated.beta_a[:, index] for index, name in enumerate(BETA_COLUMNS)},
                    "source": f"replica_a_round_{round_id}",
                }
            )
        )
        generated_frames_b.append(
            pd.DataFrame(
                {
                    **{name: generated.xyz_b[:, index] for index, name in enumerate(XYZ_COLUMNS)},
                    **{name: generated.beta_b[:, index] for index, name in enumerate(BETA_COLUMNS)},
                    "source": f"replica_b_round_{round_id}",
                }
            )
        )
    domain = pd.read_parquet(paths["domain_cells"])
    supplemental = tuple(
        CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        for row in domain[
            domain["cell_level_mm"].eq(grid.convergence_level_mm)
            & (domain["registered_pool_support"].astype(bool) | domain["tip_pool_support"].astype(bool))
        ].itertuples(index=False)
    )
    original_frontier_frame = pd.read_parquet(paths["frontier_inverse_probes"])
    frontier_evidence: list[FrontierProbeEvidence] = [
        FrontierProbeEvidence(
            CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)),
            bool(row.found_valid_inverse),
            round_id=int(row.round_id),
        )
        for row in original_frontier_frame.itertuples(index=False)
    ]
    builder = ReachProxyBuilder(
        ReachProxyConfig(
            grid=grid,
            minimum_weighted_jaccard=0.95,
            maximum_new_volume_ratio=0.01,
            maximum_boundary_change_ratio=0.02,
            maximum_frontier_new_volume_ratio=0.01,
            required_consecutive_rounds=2,
        )
    )
    new_frontier_frames: list[pd.DataFrame] = []
    # Probe each newly added round's active frontier against the cumulative
    # A/B seed bank; this is independent of the original forward occupancy.
    accumulated_rounds: list[ReachReplicaRound] = [rounds[0]]
    for extension_index, current_round in enumerate(rounds[1:], start=1):
        accumulated_rounds.append(current_round)
        preliminary = builder.build(
            accumulated_rounds,
            frontier_evidence=frontier_evidence,
            supplemental_supported_cells=supplemental,
        )
        cells = frontier_candidates(
            preliminary.proxy_upper_cells,
            grid=grid,
            maximum_count=int(config["reach_extension"]["frontier_cells_per_round"]),
        )
        seed_xyz = np.vstack([current_round.replica_a.xyz_m, current_round.replica_b.xyz_m])
        seed_beta = np.vstack(
            [
                np.vstack(beta_a_parts[: extension_index + 1]),
                np.vstack(beta_b_parts[: extension_index + 1]),
            ]
        )
        evidence, frame = _frontier_extension_rows(
            environment,
            cells,
            grid=grid,
            seed_xyz=seed_xyz,
            seed_beta=seed_beta,
            starts=int(config["reach_extension"]["frontier_starts"]),
            round_id=current_round.round_id,
        )
        frontier_evidence.extend(evidence)
        new_frontier_frames.append(frame)
    result = builder.build(
        rounds,
        frontier_evidence=frontier_evidence,
        supplemental_supported_cells=supplemental,
    )
    final_a = pd.concat([original_a, *generated_frames_a], ignore_index=True)
    final_b = pd.concat([original_b, *generated_frames_b], ignore_index=True)
    _write_parquet(final_a, stage / "replica_a_slab_round5.parquet")
    _write_parquet(final_b, stage / "replica_b_slab_round5.parquet")
    new_frontier = pd.concat(new_frontier_frames, ignore_index=True)
    _write_parquet(new_frontier, stage / "frontier_inverse_probes_round4_5.parquet")
    metrics = [asdict(row) for row in result.replica_metrics if row.round_id >= 4]
    gap = float(
        (len(result.proxy_upper_cells) - len(result.proxy_lower_cells))
        / max(1, len(result.proxy_upper_cells))
    )
    report = {
        "reach_convergence_gate": result.convergence_gate,
        "metrics_round4_5": metrics,
        "proxy_lower_cell_count": len(result.proxy_lower_cells),
        "proxy_upper_cell_count": len(result.proxy_upper_cells),
        "lower_upper_measure_gap_ratio": gap,
        "lower_upper_gap_diagnostic_pass": gap
        <= float(config["reach_extension"]["measure_gap_diagnostic_max"]),
        "replica_a_seed_lineage": seeds_a,
        "replica_b_seed_lineage": seeds_b,
        "original_round3_metrics_preserved": original_report["metrics"][-1],
        "original_gate_thresholds_unchanged": True,
    }
    _write_json(stage / "reach_extension_report.json", report)
    return _gate(
        stage / "gate.json",
        {"original_reach_convergence_gate": result.convergence_gate},
        **report,
    )


def _operational_skip(stage: Path, *, reasons: Sequence[str]) -> dict[str, Any]:
    payload = {
        "gate_pass": True,
        "scientific_gate_pass": False,
        "skipped": True,
        "reasons": tuple(sorted(set(map(str, reasons)))),
    }
    _write_json(stage / "skip.json", payload)
    _write_json(stage / "gate.json", payload)
    return payload


def stage_repaired_pilot(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["repaired_pilot"]
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"repaired Pilot stage is already sealed: {stage}")
    reasons: list[str] = []
    ablation_path = output_root / STAGE_DIRS["ablations"] / "gate.json"
    stability_path = output_root / STAGE_DIRS["stability"] / "gate.json"
    if not ablation_path.is_file() or not _read_json(ablation_path).get("gate_pass", False):
        reasons.append("patch_repair_gate_not_passed")
    if not stability_path.is_file() or not _read_json(stability_path).get("gate_pass", False):
        reasons.append("canonical_selection_stability_not_passed")
    if reasons:
        return _operational_skip(stage, reasons=reasons)
    ablation_gate = _read_json(ablation_path)
    selected_method = RepairAblation(str(ablation_gate["selected_method"]))
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    tasks = pd.read_parquet(paths["atlas_task_nodes"])
    edges = pd.read_parquet(paths["atlas_task_edges"])
    base_candidate_frame = pd.read_parquet(paths["candidate_clusters"])
    base_candidates = _candidates_from_frame(base_candidate_frame)
    base_continuation = make_predictor_corrector_continuation(
        environment,
        residual_tolerance_mm=float(config["repair"]["continuation_residual_max_mm"]),
    )
    candidates_for_method = base_candidates
    if selected_method is RepairAblation.E4_FRONTIER_ENRICHMENT:
        e3 = run_cross_cell_repair_ablation(
            RepairAblation.E3_DYNAMIC_INSERTION,
            tasks,
            edges,
            base_candidates,
            base_continuation,
            atlas_policy=_atlas_policy(config),
            repair_policy=_repair_policy(config),
        )
        candidates_for_method = _enrich_candidates(
            tasks, base_candidates, e3.graph, environment, config
        )
    started = time.time()
    repaired = run_cross_cell_repair_ablation(
        selected_method,
        tasks,
        edges,
        candidates_for_method,
        base_continuation,
        atlas_policy=_atlas_policy(config),
        repair_policy=_repair_policy(config),
    )
    candidate_frame = _candidate_frame(repaired.graph.candidates)
    node_by_id = {node.node_id: node for node in repaired.graph.task_nodes}
    cell_by_node = {
        int(row.task_node_id): (
            int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)
        )
        for row in tasks.itertuples(index=False)
    }
    active_continuation = make_segmented_continuation(
        base_continuation,
        node_by_id,
        cell_by_node,
        step_max_mm=float(config["repair"]["cartesian_step_max_mm"]),
    )
    integrated = build_workspace_atlas_integration(
        tasks,
        candidate_frame,
        environment,
        task_edges=repaired.task_edges,
        continuation=active_continuation,
        policy=_integration_policy(config),
    )
    _write_parquet(repaired.task_edges, stage / "repaired_task_edges.parquet")
    _write_parquet(candidate_frame, stage / "repaired_candidates.parquet")
    _write_parquet(_graph_edge_frame(repaired.graph), stage / "repaired_product_edges.parquet")
    for name, frame in integrated.frames.items():
        _write_parquet(frame, stage / f"{name}.parquet")
    assessments = integrated.workspace_result.cell_assessments
    total_cells = len(assessments)
    labelable_measure_ratio = float(
        sum(row.empirical_labelable_fraction for row in assessments) / max(1, total_cells)
    )
    unresolved_abstain_ratio = float(1.0 - labelable_measure_ratio)
    valid_sections = [
        section
        for section in integrated.workspace_result.section_assessments
        if section.valid
    ]
    largest_primary_region_measure_ratio = float(
        max((len(section.cell_ids) for section in valid_sections), default=0)
        / max(1, total_cells)
    )
    assessment_by_cell = {row.cell: row for row in assessments}
    parent_nodes = pd.read_parquet(paths["parent_task_nodes"])
    x_cuts = np.quantile(parent_nodes["x_m"].to_numpy(dtype=float), (1 / 3, 2 / 3))
    x_tertile_counts = {0: 0, 1: 0, 2: 0}
    for row in parent_nodes.itertuples(index=False):
        cell = CellKey(int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
        assessment = assessment_by_cell.get(cell)
        if assessment is not None and assessment.empirical_labelable_fraction > 0.0:
            x_tertile_counts[int(np.searchsorted(x_cuts, float(row.x_m), side="right"))] += 1
    stability_gate = _read_json(stability_path)
    scientific = dict(
        evaluate_repaired_pilot_gate(
            cross_cell_connection_rate=repaired.metrics.cross_cell_neighbor_pair_connection_rate,
            labelable_measure_ratio=labelable_measure_ratio,
            largest_primary_region_measure_ratio=largest_primary_region_measure_ratio,
            unresolved_abstain_ratio=unresolved_abstain_ratio,
            selection_stability_gate=bool(stability_gate["gate_pass"]),
            audit_gate=bool(integrated.audit_report.gate_pass),
            x_tertile_labelable_counts=x_tertile_counts,
        )
    )
    report = {
        "selected_method": selected_method.value,
        "scientific_gate_pass": scientific["gate_pass"],
        "scientific_checks": scientific["checks"],
        "cross_cell_metrics": asdict(repaired.metrics),
        "labelable_measure_ratio": labelable_measure_ratio,
        "largest_primary_region_measure_ratio": largest_primary_region_measure_ratio,
        "unresolved_abstain_ratio": unresolved_abstain_ratio,
        "x_tertile_labelable_counts": x_tertile_counts,
        "atlas_audit_gate_pass": integrated.audit_report.gate_pass,
        "canonical_selection_stability_gate": stability_gate["gate_pass"],
        "representation_mode": integrated.workspace_result.representation.mode.value,
        "static_inverse_authorized": integrated.workspace_result.representation.static_inverse_authorized,
        "stateful_inverse_authorized": integrated.workspace_result.representation.stateful_inverse_authorized,
        "chart_count": len(integrated.canonical_atlas.charts),
        "chart_cell_counts": [len(section.cell_ids) for section in valid_sections],
        "budget_exhausted": repaired.budget_exhausted,
        "runtime_s": time.time() - started,
    }
    _write_json(stage / "repaired_pilot_report.json", report)
    return _gate(
        stage / "gate.json",
        {"operational_completion": True},
        **report,
    )


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )


def _student_metrics(
    inverse: Any, frame: pd.DataFrame, environment: Any, *, models: Any
) -> dict[str, Any]:
    query = InverseQuery(frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float))
    prediction = inverse.predict(query)
    accepted = np.asarray(prediction.accepted, dtype=bool)
    residual = np.asarray(prediction.fk_residual_mm, dtype=float)
    finite = accepted & np.isfinite(residual)
    values = residual[finite]
    features = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    if models.mode is RepresentationMode.XYZ_GLOBAL:
        raw_beta = np.asarray(models.global_model(features, training=False), dtype=float)
    else:
        probability = np.asarray(models.router_model(features, training=False), dtype=float)
        expert = {
            chart: np.asarray(model(features, training=False), dtype=float)
            for chart, model in models.expert_models.items()
        }
        selected = np.argmax(probability, axis=1)
        raw_beta = np.vstack(
            [expert[models.chart_ids[int(chart_index)]][index] for index, chart_index in enumerate(selected)]
        )
    targets = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    dls: dict[str, float] = {}
    corrected = raw_beta.copy()
    bounds = np.asarray(environment.bounds, dtype=float)
    for step in (1, 2):
        for index in range(len(corrected)):
            error = targets[index] - _fk_one(environment, corrected[index])
            jacobian = np.asarray(environment.jacobian(corrected[index]), dtype=float)
            proposal = corrected[index] + weighted_damped_pinv(
                jacobian,
                damping=1.0e-3,
                weights=np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0]),
            ) @ error
            if _inside_bounds(proposal, bounds):
                corrected[index] = proposal
        corrected_xyz = np.asarray(environment.fk(corrected), dtype=float).reshape(-1, 3)
        corrected_residual = np.linalg.norm(corrected_xyz - targets, axis=1) * 1000.0
        dls[f"dls_{step}_fk_p95_mm"] = float(np.percentile(corrected_residual, 95))
        dls[f"dls_{step}_fk_max_mm"] = float(np.max(corrected_residual))
    return {
        "row_count": len(frame),
        "accepted_count": int(finite.sum()),
        "accepted_fraction": float(finite.mean()) if len(frame) else 0.0,
        "fk_p95_mm": float(np.percentile(values, 95)) if len(values) else math.inf,
        "fk_max_mm": float(np.max(values)) if len(values) else math.inf,
        **dls,
    }


def _training_config(config: Mapping[str, Any], seed: int) -> WorkspaceStudentTrainingConfig:
    student = config["student"]
    return WorkspaceStudentTrainingConfig(
        hidden_units=tuple(map(int, student["hidden_units"])),
        router_hidden_units=tuple(map(int, student["router_hidden_units"])),
        learning_rate=float(student["learning_rate"]),
        max_steps=int(student["max_steps"]),
        validation_interval=int(student["validation_interval"]),
        patience_intervals=int(student["patience_intervals"]),
        seed=int(seed),
    )


def _train_mode_ensemble(
    records: Sequence[Any],
    *,
    mode: RepresentationMode,
    environment: Any,
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[Any]]:
    frame = records_to_workspace_student_frame(
        records,
        jacobian_at_beta=lambda beta: np.asarray(environment.jacobian(beta), dtype=float),
    )
    train = frame[frame["split_role"].eq(SplitRole.TRAIN_CORE.value)].copy()
    validation = frame[frame["split_role"].eq(SplitRole.VALIDATION.value)].copy()
    if len(train) == 0 or len(validation) == 0:
        raise RuntimeError("Student macroblock split lacks train or validation rows")
    if mode is RepresentationMode.XYZ_ROUTER_EXPERTS:
        train_charts = set(train["chart_id"].astype(str))
        validation_charts = set(validation["chart_id"].astype(str))
        if train_charts != validation_charts:
            raise RuntimeError(
                "router expert train/validation chart support mismatch: "
                f"train_only={sorted(train_charts-validation_charts)}, "
                f"validation_only={sorted(validation_charts-train_charts)}"
            )
    _write_parquet(frame, output_dir / f"{mode.value}_student_frame.parquet")
    reports: list[dict[str, Any]] = []
    results: list[Any] = []
    geometry = _student_geometry(environment)
    for seed in map(int, config["student"]["seeds"]):
        result = train_workspace_student(
            train,
            validation,
            mode=mode,
            geometry=geometry,
            config=_training_config(config, seed),
        )
        model_dir = output_dir / f"{mode.value}_seed_{seed}"
        save_workspace_student_models(result.models, model_dir)
        _write_parquet(result.history, model_dir / "training_history.parquet")
        metrics = _student_metrics(
            result.inverse, validation, environment, models=result.models
        )
        metrics.update({"seed": seed, "mode": mode.value})
        reports.append(metrics)
        results.append(result)
    return reports, results


def stage_exploratory_student(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["exploratory_student"]
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"exploratory Student stage is already sealed: {stage}")
    pilot_path = output_root / STAGE_DIRS["repaired_pilot"] / "gate.json"
    if not pilot_path.is_file():
        return _operational_skip(stage, reasons=("missing_repaired_pilot_gate",))
    pilot = _read_json(pilot_path)
    if not pilot.get("scientific_gate_pass", False):
        return _operational_skip(stage, reasons=("repaired_pilot_scientific_gate_failed",))
    if not pilot.get("static_inverse_authorized", False):
        return _operational_skip(
            stage,
            reasons=("static_xyz_inverse_not_authorized_stateful_not_preselected",),
        )
    repaired = output_root / STAGE_DIRS["repaired_pilot"]
    task_probes = pd.read_parquet(repaired / "task_probes.parquet")
    candidates = pd.read_parquet(repaired / "repaired_candidates.parquet")
    partition = pd.read_parquet(repaired / "primary_partition.parquet")
    product_edges = pd.read_parquet(repaired / "repaired_product_edges.parquet")
    anchor_records = supervision_records_from_atlas_frames(
        task_probes=task_probes,
        candidates=candidates,
        primary_partition=partition,
        product_edges=product_edges,
    )
    paths = _source_paths(config, project_root)
    environment = load_environment(project_root, paths["robot_config"])
    replica_a = pd.read_parquet(paths["replica_a_slab"])
    replica_b = pd.read_parquet(paths["replica_b_slab"])
    targets = pd.concat(
        [
            replica_a.loc[:, XYZ_COLUMNS].assign(source="a"),
            replica_b.loc[:, XYZ_COLUMNS].assign(source="b"),
        ],
        ignore_index=True,
    ).drop_duplicates(list(XYZ_COLUMNS), keep="first")
    targets = targets.sort_values(list(XYZ_COLUMNS), kind="stable").reset_index(drop=True)
    targets["physical_point_id"] = [f"dense_{index:07d}" for index in range(len(targets))]
    dense = correct_static_targets_from_primary_sections(
        targets.loc[:, ["physical_point_id", *XYZ_COLUMNS]],
        task_probes=task_probes,
        candidates=candidates,
        primary_partition=partition,
        environment=environment,
        source_family="v14_1_exploratory_dense",
        priority=SupervisionPriority.REFINEMENT,
        maximum_rows=int(config["exploratory_dataset"]["target_rows"]),
        policy=WorkspaceAtlasFramePolicy(),
    )
    _write_parquet(dense.audit, stage / "dense_correction_audit.parquet")
    all_records = tuple(anchor_records) + tuple(dense.records)
    representation_mode = (
        RepresentationMode.XYZ_GLOBAL
        if str(pilot["representation_mode"]) == RepresentationMode.XYZ_GLOBAL.value
        else RepresentationMode.XYZ_ROUTER_EXPERTS
    )
    split_policy = MacroblockSplitPolicy(
        macroblock_mm=int(config["exploratory_dataset"]["macroblock_mm"]),
        seed=20260860,
        train_core_fraction=0.625,
        active_probe_fraction=0.075,
        validation_fraction=0.15,
        sealed_fraction=0.15,
    )
    materializer = SupervisionMaterializer(
        SupervisionBudgetPolicy(
            target_total=int(config["exploratory_dataset"]["target_rows"]),
            hard_max=int(config["exploratory_dataset"]["hard_max_rows"]),
            active_reserve=int(config["student"]["active_rows"]),
        ),
        split_policy,
    )
    preliminary = materializer.materialize(
        all_records,
        representation_mode=representation_mode,
    )
    # One registered active round: select difficult active-probe records by
    # Teacher residual and keep the reserve exact; there is no row padding.
    active_candidates = [
        row
        for row in all_records
        if split_policy.assignment_for_cell(row.cell).split_role is SplitRole.ACTIVE_PROBE
    ]
    active_candidates.sort(
        key=lambda row: (-float(row.residual_mm), row.record_id)
    )
    active_ids = [
        row.record_id
        for row in active_candidates[: int(config["student"]["active_rows"])]
    ]
    bundle = materializer.materialize(
        all_records,
        representation_mode=representation_mode,
        active_record_ids=active_ids,
    )
    _write_parquet(supervision_records_frame(bundle.supervision_records), stage / "student_supervision.parquet")
    _write_parquet(supervision_records_frame(bundle.primary_canonical), stage / "primary_canonical.parquet")
    _write_parquet(supervision_records_frame(bundle.chart_expert), stage / "chart_expert.parquet")
    _write_parquet(bundle.supervision_index, stage / "supervision_index.parquet")
    _write_json(stage / "budget_report.json", asdict(bundle.budget_report))
    minimum_rows = int(config["exploratory_dataset"]["minimum_rows"])
    if len(bundle.supervision_records) < minimum_rows:
        report = {
            "scientific_gate_pass": False,
            "reason": "insufficient_unique_supervision_without_padding",
            "row_count": len(bundle.supervision_records),
            "minimum_rows": minimum_rows,
            "padding_count": 0,
        }
        return _gate(stage / "gate.json", {"operational_completion": True}, **report)

    training_reports: list[dict[str, Any]] = []
    primary_records = tuple(
        row for row in bundle.supervision_records if row.kind.value == "static" and row.is_primary
    )
    global_reports, _global_results = _train_mode_ensemble(
        primary_records,
        mode=RepresentationMode.XYZ_GLOBAL,
        environment=environment,
        config=config,
        output_dir=stage,
    )
    training_reports.extend(global_reports)
    if len({row.chart_id for row in bundle.chart_expert}) > 1:
        router_reports, _router_results = _train_mode_ensemble(
            bundle.chart_expert,
            mode=RepresentationMode.XYZ_ROUTER_EXPERTS,
            environment=environment,
            config=config,
            output_dir=stage,
        )
        training_reports.extend(router_reports)
    metrics = pd.DataFrame.from_records(training_reports)
    _write_parquet(metrics, stage / "student_metrics.parquet")
    student = config["student"]
    per_mode = metrics.groupby("mode", sort=True).agg(
        accepted_fraction=("accepted_fraction", "min"),
        fk_p95_mm=("fk_p95_mm", "max"),
        fk_max_mm=("fk_max_mm", "max"),
    )
    mode_checks = {
        str(mode): bool(
            row.accepted_fraction >= float(student["validation_minimum_accepted_fraction"])
            and row.fk_p95_mm <= float(student["validation_fk_p95_max_mm"])
            and row.fk_max_mm <= float(student["validation_fk_max_mm"])
        )
        for mode, row in per_mode.iterrows()
    }
    report = {
        "scientific_gate_pass": bool(mode_checks and all(mode_checks.values())),
        "mode_checks": mode_checks,
        "row_count": len(bundle.supervision_records),
        "primary_row_count": len(bundle.primary_canonical),
        "expert_row_count": len(bundle.chart_expert),
        "active_row_count": bundle.budget_report.active_count,
        "padding_count": bundle.budget_report.padding_count,
        "training_seeds": list(map(int, student["seeds"])),
        "student_plus_dls_baseline_registered": True,
    }
    _write_json(stage / "student_report.json", report)
    return _gate(stage / "gate.json", {"operational_completion": True}, **report)


def stage_formal_gate(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["formal_gate"]
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"Formal Gate stage is already sealed: {stage}")
    reach_path = output_root / STAGE_DIRS["reach_extension"] / "gate.json"
    pilot_path = output_root / STAGE_DIRS["repaired_pilot"] / "gate.json"
    student_path = output_root / STAGE_DIRS["exploratory_student"] / "gate.json"
    reach = _read_json(reach_path) if reach_path.is_file() else {}
    pilot = _read_json(pilot_path) if pilot_path.is_file() else {}
    student = _read_json(student_path) if student_path.is_file() else {}
    inventory_gate_path = output_root / STAGE_DIRS["inventory"] / "gate.json"
    inventory_gate = (
        _read_json(inventory_gate_path) if inventory_gate_path.is_file() else {}
    )
    stage_source_git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    upstream_source_git_sha = inventory_gate.get("source_git_sha")
    repaired = output_root / STAGE_DIRS["repaired_pilot"]
    labelable = float(pilot.get("labelable_measure_ratio", 0.0))
    unresolved = float(pilot.get("unresolved_abstain_ratio", 1.0))
    abstention = float(pilot.get("unresolved_abstain_ratio", 1.0))
    minimum_x_bin = 0.0
    single_cell_ratio = 1.0
    largest_region_x_bins = 0
    n_min = 10**12
    if (repaired / "selected_cell_classification.parquet").is_file():
        cells = pd.read_parquet(repaired / "selected_cell_classification.parquet")
        task_nodes = pd.read_parquet(repaired / "audit_tasks.parquet")
        partition = pd.read_parquet(repaired / "primary_partition.parquet")
        charts = pd.read_parquet(repaired / "charts.parquet")
        parent_path = output_root / STAGE_DIRS["inventory"] / "patch_task_nodes.parquet"
        del parent_path
        labelable_cells = cells[cells["empirical_labelable_fraction"].gt(0.0)].copy()
        cell_fraction = {
            (int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)):
            float(row.empirical_labelable_fraction)
            for row in labelable_cells.itertuples(index=False)
        }
        x_values_by_bin: dict[int, list[float]] = {}
        assigned_by_node = partition.set_index("task_node_id")["assigned_section_id"].to_dict()
        for row in task_nodes.itertuples(index=False):
            cell = (int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz))
            x_bin = int(math.floor((float(row.x_m) - 1.015498) / 0.010))
            x_values_by_bin.setdefault(x_bin, []).append(
                cell_fraction.get(cell, 0.0) if assigned_by_node.get(int(row.task_node_id)) is not None else 0.0
            )
        minimum_x_bin = float(
            min((np.mean(values) for values in x_values_by_bin.values()), default=0.0)
        )
        valid_charts = charts[charts["section_valid"].astype(bool)]
        single_cell_sections = set(
            valid_charts.loc[valid_charts["selection_count"].le(5), "section_id"].astype(str)
        )
        assigned = partition[partition["assigned_section_id"].notna()].copy()
        single_nodes = assigned[assigned["assigned_section_id"].astype(str).isin(single_cell_sections)]
        single_cell_ratio = float(len(single_nodes) / max(1, len(assigned)))
        task_with_partition = task_nodes.merge(
            partition[["task_node_id", "assigned_section_id"]], on="task_node_id", how="left"
        )
        spans = task_with_partition[task_with_partition["assigned_section_id"].notna()].assign(
            x_bin=lambda frame: np.floor((frame["x_m"] - 1.015498) / 0.010).astype(int)
        ).groupby("assigned_section_id")["x_bin"].nunique()
        largest_region_x_bins = int(spans.max()) if len(spans) else 0
    budget_path = output_root / STAGE_DIRS["exploratory_student"] / "budget_report.json"
    if budget_path.is_file():
        budget = _read_json(budget_path)
        # Every supervised primary/expert row and the active reserve count.
        n_min = int(budget.get("deduplicated_candidate_count", 10**12))
    formal = dict(
        evaluate_formal_admission_gate(
            reach_convergence_gate=bool(reach.get("reach_convergence_gate", False)),
            selection_stability_gate=bool(pilot.get("canonical_selection_stability_gate", False)),
            labelable_measure_ratio=labelable,
            minimum_x_bin_coverage=minimum_x_bin,
            unresolved_ratio=unresolved,
            abstention_ratio=abstention,
            single_cell_chart_measure_ratio=single_cell_ratio,
            largest_region_x_bin_count=largest_region_x_bins,
            atlas_audit_gate=bool(pilot.get("atlas_audit_gate_pass", False)),
            representation_frozen=bool(pilot.get("static_inverse_authorized", False)),
            n_min=n_min,
            student_gate=bool(student.get("scientific_gate_pass", False)),
        )
    )
    report = {
        **formal,
        "labelable_measure_ratio": labelable,
        "minimum_x_bin_coverage": minimum_x_bin,
        "unresolved_ratio": unresolved,
        "abstention_ratio": abstention,
        "single_cell_chart_measure_ratio": single_cell_ratio,
        "largest_region_x_bin_count": largest_region_x_bins,
        "n_min": n_min,
        "direct_threshold_relaxation_authorized": False,
        "formal_generation_authorized": formal["gate_pass"],
        "stage_source_git_sha": stage_source_git_sha,
        "upstream_scientific_source_git_sha": upstream_source_git_sha,
        "aggregation_only_hotfix": bool(
            upstream_source_git_sha
            and str(upstream_source_git_sha) != stage_source_git_sha
        ),
    }
    _write_json(stage / "formal_admission_report.json", report)
    formal_checks = {
        "operational_completion": True,
        **{str(key): bool(value) for key, value in formal["checks"].items()},
    }
    gate_evidence = {
        key: value for key, value in report.items() if key not in {"gate_pass", "checks"}
    }
    return _gate(stage / "gate.json", formal_checks, **gate_evidence)


def stage_summary(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["summary"]
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "gate.json").is_file():
        raise FileExistsError(f"summary stage is already sealed: {stage}")
    stage_gates: dict[str, Any] = {}
    for name, directory in STAGE_DIRS.items():
        if name == "summary":
            continue
        path = output_root / directory / "gate.json"
        stage_gates[name] = _read_json(path) if path.is_file() else {"missing": True}
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.is_relative_to(stage):
            continue
        artifacts.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(stage / "artifact_manifest.json", manifest)
    formal = stage_gates.get("formal_gate", {})
    report = {
        "operational_stage_count": len(stage_gates),
        "missing_stage_names": sorted(
            name for name, gate in stage_gates.items() if gate.get("missing", False)
        ),
        "patch_repair_gate": bool(stage_gates.get("ablations", {}).get("gate_pass", False)),
        "selection_stability_gate": bool(stage_gates.get("stability", {}).get("gate_pass", False)),
        "reach_convergence_gate": bool(
            stage_gates.get("reach_extension", {}).get("reach_convergence_gate", False)
        ),
        "repaired_pilot_gate": bool(
            stage_gates.get("repaired_pilot", {}).get("scientific_gate_pass", False)
        ),
        "student_gate": bool(
            stage_gates.get("exploratory_student", {}).get("scientific_gate_pass", False)
        ),
        "formal_generation_authorized": bool(formal.get("formal_generation_authorized", False)),
        "deployment_claim": False,
        "stage_gates": stage_gates,
    }
    _write_json(stage / "summary_report.json", report)
    return _gate(
        stage / "gate.json",
        {"summary_and_manifest_complete": True},
        **report,
        artifact_count=len(artifacts),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v14_1_cross_cell_repair.yaml"),
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage", choices=STAGE_ORDER, default="inventory")
    parser.add_argument("--patch-id", default=None, help=argparse.SUPPRESS)
    return parser


STAGE_RUNNERS = {
    "inventory": stage_inventory,
    "ablations": stage_ablations,
    "stability": stage_stability,
    "reach_extension": stage_reach_extension,
    "repaired_pilot": stage_repaired_pilot,
    "exploratory_student": stage_exploratory_student,
    "formal_gate": stage_formal_gate,
    "summary": stage_summary,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.patch_id is not None:
        config["_patch_filter"] = str(args.patch_id)
    project_root = project_root_from(SOURCE_ROOT)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else project_root / str(config["output_root"])
    )
    gate = STAGE_RUNNERS[args.stage](config, project_root, output_root)
    print(json.dumps(_strict(gate), sort_keys=True, indent=2))
    return 0 if gate.get("gate_pass", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
