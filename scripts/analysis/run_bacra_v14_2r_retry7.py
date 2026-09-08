#!/usr/bin/env python3
"""Run BACRA V14.2R retry7 as a Gate-driven, gauge-locked funnel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

import run_bacra_v14_2r_stitched_atlas as base
from quasi_exp.teacher.canonical_atlas import AtlasCandidate
from quasi_exp.teacher.canonical_gauge import (
    GaugeCorrectorPolicy,
    gauge_locked_predictor_corrector,
)
from quasi_exp.teacher.holonomy_diagnostics import (
    audit_canonical_reset_edges,
    audit_cycle_all_starts,
    audit_refinement_edge_ablation,
    replay_cycle_with_prefix_trace,
)
from quasi_exp.teacher.optimized_continuation import (
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.workspace_atlas_repair import atlas_nodes_from_frames
from run_trajectory_canonical_teacher_v10 import load_environment


STAGE_DIRS = {
    "inventory": "00_inventory",
    "lineage_audit": "01_lineage_audit",
    "kr_ablation": "02_kr_ablation",
    "holonomy_diagnostics": "03_holonomy_diagnostics",
    "gauge_kernel_selection": "04_gauge_kernel_selection",
    "patch07_repair": "05_patch07_repair",
    "four_patch_gate": "06_four_patch_gate",
    "reach_round8": "07_reach_round8",
    "confirmation": "08_twelve_patch_confirmation",
    "meso_bridge": "09_meso_bridge",
    "summary": "10_summary",
    # Shared helpers use these semantic aliases.
    "replacement_confirmation": "01_replacement_confirmation",
}
STAGE_ORDER = tuple(name for name in STAGE_DIRS if name != "replacement_confirmation")
base.STAGE_DIRS.update(STAGE_DIRS)


def _retry6_root(config: Mapping[str, Any], project_root: Path) -> Path:
    return project_root / str(config["sources"]["retry6_root"])


def _prior_retry7_root(
    config: Mapping[str, Any], project_root: Path
) -> Path | None:
    value = config.get("sources", {}).get("retry7_reference_root")
    return None if value is None else project_root / str(value)


def _verify_prior_stage(
    prior_root: Path, name: str
) -> dict[str, Any]:
    stage = prior_root / STAGE_DIRS[name]
    manifest_path = stage / "completion_manifest.json"
    manifest = base._read_json(manifest_path)
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError(f"prior retry7 stage schema is invalid: {manifest_path}")
    if str(manifest.get("stage_name", "")) != str(name):
        raise RuntimeError(f"prior retry7 stage identity is invalid: {manifest_path}")
    records = tuple(manifest.get("artifacts", ()))
    if "gate.json" not in {str(record.get("path", "")) for record in records}:
        raise RuntimeError(f"prior retry7 stage has no sealed gate: {manifest_path}")
    verified: list[dict[str, Any]] = []
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"prior retry7 stage path is unsafe: {relative}")
        path = stage / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != str(record["sha256"])
        ):
            raise RuntimeError(f"prior retry7 stage artifact mismatch: {path}")
        verified.append(dict(record))
    fixed = base._read_json(
        prior_root / STAGE_DIRS["inventory"] / "source_fixed_point.json"
    )
    closure_checks = {
        "source_sha": str(manifest.get("source_sha", ""))
        == str(fixed.get("source_sha", "")),
        "config_sha256": str(manifest.get("config_sha256", ""))
        == str(fixed.get("config_sha256", "")),
        "runtime_sha256": str(manifest.get("runtime_sha256", ""))
        == str(fixed.get("runtime_sha256", "")),
    }
    if not all(closure_checks.values()):
        raise RuntimeError(
            f"prior retry7 stage fixed-point closure failed: {name} {closure_checks}"
        )
    return {
        "stage": stage,
        "manifest_path": manifest_path,
        "manifest_sha256": base.sha256_file(manifest_path),
        "source_sha": str(manifest["source_sha"]),
        "config_sha256": str(manifest["config_sha256"]),
        "runtime_sha256": str(manifest["runtime_sha256"]),
        "artifact_count": len(verified),
        "artifact_closure_sha256": base._payload_sha256(
            {"stage_name": name, "artifacts": verified}
        ),
        "records": {str(record["path"]): record for record in verified},
    }


def _complete(
    stage: Path,
    config: Mapping[str, Any],
    name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    base._write_json(stage / "gate.json", result)
    base._write_stage_completion_manifest(stage, config=config, stage_name=name)
    return result


def _require(config: Mapping[str, Any], output_root: Path, name: str) -> dict[str, Any]:
    result = base._load_validated_stage_result(
        output_root / STAGE_DIRS[name], config=config, stage_name=name
    )
    if result is None:
        raise FileNotFoundError(f"retry7 upstream stage is not sealed: {name}")
    return result


def _reuse_registered_stage(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    name: str,
) -> dict[str, Any] | None:
    registered = set(map(str, config.get("reuse_sealed_stages", ())))
    prior_root = _prior_retry7_root(config, project_root)
    if name not in registered or prior_root is None:
        return None
    closure = _verify_prior_stage(prior_root, name)
    prior_stage = Path(closure["stage"])
    records = dict(closure["records"])

    def verified_artifact(relative: Path) -> Path:
        record = records.get(relative.as_posix())
        path = prior_stage / relative
        if (
            record is None
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != str(record["sha256"])
        ):
            raise RuntimeError(
                f"sealed retry7 stage reference is absent or mismatched: {path}"
            )
        return path

    gate_path = verified_artifact(Path("gate.json"))
    prior_gate = base._read_json(gate_path)
    stage = output_root / STAGE_DIRS[name]
    stage.mkdir(parents=True, exist_ok=True)
    reference = {
        "evidence_reused": True,
        "evidence_role": "stage_completion_hash_verified_read_only_reference",
        "prior_retry7_root": str(prior_root),
        "prior_retry7_source_sha": str(closure["source_sha"]),
        "prior_stage_manifest_sha256": str(closure["manifest_sha256"]),
        "prior_stage_artifact_closure_sha256": str(
            closure["artifact_closure_sha256"]
        ),
        "prior_gate_path": str(gate_path),
        "prior_gate_sha256": base.sha256_file(gate_path),
        "prior_gate_pass": bool(prior_gate.get("gate_pass", False)),
    }
    base._write_json(stage / "reused_evidence_reference.json", reference)
    selected_kernel = None
    if name == "gauge_kernel_selection":
        selected_path = verified_artifact(Path("selected_kernel.json"))
        selected_kernel = {
            **base._read_json(selected_path),
            "policy_evidence_role": "sealed_policy_reference_not_fresh_guard",
            "prior_selected_kernel_sha256": base.sha256_file(selected_path),
        }
        base._write_json(stage / "selected_kernel.json", selected_kernel)
    payload = {
        **prior_gate,
        "gate_pass": bool(prior_gate.get("gate_pass", False)),
        "operational_completion": True,
        **reference,
    }
    if selected_kernel is not None:
        payload["selected_kernel"] = selected_kernel
    return _complete(
        stage,
        config,
        name,
        payload,
    )


def stage_inventory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["inventory"]
    stage.mkdir(parents=True, exist_ok=True)
    retry6 = _retry6_root(config, project_root)
    v14_2 = project_root / str(config["sources"]["v14_2_root"])
    v14_2_manifest = v14_2 / "08_summary/artifact_manifest.json"
    required = (
        retry6 / "11_summary/artifact_manifest.json",
        retry6 / "11_summary/gate.json",
        retry6 / "06_search_stability/patch_07/task_graph_refined/audit_v2_schedules.parquet",
        retry6 / "06_search_stability/patch_07/task_graph_refined/audit_v2_executions.parquet",
        v14_2_manifest,
        v14_2 / "04_reach_round6/replica_a_slab_round6.parquet",
        v14_2 / "04_reach_round6/replica_b_slab_round6.parquet",
        v14_2 / "04_reach_round6/reach_round6_report.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"retry7 upstream retry6 evidence is incomplete: {missing}")
    upstream_closure = base._verify_upstream_artifact_manifest(required[0])
    reach_round6_closure = base._verify_upstream_artifact_manifest(v14_2_manifest)
    prior_retry7 = _prior_retry7_root(config, project_root)
    prior_stage_closures = (
        [
            _verify_prior_stage(prior_retry7, name)
            for name in map(str, config.get("reuse_sealed_stages", ()))
        ]
        if prior_retry7 is not None
        else []
    )
    prior_retry7_closure = (
        {
            "artifact_count": sum(
                int(item["artifact_count"]) for item in prior_stage_closures
            ),
            "artifact_closure_sha256": base._payload_sha256(
                {
                    "stages": [
                        {
                            "stage": Path(item["stage"]).name,
                            "manifest_sha256": item["manifest_sha256"],
                            "artifact_closure_sha256": item[
                                "artifact_closure_sha256"
                            ],
                        }
                        for item in prior_stage_closures
                    ]
                }
            ),
        }
        if prior_stage_closures
        else None
    )
    base._write_json(
        stage / "upstream_manifest_check.json", upstream_closure
    )
    base._write_json(
        stage / "reach_round6_manifest_check.json", reach_round6_closure
    )
    if prior_retry7_closure is not None:
        base._write_json(
            stage / "prior_retry7_stage_closure_check.json",
            prior_retry7_closure,
        )
    fixed = {
        "source_sha": base._git_sha(),
        "config_sha256": base.sha256_file(Path(str(config["config_path"]))),
        "runtime_sha256": base._runtime_sha256(),
        "working_tree_clean": base._tree_clean(),
        "retry6_root": str(retry6),
        "retry6_artifact_manifest_sha256": base.sha256_file(required[0]),
        "retry6_artifact_closure_sha256": upstream_closure[
            "artifact_closure_sha256"
        ],
        "reach_round6_artifact_manifest_sha256": base.sha256_file(
            v14_2_manifest
        ),
        "reach_round6_artifact_closure_sha256": reach_round6_closure[
            "artifact_closure_sha256"
        ],
        "prior_retry7_root": (
            str(prior_retry7) if prior_retry7 is not None else None
        ),
        "prior_retry7_stage_manifest_sha256": (
            base._payload_sha256(
                {
                    "manifests": [
                        item["manifest_sha256"] for item in prior_stage_closures
                    ]
                }
            )
            if prior_stage_closures
            else None
        ),
        "prior_retry7_artifact_closure_sha256": (
            prior_retry7_closure["artifact_closure_sha256"]
            if prior_retry7_closure is not None
            else None
        ),
    }
    base._write_json(stage / "source_fixed_point.json", fixed)
    base._write_json(stage / "environment.json", base._runtime_closure())
    registered = {
        "protocol_version": "retry7",
        "stage_order": list(STAGE_ORDER),
        "diagnostic_patches": list(config["diagnostic_patch_ids"]),
        "kr_variants": ["main_K1_R5", "K4_R5", "K1_R8", "K4_R8"],
        "root_contract": "R5_is_frozen_R8_prefix",
        "worker_limit": 12,
        "threads_per_worker": 1,
        "thresholds": {
            "geometry_p95_deg": 0.5,
            "geometry_max_deg": 1.0,
            "repeat_p95_deg": 0.2,
            "fk_residual_mm": 3.0,
        },
    }
    base._write_json(stage / "registered_protocol.json", registered)
    schedules = pd.read_parquet(required[2])
    critical = schedules[schedules["audit_kind"].eq("fundamental_cycle")]
    estimate = {
        "critical_cycle_count": 1,
        "critical_all_start_direction_perturbation_executions": 10 * 2 * 7,
        "guard_cycle_executions": 96 * 2 * 3,
        "targeted_diagnostic_executions": 10 * 2 * 7 + 96 * 2 * 3,
        "registered_limit": int(
            config["audit_execution"]["targeted_diagnostic_execution_max"]
        ),
        "retry6_refined_cycle_schedule_count": int(len(critical)),
    }
    base._write_json(stage / "dry_run_cost_estimate.json", estimate)
    checks = {
        "working_tree_clean": bool(fixed["working_tree_clean"]),
        "retry6_inputs_present": not missing,
        "retry6_artifact_closure_complete": int(
            upstream_closure["artifact_count"]
        )
        > 0,
        "reach_round6_artifact_closure_complete": int(
            reach_round6_closure["artifact_count"]
        )
        > 0,
        "prior_retry7_artifact_closure_complete": (
            prior_retry7_closure is None
            or int(prior_retry7_closure["artifact_count"]) > 0
        ),
        "targeted_diagnostics_below_limit": (
            estimate["targeted_diagnostic_executions"]
            < estimate["registered_limit"]
        ),
        "twelve_single_thread_workers": config["parallel"]
        == {"patch_workers": 12, "numerical_threads_per_worker": 1},
    }
    return _complete(
        stage,
        config,
        "inventory",
        {"gate_pass": bool(all(checks.values())), "checks": checks, **fixed, **estimate},
    )


def _retry6_variant_directory(retry6: Path, patch: str, variant: str) -> Path:
    if variant == "K4_R8":
        return retry6 / "03_rooted_baseline" / patch / "baseline"
    return retry6 / "06_search_stability" / patch / variant


def stage_lineage_audit(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require(config, output_root, "inventory")
    reused = _reuse_registered_stage(
        config, project_root, output_root, "lineage_audit"
    )
    if reused is not None:
        return reused
    stage = output_root / STAGE_DIRS["lineage_audit"]
    stage.mkdir(parents=True, exist_ok=True)
    retry6 = _retry6_root(config, project_root)
    patch = "patch_07"
    variants = ("main_K1_R5", "root_dropout", "root_order2", "task_graph_refined", "K4_R8")
    lineage_rows: list[dict[str, Any]] = []
    primary_by_variant: dict[str, pd.DataFrame] = {}
    for variant in variants:
        directory = _retry6_variant_directory(retry6, patch, variant)
        report = base._read_json(directory / "report.json")
        charts = pd.read_parquet(directory / "section_charts.parquet")
        selected = set(map(str, report.get("selected_stitch_component", ())))
        for row in charts.itertuples(index=False):
            lineage_rows.append(
                {
                    "patch_id": patch,
                    "variant": variant,
                    "chart_id": str(row.chart_id),
                    "root_node_id": int(row.root_node_id),
                    "root_candidate_id": str(row.root_candidate_id),
                    "selection_count": int(row.selection_count),
                    "selected_component": str(row.chart_id) in selected,
                    "canonical_anchor_retained": bool(
                        report.get("canonical_anchor_retained", False)
                    ),
                }
            )
        primary_by_variant[variant] = pd.read_parquet(directory / "primary_atlas.parquet")
    lineage = pd.DataFrame.from_records(lineage_rows)
    base._write_parquet(lineage, stage / "root_component_lineage.parquet")
    ranking = lineage.sort_values(
        ["variant", "selected_component", "selection_count"],
        ascending=[True, False, False],
        kind="stable",
    )
    base._write_parquet(ranking, stage / "component_selection_ranking.parquet")
    contrasts = {
        "root_dropout_vs_main": ("main_K1_R5", "root_dropout"),
        "root_order2_vs_main": ("main_K1_R5", "root_order2"),
        "refined_vs_same_budget_main": ("main_K1_R5", "task_graph_refined"),
        "root_budget_K1R5_vs_K4R8_diagnostic": ("main_K1_R5", "K4_R8"),
    }
    rows = []
    node_rows = []
    for name, (left_name, right_name) in contrasts.items():
        left_dir = _retry6_variant_directory(retry6, patch, left_name)
        right_dir = _retry6_variant_directory(retry6, patch, right_name)
        metric = base._frame_stability(
            primary_by_variant[left_name],
            primary_by_variant[right_name],
            config,
            left_directory=left_dir,
            right_directory=right_dir,
        )
        rows.append({"contrast": name, "left": left_name, "right": right_name, **metric})
        left = primary_by_variant[left_name].set_index("task_node_id")
        right = primary_by_variant[right_name].set_index("task_node_id")
        for node in sorted(set(left.index) & set(right.index)):
            node_rows.append(
                {
                    "contrast": name,
                    "task_node_id": int(node),
                    "beta_gap_deg": base.beta_rms_deg(
                        left.loc[node, list(base.BETA_COLUMNS)].to_numpy(float),
                        right.loc[node, list(base.BETA_COLUMNS)].to_numpy(float),
                    ),
                    "left_abstained": bool(left.loc[node, "abstained"]),
                    "right_abstained": bool(right.loc[node, "abstained"]),
                }
            )
    comparison = pd.DataFrame.from_records(rows)
    base._write_parquet(comparison, stage / "corrected_nested_stability.parquet")
    base._write_parquet(
        pd.DataFrame.from_records(node_rows),
        stage / "nodewise_beta_field_comparison.parquet",
    )
    stable = comparison.set_index("contrast")["gate_pass"].to_dict()
    switch = bool(
        stable["root_dropout_vs_main"]
        and stable["root_order2_vs_main"]
        and not stable["root_budget_K1R5_vs_K4R8_diagnostic"]
    )
    summary = {
        "root_budget_component_switch_confirmed": switch,
        "lineage_row_count": len(lineage),
        "comparison_count": len(comparison),
        "scientific_gate": "diagnostic_only",
    }
    base._write_json(stage / "summary.json", summary)
    return _complete(
        stage,
        config,
        "lineage_audit",
        {"gate_pass": True, "operational_completion": True, **summary},
    )


def _run_patch_jobs(
    config: Mapping[str, Any], output_root: Path, stage_name: str, jobs: Sequence[tuple[str, str]]
) -> None:
    base._run_patch_jobs(
        config,
        output_root,
        stage_name,
        jobs,
        runner_path=Path(__file__).resolve(),
    )


def stage_kr_ablation(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    lineage = _require(config, output_root, "lineage_audit")
    reused = _reuse_registered_stage(
        config, project_root, output_root, "kr_ablation"
    )
    if reused is not None:
        return reused
    stage = output_root / STAGE_DIRS["kr_ablation"]
    if config.get("_patch_id"):
        report = base._execute_patch(
            config,
            project_root,
            output_root,
            str(config["_patch_id"]),
            str(config["_variant"]),
            stage / str(config["_patch_id"]) / str(config["_variant"]),
        )
        return {"gate_pass": True, "worker_report": report}
    if not lineage.get("gate_pass", False):
        return _complete(stage, config, "kr_ablation", base.write_scientific_skip(stage, "lineage_audit_failed"))
    variants = ("main_K1_R5", "K4_R5", "K1_R8", "K4_R8")
    _run_patch_jobs(config, output_root, "kr_ablation", [("patch_07", item) for item in variants])
    reports = {
        name: base._read_json(stage / "patch_07" / name / "report.json")
        for name in variants
    }
    fields = {
        name: pd.read_parquet(stage / "patch_07" / name / "primary_atlas.parquet")
        for name in variants
    }
    contrasts = (
        ("K_at_R5", "main_K1_R5", "K4_R5"),
        ("K_at_R8", "K1_R8", "K4_R8"),
        ("R_at_K1", "main_K1_R5", "K1_R8"),
        ("R_at_K4", "K4_R5", "K4_R8"),
    )
    rows = []
    for name, left, right in contrasts:
        metric = base._frame_stability(
            fields[left],
            fields[right],
            config,
            left_directory=stage / "patch_07" / left,
            right_directory=stage / "patch_07" / right,
        )
        rows.append({"contrast": name, "left": left, "right": right, **metric})
    comparison = pd.DataFrame.from_records(rows)
    base._write_parquet(comparison, stage / "kr_factorial_comparison.parquet")
    primary = pd.concat(
        [frame.assign(variant=name) for name, frame in fields.items()],
        ignore_index=True,
    )
    base._write_parquet(primary, stage / "kr_primary_fields.parquet")
    pre = pd.concat(
        [
            pd.read_parquet(stage / "patch_07" / name / "pre_pruning_hypotheses.parquet").assign(variant=name)
            for name in variants
        ],
        ignore_index=True,
    )
    base._write_parquet(pre, stage / "pre_pruning_hypotheses.parquet")
    by_contrast = comparison.set_index("contrast")["gate_pass"].astype(bool).to_dict()
    checks = {
        "K_stability": by_contrast["K_at_R5"] and by_contrast["K_at_R8"],
        "root_budget_stability_under_anchor_lock": by_contrast["R_at_K1"] and by_contrast["R_at_K4"],
        "all_anchor_components_selected": all(
            bool(report["canonical_anchor_component_selected"])
            for report in reports.values()
        ),
        "nested_R5_R8_registry": (
            reports["main_K1_R5"].get("frozen_root_registry")
            == reports["K1_R8"].get("frozen_root_registry")
            and reports["K4_R5"].get("frozen_root_registry")
            == reports["K4_R8"].get("frozen_root_registry")
        ),
    }
    anchor_report = {
        name: {
            key: report.get(key)
            for key in (
                "canonical_anchor_root_key",
                "canonical_anchor_qualified",
                "canonical_anchor_component_selected",
                "canonical_anchor_component_coverage",
                "canonical_anchor_fallback_reason",
            )
        }
        for name, report in reports.items()
    }
    base._write_json(stage / "anchor_lock_report.json", anchor_report)
    performance = {
        name: {
            "runtime_s": report.get("runtime_s"),
            "phase_runtime_s": report.get("phase_runtime_s"),
            "pre_pruning_proposal_count": report.get("pre_pruning_proposal_count"),
            "pre_pruning_cluster_count": report.get("pre_pruning_cluster_count"),
        }
        for name, report in reports.items()
    }
    base._write_json(stage / "performance_counters.json", performance)
    return _complete(
        stage,
        config,
        "kr_ablation",
        {
            "gate_pass": bool(all(checks.values())),
            "checks": checks,
            "factorial_comparison": rows,
            "patch_reports": reports,
            "true_beam_required": not checks["K_stability"],
        },
    )


def _patch_context(
    config: Mapping[str, Any], project_root: Path, patch_id: str
) -> dict[str, Any]:
    retry6 = _retry6_root(config, project_root)
    legacy_config = base.legacy.load_config(
        SOURCE_ROOT / str(config["sources"]["legacy_config"])
    )
    tasks, original_edges, _candidate_frame, _assignments = base.legacy._patch_inputs(
        legacy_config, project_root, patch_id
    )
    retry_patch = project_root / str(config["sources"]["retry4_root"]) / "01_patch_ablations" / patch_id
    refined_edges = pd.read_parquet(retry_patch / "E3_dynamic_insertion_task_edges.parquet")
    if patch_id == "patch_07":
        refined_edges = base._refine_task_graph(tasks, refined_edges)
        directory = retry6 / "06_search_stability" / patch_id / "task_graph_refined"
    else:
        directory = retry6 / "06_search_stability" / patch_id / "main_K1_R5"
    primary = pd.read_parquet(directory / "primary_atlas.parquet")
    node_rows = tasks.set_index("task_node_id")
    canonical: dict[int, AtlasCandidate] = {}
    for row in primary[~primary["abstained"].astype(bool)].itertuples(index=False):
        beta = np.asarray([getattr(row, name) for name in base.BETA_COLUMNS], dtype=float)
        canonical[int(row.task_node_id)] = AtlasCandidate(
            int(row.task_node_id),
            f"retry6_primary_{int(row.task_node_id)}",
            beta,
            0.0,
            180.0,
            1.0,
            posture_cost=float(np.linalg.norm(beta)),
            condition_number=0.0,
        )
    nodes = atlas_nodes_from_frames(tasks, refined_edges)
    node_by_id = {node.node_id: node for node in nodes}
    environment = optimized_forward(
        load_environment(project_root, SOURCE_ROOT / str(config["robot_config"]))
    )
    schedules = pd.read_parquet(directory / "audit_v2_schedules.parquet")
    executions = pd.read_parquet(directory / "audit_v2_executions.parquet")
    return {
        "tasks": tasks,
        "original_edges": original_edges,
        "refined_edges": refined_edges,
        "canonical": canonical,
        "nodes": node_by_id,
        "environment": environment,
        "schedules": schedules,
        "executions": executions,
        "directory": directory,
    }


def _critical_cycle(context: Mapping[str, Any]) -> tuple[int, ...]:
    failed_ids = set(
        context["executions"].loc[
            context["executions"]["classification"].eq("geometric_branch_disagreement"),
            "schedule_id",
        ].astype(str)
    )
    rows = context["schedules"][
        context["schedules"]["schedule_id"].astype(str).isin(failed_ids)
        & context["schedules"]["audit_kind"].eq("fundamental_cycle")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"retry7 expected exactly one critical physical cycle, got {len(rows)}")
    return tuple(map(int, rows.iloc[0]["path_node_ids"]))


def _baseline_continuation(context: Mapping[str, Any]):
    return make_optimized_predictor_corrector_continuation(
        context["environment"],
        damping=1.0e-3,
        max_corrector_iterations=100,
        residual_tolerance_mm=3.0,
    )


def stage_holonomy_diagnostics(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    kr = _require(config, output_root, "kr_ablation")
    reused = _reuse_registered_stage(
        config, project_root, output_root, "holonomy_diagnostics"
    )
    if reused is not None:
        return reused
    stage = output_root / STAGE_DIRS["holonomy_diagnostics"]
    stage.mkdir(parents=True, exist_ok=True)
    if not kr.get("gate_pass", False):
        return _complete(stage, config, "holonomy_diagnostics", base.write_scientific_skip(stage, "kr_ablation_failed"))
    context = _patch_context(config, project_root, "patch_07")
    cycle = _critical_cycle(context)
    continuation = _baseline_continuation(context)
    baseline = replay_cycle_with_prefix_trace(
        cycle,
        patch_id="patch_07",
        canonical_by_node=context["canonical"],
        task_nodes=context["nodes"],
        continuation=continuation,
        environment=context["environment"],
    )
    all_starts = audit_cycle_all_starts(
        cycle,
        patch_id="patch_07",
        canonical_by_node=context["canonical"],
        task_nodes=context["nodes"],
        continuation=continuation,
        perturbation_magnitudes_rad=config["holonomy"]["perturbation_magnitudes_rad"],
        environment=context["environment"],
    )
    reset = audit_canonical_reset_edges(
        cycle,
        patch_id="patch_07",
        canonical_by_node=context["canonical"],
        task_nodes=context["nodes"],
        continuation=continuation,
    )
    base._write_parquet(baseline, stage / "prefix_trajectories.parquet")
    base._write_parquet(reset, stage / "canonical_reset_edges.parquet")
    base._write_parquet(all_starts, stage / "all_cycle_starts.parquet")
    base._write_parquet(all_starts, stage / "perturbation_scan.parquet")
    original = {
        tuple(sorted((int(row.left_node_id), int(row.right_node_id))))
        for row in context["original_edges"].itertuples(index=False)
    }
    refined = {
        tuple(sorted((int(row.left_node_id), int(row.right_node_id))))
        for row in context["refined_edges"].itertuples(index=False)
    }
    cycle_edges = {
        tuple(sorted(edge)) for edge in zip(cycle[:-1], cycle[1:])
    }
    additions = sorted((refined - original) & cycle_edges)
    ablation = audit_refinement_edge_ablation(
        sorted(refined), additions
    )
    base._write_parquet(ablation, stage / "refinement_edge_ablation.parquet")
    threshold = float(config["holonomy"]["prefix_drift_threshold_deg"])
    drift = baseline[baseline["geometry_gap_deg"] > threshold]
    first = (
        None
        if drift.empty
        else {key: base._strict(value) for key, value in drift.iloc[0].to_dict().items()}
    )
    base._write_json(stage / "first_drift_event.json", {"event": first})
    reset_static_pass = bool(
        len(reset)
        and reset["success"].all()
        and reset["geometry_gap_deg"].max() <= 1.0
        and reset["fk_residual_mm"].max() <= 3.0
    )
    free_transport_pass = bool(
        len(all_starts)
        and all_starts["success"].all()
        and all_starts["geometry_gap_deg"].max() <= 1.0
    )
    classification = (
        "accumulated_transport_holonomy"
        if reset_static_pass and not free_transport_pass
        else "static_section_or_mixed_failure"
    )
    summary = {
        "physical_cycle": list(cycle),
        "cycle_edge_count": len(cycle) - 1,
        "canonical_reset_section_gate": reset_static_pass,
        "free_transport_gate": free_transport_pass,
        "failure_classification": classification,
        "maximum_free_transport_gap_deg": float(all_starts["geometry_gap_deg"].max()),
        "maximum_reset_edge_gap_deg": float(reset["geometry_gap_deg"].max()),
        "refinement_edge_count_on_cycle": len(additions),
        "first_drift_event": first,
    }
    base._write_json(stage / "summary.json", summary)
    return _complete(
        stage,
        config,
        "holonomy_diagnostics",
        {"gate_pass": True, "diagnostic_complete": True, **summary},
    )


def _repeat_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compare repeat endpoints only after a trace closes at its physical start.

    A failed or truncated cycle can end at a different task node.  Treating its
    last available beta as though it were the closed-cycle endpoint conflates
    solver completeness with repeat disagreement.  Incomplete traces remain a
    hard Gate failure, but do not contribute a physically meaningless beta gap.
    """

    if frame.empty:
        return {
            "max_deg": math.inf,
            "p95_deg": math.inf,
            "trace_count": 0,
            "complete_trace_count": 0,
            "incomplete_trace_count": 0,
            "comparison_count": 0,
            "gate_pass": False,
        }
    identity = [
        name
        for name in (
            "guard_patch_id",
            "guard_schedule_id",
            "physical_entity_id",
            "kernel_id",
        )
        if name in frame.columns
    ]
    trace_keys = [*identity, "start_node_id", "direction", "repeat_id"]
    ordered = frame.sort_values([*trace_keys, "prefix_index"], kind="stable")
    trace_rows = []
    for key, group in ordered.groupby(trace_keys, sort=False, dropna=False):
        last = group.iloc[-1]
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(trace_keys, key_values))
        record.update(
            {
                "complete": bool(
                    group["success"].astype(bool).all()
                    and int(last["target_node_id"]) == int(last["start_node_id"])
                ),
                **{
                    f"beta_{index}": float(last[f"beta_{index}"])
                    for index in range(6)
                },
            }
        )
        trace_rows.append(record)
    traces = pd.DataFrame.from_records(trace_rows)
    complete = traces[traces["complete"].astype(bool)]
    comparison_keys = [*identity, "start_node_id", "direction"]
    gaps: list[float] = []
    for _key, group in complete.groupby(
        comparison_keys, sort=False, dropna=False
    ):
        group = group.sort_values("repeat_id", kind="stable")
        if len(group) < 2:
            continue
        beta = group[[f"beta_{index}" for index in range(6)]].to_numpy(float)
        gaps.extend(
            base.beta_rms_deg(beta[0], beta[index])
            for index in range(1, len(beta))
        )
    maximum = max(gaps) if gaps else 0.0
    p95 = float(np.percentile(gaps, 95)) if gaps else 0.0
    incomplete = int((~traces["complete"].astype(bool)).sum())
    return {
        "max_deg": float(maximum),
        "p95_deg": p95,
        "trace_count": int(len(traces)),
        "complete_trace_count": int(len(complete)),
        "incomplete_trace_count": incomplete,
        "comparison_count": len(gaps),
        "gate_pass": bool(incomplete == 0 and maximum <= 0.2),
    }


def _repeat_max(frame: pd.DataFrame) -> float:
    return float(_repeat_metrics(frame)["max_deg"])


def _select_gauge_candidate(
    candidates: Sequence[tuple[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    *,
    maximum_localized_excess_edges: int,
    maximum_localized_geometry_deg: float = 1.10,
) -> tuple[str, Any, str] | None:
    """Select a strict repair, else a solver-complete local-abstention probe.

    The fallback is authorization to test an explicit abstention mask.  It is
    not a critical-cycle pass and cannot itself authorize deployment.
    """

    policy_by_id = dict(candidates)
    for row in metrics:
        if bool(row["critical_gate"]) and row["kernel_id"] in policy_by_id:
            return str(row["kernel_id"]), policy_by_id[str(row["kernel_id"])], "critical_pass"
    for row in metrics:
        kernel_id = str(row["kernel_id"])
        if kernel_id == "C0_baseline" or kernel_id not in policy_by_id:
            continue
        if (
            bool(row["all_traces_complete"])
            and float(row["repeat_max_deg"]) <= 0.2
            and float(row["fk_residual_max_mm"]) <= 3.0
            and 0 < int(row["geometry_excess_edge_count"])
            <= int(maximum_localized_excess_edges)
            and float(row["geometry_max_deg"])
            <= float(maximum_localized_geometry_deg)
        ):
            return kernel_id, policy_by_id[kernel_id], "localized_abstention"
    return None


def _evaluate_kernel(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    cycle: Sequence[int],
    kernel_id: str,
    continuation: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = (
        context["environment"].performance_counters()
        if hasattr(context["environment"], "performance_counters")
        else {}
    )
    started = time.perf_counter()
    frame = audit_cycle_all_starts(
        cycle,
        patch_id="patch_07",
        canonical_by_node=context["canonical"],
        task_nodes=context["nodes"],
        continuation=continuation,
        perturbation_magnitudes_rad=config["holonomy"]["perturbation_magnitudes_rad"],
        kernel_id=kernel_id,
        environment=context["environment"],
    )
    wall = time.perf_counter() - started
    after = (
        context["environment"].performance_counters()
        if hasattr(context["environment"], "performance_counters")
        else {}
    )
    counter_delta = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in set(before) | set(after)
        if key != "kinematics_cache_entry_count"
    }
    cache_requests = (
        counter_delta.get("kinematics_cache_hit_count", 0)
        + counter_delta.get("kinematics_cache_miss_count", 0)
    )
    repeat = _repeat_metrics(frame)
    excess_edges = {
        (int(row.source_node_id), int(row.target_node_id))
        for row in frame.loc[
            frame["geometry_gap_deg"].gt(1.0)
        ].itertuples(index=False)
    }
    metrics = {
        "kernel_id": kernel_id,
        "wall_time_s": wall,
        "execution_count": int(
            frame.groupby(["start_node_id", "direction", "repeat_id"]).ngroups
        ),
        "continuation_segment_calls": len(frame),
        "geometry_max_deg": float(frame["geometry_gap_deg"].max()),
        "repeat_max_deg": repeat["max_deg"],
        "repeat_p95_deg": repeat["p95_deg"],
        "complete_trace_count": repeat["complete_trace_count"],
        "incomplete_trace_count": repeat["incomplete_trace_count"],
        "all_traces_complete": repeat["incomplete_trace_count"] == 0,
        "geometry_excess_edge_count": len(excess_edges),
        "fk_residual_max_mm": float(frame["fk_residual_mm"].max()),
        "solver_iteration_count": int(frame["solver_iterations"].sum()),
        "fk_calls": counter_delta.get("fk_row_count", 0),
        "jacobian_calls": counter_delta.get("jacobian_row_count", 0),
        "kinematics_cache_hit_count": counter_delta.get(
            "kinematics_cache_hit_count", 0
        ),
        "kinematics_cache_miss_count": counter_delta.get(
            "kinematics_cache_miss_count", 0
        ),
        "kinematics_cache_hit_rate": (
            counter_delta.get("kinematics_cache_hit_count", 0) / cache_requests
            if cache_requests
            else 0.0
        ),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        / 1024.0,
        "slsqp_calls": int(kernel_id.startswith("C4_")) * len(frame),
        "bounded_ls_calls": int(kernel_id == "C0_baseline") * len(frame),
    }
    metrics["critical_gate"] = bool(
        metrics["geometry_max_deg"] <= 1.0
        and metrics["repeat_max_deg"] <= 0.2
        and metrics["fk_residual_max_mm"] <= 3.0
        and metrics["all_traces_complete"]
    )
    return frame, metrics


def _recompute_reused_kernel_metrics(
    frame: pd.DataFrame,
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    kernel_id = str(prior["kernel_id"])
    repeat = _repeat_metrics(frame)
    excess_edges = {
        (int(row.source_node_id), int(row.target_node_id))
        for row in frame.loc[
            frame["geometry_gap_deg"].gt(1.0)
        ].itertuples(index=False)
    }
    metrics = dict(prior)
    metrics.update(
        {
            "kernel_id": kernel_id,
            "execution_count": int(
                frame.groupby(
                    ["start_node_id", "direction", "repeat_id"]
                ).ngroups
            ),
            "continuation_segment_calls": len(frame),
            "geometry_max_deg": float(frame["geometry_gap_deg"].max()),
            "repeat_max_deg": repeat["max_deg"],
            "repeat_p95_deg": repeat["p95_deg"],
            "complete_trace_count": repeat["complete_trace_count"],
            "incomplete_trace_count": repeat["incomplete_trace_count"],
            "all_traces_complete": repeat["incomplete_trace_count"] == 0,
            "geometry_excess_edge_count": len(excess_edges),
            "fk_residual_max_mm": float(frame["fk_residual_mm"].max()),
            "raw_trace_reused": True,
        }
    )
    metrics["critical_gate"] = bool(
        metrics["geometry_max_deg"] <= 1.0
        and metrics["repeat_max_deg"] <= 0.2
        and metrics["fk_residual_max_mm"] <= 3.0
        and metrics["all_traces_complete"]
    )
    return metrics


def stage_gauge_kernel_selection(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    holonomy = _require(config, output_root, "holonomy_diagnostics")
    reused = _reuse_registered_stage(
        config, project_root, output_root, "gauge_kernel_selection"
    )
    if reused is not None:
        return reused
    stage = output_root / STAGE_DIRS["gauge_kernel_selection"]
    stage.mkdir(parents=True, exist_ok=True)
    if not holonomy.get("gate_pass", False):
        return _complete(stage, config, "gauge_kernel_selection", base.write_scientific_skip(stage, "holonomy_diagnostics_failed"))
    context = _patch_context(config, project_root, "patch_07")
    cycle = _critical_cycle(context)
    candidates: list[tuple[str, GaugeCorrectorPolicy | None]] = [
        ("C0_baseline", None)
    ]
    for gain in map(float, config["gauge"]["gain_candidates"]):
        candidates.append(
            (
                f"C1_predictor_proximal_g{gain:g}",
                GaugeCorrectorPolicy(
                    "predictor_proximal",
                    gain,
                    float(config["gauge"]["maximum_gauge_step_deg"]),
                    0.0,
                    5.0,
                    100,
                    beta_weights=tuple(map(float, config["rooted_section"]["beta_weights"])),
                ),
            )
        )
    selected_gain = float(config["gauge"]["gain_candidates"][0])
    candidates.extend(
        [
            (
                "C2_predictor_proximal_2p5mm",
                GaugeCorrectorPolicy(
                    "predictor_proximal", selected_gain,
                    float(config["gauge"]["maximum_gauge_step_deg"]), 0.0,
                    2.5, 200,
                    beta_weights=tuple(map(float, config["rooted_section"]["beta_weights"])),
                ),
            ),
            (
                "C3_anchor_potential",
                GaugeCorrectorPolicy(
                    "anchor_potential", selected_gain,
                    float(config["gauge"]["maximum_gauge_step_deg"]),
                    float(config["gauge"]["anchor_weight"]), 2.5, 200,
                    beta_weights=tuple(map(float, config["rooted_section"]["beta_weights"])),
                ),
            ),
            (
                "C4_proximal_slsqp",
                GaugeCorrectorPolicy(
                    "proximal_slsqp", selected_gain,
                    float(config["gauge"]["maximum_gauge_step_deg"]),
                    float(config["gauge"]["anchor_weight"]), 1.25,
                    int(config["gauge"]["proximal_slsqp_maximum_iterations"]),
                    beta_weights=tuple(map(float, config["rooted_section"]["beta_weights"])),
                ),
            ),
        ]
    )
    critical_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    selected: tuple[str, GaugeCorrectorPolicy | None] | None = None
    selection_mode: str | None = None
    baseline_wall = None
    anchor_beta = context["canonical"][int(cycle[0])].beta_rad
    prior_root = _prior_retry7_root(config, project_root)
    reuse_traces = bool(config.get("reuse_pre_abstention_gauge_traces", False))
    if reuse_traces:
        if prior_root is None:
            raise RuntimeError("gauge trace reuse requires retry7_reference_root")
        prior_stage = prior_root / STAGE_DIRS["gauge_kernel_selection"]
        prior_trace_path = prior_stage / "critical_cycle_results.parquet"
        prior_performance_path = prior_stage / "kernel_performance.parquet"
        if not prior_trace_path.is_file() or not prior_performance_path.is_file():
            raise FileNotFoundError("sealed pre-abstention gauge traces are incomplete")
        prior_trace = pd.read_parquet(prior_trace_path)
        prior_performance = pd.read_parquet(prior_performance_path).set_index(
            "kernel_id"
        )
        for kernel_id, policy in candidates:
            frame = prior_trace[
                prior_trace["kernel_id"].astype(str).eq(kernel_id)
            ].copy()
            if frame.empty or kernel_id not in prior_performance.index:
                raise RuntimeError(
                    f"sealed pre-abstention trace missing kernel {kernel_id}"
                )
            row = _recompute_reused_kernel_metrics(
                frame, prior_performance.loc[kernel_id].to_dict() | {"kernel_id": kernel_id}
            )
            critical_frames.append(frame)
            metrics.append(row)
            if baseline_wall is None:
                baseline_wall = float(row["wall_time_s"])
        base._write_json(
            stage / "critical_cycle_trace_reference.json",
            {
                "evidence_role": "sealed_raw_trace_reanalysis_not_fresh_execution",
                "prior_trace_path": str(prior_trace_path),
                "prior_trace_sha256": base.sha256_file(prior_trace_path),
                "prior_performance_path": str(prior_performance_path),
                "prior_performance_sha256": base.sha256_file(
                    prior_performance_path
                ),
            },
        )
    else:
        for kernel_id, policy in candidates:
            if policy is None:
                continuation = _baseline_continuation(context)
            else:
                continuation = gauge_locked_predictor_corrector(
                    context["environment"],
                    policy=policy,
                    anchor_beta_rad=(
                        anchor_beta
                        if policy.mode in ("anchor_potential", "proximal_slsqp")
                        else None
                    ),
                )
            frame, row = _evaluate_kernel(config, context, cycle, kernel_id, continuation)
            critical_frames.append(frame)
            row.update(
                {
                    "mode": None if policy is None else policy.mode,
                    "gauge_gain": None if policy is None else policy.gauge_gain,
                    "cartesian_step_mm": 5.0 if policy is None else policy.cartesian_step_mm,
                }
            )
            metrics.append(row)
            if baseline_wall is None:
                baseline_wall = float(row["wall_time_s"])
            if policy is not None and row["critical_gate"]:
                selected = (kernel_id, policy)
                selection_mode = "critical_pass"
                break
    if selected is None:
        fallback = _select_gauge_candidate(
            candidates,
            metrics,
            maximum_localized_excess_edges=int(
                config["gauge"]["maximum_localized_excess_edges"]
            ),
            maximum_localized_geometry_deg=float(
                config["gauge"]["maximum_localized_geometry_deg"]
            ),
        )
        if fallback is not None:
            selected = (str(fallback[0]), fallback[1])
            selection_mode = str(fallback[2])
    critical = pd.concat(critical_frames, ignore_index=True)
    if not reuse_traces:
        base._write_parquet(critical, stage / "critical_cycle_results.parquet")
    performance = pd.DataFrame.from_records(metrics)
    base._write_parquet(performance, stage / "kernel_performance.parquet")
    # A strict repair and a registered local-abstention candidate both need an
    # independent guard set.  The latter still does not pass the critical cycle.
    guard_frames = []
    if selected is not None:
        kernel_id, policy = selected
        for patch_id, count in config["holonomy"]["guard_cycle_counts"].items():
            patch_context = _patch_context(config, project_root, str(patch_id))
            schedules = patch_context["schedules"]
            executions = patch_context["executions"]
            failed = set(
                executions.loc[
                    ~executions["classification"].eq("verified"), "schedule_id"
                ].astype(str)
            )
            cycles = schedules[
                schedules["audit_kind"].eq("fundamental_cycle")
                & ~schedules["schedule_id"].astype(str).isin(failed)
            ].copy()
            cycles["path_length"] = cycles["path_node_ids"].map(len)
            cycles = cycles.sort_values(
                ["path_length", "schedule_id"], ascending=[False, True], kind="stable"
            ).head(int(count))
            patch_anchor = next(iter(patch_context["canonical"].values())).beta_rad
            continuation = gauge_locked_predictor_corrector(
                patch_context["environment"],
                policy=policy,
                anchor_beta_rad=(
                    patch_anchor
                    if policy.mode in ("anchor_potential", "proximal_slsqp")
                    else None
                ),
            )
            for schedule in cycles.itertuples(index=False):
                path = tuple(map(int, schedule.path_node_ids))
                for direction in ("forward", "reverse"):
                    for repeat_id, magnitude in enumerate((0.0, 1.0e-8, -1.0e-8)):
                        frame = replay_cycle_with_prefix_trace(
                            path,
                            patch_id=str(patch_id),
                            canonical_by_node=patch_context["canonical"],
                            task_nodes=patch_context["nodes"],
                            continuation=continuation,
                            direction=direction,
                            repeat_id=repeat_id,
                            perturbation_magnitude_rad=magnitude,
                            kernel_id=kernel_id,
                            environment=patch_context["environment"],
                        )
                        frame["guard_patch_id"] = str(patch_id)
                        frame["guard_schedule_id"] = str(schedule.schedule_id)
                        guard_frames.append(frame)
    guard = pd.concat(guard_frames, ignore_index=True) if guard_frames else pd.DataFrame()
    base._write_parquet(guard, stage / "guard_set_results.parquet")
    guard_repeat = _repeat_metrics(guard)
    guard_gate = bool(
        len(guard)
        and float(np.percentile(guard["geometry_gap_deg"], 95)) <= 0.5
        and float(guard["geometry_gap_deg"].max()) <= 1.0
        and guard_repeat["gate_pass"]
        and float(guard["fk_residual_mm"].max()) <= 3.0
        and guard["success"].all()
    )
    selection_payload: dict[str, Any]
    if selected is None:
        selection_payload = {
            "gate_pass": False,
            "proceed_to_patch07_repair": False,
            "critical_gate": False,
            "abstention_required": False,
            "kernel_id": None,
            "reason": "no_registered_gauge_kernel_repaired_critical_cycle",
        }
    else:
        kernel_id, policy = selected
        selected_row = next(row for row in metrics if row["kernel_id"] == kernel_id)
        performance_gate = float(selected_row["wall_time_s"]) <= 2.0 * max(
            1.0e-9, float(baseline_wall)
        )
        proceed = bool(guard_gate and performance_gate)
        critical_gate = bool(selected_row["critical_gate"])
        abstention_required = selection_mode == "localized_abstention"
        selection_payload = {
            "gate_pass": proceed,
            "proceed_to_patch07_repair": proceed,
            "critical_gate": critical_gate,
            "abstention_required": abstention_required,
            "kernel_id": kernel_id,
            **asdict(policy),
            "guard_gate": guard_gate,
            "guard_repeat_metrics": guard_repeat,
            "performance_gate": performance_gate,
            "selection_reason": (
                "critical_cycle_repaired"
                if critical_gate
                else "localized_geometric_excess_with_repeat_solver_guard_pass"
            ),
            "baseline_wall_time_s": baseline_wall,
            "selected_wall_time_s": selected_row["wall_time_s"],
        }
    base._write_json(stage / "selected_kernel.json", selection_payload)
    base._write_parquet(performance, stage / "kernel_ablation.parquet")
    return _complete(
        stage,
        config,
        "gauge_kernel_selection",
        {
            "gate_pass": bool(selection_payload["gate_pass"]),
            "proceed_to_patch07_repair": bool(
                selection_payload.get("proceed_to_patch07_repair", False)
            ),
            "critical_gate": bool(selection_payload.get("critical_gate", False)),
            "abstention_required": bool(
                selection_payload.get("abstention_required", False)
            ),
            "selected_kernel": selection_payload,
        },
    )


def _stability(
    config: Mapping[str, Any], stage: Path, left: str, right: str, patch: str = "patch_07"
) -> dict[str, Any]:
    left_dir, right_dir = stage / patch / left, stage / patch / right
    return base._frame_stability(
        pd.read_parquet(left_dir / "primary_atlas.parquet"),
        pd.read_parquet(right_dir / "primary_atlas.parquet"),
        config,
        left_directory=left_dir,
        right_directory=right_dir,
    )


def _patch07_repair_checks(
    gauge: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    ordinary_refined: Mapping[str, Any],
    root_budget: Mapping[str, Any],
) -> dict[str, bool]:
    full_fresh_repair = all(
        bool(row.get("gate_pass", False))
        and bool(row.get("certificate_gate", False))
        and bool(row.get("geometry_gate", False))
        and bool(row.get("solver_gate", False))
        and bool(row.get("repeat_gate", False))
        for row in reports.values()
    )
    explicit_abstention = any(
        float(row.get("abstention_ratio", 0.0)) > 0.0
        for row in reports.values()
    )
    return {
        "all_patch_certificates": all(
            bool(row.get("gate_pass", False)) for row in reports.values()
        ),
        "ordinary_refined_stability": bool(ordinary_refined["gate_pass"]),
        "R5_R8_same_anchor_stability": bool(root_budget["gate_pass"]),
        "anchor_selected": all(
            bool(row.get("canonical_anchor_component_selected", False))
            for row in reports.values()
        ),
        "registered_local_failure_resolved": (
            not bool(gauge.get("abstention_required", False))
            or explicit_abstention
            or full_fresh_repair
        ),
    }


def stage_patch07_repair(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    gauge = _require(config, output_root, "gauge_kernel_selection")
    stage = output_root / STAGE_DIRS["patch07_repair"]
    if config.get("_patch_id"):
        report = base._execute_patch(
            config, project_root, output_root, str(config["_patch_id"]),
            str(config["_variant"]), stage / str(config["_patch_id"]) / str(config["_variant"])
        )
        return {"gate_pass": True, "worker_report": report}
    reused = _reuse_registered_stage(
        config, project_root, output_root, "patch07_repair"
    )
    if reused is not None:
        return reused
    if not gauge.get("proceed_to_patch07_repair", False):
        return _complete(stage, config, "patch07_repair", base.write_scientific_skip(stage, "gauge_kernel_selection_failed"))
    variants = ("gauge_main_K1_R5", "gauge_task_graph_refined", "gauge_K1_R8")
    prior_root = _prior_retry7_root(config, project_root)
    reuse_numerical = bool(
        config.get("reuse_patch07_numerical_artifacts", False)
    )
    numerical_stage = stage
    if reuse_numerical:
        if prior_root is None:
            raise RuntimeError(
                "patch07 numerical reuse requires retry7_reference_root"
            )
        numerical_stage = prior_root / STAGE_DIRS["patch07_repair"]
        for name in variants:
            report_path = numerical_stage / "patch_07" / name / "report.json"
            completion_path = (
                numerical_stage / "patch_07" / name / "completion_manifest.json"
            )
            if not report_path.is_file() or not completion_path.is_file():
                raise FileNotFoundError(
                    f"sealed patch07 numerical artifact missing: {report_path}"
                )
        base._write_json(
            stage / "numerical_artifact_reference.json",
            {
                "evidence_role": "sealed_numerical_artifacts_reaggregated_under_corrected_gate",
                "prior_retry7_root": str(prior_root),
                "prior_patch07_stage": str(numerical_stage),
                "prior_stage_manifest_sha256": _verify_prior_stage(
                    prior_root, "patch07_repair"
                )["manifest_sha256"],
            },
        )
    else:
        _run_patch_jobs(config, output_root, "patch07_repair", [("patch_07", name) for name in variants])
    reports = {
        name: base._read_json(
            numerical_stage / "patch_07" / name / "report.json"
        )
        for name in variants
    }
    ordinary_refined = _stability(
        config, numerical_stage, variants[0], variants[1]
    )
    root_budget = _stability(
        config, numerical_stage, variants[0], variants[2]
    )
    checks = _patch07_repair_checks(
        gauge,
        reports,
        ordinary_refined=ordinary_refined,
        root_budget=root_budget,
    )
    base._write_json(stage / "stability.json", {"ordinary_refined": ordinary_refined, "root_budget": root_budget})
    return _complete(stage, config, "patch07_repair", {"gate_pass": bool(all(checks.values())), "checks": checks, "patch_reports": reports})


def stage_four_patch_gate(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    patch07 = _require(config, output_root, "patch07_repair")
    stage = output_root / STAGE_DIRS["four_patch_gate"]
    if config.get("_patch_id"):
        report = base._execute_patch(
            config, project_root, output_root, str(config["_patch_id"]),
            str(config["_variant"]), stage / str(config["_patch_id"]) / str(config["_variant"])
        )
        return {"gate_pass": True, "worker_report": report}
    reused = _reuse_registered_stage(
        config, project_root, output_root, "four_patch_gate"
    )
    if reused is not None:
        return reused
    if not patch07.get("gate_pass", False):
        return _complete(stage, config, "four_patch_gate", base.write_scientific_skip(stage, "patch07_repair_failed"))
    patches = tuple(map(str, config["diagnostic_patch_ids"]))
    variants = (
        "gauge_main_K1_R5", "gauge_K1_R8", "gauge_root_order2",
        "gauge_root_dropout", "gauge_task_graph_refined",
    )
    _run_patch_jobs(config, output_root, "four_patch_gate", [(patch, variant) for patch in patches for variant in variants])
    patch_rows = []
    comparison_rows = []
    for patch in patches:
        reports = {
            variant: base._read_json(stage / patch / variant / "report.json")
            for variant in variants
        }
        comparisons = {
            "root_budget": _stability(config, stage, variants[0], variants[1], patch),
            "root_order": _stability(config, stage, variants[0], variants[2], patch),
            "root_dropout": _stability(config, stage, variants[0], variants[3], patch),
            "graph_refinement": _stability(config, stage, variants[0], variants[4], patch),
        }
        patch_pass = bool(
            reports[variants[0]]["gate_pass"]
            and all(item["gate_pass"] for item in comparisons.values())
            and all(
                reports[variant]["canonical_anchor_component_selected"]
                for variant in variants
            )
        )
        patch_rows.append({"patch_id": patch, "gate_pass": patch_pass, "main_report": reports[variants[0]]})
        comparison_rows.extend(
            {"patch_id": patch, "contrast": name, **value}
            for name, value in comparisons.items()
        )
    base._write_parquet(pd.DataFrame.from_records(comparison_rows), stage / "corrected_search_stability.parquet")
    base._write_parquet(base._report_records_frame(patch_rows), stage / "four_patch_reports.parquet")
    passing = sum(bool(row["gate_pass"]) for row in patch_rows)
    return _complete(
        stage, config, "four_patch_gate",
        {"gate_pass": passing == 4, "passing_patch_count": passing, "required_patch_count": 4, "patch_reports": patch_rows},
    )


def _verify_reach_round_prefix(
    round6: pd.DataFrame,
    round7: pd.DataFrame,
    *,
    physical_columns: Sequence[str],
) -> bool:
    return bool(
        len(round7) > len(round6)
        and np.array_equal(
            round6.loc[:, list(physical_columns)].to_numpy(),
            round7.iloc[: len(round6)]
            .loc[:, list(physical_columns)]
            .to_numpy(),
        )
    )


def stage_reach_round8(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    four = _require(config, output_root, "four_patch_gate")
    stage = output_root / STAGE_DIRS["reach_round8"]
    reused = _reuse_registered_stage(
        config, project_root, output_root, "reach_round8"
    )
    if reused is not None:
        return reused
    if not four.get("gate_pass", False):
        return _complete(stage, config, "reach_round8", base.write_scientific_skip(stage, "four_patch_gate_failed_before_registered_round8_compute"))
    stage.mkdir(parents=True, exist_ok=True)
    retry6 = _retry6_root(config, project_root)
    round7_root = retry6 / "08_reach_round7"
    round6_root = (
        project_root
        / str(config["sources"]["v14_2_root"])
        / "04_reach_round6"
    )
    round6_a = pd.read_parquet(round6_root / "replica_a_slab_round6.parquet")
    round6_b = pd.read_parquet(round6_root / "replica_b_slab_round6.parquet")
    round7_a = pd.read_parquet(round7_root / "replica_a_slab_round7.parquet")
    round7_b = pd.read_parquet(round7_root / "replica_b_slab_round7.parquet")
    round6_report = base._read_json(round6_root / "reach_round6_report.json")
    previous_report = base._read_json(round7_root / "reach_round7_report.json")
    reach = config["reach_round8"]
    seeds6_a = tuple(map(int, round6_report["replica_a_seed_lineage"]))
    seeds6_b = tuple(map(int, round6_report["replica_b_seed_lineage"]))
    seeds_a = tuple(map(int, previous_report["replica_a_seed_lineage"]))
    seeds_b = tuple(map(int, previous_report["replica_b_seed_lineage"]))
    if seeds_a[:-1] != seeds6_a or seeds_b[:-1] != seeds6_b:
        raise RuntimeError("Round 7 seed lineage is not an extension of sealed Round 6")
    physical_columns = ["x_m", "y_m", "z_m", *base.BETA_COLUMNS]
    prefix_checks = {
        "replica_a": _verify_reach_round_prefix(
            round6_a, round7_a, physical_columns=physical_columns
        ),
        "replica_b": _verify_reach_round_prefix(
            round6_b, round7_b, physical_columns=physical_columns
        ),
    }
    if not all(prefix_checks.values()):
        raise RuntimeError(
            f"Round 7 slab payload is not an exact extension of Round 6: {prefix_checks}"
        )
    legacy_config = base.legacy.load_config(
        SOURCE_ROOT / str(config["sources"]["legacy_config"])
    )
    legacy_paths = base.legacy._paths(legacy_config, project_root)
    grid = base.legacy._workspace_grid(legacy_paths)
    environment = optimized_forward(
        load_environment(project_root, SOURCE_ROOT / str(config["robot_config"]))
    )
    spec = base.ReachSamplingRoundSpec(
        int(reach["sobol_power"]), int(reach["seed_a"]), int(reach["seed_b"])
    )
    generated = base.generate_independent_reach_samples(
        environment,
        np.asarray(environment.bounds, dtype=float),
        grid=grid,
        rounds=(spec,),
        chunk_rows=65536,
    )

    def new_frame(xyz: np.ndarray, beta: np.ndarray, source: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x_m": xyz[:, 0],
                "y_m": xyz[:, 1],
                "z_m": xyz[:, 2],
                **{
                    name: beta[:, index]
                    for index, name in enumerate(base.BETA_COLUMNS)
                },
                "source": source,
            }
        )

    final_a = pd.concat(
        [round7_a, new_frame(generated.xyz_a, generated.beta_a, "replica_a_round_8")],
        ignore_index=True,
    )
    final_b = pd.concat(
        [round7_b, new_frame(generated.xyz_b, generated.beta_b, "replica_b_round_8")],
        ignore_index=True,
    )
    rounds = (
        base.ReachReplicaRound(
            6,
            base.ReachReplica("A", seeds6_a, round6_a[["x_m", "y_m", "z_m"]].to_numpy(float)),
            base.ReachReplica("B", seeds6_b, round6_b[["x_m", "y_m", "z_m"]].to_numpy(float)),
        ),
        base.ReachReplicaRound(
            7,
            base.ReachReplica("A", seeds_a, round7_a[["x_m", "y_m", "z_m"]].to_numpy(float)),
            base.ReachReplica("B", seeds_b, round7_b[["x_m", "y_m", "z_m"]].to_numpy(float)),
        ),
        base.ReachReplicaRound(
            8,
            base.ReachReplica("A", (*seeds_a, spec.seed_a), final_a[["x_m", "y_m", "z_m"]].to_numpy(float)),
            base.ReachReplica("B", (*seeds_b, spec.seed_b), final_b[["x_m", "y_m", "z_m"]].to_numpy(float)),
        ),
    )
    builder = base.ReachProxyBuilder(
        base.ReachProxyConfig(
            grid=grid,
            minimum_weighted_jaccard=float(reach["weighted_jaccard_min"]),
            maximum_new_volume_ratio=float(reach["new_volume_ratio_max"]),
            maximum_boundary_change_ratio=1.0,
            maximum_frontier_new_volume_ratio=float(reach["frontier_new_volume_ratio_max"]),
            required_consecutive_rounds=int(reach["required_consecutive_rounds"]),
        )
    )
    frontier_evidence: list[Any] = []
    for frontier_path in (
        legacy_paths["original_frontier"],
        legacy_paths["retry4_frontier"],
        project_root / str(config["sources"]["v14_2_root"]) / "04_reach_round6/frontier_inverse_probes_round6.parquet",
        round7_root / "frontier_inverse_probes_round7.parquet",
    ):
        if not frontier_path.is_file():
            continue
        frame = pd.read_parquet(frontier_path)
        frontier_evidence.extend(
            base.FrontierProbeEvidence(
                base.CellKey(
                    int(row.cell_level_mm), int(row.cell_ix),
                    int(row.cell_iy), int(row.cell_iz),
                ),
                bool(row.found_valid_inverse),
                round_id=int(row.round_id),
            )
            for row in frame.itertuples(index=False)
        )
    domain = pd.read_parquet(legacy_paths["original_domain_cells"])
    supplemental = tuple(
        base.CellKey(
            int(row.cell_level_mm), int(row.cell_ix),
            int(row.cell_iy), int(row.cell_iz),
        )
        for row in domain[
            domain["cell_level_mm"].eq(grid.convergence_level_mm)
            & (
                domain["registered_pool_support"].astype(bool)
                | domain["tip_pool_support"].astype(bool)
            )
        ].itertuples(index=False)
    )
    preliminary = builder.build(
        rounds,
        frontier_evidence=frontier_evidence,
        supplemental_supported_cells=supplemental,
    )
    frontier_cells = base.legacy.frontier_candidates(
        preliminary.proxy_upper_cells,
        grid=grid,
        maximum_count=int(reach["frontier_cell_budget"]),
    )
    seed_xyz = np.vstack(
        [
            final_a[["x_m", "y_m", "z_m"]].to_numpy(float),
            final_b[["x_m", "y_m", "z_m"]].to_numpy(float),
        ]
    )
    seed_beta = np.vstack(
        [
            final_a.loc[:, base.BETA_COLUMNS].to_numpy(float),
            final_b.loc[:, base.BETA_COLUMNS].to_numpy(float),
        ]
    )
    new_frontier, frontier_frame = base.legacy._frontier_probe_round(
        environment,
        frontier_cells,
        grid=grid,
        seed_xyz=seed_xyz,
        seed_beta=seed_beta,
        starts=int(reach["frontier_starts_per_cell"]),
        round_id=8,
    )
    frontier_evidence.extend(new_frontier)
    result = builder.build(
        rounds,
        frontier_evidence=frontier_evidence,
        supplemental_supported_cells=supplemental,
    )
    level = grid.convergence_level_mm
    unions = [
        grid.cells_for_points(pair.replica_a.xyz_m, level_mm=level)
        | grid.cells_for_points(pair.replica_b.xyz_m, level_mm=level)
        for pair in rounds
    ]
    boundary = {
        7: base.measure_weighted_boundary_change_ratio(unions[1], unions[0]),
        8: base.measure_weighted_boundary_change_ratio(unions[2], unions[1]),
    }
    metrics = [asdict(item) for item in result.replica_metrics]
    recent = metrics[-2:]
    checks = [
        bool(
            row["volume_weighted_jaccard"] >= float(reach["weighted_jaccard_min"])
            and row["new_volume_ratio"] <= float(reach["new_volume_ratio_max"])
            and row["frontier_new_volume_ratio"]
            <= float(reach["frontier_new_volume_ratio_max"])
            and boundary[int(row["round_id"])]
            <= float(reach["boundary_change_ratio_max"])
        )
        for row in recent
    ]
    gap = (len(result.proxy_upper_cells) - len(result.proxy_lower_cells)) / max(
        1, len(result.proxy_upper_cells)
    )
    convergence = bool(
        all(checks) and gap <= float(reach["lower_upper_measure_gap_max"])
    )
    base._write_parquet(final_a, stage / "replica_a_slab_round8.parquet")
    base._write_parquet(final_b, stage / "replica_b_slab_round8.parquet")
    base._write_parquet(frontier_frame, stage / "frontier_inverse_probes_round8.parquet")
    report = {
        "reach_convergence_gate": convergence,
        "metrics_round6_8": metrics,
        "measure_weighted_boundary_change_ratio": boundary,
        "recent_round_checks": checks,
        "proxy_lower_cell_count": len(result.proxy_lower_cells),
        "proxy_upper_cell_count": len(result.proxy_upper_cells),
        "lower_upper_measure_gap_ratio": gap,
        "replica_a_seed_lineage": [*seeds_a, spec.seed_a],
        "replica_b_seed_lineage": [*seeds_b, spec.seed_b],
        "sealed_round6_round7_prefix_checks": prefix_checks,
        "frontier_round8_probe_count": len(frontier_frame),
        "frontier_round8_found_count": int(
            frontier_frame.get("found_valid_inverse", pd.Series(dtype=bool)).sum()
        ),
        "formal_reach_authorized": convergence,
    }
    base._write_json(stage / "reach_round8_report.json", report)
    return _complete(stage, config, "reach_round8", {"gate_pass": convergence, **report})


def stage_confirmation(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    four = _require(config, output_root, "four_patch_gate")
    stage = output_root / STAGE_DIRS["confirmation"]
    if config.get("_patch_id"):
        report = base._execute_patch(
            config,
            project_root,
            output_root,
            str(config["_patch_id"]),
            str(config["_variant"]),
            stage / str(config["_patch_id"]) / str(config["_variant"]),
        )
        return {"gate_pass": True, "worker_report": report}
    reused = _reuse_registered_stage(
        config, project_root, output_root, "confirmation"
    )
    if reused is not None:
        return reused
    if not four.get("gate_pass", False):
        return _complete(stage, config, "confirmation", base.write_scientific_skip(stage, "four_patch_gate_failed"))
    development = tuple(map(str, config["development_patch_ids"]))
    confirmation = tuple(map(str, config["confirmation_patch_ids"]))
    already_computed = {"patch_00", "patch_03", "patch_07"}
    fresh = tuple(
        patch for patch in (*development, *confirmation) if patch not in already_computed
    )
    variants = ("gauge_main_K1_R5", "gauge_K1_R8")
    _run_patch_jobs(
        config,
        output_root,
        "confirmation",
        [(patch, variant) for patch in fresh for variant in variants],
    )
    four_stage = output_root / STAGE_DIRS["four_patch_gate"]
    reports = []
    comparisons = []
    for patch in (*development, *confirmation):
        source = four_stage if patch in already_computed else stage
        main_dir = source / patch / variants[0]
        high_dir = source / patch / variants[1]
        main_report = base._read_json(main_dir / "report.json")
        high_report = base._read_json(high_dir / "report.json")
        stress = base._frame_stability(
            pd.read_parquet(main_dir / "primary_atlas.parquet"),
            pd.read_parquet(high_dir / "primary_atlas.parquet"),
            config,
            left_directory=main_dir,
            right_directory=high_dir,
        )
        patch_pass = bool(
            main_report["gate_pass"]
            and high_report["gate_pass"]
            and stress["gate_pass"]
            and main_report["canonical_anchor_component_selected"]
            and high_report["canonical_anchor_component_selected"]
        )
        split = "development" if patch in development else "confirmation"
        reports.append(
            {
                "patch_id": patch,
                "patch_split": split,
                "gate_pass": patch_pass,
                "main_report": main_report,
                "high_root_report": high_report,
            }
        )
        comparisons.append({"patch_id": patch, "patch_split": split, **stress})
    base._write_parquet(
        base._report_records_frame(reports),
        stage / "confirmation_patch_reports.parquet",
    )
    base._write_parquet(
        pd.DataFrame.from_records(comparisons),
        stage / "high_root_budget_stress.parquet",
    )
    development_pass = sum(
        bool(row["gate_pass"]) for row in reports if row["patch_split"] == "development"
    )
    confirmation_pass = sum(
        bool(row["gate_pass"]) for row in reports if row["patch_split"] == "confirmation"
    )
    checks = {
        "development_pass": development_pass
        >= int(config["confirmation_gate"]["development_pass_min"]),
        "sealed_confirmation_pass": confirmation_pass
        >= int(config["confirmation_gate"]["confirmation_pass_min"]),
        "all_root_budget_stress_registered": len(comparisons)
        == len(development) + len(confirmation),
        "confirmation_not_used_for_tuning": True,
    }
    return _complete(
        stage,
        config,
        "confirmation",
        {
            "gate_pass": bool(all(checks.values())),
            "checks": checks,
            "development_pass_count": development_pass,
            "confirmation_pass_count": confirmation_pass,
            "patch09_role": "diagnostic_only_excluded_from_confirmation_counts",
            "patch12_role": "sealed_replacement_confirmation",
            "threshold_changes_after_confirmation": False,
            "patch_reports": reports,
        },
    )


def stage_meso_bridge(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    confirmation = _require(config, output_root, "confirmation")
    stage = output_root / STAGE_DIRS["meso_bridge"]
    if config.get("_patch_id"):
        report = base._execute_patch(
            config,
            project_root,
            output_root,
            str(config["_patch_id"]),
            str(config["_variant"]),
            stage / str(config["_patch_id"]) / str(config["_variant"]),
        )
        return {"gate_pass": True, "worker_report": report}
    if not confirmation.get("gate_pass", False):
        return _complete(stage, config, "meso_bridge", base.write_scientific_skip(stage, "fresh_confirmation_gate_failed"))
    report = base.stage_meso_bridge(config, project_root, output_root)
    return _complete(stage, config, "meso_bridge", report)


def stage_summary(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    gates = {
        name: base._load_validated_stage_result(
            output_root / STAGE_DIRS[name], config=config, stage_name=name
        )
        for name in STAGE_ORDER[:-1]
    }
    operational = all(value is not None for value in gates.values())
    authorization = bool(
        gates.get("four_patch_gate", {}).get("gate_pass", False)
        and gates.get("confirmation", {}).get("gate_pass", False)
        and gates.get("meso_bridge", {}).get("gate_pass", False)
    )
    report = {
        "observed_facts": gates,
        "mechanism_inference": (
            gates.get("holonomy_diagnostics", {}) or {}
        ).get("failure_classification"),
        "unresolved_unknowns": [
            name for name, value in gates.items() if not bool((value or {}).get("gate_pass", False))
        ],
        "operational_completion": operational,
        "scientific_gate": bool((gates.get("four_patch_gate") or {}).get("gate_pass", False)),
        "reach_formal_status": bool((gates.get("reach_round8") or {}).get("gate_pass", False)),
        "confirmation_authorized": bool((gates.get("four_patch_gate") or {}).get("gate_pass", False)),
        "meso_authorized": bool((gates.get("confirmation") or {}).get("gate_pass", False)),
        "repaired_5k_authorized": authorization,
    }
    base._write_json(stage / "retry7_scientific_report.json", report)
    stage_performance: dict[str, Any] = {}
    patch_performance: list[dict[str, Any]] = []
    for name in STAGE_ORDER[:-1]:
        directory = output_root / STAGE_DIRS[name]
        files = [path for path in directory.rglob("*") if path.is_file()]
        shard_rows = []
        for shard_report in directory.rglob("_audit_checkpoints/*/shard_*/report.json"):
            try:
                shard_rows.append(base._read_json(shard_report))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        starts = [float(row["started_at_unix_s"]) for row in shard_rows if row.get("started_at_unix_s") is not None]
        finishes = [float(row["finished_at_unix_s"]) for row in shard_rows if row.get("finished_at_unix_s") is not None]
        audit_window = max(finishes) - min(starts) if starts and finishes else 0.0
        audit_cpu = sum(float(row.get("cpu_time_s", 0.0)) for row in shard_rows)
        stage_performance[name] = {
            "artifact_count": len(files),
            "artifact_total_bytes": int(sum(path.stat().st_size for path in files)),
            "audit_shard_count": len(shard_rows),
            "audit_shard_cpu_time_s": audit_cpu,
            "audit_shard_wall_window_s": audit_window,
            "audit_worker_utilization": (
                audit_cpu / (12.0 * audit_window) if audit_window > 0.0 else None
            ),
        }
        for patch_report in directory.glob("**/report.json"):
            try:
                payload = base._read_json(patch_report)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if "patch_id" not in payload or "variant" not in payload:
                continue
            patch_performance.append(
                {
                    "stage": name,
                    "patch_id": payload["patch_id"],
                    "variant": payload["variant"],
                    "wall_time_s": payload.get("runtime_s"),
                    "cpu_time_s": payload.get("cpu_time_s"),
                    "peak_rss_mb": payload.get("peak_rss_mb"),
                    "continuation_execution_count": payload.get(
                        "continuation_execution_count"
                    ),
                    "bounded_ls_execution_count": payload.get(
                        "bounded_ls_execution_count"
                    ),
                    "slsqp_execution_count": payload.get(
                        "slsqp_execution_count"
                    ),
                    "retry_tier_counts": payload.get("retry_tier_counts"),
                    "kinematics_counters": payload.get("kinematics_counters"),
                    "artifact_file_count": payload.get(
                        "artifact_file_count_before_completion"
                    ),
                    "artifact_total_bytes": payload.get(
                        "artifact_total_bytes_before_completion"
                    ),
                }
            )
    kernel_path = (
        output_root
        / STAGE_DIRS["gauge_kernel_selection"]
        / "kernel_performance.parquet"
    )
    performance = {
        "stage_count": len(gates),
        "worker_limit": 12,
        "threads_per_worker": 1,
        "stage_performance": stage_performance,
        "patch_performance": patch_performance,
        "gauge_kernel_performance": (
            pd.read_parquet(kernel_path).to_dict(orient="records")
            if kernel_path.is_file()
            else []
        ),
    }
    base._write_json(stage / "retry7_performance_report.json", performance)
    next_stage = {"repaired_5k_authorized": authorization, "reason": "all_required_gates" if authorization else "one_or_more_retry7_gates_failed"}
    base._write_json(stage / "retry7_next_stage_authorization.json", next_stage)
    result = {
        "gate_pass": authorization,
        "operational_completion": operational,
        "scientific_gate_pass": authorization,
        "deployment_authorized": authorization,
        **next_stage,
    }
    base._write_json(stage / "retry7_summary_gate.json", result)
    base._write_json(stage / "gate.json", result)
    # Manifest is deliberately produced after all other summary files.
    manifest_started = time.perf_counter()
    pre_manifest_artifacts = base._stage_completion_artifacts(output_root)
    performance["manifest_hashing_wall_time_s"] = time.perf_counter() - manifest_started
    performance["final_artifact_count_before_completion"] = len(
        pre_manifest_artifacts
    )
    performance["final_artifact_total_bytes_before_completion"] = int(
        sum(int(record["bytes"]) for record in pre_manifest_artifacts)
    )
    base._write_json(stage / "retry7_performance_report.json", performance)
    artifacts = base._stage_completion_artifacts(output_root)
    manifest = {
        "schema_version": 1,
        "source_sha": base._git_sha(),
        "config_sha256": base.sha256_file(Path(str(config["config_path"]))),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    base._write_json(stage / "retry7_artifact_manifest.json", manifest)
    base._write_stage_completion_manifest(stage, config=config, stage_name="summary")
    return result


STAGE_RUNNERS = {
    "inventory": stage_inventory,
    "lineage_audit": stage_lineage_audit,
    "kr_ablation": stage_kr_ablation,
    "holonomy_diagnostics": stage_holonomy_diagnostics,
    "gauge_kernel_selection": stage_gauge_kernel_selection,
    "patch07_repair": stage_patch07_repair,
    "four_patch_gate": stage_four_patch_gate,
    "reach_round8": stage_reach_round8,
    "confirmation": stage_confirmation,
    "meso_bridge": stage_meso_bridge,
    "summary": stage_summary,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SOURCE_ROOT / "configs/bacra_v14_2r_stitched_atlas_retry7.yaml"))
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--patch-id")
    parser.add_argument("--variant")
    parser.add_argument("--audit-shards-per-patch", type=int)
    parser.add_argument("--audit-token-pool")
    parser.add_argument("--audit-token-count", type=int)
    parser.add_argument("--audit-shard-bundle")
    parser.add_argument("--audit-shard-id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = base.load_config(args.config)
    project_root = base.project_root_from(SOURCE_ROOT)
    if args.audit_shard_bundle is not None:
        if args.audit_shard_id is None:
            raise SystemExit("--audit-shard-id is required")
        report = base._run_audit_shard_worker(
            config, project_root, Path(args.audit_shard_bundle).resolve(), int(args.audit_shard_id)
        )
        print(json.dumps(base._strict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage is None:
        raise SystemExit("--stage is required outside audit shard worker mode")
    config["_patch_id"] = args.patch_id
    config["_variant"] = args.variant
    config["_runner_path"] = str(Path(__file__).resolve())
    config["_audit_shards_per_patch"] = int(
        args.audit_shards_per_patch
        if args.audit_shards_per_patch is not None
        else config["audit_execution"]["logical_shard_count"]
    )
    config["_audit_token_pool"] = args.audit_token_pool
    config["_audit_token_count"] = int(
        args.audit_token_count
        if args.audit_token_count is not None
        else config["audit_execution"]["maximum_concurrent_workers"]
    )
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else project_root / str(config["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stage != "inventory":
        base._validate_resume_fixed_point(config, output_root)
    top_level = args.patch_id is None
    directory = output_root / STAGE_DIRS[args.stage]
    if top_level:
        existing = base._load_validated_stage_result(
            directory, config=config, stage_name=args.stage
        )
        if existing is not None:
            print(json.dumps(base._strict(existing), ensure_ascii=False, sort_keys=True))
            return 0
        if (directory / "completion_manifest.json").exists() or (directory / "gate.json").exists():
            raise RuntimeError(f"retry7 stage has incomplete closure: {directory}")
    result = STAGE_RUNNERS[args.stage](config, project_root, output_root)
    print(json.dumps(base._strict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
