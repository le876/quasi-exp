#!/usr/bin/env python3
"""Run BACRA V14.2R retry8 from a frozen retry7 meso partial section."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (SOURCE_ROOT / "src", SOURCE_ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

import run_bacra_v14_2r_retry7 as retry7
import run_bacra_v14_2r_stitched_atlas as base
import run_bacra_v14_2_section_first_atlas as legacy
from quasi_exp.teacher.canonical import beta_rms_deg
from quasi_exp.teacher.canonical_atlas import AtlasCandidate, AtlasTaskNode
from quasi_exp.teacher.canonical_gauge import (
    CanonicalAnchorPolicy,
    GaugeCorrectorPolicy,
    gauge_locked_predictor_corrector,
)
from quasi_exp.teacher.optimized_continuation import (
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.partial_relay import (
    MethodSpec,
    RootKind,
    RootSpec,
    authorize_exploratory_stages,
    sample_screening_schedules,
    select_spatial_relay_roots,
)
from quasi_exp.teacher.section_atlas_repair import (
    AtlasRepairPolicy,
    AuditV2Policy,
    build_registered_audit_schedules,
    diagnose_rooted_section_artifacts,
    repair_rooted_section_atlas,
    section_growth_from_frames,
)
from quasi_exp.teacher.section_first_atlas import (
    RootedSectionPolicy,
    SectionGrowthResult,
    build_section_first_atlas,
)
from quasi_exp.teacher.workspace_atlas_repair import (
    atlas_nodes_from_frames,
    make_segmented_continuation,
)


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
STAGE_DIRS = {
    "inventory": "00_inventory",
    "frozen_partial": "01_frozen_partial",
    "relay_registry": "02_relay_registry",
    "growth_funnel": "03_growth_funnel",
    "sampled_screening": "04_sampled_screening",
    "partial_certificate": "05_partial_certificate",
    "seed_labels": "06_seed_labels",
    "authorization": "07_authorization",
    "summary": "08_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def project_root_from(source_root: Path) -> Path:
    return base.project_root_from(source_root)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported retry8 config")
    config = dict(value)
    config["config_path"] = str(config_path)
    if str(config.get("protocol_version")) != "retry8":
        raise ValueError("retry8 runner requires protocol_version=retry8")
    if config["parallel"] != {
        "numerical_workers": 12,
        "numerical_threads_per_worker": 1,
    }:
        raise ValueError("retry8 requires twelve single-threaded numerical workers")
    relay = config["relay"]
    if not (
        int(relay["minimum_count"]) == 2
        and int(relay["target_count"]) == 4
        and int(relay["maximum_count"]) == 8
    ):
        raise ValueError("retry8 relay counts must be 2/4/8")
    audit = config["audit_execution"]
    if int(audit["maximum_concurrent_workers"]) != 12:
        raise ValueError("retry8 numerical concurrency is frozen at twelve")
    return config


def _paths(config: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    source = config["sources"]
    return {
        "plan": SOURCE_ROOT / str(source["reviewed_plan"]),
        "retry7": project_root / str(source["retry7_root"]),
        "robot_config": SOURCE_ROOT / str(config["robot_config"]),
    }


def _variant_directory(config: Mapping[str, Any], project_root: Path) -> Path:
    paths = _paths(config, project_root)
    return (
        paths["retry7"]
        / retry7.STAGE_DIRS["meso_bridge"]
        / "meso_512"
        / str(config["sources"]["retry7_meso_variant"])
    )


def _meso_input(config: Mapping[str, Any], project_root: Path) -> Path:
    return _paths(config, project_root)["retry7"] / retry7.STAGE_DIRS["meso_bridge"] / "input"


def _strict(value: Any) -> Any:
    return base._strict(value)


def _read_json(path: Path) -> dict[str, Any]:
    return base._read_json(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    base._write_json(path, value)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    base._write_parquet(frame, path)


def _upstream_completion_sha(output_root: Path, stage_name: str) -> str:
    index = STAGE_ORDER.index(stage_name)
    if index == 0:
        return ""
    path = output_root / STAGE_DIRS[STAGE_ORDER[index - 1]] / "completion_manifest.json"
    return base.sha256_file(path) if path.is_file() else ""


def _write_completion(
    output_root: Path, config: Mapping[str, Any], stage_name: str
) -> None:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = [
        {
            "path": path.relative_to(stage).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    ]
    fixed = _read_json(output_root / STAGE_DIRS["inventory"] / "source_fixed_point.json")
    _write_json(
        stage / "completion_manifest.json",
        {
            "schema_version": 1,
            "stage_name": stage_name,
            "source_sha": fixed["source_sha"],
            "config_sha256": fixed["config_sha256"],
            "runtime_sha256": fixed["runtime_sha256"],
            "upstream_completion_sha256": _upstream_completion_sha(
                output_root, stage_name
            ),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )


def _load_completed(
    output_root: Path, config: Mapping[str, Any], stage_name: str
) -> dict[str, Any] | None:
    stage = output_root / STAGE_DIRS[stage_name]
    manifest_path = stage / "completion_manifest.json"
    gate_path = stage / "gate.json"
    if not manifest_path.is_file() or not gate_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    fixed = _read_json(output_root / STAGE_DIRS["inventory"] / "source_fixed_point.json")
    checks = {
        "source": manifest.get("source_sha") == fixed.get("source_sha") == base._git_sha(),
        "config": manifest.get("config_sha256")
        == fixed.get("config_sha256")
        == base.sha256_file(Path(str(config["config_path"]))),
        "runtime": manifest.get("runtime_sha256")
        == fixed.get("runtime_sha256")
        == base._runtime_sha256(),
        "upstream": manifest.get("upstream_completion_sha256", "")
        == _upstream_completion_sha(output_root, stage_name),
    }
    for record in manifest.get("artifacts", ()):
        path = stage / str(record["path"])
        checks[f"artifact:{record['path']}"] = bool(
            path.is_file()
            and path.stat().st_size == int(record["bytes"])
            and base.sha256_file(path) == str(record["sha256"])
        )
    return _read_json(gate_path) if checks and all(checks.values()) else None


def _require(
    output_root: Path, config: Mapping[str, Any], stage_name: str
) -> dict[str, Any]:
    result = _load_completed(output_root, config, stage_name)
    if result is None:
        raise RuntimeError(f"retry8 stage lacks a valid completion closure: {stage_name}")
    return result


def select_frozen_lineage_chart(qualification: pd.DataFrame) -> str:
    qualified = qualification[qualification["qualified"].astype(bool)].copy()
    if qualified.empty:
        raise RuntimeError("retry7 meso contains no qualified singleton chart")
    support = pd.to_numeric(
        qualified.get("support_fraction", qualified.get("cell_count")), errors="coerce"
    ).fillna(0.0)
    qualified["_support"] = support
    def numeric_column(name: str, default: float) -> pd.Series:
        values = (
            qualified[name]
            if name in qualified.columns
            else pd.Series(default, index=qualified.index, dtype=float)
        )
        return pd.to_numeric(values, errors="coerce").fillna(default)

    qualified["_persistent"] = numeric_column("persistent_numerical_count", 0.0)
    qualified["_geometry_p95"] = numeric_column("geometry_p95_deg", math.inf)
    qualified["_geometry_max"] = numeric_column("geometry_max_deg", math.inf)
    selected = qualified.sort_values(
        ["_support", "_persistent", "_geometry_p95", "_geometry_max", "chart_id"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).iloc[0]
    return str(selected["chart_id"])


def rank_screened_methods(
    reports: pd.DataFrame, *, maximum_candidates: int
) -> tuple[str, ...]:
    eligible = reports[
        reports["is_selectable"].astype(bool)
        & reports["screen_pass"].astype(bool)
        & ~reports["method_id"].astype(str).eq("D0_raw_retry7")
    ].copy()
    eligible = eligible.sort_values(
        ["coverage_ratio", "largest_component_ratio", "method_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    ranked = list(eligible["method_id"].astype(str).head(int(maximum_candidates)))
    fallback = "P0_frozen_partial_singleton"
    if fallback in set(eligible["method_id"].astype(str)) and fallback not in ranked:
        if len(ranked) >= int(maximum_candidates):
            ranked[-1] = fallback
        else:
            ranked.append(fallback)
    return tuple(dict.fromkeys(ranked))


def should_run_r8_growth(
    *,
    r4w16_gain_over_p0: float,
    r4w32_gain_over_origin: float,
    expandable_frontier_ratio: float,
) -> bool:
    return bool(
        float(r4w16_gain_over_p0) >= 0.01
        or float(r4w32_gain_over_origin) >= 0.01
        or float(expandable_frontier_ratio) >= 0.05
    )


def _largest_component_ratio(growth: SectionGrowthResult) -> float:
    covered = set(growth.covered_node_ids)
    if not covered:
        return 0.0
    adjacency = {
        node.node_id: set(node.neighbor_node_ids) & covered
        for node in growth.task_nodes
        if node.node_id in covered
    }
    remaining = set(covered)
    largest = 0
    while remaining:
        queue = [min(remaining)]
        component: set[int] = set()
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency.get(node, set()) - component)
        remaining -= component
        largest = max(largest, len(component))
    return largest / max(1, len(growth.task_nodes))


def _load_tasks_edges(
    config: Mapping[str, Any], project_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = _meso_input(config, project_root)
    tasks = pd.read_parquet(source / "meso_task_nodes.parquet").drop(
        columns=["patch_id"], errors="ignore"
    )
    edges = pd.read_parquet(source / "meso_task_edges.parquet").drop(
        columns=["patch_id"], errors="ignore"
    )
    return tasks, edges


def _growth_from_directory(
    directory: Path, tasks: pd.DataFrame, edges: pd.DataFrame
) -> SectionGrowthResult:
    nodes = atlas_nodes_from_frames(tasks, edges)
    return section_growth_from_frames(
        nodes,
        pd.read_parquet(directory / "section_hypotheses.parquet"),
        pd.read_parquet(directory / "selected_edges.parquet"),
    )


def _root_specs_from_frame(frame: pd.DataFrame) -> tuple[RootSpec, ...]:
    return tuple(
        RootSpec(
            root_id=str(row.root_id),
            root_kind=RootKind(str(row.root_kind)),
            task_node_id=int(row.task_node_id),
            candidate_id=str(row.candidate_id),
            canonical_lineage_id=str(row.canonical_lineage_id),
            parent_root_id=(
                None if pd.isna(row.parent_root_id) else str(row.parent_root_id)
            ),
            inherited_beta_rad=np.asarray(
                [getattr(row, column) for column in BETA_COLUMNS], dtype=float
            ),
            xyz_m=np.asarray([row.x_m, row.y_m, row.z_m], dtype=float),
            inheritance_gap_deg=float(row.inheritance_gap_deg),
        )
        for row in frame.itertuples(index=False)
    )


def _candidate_from_root(root: RootSpec) -> AtlasCandidate:
    return AtlasCandidate(
        node_id=root.task_node_id,
        candidate_id=root.candidate_id,
        beta_rad=root.inherited_beta_rad,
        residual_mm=0.0,
        min_margin_deg=1.0,
        normalized_min_margin=0.1,
        posture_cost=float(np.linalg.norm(root.inherited_beta_rad)),
        condition_number=1.0,
        quality="Gold",
        solver_success=True,
        actual_bounds=True,
        diagnostics={
            "root_kind": root.root_kind.value,
            "canonical_lineage_id": root.canonical_lineage_id,
            "inherited": True,
        },
    )


def _gauge_policy(config: Mapping[str, Any], project_root: Path) -> GaugeCorrectorPolicy:
    payload = _read_json(
        _paths(config, project_root)["retry7"]
        / retry7.STAGE_DIRS["gauge_kernel_selection"]
        / "selected_kernel.json"
    )
    return GaugeCorrectorPolicy(
        mode=str(payload["mode"]),
        gauge_gain=float(payload["gauge_gain"]),
        maximum_gauge_step_deg=float(payload["maximum_gauge_step_deg"]),
        anchor_weight=float(payload["anchor_weight"]),
        cartesian_step_mm=float(payload["cartesian_step_mm"]),
        maximum_iterations=int(payload["maximum_iterations"]),
        damping=float(payload.get("damping", 1.0e-3)),
        beta_weights=tuple(map(float, config["rooted_section"]["beta_weights"])),
        residual_tolerance_mm=float(
            config["rooted_section"]["continuation_residual_max_mm"]
        ),
    )


def _environment_and_continuation(
    config: Mapping[str, Any],
    project_root: Path,
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    anchor_beta: np.ndarray,
):
    reference = base.load_environment(project_root, _paths(config, project_root)["robot_config"])
    environment = optimized_forward(reference)
    nodes = atlas_nodes_from_frames(tasks, edges)
    node_by_id = {node.node_id: node for node in nodes}
    cell_by_node = {
        int(row.task_node_id): (
            int(row.cell_level_mm),
            int(row.cell_ix),
            int(row.cell_iy),
            int(row.cell_iz),
        )
        for row in tasks.itertuples(index=False)
    }
    policy = _gauge_policy(config, project_root)
    continuation = make_segmented_continuation(
        gauge_locked_predictor_corrector(
            environment, policy=policy, anchor_beta_rad=anchor_beta
        ),
        node_by_id,
        cell_by_node,
        step_max_mm=float(policy.cartesian_step_mm),
    )
    if isinstance(config, dict):
        config["_active_gauge_policy"] = asdict(policy)
        config["_active_anchor_beta_rad"] = anchor_beta.tolist()
    return environment, nodes, continuation, policy


def _section_policy(
    config: Mapping[str, Any], *, roots: int, waves: int, beam: int
) -> RootedSectionPolicy:
    row = config["rooted_section"]
    return RootedSectionPolicy(
        beam_width=int(beam),
        root_count=int(roots),
        maximum_growth_waves=int(waves),
        parent_consensus_gold_deg=float(row["parent_consensus_gold_deg"]),
        parent_consensus_silver_deg=float(row["parent_consensus_silver_deg"]),
        continuation_residual_max_mm=float(row["continuation_residual_max_mm"]),
        reverse_return_max_deg=float(row["reverse_return_max_deg"]),
        minimum_alternative_chart_cells=int(row["minimum_alternative_chart_cells"]),
        beta_weights=tuple(map(float, row["beta_weights"])),
    )


def _repair_policy(config: Mapping[str, Any]) -> AtlasRepairPolicy:
    active = base._repair_policy(config)
    return replace(active, audit=base._audit_policy(config))


def stage_inventory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["inventory"]
    paths = _paths(config, project_root)
    required = (
        paths["plan"],
        paths["retry7"] / retry7.STAGE_DIRS["confirmation"] / "completion_manifest.json",
        paths["retry7"] / retry7.STAGE_DIRS["meso_bridge"] / "completion_manifest.json",
        paths["retry7"] / retry7.STAGE_DIRS["summary"] / "retry7_artifact_manifest.json",
        _variant_directory(config, project_root) / "chart_qualification.parquet",
        _variant_directory(config, project_root) / "section_hypotheses.parquet",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"retry8 source inventory incomplete: {missing}")
    confirmation = retry7._verify_prior_stage(paths["retry7"], "confirmation")
    meso = retry7._verify_prior_stage(paths["retry7"], "meso_bridge")
    fixed = {
        "source_sha": base._git_sha(),
        "config_sha256": base.sha256_file(Path(str(config["config_path"]))),
        "runtime_sha256": base._runtime_sha256(),
        "working_tree_clean": base._tree_clean(),
    }
    _write_json(stage / "source_fixed_point.json", fixed)
    _write_json(
        stage / "retry7_reuse_closure.json",
        {"confirmation": confirmation, "meso_bridge": meso},
    )
    result = {
        "gate_pass": bool(not missing and fixed["working_tree_clean"]),
        "operational_completion": True,
        "retry7_confirmation_reused": True,
        "retry7_meso_reused": True,
        "retry7_raw_method_role": "diagnostic_only",
        "formal_or_deployment_authorized": False,
    }
    _write_json(stage / "gate.json", result)
    return result


def stage_frozen_partial(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require(output_root, config, "inventory")
    stage = output_root / STAGE_DIRS["frozen_partial"]
    source = _variant_directory(config, project_root)
    tasks, edges = _load_tasks_edges(config, project_root)
    qualification = pd.read_parquet(source / "chart_qualification.parquet")
    selected_fragment = select_frozen_lineage_chart(qualification)
    base_chart = selected_fragment.split("__fragment_", 1)[0]
    schedules = pd.read_parquet(source / "chart_audit_v2_schedules.parquet")
    support_rows = schedules[schedules["chart_id"].astype(str).eq(selected_fragment)]
    support_nodes = set()
    for path in support_rows.get("path_node_ids", pd.Series(dtype=object)):
        support_nodes.update(map(int, path))
    hypotheses = pd.read_parquet(source / "section_hypotheses.parquet")
    hypotheses = hypotheses[
        hypotheses["chart_id"].astype(str).eq(base_chart)
        & hypotheses["selected"].astype(bool)
    ].copy()
    if support_nodes:
        hypotheses = hypotheses[
            hypotheses["task_node_id"].astype(int).isin(support_nodes)
        ].copy()
    support_nodes = set(hypotheses["task_node_id"].astype(int))
    if not support_nodes:
        raise RuntimeError("selected retry7 singleton has no persisted support")
    edges_frame = pd.read_parquet(source / "selected_edges.parquet")
    edges_frame = edges_frame[
        edges_frame["chart_id"].astype(str).eq(base_chart)
        & edges_frame["left_node_id"].astype(int).isin(support_nodes)
        & edges_frame["right_node_id"].astype(int).isin(support_nodes)
    ].copy()
    hypotheses["chart_id"] = "p0_frozen"
    edges_frame["chart_id"] = "p0_frozen"
    labels = hypotheses.merge(
        tasks[
            [
                "task_node_id",
                "x_m",
                "y_m",
                "z_m",
                "source_parent_node_id",
                "cell_level_mm",
                "cell_ix",
                "cell_iy",
                "cell_iz",
            ]
        ],
        on="task_node_id",
        how="left",
        validate="one_to_one",
    )
    lineage_id = hashlib.sha256(
        f"{selected_fragment}:{int(labels['root_node_id'].iloc[0])}:"
        f"{labels['root_candidate_id'].iloc[0]}".encode()
    ).hexdigest()[:16]
    labels["canonical_lineage_id"] = f"lineage_{lineage_id}"
    _write_parquet(hypotheses, stage / "p0_section_hypotheses.parquet")
    _write_parquet(edges_frame, stage / "p0_selected_edges.parquet")
    _write_parquet(labels, stage / "frozen_lineage_labels.parquet")
    _write_parquet(qualification, stage / "retry7_chart_qualification.parquet")
    result = {
        "gate_pass": True,
        "operational_completion": True,
        "local_chart_qualification_pass": True,
        "selected_retry7_chart": selected_fragment,
        "selected_base_chart": base_chart,
        "canonical_lineage_id": labels["canonical_lineage_id"].iloc[0],
        "origin_node_id": int(labels["root_node_id"].iloc[0]),
        "observed_partial_coverage": len(labels) / max(1, len(tasks)),
        "observed_partial_node_count": len(labels),
        "D0_is_diagnostic_only": True,
        "D0_is_selectable": False,
        "P0_is_selectable": True,
    }
    _write_json(stage / "gate.json", result)
    return result


def stage_relay_registry(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    frozen = _require(output_root, config, "frozen_partial")
    stage = output_root / STAGE_DIRS["relay_registry"]
    labels = pd.read_parquet(
        output_root / STAGE_DIRS["frozen_partial"] / "frozen_lineage_labels.parquet"
    )
    relay = config["relay"]
    roots = select_spatial_relay_roots(
        labels,
        origin_node_id=int(frozen["origin_node_id"]),
        canonical_lineage_id=str(frozen["canonical_lineage_id"]),
        target_count=int(relay["maximum_count"]),
        preferred_separation_mm=float(relay["preferred_separation_mm"]),
        minimum_separation_mm=float(relay["minimum_separation_mm"]),
    )
    records = []
    for index, root in enumerate(roots):
        records.append(
            {
                "root_index": index,
                "root_id": root.root_id,
                "root_kind": root.root_kind.value,
                "task_node_id": root.task_node_id,
                "candidate_id": root.candidate_id,
                "canonical_lineage_id": root.canonical_lineage_id,
                "parent_root_id": root.parent_root_id,
                "x_m": root.xyz_m[0],
                "y_m": root.xyz_m[1],
                "z_m": root.xyz_m[2],
                "inheritance_gap_deg": root.inheritance_gap_deg,
                **{
                    column: root.inherited_beta_rad[position]
                    for position, column in enumerate(BETA_COLUMNS)
                },
            }
        )
    registry = pd.DataFrame.from_records(records)
    relay_count = int(registry["root_kind"].eq(RootKind.SPATIAL_RELAY.value).sum())
    if relay_count >= int(relay["target_count"]):
        availability = "target_met"
    elif relay_count >= int(relay["minimum_count"]):
        availability = "availability_limited"
    elif relay_count:
        availability = "diagnostic_only"
    else:
        availability = "unavailable"
    methods = [
        MethodSpec.raw_retry7_diagnostic(),
        MethodSpec.frozen_partial_singleton(),
    ]
    method_frame = pd.DataFrame.from_records([asdict(method) for method in methods])
    _write_parquet(registry, stage / "spatial_relay_roots.parquet")
    _write_parquet(method_frame, stage / "base_method_registry.parquet")
    result = {
        "gate_pass": True,
        "operational_completion": True,
        "unique_spatial_relay_node_count": relay_count,
        "relay_availability_status": availability,
        "relay_method_testable": relay_count >= int(relay["minimum_count"]),
        "P0_partial_student_may_proceed": True,
        "alternative_branch_count_in_registry": int(
            registry["root_kind"].eq(RootKind.ALTERNATIVE_BRANCH.value).sum()
        ),
    }
    _write_json(stage / "gate.json", result)
    return result


def _write_growth(directory: Path, growth: SectionGrowthResult) -> None:
    for name, frame in growth.frames().items():
        _write_parquet(frame, directory / f"{name}.parquet")


def _growth_report(
    growth: SectionGrowthResult,
    *,
    method: Mapping[str, Any],
    p0_coverage: float,
) -> dict[str, Any]:
    total = max(1, len(growth.task_nodes))
    coverage = len(growth.covered_node_ids) / total
    frontier = set().union(*(set(chart.frontier_node_ids) for chart in growth.charts))
    failure_counts: dict[str, int] = {}
    for event in growth.events:
        failure_counts[event.reason] = failure_counts.get(event.reason, 0) + 1
    return {
        **dict(method),
        "coverage_ratio": coverage,
        "coverage_gain_over_p0": coverage - float(p0_coverage),
        "largest_component_ratio": _largest_component_ratio(growth),
        "expandable_frontier_ratio": len(frontier) / total,
        "covered_node_count": len(growth.covered_node_ids),
        "frontier_node_count": len(frontier),
        "continuation_attempt_count": growth.continuation_attempt_count,
        "rejected_continuation_count": growth.rejected_continuation_count,
        "failure_reason_counts": failure_counts,
        "cap_hit_ratio": len(growth.cap_hit_events) / total,
        "pre_pruning_hypothesis_count": sum(
            event.pre_pruning_proposal_count for event in growth.events
        ),
        "post_pruning_hypothesis_count": sum(
            event.post_pruning_hypothesis_count for event in growth.events
        ),
    }


def stage_growth_funnel(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    registry_gate = _require(output_root, config, "relay_registry")
    frozen = _require(output_root, config, "frozen_partial")
    stage = output_root / STAGE_DIRS["growth_funnel"]
    tasks, edges = _load_tasks_edges(config, project_root)
    roots = _root_specs_from_frame(
        pd.read_parquet(
            output_root / STAGE_DIRS["relay_registry"] / "spatial_relay_roots.parquet"
        )
    )
    origin = roots[0]
    environment, nodes, continuation, _policy = _environment_and_continuation(
        config, project_root, tasks, edges, origin.inherited_beta_rad
    )
    del environment
    candidates = tuple(_candidate_from_root(root) for root in roots)
    p0_coverage = float(frozen["observed_partial_coverage"])
    methods = [dict(row) for row in config["growth_funnel"]["registered_methods"]]
    reports: list[dict[str, Any]] = [
        {
            **methods[0],
            "coverage_ratio": p0_coverage,
            "coverage_gain_over_p0": 0.0,
            "largest_component_ratio": p0_coverage,
            "expandable_frontier_ratio": 0.0,
            "executed": False,
            "evidence_role": "retry7_read_only_diagnostic",
        },
        {
            **methods[1],
            "coverage_ratio": p0_coverage,
            "coverage_gain_over_p0": 0.0,
            "largest_component_ratio": p0_coverage,
            "expandable_frontier_ratio": 0.0,
            "executed": False,
            "evidence_role": "frozen_partial_fallback",
        },
    ]
    method_by_id = {str(row["method_id"]): row for row in methods}

    def run(method_id: str) -> dict[str, Any] | None:
        method = method_by_id[method_id]
        relay_count = int(method["relay_count"])
        available = len(roots) - 1
        if relay_count > available:
            report = {
                **method,
                "executed": False,
                "availability_limited": True,
                "available_relay_count": available,
                "coverage_ratio": p0_coverage,
                "coverage_gain_over_p0": 0.0,
                "largest_component_ratio": p0_coverage,
                "expandable_frontier_ratio": 0.0,
            }
            reports.append(report)
            return None
        selected = (origin, *roots[1 : relay_count + 1])
        growth = build_section_first_atlas(
            nodes,
            tuple(_candidate_from_root(root) for root in selected),
            continuation,
            root_keys=tuple(
                (root.task_node_id, root.candidate_id) for root in selected
            ),
            policy=_section_policy(
                config,
                roots=len(selected),
                waves=int(method["maximum_growth_waves"]),
                beam=int(method["beam_width"]),
            ),
        )
        directory = stage / method_id
        _write_growth(directory, growth)
        report = {
            **_growth_report(growth, method=method, p0_coverage=p0_coverage),
            "executed": True,
            "availability_limited": False,
            "available_relay_count": available,
        }
        _write_json(directory / "report.json", report)
        reports.append(report)
        return report

    g32 = run("G32_origin")
    run("R2W16")
    run("R2W32")
    r4w16 = run("R4W16")
    r4w32 = run("R4W32")
    r8 = False
    if r4w16 is not None and r4w32 is not None and g32 is not None:
        r8 = should_run_r8_growth(
            r4w16_gain_over_p0=float(r4w16["coverage_gain_over_p0"]),
            r4w32_gain_over_origin=float(r4w32["coverage_ratio"])
            - float(g32["coverage_ratio"]),
            expandable_frontier_ratio=float(r4w32["expandable_frontier_ratio"]),
        )
    if r8:
        run("R8W32")
    best = max(
        (row for row in reports if row.get("executed")),
        key=lambda row: (float(row["coverage_ratio"]), str(row["method_id"])),
        default=None,
    )
    if best is not None and float(best.get("cap_hit_ratio", 0.0)) > float(
        config["growth_funnel"]["beam4_cap_hit_ratio_trigger"]
    ):
        method_by_id["K4_best"] = {
            **method_by_id["K4_best"],
            "relay_count": int(best["relay_count"]),
            "maximum_growth_waves": int(best["maximum_growth_waves"]),
        }
        run("K4_best")
    frame = base._report_records_frame(reports)
    _write_parquet(frame, stage / "growth_method_reports.parquet")
    result = {
        "gate_pass": True,
        "operational_completion": True,
        "method_count": len(reports),
        "executed_method_count": sum(bool(row.get("executed")) for row in reports),
        "r8_triggered": r8,
        "relay_availability_status": registry_gate["relay_availability_status"],
        "P0_always_available": True,
    }
    _write_json(stage / "gate.json", result)
    return result


def _load_method_growth(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    method_id: str,
) -> SectionGrowthResult:
    tasks, edges = _load_tasks_edges(config, project_root)
    if method_id == "P0_frozen_partial_singleton":
        directory = output_root / STAGE_DIRS["frozen_partial"]
        nodes = atlas_nodes_from_frames(tasks, edges)
        return section_growth_from_frames(
            nodes,
            pd.read_parquet(directory / "p0_section_hypotheses.parquet"),
            pd.read_parquet(directory / "p0_selected_edges.parquet"),
        )
    return _growth_from_directory(
        output_root / STAGE_DIRS["growth_funnel"] / method_id,
        tasks,
        edges,
    )


def stage_sampled_screening(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require(output_root, config, "growth_funnel")
    stage = output_root / STAGE_DIRS["sampled_screening"]
    reports = pd.read_parquet(
        output_root / STAGE_DIRS["growth_funnel"] / "growth_method_reports.parquet"
    )
    tasks, edges = _load_tasks_edges(config, project_root)
    roots = _root_specs_from_frame(
        pd.read_parquet(
            output_root / STAGE_DIRS["relay_registry"] / "spatial_relay_roots.parquet"
        )
    )
    origin = roots[0]
    environment, _nodes, continuation, gauge = _environment_and_continuation(
        config, project_root, tasks, edges, origin.inherited_beta_rad
    )
    candidates = reports[
        reports["is_selectable"].astype(bool)
        & (reports["method_id"].astype(str).eq("P0_frozen_partial_singleton")
           | pd.to_numeric(reports["coverage_gain_over_p0"], errors="coerce").fillna(0.0).gt(0.0))
    ].copy()
    screen_rows: list[dict[str, Any]] = []
    screening = config["screening"]
    for record in candidates.to_dict(orient="records"):
        method_id = str(record["method_id"])
        if method_id == "P0_frozen_partial_singleton":
            screen_rows.append({**record, "screen_pass": True, "screen_role": "P0_fallback"})
            continue
        growth = _load_method_growth(config, project_root, output_root, method_id)
        full = build_registered_audit_schedules(
            growth, patch_id="meso_512", method=f"retry8_screen_{method_id}"
        )
        relay_nodes = [root.task_node_id for root in roots[1 : int(record["relay_count"]) + 1]]
        boundary_nodes = sorted(
            set().union(*(set(chart.boundary_risk_node_ids) for chart in growth.charts))
        )
        sampled = sample_screening_schedules(
            full,
            relay_node_ids=relay_nodes,
            boundary_node_ids=boundary_nodes,
            maximum_interior_edges=int(screening["maximum_interior_edges"]),
            maximum_cycles=int(screening["maximum_cycles"]),
        )
        directory = stage / method_id
        _write_parquet(sampled, directory / "screening_schedules.parquet")
        policy = replace(
            base._audit_policy(config),
            geometry_p95_max_deg=float(screening["geometry_p95_max_deg"]),
            geometry_max_deg=float(screening["geometry_max_deg"]),
            repeat_p95_max_deg=float(screening["repeat_p95_max_deg"]),
            continuation_residual_max_mm=float(screening["residual_max_mm"]),
            repeats_per_direction=int(screening["repeats_per_direction"]),
            directions=tuple(map(str, screening["directions"])),
        )
        executor = base._SubprocessAuditExecutor(
            config=config,
            project_root=project_root,
            tasks=tasks,
            task_edges=edges,
            patch_directory=directory,
            shard_count=int(config["audit_execution"]["logical_shard_count"]),
            maximum_concurrent_workers=int(
                config["audit_execution"]["maximum_concurrent_workers"]
            ),
            assignment_strategy=str(config["audit_execution"]["assignment_strategy"]),
            worker_runner_path=Path(__file__).resolve(),
        )
        executions = executor(
            growth,
            sampled,
            continuation,
            policy,
            base._registered_retry_adapter(
                environment,
                tasks,
                edges,
                gauge_policy=gauge,
                anchor_beta_rad=origin.inherited_beta_rad,
            ),
            "sampled_screening",
        )
        diagnostic = diagnose_rooted_section_artifacts(
            growth, schedules=sampled, executions=executions, policy=policy
        )
        _write_parquet(executions, directory / "screening_executions.parquet")
        residual = pd.to_numeric(executions["residual_mm"], errors="coerce")
        residual_max = float(residual.max()) if len(residual) else math.inf
        screen_pass = bool(
            diagnostic.geometry_gate
            and diagnostic.repeat_gate
            and diagnostic.solver_gate
            and residual_max <= float(screening["residual_max_mm"])
        )
        row = {
            **record,
            "screen_pass": screen_pass,
            "screen_role": "sampled_audit",
            "screen_schedule_count": len(sampled),
            "screen_execution_count": len(executions),
            "screen_geometry_p95_deg": diagnostic.geometry_metrics["p95_deg"],
            "screen_geometry_max_deg": diagnostic.geometry_metrics["max_deg"],
            "screen_repeat_p95_deg": diagnostic.repeat_metrics["p95_deg"],
            "screen_residual_max_mm": residual_max,
        }
        screen_rows.append(row)
        _write_json(directory / "report.json", row)
    screen_frame = base._report_records_frame(screen_rows)
    _write_parquet(screen_frame, stage / "screening_method_reports.parquet")
    ranked = rank_screened_methods(
        screen_frame,
        maximum_candidates=int(screening["maximum_full_certificate_candidates"]),
    )
    _write_json(stage / "full_certificate_ranking.json", {"ranked_method_ids": ranked})
    result = {
        "gate_pass": True,
        "operational_completion": True,
        "screened_method_count": len(screen_rows),
        "ranked_method_ids": ranked,
        "P0_in_full_certificate_candidates": "P0_frozen_partial_singleton" in ranked,
        "D0_excluded": "D0_raw_retry7" not in ranked,
    }
    _write_json(stage / "gate.json", result)
    return result


def _write_repaired(directory: Path, repaired: Any) -> None:
    for name, frame in repaired.frames.items():
        _write_parquet(frame, directory / f"{name}.parquet")
    _write_json(
        directory / "report.json",
        {
            "coverage_ratio": repaired.coverage_ratio,
            "largest_component_ratio": repaired.largest_coherent_region_ratio,
            "geometry_gate": repaired.geometry_gate,
            "solver_gate": repaired.solver_gate,
            "repeat_gate": repaired.repeat_gate,
            "certificate_gate": repaired.certificate_gate,
            "selected_stitch_component": repaired.selected_stitch_component,
            "unresolved_critical_entities": repaired.diagnostic.solver_metrics[
                "persistent_numerical_count"
            ],
        },
    )


def _primary_candidates(repaired: Any) -> dict[int, AtlasCandidate]:
    """Resolve the concrete candidate chosen by the materialized primary field."""

    chart_by_id = {chart.chart_id: chart for chart in repaired.growth.charts}
    result: dict[int, AtlasCandidate] = {}
    for node_id, chart_id in repaired.primary_chart_by_node.items():
        if chart_id is None:
            continue
        chart = chart_by_id[str(chart_id)]
        result[int(node_id)] = chart.selected_by_node[int(node_id)].candidate
    return result


def stage_partial_certificate(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    screening = _require(output_root, config, "sampled_screening")
    stage = output_root / STAGE_DIRS["partial_certificate"]
    tasks, edges = _load_tasks_edges(config, project_root)
    roots = _root_specs_from_frame(
        pd.read_parquet(
            output_root / STAGE_DIRS["relay_registry"] / "spatial_relay_roots.parquet"
        )
    )
    origin = roots[0]
    environment, _nodes, continuation, gauge = _environment_and_continuation(
        config, project_root, tasks, edges, origin.inherited_beta_rad
    )
    selected_method: str | None = None
    selected_repaired = None
    attempts: list[dict[str, Any]] = []
    for rank, method_id in enumerate(screening["ranked_method_ids"], start=1):
        growth = _load_method_growth(config, project_root, output_root, method_id)
        directory = stage / f"rank_{rank:02d}_{method_id}"
        executor = base._SubprocessAuditExecutor(
            config=config,
            project_root=project_root,
            tasks=tasks,
            task_edges=edges,
            patch_directory=directory,
            shard_count=int(config["audit_execution"]["logical_shard_count"]),
            maximum_concurrent_workers=int(
                config["audit_execution"]["maximum_concurrent_workers"]
            ),
            assignment_strategy=str(config["audit_execution"]["assignment_strategy"]),
            worker_runner_path=Path(__file__).resolve(),
        )
        repaired = repair_rooted_section_atlas(
            growth,
            continuation,
            patch_id="meso_512",
            method=f"retry8_full_{method_id}",
            policy=_repair_policy(config),
            retry_continuation=base._registered_retry_adapter(
                environment,
                tasks,
                edges,
                gauge_policy=gauge,
                anchor_beta_rad=origin.inherited_beta_rad,
            ),
            schedule_executor=executor,
            screening_first=False,
            canonical_root_priority=((origin.task_node_id, origin.candidate_id),),
            canonical_anchor_policy=CanonicalAnchorPolicy(
                ordered_anchor_root_keys=((origin.task_node_id, origin.candidate_id),),
                minimum_component_coverage=0.0,
                minimum_coherent_measure=0.0,
                allow_stitchable_extensions=True,
                require_anchor_selected=True,
            ),
        )
        _write_repaired(directory, repaired)
        attempts.append(
            {
                "rank": rank,
                "method_id": method_id,
                "partial_certificate_pass": repaired.certificate_gate,
                "coverage_ratio": repaired.coverage_ratio,
                "largest_component_ratio": repaired.largest_coherent_region_ratio,
            }
        )
        if repaired.certificate_gate:
            selected_method = str(method_id)
            selected_repaired = repaired
            break
    minimal_repair_used = False
    if selected_repaired is None:
        method_id = "P0_frozen_partial_singleton"
        growth = _load_method_growth(config, project_root, output_root, method_id)
        directory = stage / "p0_minimal_abstention_repair"
        active = replace(_repair_policy(config), abstention_hops=1)
        executor = base._SubprocessAuditExecutor(
            config=config,
            project_root=project_root,
            tasks=tasks,
            task_edges=edges,
            patch_directory=directory,
            shard_count=int(config["audit_execution"]["logical_shard_count"]),
            maximum_concurrent_workers=12,
            assignment_strategy=str(config["audit_execution"]["assignment_strategy"]),
            worker_runner_path=Path(__file__).resolve(),
        )
        selected_repaired = repair_rooted_section_atlas(
            growth,
            continuation,
            patch_id="meso_512",
            method="retry8_p0_minimal_abstention",
            policy=active,
            retry_continuation=base._registered_retry_adapter(
                environment,
                tasks,
                edges,
                gauge_policy=gauge,
                anchor_beta_rad=origin.inherited_beta_rad,
            ),
            schedule_executor=executor,
            canonical_root_priority=((origin.task_node_id, origin.candidate_id),),
            canonical_anchor_policy=CanonicalAnchorPolicy(
                ordered_anchor_root_keys=((origin.task_node_id, origin.candidate_id),),
                minimum_component_coverage=0.0,
                minimum_coherent_measure=0.0,
                allow_stitchable_extensions=False,
                require_anchor_selected=True,
            ),
        )
        selected_method = method_id
        minimal_repair_used = True
        _write_repaired(directory, selected_repaired)
    repaired = selected_repaired
    assert repaired is not None and selected_method is not None
    assignment = repaired.frames["primary_atlas"]
    labels = assignment[~assignment["abstained"].astype(bool)].merge(
        tasks,
        on="task_node_id",
        how="left",
        validate="one_to_one",
    )
    labels["canonical_lineage_id"] = roots[0].canonical_lineage_id
    labels["selected_method_id"] = selected_method
    primary_candidates = _primary_candidates(repaired)
    labels["candidate_id"] = labels["task_node_id"].map(
        lambda node: primary_candidates[int(node)].candidate_id
    )
    labels["source_teacher_residual_mm"] = labels["task_node_id"].map(
        lambda node: float(primary_candidates[int(node)].residual_mm)
    )
    labels["source_teacher_actual_bounds"] = labels["task_node_id"].map(
        lambda node: bool(primary_candidates[int(node)].actual_bounds)
    )
    _write_parquet(assignment, stage / "partial_primary_atlas.parquet")
    _write_parquet(labels, stage / "primary_canonical_labels.parquet")
    _write_parquet(
        repaired.frames["audit_v2_schedules"],
        stage / "partial_primary_certificate_schedules.parquet",
    )
    _write_parquet(
        repaired.frames["audit_v2_executions"],
        stage / "partial_primary_certificate.parquet",
    )
    abstention = assignment[assignment["abstained"].astype(bool)].copy()
    _write_parquet(abstention, stage / "unlabeled_abstention_nodes.parquet")
    frozen = _read_json(output_root / STAGE_DIRS["frozen_partial"] / "gate.json")
    summary = pd.DataFrame.from_records(
        [
            {
                "method_id": selected_method,
                "observed_partial_coverage": float(frozen["observed_partial_coverage"]),
                "certified_partial_coverage": repaired.coverage_ratio,
                "largest_coherent_component": repaired.largest_coherent_region_ratio,
                "partial_certificate_pass": repaired.certificate_gate,
            }
        ]
    )
    _write_parquet(summary, stage / "component_coverage_summary.parquet")
    _write_parquet(pd.DataFrame.from_records(attempts), stage / "certificate_attempts.parquet")
    result = {
        "gate_pass": bool(repaired.certificate_gate),
        "operational_completion": True,
        "local_chart_qualification_pass": bool(repaired.qualified_chart_ids),
        "partial_certificate_pass": bool(repaired.certificate_gate),
        "relay_method_validation_pass": False,
        "global_coverage_gate_pass": bool(
            repaired.coverage_ratio >= float(config["exploratory_target"]["coverage_min"])
            and repaired.largest_coherent_region_ratio
            >= float(config["exploratory_target"]["largest_component_min"])
        ),
        "selected_method_id": selected_method,
        "selected_method_relay_count": int(
            next(
                row["relay_count"]
                for row in config["growth_funnel"]["registered_methods"]
                if row["method_id"] == selected_method
            )
        ),
        "observed_partial_coverage": float(frozen["observed_partial_coverage"]),
        "certified_partial_coverage": repaired.coverage_ratio,
        "largest_coherent_component": repaired.largest_coherent_region_ratio,
        "unresolved_critical_entities": int(
            repaired.diagnostic.solver_metrics["persistent_numerical_count"]
        ),
        "minimal_abstention_repair_used": minimal_repair_used,
        "formal_or_deployment_authorized": False,
    }
    _write_json(stage / "gate.json", result)
    return result


def _seed_target_registry(
    labels: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    target_count: int,
    fractions: Sequence[float],
    shard_count: int,
) -> pd.DataFrame:
    indexed = labels.set_index("task_node_id")
    retained = set(indexed.index.astype(int))
    records: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set(
        map(tuple, np.round(labels[["x_m", "y_m", "z_m"]].to_numpy(float), 12))
    )
    registered_edges = [
        (int(row.left_node_id), int(row.right_node_id))
        for row in edges.itertuples(index=False)
        if int(row.left_node_id) in retained and int(row.right_node_id) in retained
    ]
    for left, right in registered_edges:
        left_row = indexed.loc[left]
        right_row = indexed.loc[right]
        for fraction in fractions:
            xyz = (
                (1.0 - float(fraction))
                * left_row[["x_m", "y_m", "z_m"]].to_numpy(float)
                + float(fraction)
                * right_row[["x_m", "y_m", "z_m"]].to_numpy(float)
            )
            key = tuple(np.round(xyz, 12))
            if key in seen:
                continue
            seen.add(key)
            target_id = len(records)
            records.append(
                {
                    "target_id": target_id,
                    "x_m": xyz[0],
                    "y_m": xyz[1],
                    "z_m": xyz[2],
                    "source_node_1": left,
                    "source_node_2": right,
                    "independent_parent_count": len(
                        {
                            int(left_row["source_parent_node_id"]),
                            int(right_row["source_parent_node_id"]),
                        }
                    ),
                    "shard_id": int(
                        hashlib.sha256(str(target_id).encode()).hexdigest()[:16], 16
                    )
                    % int(shard_count),
                }
            )
            if len(records) >= int(target_count):
                return pd.DataFrame.from_records(records)
    if len(records) < int(target_count) and len(labels) >= 2:
        xyz = labels[["x_m", "y_m", "z_m"]].to_numpy(float)
        tree = cKDTree(xyz)
        _distance, neighbors = tree.query(xyz, k=min(8, len(labels)))
        node_ids = labels["task_node_id"].astype(int).to_numpy()
        for left_index, row in enumerate(neighbors):
            for right_index in np.atleast_1d(row)[1:]:
                left, right = int(node_ids[left_index]), int(node_ids[int(right_index)])
                left_row, right_row = indexed.loc[left], indexed.loc[right]
                for fraction in fractions:
                    point = (
                        (1.0 - float(fraction))
                        * left_row[["x_m", "y_m", "z_m"]].to_numpy(float)
                        + float(fraction)
                        * right_row[["x_m", "y_m", "z_m"]].to_numpy(float)
                    )
                    key = tuple(np.round(point, 12))
                    if key in seen:
                        continue
                    seen.add(key)
                    target_id = len(records)
                    records.append(
                        {
                            "target_id": target_id,
                            "x_m": point[0],
                            "y_m": point[1],
                            "z_m": point[2],
                            "source_node_1": left,
                            "source_node_2": right,
                            "independent_parent_count": len(
                                {
                                    int(left_row["source_parent_node_id"]),
                                    int(right_row["source_parent_node_id"]),
                                }
                            ),
                            "shard_id": int(
                                hashlib.sha256(str(target_id).encode()).hexdigest()[:16], 16
                            )
                            % int(shard_count),
                        }
                    )
                    if len(records) >= int(target_count):
                        return pd.DataFrame.from_records(records)
    return pd.DataFrame.from_records(records)


def _source_candidate(row: Any) -> AtlasCandidate:
    return AtlasCandidate(
        node_id=int(row.task_node_id),
        candidate_id=str(row.candidate_id),
        beta_rad=np.asarray([getattr(row, column) for column in BETA_COLUMNS]),
        residual_mm=0.0,
        min_margin_deg=1.0,
        normalized_min_margin=0.1,
        condition_number=1.0,
        quality="Gold",
    )


def _execute_seed_shard(
    config: Mapping[str, Any], project_root: Path, output_root: Path, shard_id: int
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["seed_labels"]
    directory = stage / f"shard_{int(shard_id):02d}"
    targets = pd.read_parquet(stage / "seed_target_registry.parquet")
    targets = targets[targets["shard_id"].astype(int).eq(int(shard_id))]
    labels = pd.read_parquet(
        output_root / STAGE_DIRS["partial_certificate"] / "primary_canonical_labels.parquet"
    )
    indexed = labels.set_index("task_node_id")
    reference = base.load_environment(project_root, _paths(config, project_root)["robot_config"])
    environment = optimized_forward(reference)
    continuation = make_optimized_predictor_corrector_continuation(
        environment,
        damping=2.0e-3,
        max_corrector_iterations=400,
        residual_tolerance_mm=float(config["seed_labels"]["continuation_residual_max_mm"]),
    )
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        target_node = AtlasTaskNode(
            int(target.target_id),
            np.asarray([target.x_m, target.y_m, target.z_m]),
            (),
        )
        outcomes = []
        for source_node_id in (int(target.source_node_1), int(target.source_node_2)):
            source_row = indexed.loc[source_node_id]
            source = _source_candidate(source_row)
            forward = continuation(source, target_node)
            if not (
                forward.success
                and forward.actual_bounds
                and forward.residual_mm
                <= float(config["seed_labels"]["continuation_residual_max_mm"])
            ):
                continue
            endpoint = AtlasCandidate(
                node_id=target_node.node_id,
                candidate_id=f"seed_{target.target_id}_{source_node_id}",
                beta_rad=forward.beta_rad,
                residual_mm=forward.residual_mm,
                min_margin_deg=float(forward.minimum_margin_deg or 1.0),
                normalized_min_margin=0.1,
                condition_number=1.0,
                quality="Gold",
            )
            reverse_target = AtlasTaskNode(
                source_node_id,
                np.asarray([source_row.x_m, source_row.y_m, source_row.z_m]),
                (),
            )
            reverse = continuation(endpoint, reverse_target)
            reverse_gap = (
                beta_rms_deg(reverse.beta_rad, source.beta_rad)
                if reverse.success and reverse.actual_bounds
                else math.inf
            )
            if reverse_gap <= float(config["seed_labels"]["reverse_return_max_deg"]):
                outcomes.append((source_node_id, forward, reverse_gap))
        if not outcomes:
            continue
        outcome_gap = (
            beta_rms_deg(outcomes[0][1].beta_rad, outcomes[1][1].beta_rad)
            if len(outcomes) >= 2
            else math.inf
        )
        if (
            len(outcomes) >= 2
            and int(target.independent_parent_count) >= 2
            and outcome_gap
            <= float(config["seed_labels"]["multiparent_beta_gap_max_deg"])
        ):
            quality = "Gold"
        else:
            source_gaps = [
                beta_rms_deg(
                    outcomes[0][1].beta_rad,
                    indexed.loc[int(node_id), list(BETA_COLUMNS)].to_numpy(float),
                )
                for node_id in (target.source_node_1, target.source_node_2)
            ]
            if max(source_gaps) > float(
                config["seed_labels"]["silver_neighbor_jump_max_deg"]
            ):
                continue
            quality = "Silver"
        selected = min(outcomes, key=lambda row: (row[1].residual_mm, row[0]))
        rows.append(
            {
                "physical_point_id": f"seed_{int(target.target_id)}",
                "x_m": float(target.x_m),
                "y_m": float(target.y_m),
                "z_m": float(target.z_m),
                **{
                    column: float(selected[1].beta_rad[index])
                    for index, column in enumerate(BETA_COLUMNS)
                },
                "label_quality": quality,
                "parent_node_id_1": int(target.source_node_1),
                "parent_node_id_2": int(target.source_node_2),
                "independent_parent_count": int(target.independent_parent_count),
                "multiparent_beta_gap_deg": outcome_gap,
                "reverse_return_max_deg": max(row[2] for row in outcomes),
                "teacher_fk_residual_mm": float(selected[1].residual_mm),
                "actual_bounds": bool(selected[1].actual_bounds),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    _write_parquet(frame, directory / "labels.parquet")
    report = {
        "gate_pass": True,
        "shard_id": int(shard_id),
        "target_count": len(targets),
        "accepted_count": len(frame),
    }
    _write_json(directory / "report.json", report)
    return report


def stage_seed_labels(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    certificate = _require(output_root, config, "partial_certificate")
    stage = output_root / STAGE_DIRS["seed_labels"]
    if config.get("_seed_shard") is not None:
        return _execute_seed_shard(
            config, project_root, output_root, int(config["_seed_shard"])
        )
    labels = pd.read_parquet(
        output_root / STAGE_DIRS["partial_certificate"] / "primary_canonical_labels.parquet"
    )
    edges = _load_tasks_edges(config, project_root)[1]
    seed = config["seed_labels"]
    targets = _seed_target_registry(
        labels,
        edges,
        target_count=int(seed["maximum_unique_rows"]),
        fractions=tuple(map(float, seed["interpolation_fractions"])),
        shard_count=int(seed["logical_shard_count"]),
    )
    _write_parquet(targets, stage / "seed_target_registry.parquet")
    environment = base._thread_limited_environment()
    pending = list(range(int(seed["logical_shard_count"])))
    running: list[tuple[int, subprocess.Popen[str], Any]] = []
    maximum_workers = int(config["parallel"]["numerical_workers"])
    while pending or running:
        while pending and len(running) < maximum_workers:
            shard_id = pending.pop(0)
            directory = stage / f"shard_{shard_id:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / "worker.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config["config_path"]),
                    "--output-root",
                    str(output_root),
                    "--stage",
                    "seed_labels",
                    "--seed-shard",
                    str(shard_id),
                ],
                cwd=SOURCE_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((shard_id, process, handle))
        survivors = []
        for shard_id, process, handle in running:
            if process.poll() is None:
                survivors.append((shard_id, process, handle))
                continue
            handle.close()
            if process.returncode != 0:
                raise RuntimeError(f"retry8 seed-label shard failed: {shard_id}")
        running = survivors
        if running:
            time.sleep(0.2)
    frames = [
        pd.read_parquet(stage / f"shard_{shard_id:02d}" / "labels.parquet")
        for shard_id in range(int(seed["logical_shard_count"]))
    ]
    generated = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    original = labels[
        [
            "task_node_id",
            "x_m",
            "y_m",
            "z_m",
            *BETA_COLUMNS,
            "source_teacher_residual_mm",
            "source_teacher_actual_bounds",
        ]
    ].copy()
    original["physical_point_id"] = original["task_node_id"].map(
        lambda value: f"certified_task_{int(value)}"
    )
    original["label_quality"] = "Gold"
    original["parent_node_id_1"] = original["task_node_id"]
    original["parent_node_id_2"] = original["task_node_id"]
    original["independent_parent_count"] = 2
    original["multiparent_beta_gap_deg"] = 0.0
    original["reverse_return_max_deg"] = 0.0
    original["teacher_fk_residual_mm"] = original.pop(
        "source_teacher_residual_mm"
    ).astype(float)
    original["actual_bounds"] = original.pop(
        "source_teacher_actual_bounds"
    ).astype(bool)
    columns = list(generated.columns) if len(generated) else [
        "physical_point_id", "x_m", "y_m", "z_m", *BETA_COLUMNS,
        "label_quality", "parent_node_id_1", "parent_node_id_2",
        "independent_parent_count", "multiparent_beta_gap_deg",
        "reverse_return_max_deg", "teacher_fk_residual_mm", "actual_bounds",
    ]
    frame = pd.concat(
        [original.reindex(columns=columns), generated.reindex(columns=columns)],
        ignore_index=True,
    )
    frame["xyz_key"] = frame[["x_m", "y_m", "z_m"]].round(12).astype(str).agg("|".join, axis=1)
    frame = frame.sort_values(
        ["xyz_key", "label_quality", "teacher_fk_residual_mm"],
        ascending=[True, True, True],
        kind="stable",
    ).drop_duplicates("xyz_key", keep="first")
    frame = frame.head(int(seed["maximum_unique_rows"])).reset_index(drop=True)
    integrity = bool(
        certificate.get("partial_certificate_pass", False)
        and frame["actual_bounds"].astype(bool).all()
        and pd.to_numeric(frame["teacher_fk_residual_mm"], errors="coerce").le(
            float(seed["continuation_residual_max_mm"])
        ).all()
        and not frame["xyz_key"].duplicated().any()
    )
    _write_parquet(frame, stage / "primary_seed_supervision.parquet")
    counts = frame["label_quality"].value_counts().to_dict()
    result = {
        "gate_pass": integrity,
        "operational_completion": True,
        "teacher_label_integrity_failure": not integrity,
        "legal_unique_label_count": len(frame),
        "gold_label_count": int(counts.get("Gold", 0)),
        "silver_label_count": int(counts.get("Silver", 0)),
        "smoke_minimum_label_count_met": len(frame)
        >= int(config["smoke"]["minimum_strict_unique_labels"]),
        "formal_gold_only_contract": True,
    }
    _write_json(stage / "gate.json", result)
    return result


def stage_authorization(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    certificate = _require(output_root, config, "partial_certificate")
    labels = _require(output_root, config, "seed_labels")
    stage = output_root / STAGE_DIRS["authorization"]
    p0 = float(certificate["observed_partial_coverage"])
    coverage = float(certificate["certified_partial_coverage"])
    relay_count = int(certificate["selected_method_relay_count"])
    authorization = authorize_exploratory_stages(
        partial_certificate_pass=bool(certificate["partial_certificate_pass"]),
        strict_unique_label_count=int(labels["legal_unique_label_count"]),
        relay_root_count=relay_count,
        relay_coverage_gain=coverage - p0,
        relay_stitch_pass=bool(certificate["partial_certificate_pass"]),
        coverage_ratio=coverage,
        largest_component_ratio=float(certificate["largest_coherent_component"]),
        smoke_pipeline_complete=False,
        teacher_label_integrity_failure=bool(labels["teacher_label_integrity_failure"]),
        unresolved_training_implementation_failure=False,
        smoke_student_quality_pass=False,
    )
    result = {
        "gate_pass": bool(authorization.data_legality_gate),
        "operational_completion": True,
        **asdict(authorization),
        "smoke_execution_authorized": bool(
            authorization.smoke_execution_authorized
            and not labels["teacher_label_integrity_failure"]
        ),
        "five_k_teacher_execution_authorized": False,
        "five_k_authorization_pending_smoke": True,
        "formal_or_deployment_authorized": False,
    }
    _write_json(stage / "gate.json", result)
    _write_json(stage / "retry8_next_stage_authorization.json", result)
    return result


def stage_summary(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    authorization = _require(output_root, config, "authorization")
    stage = output_root / STAGE_DIRS["summary"]
    report = {
        "operational_completion": True,
        "partial_certificate_pass": authorization["data_legality_gate"],
        "relay_method_validation_pass": authorization[
            "relay_method_validation_pass"
        ],
        "global_coverage_gate_pass": authorization["global_coverage_gate_pass"],
        "smoke_execution_authorized": authorization["smoke_execution_authorized"],
        "five_k_teacher_execution_authorized": False,
        "student_claim_authorized": False,
        "formal_or_deployment_authorized": False,
    }
    _write_json(stage / "retry8_scientific_report.json", report)
    artifacts = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and not path.is_relative_to(stage)
    ]
    _write_json(
        stage / "retry8_artifact_manifest.json",
        {"schema_version": 1, "artifact_count": len(artifacts), "artifacts": artifacts},
    )
    result = {"gate_pass": True, **report}
    _write_json(stage / "gate.json", result)
    return result


STAGE_RUNNERS = {
    "inventory": stage_inventory,
    "frozen_partial": stage_frozen_partial,
    "relay_registry": stage_relay_registry,
    "growth_funnel": stage_growth_funnel,
    "sampled_screening": stage_sampled_screening,
    "partial_certificate": stage_partial_certificate,
    "seed_labels": stage_seed_labels,
    "authorization": stage_authorization,
    "summary": stage_summary,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v14_2r_retry8_partial_relay.yaml"),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--seed-shard", type=int)
    parser.add_argument("--audit-shard-bundle")
    parser.add_argument("--audit-shard-id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    project_root = project_root_from(SOURCE_ROOT)
    if args.audit_shard_bundle:
        if args.audit_shard_id is None:
            raise SystemExit("--audit-shard-id is required")
        report = base._run_audit_shard_worker(
            config,
            project_root,
            Path(args.audit_shard_bundle).resolve(),
            int(args.audit_shard_id),
        )
        print(json.dumps(_strict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage is None:
        raise SystemExit("--stage is required")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else project_root / str(config["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config["_seed_shard"] = args.seed_shard
    config["_runner_path"] = str(Path(__file__).resolve())
    if args.stage != "inventory" and args.seed_shard is None:
        fixed = _read_json(output_root / STAGE_DIRS["inventory"] / "source_fixed_point.json")
        checks = {
            "source": fixed.get("source_sha") == base._git_sha(),
            "config": fixed.get("config_sha256")
            == base.sha256_file(Path(str(config["config_path"]))),
            "runtime": fixed.get("runtime_sha256") == base._runtime_sha256(),
            "clean": base._tree_clean(),
        }
        if not all(checks.values()):
            raise RuntimeError(f"retry8 fixed-point resume failed: {checks}")
    if args.seed_shard is None:
        completed = _load_completed(output_root, config, args.stage)
        if completed is not None:
            print(json.dumps(_strict(completed), ensure_ascii=False, sort_keys=True))
            return 0
    result = STAGE_RUNNERS[args.stage](config, project_root, output_root)
    if args.seed_shard is None:
        _write_completion(output_root, config, args.stage)
    print(json.dumps(_strict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
