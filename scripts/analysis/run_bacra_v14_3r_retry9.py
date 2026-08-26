#!/usr/bin/env python3
"""Run BACRA V14.3R retry9 coverage-first Stage 0--3."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
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

from quasi_exp.teacher.canonical import weighted_damped_pinv
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import (
    BETA_COLUMNS,
    DEFAULT_BETA_WEIGHTS,
    ENDPOINT_BETA_COLUMNS,
    ExplorationThresholds,
    bounded_fundamental_cycles,
    conservative_weighted_upper_bound_deg,
    execute_exploration_audit_schedules,
    overlap_pair_metrics,
    percentile,
    select_transition_edges,
    summarize_exploration_pair,
    weighted_beta_rms_deg,
)
from quasi_exp.teacher.partial_relay import choose_macroblock_split
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.section_atlas_repair import (
    AuditV2Policy,
    RetryTier,
    section_growth_from_frames,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_inverse import InverseQuery
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    evaluate_workspace_student,
    save_workspace_student_models,
    train_workspace_student,
)

import run_bacra_v14_3_repaired_5k_student as retry8


XYZ_COLUMNS = ("x_m", "y_m", "z_m")
STAGE_DIRS = {
    "inventory": "00_inventory",
    "chart_salvage": "01_existing_chart_salvage",
    "pairwise_overlap": "02_pairwise_overlap_recovery",
    "exploratory_atlas": "03_exploratory_primary_atlas",
    "student": "04_student_smoke",
    "summary": "05_summary",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def project_root_from(source_root: Path) -> Path:
    return retry8.project_root_from(source_root)


def _environment(config: Mapping[str, Any]) -> Any:
    return retry8.optimized_forward(
        retry8.load_environment(
            project_root_from(SOURCE_ROOT),
            SOURCE_ROOT / str(config["sources"]["robot_config"]),
        )
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True
    ).strip()


def _config_sha(config: Mapping[str, Any]) -> str:
    return sha256_file(Path(str(config["config_path"])))


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported retry9 config")
    config = dict(value)
    config["config_path"] = str(config_path)
    if tuple(map(float, config["beta_coordinate_weights"])) != DEFAULT_BETA_WEIGHTS:
        raise ValueError("retry9 beta coordinate weights must be (4,4,2,2,1,1)")
    audit = config["fresh_audit"]
    if (
        int(audit["logical_shard_count"]) != 48
        or int(audit["maximum_concurrent_workers"]) != 12
        or str(audit["assignment_strategy"]) != "cost_balanced_lpt"
        or int(audit["repeats"]) != 2
    ):
        raise ValueError("retry9 fresh audit must use 48 shards, 12 workers, LPT, two repeats")
    if bool(config["claims"]["formal_authorized"]) or bool(
        config["claims"]["deployment_authorized"]
    ):
        raise ValueError("retry9 Stage 0-3 cannot authorize Formal or deployment")
    if not bool(config["student"]["beta_loss_only"]):
        raise ValueError("retry9 Student must use the pure weighted beta loss")
    if str(config["student"].get("device", "")) != "cpu":
        raise ValueError("retry9 Student device must be cpu")
    return config


def _thresholds(config: Mapping[str, Any]) -> ExplorationThresholds:
    row = config["overlap"]
    return ExplorationThresholds(
        minimum_overlap_nodes=int(row["minimum_nodes"]),
        minimum_overlap_fraction=float(row["minimum_fraction"]),
        minimum_overlap_spread_mm=float(row["minimum_spread_mm"]),
        label_weighted_p95_max_deg=float(row["label_weighted_p95_max_deg"]),
        transition_weighted_p95_max_deg=float(row["transition_weighted_p95_max_deg"]),
        cycle_weighted_p95_max_deg=float(row["cycle_weighted_p95_max_deg"]),
        repeat_weighted_p95_max_deg=float(row["repeat_weighted_p95_max_deg"]),
        catastrophic_jump_deg=float(row["catastrophic_jump_deg"]),
        catastrophic_jump_rate_max=float(row["catastrophic_jump_rate_max"]),
        persistent_failure_rate_max=float(row["persistent_failure_rate_max"]),
        residual_p95_max_mm=float(row["residual_p95_max_mm"]),
        residual_p99_max_mm=float(row["residual_p99_max_mm"]),
    )


def _audit_policy(config: Mapping[str, Any]) -> AuditV2Policy:
    row = config["fresh_audit"]
    thresholds = _thresholds(config)
    return AuditV2Policy(
        geometry_p95_max_deg=thresholds.transition_weighted_p95_max_deg,
        geometry_max_deg=thresholds.catastrophic_jump_deg,
        repeat_p95_max_deg=thresholds.repeat_weighted_p95_max_deg,
        continuation_residual_max_mm=thresholds.residual_p99_max_mm,
        repeats_per_direction=int(row["repeats"]),
        repeat_perturbation_rad=float(row["repeat_perturbation_rad"]),
        directions=("forward",),
        retry_tiers=tuple(
            RetryTier(
                tier_id=str(tier["tier_id"]),
                maximum_step_mm=float(tier["maximum_step_mm"]),
                maximum_iterations=int(tier["maximum_iterations"]),
                solver_chain=tuple(map(str, tier["solver_chain"])),
            )
            for tier in row["retry_tiers"]
        ),
    )


def _stage_upstream_digest(output_root: Path, stage_name: str) -> str:
    digest = hashlib.sha256()
    index = STAGE_ORDER.index(stage_name)
    for name in STAGE_ORDER[:index]:
        path = output_root / STAGE_DIRS[name] / "completion_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing upstream completion manifest: {path}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _seal_stage(
    output_root: Path, config: Mapping[str, Any], stage_name: str
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    artifacts = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "completion_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(stage)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "stage_name": stage_name,
        "scientific_source_fixed_point": _git_sha(),
        "config_sha256": _config_sha(config),
        "upstream_completion_sha256": _stage_upstream_digest(output_root, stage_name),
        "artifacts": artifacts,
    }
    _write_json(stage / "completion_manifest.json", manifest)
    return manifest


def _stage_is_complete(
    output_root: Path, config: Mapping[str, Any], stage_name: str
) -> bool:
    stage = output_root / STAGE_DIRS[stage_name]
    path = stage / "completion_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = _read_json(path)
        if (
            manifest["stage_name"] != stage_name
            or manifest["scientific_source_fixed_point"] != _git_sha()
            or manifest["config_sha256"] != _config_sha(config)
            or manifest["upstream_completion_sha256"]
            != _stage_upstream_digest(output_root, stage_name)
        ):
            return False
        return all(
            (stage / row["path"]).is_file()
            and sha256_file(stage / row["path"]) == row["sha256"]
            for row in manifest["artifacts"]
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _upstream_path(config: Mapping[str, Any], project_root: Path, key: str) -> Path:
    upstream = config["upstream"]
    root = project_root / str(upstream["root"])
    value = upstream[key]
    relative = value["path"] if isinstance(value, Mapping) else value
    return root / str(relative)


def _artifact_index(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["path"]): str(row["sha256"])
        for row in manifest.get("artifacts", ())
        if isinstance(row, Mapping) and "path" in row and "sha256" in row
    }


def stage_inventory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["inventory"]
    stage.mkdir(parents=True, exist_ok=True)
    upstream = config["upstream"]
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}
    consultation = SOURCE_ROOT / str(config["sources"]["consultation_input"])
    checks["consultation_sha256"] = (
        consultation.is_file()
        and sha256_file(consultation) == str(config["consultation_sha256"])
    )
    for key in (
        "completion_manifest",
        "artifact_manifest",
        "qualified_chart_executions",
        "qualified_charts",
    ):
        path = _upstream_path(config, project_root, key)
        expected = str(upstream[key]["sha256"])
        actual = sha256_file(path) if path.is_file() else None
        checks[f"{key}_sha256"] = actual == expected
        observed[key] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual}
    if not all(checks.values()):
        _write_json(stage / "gate.json", {"gate_pass": False, "checks": checks, "observed": observed})
        raise RuntimeError("retry9 inventory fixed-point closure failed")
    artifact_manifest = _read_json(_upstream_path(config, project_root, "artifact_manifest"))
    index = _artifact_index(artifact_manifest)
    for key in ("chart_schedules", "chart_labels", "task_nodes", "task_edges"):
        relative = str(upstream[key])
        path = _upstream_path(config, project_root, key)
        expected = index.get(relative)
        actual = sha256_file(path) if path.is_file() else None
        checks[f"manifest_closes_{key}"] = expected is not None and actual == expected
        observed[key] = {"path": str(path), "manifest_sha256": expected, "actual_sha256": actual}
    completion = _read_json(_upstream_path(config, project_root, "completion_manifest"))
    checks["upstream_source_fixed_point"] = (
        str(completion.get("source_sha", ""))
        == str(upstream["scientific_source_fixed_point"])
    )
    checks["upstream_config_sha256"] = (
        str(completion.get("config_sha256", "")) == str(upstream["config_sha256"])
    )
    checks["upstream_runtime_sha256"] = (
        str(completion.get("runtime_sha256", "")) == str(upstream["runtime_sha256"])
    )
    observed["upstream_claim_scope"] = {
        "scientific_source_fixed_point": upstream["scientific_source_fixed_point"],
        "config_sha256": upstream["config_sha256"],
        "runtime_sha256": upstream["runtime_sha256"],
        "completion_manifest_fields": sorted(completion),
    }
    gate = {"gate_pass": bool(all(checks.values())), "checks": checks, "observed": observed}
    _write_json(stage / "source_inventory.json", observed)
    _write_json(stage / "gate.json", gate)
    if not gate["gate_pass"]:
        raise RuntimeError("retry9 artifact manifest does not close required inputs")
    _seal_stage(output_root, config, "inventory")
    return gate


def _historical_chart_rescore(
    schedules: pd.DataFrame,
    executions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    joined = executions.merge(
        schedules.loc[:, ["schedule_id", "audit_kind"]],
        on="schedule_id",
        how="left",
        validate="many_to_one",
    )
    if joined["audit_kind"].isna().any():
        raise ValueError("historical execution references unknown schedule")
    policy = config["historical_rescore"]
    rows = []
    for chart_id, group in joined.groupby("chart_id", sort=True):
        success = group[group["solver_success"].astype(bool)]
        kind_values = {}
        for key, kinds in {
            "edge": ("edge",),
            "root_path": ("root_path",),
            "cycle": ("fundamental_cycle",),
            "multipath": ("multipath_tree", "multipath_chord"),
        }.items():
            values = success.loc[success["audit_kind"].isin(kinds), "geometry_gap_deg"].to_numpy(float)
            kind_values[key] = percentile(values, 95)
        repeat_raw = percentile(success["repeat_gap_deg"].to_numpy(float), 95)
        upper = {
            key: float(conservative_weighted_upper_bound_deg(value))
            for key, value in {**kind_values, "repeat": repeat_raw}.items()
        }
        score = (
            0.50 * upper["edge"] / float(policy["edge_scale_deg"])
            + 0.15 * upper["root_path"] / float(policy["root_path_scale_deg"])
            + 0.15 * upper["cycle"] / float(policy["cycle_scale_deg"])
            + 0.10 * upper["multipath"] / float(policy["multipath_scale_deg"])
            + 0.10 * upper["repeat"] / float(policy["repeat_scale_deg"])
        )
        persistent_rate = float(group["classification"].astype(str).eq("persistent_numerical").mean())
        raw = success["geometry_gap_deg"].to_numpy(float)
        raw_jump = float(np.mean(raw > float(policy["catastrophic_jump_deg"]))) if len(raw) else 1.0
        upper_jump = float(
            np.mean(conservative_weighted_upper_bound_deg(raw) > float(policy["catastrophic_jump_deg"]))
        ) if len(raw) else 1.0
        residual = success["residual_mm"].to_numpy(float)
        residual_p95 = percentile(residual, 95)
        residual_p99 = percentile(residual, 99)
        passed = bool(
            score <= float(policy["score_max"])
            and persistent_rate <= float(policy["persistent_failure_rate_max"])
            and raw_jump <= float(policy["catastrophic_jump_rate_max"])
            and upper_jump <= float(policy["catastrophic_jump_rate_max"])
            and residual_p95 <= float(policy["residual_p95_max_mm"])
            and residual_p99 <= float(policy["residual_p99_max_mm"])
        )
        rows.append(
            {
                "chart_id": str(chart_id),
                "label_chart_id": str(chart_id).split("__fragment_", 1)[0],
                "evidence_kind": "conservative_upper_bound",
                **{f"raw_{key}_p95_deg": value for key, value in {**kind_values, "repeat": repeat_raw}.items()},
                **{f"weighted_upper_{key}_p95_deg": value for key, value in upper.items()},
                **{f"exact_weighted_{key}_p95_deg": math.nan for key in upper},
                "exact_weighted_group_metrics_available": False,
                "consistency_score_upper": score,
                "persistent_failure_rate": persistent_rate,
                "raw_jump_gt_5deg_rate": raw_jump,
                "weighted_upper_jump_gt_5deg_rate": upper_jump,
                "residual_p95_mm": residual_p95,
                "residual_p99_mm": residual_p99,
                "exploration_salvaged": passed,
            }
        )
    return pd.DataFrame.from_records(rows)


def stage_chart_salvage(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["chart_salvage"]
    stage.mkdir(parents=True, exist_ok=True)
    schedules = pd.read_parquet(_upstream_path(config, project_root, "chart_schedules"))
    executions = pd.read_parquet(_upstream_path(config, project_root, "qualified_chart_executions"))
    labels = pd.read_parquet(_upstream_path(config, project_root, "chart_labels"))
    rescore = _historical_chart_rescore(schedules, executions, config)
    selected_evidence = tuple(
        sorted(rescore.loc[rescore["exploration_salvaged"], "chart_id"].astype(str))
    )
    selected = tuple(
        sorted(
            rescore.loc[rescore["exploration_salvaged"], "label_chart_id"]
            .astype(str)
            .unique()
        )
    )
    support_keys: set[tuple[str, int]] = set()
    selected_evidence_set = frozenset(selected_evidence)
    for row in schedules.itertuples(index=False):
        evidence_chart_id = str(row.chart_id)
        if evidence_chart_id not in selected_evidence_set:
            continue
        label_chart_id = evidence_chart_id.split("__fragment_", 1)[0]
        support_keys.update(
            (label_chart_id, int(node_id)) for node_id in row.path_task_node_ids
        )
    label_keys = list(
        zip(labels["chart_id"].astype(str), labels["task_node_id"].astype(int), strict=True)
    )
    salvaged_labels = labels[
        pd.Series((key in support_keys for key in label_keys), index=labels.index)
    ].copy()
    raw_union = int(labels["task_node_id"].nunique())
    salvaged_union = int(salvaged_labels["task_node_id"].nunique())
    gate = {
        "gate_pass": bool(selected and salvaged_union > 0),
        "evidence_kind": "historical_conservative_upper_bound",
        "salvaged_evidence_chart_ids": list(selected_evidence),
        "salvaged_chart_ids": list(selected),
        "salvaged_chart_count": len(selected),
        "raw_support_union_count": raw_union,
        "salvaged_support_union_count": salvaged_union,
        "salvage_ratio": salvaged_union / max(1, raw_union),
        "exact_weighted_group_metrics_available": False,
    }
    _write_parquet(rescore, stage / "historical_chart_rescore.parquet")
    _write_parquet(salvaged_labels, stage / "salvaged_chart_labels.parquet")
    _write_json(stage / "gate.json", gate)
    if not gate["gate_pass"]:
        raise RuntimeError("no historical chart survived retry9 conservative salvage")
    _seal_stage(output_root, config, "chart_salvage")
    return gate


def _lineage_family(value: str) -> str:
    prefix, separator, suffix = str(value).rpartition("_")
    return prefix if separator and suffix.isdigit() else str(value)


def _schedule_id(pair_id: str, chart_id: str, kind: str, path: Sequence[int]) -> str:
    identity = f"{pair_id}:{chart_id}:{kind}:{','.join(map(str, path))}"
    return "retry9_" + hashlib.sha256(identity.encode()).hexdigest()[:20]


def _build_pair_registry(
    labels: pd.DataFrame,
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = _thresholds(config)
    audit = config["fresh_audit"]
    xyz_by_node = {
        int(row.task_node_id): np.asarray([row.x_m, row.y_m, row.z_m], dtype=float)
        for row in tasks.itertuples(index=False)
    }
    by_chart = {
        str(chart): rows.sort_values("task_node_id").copy()
        for chart, rows in labels.groupby("chart_id", sort=True)
    }
    task_adjacency: dict[int, set[int]] = {}
    for row in edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        task_adjacency.setdefault(left, set()).add(right)
        task_adjacency.setdefault(right, set()).add(left)
    pair_rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    for chart_a, chart_b in itertools.combinations(sorted(by_chart), 2):
        left, right = by_chart[chart_a], by_chart[chart_b]
        family_a = _lineage_family(str(left.iloc[0]["root_candidate_id"]))
        family_b = _lineage_family(str(right.iloc[0]["root_candidate_id"]))
        pair_id = f"{chart_a}__{chart_b}"
        metrics = overlap_pair_metrics(left, right, thresholds=thresholds)
        same_lineage = family_a == family_b
        metrics["label_overlap_gate"] = bool(metrics["label_overlap_gate"] and same_lineage)
        pair_rows.append(
            {"pair_id": pair_id, "chart_a": chart_a, "chart_b": chart_b,
             "lineage_family_a": family_a, "lineage_family_b": family_b,
             "same_lineage_family": same_lineage, **metrics}
        )
        if not metrics["label_overlap_gate"]:
            continue
        common = left.merge(right, on="task_node_id", suffixes=("_a", "_b"), validate="one_to_one")
        beta_a = common.loc[:, [f"{column}_a" for column in BETA_COLUMNS]].to_numpy(float)
        beta_b = common.loc[:, [f"{column}_b" for column in BETA_COLUMNS]].to_numpy(float)
        gaps = np.asarray(weighted_beta_rms_deg(beta_a, beta_b), dtype=float)
        gap_by_node = dict(zip(common["task_node_id"].astype(int), gaps, strict=True))
        common_ids = tuple(common["task_node_id"].astype(int))
        common_set = frozenset(common_ids)
        pair_edge_tuples = sorted(
            {
                (left, right)
                for left in common_set
                for right in task_adjacency.get(left, ())
                if left < right and right in common_set
            }
        )
        pair_edges = pd.DataFrame.from_records(
            pair_edge_tuples, columns=["left_node_id", "right_node_id"]
        )
        transitions = select_transition_edges(
            common_ids, pair_edges, xyz_by_node, gap_by_node,
            maximum_edges=int(audit["maximum_transition_edges_per_pair"]),
            high_gap_edges=int(audit["high_gap_transition_edges_per_pair"]),
        )
        cycles = bounded_fundamental_cycles(
            common_ids, pair_edges,
            maximum_cycles=int(audit["maximum_cycles_per_pair"]),
            maximum_length=int(audit["maximum_cycle_length"]),
        )
        for path in transitions:
            for chart_id in (chart_a, chart_b):
                schedule_id = _schedule_id(pair_id, chart_id, "transition", path)
                schedules.append(
                    {"schedule_id": schedule_id, "pair_id": pair_id, "chart_id": chart_id,
                     "audit_kind": "transition", "path_node_ids": list(path),
                     "physical_entity_id": f"{pair_id}:edge:{path[0]}:{path[1]}",
                     "cost": len(path) * int(audit["repeats"])}
                )
        for path in cycles:
            chart_id = chart_a
            schedule_id = _schedule_id(pair_id, chart_id, "cycle", path)
            schedules.append(
                {"schedule_id": schedule_id, "pair_id": pair_id, "chart_id": chart_id,
                 "audit_kind": "cycle", "path_node_ids": list(path),
                 "physical_entity_id": f"{pair_id}:cycle:{schedule_id}",
                 "cost": len(path) * int(audit["repeats"])}
            )
    return pd.DataFrame.from_records(pair_rows), pd.DataFrame.from_records(schedules)


def _assign_lpt(schedules: pd.DataFrame, shard_count: int) -> pd.DataFrame:
    result = schedules.copy()
    if len(result) == 0:
        result["shard_id"] = pd.Series(dtype="int64")
        return result
    loads = [0] * int(shard_count)
    assignment: dict[str, int] = {}
    for row in result.sort_values(["cost", "schedule_id"], ascending=[False, True]).itertuples(index=False):
        shard = min(range(int(shard_count)), key=lambda value: (loads[value], value))
        assignment[str(row.schedule_id)] = shard
        loads[shard] += int(row.cost)
    result["shard_id"] = result["schedule_id"].map(assignment).astype(int)
    return result.sort_values(["shard_id", "schedule_id"]).reset_index(drop=True)


def _selected_edges_for_labels(labels: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for chart_id, chart_labels in labels.groupby("chart_id", sort=True):
        nodes = frozenset(chart_labels["task_node_id"].astype(int))
        for edge in edges.itertuples(index=False):
            left, right = int(edge.left_node_id), int(edge.right_node_id)
            if left in nodes and right in nodes:
                rows.append({"chart_id": str(chart_id), "left_node_id": left, "right_node_id": right})
    return pd.DataFrame.from_records(rows, columns=["chart_id", "left_node_id", "right_node_id"])


def run_audit_shard_worker(
    config: Mapping[str, Any], project_root: Path, output_root: Path, shard_id: int
) -> None:
    stage = output_root / STAGE_DIRS["pairwise_overlap"]
    shard = stage / "fresh_audit" / f"shard_{int(shard_id):02d}"
    shard.mkdir(parents=True, exist_ok=True)
    schedules = pd.read_parquet(stage / "fresh_audit_schedules.parquet")
    schedules = schedules[schedules["shard_id"].astype(int).eq(int(shard_id))].copy()
    tasks = pd.read_parquet(stage / "task_nodes.parquet")
    edges = pd.read_parquet(stage / "task_edges.parquet")
    labels = pd.read_parquet(stage / "chart_labels.parquet")
    chart_ids = frozenset(schedules["chart_id"].astype(str))
    labels = labels[labels["chart_id"].astype(str).isin(chart_ids)].copy()
    progress_path = shard / "progress.json"
    started = time.time()
    _write_json(progress_path, {"status": "running", "shard_id": int(shard_id),
                                "completed_schedules": 0, "total_schedules": len(schedules),
                                "updated_at_unix_s": started})
    if len(schedules):
        nodes = retry8.atlas_nodes_from_frames(tasks, edges)
        selected_edges = _selected_edges_for_labels(labels, edges)
        growth = section_growth_from_frames(nodes, labels, selected_edges)
        environment = _environment(config)
        _nodes, continuation = retry8._segmented_continuation(environment, tasks, edges)
        retry_adapter = retry8.v142r._registered_retry_adapter(environment, tasks, edges)
        every = int(config["fresh_audit"]["progress_every_schedules"])
        last = -every

        def progress(completed: int, total: int) -> None:
            nonlocal last
            if completed != total and completed - last < every:
                return
            _write_json(progress_path, {"status": "running", "shard_id": int(shard_id),
                                        "completed_schedules": completed, "total_schedules": total,
                                        "updated_at_unix_s": time.time()})
            last = completed

        executions = execute_exploration_audit_schedules(
            growth, schedules, continuation, _audit_policy(config),
            retry_continuation=retry_adapter,
            weights=config["beta_coordinate_weights"], progress_callback=progress,
        )
    else:
        executions = pd.DataFrame(
            columns=["schedule_id", "pair_id", "audit_kind", "chart_id", "physical_entity_id",
                     "repeat_index", "repeat_perturbation_l2_rad", "solver_success",
                     "geometry_gap_deg", "weighted_geometry_gap_deg", "weighted_repeat_gap_deg",
                     "residual_mm", "retry_tier", "registered_solver_chain",
                     "executed_solver_chain", "solver_chain_sha256", "classification",
                     "failure_source_node", "failure_target_node", "exact_bounds",
                     "oracle_used_for_pass", *ENDPOINT_BETA_COLUMNS]
        )
    path = shard / "executions.parquet"
    _write_parquet(executions, path)
    report = {"status": "complete", "shard_id": int(shard_id),
              "schedule_count": len(schedules), "execution_count": len(executions),
              "executions_sha256": sha256_file(path), "finished_at_unix_s": time.time()}
    _write_json(shard / "report.json", report)
    _write_json(progress_path, {**report, "updated_at_unix_s": time.time()})


def _run_shards(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> None:
    stage = output_root / STAGE_DIRS["pairwise_overlap"]
    count = int(config["fresh_audit"]["logical_shard_count"])
    maximum = int(config["fresh_audit"]["maximum_concurrent_workers"])
    pending = []
    for shard_id in range(count):
        root = stage / "fresh_audit" / f"shard_{shard_id:02d}"
        report_path = root / "report.json"
        execution_path = root / "executions.parquet"
        if report_path.is_file() and execution_path.is_file():
            report = _read_json(report_path)
            if report.get("status") == "complete" and report.get("executions_sha256") == sha256_file(execution_path):
                continue
        pending.append(shard_id)
    active: list[tuple[int, subprocess.Popen[Any], Any]] = []
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1"})
    while pending or active:
        while pending and len(active) < maximum:
            shard_id = pending.pop(0)
            log_path = stage / "fresh_audit" / f"shard_{shard_id:02d}" / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config["config_path"]),
                       "--output-root", str(output_root), "--audit-shard-worker", str(shard_id)]
            process = subprocess.Popen(command, cwd=SOURCE_ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
            active.append((shard_id, process, handle))
        survivors = []
        for shard_id, process, handle in active:
            code = process.poll()
            if code is None:
                survivors.append((shard_id, process, handle))
                continue
            handle.close()
            if code != 0:
                for _other_id, other, other_handle in survivors:
                    other.terminate(); other_handle.close()
                raise RuntimeError(f"fresh audit shard {shard_id} failed with exit code {code}")
        active = survivors
        if active:
            time.sleep(0.25)


def stage_pairwise_overlap(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["pairwise_overlap"]
    stage.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(output_root / STAGE_DIRS["chart_salvage"] / "salvaged_chart_labels.parquet")
    tasks = pd.read_parquet(_upstream_path(config, project_root, "task_nodes"))
    edges = pd.read_parquet(_upstream_path(config, project_root, "task_edges"))
    pair_registry, schedules = _build_pair_registry(labels, tasks, edges, config)
    schedules = _assign_lpt(schedules, int(config["fresh_audit"]["logical_shard_count"]))
    _write_parquet(pair_registry, stage / "label_overlap_pairs.parquet")
    _write_parquet(schedules, stage / "fresh_audit_schedules.parquet")
    _write_parquet(tasks, stage / "task_nodes.parquet")
    _write_parquet(edges, stage / "task_edges.parquet")
    _write_parquet(labels, stage / "chart_labels.parquet")
    _run_shards(config, project_root, output_root)
    executions = pd.concat(
        [pd.read_parquet(stage / "fresh_audit" / f"shard_{shard_id:02d}" / "executions.parquet")
         for shard_id in range(int(config["fresh_audit"]["logical_shard_count"]))],
        ignore_index=True,
    )
    _write_parquet(executions, stage / "fresh_audit_executions.parquet")
    metric_rows = []
    pair_lookup = pair_registry.set_index("pair_id", drop=False)
    for pair_id in pair_registry["pair_id"].astype(str):
        fresh = summarize_exploration_pair(
            executions[executions["pair_id"].astype(str).eq(pair_id)],
            thresholds=_thresholds(config),
        )
        label_gate = bool(pair_lookup.loc[pair_id, "label_overlap_gate"])
        metric_rows.append({"pair_id": pair_id, **fresh,
                            "label_overlap_gate": label_gate,
                            "pair_gate": bool(label_gate and fresh.get("fresh_audit_gate", False))})
    metrics = pd.DataFrame.from_records(metric_rows)
    combined = pair_registry.merge(metrics, on=["pair_id", "label_overlap_gate"], how="left", validate="one_to_one")
    _write_parquet(combined, stage / "pair_qualification.parquet")
    passed = combined[combined["pair_gate"].fillna(False).astype(bool)]
    gate = {"gate_pass": bool(len(passed)), "candidate_pair_count": len(pair_registry),
            "fresh_schedule_count": len(schedules), "fresh_execution_count": len(executions),
            "qualified_pair_count": len(passed),
            "qualified_pair_ids": sorted(passed["pair_id"].astype(str))}
    _write_json(stage / "gate.json", gate)
    _seal_stage(output_root, config, "pairwise_overlap")
    return gate


def _largest_component(vertices: Sequence[str], edges: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    adjacency = {str(vertex): set() for vertex in vertices}
    for left, right in edges:
        adjacency.setdefault(str(left), set()).add(str(right))
        adjacency.setdefault(str(right), set()).add(str(left))
    components = []
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        queue = [seed]
        seen = {seed}
        while queue:
            node = queue.pop()
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor); queue.append(neighbor)
        remaining -= seen
        components.append(tuple(sorted(seen)))
    return min(components, key=lambda value: (-len(value), value)) if components else ()


def _largest_node_component(node_ids: Sequence[int], edges: pd.DataFrame) -> frozenset[int]:
    retained = frozenset(map(int, node_ids))
    adjacency = {node: set() for node in retained}
    for row in edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left in retained and right in retained:
            adjacency[left].add(right); adjacency[right].add(left)
    components = []
    remaining = set(retained)
    while remaining:
        seed = min(remaining); queue = [seed]; seen = {seed}
        while queue:
            node = queue.pop()
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor); queue.append(neighbor)
        remaining -= seen; components.append(frozenset(seen))
    return min(components, key=lambda value: (-len(value), tuple(sorted(value)))) if components else frozenset()


def _medoid_labels(labels: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    rows = []
    catastrophic = set()
    for node_id, group in labels.groupby("task_node_id", sort=True):
        ordered = group.sort_values("chart_id").reset_index(drop=True)
        beta = ordered.loc[:, BETA_COLUMNS].to_numpy(float)
        if len(beta) > 1:
            weighted = np.asarray([[weighted_beta_rms_deg(a, b) for b in beta] for a in beta])
            raw = np.asarray([[np.sqrt(np.mean(np.square(np.degrees(a - b)))) for b in beta] for a in beta])
            if float(np.max(weighted)) > 5.0 or float(np.max(raw)) > 5.0:
                catastrophic.add(int(node_id)); continue
            index = min(range(len(ordered)), key=lambda item: (float(np.sum(weighted[item])), str(ordered.iloc[item]["chart_id"])))
        else:
            index = 0
        row = ordered.iloc[index].to_dict()
        row["supporting_chart_count"] = len(ordered)
        row["selection_method"] = "weighted_overlap_medoid"
        rows.append(row)
    return pd.DataFrame.from_records(rows), catastrophic


def stage_exploratory_atlas(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["exploratory_atlas"]
    stage.mkdir(parents=True, exist_ok=True)
    pair_stage = output_root / STAGE_DIRS["pairwise_overlap"]
    pairs = pd.read_parquet(pair_stage / "pair_qualification.parquet")
    passed = pairs[pairs["pair_gate"].fillna(False).astype(bool)]
    vertices = sorted(set(passed["chart_a"].astype(str)) | set(passed["chart_b"].astype(str)))
    component = _largest_component(vertices, list(zip(passed["chart_a"].astype(str), passed["chart_b"].astype(str))))
    labels = pd.read_parquet(pair_stage / "chart_labels.parquet")
    labels = labels[labels["chart_id"].astype(str).isin(component)].copy()
    if len(labels) == 0:
        all_tasks = pd.read_parquet(pair_stage / "task_nodes.parquet")
        empty_labels = pd.DataFrame(
            columns=["task_node_id", "chart_id", *BETA_COLUMNS, *XYZ_COLUMNS,
                     "fk_residual_mm", "exact_bounds"]
        )
        abstention = pd.DataFrame(
            {"task_node_id": all_tasks["task_node_id"].astype(int),
             "abstained": True, "reason": "no_qualified_pair_component"}
        )
        gate = {
            "state": "RED",
            "hard_checks": {"qualified_pair_component_exists": False},
            "selected_chart_component": [],
            "unique_label_count": 0,
            "coverage_fraction": 0.0,
            "zero_hop_abstention_count": 0,
            "one_hop_sensitivity_retained_count": 0,
            "catastrophic_disagreement_node_count": 0,
            "persistent_failure_endpoint_count": 0,
            "fk_p95_mm": None,
            "fk_p99_mm": None,
            "student_training_authorized": False,
            "formal_authorized": False,
            "deployment_authorized": False,
        }
        _write_parquet(empty_labels, stage / "exploratory_primary_labels.parquet")
        _write_parquet(abstention, stage / "exploratory_abstention.parquet")
        _write_json(stage / "exploratory_atlas_gate.json", gate)
        _write_json(stage / "gate.json", gate)
        _seal_stage(output_root, config, "exploratory_atlas")
        return gate
    medoid, catastrophic = _medoid_labels(labels)
    historical = pd.read_parquet(_upstream_path(config, project_root, "qualified_chart_executions"))
    fresh = pd.read_parquet(pair_stage / "fresh_audit_executions.parquet")
    bad_endpoints = set()
    for frame in (historical, fresh):
        failures = frame[frame["classification"].astype(str).eq("persistent_numerical")]
        for column in ("failure_source_node", "failure_target_node"):
            bad_endpoints.update(int(value) for value in failures[column].dropna().astype(int))
    zero_hop = bad_endpoints | catastrophic
    candidate = medoid[~medoid["task_node_id"].astype(int).isin(zero_hop)].copy()
    environment = _environment(config)
    beta = candidate.loc[:, BETA_COLUMNS].to_numpy(float)
    xyz = candidate.loc[:, XYZ_COLUMNS].to_numpy(float)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    within_bounds = np.all((beta >= bounds[:, 0] - 1e-12) & (beta <= bounds[:, 1] + 1e-12), axis=1)
    fk_xyz = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    fk_residual = np.linalg.norm(fk_xyz - xyz, axis=1) * 1000.0
    candidate["fk_residual_mm"] = fk_residual
    candidate["exact_bounds"] = within_bounds
    individual_max = float(config["exploratory_atlas"]["individual_fk_max_mm"])
    row_valid = within_bounds & np.isfinite(fk_residual) & (fk_residual <= individual_max)
    removed_physical = set(candidate.loc[~row_valid, "task_node_id"].astype(int))
    candidate = candidate.loc[row_valid].copy()
    edges = pd.read_parquet(pair_stage / "task_edges.parquet")
    coherent = _largest_node_component(candidate["task_node_id"].astype(int), edges)
    candidate = candidate[candidate["task_node_id"].astype(int).isin(coherent)].copy()
    retained = frozenset(candidate["task_node_id"].astype(int))
    one_hop = set(zero_hop)
    for row in edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left in zero_hop: one_hop.add(right)
        if right in zero_hop: one_hop.add(left)
    one_hop_retained_count = int(candidate[~candidate["task_node_id"].astype(int).isin(one_hop)].shape[0])
    all_tasks = pd.read_parquet(pair_stage / "task_nodes.parquet")
    reasons = []
    for node_id in all_tasks["task_node_id"].astype(int):
        if node_id in retained: reason = "retained"
        elif node_id in catastrophic: reason = "catastrophic_chart_disagreement"
        elif node_id in bad_endpoints: reason = "persistent_failure_endpoint"
        elif node_id in removed_physical: reason = "bounds_or_fk_gt_10mm"
        elif node_id in set(medoid["task_node_id"].astype(int)): reason = "outside_largest_coherent_component"
        else: reason = "not_covered_by_selected_chart_component"
        reasons.append({"task_node_id": node_id, "abstained": reason != "retained", "reason": reason})
    fk_values = candidate["fk_residual_mm"].to_numpy(float)
    fk_p95, fk_p99 = percentile(fk_values, 95), percentile(fk_values, 99)
    hard_checks = {
        "qualified_pair_component_exists": len(component) >= 2,
        "single_lineage_family": len({_lineage_family(value) for value in labels["root_candidate_id"].astype(str)}) <= 1,
        "catastrophic_nodes_abstained": not bool(catastrophic & set(retained)),
        "persistent_endpoints_abstained": not bool(bad_endpoints & set(retained)),
        "exact_bounds_all": bool(candidate["exact_bounds"].all()),
        "fk_p95": fk_p95 <= float(config["exploratory_atlas"]["fk_p95_max_mm"]),
        "fk_p99": fk_p99 <= float(config["exploratory_atlas"]["fk_p99_max_mm"]),
    }
    count = len(candidate)
    green = int(config["exploratory_atlas"]["green_minimum_unique_rows"])
    yellow = int(config["exploratory_atlas"]["yellow_minimum_unique_rows"])
    if not all(hard_checks.values()) or count < yellow:
        state = "RED"
    elif count < green:
        state = "YELLOW"
    else:
        state = "GREEN"
    gate = {"state": state, "hard_checks": hard_checks, "selected_chart_component": list(component),
            "unique_label_count": count, "coverage_fraction": count / int(config["exploratory_atlas"]["denominator_task_nodes"]),
            "zero_hop_abstention_count": len(zero_hop), "one_hop_sensitivity_retained_count": one_hop_retained_count,
            "catastrophic_disagreement_node_count": len(catastrophic),
            "persistent_failure_endpoint_count": len(bad_endpoints),
            "fk_p95_mm": fk_p95, "fk_p99_mm": fk_p99,
            "student_training_authorized": state == "GREEN",
            "formal_authorized": False, "deployment_authorized": False}
    _write_parquet(candidate.sort_values("task_node_id"), stage / "exploratory_primary_labels.parquet")
    _write_parquet(pd.DataFrame.from_records(reasons), stage / "exploratory_abstention.parquet")
    _write_json(stage / "exploratory_atlas_gate.json", gate)
    _write_json(stage / "gate.json", gate)
    _seal_stage(output_root, config, "exploratory_atlas")
    return gate


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )


def _two_step_dls(environment: Any, beta: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    corrected = np.asarray(beta, dtype=float).copy()
    bounds = np.asarray(environment.bounds, dtype=float)
    for _ in range(2):
        for index in range(len(corrected)):
            current_xyz = np.asarray(environment.fk(corrected[index]), dtype=float).reshape(-1, 3)[0]
            error = xyz[index] - current_xyz
            jacobian = np.asarray(environment.jacobian(corrected[index]), dtype=float).reshape(3, 6)
            step = weighted_damped_pinv(jacobian, damping=1e-3, weights=DEFAULT_BETA_WEIGHTS) @ error
            corrected[index] = np.clip(corrected[index] + step, bounds[:, 0], bounds[:, 1])
    return corrected


def stage_student(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["student"]
    stage.mkdir(parents=True, exist_ok=True)
    atlas_gate = _read_json(output_root / STAGE_DIRS["exploratory_atlas"] / "gate.json")
    if not atlas_gate["student_training_authorized"]:
        gate = {"status": "skipped", "reason": f"exploratory_atlas_{atlas_gate['state']}",
                "student_smoke_pass": False, "formal_authorized": False, "deployment_authorized": False}
        _write_json(stage / "gate.json", gate)
        _seal_stage(output_root, config, "student")
        return gate
    if str(config["student"]["device"]) != "cpu":
        raise RuntimeError("retry9 Student device contract is not CPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("retry9 Student requires CUDA_VISIBLE_DEVICES=-1 before process start")
    import tensorflow as tf

    visible_gpus = tf.config.get_visible_devices("GPU")
    if visible_gpus:
        raise RuntimeError(
            f"retry9 CPU Student unexpectedly sees {len(visible_gpus)} GPU devices"
        )
    labels = pd.read_parquet(output_root / STAGE_DIRS["exploratory_atlas"] / "exploratory_primary_labels.parquet")
    split_config = config["split"]
    split = choose_macroblock_split(
        labels, block_sizes_mm=split_config["block_sizes_mm"], split_seed=int(split_config["seed"]),
        split_fractions=split_config["fractions"],
        minimum_train_blocks=int(split_config["minimum_train_blocks"]),
        minimum_validation_blocks=int(split_config["minimum_validation_blocks"]),
        minimum_test_blocks=int(split_config["minimum_test_blocks"]),
        minimum_rows_per_split=int(split_config["minimum_rows_per_split"]),
    )
    environment = _environment(config)
    frame = split.frame.copy()
    jacobians = []
    for beta in frame.loc[:, BETA_COLUMNS].to_numpy(float):
        jacobians.append(np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6).reshape(-1))
    jacobian_array = np.asarray(jacobians)
    for index, column in enumerate(JACOBIAN_COLUMNS):
        frame[column] = jacobian_array[:, index]
    frame["record_id"] = frame["task_node_id"].map(lambda value: f"retry9_node_{int(value):06d}")
    frame["kind"] = "static"
    frame["chart_id"] = "retry9_exploratory_primary"
    frame["is_primary"] = True
    frame["sample_weight"] = 1.0
    train = frame[frame["split_role"].eq("train_core")].copy()
    validation = frame[frame["split_role"].eq("validation")].copy()
    test = frame[frame["split_role"].eq("test")].copy()
    student = config["student"]
    result = train_workspace_student(
        train, validation, mode=RepresentationMode.XYZ_GLOBAL,
        geometry=_student_geometry(environment),
        config=WorkspaceStudentTrainingConfig(
            hidden_units=tuple(map(int, student["hidden_units"])),
            learning_rate=float(student["learning_rate"]), max_steps=int(student["max_steps"]),
            validation_interval=int(student["validation_interval"]),
            patience_intervals=int(student["patience_intervals"]), seed=int(student["seed"]),
            beta_coordinate_weights=tuple(map(float, config["beta_coordinate_weights"])),
            beta_loss_only=True,
        ),
    )
    save_workspace_student_models(result.models, stage / "models")
    prediction = np.asarray(
        result.models.global_model(test.loc[:, XYZ_COLUMNS].to_numpy(np.float32), training=False),
        dtype=float,
    )
    xyz = test.loc[:, XYZ_COLUMNS].to_numpy(float)
    raw_fk = np.linalg.norm(np.asarray(environment.fk(prediction)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    corrected = _two_step_dls(environment, prediction, xyz)
    dls_fk = np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    bounds = np.asarray(environment.bounds, dtype=float)
    violation = ~np.all((prediction >= bounds[:, 0] - 1e-12) & (prediction <= bounds[:, 1] + 1e-12), axis=1)
    no_nan = bool(np.isfinite(prediction).all() and np.isfinite(raw_fk).all() and np.isfinite(dls_fk).all())
    raw_p95, dls_p95 = percentile(raw_fk, 95), percentile(dls_fk, 95)
    smoke = bool(no_nan and not np.any(violation) and (
        raw_p95 < float(student["raw_fk_p95_max_mm"])
        or dls_p95 < float(student["dls_two_step_fk_p95_max_mm"])
    ))
    predictions = test.loc[:, ["record_id", "task_node_id", *XYZ_COLUMNS, *BETA_COLUMNS]].copy()
    for index, column in enumerate(BETA_COLUMNS):
        predictions[f"predicted_{column}"] = prediction[:, index]
        predictions[f"dls2_{column}"] = corrected[:, index]
    predictions["raw_fk_residual_mm"] = raw_fk
    predictions["dls2_fk_residual_mm"] = dls_fk
    predictions["bounds_violation"] = violation
    gate = {"status": "complete", "student_smoke_pass": smoke, "no_nan": no_nan,
            "bounds_violation_count": int(np.count_nonzero(violation)),
            "raw_fk_p95_mm": raw_p95, "dls_two_step_fk_p95_mm": dls_p95,
            "train_row_count": len(train), "validation_row_count": len(validation), "test_row_count": len(test),
            "macroblock_size_mm": split.block_size_mm, "macroblock_counts": dict(split.block_counts),
            "loss": "pure_normalized_weighted_beta", "seed": int(student["seed"]),
            "training_device": "cpu", "visible_gpu_count": len(visible_gpus),
            "recovered_labels_only": True, "formal_authorized": False,
            "deployment_authorized": False, "frontier_relay_authorized": False}
    _write_parquet(frame, stage / "student_supervision.parquet")
    _write_parquet(result.history, stage / "training_history.parquet")
    _write_parquet(predictions, stage / "test_predictions.parquet")
    _write_json(stage / "gate.json", gate)
    _seal_stage(output_root, config, "student")
    return gate


def stage_summary(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS["summary"]
    stage.mkdir(parents=True, exist_ok=True)
    inventory = _read_json(output_root / STAGE_DIRS["inventory"] / "gate.json")
    salvage = _read_json(output_root / STAGE_DIRS["chart_salvage"] / "gate.json")
    pair = _read_json(output_root / STAGE_DIRS["pairwise_overlap"] / "gate.json")
    atlas = _read_json(output_root / STAGE_DIRS["exploratory_atlas"] / "gate.json")
    student = _read_json(output_root / STAGE_DIRS["student"] / "gate.json")
    gate = {
        "operational_completion": True,
        "artifact_completeness": True,
        "inventory_gate": bool(inventory["gate_pass"]),
        "chart_salvage_gate": bool(salvage["gate_pass"]),
        "pairwise_overlap_gate": bool(pair["gate_pass"]),
        "exploratory_scientific_state": atlas["state"],
        "exploratory_unique_label_count": atlas["unique_label_count"],
        "exploratory_coverage_fraction": atlas["coverage_fraction"],
        "student_status": student["status"],
        "student_smoke_pass": bool(student["student_smoke_pass"]),
        "frontier_relay_executed": False,
        "downstream_authorization": {"stage4_frontier_relay": False, "formal": False, "deployment": False},
        "formal_claim_authorized": False,
    }
    _write_json(stage / "gate.json", gate)
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and stage not in path.parents:
            artifacts.append({"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
    manifest = {"schema_version": 1, "scientific_source_fixed_point": _git_sha(),
                "config_sha256": _config_sha(config), "artifacts": artifacts}
    _write_json(stage / "artifact_manifest.json", manifest)
    completion = _seal_stage(output_root, config, "summary")
    completion.update({"operational_completion": True,
                       "gate_sha256": sha256_file(stage / "gate.json"),
                       "artifact_manifest_sha256": sha256_file(stage / "artifact_manifest.json")})
    _write_json(stage / "completion_manifest.json", completion)
    return gate


STAGE_RUNNERS = {
    "inventory": stage_inventory,
    "chart_salvage": stage_chart_salvage,
    "pairwise_overlap": stage_pairwise_overlap,
    "exploratory_atlas": stage_exploratory_atlas,
    "student": stage_student,
}


def _preflight_checkout() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=SOURCE_ROOT).returncode != 0:
        raise RuntimeError("retry9 run requires a clean scientific checkout")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=SOURCE_ROOT).returncode != 0:
        raise RuntimeError("retry9 run requires a clean scientific index")
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=SOURCE_ROOT,
                      stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("retry9 run must execute from a detached scientific fixed point")


def run(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    _preflight_checkout()
    project_root = project_root_from(SOURCE_ROOT)
    identity = {"schema_version": 1, "experiment_id": config["experiment_id"],
                "scientific_source_fixed_point": _git_sha(), "config_sha256": _config_sha(config)}
    identity_path = output_root / "run_identity.json"
    if identity_path.exists():
        if _read_json(identity_path) != identity:
            raise RuntimeError("output root belongs to a different retry9 identity")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _write_json(identity_path, identity)
    for stage_name in STAGE_ORDER[:-1]:
        if _stage_is_complete(output_root, config, stage_name):
            continue
        STAGE_RUNNERS[stage_name](config, project_root, output_root)
    if _stage_is_complete(output_root, config, "summary"):
        return _read_json(output_root / STAGE_DIRS["summary"] / "gate.json")
    return stage_summary(config, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-shard-worker", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    output_root = Path(args.output_root).resolve()
    if args.audit_shard_worker is not None:
        run_audit_shard_worker(config, project_root_from(SOURCE_ROOT), output_root, args.audit_shard_worker)
        return 0
    gate = run(config, output_root)
    print(json.dumps(gate, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
