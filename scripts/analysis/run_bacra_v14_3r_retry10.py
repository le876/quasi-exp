#!/usr/bin/env python3
"""Run BACRA V14.3R retry10 frontier-expanded atlas and staged datasets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

import run_bacra_v14_3_repaired_5k_student as retry8
import run_bacra_v14_3r_retry9 as retry9
from quasi_exp.teacher.canonical_atlas import AtlasCandidate, AtlasTaskNode
from quasi_exp.teacher.experiment import sha256_file
from quasi_exp.teacher.exploration_qualification import (
    BETA_COLUMNS,
    bounded_fundamental_cycles,
    percentile,
)
from quasi_exp.teacher.partial_relay import choose_macroblock_split
from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.retry10 import (
    METRIC_VERSION,
    XYZ_COLUMNS,
    classify_frontier_taxonomy,
    compatible_candidate_medoid,
    component_boundary_audit,
    dataset_generation_decision,
    deterministic_paired_bootstrap_delta,
    dynamic_frontier_root_registry,
    k2_retention_decision,
    macroblock_ids,
    normalized_weighted_beta_deg,
    preflight_generation_decision,
    raw_beta_max_deg,
    student_quality_decision,
)
from quasi_exp.teacher.retry10_sampling import (
    exact_nested_dataset,
    parent_domain_target_registry,
)
from quasi_exp.teacher.section_first_atlas import (
    RootedSectionPolicy,
    build_section_first_atlas,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry
from quasi_exp.teacher.workspace_atlas import RepresentationMode
from quasi_exp.teacher.workspace_student import (
    WorkspaceStudentTrainingConfig,
    save_workspace_student_models,
    train_workspace_student,
)
from quasi_exp.teacher.optimized_continuation import (
    make_optimized_predictor_corrector_continuation,
)
from quasi_exp.model.sampling import beta_to_theta
from quasi_exp.teacher.workspace_atlas_repair import (
    atlas_nodes_from_frames,
    build_shared_face_task_edges,
)


STAGE_DIRS = {
    "inventory": "00_inventory",
    "atlas_poc": "01_atlas_poc",
    "seed_datasets": "02_seed_datasets",
    "seed_students": "03_seed_students",
    "atlas_expansion": "04_atlas_expansion",
    "taxonomy_rescue": "05_taxonomy_rescue",
    "atlas_freeze": "06_atlas_freeze_hard_audit",
    "full_diagnostic": "07_full_diagnostic",
    "preflight_50k": "08_dataset_50k_preflight",
    "dataset_50k": "09_dataset_50k",
    "student_50k": "10_student_50k",
    "tension_pilot": "11_tension_pilot",
    "preflight_100k": "12_dataset_100k_preflight",
    "dataset_100k": "13_dataset_100k",
    "student_100k": "14_student_100k",
    "preflight_200k": "15_dataset_200k_preflight",
    "dataset_200k": "16_dataset_200k",
    "student_200k": "17_student_200k",
    "tension_materialization": "18_tension_materialization",
    "summary": "19_summary_report",
}
STAGE_ORDER = tuple(STAGE_DIRS)


def project_root_from(source_root: Path) -> Path:
    return retry9.project_root_from(source_root)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset | tuple):
        return list(value)
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
        raise ValueError("unsupported retry10 config")
    config = dict(value)
    config["config_path"] = str(config_path)
    if str(config.get("experiment_id")) != "bacra_v14_3r_retry10_frontier_expanded_atlas":
        raise ValueError("retry10 experiment identity mismatch")
    if str(config["metric"]["version"]) != METRIC_VERSION:
        raise ValueError("retry10 requires normalized_weighted_v1")
    if tuple(map(float, config["metric"]["beta_coordinate_weights"])) != (
        4.0, 4.0, 2.0, 2.0, 1.0, 1.0
    ):
        raise ValueError("retry10 beta weights are frozen")
    if int(config["runtime"]["maximum_concurrent_workers"]) != 12:
        raise ValueError("retry10 requires twelve numerical workers")
    if int(config["runtime"]["logical_shard_count"]) != 48:
        raise ValueError("retry10 requires 48 logical shards")
    if str(config["runtime"]["device"]) != "cpu":
        raise ValueError("retry10 requires CPU execution")
    if any(
        bool(config["claims"][key])
        for key in ("formal_authorized", "deployment_authorized", "full_workspace_authorized")
    ):
        raise ValueError("retry10 cannot authorize Formal, deployment, or full workspace")
    return config


def _upstream_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["upstream"]["root"]))


def _upstream_path(config: Mapping[str, Any], key: str) -> Path:
    row = config["upstream"][key]
    return _upstream_root(config) / str(row["path"])


def _stage_upstream_digest(output_root: Path, stage_name: str) -> str:
    digest = hashlib.sha256()
    for name in STAGE_ORDER[: STAGE_ORDER.index(stage_name)]:
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
    artifacts = [
        {
            "path": str(path.relative_to(stage)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    ]
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
    manifest_path = stage / "completion_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        return bool(
            manifest["stage_name"] == stage_name
            and manifest["scientific_source_fixed_point"] == _git_sha()
            and manifest["config_sha256"] == _config_sha(config)
            and manifest["upstream_completion_sha256"]
            == _stage_upstream_digest(output_root, stage_name)
            and all(
                (stage / row["path"]).is_file()
                and sha256_file(stage / row["path"]) == row["sha256"]
                for row in manifest["artifacts"]
            )
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _seal_gate(
    output_root: Path,
    config: Mapping[str, Any],
    stage_name: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    stage = output_root / STAGE_DIRS[stage_name]
    _write_json(stage / "gate.json", dict(gate))
    _seal_stage(output_root, config, stage_name)
    return dict(gate)


def _gate(output_root: Path, stage_name: str) -> dict[str, Any]:
    return _read_json(output_root / STAGE_DIRS[stage_name] / "gate.json")


def _environment(config: Mapping[str, Any]) -> Any:
    return retry9._environment(config)


def _source_frames(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(_upstream_path(config, "atlas_labels"))
    tasks = pd.read_parquet(_upstream_path(config, "task_nodes"))
    edges = pd.read_parquet(_upstream_path(config, "task_edges"))
    return labels, tasks, edges


def stage_inventory(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["inventory"]
    source_rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    consultation = SOURCE_ROOT / str(config["sources"]["consultation_input"])
    protocol = SOURCE_ROOT / str(config["sources"]["governing_protocol"])
    checks["consultation_sha256"] = (
        consultation.is_file()
        and sha256_file(consultation) == str(config["consultation_sha256"])
    )
    checks["governing_protocol_exists"] = protocol.is_file()
    for key, row in config["upstream"].items():
        if not isinstance(row, Mapping) or "path" not in row or "sha256" not in row:
            continue
        path = _upstream_root(config) / str(row["path"])
        observed = sha256_file(path) if path.is_file() else None
        checks[f"upstream_{key}_sha256"] = observed == str(row["sha256"])
        source_rows.append(
            {
                "source_id": key,
                "path": str(path),
                "expected_sha256": str(row["sha256"]),
                "observed_sha256": observed,
                "pass": observed == str(row["sha256"]),
            }
        )
    labels, tasks, edges = _source_frames(config)
    checks["atlas_unique"] = not labels["task_node_id"].duplicated().any()
    checks["task_denominator"] = len(tasks) == int(config["atlas"]["denominator_task_nodes"])
    checks["registered_parent_domain"] = "source_parent_node_id" in tasks
    checks["task_graph_nonempty"] = len(edges) > 0
    gate = {
        "status": "pass" if all(checks.values()) else "fail",
        "gate_pass": bool(all(checks.values())),
        "checks": checks,
        "upstream_atlas_rows": int(len(labels)),
        "task_node_count": int(len(tasks)),
        "task_edge_count": int(len(edges)),
        "metric_version": METRIC_VERSION,
        "formal_authorized": False,
        "deployment_authorized": False,
    }
    _write_parquet(pd.DataFrame.from_records(source_rows), stage / "source_inventory.parquet")
    return _seal_gate(output_root, config, "inventory", gate)


def _finite_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if math.isfinite(parsed) else float(fallback)


def _normalize_candidate_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill optional ranking metadata without masking invalid physical labels."""

    normalized = frame.copy()
    beta = normalized.loc[:, BETA_COLUMNS].to_numpy(float)
    posture_fallback = np.linalg.norm(beta, axis=1)

    def finite_column(name: str, fallback: float | np.ndarray) -> np.ndarray:
        if name in normalized:
            values = pd.to_numeric(normalized[name], errors="coerce").to_numpy(float)
        else:
            values = np.full(len(normalized), np.nan, dtype=float)
        fallback_values = np.broadcast_to(np.asarray(fallback, dtype=float), values.shape)
        return np.where(np.isfinite(values), values, fallback_values)

    normalized["residual_mm"] = finite_column("residual_mm", 0.0)
    normalized["min_margin_deg"] = finite_column("min_margin_deg", 1.0)
    normalized["normalized_min_margin"] = finite_column(
        "normalized_min_margin",
        np.maximum(normalized["min_margin_deg"].to_numpy(float), 0.0) / 180.0,
    )
    normalized["posture_cost"] = finite_column("posture_cost", posture_fallback)
    normalized["condition_number"] = finite_column("condition_number", 1.0)
    return normalized


def _atlas_candidate(row: Any, *, node_id: int | None = None) -> AtlasCandidate:
    active_node_id = int(row.task_node_id if node_id is None else node_id)
    beta_rad = np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float)
    posture_fallback = float(np.linalg.norm(beta_rad))
    residual = getattr(row, "source_teacher_residual_mm", None)
    if not math.isfinite(_finite_float(residual, math.nan)):
        residual = getattr(row, "residual_mm", 0.0)
    min_margin = _finite_float(getattr(row, "min_margin_deg", 1.0), 1.0)
    quality = getattr(row, "quality", None)
    if str(quality).lower() not in {"gold", "silver", "reject"}:
        quality = getattr(row, "label_quality", "Gold")
    if str(quality).lower() not in {"gold", "silver", "reject"}:
        quality = "Gold"
    return AtlasCandidate(
        node_id=active_node_id,
        candidate_id=str(getattr(row, "candidate_id", f"retry10_root_{active_node_id}")),
        beta_rad=beta_rad,
        residual_mm=_finite_float(residual, 0.0),
        min_margin_deg=min_margin,
        normalized_min_margin=_finite_float(
            getattr(row, "normalized_min_margin", math.nan),
            max(min_margin, 0.0) / 180.0,
        ),
        posture_cost=_finite_float(getattr(row, "posture_cost", math.nan), posture_fallback),
        condition_number=_finite_float(getattr(row, "condition_number", 1.0), 1.0),
        quality=str(quality),
        solver_success=True,
        actual_bounds=True,
    )


def _section_policy(
    config: Mapping[str, Any], *, roots: int, waves: int, beam: int
) -> RootedSectionPolicy:
    atlas = config["atlas"]
    return RootedSectionPolicy(
        beam_width=int(beam),
        root_count=int(roots),
        maximum_growth_waves=int(waves),
        parent_consensus_gold_deg=float(atlas["weighted_local_p95_max_deg"]),
        parent_consensus_silver_deg=float(atlas["raw_catastrophic_deg"]),
        continuation_residual_max_mm=float(atlas["individual_fk_max_mm"]),
        reverse_return_max_deg=float(config["sampling"]["silver"]["reverse_return_max_deg"]),
        beta_weights=tuple(map(float, config["metric"]["beta_coordinate_weights"])),
        metric_version=METRIC_VERSION,
    )


def _growth_frames(
    config: Mapping[str, Any],
    labels: pd.DataFrame,
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    roots: pd.DataFrame,
    *,
    waves: int,
    beam: int,
) -> tuple[dict[str, pd.DataFrame], Any]:
    environment = _environment(config)
    nodes, continuation = retry8._segmented_continuation(environment, tasks, edges)
    candidates = tuple(
        _atlas_candidate(next(labels[labels["task_node_id"].astype(int).eq(int(node_id))].itertuples(index=False)))
        for node_id in roots["task_node_id"].astype(int)
    )
    growth = build_section_first_atlas(
        nodes,
        candidates,
        continuation,
        root_keys=tuple((item.node_id, item.candidate_id) for item in candidates),
        policy=_section_policy(config, roots=len(candidates), waves=waves, beam=beam),
    )
    return growth.frames(), environment


def _pair_overlap_audit(candidates: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if candidates.empty:
        return pd.DataFrame()
    charts = sorted(set(candidates["chart_id"].astype(str)))
    for left_index, left_chart in enumerate(charts):
        left = candidates[candidates["chart_id"].astype(str).eq(left_chart)].set_index("task_node_id")
        for right_chart in charts[left_index + 1 :]:
            right = candidates[candidates["chart_id"].astype(str).eq(right_chart)].set_index("task_node_id")
            common = sorted(set(left.index.astype(int)) & set(right.index.astype(int)))
            weighted = [
                normalized_weighted_beta_deg(
                    left.loc[node, list(BETA_COLUMNS)].to_numpy(float),
                    right.loc[node, list(BETA_COLUMNS)].to_numpy(float),
                )
                for node in common
            ]
            raw = [
                raw_beta_max_deg(
                    left.loc[node, list(BETA_COLUMNS)].to_numpy(float),
                    right.loc[node, list(BETA_COLUMNS)].to_numpy(float),
                )
                for node in common
            ]
            records.append(
                {
                    "chart_a": left_chart,
                    "chart_b": right_chart,
                    "overlap_node_count": len(common),
                    "weighted_gap_p95_deg": percentile(weighted, 95),
                    "raw_gap_max_deg": max(raw, default=0.0),
                    "new_new_transition_audit_required": len(common) >= 8,
                    "compatibility_pass": bool(
                        not raw or (percentile(weighted, 95) <= 2.0 and max(raw) <= 5.0)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _normalize_existing_labels(labels: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    frame = labels.copy()
    if "source_parent_node_id" not in frame:
        frame = frame.merge(
            tasks[["task_node_id", "source_parent_node_id"]],
            on="task_node_id",
            how="left",
            validate="one_to_one",
        )
    frame["component_id"] = frame.get("component_id", "retry9_primary")
    frame["label_origin"] = frame.get("label_origin", "retry9_existing_atlas")
    frame["label_quality"] = frame.get("label_quality", frame.get("quality", "Gold"))
    frame["actual_bounds"] = frame.get("actual_bounds", frame.get("source_teacher_actual_bounds", True))
    frame["teacher_fk_residual_mm"] = frame.get(
        "teacher_fk_residual_mm", frame.get("source_teacher_residual_mm", frame.get("residual_mm", 0.0))
    )
    return frame


def _merge_growth(
    config: Mapping[str, Any],
    labels: pd.DataFrame,
    tasks: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    environment: Any,
    *,
    origin: str,
    allowed_node_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hypotheses = frames.get("section_hypotheses", pd.DataFrame()).copy()
    if hypotheses.empty:
        return labels.copy(), pd.DataFrame(), pd.DataFrame()
    if "selected" in hypotheses:
        hypotheses = hypotheses[hypotheses["selected"].astype(bool)]
    covered = set(labels["task_node_id"].astype(int))
    hypotheses = hypotheses[~hypotheses["task_node_id"].astype(int).isin(covered)].copy()
    if allowed_node_ids is not None:
        hypotheses = hypotheses[hypotheses["task_node_id"].astype(int).isin(allowed_node_ids)]
    if hypotheses.empty:
        return labels.copy(), pd.DataFrame(), pd.DataFrame()
    task_lookup = tasks.set_index("task_node_id")
    for name in (*XYZ_COLUMNS, "source_parent_node_id"):
        if name not in hypotheses:
            hypotheses[name] = hypotheses["task_node_id"].map(task_lookup[name])
    if "chart_id" not in hypotheses:
        hypotheses["chart_id"] = hypotheses.get("root_node_id", "retry10_chart").map(
            lambda value: f"retry10_chart_{value}"
        )
    if "candidate_id" not in hypotheses:
        hypotheses["candidate_id"] = [f"{origin}_{index}" for index in range(len(hypotheses))]
    candidates = hypotheses[
        ["task_node_id", "chart_id", "candidate_id", *XYZ_COLUMNS, "source_parent_node_id", *BETA_COLUMNS]
        + [name for name in ("residual_mm", "min_margin_deg", "normalized_min_margin", "condition_number", "actual_bounds") if name in hypotheses]
    ].copy()
    candidates = _normalize_candidate_metadata(candidates)
    selected, compatibility = compatible_candidate_medoid(
        candidates,
        weighted_max_deg=float(config["atlas"]["weighted_local_p95_max_deg"]),
        raw_max_deg=float(config["atlas"]["raw_catastrophic_deg"]),
    )
    if selected.empty:
        return labels.copy(), candidates, compatibility
    beta = selected.loc[:, BETA_COLUMNS].to_numpy(float)
    xyz = selected.loc[:, XYZ_COLUMNS].to_numpy(float)
    fk = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    selected["teacher_fk_residual_mm"] = np.linalg.norm(fk - xyz, axis=1) * 1000.0
    bounds = np.asarray(environment.bounds, dtype=float)
    selected["actual_bounds"] = np.all(
        (beta >= bounds[:, 0] - 1.0e-12) & (beta <= bounds[:, 1] + 1.0e-12), axis=1
    )
    selected = selected[
        selected["actual_bounds"].astype(bool)
        & selected["teacher_fk_residual_mm"].le(float(config["atlas"]["individual_fk_max_mm"]))
    ].copy()
    selected["component_id"] = origin
    selected["label_origin"] = origin
    selected["label_quality"] = "Gold"
    selected["quality"] = "Gold"
    merged = pd.concat([labels, selected], ignore_index=True, sort=False)
    merged = merged.drop_duplicates("task_node_id", keep="first").reset_index(drop=True)
    return merged, candidates, compatibility


def _run_atlas_batch(
    config: Mapping[str, Any],
    labels: pd.DataFrame,
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    batch_index: int,
    prior_roots: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    roots = dynamic_frontier_root_registry(
        tasks,
        edges,
        labels,
        prior_root_node_ids=prior_roots,
        root_count=int(config["atlas"]["root_count_per_batch"]),
        hops=int(config["atlas"]["dynamic_root_hops"]),
    )
    if roots.empty:
        return labels.copy(), {"roots": roots}, {"status": "exhausted", "coverage_gain": 0.0}
    before = len(labels)
    frames, environment = _growth_frames(
        config,
        labels,
        tasks,
        edges,
        roots,
        waves=int(config["atlas"]["k1_initial_waves"]),
        beam=int(config["atlas"]["k1_beam_width"]),
    )
    merged, candidates, compatibility = _merge_growth(
        config,
        labels,
        tasks,
        frames,
        environment,
        origin=f"retry10_batch_{batch_index:02d}_k1",
    )
    events = frames.get("growth_events", frames.get("events", pd.DataFrame()))
    active = bool(
        not events.empty
        and any(
            token in str(value).lower()
            for value in events.get("reason", pd.Series(dtype=str))
            for token in ("frontier", "wave", "cap")
        )
    )
    if active and int(config["atlas"]["k1_extended_waves"]) > int(config["atlas"]["k1_initial_waves"]):
        extended_frames, environment = _growth_frames(
            config,
            merged,
            tasks,
            edges,
            roots,
            waves=int(config["atlas"]["k1_extended_waves"]),
            beam=1,
        )
        merged, extended_candidates, extended_compatibility = _merge_growth(
            config,
            merged,
            tasks,
            extended_frames,
            environment,
            origin=f"retry10_batch_{batch_index:02d}_k1_extended",
        )
        candidates = pd.concat([candidates, extended_candidates], ignore_index=True, sort=False)
        compatibility = pd.concat([compatibility, extended_compatibility], ignore_index=True, sort=False)
        frames = extended_frames
        events = frames.get("growth_events", frames.get("events", events))
    cap_nodes: set[int] = set()
    cap_events = frames.get("cap_hit_events", pd.DataFrame())
    if not cap_events.empty:
        cap_column = next(
            (name for name in ("node_id", "target_node_id", "task_node_id") if name in cap_events),
            None,
        )
        if cap_column is not None:
            cap_nodes.update(cap_events[cap_column].astype(int))
    if not events.empty:
        reason = events.get("reason", pd.Series("", index=events.index)).astype(str).str.lower()
        node_column = "target_node_id" if "target_node_id" in events else "task_node_id"
        if node_column in events:
            cap_nodes = set(events.loc[reason.str.contains("cap"), node_column].astype(int))
    k2 = k2_retention_decision(recovered_node_count=0)
    if cap_nodes:
        k2_frames, environment = _growth_frames(
            config,
            merged,
            tasks,
            edges,
            roots,
            waves=int(config["atlas"]["k1_extended_waves"]),
            beam=int(config["atlas"]["k2_beam_width"]),
        )
        adjacency = {node: set() for node in tasks["task_node_id"].astype(int)}
        for edge in edges.itertuples(index=False):
            adjacency[int(edge.left_node_id)].add(int(edge.right_node_id))
            adjacency[int(edge.right_node_id)].add(int(edge.left_node_id))
        allowed = cap_nodes | {neighbor for node in cap_nodes for neighbor in adjacency.get(node, set())}
        before_k2 = len(merged)
        merged, k2_candidates, k2_compatibility = _merge_growth(
            config,
            merged,
            tasks,
            k2_frames,
            environment,
            origin=f"retry10_batch_{batch_index:02d}_k2",
            allowed_node_ids=allowed,
        )
        candidates = pd.concat([candidates, k2_candidates], ignore_index=True, sort=False)
        compatibility = pd.concat([compatibility, k2_compatibility], ignore_index=True, sort=False)
        k2 = k2_retention_decision(recovered_node_count=len(merged) - before_k2)
    denominator = int(config["atlas"]["denominator_task_nodes"])
    report = {
        "status": "complete",
        "batch_index": int(batch_index),
        "root_node_ids": roots["task_node_id"].astype(int).tolist(),
        "coverage_before": before / denominator,
        "coverage_after": len(merged) / denominator,
        "coverage_gain": (len(merged) - before) / denominator,
        "new_label_count": len(merged) - before,
        **k2,
    }
    return merged, {
        "roots": roots,
        "candidates": candidates,
        "compatibility": compatibility,
        "new_new_pair_audit": _pair_overlap_audit(candidates),
        **frames,
    }, report


def _write_batch_frames(stage: Path, batch_index: int, frames: Mapping[str, pd.DataFrame]) -> None:
    directory = stage / f"batch_{batch_index:02d}"
    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame):
            _write_parquet(frame, directory / f"{name}.parquet")


def stage_atlas_poc(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    if not _gate(output_root, "inventory")["gate_pass"]:
        return _seal_gate(output_root, config, "atlas_poc", {"status": "fail", "gate_pass": False, "reason": "inventory_failed"})
    stage = output_root / STAGE_DIRS["atlas_poc"]
    labels, tasks, edges = _source_frames(config)
    labels = _normalize_existing_labels(labels, tasks)
    merged, frames, report = _run_atlas_batch(
        config, labels, tasks, edges, batch_index=1, prior_roots=()
    )
    _write_batch_frames(stage, 1, frames)
    _write_parquet(merged, stage / "candidate_atlas_after_poc.parquet")
    _write_json(stage / "batch_report.json", report)
    hard_conflicts = int(
        frames.get("compatibility", pd.DataFrame()).get(
            "new_node_branch_conflict", pd.Series(dtype=bool)
        ).astype(bool).sum()
    )
    gate = {
        "status": "pass",
        "gate_pass": True,
        "poc_executed": True,
        "root_count": len(frames.get("roots", ())),
        "coverage_fraction": len(merged) / int(config["atlas"]["denominator_task_nodes"]),
        "coverage_gain": float(report["coverage_gain"]),
        "new_label_count": int(report.get("new_label_count", 0)),
        "branch_conflict_node_count": hard_conflicts,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "atlas_poc", gate)


def _dataset_seed_rows(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels.copy().reset_index(drop=True)
    frame["physical_point_id"] = frame["task_node_id"].map(
        lambda value: f"retry9_task_{int(value):06d}"
    )
    frame["xyz_key"] = frame.loc[:, XYZ_COLUMNS].round(12).astype(str).agg("|".join, axis=1)
    frame["label_quality"] = "Gold"
    frame["sampling_stratum"] = frame["source_parent_node_id"].map(
        lambda value: f"parent:{int(value)}:seed"
    )
    frame["parent_node_id_1"] = frame["task_node_id"].astype(int)
    frame["parent_node_id_2"] = frame["task_node_id"].astype(int)
    frame["reverse_return_max_deg"] = 0.0
    frame["multiparent_weighted_gap_deg"] = 0.0
    frame["multiparent_raw_gap_deg"] = 0.0
    frame["actual_bounds"] = frame.get("actual_bounds", True)
    frame["teacher_fk_residual_mm"] = frame.get(
        "teacher_fk_residual_mm", frame.get("source_teacher_residual_mm", 0.0)
    )
    frame["sampling_origin"] = "retry9_frozen_atlas"
    return _attach_theta(frame)


def _stable_point_id(xyz_key: str) -> str:
    return f"retry10_point_{hashlib.sha256(str(xyz_key).encode()).hexdigest()[:24]}"


def _attach_theta(frame: pd.DataFrame, *, theta_sign: float = -1.0) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    beta = result.loc[:, BETA_COLUMNS].to_numpy(float)
    theta = np.vstack([beta_to_theta(row) for row in beta]) * float(theta_sign)
    for index in range(theta.shape[1]):
        result[f"theta{index + 1}_rad"] = theta[:, index]
    return result


def _label_worker(
    config: Mapping[str, Any], worker_root: Path, shard_id: int
) -> dict[str, Any]:
    targets = pd.read_parquet(worker_root / "target_registry.parquet")
    targets = targets[targets["shard_id"].astype(int).eq(int(shard_id))]
    sources = pd.read_parquet(worker_root / "source_labels.parquet")
    indexed = sources.drop_duplicates("task_node_id").set_index("task_node_id")
    environment = _environment(config)
    continuation = make_optimized_predictor_corrector_continuation(
        environment,
        damping=2.0e-3,
        max_corrector_iterations=400,
        residual_tolerance_mm=float(config["atlas"]["individual_fk_max_mm"]),
    )
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        target_node = AtlasTaskNode(
            int(target.target_id),
            np.asarray([target.x_m, target.y_m, target.z_m], dtype=float),
            (),
        )
        outcomes: list[dict[str, Any]] = []
        for source_id in (int(target.source_node_1), int(target.source_node_2)):
            source_row = indexed.loc[source_id]
            source_candidate = _atlas_candidate(
                next(sources[sources["task_node_id"].astype(int).eq(source_id)].itertuples(index=False))
            )
            forward = continuation(source_candidate, target_node)
            if not (
                forward.success
                and forward.actual_bounds
                and np.isfinite(forward.beta_rad).all()
                and float(forward.residual_mm) <= float(config["atlas"]["individual_fk_max_mm"])
            ):
                continue
            endpoint = AtlasCandidate(
                node_id=int(target.target_id),
                candidate_id=f"target_{int(target.target_id)}_from_{source_id}",
                beta_rad=np.asarray(forward.beta_rad, dtype=float),
                residual_mm=float(forward.residual_mm),
                min_margin_deg=float(forward.minimum_margin_deg or 1.0),
                normalized_min_margin=max(0.0, float(forward.minimum_margin_deg or 1.0)) / 180.0,
                condition_number=1.0,
                quality="Gold",
                solver_success=True,
                actual_bounds=True,
            )
            reverse_node = AtlasTaskNode(
                source_id,
                source_row.loc[list(XYZ_COLUMNS)].to_numpy(float),
                (),
            )
            reverse = continuation(endpoint, reverse_node)
            reverse_gap = (
                normalized_weighted_beta_deg(reverse.beta_rad, source_candidate.beta_rad)
                if reverse.success and reverse.actual_bounds
                else math.inf
            )
            outcomes.append(
                {
                    "source_id": source_id,
                    "beta": np.asarray(forward.beta_rad, dtype=float),
                    "residual_mm": float(forward.residual_mm),
                    "reverse_gap": reverse_gap,
                }
            )
        weighted_gap = (
            normalized_weighted_beta_deg(outcomes[0]["beta"], outcomes[1]["beta"])
            if len(outcomes) >= 2
            else math.inf
        )
        raw_gap = (
            raw_beta_max_deg(outcomes[0]["beta"], outcomes[1]["beta"])
            if len(outcomes) >= 2
            else math.inf
        )
        quality: str | None = None
        selected: dict[str, Any] | None = None
        if (
            len(outcomes) >= 2
            and weighted_gap <= float(config["sampling"]["silver"]["weighted_neighbor_gap_max_deg"])
            and raw_gap <= float(config["sampling"]["silver"]["raw_neighbor_gap_max_deg"])
        ):
            quality = "Gold"
            selected = min(outcomes, key=lambda row: (row["residual_mm"], row["source_id"]))
        elif outcomes:
            selected = min(outcomes, key=lambda row: (row["residual_mm"], row["source_id"]))
            neighbor_gaps = []
            neighbor_raw = []
            for source_id in (int(target.source_node_1), int(target.source_node_2)):
                source_beta = indexed.loc[source_id, list(BETA_COLUMNS)].to_numpy(float)
                neighbor_gaps.append(normalized_weighted_beta_deg(selected["beta"], source_beta))
                neighbor_raw.append(raw_beta_max_deg(selected["beta"], source_beta))
            if (
                selected["reverse_gap"] <= float(config["sampling"]["silver"]["reverse_return_max_deg"])
                and max(neighbor_gaps) <= float(config["sampling"]["silver"]["weighted_neighbor_gap_max_deg"])
                and max(neighbor_raw) <= float(config["sampling"]["silver"]["raw_neighbor_gap_max_deg"])
            ):
                quality = "Silver"
        record: dict[str, Any] = {
            "target_id": int(target.target_id),
            "physical_point_id": _stable_point_id(str(target.xyz_key)),
            "source_parent_node_id": int(target.source_parent_node_id),
            "parent_node_id_1": int(target.source_node_1),
            "parent_node_id_2": int(target.source_node_2),
            "sampling_stratum": str(target.sampling_stratum),
            "xyz_key": str(target.xyz_key),
            "x_m": float(target.x_m),
            "y_m": float(target.y_m),
            "z_m": float(target.z_m),
            "accepted": quality is not None,
            "label_quality": quality,
            "solver_success_count": len(outcomes),
            "multiparent_weighted_gap_deg": weighted_gap,
            "multiparent_raw_gap_deg": raw_gap,
            "reverse_return_max_deg": max(
                (float(row["reverse_gap"]) for row in outcomes), default=math.inf
            ),
        }
        if quality is not None and selected is not None:
            record.update(
                {
                    **{
                        column: float(selected["beta"][index])
                        for index, column in enumerate(BETA_COLUMNS)
                    },
                    "teacher_fk_residual_mm": float(selected["residual_mm"]),
                    "actual_bounds": True,
                    "sampling_origin": "retry10_parent_domain_continuation",
                }
            )
        rows.append(record)
    frame = pd.DataFrame.from_records(rows)
    _write_parquet(frame, worker_root / f"shard_{int(shard_id):02d}" / "attempts.parquet")
    report = {
        "shard_id": int(shard_id),
        "target_count": len(targets),
        "accepted_count": int(frame.get("accepted", pd.Series(dtype=bool)).astype(bool).sum()),
    }
    _write_json(worker_root / f"shard_{int(shard_id):02d}" / "report.json", report)
    return report


def _thread_limited_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }
    )
    return environment


def _run_label_registry(
    config: Mapping[str, Any],
    directory: Path,
    sources: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    _write_parquet(sources, directory / "source_labels.parquet")
    _write_parquet(targets, directory / "target_registry.parquet")
    shard_count = int(config["runtime"]["logical_shard_count"])
    pending = list(range(shard_count))
    running: list[tuple[int, subprocess.Popen[str], Any]] = []
    while pending or running:
        while pending and len(running) < int(config["runtime"]["maximum_concurrent_workers"]):
            shard_id = pending.pop(0)
            shard = directory / f"shard_{shard_id:02d}"
            shard.mkdir(parents=True, exist_ok=True)
            handle = (shard / "worker.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config["config_path"]),
                    "--label-worker-root",
                    str(directory),
                    "--shard-id",
                    str(shard_id),
                ],
                cwd=SOURCE_ROOT,
                env=_thread_limited_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((shard_id, process, handle))
        survivors: list[tuple[int, subprocess.Popen[str], Any]] = []
        for shard_id, process, handle in running:
            if process.poll() is None:
                survivors.append((shard_id, process, handle))
                continue
            handle.close()
            if process.returncode != 0:
                raise RuntimeError(f"retry10 label worker failed: shard {shard_id}")
        running = survivors
        if running:
            time.sleep(0.2)
    frames = [
        pd.read_parquet(directory / f"shard_{shard_id:02d}" / "attempts.parquet")
        for shard_id in range(shard_count)
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _dataset_integrity(frame: pd.DataFrame, target_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            "exact_target_pass": False,
            "unique_xyz_pass": False,
            "finite_pass": False,
            "bounds_pass": False,
            "physical_point_id_unique_pass": False,
            "duplicate_count": 0,
        }
    duplicate_count = int(frame["xyz_key"].duplicated().sum())
    finite = np.isfinite(frame.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy(float)).all()
    return {
        "exact_target_pass": len(frame) == int(target_rows),
        "unique_xyz_pass": duplicate_count == 0,
        "finite_pass": bool(finite),
        "bounds_pass": bool(frame["actual_bounds"].astype(bool).all()),
        "physical_point_id_unique_pass": bool(
            "physical_point_id" in frame
            and not frame["physical_point_id"].astype(str).duplicated().any()
        ),
        "duplicate_count": duplicate_count,
    }


def stage_seed_datasets(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["seed_datasets"]
    labels, tasks, _edges = _source_frames(config)
    sources = _normalize_existing_labels(labels, tasks)
    seed = _dataset_seed_rows(sources)
    _write_parquet(seed, stage / "dataset_5105.parquet")
    scale_reports: dict[str, Any] = {}
    current = seed
    for target_rows in (10_000, 20_000):
        required = max(0, target_rows - len(current))
        targets = parent_domain_target_registry(
            sources,
            target_count=int(config["sampling"]["attempt_multiplier"]) * required,
            shard_count=int(config["runtime"]["logical_shard_count"]),
            seed=int(config["runtime"]["seed"]) + target_rows,
            excluded_xyz_keys=current["xyz_key"].astype(str),
        )
        attempts = _run_label_registry(config, stage / f"generation_{target_rows}", sources, targets)
        accepted = attempts[attempts["accepted"].astype(bool)].copy()
        accepted = _attach_theta(accepted)
        nested = exact_nested_dataset(
            current,
            accepted,
            target_rows=target_rows,
            allowed_qualities=("Gold", "Silver"),
        )
        integrity = _dataset_integrity(nested, target_rows)
        _write_parquet(attempts, stage / f"generation_attempts_{target_rows}.parquet")
        _write_parquet(nested, stage / f"dataset_{target_rows}.parquet")
        scale_reports[str(target_rows)] = {
            **integrity,
            "row_count": len(nested),
            "attempt_count": len(attempts),
            "accepted_count": len(accepted),
            "gold_count": int(nested["label_quality"].eq("Gold").sum()),
            "silver_count": int(nested["label_quality"].eq("Silver").sum()),
        }
        current = nested
    def scale_integrity_pass(report: Mapping[str, Any]) -> bool:
        return bool(
            report["exact_target_pass"]
            and report["unique_xyz_pass"]
            and report["finite_pass"]
            and report["bounds_pass"]
            and report["physical_point_id_unique_pass"]
        )

    seed_10k = scale_integrity_pass(scale_reports["10000"])
    seed_20k = scale_integrity_pass(scale_reports["20000"])
    gate = {
        "status": "pass" if seed_10k else "fail",
        "gate_pass": seed_10k,
        "dataset_20k_authorized": seed_10k,
        "seed_10k_target_pass": seed_10k,
        "seed_20k_target_pass": seed_20k,
        "maximum_legal_seed_rows": len(current),
        "scales": scale_reports,
        "row_padding_used": False,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "seed_datasets", gate)


def _student_geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=np.asarray(environment.lengths_m, dtype=np.float32),
        p_end_local_m=np.asarray(environment.p_end_local_m, dtype=np.float32),
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=np.asarray(environment.bounds, dtype=np.float32),
    )


def _prepare_student_frame(
    frame: pd.DataFrame,
    environment: Any,
    *,
    panel_macroblocks: set[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = frame.copy().reset_index(drop=True)
    prepared["macroblock_id"] = macroblock_ids(prepared, block_size_mm=40)
    available = prepared[~prepared["macroblock_id"].astype(str).isin(panel_macroblocks)].copy()
    split = choose_macroblock_split(
        available,
        block_sizes_mm=(40,),
        split_seed=int(seed),
        split_fractions=(0.70, 0.20, 0.10),
        minimum_train_blocks=3,
        minimum_validation_blocks=1,
        minimum_test_blocks=1,
        minimum_rows_per_split=20,
    )
    prepared = split.frame.copy()
    jacobians = [
        np.asarray(environment.jacobian(beta), dtype=float).reshape(3, 6).reshape(-1)
        for beta in prepared.loc[:, BETA_COLUMNS].to_numpy(float)
    ]
    jacobian_array = np.asarray(jacobians)
    for index, column in enumerate(JACOBIAN_COLUMNS):
        prepared[column] = jacobian_array[:, index]
    xyz_keys = prepared.get(
        "xyz_key",
        prepared.loc[:, XYZ_COLUMNS].round(12).astype(str).agg("|".join, axis=1),
    )
    prepared["record_id"] = xyz_keys.astype(str).map(_stable_point_id)
    prepared["kind"] = "static"
    prepared["chart_id"] = "retry10_global"
    prepared["is_primary"] = True
    prepared["sample_weight"] = 1.0
    train = prepared[prepared["split_role"].eq("train_core")].copy()
    validation = prepared[prepared["split_role"].eq("validation")].copy()
    if validation.empty:
        raise RuntimeError("retry10 Student validation split is empty")
    return prepared, train, validation


def _evaluate_model_on_panel(
    model: Any,
    panel: pd.DataFrame,
    environment: Any,
    *,
    panel_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if panel.empty:
        return pd.DataFrame(), {
            "panel_id": panel_id,
            "status": "underfilled",
            "row_count": 0,
        }
    xyz = panel.loc[:, XYZ_COLUMNS].to_numpy(np.float32)
    truth = panel.loc[:, BETA_COLUMNS].to_numpy(float)
    started = time.perf_counter()
    prediction = np.asarray(model(xyz, training=False), dtype=float)
    raw_latency = (time.perf_counter() - started) / max(1, len(panel))
    raw_fk = (
        np.linalg.norm(np.asarray(environment.fk(prediction)).reshape(-1, 3) - xyz, axis=1)
        * 1000.0
    )
    started = time.perf_counter()
    corrected = retry9._two_step_dls(environment, prediction, xyz)
    dls_latency = (time.perf_counter() - started) / max(1, len(panel))
    dls_fk = (
        np.linalg.norm(np.asarray(environment.fk(corrected)).reshape(-1, 3) - xyz, axis=1)
        * 1000.0
    )
    bounds = np.asarray(environment.bounds, dtype=float)
    violation = ~np.all(
        (prediction >= bounds[:, 0] - 1.0e-12)
        & (prediction <= bounds[:, 1] + 1.0e-12),
        axis=1,
    )
    weighted_beta = np.asarray(
        [normalized_weighted_beta_deg(left, right) for left, right in zip(prediction, truth, strict=True)]
    )
    result = panel[[name for name in ("record_id", "task_node_id", *XYZ_COLUMNS, *BETA_COLUMNS) if name in panel]].copy()
    for index, column in enumerate(BETA_COLUMNS):
        result[f"predicted_{column}"] = prediction[:, index]
        result[f"dls2_{column}"] = corrected[:, index]
    result["weighted_beta_error_deg"] = weighted_beta
    result["raw_fk_residual_mm"] = raw_fk
    result["dls2_fk_residual_mm"] = dls_fk
    result["bounds_violation"] = violation
    result["dls_success"] = (
        np.isfinite(corrected).all(axis=1)
        & ~np.any(
            (corrected < bounds[:, 0] - 1.0e-12)
            | (corrected > bounds[:, 1] + 1.0e-12),
            axis=1,
        )
        & (dls_fk <= 5.0)
    )
    no_nan = bool(
        np.isfinite(prediction).all()
        and np.isfinite(raw_fk).all()
        and np.isfinite(dls_fk).all()
    )
    report = {
        "panel_id": panel_id,
        "status": "complete",
        "row_count": len(panel),
        "no_nan": no_nan,
        "bounds_violation_count": int(np.count_nonzero(violation)),
        "weighted_beta_p95_deg": percentile(weighted_beta, 95),
        "raw_fk_p95_mm": percentile(raw_fk, 95),
        "dls_two_step_fk_p95_mm": percentile(dls_fk, 95),
        "dls_success_rate": float(result["dls_success"].mean()),
        "raw_latency_ms_per_row": raw_latency * 1000.0,
        "dls_latency_ms_per_row": dls_latency * 1000.0,
    }
    return result, report


def _panel_a(config: Mapping[str, Any]) -> pd.DataFrame:
    supervision = pd.read_parquet(_upstream_path(config, "student_supervision"))
    panel = supervision[supervision["split_role"].eq("test")].copy()
    panel["macroblock_id"] = macroblock_ids(panel, block_size_mm=40)
    panel["panel_id"] = "A_seed_domain_common"
    return panel.sort_values("record_id", kind="stable").reset_index(drop=True)


def _train_student_scale(
    config: Mapping[str, Any],
    dataset: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    directory: Path,
    *,
    scale_id: str,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("retry10 Student requires CUDA_VISIBLE_DEVICES=-1 before process start")
    import tensorflow as tf

    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("retry10 CPU Student unexpectedly sees a GPU")
    panel_blocks = {
        str(value)
        for panel in panels.values()
        for value in panel.get("macroblock_id", macroblock_ids(panel, block_size_mm=40)).astype(str)
    }
    environment = _environment(config)
    supervision, train, validation = _prepare_student_frame(
        dataset,
        environment,
        panel_macroblocks=panel_blocks,
        seed=int(config["student"]["seed"]),
    )
    student = config["student"]
    trained = train_workspace_student(
        train,
        validation,
        mode=RepresentationMode.XYZ_GLOBAL,
        geometry=_student_geometry(environment),
        config=WorkspaceStudentTrainingConfig(
            hidden_units=tuple(map(int, student["hidden_units"])),
            learning_rate=float(student["learning_rate"]),
            max_steps=int(student["max_steps"]),
            validation_interval=int(student["validation_interval"]),
            patience_intervals=int(student["patience_intervals"]),
            seed=int(student["seed"]),
            beta_coordinate_weights=tuple(map(float, config["metric"]["beta_coordinate_weights"])),
            beta_loss_only=True,
        ),
    )
    save_workspace_student_models(trained.models, directory / "models")
    _write_parquet(supervision, directory / "student_supervision.parquet")
    _write_parquet(trained.history, directory / "training_history.parquet")
    panel_reports: dict[str, Any] = {}
    for panel_id, panel in panels.items():
        predictions, report = _evaluate_model_on_panel(
            trained.models.global_model, panel, environment, panel_id=panel_id
        )
        _write_parquet(predictions, directory / f"panel_{panel_id}_predictions.parquet")
        panel_reports[panel_id] = report
    common = panel_reports.get("A", next(iter(panel_reports.values())))
    decision = student_quality_decision(
        evaluated=common["status"] == "complete",
        no_nan=bool(common.get("no_nan", False)),
        bounds_violation_count=int(common.get("bounds_violation_count", 0)),
        raw_fk_p95_mm=float(common.get("raw_fk_p95_mm", math.inf)),
        dls_two_step_fk_p95_mm=float(common.get("dls_two_step_fk_p95_mm", math.inf)),
    )
    gate = {
        "status": "complete",
        "scale_id": scale_id,
        "dataset_row_count": len(dataset),
        "train_row_count": len(train),
        "validation_row_count": len(validation),
        "from_scratch": True,
        "metric_version": METRIC_VERSION,
        "student_quality": decision.to_dict(),
        "panels": panel_reports,
        "student_scaling_claim_authorized": False,
        "student_scaling_claim_reason": "multi_seed_not_evaluated",
        "formal_authorized": False,
    }
    _write_json(directory / "gate.json", gate)
    return gate


def stage_seed_students(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["seed_students"]
    panel_a = _panel_a(config)
    _write_parquet(panel_a, stage / "panel_a_seed_domain_common.parquet")
    scale_rows = (("5k", 5_105), ("10k", 10_000), ("20k", 20_000))
    reports: dict[str, Any] = {}
    panel_metrics: dict[str, np.ndarray] = {}
    for scale_id, rows in scale_rows:
        dataset_path = output_root / STAGE_DIRS["seed_datasets"] / (
            "dataset_5105.parquet" if rows == 5_105 else f"dataset_{rows}.parquet"
        )
        dataset = pd.read_parquet(dataset_path)
        if len(dataset) < rows:
            reports[scale_id] = {
                "status": "not_evaluated",
                "reason": "exact_dataset_unavailable",
                "available_rows": len(dataset),
            }
            continue
        report = _train_student_scale(
            config, dataset, {"A": panel_a}, stage / scale_id, scale_id=scale_id
        )
        reports[scale_id] = report
        predictions = pd.read_parquet(stage / scale_id / "panel_A_predictions.parquet")
        panel_metrics[scale_id] = predictions["raw_fk_residual_mm"].to_numpy(float)
    curves: list[dict[str, Any]] = []
    available = [name for name, _rows in scale_rows if name in panel_metrics]
    for smaller, larger in zip(available, available[1:]):
        bootstrap = deterministic_paired_bootstrap_delta(
            panel_metrics[smaller], panel_metrics[larger], seed=int(config["student"]["seed"])
        )
        curves.append({"smaller_scale": smaller, "larger_scale": larger, **bootstrap})
    _write_parquet(pd.DataFrame.from_records(curves), stage / "seed_learning_curve.parquet")
    gate = {
        "status": "complete",
        "gate_pass": bool("10k" in panel_metrics),
        "panel_a_row_count": len(panel_a),
        "scales": reports,
        "seed_learning_curve_artifact_exists": True,
        "student_scaling_claim_authorized": False,
        "student_scaling_claim_reason": "multi_seed_not_evaluated",
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "seed_students", gate)


def stage_atlas_expansion(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["atlas_expansion"]
    labels = pd.read_parquet(
        output_root / STAGE_DIRS["atlas_poc"] / "candidate_atlas_after_poc.parquet"
    )
    _upstream_labels, tasks, edges = _source_frames(config)
    poc_roots = pd.read_parquet(
        output_root / STAGE_DIRS["atlas_poc"] / "batch_01" / "roots.parquet"
    )
    prior_roots = poc_roots["task_node_id"].astype(int).tolist()
    reports: list[dict[str, Any]] = []
    low_gain_batches = 0
    exhausted = False
    maximum_batches = int(config["atlas"]["maximum_batches"])
    plateau_threshold = float(config["atlas"]["plateau_gain_fraction"])
    for batch_index in range(2, maximum_batches + 1):
        labels, frames, report = _run_atlas_batch(
            config,
            labels,
            tasks,
            edges,
            batch_index=batch_index,
            prior_roots=prior_roots,
        )
        _write_batch_frames(stage, batch_index, frames)
        _write_json(stage / f"batch_{batch_index:02d}" / "report.json", report)
        reports.append(report)
        prior_roots.extend(map(int, report.get("root_node_ids", ())))
        if report["status"] == "exhausted":
            exhausted = True
            break
        if float(report["coverage_gain"]) < plateau_threshold - 1.0e-12:
            low_gain_batches += 1
        else:
            low_gain_batches = 0
        if low_gain_batches >= int(config["atlas"]["plateau_consecutive_batches"]):
            break
    plateau = low_gain_batches >= int(config["atlas"]["plateau_consecutive_batches"])
    _write_parquet(labels, stage / "candidate_atlas_after_expansion.parquet")
    _write_parquet(pd.DataFrame.from_records(reports), stage / "batch_reports.parquet")
    denominator = int(config["atlas"]["denominator_task_nodes"])
    gate = {
        "status": "complete",
        "gate_pass": True,
        "executed_batch_count": len(reports) + 1,
        "final_label_count": len(labels),
        "coverage_fraction": len(labels) / denominator,
        "coverage_plateau": plateau,
        "root_registry_exhausted": exhausted,
        "plateau_or_exhausted": plateau or exhausted or len(reports) + 1 >= maximum_batches,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "atlas_expansion", gate)


def _all_growth_candidates(output_root: Path) -> pd.DataFrame:
    paths = sorted(
        list((output_root / STAGE_DIRS["atlas_poc"]).glob("batch_*/candidates.parquet"))
        + list((output_root / STAGE_DIRS["atlas_expansion"]).glob("batch_*/candidates.parquet"))
    )
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _all_growth_events(output_root: Path) -> pd.DataFrame:
    paths = sorted(
        list((output_root / STAGE_DIRS["atlas_poc"]).glob("batch_*/growth_events.parquet"))
        + list((output_root / STAGE_DIRS["atlas_expansion"]).glob("batch_*/growth_events.parquet"))
        + list((output_root / STAGE_DIRS["atlas_poc"]).glob("batch_*/cap_hit_events.parquet"))
        + list((output_root / STAGE_DIRS["atlas_expansion"]).glob("batch_*/cap_hit_events.parquet"))
    )
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _taxonomy_evidence(
    labels: pd.DataFrame,
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    candidates: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    covered = set(labels["task_node_id"].astype(int))
    adjacency = {int(node): set() for node in tasks["task_node_id"].astype(int)}
    existing_pairs: set[tuple[int, int]] = set()
    for edge in edges.itertuples(index=False):
        left, right = sorted((int(edge.left_node_id), int(edge.right_node_id)))
        existing_pairs.add((left, right))
        adjacency[left].add(right)
        adjacency[right].add(left)
    refined = build_shared_face_task_edges(tasks, edges)
    missing_refined_nodes: set[int] = set()
    for edge in refined.itertuples(index=False):
        pair = tuple(sorted((int(edge.left_node_id), int(edge.right_node_id))))
        if pair not in existing_pairs:
            missing_refined_nodes.update(pair)
    candidate_groups = (
        candidates.groupby("task_node_id", sort=True) if not candidates.empty else {}
    )
    event_node_column = next(
        (name for name in ("target_node_id", "task_node_id", "node_id") if name in events),
        None,
    )
    cap_nodes: set[int] = set()
    if event_node_column is not None:
        reason = events.get("reason", pd.Series("", index=events.index)).astype(str).str.lower()
        cap_nodes = set(events.loc[reason.str.contains("cap"), event_node_column].astype(int))
    records: list[dict[str, Any]] = []
    for node_id in sorted(set(tasks["task_node_id"].astype(int)) - covered):
        if not candidates.empty and node_id in candidate_groups.groups:
            group = candidate_groups.get_group(node_id)
            beta = group.loc[:, BETA_COLUMNS].to_numpy(float)
            raw_gap = max(
                (
                    raw_beta_max_deg(beta[left], beta[right])
                    for left in range(len(beta))
                    for right in range(left + 1, len(beta))
                ),
                default=0.0,
            )
            feasible_count = len(group)
            finite_candidate = bool(np.isfinite(beta).all())
        else:
            raw_gap = 0.0
            feasible_count = 0
            finite_candidate = False
        retained_neighbors = adjacency[node_id] & covered
        records.append(
            {
                "task_node_id": node_id,
                "feasible_label_count": feasible_count,
                "raw_candidate_gap_max_deg": raw_gap,
                "catastrophic_neighbor_jump": raw_gap > 5.0,
                "cap_hit": node_id in cap_nodes or bool(adjacency[node_id] & cap_nodes),
                "pruned_hypothesis_reachable": node_id in cap_nodes,
                "registered_retained_neighbor": bool(retained_neighbors),
                "local_continuation_succeeded": False,
                "local_continuation_exhausted_or_pending": bool(retained_neighbors),
                "edge_predicate_satisfied_missing_edge": node_id in missing_refined_nodes,
                "finite_in_bounds_fk_valid_label": finite_candidate,
                "certified_primary_path": False,
            }
        )
    return pd.DataFrame.from_records(records)


def _targeted_rescue(
    config: Mapping[str, Any],
    labels: pd.DataFrame,
    tasks: pd.DataFrame,
    edges: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    environment = _environment(config)
    continuation = make_optimized_predictor_corrector_continuation(
        environment,
        damping=2.0e-3,
        max_corrector_iterations=400,
        residual_tolerance_mm=float(config["atlas"]["individual_fk_max_mm"]),
    )
    indexed = labels.drop_duplicates("task_node_id").set_index("task_node_id")
    task_lookup = tasks.set_index("task_node_id")
    adjacency = {int(node): set() for node in tasks["task_node_id"].astype(int)}
    for edge in build_shared_face_task_edges(tasks, edges).itertuples(index=False):
        adjacency[int(edge.left_node_id)].add(int(edge.right_node_id))
        adjacency[int(edge.right_node_id)].add(int(edge.left_node_id))
    records: list[dict[str, Any]] = []
    eligible = taxonomy[
        taxonomy["taxonomy"].isin(("frontier_reachable", "graph_limited", "beam_limited"))
    ]
    for row in eligible.sort_values(["taxonomy_priority", "task_node_id"]).itertuples(index=False):
        node_id = int(row.task_node_id)
        neighbors = sorted(adjacency[node_id] & set(indexed.index.astype(int)))
        if not neighbors:
            continue
        target_row = task_lookup.loc[node_id]
        target = AtlasTaskNode(
            node_id,
            target_row.loc[list(XYZ_COLUMNS)].to_numpy(float),
            (),
        )
        proposals: list[dict[str, Any]] = []
        for source_id in neighbors[:4]:
            source = _atlas_candidate(indexed.loc[source_id], node_id=source_id)
            outcome = continuation(source, target)
            if outcome.success and outcome.actual_bounds and outcome.residual_mm <= float(config["atlas"]["individual_fk_max_mm"]):
                proposals.append(
                    {
                        "source_id": source_id,
                        "beta": np.asarray(outcome.beta_rad, dtype=float),
                        "residual_mm": float(outcome.residual_mm),
                    }
                )
        if not proposals:
            continue
        candidate = min(proposals, key=lambda value: (value["residual_mm"], value["source_id"]))
        compatible = all(
            normalized_weighted_beta_deg(
                candidate["beta"], indexed.loc[neighbor, list(BETA_COLUMNS)].to_numpy(float)
            ) <= float(config["atlas"]["weighted_local_p95_max_deg"])
            and raw_beta_max_deg(
                candidate["beta"], indexed.loc[neighbor, list(BETA_COLUMNS)].to_numpy(float)
            ) <= float(config["atlas"]["raw_catastrophic_deg"])
            for neighbor in neighbors[:4]
        )
        if not compatible:
            continue
        record = {
            "task_node_id": node_id,
            "chart_id": "retry10_targeted_r2",
            "candidate_id": f"retry10_r2_{node_id}",
            "source_parent_node_id": int(target_row.source_parent_node_id),
            **{name: float(target_row[name]) for name in XYZ_COLUMNS},
            **{
                column: float(candidate["beta"][index])
                for index, column in enumerate(BETA_COLUMNS)
            },
            "residual_mm": candidate["residual_mm"],
            "teacher_fk_residual_mm": candidate["residual_mm"],
            "min_margin_deg": 1.0,
            "normalized_min_margin": 1.0 / 180.0,
            "posture_cost": float(np.linalg.norm(candidate["beta"])),
            "condition_number": 1.0,
            "actual_bounds": True,
            "quality": "Gold",
            "label_quality": "Gold",
            "label_origin": f"taxonomy_rescue:{row.taxonomy}",
            "component_id": "retry10_rescue_pending",
        }
        records.append(record)
        for name, value in record.items():
            if name != "task_node_id":
                indexed.loc[node_id, name] = value
    rescued = pd.DataFrame.from_records(records)
    merged = pd.concat([labels, rescued], ignore_index=True, sort=False)
    merged = merged.drop_duplicates("task_node_id", keep="first").reset_index(drop=True)
    return merged, rescued


def _assign_components(labels: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    frame = labels.copy()
    retained = set(frame["task_node_id"].astype(int))
    adjacency = {node: set() for node in retained}
    for edge in edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        if left in retained and right in retained:
            adjacency[left].add(right)
            adjacency[right].add(left)
    assignments: dict[int, str] = {}
    components: list[set[int]] = []
    unseen = set(retained)
    while unseen:
        start = min(unseen)
        component = {start}
        frontier = {start}
        while frontier:
            frontier = {
                neighbor
                for node in frontier
                for neighbor in adjacency[node]
                if neighbor not in component
            }
            component.update(frontier)
        unseen -= component
        components.append(component)
    components.sort(key=lambda value: (-len(value), min(value)))
    for index, component in enumerate(components):
        for node in component:
            assignments[node] = f"component_{index:03d}"
    frame["component_id"] = frame["task_node_id"].astype(int).map(assignments)
    return frame


def stage_taxonomy_rescue(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["taxonomy_rescue"]
    labels = pd.read_parquet(
        output_root / STAGE_DIRS["atlas_expansion"] / "candidate_atlas_after_expansion.parquet"
    )
    _base, tasks, edges = _source_frames(config)
    candidates = _all_growth_candidates(output_root)
    events = _all_growth_events(output_root)
    evidence = _taxonomy_evidence(labels, tasks, edges, candidates, events)
    taxonomy = classify_frontier_taxonomy(evidence)
    merged, rescued = _targeted_rescue(config, labels, tasks, edges, taxonomy)
    refined_edges = build_shared_face_task_edges(tasks, edges)
    merged = _assign_components(merged, refined_edges)
    boundary = component_boundary_audit(
        merged,
        refined_edges,
        weighted_p95_max_deg=float(config["atlas"]["weighted_local_p95_max_deg"]),
        raw_catastrophic_deg=float(config["atlas"]["raw_catastrophic_deg"]),
        raw_catastrophic_rate_max=float(config["atlas"]["raw_catastrophic_rate_max"]),
    )
    _write_parquet(evidence, stage / "frontier_taxonomy_evidence.parquet")
    _write_parquet(taxonomy, stage / "frontier_taxonomy.parquet")
    _write_parquet(rescued, stage / "targeted_r2_rescued_labels.parquet")
    _write_parquet(refined_edges, stage / "refined_registered_task_edges.parquet")
    _write_parquet(boundary, stage / "cross_component_boundary_audit.parquet")
    _write_parquet(merged, stage / "candidate_atlas_after_rescue.parquet")
    gate = {
        "status": "complete",
        "gate_pass": True,
        "taxonomy_complete": len(taxonomy) + len(labels) == len(tasks),
        "taxonomy_counts": taxonomy["taxonomy"].value_counts().to_dict(),
        "rescued_label_count": len(rescued),
        "candidate_label_count": len(merged),
        "component_count": int(merged["component_id"].nunique()),
        "component_incompatibility_count": int(
            (~boundary.get("component_global_student_compatible", pd.Series(dtype=bool)).astype(bool)).sum()
        ),
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "taxonomy_rescue", gate)


def _local_edge_audit(labels: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    indexed = labels.drop_duplicates("task_node_id").set_index("task_node_id")
    records: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        left, right = int(edge.left_node_id), int(edge.right_node_id)
        if left not in indexed.index or right not in indexed.index:
            continue
        left_beta = indexed.loc[left, list(BETA_COLUMNS)].to_numpy(float)
        right_beta = indexed.loc[right, list(BETA_COLUMNS)].to_numpy(float)
        records.append(
            {
                "left_node_id": left,
                "right_node_id": right,
                "weighted_gap_deg": normalized_weighted_beta_deg(left_beta, right_beta),
                "raw_gap_deg": raw_beta_max_deg(left_beta, right_beta),
            }
        )
    return pd.DataFrame.from_records(records)


def _freeze_panel(
    labels: pd.DataFrame,
    *,
    excluded_blocks: set[str],
    target_rows: int,
    ranking: Sequence[str],
) -> pd.DataFrame:
    frame = labels.copy()
    frame["macroblock_id"] = macroblock_ids(frame, block_size_mm=40)
    frame = frame[~frame["macroblock_id"].astype(str).isin(excluded_blocks)].copy()
    if frame.empty:
        return frame
    available_ranking = [name for name in ranking if name in frame]
    if available_ranking:
        frame = frame.sort_values(
            available_ranking + ["task_node_id"],
            ascending=[False] * len(available_ranking) + [True],
            kind="stable",
        )
    selected_blocks: list[str] = []
    count = 0
    for block, group in frame.groupby("macroblock_id", sort=False):
        selected_blocks.append(str(block))
        count += len(group)
        if count >= int(target_rows):
            break
    return frame[frame["macroblock_id"].astype(str).isin(selected_blocks)].copy()


def stage_atlas_freeze(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["atlas_freeze"]
    labels = pd.read_parquet(
        output_root / STAGE_DIRS["taxonomy_rescue"] / "candidate_atlas_after_rescue.parquet"
    )
    edges = pd.read_parquet(
        output_root / STAGE_DIRS["taxonomy_rescue"] / "refined_registered_task_edges.parquet"
    )
    environment = _environment(config)
    beta = labels.loc[:, BETA_COLUMNS].to_numpy(float)
    xyz = labels.loc[:, XYZ_COLUMNS].to_numpy(float)
    bounds = np.asarray(environment.bounds, dtype=float)
    finite = np.isfinite(beta).all(axis=1) & np.isfinite(xyz).all(axis=1)
    actual_bounds = np.all(
        (beta >= bounds[:, 0] - 1.0e-12) & (beta <= bounds[:, 1] + 1.0e-12), axis=1
    )
    fk = np.linalg.norm(np.asarray(environment.fk(beta)).reshape(-1, 3) - xyz, axis=1) * 1000.0
    labels["teacher_fk_residual_mm"] = fk
    labels["actual_bounds"] = actual_bounds
    hard_row = finite & actual_bounds & (fk <= float(config["atlas"]["individual_fk_max_mm"]))
    quarantined = labels[~hard_row].copy()
    labels = labels[hard_row].copy()
    local = _local_edge_audit(labels, edges)
    catastrophic_edges = local[
        local["raw_gap_deg"].gt(float(config["atlas"]["raw_catastrophic_deg"]))
    ]
    catastrophic_nodes = set(catastrophic_edges.get("left_node_id", ())) | set(
        catastrophic_edges.get("right_node_id", ())
    )
    if catastrophic_nodes:
        quarantined = pd.concat(
            [quarantined, labels[labels["task_node_id"].astype(int).isin(catastrophic_nodes)]],
            ignore_index=True,
            sort=False,
        )
        labels = labels[~labels["task_node_id"].astype(int).isin(catastrophic_nodes)].copy()
        local = _local_edge_audit(labels, edges)
    labels = _assign_components(labels, edges)
    boundary = component_boundary_audit(
        labels,
        edges,
        weighted_p95_max_deg=float(config["atlas"]["weighted_local_p95_max_deg"]),
        raw_catastrophic_deg=float(config["atlas"]["raw_catastrophic_deg"]),
        raw_catastrophic_rate_max=float(config["atlas"]["raw_catastrophic_rate_max"]),
    )
    incompatible = set()
    for row in boundary.itertuples(index=False):
        if not bool(row.component_global_student_compatible):
            incompatible.add(str(row.component_b))
    labels["component_global_student_compatible"] = ~labels["component_id"].astype(str).isin(incompatible)
    global_labels = labels[labels["component_global_student_compatible"].astype(bool)].copy()
    fk_p95, fk_p99 = percentile(labels["teacher_fk_residual_mm"], 95), percentile(labels["teacher_fk_residual_mm"], 99)
    jump_rate = float(
        local["raw_gap_deg"].gt(float(config["atlas"]["raw_catastrophic_deg"])).mean()
    ) if len(local) else 0.0
    weighted_p95 = percentile(local.get("weighted_gap_deg", ()), 95)
    duplicate_count = int(labels["task_node_id"].duplicated().sum())
    hard_pass = bool(
        not len(quarantined)
        and duplicate_count == 0
        and fk_p95 <= float(config["atlas"]["fk_p95_max_mm"])
        and fk_p99 <= float(config["atlas"]["fk_p99_max_mm"])
        and weighted_p95 <= float(config["atlas"]["weighted_local_p95_max_deg"])
        and jump_rate <= float(config["atlas"]["raw_catastrophic_rate_max"])
    )
    # Quarantine is a successful local rollback when the re-audited frozen atlas passes.
    reaudited_pass = bool(
        duplicate_count == 0
        and fk_p95 <= float(config["atlas"]["fk_p95_max_mm"])
        and fk_p99 <= float(config["atlas"]["fk_p99_max_mm"])
        and weighted_p95 <= float(config["atlas"]["weighted_local_p95_max_deg"])
        and jump_rate <= float(config["atlas"]["raw_catastrophic_rate_max"])
    )
    panel_a = _panel_a(config)
    excluded = set(panel_a["macroblock_id"].astype(str))
    new_labels = global_labels[
        ~global_labels.get("label_origin", pd.Series("", index=global_labels.index)).astype(str).eq("retry9_existing_atlas")
    ]
    panel_b = _freeze_panel(
        new_labels,
        excluded_blocks=excluded,
        target_rows=int(config["evaluation_panels"]["target_rows_each"]),
        ranking=("normalized_min_margin",),
    )
    excluded.update(panel_b.get("macroblock_id", pd.Series(dtype=str)).astype(str))
    stress = global_labels.copy()
    stress["stress_score"] = pd.to_numeric(
        stress.get("condition_number", pd.Series(0.0, index=stress.index)), errors="coerce"
    ).fillna(0.0)
    stress["stress_score"] += stress.get("label_origin", pd.Series("", index=stress.index)).astype(str).str.contains("rescue").astype(float) * 1.0e6
    panel_c = _freeze_panel(
        stress,
        excluded_blocks=excluded,
        target_rows=int(config["evaluation_panels"]["target_rows_each"]),
        ranking=("stress_score",),
    )
    panel_b["panel_id"] = "B_expanded_atlas"
    panel_c["panel_id"] = "C_frontier_boundary_stress"
    _write_parquet(labels, stage / "frozen_candidate_atlas.parquet")
    _write_parquet(global_labels, stage / "global_student_compatible_atlas.parquet")
    _write_parquet(quarantined, stage / "quarantined_rows.parquet")
    _write_parquet(local, stage / "local_edge_integrity_audit.parquet")
    _write_parquet(boundary, stage / "cross_component_boundary_audit.parquet")
    _write_parquet(panel_a, stage / "panel_a.parquet")
    _write_parquet(panel_b, stage / "panel_b.parquet")
    _write_parquet(panel_c, stage / "panel_c.parquet")
    denominator = int(config["atlas"]["denominator_task_nodes"])
    gate = {
        "status": "pass" if reaudited_pass else "fail",
        "gate_pass": reaudited_pass,
        "atlas_expansion_valid": reaudited_pass,
        "atlas_frozen": reaudited_pass,
        "exploration_hard_guards_pass": reaudited_pass,
        "initial_hard_pass_without_quarantine": hard_pass,
        "quarantined_row_count": len(quarantined),
        "frozen_label_count": len(labels),
        "global_student_label_count": len(global_labels),
        "coverage_fraction": len(labels) / denominator,
        "fk_p95_mm": fk_p95,
        "fk_p99_mm": fk_p99,
        "local_weighted_p95_deg": weighted_p95,
        "catastrophic_edge_rate": jump_rate,
        "duplicate_count": duplicate_count,
        "panel_rows": {"A": len(panel_a), "B": len(panel_b), "C": len(panel_c)},
        "panel_underfill_blocks_student_claim_only": True,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "atlas_freeze", gate)


def stage_full_diagnostic(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["full_diagnostic"]
    freeze = _gate(output_root, "atlas_freeze")
    labels = pd.read_parquet(output_root / STAGE_DIRS["atlas_freeze"] / "frozen_candidate_atlas.parquet")
    edges = pd.read_parquet(output_root / STAGE_DIRS["taxonomy_rescue"] / "refined_registered_task_edges.parquet")
    cycles = bounded_fundamental_cycles(
        labels["task_node_id"].astype(int), edges, maximum_cycles=128, maximum_length=32
    )
    cycle_rows = [
        {"cycle_id": index, "path_node_ids": path, "length": len(path) - 1}
        for index, path in enumerate(cycles)
    ]
    _write_parquet(pd.DataFrame.from_records(cycle_rows), stage / "registered_long_cycle_diagnostics.parquet")
    gate = {
        "status": "complete",
        "gate_pass": bool(freeze["exploration_hard_guards_pass"]),
        "exploration_hard_guard_failure": not bool(freeze["exploration_hard_guards_pass"]),
        "formal_diagnostic_failure": True,
        "formal_failure_reasons": [
            "reach_measure_not_evaluated",
            "formal_repeat_max_not_evaluated",
            "long_transport_execution_not_evaluated",
        ],
        "registered_cycle_count": len(cycles),
        "exploratory_materialization_may_proceed": bool(freeze["exploration_hard_guards_pass"]),
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "full_diagnostic", gate)


def _target_for_stage(stage_name: str) -> int:
    return {
        "preflight_50k": 50_000,
        "dataset_50k": 50_000,
        "student_50k": 50_000,
        "preflight_100k": 100_000,
        "dataset_100k": 100_000,
        "student_100k": 100_000,
        "preflight_200k": 200_000,
        "dataset_200k": 200_000,
        "student_200k": 200_000,
    }[stage_name]


def _prior_dataset(output_root: Path, target_rows: int) -> tuple[pd.DataFrame, bool]:
    if int(target_rows) == 50_000:
        gate = _gate(output_root, "seed_datasets")
        preferred = 20_000 if gate["seed_20k_target_pass"] else 10_000
        return (
            pd.read_parquet(output_root / STAGE_DIRS["seed_datasets"] / f"dataset_{preferred}.parquet"),
            bool(gate["seed_10k_target_pass"]),
        )
    prior = 50_000 if int(target_rows) == 100_000 else 100_000
    stage_name = f"dataset_{prior // 1000}k"
    gate = _gate(output_root, stage_name)
    path = output_root / STAGE_DIRS[stage_name] / f"dataset_{prior}.parquet"
    return (pd.read_parquet(path) if path.is_file() else pd.DataFrame(), bool(gate.get("dataset_complete", False)))


def _coarse_sampling_strata(targets: pd.DataFrame) -> pd.DataFrame:
    frame = targets.copy()
    if frame.empty:
        return frame
    ranks = pd.Series(pd.qcut(frame["x_m"].rank(method="first"), 3, labels=False), index=frame.index)
    parent_support = frame.groupby("source_parent_node_id")["target_id"].transform("count")
    support_tier = pd.qcut(parent_support.rank(method="first"), 2, labels=False)
    frame["sampling_stratum"] = [
        f"x{int(x)}:support{int(support)}"
        for x, support in zip(ranks, support_tier, strict=True)
    ]
    return frame


def _x_tertile_coverage(labels: pd.DataFrame, tasks: pd.DataFrame) -> float:
    ranked = tasks.copy()
    ranked["x_tertile"] = pd.qcut(
        ranked["x_m"].rank(method="first"), 3, labels=False
    )
    covered = set(labels["task_node_id"].astype(int))
    fractions = [
        float(group["task_node_id"].astype(int).isin(covered).mean())
        for _tertile, group in ranked.groupby("x_tertile")
    ]
    return min(fractions, default=0.0)


def _preflight_stage(
    config: Mapping[str, Any], output_root: Path, stage_name: str
) -> dict[str, Any]:
    target_rows = _target_for_stage(stage_name)
    stage = output_root / STAGE_DIRS[stage_name]
    freeze = _gate(output_root, "atlas_freeze")
    expansion = _gate(output_root, "atlas_expansion")
    taxonomy_gate = _gate(output_root, "taxonomy_rescue")
    seed_students = _gate(output_root, "seed_students")
    prior, prior_complete = _prior_dataset(output_root, target_rows)
    sources = pd.read_parquet(
        output_root / STAGE_DIRS["atlas_freeze"] / "global_student_compatible_atlas.parquet"
    )
    _base, tasks, _edges = _source_frames(config)
    targets = parent_domain_target_registry(
        sources,
        target_count=int(config["sampling"]["preflight_rows"]),
        shard_count=int(config["runtime"]["logical_shard_count"]),
        seed=int(config["runtime"]["seed"]) + target_rows + 1,
        excluded_xyz_keys=prior.get("xyz_key", pd.Series(dtype=str)).astype(str),
    )
    targets = _coarse_sampling_strata(targets)
    attempts = _run_label_registry(config, stage / "workers", sources, targets)
    strata_rows: list[dict[str, Any]] = []
    for stratum, group in attempts.groupby("sampling_stratum", sort=True):
        strata_rows.append(
            {
                "stratum_id": str(stratum),
                "target_weight": len(group) / max(1, len(attempts)),
                "trial_count": len(group),
                "gold_count": int(group["label_quality"].eq("Gold").sum()),
                "silver_count": int(group["label_quality"].eq("Silver").sum()),
                "no_solution_count": int(group["solver_success_count"].eq(0).sum()),
                "branch_disagreement_count": int(
                    group["multiparent_raw_gap_deg"].gt(
                        float(config["atlas"]["raw_catastrophic_deg"])
                    ).sum()
                ),
            }
        )
    strata = pd.DataFrame.from_records(strata_rows)
    allowed = tuple(config["sampling"]["qualities"][f"dataset_{target_rows // 1000}k"])
    preflight, estimate = preflight_generation_decision(
        strata,
        required_new_rows=max(0, target_rows - len(prior)),
        allowed_qualities=allowed,
        attempt_multiplier=int(config["sampling"]["attempt_multiplier"]),
    )
    decision, dataset_id = dataset_generation_decision(
        target_rows=target_rows,
        atlas_frozen=bool(freeze["atlas_frozen"]),
        exploration_hard_guards_pass=bool(freeze["exploration_hard_guards_pass"]),
        coverage_fraction=float(freeze["coverage_fraction"]),
        plateau_or_exhausted=bool(expansion["plateau_or_exhausted"]),
        seed_rows=int(_gate(output_root, "seed_datasets")["maximum_legal_seed_rows"]),
        seed_learning_curve_artifact_exists=bool(seed_students["seed_learning_curve_artifact_exists"]),
        prior_dataset_complete=prior_complete,
        taxonomy_complete=bool(taxonomy_gate["taxonomy_complete"]),
        preflight_pass=preflight.authorized,
        x_tertile_minimum_coverage=_x_tertile_coverage(sources, tasks),
    )
    _write_parquet(targets, stage / "preflight_target_registry.parquet")
    _write_parquet(attempts, stage / "preflight_attempts.parquet")
    _write_parquet(strata, stage / "preflight_strata.parquet")
    gate = {
        "status": "pass" if decision.authorized else "not_authorized",
        "gate_pass": decision.authorized,
        f"dataset_{target_rows // 1000}k_generation_authorized": decision.authorized,
        "preflight_decision": preflight.to_dict(),
        "dataset_decision": decision.to_dict(),
        "dataset_id": dataset_id,
        "estimate": estimate,
        "gold_acceptance_rate": float(attempts["label_quality"].eq("Gold").mean()),
        "silver_acceptance_rate": float(attempts["label_quality"].eq("Silver").mean()),
        "no_solution_rate": float(attempts["solver_success_count"].eq(0).mean()),
        "branch_disagreement_rate": float(
            attempts["multiparent_raw_gap_deg"].gt(float(config["atlas"]["raw_catastrophic_deg"])).mean()
        ),
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, stage_name, gate)


def stage_preflight_50k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _preflight_stage(config, output_root, "preflight_50k")


def stage_preflight_100k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _preflight_stage(config, output_root, "preflight_100k")


def stage_preflight_200k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _preflight_stage(config, output_root, "preflight_200k")


def _dataset_stage(
    config: Mapping[str, Any], output_root: Path, stage_name: str
) -> dict[str, Any]:
    target_rows = _target_for_stage(stage_name)
    stage = output_root / STAGE_DIRS[stage_name]
    preflight_name = f"preflight_{target_rows // 1000}k"
    preflight = _gate(output_root, preflight_name)
    authorization_key = f"dataset_{target_rows // 1000}k_generation_authorized"
    if not bool(preflight.get(authorization_key, False)):
        return _seal_gate(
            output_root,
            config,
            stage_name,
            {
                "status": "not_authorized",
                "gate_pass": False,
                "dataset_complete": False,
                authorization_key: False,
                "reason": "preflight_or_coverage_path_not_authorized",
                "formal_authorized": False,
            },
        )
    prior, prior_complete = _prior_dataset(output_root, target_rows)
    if not prior_complete:
        raise RuntimeError(f"retry10 prior dataset incomplete for {target_rows}")
    sources = pd.read_parquet(
        output_root / STAGE_DIRS["atlas_freeze"] / "global_student_compatible_atlas.parquet"
    )
    required = target_rows - len(prior)
    targets = parent_domain_target_registry(
        sources,
        target_count=int(config["sampling"]["attempt_multiplier"]) * required,
        shard_count=int(config["runtime"]["logical_shard_count"]),
        seed=int(config["runtime"]["seed"]) + target_rows + 2,
        excluded_xyz_keys=prior.get("xyz_key", pd.Series(dtype=str)).astype(str),
    )
    targets = _coarse_sampling_strata(targets)
    attempts = _run_label_registry(config, stage / "workers", sources, targets)
    allowed = tuple(config["sampling"]["qualities"][f"dataset_{target_rows // 1000}k"])
    accepted = attempts[
        attempts["accepted"].astype(bool)
        & attempts["label_quality"].astype(str).isin(set(map(str, allowed)))
    ].copy()
    accepted = _attach_theta(accepted)
    dataset = exact_nested_dataset(
        prior,
        accepted,
        target_rows=target_rows,
        allowed_qualities=allowed,
    )
    integrity = _dataset_integrity(dataset, target_rows)
    strata_present = set(dataset["sampling_stratum"].astype(str))
    expected_strata = set(targets["sampling_stratum"].astype(str))
    quota_pass = expected_strata <= strata_present
    complete = bool(
        integrity["exact_target_pass"]
        and integrity["unique_xyz_pass"]
        and integrity["finite_pass"]
        and integrity["bounds_pass"]
        and integrity["physical_point_id_unique_pass"]
        and quota_pass
    )
    _write_parquet(targets, stage / "target_registry.parquet")
    _write_parquet(attempts, stage / "generation_attempts.parquet")
    _write_parquet(dataset, stage / f"dataset_{target_rows}.parquet")
    gate = {
        "status": "pass" if complete else "underfilled",
        "gate_pass": complete,
        "dataset_complete": complete,
        authorization_key: True,
        "dataset_id": preflight["dataset_id"],
        "target_rows": target_rows,
        "actual_rows": len(dataset),
        "allowed_qualities": list(allowed),
        "gold_count": int(dataset["label_quality"].eq("Gold").sum()),
        "silver_count": int(dataset["label_quality"].eq("Silver").sum()),
        "attempt_count": len(attempts),
        "accepted_count": len(accepted),
        "sampling_quota_pass": quota_pass,
        "integrity": integrity,
        "row_padding_used": False,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, stage_name, gate)


def stage_dataset_50k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _dataset_stage(config, output_root, "dataset_50k")


def stage_dataset_100k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _dataset_stage(config, output_root, "dataset_100k")


def stage_dataset_200k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _dataset_stage(config, output_root, "dataset_200k")


def _frozen_panels(output_root: Path) -> dict[str, pd.DataFrame]:
    root = output_root / STAGE_DIRS["atlas_freeze"]
    return {
        key: pd.read_parquet(root / f"panel_{key.lower()}.parquet")
        for key in ("A", "B", "C")
    }


def _student_stage(
    config: Mapping[str, Any], output_root: Path, stage_name: str
) -> dict[str, Any]:
    target_rows = _target_for_stage(stage_name)
    dataset_name = f"dataset_{target_rows // 1000}k"
    dataset_gate = _gate(output_root, dataset_name)
    if not dataset_gate.get("dataset_complete", False):
        return _seal_gate(
            output_root,
            config,
            stage_name,
            {
                "status": "not_evaluated",
                "gate_pass": False,
                f"student_{target_rows // 1000}k_quality_pass": False,
                "reason": "dataset_unavailable",
                "student_scaling_claim_authorized": False,
                "formal_authorized": False,
            },
        )
    dataset = pd.read_parquet(
        output_root / STAGE_DIRS[dataset_name] / f"dataset_{target_rows}.parquet"
    )
    stage = output_root / STAGE_DIRS[stage_name]
    report = _train_student_scale(
        config,
        dataset,
        _frozen_panels(output_root),
        stage,
        scale_id=f"{target_rows // 1000}k",
    )
    quality = bool(report["student_quality"]["authorized"])
    report["gate_pass"] = quality
    report[f"student_{target_rows // 1000}k_quality_pass"] = quality
    _write_json(stage / "gate.json", report)
    _seal_stage(output_root, config, stage_name)
    return report


def stage_student_50k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _student_stage(config, output_root, "student_50k")


def stage_student_100k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _student_stage(config, output_root, "student_100k")


def stage_student_200k(config: Mapping[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    del project_root
    return _student_stage(config, output_root, "student_200k")


def _tension_worker(
    config: Mapping[str, Any], worker_root: Path, shard_id: int
) -> dict[str, Any]:
    from quasi_exp.io import load_config as load_robot_config, load_robot_inputs
    from quasi_exp.model.quasi_static import QuasiStaticModel
    from quasi_exp.opt.tension_labeler import solve_tension_label

    frame = pd.read_parquet(worker_root / "tension_registry.parquet")
    frame = frame[frame["shard_id"].astype(int).eq(int(shard_id))]
    robot_path = SOURCE_ROOT / str(config["sources"]["tension_robot_config"])
    robot = load_robot_config(robot_path)
    inputs = load_robot_inputs(robot)
    model = QuasiStaticModel(robot, inputs)
    rms_threshold = float(config["tension"]["residual_max"])
    pso_config = robot.get("pso", {})
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        started = time.perf_counter()
        beta = np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float)
        theta_raw = beta_to_theta(beta)
        try:
            cache = model.build_cache(theta_raw)
            solved = solve_tension_label(
                model=model,
                cache=cache,
                pso_cfg=pso_config,
                pso_seed=int(config["runtime"]["seed"]) + int(row.tension_candidate_id),
                rms_thresh=rms_threshold,
            )
            tension = np.asarray(solved.T_base_12, dtype=float).reshape(12)
            residual = float(solved.meta.get("rms_rnorm", math.inf))
            success = bool(solved.ok and np.isfinite(tension).all() and residual < rms_threshold)
            meta = dict(solved.meta)
        except Exception as error:  # worker records numerical failures as evidence
            tension = np.full(12, np.nan)
            residual = math.inf
            success = False
            meta = {"exception_type": type(error).__name__, "exception_message": str(error)}
        record = row._asdict()
        record.update(
            {
                **{f"T{index + 1}_N": float(tension[index]) for index in range(12)},
                "tension_success": success,
                "tension_residual": residual,
                "tension_elapsed_s": time.perf_counter() - started,
                "tension_solver_method": str(robot.get("tension_labeler", {}).get("method", "unknown")),
                "tension_meta_json": json.dumps(meta, sort_keys=True, default=_json_default),
            }
        )
        rows.append(record)
    result = pd.DataFrame.from_records(rows)
    _write_parquet(result, worker_root / f"shard_{int(shard_id):02d}" / "tension.parquet")
    report = {
        "shard_id": int(shard_id),
        "target_count": len(frame),
        "success_count": int(result.get("tension_success", pd.Series(dtype=bool)).astype(bool).sum()),
    }
    _write_json(worker_root / f"shard_{int(shard_id):02d}" / "report.json", report)
    return report


def _run_tension_registry(
    config: Mapping[str, Any], directory: Path, registry: pd.DataFrame
) -> pd.DataFrame:
    frame = registry.copy().reset_index(drop=True)
    frame["tension_candidate_id"] = np.arange(len(frame), dtype=np.int64)
    frame["shard_id"] = frame["tension_candidate_id"].map(
        lambda value: int(
            hashlib.sha256(f"tension:{int(config['runtime']['seed'])}:{int(value)}".encode()).hexdigest()[:16],
            16,
        )
        % int(config["runtime"]["logical_shard_count"])
    )
    _write_parquet(frame, directory / "tension_registry.parquet")
    pending = list(range(int(config["runtime"]["logical_shard_count"])))
    running: list[tuple[int, subprocess.Popen[str], Any]] = []
    while pending or running:
        while pending and len(running) < int(config["tension"]["maximum_concurrent_workers"]):
            shard_id = pending.pop(0)
            shard = directory / f"shard_{shard_id:02d}"
            shard.mkdir(parents=True, exist_ok=True)
            handle = (shard / "worker.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config["config_path"]),
                    "--tension-worker-root",
                    str(directory),
                    "--shard-id",
                    str(shard_id),
                ],
                cwd=SOURCE_ROOT,
                env=_thread_limited_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((shard_id, process, handle))
        survivors: list[tuple[int, subprocess.Popen[str], Any]] = []
        for shard_id, process, handle in running:
            if process.poll() is None:
                survivors.append((shard_id, process, handle))
                continue
            handle.close()
            if process.returncode != 0:
                raise RuntimeError(f"retry10 tension worker failed: shard {shard_id}")
        running = survivors
        if running:
            time.sleep(0.2)
    frames = [
        pd.read_parquet(directory / f"shard_{shard_id:02d}" / "tension.parquet")
        for shard_id in range(int(config["runtime"]["logical_shard_count"]))
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _tension_pilot_registry(
    dataset: pd.DataFrame, *, target_rows: int, seed: int
) -> pd.DataFrame:
    frame = dataset.copy()
    frame["macroblock_id"] = macroblock_ids(frame, block_size_mm=40)
    frame["x_tertile"] = pd.qcut(frame["x_m"].rank(method="first"), 3, labels=False)
    frame["boundary_role"] = np.where(
        frame["sampling_stratum"].astype(str).str.contains("f0|f4"), "boundary", "interior"
    )
    frame["pilot_stratum"] = (
        frame["x_tertile"].astype(str)
        + ":"
        + frame["boundary_role"].astype(str)
        + ":"
        + frame["label_quality"].astype(str)
    )
    selected: list[pd.DataFrame] = []
    strata = sorted(frame["pilot_stratum"].unique())
    quota = int(math.ceil(int(target_rows) / max(1, len(strata))))
    for stratum in strata:
        group = frame[frame["pilot_stratum"].eq(stratum)].copy()
        group["_rank"] = group["xyz_key"].astype(str).map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest()
        )
        selected.append(group.sort_values("_rank", kind="stable").head(quota * 2))
    result = pd.concat(selected, ignore_index=True, sort=False).drop(columns="_rank")
    return result.head(int(target_rows) * 2).reset_index(drop=True)


def stage_tension_pilot(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["tension_pilot"]
    dataset_gate = _gate(output_root, "dataset_50k")
    if not dataset_gate.get("dataset_complete", False):
        return _seal_gate(
            output_root,
            config,
            "tension_pilot",
            {
                "status": "not_evaluated",
                "gate_pass": False,
                "tension_pilot_pass": False,
                "reason": "50k_beta_theta_dataset_unavailable",
                "beta_theta_dataset_authorization_unchanged": True,
                "formal_authorized": False,
            },
        )
    dataset = pd.read_parquet(output_root / STAGE_DIRS["dataset_50k"] / "dataset_50000.parquet")
    registry = _tension_pilot_registry(
        dataset,
        target_rows=int(config["tension"]["pilot_rows"]),
        seed=int(config["runtime"]["seed"]),
    )
    solved = _run_tension_registry(config, stage / "workers", registry)
    successes = solved[solved["tension_success"].astype(bool)].copy()
    selected: list[pd.DataFrame] = []
    replacement_count = 0
    per_stratum_target = (
        registry.head(int(config["tension"]["pilot_rows"]))["pilot_stratum"].value_counts().to_dict()
    )
    for stratum, target in sorted(per_stratum_target.items()):
        available = successes[successes["pilot_stratum"].eq(stratum)].sort_values(
            "tension_candidate_id", kind="stable"
        )
        chosen = available.head(int(target))
        replacement_count += int(
            chosen["tension_candidate_id"].ge(int(config["tension"]["pilot_rows"])).sum()
        )
        selected.append(chosen)
    pilot = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()
    pilot = pilot.head(int(config["tension"]["pilot_rows"]))
    success_rate = float(solved["tension_success"].astype(bool).mean()) if len(solved) else 0.0
    elapsed = solved.loc[solved["tension_success"].astype(bool), "tension_elapsed_s"].to_numpy(float)
    projected_hours = (
        200_000 * float(np.mean(elapsed)) / int(config["tension"]["maximum_concurrent_workers"]) / 3600.0
        if len(elapsed)
        else math.inf
    )
    tension_columns = [f"T{index}_N" for index in range(1, 13)]
    bounds = tuple(map(float, config["tension"]["tension_bounds_n"]))
    bounds_pass = bool(
        len(pilot)
        and np.all(
            (pilot.loc[:, tension_columns].to_numpy(float) >= bounds[0] - 1.0e-12)
            & (pilot.loc[:, tension_columns].to_numpy(float) <= bounds[1] + 1.0e-12)
        )
    )
    passed = bool(
        len(pilot) == int(config["tension"]["pilot_rows"])
        and success_rate >= float(config["tension"]["success_rate_min"])
        and bounds_pass
        and percentile(pilot["tension_residual"], 95) < float(config["tension"]["residual_max"])
        and projected_hours <= float(config["tension"]["projected_200k_budget_hours"])
    )
    _write_parquet(solved, stage / "tension_pilot_attempts.parquet")
    _write_parquet(pilot, stage / "tension_pilot_labels.parquet")
    gate = {
        "status": "pass" if passed else "fail",
        "gate_pass": passed,
        "tension_pilot_pass": passed,
        "pilot_target_rows": int(config["tension"]["pilot_rows"]),
        "pilot_complete_rows": len(pilot),
        "solve_success_rate": success_rate,
        "residual_p50": percentile(pilot.get("tension_residual", ()), 50),
        "residual_p95": percentile(pilot.get("tension_residual", ()), 95),
        "runtime_p50_s": percentile(elapsed, 50),
        "runtime_p95_s": percentile(elapsed, 95),
        "projected_200k_wall_hours": projected_hours,
        "tension_bounds_pass": bounds_pass,
        "replacement_count": replacement_count,
        "replacement_scope": config["tension"]["replacement_scope"],
        "beta_theta_dataset_authorization_unchanged": True,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "tension_pilot", gate)


def stage_tension_materialization(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["tension_materialization"]
    pilot = _gate(output_root, "tension_pilot")
    if not pilot.get("tension_pilot_pass", False):
        return _seal_gate(
            output_root,
            config,
            "tension_materialization",
            {
                "status": "not_authorized",
                "gate_pass": False,
                "tension_materialization_complete": False,
                "reason": "tension_pilot_failed_or_not_evaluated",
                "beta_theta_dataset_authorization_unchanged": True,
                "formal_authorized": False,
            },
        )
    reports: dict[str, Any] = {}
    all_complete = True
    for target_rows in (50_000, 100_000, 200_000):
        dataset_name = f"dataset_{target_rows // 1000}k"
        dataset_gate = _gate(output_root, dataset_name)
        if not dataset_gate.get("dataset_complete", False):
            reports[str(target_rows)] = {"status": "not_applicable", "reason": "dataset_unavailable"}
            continue
        dataset = pd.read_parquet(
            output_root / STAGE_DIRS[dataset_name] / f"dataset_{target_rows}.parquet"
        )
        dataset["macroblock_id"] = macroblock_ids(dataset, block_size_mm=40)
        dataset["replacement_key"] = (
            dataset["source_parent_node_id"].astype(str)
            + "|"
            + dataset["macroblock_id"].astype(str)
            + "|"
            + dataset["sampling_stratum"].astype(str)
        )
        dataset["replacement_role"] = "primary"
        attempts_path = output_root / STAGE_DIRS[dataset_name] / "generation_attempts.parquet"
        backups = pd.read_parquet(attempts_path)
        backups = backups[backups["accepted"].astype(bool)].copy()
        backups["macroblock_id"] = macroblock_ids(backups, block_size_mm=40)
        backups["replacement_key"] = (
            backups["source_parent_node_id"].astype(str)
            + "|"
            + backups["macroblock_id"].astype(str)
            + "|"
            + backups["sampling_stratum"].astype(str)
        )
        backups = backups[~backups["xyz_key"].astype(str).isin(set(dataset["xyz_key"].astype(str)))]
        backups = (
            backups.sort_values(["replacement_key", "xyz_key"], kind="stable")
            .groupby("replacement_key", sort=True)
            .head(2)
        )
        backups["replacement_role"] = "same_stratum_backup"
        registry = pd.concat([dataset, backups], ignore_index=True, sort=False)
        solved = _run_tension_registry(config, stage / f"scale_{target_rows}" / "workers", registry)
        primary = solved[solved["replacement_role"].eq("primary")].sort_values(
            "tension_candidate_id", kind="stable"
        )
        successful_backup = solved[
            solved["replacement_role"].eq("same_stratum_backup")
            & solved["tension_success"].astype(bool)
        ].copy()
        completed_rows: list[pd.Series] = []
        used_backup_ids: set[int] = set()
        replacement_count = 0
        for row in primary.itertuples(index=False):
            if bool(row.tension_success):
                completed_rows.append(pd.Series(row._asdict()))
                continue
            available = successful_backup[
                successful_backup["replacement_key"].eq(str(row.replacement_key))
                & ~successful_backup["tension_candidate_id"].astype(int).isin(used_backup_ids)
            ]
            if available.empty:
                continue
            replacement = available.sort_values("tension_candidate_id", kind="stable").iloc[0]
            used_backup_ids.add(int(replacement["tension_candidate_id"]))
            completed_rows.append(replacement)
            replacement_count += 1
        completed = pd.DataFrame(completed_rows).reset_index(drop=True)
        complete = len(completed) == target_rows
        all_complete = all_complete and complete
        _write_parquet(completed, stage / f"dataset_{target_rows}_with_tension.parquet")
        reports[str(target_rows)] = {
            "status": "pass" if complete else "underfilled",
            "target_rows": target_rows,
            "complete_rows": len(completed),
            "failed_rows": int((~solved["tension_success"].astype(bool)).sum()),
            "same_stratum_replacement_count": replacement_count,
            "replacement_scope": config["tension"]["replacement_scope"],
        }
    gate = {
        "status": "pass" if all_complete else "underfilled",
        "gate_pass": all_complete,
        "tension_materialization_complete": all_complete,
        "scales": reports,
        "beta_theta_dataset_authorization_unchanged": True,
        "formal_authorized": False,
    }
    return _seal_gate(output_root, config, "tension_materialization", gate)


def _report_html(gate: Mapping[str, Any]) -> str:
    dataset_rows = "".join(
        f"<tr><td>{name}</td><td>{value.get('status')}</td><td>{value.get('actual_rows', value.get('target_rows', '—'))}</td></tr>"
        for name, value in gate["dataset_authorizations"].items()
    )
    student_rows = "".join(
        f"<tr><td>{name}</td><td>{value.get('status')}</td><td>{value.get('quality_pass')}</td></tr>"
        for name, value in gate["student_authorizations"].items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>retry10 Frontier-Expanded Atlas</title>
<style>
@page {{ size: A4 landscape; margin: 12mm; }}
body {{ font-family: 'Noto Sans CJK SC', sans-serif; margin:0; color:#172033; background:#eef2f7; }}
.page {{ max-width:1280px; margin:auto; background:white; min-height:100vh; display:grid; grid-template-columns:260px 1fr; }}
aside {{ background:#14213d; color:white; padding:28px 22px; }} main {{ padding:30px 38px; }}
h1 {{ font-size:27px; margin:0 0 8px; }} h2 {{ color:#1d4ed8; margin-top:26px; }}
.metric {{ background:#eff6ff; border-left:5px solid #2563eb; padding:14px 18px; margin:12px 0; }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid #dbe2ea; padding:9px; text-align:left; }}
.fail {{ color:#b91c1c; }} .small {{ font-size:12px; color:#64748b; }} aside .small {{ color:#cbd5e1; }}
</style></head><body><div class="page"><aside><h1>BACRA retry10</h1>
<p>Frontier-Expanded Atlas</p><p class="small">Pilot-domain exploratory evidence</p>
<hr><p>Formal: <b>false</b></p><p>Deployment: <b>false</b></p><p>Full workspace: <b>false</b></p>
</aside><main><h1>实验终态摘要</h1>
<div class="metric">Frozen atlas: <b>{gate['atlas_authorization']['frozen_label_count']}</b> labels，coverage <b>{100.0 * gate['atlas_authorization']['coverage_fraction']:.2f}%</b></div>
<p>本文仅报告固定 25,000-node Pilot task domain；不外推为完整连续 Omega200。</p>
<h2>Dataset authorization</h2><table><tr><th>规模</th><th>状态</th><th>rows</th></tr>{dataset_rows}</table>
<h2>Student authorization</h2><table><tr><th>规模</th><th>状态</th><th>quality pass</th></tr>{student_rows}</table>
<h2>物理标签与 Formal 边界</h2><p>Tension pilot: {gate['physics_label_authorization']['tension_pilot_pass']}；Tension materialization: {gate['physics_label_authorization']['tension_materialization_complete']}。</p>
<p class="fail">formal_authorization=false；deployment_authorization=false；student_scaling_claim_authorized=false（single seed）。</p>
<p class="small">scientific_source_fixed_point={gate['scientific_source_fixed_point']}<br>config_sha256={gate['config_sha256']}</p>
</main></div></body></html>"""


def stage_summary(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    del project_root
    stage = output_root / STAGE_DIRS["summary"]
    freeze = _gate(output_root, "atlas_freeze")
    datasets: dict[str, Any] = {}
    students: dict[str, Any] = {}
    for scale in (50, 100, 200):
        dataset = _gate(output_root, f"dataset_{scale}k")
        student = _gate(output_root, f"student_{scale}k")
        datasets[f"{scale}k"] = {
            "status": dataset["status"],
            "generation_authorized": bool(
                dataset.get(f"dataset_{scale}k_generation_authorized", False)
            ),
            "complete": bool(dataset.get("dataset_complete", False)),
            "actual_rows": int(dataset.get("actual_rows", 0)),
            "dataset_id": dataset.get("dataset_id"),
        }
        students[f"{scale}k"] = {
            "status": student["status"],
            "quality_pass": bool(student.get(f"student_{scale}k_quality_pass", False)),
        }
    tension_pilot = _gate(output_root, "tension_pilot")
    tension_full = _gate(output_root, "tension_materialization")
    gate = {
        "governance_structure": "registered_retry10_protocol_config_runner",
        "scientific_contract": "pilot_domain_frontier_expansion_and_staged_materialization",
        "operational_completion": True,
        "artifact_completeness": True,
        "scientific_source_fixed_point": _git_sha(),
        "config_sha256": _config_sha(config),
        "atlas_authorization": {
            "atlas_expansion_valid": bool(freeze["atlas_expansion_valid"]),
            "atlas_frozen": bool(freeze["atlas_frozen"]),
            "frozen_label_count": int(freeze["frozen_label_count"]),
            "coverage_fraction": float(freeze["coverage_fraction"]),
        },
        "dataset_authorizations": datasets,
        "student_authorizations": students,
        "student_scaling_claim_authorized": False,
        "student_scaling_claim_reason": "multi_seed_not_evaluated",
        "physics_label_authorization": {
            "theta_generation_complete": any(value["complete"] for value in datasets.values()),
            "tension_pilot_pass": bool(tension_pilot.get("tension_pilot_pass", False)),
            "tension_materialization_complete": bool(
                tension_full.get("tension_materialization_complete", False)
            ),
        },
        "formal_authorization": False,
        "deployment_authorization": False,
        "full_workspace_authorization": False,
    }
    _write_json(stage / "gate.json", gate)
    html_path = stage / "retry10_advisor_report.html"
    html_path.write_text(_report_html(gate), encoding="utf-8")
    pdf_status: dict[str, Any]
    try:
        from weasyprint import HTML

        pdf_path = stage / "retry10_advisor_report.pdf"
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        pdf_status = {"generated": True, "path": pdf_path.name, "sha256": sha256_file(pdf_path)}
    except Exception as error:  # report remains auditable when PDF runtime is absent
        pdf_status = {
            "generated": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }
    _write_json(stage / "report_render.json", pdf_status)
    artifacts = [
        {
            "path": str(path.relative_to(output_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and stage not in path.parents
    ]
    _write_json(
        stage / "artifact_manifest.json",
        {
            "schema_version": 1,
            "scientific_source_fixed_point": _git_sha(),
            "config_sha256": _config_sha(config),
            "artifacts": artifacts,
        },
    )
    completion = _seal_stage(output_root, config, "summary")
    completion.update(
        {
            "operational_completion": True,
            "gate_sha256": sha256_file(stage / "gate.json"),
            "artifact_manifest_sha256": sha256_file(stage / "artifact_manifest.json"),
        }
    )
    _write_json(stage / "completion_manifest.json", completion)
    return gate


STAGE_RUNNERS: dict[str, Callable[[Mapping[str, Any], Path, Path], dict[str, Any]]] = {
    "inventory": stage_inventory,
    "atlas_poc": stage_atlas_poc,
    "seed_datasets": stage_seed_datasets,
    "seed_students": stage_seed_students,
    "atlas_expansion": stage_atlas_expansion,
    "taxonomy_rescue": stage_taxonomy_rescue,
    "atlas_freeze": stage_atlas_freeze,
    "full_diagnostic": stage_full_diagnostic,
    "preflight_50k": stage_preflight_50k,
    "dataset_50k": stage_dataset_50k,
    "student_50k": stage_student_50k,
    "tension_pilot": stage_tension_pilot,
    "preflight_100k": stage_preflight_100k,
    "dataset_100k": stage_dataset_100k,
    "student_100k": stage_student_100k,
    "preflight_200k": stage_preflight_200k,
    "dataset_200k": stage_dataset_200k,
    "student_200k": stage_student_200k,
    "tension_materialization": stage_tension_materialization,
    "summary": stage_summary,
}


def _preflight_checkout() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=SOURCE_ROOT).returncode != 0:
        raise RuntimeError("retry10 run requires a clean scientific checkout")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=SOURCE_ROOT).returncode != 0:
        raise RuntimeError("retry10 run requires a clean scientific index")
    if subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=SOURCE_ROOT,
        stdout=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError("retry10 run must execute from a detached scientific fixed point")


def _ensure_identity(config: Mapping[str, Any], output_root: Path) -> None:
    identity = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "scientific_source_fixed_point": _git_sha(),
        "config_sha256": _config_sha(config),
    }
    path = output_root / "run_identity.json"
    if path.exists():
        if _read_json(path) != identity:
            raise RuntimeError("output root belongs to a different retry10 identity")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _write_json(path, identity)


def run(
    config: Mapping[str, Any], output_root: Path, *, only_stage: str | None = None
) -> dict[str, Any]:
    _preflight_checkout()
    _ensure_identity(config, output_root)
    project_root = project_root_from(SOURCE_ROOT)
    stages = STAGE_ORDER if only_stage is None else (only_stage,)
    for stage_name in stages:
        if _stage_is_complete(output_root, config, stage_name):
            continue
        if only_stage is not None:
            for predecessor in STAGE_ORDER[: STAGE_ORDER.index(stage_name)]:
                if not _stage_is_complete(output_root, config, predecessor):
                    raise RuntimeError(f"retry10 predecessor is not sealed: {predecessor}")
        STAGE_RUNNERS[stage_name](config, project_root, output_root)
    terminal = output_root / STAGE_DIRS["summary"] / "gate.json"
    if terminal.is_file():
        return _read_json(terminal)
    return _gate(output_root, stages[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=STAGE_ORDER)
    parser.add_argument("--validate-stage", choices=STAGE_ORDER)
    parser.add_argument("--label-worker-root")
    parser.add_argument("--tension-worker-root")
    parser.add_argument("--shard-id", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.label_worker_root is not None:
        if args.shard_id is None:
            raise ValueError("label worker requires --shard-id")
        _label_worker(config, Path(args.label_worker_root).resolve(), args.shard_id)
        return 0
    if args.tension_worker_root is not None:
        if args.shard_id is None:
            raise ValueError("tension worker requires --shard-id")
        _tension_worker(config, Path(args.tension_worker_root).resolve(), args.shard_id)
        return 0
    if args.output_root is None:
        raise ValueError("retry10 main execution requires --output-root")
    output_root = Path(args.output_root).resolve()
    if args.validate_stage is not None:
        return 0 if _stage_is_complete(output_root, config, args.validate_stage) else 1
    gate = run(config, output_root, only_stage=args.stage)
    print(json.dumps(gate, sort_keys=True, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
