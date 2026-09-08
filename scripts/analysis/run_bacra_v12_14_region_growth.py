#!/usr/bin/env python3
"""Run BACRA V12.14 region-grown canonical workspace experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = SOURCE_ROOT.parent.parent
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import run_bacra_v12 as v12
import run_bacra_v12_7_canonical_student as canonical_runner
from quasi_exp.teacher.bacra_exploratory_expansion import (
    fine_tune_exploratory_student,
    point_margin_deg,
)
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.gold_set_student import save_uncompiled_model
from quasi_exp.teacher.region_growth import (
    JACOBIAN_COLUMNS,
    RegionLabelPolicy,
    add_jacobians,
    aggregate_capability_voxels,
    beta_rms_deg,
    build_consistent_edges,
    canonicalize_seed_set,
    consistency_audit,
    continue_from_parent,
    farthest_point_seed_indices,
    plan_dense_targets,
    plan_parent_rows,
    reduce_parent_candidates,
    select_capability_region,
    select_sparse_targets,
    spatial_train_validation_split,
    voxel_indices,
)
from run_trajectory_canonical_teacher_v10 import (
    load_environment,
    runtime_fingerprint,
)


PROTOCOL_ID = (
    "branch-aware-canonical-region-atlas-v12.14-"
    "region-grown-canonical-workspace-dataset"
)
CLAIM_SCOPE = "simulation_exploratory_region_generalization"
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)
PREDICTED_BETA_COLUMNS = tuple(f"predicted_{name}" for name in BETA_COLUMNS)


def _source_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _output_root(
    config: Mapping[str, Any], project_root: Path, override: str | None
) -> Path:
    return (
        Path(override).resolve()
        if override
        else _source_path(project_root, str(config["output_root"]))
    )


def _environment(config: Mapping[str, Any], project_root: Path):
    return load_environment(
        project_root, _source_path(project_root, str(config["robot_config"]))
    )


def _policy(config: Mapping[str, Any]) -> RegionLabelPolicy:
    values = config["v12_14"]["region"]
    from quasi_exp.teacher.canonical import TeacherPolicy

    return RegionLabelPolicy(
        parent_distance_max_mm=float(values["parent_distance_max_mm"]),
        parent_attempt_count=int(values["parent_attempt_count"]),
        successful_parent_minimum=int(values["successful_parent_minimum"]),
        continuation_substep_mm=float(values["continuation_substep_mm"]),
        gold_residual_max_mm=float(values["gold_residual_max_mm"]),
        gold_candidate_gap_max_deg=float(
            values["gold_candidate_gap_max_deg"]
        ),
        silver_residual_max_mm=float(values["silver_residual_max_mm"]),
        silver_candidate_gap_max_deg=float(
            values["silver_candidate_gap_max_deg"]
        ),
        edge_distance_max_mm=float(values["edge_distance_max_mm"]),
        teacher_policy=TeacherPolicy(
            tracking_tolerance_mm=float(values["silver_residual_max_mm"]),
            max_corrector_iterations=80,
            safe_margin_repulsion_step_deg=0.0,
            beta_weights=(4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
        ),
    )


def _gate(
    path: Path,
    checks: Mapping[str, bool],
    *,
    semantics: str,
    **evidence: Any,
) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "claim_scope": CLAIM_SCOPE,
        "deployment_claim_gate_pass": False,
        "gate_semantics": str(semantics),
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, payload)
    return payload


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    v12._atomic_parquet(frame, path)


def _require_gate(output_root: Path, relative: str) -> dict[str, Any]:
    path = output_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required upstream gate is missing: {path}")
    payload = json.loads(path.read_text())
    if not payload["gate_pass"]:
        raise RuntimeError(f"upstream gate failed: {path}")
    return payload


def _source_files(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    values = config["v12_14"]
    return {
        name: _source_path(project_root, str(values[name]))
        for name in ("source_d3", "source_final_teacher", "source_capability_pool")
    }


def stage_protocol(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / "00_protocol"
    stage.mkdir(parents=True, exist_ok=False)
    sources = _source_files(config, project_root)
    expected = dict(config["v12_14"]["expected_source_sha256"])
    actual = {
        name: sha256_file(path) if path.is_file() else "missing"
        for name, path in sources.items()
    }
    status = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    config_snapshot = {
        key: value for key, value in config.items() if key != "config_path"
    }
    atomic_write_json(stage / "frozen_config.json", config_snapshot)
    implementation = {
        "region_growth": SOURCE_ROOT
        / "src/quasi_exp/teacher/region_growth.py",
        "exploratory_student": SOURCE_ROOT
        / "src/quasi_exp/teacher/bacra_exploratory_expansion.py",
        "runner": Path(__file__).resolve(),
        "config": Path(str(config["config_path"])).resolve(),
    }
    manifest = {
        "source_artifacts": {
            name: {
                "path": str(path),
                "sha256": actual[name],
                "expected_sha256": expected[name],
            }
            for name, path in sources.items()
        },
        "implementation": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in implementation.items()
        },
        "git_sha": git_sha,
        "source_worktree": str(SOURCE_ROOT),
    }
    atomic_write_json(stage / "source_manifest.json", manifest)
    atomic_write_json(
        stage / "runtime.json",
        {
            **runtime_fingerprint(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "python": sys.executable,
        },
    )
    return _gate(
        stage / "gate.json",
        {
            "source_files_exist": all(path.is_file() for path in sources.values()),
            "source_hashes_match": actual == expected,
            "source_worktree_clean": status.strip() == "",
            "protocol_claim_scope_is_exploratory": config["claim_scope"]
            == CLAIM_SCOPE,
            "deployment_claim_disabled": config["deployment_claim_gate_pass"]
            is False,
        },
        semantics="v12_14_clean_fixed_point_and_source_closure",
        source_sha256=actual,
        implementation_sha256={
            name: sha256_file(path) for name, path in implementation.items()
        },
        git_sha=git_sha,
        chart_scope=str(config["v12_14"].get("chart_scope", "chart_A_only")),
        capability_pool_beta_role="support_only_never_canonical_label",
    )


def _d3_conditioning(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(JACOBIAN_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"D3 lacks Jacobian columns: {missing}")
    jacobian = frame.loc[:, JACOBIAN_COLUMNS].to_numpy(dtype=float).reshape(
        -1, 3, 6
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return singular[:, 0] / np.maximum(singular[:, -1], 1.0e-12)


def stage_seed_cleanup(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_gate(output_root, "00_protocol/gate.json")
    stage = output_root / "01_seed_cleanup"
    stage.mkdir(parents=True, exist_ok=True)
    source = _source_files(config, project_root)["source_d3"]
    d3 = pd.read_parquet(source)
    policy = config["v12_14"]
    canonical_priority = policy.get("canonical_source_priority")
    if "dataset_source" not in d3 and canonical_priority is not None:
        if "chart_id" not in d3:
            raise ValueError(
                "chart-scoped seed data requires chart_id or dataset_source"
            )
        d3 = d3.copy()
        d3["dataset_source"] = d3["chart_id"].astype(str)
    clean, duplicates = canonicalize_seed_set(
        d3,
        source_priority=(
            None
            if canonical_priority is None
            else {
                str(key): int(value)
                for key, value in dict(canonical_priority).items()
            }
        ),
    )
    kappa = _d3_conditioning(clean)
    clean["kappa"] = kappa
    count = int(policy["region"]["seed_count"])
    indices = farthest_point_seed_indices(
        clean.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        count,
        seed=int(policy["seeds"]["seed_fps"]),
    )
    seeds = clean.iloc[indices].copy().reset_index(drop=True)
    seeds.insert(
        0, "node_id", [f"seed:{index:04d}" for index in range(len(seeds))]
    )
    chart_scope = str(policy.get("chart_scope", "chart_A_only"))
    seeds["node_origin"] = f"{chart_scope}_farthest_point_seed"
    duplicate_curve = (
        duplicates.groupby(
            ["duplicate_reason", "family_id", "selected_family_id"],
            dropna=False,
            sort=True,
        )
        .agg(
            duplicate_rows=("phase_idx", "size"),
            beta_rms_p95_deg=("beta_rms_to_selected_deg", lambda x: np.percentile(x, 95)),
            beta_rms_max_deg=("beta_rms_to_selected_deg", "max"),
        )
        .reset_index()
    )
    duplicate_label = duplicates.loc[
        :,
        [
            "family_id",
            "selected_family_id",
            "phase_idx",
            "duplicate_reason",
            "beta_rms_to_selected_deg",
        ],
    ]
    duplicate_curve.to_csv(stage / "duplicate_curve_report.csv", index=False)
    duplicate_label.to_csv(stage / "duplicate_label_report.csv", index=False)
    _atomic_parquet(clean, stage / "canonical_seed_dataset.parquet")
    _atomic_parquet(seeds, stage / "seed_nodes.parquet")
    cutoff = float(
        np.percentile(kappa, 95)
        * float(policy["region"]["conditioning_p95_multiplier"])
    )
    report = {
        "source_rows": int(len(d3)),
        "canonical_rows": int(len(clean)),
        "duplicate_rows_removed": int(len(d3) - len(clean)),
        "seed_count": int(len(seeds)),
        "clean_d3_kappa_p95": float(np.percentile(kappa, 95)),
        "conditioning_cutoff": cutoff,
    }
    atomic_write_json(stage / "cleanup_report.json", report)
    return _gate(
        stage / "gate.json",
        {
            "canonical_rows_match_registered": len(clean) == int(
                policy["expected_canonical_seed_rows"]
            ),
            "seed_count_match": len(seeds) == count,
            "task_phase_keys_unique": not clean.duplicated(
                [*XYZ_COLUMNS, "phase_idx"]
            ).any(),
            "all_labels_finite": np.isfinite(
                clean.loc[:, [*XYZ_COLUMNS, *BETA_COLUMNS]].to_numpy()
            ).all(),
        },
        semantics=f"{chart_scope}_seed_canonicalization",
        **report,
    )


def stage_capability_region(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    cleanup = _require_gate(output_root, "01_seed_cleanup/gate.json")
    stage = output_root / "02_capability_region"
    stage.mkdir(parents=True, exist_ok=True)
    values = config["v12_14"]
    pool = pd.read_parquet(
        _source_files(config, project_root)["source_capability_pool"],
        columns=[
            *XYZ_COLUMNS,
            *BETA_COLUMNS,
            "minimum_margin_deg",
            "kappa",
        ],
    )
    # The beta columns are deliberately discarded before aggregation.
    support = pool.loc[:, [*XYZ_COLUMNS, "minimum_margin_deg", "kappa"]]
    cutoff = float(cleanup["conditioning_cutoff"])
    voxel_map = aggregate_capability_voxels(
        support,
        voxel_size_mm=float(values["region"]["voxel_size_mm"]),
        conditioning_max=cutoff,
    )
    seeds = pd.read_parquet(output_root / "01_seed_cleanup/seed_nodes.parquet")
    selected, seed_support = select_capability_region(
        voxel_map,
        seeds.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        voxel_size_mm=float(values["region"]["voxel_size_mm"]),
        thickness_layers=int(values["region"]["thickness_layers"]),
    )
    _atomic_parquet(voxel_map, stage / "voxel_map.parquet")
    _atomic_parquet(selected, stage / "selected_region_voxels.parquet")
    seed_support.to_csv(stage / "seed_support.csv", index=False)
    components = (
        selected.groupby("region_wave", sort=True)
        .agg(voxel_count=("voxel_key", "size"))
        .reset_index()
    )
    components.to_csv(stage / "connected_components.csv", index=False)
    report = {
        "source_pool_rows": int(len(pool)),
        "conditioning_retained_rows": int(
            np.count_nonzero(np.isfinite(pool["kappa"]) & pool["kappa"].le(cutoff))
        ),
        "occupied_voxel_count": int(len(voxel_map)),
        "selected_region_voxel_count": int(len(selected)),
        "conditioning_cutoff": cutoff,
        "voxel_size_mm": float(values["region"]["voxel_size_mm"]),
        "adjacency": int(values["region"]["adjacency"]),
        "thickness_layers": int(values["region"]["thickness_layers"]),
        "capability_beta_used_as_label": False,
        "seed_support_distance_p95_mm": float(
            np.percentile(seed_support["support_distance_mm"], 95)
        ),
        "seed_support_distance_max_mm": float(
            seed_support["support_distance_mm"].max()
        ),
    }
    atomic_write_json(stage / "selected_region.json", report)
    return _gate(
        stage / "gate.json",
        {
            "formal_5mm_voxel_map_nonempty": len(voxel_map) > 0,
            "selected_region_covers_sparse_target": len(selected)
            >= int(values["region"]["sparse_target_count"]),
            "capability_beta_not_used_as_label": True,
            "seed_support_mapping_complete_and_finite": len(seed_support)
            == len(seeds)
            and np.isfinite(seed_support["support_distance_mm"]).all(),
        },
        semantics="branch_agnostic_capability_support_region",
        **report,
    )


def _worker_sparse(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    env = _environment(config, Path(args.project_root).resolve())
    policy = _policy(config)
    work = pd.read_parquet(args.worker_input)
    rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        result = continue_from_parent(
            env,
            np.asarray(
                [row.target_x_m, row.target_y_m, row.target_z_m], dtype=float
            ),
            np.asarray(
                [row.parent_x_m, row.parent_y_m, row.parent_z_m], dtype=float
            ),
            np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float),
            policy=policy,
        )
        rows.append(
            {
                "target_id": int(row.target_id),
                "parent_id": str(row.parent_id),
                "parent_rank": int(row.parent_rank),
                **result,
            }
        )
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(pd.DataFrame(rows), destination / "candidates.parquet")
    atomic_write_json(
        destination / "report.json",
        {
            "task_rows": int(len(work)),
            "success_rows": int(sum(bool(row["success"]) for row in rows)),
        },
    )
    return 0


def _slice_frame(
    frame: pd.DataFrame,
    *,
    stage: Path,
    task_prefix: str,
    worker: str,
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    python: Path,
    requested_workers: int,
) -> tuple[list[Path], dict[str, Any]]:
    if len(frame) == 0:
        return [], {"requested_workers": requested_workers, "effective_workers": 0}
    groups = np.array_split(
        np.arange(len(frame)),
        min(int(requested_workers), int(frame["target_id"].nunique())),
    )
    commands = []
    outputs = []
    for index, positions in enumerate(groups):
        if len(positions) == 0:
            continue
        task_id = f"{task_prefix}_{index:02d}"
        input_path = stage / "work" / task_id / "input.parquet"
        output_path = stage / "work" / task_id / "output"
        _atomic_parquet(frame.iloc[positions].copy(), input_path)
        commands.append(
            (
                task_id,
                [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    worker,
                    "--config",
                    str(config["config_path"]),
                    "--preset",
                    str(config["preset"]),
                    "--project-root",
                    str(project_root),
                    "--output-root",
                    str(output_root),
                    "--worker-input",
                    str(input_path),
                    "--worker-output",
                    str(output_path),
                ],
                stage / "logs" / f"{task_id}.log",
            )
        )
        outputs.append(output_path / "candidates.parquet")
    manifest = v12._run_subprocess_tasks(
        commands,
        requested_workers=int(requested_workers),
        manifest_path=stage / f"{task_prefix}_parallel_manifest.json",
    )
    return outputs, manifest


def _seed_anchor_frame(output_root: Path, voxel_size_mm: float) -> pd.DataFrame:
    seeds = pd.read_parquet(
        output_root / "01_seed_cleanup/canonical_seed_dataset.parquet"
    )
    seeds.insert(
        0,
        "node_id",
        [f"canonical:{index:06d}" for index in range(len(seeds))],
    )
    voxels = voxel_indices(seeds.loc[:, XYZ_COLUMNS].to_numpy(), voxel_size_mm)
    from quasi_exp.teacher.region_growth import pack_voxels

    seeds["voxel_key"] = pack_voxels(voxels)
    seeds["quality_class"] = seeds.get("quality_class", "Gold")
    seeds["candidate_gap_max_deg"] = 0.0
    seeds["node_origin"] = "canonical_chart_anchor"
    return seeds


def stage_sparse_growth(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "02_capability_region/gate.json")
    stage = output_root / "03_sparse_region_growth"
    stage.mkdir(parents=True, exist_ok=True)
    values = config["v12_14"]
    region_values = values["region"]
    voxel_size = float(region_values["voxel_size_mm"])
    canonical_anchors = _seed_anchor_frame(output_root, voxel_size)
    region_seeds = pd.read_parquet(
        output_root / "01_seed_cleanup/seed_nodes.parquet"
    )
    seed_voxels = voxel_indices(
        region_seeds.loc[:, XYZ_COLUMNS].to_numpy(), voxel_size
    )
    from quasi_exp.teacher.region_growth import pack_voxels

    region_seeds["voxel_key"] = pack_voxels(seed_voxels)
    region_seeds["candidate_gap_max_deg"] = 0.0
    region_seeds["candidate_gap_p95_deg"] = 0.0
    region_seeds["quality_class"] = region_seeds["quality_class"].astype(str)
    region_seeds["node_origin"] = (
        f"{values.get('chart_scope', 'chart_A_only')}_farthest_point_seed"
    )
    region = pd.read_parquet(
        output_root / "02_capability_region/selected_region_voxels.parquet"
    )
    targets = select_sparse_targets(
        region,
        region_seeds.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        count=int(region_values["sparse_target_count"]),
        seed=int(values["seeds"]["sparse_target"]),
        parent_distance_max_mm=float(region_values["parent_distance_max_mm"]),
        required_parent_support_count=int(
            region_values["successful_parent_minimum"]
        ),
        initial_parent_xyz_m=canonical_anchors.loc[
            :, XYZ_COLUMNS
        ].to_numpy(dtype=float),
    )
    _atomic_parquet(targets, stage / "sparse_target_inventory.parquet")
    anchors = canonical_anchors.copy()
    accepted_parts: list[pd.DataFrame] = []
    rejected_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    policy = _policy(config)
    for growth_batch in sorted(map(int, targets["growth_batch"].unique())):
        wave_targets = targets.loc[
            targets["growth_batch"].eq(growth_batch)
        ].reset_index(
            drop=True
        )
        parents = plan_parent_rows(wave_targets, anchors, policy=policy)
        if len(parents) == 0:
            rejected = wave_targets.copy()
            rejected["reason"] = "no_parent_within_registered_distance"
            rejected_parts.append(rejected)
            continue
        outputs, manifest = _slice_frame(
            parents,
            stage=stage,
            task_prefix=f"batch_{growth_batch:02d}",
            worker="sparse",
            config=config,
            project_root=project_root,
            output_root=output_root,
            python=python,
            requested_workers=int(values["parallel"]["cpu_workers"]),
        )
        manifests.append(manifest)
        candidates = pd.concat(
            [pd.read_parquet(path) for path in outputs], ignore_index=True
        )
        candidates["consensus_selected"] = False
        accepted_rows = []
        rejected_rows = []
        target_lookup = wave_targets.set_index("target_id")
        for target_id in wave_targets["target_id"]:
            group = candidates.loc[candidates["target_id"].eq(int(target_id))]
            reduced = reduce_parent_candidates(group, policy=policy)
            for candidate_row in reduced.get("consensus_candidate_rows", []):
                candidates.loc[int(candidate_row), "consensus_selected"] = True
            base = target_lookup.loc[int(target_id)].to_dict()
            row = {
                "node_id": f"region:{int(target_id):06d}",
                "target_id": int(target_id),
                **base,
                **{
                    key: value
                    for key, value in reduced.items()
                    if key
                    not in {
                        "selected_candidate_row",
                        "consensus_candidate_rows",
                    }
                },
            }
            (accepted_rows if reduced["accepted"] else rejected_rows).append(row)
        accepted = pd.DataFrame(accepted_rows)
        rejected = pd.DataFrame(rejected_rows)
        if len(accepted):
            accepted_parts.append(accepted)
            anchors = pd.concat([anchors, accepted], ignore_index=True, sort=False)
        if len(rejected):
            rejected_parts.append(rejected)
        candidate_parts.append(candidates)
    accepted = (
        pd.concat(accepted_parts, ignore_index=True)
        if accepted_parts
        else pd.DataFrame()
    )
    rejected = (
        pd.concat(rejected_parts, ignore_index=True)
        if rejected_parts
        else pd.DataFrame()
    )
    candidates = (
        pd.concat(candidate_parts, ignore_index=True)
        if candidate_parts
        else pd.DataFrame()
    )
    graph_nodes = pd.concat(
        [region_seeds, accepted], ignore_index=True, sort=False
    )
    if len(graph_nodes):
        edges, component = build_consistent_edges(
            graph_nodes,
            distance_max_mm=float(region_values["edge_distance_max_mm"]),
            gap_max_deg=float(region_values["silver_candidate_gap_max_deg"]),
        )
    else:
        edges, component = pd.DataFrame(), accepted
    accepted_ids = set(map(str, component.get("node_id", [])))
    if len(accepted):
        outside = accepted.loc[~accepted["node_id"].astype(str).isin(accepted_ids)].copy()
        if len(outside):
            outside["reason"] = "outside_largest_consistent_component"
            rejected = pd.concat([rejected, outside], ignore_index=True, sort=False)
    _atomic_parquet(component, stage / "sparse_nodes.parquet")
    _atomic_parquet(edges, stage / "sparse_edges.parquet")
    _atomic_parquet(candidates, stage / "multiparent_candidates.parquet")
    _atomic_parquet(rejected, stage / "rejected_nodes.parquet")
    prefix = targets["target_id"].lt(int(region_values["sparse_prefix_count"]))
    prefix_accept = int(
        component["target_id"].isin(targets.loc[prefix, "target_id"]).sum()
    ) if len(component) else 0
    component_new = component.loc[
        component["node_id"].astype(str).str.startswith("region:")
    ]
    report = {
        "target_count": int(len(targets)),
        "accepted_before_component": int(len(accepted)),
        "largest_component_node_count": int(len(component)),
        "largest_component_new_node_count": int(len(component_new)),
        "largest_component_seed_node_count": int(
            len(component) - len(component_new)
        ),
        "rejected_count": int(len(rejected)),
        "acceptance_rate": float(len(accepted) / len(targets)),
        "largest_component_rate": float(len(component_new) / len(targets)),
        "prefix_target_count": int(prefix.sum()),
        "prefix_component_accept_count": prefix_accept,
        "target_stratum_counts": {
            str(key): int(value)
            for key, value in targets["target_stratum"].value_counts().items()
        },
        "selection_role_counts": {
            str(key): int(value)
            for key, value in targets["selection_role"].value_counts().items()
        },
        "growth_batch_count": int(targets["growth_batch"].nunique()),
        "planned_farthest_parent_distance_max_mm": (
            float(
                targets["planned_farthest_parent_distance_mm"]
                .dropna()
                .max()
            )
            if targets["planned_farthest_parent_distance_mm"].notna().any()
            else None
        ),
        "multiparent_candidate_gap_p95_deg": (
            float(
                np.percentile(
                    component_new["candidate_gap_max_deg"], 95
                )
            )
            if len(component_new)
            else None
        ),
        "wave_parallel_evidence": manifests,
        "policy_fingerprint": policy.fingerprint,
    }
    atomic_write_json(stage / "growth_report.json", report)
    return _gate(
        stage / "gate.json",
        {
            "sparse_acceptance_rate": report["acceptance_rate"]
            >= float(region_values["required_sparse_acceptance_rate"]),
            "largest_component_minimum": len(component)
            >= int(region_values["required_component_nodes"]),
            "multiparent_consistency_p95": report[
                "multiparent_candidate_gap_p95_deg"
            ]
            is not None
            and report["multiparent_candidate_gap_p95_deg"]
            <= float(config["v12_14"]["consistency"]["multiparent_p95_deg"]),
            "no_capability_pool_beta_labels": True,
        },
        semantics="multi_parent_region_growth_admission",
        **report,
    )


def stage_consistency(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    sparse_gate = _require_gate(output_root, "03_sparse_region_growth/gate.json")
    stage = output_root / "04_consistency_audit"
    stage.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(output_root / "03_sparse_region_growth/sparse_nodes.parquet")
    edges = pd.read_parquet(output_root / "03_sparse_region_growth/sparse_edges.parquet")
    candidates = pd.read_parquet(
        output_root / "03_sparse_region_growth/multiparent_candidates.parquet"
    )
    values = config["v12_14"]["consistency"]
    triangles, paths, report = consistency_audit(
        nodes,
        edges,
        candidates,
        triangle_count=int(values["triangle_sample_count"]),
        two_path_count=int(values["two_path_sample_count"]),
        seed=int(config["v12_14"]["seeds"]["consistency"]),
        environment=_environment(config, project_root),
        policy=_policy(config),
    )
    triangles.to_csv(stage / "loop_consistency.csv", index=False)
    paths.to_csv(stage / "two_path_consistency.csv", index=False)
    conflicts = edges.loc[~edges["accepted"].astype(bool)].copy()
    _atomic_parquet(conflicts, stage / "conflict_voxels.parquet")
    atomic_write_json(stage / "audit_report.json", report)
    return _gate(
        stage / "gate.json",
        {
            "triangle_sample_available": len(triangles)
            >= int(values["triangle_sample_count"]),
            "triangle_continuation_all_successful": report["triangle_failed"]
            == 0,
            "two_path_sample_available": len(paths)
            >= int(values["two_path_sample_count"]),
            "triangle_p95": report["triangle_p95_deg"] is not None
            and report["triangle_p95_deg"]
            <= float(values["triangle_p95_deg"]),
            "triangle_max": report["triangle_max_deg"] is not None
            and report["triangle_max_deg"]
            <= float(values["triangle_max_deg"]),
            "two_path_p95": report["two_path_p95_deg"] is not None
            and report["two_path_p95_deg"]
            <= float(values["two_path_p95_deg"]),
            "upstream_multiparent_p95": sparse_gate[
                "multiparent_candidate_gap_p95_deg"
            ]
            is not None
            and sparse_gate["multiparent_candidate_gap_p95_deg"]
            <= float(values["multiparent_p95_deg"]),
        },
        semantics="local_predictor_corrector_path_and_loop_consistency",
        **report,
        conflict_edge_count=int(len(conflicts)),
    )


def _dense_parent_frame(plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    anchor_count = len(
        [column for column in plan.columns if column.startswith("anchor_") and column.endswith("_node_id")]
    )
    for target in plan.itertuples(index=False):
        for rank in range(anchor_count):
            rows.append(
                {
                    "target_id": int(target.target_id),
                    "target_x_m": float(target.x_m),
                    "target_y_m": float(target.y_m),
                    "target_z_m": float(target.z_m),
                    "parent_rank": rank,
                    "parent_id": str(getattr(target, f"anchor_{rank}_node_id")),
                    "parent_x_m": float(getattr(target, f"anchor_{rank}_x_m")),
                    "parent_y_m": float(getattr(target, f"anchor_{rank}_y_m")),
                    "parent_z_m": float(getattr(target, f"anchor_{rank}_z_m")),
                    "parent_distance_mm": float(
                        getattr(target, f"anchor_{rank}_distance_mm")
                    ),
                    **{
                        name: float(getattr(target, f"anchor_{rank}_{name}"))
                        for name in BETA_COLUMNS
                    },
                }
            )
        if bool(target.tetrahedral_prediction_available):
            rows.append(
                {
                    "target_id": int(target.target_id),
                    "target_x_m": float(target.x_m),
                    "target_y_m": float(target.y_m),
                    "target_z_m": float(target.z_m),
                    "parent_rank": anchor_count,
                    "parent_id": f"tetrahedron:{int(target.tetrahedron_idx)}",
                    "parent_x_m": float(target.x_m),
                    "parent_y_m": float(target.y_m),
                    "parent_z_m": float(target.z_m),
                    "parent_distance_mm": 0.0,
                    **{
                        name: float(getattr(target, f"tetrahedral_{name}"))
                        for name in BETA_COLUMNS
                    },
                }
            )
    return pd.DataFrame(rows)


def stage_dense(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "04_consistency_audit/gate.json")
    stage = output_root / "05_dense_region"
    stage.mkdir(parents=True, exist_ok=True)
    values = config["v12_14"]
    region = pd.read_parquet(
        output_root / "02_capability_region/selected_region_voxels.parquet"
    )
    sparse = pd.read_parquet(output_root / "03_sparse_region_growth/sparse_nodes.parquet")
    support_distance, _support_index = cKDTree(
        sparse.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    ).query(region.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), k=2)
    region["second_anchor_support_distance_mm"] = (
        np.asarray(support_distance)[:, 1] * 1000.0
    )
    supported_region = region.loc[
        region["second_anchor_support_distance_mm"].le(
            float(values["region"]["parent_distance_max_mm"])
        )
    ].copy()
    dense_policy = values["dense"]
    formal = int(dense_policy["formal_accepted_count"])
    attempt_count = formal * int(dense_policy["attempt_multiplier"])
    if attempt_count < 1:
        raise RuntimeError(
            "sparse chart has no voxel with two registered local anchors"
        )
    plan = plan_dense_targets(
        supported_region,
        sparse,
        attempt_count=attempt_count,
        seed=int(values["seeds"]["dense_target"]),
        voxel_size_mm=float(values["region"]["voxel_size_mm"]),
        parent_distance_max_mm=float(
            values["region"]["parent_distance_max_mm"]
        ),
        minimum_anchor_count=int(
            values["dense"]["independent_predictor_minimum"]
        ),
    )
    _atomic_parquet(plan, stage / "dense_target_inventory.parquet")
    parents = _dense_parent_frame(plan)
    outputs, manifest = _slice_frame(
        parents,
        stage=stage,
        task_prefix="dense",
        worker="sparse",
        config=config,
        project_root=project_root,
        output_root=output_root,
        python=python,
        requested_workers=int(values["parallel"]["cpu_workers"]),
    )
    candidates = pd.concat(
        [pd.read_parquet(path) for path in outputs], ignore_index=True
    )
    plan_lookup = plan.set_index("target_id")
    policy = _policy(config)
    accepted_rows = []
    rejected_rows = []
    for target_id, group in candidates.groupby("target_id", sort=True):
        reduced = reduce_parent_candidates(
            group, policy=policy, required_successes=2
        )
        base = plan_lookup.loc[int(target_id)].to_dict()
        row = {
            "node_id": f"dense:{int(target_id):07d}",
            "target_id": int(target_id),
            **base,
            **{
                key: value
                for key, value in reduced.items()
                if key
                not in {
                    "selected_candidate_row",
                    "consensus_candidate_rows",
                }
            },
        }
        (accepted_rows if reduced["accepted"] else rejected_rows).append(row)
    accepted = pd.DataFrame(accepted_rows)
    rejected = pd.DataFrame(rejected_rows)
    accepted = accepted.sort_values("target_id", kind="stable")
    prefix_count = int(dense_policy["prefix_accepted_count"])
    formal_frame = accepted.iloc[:formal].copy()
    _atomic_parquet(
        formal_frame.iloc[:prefix_count], stage / "dense_region_50k.parquet"
    )
    _atomic_parquet(formal_frame, stage / "dense_region_100k.parquet")
    _atomic_parquet(candidates, stage / "dense_multiparent_candidates.parquet")
    _atomic_parquet(rejected, stage / "dense_rejected.parquet")
    report = {
        "attempt_count": int(len(plan)),
        "supported_region_voxel_count": int(len(supported_region)),
        "accepted_count": int(len(accepted)),
        "formal_saved_count": int(len(formal_frame)),
        "prefix_saved_count": int(min(prefix_count, len(formal_frame))),
        "acceptance_rate": float(len(accepted) / len(plan)),
        "candidate_gap_p95_deg": (
            float(np.percentile(accepted["candidate_gap_max_deg"], 95))
            if len(accepted)
            else None
        ),
        "parallel_evidence": manifest,
        "tetrahedral_target_count": int(
            plan["tetrahedral_prediction_available"].sum()
        ),
        "nearest_anchor_predictors_per_target": 4,
        "tetrahedral_predictor_used_when_available": True,
        "continuous_points_sampled_within_capability_voxels": True,
    }
    atomic_write_json(stage / "dense_acceptance_report.json", report)
    return _gate(
        stage / "gate.json",
        {
            "dense_prefix_complete": len(formal_frame) >= prefix_count,
            "dense_formal_complete": len(formal_frame) >= formal,
            "dense_acceptance_rate": report["acceptance_rate"]
            >= float(values["region"]["required_sparse_acceptance_rate"]),
            "dense_multiparent_p95": report["candidate_gap_p95_deg"]
            is not None
            and report["candidate_gap_p95_deg"]
            <= float(values["consistency"]["multiparent_p95_deg"]),
        },
        semantics="dense_region_multi_anchor_admission",
        **report,
    )


def _worker_jacobian(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    env = _environment(config, Path(args.project_root).resolve())
    frame = pd.read_parquet(args.worker_input)
    output = add_jacobians(frame, env)
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output, destination / "candidates.parquet")
    atomic_write_json(destination / "report.json", {"rows": int(len(output))})
    return 0


def _parallel_add_jacobians(
    frame: pd.DataFrame,
    *,
    name: str,
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    stage: Path,
    python: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sliced = frame.copy()
    sliced["target_id"] = np.arange(len(sliced), dtype=np.int64)
    outputs, manifest = _slice_frame(
        sliced,
        stage=stage,
        task_prefix=f"jacobian_{name}",
        worker="jacobian",
        config=config,
        project_root=project_root,
        output_root=output_root,
        python=python,
        requested_workers=int(config["v12_14"]["parallel"]["cpu_workers"]),
    )
    output = pd.concat([pd.read_parquet(path) for path in outputs], ignore_index=True)
    output = output.sort_values("target_id", kind="stable").drop(columns="target_id")
    return output.reset_index(drop=True), manifest


def _deterministic_sample(
    frame: pd.DataFrame, count: int, *, seed: int
) -> pd.DataFrame:
    if count > len(frame):
        raise ValueError(f"requested {count} rows from only {len(frame)}")
    order = np.random.default_rng(int(seed)).permutation(len(frame))[:count]
    return frame.iloc[order].copy().reset_index(drop=True)


def _prepare_dataset(
    frame: pd.DataFrame,
    *,
    name: str,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = config["v12_14"]["dataset"]
    output = frame.copy()
    output["dataset_version"] = str(name)
    weights = dict(values["quality_weights"])
    output["sample_weight"] = output["quality_class"].map(weights)
    if output["sample_weight"].isna().any():
        raise ValueError(f"{name} contains an unregistered quality class")
    split, report = spatial_train_validation_split(
        output,
        macro_voxel_mm=float(values["macro_voxel_mm"]),
        validation_fraction=float(values["validation_fraction"]),
        buffer_mm=float(values["split_buffer_mm"]),
        seed=int(seed),
    )
    return split, report


def stage_dataset(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "05_dense_region/gate.json")
    stage = output_root / "06_datasets"
    stage.mkdir(parents=True, exist_ok=True)
    values = config["v12_14"]
    policy = values["dataset"]
    clean = pd.read_parquet(
        output_root / "01_seed_cleanup/canonical_seed_dataset.parquet"
    )
    clean = clean.copy()
    clean["quality_class"] = clean["quality_class"].replace(
        {"Gold": "Gold", "Silver": "Silver"}
    )
    clean["sampling_bucket"] = "old"
    dense = pd.read_parquet(output_root / "05_dense_region/dense_region_100k.parquet")
    maximum_wave = int(dense["region_wave"].max())
    dense["sampling_bucket"] = np.where(
        dense["region_wave"].ge(maximum_wave), "boundary", "interior"
    )
    dense, jacobian_manifest = _parallel_add_jacobians(
        dense,
        name="dense100k",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=stage,
        python=python,
    )
    base_seed = int(values["seeds"]["dense_target"])
    datasets = {
        "R0": _deterministic_sample(
            clean, int(policy["R0_rows"]), seed=base_seed
        ),
        "R1_equal": _deterministic_sample(
            dense, int(policy["R1_equal_rows"]), seed=base_seed + 1
        ),
        "R1_full": _deterministic_sample(
            dense, int(policy["R1_full_rows"]), seed=base_seed + 2
        ),
    }
    old = _deterministic_sample(
        clean, int(policy["R2_old_rows"]), seed=base_seed + 3
    )
    region = _deterministic_sample(
        dense, int(policy["R2_region_rows"]), seed=base_seed + 4
    )
    datasets["R2"] = pd.concat([old, region], ignore_index=True, sort=False)
    manifests = []
    for offset, (name, frame) in enumerate(datasets.items()):
        prepared, split_report = _prepare_dataset(
            frame, name=name, config=config, seed=base_seed + 100 + offset
        )
        path = stage / f"{name}_dataset.parquet"
        _atomic_parquet(prepared, path)
        manifest = {
            "dataset_version": name,
            "row_count": int(len(prepared)),
            "source_counts": {
                str(key): int(value)
                for key, value in prepared["sampling_bucket"].value_counts().items()
            },
            "quality_counts": {
                str(key): int(value)
                for key, value in prepared["quality_class"].value_counts().items()
            },
            "split": split_report,
            "sha256": sha256_file(path),
            "claim_scope": CLAIM_SCOPE,
        }
        atomic_write_json(stage / f"{name}_manifest.json", manifest)
        manifests.append(manifest)
    pd.DataFrame(
        [
            {
                "dataset_version": item["dataset_version"],
                **item["split"],
            }
            for item in manifests
        ]
    ).to_csv(stage / "spatial_split_manifest.csv", index=False)
    return _gate(
        stage / "gate.json",
        {
            "R0_complete": manifests[0]["row_count"] == int(policy["R0_rows"]),
            "R1_equal_complete": manifests[1]["row_count"]
            == int(policy["R1_equal_rows"]),
            "R1_full_complete": manifests[2]["row_count"]
            == int(policy["R1_full_rows"]),
            "R2_complete": manifests[3]["row_count"]
            == int(policy["R2_old_rows"]) + int(policy["R2_region_rows"]),
            "all_splits_nonempty": all(
                item["split"]["train_rows"] > 0
                and item["split"]["validation_rows"] > 0
                for item in manifests
            ),
        },
        semantics="voxel_balanced_region_dataset_assembly",
        dataset_manifests=manifests,
        jacobian_parallel_evidence=jacobian_manifest,
    )


def _gpu_probe(python: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json,tensorflow as tf;"
                "g=tf.config.list_physical_devices('GPU');"
                "print(json.dumps({'tensorflow':tf.__version__,"
                "'gpus':[x.name for x in g]}))"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    lines = process.stdout.strip().splitlines()
    payload = json.loads(lines[-1]) if lines else {"gpus": []}
    return {
        "return_code": int(process.returncode),
        "tensorflow": payload.get("tensorflow"),
        "gpus": payload.get("gpus", []),
        "gpu_available": bool(process.returncode == 0 and payload.get("gpus")),
        "stderr_tail": process.stderr[-2000:],
    }


def _prediction_metrics(environment, beta: np.ndarray, target: np.ndarray) -> dict[str, float]:
    achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    error = np.linalg.norm(achieved - np.asarray(target, dtype=float), axis=1) * 1000.0
    margin = point_margin_deg(beta, environment.bounds)
    return {
        "fk_p50_mm": float(np.percentile(error, 50)),
        "fk_p95_mm": float(np.percentile(error, 95)),
        "fk_max_mm": float(np.max(error)),
        "minimum_joint_margin_deg": float(np.min(margin)),
        "actual_bounds": bool(np.min(margin) >= -1.0e-9),
    }


def _sampling_weights(
    config: Mapping[str, Any], dataset_version: str, frame: pd.DataFrame
) -> dict[str, float]:
    dataset_version = str(dataset_version).removesuffix("_enriched")
    registered = dict(
        config["v12_14"]["dataset"]["batch_proportions"][dataset_version]
    )
    observed = set(map(str, frame["sampling_bucket"].unique()))
    available = {key: float(value) for key, value in registered.items() if key in observed}
    total = sum(available.values())
    if set(available) != observed or total <= 0.0:
        raise ValueError(
            f"batch proportions do not cover {dataset_version}: {sorted(observed)}"
        )
    return {key: value / total for key, value in available.items()}


def _worker_train(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("V12.14 Student training requires a visible GPU")
    seed = int(args.seed)
    dataset = pd.read_parquet(args.dataset)
    train = dataset.loc[dataset["split"].eq("train")].copy()
    validation = dataset.loc[dataset["split"].eq("validation")].copy()
    geometry = canonical_runner._geometry(config, project_root)
    initial_model = (
        Path(args.initial_model).resolve()
        if args.initial_model
        else (
            _source_path(
                project_root, config["v12_14"]["source_v12_11_root"]
            )
            / f"01_training/seed_{seed}/model.keras"
        )
    )
    model = tf.keras.models.load_model(initial_model, compile=False)
    training = config["v12_14"]["training"]
    model, history, report = fine_tune_exploratory_student(
        model,
        train,
        validation,
        geometry=geometry,
        seed=seed,
        row_loss_weight=float(training["row_loss_weight"]),
        learning_rate=float(training["learning_rate"]),
        batch_size=int(training["batch_size"]),
        max_steps=int(training["max_optimizer_steps"]),
        validation_interval=int(training["validation_interval_steps"]),
        patience_intervals=int(training["patience_intervals"]),
        sampling_weights=_sampling_weights(
            config, str(args.dataset_version), train
        ),
    )
    prediction = np.asarray(
        model.predict(
            validation.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
            batch_size=2048,
            verbose=0,
        ),
        dtype=float,
    )
    report.update(
        {
            "dataset_version": str(args.dataset_version),
            "initial_model": str(initial_model),
            "device": devices[0].name,
            "validation_metrics": _prediction_metrics(
                _environment(config, project_root),
                prediction,
                validation.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
            ),
        }
    )
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    save_uncompiled_model(model, destination / "model.keras")
    history.to_csv(destination / "history.csv", index=False)
    report["model_sha256"] = sha256_file(destination / "model.keras")
    atomic_write_json(destination / "report.json", report)
    return 0


def _training_commands(
    dataset_versions: Sequence[str],
    *,
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    stage: Path,
    python: Path,
    dataset_paths: Mapping[str, Path] | None = None,
    initial_model_paths: Mapping[int, Path] | None = None,
) -> list[tuple[str, list[str], Path]]:
    commands = []
    for dataset_version in dataset_versions:
        for seed in config["v12_14"]["seeds"]["training"]:
            task_id = f"{dataset_version}_seed{int(seed)}"
            commands.append(
                (
                    task_id,
                    [
                        str(python),
                        str(Path(__file__).resolve()),
                        "--worker",
                        "train",
                        "--config",
                        str(config["config_path"]),
                        "--preset",
                        str(config["preset"]),
                        "--project-root",
                        str(project_root),
                        "--output-root",
                        str(output_root),
                        "--dataset",
                        str(output_root / f"06_datasets/{dataset_version}_dataset.parquet"),
                        "--dataset-version",
                        dataset_version,
                        "--seed",
                        str(seed),
                        "--worker-output",
                        str(stage / task_id),
                    ],
                    stage / "logs" / f"{task_id}.log",
                )
            )
            if dataset_paths is not None:
                dataset_position = commands[-1][1].index("--dataset") + 1
                commands[-1][1][dataset_position] = str(
                    dataset_paths[dataset_version]
                )
            if initial_model_paths is not None:
                commands[-1][1].extend(
                    [
                        "--initial-model",
                        str(initial_model_paths[int(seed)]),
                    ]
                )
    return commands


def stage_train(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "06_datasets/gate.json")
    stage = output_root / "07_students"
    stage.mkdir(parents=True, exist_ok=True)
    probe = _gpu_probe(python)
    if not probe["gpu_available"]:
        return _gate(
            stage / "gate.json",
            {"gpu_visible": False},
            semantics="single_gpu_serial_base_student_training",
            gpu_probe=probe,
            stopped_before_training=True,
        )
    versions = ("R0", "R1_equal", "R1_full", "R2")
    commands = _training_commands(
        versions,
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=stage,
        python=python,
    )
    parallel = v12._run_subprocess_tasks(
        commands,
        requested_workers=1,
        manifest_path=stage / "training_parallel_manifest.json",
    )
    rows = []
    for task_id, _command, _log in commands:
        report = json.loads((stage / task_id / "report.json").read_text())
        rows.append({"task_id": task_id, **report})
    metrics = pd.json_normalize(rows)
    metrics.to_csv(stage / "metrics_per_seed.csv", index=False)
    eligible = metrics.loc[metrics["dataset_version"].isin(["R1_full", "R2"])]
    ranked = (
        eligible.groupby("dataset_version", sort=True)[
            "validation_metrics.fk_p95_mm"
        ]
        .mean()
        .sort_values(kind="stable")
    )
    selected = str(ranked.index[0])
    selection = {
        "selection_scope": ["R1_full", "R2"],
        "criterion": "mean_validation_fk_p95_mm",
        "selected_dataset_version": selected,
        "mean_validation_fk_p95_mm": {
            str(key): float(value) for key, value in ranked.items()
        },
        "model_roots": {
            str(int(seed)): str(stage / f"{selected}_seed{int(seed)}")
            for seed in config["v12_14"]["seeds"]["training"]
        },
    }
    atomic_write_json(stage / "selected_model.json", selection)
    return _gate(
        stage / "gate.json",
        {
            "gpu_visible": True,
            "all_twelve_base_tasks_complete": len(metrics) == 12,
            "all_models_saved": all(
                (stage / task_id / "model.keras").is_file()
                for task_id, _command, _log in commands
            ),
            "selection_limited_to_registered_region_datasets": selected
            in {"R1_full", "R2"},
        },
        semantics="single_gpu_serial_base_student_training",
        gpu_probe=probe,
        selected_dataset_version=selected,
        parallel_evidence=parallel,
    )


def _random_targets_from_region(
    dense: pd.DataFrame,
    *,
    attempt_count: int,
    voxel_size_mm: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(dense), size=int(attempt_count))
    center = dense.iloc[selected].loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    jitter = rng.uniform(
        -0.45 * float(voxel_size_mm) / 1000.0,
        0.45 * float(voxel_size_mm) / 1000.0,
        size=center.shape,
    )
    target = center + jitter
    output = pd.DataFrame(target, columns=XYZ_COLUMNS)
    output.insert(0, "target_id", np.arange(len(output), dtype=np.int64))
    output["source_dense_node_id"] = dense.iloc[selected]["node_id"].to_numpy()
    output["source_voxel_key"] = dense.iloc[selected]["voxel_key"].to_numpy()
    return output


def _label_targets(
    targets: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    name: str,
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    stage: Path,
    python: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    policy = _policy(config)
    parents = plan_parent_rows(targets, anchors, policy=policy)
    outputs, manifest = _slice_frame(
        parents,
        stage=stage,
        task_prefix=f"label_{name}",
        worker="sparse",
        config=config,
        project_root=project_root,
        output_root=output_root,
        python=python,
        requested_workers=int(config["v12_14"]["parallel"]["cpu_workers"]),
    )
    candidates = (
        pd.concat([pd.read_parquet(path) for path in outputs], ignore_index=True)
        if outputs
        else pd.DataFrame()
    )
    lookup = targets.set_index("target_id")
    accepted_rows = []
    rejected_rows = []
    for target_id in targets["target_id"]:
        group = (
            candidates.loc[candidates["target_id"].eq(int(target_id))]
            if len(candidates)
            else pd.DataFrame(columns=["success", "residual_mm", *BETA_COLUMNS])
        )
        reduced = reduce_parent_candidates(
            group, policy=policy, required_successes=2
        )
        row = {
            **lookup.loc[int(target_id)].to_dict(),
            "target_id": int(target_id),
            **{
                key: value
                for key, value in reduced.items()
                if key
                not in {
                    "selected_candidate_row",
                    "consensus_candidate_rows",
                }
            },
        }
        (accepted_rows if reduced["accepted"] else rejected_rows).append(row)
    return (
        pd.DataFrame(accepted_rows),
        pd.DataFrame(rejected_rows),
        manifest,
    )


class _ResidualCompositePredictor:
    def __init__(self, base: Any, region: Any, alpha: float) -> None:
        self.base = base
        self.region = region
        self.alpha = float(alpha)

    def predict(
        self, target: np.ndarray, *, batch_size: int, verbose: int
    ) -> np.ndarray:
        base = np.asarray(
            self.base.predict(target, batch_size=batch_size, verbose=verbose),
            dtype=float,
        )
        region = np.asarray(
            self.region.predict(target, batch_size=batch_size, verbose=verbose),
            dtype=float,
        )
        return base + self.alpha * (region - base)


def _load_models(model_roots: Mapping[str, str]) -> dict[int, Any]:
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("V12.14 evaluation requires a visible GPU")
    models = {}
    for seed, root_value in model_roots.items():
        root = Path(root_value)
        composite_path = root / "composite_model.json"
        if composite_path.is_file():
            composite = json.loads(composite_path.read_text())
            if composite.get("model_mode") != "output_residual_composite":
                raise ValueError(
                    f"unsupported composite model mode: {composite.get('model_mode')}"
                )
            models[int(seed)] = _ResidualCompositePredictor(
                tf.keras.models.load_model(
                    composite["base_model"], compile=False
                ),
                tf.keras.models.load_model(
                    composite["region_model"], compile=False
                ),
                float(composite["alpha"]),
            )
        else:
            models[int(seed)] = tf.keras.models.load_model(
                root / "model.keras", compile=False
            )
    return models


def _family_seed_gate(
    metrics: pd.DataFrame,
    *,
    family_column: str,
    pass_column: str,
    required_seed_passes: int,
) -> pd.Series:
    return (
        metrics.groupby(family_column, sort=True)[pass_column]
        .sum()
        .ge(int(required_seed_passes))
    )


def _evaluate_model_set(
    reference: pd.DataFrame,
    *,
    model_roots: Mapping[str, str],
    environment,
    family_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = reference.reset_index(drop=True)
    models = _load_models(model_roots)
    target = reference.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    detail_parts = []
    metric_rows = []
    groups = (
        [(None, reference.index.to_numpy())]
        if family_column is None
        else [
            (family, group.index.to_numpy())
            for family, group in reference.groupby(family_column, sort=True)
        ]
    )
    for seed, model in models.items():
        beta = np.asarray(
            model.predict(target, batch_size=2048, verbose=0), dtype=float
        )
        achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
        error = (
            np.linalg.norm(achieved - target.astype(float), axis=1) * 1000.0
        )
        margin = point_margin_deg(beta, environment.bounds)
        detail = reference[
            [
                column
                for column in (
                    "target_id",
                    "family_id",
                    "phase_idx",
                    "source_voxel_key",
                    *XYZ_COLUMNS,
                )
                if column in reference.columns
            ]
        ].copy()
        detail["seed"] = int(seed)
        for index, column in enumerate(PREDICTED_BETA_COLUMNS):
            detail[column] = beta[:, index]
        detail["achieved_x_m"] = achieved[:, 0]
        detail["achieved_y_m"] = achieved[:, 1]
        detail["achieved_z_m"] = achieved[:, 2]
        detail["fk_error_mm"] = error
        detail["minimum_joint_margin_deg"] = margin
        detail_parts.append(detail)
        for family, positions in groups:
            values = error[positions]
            margins = margin[positions]
            metric_rows.append(
                {
                    "seed": int(seed),
                    **(
                        {}
                        if family_column is None
                        else {family_column: str(family)}
                    ),
                    "row_count": int(len(positions)),
                    "fk_p50_mm": float(np.percentile(values, 50)),
                    "fk_p95_mm": float(np.percentile(values, 95)),
                    "fk_max_mm": float(np.max(values)),
                    "minimum_joint_margin_deg": float(np.min(margins)),
                    "actual_bounds": bool(np.min(margins) >= -1.0e-9),
                }
            )
    return pd.DataFrame(metric_rows), pd.concat(detail_parts, ignore_index=True)


def stage_development_random(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    train_gate = _require_gate(output_root, "07_students/gate.json")
    stage = output_root / "08_random_point_test"
    stage.mkdir(parents=True, exist_ok=True)
    values = config["v12_14"]
    dense = pd.read_parquet(output_root / "05_dense_region/dense_region_100k.parquet")
    sparse = pd.read_parquet(
        output_root / "03_sparse_region_growth/sparse_nodes.parquet"
    )
    label_anchors = pd.concat([sparse, dense], ignore_index=True, sort=False)
    required = int(values["training"]["random_test_rows"])
    targets = _random_targets_from_region(
        dense,
        attempt_count=required * 2,
        voxel_size_mm=float(values["region"]["voxel_size_mm"]),
        seed=int(values["seeds"]["development_random"]),
    )
    accepted, rejected, manifest = _label_targets(
        targets,
        label_anchors,
        name="development_random",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=stage,
        python=python,
    )
    reference = accepted.sort_values("target_id", kind="stable").iloc[:required].copy()
    reference["random_set_role"] = "development_active_enrichment_only"
    _atomic_parquet(targets, stage / "random_targets.parquet")
    _atomic_parquet(reference, stage / "development_teacher_reference.parquet")
    _atomic_parquet(rejected, stage / "development_rejected_targets.parquet")
    selection = json.loads(
        (output_root / "07_students/selected_model.json").read_text()
    )
    metrics, details = _evaluate_model_set(
        reference,
        model_roots=selection["model_roots"],
        environment=_environment(config, project_root),
    )
    metrics["random_set_role"] = "development_active_enrichment_only"
    metrics.to_csv(stage / "random_point_metrics.csv", index=False)
    _atomic_parquet(details, stage / "development_prediction_details.parquet")
    support = (
        details.groupby(["seed", "source_voxel_key"], sort=True)
        .agg(mean_fk_error_mm=("fk_error_mm", "mean"))
        .reset_index()
        if "source_voxel_key" in details
        else pd.DataFrame()
    )
    support.to_csv(stage / "error_vs_support.csv", index=False)
    return _gate(
        stage / "gate.json",
        {
            "development_reference_complete": len(reference) == required,
            "development_set_not_final_evidence": True,
            "three_locked_models_evaluated": metrics["seed"].nunique() == 3,
        },
        semantics="development_random_points_for_single_active_enrichment",
        accepted_teacher_rows=int(len(accepted)),
        required_rows=required,
        model_source_dataset=train_gate["selected_dataset_version"],
        parallel_evidence=manifest,
    )


def stage_active_enrichment(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "08_random_point_test/gate.json")
    stage = output_root / "10_active_enrichment"
    stage.mkdir(parents=True, exist_ok=True)
    values = config["v12_14"]
    selection = json.loads(
        (output_root / "07_students/selected_model.json").read_text()
    )
    selected_version = str(selection["selected_dataset_version"])
    detail = pd.read_parquet(
        output_root / "08_random_point_test/development_prediction_details.parquet"
    )
    score = (
        detail.groupby("target_id", sort=True)
        .agg(
            mean_fk_error_mm=("fk_error_mm", "mean"),
            source_voxel_key=("source_voxel_key", "first"),
        )
        .reset_index()
        .sort_values(["mean_fk_error_mm", "target_id"], ascending=[False, True], kind="stable")
    )
    top_count = max(
        1,
        int(
            math.ceil(
                len(score)
                * float(values["training"]["active_top_voxel_fraction"])
            )
        ),
    )
    high = score.iloc[:top_count].copy()
    high.to_csv(stage / "high_error_voxels.csv", index=False)
    dense = pd.read_parquet(output_root / "05_dense_region/dense_region_100k.parquet")
    sparse = pd.read_parquet(
        output_root / "03_sparse_region_growth/sparse_nodes.parquet"
    )
    label_anchors = pd.concat([sparse, dense], ignore_index=True, sort=False)
    high_dense = dense.loc[
        dense["voxel_key"].isin(high["source_voxel_key"])
    ].copy()
    required = int(values["training"]["active_rows"])
    targets = _random_targets_from_region(
        high_dense,
        attempt_count=required * 2,
        voxel_size_mm=float(values["region"]["voxel_size_mm"]),
        seed=int(values["seeds"]["enrichment"]),
    )
    accepted, rejected, label_manifest = _label_targets(
        targets,
        label_anchors,
        name="active_enrichment",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=stage,
        python=python,
    )
    enrichment = (
        accepted.sort_values("target_id", kind="stable").iloc[:required].copy()
    )
    maximum_wave = int(dense["region_wave"].max())
    wave_lookup = dense.groupby("voxel_key", sort=True)[
        "region_wave"
    ].first()
    enrichment["region_wave"] = enrichment["source_voxel_key"].map(wave_lookup)
    enrichment["sampling_bucket"] = np.where(
        enrichment["region_wave"].ge(maximum_wave), "boundary", "interior"
    )
    enrichment, jacobian_manifest = _parallel_add_jacobians(
        enrichment,
        name="enrichment",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=stage,
        python=python,
    )
    enrichment["quality_class"] = enrichment["quality_class"].astype(str)
    _atomic_parquet(enrichment, stage / "enrichment_dataset.parquet")
    base = pd.read_parquet(
        output_root / f"06_datasets/{selected_version}_dataset.parquet"
    )
    enrichment["dataset_version"] = f"{selected_version}_enriched"
    enrichment["sample_weight"] = enrichment["quality_class"].map(
        dict(values["dataset"]["quality_weights"])
    )
    enriched = pd.concat([base, enrichment], ignore_index=True, sort=False)
    enriched, split_report = spatial_train_validation_split(
        enriched,
        macro_voxel_mm=float(values["dataset"]["macro_voxel_mm"]),
        validation_fraction=float(values["dataset"]["validation_fraction"]),
        buffer_mm=float(values["dataset"]["split_buffer_mm"]),
        seed=int(values["seeds"]["enrichment"]) + 1,
    )
    dataset_version = f"{selected_version}_enriched"
    dataset_path = stage / f"{dataset_version}_dataset.parquet"
    _atomic_parquet(enriched, dataset_path)
    probe = _gpu_probe(python)
    if not probe["gpu_available"]:
        return _gate(
            stage / "gate.json",
            {"gpu_visible": False},
            semantics="single_registered_active_enrichment",
            stopped_before_retraining=True,
            gpu_probe=probe,
        )
    commands = _training_commands(
        [dataset_version],
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=stage,
        python=python,
        dataset_paths={dataset_version: dataset_path},
    )
    training_manifest = v12._run_subprocess_tasks(
        commands,
        requested_workers=1,
        manifest_path=stage / "retraining_parallel_manifest.json",
    )
    retrained_rows = []
    for task_id, _command, _log in commands:
        report = json.loads((stage / task_id / "report.json").read_text())
        retrained_rows.append({"task_id": task_id, **report})
    retrained = pd.json_normalize(retrained_rows)
    retrained.to_csv(stage / "retrained_metrics.csv", index=False)
    base_metrics = pd.read_csv(output_root / "07_students/metrics_per_seed.csv")
    base_selected = base_metrics.loc[
        base_metrics["dataset_version"].eq(selected_version)
    ]
    base_score = float(base_selected["validation_metrics.fk_p95_mm"].mean())
    enriched_score = float(retrained["validation_metrics.fk_p95_mm"].mean())
    use_enriched = enriched_score < base_score
    model_roots = (
        {
            str(int(seed)): str(stage / f"{dataset_version}_seed{int(seed)}")
            for seed in values["seeds"]["training"]
        }
        if use_enriched
        else selection["model_roots"]
    )
    lock = {
        "model_locked_before_final_random_and_trajectory_generation": True,
        "selected_variant": "enriched" if use_enriched else "base",
        "selected_dataset_version": (
            dataset_version if use_enriched else selected_version
        ),
        "selection_criterion": "mean_validation_fk_p95_mm",
        "base_mean_validation_fk_p95_mm": base_score,
        "enriched_mean_validation_fk_p95_mm": enriched_score,
        "model_roots": model_roots,
        "model_sha256": {
            seed: sha256_file(Path(root) / "model.keras")
            for seed, root in model_roots.items()
        },
    }
    atomic_write_json(stage / "final_model_lock.json", lock)
    return _gate(
        stage / "gate.json",
        {
            "teacher_consistent_enrichment_complete": len(enrichment) == required,
            "exactly_one_active_enrichment_round": True,
            "three_retrained_seeds_complete": len(retrained) == 3,
            "model_locked_before_final_sets": True,
        },
        semantics="single_registered_active_enrichment_and_model_lock",
        enrichment_rows=int(len(enrichment)),
        split=split_report,
        teacher_parallel_evidence=label_manifest,
        jacobian_parallel_evidence=jacobian_manifest,
        training_parallel_evidence=training_manifest,
        model_lock=lock,
    )


def _local_path(
    kind: str,
    center: np.ndarray,
    basis: np.ndarray,
    *,
    phase_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    phase = np.linspace(0.0, 2.0 * math.pi, int(phase_count), endpoint=False)
    q1, q2, q3 = basis.T
    if kind == "ellipse":
        target = (
            center
            + 0.0030 * np.cos(phase)[:, None] * q1
            + 0.0020 * np.sin(phase)[:, None] * q2
        )
        return target, True
    if kind == "circle":
        target = (
            center
            + 0.0032 * np.cos(phase)[:, None] * q1
            + 0.0032 * np.sin(phase)[:, None] * q2
        )
        return target, True
    if kind == "lissajous":
        target = (
            center
            + 0.0028 * np.sin(phase)[:, None] * q1
            + 0.0024 * np.sin(2.0 * phase + 0.4)[:, None] * q2
            + 0.0018 * np.sin(3.0 * phase + 0.7)[:, None] * q3
        )
        return target, True
    from scipy.interpolate import splprep, splev

    if kind == "closed_bspline":
        control_phase = np.linspace(0.0, 2.0 * math.pi, 9)
        radial = rng.uniform(0.0020, 0.0038, size=8)
        controls = (
            center
            + radial[:, None] * np.cos(control_phase[:-1])[:, None] * q1
            + radial[::-1, None] * np.sin(control_phase[:-1])[:, None] * q2
            + rng.uniform(-0.0012, 0.0012, size=(8, 1)) * q3
        )
        controls = np.vstack([controls, controls[0]])
        spline, _ = splprep(controls.T, s=0.0, per=True, k=3)
        target = np.column_stack(splev(np.linspace(0.0, 1.0, phase_count, endpoint=False), spline))
        return target, True
    if kind == "open_bspline":
        scalar = np.linspace(-1.0, 1.0, 7)
        controls = (
            center
            + 0.0040 * scalar[:, None] * q1
            + rng.uniform(-0.0020, 0.0020, size=(7, 1)) * q2
            + rng.uniform(-0.0015, 0.0015, size=(7, 1)) * q3
        )
        spline, _ = splprep(controls.T, s=0.0, per=False, k=3)
        target = np.column_stack(splev(np.linspace(0.0, 1.0, phase_count), spline))
        return target, False
    raise ValueError(f"unknown path kind: {kind}")


def _generate_path_catalog(
    dense: pd.DataFrame,
    *,
    phase_count: int,
    per_type: int,
    support_max_mm: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    xyz = dense.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    tree = cKDTree(xyz)
    interior = np.ones(len(dense), dtype=bool)
    if "quality_class" in dense:
        interior &= dense["quality_class"].eq("RegionGold").to_numpy()
    if "candidate_gap_max_deg" in dense:
        interior &= dense["candidate_gap_max_deg"].le(0.5).to_numpy()
    if "support_margin_min_deg" in dense:
        margin = dense["support_margin_min_deg"].to_numpy(dtype=float)
        interior &= margin >= float(np.median(margin[np.isfinite(margin)]))
    source_indices = np.flatnonzero(interior)
    if len(source_indices) < 1:
        raise RuntimeError("dense chart contains no registered interior path source")
    kinds = ("ellipse", "circle", "lissajous", "closed_bspline", "open_bspline")
    catalog_rows = []
    target_parts = []
    global_id = 0
    for kind in kinds:
        for local_id in range(int(per_type)):
            accepted = None
            closed = kind != "open_bspline"
            source_node = None
            support = math.inf
            for _attempt in range(300):
                source_index = int(rng.choice(source_indices))
                center = xyz[source_index]
                _distance, neighbor = tree.query(center, k=min(64, len(xyz)))
                local = xyz[np.asarray(neighbor)] - center
                _u, _s, vh = np.linalg.svd(local, full_matrices=False)
                basis = vh.T
                candidate, closed = _local_path(
                    kind,
                    center,
                    basis,
                    phase_count=phase_count,
                    rng=rng,
                )
                distance, _ = tree.query(candidate, k=1)
                support = float(np.max(distance) * 1000.0)
                if support <= float(support_max_mm):
                    accepted = candidate
                    source_node = str(dense.iloc[source_index]["node_id"])
                    break
            if accepted is None:
                raise RuntimeError(
                    f"unable to generate support-contained {kind} path"
                )
            family_id = f"unseen_{kind}_{local_id:02d}"
            part = pd.DataFrame(accepted, columns=XYZ_COLUMNS)
            part.insert(0, "phase_idx", np.arange(len(part), dtype=np.int64))
            part.insert(0, "family_id", family_id)
            part.insert(
                0,
                "target_id",
                np.arange(global_id, global_id + len(part), dtype=np.int64),
            )
            target_parts.append(part)
            catalog_rows.append(
                {
                    "family_id": family_id,
                    "trajectory_type": kind,
                    "closed": bool(closed),
                    "phase_count": int(len(part)),
                    "source_dense_node_id": source_node,
                    "support_distance_max_mm": support,
                    "generation_seed": int(seed),
                    "generated_after_model_lock": True,
                }
            )
            global_id += len(part)
    return pd.DataFrame(catalog_rows), pd.concat(target_parts, ignore_index=True)


def _plot_tracking(
    reference: pd.DataFrame,
    details: pd.DataFrame,
    catalog: pd.DataFrame,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    for family_id, family in reference.groupby("family_id", sort=True):
        info = catalog.loc[catalog["family_id"].eq(family_id)].iloc[0]
        target = family.sort_values("phase_idx").loc[:, XYZ_COLUMNS].to_numpy()
        centered = target - target.mean(axis=0)
        _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
        planar = bool(singular[-1] <= max(singular[0] * 0.02, 1.0e-8))
        if planar:
            figure = plt.figure(figsize=(10.5, 5.5))
            axis = figure.add_axes([0.08, 0.12, 0.58, 0.80])
            inset = figure.add_axes([0.72, 0.54, 0.24, 0.36], projection="3d")
            basis = vh[:2]
            target_uv = centered @ basis.T
            axis.plot(target_uv[:, 0] * 1000.0, target_uv[:, 1] * 1000.0, "k--", label="Teacher")
            for seed, group in details.loc[
                details["family_id"].eq(family_id)
            ].groupby("seed", sort=True):
                achieved = group.sort_values("phase_idx")[
                    ["achieved_x_m", "achieved_y_m", "achieved_z_m"]
                ].to_numpy()
                achieved_uv = (achieved - target.mean(axis=0)) @ basis.T
                axis.plot(
                    achieved_uv[:, 0] * 1000.0,
                    achieved_uv[:, 1] * 1000.0,
                    label=f"Student {seed}",
                )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("ellipse-plane u (mm)")
            axis.set_ylabel("ellipse-plane v (mm)")
            axis.legend(fontsize=7)
            inset.plot(target[:, 0], target[:, 1], target[:, 2], "k--")
            inset.scatter([0.0], [0.0], [0.0], marker="o", color="tab:red", s=20)
            inset.quiver(0, 0, 0, 0.15, 0, 0, color="r")
            inset.quiver(0, 0, 0, 0, 0.15, 0, color="g")
            inset.quiver(0, 0, 0, 0, 0, 0.15, color="b")
            inset.set_title("robot workspace frame", fontsize=8)
        else:
            figure = plt.figure(figsize=(8, 7))
            axis = figure.add_subplot(111, projection="3d")
            axis.plot(target[:, 0], target[:, 1], target[:, 2], "k--", label="Teacher")
            for seed, group in details.loc[
                details["family_id"].eq(family_id)
            ].groupby("seed", sort=True):
                achieved = group.sort_values("phase_idx")[
                    ["achieved_x_m", "achieved_y_m", "achieved_z_m"]
                ].to_numpy()
                axis.plot(
                    achieved[:, 0],
                    achieved[:, 1],
                    achieved[:, 2],
                    label=f"Student {seed}",
                )
            spans = np.ptp(target, axis=0)
            radius = max(float(spans.max()) * 0.6, 0.005)
            center = target.mean(axis=0)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.legend(fontsize=7)
        figure.suptitle(f"{family_id} ({info['trajectory_type']})")
        figure.savefig(destination / f"{family_id}.png", dpi=160)
        plt.close(figure)


def stage_final_evaluation(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    active_gate = _require_gate(output_root, "10_active_enrichment/gate.json")
    values = config["v12_14"]
    lock = json.loads(
        (output_root / "10_active_enrichment/final_model_lock.json").read_text()
    )
    dense = pd.read_parquet(output_root / "05_dense_region/dense_region_100k.parquet")
    sparse = pd.read_parquet(
        output_root / "03_sparse_region_growth/sparse_nodes.parquet"
    )
    label_anchors = pd.concat([sparse, dense], ignore_index=True, sort=False)
    environment = _environment(config, project_root)

    random_stage = output_root / "08_random_point_test"
    required = int(values["training"]["random_test_rows"])
    final_targets = _random_targets_from_region(
        dense,
        attempt_count=required * 2,
        voxel_size_mm=float(values["region"]["voxel_size_mm"]),
        seed=int(
            lock.get(
                "final_random_seed",
                values["seeds"]["final_random"],
            )
        ),
    )
    final_reference, final_rejected, random_manifest = _label_targets(
        final_targets,
        label_anchors,
        name="final_random",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=random_stage,
        python=python,
    )
    final_reference = final_reference.sort_values(
        "target_id", kind="stable"
    ).iloc[:required].copy()
    final_reference["random_set_role"] = "final_independent_after_model_lock"
    _atomic_parquet(final_targets, random_stage / "final_random_targets.parquet")
    _atomic_parquet(
        final_reference, random_stage / "final_teacher_reference.parquet"
    )
    _atomic_parquet(
        final_rejected, random_stage / "final_rejected_targets.parquet"
    )
    random_metrics, random_details = _evaluate_model_set(
        final_reference,
        model_roots=lock["model_roots"],
        environment=environment,
    )
    random_metrics["random_set_role"] = "final_independent_after_model_lock"
    random_metrics.to_csv(
        random_stage / "final_random_point_metrics.csv", index=False
    )
    _atomic_parquet(
        random_details, random_stage / "final_prediction_details.parquet"
    )
    training = values["training"]
    random_pass = (
        random_metrics["fk_p95_mm"].le(float(training["fk_p95_mm"]))
        & random_metrics["fk_max_mm"].le(float(training["fk_max_mm"]))
        & random_metrics["actual_bounds"].astype(bool)
    )

    path_stage = output_root / "09_unseen_trajectory_test"
    path_stage.mkdir(parents=True, exist_ok=True)
    catalog, targets = _generate_path_catalog(
        dense,
        phase_count=int(values["trajectories"]["phase_count"]),
        per_type=int(values["trajectories"]["per_type"]),
        support_max_mm=float(values["region"]["parent_distance_max_mm"]) / 2.0,
        seed=int(
            lock.get(
                "trajectory_seed",
                values["seeds"]["trajectory"],
            )
        ),
    )
    catalog.to_csv(path_stage / "trajectory_catalog.csv", index=False)
    teacher, path_rejected, path_manifest = _label_targets(
        targets,
        label_anchors,
        name="unseen_paths",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=path_stage,
        python=python,
    )
    family_counts = teacher.groupby("family_id", sort=True).size()
    complete_ids = family_counts.loc[
        family_counts.eq(int(values["trajectories"]["phase_count"]))
    ].index
    teacher = teacher.loc[teacher["family_id"].isin(complete_ids)].copy()
    _atomic_parquet(teacher, path_stage / "teacher_references.parquet")
    _atomic_parquet(path_rejected, path_stage / "rejected_path_points.parquet")
    path_metrics, path_details = _evaluate_model_set(
        teacher,
        model_roots=lock["model_roots"],
        environment=environment,
        family_column="family_id",
    )
    path_metrics["seed_gate_pass"] = (
        path_metrics["fk_p95_mm"].le(float(training["fk_p95_mm"]))
        & path_metrics["fk_max_mm"].le(float(training["fk_max_mm"]))
        & path_metrics["actual_bounds"].astype(bool)
    )
    path_pass = _family_seed_gate(
        path_metrics,
        family_column="family_id",
        pass_column="seed_gate_pass",
        required_seed_passes=int(training["required_seed_passes"]),
    )
    path_metrics.to_csv(path_stage / "trajectory_metrics.csv", index=False)
    _atomic_parquet(path_details, path_stage / "student_tracking.parquet")
    _plot_tracking(teacher, path_details, catalog, path_stage / "plots")

    final8 = pd.read_parquet(
        _source_files(config, project_root)["source_final_teacher"]
    )
    final8_metrics, final8_details = _evaluate_model_set(
        final8,
        model_roots=lock["model_roots"],
        environment=environment,
        family_column="family_id",
    )
    catalog_path = (
        _source_files(config, project_root)["source_final_teacher"].parent
        / "new_family_catalog.parquet"
    )
    final_catalog = pd.read_parquet(catalog_path)[
        ["family_id", "major_semiaxis_m"]
    ]
    final8_metrics = final8_metrics.merge(
        final_catalog, on="family_id", validate="many_to_one"
    )
    final8_metrics["fk_p95_relative"] = (
        final8_metrics["fk_p95_mm"]
        / (final8_metrics["major_semiaxis_m"] * 1000.0)
    )
    final8_metrics["fk_max_relative"] = (
        final8_metrics["fk_max_mm"]
        / (final8_metrics["major_semiaxis_m"] * 1000.0)
    )
    final8_metrics["radius_gate_pass"] = (
        final8_metrics["fk_p95_relative"].le(
            float(values["trajectories"]["final_ellipse_p95_relative"])
        )
        & final8_metrics["fk_max_relative"].le(
            float(values["trajectories"]["final_ellipse_max_relative"])
        )
    )
    final8_pass = _family_seed_gate(
        final8_metrics,
        family_column="family_id",
        pass_column="radius_gate_pass",
        required_seed_passes=int(training["required_seed_passes"]),
    )
    final8_metrics.to_csv(path_stage / "final8_retention_metrics.csv", index=False)
    _atomic_parquet(final8_details, path_stage / "final8_student_tracking.parquet")

    summary_stage = output_root / "11_summary"
    summary_stage.mkdir(parents=True, exist_ok=True)
    dataset_metrics = pd.read_csv(output_root / "07_students/metrics_per_seed.csv")
    dataset_metrics.to_csv(summary_stage / "dataset_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected_region_voxels": int(
                    json.loads(
                        (
                            output_root
                            / "02_capability_region/selected_region.json"
                        ).read_text()
                    )["selected_region_voxel_count"]
                ),
                "sparse_component_nodes": int(
                    json.loads(
                        (
                            output_root
                            / "03_sparse_region_growth/growth_report.json"
                        ).read_text()
                    )["largest_component_node_count"]
                ),
                "dense_rows": int(len(dense)),
            }
        ]
    ).to_csv(summary_stage / "coverage_report.csv", index=False)
    generalization = pd.DataFrame(
        [
            {
                "final_random_seed_passes": int(random_pass.sum()),
                "unseen_trajectory_passes": int(path_pass.sum()),
                "unseen_trajectory_count": int(len(path_pass)),
                "final8_family_seed_passes": int(
                    final8_metrics["radius_gate_pass"].sum()
                ),
                "final8_family_seed_count": int(len(final8_metrics)),
                "final8_retained_family_count": int(final8_pass.sum()),
                "final8_family_count": int(len(final8_pass)),
            }
        ]
    )
    generalization.to_csv(
        summary_stage / "student_generalization_summary.csv", index=False
    )
    if config["preset"] == "smoke":
        final_checks = {
            "final_random_teacher_rows_complete": len(final_reference)
            == required,
            "final_random_three_models_evaluated": len(random_metrics) == 3,
            "all_smoke_unseen_paths_teacher_complete": len(complete_ids)
            == 5 * int(values["trajectories"]["per_type"]),
            "all_smoke_unseen_paths_evaluated": path_metrics[
                "family_id"
            ].nunique()
            == 5 * int(values["trajectories"]["per_type"]),
            "current_final8_evaluation_complete": len(final8_metrics) == 24,
            "model_was_locked_before_final_generation": bool(
                lock[
                    "model_locked_before_final_random_and_trajectory_generation"
                ]
            ),
        }
    else:
        final_checks = {
            "final_random_teacher_rows_complete": len(final_reference)
            == required,
            "final_random_seed_gate": int(random_pass.sum())
            >= int(training["required_seed_passes"]),
            "twenty_teacher_feasible_unseen_paths": len(complete_ids)
            == 5 * int(values["trajectories"]["per_type"]),
            "unseen_path_gate": int(path_pass.sum())
            >= int(values["trajectories"]["required_pass_count"]),
            "current_final8_retained": bool(final8_pass.all()),
            "model_was_locked_before_final_generation": bool(
                lock[
                    "model_locked_before_final_random_and_trajectory_generation"
                ]
            ),
        }
    recommendation = (
        "# V12.14 final recommendation\n\n"
        f"- final random seed pass: {int(random_pass.sum())}/3\n"
        f"- unseen path pass: {int(path_pass.sum())}/{len(path_pass)}\n"
        f"- V12.13 final ellipse family-seed retention: "
        f"{int(final8_metrics['radius_gate_pass'].sum())}/{len(final8_metrics)}\n"
        f"- overall exploratory gate: {all(final_checks.values())}\n\n"
        "This artifact supports simulation region-generalization diagnostics "
        "only; deployment remains false.\n"
    )
    (summary_stage / "final_recommendation.md").write_text(recommendation)
    return _gate(
        summary_stage / "gate.json",
        final_checks,
        semantics="v12_14_final_independent_region_generalization",
        random_parallel_evidence=random_manifest,
        path_parallel_evidence=path_manifest,
        selected_model_lock=lock,
        random_seed_passes=int(random_pass.sum()),
        unseen_path_passes=int(path_pass.sum()),
        unseen_path_count=int(len(path_pass)),
        final8_family_seed_passes=int(
            final8_metrics["radius_gate_pass"].sum()
        ),
        final8_family_seed_count=int(len(final8_metrics)),
        final8_family_passes=int(final8_pass.sum()),
        final8_family_count=int(len(final8_pass)),
    )


STAGES = (
    "protocol",
    "seed_cleanup",
    "capability_region",
    "sparse_growth",
    "consistency",
    "dense_generation",
    "dataset_assemble",
    "train",
    "development_random",
    "active_enrich",
    "final_test",
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = _output_root(config, project_root, args.output_root)
    stages = {
        "protocol": lambda: stage_protocol(config, project_root, output_root),
        "seed_cleanup": lambda: stage_seed_cleanup(
            config, project_root, output_root
        ),
        "capability_region": lambda: stage_capability_region(
            config, project_root, output_root
        ),
        "sparse_growth": lambda: stage_sparse_growth(
            config, project_root, output_root, python=Path(args.python)
        ),
        "consistency": lambda: stage_consistency(
            config, project_root, output_root
        ),
        "dense_generation": lambda: stage_dense(
            config, project_root, output_root, python=Path(args.python)
        ),
        "dataset_assemble": lambda: stage_dataset(
            config, project_root, output_root, python=Path(args.python)
        ),
        "train": lambda: stage_train(
            config, project_root, output_root, python=Path(args.python)
        ),
        "development_random": lambda: stage_development_random(
            config, project_root, output_root, python=Path(args.python)
        ),
        "active_enrich": lambda: stage_active_enrichment(
            config, project_root, output_root, python=Path(args.python)
        ),
        "final_test": lambda: stage_final_evaluation(
            config, project_root, output_root, python=Path(args.python)
        ),
    }
    requested = STAGES if args.stage == "all" else (str(args.stage),)
    reports = {}
    for name in requested:
        report = stages[name]()
        reports[name] = report
        if not report["gate_pass"]:
            break
    summary = {
        "protocol_id": PROTOCOL_ID,
        "claim_scope": CLAIM_SCOPE,
        "requested_stage": str(args.stage),
        "completed_stages": list(reports),
        "stage_gate_pass": {
            name: bool(report["gate_pass"]) for name, report in reports.items()
        },
    }
    atomic_write_json(output_root / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v12_14_region_growth.yaml"),
    )
    parser.add_argument("--preset", choices=("formal", "smoke"), default="formal")
    parser.add_argument(
        "--project-root", default=str(DEFAULT_PROJECT_ROOT)
    )
    parser.add_argument("--output-root")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--worker", choices=("sparse", "jacobian", "train"))
    parser.add_argument("--worker-input")
    parser.add_argument("--worker-output")
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-version")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-model")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker == "sparse":
        return _worker_sparse(args)
    if args.worker == "jacobian":
        return _worker_jacobian(args)
    if args.worker == "train":
        return _worker_train(args)
    summary = run(args)
    return 0 if all(summary["stage_gate_pass"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
