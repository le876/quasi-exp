#!/usr/bin/env python3
"""Build deterministic, evidence-only aggregates for the BACRA V14 Q17 handoff.

The script reads the complete Pilot tables used by the three scientific gates.
It does not resample data, change thresholds, rerun IK, or upgrade any gate.
The output is a compact cross-file audit plus the complete set of robust
cross-cell product edges, with hashes and table schemas for every source.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/mnt/ML_projects/quasi_exp")
SOURCE_ROOT = PROJECT_ROOT / ".worktrees/bacra-v14-omega200-workspace-atlas"
PILOT_ROOT = PROJECT_ROOT / "runs/bacra_v14_omega200_workspace_atlas_pilot"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/gpt5pro_handoff_evidence/Q17"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def parquet_inventory(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
    }


def file_inventory(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_longrun_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    start_match = re.search(r"^\[longrun\] start=(.+)$", text, flags=re.MULTILINE)
    end_match = re.search(r"^\[longrun\] end=(.+) rc=(\d+)$", text, flags=re.MULTILINE)
    result: dict[str, Any] = {"source": file_inventory(path)}
    if start_match:
        result["start_local"] = start_match.group(1)
    if end_match:
        result["end_local"] = end_match.group(1)
        result["return_code"] = int(end_match.group(2))
    if start_match and end_match:
        start = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_match.group(1), "%Y-%m-%d %H:%M:%S")
        result["wall_time_s"] = float((end - start).total_seconds())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {output}")
    staging = output.with_name(output.name + f".staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    try:
        paths = {
            "frozen_config": PILOT_ROOT / "00_protocol/frozen_config.json",
            "source_manifest": PILOT_ROOT / "00_protocol/source_manifest.json",
            "workspace_proxy_report": PILOT_ROOT / "01_workspace_proxy/workspace_proxy_report.json",
            "occupancy_convergence": PILOT_ROOT / "01_workspace_proxy/occupancy_convergence.json",
            "replica_a": PILOT_ROOT / "01_workspace_proxy/replica_a_slab.parquet",
            "replica_b": PILOT_ROOT / "01_workspace_proxy/replica_b_slab.parquet",
            "frontier_probes": PILOT_ROOT / "01_workspace_proxy/frontier_inverse_probes.parquet",
            "registry_report": PILOT_ROOT / "02_domain_registry/registry_report.json",
            "pilot_task_probes": PILOT_ROOT / "02_domain_registry/pilot_task_probes.parquet",
            "candidate_report": PILOT_ROOT / "03_candidate_bank/candidate_report.json",
            "branch_audit": PILOT_ROOT / "03_candidate_bank/branch_saturation_audit.parquet",
            "cell_candidates": PILOT_ROOT / "03_candidate_bank/cell_candidates.parquet",
            "saturation_candidates": PILOT_ROOT / "03_candidate_bank/saturation_candidates.parquet",
            "atlas_report": PILOT_ROOT / "04_workspace_atlas/atlas_report.json",
            "workspace_atlas_report": PILOT_ROOT / "04_workspace_atlas/workspace_atlas_report.json",
            "atlas_task_nodes": PILOT_ROOT / "04_workspace_atlas/atlas_task_nodes.parquet",
            "atlas_candidates": PILOT_ROOT / "04_workspace_atlas/atlas_candidates.parquet",
            "product_edges": PILOT_ROOT / "04_workspace_atlas/product_edges.parquet",
            "charts": PILOT_ROOT / "04_workspace_atlas/charts.parquet",
            "chart_overlaps": PILOT_ROOT / "04_workspace_atlas/chart_overlaps.parquet",
            "selected_cells": PILOT_ROOT / "04_workspace_atlas/selected_cell_classification.parquet",
            "primary_partition": PILOT_ROOT / "04_workspace_atlas/primary_chart_partition.parquet",
            "task_probe_labels": PILOT_ROOT / "04_workspace_atlas/task_probe_labels.parquet",
            "budget_report": PILOT_ROOT / "05_budget_allocation/budget_report.json",
            "dataset_report": PILOT_ROOT / "06_dataset/dataset_report.json",
            "student_report": PILOT_ROOT / "07_student/student_report.json",
            "summary_report": PILOT_ROOT / "11_summary/summary_report.json",
            "pilot_gate": PILOT_ROOT / "11_summary/pilot_gate.json",
            "artifact_manifest": PILOT_ROOT / "11_summary/artifact_manifest.json",
            "pilot_log": PROJECT_ROOT / "runs/maintenance/bacra_v14_pilot.log",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing Q17 sources: {missing}")

        config = strict_json(paths["frozen_config"])
        proxy = strict_json(paths["workspace_proxy_report"])
        convergence = read_json(paths["occupancy_convergence"])
        registry = strict_json(paths["registry_report"])
        candidate_report = strict_json(paths["candidate_report"])
        atlas_report = strict_json(paths["atlas_report"])
        workspace_atlas = strict_json(paths["workspace_atlas_report"])
        budget = strict_json(paths["budget_report"])
        dataset = strict_json(paths["dataset_report"])
        student = strict_json(paths["student_report"])
        summary = strict_json(paths["summary_report"])
        pilot_gate = strict_json(paths["pilot_gate"])

        frames = {
            key: pd.read_parquet(paths[key])
            for key in (
                "replica_a",
                "replica_b",
                "frontier_probes",
                "pilot_task_probes",
                "branch_audit",
                "cell_candidates",
                "saturation_candidates",
                "atlas_task_nodes",
                "atlas_candidates",
                "product_edges",
                "charts",
                "chart_overlaps",
                "selected_cells",
                "primary_partition",
                "task_probe_labels",
            )
        }

        nodes = frames["atlas_task_nodes"]
        edges = frames["product_edges"]
        cell_columns = ["cell_level_mm", "cell_ix", "cell_iy", "cell_iz"]
        node_cells = {
            str(row.task_node_id): tuple(getattr(row, column) for column in cell_columns)
            for row in nodes.itertuples(index=False)
        }
        left_cells = edges["left_task_node_id"].astype(str).map(node_cells)
        right_cells = edges["right_task_node_id"].astype(str).map(node_cells)
        if left_cells.isna().any() or right_cells.isna().any():
            raise RuntimeError("product edge references missing task nodes")
        cross_mask = left_cells != right_cells
        cross_edges = edges.loc[cross_mask].copy()
        for index, column in enumerate(cell_columns):
            cross_edges[f"left_{column}"] = [value[index] for value in left_cells[cross_mask]]
            cross_edges[f"right_{column}"] = [value[index] for value in right_cells[cross_mask]]
        cross_output = staging / "q17_cross_cell_robust_edges.csv"
        cross_edges.to_csv(cross_output, index=False)

        branch = frames["branch_audit"]
        selected = frames["selected_cells"]
        partition = frames["primary_partition"]
        overlaps = frames["chart_overlaps"]
        charts = frames["charts"]

        latest_metric = proxy["metrics"][-1]
        reach_thresholds = {
            "minimum_weighted_jaccard": float(config["reach"]["minimum_weighted_jaccard"]),
            "maximum_new_volume_ratio": float(config["reach"]["maximum_new_volume_ratio"]),
            "maximum_boundary_change_ratio": float(config["reach"]["maximum_boundary_change_ratio"]),
            "maximum_frontier_new_volume_ratio": float(config["reach"]["maximum_frontier_new_volume_ratio"]),
            "required_consecutive_rounds": int(config["reach"]["required_consecutive_rounds"]),
        }

        formal_reasons: list[str] = []
        if not bool(pilot_gate.get("gate_pass", False)):
            formal_reasons.append("pilot_gate_failed")
        if config["formal_requirements"].get("require_reach_convergence", True) and not bool(
            pilot_gate.get("reach_convergence_gate", False)
        ):
            formal_reasons.append("reach_convergence_not_proven")
        if config["formal_requirements"].get("require_branch_saturation", True) and not bool(
            pilot_gate.get("branch_saturation_gate", False)
        ):
            formal_reasons.append("branch_discovery_not_saturated")
        if config["formal_requirements"].get("require_representation_gate", True) and not bool(
            pilot_gate.get("representation_gate_pass", False)
        ):
            formal_reasons.append("deployable_representation_not_authorized")

        source_inventory: dict[str, Any] = {}
        for key, path in paths.items():
            if key in frames:
                source_inventory[key] = parquet_inventory(path, frames[key])
            else:
                source_inventory[key] = file_inventory(path)

        output_value = {
            "schema_version": 1,
            "evidence_level": "diagnostic_only_post_hoc_full_table_aggregation",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "claim_scope": "summarizes_existing_v14_pilot_artifacts_without_changing_any_gate",
            "generation": {
                "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
                "command": f"{Path(os.sys.executable).resolve()} {Path(__file__).resolve()} --output {output}",
                "sampling": "none; every row of each listed Parquet source was read",
                "random_seed": None,
                "source_root": str(SOURCE_ROOT),
                "pilot_root": str(PILOT_ROOT),
            },
            "fixed_point": {
                "branch": git_text("branch", "--show-current"),
                "head": git_text("rev-parse", "HEAD"),
                "working_tree_clean": git_text("status", "--porcelain") == "",
                "implementation_commit_subject": git_text("show", "-s", "--format=%s", "HEAD"),
            },
            "longrun": parse_longrun_log(paths["pilot_log"]),
            "overall": {
                "operational_stage_gate_pass": summary["stage_gate_pass"],
                "all_operational_stages_pass": all(bool(value) for value in summary["stage_gate_pass"].values()),
                "pilot_gate_pass": bool(summary["pilot_gate_pass"]),
                "scientific_gate_pass": bool(summary["scientific_gate_pass"]),
                "formal_authorized_by_this_run": bool(summary["formal_authorized_by_this_run"]),
                "deployment_claim_gate_pass": bool(summary["deployment_claim_gate_pass"]),
                "formal_prerequisite_reasons": formal_reasons,
            },
            "reach": {
                "report": proxy,
                "convergence_record": convergence,
                "thresholds": reach_thresholds,
                "latest_round": latest_metric,
                "latest_round_checks": {
                    "weighted_jaccard": bool(
                        latest_metric["volume_weighted_jaccard"] >= reach_thresholds["minimum_weighted_jaccard"]
                    ),
                    "new_volume": bool(
                        latest_metric["new_volume_ratio"] <= reach_thresholds["maximum_new_volume_ratio"]
                    ),
                    "boundary_change": bool(
                        latest_metric["boundary_change_ratio"] <= reach_thresholds["maximum_boundary_change_ratio"]
                    ),
                    "frontier_new_volume": bool(
                        latest_metric["frontier_new_volume_ratio"]
                        <= reach_thresholds["maximum_frontier_new_volume_ratio"]
                    ),
                },
                "interpretation_boundary": [
                    "The A/B replicas are independent scrambled Sobol lineages, but finite occupancy is still an empirical proxy.",
                    "A numerical IK failure was not converted into certified unreachability.",
                    "This aggregation does not determine whether more rounds or a different reach-discovery method is scientifically preferable.",
                ],
            },
            "branch_discovery": {
                "candidate_report": candidate_report,
                "audit_rows": int(len(branch)),
                "new_stable_family_counts": counts(branch["new_stable_branch"]),
                "base_cluster_count": numeric_summary(branch["base_cluster_count"]),
                "saturation_cluster_count": numeric_summary(branch["saturation_cluster_count"]),
                "new_branch_max_gap_deg": numeric_summary(branch["new_branch_max_gap_deg"]),
                "base_solver_attempt_count": numeric_summary(branch["base_solver_attempt_count"]),
                "saturation_solver_attempt_count": numeric_summary(branch["saturation_solver_attempt_count"]),
                "interpretation_boundary": [
                    "The persisted field new_stable_branch is triggered by the configured candidate-family gap criterion.",
                    "A candidate family separated by more than one degree is not by itself a proof of a distinct topological inverse component.",
                    "The current artifacts do not classify distant candidates as same-fiber versus distinct-component solutions.",
                ],
            },
            "atlas_and_representation": {
                "workspace_atlas_report": workspace_atlas,
                "atlas_report": atlas_report,
                "selected_cell_status_counts": counts(selected["representative_status"]),
                "selected_cell_resolution_counts": counts(selected["resolution"]),
                "selected_nonzero_labelable_fraction_count": int(
                    (selected["empirical_labelable_fraction"] > 0).sum()
                ),
                "selected_fully_labelable_fraction_count": int(
                    (selected["empirical_labelable_fraction"] == 1).sum()
                ),
                "primary_partition_kind_counts": counts(partition["partition_kind"]),
                "primary_assigned_section_counts": counts(partition["assigned_section_id"]),
                "chart_count": int(len(charts)),
                "chart_selection_count": numeric_summary(charts["selection_count"]),
                "overlap_kind_counts": counts(overlaps["kind"]),
                "overlap_beta_gap_p95_deg": numeric_summary(overlaps["beta_gap_p95_deg"]),
                "robust_product_edges": {
                    "total": int(len(edges)),
                    "within_cell": int((~cross_mask).sum()),
                    "cross_cell": int(cross_mask.sum()),
                    "unique_cross_cell_pairs": int(
                        pd.DataFrame({"left": left_cells[cross_mask], "right": right_cells[cross_mask]})
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "cross_cell_csv": "q17_cross_cell_robust_edges.csv",
                },
                "interpretation_boundary": [
                    "The edge counts establish sparse cross-cell graph support but do not identify its cause.",
                    "Passing local path, cycle, multipath, and repeat audits does not establish domain-scale primary coverage.",
                    "Non-stitchable overlaps rule out an evidence-free claim that FK residual can select a canonical branch.",
                ],
            },
            "budget_dataset_student": {
                "budget_report": budget,
                "dataset_report": dataset,
                "student_report": student,
                "interpretation_boundary": "Budget arithmetic passed, but representation blocking prevented supervision materialization and Student training.",
            },
            "derived_outputs": {
                "q17_cross_cell_robust_edges.csv": {
                    "method": "complete filter of product_edges where the joined 10 mm cell tuple differs",
                    "source_rows": int(len(edges)),
                    "output_rows": int(len(cross_edges)),
                    "sampling": "none",
                }
            },
            "source_inventory": source_inventory,
            "non_claims": [
                "No gate is relaxed or passed by this post-hoc aggregation.",
                "No Formal run, 50k Pilot supervision dataset, 200k Formal dataset, Student, sealed spatial evaluation, or trajectory evaluation is created here.",
                "The aggregate does not prove a root cause for reach non-convergence, branch non-saturation, or sparse cross-cell edges.",
            ],
        }
        summary_output = staging / "q17_pilot_gate_audit.json"
        summary_output.write_text(
            json.dumps(jsonable(output_value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output": str(output),
                "files": sorted(path.name for path in output.iterdir()),
                "summary_sha256": sha256_file(output / "q17_pilot_gate_audit.json"),
                "cross_cell_csv_sha256": sha256_file(output / "q17_cross_cell_robust_edges.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
