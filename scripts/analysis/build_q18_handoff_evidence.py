#!/usr/bin/env python3
"""Build deterministic BACRA V14.1 retry4 evidence for the Q18 handoff.

This script performs a read-only, full-table aggregation of the existing
retry4 artifacts.  It does not rerun IK, alter thresholds, resample rows, or
change any scientific Gate.  It also exports the exact scientific source files
from the upstream fixed point and the later aggregation-only Formal fix so GPT-5
Pro can distinguish the two lineages.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/mnt/ML_projects/quasi_exp")
SOURCE_ROOT = PROJECT_ROOT / ".worktrees/bacra-v14-1-cross-cell-repair"
RUN_ROOT = PROJECT_ROOT / "runs/bacra_v14_1_cross_cell_repair_retry4"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/gpt5pro_handoff_evidence/Q18"
SCIENTIFIC_SHA = "5239d245d823223eaf282ba419eccb505e3e47e2"
FORMAL_AGGREGATION_SHA = "98dd93cb4e3ab48cf75314868fbe7207a56c01b4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            raise ValueError(f"non-finite value: {value}")
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_bytes(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout


def file_record(path: Path, *, relative_to: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def parquet_record(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        **file_record(path),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
    }


def quantile(values: pd.Series, probability: float) -> float:
    return float(pd.to_numeric(values, errors="raise").quantile(probability))


def patch_rows(frame: pd.DataFrame, ablation: str) -> pd.DataFrame:
    rows = frame.loc[frame["ablation"] == ablation].copy()
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 rows for {ablation}, got {len(rows)}")
    return rows


def aggregate_patch_method(frame: pd.DataFrame, ablation: str) -> dict[str, Any]:
    rows = patch_rows(frame, ablation)
    return {
        "patch_count": int(len(rows)),
        "candidate_count_median": float(rows["candidate_count"].median()),
        "robust_edge_count_median": float(rows["robust_edge_count"].median()),
        "continuation_attempt_count_median": float(
            rows["continuation_attempt_count"].median()
        ),
        "fresh_audit_execution_count_median": float(
            rows["fresh_audit_execution_count"].median()
        ),
        "cross_cell_connection_rate_min": float(
            rows["cross_cell_neighbor_pair_connection_rate"].min()
        ),
        "cross_cell_connection_rate_median": float(
            rows["cross_cell_neighbor_pair_connection_rate"].median()
        ),
        "largest_component_ratio_min": float(rows["largest_component_ratio"].min()),
        "largest_component_ratio_median": float(
            rows["largest_component_ratio"].median()
        ),
        "connected_cell_ratio_min": float(rows["connected_cell_ratio"].min()),
        "audit_pass_count": int(rows["audit_gate_pass"].astype(bool).sum()),
        "budget_exhausted_count": int(rows["budget_exhausted"].astype(bool).sum()),
        "runtime_s_median": float(rows["runtime_s"].median()),
        "runtime_s_sum_across_patches": float(rows["runtime_s"].sum()),
    }


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("artifacts", [])
    missing: list[str] = []
    size_mismatch: list[str] = []
    sha_mismatch: list[str] = []
    for row in rows:
        relative = str(row["path"])
        path = RUN_ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            size_mismatch.append(relative)
        if sha256_file(path) != str(row["sha256"]):
            sha_mismatch.append(relative)
    return {
        "declared_artifact_count": int(manifest["artifact_count"]),
        "manifest_row_count": int(len(rows)),
        "missing_count": int(len(missing)),
        "size_mismatch_count": int(len(size_mismatch)),
        "sha256_mismatch_count": int(len(sha_mismatch)),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "sha256_mismatch": sha_mismatch,
        "all_rows_verified": not (missing or size_mismatch or sha_mismatch),
    }


def retry3_retry4_overlap_check(columns: Iterable[str]) -> dict[str, Any]:
    retry3_path = (
        PROJECT_ROOT
        / "runs/bacra_v14_1_cross_cell_repair_retry3/01_patch_ablations/ablation_metrics.parquet"
    )
    if not retry3_path.is_file():
        return {"available": False, "reason": "retry3_ablation_metrics_missing"}
    retry3 = pd.read_parquet(retry3_path)
    retry4 = pd.read_parquet(RUN_ROOT / "01_patch_ablations/ablation_metrics.parquet")
    keys = ["patch_id", "ablation"]
    shared = retry3.merge(retry4, on=keys, suffixes=("_retry3", "_retry4"))
    compare_columns = [column for column in columns if column != "runtime_s"]
    mismatch: dict[str, int] = {}
    for column in compare_columns:
        left = shared[f"{column}_retry3"]
        right = shared[f"{column}_retry4"]
        equal = left.astype(str) == right.astype(str)
        mismatch[column] = int((~equal).sum())
    return {
        "available": True,
        "shared_patch_method_row_count": int(len(shared)),
        "shared_patch_ids": sorted(str(value) for value in shared["patch_id"].unique()),
        "excluded_columns": ["runtime_s"],
        "compared_column_count": int(len(compare_columns)),
        "mismatch_counts": mismatch,
        "all_compared_values_equal": all(value == 0 for value in mismatch.values()),
        "source": file_record(retry3_path),
    }


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
        artifact_paths = {
            "protocol": RUN_ROOT / "00_patch_inventory/protocol.json",
            "inventory_gate": RUN_ROOT / "00_patch_inventory/gate.json",
            "ablation_metrics": RUN_ROOT / "01_patch_ablations/ablation_metrics.parquet",
            "ablation_gate": RUN_ROOT / "01_patch_ablations/gate.json",
            "stability_observations": RUN_ROOT
            / "02_selection_stability/selection_observations.parquet",
            "stability_gate": RUN_ROOT / "02_selection_stability/gate.json",
            "frontier_probes": RUN_ROOT
            / "03_reach_extension/frontier_inverse_probes_round4_5.parquet",
            "reach_gate": RUN_ROOT / "03_reach_extension/gate.json",
            "repaired_pilot_gate": RUN_ROOT / "04_repaired_5k_pilot/gate.json",
            "student_gate": RUN_ROOT / "05_exploratory_student/gate.json",
            "formal_gate": RUN_ROOT / "06_formal_admission/gate.json",
            "summary": RUN_ROOT / "07_summary/summary_report.json",
            "summary_gate": RUN_ROOT / "07_summary/gate.json",
            "artifact_manifest": RUN_ROOT / "07_summary/artifact_manifest.json",
            "ablation_log": PROJECT_ROOT
            / "runs/maintenance/bacra_v14_1_retry4_ablations_12w.log",
            "reach_log": PROJECT_ROOT / "runs/maintenance/bacra_v14_1_retry4_reach.log",
            "stability_log": PROJECT_ROOT
            / "runs/maintenance/bacra_v14_1_retry4_stability.log",
            "reviewed_plan": PROJECT_ROOT
            / "docs/17-pro提问-V14三项Gate审计与Formal准入实验方案.md",
        }
        missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing Q18 sources: {missing}")

        ablations = pd.read_parquet(artifact_paths["ablation_metrics"])
        stability = pd.read_parquet(artifact_paths["stability_observations"])
        frontier = pd.read_parquet(artifact_paths["frontier_probes"])
        protocol = read_json(artifact_paths["protocol"])
        ablation_gate = read_json(artifact_paths["ablation_gate"])
        stability_gate = read_json(artifact_paths["stability_gate"])
        reach_gate = read_json(artifact_paths["reach_gate"])
        repaired_pilot_gate = read_json(artifact_paths["repaired_pilot_gate"])
        student_gate = read_json(artifact_paths["student_gate"])
        formal_gate = read_json(artifact_paths["formal_gate"])
        summary_gate = read_json(artifact_paths["summary_gate"])
        manifest = read_json(artifact_paths["artifact_manifest"])

        expected_rows = {
            "ablation_metrics": 60,
            "stability_observations": 2000,
            "frontier_probes": 256,
        }
        actual_rows = {
            "ablation_metrics": int(len(ablations)),
            "stability_observations": int(len(stability)),
            "frontier_probes": int(len(frontier)),
        }
        if actual_rows != expected_rows:
            raise RuntimeError(f"unexpected source row counts: {actual_rows}")

        source_files = {
            "v14_1_scientific_config_5239d24.yaml": (
                SCIENTIFIC_SHA,
                "configs/bacra_v14_1_cross_cell_repair.yaml",
            ),
            "v14_1_scientific_runner_5239d24.py": (
                SCIENTIFIC_SHA,
                "scripts/analysis/run_bacra_v14_1_cross_cell_repair.py",
            ),
            "v14_1_workspace_atlas_repair_5239d24.py": (
                SCIENTIFIC_SHA,
                "src/quasi_exp/teacher/workspace_atlas_repair.py",
            ),
            "v14_1_workspace_atlas_integration_5239d24.py": (
                SCIENTIFIC_SHA,
                "src/quasi_exp/teacher/workspace_atlas_integration.py",
            ),
            "v14_1_workspace_atlas_5239d24.py": (
                SCIENTIFIC_SHA,
                "src/quasi_exp/teacher/workspace_atlas.py",
            ),
            "v14_1_formal_runner_98dd93c.py": (
                FORMAL_AGGREGATION_SHA,
                "scripts/analysis/run_bacra_v14_1_cross_cell_repair.py",
            ),
            "v14_1_runner_tests_98dd93c.py": (
                FORMAL_AGGREGATION_SHA,
                "tests/test_bacra_v14_1_repair_runner.py",
            ),
            "v14_1_repair_tests_98dd93c.py": (
                FORMAL_AGGREGATION_SHA,
                "tests/test_workspace_atlas_repair_v14_1.py",
            ),
        }
        exported_sources: dict[str, Any] = {}
        for filename, (commit, relative_path) in source_files.items():
            destination = staging / filename
            destination.write_bytes(git_bytes(commit, relative_path))
            exported_sources[filename] = {
                **file_record(destination, relative_to=staging),
                "git_commit": commit,
                "repository_path": relative_path,
            }

        diff_name = "v14_1_5239d24_to_98dd93c.patch"
        diff_path = staging / diff_name
        diff_path.write_bytes(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(SOURCE_ROOT),
                    "diff",
                    f"{SCIENTIFIC_SHA}..{FORMAL_AGGREGATION_SHA}",
                    "--",
                    "configs/bacra_v14_1_cross_cell_repair.yaml",
                    "scripts/analysis/run_bacra_v14_1_cross_cell_repair.py",
                    "tests/test_bacra_v14_1_repair_runner.py",
                ],
                check=True,
                capture_output=True,
            ).stdout
        )

        stability_budget: dict[str, Any] = {}
        for budget, rows in stability.groupby("budget", sort=True):
            stability_budget[str(int(budget))] = {
                "row_count": int(len(rows)),
                "available_cluster_count_min": int(rows["available_cluster_count"].min()),
                "available_cluster_count_median": float(
                    rows["available_cluster_count"].median()
                ),
                "available_cluster_count_p95": quantile(
                    rows["available_cluster_count"], 0.95
                ),
                "available_cluster_count_max": int(rows["available_cluster_count"].max()),
                "same_fiber_component_count_median": float(
                    rows["same_fiber_component_count"].median()
                ),
                "same_fiber_component_count_max": int(
                    rows["same_fiber_component_count"].max()
                ),
            }

        frontier_by_round: dict[str, Any] = {}
        for round_id, rows in frontier.groupby("round_id", sort=True):
            frontier_by_round[str(int(round_id))] = {
                "probe_count": int(len(rows)),
                "found_valid_inverse_count": int(rows["found_valid_inverse"].astype(bool).sum()),
                "candidate_attempt_count_sum": int(rows["candidate_attempt_count"].sum()),
                "gold_candidate_count_sum": int(rows["gold_candidate_count"].sum()),
                "silver_candidate_count_sum": int(rows["silver_candidate_count"].sum()),
            }

        source_inventory: dict[str, Any] = {}
        for key, path in artifact_paths.items():
            if key == "ablation_metrics":
                source_inventory[key] = parquet_record(path, ablations)
            elif key == "stability_observations":
                source_inventory[key] = parquet_record(path, stability)
            elif key == "frontier_probes":
                source_inventory[key] = parquet_record(path, frontier)
            else:
                source_inventory[key] = file_record(path)

        output_value = {
            "schema_version": 1,
            "evidence_level": "diagnostic_only_post_hoc_full_table_aggregation",
            "claim_scope": (
                "summarizes existing V14.1 retry4 artifacts without changing thresholds, "
                "rerunning IK, or authorizing downstream stages"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation": {
                "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
                "command": (
                    f"{Path(os.sys.executable).resolve()} {Path(__file__).resolve()} "
                    f"--output {output}"
                ),
                "sampling": "none; all rows from all three attached Parquet sources",
                "random_seed": None,
            },
            "fixed_points": {
                "scientific_numeric_stages_git_sha": SCIENTIFIC_SHA,
                "formal_aggregation_git_sha": FORMAL_AGGREGATION_SHA,
                "current_source_head": git_text("rev-parse", "HEAD"),
                "current_source_branch": git_text("branch", "--show-current"),
                "current_source_worktree_clean": git_text("status", "--porcelain") == "",
                "artifact_protocol_source_git_sha": protocol["source_git_sha"],
                "formal_gate_upstream_scientific_source_git_sha": formal_gate[
                    "upstream_scientific_source_git_sha"
                ],
                "formal_gate_stage_source_git_sha": formal_gate["stage_source_git_sha"],
                "aggregation_only_hotfix": bool(formal_gate["aggregation_only_hotfix"]),
                "source_diff": {
                    **file_record(diff_path, relative_to=staging),
                    "filename": diff_name,
                },
            },
            "operational_integrity": {
                "artifact_manifest_verification": verify_manifest(manifest),
                "input_closure_all_match": bool(
                    read_json(artifact_paths["inventory_gate"])["input_closure"]["all_match"]
                ),
                "summary_gate_pass_is_operational_only": bool(summary_gate["gate_pass"]),
                "formal_generation_authorized": bool(
                    summary_gate["formal_generation_authorized"]
                ),
                "deployment_claim": bool(summary_gate["deployment_claim"]),
            },
            "patch_repair": {
                "ablation_gate": ablation_gate,
                "method_aggregates": {
                    ablation: aggregate_patch_method(ablations, ablation)
                    for ablation in sorted(ablations["ablation"].unique())
                },
                "row_level_fields_do_not_include_fresh_audit_submetrics": bool(
                    "audit_gate_pass" in ablations.columns
                    and not any(
                        token in str(column)
                        for column in ablations.columns
                        for token in ("cycle_p95", "cycle_max", "path_p95", "multipath")
                    )
                ),
                "retry3_retry4_shared_row_check": retry3_retry4_overlap_check(
                    ablations.columns
                ),
            },
            "canonical_selection_stability": {
                "gate": stability_gate,
                "by_budget": stability_budget,
                "candidate_cluster_growth_is_observed_but_not_itself_a_topological_branch_count": True,
            },
            "reach": {
                "gate": reach_gate,
                "frontier_probe_aggregates": frontier_by_round,
                "round5_threshold_margins": {
                    "weighted_jaccard_minus_minimum": float(
                        reach_gate["metrics_round4_5"][1]["volume_weighted_jaccard"] - 0.95
                    ),
                    "new_volume_minus_maximum": float(
                        reach_gate["metrics_round4_5"][1]["new_volume_ratio"] - 0.01
                    ),
                    "boundary_change_minus_maximum": float(
                        reach_gate["metrics_round4_5"][1]["boundary_change_ratio"] - 0.02
                    ),
                    "frontier_new_volume_minus_maximum": float(
                        reach_gate["metrics_round4_5"][1]["frontier_new_volume_ratio"] - 0.01
                    ),
                },
            },
            "downstream": {
                "repaired_pilot": repaired_pilot_gate,
                "student": student_gate,
                "formal": formal_gate,
                "formal_sentinel_warning": (
                    "labelable=0, unresolved=1, abstention=1, and n_min=1e12 are "
                    "fail-closed defaults after skipped upstream stages, not measured "
                    "workspace performance"
                ),
            },
            "source_inventory": source_inventory,
            "exported_sources": exported_sources,
        }
        audit_path = staging / "q18_retry4_full_table_audit.json"
        audit_path.write_text(
            json.dumps(strict(output_value), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({"output": str(output), "files": sorted(p.name for p in output.iterdir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
