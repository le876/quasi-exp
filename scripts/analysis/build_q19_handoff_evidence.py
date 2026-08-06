#!/usr/bin/env python3
"""Build independently auditable Q19 evidence from V14.1/V14.2 artifacts.

The script deliberately separates complete reproducibility archives from the
compact, directly readable tables used for the GPT-5 Pro decision.  Every
source artifact is checked against its sealed manifest before any derived file
is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V14_WORKTREE = PROJECT_ROOT / ".worktrees/bacra-v14-1-cross-cell-repair"
V141_ROOT = PROJECT_ROOT / "runs/bacra_v14_1_cross_cell_repair_retry4"
V142_ROOT = PROJECT_ROOT / "runs/bacra_v14_2_section_first_atlas"
OUTPUT_ROOT = PROJECT_ROOT / "runs/gpt5pro_handoff_evidence/Q19"
PYTHON = Path("/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python")
CONDA = Path("/mnt/ML_projects/conda/anaconda3/bin/conda")

V141_SCIENTIFIC_SHA = "5239d245d823223eaf282ba419eccb505e3e47e2"
V141_FORMAL_SHA = "98dd93cb4e3ab48cf75314868fbe7207a56c01b4"
V142_PARENT_SHA = "68b44f4272ca8f491e6904887a9fe49a0395bd4c"
V142_RUNTIME_SHA = "9223010a1e5b8c99c58d89d86b681c23f0262855"
PATCHES = ("patch_00", "patch_03", "patch_07", "patch_09")
AUDIT_FILES = (
    "fresh_edge_audit",
    "fresh_path_audit",
    "fresh_cycle_audit",
    "fresh_multipath_audit",
    "fresh_repeat_direction_audit",
)
GEOMETRIC_FAILURE_REASONS = frozenset(
    {
        "selected_endpoint_gap",
        "cycle_return_gap",
        "multipath_gap",
        "path_endpoint_gap",
        "repeat_gap",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def run_text(command: Sequence[str], *, cwd: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def git(*args: str) -> str:
    return run_text(("git", *args), cwd=V14_WORKTREE).strip()


@dataclass(frozen=True)
class ManifestClosure:
    artifact_count: int
    verified_count: int
    total_bytes: int
    failures: tuple[Mapping[str, Any], ...]


def verify_manifest(
    root: Path,
    manifest_path: Path,
    *,
    size_key: str,
) -> tuple[ManifestClosure, list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    rows = list(manifest["artifacts"])
    failures: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    total_bytes = 0
    for row in rows:
        relative = Path(str(row["path"]))
        path = root / relative
        expected_size = int(row[size_key])
        expected_sha = str(row["sha256"])
        actual_size = path.stat().st_size if path.is_file() else None
        actual_sha = sha256_file(path) if path.is_file() else None
        if actual_size != expected_size or actual_sha != expected_sha:
            failures.append(
                {
                    "path": str(relative),
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                }
            )
        total_bytes += expected_size
        normalized.append(
            {
                "path": str(relative),
                "size_bytes": expected_size,
                "sha256": expected_sha,
            }
        )
    declared = int(manifest.get("artifact_count", len(rows)))
    if declared != len(rows):
        failures.append({"artifact_count_declared": declared, "artifact_count_rows": len(rows)})
    closure = ManifestClosure(
        artifact_count=declared,
        verified_count=len(rows) - len([row for row in failures if "path" in row]),
        total_bytes=total_bytes,
        failures=tuple(failures),
    )
    if failures:
        raise RuntimeError(f"manifest verification failed for {manifest_path}: {failures[:3]}")
    return closure, normalized


def deterministic_zip(
    target: Path,
    members: Iterable[tuple[Path, str]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    records = []
    with zipfile.ZipFile(temporary, "w", compression=compression, allowZip64=True) as archive:
        for source, archive_name in sorted(members, key=lambda item: item[1]):
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, archive_name)
            records.append(
                {
                    "archive_name": archive_name,
                    "source_path": str(source),
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
    temporary.replace(target)
    return {
        "path": str(target),
        "member_count": len(records),
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "members": records,
    }


def build_source_archives() -> list[dict[str, Any]]:
    outputs = []
    for name, revision in (
        ("v14_1_scientific_source_5239d24.zip", V141_SCIENTIFIC_SHA),
        ("v14_1_formal_hotfix_source_98dd93c.zip", V141_FORMAL_SHA),
        ("v14_2_section_first_source_9223010.zip", V142_RUNTIME_SHA),
    ):
        target = OUTPUT_ROOT / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={temporary}", revision],
            cwd=V14_WORKTREE,
            check=True,
        )
        temporary.replace(target)
        outputs.append(
            {
                "path": str(target),
                "git_revision": revision,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    patch_path = OUTPUT_ROOT / "v14_2_parent_68b44f4_to_9223010.patch"
    write_text(patch_path, git("diff", "--binary", V142_PARENT_SHA, V142_RUNTIME_SHA) + "\n")
    outputs.append(
        {
            "path": str(patch_path),
            "git_revision": f"{V142_PARENT_SHA}..{V142_RUNTIME_SHA}",
            "size_bytes": patch_path.stat().st_size,
            "sha256": sha256_file(patch_path),
        }
    )
    return outputs


def build_retry4_archives(
    manifest_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact_members = [
        (V141_ROOT / str(row["path"]), f"runs/bacra_v14_1_cross_cell_repair_retry4/{row['path']}")
        for row in manifest_rows
    ]
    artifact_archive = deterministic_zip(
        OUTPUT_ROOT / "v14_1_retry4_complete_410_artifacts.zip",
        artifact_members,
    )

    input_members = []
    input_records = []
    expected = protocol["input_closure"]["expected_sha256"]
    for key, value in sorted(protocol["input_closure"]["paths"].items()):
        source = Path(str(value)).resolve()
        relative = source.relative_to(PROJECT_ROOT)
        actual_sha = sha256_file(source)
        if actual_sha != str(expected[key]):
            raise RuntimeError(f"sealed input hash mismatch: {key}")
        input_members.append((source, str(relative)))
        input_records.append(
            {
                "key": key,
                "project_relative_path": str(relative),
                "size_bytes": source.stat().st_size,
                "sha256": actual_sha,
            }
        )
    input_manifest_path = OUTPUT_ROOT / "v14_1_retry4_sealed_inputs_manifest.json"
    write_json(
        input_manifest_path,
        {
            "schema_version": 1,
            "source_protocol": str(V141_ROOT / "00_patch_inventory/protocol.json"),
            "input_count": len(input_records),
            "inputs": input_records,
        },
    )
    input_members.append((input_manifest_path, "SEALED_INPUTS_MANIFEST.json"))
    input_archive = deterministic_zip(
        OUTPUT_ROOT / "v14_1_retry4_complete_sealed_inputs.zip",
        input_members,
    )
    return artifact_archive, input_archive, read_json(input_manifest_path)


def beta_rms_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.degrees(left - right)), axis=1))


def selected_labels(directory: Path) -> pd.DataFrame:
    primary = pd.read_parquet(directory / "primary_section.parquet")
    hypotheses = pd.read_parquet(directory / "section_hypotheses.parquet")
    selected = hypotheses.loc[hypotheses["selected"]].copy()
    output = primary.merge(
        selected,
        left_on=("task_node_id", "primary_chart_id"),
        right_on=("task_node_id", "chart_id"),
        how="left",
        validate="one_to_one",
    )
    if output.loc[~output["abstained"], "candidate_id"].isna().any():
        raise RuntimeError(f"missing selected primary labels in {directory}")
    return output


def build_v142_tables() -> dict[str, Any]:
    mechanism_root = V142_ROOT / "02_mechanism_experiment"
    task_nodes = pd.read_parquet(V141_ROOT / "00_patch_inventory/patch_task_nodes.parquet")
    audit_frames = []
    hypothesis_frames = []
    primary_frames = []
    edge_frames = []
    comparison_frames = []
    report_rows = []
    stability_rows = []

    for patch in PATCHES:
        s4 = mechanism_root / patch / "S4"
        s8 = mechanism_root / patch / "S8"
        patch_tasks = task_nodes.loc[task_nodes["patch_id"] == patch].copy()
        for audit_name in AUDIT_FILES:
            source = s4 / f"{audit_name}.parquet"
            frame = pd.read_parquet(source)
            frame["patch_id"] = patch
            frame["audit_file"] = audit_name
            frame["source_relative_path"] = str(source.relative_to(PROJECT_ROOT))
            frame["source_sha256"] = sha256_file(source)
            frame["failure_class"] = np.where(
                frame["passed"],
                "passed",
                np.where(
                    frame["failure_reason"].isin(GEOMETRIC_FAILURE_REASONS),
                    "geometric_gap",
                    "numerical_solver_failure",
                ),
            )
            audit_frames.append(frame)

        hypotheses = pd.read_parquet(s4 / "section_hypotheses.parquet")
        hypotheses.insert(0, "patch_id", patch)
        hypotheses["source_relative_path"] = str(
            (s4 / "section_hypotheses.parquet").relative_to(PROJECT_ROOT)
        )
        hypotheses["source_sha256"] = sha256_file(s4 / "section_hypotheses.parquet")
        hypothesis_frames.append(hypotheses)

        primary = selected_labels(s4)
        primary.insert(0, "patch_id", patch)
        primary = primary.merge(
            patch_tasks,
            on=("patch_id", "task_node_id"),
            how="left",
            validate="one_to_one",
            suffixes=("", "_task"),
        )
        primary_frames.append(primary)

        edges = pd.read_parquet(s4 / "selected_edges.parquet")
        primary_ids = pd.read_parquet(s4 / "primary_section.parquet").set_index("task_node_id")[
            "primary_chart_id"
        ]
        edges.insert(0, "patch_id", patch)
        edges["left_primary_chart_id"] = edges["left_node_id"].map(primary_ids)
        edges["right_primary_chart_id"] = edges["right_node_id"].map(primary_ids)
        edges["edge_in_primary_partition"] = (
            (edges["chart_id"] == edges["left_primary_chart_id"])
            & (edges["chart_id"] == edges["right_primary_chart_id"])
        )
        edge_frames.append(edges)

        left = selected_labels(s4)
        right = selected_labels(s8)
        beta_columns = [f"beta{index}_rad" for index in range(1, 7)]
        comparison = left[["task_node_id", "primary_chart_id", "abstained", *beta_columns]].merge(
            right[["task_node_id", "primary_chart_id", "abstained", *beta_columns]],
            on="task_node_id",
            how="outer",
            validate="one_to_one",
            suffixes=("_s4", "_s8"),
        )
        left_beta = comparison[[f"{name}_s4" for name in beta_columns]].to_numpy(dtype=float)
        right_beta = comparison[[f"{name}_s8" for name in beta_columns]].to_numpy(dtype=float)
        comparison.insert(0, "patch_id", patch)
        comparison["selected_beta_gap_deg"] = beta_rms_deg(left_beta, right_beta)
        comparison["chart_assignment_equal"] = (
            comparison["primary_chart_id_s4"] == comparison["primary_chart_id_s8"]
        )
        comparison["abstention_equal"] = comparison["abstained_s4"] == comparison["abstained_s8"]
        comparison_frames.append(comparison)

        for method in ("F0", "F1", "S2", "S4", "S4C", "S8"):
            source = (
                V142_ROOT / "01_fresh_audit_diagnostics" / patch / method / "report.json"
                if method == "F0"
                else mechanism_root / patch / method / "report.json"
            )
            if source.is_file():
                row = dict(read_json(source))
                row["source_relative_path"] = str(source.relative_to(PROJECT_ROOT))
                row["source_sha256"] = sha256_file(source)
                report_rows.append(row)
        stability_path = mechanism_root / patch / "S4_vs_S8_stability.json"
        stability = dict(read_json(stability_path))
        stability["source_relative_path"] = str(stability_path.relative_to(PROJECT_ROOT))
        stability["source_sha256"] = sha256_file(stability_path)
        stability_rows.append(stability)

    audits = pd.concat(audit_frames, ignore_index=True)
    hypotheses = pd.concat(hypothesis_frames, ignore_index=True)
    primary = pd.concat(primary_frames, ignore_index=True)
    edges = pd.concat(edge_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    reports = pd.DataFrame.from_records(report_rows)
    stability = pd.DataFrame.from_records(stability_rows)

    paths = {
        "fresh_audits": OUTPUT_ROOT / "v14_2_s4_full_fresh_audits.parquet",
        "failed_audits": OUTPUT_ROOT / "v14_2_s4_failed_fresh_audits.parquet",
        "hypotheses": OUTPUT_ROOT / "v14_2_s4_full_chart_hypotheses.parquet",
        "primary": OUTPUT_ROOT / "v14_2_s4_primary_atlas.parquet",
        "edges": OUTPUT_ROOT / "v14_2_s4_full_selected_edges.parquet",
        "comparison": OUTPUT_ROOT / "v14_2_s4_vs_s8_full_comparison.parquet",
        "reports": OUTPUT_ROOT / "v14_2_all_mechanism_reports.parquet",
        "stability": OUTPUT_ROOT / "v14_2_s4_vs_s8_stability.parquet",
    }
    write_parquet(paths["fresh_audits"], audits)
    write_parquet(paths["failed_audits"], audits.loc[~audits["passed"]].copy())
    write_parquet(paths["hypotheses"], hypotheses)
    write_parquet(paths["primary"], primary)
    write_parquet(paths["edges"], edges)
    write_parquet(paths["comparison"], comparisons)
    write_parquet(paths["reports"], reports)
    write_parquet(paths["stability"], stability)

    failed = audits.loc[~audits["passed"]]
    failure_summary = (
        audits.groupby(["patch_id", "audit_file", "failure_class"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "tables": {
            key: {
                "path": str(path),
                "rows": int(len(pd.read_parquet(path))),
                "columns": int(len(pd.read_parquet(path).columns)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
        "fresh_audit": {
            "row_count": int(len(audits)),
            "passed_row_count": int(audits["passed"].sum()),
            "failed_row_count": int((~audits["passed"]).sum()),
            "numerical_solver_failure_count": int(
                (failed["failure_class"] == "numerical_solver_failure").sum()
            ),
            "geometric_gap_failure_count": int(
                (failed["failure_class"] == "geometric_gap").sum()
            ),
            "failure_summary": failure_summary,
        },
        "s4_s8": {
            "row_count": int(len(comparisons)),
            "coverage_equal": bool(comparisons["abstention_equal"].all()),
            "chart_assignment_equal": bool(comparisons["chart_assignment_equal"].all()),
            "selected_beta_p95_deg": float(np.percentile(comparisons["selected_beta_gap_deg"], 95)),
            "selected_beta_max_deg": float(comparisons["selected_beta_gap_deg"].max()),
        },
        "primary_atlas": {
            "row_count": int(len(primary)),
            "covered_row_count": int((~primary["abstained"]).sum()),
            "patch_count": int(primary["patch_id"].nunique()),
            "chart_count_by_patch": {
                str(key): int(value)
                for key, value in primary.groupby("patch_id")["primary_chart_id"].nunique().items()
            },
        },
    }


def build_runtime_evidence(protocol: Mapping[str, Any]) -> dict[str, Any]:
    package_names = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "tensorflow")
    current_packages = {}
    for name in package_names:
        try:
            current_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            current_packages[name] = None
    conda_explicit = run_text(
        (str(CONDA), "list", "--explicit", "--prefix", str(PYTHON.parent.parent))
    )
    pip_freeze = run_text((str(PYTHON), "-m", "pip", "freeze"))
    write_text(OUTPUT_ROOT / "runtime_conda_explicit.txt", conda_explicit)
    write_text(OUTPUT_ROOT / "runtime_pip_freeze.txt", pip_freeze)
    history_source = PYTHON.parent.parent / "conda-meta/history"
    history_target = OUTPUT_ROOT / "runtime_conda_history.txt"
    shutil.copyfile(history_source, history_target)

    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    try:
        gpu = run_text(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            )
        ).strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = []

    evidence = {
        "schema_version": 1,
        "sealed_retry4_runtime": protocol["runtime"],
        "current_reproduction_environment": {
            "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "executable": str(PYTHON),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "logical_cpu_count": __import__("os").cpu_count(),
            "gpu": gpu,
            "packages": current_packages,
            "conda_explicit_sha256": sha256_file(OUTPUT_ROOT / "runtime_conda_explicit.txt"),
            "pip_freeze_sha256": sha256_file(OUTPUT_ROOT / "runtime_pip_freeze.txt"),
            "conda_history_sha256": sha256_file(history_target),
        },
        "thread_policy": {
            "patch_workers": 12,
            "numerical_threads_per_worker": 1,
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
    }
    write_json(OUTPUT_ROOT / "runtime_environment.json", evidence)
    return evidence


def build_reproduction_record() -> None:
    command_paths = (
        PROJECT_ROOT / "runs/maintenance/v14_1_retry4_ablations_state/current.command.sh",
        PROJECT_ROOT / "runs/maintenance/v14_1_retry4_stability_state/current.command.sh",
        PROJECT_ROOT / "runs/maintenance/v14_1_retry4_reach_state/current.command.sh",
    )
    blocks = []
    for path in command_paths:
        blocks.append(f"## {path.name}: {path.parent.name}\n\n```bash\n{path.read_text(encoding='utf-8').rstrip()}\n```\n")
    lineage = git("log", "--oneline", "--decorate=no", f"{V141_SCIENTIFIC_SHA}^..{V142_RUNTIME_SHA}")
    body = """# BACRA V14.1 retry4 / V14.2 reproduction record

This file preserves the three retry4 long-run command wrappers that still
exist in the canonical launcher state directories.  A scientific Gate failure
returned process code 2 by design; it is not an operational crash.  Inventory,
downstream skip/Formal aggregation, and summary are represented by the source
runner, sealed protocol, stage Gates, and complete artifact archive; their
original shell wrappers were not retained as separate files.

The complete source snapshots, sealed inputs, and retry4 artifacts in this
handoff are intended to be extracted below the project root so the recorded
relative paths are restored.  Absolute environment and project paths remain
the paths used by the original run.

## Relevant Git lineage

```text
""" + lineage + "\n```\n\n" + "\n".join(blocks)
    write_text(OUTPUT_ROOT / "reproduction_record.md", body)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = read_json(V141_ROOT / "00_patch_inventory/protocol.json")
    v141_closure, v141_rows = verify_manifest(
        V141_ROOT,
        V141_ROOT / "07_summary/artifact_manifest.json",
        size_key="size_bytes",
    )
    v142_closure, _ = verify_manifest(
        V142_ROOT,
        V142_ROOT / "08_summary/artifact_manifest.json",
        size_key="bytes",
    )
    if protocol["source_git_sha"] != V141_SCIENTIFIC_SHA:
        raise RuntimeError("retry4 protocol source SHA drift")
    if git("rev-parse", "HEAD") != V142_RUNTIME_SHA:
        raise RuntimeError("V14.2 worktree HEAD drift")
    if git("status", "--short"):
        raise RuntimeError("V14.2 worktree must be clean")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", V142_PARENT_SHA, V142_RUNTIME_SHA],
        cwd=V14_WORKTREE,
        check=True,
    )

    source_archives = build_source_archives()
    artifact_archive, input_archive, input_manifest = build_retry4_archives(
        v141_rows,
        protocol,
    )
    v142_tables = build_v142_tables()
    runtime = build_runtime_evidence(protocol)
    build_reproduction_record()

    gate_paths = {
        "v14_1_summary": V141_ROOT / "07_summary/summary_report.json",
        "v14_1_ablation": V141_ROOT / "01_patch_ablations/gate.json",
        "v14_1_stability": V141_ROOT / "02_selection_stability/gate.json",
        "v14_1_reach": V141_ROOT / "03_reach_extension/gate.json",
        "v14_1_formal": V141_ROOT / "06_formal_admission/gate.json",
        "v14_2_diagnostics": V142_ROOT / "01_fresh_audit_diagnostics/gate.json",
        "v14_2_mechanisms": V142_ROOT / "02_mechanism_experiment/gate.json",
        "v14_2_confirmation": V142_ROOT / "03_twelve_patch_confirmation/gate.json",
        "v14_2_reach": V142_ROOT / "04_reach_round6/gate.json",
        "v14_2_repaired_pilot": V142_ROOT / "05_repaired_5k_pilot/gate.json",
        "v14_2_student": V142_ROOT / "06_exploratory_student/gate.json",
        "v14_2_formal": V142_ROOT / "07_formal_admission/gate.json",
        "v14_2_summary": V142_ROOT / "08_summary/summary_report.json",
    }
    summary = {
        "schema_version": 1,
        "evidence_cutoff": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fixed_points": {
            "v14_1_scientific": V141_SCIENTIFIC_SHA,
            "v14_1_formal_hotfix": V141_FORMAL_SHA,
            "v14_2_requested_parent": V142_PARENT_SHA,
            "v14_2_runtime": V142_RUNTIME_SHA,
            "v14_2_parent_is_ancestor": True,
            "v14_2_worktree_clean": True,
        },
        "manifest_closure": {
            "v14_1_retry4": vars(v141_closure),
            "v14_2": vars(v142_closure),
        },
        "complete_archives": {
            "v14_1_retry4_artifacts": artifact_archive,
            "v14_1_retry4_sealed_inputs": input_archive,
            "source_archives": source_archives,
        },
        "sealed_input_manifest": input_manifest,
        "v14_2_tables": v142_tables,
        "runtime": runtime,
        "gates": {
            name: {
                "source_path": str(path),
                "source_sha256": sha256_file(path),
                "value": read_json(path),
            }
            for name, path in gate_paths.items()
        },
        "provenance_boundary": {
            "v14_2_inventory_source_git_sha": read_json(
                V142_ROOT / "00_inventory/gate.json"
            )["protocol"]["source_git_sha"],
            "v14_2_successful_retry_runtime_sha": V142_RUNTIME_SHA,
            "two_fixed_point_diagnostic_retry": True,
            "formal_eligible": False,
        },
    }
    write_json(OUTPUT_ROOT / "q19_verified_evidence_summary.json", summary)
    print(json.dumps({
        "output_root": str(OUTPUT_ROOT),
        "v14_1_artifacts": vars(v141_closure),
        "v14_2_artifacts": vars(v142_closure),
        "fresh_audit": v142_tables["fresh_audit"],
        "s4_s8": v142_tables["s4_s8"],
    }, ensure_ascii=False, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
