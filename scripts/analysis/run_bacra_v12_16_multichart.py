#!/usr/bin/env python3
"""Run BACRA V12.16A chart-B region readiness experiment.

V12.16A is intentionally a data-readiness checkpoint.  It expands the
registered chart-B stress02 cycle into a connected multi-parent region and
freezes whole spatial blocks before any multi-chart Student is trained.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = SOURCE_ROOT.parent.parent
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import run_bacra_v12 as v12
import run_bacra_v12_14_region_growth as v14
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.retention_distillation import (
    SpatialBlockPolicy,
    assign_whole_spatial_blocks,
)
from run_trajectory_canonical_teacher_v10 import (
    runtime_fingerprint,
)


PROTOCOL_ID = (
    "branch-aware-canonical-region-atlas-v12.16-"
    "chart-b-region-spatial-seal"
)
CLAIM_SCOPE = "simulation_chart_b_region_readiness"
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)
STAGES = ("protocol", "chart_b_region", "spatial_seal")


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


def _values(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["v12_16"]


def _source_files(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    values = _values(config)
    v15 = _source_path(
        project_root, str(values["source_v12_15_root"])
    )
    return {
        "v12_15_gate": v15 / "08_summary/gate.json",
        "v12_15_model_lock": v15 / "05_model_lock/final_model_lock.json",
        "v12_15_training_dataset": v15
        / "02_distillation_dataset/train_validation.parquet",
        "chart_b_gate": _source_path(
            project_root, str(values["source_chart_b_gate"])
        ),
        "chart_b_centerline": _source_path(
            project_root, str(config["v12_14"]["source_d3"])
        ),
    }


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


def _require_gate(output_root: Path, relative: str) -> dict[str, Any]:
    path = output_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required upstream gate is missing: {path}")
    payload = json.loads(path.read_text())
    if not bool(payload["gate_pass"]):
        raise RuntimeError(f"upstream gate failed: {path}")
    return payload


def stage_protocol(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    stage = output_root / "00_protocol"
    stage.mkdir(parents=True, exist_ok=False)
    sources = _source_files(config, project_root)
    expected = dict(_values(config)["expected_source_sha256"])
    actual = {
        name: sha256_file(path) if path.is_file() else "missing"
        for name, path in sources.items()
    }
    source_gates: dict[str, Any] = {}
    for name in ("v12_15_gate", "chart_b_gate"):
        path = sources[name]
        source_gates[name] = (
            json.loads(path.read_text()) if path.is_file() else {}
        )
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
    implementation = {
        "runner": Path(__file__).resolve(),
        "config": Path(str(config["config_path"])).resolve(),
        "region_runner": SOURCE_ROOT
        / "scripts/analysis/run_bacra_v12_14_region_growth.py",
        "region_module": SOURCE_ROOT
        / "src/quasi_exp/teacher/region_growth.py",
        "spatial_seal_module": SOURCE_ROOT
        / "src/quasi_exp/teacher/retention_distillation.py",
    }
    atomic_write_json(
        stage / "frozen_config.json",
        {key: value for key, value in config.items() if key != "config_path"},
    )
    atomic_write_json(
        stage / "source_manifest.json",
        {
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
        },
    )
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
            "v12_15_source_gate_pass": bool(
                source_gates["v12_15_gate"].get("gate_pass")
            ),
            "chart_b_known_expert_gate_pass": bool(
                source_gates["chart_b_gate"].get("gate_pass")
            ),
            "chart_b_centerline_has_registered_identity": bool(
                pd.read_parquet(
                    sources["chart_b_centerline"], columns=["chart_id"]
                )["chart_id"].eq("chart_B").all()
            ),
            "deployment_claim_disabled": config["deployment_claim_gate_pass"]
            is False,
        },
        semantics="v12_16_chart_b_region_source_closure",
        source_sha256=actual,
        implementation_sha256={
            name: sha256_file(path) for name, path in implementation.items()
        },
        git_sha=git_sha,
        upstream_claim_scope={
            name: source_gates[name].get("claim_scope")
            for name in source_gates
        },
    )


def _v14_chart_b_config(config: Mapping[str, Any]) -> dict[str, Any]:
    compatible = copy.deepcopy(dict(config))
    compatible["claim_scope"] = v14.CLAIM_SCOPE
    compatible["deployment_claim_gate_pass"] = False
    return compatible


def stage_chart_b_region(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "00_protocol/gate.json")
    stage = output_root / "01_chart_b_region"
    compatible = _v14_chart_b_config(config)
    runners = (
        ("protocol", lambda: v14.stage_protocol(compatible, project_root, stage)),
        (
            "seed_cleanup",
            lambda: v14.stage_seed_cleanup(compatible, project_root, stage),
        ),
        (
            "capability_region",
            lambda: v14.stage_capability_region(
                compatible, project_root, stage
            ),
        ),
        (
            "sparse_growth",
            lambda: v14.stage_sparse_growth(
                compatible, project_root, stage, python=python
            ),
        ),
        (
            "consistency",
            lambda: v14.stage_consistency(
                compatible, project_root, stage
            ),
        ),
        (
            "dense_generation",
            lambda: v14.stage_dense(
                compatible, project_root, stage, python=python
            ),
        ),
        (
            "dataset_assemble",
            lambda: v14.stage_dataset(
                compatible, project_root, stage, python=python
            ),
        ),
    )
    reports: dict[str, Any] = {}
    for name, runner in runners:
        report = runner()
        reports[name] = report
        if not bool(report["gate_pass"]):
            break
    completed = list(reports)
    expected = [name for name, _runner in runners]
    nested_pass = completed == expected and all(
        bool(report["gate_pass"]) for report in reports.values()
    )
    dense_path = stage / "05_dense_region/dense_region_100k.parquet"
    dataset_path = stage / "06_datasets/R2_dataset.parquet"
    dense_rows = (
        len(pd.read_parquet(dense_path, columns=["x_m"]))
        if dense_path.is_file()
        else 0
    )
    dataset_rows = (
        len(pd.read_parquet(dataset_path, columns=["x_m"]))
        if dataset_path.is_file()
        else 0
    )
    return _gate(
        stage / "gate.json",
        {
            "all_registered_region_stages_complete": nested_pass,
            "chart_b_dense_region_written": dense_rows
            == int(config["v12_14"]["dense"]["formal_accepted_count"]),
            "chart_b_R2_dataset_written": dataset_rows
            == int(config["v12_14"]["dataset"]["R2_old_rows"])
            + int(config["v12_14"]["dataset"]["R2_region_rows"]),
        },
        semantics="chart_b_connected_region_readiness",
        completed_stages=completed,
        nested_stage_gate_pass={
            name: bool(report["gate_pass"])
            for name, report in reports.items()
        },
        dense_rows=int(dense_rows),
        dataset_rows=int(dataset_rows),
        chart_scope="chart_B",
    )


def _spatial_policy(config: Mapping[str, Any]) -> SpatialBlockPolicy:
    values = _values(config)["spatial_holdout"]
    return SpatialBlockPolicy(
        macro_voxel_mm=float(values["macro_voxel_mm"]),
        validation_fraction=float(values["validation_fraction"]),
        sealed_holdout_fraction=float(values["sealed_holdout_fraction"]),
        train_buffer_mm=float(values["train_buffer_mm"]),
    )


def stage_spatial_seal(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    del project_root
    _require_gate(output_root, "01_chart_b_region/gate.json")
    stage = output_root / "02_chart_b_spatial_seal"
    stage.mkdir(parents=True, exist_ok=False)
    source = output_root / "01_chart_b_region/06_datasets/R2_dataset.parquet"
    frame = pd.read_parquet(source)
    partitioned, report = assign_whole_spatial_blocks(
        frame,
        region_mask=frame["sampling_bucket"].ne("old").to_numpy(),
        policy=_spatial_policy(config),
        seed=int(_values(config)["seeds"]["spatial_partition"]),
    )
    partitioned["chart_id"] = "chart_B"
    partitioned["chart_feature"] = 1.0
    admitted = partitioned.loc[
        partitioned["v12_15_split"].isin(["train", "validation"])
    ].copy()
    v14._atomic_parquet(admitted, stage / "train_validation.parquet")
    registry = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "chart_id": "chart_B",
        "partition_seed": int(_values(config)["seeds"]["spatial_partition"]),
        "macro_voxel_mm": float(
            _values(config)["spatial_holdout"]["macro_voxel_mm"]
        ),
        "sealed_block_keys": report["sealed_block_keys"],
        "sealed_rows": int(report["sealed_rows"]),
        "source_dataset_sha256": sha256_file(source),
        "sealed_labels_materialized_before_model_lock": False,
    }
    atomic_write_json(stage / "sealed_block_registry.json", registry)
    atomic_write_json(stage / "partition_report.json", report)
    source_blocks = set(map(int, report["sealed_block_keys"]))
    admitted_blocks = set(
        map(
            int,
            admitted.loc[
                admitted["v12_15_split"].eq("train"),
                "spatial_block_key",
            ].unique(),
        )
    )
    return _gate(
        stage / "gate.json",
        {
            "chart_b_has_train_blocks": report["train_macro_block_count"] > 0,
            "chart_b_has_validation_blocks": report[
                "validation_macro_block_count"
            ]
            > 0,
            "chart_b_has_sealed_blocks": report["sealed_macro_block_count"] > 0,
            "sealed_rows_excluded_from_training_artifact": not admitted[
                "v12_15_split"
            ].eq("sealed_holdout").any(),
            "buffer_rows_excluded_from_training_artifact": not admitted[
                "v12_15_split"
            ].eq("buffer_excluded").any(),
            "train_and_sealed_blocks_disjoint": source_blocks.isdisjoint(
                admitted_blocks
            ),
            "chart_identity_explicit": admitted["chart_id"].eq("chart_B").all(),
            "sealed_labels_not_materialized": not (
                stage / "sealed_teacher_reference.parquet"
            ).exists(),
        },
        semantics="chart_b_whole_spatial_block_pretraining_seal",
        partition_report=report,
        training_artifact_rows=int(len(admitted)),
        training_artifact_sha256=sha256_file(
            stage / "train_validation.parquet"
        ),
        sealed_registry_sha256=sha256_file(
            stage / "sealed_block_registry.json"
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = _output_root(config, project_root, args.output_root)
    stages = {
        "protocol": lambda: stage_protocol(config, project_root, output_root),
        "chart_b_region": lambda: stage_chart_b_region(
            config,
            project_root,
            output_root,
            python=Path(args.python),
        ),
        "spatial_seal": lambda: stage_spatial_seal(
            config, project_root, output_root
        ),
    }
    requested = STAGES if args.stage == "all" else (str(args.stage),)
    reports: dict[str, Any] = {}
    for name in requested:
        report = stages[name]()
        reports[name] = report
        if not bool(report["gate_pass"]):
            break
    summary = {
        "protocol_id": PROTOCOL_ID,
        "claim_scope": CLAIM_SCOPE,
        "requested_stage": str(args.stage),
        "completed_stages": list(reports),
        "stage_gate_pass": {
            name: bool(report["gate_pass"])
            for name, report in reports.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(SOURCE_ROOT / "configs/bacra_v12_16_multichart.yaml"),
    )
    parser.add_argument("--preset", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--output-root")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    return 0 if all(summary["stage_gate_pass"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
