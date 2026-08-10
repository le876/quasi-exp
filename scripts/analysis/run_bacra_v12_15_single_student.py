#!/usr/bin/env python3
"""Run BACRA V12.15 single-Student retention distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
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
import run_bacra_v12_14_region_growth as v14
import run_bacra_v12_7_canonical_student as canonical_runner
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.gold_set_student import save_uncompiled_model
from quasi_exp.teacher.retention_distillation import (
    DISTILL_BETA_COLUMNS,
    SpatialBlockPolicy,
    assign_whole_spatial_blocks,
    spatial_block_keys,
    train_retention_distilled_student,
)
from run_trajectory_canonical_teacher_v10 import runtime_fingerprint


PROTOCOL_ID = (
    "branch-aware-canonical-region-atlas-v12.15-"
    "single-student-spatial-block-holdout"
)
CLAIM_SCOPE = "simulation_single_chart_spatial_block_generalization"
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)


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
    return config["v12_15"]


def _source_root(config: Mapping[str, Any], project_root: Path) -> Path:
    return _source_path(
        project_root, str(_values(config)["source_v12_14_root"])
    )


def _source_files(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    root = _source_root(config, project_root)
    return {
        "v12_14_gate": root / "11_summary/gate.json",
        "v12_14_model_lock": root
        / "10_active_enrichment/final_model_lock.json",
        "v12_14_R2_dataset": root / "06_datasets/R2_dataset.parquet",
        "final8_teacher": _source_path(
            project_root, str(_values(config)["source_final8_teacher"])
        ),
        "final8_catalog": _source_path(
            project_root, str(_values(config)["source_final8_catalog"])
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
    if not payload["gate_pass"]:
        raise RuntimeError(f"upstream gate failed: {path}")
    return payload


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    v14._atomic_parquet(frame, path)


def _spatial_policy(config: Mapping[str, Any]) -> SpatialBlockPolicy:
    values = _values(config)["spatial_holdout"]
    return SpatialBlockPolicy(
        macro_voxel_mm=float(values["macro_voxel_mm"]),
        validation_fraction=float(values["validation_fraction"]),
        sealed_holdout_fraction=float(values["sealed_holdout_fraction"]),
        train_buffer_mm=float(values["train_buffer_mm"]),
    )


def _partition_source(
    config: Mapping[str, Any], project_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(
        _source_files(config, project_root)["v12_14_R2_dataset"]
    )
    region_mask = frame["sampling_bucket"].ne("old").to_numpy()
    return assign_whole_spatial_blocks(
        frame,
        region_mask=region_mask,
        policy=_spatial_policy(config),
        seed=int(_values(config)["seeds"]["spatial_partition"]),
    )


def stage_protocol(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    stage = output_root / "00_protocol"
    stage.mkdir(parents=True, exist_ok=False)
    sources = _source_files(config, project_root)
    expected = dict(_values(config)["expected_source_sha256"])
    actual = {
        name: sha256_file(path) if path.is_file() else "missing"
        for name, path in sources.items()
    }
    source_gate = (
        json.loads(sources["v12_14_gate"].read_text())
        if sources["v12_14_gate"].is_file()
        else {}
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
    config_snapshot = {
        key: value for key, value in config.items() if key != "config_path"
    }
    atomic_write_json(stage / "frozen_config.json", config_snapshot)
    implementation = {
        "runner": Path(__file__).resolve(),
        "distillation": SOURCE_ROOT
        / "src/quasi_exp/teacher/retention_distillation.py",
        "v12_14_runner": SOURCE_ROOT
        / "scripts/analysis/run_bacra_v12_14_region_growth.py",
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
            "v12_14_formal_gate_passed": bool(source_gate.get("gate_pass")),
            "source_worktree_clean": status.strip() == "",
            "single_chart_claim_scope_frozen": config["claim_scope"]
            == CLAIM_SCOPE,
            "deployment_claim_disabled": config["deployment_claim_gate_pass"]
            is False,
        },
        semantics="v12_15_clean_fixed_point_and_v12_14_source_closure",
        source_sha256=actual,
        implementation_sha256={
            name: sha256_file(path) for name, path in implementation.items()
        },
        git_sha=git_sha,
        source_v12_14_gate_semantics=source_gate.get("gate_semantics"),
        chart_scope="chart_A_only",
    )


def stage_spatial_seal(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_gate(output_root, "00_protocol/gate.json")
    stage = output_root / "01_spatial_seal"
    stage.mkdir(parents=True, exist_ok=True)
    _partitioned, report = _partition_source(config, project_root)
    registry = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "partition_seed": report["partition_seed"],
        "macro_voxel_mm": report["macro_voxel_mm"],
        "sealed_block_keys": report["sealed_block_keys"],
        "sealed_block_count": report["sealed_macro_block_count"],
        "sealed_labels_opened": False,
        "role": "model_locked_spatial_holdout",
        "source_dataset_sha256": _values(config)["expected_source_sha256"][
            "v12_14_R2_dataset"
        ],
    }
    atomic_write_json(stage / "sealed_block_registry.json", registry)
    atomic_write_json(stage / "partition_report.json", report)
    return _gate(
        stage / "gate.json",
        {
            "whole_blocks_registered": report["sealed_macro_block_count"] > 0,
            "validation_blocks_registered": report[
                "validation_macro_block_count"
            ]
            > 0,
            "training_blocks_remain": report["train_macro_block_count"] > 0,
            "sealed_labels_not_materialized": not (
                stage / "sealed_teacher_reference.parquet"
            ).exists(),
        },
        semantics="whole_spatial_blocks_registered_before_training",
        partition_report=report,
        sealed_registry_sha256=sha256_file(
            stage / "sealed_block_registry.json"
        ),
    )


def stage_dataset(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    seal_gate = _require_gate(output_root, "01_spatial_seal/gate.json")
    stage = output_root / "02_distillation_dataset"
    stage.mkdir(parents=True, exist_ok=True)
    partitioned, report = _partition_source(config, project_root)
    usable = partitioned.loc[
        partitioned["v12_15_split"].isin(["train", "validation"])
    ].copy()
    path = stage / "train_validation.parquet"
    _atomic_parquet(usable, path)
    sealed_keys = set(
        json.loads(
            (output_root / "01_spatial_seal/sealed_block_registry.json").read_text()
        )["sealed_block_keys"]
    )
    leaked = usable["spatial_block_key"].isin(sealed_keys)
    manifest = {
        "schema_version": 1,
        "row_count": int(len(usable)),
        "train_rows": int(usable["v12_15_split"].eq("train").sum()),
        "validation_rows": int(
            usable["v12_15_split"].eq("validation").sum()
        ),
        "sampling_bucket_counts": {
            str(key): int(value)
            for key, value in usable["sampling_bucket"].value_counts().items()
        },
        "sha256": sha256_file(path),
        "sealed_registry_sha256": seal_gate["sealed_registry_sha256"],
        "sealed_rows_written": int(leaked.sum()),
        "distill_targets_materialized_by_seed_worker": True,
    }
    atomic_write_json(stage / "dataset_manifest.json", manifest)
    return _gate(
        stage / "gate.json",
        {
            "train_rows_present": manifest["train_rows"] > 0,
            "validation_rows_present": manifest["validation_rows"] > 0,
            "all_sampling_buckets_present": set(
                manifest["sampling_bucket_counts"]
            )
            == {"old", "interior", "boundary"},
            "no_sealed_block_rows_written": not leaked.any(),
            "sealed_labels_still_unopened": not (
                output_root
                / "01_spatial_seal/sealed_teacher_reference.parquet"
            ).exists(),
        },
        semantics="retention_replay_dataset_excluding_registered_spatial_blocks",
        dataset_manifest=manifest,
        partition_report=report,
    )


def _load_composite_targets(
    frame: pd.DataFrame,
    *,
    seed: int,
    model_lock: Mapping[str, Any],
) -> np.ndarray:
    import tensorflow as tf

    source = model_lock["model_sources"][str(int(seed))]
    base = tf.keras.models.load_model(source["base_model"], compile=False)
    region = tf.keras.models.load_model(source["region_model"], compile=False)
    predictor = v14._ResidualCompositePredictor(
        base, region, float(model_lock["alpha"])
    )
    return np.asarray(
        predictor.predict(
            frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
            batch_size=2048,
            verbose=0,
        ),
        dtype=float,
    )


def _worker_distill(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("V12.15 distillation requires a visible GPU")
    values = _values(config)
    variant = dict(values["distillation"]["variants"][str(args.variant)])
    seed = int(args.seed)
    dataset = pd.read_parquet(args.dataset)
    source_lock_path = _source_files(config, project_root)[
        "v12_14_model_lock"
    ]
    source_lock = json.loads(source_lock_path.read_text())
    composite_target = _load_composite_targets(
        dataset, seed=seed, model_lock=source_lock
    )
    for index, column in enumerate(DISTILL_BETA_COLUMNS):
        dataset[column] = composite_target[:, index]
    environment = v14._environment(config, project_root)
    composite_margin = v14.point_margin_deg(
        composite_target, environment.bounds
    )
    training = values["distillation"]
    margin_tail = (
        dataset["v12_15_split"].eq("train").to_numpy()
        & (
            composite_margin
            < float(training["margin_tail_threshold_deg"])
        )
    )
    dataset.loc[margin_tail, "sample_weight"] *= float(
        training["margin_tail_sample_weight_multiplier"]
    )
    train = dataset.loc[dataset["v12_15_split"].eq("train")].copy()
    validation = dataset.loc[
        dataset["v12_15_split"].eq("validation")
    ].copy()
    source = source_lock["model_sources"][str(seed)]
    initial_model = Path(source["base_model"]).resolve()
    model = tf.keras.models.load_model(initial_model, compile=False)
    model, history, report = train_retention_distilled_student(
        model,
        train,
        validation,
        geometry=canonical_runner._geometry(config, project_root),
        seed=seed,
        learning_rate=float(variant["learning_rate"]),
        batch_size=int(training["batch_size"]),
        max_steps=int(variant["max_optimizer_steps"]),
        validation_interval=int(training["validation_interval_steps"]),
        patience_intervals=int(training["patience_intervals"]),
        sampling_weights=dict(training["sampling_weights"]),
        teacher_beta_weight=float(variant["teacher_beta_weight"]),
        distill_beta_weight=float(variant["distill_beta_weight"]),
        fk_weight=float(variant["fk_weight"]),
        margin_weight=float(variant["margin_weight"]),
        row_loss_weight=float(variant["row_loss_weight"]),
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
            "variant": str(args.variant),
            "model_mode": "single_bounded_mlp",
            "initial_model": str(initial_model),
            "initial_model_sha256": sha256_file(initial_model),
            "source_composite_alpha": float(source_lock["alpha"]),
            "source_base_sha256": source["base_model_sha256"],
            "source_region_sha256": source["region_model_sha256"],
            "source_model_lock_sha256": sha256_file(source_lock_path),
            "device": devices[0].name,
            "margin_tail_threshold_deg": float(
                training["margin_tail_threshold_deg"]
            ),
            "margin_tail_sample_weight_multiplier": float(
                training["margin_tail_sample_weight_multiplier"]
            ),
            "margin_tail_training_rows": int(np.count_nonzero(margin_tail)),
            "validation_metrics": v14._prediction_metrics(
                environment,
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


def _distillation_commands(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    stage: Path,
    python: Path,
) -> list[tuple[str, list[str], Path]]:
    commands = []
    dataset = output_root / "02_distillation_dataset/train_validation.parquet"
    for variant in sorted(_values(config)["distillation"]["variants"]):
        for seed in _values(config)["seeds"]["training"]:
            task_id = f"{variant}_seed{int(seed)}"
            commands.append(
                (
                    task_id,
                    [
                        str(python),
                        str(Path(__file__).resolve()),
                        "--worker",
                        "distill",
                        "--config",
                        str(config["config_path"]),
                        "--preset",
                        str(config["preset"]),
                        "--project-root",
                        str(project_root),
                        "--dataset",
                        str(dataset),
                        "--variant",
                        str(variant),
                        "--seed",
                        str(int(seed)),
                        "--worker-output",
                        str(stage / task_id),
                    ],
                    stage / "logs" / f"{task_id}.log",
                )
            )
    return commands


def stage_train(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "02_distillation_dataset/gate.json")
    stage = output_root / "03_single_students"
    stage.mkdir(parents=True, exist_ok=True)
    probe = v14._gpu_probe(python)
    if not probe["gpu_available"]:
        return _gate(
            stage / "gate.json",
            {"gpu_visible": False},
            semantics="serial_single_gpu_single_student_distillation",
            gpu_probe=probe,
        )
    commands = _distillation_commands(
        config, project_root, output_root, stage, python
    )
    manifest = v12._run_subprocess_tasks(
        commands,
        requested_workers=int(
            _values(config)["parallel"]["gpu_training_workers"]
        ),
        manifest_path=stage / "training_parallel_manifest.json",
    )
    rows = []
    for task_id, _command, _log in commands:
        rows.append(
            {
                "task_id": task_id,
                **json.loads((stage / task_id / "report.json").read_text()),
            }
        )
    metrics = pd.json_normalize(rows)
    metrics.to_csv(stage / "training_metrics.csv", index=False)
    expected = (
        len(_values(config)["distillation"]["variants"])
        * len(_values(config)["seeds"]["training"])
    )
    return _gate(
        stage / "gate.json",
        {
            "gpu_visible": True,
            "all_registered_tasks_complete": len(metrics) == expected,
            "all_outputs_are_single_models": all(
                row["model_mode"] == "single_bounded_mlp" for row in rows
            ),
            "all_model_files_saved": all(
                (stage / task_id / "model.keras").is_file()
                for task_id, _command, _log in commands
            ),
            "sealed_holdout_still_unopened": not (
                output_root
                / "01_spatial_seal/sealed_teacher_reference.parquet"
            ).exists(),
        },
        semantics="serial_single_gpu_single_student_distillation",
        gpu_probe=probe,
        training_parallel_evidence=manifest,
        task_count=int(len(metrics)),
    )


def _model_roots(
    output_root: Path, variant: str, seeds: Sequence[int]
) -> dict[str, str]:
    return {
        str(int(seed)): str(
            output_root / f"03_single_students/{variant}_seed{int(seed)}"
        )
        for seed in seeds
    }


def _final8_reference(
    config: Mapping[str, Any], project_root: Path
) -> pd.DataFrame:
    sources = _source_files(config, project_root)
    teacher = pd.read_parquet(sources["final8_teacher"])
    catalog = pd.read_parquet(sources["final8_catalog"])[
        ["family_id", "major_semiaxis_m"]
    ]
    return teacher.merge(
        catalog,
        on="family_id",
        how="left",
        validate="many_to_one",
    )


def _apply_point_gate(
    metrics: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    output = metrics.copy()
    policy = _values(config)["selection"]
    output["seed_gate_pass"] = (
        output["fk_p95_mm"].le(float(policy["fk_p95_mm"]))
        & output["fk_max_mm"].le(float(policy["fk_max_mm"]))
        & output["minimum_joint_margin_deg"].ge(
            float(policy["minimum_joint_margin_deg"])
        )
        & output["actual_bounds"].astype(bool)
    )
    return output


def _apply_final8_gate(
    metrics: pd.DataFrame,
    reference: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    output = metrics.merge(
        reference.groupby("family_id", sort=True)["major_semiaxis_m"]
        .first()
        .rename("major_semiaxis_m"),
        on="family_id",
        how="left",
        validate="many_to_one",
    )
    policy = _values(config)["selection"]
    output["fk_p95_relative"] = output["fk_p95_mm"] / (
        output["major_semiaxis_m"] * 1000.0
    )
    output["fk_max_relative"] = output["fk_max_mm"] / (
        output["major_semiaxis_m"] * 1000.0
    )
    output["radius_gate_pass"] = (
        output["fk_p95_relative"].le(
            float(policy["final_ellipse_p95_relative"])
        )
        & output["fk_max_relative"].le(
            float(policy["final_ellipse_max_relative"])
        )
        & output["actual_bounds"].astype(bool)
    )
    family = v14._family_seed_gate(
        output,
        family_column="family_id",
        pass_column="radius_gate_pass",
        required_seed_passes=int(policy["required_seed_passes"]),
    )
    return output, family


def stage_select_and_lock(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "03_single_students/gate.json")
    stage = output_root / "04_development_selection"
    stage.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(
        output_root / "02_distillation_dataset/train_validation.parquet"
    )
    validation = dataset.loc[
        dataset["v12_15_split"].eq("validation")
    ].reset_index(drop=True)
    final8 = _final8_reference(config, project_root)
    environment = v14._environment(config, project_root)
    seeds = [int(value) for value in _values(config)["seeds"]["training"]]
    candidate_rows = []
    validation_parts = []
    final8_parts = []
    for variant in sorted(_values(config)["distillation"]["variants"]):
        roots = _model_roots(output_root, variant, seeds)
        validation_metrics, _ = v14._evaluate_model_set(
            validation, model_roots=roots, environment=environment
        )
        validation_metrics = _apply_point_gate(validation_metrics, config)
        validation_metrics.insert(0, "variant", variant)
        validation_parts.append(validation_metrics)
        final8_metrics, _ = v14._evaluate_model_set(
            final8,
            model_roots=roots,
            environment=environment,
            family_column="family_id",
        )
        final8_metrics, family = _apply_final8_gate(
            final8_metrics, final8, config
        )
        final8_metrics.insert(0, "variant", variant)
        final8_parts.append(final8_metrics)
        validation_passes = int(validation_metrics["seed_gate_pass"].sum())
        family_passes = int(family.sum())
        candidate_rows.append(
            {
                "variant": variant,
                "validation_seed_passes": validation_passes,
                "validation_seed_count": int(len(validation_metrics)),
                "final8_family_passes": family_passes,
                "final8_family_count": int(len(family)),
                "validation_worst_fk_p95_mm": float(
                    validation_metrics["fk_p95_mm"].max()
                ),
                "validation_worst_fk_max_mm": float(
                    validation_metrics["fk_max_mm"].max()
                ),
                "eligible": bool(
                    validation_passes
                    >= int(
                        _values(config)["selection"][
                            "required_seed_passes"
                        ]
                    )
                    and family.all()
                ),
            }
        )
    candidate = pd.DataFrame(candidate_rows).sort_values(
        [
            "eligible",
            "validation_seed_passes",
            "final8_family_passes",
            "validation_worst_fk_p95_mm",
            "validation_worst_fk_max_mm",
            "variant",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    )
    selected = candidate.iloc[0]
    candidate.to_csv(stage / "candidate_ranking.csv", index=False)
    pd.concat(validation_parts, ignore_index=True).to_csv(
        stage / "validation_block_metrics.csv", index=False
    )
    pd.concat(final8_parts, ignore_index=True).to_csv(
        stage / "final8_development_retention_metrics.csv", index=False
    )
    selection = {
        "schema_version": 1,
        "selected_variant": str(selected["variant"]),
        "selected_eligible": bool(selected["eligible"]),
        "smoke_numeric_gate_bypassed": config["preset"] == "smoke",
        "selection_uses_sealed_holdout": False,
        "selection_sets": [
            "nonsealed_validation_macro_blocks",
            "legacy_v12_13_final8_retention_regression",
        ],
        "ranking": candidate.to_dict(orient="records"),
    }
    atomic_write_json(stage / "selection.json", selection)
    smoke = config["preset"] == "smoke"
    gate = _gate(
        stage / "gate.json",
        {
            "registered_variants_evaluated": len(candidate)
            == len(_values(config)["distillation"]["variants"]),
            "formal_joint_eligibility_or_smoke_schema_only": bool(
                smoke or candidate["eligible"].any()
            ),
            "formal_selected_eligible_or_smoke_schema_only": bool(
                smoke or selected["eligible"]
            ),
            "selection_excludes_sealed_holdout": True,
            "sealed_holdout_still_unopened": not (
                output_root
                / "01_spatial_seal/sealed_teacher_reference.parquet"
            ).exists(),
        },
        semantics="joint_region_validation_and_legacy_retention_selection",
        selection=selection,
    )
    if not gate["gate_pass"]:
        return gate

    lock_stage = output_root / "05_model_lock"
    lock_stage.mkdir(parents=True, exist_ok=True)
    roots = _model_roots(output_root, str(selected["variant"]), seeds)
    locked_roots = {}
    model_hashes = {}
    for seed, source_root in roots.items():
        destination = lock_stage / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source_root) / "model.keras", destination / "model.keras")
        locked_roots[seed] = str(destination)
        model_hashes[seed] = sha256_file(destination / "model.keras")
    model_lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_mode": "single_bounded_mlp",
        "selected_variant": str(selected["variant"]),
        "model_roots": locked_roots,
        "model_sha256": model_hashes,
        "model_locked_before_sealed_holdout_open": True,
        "sealed_registry_sha256": sha256_file(
            output_root / "01_spatial_seal/sealed_block_registry.json"
        ),
        "selection_sha256": sha256_file(stage / "selection.json"),
        "implementation_commit": subprocess.run(
            ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
    }
    atomic_write_json(lock_stage / "final_model_lock.json", model_lock)
    return _gate(
        lock_stage / "gate.json",
        {
            "three_single_models_locked": len(model_hashes) == 3,
            "no_composite_descriptor_in_locked_roots": all(
                not (Path(root) / "composite_model.json").exists()
                for root in locked_roots.values()
            ),
            "locked_before_sealed_holdout_open": True,
        },
        semantics="single_student_model_lock",
        model_lock=model_lock,
    )


def _targets_inside_sealed_blocks(
    source: pd.DataFrame,
    *,
    sealed_keys: set[int],
    macro_voxel_mm: float,
    row_count: int,
    jitter_mm: float,
    seed: int,
) -> pd.DataFrame:
    source_keys = spatial_block_keys(
        source.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), macro_voxel_mm
    )
    pool = source.loc[np.isin(source_keys, list(sealed_keys))].copy()
    if pool.empty:
        raise RuntimeError("sealed blocks contain no source anchors")
    rng = np.random.default_rng(int(seed))
    rows = []
    attempts = 0
    maximum = max(int(row_count) * 20, 1000)
    while len(rows) < int(row_count) * 2 and attempts < maximum:
        attempts += 1
        index = int(rng.integers(0, len(pool)))
        center = pool.iloc[index].loc[list(XYZ_COLUMNS)].to_numpy(dtype=float)
        target = center + rng.uniform(
            -float(jitter_mm) / 1000.0,
            float(jitter_mm) / 1000.0,
            size=3,
        )
        key = int(spatial_block_keys(target[None, :], macro_voxel_mm)[0])
        if key not in sealed_keys:
            continue
        rows.append(
            {
                "target_id": len(rows),
                **{
                    column: float(target[position])
                    for position, column in enumerate(XYZ_COLUMNS)
                },
                "source_dense_node_id": str(pool.iloc[index]["node_id"]),
                "source_voxel_key": int(pool.iloc[index]["voxel_key"]),
                "spatial_block_key": key,
            }
        )
    if len(rows) < int(row_count):
        raise RuntimeError("unable to generate enough sealed-block targets")
    return pd.DataFrame(rows)


def _sealed_path_catalog(
    dense: pd.DataFrame,
    *,
    sealed_keys: set[int],
    macro_voxel_mm: float,
    phase_count: int,
    per_type: int,
    support_max_mm: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = spatial_block_keys(
        dense.loc[:, XYZ_COLUMNS].to_numpy(dtype=float), macro_voxel_mm
    )
    pool = dense.loc[np.isin(keys, list(sealed_keys))].copy().reset_index(
        drop=True
    )
    xyz = pool.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    if len(pool) < 64:
        raise RuntimeError("sealed block support is too small for path generation")
    tree = cKDTree(xyz)
    rng = np.random.default_rng(int(seed))
    kinds = ("ellipse", "circle", "lissajous", "closed_bspline", "open_bspline")
    catalog_rows = []
    target_parts = []
    global_id = 0
    for kind in kinds:
        for local_id in range(int(per_type)):
            accepted = None
            source_node = None
            support = math.inf
            closed = kind != "open_bspline"
            for _attempt in range(1200):
                source_index = int(rng.integers(0, len(pool)))
                center = xyz[source_index]
                _distance, neighbor = tree.query(center, k=min(64, len(xyz)))
                local = xyz[np.asarray(neighbor)] - center
                _u, _s, vh = np.linalg.svd(local, full_matrices=False)
                candidate, closed = v14._local_path(
                    kind,
                    center,
                    vh.T,
                    phase_count=int(phase_count),
                    rng=rng,
                )
                candidate_keys = spatial_block_keys(
                    candidate, macro_voxel_mm
                )
                if not np.isin(candidate_keys, list(sealed_keys)).all():
                    continue
                distance, _ = tree.query(candidate, k=1)
                support = float(np.max(distance) * 1000.0)
                if support <= float(support_max_mm):
                    accepted = candidate
                    source_node = str(pool.iloc[source_index]["node_id"])
                    break
            if accepted is None:
                raise RuntimeError(
                    f"unable to generate sealed-block {kind} path"
                )
            family_id = f"sealed_{kind}_{local_id:02d}"
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
                    "all_points_in_registered_sealed_blocks": True,
                }
            )
            global_id += len(part)
    return pd.DataFrame(catalog_rows), pd.concat(
        target_parts, ignore_index=True
    )


def stage_final(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "05_model_lock/gate.json")
    values = _values(config)
    source_root = _source_root(config, project_root)
    lock = json.loads(
        (output_root / "05_model_lock/final_model_lock.json").read_text()
    )
    registry_path = output_root / "01_spatial_seal/sealed_block_registry.json"
    registry = json.loads(registry_path.read_text())
    sealed_keys = set(map(int, registry["sealed_block_keys"]))
    macro_mm = float(registry["macro_voxel_mm"])
    dense = pd.read_parquet(source_root / "05_dense_region/dense_region_100k.parquet")
    sparse = pd.read_parquet(
        source_root / "03_sparse_region_growth/sparse_nodes.parquet"
    )
    anchors = pd.concat([sparse, dense], ignore_index=True, sort=False)
    environment = v14._environment(config, project_root)

    random_stage = output_root / "06_sealed_block_test"
    random_stage.mkdir(parents=True, exist_ok=True)
    required = int(values["spatial_holdout"]["random_rows"])
    targets = _targets_inside_sealed_blocks(
        dense,
        sealed_keys=sealed_keys,
        macro_voxel_mm=macro_mm,
        row_count=required,
        jitter_mm=float(values["spatial_holdout"]["random_jitter_mm"]),
        seed=int(values["seeds"]["sealed_random"]),
    )
    accepted, rejected, random_manifest = v14._label_targets(
        targets,
        anchors,
        name="sealed_random",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=random_stage,
        python=python,
    )
    reference = accepted.sort_values("target_id", kind="stable").iloc[
        :required
    ].copy()
    reference["holdout_role"] = "opened_after_model_lock"
    _atomic_parquet(reference, random_stage / "sealed_teacher_reference.parquet")
    _atomic_parquet(rejected, random_stage / "rejected_targets.parquet")
    random_metrics, random_details = v14._evaluate_model_set(
        reference,
        model_roots=lock["model_roots"],
        environment=environment,
    )
    random_metrics = _apply_point_gate(random_metrics, config)
    random_metrics.to_csv(random_stage / "metrics.csv", index=False)
    _atomic_parquet(random_details, random_stage / "tracking.parquet")

    path_stage = output_root / "07_sealed_trajectory_test"
    path_stage.mkdir(parents=True, exist_ok=True)
    catalog, path_targets = _sealed_path_catalog(
        dense,
        sealed_keys=sealed_keys,
        macro_voxel_mm=macro_mm,
        phase_count=int(values["trajectories"]["phase_count"]),
        per_type=int(values["trajectories"]["per_type"]),
        support_max_mm=float(values["trajectories"]["support_max_mm"]),
        seed=int(values["seeds"]["sealed_trajectory"]),
    )
    catalog.to_csv(path_stage / "trajectory_catalog.csv", index=False)
    path_reference, path_rejected, path_manifest = v14._label_targets(
        path_targets,
        anchors,
        name="sealed_paths",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=path_stage,
        python=python,
    )
    counts = path_reference.groupby("family_id", sort=True).size()
    complete_ids = counts[
        counts.eq(int(values["trajectories"]["phase_count"]))
    ].index
    path_reference = path_reference.loc[
        path_reference["family_id"].isin(complete_ids)
    ].copy()
    _atomic_parquet(
        path_reference, path_stage / "teacher_reference.parquet"
    )
    _atomic_parquet(path_rejected, path_stage / "rejected_targets.parquet")
    path_metrics, path_details = v14._evaluate_model_set(
        path_reference,
        model_roots=lock["model_roots"],
        environment=environment,
        family_column="family_id",
    )
    path_metrics = _apply_point_gate(path_metrics, config)
    path_metrics.to_csv(path_stage / "trajectory_metrics.csv", index=False)
    _atomic_parquet(path_details, path_stage / "student_tracking.parquet")
    path_family_pass = v14._family_seed_gate(
        path_metrics,
        family_column="family_id",
        pass_column="seed_gate_pass",
        required_seed_passes=int(values["selection"]["required_seed_passes"]),
    )
    v14._plot_tracking(
        path_reference, path_details, catalog, path_stage / "plots"
    )

    final8 = _final8_reference(config, project_root)
    final8_metrics, _ = v14._evaluate_model_set(
        final8,
        model_roots=lock["model_roots"],
        environment=environment,
        family_column="family_id",
    )
    final8_metrics, final8_family_pass = _apply_final8_gate(
        final8_metrics, final8, config
    )
    final8_metrics.to_csv(
        path_stage / "final8_retention_metrics.csv", index=False
    )

    summary_stage = output_root / "08_summary"
    summary_stage.mkdir(parents=True, exist_ok=True)
    expected_paths = 5 * int(values["trajectories"]["per_type"])
    if config["preset"] == "smoke":
        checks = {
            "locked_artifacts_are_single_models": lock["model_mode"]
            == "single_bounded_mlp",
            "sealed_registry_matches_locked_hash": sha256_file(registry_path)
            == lock["sealed_registry_sha256"],
            "sealed_random_teacher_rows_complete": len(reference) == required,
            "three_smoke_models_evaluated": len(random_metrics) == 3,
            "all_smoke_paths_teacher_complete": len(complete_ids)
            == expected_paths,
            "all_smoke_paths_evaluated": path_metrics[
                "family_id"
            ].nunique()
            == expected_paths,
            "current_final8_evaluation_complete": len(final8_metrics) == 24,
            "all_path_points_registered_sealed": bool(
                catalog["all_points_in_registered_sealed_blocks"].all()
            ),
            "model_was_locked_before_holdout_open": bool(
                lock["model_locked_before_sealed_holdout_open"]
            ),
        }
    else:
        checks = {
            "locked_artifacts_are_single_models": lock["model_mode"]
            == "single_bounded_mlp",
            "sealed_registry_matches_locked_hash": sha256_file(registry_path)
            == lock["sealed_registry_sha256"],
            "sealed_random_teacher_rows_complete": len(reference) == required,
            "sealed_random_seed_gate": int(
                random_metrics["seed_gate_pass"].sum()
            )
            >= int(values["selection"]["required_seed_passes"]),
            "sealed_paths_teacher_complete": len(complete_ids)
            == expected_paths,
            "sealed_path_gate": int(path_family_pass.sum())
            >= int(values["trajectories"]["required_pass_count"]),
            "current_final8_retained": bool(final8_family_pass.all()),
            "all_path_points_registered_sealed": bool(
                catalog["all_points_in_registered_sealed_blocks"].all()
            ),
            "model_was_locked_before_holdout_open": bool(
                lock["model_locked_before_sealed_holdout_open"]
            ),
        }
    recommendation = (
        "# V12.15 final recommendation\n\n"
        f"- sealed spatial-block random seed pass: "
        f"{int(random_metrics['seed_gate_pass'].sum())}/3\n"
        f"- sealed unseen path pass: "
        f"{int(path_family_pass.sum())}/{len(path_family_pass)}\n"
        f"- V12.13 final ellipse family-seed retention: "
        f"{int(final8_metrics['radius_gate_pass'].sum())}/"
        f"{len(final8_metrics)}\n"
        f"- final8 retained families: "
        f"{int(final8_family_pass.sum())}/{len(final8_family_pass)}\n"
        f"- overall exploratory gate: {all(checks.values())}\n\n"
        "The locked artifact is one bounded MLP per seed.  This supports "
        "single-chart spatial-block generalization in simulation; it does "
        "not support workspace-wide, multi-chart, or deployment claims.\n"
    )
    (summary_stage / "final_recommendation.md").write_text(recommendation)
    pd.DataFrame(
        [
            {
                "sealed_random_seed_passes": int(
                    random_metrics["seed_gate_pass"].sum()
                ),
                "sealed_path_passes": int(path_family_pass.sum()),
                "sealed_path_count": int(len(path_family_pass)),
                "final8_family_seed_passes": int(
                    final8_metrics["radius_gate_pass"].sum()
                ),
                "final8_family_seed_count": int(len(final8_metrics)),
                "final8_family_passes": int(final8_family_pass.sum()),
                "final8_family_count": int(len(final8_family_pass)),
            }
        ]
    ).to_csv(summary_stage / "generalization_summary.csv", index=False)
    return _gate(
        summary_stage / "gate.json",
        checks,
        semantics="v12_15_locked_single_student_spatial_block_generalization",
        selected_model_lock=lock,
        random_parallel_evidence=random_manifest,
        path_parallel_evidence=path_manifest,
        sealed_random_seed_passes=int(
            random_metrics["seed_gate_pass"].sum()
        ),
        sealed_path_passes=int(path_family_pass.sum()),
        sealed_path_count=int(len(path_family_pass)),
        final8_family_seed_passes=int(
            final8_metrics["radius_gate_pass"].sum()
        ),
        final8_family_seed_count=int(len(final8_metrics)),
        final8_family_passes=int(final8_family_pass.sum()),
        final8_family_count=int(len(final8_family_pass)),
    )


STAGES = (
    "protocol",
    "spatial_seal",
    "dataset",
    "train",
    "select_lock",
    "final",
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = _output_root(config, project_root, args.output_root)
    python = Path(args.python).resolve()
    requested = str(args.stage)
    selected = list(STAGES) if requested == "all" else [requested]
    gates = {}
    for stage in selected:
        if stage == "protocol":
            gate = stage_protocol(config, project_root, output_root)
        elif stage == "spatial_seal":
            gate = stage_spatial_seal(config, project_root, output_root)
        elif stage == "dataset":
            gate = stage_dataset(config, project_root, output_root)
        elif stage == "train":
            gate = stage_train(
                config, project_root, output_root, python=python
            )
        elif stage == "select_lock":
            gate = stage_select_and_lock(config, project_root, output_root)
        elif stage == "final":
            gate = stage_final(
                config, project_root, output_root, python=python
            )
        else:
            raise ValueError(f"unknown stage: {stage}")
        gates[stage] = bool(gate["gate_pass"])
        if not gate["gate_pass"]:
            break
    summary = {
        "protocol_id": PROTOCOL_ID,
        "claim_scope": CLAIM_SCOPE,
        "requested_stage": requested,
        "completed_stages": list(gates),
        "stage_gate_pass": gates,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            SOURCE_ROOT / "configs/bacra_v12_15_single_student.yaml"
        ),
    )
    parser.add_argument("--preset", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--output-root")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--worker", choices=("distill",))
    parser.add_argument("--dataset")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker == "distill":
        return _worker_distill(args)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(summary["stage_gate_pass"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
