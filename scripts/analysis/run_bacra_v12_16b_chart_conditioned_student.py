#!/usr/bin/env python3
"""Run BACRA V12.16B known-chart single-Student distillation."""

from __future__ import annotations

import argparse
import json
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

import run_bacra_v12 as v12
import run_bacra_v12_14_region_growth as v14
import run_bacra_v12_15_single_student as v15
import run_bacra_v12_7_canonical_student as canonical_runner
from quasi_exp.teacher.bacra_exploratory_expansion import point_margin_deg
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.gold_set_student import save_uncompiled_model
from quasi_exp.teacher.multichart_distillation import (
    build_chart_conditioned_model,
    predict_chart_conditioned,
    train_chart_conditioned_student,
)
from quasi_exp.teacher.retention_distillation import (
    DISTILL_BETA_COLUMNS,
    spatial_block_keys,
)
from run_trajectory_canonical_teacher_v10 import runtime_fingerprint


PROTOCOL_ID = (
    "branch-aware-canonical-region-atlas-v12.16b-"
    "known-chart-single-student"
)
CLAIM_SCOPE = "simulation_known_chart_multichart_spatial_generalization"
DEFAULT_PYTHON = Path(
    "/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python"
)
PREDICTED_BETA_COLUMNS = tuple(f"predicted_{name}" for name in BETA_COLUMNS)


def _source_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _values(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["v12_16b"]


def _output_root(
    config: Mapping[str, Any], project_root: Path, override: str | None
) -> Path:
    return (
        Path(override).resolve()
        if override
        else _source_path(project_root, str(config["output_root"]))
    )


def _source_roots(
    config: Mapping[str, Any], project_root: Path
) -> tuple[Path, Path]:
    values = _values(config)
    return (
        _source_path(project_root, values["source_v12_16a_root"]),
        _source_path(project_root, values["source_v12_15_root"]),
    )


def _source_files(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    chart_b, chart_a = _source_roots(config, project_root)
    values = _values(config)
    return {
        "v12_16a_region_gate": chart_b / "01_chart_b_region/gate.json",
        "v12_16a_spatial_gate": chart_b
        / "02_chart_b_spatial_seal/gate.json",
        "v12_16a_registry": chart_b
        / "02_chart_b_spatial_seal/sealed_block_registry.json",
        "v12_16a_train_validation": chart_b
        / "02_chart_b_spatial_seal/train_validation.parquet",
        "v12_16a_R2_dataset": chart_b
        / "01_chart_b_region/06_datasets/R2_dataset.parquet",
        "v12_15_gate": chart_a / "08_summary/gate.json",
        "v12_15_model_lock": chart_a
        / "05_model_lock/final_model_lock.json",
        "v12_15_training_dataset": chart_a
        / "02_distillation_dataset/train_validation.parquet",
        "v12_15_sealed_random": chart_a
        / "06_sealed_block_test/sealed_teacher_reference.parquet",
        "v12_15_sealed_paths": chart_a
        / "07_sealed_trajectory_test/teacher_reference.parquet",
        "final8_teacher": _source_path(
            project_root, values["source_final8_teacher"]
        ),
        "final8_catalog": _source_path(
            project_root, values["source_final8_catalog"]
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
        "gate_semantics": semantics,
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


def _model_roots(output_root: Path, seeds: Sequence[int]) -> dict[str, str]:
    return {
        str(int(seed)): str(output_root / f"02_students/seed_{int(seed)}")
        for seed in seeds
    }


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
    chart_b_gate = (
        json.loads(sources["v12_16a_spatial_gate"].read_text())
        if sources["v12_16a_spatial_gate"].is_file()
        else {}
    )
    chart_a_gate = (
        json.loads(sources["v12_15_gate"].read_text())
        if sources["v12_15_gate"].is_file()
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
    atomic_write_json(
        stage / "frozen_config.json",
        {key: value for key, value in config.items() if key != "config_path"},
    )
    implementation = {
        "runner": Path(__file__).resolve(),
        "distillation": SOURCE_ROOT
        / "src/quasi_exp/teacher/multichart_distillation.py",
        "config": Path(str(config["config_path"])).resolve(),
    }
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
            "chart_a_source_gate_passed": bool(chart_a_gate.get("gate_pass")),
            "chart_b_source_gate_passed": bool(chart_b_gate.get("gate_pass")),
            "source_worktree_clean": status.strip() == "",
            "known_chart_claim_scope_frozen": config["claim_scope"]
            == CLAIM_SCOPE,
            "deployment_claim_disabled": config[
                "deployment_claim_gate_pass"
            ]
            is False,
        },
        semantics="clean_fixed_point_and_two_chart_source_closure",
        source_sha256=actual,
        git_sha=git_sha,
        chart_identity_input="known_registered_chart_id",
        automatic_chart_classifier_in_scope=False,
    )


def stage_dataset(
    config: Mapping[str, Any], project_root: Path, output_root: Path
) -> dict[str, Any]:
    _require_gate(output_root, "00_protocol/gate.json")
    stage = output_root / "01_multichart_dataset"
    stage.mkdir(parents=True, exist_ok=True)
    sources = _source_files(config, project_root)
    chart_a = pd.read_parquet(sources["v12_15_training_dataset"]).copy()
    chart_b = pd.read_parquet(sources["v12_16a_train_validation"]).copy()
    chart_a["chart_id"] = "chart_A"
    chart_a["chart_feature"] = 0.0
    chart_b["chart_id"] = "chart_B"
    chart_b["chart_feature"] = 1.0
    columns = sorted(set(chart_a.columns) | set(chart_b.columns))
    dataset = pd.concat(
        [
            chart_a.reindex(columns=columns),
            chart_b.reindex(columns=columns),
        ],
        ignore_index=True,
    )
    path = stage / "train_validation.parquet"
    _atomic_parquet(dataset, path)
    counts = (
        dataset.groupby(["chart_id", "v12_15_split"], sort=True)
        .size()
        .to_dict()
    )
    registry = json.loads(sources["v12_16a_registry"].read_text())
    manifest = {
        "row_count": int(len(dataset)),
        "chart_split_rows": {
            f"{chart}/{split}": int(value)
            for (chart, split), value in counts.items()
        },
        "sha256": sha256_file(path),
        "chart_b_sealed_registry_sha256": sha256_file(
            sources["v12_16a_registry"]
        ),
        "chart_b_sealed_block_count": int(registry["sealed_block_count"]),
        "chart_b_sealed_labels_opened": False,
    }
    atomic_write_json(stage / "dataset_manifest.json", manifest)
    expected_groups = {
        ("chart_A", "train"),
        ("chart_A", "validation"),
        ("chart_B", "train"),
        ("chart_B", "validation"),
    }
    return _gate(
        stage / "gate.json",
        {
            "both_charts_and_splits_present": set(counts) == expected_groups,
            "chart_feature_mapping_exact": bool(
                dataset.loc[dataset["chart_id"].eq("chart_A"), "chart_feature"]
                .eq(0.0)
                .all()
                and dataset.loc[
                    dataset["chart_id"].eq("chart_B"), "chart_feature"
                ]
                .eq(1.0)
                .all()
            ),
            "only_train_validation_rows_written": bool(
                dataset["v12_15_split"]
                .isin(["train", "validation"])
                .all()
            ),
            "chart_b_sealed_labels_still_unopened": True,
        },
        semantics="known_chart_training_union_without_chart_b_sealed_blocks",
        dataset_manifest=manifest,
    )


def _worker_train(args: argparse.Namespace) -> int:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("V12.16B training requires a visible GPU")
    values = _values(config)
    seed = int(args.seed)
    dataset = pd.read_parquet(args.dataset)
    source_lock_path = _source_files(config, project_root)[
        "v12_15_model_lock"
    ]
    source_lock = json.loads(source_lock_path.read_text())
    source_path = (
        Path(source_lock["model_roots"][str(seed)]) / "model.keras"
    )
    source_model = tf.keras.models.load_model(source_path, compile=False)
    chart_a = dataset["chart_id"].eq("chart_A").to_numpy()
    distill = dataset.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    distill[chart_a] = np.asarray(
        source_model.predict(
            dataset.loc[chart_a, XYZ_COLUMNS].to_numpy(dtype=np.float32),
            batch_size=2048,
            verbose=0,
        ),
        dtype=float,
    )
    for index, column in enumerate(DISTILL_BETA_COLUMNS):
        dataset[column] = distill[:, index]
    geometry = canonical_runner._geometry(config, project_root)
    training = values["training"]
    margin = point_margin_deg(
        dataset.loc[:, BETA_COLUMNS].to_numpy(dtype=float),
        geometry.beta_bounds_rad,
    )
    margin_tail = (
        dataset["v12_15_split"].eq("train").to_numpy()
        & (margin < float(training["margin_tail_threshold_deg"]))
    )
    dataset.loc[margin_tail, "sample_weight"] *= float(
        training["margin_tail_sample_weight_multiplier"]
    )
    train = dataset.loc[dataset["v12_15_split"].eq("train")].copy()
    validation = dataset.loc[
        dataset["v12_15_split"].eq("validation")
    ].copy()
    model = build_chart_conditioned_model(source_model)
    probe_xyz = validation.loc[
        validation["chart_id"].eq("chart_A"), XYZ_COLUMNS
    ].to_numpy(dtype=np.float32)[:2048]
    initial_source = np.asarray(
        source_model.predict(probe_xyz, batch_size=2048, verbose=0),
        dtype=float,
    )
    initial_chart = predict_chart_conditioned(model, probe_xyz, 0.0)
    initialization_max_abs_deg = float(
        np.max(np.abs(np.rad2deg(initial_source - initial_chart)))
    )
    model, history, report = train_chart_conditioned_student(
        model,
        train,
        validation,
        geometry=geometry,
        seed=seed,
        learning_rate=float(training["learning_rate"]),
        batch_size=int(training["batch_size"]),
        max_steps=int(training["max_optimizer_steps"]),
        validation_interval=int(training["validation_interval_steps"]),
        patience_intervals=int(training["patience_intervals"]),
        chart_sampling_weights=dict(training["chart_sampling_weights"]),
        teacher_beta_weight=float(training["teacher_beta_weight"]),
        distill_beta_weight=float(training["distill_beta_weight"]),
        fk_weight=float(training["fk_weight"]),
        margin_weight=float(training["margin_weight"]),
        row_loss_weight=float(training["row_loss_weight"]),
    )
    report.update(
        {
            "source_model": str(source_path),
            "source_model_sha256": sha256_file(source_path),
            "source_model_lock_sha256": sha256_file(source_lock_path),
            "device": devices[0].name,
            "initial_chart_a_equivalence_max_abs_deg": (
                initialization_max_abs_deg
            ),
            "margin_tail_training_rows": int(np.count_nonzero(margin_tail)),
        }
    )
    destination = Path(args.worker_output)
    destination.mkdir(parents=True, exist_ok=True)
    save_uncompiled_model(model, destination / "model.keras")
    history.to_csv(destination / "history.csv", index=False)
    report["model_sha256"] = sha256_file(destination / "model.keras")
    atomic_write_json(destination / "report.json", report)
    return 0


def stage_train(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "01_multichart_dataset/gate.json")
    stage = output_root / "02_students"
    stage.mkdir(parents=True, exist_ok=True)
    probe = v14._gpu_probe(python)
    if not probe["gpu_available"]:
        return _gate(
            stage / "gate.json",
            {"gpu_visible": False},
            semantics="serial_single_gpu_known_chart_distillation",
            gpu_probe=probe,
        )
    commands = []
    dataset = output_root / "01_multichart_dataset/train_validation.parquet"
    for seed in _values(config)["seeds"]["training"]:
        task_id = f"seed_{int(seed)}"
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
                    "--dataset",
                    str(dataset),
                    "--seed",
                    str(int(seed)),
                    "--worker-output",
                    str(stage / task_id),
                ],
                stage / "logs" / f"{task_id}.log",
            )
        )
    manifest = v12._run_subprocess_tasks(
        commands,
        requested_workers=int(
            _values(config)["parallel"]["gpu_training_workers"]
        ),
        manifest_path=stage / "training_parallel_manifest.json",
    )
    reports = [
        json.loads((stage / task_id / "report.json").read_text())
        for task_id, _command, _log in commands
    ]
    pd.json_normalize(reports).to_csv(
        stage / "training_metrics.csv", index=False
    )
    return _gate(
        stage / "gate.json",
        {
            "gpu_visible": True,
            "three_seed_tasks_complete": len(reports) == 3,
            "all_models_are_known_chart_single_mlp": all(
                row["model_mode"] == "known_chart_single_bounded_mlp"
                for row in reports
            ),
            "chart_a_initialization_is_exact": all(
                row["initial_chart_a_equivalence_max_abs_deg"] <= 1.0e-5
                for row in reports
            ),
            "chart_b_sealed_labels_still_unopened": True,
        },
        semantics="serial_single_gpu_known_chart_distillation",
        gpu_probe=probe,
        training_parallel_evidence=manifest,
    )


def _load_models(model_roots: Mapping[str, str]) -> dict[int, Any]:
    import tensorflow as tf

    return {
        int(seed): tf.keras.models.load_model(
            Path(root) / "model.keras", compile=False
        )
        for seed, root in model_roots.items()
    }


def _evaluate_model_set(
    reference: pd.DataFrame,
    *,
    model_roots: Mapping[str, str],
    environment: Any,
    chart_feature: float,
    family_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = reference.reset_index(drop=True)
    target = reference.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    groups = (
        [(None, reference.index.to_numpy())]
        if family_column is None
        else [
            (family, group.index.to_numpy())
            for family, group in reference.groupby(family_column, sort=True)
        ]
    )
    metric_rows = []
    detail_parts = []
    for seed, model in _load_models(model_roots).items():
        beta = predict_chart_conditioned(model, target, chart_feature)
        achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
        error = np.linalg.norm(achieved - target, axis=1) * 1000.0
        margin = point_margin_deg(beta, environment.bounds)
        detail = reference[
            [
                column
                for column in (
                    "target_id",
                    "family_id",
                    "phase_idx",
                    "spatial_block_key",
                    *XYZ_COLUMNS,
                )
                if column in reference.columns
            ]
        ].copy()
        detail["seed"] = seed
        detail["chart_feature"] = float(chart_feature)
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
                    "seed": seed,
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


def _final8_reference(
    config: Mapping[str, Any], project_root: Path
) -> pd.DataFrame:
    sources = _source_files(config, project_root)
    teacher = pd.read_parquet(sources["final8_teacher"])
    catalog = pd.read_parquet(sources["final8_catalog"])[
        ["family_id", "major_semiaxis_m"]
    ]
    return teacher.merge(
        catalog, on="family_id", how="left", validate="many_to_one"
    )


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
    _require_gate(output_root, "02_students/gate.json")
    stage = output_root / "03_development_selection"
    stage.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(
        output_root / "01_multichart_dataset/train_validation.parquet"
    )
    validation = dataset.loc[
        dataset["v12_15_split"].eq("validation")
    ].copy()
    environment = v14._environment(config, project_root)
    seeds = [int(value) for value in _values(config)["seeds"]["training"]]
    roots = _model_roots(output_root, seeds)
    parts = []
    passes = {}
    for chart_id, feature in (("chart_A", 0.0), ("chart_B", 1.0)):
        reference = validation.loc[validation["chart_id"].eq(chart_id)]
        metrics, _details = _evaluate_model_set(
            reference,
            model_roots=roots,
            environment=environment,
            chart_feature=feature,
        )
        metrics = _apply_point_gate(metrics, config)
        metrics.insert(0, "chart_id", chart_id)
        parts.append(metrics)
        passes[chart_id] = int(metrics["seed_gate_pass"].sum())
    metrics = pd.concat(parts, ignore_index=True)
    metrics.to_csv(stage / "validation_metrics.csv", index=False)
    required = int(_values(config)["selection"]["required_seed_passes"])
    smoke = config["preset"] == "smoke"
    selection = {
        "model_mode": "known_chart_single_bounded_mlp",
        "known_chart_id_required": True,
        "automatic_chart_classifier": False,
        "selection_uses_chart_b_sealed_blocks": False,
        "validation_seed_passes": passes,
        "smoke_numeric_gate_bypassed": smoke,
    }
    atomic_write_json(stage / "selection.json", selection)
    gate = _gate(
        stage / "gate.json",
        {
            "both_chart_validation_sets_evaluated": len(parts) == 2,
            "chart_a_gate_or_smoke": smoke or passes["chart_A"] >= required,
            "chart_b_gate_or_smoke": smoke or passes["chart_B"] >= required,
            "selection_excludes_chart_b_sealed_blocks": True,
        },
        semantics="known_chart_two_region_validation_before_lock",
        selection=selection,
    )
    if not gate["gate_pass"]:
        return gate
    lock_stage = output_root / "04_model_lock"
    lock_stage.mkdir(parents=True, exist_ok=True)
    locked_roots = {}
    hashes = {}
    for seed, root in roots.items():
        destination = lock_stage / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(root) / "model.keras", destination / "model.keras")
        locked_roots[seed] = str(destination)
        hashes[seed] = sha256_file(destination / "model.keras")
    registry_path = _source_files(config, project_root)["v12_16a_registry"]
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_mode": "known_chart_single_bounded_mlp",
        "known_chart_id_required": True,
        "automatic_chart_classifier": False,
        "model_roots": locked_roots,
        "model_sha256": hashes,
        "model_locked_before_chart_b_sealed_open": True,
        "chart_b_sealed_registry_sha256": sha256_file(registry_path),
        "selection_sha256": sha256_file(stage / "selection.json"),
        "implementation_commit": subprocess.run(
            ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
    }
    atomic_write_json(lock_stage / "final_model_lock.json", lock)
    return _gate(
        lock_stage / "gate.json",
        {
            "three_single_models_locked": len(hashes) == 3,
            "known_chart_identity_is_explicit": True,
            "automatic_classifier_not_claimed": True,
            "locked_before_chart_b_sealed_open": True,
        },
        semantics="known_chart_single_student_model_lock",
        model_lock=lock,
    )


def _chart_b_sealed_reference(
    config: Mapping[str, Any], project_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = _source_files(config, project_root)
    registry = json.loads(sources["v12_16a_registry"].read_text())
    sealed = set(map(int, registry["sealed_block_keys"]))
    frame = pd.read_parquet(sources["v12_16a_R2_dataset"]).copy()
    keys = spatial_block_keys(
        frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float),
        float(registry["macro_voxel_mm"]),
    )
    reference = frame.loc[np.isin(keys, list(sealed))].copy()
    reference["spatial_block_key"] = keys[np.isin(keys, list(sealed))]
    reference["chart_id"] = "chart_B"
    reference["chart_feature"] = 1.0
    return reference, registry


def stage_final(
    config: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    *,
    python: Path,
) -> dict[str, Any]:
    _require_gate(output_root, "04_model_lock/gate.json")
    values = _values(config)
    sources = _source_files(config, project_root)
    lock = json.loads(
        (output_root / "04_model_lock/final_model_lock.json").read_text()
    )
    roots = lock["model_roots"]
    environment = v14._environment(config, project_root)

    stage = output_root / "05_locked_evaluation"
    stage.mkdir(parents=True, exist_ok=True)
    chart_b_reference, registry = _chart_b_sealed_reference(
        config, project_root
    )
    _atomic_parquet(
        chart_b_reference, stage / "chart_b_sealed_teacher_reference.parquet"
    )
    chart_b_metrics, chart_b_details = _evaluate_model_set(
        chart_b_reference,
        model_roots=roots,
        environment=environment,
        chart_feature=1.0,
    )
    chart_b_metrics = _apply_point_gate(chart_b_metrics, config)
    chart_b_metrics.to_csv(stage / "chart_b_sealed_metrics.csv", index=False)
    _atomic_parquet(
        chart_b_details, stage / "chart_b_sealed_tracking.parquet"
    )

    chart_a_random = pd.read_parquet(sources["v12_15_sealed_random"])
    chart_a_random_metrics, chart_a_random_details = _evaluate_model_set(
        chart_a_random,
        model_roots=roots,
        environment=environment,
        chart_feature=0.0,
    )
    chart_a_random_metrics = _apply_point_gate(
        chart_a_random_metrics, config
    )
    chart_a_random_metrics.to_csv(
        stage / "chart_a_sealed_random_metrics.csv", index=False
    )
    _atomic_parquet(
        chart_a_random_details,
        stage / "chart_a_sealed_random_tracking.parquet",
    )

    chart_a_paths = pd.read_parquet(sources["v12_15_sealed_paths"])
    chart_a_path_metrics, chart_a_path_details = _evaluate_model_set(
        chart_a_paths,
        model_roots=roots,
        environment=environment,
        chart_feature=0.0,
        family_column="family_id",
    )
    chart_a_path_metrics = _apply_point_gate(
        chart_a_path_metrics, config
    )
    chart_a_path_metrics.to_csv(
        stage / "chart_a_sealed_path_metrics.csv", index=False
    )
    _atomic_parquet(
        chart_a_path_details,
        stage / "chart_a_sealed_path_tracking.parquet",
    )
    chart_a_path_family = v14._family_seed_gate(
        chart_a_path_metrics,
        family_column="family_id",
        pass_column="seed_gate_pass",
        required_seed_passes=int(values["selection"]["required_seed_passes"]),
    )

    chart_b_root, _chart_a_root = _source_roots(config, project_root)
    dense = pd.read_parquet(
        chart_b_root
        / "01_chart_b_region/05_dense_region/dense_region_100k.parquet"
    )
    sparse = pd.read_parquet(
        chart_b_root
        / "01_chart_b_region/03_sparse_region_growth/sparse_nodes.parquet"
    )
    sealed_keys = set(map(int, registry["sealed_block_keys"]))
    catalog, targets = v15._sealed_path_catalog(
        dense,
        sealed_keys=sealed_keys,
        macro_voxel_mm=float(registry["macro_voxel_mm"]),
        phase_count=int(values["trajectories"]["phase_count"]),
        per_type=int(values["trajectories"]["per_type"]),
        support_max_mm=float(values["trajectories"]["support_max_mm"]),
        seed=int(values["seeds"]["chart_b_trajectory"]),
    )
    catalog.to_csv(stage / "chart_b_path_catalog.csv", index=False)
    anchors = pd.concat([sparse, dense], ignore_index=True, sort=False)
    chart_b_paths, rejected, path_parallel = v14._label_targets(
        targets,
        anchors,
        name="chart_b_sealed_paths",
        config=config,
        project_root=project_root,
        output_root=output_root,
        stage=stage,
        python=python,
    )
    counts = chart_b_paths.groupby("family_id", sort=True).size()
    complete_ids = counts[
        counts.eq(int(values["trajectories"]["phase_count"]))
    ].index
    chart_b_paths = chart_b_paths.loc[
        chart_b_paths["family_id"].isin(complete_ids)
    ].copy()
    _atomic_parquet(
        chart_b_paths, stage / "chart_b_path_teacher_reference.parquet"
    )
    _atomic_parquet(rejected, stage / "chart_b_path_rejected.parquet")
    chart_b_path_metrics, chart_b_path_details = _evaluate_model_set(
        chart_b_paths,
        model_roots=roots,
        environment=environment,
        chart_feature=1.0,
        family_column="family_id",
    )
    chart_b_path_metrics = _apply_point_gate(
        chart_b_path_metrics, config
    )
    chart_b_path_metrics.to_csv(
        stage / "chart_b_path_metrics.csv", index=False
    )
    _atomic_parquet(
        chart_b_path_details, stage / "chart_b_path_tracking.parquet"
    )
    chart_b_path_family = v14._family_seed_gate(
        chart_b_path_metrics,
        family_column="family_id",
        pass_column="seed_gate_pass",
        required_seed_passes=int(values["selection"]["required_seed_passes"]),
    )
    v14._plot_tracking(
        chart_b_paths,
        chart_b_path_details,
        catalog,
        stage / "chart_b_path_plots",
    )

    final8 = _final8_reference(config, project_root)
    final8_metrics, _ = _evaluate_model_set(
        final8,
        model_roots=roots,
        environment=environment,
        chart_feature=0.0,
        family_column="family_id",
    )
    final8_metrics, final8_family = _apply_final8_gate(
        final8_metrics, final8, config
    )
    final8_metrics.to_csv(stage / "final8_retention_metrics.csv", index=False)

    expected_paths = 5 * int(values["trajectories"]["per_type"])
    required = int(values["selection"]["required_seed_passes"])
    smoke = config["preset"] == "smoke"
    if smoke:
        checks = {
            "lock_is_known_chart_single_model": lock["model_mode"]
            == "known_chart_single_bounded_mlp",
            "chart_b_registry_matches_lock": sha256_file(
                sources["v12_16a_registry"]
            )
            == lock["chart_b_sealed_registry_sha256"],
            "chart_b_sealed_points_opened_after_lock": len(chart_b_reference) > 0,
            "all_smoke_chart_b_paths_teacher_complete": len(complete_ids)
            == expected_paths,
            "all_reference_sets_evaluated": bool(
                len(chart_b_metrics) == 3
                and len(chart_a_random_metrics) == 3
                and len(final8_metrics) == 24
            ),
            "known_chart_identity_required": bool(
                lock["known_chart_id_required"]
            ),
            "automatic_classifier_not_claimed": not bool(
                lock["automatic_chart_classifier"]
            ),
        }
    else:
        checks = {
            "lock_is_known_chart_single_model": lock["model_mode"]
            == "known_chart_single_bounded_mlp",
            "chart_b_registry_matches_lock": sha256_file(
                sources["v12_16a_registry"]
            )
            == lock["chart_b_sealed_registry_sha256"],
            "chart_b_sealed_point_gate": int(
                chart_b_metrics["seed_gate_pass"].sum()
            )
            >= required,
            "chart_b_paths_teacher_complete": len(complete_ids)
            == expected_paths,
            "chart_b_path_gate": int(chart_b_path_family.sum())
            >= int(values["trajectories"]["required_pass_count"]),
            "chart_a_sealed_random_retained": int(
                chart_a_random_metrics["seed_gate_pass"].sum()
            )
            >= required,
            "chart_a_sealed_paths_retained": bool(chart_a_path_family.all()),
            "chart_a_final8_retained": bool(final8_family.all()),
            "model_was_locked_before_chart_b_sealed_open": bool(
                lock["model_locked_before_chart_b_sealed_open"]
            ),
            "known_chart_identity_required": bool(
                lock["known_chart_id_required"]
            ),
            "automatic_classifier_not_claimed": not bool(
                lock["automatic_chart_classifier"]
            ),
        }
    summary = {
        "chart_b_sealed_seed_passes": int(
            chart_b_metrics["seed_gate_pass"].sum()
        ),
        "chart_b_path_passes": int(chart_b_path_family.sum()),
        "chart_b_path_count": int(len(chart_b_path_family)),
        "chart_a_random_seed_passes": int(
            chart_a_random_metrics["seed_gate_pass"].sum()
        ),
        "chart_a_path_passes": int(chart_a_path_family.sum()),
        "chart_a_path_count": int(len(chart_a_path_family)),
        "final8_family_passes": int(final8_family.sum()),
        "final8_family_count": int(len(final8_family)),
    }
    pd.DataFrame([summary]).to_csv(
        stage / "generalization_summary.csv", index=False
    )
    summary_stage = output_root / "06_summary"
    summary_stage.mkdir(parents=True, exist_ok=True)
    return _gate(
        summary_stage / "gate.json",
        checks,
        semantics="locked_known_chart_single_student_multichart_generalization",
        summary=summary,
        chart_b_path_parallel_evidence=path_parallel,
        model_lock=lock,
    )


STAGES = ("protocol", "dataset", "train", "select_lock", "final")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = v12.load_protocol_config(args.config, args.preset)
    project_root = Path(args.project_root).resolve()
    output_root = _output_root(config, project_root, args.output_root)
    python = Path(args.python).resolve()
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    gates = {}
    for stage in selected:
        if stage == "protocol":
            gate = stage_protocol(config, project_root, output_root)
        elif stage == "dataset":
            gate = stage_dataset(config, project_root, output_root)
        elif stage == "train":
            gate = stage_train(
                config, project_root, output_root, python=python
            )
        elif stage == "select_lock":
            gate = stage_select_and_lock(
                config, project_root, output_root
            )
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
        "requested_stage": args.stage,
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
            SOURCE_ROOT
            / "configs/bacra_v12_16b_chart_conditioned_student.yaml"
        ),
    )
    parser.add_argument(
        "--preset", choices=("formal", "smoke"), default="formal"
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--output-root")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--worker", choices=("train",))
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker == "train":
        return _worker_train(args)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(summary["stage_gate_pass"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
