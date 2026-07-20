#!/usr/bin/env python3
"""Screen and finalise S0/S1/S3 students on the frozen large-scale splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.student import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.student_tracking_tf import (
    StudentGeometry,
    autoregressive_rollout,
    build_gru_model,
    build_static_model,
    compile_student,
    make_cyclic_windows,
    packed_targets,
)

from run_trajectory_canonical_teacher_v10 import load_environment, project_root_from, runtime_fingerprint


SCREEN_SEEDS = (20260720, 20260721, 20260722)
FINAL_SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    student: str
    output_mode: str
    lambda_fk: float

    @property
    def is_autoregressive(self) -> bool:
        return self.student == "S3"


CANDIDATES = (
    Candidate("S0_identity", "S0", "identity", 0.0),
    Candidate("S0_tanh", "S0", "tanh", 0.0),
    Candidate("S1_l0p1_identity", "S1", "identity", 0.1),
    Candidate("S1_l0p1_tanh", "S1", "tanh", 0.1),
    Candidate("S1_l1_identity", "S1", "identity", 1.0),
    Candidate("S1_l1_tanh", "S1", "tanh", 1.0),
    Candidate("S3_gru_l0p1_tanh", "S3", "tanh", 0.1),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_split(output_root: Path, scale_slug: str, split: str) -> pd.DataFrame:
    filename = {"train": "train.parquet", "validation": "validation.parquet", "test": "sealed_test.parquet"}[split]
    path = output_root / scale_slug / "teacher_dense/splits" / filename
    frame = pd.read_parquet(path).sort_values("phase_idx", kind="stable").reset_index(drop=True)
    if split != "test" and frame["split"].astype(str).eq("test").any():
        raise RuntimeError("sealed test rows leaked into a pre-selection split")
    return frame


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["label_eligible"].astype(bool)].reset_index(drop=True)


def _geometry(environment: Any) -> StudentGeometry:
    return StudentGeometry(
        lengths_m=environment.lengths_m,
        p_end_local_m=environment.p_end_local_m,
        theta_sign=environment.theta_sign,
        beta_bounds_rad=environment.bounds,
    )


def _fit_static(
    tf: Any,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    candidate: Candidate,
    geometry: StudentGeometry,
    *,
    seed: int,
    epochs: int,
) -> tuple[Any, dict[str, Any]]:
    tf.keras.utils.set_random_seed(int(seed))
    x_train = train.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    x_val = validation.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    model = build_static_model(x_train, geometry=geometry, output_mode=candidate.output_mode)
    compile_student(model, geometry=geometry, lambda_fk=candidate.lambda_fk, learning_rate=1.0e-3)
    started = time.perf_counter()
    history = model.fit(
        x_train,
        packed_targets(train),
        validation_data=(x_val, packed_targets(validation)),
        epochs=int(epochs),
        batch_size=min(128, len(train)),
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=30, min_delta=1.0e-6, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    return model, {
        "epochs_ran": int(len(history.history["loss"])),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "wall_time_s": float(time.perf_counter() - started),
    }


def _fit_gru(
    tf: Any,
    train_frames: Iterable[pd.DataFrame],
    validation: pd.DataFrame,
    candidate: Candidate,
    geometry: StudentGeometry,
    *,
    seed: int,
    epochs: int,
    window_size: int,
) -> tuple[Any, Any, dict[str, Any]]:
    train_list = list(train_frames)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for frame in train_list:
        x, y = make_cyclic_windows(frame, window_size=window_size, include_reverse=True)
        features.append(x)
        targets.append(y)
    train_x = np.concatenate(features, axis=0)
    train_y = np.concatenate(targets, axis=0)
    val_x, val_y = make_cyclic_windows(validation, window_size=window_size, include_reverse=False)
    tf.keras.utils.set_random_seed(int(seed))
    model = build_gru_model(train_x, geometry=geometry, output_mode=candidate.output_mode)
    compile_student(model, geometry=geometry, lambda_fk=candidate.lambda_fk, learning_rate=5.0e-4)
    started = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=int(epochs),
        batch_size=64,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, min_delta=1.0e-6, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    # Autonomous rollout is bootstrapped by a separately trained S1 static
    # inverse.  No teacher beta is read after this initial prediction.
    init_candidate = Candidate("S1_rollout_init", "S1", "tanh", 1.0)
    init_train = pd.concat(train_list, ignore_index=True)
    init_model, init_history = _fit_static(
        tf,
        init_train,
        validation,
        init_candidate,
        geometry,
        seed=seed + 1000,
        epochs=max(100, epochs),
    )
    return model, init_model, {
        "epochs_ran": int(len(history.history["loss"])),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "wall_time_s": float(time.perf_counter() - started),
        "window_size": int(window_size),
        "training_windows": int(len(train_x)),
        "rollout_initialiser": init_history,
    }


def _fit_fixed_epochs(
    tf: Any,
    train_frames: list[pd.DataFrame],
    candidate: Candidate,
    geometry: StudentGeometry,
    *,
    seed: int,
    epochs: int,
    window_size: int,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Fit without consulting a validation set (used by the 0.75 m control)."""

    tf.keras.utils.set_random_seed(int(seed))
    started = time.perf_counter()
    init_model = None
    if candidate.is_autoregressive:
        windows = [
            make_cyclic_windows(frame, window_size=window_size, include_reverse=True)
            for frame in train_frames
        ]
        x = np.concatenate([value[0] for value in windows], axis=0)
        y = np.concatenate([value[1] for value in windows], axis=0)
        model = build_gru_model(x, geometry=geometry, output_mode=candidate.output_mode)
        compile_student(model, geometry=geometry, lambda_fk=candidate.lambda_fk, learning_rate=5.0e-4)
        history = model.fit(x, y, epochs=epochs, batch_size=64, shuffle=True, verbose=0)
        init_candidate = Candidate("S1_rollout_init", "S1", "tanh", 1.0)
        full = pd.concat(train_frames, ignore_index=True)
        init_model = build_static_model(
            full.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
            geometry=geometry,
            output_mode=init_candidate.output_mode,
        )
        compile_student(init_model, geometry=geometry, lambda_fk=1.0, learning_rate=1.0e-3)
        init_model.fit(
            full.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32),
            packed_targets(full),
            epochs=max(100, epochs),
            batch_size=min(128, len(full)),
            shuffle=True,
            verbose=0,
        )
        rows = len(x)
    else:
        full = pd.concat(train_frames, ignore_index=True)
        xyz = full.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
        model = build_static_model(xyz, geometry=geometry, output_mode=candidate.output_mode)
        compile_student(model, geometry=geometry, lambda_fk=candidate.lambda_fk, learning_rate=1.0e-3)
        history = model.fit(
            xyz,
            packed_targets(full),
            epochs=epochs,
            batch_size=min(128, len(full)),
            shuffle=True,
            verbose=0,
        )
        rows = len(full)
    return model, init_model, {
        "epochs_ran": int(epochs),
        "best_epoch": int(epochs),
        "best_val_loss": None,
        "final_train_loss": float(history.history["loss"][-1]),
        "wall_time_s": float(time.perf_counter() - started),
        "rows_or_windows": int(rows),
        "epoch_budget_source": "0.5m_screen_median",
    }


def _predict(
    model: Any,
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    init_model: Any | None,
    window_size: int,
) -> np.ndarray:
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
    if not candidate.is_autoregressive:
        return np.asarray(model.predict(xyz, verbose=0), dtype=float)
    if init_model is None:
        raise ValueError("S3 rollout requires a static initialiser")
    # ``initial_beta_rad`` is beta_(t-1); for a cyclic traversal its state is
    # predicted at the traversal's final target, not at the first target.
    initial = np.asarray(init_model.predict(xyz[-1:], verbose=0), dtype=float)[0]
    return autoregressive_rollout(
        model,
        xyz,
        initial_beta_rad=initial,
        window_size=window_size,
    ).astype(float)


def _prediction_variants(
    model: Any,
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    init_model: Any | None,
    window_size: int,
) -> list[tuple[str, np.ndarray]]:
    """Return canonical predictions, plus cut/direction stress rollouts for S3."""

    canonical = frame.sort_values("phase_idx", kind="stable").reset_index(drop=True)
    if not candidate.is_autoregressive:
        return [("forward_cut0", _predict(model, canonical, candidate, init_model=init_model, window_size=window_size))]
    count = len(canonical)
    base = np.arange(count)
    variants: list[tuple[str, np.ndarray]] = []
    for direction in ("forward", "reverse"):
        for cut in (0, count // 4, count // 2, (3 * count) // 4):
            indices = np.roll(base, -cut)
            if direction == "reverse":
                indices = indices[::-1]
            traversal = canonical.iloc[indices].reset_index(drop=True)
            traversed_prediction = _predict(
                model,
                traversal,
                candidate,
                init_model=init_model,
                window_size=window_size,
            )
            restored = np.empty_like(traversed_prediction)
            restored[indices] = traversed_prediction
            variants.append((f"{direction}_cut{cut}", restored))
    return variants


def _aggregate_variant_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics) == 1:
        return dict(metrics[0])
    return {
        "rows": int(metrics[0]["rows"]),
        "tracking_residual_p50_mm": float(max(row["tracking_residual_p50_mm"] for row in metrics)),
        "tracking_residual_p95_mm": float(max(row["tracking_residual_p95_mm"] for row in metrics)),
        "tracking_residual_max_mm": float(max(row["tracking_residual_max_mm"] for row in metrics)),
        "tracking_success_rate_3mm": float(min(row["tracking_success_rate_3mm"] for row in metrics)),
        "beta_rms_p95_deg": float(max(row["beta_rms_p95_deg"] for row in metrics)),
        "beta_rms_max_deg": float(max(row["beta_rms_max_deg"] for row in metrics)),
        "joint_bounds_rate": float(min(row["joint_bounds_rate"] for row in metrics)),
        "rollout_variant_count": int(len(metrics)),
        "aggregation": "worst_of_four_cuts_times_two_directions",
    }


def evaluate_prediction(
    frame: pd.DataFrame,
    beta_pred: np.ndarray,
    environment: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ordered = frame.sort_values("phase_idx", kind="stable").reset_index(drop=True)
    prediction = np.asarray(beta_pred, dtype=float).reshape(len(ordered), 6)
    target = ordered.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    teacher_beta = ordered.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    achieved = environment.fk(prediction)
    residual_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
    beta_rms_deg = np.rad2deg(np.sqrt(np.mean(np.square(prediction - teacher_beta), axis=1)))
    bounds = np.asarray(environment.bounds, dtype=float)
    in_bounds = np.all(
        (prediction >= bounds[:, 0][None, :]) & (prediction <= bounds[:, 1][None, :]), axis=1
    )
    metric = {
        "rows": int(len(ordered)),
        "tracking_residual_p50_mm": float(np.percentile(residual_mm, 50)),
        "tracking_residual_p95_mm": float(np.percentile(residual_mm, 95)),
        "tracking_residual_max_mm": float(np.max(residual_mm)),
        "tracking_success_rate_3mm": float(np.mean(residual_mm <= 3.0)),
        "beta_rms_p95_deg": float(np.percentile(beta_rms_deg, 95)),
        "beta_rms_max_deg": float(np.max(beta_rms_deg)),
        "joint_bounds_rate": float(np.mean(in_bounds)),
    }
    result = ordered.loc[:, ["phase_idx", "phase_rad", *XYZ_COLUMNS]].copy()
    for index in range(6):
        result[f"teacher_beta{index + 1}_rad"] = teacher_beta[:, index]
        result[f"predicted_beta{index + 1}_rad"] = prediction[:, index]
    result[["achieved_x_m", "achieved_y_m", "achieved_z_m"]] = achieved
    result["tracking_residual_mm"] = residual_mm
    result["beta_rms_deg"] = beta_rms_deg
    result["within_joint_bounds"] = in_bounds
    return metric, result


def _train_one(
    tf: Any,
    *,
    candidate: Candidate,
    scope: str,
    seed: int,
    train_0p5: pd.DataFrame,
    validation_0p5: pd.DataFrame,
    geometry: StudentGeometry,
    output: Path,
    epochs_static: int,
    epochs_gru: int,
    window_size: int,
) -> dict[str, Any]:
    report_path = output / "report.json"
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    print(
        f"[train] scope={scope} candidate={candidate.candidate_id} seed={seed}",
        flush=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    if scope != "scale0p5":
        raise ValueError("pre-selection training is restricted to scale0p5")
    train_frames = [train_0p5]
    train = pd.concat(train_frames, ignore_index=True)
    init_model = None
    if candidate.is_autoregressive:
        model, init_model, history = _fit_gru(
            tf,
            train_frames,
            validation_0p5,
            candidate,
            geometry,
            seed=seed,
            epochs=epochs_gru,
            window_size=window_size,
        )
    else:
        model, history = _fit_static(
            tf,
            train,
            validation_0p5,
            candidate,
            geometry,
            seed=seed,
            epochs=epochs_static,
        )
    variants = _prediction_variants(
        model,
        validation_0p5,
        candidate,
        init_model=init_model,
        window_size=window_size,
    )
    variant_results = [
        (name, *evaluate_prediction(validation_0p5, beta, _ENVIRONMENT_HOLDER[0]))
        for name, beta in variants
    ]
    metrics = _aggregate_variant_metrics([row[1] for row in variant_results])
    prediction = variant_results[0][2]
    model.save(output / "model.keras")
    if init_model is not None:
        init_model.save(output / "rollout_initialiser.keras")
    prediction.to_parquet(output / "validation_0p5_predictions.parquet", index=False, compression="zstd")
    report = {
        "candidate": asdict(candidate),
        "scope": scope,
        "seed": int(seed),
        "selection_data": "0.5m_validation_only",
        "stress_0p75_evaluated_during_selection": False,
        "history": history,
        "validation_0p5": metrics,
        "rollout_variants": {name: value for name, value, _prediction in variant_results},
    }
    atomic_write_json(report_path, report)
    print(
        f"[done] scope={scope} candidate={candidate.candidate_id} seed={seed} "
        f"p95={metrics['tracking_residual_p95_mm']:.3f} mm",
        flush=True,
    )
    tf.keras.backend.clear_session()
    return report


# Kept process-local so evaluate_prediction continues to use the authoritative
# NumPy environment without serialising it through Keras callbacks.
_ENVIRONMENT_HOLDER: list[Any] = []


def run_screen(args: argparse.Namespace, tf: Any, environment: Any, output_root: Path) -> dict[str, Any]:
    train_0p5 = _eligible(_read_split(output_root, "a0p500m", "train"))
    validation_0p5 = _eligible(_read_split(output_root, "a0p500m", "validation"))
    geometry = _geometry(environment)
    reports: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for seed in SCREEN_SEEDS:
            path = output_root / "screen/scale0p5" / candidate.candidate_id / f"seed_{seed}"
            reports.append(
                _train_one(
                    tf,
                    candidate=candidate,
                    scope="scale0p5",
                    seed=seed,
                    train_0p5=train_0p5,
                    validation_0p5=validation_0p5,
                    geometry=geometry,
                    output=path,
                    epochs_static=args.epochs_static,
                    epochs_gru=args.epochs_gru,
                    window_size=args.window_size,
                )
            )
    rows = []
    for report in reports:
        rows.append(
            {
                "scope": report["scope"],
                "candidate_id": report["candidate"]["candidate_id"],
                "seed": report["seed"],
                **report["validation_0p5"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_root / "screen/screen_results.csv", index=False)
    aggregate = (
        table.groupby("candidate_id", as_index=False)
        .agg(
            tracking_residual_p95_mm=("tracking_residual_p95_mm", "median"),
            tracking_residual_max_mm=("tracking_residual_max_mm", "median"),
            tracking_success_rate_3mm=("tracking_success_rate_3mm", "median"),
            joint_bounds_rate=("joint_bounds_rate", "median"),
        )
    )
    aggregate["selection_score"] = (
        aggregate["tracking_residual_p95_mm"]
        + 0.1 * aggregate["tracking_residual_max_mm"]
        + 1000.0 * (1.0 - aggregate["joint_bounds_rate"])
    )
    aggregate = aggregate.sort_values(
        ["selection_score", "tracking_residual_p95_mm", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    aggregate.to_csv(output_root / "screen/scale0p5_aggregate.csv", index=False)
    winner_id = str(aggregate.iloc[0]["candidate_id"])
    selected = {
        "candidate": asdict(next(value for value in CANDIDATES if value.candidate_id == winner_id)),
        "selection_metrics": aggregate.iloc[0].to_dict(),
    }
    selection = {
        "protocol_id": "large-scale-student-tracking-v10.1",
        "selection_locked": True,
        "selection_benchmark": "0.5m_validation_only",
        "sealed_test_opened": False,
        "stress_0p75_evaluated": False,
        "selected": selected,
    }
    atomic_write_json(output_root / "screen/selection.json", selection)
    return selection


def run_final(args: argparse.Namespace, tf: Any, environment: Any, output_root: Path) -> dict[str, Any]:
    selection_path = output_root / "screen/selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("screen selection must be frozen before sealed-test evaluation")
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if not selection.get("selection_locked") or selection.get("sealed_test_opened"):
        raise RuntimeError("invalid selection state before sealed-test evaluation")
    train_0p5 = _eligible(_read_split(output_root, "a0p500m", "train"))
    train_0p75 = _eligible(_read_split(output_root, "a0p750m", "train"))
    validation_0p5 = _eligible(_read_split(output_root, "a0p500m", "validation"))
    test = {
        "a0p500m": _read_split(output_root, "a0p500m", "test"),
        "a0p750m": _read_split(output_root, "a0p750m", "test"),
    }
    geometry = _geometry(environment)
    final_reports: list[dict[str, Any]] = []
    scopes = {
        "joint": (selection["selected"]["candidate"], [train_0p5, train_0p75]),
        "scale0p5": (selection["selected"]["candidate"], [train_0p5]),
        "scale0p75": (selection["selected"]["candidate"], [train_0p75]),
    }
    for scope, (candidate_dict, train_frames) in scopes.items():
        candidate = Candidate(**candidate_dict)
        validation = validation_0p5
        train = pd.concat(train_frames, ignore_index=True)
        screen_reports = sorted(
            (
                output_root
                / "screen/scale0p5"
                / selection["selected"]["candidate"]["candidate_id"]
            ).glob("seed_*/report.json")
        )
        frozen_epoch_values = [
            int(_report["history"]["best_epoch"])
            for _report in (_read_json(path) for path in screen_reports)
        ]
        frozen_epochs = (
            max(1, int(round(float(np.median(frozen_epoch_values)))))
            if frozen_epoch_values
            else (args.epochs_gru if candidate.is_autoregressive else args.epochs_static)
        )
        for seed in FINAL_SEEDS:
            out = output_root / "final" / scope / f"seed_{seed}"
            report_path = out / "report.json"
            if report_path.exists():
                with report_path.open("r", encoding="utf-8") as handle:
                    final_reports.append(json.load(handle))
                continue
            out.mkdir(parents=True, exist_ok=True)
            init_model = None
            if scope == "scale0p75":
                model, init_model, history = _fit_fixed_epochs(
                    tf,
                    train_frames,
                    candidate,
                    geometry,
                    seed=seed,
                    epochs=frozen_epochs,
                    window_size=args.window_size,
                )
            elif candidate.is_autoregressive:
                model, init_model, history = _fit_gru(
                    tf,
                    train_frames,
                    validation,
                    candidate,
                    geometry,
                    seed=seed,
                    epochs=args.epochs_gru,
                    window_size=args.window_size,
                )
            else:
                model, history = _fit_static(
                    tf,
                    train,
                    validation,
                    candidate,
                    geometry,
                    seed=seed,
                    epochs=args.epochs_static,
                )
            evaluations: dict[str, Any] = {}
            scales = ("a0p500m", "a0p750m") if scope == "joint" else (("a0p500m",) if scope == "scale0p5" else ("a0p750m",))
            for scale in scales:
                variants = _prediction_variants(
                    model,
                    test[scale],
                    candidate,
                    init_model=init_model,
                    window_size=args.window_size,
                )
                variant_results = [
                    (name, *evaluate_prediction(test[scale], beta, environment))
                    for name, beta in variants
                ]
                metrics = _aggregate_variant_metrics([row[1] for row in variant_results])
                metrics["rollout_variants"] = {
                    name: value for name, value, _prediction in variant_results
                }
                prediction = variant_results[0][2]
                prediction.to_parquet(out / f"{scale}_sealed_test_predictions.parquet", index=False, compression="zstd")
                evaluations[scale] = metrics
            model.save(out / "model.keras")
            if init_model is not None:
                init_model.save(out / "rollout_initialiser.keras")
            report = {
                "scope": scope,
                "candidate": asdict(candidate),
                "seed": seed,
                "history": history,
                "sealed_test_evaluations": evaluations,
            }
            atomic_write_json(report_path, report)
            final_reports.append(report)
            tf.keras.backend.clear_session()
    rows = []
    for report in final_reports:
        for scale, metrics in report["sealed_test_evaluations"].items():
            rows.append({"scope": report["scope"], "seed": report["seed"], "scale": scale, **metrics})
    table = pd.DataFrame(rows)
    table.to_csv(output_root / "final/final_results.csv", index=False)
    summary_rows = (
        table.groupby(["scope", "scale"], as_index=False)
        .agg(
            residual_p95_median_mm=("tracking_residual_p95_mm", "median"),
            residual_p95_min_mm=("tracking_residual_p95_mm", "min"),
            residual_p95_max_mm=("tracking_residual_p95_mm", "max"),
            residual_max_median_mm=("tracking_residual_max_mm", "median"),
            success_rate_median=("tracking_success_rate_3mm", "median"),
            joint_bounds_rate_median=("joint_bounds_rate", "median"),
        )
    )
    summary_rows.to_csv(output_root / "final/final_summary.csv", index=False)
    summary = {
        "protocol_id": "large-scale-student-tracking-v10.1",
        "selection_sha256": sha256_file(selection_path),
        "sealed_test_opened_after_selection": True,
        "final_seed_count": len(FINAL_SEEDS),
        "records": summary_rows.to_dict(orient="records"),
    }
    atomic_write_json(output_root / "final/final_summary.json", summary)
    return summary


def main() -> None:
    import tensorflow as tf

    repo_root = Path(__file__).resolve().parents[2]
    project_root = project_root_from(repo_root)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "final", "all"), default="all")
    parser.add_argument(
        "--robot-config",
        default=project_root / "configs/robot_rods_only_standard_100k.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        default=project_root / "runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking",
        type=Path,
    )
    parser.add_argument("--epochs-static", type=int, default=300)
    parser.add_argument("--epochs-gru", type=int, default=180)
    parser.add_argument("--window-size", type=int, default=32)
    args = parser.parse_args()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("large-scale student training requires a TensorFlow GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    environment = load_environment(project_root, args.robot_config.resolve())
    _ENVIRONMENT_HOLDER[:] = [environment]
    report: dict[str, Any] = {
        "runtime": runtime_fingerprint(),
        "gpu_devices": [str(value) for value in gpus],
    }
    if args.stage in {"screen", "all"}:
        report["screen"] = run_screen(args, tf, environment, output_root)
    if args.stage in {"final", "all"}:
        report["final"] = run_final(args, tf, environment, output_root)
    atomic_write_json(output_root / "training_run_summary.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
