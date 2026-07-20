#!/usr/bin/env python3
"""Refit selected students on all labels and evaluate 1440 odd interstitial phases."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.large_scale import EllipseChallenge
from quasi_exp.teacher.student import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.student_tracking_tf import (
    autoregressive_rollout,
    build_gru_model,
    build_static_model,
    compile_student,
    make_cyclic_windows,
    packed_targets,
)

from run_trajectory_canonical_teacher_v10 import load_environment, project_root_from, runtime_fingerprint
from train_large_scale_students_v10 import Candidate, _geometry


DEPLOYMENT_SEED = 20260740


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _epoch_budget(output_root: Path, scope: str, fallback: int) -> int:
    values = []
    for report_path in sorted((output_root / "final" / scope).glob("seed_*/report.json")):
        report = _load_json(report_path)
        values.append(int(report["history"]["best_epoch"]))
    return max(1, int(round(float(np.median(values))))) if values else int(fallback)


def _fit_fixed(
    tf: Any,
    frames: list[pd.DataFrame],
    candidate: Candidate,
    geometry: Any,
    *,
    epochs: int,
    window_size: int,
) -> tuple[Any, Any | None, dict[str, Any]]:
    tf.keras.utils.set_random_seed(DEPLOYMENT_SEED)
    started = time.perf_counter()
    init_model = None
    if candidate.student == "S3":
        windows = [
            make_cyclic_windows(frame, window_size=window_size, include_reverse=True)
            for frame in frames
        ]
        x = np.concatenate([value[0] for value in windows], axis=0)
        y = np.concatenate([value[1] for value in windows], axis=0)
        model = build_gru_model(x, geometry=geometry, output_mode=candidate.output_mode)
        compile_student(model, geometry=geometry, lambda_fk=candidate.lambda_fk, learning_rate=5.0e-4)
        history = model.fit(x, y, epochs=epochs, batch_size=64, shuffle=True, verbose=0)
        full = pd.concat(frames, ignore_index=True)
        xyz = full.loc[:, XYZ_COLUMNS].to_numpy(dtype=np.float32)
        init_model = build_static_model(xyz, geometry=geometry, output_mode="tanh")
        compile_student(init_model, geometry=geometry, lambda_fk=1.0, learning_rate=1.0e-3)
        init_model.fit(
            xyz,
            packed_targets(full),
            epochs=max(epochs, 100),
            batch_size=min(128, len(full)),
            shuffle=True,
            verbose=0,
        )
        row_count = int(len(x))
    else:
        full = pd.concat(frames, ignore_index=True)
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
        row_count = int(len(full))
    return model, init_model, {
        "epochs": int(epochs),
        "rows_or_windows": row_count,
        "final_loss": float(history.history["loss"][-1]),
        "wall_time_s": float(time.perf_counter() - started),
    }


def _interstitial_targets(
    *,
    pose: dict[str, Any],
    minor_to_major_ratio: float,
    phase_count: int = 2880,
) -> tuple[np.ndarray, np.ndarray]:
    challenge = EllipseChallenge.from_major_semiaxis_m(
        float(pose["major_semiaxis_m"]),
        minor_to_major_ratio=minor_to_major_ratio,
    )
    dense = challenge.generate_targets(
        center_m=np.asarray(pose["center_m"], dtype=float),
        major_direction=np.asarray(pose["major_direction"], dtype=float),
        minor_direction=np.asarray(pose["minor_direction"], dtype=float),
        phase_count=phase_count,
    )
    indices = np.arange(1, phase_count, 2, dtype=np.int64)
    return dense[indices], indices * (2.0 * np.pi / phase_count)


def _evaluate_unlabelled(
    target: np.ndarray,
    phase: np.ndarray,
    beta: np.ndarray,
    environment: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    achieved = environment.fk(beta)
    residual = np.linalg.norm(achieved - target, axis=1) * 1000.0
    bounds = np.asarray(environment.bounds, dtype=float)
    in_bounds = np.all(
        (beta >= bounds[:, 0][None, :]) & (beta <= bounds[:, 1][None, :]), axis=1
    )
    metrics = {
        "interstitial_phase_count": int(len(target)),
        "tracking_residual_p50_mm": float(np.percentile(residual, 50)),
        "tracking_residual_p95_mm": float(np.percentile(residual, 95)),
        "tracking_residual_max_mm": float(np.max(residual)),
        "tracking_success_rate_3mm": float(np.mean(residual <= 3.0)),
        "joint_bounds_rate": float(np.mean(in_bounds)),
    }
    frame = pd.DataFrame(
        {
            "phase_rad": phase,
            "target_x_m": target[:, 0],
            "target_y_m": target[:, 1],
            "target_z_m": target[:, 2],
            "achieved_x_m": achieved[:, 0],
            "achieved_y_m": achieved[:, 1],
            "achieved_z_m": achieved[:, 2],
            "tracking_residual_mm": residual,
            "within_joint_bounds": in_bounds,
        }
    )
    for index in range(6):
        frame[f"predicted_beta{index + 1}_rad"] = beta[:, index]
    return metrics, frame


def _set_equal_3d(ax: Any, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    radius = 0.55 * float(np.max(high - low))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_tracking(frame: pd.DataFrame, pose: dict[str, Any], title: str, output: Path) -> None:
    target = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    achieved = frame.loc[:, ["achieved_x_m", "achieved_y_m", "achieved_z_m"]].to_numpy(dtype=float)
    center = np.asarray(pose["center_m"], dtype=float)
    major = np.asarray(pose["major_direction"], dtype=float)
    minor = np.asarray(pose["minor_direction"], dtype=float)
    major /= np.linalg.norm(major)
    minor /= np.linalg.norm(minor)
    target_face = np.column_stack([(target - center) @ major, (target - center) @ minor])
    achieved_face = np.column_stack([(achieved - center) @ major, (achieved - center) @ minor])
    phase = frame["phase_rad"].to_numpy(dtype=float)
    residual = frame["tracking_residual_mm"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(target_face[:, 0], target_face[:, 1], color="#1f77b4", linewidth=2.5, label="Target")
    ax0.plot(achieved_face[:, 0], achieved_face[:, 1], color="#d62728", linewidth=1.5, label="Model FK")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("Major-axis coordinate (m)")
    ax0.set_ylabel("Minor-axis coordinate (m)")
    ax0.set_title("Face-on ellipse")
    ax0.grid(alpha=0.25)
    ax0.legend()

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax1.plot(*target.T, color="#1f77b4", linewidth=2.5, label="Target")
    ax1.plot(*achieved.T, color="#d62728", linewidth=1.3, label="Model FK")
    ax1.view_init(elev=24, azim=-58)
    _set_equal_3d(ax1, np.vstack([target, achieved]))
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Tilted 3-D view")
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(phase, residual, color="#7b3294", linewidth=1.4)
    ax2.axhline(3.0, color="#444444", linestyle="--", linewidth=1.2, label="3 mm")
    ax2.set_xlabel("Phase (rad)")
    ax2.set_ylabel("FK tracking residual (mm)")
    ax2.set_title("Interstitial phase error")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig.suptitle(title, fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    import yaml
    import tensorflow as tf

    repo_root = Path(__file__).resolve().parents[2]
    project_root = project_root_from(repo_root)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-config",
        default=project_root / "configs/robot_rods_only_standard_100k.yaml",
        type=Path,
    )
    parser.add_argument(
        "--challenge-config",
        default=repo_root / "configs/large_scale_ellipse_challenge_v10.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        default=project_root / "runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking",
        type=Path,
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--fallback-epochs", type=int, default=180)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    selection_path = output_root / "screen/selection.json"
    final_path = output_root / "final/final_summary.json"
    if not selection_path.exists() or not final_path.exists():
        raise FileNotFoundError("frozen selection and five-seed final summary are required")
    selection = _load_json(selection_path)
    with args.challenge_config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("deployment refit requires a TensorFlow GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    environment = load_environment(project_root, args.robot_config.resolve())
    geometry = _geometry(environment)
    dense = {
        slug: pd.read_parquet(output_root / slug / "teacher_dense/dense_teacher.parquet")
        for slug in ("a0p500m", "a0p750m")
    }
    dense = {
        slug: frame.loc[frame["label_eligible"].astype(bool)].sort_values("phase_idx").reset_index(drop=True)
        for slug, frame in dense.items()
    }
    scopes = {
        "joint": (selection["selected"]["joint"]["candidate"], ["a0p500m", "a0p750m"]),
        "scale0p5": (selection["selected"]["scale0p5"]["candidate"], ["a0p500m"]),
        "scale0p75": (selection["selected"]["scale0p5"]["candidate"], ["a0p750m"]),
    }
    reports: list[dict[str, Any]] = []
    for scope, (candidate_dict, scale_slugs) in scopes.items():
        candidate = Candidate(**candidate_dict)
        epochs = _epoch_budget(output_root, scope, args.fallback_epochs)
        model, init_model, training = _fit_fixed(
            tf,
            [dense[slug] for slug in scale_slugs],
            candidate,
            geometry,
            epochs=epochs,
            window_size=args.window_size,
        )
        scope_dir = output_root / "deployment" / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        model_path = scope_dir / "model.keras"
        model.save(model_path)
        if init_model is not None:
            init_model.save(scope_dir / "rollout_initialiser.keras")
        scale_reports: dict[str, Any] = {}
        for slug in scale_slugs:
            source_pose = (
                project_root
                / "runs/trajectory_canonical_teacher_v10/07_large_scale_challenge"
                / slug
                / "pose_report.json"
            )
            pose = _load_json(source_pose)
            target, phase = _interstitial_targets(
                pose=pose,
                minor_to_major_ratio=float(config["minor_to_major_ratio"]),
            )
            if candidate.student == "S3":
                if init_model is None:
                    raise RuntimeError("missing S3 rollout initialiser")
                initial = np.asarray(init_model.predict(target[:1], verbose=0), dtype=float)[0]
                beta = autoregressive_rollout(
                    model,
                    target,
                    initial_beta_rad=initial,
                    window_size=args.window_size,
                )
            else:
                beta = np.asarray(model.predict(target, verbose=0), dtype=float)
            metrics, prediction = _evaluate_unlabelled(target, phase, beta, environment)
            prediction_path = scope_dir / f"{slug}_interstitial_predictions.parquet"
            prediction.to_parquet(prediction_path, index=False, compression="zstd")
            plot_path = scope_dir / f"{slug}_tracking.png"
            plot_tracking(
                prediction,
                pose,
                f"{slug.replace('a0p', 'a=0.').replace('m', ' m')} — {scope} — {candidate.candidate_id}",
                plot_path,
            )
            scale_reports[slug] = {
                **metrics,
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "plot_path": str(plot_path.resolve()),
                "pose_path": str(source_pose.resolve()),
                "pose_sha256": sha256_file(source_pose),
            }
        report = {
            "scope": scope,
            "candidate": asdict(candidate),
            "deployment_seed": DEPLOYMENT_SEED,
            "training": training,
            "trained_on_all_eligible_dense_labels": {
                slug: int(len(dense[slug])) for slug in scale_slugs
            },
            "evaluation_grid": "2880-grid odd interstitial phases (1440 unseen points)",
            "model_path": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "scales": scale_reports,
        }
        atomic_write_json(scope_dir / "report.json", report)
        reports.append(report)
        tf.keras.backend.clear_session()
    manifest = {
        "protocol_id": "large-scale-student-tracking-v10.1-deployment",
        "runtime": runtime_fingerprint(),
        "selection_sha256": sha256_file(selection_path),
        "final_summary_sha256": sha256_file(final_path),
        "gpu_preflight_sha256": sha256_file(output_root / "gpu_preflight.json"),
        "worker_code_sha256": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str((repo_root / "src/quasi_exp/teacher/student_tracking_tf.py").resolve()): sha256_file(
                repo_root / "src/quasi_exp/teacher/student_tracking_tf.py"
            ),
        },
        "reports": reports,
    }
    atomic_write_json(output_root / "deployment/deployment_summary.json", manifest)
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
