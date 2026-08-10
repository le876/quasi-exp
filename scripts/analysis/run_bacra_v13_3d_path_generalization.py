#!/usr/bin/env python3
"""Evaluate locked V13 Students on preregistered non-planar 3D shell paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from quasi_exp.teacher.ellipsoidal_shell import EllipsoidSpec
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from run_bacra_v13_ellipsoidal_shell_atlas import load_config, project_root_from
from run_trajectory_canonical_teacher_v10 import load_environment


PROTOCOL_ID = "bacra-v13-locked-student-nonplanar-3d-path-generalization"
CLAIM_SCOPE = "simulation_locked_student_nonplanar_3d_shell_path_generalization"
PATH_SPECS: tuple[dict[str, Any], ...] = (
    {"shape": "tilted_ellipse", "inclination_deg": 5.0, "azimuth_deg": 15.0, "axis_ratio": 0.35, "rho_mm": 4.0},
    {"shape": "tilted_ellipse", "inclination_deg": 28.0, "azimuth_deg": 80.0, "axis_ratio": 0.50, "rho_mm": 8.0},
    {"shape": "tilted_ellipse", "inclination_deg": 51.0, "azimuth_deg": 150.0, "axis_ratio": 0.70, "rho_mm": 12.0},
    {"shape": "tilted_ellipse", "inclination_deg": 74.0, "azimuth_deg": 235.0, "axis_ratio": 0.90, "rho_mm": 16.0},
    {"shape": "rounded_superellipse", "inclination_deg": 12.0, "azimuth_deg": 40.0, "axis_ratio": 0.40, "rho_mm": 8.0},
    {"shape": "rounded_superellipse", "inclination_deg": 34.0, "azimuth_deg": 105.0, "axis_ratio": 0.55, "rho_mm": 12.0},
    {"shape": "rounded_superellipse", "inclination_deg": 57.0, "azimuth_deg": 175.0, "axis_ratio": 0.75, "rho_mm": 16.0},
    {"shape": "rounded_superellipse", "inclination_deg": 78.0, "azimuth_deg": 265.0, "axis_ratio": 0.90, "rho_mm": 4.0},
    {"shape": "peanut", "inclination_deg": 18.0, "azimuth_deg": 0.0, "axis_ratio": 0.35, "rho_mm": 12.0},
    {"shape": "peanut", "inclination_deg": 39.0, "azimuth_deg": 65.0, "axis_ratio": 0.50, "rho_mm": 16.0},
    {"shape": "peanut", "inclination_deg": 61.0, "azimuth_deg": 140.0, "axis_ratio": 0.70, "rho_mm": 4.0},
    {"shape": "peanut", "inclination_deg": 82.0, "azimuth_deg": 220.0, "axis_ratio": 0.85, "rho_mm": 8.0},
    {"shape": "spherical_lissajous", "inclination_deg": 8.0, "azimuth_deg": 55.0, "axis_ratio": 0.40, "rho_mm": 16.0},
    {"shape": "spherical_lissajous", "inclination_deg": 31.0, "azimuth_deg": 115.0, "axis_ratio": 0.55, "rho_mm": 4.0},
    {"shape": "spherical_lissajous", "inclination_deg": 54.0, "azimuth_deg": 195.0, "axis_ratio": 0.75, "rho_mm": 8.0},
    {"shape": "spherical_lissajous", "inclination_deg": 76.0, "azimuth_deg": 285.0, "axis_ratio": 0.90, "rho_mm": 12.0},
)


def _gate(path: Path, checks: Mapping[str, bool], *, semantics: str, **evidence: Any) -> dict[str, Any]:
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


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _basis(inclination_deg: float, azimuth_deg: float) -> np.ndarray:
    inclination = np.deg2rad(float(inclination_deg))
    azimuth = np.deg2rad(float(azimuth_deg))
    normal = np.asarray((
        np.sin(inclination) * np.cos(azimuth),
        np.sin(inclination) * np.sin(azimuth),
        np.cos(inclination),
    ))
    reference = np.asarray((0.0, 0.0, 1.0)) if abs(normal[2]) < 0.9 else np.asarray((1.0, 0.0, 0.0))
    axis0 = np.cross(reference, normal)
    axis0 /= np.linalg.norm(axis0)
    axis1 = np.cross(normal, axis0)
    return np.column_stack((axis0, axis1, normal))


def _shape_coordinates(shape: str, phase: np.ndarray, axis_ratio: float) -> np.ndarray:
    cosine, sine = np.cos(phase), np.sin(phase)
    if shape == "tilted_ellipse":
        x, y, z = cosine, axis_ratio * sine, 0.24 * np.sin(2.0 * phase + 0.25)
    elif shape == "rounded_superellipse":
        x = np.sign(cosine) * np.sqrt(np.abs(cosine))
        y = axis_ratio * np.sign(sine) * np.sqrt(np.abs(sine))
        z = 0.20 * np.sin(3.0 * phase + 0.35)
    elif shape == "peanut":
        radial = 1.0 + 0.28 * np.cos(2.0 * phase)
        x, y, z = radial * cosine, axis_ratio * radial * sine, 0.26 * np.sin(2.0 * phase + 0.45)
    elif shape == "spherical_lissajous":
        x, y, z = cosine, axis_ratio * np.sin(2.0 * phase), 0.30 * np.sin(3.0 * phase + 0.20)
    else:
        raise ValueError(f"unknown registered 3D path shape: {shape}")
    return np.column_stack((x, y, z))


def build_path(spec: EllipsoidSpec, path_spec: Mapping[str, Any], phase_count: int) -> dict[str, np.ndarray | float]:
    phase = np.linspace(0.0, 2.0 * np.pi, int(phase_count), endpoint=False)
    coordinates = _shape_coordinates(str(path_spec["shape"]), phase, float(path_spec["axis_ratio"]))
    basis = _basis(float(path_spec["inclination_deg"]), float(path_spec["azimuth_deg"]))
    raw = 0.42 * basis[:, 2] + 0.82 * coordinates[:, :2] @ basis[:, :2].T + 0.55 * coordinates[:, 2, None] * basis[:, 2]
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    surface = spec.surface_points(unit)
    normal = spec.outward_normals(unit)
    shape_frequency = {"tilted_ellipse": 1, "rounded_superellipse": 2, "peanut": 3, "spherical_lissajous": 4}[str(path_spec["shape"])]
    rho_m = float(path_spec["rho_mm"]) * 1.0e-3 * np.sin(shape_frequency * phase + 0.37)
    target = surface + rho_m[:, None] * normal
    singular = np.linalg.svd(target - target.mean(axis=0), compute_uv=False)
    centered = target - target.mean(axis=0)
    principal = np.linalg.svd(centered, full_matrices=False)[2]
    projected = centered @ principal.T
    scale_m = 0.5 * float(np.max(projected[:, 0]) - np.min(projected[:, 0]))
    return {
        "phase": phase,
        "unit": unit,
        "surface": surface,
        "normal": normal,
        "rho_m": rho_m,
        "target": target,
        "scale_m": scale_m,
        "observed_axis_ratio": float(singular[1] / singular[0]),
        "nonplanarity": float(singular[2] / singular[0]),
    }


def _registered_catalog(spec: EllipsoidSpec, phase_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_rows: list[dict[str, Any]] = []
    point_rows: list[pd.DataFrame] = []
    for path_id, path_spec in enumerate(PATH_SPECS):
        path = build_path(spec, path_spec, phase_count)
        inclination_bin = min(3, int(float(path_spec["inclination_deg"]) // 22.5))
        axis_ratio_bin = min(3, int((float(path_spec["axis_ratio"]) - 0.30) // 0.175))
        catalog_rows.append({
            "path_id": path_id,
            **path_spec,
            "inclination_bin": inclination_bin,
            "axis_ratio_bin": axis_ratio_bin,
            "scale_m": path["scale_m"],
            "observed_axis_ratio": path["observed_axis_ratio"],
            "nonplanarity": path["nonplanarity"],
        })
        target = np.asarray(path["target"])
        point_rows.append(pd.DataFrame({
            "path_id": path_id,
            "phase_idx": np.arange(phase_count, dtype=np.int64),
            "phase_rad": path["phase"],
            "x_m": target[:, 0], "y_m": target[:, 1], "z_m": target[:, 2],
            "rho_mm": np.asarray(path["rho_m"]) * 1000.0,
        }))
    return pd.DataFrame(catalog_rows), pd.concat(point_rows, ignore_index=True)


def _save_visualizations(catalog: pd.DataFrame, points: pd.DataFrame, output: Path, primary_seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage = output / "03_visualization"
    stage.mkdir()
    figure = plt.figure(figsize=(16, 16), constrained_layout=True)
    for path_id, row in catalog.set_index("path_id").iterrows():
        axis = figure.add_subplot(4, 4, int(path_id) + 1, projection="3d")
        subset = points.loc[(points["path_id"] == path_id) & (points["seed"] == primary_seed)]
        axis.plot(subset["x_m"], subset["y_m"], subset["z_m"], color="#202735", linewidth=2.0, label="target")
        axis.plot(subset["achieved_x_m"], subset["achieved_y_m"], subset["achieved_z_m"], color="#e86e4d", linewidth=1.1, label="Student")
        axis.set_title(f"{row['shape']}  tilt={row['inclination_deg']:.0f}°  r={row['axis_ratio']:.2f}", fontsize=9)
        axis.set_xticks([]); axis.set_yticks([]); axis.set_zticks([])
        axis.set_box_aspect((1, 1, 1))
    figure.suptitle(f"V13 preregistered non-planar 3D paths — Student seed {primary_seed}")
    figure.savefig(stage / "tracking_grid.png", dpi=170)
    plt.close(figure)

    metrics = points.groupby(["seed", "path_id"], sort=True)["student_fk_error_mm"].quantile(0.95).reset_index()
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for seed, subset in metrics.groupby("seed", sort=True):
        axis.plot(subset["path_id"], subset["student_fk_error_mm"], marker="o", label=f"seed {seed}")
    axis.axhline(5.0, color="#c03d3e", linestyle="--", label="5 mm Gate")
    axis.set_xlabel("path_id"); axis.set_ylabel("FK error P95 (mm)")
    axis.set_title("Locked Student performance across all 3D paths")
    axis.grid(alpha=0.25); axis.legend(ncol=4)
    figure.savefig(stage / "path_error_p95.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    shell_root = Path(args.shell_root).resolve()
    student_root = Path(args.student_root).resolve()
    output_root = Path(args.output).resolve()
    if output_root.exists():
        raise FileExistsError(f"3D path output already exists: {output_root}")
    output_root.mkdir(parents=True)
    config = load_config(args.config, "formal")
    path_config = config["trajectory_3d"]
    shell_gate_path = shell_root / "05_summary/gate.json"
    shell_gate = json.loads(shell_gate_path.read_text())
    replay_manifest_path = shell_root / "00_protocol/source_manifest.json"
    replay_manifest = json.loads(replay_manifest_path.read_text())
    ellipsoid_path = Path(replay_manifest["source_artifacts"]["ellipsoid"]["path"])
    student_gate_path = student_root / "04_summary/gate.json"
    model_lock_path = student_root / "02_model_lock/model_lock.json"
    student_gate = json.loads(student_gate_path.read_text())
    model_lock = json.loads(model_lock_path.read_text())
    spec_payload = json.loads(ellipsoid_path.read_text())
    spec = EllipsoidSpec(spec_payload["center_m"], spec_payload["semiaxes_m"], spec_payload["rotation"])
    phase_count = int(path_config["phase_count"])
    catalog, targets = _registered_catalog(spec, phase_count)
    protocol = output_root / "00_protocol"
    protocol.mkdir()
    _atomic_parquet(catalog, protocol / "path_catalog.parquet")
    _atomic_parquet(targets, protocol / "target_paths.parquet")
    git_sha = subprocess.run(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    protocol_gate = _gate(
        protocol / "gate.json",
        {
            "formal_shell_gate_pass": bool(shell_gate["gate_pass"]),
            "student_gate_pass": bool(student_gate["gate_pass"]),
            "model_lock_dataset_matches_shell": model_lock["source_dataset_sha256"] == shell_gate["dataset_sha256"],
            "ellipsoid_bytes_match_manifest": sha256_file(ellipsoid_path) == replay_manifest["source_artifacts"]["ellipsoid"]["sha256"],
            "worktree_clean": git_status == "",
            "registered_path_count": len(catalog) == int(path_config["path_count"]),
            "registered_phase_count": phase_count == 720,
            "four_registered_shapes": catalog["shape"].nunique() == int(path_config["shape_count"]),
            "radial_amplitudes_registered": sorted(catalog["rho_mm"].unique()) == sorted(path_config["radial_amplitudes_mm"]),
            "targets_within_accepted_half_thickness": float(targets["rho_mm"].abs().max()) <= float(path_config["shell_half_thickness_mm"]) + 1.0e-9,
            "all_paths_nonplanar": bool((catalog["nonplanarity"] >= float(path_config["nonplanarity_min"])).all()),
            "automatic_chart_classifier_disabled": config["student"]["automatic_chart_classifier"] is False,
        },
        semantics="preregistered_nonplanar_3d_paths_before_locked_model_inference",
        git_sha=git_sha,
        shell_dataset_sha256=shell_gate["dataset_sha256"],
        shell_gate_sha256=sha256_file(shell_gate_path),
        student_gate_sha256=sha256_file(student_gate_path),
        model_lock_sha256=sha256_file(model_lock_path),
        ellipsoid_sha256=sha256_file(ellipsoid_path),
        path_catalog_sha256=sha256_file(protocol / "path_catalog.parquet"),
        target_paths_sha256=sha256_file(protocol / "target_paths.parquet"),
    )
    if not protocol_gate["gate_pass"]:
        report = {"protocol_id": PROTOCOL_ID, "gate_pass": False, "stopped_after": "protocol", "output_root": str(output_root)}
        atomic_write_json(output_root / "run_report.json", report)
        return report

    environment = load_environment(project_root, project_root / str(config["robot_config"]))
    import tensorflow as tf

    target_xyz = targets.loc[:, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    features = np.column_stack((target_xyz.astype(np.float32), np.ones((len(target_xyz), 1), dtype=np.float32)))
    result_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for raw_seed, lock in sorted(model_lock["models"].items(), key=lambda item: int(item[0])):
        seed = int(raw_seed)
        model_path = Path(lock["path"])
        if sha256_file(model_path) != lock["sha256"]:
            raise RuntimeError(f"model bytes changed after lock: seed {seed}")
        model = tf.keras.models.load_model(model_path, compile=False)
        beta = np.asarray(model.predict(features, batch_size=2048, verbose=0), dtype=float)
        achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
        error_mm = np.linalg.norm(achieved - target_xyz, axis=1) * 1000.0
        details = targets.copy()
        details.insert(0, "seed", seed)
        details["achieved_x_m"] = achieved[:, 0]; details["achieved_y_m"] = achieved[:, 1]; details["achieved_z_m"] = achieved[:, 2]
        details["student_fk_error_mm"] = error_mm
        for joint in range(beta.shape[1]):
            details[f"predicted_beta{joint + 1}_rad"] = beta[:, joint]
        result_frames.append(details)
        for path_id, subset in details.groupby("path_id", sort=True):
            row = catalog.loc[catalog["path_id"] == path_id].iloc[0]
            beta_path = subset.loc[:, [f"predicted_beta{joint + 1}_rad" for joint in range(beta.shape[1])]].to_numpy()
            beta_steps = np.diff(np.vstack((beta_path, beta_path[:1])), axis=0)
            metric_frames.append(pd.DataFrame([{
                "seed": seed, "path_id": int(path_id),
                "fk_p50_mm": float(np.percentile(subset["student_fk_error_mm"], 50)),
                "fk_p95_mm": float(np.percentile(subset["student_fk_error_mm"], 95)),
                "fk_max_mm": float(subset["student_fk_error_mm"].max()),
                "relative_p95": float(np.percentile(subset["student_fk_error_mm"], 95) / (float(row["scale_m"]) * 1000.0)),
                "relative_max": float(subset["student_fk_error_mm"].max() / (float(row["scale_m"]) * 1000.0)),
                "beta_step_p95_deg": float(np.percentile(np.linalg.norm(beta_steps, axis=1), 95) * 180.0 / np.pi),
                "beta_step_max_deg": float(np.max(np.linalg.norm(beta_steps, axis=1)) * 180.0 / np.pi),
                "predictions_in_bounds": bool(np.all(beta_path >= environment.bounds[:, 0] - 1.0e-9) and np.all(beta_path <= environment.bounds[:, 1] + 1.0e-9)),
            }]))
    results = pd.concat(result_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True).merge(catalog, on="path_id", how="left", validate="many_to_one")
    metrics["path_pass"] = (
        (metrics["fk_p95_mm"] <= float(path_config["absolute_error_p95_max_mm"]))
        & (metrics["fk_max_mm"] <= float(path_config["absolute_error_max_mm"]))
        & (metrics["relative_p95"] <= float(path_config["relative_error_p95_max"]))
        & (metrics["relative_max"] <= float(path_config["relative_error_max"]))
        & metrics["predictions_in_bounds"]
    )
    result_stage = output_root / "01_results"
    result_stage.mkdir()
    _atomic_parquet(results, result_stage / "point_metrics.parquet")
    _atomic_parquet(metrics, result_stage / "path_metrics.parquet")
    group_columns = ("shape", "inclination_bin", "axis_ratio_bin", "rho_mm")
    group_pass = {column: {str(key): float(value) for key, value in metrics.groupby(column, sort=True)["path_pass"].mean().items()} for column in group_columns}
    summary = output_root / "02_summary"
    summary.mkdir()
    gate = _gate(
        summary / "gate.json",
        {
            "all_registered_models_evaluated": metrics["seed"].nunique() == len(model_lock["models"]),
            "all_registered_paths_evaluated": metrics["path_id"].nunique() == int(path_config["path_count"]),
            "all_predictions_in_bounds": bool(metrics["predictions_in_bounds"].all()),
            "overall_path_pass": float(metrics["path_pass"].mean()) >= float(path_config["overall_path_pass_min"]),
            "each_nonempty_bin_pass": all(value >= float(path_config["nonempty_bin_pass_min"]) for groups in group_pass.values() for value in groups.values()),
            "absolute_point_p95": float(np.percentile(results["student_fk_error_mm"], 95)) <= float(path_config["absolute_error_p95_max_mm"]),
            "absolute_point_max": float(results["student_fk_error_mm"].max()) <= float(path_config["absolute_error_max_mm"]),
            "source_shell_and_student_gates_remain_passed": bool(shell_gate["gate_pass"] and student_gate["gate_pass"]),
        },
        semantics="locked_student_nonplanar_3d_shell_path_generalization",
        path_count=int(catalog.shape[0]),
        point_count=int(results.shape[0]),
        seed_count=int(metrics["seed"].nunique()),
        shape_count=int(catalog["shape"].nunique()),
        inclination_min_deg=float(catalog["inclination_deg"].min()),
        inclination_max_deg=float(catalog["inclination_deg"].max()),
        design_axis_ratio_min=float(catalog["axis_ratio"].min()),
        design_axis_ratio_max=float(catalog["axis_ratio"].max()),
        observed_axis_ratio_min=float(catalog["observed_axis_ratio"].min()),
        observed_axis_ratio_max=float(catalog["observed_axis_ratio"].max()),
        nonplanarity_min=float(catalog["nonplanarity"].min()),
        nonplanarity_max=float(catalog["nonplanarity"].max()),
        radial_amplitude_min_mm=float(catalog["rho_mm"].min()),
        radial_amplitude_max_mm=float(catalog["rho_mm"].max()),
        path_pass_ratio=float(metrics["path_pass"].mean()),
        point_fk_p50_mm=float(np.percentile(results["student_fk_error_mm"], 50)),
        point_fk_p95_mm=float(np.percentile(results["student_fk_error_mm"], 95)),
        point_fk_max_mm=float(results["student_fk_error_mm"].max()),
        worst_path_fk_p95_mm=float(metrics["fk_p95_mm"].max()),
        worst_path_fk_max_mm=float(metrics["fk_max_mm"].max()),
        beta_step_p95_deg=float(np.percentile(metrics["beta_step_p95_deg"], 95)),
        beta_step_max_deg=float(metrics["beta_step_max_deg"].max()),
        group_pass_ratios=group_pass,
    )
    _save_visualizations(catalog, results, output_root, min(int(value) for value in model_lock["models"]))
    report = {"protocol_id": PROTOCOL_ID, "gate_pass": bool(gate["gate_pass"]), "stopped_after": None if gate["gate_pass"] else "path_summary", "output_root": str(output_root)}
    atomic_write_json(output_root / "run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SOURCE_ROOT / "configs/bacra_v13_ellipsoidal_shell_atlas.yaml"))
    parser.add_argument("--project-root", default=str(project_root_from(SOURCE_ROOT)))
    parser.add_argument("--shell-root", required=True)
    parser.add_argument("--student-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report.get("gate_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
