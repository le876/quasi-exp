#!/usr/bin/env python3
"""Evaluate locked V13 Students on analytical ellipsoid plane sections."""

from __future__ import annotations

import argparse
import json
import math
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
from scipy.stats import norm, qmc

from quasi_exp.teacher.dense_chart_sampling import XYZ_COLUMNS
from quasi_exp.teacher.ellipsoidal_shell import EllipsoidSpec, plane_section
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from run_bacra_v13_ellipsoidal_shell_atlas import load_config, project_root_from
from run_trajectory_canonical_teacher_v10 import load_environment


PROTOCOL_ID = "bacra-v13-locked-student-analytical-ellipse-generalization"
CLAIM_SCOPE = "simulation_locked_student_analytical_ellipse_generalization"


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


def _section_candidates(spec: EllipsoidSpec, count: int) -> pd.DataFrame:
    engine = qmc.Sobol(d=4, scramble=True, seed=20260830)
    sample = engine.random_base2(int(math.ceil(math.log2(count))))[:count]
    normals = norm.ppf(np.clip(sample[:, :3], 1.0e-9, 1.0 - 1.0e-9))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offset_fraction = -0.97 + 1.94 * sample[:, 3]
    rows: list[dict[str, Any]] = []
    for candidate_id, (normal_vector, fraction) in enumerate(zip(normals, offset_fraction)):
        support = float(np.linalg.norm(spec.linear_map.T @ normal_vector))
        offset = float(normal_vector @ spec.center_m + fraction * support)
        section = plane_section(spec, normal_vector, offset)
        if section is None:
            continue
        major_axis = section.axes_world[:, 0].copy()
        first_nonzero = int(np.flatnonzero(np.abs(major_axis) > 1.0e-12)[0])
        if major_axis[first_nonzero] < 0.0:
            major_axis *= -1.0
        orientation = float(np.mod(np.arctan2(major_axis[1], major_axis[0]), np.pi))
        rows.append(
            {
                "candidate_id": candidate_id,
                "normal_x": normal_vector[0], "normal_y": normal_vector[1], "normal_z": normal_vector[2],
                "plane_offset_m": offset,
                "offset_fraction": fraction,
                "center_x_m": section.center_m[0], "center_y_m": section.center_m[1], "center_z_m": section.center_m[2],
                "axis0_x": section.axes_world[0, 0], "axis0_y": section.axes_world[1, 0], "axis0_z": section.axes_world[2, 0],
                "axis1_x": section.axes_world[0, 1], "axis1_y": section.axes_world[1, 1], "axis1_z": section.axes_world[2, 1],
                "semimajor_m": section.semiaxes_m[0],
                "semiminor_m": section.semiaxes_m[1],
                "axis_ratio": section.semiaxes_m[1] / section.semiaxes_m[0],
                "orientation_rad": orientation,
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["axis_ratio"].between(0.30, 1.0)].copy()
    frame["size_bin"] = pd.qcut(frame["semimajor_m"], 5, labels=False, duplicates="drop")
    frame["axis_ratio_bin"] = np.minimum(
        3, np.floor((frame["axis_ratio"] - 0.30) / 0.175).astype(int)
    )
    frame["orientation_bin"] = np.minimum(
        11, np.floor(frame["orientation_rad"] / np.pi * 12).astype(int)
    )
    frame["offset_bin"] = np.minimum(
        3, np.floor((frame["offset_fraction"] + 0.97) / 1.94 * 4).astype(int)
    )
    return frame.sort_values(
        ["size_bin", "axis_ratio_bin", "orientation_bin", "offset_bin", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


def _select_catalog(candidates: pd.DataFrame, cycle_count: int) -> pd.DataFrame:
    if cycle_count > len(candidates):
        raise ValueError(f"cycle_count={cycle_count} exceeds candidate_count={len(candidates)}")
    selected: list[int] = [
        int(candidates["semimajor_m"].idxmin()),
        int(candidates["semimajor_m"].idxmax()),
    ]
    for column in ("size_bin", "axis_ratio_bin", "orientation_bin", "offset_bin"):
        for _, group in candidates.groupby(column, sort=True):
            index = int(group.index[0])
            if index not in selected:
                selected.append(index)
    selected = selected[:cycle_count]
    feature = candidates.loc[:, ["semimajor_m", "axis_ratio", "normal_x", "normal_y", "normal_z", "offset_fraction"]].to_numpy(dtype=float)
    feature = (feature - feature.mean(axis=0)) / np.maximum(feature.std(axis=0), 1.0e-12)
    distance = np.min(
        np.linalg.norm(feature[:, None, :] - feature[np.asarray(selected)][None, :, :], axis=2),
        axis=1,
    )
    while len(selected) < cycle_count:
        distance[np.asarray(selected, dtype=int)] = -1.0
        chosen = int(np.argmax(distance))
        selected.append(chosen)
        distance = np.minimum(distance, np.linalg.norm(feature - feature[chosen], axis=1))
    catalog = candidates.loc[selected].copy().reset_index(drop=True)
    catalog.insert(0, "cycle_id", np.arange(len(catalog), dtype=np.int64))
    return catalog


def _cycle_points(row: Any, phase_count: int) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, int(phase_count), endpoint=False)
    center = np.asarray((row.center_x_m, row.center_y_m, row.center_z_m))
    axis0 = np.asarray((row.axis0_x, row.axis0_y, row.axis0_z))
    axis1 = np.asarray((row.axis1_x, row.axis1_y, row.axis1_z))
    return center + np.cos(phase)[:, None] * float(row.semimajor_m) * axis0 + np.sin(phase)[:, None] * float(row.semiminor_m) * axis1


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    shell_root = Path(args.shell_root).resolve()
    student_root = Path(args.student_root).resolve()
    output_root = Path(args.output).resolve()
    if output_root.exists():
        raise FileExistsError(f"ellipse output already exists: {output_root}")
    output_root.mkdir(parents=True)
    config = load_config(args.config, "formal")
    ellipse_config = config["ellipse"]
    shell_summary = json.loads((shell_root / "05_summary/gate.json").read_text())
    replay_manifest = json.loads((shell_root / "00_protocol/source_manifest.json").read_text())
    source_shell_root = Path(replay_manifest["source_run_root"])
    ellipsoid_path = source_shell_root / "01_shell_search/ellipsoid.json"
    student_gate_path = student_root / "04_summary/gate.json"
    model_lock_path = student_root / "02_model_lock/model_lock.json"
    student_gate = json.loads(student_gate_path.read_text())
    model_lock = json.loads(model_lock_path.read_text())
    git_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    protocol = output_root / "00_protocol"
    protocol.mkdir()
    source_dataset_hash = str(shell_summary["dataset_sha256"])
    protocol_gate = _gate(
        protocol / "gate.json",
        {
            "formal_shell_gate_pass": bool(shell_summary["gate_pass"]),
            "student_gate_pass": bool(student_gate["gate_pass"]),
            "model_lock_dataset_matches_shell": model_lock["source_dataset_sha256"] == source_dataset_hash,
            "ellipsoid_source_exists": ellipsoid_path.is_file(),
            "worktree_clean": git_status == "",
            "automatic_chart_classifier_disabled": True,
        },
        semantics="locked_models_and_analytical_ellipsoid_before_cycles",
        git_sha=git_sha,
        shell_dataset_sha256=source_dataset_hash,
        student_gate_sha256=sha256_file(student_gate_path),
        model_lock_sha256=sha256_file(model_lock_path),
        ellipsoid_sha256=sha256_file(ellipsoid_path),
    )
    if not protocol_gate["gate_pass"]:
        return {"gate_pass": False, "stopped_after": "protocol"}
    payload = json.loads(ellipsoid_path.read_text())
    spec = EllipsoidSpec(payload["center_m"], payload["semiaxes_m"], payload["rotation"])
    candidates = _section_candidates(spec, int(ellipse_config["candidate_count"]))
    catalog = _select_catalog(candidates, int(ellipse_config["cycle_count"]))
    catalog["role"] = np.where(
        catalog["cycle_id"] < int(ellipse_config["development_cycles"]),
        "development",
        "sealed",
    )
    catalog_stage = output_root / "01_catalog"
    catalog_stage.mkdir()
    _atomic_parquet(candidates, catalog_stage / "section_candidates.parquet")
    _atomic_parquet(catalog, catalog_stage / "selected_cycles.parquet")
    phase_count = int(ellipse_config["phase_count"])
    target = np.vstack([_cycle_points(row, phase_count) for row in catalog.itertuples(index=False)])
    cycle_ids = np.repeat(catalog["cycle_id"].to_numpy(dtype=int), phase_count)
    environment = load_environment(project_root, project_root / str(config["robot_config"]))
    import tensorflow as tf

    point_frames: list[pd.DataFrame] = []
    cycle_frames: list[pd.DataFrame] = []
    for raw_seed, lock in sorted(model_lock["models"].items(), key=lambda item: int(item[0])):
        seed = int(raw_seed)
        model_path = Path(lock["path"])
        if sha256_file(model_path) != lock["sha256"]:
            raise RuntimeError(f"model bytes changed after lock: seed {seed}")
        model = tf.keras.models.load_model(model_path, compile=False)
        features = np.column_stack((target.astype(np.float32), np.ones((len(target), 1), dtype=np.float32)))
        beta = np.asarray(model.predict(features, batch_size=2048, verbose=0), dtype=float)
        achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
        error_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
        semimajor = catalog.set_index("cycle_id").loc[cycle_ids, "semimajor_m"].to_numpy()
        relative = error_mm / (semimajor * 1000.0)
        point_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "cycle_id": cycle_ids,
                    "phase_idx": np.tile(np.arange(phase_count), len(catalog)),
                    "x_m": target[:, 0], "y_m": target[:, 1], "z_m": target[:, 2],
                    "student_fk_error_mm": error_mm,
                    "relative_semimajor_error": relative,
                }
            )
        )
        metrics = (
            pd.DataFrame({"cycle_id": cycle_ids, "relative": relative, "error_mm": error_mm})
            .groupby("cycle_id", sort=True)
            .agg(
                relative_p95=("relative", lambda value: np.percentile(value, 95)),
                relative_max=("relative", "max"),
                fk_p95_mm=("error_mm", lambda value: np.percentile(value, 95)),
                fk_max_mm=("error_mm", "max"),
            )
            .reset_index()
        )
        metrics.insert(0, "seed", seed)
        cycle_frames.append(metrics)
    points = pd.concat(point_frames, ignore_index=True)
    cycles = pd.concat(cycle_frames, ignore_index=True).merge(catalog, on="cycle_id", how="left", validate="many_to_one")
    cycles["cycle_pass"] = (
        cycles["relative_p95"] <= float(ellipse_config["relative_error_p95_max"])
    ) & (cycles["relative_max"] <= float(ellipse_config["relative_error_max"]))
    result_stage = output_root / "02_results"
    result_stage.mkdir()
    _atomic_parquet(points, result_stage / "point_metrics.parquet")
    _atomic_parquet(cycles, result_stage / "cycle_metrics.parquet")
    sealed_cycles = cycles.loc[cycles["role"] == "sealed"].copy()
    sealed_points = points.loc[
        points["cycle_id"].isin(catalog.loc[catalog["role"] == "sealed", "cycle_id"])
    ].copy()
    worst_cycle = sealed_cycles.groupby("cycle_id", sort=True)["cycle_pass"].all()
    bin_columns = ("size_bin", "axis_ratio_bin", "orientation_bin", "offset_bin")
    bin_pass = {
        column: sealed_cycles.groupby(column, sort=True)["cycle_pass"].mean().to_dict()
        for column in bin_columns
    }
    semimajor_ratio = float(catalog["semimajor_m"].max() / catalog["semimajor_m"].min())
    summary = output_root / "03_summary"
    summary.mkdir()
    gate = _gate(
        summary / "gate.json",
        {
            "registered_cycle_count": len(catalog) == int(ellipse_config["cycle_count"]),
            "registered_phase_count": phase_count == 720,
            "registered_development_cycle_count": int((catalog["role"] == "development").sum()) == int(ellipse_config["development_cycles"]),
            "registered_sealed_cycle_count": int((catalog["role"] == "sealed").sum()) == int(ellipse_config["sealed_cycles"]),
            "semimajor_range_threefold": semimajor_ratio >= 3.0,
            "axis_ratio_domain": bool(catalog["axis_ratio"].between(0.30, 1.0).all()),
            "orientation_coverage": catalog["orientation_bin"].nunique() >= 8,
            "offset_coverage": catalog["offset_bin"].nunique() >= 3,
            "overall_cycle_pass": float(worst_cycle.mean()) >= float(ellipse_config["overall_cycle_pass_min"]),
            "each_nonempty_bin_pass": all(
                float(value) >= float(ellipse_config["nonempty_bin_pass_min"])
                for groups in bin_pass.values()
                for value in groups.values()
            ),
            "sealed_point_p95": float(np.percentile(sealed_points["relative_semimajor_error"], 95)) <= float(ellipse_config["relative_error_p95_max"]),
            "sealed_point_max": float(sealed_points["relative_semimajor_error"].max()) <= float(ellipse_config["relative_error_max"]),
            "random_shell_sealed_gate_already_passed": bool(student_gate["gate_pass"]),
        },
        semantics="locked_student_analytical_plane_section_ellipse_generalization",
        cycle_count=len(catalog),
        point_count=len(points),
        semimajor_min_m=float(catalog["semimajor_m"].min()),
        semimajor_max_m=float(catalog["semimajor_m"].max()),
        semimajor_range_ratio=semimajor_ratio,
        axis_ratio_min=float(catalog["axis_ratio"].min()),
        axis_ratio_max=float(catalog["axis_ratio"].max()),
        orientation_bin_count=int(catalog["orientation_bin"].nunique()),
        offset_bin_count=int(catalog["offset_bin"].nunique()),
        cycle_pass_ratio=float(worst_cycle.mean()),
        sealed_cycle_count=int(len(worst_cycle)),
        sealed_point_count=int(len(sealed_points)),
        sealed_point_relative_p95=float(np.percentile(sealed_points["relative_semimajor_error"], 95)),
        sealed_point_relative_max=float(sealed_points["relative_semimajor_error"].max()),
        bin_pass_ratios=bin_pass,
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "output_root": str(output_root),
        "gate_pass": bool(gate["gate_pass"]),
        "stopped_after": None if gate["gate_pass"] else "ellipse_summary",
    }
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
