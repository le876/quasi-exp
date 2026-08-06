#!/usr/bin/env python3
"""Build read-only Q16 evidence from locked BACRA V13 artifacts.

This script does not upgrade any formal Gate.  It derives two diagnostic
comparisons needed by the Q16 GPT-5 Pro handoff:

1. compare the 200k V13 shell dataset with a 1,048,576-point Sobol forward-FK
   capability proxy inside the user-defined zero-pose x-backward-200-mm slab;
2. replay the three locked V13 Students on the historical final8 Teacher
   targets to expose out-of-domain behaviour.

All source bytes used for the comparisons are hashed into the output summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path("/mnt/ML_projects/quasi_exp")
V13_SOURCE_ROOT = PROJECT_ROOT / ".worktrees/bacra-v13-ellipsoidal-shell-atlas"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/gpt5pro_handoff_evidence/Q16"

SHELL_DATASET = (
    PROJECT_ROOT
    / "runs/bacra_v13_ellipsoidal_shell_atlas_formal_dense_replay1_20260801"
    / "04_dense_dataset/A3_shell_dataset.parquet"
)
SHELL_DATASET_GATE = SHELL_DATASET.parents[1] / "05_summary/gate.json"
CAPABILITY_POOL = (
    PROJECT_ROOT
    / "runs/branch_aware_canonical_region_atlas_v12_pilot/01_capability"
    / "capability_map.parquet"
)
SEARCH_CSV = (
    PROJECT_ROOT
    / "runs/bacra_v13_ellipsoidal_shell_atlas_formal_20260801"
    / "01_shell_search/ellipsoid_search.csv"
)
ELLIPSOID_JSON = SEARCH_CSV.with_name("ellipsoid.json")
RADIAL_GATE = SEARCH_CSV.parents[1] / "03_radial_labels/gate.json"
STUDENT_GATE = (
    PROJECT_ROOT
    / "runs/bacra_v13_formal_shell_student_20260801/04_summary/gate.json"
)
MODEL_LOCK = STUDENT_GATE.parents[1] / "02_model_lock/model_lock.json"
LEGACY_REFERENCE = (
    PROJECT_ROOT
    / "runs/bacra_v12_13_exploratory_expansion/06_final/teacher_reference.parquet"
)
LEGACY_CATALOG = LEGACY_REFERENCE.with_name("new_family_catalog.csv")

EXPECTED_SHELL_SHA256 = "528c8a6dd902c6d665a1d28378c30eb19c6969f314a44399c18872e1a1a6f1a8"
EXPECTED_CAPABILITY_SHA256 = "ce71915dd337837b08d6e6959ba81eb7b8122470d0d439cee8677c43b91d5d74"
EXPECTED_MODEL_LOCK_SHA256 = "50506e7d48ed690ded0bc07d620457ca17eb49fb0acb7ffa8020d5655c4b73fb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(V13_SOURCE_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def quantiles(values: Iterable[float], points: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        f"p{int(round(point * 100)):02d}": float(np.quantile(array, point))
        for point in points
    }


def xyz_profile(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ("x_m", "y_m", "z_m"):
        values = frame[column].to_numpy(dtype=float)
        result[column] = {
            "min": float(values.min()),
            **quantiles(values, (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)),
            "max": float(values.max()),
            "span_mm": float((values.max() - values.min()) * 1000.0),
        }
    return result


def unique_voxels(xyz_m: np.ndarray, size_m: float) -> np.ndarray:
    return np.unique(np.floor(np.asarray(xyz_m, dtype=float) / float(size_m)).astype(np.int64), axis=0)


def voxel_overlap(capability_xyz: np.ndarray, shell_xyz: np.ndarray, size_mm: float) -> dict[str, Any]:
    size_m = float(size_mm) / 1000.0
    capability = unique_voxels(capability_xyz, size_m)
    shell = unique_voxels(shell_xyz, size_m)
    capability_set = {tuple(row) for row in capability}
    shell_set = {tuple(row) for row in shell}
    overlap = len(capability_set & shell_set)
    return {
        "voxel_size_mm": float(size_mm),
        "capability_target_voxel_count": int(len(capability)),
        "shell_voxel_count_all_xyz": int(len(shell)),
        "overlap_voxel_count": int(overlap),
        "capability_target_voxel_coverage_ratio": float(overlap / len(capability)),
    }


def build_coverage_profile(output: Path, environment: Any) -> tuple[dict[str, Any], np.ndarray]:
    shell_gate = json.loads(SHELL_DATASET_GATE.read_text(encoding="utf-8"))
    radial_gate = json.loads(RADIAL_GATE.read_text(encoding="utf-8"))
    shell = pd.read_parquet(SHELL_DATASET)
    capability = pd.read_parquet(
        CAPABILITY_POOL,
        columns=["x_m", "y_m", "z_m", "minimum_margin_deg", "capability_tier"],
    )

    zero_beta = np.zeros((1, 6), dtype=float)
    zero_xyz = np.asarray(environment.fk(zero_beta), dtype=float).reshape(3)
    target_x_min = float(zero_xyz[0] - 0.200)
    target_x_max = float(zero_xyz[0])
    target_mask = capability["x_m"].between(target_x_min, target_x_max, inclusive="both")
    target_capability = capability.loc[target_mask].copy()
    target_capability_xyz = target_capability[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    shell_xyz = shell[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    shell_in_slab = shell["x_m"].between(target_x_min, target_x_max, inclusive="both")

    voxel_audits = [
        voxel_overlap(target_capability_xyz, shell_xyz, size_mm)
        for size_mm in (5.0, 10.0, 20.0)
    ]

    capability_voxels_10mm = unique_voxels(target_capability_xyz, 0.010)
    capability_voxel_centers = (capability_voxels_10mm.astype(float) + 0.5) * 0.010
    nearest_mm = cKDTree(shell_xyz).query(capability_voxel_centers, k=1, workers=-1)[0] * 1000.0

    x_edges = np.linspace(target_x_min, target_x_max, 21)
    capability_x_counts, _ = np.histogram(target_capability["x_m"], bins=x_edges)
    shell_x_counts, _ = np.histogram(shell.loc[shell_in_slab, "x_m"], bins=x_edges)
    x_bins = [
        {
            "x_lower_m": float(x_edges[index]),
            "x_upper_m": float(x_edges[index + 1]),
            "capability_proxy_rows": int(capability_x_counts[index]),
            "shell_rows": int(shell_x_counts[index]),
        }
        for index in range(len(x_edges) - 1)
    ]

    rho = shell["rho_m"].to_numpy(dtype=float)
    profile = {
        "schema_version": 1,
        "evidence_level": "diagnostic_only",
        "claim_scope": "empirical_workspace_coverage_audit_not_a_formal_reachability_proof",
        "definition": {
            "zero_pose_beta_rad": [0.0] * 6,
            "zero_pose_endpoint_m": zero_xyz.tolist(),
            "user_target_x_slab_m": [target_x_min, target_x_max],
            "user_target_description": "all robot-workspace positions whose x lies within 200 mm backward from the zero-pose endpoint",
            "capability_proxy_semantics": "1,048,576 Sobol beta samples forward-mapped under registered mechanical beta bounds; not an exhaustive proof of Reach(FK)",
        },
        "source_hashes": {
            "shell_dataset_sha256": sha256_file(SHELL_DATASET),
            "shell_dataset_gate_sha256": sha256_file(SHELL_DATASET_GATE),
            "capability_pool_sha256": sha256_file(CAPABILITY_POOL),
            "ellipsoid_json_sha256": sha256_file(ELLIPSOID_JSON),
            "radial_gate_sha256": sha256_file(RADIAL_GATE),
        },
        "current_shell_dataset": {
            "row_count": int(len(shell)),
            "column_count": int(len(shell.columns)),
            "chart_counts": {str(key): int(value) for key, value in shell["chart_id"].value_counts().sort_index().items()},
            "split_counts": {str(key): int(value) for key, value in shell["split"].value_counts().sort_index().items()},
            "quality_counts": {str(key): int(value) for key, value in shell["quality_class"].value_counts().sort_index().items()},
            "xyz_profile": xyz_profile(shell),
            "rho_m": {
                "min": float(rho.min()),
                **quantiles(rho, (0.01, 0.10, 0.50, 0.90, 0.99)),
                "max": float(rho.max()),
                "observed_total_span_mm": float((rho.max() - rho.min()) * 1000.0),
            },
            "rows_inside_user_target_x_slab": int(shell_in_slab.sum()),
            "fraction_inside_user_target_x_slab": float(shell_in_slab.mean()),
            "formal_dense_gate": shell_gate,
            "formal_radial_gate": radial_gate,
            "thickness_interpretation_correction": "radial levels and rho span are approximately -20..+20 mm. The Gate field is generated from symmetric_half_thickness(), so a reported 20 mm is half-thickness/reach, not 20 mm total thickness.",
        },
        "capability_proxy_in_target_x_slab": {
            "source_row_count": int(len(capability)),
            "target_row_count": int(len(target_capability)),
            "target_row_fraction": float(len(target_capability) / len(capability)),
            "tier_counts": {str(key): int(value) for key, value in target_capability["capability_tier"].value_counts().sort_index().items()},
            "xyz_profile": xyz_profile(target_capability),
            "voxel_overlap_audits": voxel_audits,
            "nearest_current_shell_distance_from_10mm_capability_voxel_centers_mm": {
                "min": float(nearest_mm.min()),
                **quantiles(nearest_mm, (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)),
                "max": float(nearest_mm.max()),
            },
            "x_bins_10mm": x_bins,
        },
        "interpretation_boundary": [
            "Voxel overlap measures empirical support against the Sobol capability proxy, not continuous-volume coverage of the exact reachable set.",
            "Coordinate bounds and projections do not imply that the interior is safely canonical-labelable or single-valued.",
            "The full target domain must still be partitioned into safe single-valued, multichart/multivalued, unsafe, unresolved, and unreachable subsets.",
        ],
    }
    if profile["source_hashes"]["shell_dataset_sha256"] != EXPECTED_SHELL_SHA256:
        raise RuntimeError("locked shell dataset bytes changed")
    if profile["source_hashes"]["capability_pool_sha256"] != EXPECTED_CAPABILITY_SHA256:
        raise RuntimeError("locked capability pool bytes changed")
    atomic_json(output / "q16_workspace_coverage_profile.json", profile)
    return profile, nearest_mm


def build_search_audit(output: Path) -> dict[str, Any]:
    frame = pd.read_csv(SEARCH_CSV)
    threshold_mask = (
        (frame["center_support_ratio"] >= 0.90)
        & (frame["minus_support_ratio"] >= 0.80)
        & (frame["plus_support_ratio"] >= 0.80)
        & (frame["connected_surface_ratio"] >= 0.80)
        & frame["envelope_pass"].astype(bool)
    )
    passed = frame.loc[threshold_mask].copy()

    bins = [0.0, 0.15, 0.30, 0.45, np.inf]
    labels = ["[0,150)", "[150,300)", "[300,450)", "[450,inf)"]
    groups = pd.cut(frame["semiaxis_a_m"], bins=bins, labels=labels, right=False)
    by_scale: list[dict[str, Any]] = []
    for label in labels:
        subset = frame.loc[groups == label]
        subset_pass = threshold_mask.loc[subset.index]
        by_scale.append(
            {
                "semimajor_bin_mm": label,
                "candidate_count": int(len(subset)),
                "formal_prefilter_pass_count": int(subset_pass.sum()),
                "max_center_support_ratio": float(subset["center_support_ratio"].max()) if len(subset) else None,
                "max_minus_support_ratio": float(subset["minus_support_ratio"].max()) if len(subset) else None,
                "max_plus_support_ratio": float(subset["plus_support_ratio"].max()) if len(subset) else None,
                "max_connected_surface_ratio": float(subset["connected_surface_ratio"].max()) if len(subset) else None,
            }
        )

    selected = frame.iloc[0]
    audit = {
        "schema_version": 1,
        "evidence_level": "diagnostic_read_of_formal_search_table",
        "source_sha256": sha256_file(SEARCH_CSV),
        "candidate_count": int(len(frame)),
        "formal_prefilter_definition": {
            "center_support_ratio_min": 0.90,
            "minus_support_ratio_min": 0.80,
            "plus_support_ratio_min": 0.80,
            "connected_surface_ratio_min": 0.80,
            "envelope_pass_required": True,
        },
        "formal_prefilter_pass_count": int(threshold_mask.sum()),
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_semiaxes_mm": [
            float(selected["semiaxis_a_m"] * 1000.0),
            float(selected["semiaxis_b_m"] * 1000.0),
            float(selected["semiaxis_c_m"] * 1000.0),
        ],
        "largest_semimajor_among_prefilter_pass_mm": float(passed["semiaxis_a_m"].max() * 1000.0),
        "pass_count_semimajor_at_least_300mm": int((threshold_mask & (frame["semiaxis_a_m"] >= 0.300)).sum()),
        "pass_count_semimajor_at_least_450mm": int((threshold_mask & (frame["semiaxis_a_m"] >= 0.450)).sum()),
        "by_semimajor_scale": by_scale,
        "boundary": "This table audits the frozen candidate search only. It does not prove that a different region representation, chart construction, or sampling strategy cannot cover a larger domain.",
    }
    atomic_json(output / "q16_shell_search_scale_audit.json", audit)
    return audit


def build_legacy_ood_replay(output: Path, environment: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    import tensorflow as tf

    shell = pd.read_parquet(SHELL_DATASET, columns=["x_m", "y_m", "z_m"])
    reference = pd.read_parquet(LEGACY_REFERENCE)
    catalog = pd.read_csv(LEGACY_CATALOG)
    model_lock = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))
    if sha256_file(MODEL_LOCK) != EXPECTED_MODEL_LOCK_SHA256:
        raise RuntimeError("locked V13 model lock bytes changed")

    target = reference[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    features = np.column_stack(
        (target.astype(np.float32), np.ones((len(target), 1), dtype=np.float32))
    )
    semimajor_by_family = catalog.set_index("family_id")["major_semiaxis_m"].to_dict()
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    rows: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []

    for raw_seed, lock in sorted(model_lock["models"].items(), key=lambda item: int(item[0])):
        seed = int(raw_seed)
        model_path = Path(lock["path"])
        actual_sha = sha256_file(model_path)
        if actual_sha != lock["sha256"]:
            raise RuntimeError(f"locked model bytes changed for seed {seed}")
        model = tf.keras.models.load_model(model_path, compile=False)
        beta = np.asarray(model.predict(features, batch_size=2048, verbose=0), dtype=float)
        achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
        error_mm = np.linalg.norm(achieved - target, axis=1) * 1000.0
        in_bounds = np.all(beta >= bounds[:, 0] - 1.0e-12, axis=1) & np.all(
            beta <= bounds[:, 1] + 1.0e-12, axis=1
        )
        overall.append(
            {
                "seed": seed,
                "row_count": int(len(error_mm)),
                "fk_p50_mm": float(np.percentile(error_mm, 50)),
                "fk_p95_mm": float(np.percentile(error_mm, 95)),
                "fk_max_mm": float(error_mm.max()),
                "all_predictions_in_bounds": bool(in_bounds.all()),
            }
        )
        metric_frame = pd.DataFrame(
            {
                "family_id": reference["family_id"].astype(str),
                "error_mm": error_mm,
                "in_bounds": in_bounds,
            }
        )
        for family_id, subset in metric_frame.groupby("family_id", sort=True):
            semimajor_m = float(semimajor_by_family[family_id])
            p95 = float(np.percentile(subset["error_mm"], 95))
            maximum = float(subset["error_mm"].max())
            rows.append(
                {
                    "seed": seed,
                    "family_id": family_id,
                    "row_count": int(len(subset)),
                    "major_semiaxis_m": semimajor_m,
                    "fk_p50_mm": float(np.percentile(subset["error_mm"], 50)),
                    "fk_p95_mm": p95,
                    "fk_max_mm": maximum,
                    "fk_p95_relative": p95 / (semimajor_m * 1000.0),
                    "fk_max_relative": maximum / (semimajor_m * 1000.0),
                    "all_predictions_in_bounds": bool(subset["in_bounds"].all()),
                    "historical_relative_gate_pass": bool(
                        p95 / (semimajor_m * 1000.0) <= 0.01
                        and maximum / (semimajor_m * 1000.0) <= 0.02
                    ),
                }
            )
        tf.keras.backend.clear_session()

    metrics = pd.DataFrame(rows).sort_values(["seed", "family_id"], kind="stable")
    atomic_csv(metrics, output / "q16_v13_on_legacy_final8_ood_replay.csv")

    shell_tree = cKDTree(shell[["x_m", "y_m", "z_m"]].to_numpy(dtype=float))
    target_distance_mm = shell_tree.query(target, k=1, workers=-1)[0] * 1000.0
    distance_frame = pd.DataFrame(
        {"family_id": reference["family_id"].astype(str), "nearest_shell_distance_mm": target_distance_mm}
    )
    distance_rows: list[dict[str, Any]] = []
    for family_id, subset in distance_frame.groupby("family_id", sort=True):
        distance_rows.append(
            {
                "family_id": family_id,
                "row_count": int(len(subset)),
                "nearest_shell_min_mm": float(subset["nearest_shell_distance_mm"].min()),
                "nearest_shell_p50_mm": float(np.percentile(subset["nearest_shell_distance_mm"], 50)),
                "nearest_shell_p95_mm": float(np.percentile(subset["nearest_shell_distance_mm"], 95)),
                "nearest_shell_max_mm": float(subset["nearest_shell_distance_mm"].max()),
            }
        )
    distance_metrics = pd.DataFrame(distance_rows)
    atomic_csv(distance_metrics, output / "q16_legacy_final8_geometry_distance.csv")

    summary = {
        "schema_version": 1,
        "evidence_level": "post_hoc_diagnostic_only",
        "claim_scope": "locked_v13_student_ood_replay_without_retraining_or_gate_upgrade",
        "source_hashes": {
            "legacy_teacher_reference_sha256": sha256_file(LEGACY_REFERENCE),
            "legacy_family_catalog_sha256": sha256_file(LEGACY_CATALOG),
            "v13_model_lock_sha256": sha256_file(MODEL_LOCK),
            "v13_shell_dataset_sha256": sha256_file(SHELL_DATASET),
        },
        "legacy_reference_rows": int(len(reference)),
        "legacy_family_count": int(reference["family_id"].nunique()),
        "legacy_major_semiaxis_range_mm": [
            float(catalog["major_semiaxis_m"].min() * 1000.0),
            float(catalog["major_semiaxis_m"].max() * 1000.0),
        ],
        "overall_by_seed": overall,
        "historical_family_seed_pass_count": int(metrics["historical_relative_gate_pass"].sum()),
        "historical_family_seed_total": int(len(metrics)),
        "all_seed_family_pass_count": int(
            metrics.groupby("family_id", sort=True)["historical_relative_gate_pass"].all().sum()
        ),
        "all_seed_family_total": int(metrics["family_id"].nunique()),
        "legacy_target_to_current_shell_distance_mm": {
            "min": float(target_distance_mm.min()),
            **quantiles(target_distance_mm, (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)),
            "max": float(target_distance_mm.max()),
        },
        "interpretation": "The locked V13 models remain mechanically bounded but fail the historical relative tracking gate far outside the local shell. This supports domain omission/OOD extrapolation as a concrete counterexample; it is not a preregistered formal V13 retention result and does not by itself prove a unique causal mechanism.",
    }
    atomic_json(output / "q16_legacy_ood_summary.json", summary)
    return summary, metrics


def plot_workspace_coverage(output: Path, profile: dict[str, Any], nearest_mm: np.ndarray) -> None:
    shell = pd.read_parquet(SHELL_DATASET, columns=["x_m", "y_m", "z_m"])
    capability = pd.read_parquet(CAPABILITY_POOL, columns=["x_m", "y_m", "z_m"])
    x_min, x_max = profile["definition"]["user_target_x_slab_m"]
    capability = capability.loc[capability["x_m"].between(x_min, x_max, inclusive="both")]
    rng = np.random.default_rng(20260804)
    capability_draw = capability.iloc[
        np.sort(rng.choice(len(capability), size=min(60000, len(capability)), replace=False))
    ]
    shell_draw = shell.iloc[np.sort(rng.choice(len(shell), size=min(30000, len(shell)), replace=False))]

    figure, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    projections = (("x_m", "y_m"), ("x_m", "z_m"), ("y_m", "z_m"))
    for axis, (horizontal, vertical) in zip(axes.flat[:3], projections):
        axis.scatter(
            capability_draw[horizontal], capability_draw[vertical],
            s=1, alpha=0.035, color="#737373", rasterized=True,
            label="Sobol FK capability proxy in target x-slab",
        )
        axis.scatter(
            shell_draw[horizontal], shell_draw[vertical],
            s=2, alpha=0.12, color="#0072B2", rasterized=True,
            label="current V13 shell dataset",
        )
        axis.set_xlabel(horizontal.replace("_m", " (m)"))
        axis.set_ylabel(vertical.replace("_m", " (m)"))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        if horizontal == "x_m":
            axis.axvline(x_min, color="#D55E00", linestyle="--", linewidth=1)
            axis.axvline(x_max, color="#D55E00", linestyle="--", linewidth=1)
        if (horizontal, vertical) == ("x_m", "y_m"):
            zero = profile["definition"]["zero_pose_endpoint_m"]
            axis.scatter([zero[0]], [zero[1]], marker="*", s=140, color="#CC3311", label="zero-pose endpoint")
            axis.legend(loc="best", fontsize=8)

    axis = axes.flat[3]
    sorted_distance = np.sort(nearest_mm)
    cumulative = np.arange(1, len(sorted_distance) + 1) / len(sorted_distance)
    axis.plot(sorted_distance, cumulative, color="#0072B2", linewidth=2)
    p50 = float(np.percentile(nearest_mm, 50))
    p95 = float(np.percentile(nearest_mm, 95))
    axis.axvline(p50, color="#E69F00", linestyle="--", label=f"P50={p50:.1f} mm")
    axis.axvline(p95, color="#D55E00", linestyle="--", label=f"P95={p95:.1f} mm")
    axis.set_xlabel("nearest distance from target 10 mm capability voxel center to V13 shell (mm)")
    axis.set_ylabel("empirical CDF")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.suptitle("Q16 diagnostic: local 200k shell versus the requested 200 mm x-slab", fontsize=15)
    temporary = output / "q16_workspace_coverage.png.tmp.png"
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, output / "q16_workspace_coverage.png")


def plot_local_vs_ood(output: Path, legacy_summary: dict[str, Any]) -> None:
    student_gate = json.loads(STUDENT_GATE.read_text(encoding="utf-8"))
    local = {
        int(row["seed"]): float(row["sealed_fk_p95_mm"])
        for row in student_gate["sealed_reports"]
    }
    legacy = {
        int(row["seed"]): float(row["fk_p95_mm"])
        for row in legacy_summary["overall_by_seed"]
    }
    seeds = sorted(local)
    x = np.arange(len(seeds))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    local_bars = axis.bar(x - width / 2, [local[seed] for seed in seeds], width, label="V13 sealed local shell", color="#009E73")
    legacy_bars = axis.bar(x + width / 2, [legacy[seed] for seed in seeds], width, label="legacy final8 OOD replay", color="#D55E00")
    axis.set_yscale("log")
    axis.set_ylabel("FK P95 error (mm, log scale)")
    axis.set_xticks(x, [str(seed) for seed in seeds])
    axis.set_xlabel("locked Student seed")
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    for bars in (local_bars, legacy_bars):
        for bar in bars:
            value = float(bar.get_height())
            axis.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    axis.set_title(
        "Q16 diagnostic: local learnability does not imply workspace-scale generalization",
        pad=72,
    )
    temporary = output / "q16_local_vs_legacy_ood.png.tmp.png"
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, output / "q16_local_vs_legacy_ood.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if str(V13_SOURCE_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(V13_SOURCE_ROOT / "src"))
    if str(V13_SOURCE_ROOT / "scripts/analysis") not in sys.path:
        sys.path.insert(0, str(V13_SOURCE_ROOT / "scripts/analysis"))
    from run_trajectory_canonical_teacher_v10 import load_environment

    environment = load_environment(
        V13_SOURCE_ROOT,
        V13_SOURCE_ROOT / "configs/robot_rods_only_standard_100k.yaml",
    )
    coverage, nearest_mm = build_coverage_profile(output, environment)
    search = build_search_audit(output)
    legacy, _metrics = build_legacy_ood_replay(output, environment)
    plot_workspace_coverage(output, coverage, nearest_mm)
    plot_local_vs_ood(output, legacy)

    manifest = {
        "schema_version": 1,
        "evidence_level": "diagnostic_only",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "interpreter": sys.executable,
        "v13_source_root": str(V13_SOURCE_ROOT),
        "v13_source_head": git_value("rev-parse", "HEAD"),
        "v13_source_status_porcelain": git_value("status", "--porcelain"),
        "formal_artifact_commits": {
            "shell_source": "7e495db80096f89fe68971fe6c4e15bab0d2b92e",
            "dense_replay": "b9df6917171c636eefa43b11912941fcb38522d8",
            "student": "77f23d598a81e24b5bad88610aad26893edc9ec1",
            "ellipse_retry1": "301118d494325442d8d0832da020ab9ee5d59d2d",
            "three_dimensional_paths": "413d322c75bd6f97851b0352289e9e49156b367a",
        },
        "key_results": {
            "zero_pose_endpoint_m": coverage["definition"]["zero_pose_endpoint_m"],
            "target_x_slab_m": coverage["definition"]["user_target_x_slab_m"],
            "target_capability_proxy_voxel_coverage_10mm": coverage["capability_proxy_in_target_x_slab"]["voxel_overlap_audits"][1]["capability_target_voxel_coverage_ratio"],
            "target_capability_proxy_nearest_shell_p50_mm": coverage["capability_proxy_in_target_x_slab"]["nearest_current_shell_distance_from_10mm_capability_voxel_centers_mm"]["p50"],
            "target_capability_proxy_nearest_shell_p95_mm": coverage["capability_proxy_in_target_x_slab"]["nearest_current_shell_distance_from_10mm_capability_voxel_centers_mm"]["p95"],
            "formal_search_prefilter_pass_count": search["formal_prefilter_pass_count"],
            "largest_semimajor_among_prefilter_pass_mm": search["largest_semimajor_among_prefilter_pass_mm"],
            "legacy_family_seed_pass_count": legacy["historical_family_seed_pass_count"],
            "legacy_family_seed_total": legacy["historical_family_seed_total"],
        },
        "outputs": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "q16_evidence_manifest.json":
            manifest["outputs"][path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    atomic_json(output / "q16_evidence_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
