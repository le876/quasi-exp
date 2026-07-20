#!/usr/bin/env python3
"""Run the gate-driven V11 generalized 0.5 m ellipse-region experiment.

Every stage has a single boolean gate backed by raw boolean checks.  A failed
gate stops the chain; it is evidence, never a request to delete rows or relax a
threshold.  Stage directories are resumable, while ``COMPLETED`` makes a final
output immutable and hash-verifiable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import yaml

from quasi_exp.teacher.canonical import (
    CanonicalTeacher,
    TeacherPolicy,
    TeacherVariant,
    TrajectorySpec,
)
from quasi_exp.teacher.dataset import evaluate_centerline_gate, trajectory_frame
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.large_scale import ReachabilityAtlas
from quasi_exp.teacher.region import (
    CanonicalRegionTeacher,
    EllipseFamilySpec,
    TeacherSurface,
    TubeCrossSection,
)
from quasi_exp.teacher.region_artifacts import finalize_experiment
from quasi_exp.teacher.region_audit import (
    audit_cartesian_coverage,
    audit_cross_family_conflicts,
    evaluate_teacher_surface_gate,
    select_student_representation,
)
from quasi_exp.teacher.region_protocol import (
    FamilyCatalog,
    generate_family_catalog,
    nested_family_sample_indices,
    promote_complete_families,
    require_boolean_gate_tree,
)

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


STAGES = (
    "protocol",
    "anchor",
    "tube",
    "core",
    "pilot",
    "representation",
    "formal",
    "train",
    "evaluate",
    "verify",
)
STAGE_DIRS = {
    "protocol": "00_protocol",
    "anchor": "01_anchor",
    "tube": "02_tube",
    "core": "03_core",
    "pilot": "04_pilot",
    "representation": "05_representation",
    "formal": "06_formal",
    "train": "07_train",
    "evaluate": "08_evaluate",
}
BETA_COLUMNS = tuple(f"teacher_beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("target_x_m", "target_y_m", "target_z_m")


def project_root_from(path: Path) -> Path:
    resolved = path.resolve()
    if ".worktrees" in resolved.parts:
        index = resolved.parts.index(".worktrees")
        return Path(*resolved.parts[:index])
    return resolved


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def load_protocol_config(path: str | Path, *, preset: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V11 protocol YAML must contain a mapping")
    normalized = str(preset).lower()
    if normalized not in {"smoke", "formal"}:
        raise ValueError("preset must be smoke or formal")
    if normalized == "smoke":
        payload = _deep_merge(payload, payload.get("smoke", {}))
    payload.pop("smoke", None)
    payload["preset"] = normalized
    payload["config_path"] = str(config_path)
    return payload


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_valid_gate(path: str | Path) -> dict[str, Any] | None:
    gate_path = Path(path)
    if not gate_path.is_file():
        return None
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
        require_boolean_gate_tree(payload)
        for relative, expected_sha in payload.get("artifact_sha256", {}).items():
            artifact = gate_path.parent / str(relative)
            if not artifact.is_file() or sha256_file(artifact) != str(expected_sha):
                return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload


def _write_gate(path: Path, *, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    normalized = {str(key): bool(value) for key, value in checks.items()}
    artifact_hashes = {
        artifact.relative_to(path.parent).as_posix(): sha256_file(artifact)
        for artifact in sorted(path.parent.rglob("*"))
        if artifact.is_file() and artifact.resolve() != path.resolve()
    }
    report = {
        **evidence,
        "artifact_sha256": artifact_hashes,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
    }
    atomic_write_json(path, report)
    return report


def _baseline_family(project_root: Path, config: Mapping[str, Any]) -> EllipseFamilySpec:
    pose_path = (
        project_root
        / str(config["v10_evidence_root"])
        / "a0p500m/pose_report.json"
    )
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    return EllipseFamilySpec(
        family_id="anchor_v10_a0p500m",
        center_m=np.asarray(pose["center_m"], dtype=float),
        major_direction=np.asarray(pose["major_direction"], dtype=float),
        minor_direction=np.asarray(pose["minor_direction"], dtype=float),
        major_semiaxis_m=float(pose["major_semiaxis_m"]),
        minor_semiaxis_m=float(pose["minor_semiaxis_m"]),
        metadata={"source": str(pose_path)},
    )


def _catalog_from_csv(path: Path, *, seed: int) -> FamilyCatalog:
    frame = pd.read_csv(path)
    frame["is_primary"] = frame["is_primary"].map(
        lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
    )
    return FamilyCatalog(frame, seed=int(seed))


def run_protocol_stage(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    output: Path,
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["protocol"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_family(project_root, config)
    catalog = generate_family_catalog(
        baseline, seed=int(config["seeds"]["family"])
    )
    catalog_path = stage / "family_catalog.csv"
    catalog.frame.to_csv(catalog_path, index=False)
    split = catalog.frame[catalog.frame["is_primary"]].copy()
    split = split[["family_id", "role", "catalog_order"]].sort_values(
        "catalog_order", kind="stable"
    )
    split.to_csv(stage / "split_manifest.csv", index=False)
    resolved_config = copy.deepcopy(dict(config))
    resolved_config.pop("config_path", None)
    (stage / "protocol_v11.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    input_files = {
        "protocol_config": sha256_file(str(config["config_path"])),
        "robot_config": sha256_file(project_root / str(config["robot_config"])),
        "v10_pose": sha256_file(
            project_root
            / str(config["v10_evidence_root"])
            / "a0p500m/pose_report.json"
        ),
        "v10_centerline": sha256_file(
            project_root
            / str(config["v10_evidence_root"])
            / "a0p500m/formal/T3/centerline.parquet"
        ),
    }
    manifest_payload = {
        "protocol_id": str(config["protocol_id"]),
        "source_fixed_point": str(config["source_fixed_point"]),
        "preset": str(config["preset"]),
        "catalog_fingerprint": catalog.fingerprint,
        "catalog_seal_token_sha256": hashlib.sha256(
            catalog.seal_token.encode("utf-8")
        ).hexdigest(),
        "input_sha256": input_files,
        "worker_code_sha256": {
            str(Path(__file__).resolve().relative_to(source_root)): sha256_file(__file__),
            "src/quasi_exp/teacher/region.py": sha256_file(
                source_root / "src/quasi_exp/teacher/region.py"
            ),
            "src/quasi_exp/teacher/region_protocol.py": sha256_file(
                source_root / "src/quasi_exp/teacher/region_protocol.py"
            ),
        },
        "runtime": runtime_fingerprint(),
    }
    manifest_payload["protocol_sha256"] = _canonical_sha(manifest_payload)
    atomic_write_json(stage / "artifact_manifest.json", manifest_payload)
    role_counts = split.groupby("role").size().to_dict()
    checks = {
        "catalog_has_72_unique_families": bool(
            len(catalog.frame) == 72 and catalog.frame["family_id"].is_unique
        ),
        "primary_role_quota_14_5_5": bool(
            role_counts == {"train": 14, "validation": 5, "virgin_test": 5}
        ),
        "sealed_test_disjoint": bool(
            set(catalog.authorized_family_ids("model_selection")).isdisjoint(
                set(catalog.primary_ids("virgin_test"))
            )
        ),
        "manifest_hash_present": bool(
            len(manifest_payload["protocol_sha256"]) == 64
        ),
        "source_fixed_point_frozen": bool(
            str(config["source_fixed_point"])
            == "7a73f2acbdc8c90f7872bb2b20691d5a65abe434"
        ),
    }
    return _write_gate(
        gate_path,
        checks=checks,
        protocol_id=str(config["protocol_id"]),
        catalog_fingerprint=catalog.fingerprint,
        split_counts={key: int(value) for key, value in role_counts.items()},
    )


def _teacher_policy(config: Mapping[str, Any], *, seed: int | None = None) -> TeacherPolicy:
    values = dict(config["teacher_policy"])
    values["variant"] = TeacherVariant(str(values.get("variant", "T3")))
    values["solver_seed"] = int(
        config["seeds"]["solver"] if seed is None else seed
    )
    return TeacherPolicy(**values)


def _reachability_atlas(project_root: Path, config: Mapping[str, Any]) -> ReachabilityAtlas:
    path = project_root / str(config["v10_evidence_root"]) / "reachability_atlas.parquet"
    frame = pd.read_parquet(path)
    return ReachabilityAtlas(
        xyz_m=frame[["x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        beta_rad=frame[[f"beta{index}_rad" for index in range(1, 7)]].to_numpy(dtype=float),
    )


def _v10_centerline(project_root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    path = (
        project_root
        / str(config["v10_evidence_root"])
        / "a0p500m/formal/T3/centerline.parquet"
    )
    return pd.read_parquet(path).sort_values("phase_idx", kind="stable").reset_index(drop=True)


def _subsample_cyclic(values: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(values)
    requested = int(count)
    if len(source) == requested:
        return source.copy()
    positions = np.floor(np.arange(requested) * len(source) / requested).astype(int)
    return source[positions].copy()


def _family_payload(family: EllipseFamilySpec) -> dict[str, Any]:
    return {
        "family_id": family.family_id,
        "center_m": family.center_m.tolist(),
        "major_direction": family.major_direction.tolist(),
        "minor_direction": family.minor_direction.tolist(),
        "major_semiaxis_m": family.major_semiaxis_m,
        "minor_semiaxis_m": family.minor_semiaxis_m,
        "fingerprint": family.fingerprint,
    }


def _family_from_payload(payload: Mapping[str, Any]) -> EllipseFamilySpec:
    return EllipseFamilySpec(
        family_id=str(payload["family_id"]),
        center_m=np.asarray(payload["center_m"], dtype=float),
        major_direction=np.asarray(payload["major_direction"], dtype=float),
        minor_direction=np.asarray(payload["minor_direction"], dtype=float),
        major_semiaxis_m=float(payload["major_semiaxis_m"]),
        minor_semiaxis_m=float(payload["minor_semiaxis_m"]),
    )


def _anchor_candidates(
    baseline: EllipseFamilySpec, config: Mapping[str, Any]
) -> list[tuple[str, EllipseFamilySpec]]:
    anchor = config["anchor"]
    rng = qmc.Sobol(d=6, scramble=True, seed=int(config["seeds"]["anchor"]))
    total = int(anchor["center_proposals"]) + int(anchor["center_plane_proposals"])
    exponent = max(0, int(math.ceil(math.log2(max(total, 1)))))
    unit = rng.random_base2(exponent)[:total]
    q1_range = np.asarray(anchor["center_q1_mm"], dtype=float) / 1000.0
    q2_range = np.asarray(anchor["center_q2_mm"], dtype=float) / 1000.0
    normal_range = np.asarray(anchor["center_normal_mm"], dtype=float) / 1000.0
    tilt_range = np.asarray(anchor["tilt_deg"], dtype=float)
    axes = np.column_stack(
        [baseline.major_direction, baseline.minor_direction, baseline.plane_normal]
    )
    candidates = [("A0_000", baseline)]
    for index, row in enumerate(unit):
        center = (
            baseline.center_m
            + np.interp(row[0], [0.0, 1.0], q1_range) * baseline.major_direction
            + np.interp(row[1], [0.0, 1.0], q2_range) * baseline.minor_direction
            + np.interp(row[2], [0.0, 1.0], normal_range) * baseline.plane_normal
        )
        is_center_only = index < int(anchor["center_proposals"])
        if is_center_only:
            major = baseline.major_direction
            minor = baseline.minor_direction
            prefix = "A1"
        else:
            angles = np.interp(row[3:6], [0.0, 1.0], tilt_range)
            rotation = Rotation.from_rotvec(axes @ np.deg2rad(angles))
            major = rotation.apply(baseline.major_direction)
            minor = rotation.apply(baseline.minor_direction)
            prefix = "A2"
        candidates.append(
            (
                f"{prefix}_{index:03d}",
                EllipseFamilySpec(
                    family_id=f"anchor_{prefix}_{index:03d}",
                    center_m=center,
                    major_direction=major,
                    minor_direction=minor,
                    major_semiaxis_m=baseline.major_semiaxis_m,
                    minor_semiaxis_m=baseline.minor_semiaxis_m,
                ),
            )
        )
    return candidates


def _centerline_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    source = config["gates"]["teacher_surface"]
    return {
        "residual_p95_mm": float(source["residual_p95_mm"]),
        "residual_max_mm": float(source["residual_max_mm"]),
        "delta_beta_rms_p95_deg": float(source["phase_beta_rms_p95_deg"]),
        "delta_beta_rms_max_deg": 2.0,
        "acceleration_beta_rms_p95_deg": float(
            source["acceleration_beta_rms_p95_deg"]
        ),
        "seam_beta_rms_deg": float(source["seam_beta_rms_max_deg"]),
        "joint_margin_min_deg": float(source["joint_margin_min_deg"]),
    }


def _solve_centerline_artifact(
    *,
    family: EllipseFamilySpec,
    phase_count: int,
    environment: Any,
    atlas: ReachabilityAtlas,
    policy: TeacherPolicy,
    config: Mapping[str, Any],
    directory: Path,
    initial_beta: np.ndarray | None = None,
    traversal_direction: str = "forward",
    cyclic_cut: int = 0,
) -> tuple[Any, dict[str, Any]]:
    report_path = directory / "report.json"
    parquet_path = directory / "centerline.parquet"
    cached = read_valid_gate(report_path)
    if cached is not None and parquet_path.is_file():
        return None, cached
    targets = family.centerline(phase_count=int(phase_count))
    if initial_beta is None:
        initial_beta = atlas.match_targets(targets).initial_beta_path_rad
    trajectory = CanonicalTeacher(environment).solve(
        TrajectorySpec(
            trajectory_id=f"{family.family_id}:centerline",
            family_id=family.family_id,
            radius_mm=float(family.major_semiaxis_m * 1000.0),
            target_xyz_m=targets,
            traversal_direction=traversal_direction,
            cyclic_cut=int(cyclic_cut),
        ),
        policy,
        root_beta=np.asarray(initial_beta, dtype=float)[0],
        initial_beta_path=np.asarray(initial_beta, dtype=float),
    )
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(trajectory_frame(trajectory, environment), parquet_path)
    original = evaluate_centerline_gate(
        trajectory, thresholds=_centerline_thresholds(config)
    )
    report = {
        "family": _family_payload(family),
        "metrics": dict(trajectory.metrics),
        "provenance": dict(trajectory.provenance),
        "checks": {key: bool(value) for key, value in original["checks"].items()},
        "gate_pass": bool(original["centerline_gate_pass"]),
    }
    atomic_write_json(report_path, report)
    return trajectory, report


def _aligned_beta_gap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    key = ["phase_idx"]
    merged = left[key + list(BETA_COLUMNS)].merge(
        right[key + list(BETA_COLUMNS)],
        on=key,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    a = merged[[f"{name}_left" for name in BETA_COLUMNS]].to_numpy(dtype=float)
    b = merged[[f"{name}_right" for name in BETA_COLUMNS]].to_numpy(dtype=float)
    gap = np.rad2deg(np.sqrt(np.mean(np.square(a - b), axis=1)))
    return {
        "beta_gap_rms_p95_deg": float(np.percentile(gap, 95)),
        "beta_gap_rms_max_deg": float(np.max(gap)),
    }


def run_anchor_stage(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    output: Path,
) -> dict[str, Any]:
    protocol_gate = read_valid_gate(output / STAGE_DIRS["protocol"] / "gate.json")
    if protocol_gate is None or not protocol_gate["gate_pass"]:
        raise RuntimeError("anchor stage requires a passing protocol gate")
    stage = output / STAGE_DIRS["anchor"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    environment = load_environment(
        project_root, project_root / str(config["robot_config"])
    )
    atlas = _reachability_atlas(project_root, config)
    baseline = _baseline_family(project_root, config)
    candidates = _anchor_candidates(baseline, config)
    proposal_count = int(config["anchor"]["proposal_phase_count"])
    proposal_rows = []
    safe_margin = float(config["teacher_policy"]["safe_joint_margin_deg"])
    for candidate_id, family in candidates:
        match = atlas.match_targets(
            family.centerline(phase_count=proposal_count),
            beta_bounds_rad=environment.bounds,
        )
        margin = float(match.metrics.get("nearest_joint_margin_min_deg", -math.inf))
        score = float(
            match.metrics["nearest_distance_p95_mm"]
            + 0.25 * match.metrics["nearest_distance_max_mm"]
            + 20.0 * max(safe_margin - margin, 0.0)
        )
        proposal_rows.append(
            {
                "candidate_id": candidate_id,
                "proposal_score": score,
                **_family_payload(family),
                **match.metrics,
            }
        )
    proposals = pd.DataFrame(proposal_rows).sort_values(
        ["proposal_score", "candidate_id"], kind="stable"
    )
    proposals.to_csv(stage / "proposal_catalog.csv", index=False)
    lookup = dict(candidates)
    proposal_score = proposals.set_index("candidate_id")["proposal_score"].to_dict()
    trial_metadata: dict[str, dict[str, Any]] = {
        candidate_id: {
            "family_id": candidate_id,
            "root_index": 0,
            "direction": "forward",
            "cut_fraction": 0.0,
            "policy_seed_offset": 0,
            "proposal_score": float(proposal_score[candidate_id]),
            "search_group": candidate_id.split("_")[0],
        }
        for candidate_id in lookup
    }
    screen_limit = int(config["anchor"]["screen_candidate_limit"])
    selected_ids: list[str] = ["A0_000"]
    center_ranked = [
        value for value in proposals["candidate_id"].astype(str) if value.startswith("A1_")
    ]
    plane_ranked = [
        value for value in proposals["candidate_id"].astype(str) if value.startswith("A2_")
    ]
    selected_ids.extend(center_ranked[: min(8, max(screen_limit - len(selected_ids), 0))])
    selected_ids.extend(plane_ranked[: min(8, max(screen_limit - len(selected_ids), 0))])
    screen_phase = int(config["anchor"]["screen_phase_count"])
    configured_cuts = tuple(int(value) for value in config["anchor"]["cuts"])
    directions = tuple(str(value) for value in config["anchor"]["directions"])
    a3_trials = []
    for root_index in range(8):
        for direction in directions:
            for cut in configured_cuts:
                trial_id = f"A3_r{root_index}_{direction}_c{cut:04d}"
                lookup[trial_id] = baseline
                trial_metadata[trial_id] = {
                    "family_id": "A0_000",
                    "root_index": root_index,
                    "direction": direction,
                    "cut_fraction": float(cut / max(screen_phase, 1)),
                    "policy_seed_offset": 100 + root_index,
                    "proposal_score": float(proposal_score["A0_000"]),
                    "search_group": "A3",
                }
                a3_trials.append(trial_id)
    a3_priority = [
        f"A3_r{root_index}_forward_c{configured_cuts[0]:04d}"
        for root_index in range(8)
    ]
    selected_ids.extend(
        a3_priority[: min(8, max(screen_limit - len(selected_ids), 0))]
    )
    a4_trials = []
    for geometry_id in plane_ranked[:16]:
        for root_index in range(4):
            for direction in directions:
                for cut in configured_cuts[:2]:
                    trial_id = (
                        f"A4_{geometry_id}_r{root_index}_{direction}_c{cut:04d}"
                    )
                    lookup[trial_id] = lookup[geometry_id]
                    trial_metadata[trial_id] = {
                        "family_id": geometry_id,
                        "root_index": root_index,
                        "direction": direction,
                        "cut_fraction": float(cut / max(screen_phase, 1)),
                        "policy_seed_offset": 1000 + root_index,
                        "proposal_score": float(proposal_score[geometry_id]),
                        "search_group": "A4",
                    }
                    a4_trials.append(trial_id)
    a4_trials.sort(
        key=lambda value: (trial_metadata[value]["proposal_score"], value)
    )
    selected_ids.extend(a4_trials[: max(screen_limit - len(selected_ids), 0)])
    selected_ids = selected_ids[:screen_limit]
    pd.DataFrame(
        [{"candidate_id": value, **trial_metadata[value]} for value in selected_ids]
    ).to_csv(stage / "screen_trial_manifest.csv", index=False)
    v10 = _v10_centerline(project_root, config)
    v10_beta = v10[list(BETA_COLUMNS)].to_numpy(dtype=float)
    screen_rows = []
    for candidate_id in selected_ids:
        family = lookup[candidate_id]
        trial = trial_metadata[candidate_id]
        count = int(config["anchor"]["screen_phase_count"])
        initial = None
        if candidate_id == "A0_000":
            initial = _subsample_cyclic(
                v10_beta, count
            )
        else:
            target = family.centerline(phase_count=count)
            initial = atlas.match_targets(target).initial_beta_path_rad
            root_count = min(8, len(atlas.xyz_m))
            _distance, root_indices = cKDTree(atlas.xyz_m).query(
                target[0], k=root_count
            )
            root_indices = np.atleast_1d(root_indices)
            initial[0] = atlas.beta_rad[
                int(root_indices[int(trial["root_index"]) % len(root_indices)])
            ]
        _trajectory, report = _solve_centerline_artifact(
            family=family,
            phase_count=count,
            environment=environment,
            atlas=atlas,
            policy=_teacher_policy(
                config,
                seed=int(config["seeds"]["solver"])
                + int(trial["policy_seed_offset"]),
            ),
            config=config,
            directory=stage / "screen" / candidate_id,
            initial_beta=initial,
            traversal_direction=str(trial["direction"]),
            cyclic_cut=int(round(float(trial["cut_fraction"]) * count)) % count,
        )
        screen_rows.append(
            {
                "candidate_id": candidate_id,
                "search_group": str(trial["search_group"]),
                "gate_pass": bool(report["gate_pass"]),
                **{key: float(value) for key, value in report["metrics"].items()},
            }
        )
    screen = pd.DataFrame(screen_rows).sort_values(
        ["gate_pass", "joint_margin_min_deg", "residual_p95_mm"],
        ascending=[False, False, True],
        kind="stable",
    )
    screen.to_csv(stage / "screen_ranking.csv", index=False)
    verify_ids = screen.head(int(config["anchor"]["verify_top_count"]))[
        "candidate_id"
    ].astype(str)
    verify_rows = []
    for candidate_id in verify_ids:
        family = lookup[candidate_id]
        trial = trial_metadata[candidate_id]
        count = int(config["anchor"]["verify_phase_count"])
        target = family.centerline(phase_count=count)
        primary_initial = atlas.match_targets(target).initial_beta_path_rad
        root_count = min(8, len(atlas.xyz_m))
        _distance, root_indices = cKDTree(atlas.xyz_m).query(target[0], k=root_count)
        root_indices = np.atleast_1d(root_indices)
        primary_initial[0] = atlas.beta_rad[
            int(root_indices[int(trial["root_index"]) % len(root_indices)])
        ]
        primary_policy = _teacher_policy(
            config,
            seed=int(config["seeds"]["solver"])
            + int(trial["policy_seed_offset"]),
        )
        primary_dir = stage / "verify" / candidate_id / "primary"
        primary_trajectory, primary_report = _solve_centerline_artifact(
            family=family,
            phase_count=count,
            environment=environment,
            atlas=atlas,
            policy=primary_policy,
            config=config,
            directory=primary_dir,
            initial_beta=primary_initial,
            traversal_direction=str(trial["direction"]),
            cyclic_cut=int(round(float(trial["cut_fraction"]) * count)) % count,
        )
        primary_frame = pd.read_parquet(primary_dir / "centerline.parquet")
        audit_gaps = []
        directions = tuple(config["anchor"]["directions"])
        cuts = tuple(int(value) % count for value in config["anchor"]["cuts"])
        variants = []
        for direction in directions:
            for cut in cuts:
                if str(direction) == "forward" and cut == 0:
                    continue
                variants.append((str(direction), cut))
        # The primary seed is repeated once independently; traversal variants
        # are then audited against the same canonical phase ordering.
        repeat_dir = stage / "verify" / candidate_id / "repeat"
        _solve_centerline_artifact(
            family=family,
            phase_count=count,
            environment=environment,
            atlas=atlas,
            policy=_teacher_policy(config, seed=int(config["seeds"]["solver"]) + 1),
            config=config,
            directory=repeat_dir,
            initial_beta=primary_initial,
            traversal_direction=str(trial["direction"]),
            cyclic_cut=int(round(float(trial["cut_fraction"]) * count)) % count,
        )
        repeat_gap = _aligned_beta_gap(
            primary_frame, pd.read_parquet(repeat_dir / "centerline.parquet")
        )
        for direction, cut in variants:
            variant_dir = (
                stage / "verify" / candidate_id / f"{direction}_cut{cut:04d}"
            )
            _solve_centerline_artifact(
                family=family,
                phase_count=count,
                environment=environment,
                atlas=atlas,
                policy=primary_policy,
                config=config,
                directory=variant_dir,
                initial_beta=primary_initial,
                traversal_direction=direction,
                cyclic_cut=cut,
            )
            audit_gaps.append(
                _aligned_beta_gap(
                    primary_frame,
                    pd.read_parquet(variant_dir / "centerline.parquet"),
                )["beta_gap_rms_p95_deg"]
            )
        reverse_cut_p95 = max(audit_gaps, default=0.0)
        repeat_limit = float(
            config["gates"]["repeatability"]["repeat_beta_rms_p95_deg"]
        )
        reverse_limit = float(
            config["gates"]["repeatability"]["reverse_cut_beta_rms_p95_deg"]
        )
        full_pass = bool(
            primary_report["gate_pass"]
            and repeat_gap["beta_gap_rms_p95_deg"] <= repeat_limit
            and reverse_cut_p95 <= reverse_limit
        )
        verify_rows.append(
            {
                "candidate_id": candidate_id,
                "teacher_gate_pass": bool(primary_report["gate_pass"]),
                "repeat_beta_rms_p95_deg": repeat_gap["beta_gap_rms_p95_deg"],
                "reverse_cut_beta_rms_p95_deg": reverse_cut_p95,
                "full_gate_pass": full_pass,
                **{key: float(value) for key, value in primary_report["metrics"].items()},
            }
        )
    verification = pd.DataFrame(verify_rows).sort_values(
        ["full_gate_pass", "joint_margin_min_deg", "residual_p95_mm"],
        ascending=[False, False, True],
        kind="stable",
    )
    verification.to_csv(stage / "verification_ranking.csv", index=False)
    passing = verification[verification["full_gate_pass"]]
    selected_family = None
    if not passing.empty:
        selected_id = str(passing.iloc[0]["candidate_id"])
        selected_family = lookup[selected_id]
        selected_dir = stage / "verify" / selected_id / "primary"
        shutil.copy2(selected_dir / "centerline.parquet", stage / "selected_centerline.parquet")
        atomic_write_json(stage / "selected_anchor.json", _family_payload(selected_family))
    checks = {
        "at_least_one_720_phase_candidate_passes": bool(selected_family is not None),
        "selected_margin_at_least_1p5_deg": bool(
            selected_family is not None
            and float(passing.iloc[0]["joint_margin_min_deg"]) >= 1.5
        ),
        "selected_repeatability_passes": bool(
            selected_family is not None
            and float(passing.iloc[0]["repeat_beta_rms_p95_deg"])
            <= float(config["gates"]["repeatability"]["repeat_beta_rms_p95_deg"])
        ),
        "selected_reverse_cut_passes": bool(
            selected_family is not None
            and float(passing.iloc[0]["reverse_cut_beta_rms_p95_deg"])
            <= float(config["gates"]["repeatability"]["reverse_cut_beta_rms_p95_deg"])
        ),
    }
    return _write_gate(
        gate_path,
        checks=checks,
        proposal_count=int(len(proposals)),
        screen_count=int(len(screen)),
        verification_count=int(len(verification)),
        selected_family=(
            None if selected_family is None else _family_payload(selected_family)
        ),
    )


def _surface_local_metrics(surface: TeacherSurface) -> dict[str, float]:
    beta = surface.beta_rad
    points = surface.cross_section.points
    physical = np.column_stack(
        [
            points[:, 0] * float(surface.radial_radius_mm),
            points[:, 1] * float(surface.plane_radius_mm),
        ]
    )
    distances = []
    gaps = []
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            distance = float(np.linalg.norm(physical[left] - physical[right]))
            gap = np.rad2deg(
                np.sqrt(np.mean(np.square(beta[left] - beta[right]), axis=1))
            )
            distances.extend([distance] * len(gap))
            gaps.extend(gap.tolist())
    distances_array = np.asarray(distances, dtype=float)
    gaps_array = np.asarray(gaps, dtype=float)
    output = {}
    for radius in (5.0, 10.0):
        selected = gaps_array[distances_array <= radius + 1e-12]
        output[f"local_{int(radius)}mm_beta_gap_p95_deg"] = (
            float(np.percentile(selected, 95)) if len(selected) else 0.0
        )
    return output


def _solve_surface_artifact(
    *,
    family: EllipseFamilySpec,
    phase_count: int,
    cross_section: TubeCrossSection,
    radial_radius_mm: float,
    plane_radius_mm: float,
    environment: Any,
    policy: TeacherPolicy,
    config: Mapping[str, Any],
    directory: Path,
    centerline_seed: np.ndarray | None,
    traversal_direction: str = "forward",
    cyclic_cut: int = 0,
) -> tuple[TeacherSurface | None, dict[str, Any]]:
    report_path = directory / "report.json"
    parquet_path = directory / "surface.parquet"
    cached = read_valid_gate(report_path)
    if cached is not None and parquet_path.is_file():
        return None, cached
    surface = CanonicalRegionTeacher(environment).solve_surface(
        family=family,
        cross_section=cross_section,
        phase_count=int(phase_count),
        radial_radius_mm=float(radial_radius_mm),
        plane_radius_mm=float(plane_radius_mm),
        policy=policy,
        root_beta=(
            np.asarray(centerline_seed, dtype=float)[0]
            if centerline_seed is not None
            else np.zeros(6, dtype=float)
        ),
        centerline_seed=centerline_seed,
        sweep_directions=tuple(config["tube"]["sweep_directions"]),
        traversal_direction=traversal_direction,
        cyclic_cut=int(cyclic_cut),
    )
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(surface.to_frame(), parquet_path)
    gate = evaluate_teacher_surface_gate(
        surface.metrics, thresholds=config["gates"]["teacher_surface"]
    )
    local = _surface_local_metrics(surface)
    local_checks = {
        "local_5mm_consistency": bool(
            local["local_5mm_beta_gap_p95_deg"]
            <= float(config["gates"]["local_consistency"]["gap_5mm_p95_deg"])
        ),
        "local_10mm_consistency": bool(
            local["local_10mm_beta_gap_p95_deg"]
            <= float(config["gates"]["local_consistency"]["gap_10mm_p95_deg"])
        ),
    }
    report = {
        "family": _family_payload(family),
        "radial_radius_mm": float(radial_radius_mm),
        "plane_radius_mm": float(plane_radius_mm),
        "phase_count": int(phase_count),
        "cross_section_count": int(len(cross_section.points)),
        "cross_section_fingerprint": cross_section.fingerprint,
        "metrics": {**dict(surface.metrics), **local},
        "provenance": dict(surface.provenance),
        "checks": {**gate["checks"], **local_checks},
        "gate_pass": bool(all({**gate["checks"], **local_checks}.values())),
    }
    atomic_write_json(report_path, report)
    return surface, report


def _surface_aligned_gap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    keys = ["cross_section_idx", "phase_idx"]
    merged = left[keys + list(BETA_COLUMNS)].merge(
        right[keys + list(BETA_COLUMNS)],
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    a = merged[[f"{name}_left" for name in BETA_COLUMNS]].to_numpy(dtype=float)
    b = merged[[f"{name}_right" for name in BETA_COLUMNS]].to_numpy(dtype=float)
    gap = np.rad2deg(np.sqrt(np.mean(np.square(a - b), axis=1)))
    return {
        "aligned_count": int(len(merged)),
        "beta_gap_rms_p95_deg": float(np.percentile(gap, 95)),
        "beta_gap_rms_max_deg": float(np.max(gap)),
    }


def _load_selected_anchor(output: Path) -> tuple[EllipseFamilySpec, np.ndarray]:
    stage = output / STAGE_DIRS["anchor"]
    family = _family_from_payload(
        json.loads((stage / "selected_anchor.json").read_text(encoding="utf-8"))
    )
    centerline = pd.read_parquet(stage / "selected_centerline.parquet").sort_values(
        "phase_idx", kind="stable"
    )
    return family, centerline[list(BETA_COLUMNS)].to_numpy(dtype=float)


def run_tube_stage(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    output: Path,
) -> dict[str, Any]:
    anchor_gate = read_valid_gate(output / STAGE_DIRS["anchor"] / "gate.json")
    if anchor_gate is None or not anchor_gate["gate_pass"]:
        raise RuntimeError("tube stage requires a passing 720-phase anchor gate")
    stage = output / STAGE_DIRS["tube"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    environment = load_environment(
        project_root, project_root / str(config["robot_config"])
    )
    family, anchor_beta = _load_selected_anchor(output)
    master = TubeCrossSection.master(seed=int(config["seeds"]["cross_section"]))
    policy = _teacher_policy(config)
    candidates = [tuple(map(float, row)) for row in config["tube"]["frontier_mm"]]
    screen_rows = []
    for radial, plane in candidates:
        count = int(config["tube"]["screen_phase_count"])
        cross = master.prefix(int(config["tube"]["screen_cross_section_count"]))
        _surface, report = _solve_surface_artifact(
            family=family,
            phase_count=count,
            cross_section=cross,
            radial_radius_mm=radial,
            plane_radius_mm=plane,
            environment=environment,
            policy=policy,
            config=config,
            directory=stage / "screen" / f"r{radial:g}_p{plane:g}",
            centerline_seed=_subsample_cyclic(anchor_beta, count),
        )
        screen_rows.append(
            {
                "radial_radius_mm": radial,
                "plane_radius_mm": plane,
                "area_score_mm2": radial * plane,
                "gate_pass": bool(report["gate_pass"]),
                **{key: float(value) for key, value in report["metrics"].items()},
            }
        )
    one_by_one = [
        row
        for row in screen_rows
        if math.isclose(float(row["radial_radius_mm"]), 1.0)
        and math.isclose(float(row["plane_radius_mm"]), 1.0)
    ]
    if one_by_one and not bool(one_by_one[0]["gate_pass"]):
        tested = {(float(left), float(right)) for left, right in candidates}
        widths = tuple(float(value) for value in config["tube"]["fallback_widths_mm"])
        for radial in widths:
            for plane in widths:
                if (radial, plane) in tested:
                    continue
                count = int(config["tube"]["screen_phase_count"])
                cross = master.prefix(int(config["tube"]["screen_cross_section_count"]))
                _surface, report = _solve_surface_artifact(
                    family=family,
                    phase_count=count,
                    cross_section=cross,
                    radial_radius_mm=radial,
                    plane_radius_mm=plane,
                    environment=environment,
                    policy=policy,
                    config=config,
                    directory=stage / "screen" / f"r{radial:g}_p{plane:g}",
                    centerline_seed=_subsample_cyclic(anchor_beta, count),
                )
                screen_rows.append(
                    {
                        "radial_radius_mm": radial,
                        "plane_radius_mm": plane,
                        "area_score_mm2": radial * plane,
                        "gate_pass": bool(report["gate_pass"]),
                        **{
                            key: float(value)
                            for key, value in report["metrics"].items()
                        },
                    }
                )
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(stage / "screen_frontier.csv", index=False)
    screen_pass = screen[screen["gate_pass"]].sort_values(
        ["area_score_mm2", "radial_radius_mm", "plane_radius_mm"],
        ascending=[False, True, True],
        kind="stable",
    )
    dense_rows = []
    # Dense-audit all screen-pass contenders; no failed phase or point is removed.
    for row in screen_pass.itertuples(index=False):
        radial = float(row.radial_radius_mm)
        plane = float(row.plane_radius_mm)
        count = int(config["tube"]["dense_phase_count"])
        cross = master.prefix(int(config["tube"]["dense_cross_section_count"]))
        _surface, report = _solve_surface_artifact(
            family=family,
            phase_count=count,
            cross_section=cross,
            radial_radius_mm=radial,
            plane_radius_mm=plane,
            environment=environment,
            policy=policy,
            config=config,
            directory=stage / "dense" / f"r{radial:g}_p{plane:g}" / "primary",
            centerline_seed=_subsample_cyclic(anchor_beta, count),
        )
        dense_rows.append(
            {
                "radial_radius_mm": radial,
                "plane_radius_mm": plane,
                "area_score_mm2": radial * plane,
                "gate_pass": bool(report["gate_pass"]),
                **{key: float(value) for key, value in report["metrics"].items()},
            }
        )
    dense = pd.DataFrame(dense_rows)
    dense.to_csv(stage / "dense_frontier.csv", index=False)
    dense_pass = (
        dense[dense["gate_pass"]].sort_values(
            ["area_score_mm2", "radial_radius_mm", "plane_radius_mm"],
            ascending=[False, True, True],
            kind="stable",
        )
        if not dense.empty
        else dense
    )
    selected = None
    repeat_gap = math.inf
    reverse_gap = math.inf
    if not dense_pass.empty:
        best = dense_pass.iloc[0]
        radial = float(best["radial_radius_mm"])
        plane = float(best["plane_radius_mm"])
        count = int(config["tube"]["dense_phase_count"])
        cross = master.prefix(int(config["tube"]["dense_cross_section_count"]))
        primary_dir = stage / "dense" / f"r{radial:g}_p{plane:g}" / "primary"
        primary = pd.read_parquet(primary_dir / "surface.parquet")
        repeat_dir = stage / "dense" / f"r{radial:g}_p{plane:g}" / "repeat"
        _solve_surface_artifact(
            family=family,
            phase_count=count,
            cross_section=cross,
            radial_radius_mm=radial,
            plane_radius_mm=plane,
            environment=environment,
            policy=_teacher_policy(config, seed=int(config["seeds"]["solver"]) + 1),
            config=config,
            directory=repeat_dir,
            centerline_seed=_subsample_cyclic(anchor_beta, count),
        )
        repeat_gap = _surface_aligned_gap(
            primary, pd.read_parquet(repeat_dir / "surface.parquet")
        )["beta_gap_rms_p95_deg"]
        reverse_dir = stage / "dense" / f"r{radial:g}_p{plane:g}" / "reverse_cut"
        _solve_surface_artifact(
            family=family,
            phase_count=count,
            cross_section=cross,
            radial_radius_mm=radial,
            plane_radius_mm=plane,
            environment=environment,
            policy=policy,
            config=config,
            directory=reverse_dir,
            centerline_seed=_subsample_cyclic(anchor_beta, count),
            traversal_direction="reverse",
            cyclic_cut=count // 4,
        )
        reverse_gap = _surface_aligned_gap(
            primary, pd.read_parquet(reverse_dir / "surface.parquet")
        )["beta_gap_rms_p95_deg"]
        repeat_pass = repeat_gap <= float(
            config["gates"]["repeatability"]["repeat_beta_rms_p95_deg"]
        )
        reverse_pass = reverse_gap <= float(
            config["gates"]["repeatability"]["reverse_cut_beta_rms_p95_deg"]
        )
        if repeat_pass and reverse_pass:
            selected = {
                "radial_radius_mm": radial,
                "plane_radius_mm": plane,
                "phase_count": count,
                "cross_section_count": int(len(cross.points)),
                "repeat_beta_rms_p95_deg": repeat_gap,
                "reverse_cut_beta_rms_p95_deg": reverse_gap,
            }
            atomic_write_json(stage / "selected_tube.json", selected)
            shutil.copy2(primary_dir / "surface.parquet", stage / "selected_surface.parquet")
    half_pass = bool(
        not screen[
            np.isclose(screen["radial_radius_mm"], 0.5)
            & np.isclose(screen["plane_radius_mm"], 0.5)
        ].empty
        and screen.loc[
            np.isclose(screen["radial_radius_mm"], 0.5)
            & np.isclose(screen["plane_radius_mm"], 0.5),
            "gate_pass",
        ].astype(bool).all()
    )
    checks = {
        "minimum_0p5_by_0p5_tube_passes": half_pass,
        "at_least_one_dense_tube_passes": bool(selected is not None),
        "selected_repeatability_passes": bool(
            selected is not None
            and repeat_gap
            <= float(config["gates"]["repeatability"]["repeat_beta_rms_p95_deg"])
        ),
        "selected_reverse_cut_passes": bool(
            selected is not None
            and reverse_gap
            <= float(config["gates"]["repeatability"]["reverse_cut_beta_rms_p95_deg"])
        ),
    }
    return _write_gate(
        gate_path,
        checks=checks,
        selected_tube=selected,
        screen_candidate_count=int(len(screen)),
        dense_candidate_count=int(len(dense)),
    )


def _load_selected_tube(output: Path) -> dict[str, Any]:
    return json.loads(
        (output / STAGE_DIRS["tube"] / "selected_tube.json").read_text(
            encoding="utf-8"
        )
    )


def run_core_stage(
    *, config: Mapping[str, Any], source_root: Path, project_root: Path, output: Path
) -> dict[str, Any]:
    tube_gate = read_valid_gate(output / STAGE_DIRS["tube"] / "gate.json")
    if tube_gate is None or not tube_gate["gate_pass"]:
        raise RuntimeError("core stage requires a passing complete-tube gate")
    stage = output / STAGE_DIRS["core"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    environment = load_environment(project_root, project_root / str(config["robot_config"]))
    family, anchor_beta = _load_selected_anchor(output)
    tube = _load_selected_tube(output)
    master = TubeCrossSection.master(seed=int(config["seeds"]["cross_section"]))
    cross = master.prefix(int(config["core"]["cross_section_count"]))
    count = int(config["core"]["phase_count"])
    surface, report = _solve_surface_artifact(
        family=family,
        phase_count=count,
        cross_section=cross,
        radial_radius_mm=float(tube["radial_radius_mm"]),
        plane_radius_mm=float(tube["plane_radius_mm"]),
        environment=environment,
        policy=_teacher_policy(config),
        config=config,
        directory=stage / "primary",
        centerline_seed=_subsample_cyclic(anchor_beta, count),
    )
    primary = pd.read_parquet(stage / "primary/surface.parquet")
    _atomic_parquet(primary, stage / "D_core_0p5m.parquet")
    repeat_dir = stage / "repeat"
    reverse_dir = stage / "reverse_cut"
    _solve_surface_artifact(
        family=family,
        phase_count=count,
        cross_section=cross,
        radial_radius_mm=float(tube["radial_radius_mm"]),
        plane_radius_mm=float(tube["plane_radius_mm"]),
        environment=environment,
        policy=_teacher_policy(config, seed=int(config["seeds"]["solver"]) + 1),
        config=config,
        directory=repeat_dir,
        centerline_seed=_subsample_cyclic(anchor_beta, count),
    )
    _solve_surface_artifact(
        family=family,
        phase_count=count,
        cross_section=cross,
        radial_radius_mm=float(tube["radial_radius_mm"]),
        plane_radius_mm=float(tube["plane_radius_mm"]),
        environment=environment,
        policy=_teacher_policy(config),
        config=config,
        directory=reverse_dir,
        centerline_seed=_subsample_cyclic(anchor_beta, count),
        traversal_direction="reverse",
        cyclic_cut=count // 4,
    )
    repeat_gap = _surface_aligned_gap(
        primary, pd.read_parquet(repeat_dir / "surface.parquet")
    )
    reverse_gap = _surface_aligned_gap(
        primary, pd.read_parquet(reverse_dir / "surface.parquet")
    )
    atomic_write_json(
        stage / "invariance_report.json",
        {"repeat": repeat_gap, "reverse_cut": reverse_gap},
    )
    # Coverage is measured against a separate phase/cross-section design.
    probe_cross = TubeCrossSection.master(seed=int(config["seeds"]["audit"])).prefix(
        int(config["core"]["coverage_probe_cross_section_count"])
    )
    probe = family.tube_targets(
        phase_count=int(config["core"]["coverage_probe_phase_count"]),
        normalized_offsets=probe_cross.points,
        radial_radius_mm=float(tube["radial_radius_mm"]),
        plane_radius_mm=float(tube["plane_radius_mm"]),
        phase_offset_rad=math.pi / int(config["core"]["coverage_probe_phase_count"]),
    ).reshape(-1, 3)
    coverage = audit_cartesian_coverage(
        primary[list(XYZ_COLUMNS)].to_numpy(dtype=float),
        probe,
        p95_limit_mm=float(config["gates"]["coverage"]["nearest_p95_mm"]),
        max_limit_mm=float(config["gates"]["coverage"]["nearest_max_mm"]),
    )
    atomic_write_json(stage / "coverage_report.json", coverage)
    checks = {
        "all_protocol_rows_present": bool(
            len(primary)
            == int(config["core"]["phase_count"])
            * int(config["core"]["cross_section_count"])
        ),
        "teacher_surface_gate_passes": bool(report["gate_pass"]),
        "independent_coverage_gate_passes": bool(coverage["coverage_gate_pass"]),
        "full_surface_repeatability_passes": bool(
            repeat_gap["beta_gap_rms_p95_deg"]
            <= float(config["gates"]["repeatability"]["repeat_beta_rms_p95_deg"])
        ),
        "full_surface_reverse_cut_passes": bool(
            reverse_gap["beta_gap_rms_p95_deg"]
            <= float(config["gates"]["repeatability"]["reverse_cut_beta_rms_p95_deg"])
        ),
        "core_is_control_only": True,
    }
    return _write_gate(
        gate_path,
        checks=checks,
        rows=int(len(primary)),
        teacher_report=report,
        coverage_report=coverage,
        invariance_report={"repeat": repeat_gap, "reverse_cut": reverse_gap},
    )


def run_pilot_stage(
    *, config: Mapping[str, Any], source_root: Path, project_root: Path, output: Path
) -> dict[str, Any]:
    core_gate = read_valid_gate(output / STAGE_DIRS["core"] / "gate.json")
    if core_gate is None or not core_gate["gate_pass"]:
        raise RuntimeError("pilot stage requires a passing core-tube gate")
    stage = output / STAGE_DIRS["pilot"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    frozen_catalog = _catalog_from_csv(
        output / STAGE_DIRS["protocol"] / "family_catalog.csv",
        seed=int(config["seeds"]["family"]),
    )
    selected_anchor, _anchor_beta = _load_selected_anchor(output)
    catalog = frozen_catalog.reanchor(selected_anchor)
    catalog.frame.to_csv(stage / "applied_family_catalog.csv", index=False)
    atomic_write_json(
        stage / "applied_catalog_manifest.json",
        {
            "frozen_catalog_fingerprint": frozen_catalog.fingerprint,
            "applied_catalog_fingerprint": catalog.fingerprint,
            "selected_anchor_fingerprint": selected_anchor.fingerprint,
            "roles_and_ids_unchanged": bool(
                catalog.frame[["family_id", "role", "is_primary", "reserve_rank"]].equals(
                    frozen_catalog.frame[["family_id", "role", "is_primary", "reserve_rank"]]
                )
            ),
        },
    )
    tube = _load_selected_tube(output)
    environment = load_environment(project_root, project_root / str(config["robot_config"]))
    atlas = _reachability_atlas(project_root, config)
    cross = TubeCrossSection.master(seed=int(config["seeds"]["cross_section"])).prefix(
        int(config["pilot"]["cross_section_count"])
    )
    count = int(config["pilot"]["phase_count"])
    quality = catalog.frame[["family_id", "role", "is_primary", "reserve_rank"]].copy()
    quality["solved"] = False
    quality["complete_gate_pass"] = False
    report_metrics: dict[str, dict[str, Any]] = {}
    quota = {"train": 14, "validation": 5, "virgin_test": 5}
    for role, required in quota.items():
        role_rows = catalog.frame[catalog.frame["role"].eq(role)].copy()
        role_rows["priority"] = np.where(
            role_rows["is_primary"], role_rows["catalog_order"], 10_000 + role_rows["reserve_rank"]
        )
        role_rows = role_rows.sort_values("priority", kind="stable")
        passed = 0
        for row in role_rows.itertuples(index=False):
            if passed >= required:
                break
            family_id = str(row.family_id)
            family = catalog.family(family_id)
            seed_path = atlas.match_targets(
                family.centerline(phase_count=count)
            ).initial_beta_path_rad
            _surface, report = _solve_surface_artifact(
                family=family,
                phase_count=count,
                cross_section=cross,
                radial_radius_mm=float(tube["radial_radius_mm"]),
                plane_radius_mm=float(tube["plane_radius_mm"]),
                environment=environment,
                policy=_teacher_policy(config),
                config=config,
                directory=stage / "families" / family_id,
                centerline_seed=seed_path,
            )
            mask = quality["family_id"].astype(str).eq(family_id)
            quality.loc[mask, "solved"] = True
            quality.loc[mask, "complete_gate_pass"] = bool(report["gate_pass"])
            report_metrics[family_id] = report
            if report["gate_pass"]:
                passed += 1
    quality.to_csv(stage / "family_quality.csv", index=False)
    selection = None
    try:
        selection = promote_complete_families(catalog, quality)
    except RuntimeError:
        selection = None
    frames: list[pd.DataFrame] = []
    if selection is not None:
        selection.to_csv(stage / "selected_families.csv", index=False)
        for row in selection.itertuples(index=False):
            frame = pd.read_parquet(stage / "families" / str(row.family_id) / "surface.parquet")
            frame["role"] = str(row.role)
            frame["is_primary_family"] = bool(row.is_primary)
            frames.append(frame)
    train_validation = pd.DataFrame()
    virgin = pd.DataFrame()
    conflict_report: dict[str, Any] = {
        "cross_family_pair_count": 0,
        "conflict_pair_count": 0,
        "static_representation_gate_pass": False,
    }
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        train_validation = combined[combined["role"].isin(["train", "validation"])].copy()
        virgin = combined[combined["role"].eq("virgin_test")].copy()
        _atomic_parquet(
            train_validation, stage / "D_family_pilot_train_validation.parquet"
        )
        _atomic_parquet(virgin, stage / "sealed/D_family_pilot_virgin_test.parquet")
        conflict_report, conflicts = audit_cross_family_conflicts(
            train_validation,
            xyz_radius_mm=float(config["gates"]["conflicts"]["xyz_radius_mm"]),
            beta_gap_threshold_deg=float(
                config["gates"]["conflicts"]["beta_gap_threshold_deg"]
            ),
        )
        _atomic_parquet(conflicts, stage / "cross_family_conflicts.parquet")
        atomic_write_json(stage / "conflict_report.json", conflict_report)
    role_counts = (
        selection.groupby("role").size().to_dict() if selection is not None else {}
    )
    checks = {
        "exact_14_5_5_complete_families": bool(
            role_counts == {"train": 14, "validation": 5, "virgin_test": 5}
        ),
        "all_selected_families_are_complete": bool(
            selection is not None
            and all(
                bool(report_metrics[str(family_id)]["gate_pass"])
                for family_id in selection["family_id"]
            )
        ),
        "train_validation_and_test_stored_separately": bool(
            not train_validation.empty
            and not virgin.empty
            and (stage / "sealed/D_family_pilot_virgin_test.parquet").is_file()
        ),
        "conflict_audit_completed_on_train_validation_only": bool(
            frames
            and set(train_validation["family_id"]).isdisjoint(set(virgin["family_id"]))
        ),
    }
    return _write_gate(
        gate_path,
        checks=checks,
        selected_role_counts={key: int(value) for key, value in role_counts.items()},
        train_validation_rows=int(len(train_validation)),
        sealed_test_rows=int(len(virgin)),
        conflict_report=conflict_report,
    )


def run_representation_stage(
    *, config: Mapping[str, Any], source_root: Path, project_root: Path, output: Path
) -> dict[str, Any]:
    pilot_gate = read_valid_gate(output / STAGE_DIRS["pilot"] / "gate.json")
    if pilot_gate is None or not pilot_gate["gate_pass"]:
        raise RuntimeError("representation stage requires a passing family-pilot gate")
    stage = output / STAGE_DIRS["representation"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    pilot_stage = output / STAGE_DIRS["pilot"]
    conflict = json.loads((pilot_stage / "conflict_report.json").read_text(encoding="utf-8"))
    diagnostics = dict(conflict)
    # Chart/state evidence is deliberately unavailable unless separately
    # demonstrated.  T3's default chart id of zero is not counted as evidence.
    diagnostics.update(
        {
            "stable_chart_count": 0,
            "chart_repeat_ari": 0.0,
            "chart_xyz_macro_f1": 0.0,
            "ambiguous_voxel_count": int(conflict.get("conflict_pair_count", 0)),
            "previous_beta_resolves_conflicts": False,
        }
    )
    representation = select_student_representation(diagnostics)
    decision = {
        "representation": representation,
        "decision_order": ["static", "chart_expert", "stateful", "stop"],
        "diagnostics": diagnostics,
        "model_selection_family_roles": ["train", "validation"],
        "virgin_test_opened": False,
        "static_model": {
            "inputs": ["target_x_m", "target_y_m", "target_z_m"],
            "outputs": list(BETA_COLUMNS),
            "hidden_units": list(config["representation"]["static_hidden_units"]),
            "activation": "gelu",
            "bounded_output": "tanh",
            "fk_loss_lambda": float(config["training"]["lambda_fk"]),
        },
    }
    atomic_write_json(stage / "representation_decision.json", decision)
    checks = {
        "decision_uses_only_train_validation_conflict_audit": True,
        "static_selected_only_if_zero_conflicts": bool(
            representation != "static" or int(conflict.get("conflict_pair_count", 1)) == 0
        ),
        "selected_representation_is_implementable": bool(
            representation in {"static", "chart_expert", "stateful"}
        ),
        "virgin_test_remains_sealed": True,
    }
    return _write_gate(
        gate_path,
        checks=checks,
        representation=representation,
        diagnostics=diagnostics,
    )


def run_formal_stage(
    *, config: Mapping[str, Any], source_root: Path, project_root: Path, output: Path
) -> dict[str, Any]:
    representation_gate = read_valid_gate(
        output / STAGE_DIRS["representation"] / "gate.json"
    )
    if representation_gate is None or not representation_gate["gate_pass"]:
        raise RuntimeError("formal dataset requires a locked implementable representation")
    stage = output / STAGE_DIRS["formal"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(output / STAGE_DIRS["pilot"] / "selected_families.csv")
    catalog = _catalog_from_csv(
        output / STAGE_DIRS["pilot"] / "applied_family_catalog.csv",
        seed=int(config["seeds"]["family"]),
    )
    tube = _load_selected_tube(output)
    environment = load_environment(project_root, project_root / str(config["robot_config"]))
    atlas = _reachability_atlas(project_root, config)
    count = int(config["formal"]["phase_count"])
    cross = TubeCrossSection.master(seed=int(config["seeds"]["cross_section"])).prefix(
        int(config["formal"]["cross_section_count"])
    )
    family_frames: list[pd.DataFrame] = []
    family_reports = []
    centerline_invariance = []
    sentinel_invariance = []
    for selected_row in selection.itertuples(index=False):
        family_id = str(selected_row.family_id)
        role = str(selected_row.role)
        family = catalog.family(family_id)
        seed_path = atlas.match_targets(
            family.centerline(phase_count=count)
        ).initial_beta_path_rad
        family_dir = stage / "families" / family_id / "primary"
        _surface, report = _solve_surface_artifact(
            family=family,
            phase_count=count,
            cross_section=cross,
            radial_radius_mm=float(tube["radial_radius_mm"]),
            plane_radius_mm=float(tube["plane_radius_mm"]),
            environment=environment,
            policy=_teacher_policy(config),
            config=config,
            directory=family_dir,
            centerline_seed=seed_path,
        )
        frame = pd.read_parquet(family_dir / "surface.parquet")
        frame["role"] = role
        family_frames.append(frame)
        family_reports.append(
            {"family_id": family_id, "role": role, "gate_pass": bool(report["gate_pass"])}
        )
        primary_center = frame[frame["cross_section_idx"].eq(0)].copy()
        repeat_dir = stage / "families" / family_id / "centerline_repeat"
        _solve_centerline_artifact(
            family=family,
            phase_count=count,
            environment=environment,
            atlas=atlas,
            policy=_teacher_policy(config, seed=int(config["seeds"]["solver"]) + 1),
            config=config,
            directory=repeat_dir,
            initial_beta=seed_path,
        )
        repeat_center = pd.read_parquet(repeat_dir / "centerline.parquet")
        gap = _aligned_beta_gap(primary_center, repeat_center)
        centerline_invariance.append(
            {"family_id": family_id, "role": role, **gap}
        )
        if role == "virgin_test":
            repeat_surface_dir = stage / "families" / family_id / "surface_repeat"
            reverse_surface_dir = stage / "families" / family_id / "surface_reverse_cut"
            _solve_surface_artifact(
                family=family,
                phase_count=count,
                cross_section=cross,
                radial_radius_mm=float(tube["radial_radius_mm"]),
                plane_radius_mm=float(tube["plane_radius_mm"]),
                environment=environment,
                policy=_teacher_policy(
                    config, seed=int(config["seeds"]["solver"]) + 1
                ),
                config=config,
                directory=repeat_surface_dir,
                centerline_seed=seed_path,
            )
            _solve_surface_artifact(
                family=family,
                phase_count=count,
                cross_section=cross,
                radial_radius_mm=float(tube["radial_radius_mm"]),
                plane_radius_mm=float(tube["plane_radius_mm"]),
                environment=environment,
                policy=_teacher_policy(config),
                config=config,
                directory=reverse_surface_dir,
                centerline_seed=seed_path,
                traversal_direction="reverse",
                cyclic_cut=count // 4,
            )
            sentinel_invariance.append(
                {
                    "family_id": family_id,
                    "repeat_beta_rms_p95_deg": _surface_aligned_gap(
                        frame,
                        pd.read_parquet(repeat_surface_dir / "surface.parquet"),
                    )["beta_gap_rms_p95_deg"],
                    "reverse_cut_beta_rms_p95_deg": _surface_aligned_gap(
                        frame,
                        pd.read_parquet(reverse_surface_dir / "surface.parquet"),
                    )["beta_gap_rms_p95_deg"],
                }
            )
    quality = pd.DataFrame(family_reports)
    quality.to_csv(stage / "family_quality.csv", index=False)
    pd.DataFrame(centerline_invariance).to_csv(
        stage / "centerline_invariance.csv", index=False
    )
    pd.DataFrame(sentinel_invariance).to_csv(
        stage / "test_surface_invariance.csv", index=False
    )
    combined = pd.concat(family_frames, ignore_index=True, sort=False)
    train = combined[combined["role"].eq("train")].copy()
    validation = combined[combined["role"].eq("validation")].copy()
    virgin = combined[combined["role"].eq("virgin_test")].copy()
    _atomic_parquet(train, stage / "train/D_full_252k.parquet")
    _atomic_parquet(validation, stage / "validation/D_validation_90k.parquet")
    _atomic_parquet(virgin, stage / "sealed/D_virgin_test_90k.parquet")
    sizes = tuple(
        min(int(value), len(train)) for value in config["formal"]["nested_train_sizes"]
    )
    sizes = tuple(dict.fromkeys(sizes))
    subsets = nested_family_sample_indices(
        train, sizes=sizes, seed=int(config["seeds"]["formal_train"][0])
    )
    subset_manifest = []
    for requested, indices in subsets.items():
        name = "D_full_252k" if requested == len(train) else f"D{requested // 1000}k"
        path = stage / "train" / f"{name}.parquet"
        _atomic_parquet(train.loc[indices].reset_index(drop=True), path)
        subset_manifest.append(
            {
                "name": name,
                "rows": int(requested),
                "sha256": sha256_file(path),
                "family_count": int(train.loc[indices, "family_id"].nunique()),
            }
        )
    atomic_write_json(stage / "nested_subset_manifest.json", subset_manifest)
    repeat_limit = float(
        config["gates"]["repeatability"]["repeat_beta_rms_p95_deg"]
    )
    reverse_limit = float(
        config["gates"]["repeatability"]["reverse_cut_beta_rms_p95_deg"]
    )
    checks = {
        "exact_24_complete_families": bool(
            len(quality) == 24 and quality["gate_pass"].astype(bool).all()
        ),
        "exact_14_5_5_group_split": bool(
            combined.groupby("role")["family_id"].nunique().to_dict()
            == {"train": 14, "validation": 5, "virgin_test": 5}
        ),
        "formal_row_counts_match_protocol": bool(
            len(train) == 14 * count * len(cross.points)
            and len(validation) == 5 * count * len(cross.points)
            and len(virgin) == 5 * count * len(cross.points)
        ),
        "all_centerline_repeats_pass": bool(
            centerline_invariance
            and max(row["beta_gap_rms_p95_deg"] for row in centerline_invariance)
            <= repeat_limit
        ),
        "five_test_surface_sentinels_pass": bool(
            len(sentinel_invariance) == 5
            and max(row["repeat_beta_rms_p95_deg"] for row in sentinel_invariance)
            <= repeat_limit
            and max(row["reverse_cut_beta_rms_p95_deg"] for row in sentinel_invariance)
            <= reverse_limit
        ),
        "nested_subsets_include_all_14_train_families": bool(
            subset_manifest and all(row["family_count"] == 14 for row in subset_manifest)
        ),
    }
    return _write_gate(
        gate_path,
        checks=checks,
        train_rows=int(len(train)),
        validation_rows=int(len(validation)),
        sealed_test_rows=int(len(virgin)),
        nested_subsets=subset_manifest,
    )


def _tracking_metrics(
    target_xyz_m: np.ndarray, achieved_xyz_m: np.ndarray, *, max_limit_mm: float
) -> dict[str, Any]:
    target = np.asarray(target_xyz_m, dtype=float).reshape(-1, 3)
    achieved = np.asarray(achieved_xyz_m, dtype=float).reshape(-1, 3)
    delta_mm = (achieved - target) * 1000.0
    norm_mm = np.linalg.norm(delta_mm, axis=1)
    axis_mean = np.mean(np.abs(delta_mm), axis=0)
    return {
        "rows": int(len(target)),
        "tracking_p95_mm": float(np.percentile(norm_mm, 95)),
        "tracking_max_mm": float(np.max(norm_mm)),
        "axis_mean_abs_mm": axis_mean.tolist(),
        "max_limit_mm": float(max_limit_mm),
    }


def _student_gate(
    metrics: Mapping[str, Any],
    *,
    p95_limit_mm: float,
    axis_limit_mm: float,
) -> dict[str, Any]:
    checks = {
        "tracking_p95": bool(
            float(metrics["tracking_p95_mm"]) <= float(p95_limit_mm)
        ),
        "tracking_max": bool(
            float(metrics["tracking_max_mm"]) <= float(metrics["max_limit_mm"])
        ),
        "per_axis_mean": bool(
            max(float(value) for value in metrics["axis_mean_abs_mm"])
            <= float(axis_limit_mm)
        ),
    }
    return {"checks": checks, "gate_pass": bool(all(checks.values()))}


def _train_one_static_model(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    environment: Any,
    config: Mapping[str, Any],
    seed: int,
    directory: Path,
) -> dict[str, Any]:
    import tensorflow as tf

    from quasi_exp.teacher.student_tracking_tf import (
        StudentGeometry,
        build_static_model,
        compile_student,
        packed_targets,
    )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    geometry = StudentGeometry(
        lengths_m=environment.lengths_m,
        p_end_local_m=environment.p_end_local_m,
        theta_sign=float(environment.theta_sign),
        beta_bounds_rad=environment.bounds,
    )
    train_xyz = train[list(XYZ_COLUMNS)].to_numpy(dtype=np.float32)
    validation_xyz = validation[list(XYZ_COLUMNS)].to_numpy(dtype=np.float32)
    model = build_static_model(
        train_xyz,
        geometry=geometry,
        output_mode="tanh",
        hidden_units=tuple(int(value) for value in config["representation"]["static_hidden_units"]),
    )
    compile_student(
        model,
        geometry=geometry,
        lambda_fk=float(config["training"]["lambda_fk"]),
        learning_rate=float(config["training"]["learning_rate"]),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["training"]["patience"]),
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        train_xyz,
        packed_targets(train),
        validation_data=(validation_xyz, packed_targets(validation)),
        epochs=int(config["training"]["max_epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.keras"
    model.save(model_path)
    prediction = np.asarray(
        model.predict(
            validation_xyz,
            batch_size=max(256, int(config["training"]["batch_size"])),
            verbose=0,
        ),
        dtype=float,
    )
    achieved = environment.fk(prediction)
    semiaxis = float(validation.get("major_semiaxis_m", pd.Series([0.46])).min())
    max_limit = min(
        float(config["gates"]["student"]["absolute_max_mm"]),
        1000.0
        * float(config["gates"]["student"]["relative_max_fraction"])
        * semiaxis,
    )
    metrics = _tracking_metrics(validation_xyz, achieved, max_limit_mm=max_limit)
    gate = _student_gate(
        metrics,
        p95_limit_mm=float(config["gates"]["student"]["interpolation_p95_mm"]),
        axis_limit_mm=float(config["gates"]["student"]["per_axis_mean_mm"]),
    )
    report = {
        "seed": int(seed),
        "epochs": int(len(history.history.get("loss", []))),
        "best_epoch": int(np.argmin(history.history.get("val_loss", [math.inf])) + 1),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "validation_metrics": metrics,
        "checks": gate["checks"],
        "gate_pass": gate["gate_pass"],
        "model_sha256": sha256_file(model_path),
    }
    atomic_write_json(directory / "report.json", report)
    return report


def run_train_stage(
    *, config: Mapping[str, Any], source_root: Path, project_root: Path, output: Path
) -> dict[str, Any]:
    formal_gate = read_valid_gate(output / STAGE_DIRS["formal"] / "gate.json")
    if formal_gate is None or not formal_gate["gate_pass"]:
        raise RuntimeError("training requires a passing formal-dataset gate")
    representation = json.loads(
        (output / STAGE_DIRS["representation"] / "representation_decision.json").read_text(
            encoding="utf-8"
        )
    )["representation"]
    if representation != "static":
        raise RuntimeError(
            f"formal V11 static trainer cannot run for representation={representation}; "
            "the representation gate should have stopped this chain"
        )
    stage = output / STAGE_DIRS["train"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    formal = output / STAGE_DIRS["formal"]
    validation = pd.read_parquet(formal / "validation/D_validation_90k.parquet")
    catalog = _catalog_from_csv(
        output / STAGE_DIRS["pilot"] / "applied_family_catalog.csv",
        seed=int(config["seeds"]["family"]),
    )
    dimension_lookup = catalog.frame.set_index("family_id")["major_semiaxis_m"]
    validation["major_semiaxis_m"] = validation["family_id"].map(dimension_lookup)
    subset_manifest = json.loads(
        (formal / "nested_subset_manifest.json").read_text(encoding="utf-8")
    )
    environment = load_environment(project_root, project_root / str(config["robot_config"]))
    seeds = tuple(int(value) for value in config["seeds"]["formal_train"])
    learning_curve = []
    for subset in subset_manifest:
        train_path = formal / "train" / f"{subset['name']}.parquet"
        train = pd.read_parquet(train_path)
        reports = []
        for seed in seeds:
            directory = stage / "learning_curve" / str(subset["name"]) / f"seed_{seed}"
            cached_report = read_valid_gate(directory / "report.json")
            report = (
                cached_report
                if cached_report is not None and (directory / "model.keras").is_file()
                else _train_one_static_model(
                    train=train,
                    validation=validation,
                    environment=environment,
                    config=config,
                    seed=seed,
                    directory=directory,
                )
            )
            reports.append(report)
        learning_curve.append(
            {
                "name": subset["name"],
                "rows": int(subset["rows"]),
                "seed_pass_count": int(sum(bool(row["gate_pass"]) for row in reports)),
                "median_tracking_p95_mm": float(
                    np.median(
                        [row["validation_metrics"]["tracking_p95_mm"] for row in reports]
                    )
                ),
                "reports": reports,
            }
        )
    atomic_write_json(stage / "learning_curve.json", learning_curve)
    required = int(config["gates"]["student"]["required_seed_passes"])
    passing = [row for row in learning_curve if row["seed_pass_count"] >= required]
    selected = min(passing, key=lambda row: row["rows"]) if passing else None
    final_reports = []
    if selected is not None:
        train = pd.read_parquet(formal / "train" / f"{selected['name']}.parquet")
        final_train = pd.concat([train, validation.drop(columns=["major_semiaxis_m"])], ignore_index=True)
        # Model lock is written before any sealed-test loader is invoked.
        prelock = {
            "representation": "static",
            "selected_subset": selected["name"],
            "selected_rows": int(selected["rows"]),
            "selection_seed_pass_count": int(selected["seed_pass_count"]),
            "hyperparameters": dict(config["training"]),
            "seeds": list(seeds),
            "catalog_fingerprint": catalog.fingerprint,
            "virgin_test_opened": False,
        }
        atomic_write_json(stage / "MODEL_LOCKED.json", prelock)
        for seed in seeds:
            directory = stage / "final" / f"seed_{seed}"
            cached_report = read_valid_gate(directory / "report.json")
            report = (
                cached_report
                if cached_report is not None and (directory / "model.keras").is_file()
                else _train_one_static_model(
                    train=final_train,
                    validation=validation,
                    environment=environment,
                    config=config,
                    seed=seed,
                    directory=directory,
                )
            )
            final_reports.append(report)
        lock = {
            **prelock,
            "final_model_sha256": {
                str(seed): sha256_file(stage / "final" / f"seed_{seed}/model.keras")
                for seed in seeds
            },
        }
        atomic_write_json(stage / "model_lock_manifest.json", lock)
    checks = {
        "at_least_one_nested_size_passes_4_of_5_seeds": bool(selected is not None),
        "model_locked_before_sealed_test": bool((stage / "MODEL_LOCKED.json").is_file()),
        "five_final_train_plus_validation_models_exist": bool(
            selected is not None
            and len(final_reports) == 5
            and all(
                (stage / "final" / f"seed_{seed}/model.keras").is_file()
                for seed in seeds
            )
        ),
        "representation_is_static_single_valued": bool(representation == "static"),
    }
    return _write_gate(
        gate_path,
        checks=checks,
        selected_learning_curve_point=selected,
        final_training_reports=final_reports,
    )


def _evaluation_targets(
    family: EllipseFamilySpec,
    *,
    cross_section: TubeCrossSection,
    phase_count: int,
    tube: Mapping[str, Any],
) -> pd.DataFrame:
    values = family.tube_targets(
        phase_count=int(phase_count),
        normalized_offsets=cross_section.points,
        radial_radius_mm=float(tube["radial_radius_mm"]),
        plane_radius_mm=float(tube["plane_radius_mm"]),
        phase_offset_rad=math.pi / int(phase_count),
    )
    node = np.repeat(np.arange(len(cross_section.points)), int(phase_count))
    phase = np.tile(np.arange(int(phase_count)), len(cross_section.points))
    frame = pd.DataFrame(values.reshape(-1, 3), columns=XYZ_COLUMNS)
    frame["family_id"] = family.family_id
    frame["phase_idx"] = phase
    frame["cross_section_idx"] = node
    frame["cross_section_radius"] = np.linalg.norm(cross_section.points[node], axis=1)
    frame["major_semiaxis_m"] = family.major_semiaxis_m
    return frame


def _category_families(
    family: EllipseFamilySpec, *, category: str
) -> EllipseFamilySpec:
    if category == "family_interpolation":
        return family
    if category == "center_ood":
        return EllipseFamilySpec(
            family_id=f"{family.family_id}:center_ood",
            center_m=family.center_m + 0.015 * family.major_direction,
            major_direction=family.major_direction,
            minor_direction=family.minor_direction,
            major_semiaxis_m=family.major_semiaxis_m,
            minor_semiaxis_m=family.minor_semiaxis_m,
        )
    if category == "plane_ood":
        rotation = Rotation.from_rotvec(np.deg2rad(4.0) * family.major_direction)
        return EllipseFamilySpec(
            family_id=f"{family.family_id}:plane_ood",
            center_m=family.center_m,
            major_direction=rotation.apply(family.major_direction),
            minor_direction=rotation.apply(family.minor_direction),
            major_semiaxis_m=family.major_semiaxis_m,
            minor_semiaxis_m=family.minor_semiaxis_m,
        )
    if category == "axis_ratio_ood":
        ratio = 0.30 if family.axis_ratio >= 0.335 else 0.37
        return EllipseFamilySpec(
            family_id=f"{family.family_id}:ratio_ood",
            center_m=family.center_m,
            major_direction=family.major_direction,
            minor_direction=family.minor_direction,
            major_semiaxis_m=family.major_semiaxis_m,
            minor_semiaxis_m=family.major_semiaxis_m * ratio,
        )
    if category == "exact_0p5m_unseen_family":
        return EllipseFamilySpec(
            family_id=f"{family.family_id}:exact_0p5m",
            center_m=family.center_m,
            major_direction=family.major_direction,
            minor_direction=family.minor_direction,
            major_semiaxis_m=0.5,
            minor_semiaxis_m=0.5 * family.axis_ratio,
        )
    raise ValueError(f"unknown evaluation category: {category}")


def run_evaluate_stage(
    *, config: Mapping[str, Any], source_root: Path, project_root: Path, output: Path
) -> dict[str, Any]:
    train_gate = read_valid_gate(output / STAGE_DIRS["train"] / "gate.json")
    if train_gate is None or not train_gate["gate_pass"]:
        raise RuntimeError("sealed evaluation requires a passing model-lock gate")
    stage = output / STAGE_DIRS["evaluate"]
    gate_path = stage / "gate.json"
    cached = read_valid_gate(gate_path)
    if cached is not None:
        return cached
    stage.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf

    catalog = _catalog_from_csv(
        output / STAGE_DIRS["pilot"] / "applied_family_catalog.csv",
        seed=int(config["seeds"]["family"]),
    )
    selection = pd.read_csv(output / STAGE_DIRS["pilot"] / "selected_families.csv")
    selected_test = tuple(
        selection.loc[selection["role"].eq("virgin_test"), "family_id"].astype(str)
    )
    catalog.authorized_family_ids(
        "sealed_evaluation", unseal_token=catalog.seal_token
    )
    sealed_role_ids = set(
        catalog.frame.loc[
            catalog.frame["role"].eq("virgin_test"), "family_id"
        ].astype(str)
    )
    if not set(selected_test).issubset(sealed_role_ids):
        raise PermissionError("selected test promotion is not in the sealed role")
    tube = _load_selected_tube(output)
    count = int(config["formal"]["sealed_eval_phase_count"])
    cross = TubeCrossSection.master(seed=int(config["seeds"]["audit"])).prefix(
        int(config["formal"]["sealed_eval_cross_section_count"])
    )
    categories = (
        "family_interpolation",
        "center_ood",
        "plane_ood",
        "axis_ratio_ood",
        "exact_0p5m_unseen_family",
    )
    target_sets = {}
    for category in categories:
        target_sets[category] = pd.concat(
            [
                _evaluation_targets(
                    _category_families(catalog.family(family_id), category=category),
                    cross_section=cross,
                    phase_count=count,
                    tube=tube,
                )
                for family_id in selected_test
            ],
            ignore_index=True,
        )
    boundary = target_sets["family_interpolation"]
    target_sets["tube_boundary"] = boundary[
        boundary["cross_section_radius"] >= 1.0 - 1.0e-10
    ].copy()
    auxiliary_family_id = str(
        selection.loc[selection["role"].eq("train"), "family_id"].iloc[0]
    )
    target_sets["phase_interpolation_auxiliary"] = _evaluation_targets(
        catalog.family(auxiliary_family_id),
        cross_section=cross,
        phase_count=count,
        tube=tube,
    )
    required_categories = set(categories) | {"tube_boundary"}
    environment = load_environment(project_root, project_root / str(config["robot_config"]))
    seeds = tuple(int(value) for value in config["seeds"]["formal_train"])
    per_seed = []
    for seed in seeds:
        model = tf.keras.models.load_model(
            output / STAGE_DIRS["train"] / "final" / f"seed_{seed}/model.keras",
            compile=False,
        )
        category_reports = {}
        for category, target_frame in target_sets.items():
            xyz = target_frame[list(XYZ_COLUMNS)].to_numpy(dtype=np.float32)
            beta = np.asarray(model.predict(xyz, batch_size=2048, verbose=0), dtype=float)
            achieved = environment.fk(beta)
            out_of_bounds_count = int(
                np.count_nonzero(
                    (beta < environment.bounds[:, 0][None, :] - 1.0e-9)
                    | (beta > environment.bounds[:, 1][None, :] + 1.0e-9)
                )
            )
            max_limit = min(
                float(config["gates"]["student"]["absolute_max_mm"]),
                1000.0
                * float(config["gates"]["student"]["relative_max_fraction"])
                * float(target_frame["major_semiaxis_m"].min()),
            )
            metrics = _tracking_metrics(xyz, achieved, max_limit_mm=max_limit)
            p95_limit = (
                float(config["gates"]["student"]["interpolation_p95_mm"])
                if category
                in {
                    "family_interpolation",
                    "tube_boundary",
                    "phase_interpolation_auxiliary",
                }
                else float(config["gates"]["student"]["near_ood_p95_mm"])
            )
            gate = _student_gate(
                metrics,
                p95_limit_mm=p95_limit,
                axis_limit_mm=float(config["gates"]["student"]["per_axis_mean_mm"]),
            )
            gate["checks"]["within_bounds"] = bool(out_of_bounds_count == 0)
            gate["gate_pass"] = bool(all(gate["checks"].values()))
            category_reports[category] = {
                **metrics,
                **gate,
                "out_of_bounds_count": out_of_bounds_count,
            }
        seed_pass = bool(
            all(category_reports[name]["gate_pass"] for name in required_categories)
        )
        per_seed.append(
            {"seed": seed, "categories": category_reports, "all_category_pass": seed_pass}
        )
    atomic_write_json(stage / "sealed_evaluation.json", per_seed)
    # The 0.75 m artifact is opened only after model lock and is never part of
    # any check above.  It remains a descriptive scale stress test.
    stress_path = (
        project_root
        / str(config["v10_evidence_root"])
        / "a0p750m/pose_targets.parquet"
    )
    stress = pd.read_parquet(stress_path)
    stress_xyz = stress[["target_x_m", "target_y_m", "target_z_m"]].to_numpy(dtype=np.float32)
    stress_reports = []
    for seed in seeds:
        model = tf.keras.models.load_model(
            output / STAGE_DIRS["train"] / "final" / f"seed_{seed}/model.keras",
            compile=False,
        )
        achieved = environment.fk(
            np.asarray(model.predict(stress_xyz, batch_size=2048, verbose=0), dtype=float)
        )
        stress_reports.append(
            {
                "seed": seed,
                **_tracking_metrics(stress_xyz, achieved, max_limit_mm=1.0e9),
            }
        )
    atomic_write_json(
        stage / "scale_stress_0p75m.json",
        {
            "used_for_model_selection": False,
            "known_teacher_gap_count": 22,
            "reports": stress_reports,
        },
    )
    pass_count = int(sum(row["all_category_pass"] for row in per_seed))
    checks = {
        "sealed_test_opened_after_model_lock": bool(
            (output / STAGE_DIRS["train"] / "MODEL_LOCKED.json").is_file()
        ),
        "all_required_categories_reported": bool(
            all(required_categories.issubset(set(row["categories"])) for row in per_seed)
        ),
        "same_seed_passes_all_categories_for_at_least_4_of_5": bool(
            pass_count >= int(config["gates"]["student"]["required_seed_passes"])
        ),
        "zero_out_of_bounds_by_construction": bool(
            all(
                int(category["out_of_bounds_count"]) == 0
                for seed_report in per_seed
                for category in seed_report["categories"].values()
            )
        ),
        "0p75m_used_only_after_lock_as_stress": True,
    }
    return _write_gate(
        gate_path,
        checks=checks,
        seed_all_category_pass_count=pass_count,
        seed_reports=per_seed,
        scale_stress_file="scale_stress_0p75m.json",
    )


def run_verify_stage(*, output: Path) -> dict[str, Any]:
    required = tuple(
        Path(STAGE_DIRS[stage]) / "gate.json"
        for stage in STAGE_DIRS
    )
    return finalize_experiment(output, required_gate_files=required)


STAGE_RUNNERS = {
    "protocol": run_protocol_stage,
    "anchor": run_anchor_stage,
    "tube": run_tube_stage,
    "core": run_core_stage,
    "pilot": run_pilot_stage,
    "representation": run_representation_stage,
    "formal": run_formal_stage,
    "train": run_train_stage,
    "evaluate": run_evaluate_stage,
}


def parse_args() -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    project_root = project_root_from(source_root)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--preset", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--config",
        type=Path,
        default=source_root / "configs/generalized_ellipse_region_v11.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[2]
    project_root = Path(args.project_root).resolve()
    config = load_protocol_config(args.config, preset=args.preset)
    output = (
        Path(args.output).resolve()
        if args.output is not None
        else project_root / str(config["output_root"])
    )
    if (output / "COMPLETED").exists():
        raise RuntimeError("V11 output is finalized; use a fresh output root")
    common = {
        "config": config,
        "source_root": source_root,
        "project_root": project_root,
        "output": output,
    }
    if args.stage == "verify":
        report = run_verify_stage(output=output)
    elif args.stage in STAGE_RUNNERS:
        report = STAGE_RUNNERS[args.stage](**common)
    else:
        reports = {}
        for stage_name in STAGE_DIRS:
            report = STAGE_RUNNERS[stage_name](**common)
            reports[stage_name] = report
            if not bool(report["gate_pass"]):
                reports["stopped_after"] = stage_name
                reports["stop_reason"] = "hard_gate_failed"
                print(json.dumps(reports, indent=2, allow_nan=False))
                return
        reports["verify"] = run_verify_stage(output=output)
        report = reports
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
