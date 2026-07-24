#!/usr/bin/env python3
"""Run V11.4 viability-first, cycle-first inverse-branch discovery.

The protocol consumes the immutable V11.3 evidence, maps the root IK fibre,
tests bidirectional continuation survival, builds hard-feasible candidate
layers without raw padding, and selects a closed cycle before applying any
canonical posture cost.  Strict formal failure terminates before tube/data/
student work; strict formal success emits a bridge to the frozen V11 pipeline.
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

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.stats import qmc
import yaml

from quasi_exp.teacher.branch_identity import (
    BETA_COLUMNS,
    analyze_branch_variants,
    select_canonical_root_phase,
)
from quasi_exp.teacher.canonical import (
    TeacherPolicy,
    TeacherVariant,
    _correct_target,
    _environment_jacobian,
)
from quasi_exp.teacher.experiment import atomic_write_json, sha256_file
from quasi_exp.teacher.full_loop_feasibility import (
    StrictCycleGate,
    ValidatedRootGraph,
    candidate_search_budgets,
    classify_full_loop_outcome,
    cluster_full_loop_cycles,
    connectivity_sensitivity,
    filter_hard_feasible_candidates,
    formal_decision,
    geodesic_farthest_representatives,
    rank_viability,
    solve_sparse_cycle,
    strict_cycle_metrics,
    validated_mutual_knn_graph,
    weighted_rms_gap_deg,
)
from quasi_exp.teacher.large_scale import ReachabilityAtlas


if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_generalized_ellipse_branch_identity_v11_3 as v113
import run_generalized_ellipse_region_v11 as v11
from run_trajectory_canonical_teacher_v10 import load_environment, runtime_fingerprint


STAGE_DIRS = {
    "protocol": "00_protocol",
    "root_fiber": "01_root_fiber",
    "viability": "02_viability",
    "pilot": "03_pilot_matrix",
    "formal": "04_formal",
    "bridge": "05_downstream_bridge",
}

CANDIDATE_EVIDENCE_COLUMNS = (
    "candidate_source",
    "candidate_beta",
    "corrector_success",
    "residual_mm",
    "joint_margin_deg",
    "within_bounds",
    "reference_gap_deg",
    "previous_gap_deg",
    "trust_pass",
    "kappa",
)

ROOT_REPRESENTATIVE_COLUMNS = (
    "root_seed_count",
    "representative_idx",
    "root_seed_idx",
)


def representative_inventory_table(
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Materialize a stable root-representative CSV schema, including empties."""

    return pd.DataFrame(rows, columns=ROOT_REPRESENTATIVE_COLUMNS)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def load_config(path: str | Path, *, preset: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V11.4 config must contain a mapping")
    normalized = str(preset).lower()
    if normalized not in {"smoke", "formal"}:
        raise ValueError("preset must be smoke or formal")
    if normalized == "smoke":
        payload = _deep_merge(payload, payload.get("smoke", {}))
    payload.pop("smoke", None)
    payload["preset"] = normalized
    payload["config_path"] = str(config_path)
    return payload


def project_root_from(source_root: Path) -> Path:
    return v113.project_root_from(source_root)


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


def _write_gate(
    path: Path, *, checks: Mapping[str, bool], **evidence: Any
) -> dict[str, Any]:
    normalized = {str(name): bool(value) for name, value in checks.items()}
    artifacts = {
        item.relative_to(path.parent).as_posix(): sha256_file(item)
        for item in sorted(path.parent.rglob("*"))
        if item.is_file() and item.resolve() != path.resolve()
    }
    payload = {
        **evidence,
        "checks": normalized,
        "gate_pass": bool(all(normalized.values())),
        "artifact_sha256": artifacts,
    }
    atomic_write_json(path, payload)
    return payload


def _directory_artifact_manifest(
    directory: Path, *, exclude: Sequence[Path] = ()
) -> dict[str, str]:
    excluded = {value.resolve() for value in exclude}
    return {
        item.relative_to(directory).as_posix(): sha256_file(item)
        for item in sorted(directory.rglob("*"))
        if item.is_file() and item.resolve() not in excluded
    }


def _artifact_manifest_is_valid(
    directory: Path, manifest: Mapping[str, str]
) -> bool:
    return bool(manifest) and all(
        (directory / relative).is_file()
        and sha256_file(directory / relative) == str(expected)
        for relative, expected in manifest.items()
    )


def root_evidence_payload(
    *,
    candidate_id: str,
    phase_idx: int,
    selected_row: Mapping[str, Any],
    primary_row: Mapping[str, Any],
    selected_beta: Sequence[float],
    primary_beta: Sequence[float],
) -> dict[str, Any]:
    """Keep selected-root evidence separate from primary-root evidence."""

    return {
        "candidate_id": str(candidate_id),
        "phase_idx": int(phase_idx),
        "selected_root_beta": list(map(float, selected_beta)),
        "selected_root_residual_mm": float(selected_row["residual_mm"]),
        "selected_root_margin_deg": float(selected_row["joint_margin_deg"]),
        "selected_root_kappa": float(selected_row["kappa"]),
        "primary_root_beta": list(map(float, primary_beta)),
        "primary_root_residual_mm": float(primary_row["residual_mm"]),
        "primary_root_margin_deg": float(primary_row["joint_margin_deg"]),
        "primary_root_kappa": float(primary_row["kappa"]),
    }


def _source_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_artifact_root": str(config["source_anchor_root"]),
    }


def _load_source_frames(
    config: Mapping[str, Any], project_root: Path, candidate_id: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return v113._load_source_frames(  # noqa: SLF001 - protocol extension seam
        _source_config(config), project_root, candidate_id
    )


def _aligned_source_frames(
    config: Mapping[str, Any],
    project_root: Path,
    candidate_id: str,
    *,
    count: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    primary, variants = _load_source_frames(config, project_root, candidate_id)
    return v113._aligned_source_variants(  # noqa: SLF001
        primary, variants, count=int(count)
    )


def _frame_beta(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(BETA_COLUMNS)].to_numpy(float)


def _frame_target(frame: pd.DataFrame) -> np.ndarray:
    return frame[["target_x_m", "target_y_m", "target_z_m"]].to_numpy(float)


def _teacher_policy(config: Mapping[str, Any], *, seed: int | None = None) -> TeacherPolicy:
    values = dict(config["teacher_policy"])
    values["variant"] = TeacherVariant(str(values.get("variant", "T3")))
    if seed is not None:
        values["solver_seed"] = int(seed)
    return TeacherPolicy(**values)


def _strict_gate(config: Mapping[str, Any]) -> StrictCycleGate:
    return StrictCycleGate(**dict(config["strict_cycle_gate"]))


def _relaxed_gate(config: Mapping[str, Any]) -> StrictCycleGate:
    return StrictCycleGate(**dict(config["diagnostic_relaxed_gate"]))


def _reachability_atlas(
    project_root: Path, config: Mapping[str, Any]
) -> ReachabilityAtlas:
    return v113._reachability_atlas(  # noqa: SLF001
        project_root,
        {
            "v10_evidence_root": config["v10_evidence_root"],
        },
    )


def _beta_metrics(
    environment: Any, target: np.ndarray, beta: np.ndarray
) -> dict[str, Any]:
    value = np.asarray(beta, dtype=float).reshape(6)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    achieved = np.asarray(environment.fk(value.reshape(1, 6)), dtype=float).reshape(
        -1, 3
    )[0]
    residual = float(
        np.linalg.norm(achieved - np.asarray(target, dtype=float).reshape(3))
        * 1000.0
    )
    margin = float(
        np.rad2deg(
            np.min(
                np.minimum(value - bounds[:, 0], bounds[:, 1] - value)
            )
        )
    )
    within = bool(
        np.all(value >= bounds[:, 0] - 1.0e-12)
        and np.all(value <= bounds[:, 1] + 1.0e-12)
    )
    singular = np.linalg.svd(
        _environment_jacobian(environment, value), compute_uv=False
    )
    kappa = float(singular[0] / max(float(singular[-1]), 1.0e-12))
    return {
        "residual_mm": residual,
        "joint_margin_deg": margin,
        "within_bounds": within,
        "kappa": kappa,
    }


def _candidate_record(
    *,
    environment: Any,
    target: np.ndarray,
    beta: np.ndarray,
    candidate_source: str,
    corrector_success: bool,
    reference_beta: np.ndarray | None = None,
    previous_beta: np.ndarray | None = None,
    trust_radius_deg: float = math.inf,
) -> dict[str, Any]:
    metrics = _beta_metrics(environment, target, beta)
    reference_gap = (
        0.0
        if reference_beta is None
        else float(weighted_rms_gap_deg(beta, reference_beta))
    )
    previous_gap = (
        0.0
        if previous_beta is None
        else float(weighted_rms_gap_deg(beta, previous_beta))
    )
    return {
        "candidate_source": str(candidate_source),
        "candidate_beta": json.dumps(np.asarray(beta, dtype=float).tolist()),
        "corrector_success": bool(corrector_success),
        **metrics,
        "reference_gap_deg": reference_gap,
        "previous_gap_deg": previous_gap,
        "trust_pass": bool(reference_gap <= float(trust_radius_deg) + 1.0e-12),
    }


def _correct_seed(
    *,
    environment: Any,
    target: np.ndarray,
    seed: np.ndarray,
    policy: TeacherPolicy,
    candidate_source: str,
    reference_beta: np.ndarray | None = None,
    previous_beta: np.ndarray | None = None,
    trust_radius_deg: float = math.inf,
) -> tuple[np.ndarray, dict[str, Any]]:
    beta, _residual, iterations, success = _correct_target(
        environment, target, seed, policy
    )
    row = _candidate_record(
        environment=environment,
        target=target,
        beta=beta,
        candidate_source=candidate_source,
        corrector_success=bool(success),
        reference_beta=reference_beta,
        previous_beta=previous_beta,
        trust_radius_deg=trust_radius_deg,
    )
    row["corrector_iterations"] = int(iterations)
    return beta, row


def _root_phase(
    environment: Any, primary: pd.DataFrame, variants: Mapping[str, pd.DataFrame]
) -> int:
    report = analyze_branch_variants(
        primary,
        variants,
        jacobian=lambda beta: environment.jacobian(beta),
        cluster_threshold_deg=0.5,
    )
    return int(
        select_canonical_root_phase(
            primary, branch_clusters=report.branch_clusters
        )["phase_idx"]
    )


def _round_robin_sources(
    banks: Mapping[str, Sequence[np.ndarray]], total: int
) -> list[tuple[str, np.ndarray]]:
    names = tuple(banks)
    positions = {name: 0 for name in names}
    output: list[tuple[str, np.ndarray]] = []
    while len(output) < int(total):
        progressed = False
        for name in names:
            index = positions[name]
            if index >= len(banks[name]):
                continue
            output.append((name, np.asarray(banks[name][index], dtype=float)))
            positions[name] += 1
            progressed = True
            if len(output) >= int(total):
                break
        if not progressed:
            raise ValueError("root seed source quotas do not cover requested budget")
    return output


def _root_seed_bank(
    *,
    config: Mapping[str, Any],
    environment: Any,
    atlas: ReachabilityAtlas,
    target: np.ndarray,
    canonical_primary_beta: np.ndarray,
    max_count: int,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    quotas = {
        name: int(value)
        for name, value in config["root_fiber"]["source_quotas"].items()
    }
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    tree = cKDTree(atlas.xyz_m)
    atlas_count = min(quotas["atlas"], len(atlas.xyz_m))
    _distance, indices = tree.query(target, k=max(atlas_count, 1))
    atlas_values = atlas.beta_rad[np.atleast_1d(indices)[:atlas_count]]
    sobol_count = quotas["sobol"]
    sobol = qmc.Sobol(d=6, scramble=True, seed=int(seed))
    sobol_unit = sobol.random(sobol_count)
    sobol_values = qmc.scale(sobol_unit, bounds[:, 0], bounds[:, 1])
    jacobian = _environment_jacobian(environment, canonical_primary_beta)
    _u, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * max(
        float(singular[0]), 1.0
    )
    null_basis = vh[int(np.sum(singular > tolerance)) :]
    if len(null_basis) == 0:
        null_basis = vh[-3:]
    null_values = []
    for index in range(quotas["nullspace"]):
        basis = null_basis[index % len(null_basis)]
        shell = 1 + index // (2 * len(null_basis))
        sign = 1.0 if (index // len(null_basis)) % 2 == 0 else -1.0
        null_values.append(
            np.clip(
                canonical_primary_beta + sign * math.radians(0.5 * shell) * basis,
                bounds[:, 0],
                bounds[:, 1],
            )
        )
    rng = np.random.default_rng(int(seed) + 1)
    predictor_values = []
    walk = np.asarray(canonical_primary_beta, dtype=float).copy()
    for _index in range(quotas["predictor_walk"]):
        coefficients = rng.normal(size=len(null_basis))
        coefficients /= max(float(np.linalg.norm(coefficients)), 1.0e-12)
        walk = np.clip(
            walk + math.radians(0.5) * coefficients @ null_basis,
            bounds[:, 0],
            bounds[:, 1],
        )
        predictor_values.append(walk.copy())
    banks = {
        "primary": [np.asarray(canonical_primary_beta, dtype=float)],
        "atlas": list(atlas_values),
        "sobol": list(sobol_values),
        "nullspace": null_values,
        "predictor_walk": predictor_values,
    }
    return _round_robin_sources(banks, int(max_count))


def run_protocol(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    output: Path,
    **_unused: Any,
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["protocol"]
    stage.mkdir(parents=True, exist_ok=True)
    source_completion = (
        project_root
        / str(config["source_artifact_root"])
        / "BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json"
    )
    source_payload = json.loads(source_completion.read_text(encoding="utf-8"))
    resolved = copy.deepcopy(dict(config))
    resolved.pop("config_path", None)
    protocol_path = stage / "protocol_v11_4.yaml"
    protocol_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    inputs = {
        "config": sha256_file(Path(str(config["config_path"]))),
        "runner": sha256_file(Path(__file__).resolve()),
        "full_loop_module": sha256_file(
            source_root / "src/quasi_exp/teacher/full_loop_feasibility.py"
        ),
        "source_completion": sha256_file(source_completion),
    }
    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "source_protocol_id": str(config["source_protocol_id"]),
        "input_sha256": inputs,
        "runtime": runtime_fingerprint(),
        "manifest_sha256": _canonical_sha(inputs),
    }
    atomic_write_json(stage / "artifact_manifest.json", manifest)
    return _write_gate(
        stage / "gate.json",
        checks={
            "source_v11_3_complete": bool(
                source_payload["branch_identity_experiment_complete"]
            ),
            "strict_gate_unchanged": dict(config["strict_cycle_gate"])
            == {
                "residual_p95_mm": 1.0,
                "residual_max_mm": 3.0,
                "joint_margin_min_deg": 1.5,
                "delta_beta_rms_p95_deg": 1.0,
                "delta_beta_rms_max_deg": 2.0,
                "acceleration_beta_rms_p95_deg": 0.5,
                "seam_beta_rms_deg": 1.0,
            },
            "raw_seed_padding_prohibited": bool(
                config["protocol_constraints"]["prohibit_raw_seed_padding"]
            ),
            "static_conflict_fail_closed": (
                config["downstream"]["static_mapping_conflict_policy"]
                == "fail_closed"
            ),
        },
        protocol_id=str(config["protocol_id"]),
        source_protocol_id=str(config["source_protocol_id"]),
        manifest_sha256=manifest["manifest_sha256"],
    )


def run_root_fiber(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    atlas: ReachabilityAtlas,
    output: Path,
    **_unused: Any,
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["root_fiber"]
    nested = list(map(int, config["root_fiber"]["nested_seed_counts"]))
    max_count = max(nested)
    summaries: list[dict[str, Any]] = []
    for candidate_offset, candidate_id in enumerate(map(str, config["candidates"])):
        primary, variants = _aligned_source_frames(
            config,
            project_root,
            candidate_id,
            count=int(config["phase_counts"]["pilot"]),
        )
        phase = _root_phase(environment, primary, variants)
        target = _frame_target(primary)[phase]
        primary_beta = _frame_beta(primary)[phase]
        source_root = json.loads(
            (
                project_root
                / str(config["source_artifact_root"])
                / "01_canonical_root"
                / candidate_id
                / "canonical_root.json"
            ).read_text(encoding="utf-8")
        )
        source_pilot_count = 180
        source_phase = int(
            round(int(source_root["phase_idx"]) * len(primary) / source_pilot_count)
        ) % len(primary)
        if source_phase != phase:
            phase = source_phase
            target = _frame_target(primary)[phase]
            primary_beta = _frame_beta(primary)[phase]
        canonical_primary_beta, _canonical_primary_record = _correct_seed(
            environment=environment,
            target=target,
            seed=np.asarray(source_root["canonical_root_beta"], dtype=float),
            policy=_teacher_policy(
                config, seed=int(config["seeds"]["root"]) + candidate_offset
            ),
            candidate_source="primary",
            reference_beta=primary_beta,
        )
        policy = _teacher_policy(
            config, seed=int(config["seeds"]["root"]) + candidate_offset
        )
        seeds = _root_seed_bank(
            config=config,
            environment=environment,
            atlas=atlas,
            target=target,
            canonical_primary_beta=canonical_primary_beta,
            max_count=max_count,
            seed=int(config["seeds"]["root"]) + candidate_offset,
        )
        solved: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for seed_index, (source, seed_beta) in enumerate(seeds):
            beta, row = _correct_seed(
                environment=environment,
                target=target,
                seed=seed_beta,
                policy=policy,
                candidate_source=source,
                reference_beta=primary_beta,
            )
            row["root_seed_idx"] = seed_index
            for joint in range(6):
                row[f"beta{joint + 1}_rad"] = float(beta[joint])
            solved.append(beta)
            rows.append(row)
        beta_all = np.vstack(solved)
        table = pd.DataFrame(rows)
        candidate_dir = stage / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(table, candidate_dir / "root_candidates.parquet")
        sensitivity_rows = []
        nested_summary = []
        for count in nested:
            prefix = table.iloc[:count]
            solved_mask = (
                prefix["corrector_success"].astype(bool).to_numpy()
                & prefix["within_bounds"].astype(bool).to_numpy()
                & (
                    prefix["residual_mm"].to_numpy(float)
                    <= float(config["hard_feasibility"]["residual_max_mm"])
                )
            )
            beta = beta_all[:count][solved_mask]
            if len(beta):
                sensitivity = connectivity_sensitivity(
                    beta,
                    thresholds_deg=config["root_fiber"][
                        "connectivity_eps_deg"
                    ],
                    beta_weights=config["cycle"]["beta_weights"],
                )
            else:
                sensitivity = pd.DataFrame(
                    {
                        "epsilon_deg": config["root_fiber"][
                            "connectivity_eps_deg"
                        ],
                        "connected_component_count": 0,
                        "largest_component_size": 0,
                    }
                )
            sensitivity["root_seed_count"] = count
            sensitivity_rows.append(sensitivity)
            nested_summary.append(
                {
                    "root_seed_count": count,
                    "corrector_success_count": int(np.count_nonzero(solved_mask)),
                    "component_count_at_0p5deg": int(
                        sensitivity.loc[
                            np.isclose(sensitivity["epsilon_deg"], 0.5),
                            "connected_component_count",
                        ].iloc[0]
                    ),
                }
            )
        sensitivity_all = pd.concat(sensitivity_rows, ignore_index=True)
        sensitivity_all.to_csv(
            candidate_dir / "connectivity_sensitivity.csv", index=False
        )
        hard_mask = (
            table["corrector_success"].astype(bool).to_numpy()
            & (table["residual_mm"].to_numpy(float) <= 3.0)
            & (table["joint_margin_deg"].to_numpy(float) >= 1.5)
            & table["within_bounds"].astype(bool).to_numpy()
        )
        hard_indices_raw = np.flatnonzero(hard_mask)
        hard_indices_list: list[int] = []
        for index in hard_indices_raw:
            if any(
                float(
                    weighted_rms_gap_deg(
                        beta_all[index],
                        beta_all[previous],
                        beta_weights=config["cycle"]["beta_weights"],
                    )
                )
                < float(
                    config["hard_feasibility"]["distinct_threshold_deg"]
                )
                - 1.0e-12
                for previous in hard_indices_list
            ):
                continue
            hard_indices_list.append(int(index))
        hard_indices = np.asarray(hard_indices_list, dtype=np.int64)
        hard_beta = beta_all[hard_indices]

        def interpolation_is_feasible(
            left: np.ndarray, right: np.ndarray, fraction: float
        ) -> bool:
            seed_beta = (1.0 - fraction) * left + fraction * right
            corrected, row = _correct_seed(
                environment=environment,
                target=target,
                seed=seed_beta,
                policy=policy,
                candidate_source="geodesic_interpolation",
            )
            return bool(
                row["corrector_success"]
                and row["residual_mm"]
                <= float(config["hard_feasibility"]["residual_max_mm"])
                and row["joint_margin_deg"]
                >= float(config["hard_feasibility"]["joint_margin_min_deg"])
                and row["within_bounds"]
                and float(
                    weighted_rms_gap_deg(
                        corrected,
                        seed_beta,
                        beta_weights=config["cycle"]["beta_weights"],
                    )
                )
                <= float(
                    config["hard_feasibility"]["distinct_threshold_deg"]
                )
                + 1.0e-12
            )

        graph = (
            validated_mutual_knn_graph(
                hard_beta,
                k=int(config["root_fiber"]["mutual_knn_k"]),
                max_edge_deg=float(
                    config["root_fiber"]["geodesic_max_edge_deg"]
                ),
                interpolation_fractions=config["root_fiber"][
                    "interpolation_fractions"
                ],
                interpolation_is_feasible=interpolation_is_feasible,
                beta_weights=config["cycle"]["beta_weights"],
            )
            if len(hard_beta)
            else ValidatedRootGraph(
                edge_table=pd.DataFrame(
                    columns=[
                        "left_idx",
                        "right_idx",
                        "distance_deg",
                        "validated",
                        "interpolation_fraction_count",
                    ]
                ),
                component_labels=np.zeros(0, dtype=np.int64),
                connected_component_count=0,
            )
        )
        graph.edge_table.to_csv(candidate_dir / "validated_geodesic_edges.csv", index=False)
        direct_geodesic = np.full(
            (len(hard_beta), len(hard_beta)), np.inf, dtype=float
        )
        np.fill_diagonal(direct_geodesic, 0.0)
        for edge in graph.edge_table.itertuples(index=False):
            direct_geodesic[int(edge.left_idx), int(edge.right_idx)] = float(
                edge.distance_deg
            )
            direct_geodesic[int(edge.right_idx), int(edge.left_idx)] = float(
                edge.distance_deg
            )
        graph_nodes = pd.DataFrame(
            {
                "hard_candidate_idx": np.arange(len(hard_beta), dtype=np.int64),
                "root_seed_idx": hard_indices,
                "geodesic_component_id": graph.component_labels,
            }
        )
        graph_nodes.to_csv(candidate_dir / "validated_geodesic_nodes.csv", index=False)
        primary_hard_position = (
            int(np.flatnonzero(hard_indices == 0)[0])
            if np.any(hard_indices == 0)
            else 0
        )
        representative_rows: list[dict[str, Any]] = []
        representative_seed_indices = np.zeros(0, dtype=np.int64)
        for count in sorted(
            set(
                [
                    *nested,
                    *map(int, config["pilot_matrix"]["root_seed_counts"]),
                ]
            )
        ):
            prefix_hard_indices = hard_indices[hard_indices < count]
            prefix_beta = beta_all[prefix_hard_indices]
            prefix_positions = np.flatnonzero(hard_indices < count)
            prefix_geodesic = (
                shortest_path(
                    direct_geodesic[
                        np.ix_(prefix_positions, prefix_positions)
                    ],
                    directed=False,
                )
                if len(prefix_positions)
                else np.zeros((0, 0))
            )
            required_position = (
                int(np.flatnonzero(prefix_hard_indices == 0)[0])
                if np.any(prefix_hard_indices == 0)
                else 0
            )
            local = geodesic_farthest_representatives(
                prefix_beta,
                max_count=int(config["root_fiber"]["representative_count"]),
                required_indices=[required_position] if len(prefix_beta) else [],
                beta_weights=config["cycle"]["beta_weights"],
                geodesic_distance_deg=prefix_geodesic,
            )
            selected_indices = prefix_hard_indices[local]
            if count == max_count:
                representative_seed_indices = selected_indices
            representative_rows.extend(
                {
                    "root_seed_count": int(count),
                    "representative_idx": int(index),
                    "root_seed_idx": int(seed_index),
                }
                for index, seed_index in enumerate(selected_indices)
            )
        representative_inventory_table(representative_rows).to_csv(
            candidate_dir / "root_representatives.csv", index=False
        )
        selection_pool = (
            table.loc[hard_indices]
            if len(hard_indices)
            else table.sort_values(
                ["corrector_success", "residual_mm"],
                ascending=[False, True],
                kind="stable",
            ).head(1)
        )
        selected_seed_idx = int(
            selection_pool.sort_values(
                ["joint_margin_deg", "residual_mm", "kappa"],
                ascending=[False, True, True],
                kind="stable",
            ).index[0]
        )
        selected_row = table.iloc[selected_seed_idx]
        primary_row = _candidate_record(
            environment=environment,
            target=target,
            beta=primary_beta,
            candidate_source="source_centerline_primary",
            corrector_success=True,
        )
        payload = root_evidence_payload(
            candidate_id=candidate_id,
            phase_idx=phase,
            selected_row=selected_row,
            primary_row=primary_row,
            selected_beta=beta_all[selected_seed_idx],
            primary_beta=primary_beta,
        )
        payload.update(
            {
                "root_seed_count": max_count,
                "hard_feasible_root_count": int(len(hard_indices)),
                "validated_geodesic_component_count": int(
                    graph.connected_component_count
                ),
                "representative_root_seed_indices": representative_seed_indices.tolist(),
                "nested_summary": nested_summary,
            }
        )
        atomic_write_json(candidate_dir / "root_fiber_summary.json", payload)
        summaries.append(payload)
    return _write_gate(
        stage / "gate.json",
        checks={
            "two_candidates_mapped": len(summaries) == 2,
            "nested_seed_inventory_complete": all(
                row["root_seed_count"] == max_count for row in summaries
            ),
            "hard_root_inventory_recorded_for_each_candidate": all(
                "hard_feasible_root_count" in row for row in summaries
            ),
            "representative_inventory_recorded_for_each_candidate": all(
                "representative_root_seed_indices" in row for row in summaries
            ),
        },
        candidates=summaries,
        root_seed_counts=nested,
        connectivity_eps_deg=list(
            map(float, config["root_fiber"]["connectivity_eps_deg"])
        ),
    )


def _lineage_order(phase_count: int, root_phase: int, direction: str) -> np.ndarray:
    step = 1 if str(direction) == "forward" else -1
    return (
        int(root_phase) + step * np.arange(int(phase_count), dtype=np.int64)
    ) % int(phase_count)


def _continuation_seed_bank(
    *,
    environment: Any,
    target: np.ndarray,
    previous_target: np.ndarray,
    previous_beta: np.ndarray,
    before_beta: np.ndarray | None,
    reference_beta: np.ndarray,
    candidate_budget: int,
    rng: np.random.Generator,
    local_seed_std_deg: float,
) -> list[tuple[str, np.ndarray]]:
    seeds: list[tuple[str, np.ndarray]] = [
        ("lineage_previous", np.asarray(previous_beta, dtype=float)),
        ("source_reference", np.asarray(reference_beta, dtype=float)),
    ]
    if before_beta is not None:
        seeds.append(
            (
                "secant_predictor",
                np.asarray(previous_beta)
                + (np.asarray(previous_beta) - np.asarray(before_beta)),
            )
        )
    jacobian = _environment_jacobian(environment, previous_beta)
    pinv = np.linalg.pinv(jacobian, rcond=1.0e-10)
    seeds.append(
        (
            "jacobian_predictor",
            np.asarray(previous_beta)
            + pinv
            @ (
                np.asarray(target, dtype=float)
                - np.asarray(previous_target, dtype=float)
            ),
        )
    )
    _u, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * max(
        float(singular[0]), 1.0
    )
    null_basis = vh[int(np.sum(singular > tolerance)) :]
    while len(seeds) < int(candidate_budget):
        if len(null_basis):
            coefficients = rng.normal(size=len(null_basis))
            coefficients /= max(float(np.linalg.norm(coefficients)), 1.0e-12)
            perturbation = coefficients @ null_basis
            source = "nullspace_local"
        else:
            perturbation = rng.normal(size=6)
            perturbation /= max(float(np.linalg.norm(perturbation)), 1.0e-12)
            source = "full_beta_local"
        seeds.append(
            (
                source,
                np.asarray(previous_beta)
                + math.radians(float(local_seed_std_deg)) * perturbation,
            )
        )
    return seeds[: int(candidate_budget)]


def _solve_lineage(
    *,
    config: Mapping[str, Any],
    environment: Any,
    target: np.ndarray,
    reference: np.ndarray,
    root_phase: int,
    root_beta: np.ndarray,
    root_seed_idx: int,
    direction: str,
    policy: TeacherPolicy,
    solver_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    count = len(target)
    order = _lineage_order(count, root_phase, direction)
    selected = np.full((count, 6), np.nan)
    selected[int(root_phase)] = np.asarray(root_beta, dtype=float)
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    root_record = _candidate_record(
        environment=environment,
        target=target[int(root_phase)],
        beta=root_beta,
        candidate_source="root_representative",
        corrector_success=True,
        reference_beta=root_beta,
        previous_beta=root_beta,
        trust_radius_deg=float(config["viability"]["trust_radius_deg"]),
    )
    root_record.update(
        {
            "phase_idx": int(root_phase),
            "traversal_idx": 0,
            "selected": True,
            "root_seed_idx": int(root_seed_idx),
            "direction": str(direction),
        }
    )
    candidate_rows.append(root_record.copy())
    selected_rows.append(root_record.copy())
    rng = np.random.default_rng(int(solver_seed))
    failure_phase: int | None = None
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    for traversal_idx in range(1, count):
        phase = int(order[traversal_idx])
        previous_phase = int(order[traversal_idx - 1])
        before_phase = (
            int(order[traversal_idx - 2]) if traversal_idx >= 2 else None
        )
        previous_beta = selected[previous_phase]
        seeds = _continuation_seed_bank(
            environment=environment,
            target=target[phase],
            previous_target=target[previous_phase],
            previous_beta=previous_beta,
            before_beta=(
                selected[before_phase] if before_phase is not None else None
            ),
            reference_beta=reference[phase],
            candidate_budget=int(config["viability"]["candidate_budget"]),
            rng=rng,
            local_seed_std_deg=float(
                config["viability"]["local_seed_std_deg"]
            ),
        )
        phase_beta: list[np.ndarray] = []
        phase_rows: list[dict[str, Any]] = []
        for source, seed_beta in seeds:
            beta, row = _correct_seed(
                environment=environment,
                target=target[phase],
                seed=np.clip(seed_beta, bounds[:, 0], bounds[:, 1]),
                policy=policy,
                candidate_source=source,
                reference_beta=previous_beta,
                previous_beta=previous_beta,
                trust_radius_deg=float(config["viability"]["trust_radius_deg"]),
            )
            row.update(
                {
                    "phase_idx": phase,
                    "traversal_idx": traversal_idx,
                    "selected": False,
                    "root_seed_idx": int(root_seed_idx),
                    "direction": str(direction),
                }
            )
            phase_beta.append(beta)
            phase_rows.append(row)
        phase_table = pd.DataFrame(phase_rows)
        hard = (
            phase_table["corrector_success"].astype(bool).to_numpy()
            & phase_table["within_bounds"].astype(bool).to_numpy()
            & phase_table["trust_pass"].astype(bool).to_numpy()
            & (
                phase_table["residual_mm"].to_numpy(float)
                <= float(config["hard_feasibility"]["residual_max_mm"])
            )
            & (
                phase_table["joint_margin_deg"].to_numpy(float)
                >= float(config["hard_feasibility"]["joint_margin_min_deg"])
            )
        )
        feasible = np.flatnonzero(hard)
        if len(feasible) == 0:
            failure_phase = phase
            candidate_rows.extend(phase_rows)
            break
        choice_table = phase_table.iloc[feasible].sort_values(
            ["previous_gap_deg", "residual_mm", "joint_margin_deg", "kappa"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        chosen = int(choice_table.index[0])
        phase_rows[chosen]["selected"] = True
        selected[phase] = phase_beta[chosen]
        selected_rows.append(phase_rows[chosen].copy())
        candidate_rows.extend(phase_rows)
    solved_table = pd.DataFrame(selected_rows).sort_values(
        "traversal_idx", kind="stable"
    )
    solved_count = len(solved_table)
    complete_inventory = solved_count == count
    seam_deg = (
        float(
            weighted_rms_gap_deg(
                selected[int(order[-1])],
                selected[int(root_phase)],
                beta_weights=config["cycle"]["beta_weights"],
            )
        )
        if complete_inventory
        else math.inf
    )
    completed_360 = bool(
        complete_inventory
        and seam_deg
        <= float(config["strict_cycle_gate"]["delta_beta_rms_max_deg"])
    )
    beta_solved = np.vstack(
        [
            json.loads(value)
            for value in solved_table["candidate_beta"].astype(str).tolist()
        ]
    )
    velocity = (
        weighted_rms_gap_deg(
            beta_solved[1:],
            beta_solved[:-1],
            beta_weights=config["cycle"]["beta_weights"],
        )
        if len(beta_solved) >= 2
        else np.asarray([1.0e12])
    )
    if completed_360:
        velocity = np.r_[velocity, seam_deg]
    acceleration = (
        weighted_rms_gap_deg(
            beta_solved[2:] - 2.0 * beta_solved[1:-1] + beta_solved[:-2],
            np.zeros_like(beta_solved[2:]),
            beta_weights=config["cycle"]["beta_weights"],
        )
        if len(beta_solved) >= 3
        else np.asarray([1.0e12])
    )
    span_deg = 360.0 * solved_count / count
    checkpoint = {
        f"survived_{int(degree)}deg": bool(span_deg + 1.0e-12 >= float(degree))
        for degree in config["viability"]["arc_lengths_deg"]
    }
    summary = {
        "root_seed_idx": int(root_seed_idx),
        "direction": str(direction),
        "completed_360": completed_360,
        "strict_feasible_ratio": float(solved_count / count),
        "solved_phase_count": int(solved_count),
        "first_failure_phase_idx": (
            None if failure_phase is None else int(failure_phase)
        ),
        "joint_margin_min_deg": float(solved_table["joint_margin_deg"].min()),
        "residual_max_mm": float(solved_table["residual_mm"].max()),
        "residual_p95_mm": float(
            np.percentile(solved_table["residual_mm"], 95)
        ),
        "velocity_p95_deg": float(np.percentile(velocity, 95)),
        "velocity_max_deg": float(np.max(velocity)),
        "acceleration_p95_deg": float(np.percentile(acceleration, 95)),
        "seam_deg": (seam_deg if math.isfinite(seam_deg) else 1.0e12),
        "kappa_p95": float(np.percentile(solved_table["kappa"], 95)),
        **checkpoint,
    }
    return summary, solved_table, pd.DataFrame(candidate_rows)


def _aggregate_bidirectional_viability(
    direction_table: pd.DataFrame,
    *,
    environment: Any,
    root_beta_by_seed: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    rows: list[dict[str, Any]] = []
    for root_seed_idx, group in direction_table.groupby(
        "root_seed_idx", sort=True
    ):
        by_direction = group.set_index("direction")
        beta = np.asarray(root_beta_by_seed[int(root_seed_idx)], dtype=float)
        posture = float(np.mean(np.square((beta - midpoint) / span)))
        rows.append(
            {
                "root_candidate_idx": int(root_seed_idx),
                "completed_360": bool(group["completed_360"].all())
                and set(group["direction"]) == {"forward", "reverse"},
                "strict_feasible_ratio": float(
                    group["strict_feasible_ratio"].min()
                ),
                "joint_margin_min_deg": float(
                    group["joint_margin_min_deg"].min()
                ),
                "residual_max_mm": float(group["residual_max_mm"].max()),
                "residual_p95_mm": float(group["residual_p95_mm"].max()),
                "velocity_p95_deg": float(group["velocity_p95_deg"].max()),
                "acceleration_p95_deg": float(
                    group["acceleration_p95_deg"].max()
                ),
                "posture_cost": posture,
                "kappa_p95": float(group["kappa_p95"].max()),
                "forward_solved_phase_count": int(
                    by_direction.loc["forward", "solved_phase_count"]
                ),
                "reverse_solved_phase_count": int(
                    by_direction.loc["reverse", "solved_phase_count"]
                ),
            }
        )
    return rank_viability(pd.DataFrame(rows))


def run_viability(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    output: Path,
    **_unused: Any,
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["viability"]
    all_summaries: list[dict[str, Any]] = []
    for candidate_offset, candidate_id in enumerate(map(str, config["candidates"])):
        primary, _variants = _aligned_source_frames(
            config,
            project_root,
            candidate_id,
            count=int(config["phase_counts"]["pilot"]),
        )
        target = _frame_target(primary)
        reference = _frame_beta(primary)
        root_summary = json.loads(
            (
                output
                / STAGE_DIRS["root_fiber"]
                / candidate_id
                / "root_fiber_summary.json"
            ).read_text(encoding="utf-8")
        )
        root_phase = int(root_summary["phase_idx"])
        candidates = pd.read_parquet(
            output
            / STAGE_DIRS["root_fiber"]
            / candidate_id
            / "root_candidates.parquet"
        )
        representative_inventory = pd.read_csv(
            output
            / STAGE_DIRS["root_fiber"]
            / candidate_id
            / "root_representatives.csv"
        )
        beta_columns = [f"beta{joint}_rad" for joint in range(1, 7)]
        beta_by_seed = {
            int(index): row[beta_columns].to_numpy(float)
            for index, row in candidates.iterrows()
        }
        for root_seed_count in map(
            int, config["pilot_matrix"]["root_seed_counts"]
        ):
            inventory = representative_inventory[
                representative_inventory["root_seed_count"] == root_seed_count
            ]
            direction_rows: list[dict[str, Any]] = []
            selected_frames: list[pd.DataFrame] = []
            evidence_frames: list[pd.DataFrame] = []
            for representative_order, seed_index in enumerate(
                inventory["root_seed_idx"].astype(int).tolist()
            ):
                for direction_index, direction in enumerate(
                    ("forward", "reverse")
                ):
                    policy = _teacher_policy(
                        config,
                        seed=(
                            int(config["seeds"]["viability"])
                            + 100000 * candidate_offset
                            + 100 * representative_order
                            + direction_index
                        ),
                    )
                    summary, selected, evidence = _solve_lineage(
                        config=config,
                        environment=environment,
                        target=target,
                        reference=reference,
                        root_phase=root_phase,
                        root_beta=beta_by_seed[seed_index],
                        root_seed_idx=seed_index,
                        direction=direction,
                        policy=policy,
                        solver_seed=int(policy.solver_seed),
                    )
                    direction_rows.append(summary)
                    selected["root_seed_count"] = root_seed_count
                    evidence["root_seed_count"] = root_seed_count
                    selected_frames.append(selected)
                    evidence_frames.append(evidence)
            direction_table = pd.DataFrame(direction_rows)
            if direction_table.empty:
                ranked = pd.DataFrame(
                    columns=[
                        "root_candidate_idx",
                        "completed_360",
                        "strict_feasible_ratio",
                        "joint_margin_min_deg",
                        "residual_max_mm",
                        "residual_p95_mm",
                        "velocity_p95_deg",
                        "acceleration_p95_deg",
                        "posture_cost",
                        "kappa_p95",
                        "forward_solved_phase_count",
                        "reverse_solved_phase_count",
                        "viability_rank",
                    ]
                )
            else:
                ranked = _aggregate_bidirectional_viability(
                    direction_table,
                    environment=environment,
                    root_beta_by_seed=beta_by_seed,
                )
            directory = stage / candidate_id / f"Nroot_{root_seed_count:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            direction_table.to_csv(
                directory / "direction_survival_summary.csv", index=False
            )
            ranked.to_csv(directory / "viability_ranking.csv", index=False)
            selected_table = (
                pd.concat(selected_frames, ignore_index=True)
                if selected_frames
                else pd.DataFrame(
                    columns=[
                        *CANDIDATE_EVIDENCE_COLUMNS,
                        "phase_idx",
                        "traversal_idx",
                        "selected",
                        "root_seed_idx",
                        "direction",
                        "root_seed_count",
                    ]
                )
            )
            evidence_table = (
                pd.concat(evidence_frames, ignore_index=True)
                if evidence_frames
                else selected_table.copy()
            )
            _atomic_parquet(
                selected_table,
                directory / "selected_lineages.parquet",
            )
            _atomic_parquet(
                evidence_table,
                directory / "candidate_evidence.parquet",
            )
            top = ranked.head(int(config["viability"]["top_lineage_count"]))
            top.to_csv(
                directory / "top_survival_ranked_roots.csv", index=False
            )
            full_loop = ranked[ranked["completed_360"].astype(bool)].copy()
            full_loop.to_csv(
                directory / "full_loop_viable_roots.csv", index=False
            )
            summary = {
                "candidate_id": candidate_id,
                "root_seed_count": root_seed_count,
                "representative_count": int(len(inventory)),
                "bidirectional_full_loop_root_count": int(
                    ranked["completed_360"].sum()
                ),
                "best_strict_feasible_ratio": float(
                    ranked["strict_feasible_ratio"].max()
                    if not ranked.empty
                    else 0.0
                ),
                "top_root_seed_indices": top[
                    "root_candidate_idx"
                ].astype(int).tolist(),
            }
            atomic_write_json(directory / "gate.json", summary)
            all_summaries.append(summary)
    return _write_gate(
        stage / "gate.json",
        checks={
            "two_candidates_complete": set(
                row["candidate_id"] for row in all_summaries
            )
            == set(map(str, config["candidates"])),
            "full_root_matrix_complete": len(all_summaries)
            == len(config["candidates"])
            * len(config["pilot_matrix"]["root_seed_counts"]),
            "candidate_evidence_columns_complete": all(
                set(CANDIDATE_EVIDENCE_COLUMNS).issubset(
                    pd.read_parquet(
                        stage
                        / row["candidate_id"]
                        / f"Nroot_{int(row['root_seed_count']):04d}"
                        / "candidate_evidence.parquet"
                    ).columns
                )
                for row in all_summaries
            ),
        },
        summaries=all_summaries,
    )


def _global_sobol_seeds(
    bounds: np.ndarray, *, count: int, seed: int
) -> np.ndarray:
    sampler = qmc.Sobol(d=6, scramble=True, seed=int(seed))
    return qmc.scale(
        sampler.random(int(count)),
        np.asarray(bounds)[:, 0],
        np.asarray(bounds)[:, 1],
    )


def _phase_seed_inventory(
    *,
    environment: Any,
    target: np.ndarray,
    phase: int,
    reference: np.ndarray,
    source_paths: Sequence[tuple[str, np.ndarray, int | None]],
    budget: int,
    seed: int,
    nullspace_step_deg: float,
) -> list[tuple[str, np.ndarray, int | None]]:
    inventory = list(source_paths)
    current = reference[phase]
    previous = reference[(phase - 1) % len(reference)]
    before = reference[(phase - 2) % len(reference)]
    inventory.extend(
        [
            ("source_reference", current.copy(), None),
            ("secant_predictor", previous + (previous - before), None),
        ]
    )
    jacobian = _environment_jacobian(environment, current)
    _u, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * max(
        float(singular[0]), 1.0
    )
    null_basis = vh[int(np.sum(singular > tolerance)) :]
    for direction in null_basis:
        inventory.append(
            (
                "nullspace_local",
                current + math.radians(float(nullspace_step_deg)) * direction,
                None,
            )
        )
        inventory.append(
            (
                "nullspace_local",
                current - math.radians(float(nullspace_step_deg)) * direction,
                None,
            )
        )
    missing = max(int(budget) - len(inventory), 0)
    if missing:
        bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
        inventory.extend(
            ("full_beta_sobol", value, None)
            for value in _global_sobol_seeds(
                bounds, count=missing, seed=int(seed)
            )
        )
    return inventory[: int(budget)]


def _generate_hard_layers(
    *,
    config: Mapping[str, Any],
    environment: Any,
    target: np.ndarray,
    reference: np.ndarray,
    variants: Mapping[str, np.ndarray],
    selected_lineages: pd.DataFrame,
    top_root_seed_indices: set[int],
    candidate_cap: int,
    solver_seed: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    maximum_cap = int(candidate_cap)
    targeted_budgets = candidate_search_budgets(
        candidate_cap=maximum_cap,
        initial_budget_floor=int(
            config["candidate_generation"]["initial_budget_floor"]
        ),
        initial_budget_multiplier=int(
            config["candidate_generation"]["initial_budget_multiplier"]
        ),
        targeted_cumulative_budgets=config["candidate_generation"][
            "targeted_cumulative_budgets"
        ],
    )
    initial_budget = targeted_budgets[0]
    policy = _teacher_policy(config, seed=int(solver_seed))
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    layers: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    evidence_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    for phase in range(len(target)):
        lineage_phase = selected_lineages[
            selected_lineages["phase_idx"].astype(int) == phase
        ]
        lineage_sources: list[tuple[str, np.ndarray, int | None]] = []
        for _, row in lineage_phase.iterrows():
            root_seed_idx = int(row["root_seed_idx"])
            if root_seed_idx not in top_root_seed_indices:
                continue
            lineage_sources.append(
                (
                    f"viability_{row['direction']}",
                    np.asarray(json.loads(row["candidate_beta"]), dtype=float),
                    root_seed_idx,
                )
            )
        for name, path in variants.items():
            lineage_sources.append((f"source_{name}", path[phase], None))
        all_beta: list[np.ndarray] = []
        all_rows: list[dict[str, Any]] = []
        attempted = 0
        layer = filter_hard_feasible_candidates(
            np.zeros((0, 6)),
            pd.DataFrame(
                columns=[
                    "corrector_success",
                    "residual_mm",
                    "joint_margin_deg",
                    "within_bounds",
                ]
            ),
            cap=maximum_cap,
            residual_max_mm=float(
                config["hard_feasibility"]["residual_max_mm"]
            ),
            margin_min_deg=float(
                config["hard_feasibility"]["joint_margin_min_deg"]
            ),
            distinct_threshold_deg=float(
                config["hard_feasibility"]["distinct_threshold_deg"]
            ),
            beta_weights=config["cycle"]["beta_weights"],
        )
        targeted_search_used = False
        for cumulative_budget in targeted_budgets:
            if cumulative_budget <= attempted:
                continue
            if attempted >= initial_budget:
                targeted_search_used = True
            inventory = _phase_seed_inventory(
                environment=environment,
                target=target,
                phase=phase,
                reference=reference,
                source_paths=lineage_sources if attempted == 0 else [],
                budget=cumulative_budget - attempted,
                seed=int(solver_seed) + 1009 * phase + attempted,
                nullspace_step_deg=float(
                    config["candidate_generation"]["local_nullspace_step_deg"]
                ),
            )
            for source, seed_beta, origin_root in inventory:
                beta, row = _correct_seed(
                    environment=environment,
                    target=target[phase],
                    seed=np.clip(seed_beta, bounds[:, 0], bounds[:, 1]),
                    policy=policy,
                    candidate_source=source,
                    reference_beta=reference[phase],
                    trust_radius_deg=math.inf,
                )
                row.update(
                    {
                        "phase_idx": phase,
                        "attempt_idx": len(all_rows),
                        "origin_root_seed_idx": origin_root,
                        "targeted_global_search": bool(attempted >= initial_budget),
                    }
                )
                all_beta.append(beta)
                all_rows.append(row)
            attempted = cumulative_budget
            layer = filter_hard_feasible_candidates(
                np.vstack(all_beta),
                pd.DataFrame(all_rows),
                cap=maximum_cap,
                residual_max_mm=float(
                    config["hard_feasibility"]["residual_max_mm"]
                ),
                margin_min_deg=float(
                    config["hard_feasibility"]["joint_margin_min_deg"]
                ),
                distinct_threshold_deg=float(
                    config["hard_feasibility"]["distinct_threshold_deg"]
                ),
                beta_weights=config["cycle"]["beta_weights"],
            )
            if len(layer.beta_rad) > 0:
                break
        selected_table = layer.table.copy()
        selected_table["phase_idx"] = phase
        selected_table["layer_candidate_idx"] = np.arange(
            len(selected_table), dtype=np.int64
        )
        for joint in range(6):
            selected_table[f"beta{joint + 1}_rad"] = layer.beta_rad[:, joint]
        selected_frames.append(selected_table)
        evidence = pd.DataFrame(all_rows)
        evidence["hard_selected"] = False
        if len(selected_table):
            evidence.loc[
                selected_table["source_candidate_idx"].astype(int), "hard_selected"
            ] = True
        evidence_frames.append(evidence)
        layers.append(layer.beta_rad)
        residuals.append(
            selected_table["residual_mm"].to_numpy(float)
            if len(selected_table)
            else np.zeros(0)
        )
        count_rows.append(
            {
                "phase_idx": phase,
                "feasible_candidate_count": int(len(layer.beta_rad)),
                "requested_cap": maximum_cap,
                "candidate_shortfall": int(layer.shortfall),
                "corrector_attempt_count": int(len(all_rows)),
                "targeted_global_search_used": bool(targeted_search_used),
                "targeted_search_complete": bool(
                    len(layer.beta_rad) > 0
                    or attempted >= max(targeted_budgets)
                ),
            }
        )
    return (
        layers,
        residuals,
        pd.concat(selected_frames, ignore_index=True),
        pd.DataFrame(count_rows),
        pd.concat(evidence_frames, ignore_index=True),
    )


def _cycle_metrics(
    environment: Any,
    target: np.ndarray,
    beta: np.ndarray,
    *,
    beta_weights: Sequence[float],
) -> dict[str, float]:
    achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    margin = np.rad2deg(
        np.min(
            np.minimum(
                beta - bounds[:, 0][None, :],
                bounds[:, 1][None, :] - beta,
            ),
            axis=1,
        )
    )
    return strict_cycle_metrics(
        beta,
        target_xyz_m=target,
        achieved_xyz_m=achieved,
        joint_margin_deg=margin,
        beta_weights=beta_weights,
    )


def _correct_whole_cycle(
    *,
    config: Mapping[str, Any],
    environment: Any,
    target: np.ndarray,
    initial_beta: np.ndarray,
    solver_seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    initial = np.asarray(initial_beta, dtype=float).copy()
    policy = _teacher_policy(config, seed=int(solver_seed))
    count = len(target)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    safe = math.radians(
        float(config["hard_feasibility"]["joint_margin_min_deg"])
    )
    lower = np.tile(bounds[:, 0] + safe, count)
    upper = np.tile(bounds[:, 1] - safe, count)
    x0 = np.clip(initial.reshape(-1), lower, upper)
    beta_weights = np.asarray(config["cycle"]["beta_weights"], dtype=float)
    joint_scale = np.sqrt(beta_weights / np.sum(beta_weights)) * 180.0 / math.pi
    velocity_scale = (
        math.sqrt(float(config["cycle"]["lambda_velocity"])) * joint_scale
    )
    acceleration_scale = (
        math.sqrt(float(config["cycle"]["lambda_acceleration"])) * joint_scale
    )
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    posture_scale = math.sqrt(
        float(config["teacher_policy"]["lambda_posture"])
    )
    residual_target_mm = float(
        config["strict_cycle_gate"]["residual_p95_mm"]
    )
    current_penalty = 1.0

    def objective(flat: np.ndarray) -> np.ndarray:
        beta = np.asarray(flat, dtype=float).reshape(count, 6)
        achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
        task = (achieved - target) * 1000.0
        velocity = (np.roll(beta, -1, axis=0) - beta) * velocity_scale
        acceleration = (
            np.roll(beta, -1, axis=0)
            - 2.0 * beta
            + np.roll(beta, 1, axis=0)
        ) * acceleration_scale
        posture = ((beta - midpoint) / span) * posture_scale
        residual_norm = np.linalg.norm(task, axis=1)
        tracking_violation = (
            np.maximum(residual_norm - residual_target_mm, 0.0)
            * math.sqrt(current_penalty)
        )
        return np.concatenate(
            [
                task.reshape(-1),
                velocity.reshape(-1),
                acceleration.reshape(-1),
                posture.reshape(-1),
                tracking_violation,
            ]
        )

    def jacobian(flat: np.ndarray) -> Any:
        beta = np.asarray(flat, dtype=float).reshape(count, 6)
        achieved = np.asarray(environment.fk(beta), dtype=float).reshape(-1, 3)
        task = (achieved - target) * 1000.0
        rows = 3 * count + 12 * count + 6 * count + count
        matrix = lil_matrix((rows, 6 * count), dtype=float)
        task_jacobians: list[np.ndarray] = []
        for phase in range(count):
            task_jacobian = (
                _environment_jacobian(environment, beta[phase]) * 1000.0
            )
            task_jacobians.append(task_jacobian)
            matrix[
                3 * phase : 3 * phase + 3,
                6 * phase : 6 * phase + 6,
            ] = task_jacobian
        velocity_offset = 3 * count
        acceleration_offset = velocity_offset + 6 * count
        posture_offset = acceleration_offset + 6 * count
        violation_offset = posture_offset + 6 * count
        for phase in range(count):
            next_phase = (phase + 1) % count
            previous_phase = (phase - 1) % count
            for joint in range(6):
                velocity_row = velocity_offset + 6 * phase + joint
                matrix[velocity_row, 6 * phase + joint] = -velocity_scale[joint]
                matrix[velocity_row, 6 * next_phase + joint] = velocity_scale[joint]
                acceleration_row = acceleration_offset + 6 * phase + joint
                matrix[
                    acceleration_row, 6 * previous_phase + joint
                ] = acceleration_scale[joint]
                matrix[
                    acceleration_row, 6 * phase + joint
                ] = -2.0 * acceleration_scale[joint]
                matrix[
                    acceleration_row, 6 * next_phase + joint
                ] = acceleration_scale[joint]
                matrix[
                    posture_offset + 6 * phase + joint,
                    6 * phase + joint,
                ] = posture_scale / span[joint]
            residual_norm = float(np.linalg.norm(task[phase]))
            if residual_norm > residual_target_mm:
                matrix[
                    violation_offset + phase,
                    6 * phase : 6 * phase + 6,
                ] = (
                    math.sqrt(current_penalty)
                    * (task[phase] / residual_norm)
                    @ task_jacobians[phase]
                )
        return matrix.tocsr()

    optimization = None
    total_nfev = 0
    penalty_schedule = tuple(
        float(value)
        for value in config["whole_curve"]["residual_penalty_schedule"]
    )
    for penalty in penalty_schedule:
        current_penalty = penalty
        optimization = least_squares(
            objective,
            x0,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            max_nfev=int(config["whole_curve"]["least_squares_max_nfev"]),
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
            verbose=0,
        )
        total_nfev += int(optimization.nfev)
        x0 = optimization.x
    if optimization is None:
        raise ValueError("whole-curve residual penalty schedule must be non-empty")
    selected = optimization.x.reshape(count, 6)
    rows: list[dict[str, Any]] = []
    # A final hard IK projection is accepted phase-wise only when it preserves
    # both the registered residual and margin constraints.
    for phase in range(count):
        beta, row = _correct_seed(
            environment=environment,
            target=target[phase],
            seed=selected[phase],
            policy=policy,
            candidate_source="bounded_whole_trajectory_penalty_least_squares",
            reference_beta=selected[phase],
            trust_radius_deg=float(config["viability"]["trust_radius_deg"]),
        )
        hard = bool(
            row["corrector_success"]
            and row["within_bounds"]
            and row["residual_mm"]
            <= float(config["hard_feasibility"]["residual_max_mm"])
            and row["joint_margin_deg"]
            >= float(config["hard_feasibility"]["joint_margin_min_deg"])
        )
        if hard:
            selected[phase] = beta
        row.update(
            {
                "phase_idx": phase,
                "accepted": hard,
                "optimizer": (
                    "bounded_whole_trajectory_penalty_least_squares"
                ),
                "optimizer_success": bool(optimization.success),
                "optimizer_status": int(optimization.status),
                "optimizer_nfev": total_nfev,
                "optimizer_cost": float(optimization.cost),
                "residual_penalty_schedule": json.dumps(
                    penalty_schedule
                ),
            }
        )
        rows.append(row)
    return selected, pd.DataFrame(rows)


def _minimal_slack(
    metrics: Mapping[str, float], config: Mapping[str, Any]
) -> dict[str, float]:
    gate = config["strict_cycle_gate"]
    return {
        "residual_p95_slack_mm": max(
            float(metrics["residual_p95_mm"])
            - float(gate["residual_p95_mm"]),
            0.0,
        ),
        "residual_max_slack_mm": max(
            float(metrics["residual_max_mm"]) - float(gate["residual_max_mm"]),
            0.0,
        ),
        "margin_slack_deg": max(
            float(gate["joint_margin_min_deg"])
            - float(metrics["joint_margin_min_deg"]),
            0.0,
        ),
        "velocity_p95_slack_deg": max(
            float(metrics["delta_beta_rms_p95_deg"])
            - float(gate["delta_beta_rms_p95_deg"]),
            0.0,
        ),
        "velocity_max_slack_deg": max(
            float(metrics["delta_beta_rms_max_deg"])
            - float(gate["delta_beta_rms_max_deg"]),
            0.0,
        ),
        "acceleration_p95_slack_deg": max(
            float(metrics["acceleration_beta_rms_p95_deg"])
            - float(gate["acceleration_beta_rms_p95_deg"]),
            0.0,
        ),
        "seam_slack_deg": max(
            float(metrics["seam_beta_rms_deg"])
            - float(gate["seam_beta_rms_deg"]),
            0.0,
        ),
    }


def _minimal_slack_by_phase(
    candidate_evidence: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Solve the discrete per-phase slack diagnostic over all searched IK nodes."""

    rows: list[dict[str, Any]] = []
    residual_target = float(config["strict_cycle_gate"]["residual_p95_mm"])
    residual_hard = float(config["strict_cycle_gate"]["residual_max_mm"])
    margin_target = float(config["strict_cycle_gate"]["joint_margin_min_deg"])
    for phase, group in candidate_evidence.groupby("phase_idx", sort=True):
        scored = group.copy()
        scored["residual_p95_target_slack_mm"] = np.maximum(
            scored["residual_mm"].to_numpy(float) - residual_target, 0.0
        )
        scored["residual_hard_slack_mm"] = np.maximum(
            scored["residual_mm"].to_numpy(float) - residual_hard, 0.0
        )
        scored["margin_slack_deg"] = np.maximum(
            margin_target - scored["joint_margin_deg"].to_numpy(float), 0.0
        )
        invalid = ~(
            scored["corrector_success"].astype(bool).to_numpy()
            & scored["within_bounds"].astype(bool).to_numpy()
        )
        scored["slack_objective"] = (
            np.square(scored["residual_p95_target_slack_mm"])
            + 10.0 * np.square(scored["residual_hard_slack_mm"])
            + 10.0 * np.square(scored["margin_slack_deg"])
            + invalid.astype(float) * 1.0e6
        )
        best = scored.sort_values(
            ["slack_objective", "residual_mm", "joint_margin_deg"],
            ascending=[True, True, False],
            kind="stable",
        ).iloc[0]
        rows.append(
            {
                "phase_idx": int(phase),
                "candidate_source": str(best["candidate_source"]),
                "residual_mm": float(best["residual_mm"]),
                "joint_margin_deg": float(best["joint_margin_deg"]),
                "residual_p95_target_slack_mm": float(
                    best["residual_p95_target_slack_mm"]
                ),
                "residual_hard_slack_mm": float(
                    best["residual_hard_slack_mm"]
                ),
                "margin_slack_deg": float(best["margin_slack_deg"]),
                "slack_objective": float(best["slack_objective"]),
                "corrector_success": bool(best["corrector_success"]),
                "within_bounds": bool(best["within_bounds"]),
            }
        )
    return pd.DataFrame(rows)


def _slack_phase_summary(frame: pd.DataFrame) -> dict[str, Any]:
    active = frame[frame["slack_objective"] > 0.0].sort_values(
        "slack_objective", ascending=False, kind="stable"
    )
    return {
        "minimal_slack_objective_sum": float(frame["slack_objective"].sum()),
        "minimal_slack_objective_max": float(frame["slack_objective"].max()),
        "slack_phase_count": int(len(active)),
        "slack_concentrated_phase_indices": active["phase_idx"]
        .head(32)
        .astype(int)
        .tolist(),
        "residual_hard_slack_phase_indices": frame.loc[
            frame["residual_hard_slack_mm"] > 0.0, "phase_idx"
        ]
        .astype(int)
        .tolist(),
        "margin_slack_phase_indices": frame.loc[
            frame["margin_slack_deg"] > 0.0, "phase_idx"
        ]
        .astype(int)
        .tolist(),
    }


def _evaluate_cycle(
    *,
    config: Mapping[str, Any],
    environment: Any,
    target: np.ndarray,
    beta: np.ndarray,
) -> dict[str, Any]:
    metrics = _cycle_metrics(
        environment,
        target,
        beta,
        beta_weights=config["cycle"]["beta_weights"],
    )
    strict = _strict_gate(config).evaluate(metrics)
    relaxed = _relaxed_gate(config).evaluate(metrics)
    return {
        "numerical_metrics": metrics,
        "strict_checks": strict["checks"],
        "gate_pass": bool(strict["gate_pass"]),
        "diagnostic_relaxed_cycle_pass": bool(relaxed["gate_pass"]),
        "minimal_slack": _minimal_slack(metrics, config),
    }


def _restore_phase_order(
    ordered_beta: np.ndarray, phase_order: np.ndarray
) -> np.ndarray:
    restored = np.empty_like(ordered_beta)
    restored[np.asarray(phase_order, dtype=np.int64)] = ordered_beta
    return restored


def _write_cycle_artifact(
    *,
    directory: Path,
    beta: np.ndarray,
    target: np.ndarray,
    evaluation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "phase_idx": np.arange(len(beta), dtype=np.int64),
            "target_x_m": target[:, 0],
            "target_y_m": target[:, 1],
            "target_z_m": target[:, 2],
        }
    )
    for joint in range(6):
        frame[f"teacher_beta{joint + 1}_rad"] = beta[:, joint]
    _atomic_parquet(frame, directory / "cycle.parquet")
    atomic_write_json(
        directory / "gate.json",
        {
            **dict(metadata),
            **dict(evaluation),
        },
    )


def _write_full_curve_cluster_artifacts(
    *,
    directory: Path,
    ordered_layers: Sequence[np.ndarray],
    top_cycles: Sequence[Mapping[str, Any]],
    selected_beta: np.ndarray | None,
    beta_weights: Sequence[float],
    threshold_deg: float = 1.0,
) -> dict[str, Any]:
    curves: list[np.ndarray] = []
    identifiers: list[str] = []
    if selected_beta is not None and len(selected_beta):
        curves.append(np.asarray(selected_beta, dtype=float).reshape(-1, 6))
        identifiers.append("selected_cycle")
    for index, cycle in enumerate(top_cycles):
        indices = np.asarray(cycle["layer_candidate_indices"], dtype=np.int64)
        if len(indices) != len(ordered_layers):
            continue
        curves.append(
            np.vstack(
                [
                    ordered_layers[phase][indices[phase]]
                    for phase in range(len(ordered_layers))
                ]
            )
        )
        identifiers.append(f"top_cycle_{index:02d}")
    clustered = cluster_full_loop_cycles(
        curves,
        threshold_deg=float(threshold_deg),
        beta_weights=beta_weights,
    )
    membership = pd.DataFrame(
        {
            "cycle_id": identifiers,
            "curve_cluster_id": clustered.component_labels,
        }
    )
    pairs = [
        {
            "left_cycle_id": identifiers[left],
            "right_cycle_id": identifiers[right],
            "curve_gap_p95_deg": float(
                clustered.distance_deg[left, right]
            ),
        }
        for left in range(len(identifiers))
        for right in range(left + 1, len(identifiers))
    ]
    directory.mkdir(parents=True, exist_ok=True)
    membership.to_csv(
        directory / "full_curve_cluster_membership.csv", index=False
    )
    pd.DataFrame(
        pairs,
        columns=[
            "left_cycle_id",
            "right_cycle_id",
            "curve_gap_p95_deg",
        ],
    ).to_csv(directory / "full_curve_pairwise_distance.csv", index=False)
    payload = {
        "curve_count": len(identifiers),
        "curve_cluster_count": clustered.connected_component_count,
        "cluster_threshold_deg": float(threshold_deg),
    }
    atomic_write_json(directory / "full_curve_cluster_summary.json", payload)
    return payload


def _restricted_root_layers(
    layers: Sequence[np.ndarray],
    residuals: Sequence[np.ndarray],
    *,
    root_nodes: pd.DataFrame,
    mode: str,
    root_candidates: pd.DataFrame,
    allowed_root_indices: set[int],
    beta_weights: Sequence[float],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    output_layers = [np.asarray(value).copy() for value in layers]
    output_residuals = [np.asarray(value).copy() for value in residuals]
    if len(output_layers[0]) == 0 or mode == "all":
        return output_layers, output_residuals
    allowed: np.ndarray
    origin = pd.to_numeric(
        root_nodes["origin_root_seed_idx"], errors="coerce"
    ).to_numpy(float)
    if mode == "current":
        allowed = np.flatnonzero(origin == 0)
        target_roots = root_candidates.iloc[[0]]
    elif mode == "roots":
        allowed = np.flatnonzero(
            np.asarray(
                [
                    np.isfinite(value) and int(value) in allowed_root_indices
                    for value in origin
                ],
                dtype=bool,
            )
        )
        target_roots = root_candidates.iloc[sorted(allowed_root_indices)]
    else:
        raise ValueError(f"unknown root restriction mode: {mode}")
    if len(allowed) == 0:
        if target_roots.empty:
            output_layers[0] = np.zeros((0, 6), dtype=float)
            output_residuals[0] = np.zeros(0, dtype=float)
            return output_layers, output_residuals
        root_beta = target_roots[
            [f"beta{joint}_rad" for joint in range(1, 7)]
        ].to_numpy(float)
        gap = weighted_rms_gap_deg(
            output_layers[0][:, None, :],
            root_beta[None, :, :],
            beta_weights=beta_weights,
        )
        allowed = np.asarray([int(np.unravel_index(np.argmin(gap), gap.shape)[0])])
    output_layers[0] = output_layers[0][allowed]
    output_residuals[0] = output_residuals[0][allowed]
    return output_layers, output_residuals


def _best_corrected_top_cycle(
    *,
    config: Mapping[str, Any],
    environment: Any,
    ordered_target: np.ndarray,
    ordered_layers: Sequence[np.ndarray],
    solution: Any,
    solver_seed: int,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    choices: list[tuple[np.ndarray, dict[str, Any], pd.DataFrame]] = []
    for cycle_index, cycle in enumerate(solution.top_cycles):
        indices = np.asarray(cycle["layer_candidate_indices"], dtype=np.int64)
        initial = np.vstack(
            [
                ordered_layers[phase][indices[phase]]
                for phase in range(len(ordered_layers))
            ]
        )
        corrected, evidence = _correct_whole_cycle(
            config=config,
            environment=environment,
            target=ordered_target,
            initial_beta=initial,
            solver_seed=int(solver_seed) + cycle_index,
        )
        evaluation = _evaluate_cycle(
            config=config,
            environment=environment,
            target=ordered_target,
            beta=corrected,
        )
        choices.append((corrected, evaluation, evidence))
    choices.sort(
        key=lambda item: (
            not bool(item[1]["gate_pass"]),
            -sum(bool(value) for value in item[1]["strict_checks"].values()),
            sum(float(value) for value in item[1]["minimal_slack"].values()),
        )
    )
    return choices[0]


def _old_e0_evidence(
    config: Mapping[str, Any], project_root: Path, candidate_id: str
) -> dict[str, Any]:
    path = (
        project_root
        / str(config["source_artifact_root"])
        / "03_formal"
        / candidate_id
        / "BI-3"
        / "gate.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(payload["numerical_metrics"])
    strict = _strict_gate(config).evaluate(metrics)
    relaxed = _relaxed_gate(config).evaluate(metrics)
    return {
        "numerical_metrics": metrics,
        "strict_checks": strict["checks"],
        "gate_pass": bool(strict["gate_pass"]),
        "diagnostic_relaxed_cycle_pass": bool(relaxed["gate_pass"]),
        "minimal_slack": _minimal_slack(metrics, config),
        "source_gate_path": str(path),
    }


def _pilot_task_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic, numerically independent Pilot task slices."""

    root_seed_counts = list(map(int, config["pilot_matrix"]["root_seed_counts"]))
    first_root_seed_count = root_seed_counts[0]
    tasks = []
    for candidate_offset, candidate_id in enumerate(map(str, config["candidates"])):
        for root_seed_count in root_seed_counts:
            tasks.append(
                {
                    "task_id": (
                        f"{candidate_offset:02d}_{candidate_id}"
                        f"_Nroot_{root_seed_count:04d}"
                    ),
                    "candidate_id": candidate_id,
                    "candidate_offset": candidate_offset,
                    "root_seed_count": root_seed_count,
                    "include_e0": root_seed_count == first_root_seed_count,
                }
            )
    return tasks


def _sort_pilot_ranking(
    ranking: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Stably order Pilot rows without depending on worker completion order."""

    candidate_order = {
        value: index for index, value in enumerate(map(str, config["candidates"]))
    }
    experiment_order = {
        value: index
        for index, value in enumerate(("E0", "E1", "E2", "E3", "E4"))
    }
    ordered = ranking.copy()
    ordered["_candidate_order"] = (
        ordered["candidate_id"].map(candidate_order).fillna(len(candidate_order))
    )
    ordered["_experiment_order"] = (
        ordered["experiment"].map(experiment_order).fillna(len(experiment_order))
    )
    ordered = ordered.sort_values(
        [
            "_candidate_order",
            "root_seed_count",
            "candidate_cap",
            "_experiment_order",
            "edge_limit_deg",
        ],
        kind="stable",
        na_position="first",
    )
    return ordered.drop(
        columns=["_candidate_order", "_experiment_order"]
    ).reset_index(drop=True)


def _pilot_slice_paths(
    stage: Path, task: Mapping[str, Any]
) -> tuple[Path, Path]:
    slice_dir = stage / "_parallel" / "slices" / str(task["task_id"])
    return slice_dir / "pilot_rows.csv", slice_dir / "gate.json"


def _run_pilot_slice(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    output: Path,
    pilot_task: Mapping[str, Any],
    **_unused: Any,
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["pilot"]
    ranking_rows: list[dict[str, Any]] = []
    maximum_cap = max(map(int, config["pilot_matrix"]["candidate_caps"]))
    for candidate_offset, candidate_id in enumerate(map(str, config["candidates"])):
        if candidate_id != str(pilot_task["candidate_id"]):
            continue
        if candidate_offset != int(pilot_task["candidate_offset"]):
            raise RuntimeError(
                "Pilot task candidate_offset does not match frozen candidate order"
            )
        primary, variants_frame = _aligned_source_frames(
            config,
            project_root,
            candidate_id,
            count=int(config["phase_counts"]["pilot"]),
        )
        target = _frame_target(primary)
        reference = _frame_beta(primary)
        variants = {
            name: _frame_beta(frame) for name, frame in variants_frame.items()
        }
        root_summary = json.loads(
            (
                output
                / STAGE_DIRS["root_fiber"]
                / candidate_id
                / "root_fiber_summary.json"
            ).read_text(encoding="utf-8")
        )
        root_phase = int(root_summary["phase_idx"])
        phase_order = _lineage_order(len(target), root_phase, "forward")
        root_candidates = pd.read_parquet(
            output
            / STAGE_DIRS["root_fiber"]
            / candidate_id
            / "root_candidates.parquet"
        )
        if bool(pilot_task["include_e0"]):
            e0 = _old_e0_evidence(config, project_root, candidate_id)
            e0_source = pd.read_parquet(
                project_root
                / str(config["source_artifact_root"])
                / "03_formal"
                / candidate_id
                / "BI-3"
                / "canonical_consensus_branch.parquet"
            )
            e0_beta = _frame_beta(
                v113._subsample_frame(e0_source, len(target))  # noqa: SLF001
            )
            ranking_rows.append(
                {
                    "candidate_id": candidate_id,
                    "experiment": "E0",
                    "root_seed_count": 64,
                    "candidate_cap": 0,
                    "edge_limit_deg": math.nan,
                    "cycle_found": True,
                    "gate_pass": bool(e0["gate_pass"]),
                    "passed_check_count": sum(
                        bool(value) for value in e0["strict_checks"].values()
                    ),
                    "minimal_slack_total": sum(e0["minimal_slack"].values()),
                    "empty_feasible_layer_count": 0,
                    "targeted_search_complete": True,
                    "outcome": classify_full_loop_outcome(
                        strict_cycle_pass=bool(e0["gate_pass"]),
                        empty_feasible_layer_indices=[],
                        targeted_search_complete=True,
                        diagnostic_relaxed_cycle_pass=bool(
                            e0["diagnostic_relaxed_cycle_pass"]
                        ),
                    ),
                }
            )
            _write_cycle_artifact(
                directory=stage / candidate_id / "E0",
                beta=e0_beta,
                target=target,
                evaluation=e0,
                metadata={"experiment": "E0", "source": "V11.3_BI-3"},
            )
        for root_seed_count in map(
            int, config["pilot_matrix"]["root_seed_counts"]
        ):
            if root_seed_count != int(pilot_task["root_seed_count"]):
                continue
            viability_dir = (
                output
                / STAGE_DIRS["viability"]
                / candidate_id
                / f"Nroot_{root_seed_count:04d}"
            )
            top = pd.read_csv(
                viability_dir / "top_survival_ranked_roots.csv"
            )
            survival_root_indices = set(
                top["root_candidate_idx"].astype(int).tolist()
            )
            full_loop = pd.read_csv(
                viability_dir / "full_loop_viable_roots.csv"
            )
            full_loop_root_indices = set(
                full_loop["root_candidate_idx"].astype(int).tolist()
            )
            generation_root_indices = (
                survival_root_indices | full_loop_root_indices
            )
            selected_lineages = pd.read_parquet(
                viability_dir / "selected_lineages.parquet"
            )
            (
                full_layers,
                full_residuals,
                nodes,
                counts,
                candidate_evidence,
            ) = _generate_hard_layers(
                config=config,
                environment=environment,
                target=target,
                reference=reference,
                variants=variants,
                selected_lineages=selected_lineages,
                top_root_seed_indices=generation_root_indices,
                candidate_cap=maximum_cap,
                solver_seed=(
                    int(config["seeds"]["graph"])
                    + 100000 * candidate_offset
                    + root_seed_count
                ),
            )
            graph_dir = (
                stage / candidate_id / f"Nroot_{root_seed_count:04d}" / "hard_graph"
            )
            graph_dir.mkdir(parents=True, exist_ok=True)
            _atomic_parquet(nodes, graph_dir / "hard_feasible_nodes.parquet")
            _atomic_parquet(
                candidate_evidence, graph_dir / "candidate_evidence.parquet"
            )
            slack_by_phase = _minimal_slack_by_phase(
                candidate_evidence, config
            )
            slack_by_phase.to_csv(
                graph_dir / "minimal_slack_by_phase.csv", index=False
            )
            atomic_write_json(
                graph_dir / "minimal_slack_summary.json",
                _slack_phase_summary(slack_by_phase),
            )
            counts.to_csv(
                graph_dir / "feasible_candidate_count_by_phase.csv", index=False
            )
            empty = counts.loc[
                counts["feasible_candidate_count"] == 0, "phase_idx"
            ].astype(int).tolist()
            targeted_search_complete = bool(
                counts.loc[
                    counts["feasible_candidate_count"] == 0,
                    "targeted_search_complete",
                ].astype(bool).all()
            )
            atomic_write_json(
                graph_dir / "empty_feasible_layers.json",
                {
                    "empty_feasible_layer_indices": empty,
                    "targeted_search_complete": targeted_search_complete,
                },
            )
            for candidate_cap in map(
                int, config["pilot_matrix"]["candidate_caps"]
            ):
                capped_layers = [value[:candidate_cap] for value in full_layers]
                capped_residuals = [
                    value[:candidate_cap] for value in full_residuals
                ]
                ordered_layers = [
                    capped_layers[int(phase)] for phase in phase_order
                ]
                ordered_residuals = [
                    capped_residuals[int(phase)] for phase in phase_order
                ]
                root_nodes = nodes[
                    nodes["phase_idx"].astype(int) == root_phase
                ].sort_values("layer_candidate_idx", kind="stable").head(
                    candidate_cap
                )
                cycle_cache: dict[
                    tuple[str, tuple[int, ...], float],
                    tuple[Any, list[np.ndarray], list[np.ndarray]],
                ] = {}
                for experiment in ("E1", "E2", "E3", "E4"):
                    restriction = {
                        "E1": "current",
                        "E2": "roots",
                        "E3": "roots",
                        "E4": "roots",
                    }[experiment]
                    experiment_root_indices = (
                        survival_root_indices
                        if experiment in {"E2", "E3"}
                        else full_loop_root_indices
                        if experiment == "E4"
                        else set()
                    )
                    experiment_layers, experiment_residuals = (
                        _restricted_root_layers(
                            ordered_layers,
                            ordered_residuals,
                            root_nodes=root_nodes,
                            mode=restriction,
                            root_candidates=root_candidates,
                            allowed_root_indices=experiment_root_indices,
                            beta_weights=config["cycle"]["beta_weights"],
                        )
                    )
                    for edge_limit in map(
                        float, config["pilot_matrix"]["edge_limits_deg"]
                    ):
                        cache_key = (
                            restriction,
                            tuple(sorted(experiment_root_indices)),
                            edge_limit,
                        )
                        if cache_key not in cycle_cache:
                            cycle_cache[cache_key] = (
                                solve_sparse_cycle(
                                    experiment_layers,
                                    experiment_residuals,
                                    beta_weights=config["cycle"]["beta_weights"],
                                    edge_limit_deg=edge_limit,
                                    lambda_velocity=float(
                                        config["cycle"]["lambda_velocity"]
                                    ),
                                    lambda_acceleration=float(
                                        config["cycle"]["lambda_acceleration"]
                                    ),
                                    top_m=int(config["cycle"]["top_m"]),
                                    max_states_per_root=int(
                                        config["cycle"]["max_states_per_root"]
                                    ),
                                ),
                                experiment_layers,
                                experiment_residuals,
                            )
                        solution, experiment_layers, experiment_residuals = cycle_cache[
                            cache_key
                        ]
                        run_dir = (
                            stage
                            / candidate_id
                            / f"Nroot_{root_seed_count:04d}"
                            / f"K_{candidate_cap:02d}"
                            / experiment
                            / f"edge_{edge_limit:g}deg"
                        )
                        if solution.success:
                            correction_evidence = pd.DataFrame()
                            if experiment in {"E3", "E4"}:
                                ordered_beta, evaluation, correction_evidence = (
                                    _best_corrected_top_cycle(
                                        config=config,
                                        environment=environment,
                                        ordered_target=target[phase_order],
                                        ordered_layers=experiment_layers,
                                        solution=solution,
                                        solver_seed=(
                                            int(config["seeds"]["graph"])
                                            + root_seed_count
                                            + candidate_cap
                                        ),
                                    )
                                )
                            else:
                                ordered_beta = solution.beta_rad
                                evaluation = _evaluate_cycle(
                                    config=config,
                                    environment=environment,
                                    target=target[phase_order],
                                    beta=ordered_beta,
                                )
                            beta = _restore_phase_order(
                                ordered_beta, phase_order
                            )
                            if not correction_evidence.empty:
                                _atomic_parquet(
                                    correction_evidence,
                                    run_dir / "whole_curve_candidate_evidence.parquet",
                                )
                        else:
                            diagnostic = solve_sparse_cycle(
                                experiment_layers,
                                experiment_residuals,
                                beta_weights=config["cycle"]["beta_weights"],
                                edge_limit_deg=float(
                                    config["diagnostic_relaxed_gate"][
                                        "delta_beta_rms_max_deg"
                                    ]
                                ),
                                lambda_velocity=float(
                                    config["cycle"]["lambda_velocity"]
                                ),
                                lambda_acceleration=float(
                                    config["cycle"]["lambda_acceleration"]
                                ),
                                top_m=1,
                                max_states_per_root=int(
                                    config["cycle"]["max_states_per_root"]
                                ),
                            )
                            if diagnostic.success:
                                beta = _restore_phase_order(
                                    diagnostic.beta_rad, phase_order
                                )
                                evaluation = _evaluate_cycle(
                                    config=config,
                                    environment=environment,
                                    target=target[phase_order],
                                    beta=diagnostic.beta_rad,
                                )
                            else:
                                beta = np.zeros((0, 6))
                                evaluation = {
                                    "numerical_metrics": {},
                                    "strict_checks": {
                                        name: False
                                        for name in _strict_gate(
                                            config
                                        ).__dataclass_fields__
                                    },
                                    "gate_pass": False,
                                    "diagnostic_relaxed_cycle_pass": False,
                                    "minimal_slack": {
                                        "no_cycle_within_relaxed_edge_limit": 1.0e6
                                    },
                                }
                        outcome = classify_full_loop_outcome(
                            strict_cycle_pass=bool(evaluation["gate_pass"]),
                            empty_feasible_layer_indices=empty,
                            targeted_search_complete=targeted_search_complete,
                            diagnostic_relaxed_cycle_pass=bool(
                                evaluation[
                                    "diagnostic_relaxed_cycle_pass"
                                ]
                            ),
                        )
                        metadata = {
                            "candidate_id": candidate_id,
                            "experiment": experiment,
                            "root_seed_count": root_seed_count,
                            "candidate_cap": candidate_cap,
                            "edge_limit_deg": edge_limit,
                            "cycle_found": bool(solution.success),
                            "cycle_reason": str(solution.reason),
                            "state_pruning_count": int(
                                solution.state_pruning_count
                            ),
                            "state_dominance_count": int(
                                solution.state_dominance_count
                            ),
                            "max_states_per_root": solution.max_states_per_root,
                            "outcome": outcome,
                            "empty_feasible_layer_indices": empty,
                            "root_selection": (
                                "current_root"
                                if experiment == "E1"
                                else "survival_ranked_top16"
                                if experiment in {"E2", "E3"}
                                else "full_loop_viable_only"
                            ),
                            "allowed_root_seed_indices": sorted(
                                experiment_root_indices
                            ),
                            "top_cycles": list(solution.top_cycles),
                        }
                        if solution.success:
                            metadata["full_curve_cluster_summary"] = (
                                _write_full_curve_cluster_artifacts(
                                    directory=run_dir,
                                    ordered_layers=experiment_layers,
                                    top_cycles=solution.top_cycles,
                                    selected_beta=ordered_beta,
                                    beta_weights=config["cycle"][
                                        "beta_weights"
                                    ],
                                )
                            )
                            _write_cycle_artifact(
                                directory=run_dir,
                                beta=beta,
                                target=target,
                                evaluation=evaluation,
                                metadata=metadata,
                            )
                        else:
                            run_dir.mkdir(parents=True, exist_ok=True)
                            atomic_write_json(
                                run_dir / "gate.json",
                                {**metadata, **evaluation},
                            )
                        slack_total = sum(
                            float(value)
                            for value in evaluation["minimal_slack"].values()
                        )
                        ranking_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "experiment": experiment,
                                "root_seed_count": root_seed_count,
                                "candidate_cap": candidate_cap,
                                "edge_limit_deg": edge_limit,
                                "cycle_found": bool(solution.success),
                                "gate_pass": bool(evaluation["gate_pass"]),
                                "passed_check_count": sum(
                                    bool(value)
                                    for value in evaluation[
                                        "strict_checks"
                                    ].values()
                                ),
                                "minimal_slack_total": slack_total,
                                "empty_feasible_layer_count": len(empty),
                                "targeted_search_complete": (
                                    targeted_search_complete
                                ),
                                "outcome": outcome,
                            }
                        )
    ranking = _sort_pilot_ranking(pd.DataFrame(ranking_rows), config)
    rows_path, gate_path = _pilot_slice_paths(stage, pilot_task)
    rows_path.parent.mkdir(parents=True, exist_ok=False)
    ranking.to_csv(rows_path, index=False)
    expected_matrix_rows = (
        len(config["pilot_matrix"]["candidate_caps"])
        * 4
        * len(config["pilot_matrix"]["edge_limits_deg"])
    )
    if bool(pilot_task["include_e0"]):
        expected_matrix_rows += 1
    artifact_gate_count = 0
    for row in ranking.to_dict(orient="records"):
        if str(row["experiment"]) == "E0":
            artifact_gate = stage / str(row["candidate_id"]) / "E0" / "gate.json"
        else:
            artifact_gate = (
                stage
                / str(row["candidate_id"])
                / f"Nroot_{int(row['root_seed_count']):04d}"
                / f"K_{int(row['candidate_cap']):02d}"
                / str(row["experiment"])
                / f"edge_{float(row['edge_limit_deg']):g}deg"
                / "gate.json"
            )
        artifact_gate_count += int(artifact_gate.is_file())
    return _write_gate(
        gate_path,
        checks={
            "task_candidate_matches": set(ranking["candidate_id"])
            == {str(pilot_task["candidate_id"])},
            "task_root_seed_matches": set(
                ranking.loc[
                    ranking["experiment"] != "E0", "root_seed_count"
                ].astype(int)
            )
            == {int(pilot_task["root_seed_count"])},
            "expected_matrix_rows_complete": len(ranking)
            == expected_matrix_rows,
            "all_row_artifact_gates_present": artifact_gate_count
            == len(ranking),
        },
        task=dict(pilot_task),
        row_count=int(len(ranking)),
        artifact_gate_count=artifact_gate_count,
        targeted_empty_layer_search_complete=bool(
            ranking.loc[
                ranking["experiment"] != "E0",
                "targeted_search_complete",
            ]
            .astype(bool)
            .all()
        ),
        rows_sha256=sha256_file(rows_path),
    )


def _finalize_pilot_ranking(
    *,
    config: Mapping[str, Any],
    stage: Path,
    ranking: pd.DataFrame,
) -> dict[str, Any]:
    ranking = _sort_pilot_ranking(ranking, config)
    ranking.to_csv(stage / "pilot_matrix.csv", index=False)
    eligible = ranking[ranking["experiment"].isin(["E1", "E2", "E3", "E4"])]
    methods = (
        eligible.groupby(
            [
                "experiment",
                "root_seed_count",
                "candidate_cap",
                "edge_limit_deg",
            ],
            as_index=False,
        )
        .agg(
            candidate_count=("candidate_id", "nunique"),
            strict_pass_count=("gate_pass", "sum"),
            cycle_found_count=("cycle_found", "sum"),
            passed_check_count=("passed_check_count", "sum"),
            worst_minimal_slack=("minimal_slack_total", "max"),
            total_minimal_slack=("minimal_slack_total", "sum"),
            empty_layer_count=("empty_feasible_layer_count", "sum"),
            targeted_search_complete=(
                "targeted_search_complete",
                "all",
            ),
        )
        .sort_values(
            [
                "strict_pass_count",
                "cycle_found_count",
                "passed_check_count",
                "empty_layer_count",
                "worst_minimal_slack",
                "total_minimal_slack",
            ],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    priority = {"E1": 1, "E2": 2, "E3": 3, "E4": 4}
    methods["capability_priority"] = methods["experiment"].map(priority).astype(int)
    methods = methods.sort_values(
        [
            "strict_pass_count",
            "cycle_found_count",
            "passed_check_count",
            "empty_layer_count",
            "worst_minimal_slack",
            "total_minimal_slack",
            "capability_priority",
            "root_seed_count",
            "candidate_cap",
            "edge_limit_deg",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ],
        kind="stable",
    ).reset_index(drop=True)
    methods.to_csv(stage / "shared_method_ranking.csv", index=False)
    selected = methods.iloc[0].to_dict()
    selected = {
        key: (
            int(value)
            if key in {"root_seed_count", "candidate_cap"}
            else float(value)
            if key == "edge_limit_deg"
            else str(value)
            if key == "experiment"
            else value
        )
        for key, value in selected.items()
    }
    atomic_write_json(stage / "selected_formal_method.json", selected)
    return _write_gate(
        stage / "gate.json",
        checks={
            "two_candidates_complete": set(ranking["candidate_id"])
            == set(map(str, config["candidates"])),
            "five_experiments_present": set(ranking["experiment"])
            == {"E0", "E1", "E2", "E3", "E4"},
            "shared_formal_method_selected": bool(selected),
            "targeted_empty_layer_search_complete": bool(
                eligible["targeted_search_complete"].astype(bool).all()
            ),
        },
        selected_formal_method=selected,
        matrix_row_count=int(len(ranking)),
        strict_pass_row_count=int(ranking["gate_pass"].sum()),
    )


def _pilot_slice_is_valid(stage: Path, task: Mapping[str, Any]) -> bool:
    rows_path, gate_path = _pilot_slice_paths(stage, task)
    if not rows_path.is_file() or not gate_path.is_file():
        return False
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        gate.get("gate_pass")
        and gate.get("task") == dict(task)
        and gate.get("rows_sha256") == sha256_file(rows_path)
    )


def _pilot_task_has_partial_artifacts(
    stage: Path, task: Mapping[str, Any]
) -> bool:
    candidate_root = (
        stage
        / str(task["candidate_id"])
        / f"Nroot_{int(task['root_seed_count']):04d}"
    )
    e0_root = stage / str(task["candidate_id"]) / "E0"
    return candidate_root.exists() or (
        bool(task["include_e0"]) and e0_root.exists()
    )


def _read_process_usage(pid: int) -> tuple[float, int]:
    """Return sampled CPU seconds and RSS bytes for one Linux worker."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        clock_ticks = float(os.sysconf("SC_CLK_TCK"))
        cpu_seconds = (float(fields[11]) + float(fields[12])) / clock_ticks
        rss_kib = 0
        for line in Path(f"/proc/{pid}/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
        return cpu_seconds, rss_kib * 1024
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return 0.0, 0


def _host_memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[f"{key.lower()}_bytes"] = int(raw.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _tail_text(path: Path, *, character_limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[
            -character_limit:
        ]
    except OSError:
        return ""


def _run_pilot_subprocess_tasks(
    *,
    source_root: Path,
    project_root: Path,
    output: Path,
    config_path: Path,
    preset: str,
    task_files: Sequence[Path],
    task_specs: Sequence[Mapping[str, Any]],
    effective_workers: int,
    requested_workers: int,
    configured_max_workers: int,
    poll_seconds: float,
    per_worker_blas_threads: int,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run Pilot slices as independent subprocesses and record resource evidence."""

    task_by_path = {
        path: dict(spec) for path, spec in zip(task_files, task_specs, strict=True)
    }
    logs_dir = manifest_path.parent / "_parallel" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    started_wall = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records: dict[str, dict[str, Any]] = {}
    pending: list[Path] = []
    stage = output / STAGE_DIRS["pilot"]
    for task_file in task_files:
        task = task_by_path[task_file]
        if _pilot_slice_is_valid(stage, task):
            records[str(task["task_id"])] = {
                **task,
                "status": "reused_valid_slice",
                "exit_code": 0,
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "cpu_utilization_percent": 0.0,
                "peak_rss_bytes": 0,
            }
        else:
            if _pilot_task_has_partial_artifacts(stage, task):
                raise RuntimeError(
                    "Refusing to overwrite incomplete Pilot slice artifacts: "
                    f"{task['task_id']}"
                )
            pending.append(task_file)
    worker_limit_reasons = []
    if effective_workers < requested_workers:
        worker_limit_reasons.append(
            "effective_workers_capped_by_configured_max_or_independent_task_count"
        )
    if len(task_specs) == 4:
        worker_limit_reasons.append(
            "current_granularity_has_four_candidate_root_seed_slices"
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "execution_model": "independent_subprocess_workers",
        "started_utc": started_utc,
        "status": "running",
        "requested_workers": int(requested_workers),
        "configured_max_workers": int(configured_max_workers),
        "effective_workers": int(effective_workers),
        "independent_task_count": len(task_specs),
        "per_worker_blas_threads": int(per_worker_blas_threads),
        "logical_cpu_count": os.cpu_count(),
        "worker_limit_reasons": worker_limit_reasons,
        "host_memory_start": _host_memory_snapshot(),
        "peak_concurrent_worker_rss_bytes": 0,
        "tasks": [],
    }
    atomic_write_json(manifest_path, manifest)
    active: dict[subprocess.Popen[Any], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    env = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(per_worker_blas_threads)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-v11-4")
    while pending or active:
        while pending and len(active) < effective_workers:
            task_file = pending.pop(0)
            task = task_by_path[task_file]
            stdout_path = logs_dir / f"{task['task_id']}.stdout.log"
            stderr_path = logs_dir / f"{task['task_id']}.stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config_path),
                    "--preset",
                    str(preset),
                    "--project-root",
                    str(project_root),
                    "--output",
                    str(output),
                    "--pilot-worker-task",
                    str(task_file),
                ],
                cwd=str(source_root),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=env,
            )
            active[process] = {
                "task": task,
                "started": time.monotonic(),
                "pid": int(process.pid),
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
                "peak_rss_bytes": 0,
                "cpu_seconds": 0.0,
            }
        concurrent_rss = 0
        for process, state in active.items():
            cpu_seconds, rss_bytes = _read_process_usage(int(process.pid))
            state["cpu_seconds"] = max(state["cpu_seconds"], cpu_seconds)
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss_bytes)
            concurrent_rss += rss_bytes
        manifest["peak_concurrent_worker_rss_bytes"] = max(
            int(manifest["peak_concurrent_worker_rss_bytes"]), concurrent_rss
        )
        completed = [
            process for process in active if process.poll() is not None
        ]
        for process in completed:
            state = active.pop(process)
            state["stdout_handle"].close()
            state["stderr_handle"].close()
            wall_seconds = max(time.monotonic() - state["started"], 1.0e-9)
            cpu_seconds = float(state["cpu_seconds"])
            task = state["task"]
            record = {
                **task,
                "status": (
                    "completed" if process.returncode == 0 else "failed"
                ),
                "pid": int(state["pid"]),
                "exit_code": int(process.returncode),
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "cpu_utilization_percent": 100.0
                * cpu_seconds
                / wall_seconds,
                "peak_rss_bytes": int(state["peak_rss_bytes"]),
                "stdout_log": str(state["stdout_path"]),
                "stderr_log": str(state["stderr_path"]),
            }
            records[str(task["task_id"])] = record
            if process.returncode != 0:
                failures.append(
                    {
                        **record,
                        "stdout_tail": _tail_text(state["stdout_path"]),
                        "stderr_tail": _tail_text(state["stderr_path"]),
                    }
                )
            manifest["tasks"] = [
                records[str(spec["task_id"])]
                for spec in task_specs
                if str(spec["task_id"]) in records
            ]
            atomic_write_json(manifest_path, manifest)
        if pending or active:
            time.sleep(max(0.1, float(poll_seconds)))
    elapsed = max(time.monotonic() - started_wall, 1.0e-9)
    total_cpu = sum(float(record["cpu_seconds"]) for record in records.values())
    manifest.update(
        {
            "status": "failed" if failures else "completed",
            "completed_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "wall_seconds": elapsed,
            "total_worker_cpu_seconds": total_cpu,
            "aggregate_worker_cpu_utilization_percent": 100.0
            * total_cpu
            / elapsed,
            "mean_effective_worker_utilization_percent": 100.0
            * total_cpu
            / (elapsed * effective_workers),
            "host_memory_end": _host_memory_snapshot(),
            "tasks": [
                records[str(spec["task_id"])] for spec in task_specs
            ],
            "failures": failures,
        }
    )
    atomic_write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(
            "Pilot subprocess worker failures: "
            + json.dumps(failures, ensure_ascii=False)
        )
    return manifest


def run_pilot(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    environment: Any,
    output: Path,
    config_path: Path,
    preset: str,
    pilot_workers: int | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Execute and deterministically merge coarse-grained Pilot slices."""

    stage = output / STAGE_DIRS["pilot"]
    parallel_root = stage / "_parallel"
    tasks_root = parallel_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    task_specs = _pilot_task_specs(config)
    task_files = []
    for task in task_specs:
        task_file = tasks_root / f"{task['task_id']}.json"
        atomic_write_json(task_file, task)
        task_files.append(task_file)
    configured_workers = int(config["pilot_matrix"]["parallel_workers"])
    configured_max_workers = int(
        config["pilot_matrix"]["max_parallel_workers"]
    )
    requested_workers = (
        configured_workers if pilot_workers is None else int(pilot_workers)
    )
    if requested_workers < 1:
        raise ValueError("Pilot worker count must be positive")
    effective_workers = min(
        requested_workers, configured_max_workers, len(task_specs)
    )
    _run_pilot_subprocess_tasks(
        source_root=source_root,
        project_root=project_root,
        output=output,
        config_path=config_path,
        preset=preset,
        task_files=task_files,
        task_specs=task_specs,
        effective_workers=effective_workers,
        requested_workers=requested_workers,
        configured_max_workers=configured_max_workers,
        poll_seconds=float(
            config["pilot_matrix"]["monitoring_poll_seconds"]
        ),
        per_worker_blas_threads=int(
            config["pilot_matrix"]["per_worker_blas_threads"]
        ),
        manifest_path=stage / "pilot_parallel_manifest.json",
    )
    frames = []
    for task in task_specs:
        if not _pilot_slice_is_valid(stage, task):
            raise RuntimeError(
                f"Pilot slice missing or invalid after worker completion: "
                f"{task['task_id']}"
            )
        rows_path, _gate_path = _pilot_slice_paths(stage, task)
        frames.append(pd.read_csv(rows_path))
    ranking = _sort_pilot_ranking(pd.concat(frames, ignore_index=True), config)
    return _finalize_pilot_ranking(
        config=config,
        stage=stage,
        ranking=ranking,
    )


def _audit_variant_specs(
    phase_count: int,
    cut_phase_indices: Sequence[int],
    cut_angles_deg: Sequence[float],
) -> list[tuple[str, str, int, int]]:
    normalized_cuts = sorted(
        {
            int(value) % int(phase_count)
            for value in cut_phase_indices
        }
        | {
            int(round(float(value) * int(phase_count) / 360.0))
            % int(phase_count)
            for value in cut_angles_deg
        }
    )
    output = [
        ("primary", "forward", 0, 0),
        ("repeat", "forward", 0, 0),
    ]
    output.extend(
        (f"forward_cut{cut:04d}", "forward", cut, index + 1)
        for index, cut in enumerate(normalized_cuts)
        if cut != 0
    )
    output.extend(
        (f"reverse_cut{cut:04d}", "reverse", cut, index + 100)
        for index, cut in enumerate(normalized_cuts)
    )
    return output


def _solve_cycle_audit_variant(
    *,
    config: Mapping[str, Any],
    environment: Any,
    target: np.ndarray,
    canonical_beta: np.ndarray,
    name: str,
    direction: str,
    cut: int,
    solver_seed: int,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    count = len(target)
    order = _lineage_order(count, int(cut), direction)
    selected = np.full_like(canonical_beta, np.nan)
    evidence_rows: list[dict[str, Any]] = []
    solver_failures: list[int] = []
    trust_violations = 0
    policy = _teacher_policy(config, seed=int(solver_seed))
    rng = np.random.default_rng(int(solver_seed))
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    for traversal_idx, raw_phase in enumerate(order):
        phase = int(raw_phase)
        previous_phase = int(order[traversal_idx - 1]) if traversal_idx else phase
        previous_beta = (
            selected[previous_phase]
            if traversal_idx
            else canonical_beta[phase]
        )
        before_beta = (
            selected[int(order[traversal_idx - 2])]
            if traversal_idx >= 2
            else None
        )
        seeds = _continuation_seed_bank(
            environment=environment,
            target=target[phase],
            previous_target=(
                target[previous_phase] if traversal_idx else target[phase]
            ),
            previous_beta=previous_beta,
            before_beta=before_beta,
            reference_beta=canonical_beta[phase],
            candidate_budget=int(config["viability"]["candidate_budget"]),
            rng=rng,
            local_seed_std_deg=float(config["viability"]["local_seed_std_deg"]),
        )
        beta_values: list[np.ndarray] = []
        phase_rows: list[dict[str, Any]] = []
        for source, seed_beta in seeds:
            beta, row = _correct_seed(
                environment=environment,
                target=target[phase],
                seed=np.clip(seed_beta, bounds[:, 0], bounds[:, 1]),
                policy=policy,
                candidate_source=source,
                reference_beta=canonical_beta[phase],
                previous_beta=previous_beta,
                trust_radius_deg=float(
                    config["formal_audit"]["trust_radius_deg"]
                ),
            )
            row.update(
                {
                    "variant": str(name),
                    "phase_idx": phase,
                    "traversal_idx": traversal_idx,
                    "selected": False,
                }
            )
            beta_values.append(beta)
            phase_rows.append(row)
        table = pd.DataFrame(phase_rows)
        hard = (
            table["corrector_success"].astype(bool).to_numpy()
            & table["within_bounds"].astype(bool).to_numpy()
            & table["trust_pass"].astype(bool).to_numpy()
            & (
                table["residual_mm"].to_numpy(float)
                <= float(config["hard_feasibility"]["residual_max_mm"])
            )
            & (
                table["joint_margin_deg"].to_numpy(float)
                >= float(config["hard_feasibility"]["joint_margin_min_deg"])
            )
        )
        feasible = np.flatnonzero(hard)
        if len(feasible) == 0:
            solver_failures.append(phase)
            if not table["trust_pass"].astype(bool).any():
                trust_violations += 1
            evidence_rows.extend(phase_rows)
            break
        choice = (
            table.iloc[feasible]
            .sort_values(
                [
                    "reference_gap_deg",
                    "previous_gap_deg",
                    "residual_mm",
                    "joint_margin_deg",
                ],
                ascending=[True, True, True, False],
                kind="stable",
            )
            .index[0]
        )
        phase_rows[int(choice)]["selected"] = True
        selected[phase] = beta_values[int(choice)]
        evidence_rows.extend(phase_rows)
    success = bool(not solver_failures and np.isfinite(selected).all())
    report = {
        "variant": str(name),
        "direction": str(direction),
        "cut": int(cut),
        "solver_success": success,
        "trust_violation_count": int(trust_violations),
        "solver_failure_phase_indices": solver_failures,
        "reference_copy_phase_indices": [],
    }
    return selected, report, pd.DataFrame(evidence_rows)


def _audit_formal_cycle(
    *,
    config: Mapping[str, Any],
    environment: Any,
    target: np.ndarray,
    canonical_beta: np.ndarray,
    solver_seed: int,
    directory: Path,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    variant_paths: dict[str, np.ndarray] = {}
    for name, direction, cut, seed_offset in _audit_variant_specs(
        len(target),
        config["formal_audit"]["cut_phase_indices"],
        config["formal_audit"]["cut_angles_deg"],
    ):
        variant_seed = (
            int(solver_seed)
            if name in {"primary", "repeat"}
            else int(solver_seed) + seed_offset
        )
        beta, report, evidence = _solve_cycle_audit_variant(
            config=config,
            environment=environment,
            target=target,
            canonical_beta=canonical_beta,
            name=name,
            direction=direction,
            cut=cut,
            solver_seed=variant_seed,
        )
        reports.append(report)
        evidence_frames.append(evidence)
        variant_paths[name] = beta
    summary_rows: list[dict[str, Any]] = []
    primary = variant_paths["primary"]
    for name, beta in variant_paths.items():
        comparison = primary if name == "repeat" else canonical_beta
        if not np.isfinite(beta).all():
            summary_rows.append(
                {
                    "variant": name,
                    "gap_p95_deg": 1.0e12,
                    "gap_max_deg": 1.0e12,
                }
            )
            continue
        gap = weighted_rms_gap_deg(
            beta,
            comparison,
            beta_weights=config["cycle"]["beta_weights"],
        )
        summary_rows.append(
            {
                "variant": name,
                "gap_p95_deg": float(np.percentile(gap, 95)),
                "gap_max_deg": float(np.max(gap)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    report_table = pd.DataFrame(reports)
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "variant_gap_summary.csv", index=False)
    atomic_write_json(directory / "variant_solver_reports.json", reports)
    _atomic_parquet(
        pd.concat(evidence_frames, ignore_index=True),
        directory / "candidate_evidence.parquet",
    )
    variant_root = directory / "variants"
    for name, beta in variant_paths.items():
        if not np.isfinite(beta).all():
            continue
        frame = pd.DataFrame({"phase_idx": np.arange(len(beta))})
        for joint in range(6):
            frame[f"teacher_beta{joint + 1}_rad"] = beta[:, joint]
        _atomic_parquet(frame, variant_root / name / "centerline.parquet")
    curve_names = ["canonical", *[
        name for name, beta in variant_paths.items() if np.isfinite(beta).all()
    ]]
    curve_values = [
        canonical_beta,
        *[
            beta
            for beta in variant_paths.values()
            if np.isfinite(beta).all()
        ],
    ]
    curve_clusters = cluster_full_loop_cycles(
        curve_values,
        threshold_deg=float(
            config["formal_audit"]["variant_beta_rms_p95_deg"]
        ),
        beta_weights=config["cycle"]["beta_weights"],
    )
    pd.DataFrame(
        {
            "curve_id": curve_names,
            "curve_cluster_id": curve_clusters.component_labels,
        }
    ).to_csv(directory / "full_curve_cluster_membership.csv", index=False)
    pd.DataFrame(
        [
            {
                "left_curve_id": curve_names[left],
                "right_curve_id": curve_names[right],
                "curve_gap_p95_deg": float(
                    curve_clusters.distance_deg[left, right]
                ),
            }
            for left in range(len(curve_names))
            for right in range(left + 1, len(curve_names))
        ]
    ).to_csv(directory / "full_curve_pairwise_distance.csv", index=False)
    repeat = summary[summary["variant"] == "repeat"]
    traversal = summary[summary["variant"] != "repeat"]
    checks = {
        "all_solver_success": bool(report_table["solver_success"].all()),
        "no_trust_violations": bool(
            (report_table["trust_violation_count"] == 0).all()
        ),
        "no_reference_copy": bool(
            all(not values for values in report_table["reference_copy_phase_indices"])
        ),
        "repeat_p95": bool(
            len(repeat) == 1
            and float(repeat.iloc[0]["gap_p95_deg"])
            <= float(config["formal_audit"]["repeat_beta_rms_p95_deg"])
        ),
        "traversal_p95": bool(
            not traversal.empty
            and float(traversal["gap_p95_deg"].max())
            <= float(config["formal_audit"]["variant_beta_rms_p95_deg"])
        ),
        "traversal_max": bool(
            not traversal.empty
            and float(traversal["gap_max_deg"].max())
            <= float(config["formal_audit"]["variant_beta_rms_max_deg"])
        ),
        "single_full_curve_cluster": bool(
            curve_clusters.connected_component_count == 1
        ),
    }
    payload = {
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
        "variant_summary": summary.to_dict("records"),
        "variant_reports": reports,
        "full_curve_cluster_count": (
            curve_clusters.connected_component_count
        ),
    }
    atomic_write_json(directory / "gate.json", payload)
    return payload


def _formal_lineages(
    *,
    config: Mapping[str, Any],
    environment: Any,
    target: np.ndarray,
    reference: np.ndarray,
    root_phase: int,
    root_candidates: pd.DataFrame,
    top_root_indices: Sequence[int],
    solver_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beta_columns = [f"beta{joint}_rad" for joint in range(1, 7)]
    summaries: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    for order, root_seed_idx in enumerate(map(int, top_root_indices)):
        seed_beta = root_candidates.iloc[root_seed_idx][beta_columns].to_numpy(float)
        root_beta, _record = _correct_seed(
            environment=environment,
            target=target[root_phase],
            seed=seed_beta,
            policy=_teacher_policy(config, seed=int(solver_seed) + order),
            candidate_source="formal_root_recorrection",
            reference_beta=reference[root_phase],
        )
        root_hard = bool(
            _record["corrector_success"]
            and _record["within_bounds"]
            and float(_record["residual_mm"])
            <= float(config["hard_feasibility"]["residual_max_mm"])
            and float(_record["joint_margin_deg"])
            >= float(config["hard_feasibility"]["joint_margin_min_deg"])
        )
        if not root_hard:
            for direction in ("forward", "reverse"):
                failed_record = {
                    **_record,
                    "phase_idx": int(root_phase),
                    "traversal_idx": 0,
                    "selected": False,
                    "root_seed_idx": int(root_seed_idx),
                    "direction": direction,
                    "root_recorrection_hard_feasible": False,
                }
                evidence_frames.append(pd.DataFrame([failed_record]))
                summaries.append(
                    {
                        "root_seed_idx": int(root_seed_idx),
                        "direction": direction,
                        "completed_360": False,
                        "strict_feasible_ratio": 0.0,
                        "solved_phase_count": 0,
                        "first_failure_phase_idx": int(root_phase),
                        "joint_margin_min_deg": float(
                            _record["joint_margin_deg"]
                        ),
                        "residual_max_mm": float(_record["residual_mm"]),
                        "residual_p95_mm": float(_record["residual_mm"]),
                        "velocity_p95_deg": 1.0e12,
                        "velocity_max_deg": 1.0e12,
                        "acceleration_p95_deg": 1.0e12,
                        "seam_deg": 1.0e12,
                        "kappa_p95": float(_record["kappa"]),
                        "root_recorrection_hard_feasible": False,
                    }
                )
            continue
        for direction_offset, direction in enumerate(("forward", "reverse")):
            summary, selected, evidence = _solve_lineage(
                config=config,
                environment=environment,
                target=target,
                reference=reference,
                root_phase=root_phase,
                root_beta=root_beta,
                root_seed_idx=root_seed_idx,
                direction=direction,
                policy=_teacher_policy(
                    config,
                    seed=int(solver_seed) + 100 * order + direction_offset,
                ),
                solver_seed=int(solver_seed) + 100 * order + direction_offset,
            )
            summaries.append(summary)
            selected_frames.append(selected)
            evidence_frames.append(evidence)
    empty_columns = [
        *CANDIDATE_EVIDENCE_COLUMNS,
        "phase_idx",
        "traversal_idx",
        "selected",
        "root_seed_idx",
        "direction",
    ]
    return (
        pd.DataFrame(summaries),
        (
            pd.concat(selected_frames, ignore_index=True)
            if selected_frames
            else pd.DataFrame(columns=empty_columns)
        ),
        (
            pd.concat(evidence_frames, ignore_index=True)
            if evidence_frames
            else pd.DataFrame(columns=empty_columns)
        ),
    )


def _formal_task_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"{candidate_offset:02d}_{candidate_id}",
            "candidate_id": candidate_id,
            "candidate_offset": candidate_offset,
        }
        for candidate_offset, candidate_id in enumerate(
            map(str, config["candidates"])
        )
    ]


def _run_formal_candidate(
    *,
    config: Mapping[str, Any],
    project_root: Path,
    environment: Any,
    output: Path,
    formal_task: Mapping[str, Any],
    **_unused: Any,
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["formal"]
    selected_method_path = (
        output
        / STAGE_DIRS["pilot"]
        / "selected_formal_method.json"
    )
    selected_method = json.loads(
        selected_method_path.read_text(encoding="utf-8")
    )
    experiment = str(selected_method["experiment"])
    root_seed_count = int(selected_method["root_seed_count"])
    candidate_cap = int(selected_method["candidate_cap"])
    edge_limit = float(selected_method["edge_limit_deg"])
    candidate_reports: list[dict[str, Any]] = []
    for candidate_offset, candidate_id in enumerate(map(str, config["candidates"])):
        if candidate_id != str(formal_task["candidate_id"]):
            continue
        if candidate_offset != int(formal_task["candidate_offset"]):
            raise RuntimeError(
                "Formal task candidate_offset does not match frozen candidate order"
            )
        candidate_dir = stage / candidate_id
        root_summary_path = (
            output
            / STAGE_DIRS["root_fiber"]
            / candidate_id
            / "root_fiber_summary.json"
        )
        viability_gate_path = (
            output
            / STAGE_DIRS["viability"]
            / candidate_id
            / f"Nroot_{root_seed_count:04d}"
            / "gate.json"
        )
        root_candidates_path = (
            output
            / STAGE_DIRS["root_fiber"]
            / candidate_id
            / "root_candidates.parquet"
        )
        viability_directory = viability_gate_path.parent
        source_anchor_candidate = (
            project_root
            / str(config["source_anchor_root"])
            / "verify"
            / candidate_id
        )
        source_centerline_paths = sorted(
            source_anchor_candidate.glob("*/centerline.parquet")
        )
        if not source_centerline_paths:
            raise FileNotFoundError(
                f"no source centerline artifacts under {source_anchor_candidate}"
            )
        source_artifact = (
            project_root / str(config["source_artifact_root"])
        )
        source_inputs = [
            Path(str(config["config_path"])),
            selected_method_path,
            root_summary_path,
            root_candidates_path,
            viability_gate_path,
            viability_directory / "top_survival_ranked_roots.csv",
            viability_directory / "full_loop_viable_roots.csv",
            viability_directory / "selected_lineages.parquet",
            viability_directory / "candidate_evidence.parquet",
            source_artifact / "BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json",
            source_artifact / "03_formal" / "gate.json",
            source_artifact
            / "03_formal"
            / candidate_id
            / "BI-3"
            / "canonical_consensus_branch.parquet",
            *source_centerline_paths,
        ]
        input_sha256 = {
            path.relative_to(project_root).as_posix()
            if path.is_relative_to(project_root)
            else str(path): sha256_file(path)
            for path in source_inputs
        }
        resolved_config = copy.deepcopy(dict(config))
        resolved_config.pop("config_path", None)
        candidate_fingerprint = _canonical_sha(
            {
                "protocol_id": config["protocol_id"],
                "preset": config["preset"],
                "candidate_id": candidate_id,
                "selected_method": selected_method,
                "resolved_config": resolved_config,
                "input_sha256": input_sha256,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "module_sha256": sha256_file(
                    SOURCE_ROOT
                    / "src/quasi_exp/teacher/full_loop_feasibility.py"
                ),
            }
        )
        report_path = candidate_dir / "formal_report.json"
        if report_path.is_file():
            cached_report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            if (
                cached_report.get("candidate_fingerprint")
                == candidate_fingerprint
                and _artifact_manifest_is_valid(
                    candidate_dir,
                    cached_report.get("candidate_artifact_sha256", {}),
                )
            ):
                candidate_reports.append(cached_report)
                continue
        primary, variants_frame = _aligned_source_frames(
            config,
            project_root,
            candidate_id,
            count=int(config["phase_counts"]["formal"]),
        )
        target = _frame_target(primary)
        reference = _frame_beta(primary)
        variants = {
            name: _frame_beta(frame) for name, frame in variants_frame.items()
        }
        root_summary = json.loads(
            root_summary_path.read_text(encoding="utf-8")
        )
        root_phase = int(
            round(
                int(root_summary["phase_idx"])
                * len(target)
                / int(config["phase_counts"]["pilot"])
            )
        ) % len(target)
        root_candidates = pd.read_parquet(
            output
            / STAGE_DIRS["root_fiber"]
            / candidate_id
            / "root_candidates.parquet"
        )
        top = pd.read_csv(
            output
            / STAGE_DIRS["viability"]
            / candidate_id
            / f"Nroot_{root_seed_count:04d}"
            / "top_survival_ranked_roots.csv"
        )
        full_loop = pd.read_csv(
            output
            / STAGE_DIRS["viability"]
            / candidate_id
            / f"Nroot_{root_seed_count:04d}"
            / "full_loop_viable_roots.csv"
        )
        survival_root_indices = set(
            top["root_candidate_idx"].astype(int).tolist()
        )
        full_loop_root_indices = set(
            full_loop["root_candidate_idx"].astype(int).tolist()
        )
        formal_root_indices = (
            [0]
            if experiment == "E1"
            else sorted(survival_root_indices)
            if experiment in {"E2", "E3"}
            else sorted(full_loop_root_indices)
        )
        (
            direction_summary,
            selected_lineages,
            lineage_evidence,
        ) = _formal_lineages(
            config=config,
            environment=environment,
            target=target,
            reference=reference,
            root_phase=root_phase,
            root_candidates=root_candidates,
            top_root_indices=formal_root_indices,
            solver_seed=int(config["seeds"]["viability"])
            + 1000000
            + candidate_offset * 100000,
        )
        candidate_dir.mkdir(parents=True, exist_ok=True)
        direction_summary.to_csv(
            candidate_dir / "direction_survival_summary.csv", index=False
        )
        _atomic_parquet(
            selected_lineages, candidate_dir / "selected_lineages.parquet"
        )
        _atomic_parquet(
            lineage_evidence,
            candidate_dir / "lineage_candidate_evidence.parquet",
        )
        (
            layers,
            residuals,
            nodes,
            counts,
            graph_evidence,
        ) = _generate_hard_layers(
            config=config,
            environment=environment,
            target=target,
            reference=reference,
            variants=variants,
            selected_lineages=selected_lineages,
            top_root_seed_indices=set(formal_root_indices),
            candidate_cap=candidate_cap,
            solver_seed=int(config["seeds"]["graph"])
            + 1000000
            + candidate_offset * 100000,
        )
        _atomic_parquet(nodes, candidate_dir / "hard_feasible_nodes.parquet")
        _atomic_parquet(
            graph_evidence, candidate_dir / "graph_candidate_evidence.parquet"
        )
        slack_by_phase = _minimal_slack_by_phase(graph_evidence, config)
        slack_by_phase.to_csv(
            candidate_dir / "minimal_slack_by_phase.csv", index=False
        )
        slack_phase_summary = _slack_phase_summary(slack_by_phase)
        atomic_write_json(
            candidate_dir / "minimal_slack_summary.json",
            slack_phase_summary,
        )
        counts.to_csv(
            candidate_dir / "feasible_candidate_count_by_phase.csv", index=False
        )
        empty = counts.loc[
            counts["feasible_candidate_count"] == 0, "phase_idx"
        ].astype(int).tolist()
        targeted_search_complete = bool(
            counts.loc[
                counts["feasible_candidate_count"] == 0,
                "targeted_search_complete",
            ].astype(bool).all()
        )
        phase_order = _lineage_order(len(target), root_phase, "forward")
        ordered_layers = [layers[int(phase)] for phase in phase_order]
        ordered_residuals = [residuals[int(phase)] for phase in phase_order]
        root_nodes = nodes[
            nodes["phase_idx"].astype(int) == root_phase
        ].sort_values("layer_candidate_idx", kind="stable")
        restriction = {
            "E1": "current",
            "E2": "roots",
            "E3": "roots",
            "E4": "roots",
        }[experiment]
        allowed_root_indices = (
            survival_root_indices
            if experiment in {"E2", "E3"}
            else full_loop_root_indices
            if experiment == "E4"
            else set()
        )
        experiment_layers, experiment_residuals = _restricted_root_layers(
            ordered_layers,
            ordered_residuals,
            root_nodes=root_nodes,
            mode=restriction,
            root_candidates=root_candidates,
            allowed_root_indices=allowed_root_indices,
            beta_weights=config["cycle"]["beta_weights"],
        )
        solution = solve_sparse_cycle(
            experiment_layers,
            experiment_residuals,
            beta_weights=config["cycle"]["beta_weights"],
            edge_limit_deg=edge_limit,
            lambda_velocity=float(config["cycle"]["lambda_velocity"]),
            lambda_acceleration=float(config["cycle"]["lambda_acceleration"]),
            top_m=int(config["cycle"]["top_m"]),
            max_states_per_root=int(
                config["cycle"]["max_states_per_root"]
            ),
        )
        audit: dict[str, Any] = {
            "checks": {"cycle_exists_before_audit": False},
            "gate_pass": False,
            "variant_summary": [],
            "variant_reports": [],
        }
        exact_cycle_fallback_attempted = False
        if solution.success:
            if experiment in {"E3", "E4"}:
                ordered_beta, evaluation, correction_evidence = (
                    _best_corrected_top_cycle(
                        config=config,
                        environment=environment,
                        ordered_target=target[phase_order],
                        ordered_layers=experiment_layers,
                        solution=solution,
                        solver_seed=int(config["seeds"]["graph"])
                        + 2000000
                        + candidate_offset,
                    )
                )
                _atomic_parquet(
                    correction_evidence,
                    candidate_dir / "whole_curve_candidate_evidence.parquet",
                )
            else:
                ordered_beta = solution.beta_rad
                evaluation = _evaluate_cycle(
                    config=config,
                    environment=environment,
                    target=target[phase_order],
                    beta=ordered_beta,
                )
            if (
                not bool(evaluation["gate_pass"])
                and int(solution.state_pruning_count) > 0
            ):
                exact_cycle_fallback_attempted = True
                exact_solution = solve_sparse_cycle(
                    experiment_layers,
                    experiment_residuals,
                    beta_weights=config["cycle"]["beta_weights"],
                    edge_limit_deg=edge_limit,
                    lambda_velocity=float(config["cycle"]["lambda_velocity"]),
                    lambda_acceleration=float(
                        config["cycle"]["lambda_acceleration"]
                    ),
                    top_m=int(config["cycle"]["top_m"]),
                    max_states_per_root=int(
                        config["cycle"][
                            "fallback_max_states_per_root"
                        ]
                    ),
                )
                if exact_solution.success:
                    if experiment in {"E3", "E4"}:
                        (
                            exact_beta,
                            exact_evaluation,
                            exact_correction_evidence,
                        ) = _best_corrected_top_cycle(
                            config=config,
                            environment=environment,
                            ordered_target=target[phase_order],
                            ordered_layers=experiment_layers,
                            solution=exact_solution,
                            solver_seed=int(config["seeds"]["graph"])
                            + 3000000
                            + candidate_offset,
                        )
                    else:
                        exact_beta = exact_solution.beta_rad
                        exact_evaluation = _evaluate_cycle(
                            config=config,
                            environment=environment,
                            target=target[phase_order],
                            beta=exact_beta,
                        )
                        exact_correction_evidence = pd.DataFrame()
                    current_key = (
                        not bool(evaluation["gate_pass"]),
                        -sum(
                            bool(value)
                            for value in evaluation["strict_checks"].values()
                        ),
                        sum(evaluation["minimal_slack"].values()),
                    )
                    exact_key = (
                        not bool(exact_evaluation["gate_pass"]),
                        -sum(
                            bool(value)
                            for value in exact_evaluation[
                                "strict_checks"
                            ].values()
                        ),
                        sum(exact_evaluation["minimal_slack"].values()),
                    )
                    if exact_key < current_key:
                        solution = exact_solution
                        ordered_beta = exact_beta
                        evaluation = exact_evaluation
                        if not exact_correction_evidence.empty:
                            _atomic_parquet(
                                exact_correction_evidence,
                                candidate_dir
                                / "whole_curve_candidate_evidence_exact.parquet",
                            )
            beta = _restore_phase_order(ordered_beta, phase_order)
            audit = _audit_formal_cycle(
                config=config,
                environment=environment,
                target=target,
                canonical_beta=beta,
                solver_seed=int(config["seeds"]["audit"]) + candidate_offset,
                directory=candidate_dir / "audit",
            )
            combined_pass = bool(
                evaluation["gate_pass"] and audit["gate_pass"]
            )
            curve_clusters = _write_full_curve_cluster_artifacts(
                directory=candidate_dir / "selected_cycle",
                ordered_layers=experiment_layers,
                top_cycles=solution.top_cycles,
                selected_beta=ordered_beta,
                beta_weights=config["cycle"]["beta_weights"],
            )
            _write_cycle_artifact(
                directory=candidate_dir / "selected_cycle",
                beta=beta,
                target=target,
                evaluation={
                    **evaluation,
                    "gate_pass": combined_pass,
                    "cycle_gate_pass": bool(evaluation["gate_pass"]),
                    "audit_gate_pass": bool(audit["gate_pass"]),
                },
                metadata={
                    "selected_method": selected_method,
                    "cycle_found": True,
                    "state_pruning_count": int(
                        solution.state_pruning_count
                    ),
                    "state_dominance_count": int(
                        solution.state_dominance_count
                    ),
                    "max_states_per_root": solution.max_states_per_root,
                    "exact_cycle_fallback_attempted": exact_cycle_fallback_attempted,
                    "top_cycles": list(solution.top_cycles),
                    "full_curve_cluster_summary": curve_clusters,
                },
            )
        else:
            diagnostic = solve_sparse_cycle(
                experiment_layers,
                experiment_residuals,
                beta_weights=config["cycle"]["beta_weights"],
                edge_limit_deg=float(
                    config["diagnostic_relaxed_gate"][
                        "delta_beta_rms_max_deg"
                    ]
                ),
                lambda_velocity=float(config["cycle"]["lambda_velocity"]),
                lambda_acceleration=float(
                    config["cycle"]["lambda_acceleration"]
                ),
                top_m=1,
                max_states_per_root=int(
                    config["cycle"]["max_states_per_root"]
                ),
            )
            if diagnostic.success:
                evaluation = _evaluate_cycle(
                    config=config,
                    environment=environment,
                    target=target[phase_order],
                    beta=diagnostic.beta_rad,
                )
            else:
                evaluation = {
                    "numerical_metrics": {},
                    "strict_checks": {},
                    "gate_pass": False,
                    "diagnostic_relaxed_cycle_pass": False,
                    "minimal_slack": {
                        "no_cycle_within_relaxed_edge_limit": 1.0e6
                    },
                }
            combined_pass = False
        numerical_outcome = classify_full_loop_outcome(
            strict_cycle_pass=bool(evaluation["gate_pass"]),
            empty_feasible_layer_indices=empty,
            targeted_search_complete=targeted_search_complete,
            diagnostic_relaxed_cycle_pass=bool(
                evaluation["diagnostic_relaxed_cycle_pass"]
            ),
        )
        negative_search_incomplete = bool(
            solution.success
            and not evaluation["gate_pass"]
            and (
                int(solution.state_pruning_count) > 0
                or int(solution.state_dominance_count) > 0
            )
        )
        if (
            negative_search_incomplete
            and not empty
            and targeted_search_complete
        ):
            numerical_outcome = "SEARCH_INCOMPLETE"
        decision = formal_decision(
            numerical_outcome=numerical_outcome,
            audit_pass=bool(audit["gate_pass"]),
        )
        report = {
            "candidate_id": candidate_id,
            "candidate_fingerprint": candidate_fingerprint,
            "input_sha256": input_sha256,
            "selected_method": selected_method,
            "phase_count": len(target),
            "cycle_found": bool(solution.success),
            "cycle_state_pruning_count": int(solution.state_pruning_count),
            "cycle_state_dominance_count": int(
                solution.state_dominance_count
            ),
            "cycle_negative_search_complete": not (
                negative_search_incomplete
            ),
            "exact_cycle_fallback_attempted": exact_cycle_fallback_attempted,
            "empty_feasible_layer_indices": empty,
            "targeted_search_complete": targeted_search_complete,
            "cycle_gate_pass": bool(evaluation["gate_pass"]),
            "audit_gate_pass": bool(audit["gate_pass"]),
            "formal_pass": combined_pass,
            "numerical_outcome": numerical_outcome,
            "audit_status": "pass" if audit["gate_pass"] else "fail",
            "formal_decision": decision,
            "outcome": decision,
            "numerical_metrics": evaluation["numerical_metrics"],
            "minimal_slack": evaluation["minimal_slack"],
            "minimal_slack_phase_diagnostic": slack_phase_summary,
        }
        report["candidate_artifact_sha256"] = _directory_artifact_manifest(
            candidate_dir,
            exclude=[report_path],
        )
        atomic_write_json(candidate_dir / "formal_report.json", report)
        candidate_reports.append(report)
    if len(candidate_reports) != 1:
        raise RuntimeError("Formal worker did not produce exactly one candidate report")
    return candidate_reports[0]


def _finalize_formal_reports(
    *,
    config: Mapping[str, Any],
    output: Path,
    selected_method: Mapping[str, Any],
    candidate_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage = output / STAGE_DIRS["formal"]
    selected_method = dict(selected_method)
    candidate_reports = [dict(report) for report in candidate_reports]
    passing = [
        row["candidate_id"] for row in candidate_reports if row["formal_pass"]
    ]
    decision = {
        row["candidate_id"]: row["formal_decision"]
        for row in candidate_reports
    }
    gate = _write_gate(
        stage / "gate.json",
        checks={
            "two_candidates_rerun_at_720": len(candidate_reports) == 2
            and all(
                row["phase_count"] == int(config["phase_counts"]["formal"])
                for row in candidate_reports
            ),
            "shared_method_used": all(
                row["selected_method"] == selected_method
                for row in candidate_reports
            ),
            "targeted_search_complete": all(
                row["targeted_search_complete"] for row in candidate_reports
            ),
            "at_least_one_strict_formal_pass": bool(passing),
        },
        selected_method=selected_method,
        candidate_reports=candidate_reports,
        decision_by_candidate=decision,
        passing_candidates=passing,
        downstream_authorized=bool(passing),
    )
    atomic_write_json(
        output / "FULL_LOOP_EXPERIMENT_COMPLETED.json",
        {
            "protocol_id": str(config["protocol_id"]),
            "full_loop_experiment_complete": True,
            "formal_gate_pass": bool(gate["gate_pass"]),
            "decision_by_candidate": decision,
            "passing_candidates": passing,
            "downstream_authorized": bool(passing),
        },
    )
    return gate


def _run_formal_subprocess_tasks(
    *,
    source_root: Path,
    project_root: Path,
    output: Path,
    config_path: Path,
    preset: str,
    task_files: Sequence[Path],
    task_specs: Sequence[Mapping[str, Any]],
    effective_workers: int,
    poll_seconds: float,
    per_worker_blas_threads: int,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run independent 720-phase candidate jobs with resource observations."""

    task_by_path = {
        path: dict(spec) for path, spec in zip(task_files, task_specs, strict=True)
    }
    logs_dir = manifest_path.parent / "_parallel" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    started_wall = time.monotonic()
    records: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "execution_model": "independent_subprocess_workers",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "effective_workers": int(effective_workers),
        "independent_task_count": len(task_specs),
        "per_worker_blas_threads": int(per_worker_blas_threads),
        "logical_cpu_count": os.cpu_count(),
        "host_memory_start": _host_memory_snapshot(),
        "peak_concurrent_worker_rss_bytes": 0,
        "tasks": [],
    }
    atomic_write_json(manifest_path, manifest)
    env = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(per_worker_blas_threads)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-v11-4")
    pending = list(task_files)
    active: dict[subprocess.Popen[Any], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    while pending or active:
        while pending and len(active) < effective_workers:
            task_file = pending.pop(0)
            task = task_by_path[task_file]
            stdout_path = logs_dir / f"{task['task_id']}.stdout.log"
            stderr_path = logs_dir / f"{task['task_id']}.stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config_path),
                    "--preset",
                    str(preset),
                    "--project-root",
                    str(project_root),
                    "--output",
                    str(output),
                    "--formal-worker-task",
                    str(task_file),
                ],
                cwd=str(source_root),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                env=env,
            )
            active[process] = {
                "task": task,
                "started": time.monotonic(),
                "pid": int(process.pid),
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
                "peak_rss_bytes": 0,
                "cpu_seconds": 0.0,
            }
        concurrent_rss = 0
        for process, state in active.items():
            cpu_seconds, rss_bytes = _read_process_usage(int(process.pid))
            state["cpu_seconds"] = max(state["cpu_seconds"], cpu_seconds)
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss_bytes)
            concurrent_rss += rss_bytes
        manifest["peak_concurrent_worker_rss_bytes"] = max(
            int(manifest["peak_concurrent_worker_rss_bytes"]), concurrent_rss
        )
        completed = [
            process for process in active if process.poll() is not None
        ]
        for process in completed:
            state = active.pop(process)
            state["stdout_handle"].close()
            state["stderr_handle"].close()
            wall_seconds = max(time.monotonic() - state["started"], 1.0e-9)
            cpu_seconds = float(state["cpu_seconds"])
            task = state["task"]
            record = {
                **task,
                "status": (
                    "completed" if process.returncode == 0 else "failed"
                ),
                "pid": int(state["pid"]),
                "exit_code": int(process.returncode),
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "cpu_utilization_percent": 100.0
                * cpu_seconds
                / wall_seconds,
                "peak_rss_bytes": int(state["peak_rss_bytes"]),
                "stdout_log": str(state["stdout_path"]),
                "stderr_log": str(state["stderr_path"]),
            }
            records[str(task["task_id"])] = record
            if process.returncode != 0:
                failures.append(
                    {
                        **record,
                        "stdout_tail": _tail_text(state["stdout_path"]),
                        "stderr_tail": _tail_text(state["stderr_path"]),
                    }
                )
            manifest["tasks"] = [
                records[str(spec["task_id"])]
                for spec in task_specs
                if str(spec["task_id"]) in records
            ]
            atomic_write_json(manifest_path, manifest)
        if pending or active:
            time.sleep(max(0.1, float(poll_seconds)))
    elapsed = max(time.monotonic() - started_wall, 1.0e-9)
    total_cpu = sum(float(record["cpu_seconds"]) for record in records.values())
    manifest.update(
        {
            "status": "failed" if failures else "completed",
            "completed_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "wall_seconds": elapsed,
            "total_worker_cpu_seconds": total_cpu,
            "aggregate_worker_cpu_utilization_percent": 100.0
            * total_cpu
            / elapsed,
            "mean_effective_worker_utilization_percent": 100.0
            * total_cpu
            / (elapsed * effective_workers),
            "host_memory_end": _host_memory_snapshot(),
            "tasks": [
                records[str(spec["task_id"])] for spec in task_specs
            ],
            "failures": failures,
        }
    )
    atomic_write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(
            "Formal subprocess worker failures: "
            + json.dumps(failures, ensure_ascii=False)
        )
    return manifest


def run_formal(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    environment: Any,
    output: Path,
    config_path: Path,
    preset: str,
    pilot_workers: int | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Execute two independent Formal candidates and merge in frozen order."""

    stage = output / STAGE_DIRS["formal"]
    tasks_root = stage / "_parallel" / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    task_specs = _formal_task_specs(config)
    task_files = []
    for task in task_specs:
        task_file = tasks_root / f"{task['task_id']}.json"
        atomic_write_json(task_file, task)
        task_files.append(task_file)
    configured_workers = int(config["formal_audit"]["parallel_workers"])
    requested_workers = (
        configured_workers if pilot_workers is None else int(pilot_workers)
    )
    effective_workers = min(
        max(1, requested_workers), configured_workers, len(task_specs)
    )
    _run_formal_subprocess_tasks(
        source_root=source_root,
        project_root=project_root,
        output=output,
        config_path=config_path,
        preset=preset,
        task_files=task_files,
        task_specs=task_specs,
        effective_workers=effective_workers,
        poll_seconds=float(
            config["formal_audit"]["monitoring_poll_seconds"]
        ),
        per_worker_blas_threads=int(
            config["formal_audit"]["per_worker_blas_threads"]
        ),
        manifest_path=stage / "formal_parallel_manifest.json",
    )
    candidate_reports = []
    for task in task_specs:
        candidate_dir = stage / str(task["candidate_id"])
        report_path = candidate_dir / "formal_report.json"
        if not report_path.is_file():
            raise RuntimeError(
                f"Formal candidate report missing: {task['candidate_id']}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("candidate_id") != str(task["candidate_id"])
            or not _artifact_manifest_is_valid(
                candidate_dir,
                report.get("candidate_artifact_sha256", {}),
            )
        ):
            raise RuntimeError(
                f"Formal candidate artifact validation failed: "
                f"{task['candidate_id']}"
            )
        candidate_reports.append(report)
    selected_method = json.loads(
        (
            output
            / STAGE_DIRS["pilot"]
            / "selected_formal_method.json"
        ).read_text(encoding="utf-8")
    )
    return _finalize_formal_reports(
        config=config,
        output=output,
        selected_method=selected_method,
        candidate_reports=candidate_reports,
    )


def _derived_downstream_config(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    directory: Path,
) -> Path:
    base_path = source_root / str(config["downstream"]["base_config"])
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    payload["protocol_id"] = "generalized-ellipse-region-v11.4-downstream"
    output_root = str(config["downstream"]["output_root"])
    if str(config["preset"]) == "smoke":
        output_root += "_smoke"
    payload["output_root"] = output_root
    payload["tube"]["frontier_mm"] = copy.deepcopy(
        config["downstream"]["tube_frontier_mm"]
    )
    payload["tube"]["fallback_widths_mm"] = copy.deepcopy(
        config["downstream"]["tube_fallback_widths_mm"]
    )
    payload["formal"]["nested_train_sizes"] = copy.deepcopy(
        config["downstream"]["student_nested_train_sizes"]
    )
    payload["representation"]["static_hidden_units"] = copy.deepcopy(
        config["downstream"]["student_hidden_units"]
    )
    payload["training"].update(
        {
            "max_epochs": int(config["downstream"]["student_max_epochs"]),
            "patience": int(config["downstream"]["student_patience"]),
            "batch_size": int(config["downstream"]["student_batch_size"]),
        }
    )
    payload["gates"]["student"].update(
        {
            "seed_count": int(config["downstream"]["student_seed_count"]),
            "required_seed_passes": int(
                config["downstream"]["student_required_seed_passes"]
            ),
        }
    )
    path = directory / "generalized_ellipse_region_v11_4_downstream.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _bootstrap_downstream_anchor(
    *,
    config: Mapping[str, Any],
    downstream_config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    branch_output: Path,
    downstream_output: Path,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    stage = downstream_output / v11.STAGE_DIRS["anchor"]
    stage.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    selected_centerline: pd.DataFrame | None = None
    selected_family: Mapping[str, Any] | None = None
    for candidate_id in map(str, candidate_ids):
        task = json.loads(
            (
                project_root
                / str(config["source_anchor_root"])
                / "verify_tasks"
                / f"{candidate_id}.json"
            ).read_text(encoding="utf-8")
        )
        family = task["family"]
        cycle = pd.read_parquet(
            branch_output
            / STAGE_DIRS["formal"]
            / candidate_id
            / "selected_cycle"
            / "cycle.parquet"
        ).sort_values("phase_idx", kind="stable")
        centerline = cycle[["phase_idx", *BETA_COLUMNS]].copy()
        relative = Path("bridged") / candidate_id / "centerline.parquet"
        _atomic_parquet(centerline, stage / relative)
        payloads.append(
            {
                "candidate_id": candidate_id,
                "family": family,
                "centerline_path": relative.as_posix(),
                "source_protocol_id": str(config["protocol_id"]),
            }
        )
        if selected_centerline is None:
            selected_centerline = centerline
            selected_family = family
    if selected_centerline is None or selected_family is None:
        raise ValueError("at least one passing candidate is required")
    _atomic_parquet(
        selected_centerline, stage / "selected_centerline.parquet"
    )
    atomic_write_json(stage / "selected_anchor.json", selected_family)
    atomic_write_json(stage / "passing_anchors.json", payloads)
    cache = v11._stage_cache_fingerprint(  # noqa: SLF001
        config=downstream_config,
        source_root=source_root,
        output=downstream_output,
        stage_name="anchor",
    )
    return v11._write_gate(  # noqa: SLF001
        stage / "gate.json",
        checks={
            "v11_4_formal_cycle_passed": True,
            "phase_inventory_720": len(selected_centerline)
            == int(config["phase_counts"]["formal"]),
            "all_passing_anchors_bridged": len(payloads)
            == len(candidate_ids),
            "centerline_has_no_phase_deletion": selected_centerline[
                "phase_idx"
            ].tolist()
            == list(range(len(selected_centerline))),
        },
        cache_fingerprint=cache,
        selected_candidate_id=str(candidate_ids[0]),
        passing_candidate_ids=list(map(str, candidate_ids)),
        source_full_loop_gate=sha256_file(
            branch_output / STAGE_DIRS["formal"] / "gate.json"
        ),
    )


def _rewrite_downstream_tube_gate(
    *,
    tube_gate_path: Path,
    original_tube_gate: Mapping[str, Any],
    downstream_config: Mapping[str, Any],
    source_root: Path,
    downstream_output: Path,
    selected_tube: Mapping[str, Any],
    half_mm_full_audit_pass: bool,
) -> dict[str, Any]:
    """Re-seal the V11 tube gate after V11.4 selects an audited surface."""

    preserved = {
        key: value
        for key, value in original_tube_gate.items()
        if key
        not in {
            "artifact_sha256",
            "cache_fingerprint",
            "checks",
            "gate_pass",
            "selected_tube",
        }
    }
    return v11._write_gate(  # noqa: SLF001
        tube_gate_path,
        checks={
            **original_tube_gate["checks"],
            "minimum_0p5_by_0p5_tube_passes": bool(
                half_mm_full_audit_pass
            ),
            "v11_4_selected_surface_full_audit_passes": True,
        },
        cache_fingerprint=v11._stage_cache_fingerprint(  # noqa: SLF001
            config=downstream_config,
            source_root=source_root,
            output=downstream_output,
            stage_name="tube",
        ),
        selected_tube=dict(selected_tube),
        **preserved,
    )


def _downstream_full_audit_task_specs(
    frontier: pd.DataFrame, *, cuts: Sequence[int]
) -> list[dict[str, Any]]:
    """Build a stable, deterministic task order for dense branch audits."""

    specs: list[dict[str, Any]] = []
    stable_cuts = sorted({int(cut) for cut in cuts})
    for candidate_index, candidate in frontier.reset_index(drop=True).iterrows():
        for direction in ("forward", "reverse"):
            for cut in stable_cuts:
                specs.append(
                    {
                        "task_id": (
                            f"full_branch_c{candidate_index:03d}"
                            f"_{direction}_cut{cut:04d}"
                        ),
                        "candidate_index": int(candidate_index),
                        "anchor_id": str(candidate["anchor_id"]),
                        "radial_radius_mm": float(
                            candidate["radial_radius_mm"]
                        ),
                        "plane_radius_mm": float(
                            candidate["plane_radius_mm"]
                        ),
                        "direction": direction,
                        "cut": int(cut),
                        "variant": f"{direction}_cut{cut:04d}",
                        "policy_seed_offset": (
                            100000 * int(candidate_index)
                            + int(cut)
                            + (10000 if direction == "reverse" else 0)
                        ),
                    }
                )
    return specs


def _run_downstream_full_branch_audit(
    *,
    config: Mapping[str, Any],
    downstream_config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    downstream_output: Path,
    directory: Path,
) -> dict[str, Any]:
    environment = load_environment(
        project_root,
        project_root / str(downstream_config["robot_config"]),
    )
    tube_stage = downstream_output / v11.STAGE_DIRS["tube"]
    tube_gate_path = tube_stage / "gate.json"
    original_tube_gate = json.loads(
        tube_gate_path.read_text(encoding="utf-8")
    )
    dense = pd.read_csv(tube_stage / "dense_frontier.csv")
    dense = dense[dense["gate_pass"].astype(bool)].copy()
    anchors = {
        anchor_id: (family, centerline)
        for anchor_id, family, centerline in v11._load_passing_anchors(  # noqa: SLF001
            downstream_output
        )
    }
    count = int(downstream_config["tube"]["dense_phase_count"])
    cross = v11.TubeCrossSection.master(
        seed=int(downstream_config["seeds"]["cross_section"])
    ).prefix(int(downstream_config["tube"]["dense_cross_section_count"]))
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    cuts = {
        int(value) % count
        for value in config["formal_audit"]["cut_phase_indices"]
    } | {
        int(round(float(value) * count / 360.0)) % count
        for value in config["formal_audit"]["cut_angles_deg"]
    }
    frontier = dense[
        (
            np.isclose(dense["radial_radius_mm"], 0.5)
            & np.isclose(dense["plane_radius_mm"], 0.5)
        )
        | (
            np.isclose(dense["radial_radius_mm"], 1.0)
            & np.isclose(dense["plane_radius_mm"], 1.0)
        )
    ]
    specs = _downstream_full_audit_task_specs(frontier, cuts=cuts)
    centerline_paths = v11._tube_centerline_paths(  # noqa: SLF001
        downstream_output
    )
    task_files: list[Path] = []
    variant_directories: dict[str, Path] = {}
    for spec in specs:
        anchor_id = str(spec["anchor_id"])
        family, _centerline = anchors[anchor_id]
        radial = float(spec["radial_radius_mm"])
        plane = float(spec["plane_radius_mm"])
        variant_directory = (
            directory
            / anchor_id
            / f"r{radial:g}_p{plane:g}"
            / str(spec["variant"])
        )
        variant_directories[str(spec["task_id"])] = variant_directory
        task_files.append(
            v11._write_tube_surface_task(  # noqa: SLF001
                stage=directory,
                task_id=str(spec["task_id"]),
                config=downstream_config,
                project_root=project_root,
                family=family,
                centerline_path=centerline_paths[anchor_id],
                phase_count=count,
                cross_section_count=int(len(cross.points)),
                radial_radius_mm=radial,
                plane_radius_mm=plane,
                directory=variant_directory,
                policy_seed_offset=int(spec["policy_seed_offset"]),
                traversal_direction=str(spec["direction"]),
                cyclic_cut=int(spec["cut"]),
            )
        )
    parallel_manifest_path = directory / "full_branch_parallel_manifest.json"
    requested_workers = int(
        config["downstream"]["full_branch_audit_parallel_workers"]
    )
    atomic_write_json(
        parallel_manifest_path,
        {
            "schema_version": 1,
            "execution_model": (
                "deterministic_independent_subprocess_workers"
            ),
            "status": "running",
            "requested_workers": requested_workers,
            "independent_task_count": len(task_files),
            "per_worker_blas_threads": 1,
            "task_ids": [str(spec["task_id"]) for spec in specs],
        },
    )
    try:
        batch_report = v11._run_subprocess_tasks(  # noqa: SLF001
            task_files,
            worker_name="tube-surface",
            max_workers=requested_workers,
        )
    except v11.ParallelWorkerError as error:
        atomic_write_json(
            parallel_manifest_path,
            {
                "schema_version": 1,
                "execution_model": (
                    "deterministic_independent_subprocess_workers"
                ),
                "status": "failed",
                "independent_task_count": len(task_files),
                "per_worker_blas_threads": 1,
                **error.report,
            },
        )
        raise
    atomic_write_json(
        parallel_manifest_path,
        {
            "schema_version": 1,
            "execution_model": (
                "deterministic_independent_subprocess_workers"
            ),
            "status": "completed",
            "independent_task_count": len(task_files),
            "per_worker_blas_threads": 1,
            **batch_report,
        },
    )

    rows_by_candidate: dict[int, list[dict[str, Any]]] = {
        int(index): [] for index in range(len(frontier))
    }
    for spec in specs:
        anchor_id = str(spec["anchor_id"])
        radial = float(spec["radial_radius_mm"])
        plane = float(spec["plane_radius_mm"])
        size_name = f"r{radial:g}_p{plane:g}"
        primary_path = (
            tube_stage
            / "anchors"
            / anchor_id
            / "dense"
            / size_name
            / "primary"
            / "surface.parquet"
        )
        variant_directory = variant_directories[str(spec["task_id"])]
        report = json.loads(
            (variant_directory / "report.json").read_text(encoding="utf-8")
        )
        gap = v11._surface_aligned_gap(  # noqa: SLF001
            pd.read_parquet(primary_path),
            pd.read_parquet(variant_directory / "surface.parquet"),
        )
        row = {
            "anchor_id": anchor_id,
            "radial_radius_mm": radial,
            "plane_radius_mm": plane,
            "variant": str(spec["variant"]),
            "solver_gate_pass": bool(report["gate_pass"]),
            **gap,
        }
        rows.append(row)
        rows_by_candidate[int(spec["candidate_index"])].append(row)

    for candidate_index, candidate in frontier.reset_index(drop=True).iterrows():
        anchor_id = str(candidate["anchor_id"])
        radial = float(candidate["radial_radius_mm"])
        plane = float(candidate["plane_radius_mm"])
        size_name = f"r{radial:g}_p{plane:g}"
        primary_path = (
            tube_stage
            / "anchors"
            / anchor_id
            / "dense"
            / size_name
            / "primary"
            / "surface.parquet"
        )
        one_rows = rows_by_candidate[int(candidate_index)]
        one_table = pd.DataFrame(one_rows)
        audit_pass = bool(
            len(one_table) == 2 * len(cuts)
            and one_table["solver_gate_pass"].all()
            and (one_table["beta_gap_rms_p95_deg"] <= 1.0).all()
            and (one_table["beta_gap_rms_max_deg"] <= 2.0).all()
        )
        candidate_rows.append(
            {
                "anchor_id": anchor_id,
                "radial_radius_mm": radial,
                "plane_radius_mm": plane,
                "dense_gate_pass": True,
                "full_branch_audit_pass": audit_pass,
                "primary_surface_path": str(primary_path),
            }
        )
    table = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / "full_branch_audit.csv", index=False)
    candidates.to_csv(
        directory / "audited_dense_candidates.csv", index=False
    )
    eligible_half = candidates[
        np.isclose(candidates["radial_radius_mm"], 0.5)
        & np.isclose(candidates["plane_radius_mm"], 0.5)
        & candidates["full_branch_audit_pass"].astype(bool)
    ] if not candidates.empty else candidates
    eligible_one = candidates[
        np.isclose(candidates["radial_radius_mm"], 1.0)
        & np.isclose(candidates["plane_radius_mm"], 1.0)
        & candidates["full_branch_audit_pass"].astype(bool)
    ] if not candidates.empty else candidates
    selected_row = (
        eligible_one.iloc[0]
        if not eligible_one.empty
        else (eligible_half.iloc[0] if not eligible_half.empty else None)
    )
    if selected_row is not None:
        selected_anchor_id = str(selected_row["anchor_id"])
        selected_family, selected_centerline = anchors[selected_anchor_id]
        shutil.copy2(
            Path(str(selected_row["primary_surface_path"])),
            tube_stage / "selected_surface.parquet",
        )
        selected_tube_payload = {
            "anchor_id": selected_anchor_id,
            "anchor_family": v11._family_payload(  # noqa: SLF001
                selected_family
            ),
            "radial_radius_mm": float(
                selected_row["radial_radius_mm"]
            ),
            "plane_radius_mm": float(selected_row["plane_radius_mm"]),
            "phase_count": count,
            "cross_section_count": int(len(cross.points)),
            "v11_4_full_branch_audit_pass": True,
        }
        atomic_write_json(
            tube_stage / "selected_tube.json",
            selected_tube_payload,
        )
        atomic_write_json(
            tube_stage / "selected_region_anchor.json",
            v11._family_payload(selected_family),  # noqa: SLF001
        )
        centerline_frame = pd.DataFrame(
            selected_centerline, columns=BETA_COLUMNS
        )
        centerline_frame.insert(
            0, "phase_idx", np.arange(len(selected_centerline))
        )
        _atomic_parquet(
            centerline_frame,
            tube_stage / "selected_region_centerline.parquet",
        )
        _rewrite_downstream_tube_gate(
            tube_gate_path=tube_gate_path,
            original_tube_gate=original_tube_gate,
            downstream_config=downstream_config,
            source_root=source_root,
            downstream_output=downstream_output,
            selected_tube=selected_tube_payload,
            half_mm_full_audit_pass=not eligible_half.empty,
        )
    return _write_gate(
        directory / "gate.json",
        checks={
            "half_mm_dense_candidate_exists": bool(
                not candidates.empty
                and (
                    np.isclose(candidates["radial_radius_mm"], 0.5)
                    & np.isclose(candidates["plane_radius_mm"], 0.5)
                ).any()
            ),
            "half_mm_full_branch_audit_passes": bool(
                not eligible_half.empty
            ),
            "selected_surface_full_branch_audit_passes": bool(
                selected_row is not None
            ),
        },
        variant_summary=table.to_dict("records"),
        candidate_summary=candidates.to_dict("records"),
        selected_candidate=(
            None if selected_row is None else selected_row.to_dict()
        ),
    )


def run_bridge(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    project_root: Path,
    output: Path,
    **_unused: Any,
) -> dict[str, Any]:
    formal_gate = json.loads(
        (output / STAGE_DIRS["formal"] / "gate.json").read_text(
            encoding="utf-8"
        )
    )
    passing = list(map(str, formal_gate.get("passing_candidates", [])))
    if not passing:
        raise RuntimeError(
            "downstream bridge requires at least one V11.4 outcome-A candidate"
        )
    stage = output / STAGE_DIRS["bridge"]
    stage.mkdir(parents=True, exist_ok=True)
    derived_path = _derived_downstream_config(
        config=config, source_root=source_root, directory=stage
    )
    downstream_config = v11.load_protocol_config(
        derived_path, preset=str(config["preset"])
    )
    downstream_output = (
        project_root / str(downstream_config["output_root"])
    )
    downstream_output.mkdir(parents=True, exist_ok=True)
    common = {
        "config": downstream_config,
        "source_root": source_root,
        "project_root": project_root,
        "output": downstream_output,
    }
    reports: dict[str, Any] = {}
    reports["protocol"] = v11.run_protocol_stage(**common)
    if not reports["protocol"]["gate_pass"]:
        return _write_gate(
            stage / "gate.json",
            checks={"downstream_protocol_pass": False},
            downstream_output=str(downstream_output),
            reports=reports,
        )
    reports["anchor"] = _bootstrap_downstream_anchor(
        config=config,
        downstream_config=downstream_config,
        source_root=source_root,
        project_root=project_root,
        branch_output=output,
        downstream_output=downstream_output,
        candidate_ids=passing,
    )
    if not reports["anchor"]["gate_pass"]:
        return _write_gate(
            stage / "gate.json",
            checks={"downstream_anchor_bridge_pass": False},
            downstream_output=str(downstream_output),
            reports=reports,
        )
    downstream_stages = [
        ("tube", v11.run_tube_stage),
        ("core", v11.run_core_stage),
        ("pilot", v11.run_pilot_stage),
        ("representation", v11.run_representation_stage),
        ("formal", v11.run_formal_stage),
        ("train", v11.run_train_stage),
        ("evaluate", v11.run_evaluate_stage),
    ]
    reports["tube"] = v11.run_tube_stage(**common)
    if not reports["tube"]["gate_pass"]:
        return _write_gate(
            stage / "gate.json",
            checks={"minimum_tube_pass": False},
            downstream_output=str(downstream_output),
            reports=reports,
        )
    reports["full_tube_branch_audit"] = _run_downstream_full_branch_audit(
        config=config,
        downstream_config=downstream_config,
        source_root=source_root,
        project_root=project_root,
        downstream_output=downstream_output,
        directory=stage / "full_tube_branch_audit",
    )
    if not reports["full_tube_branch_audit"]["gate_pass"]:
        return _write_gate(
            stage / "gate.json",
            checks={"full_tube_branch_audit_pass": False},
            downstream_output=str(downstream_output),
            reports=reports,
        )
    for name, runner in downstream_stages[1:]:
        reports[name] = runner(**common)
        if not reports[name]["gate_pass"]:
            return _write_gate(
                stage / "gate.json",
                checks={f"downstream_{name}_pass": False},
                downstream_output=str(downstream_output),
                stopped_after=name,
                reports=reports,
            )
    reports["verify"] = v11.run_verify_stage(output=downstream_output)
    gate = _write_gate(
        stage / "gate.json",
        checks={
            "minimum_0p5_tube_passed": bool(
                reports["full_tube_branch_audit"]["checks"][
                    "half_mm_full_branch_audit_passes"
                ]
            ),
            "full_tube_branch_audit_passed": bool(
                reports["full_tube_branch_audit"]["gate_pass"]
            ),
            "static_representation_selected": (
                reports["representation"].get("representation") == "static"
            ),
            "formal_dataset_passed": bool(reports["formal"]["gate_pass"]),
            "student_4_of_5_passed": bool(reports["train"]["gate_pass"]),
            "sealed_evaluation_passed": bool(reports["evaluate"]["gate_pass"]),
            "downstream_verify_passed": bool(reports["verify"]["gate_pass"]),
        },
        downstream_output=str(downstream_output),
        reports=reports,
    )
    atomic_write_json(
        output / "V11_4_END_TO_END_COMPLETED.json",
        {
            "protocol_id": str(config["protocol_id"]),
            "gate_pass": bool(gate["gate_pass"]),
            "downstream_output": str(downstream_output),
        },
    )
    return gate


STAGE_RUNNERS = {
    "protocol": run_protocol,
    "root_fiber": run_root_fiber,
    "viability": run_viability,
    "pilot": run_pilot,
    "formal": run_formal,
    "bridge": run_bridge,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generalized_ellipse_region_v11_full_loop.yaml",
    )
    parser.add_argument("--preset", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--stage",
        choices=(
            "branch",
            "all",
            "protocol",
            "root_fiber",
            "viability",
            "pilot",
            "formal",
            "bridge",
        ),
        default="branch",
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override coarse-grained Pilot subprocess worker count.",
    )
    parser.add_argument(
        "--pilot-worker-task",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--formal-worker-task",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[2]
    project_root = (
        project_root_from(source_root)
        if args.project_root is None
        else Path(args.project_root).resolve()
    )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = source_root / config_path
    config = load_config(config_path, preset=args.preset)
    output = (
        project_root / str(config["output_root"])
        if args.output is None
        else Path(args.output).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    environment = load_environment(
        project_root, project_root / str(config["robot_config"])
    )
    if args.pilot_worker_task is not None:
        task_path = Path(args.pilot_worker_task).resolve()
        pilot_task = json.loads(task_path.read_text(encoding="utf-8"))
        expected_tasks = {
            str(task["task_id"]): task for task in _pilot_task_specs(config)
        }
        if (
            str(pilot_task.get("task_id")) not in expected_tasks
            or expected_tasks[str(pilot_task["task_id"])] != pilot_task
        ):
            raise RuntimeError("Pilot worker task does not match resolved config")
        report = _run_pilot_slice(
            config=config,
            project_root=project_root,
            environment=environment,
            output=output,
            pilot_task=pilot_task,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    if args.formal_worker_task is not None:
        task_path = Path(args.formal_worker_task).resolve()
        formal_task = json.loads(task_path.read_text(encoding="utf-8"))
        expected_tasks = {
            str(task["task_id"]): task for task in _formal_task_specs(config)
        }
        if (
            str(formal_task.get("task_id")) not in expected_tasks
            or expected_tasks[str(formal_task["task_id"])] != formal_task
        ):
            raise RuntimeError("Formal worker task does not match resolved config")
        report = _run_formal_candidate(
            config=config,
            project_root=project_root,
            environment=environment,
            output=output,
            formal_task=formal_task,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    atlas = _reachability_atlas(project_root, config)
    common = {
        "config": config,
        "config_path": config_path.resolve(),
        "preset": str(args.preset),
        "pilot_workers": args.workers,
        "source_root": source_root,
        "project_root": project_root,
        "environment": environment,
        "atlas": atlas,
        "output": output,
    }
    if args.stage in {"branch", "all"}:
        reports: dict[str, Any] = {}
        for name in ("protocol", "root_fiber", "viability", "pilot", "formal"):
            report = STAGE_RUNNERS[name](**common)
            reports[name] = report
            if not bool(report["gate_pass"]):
                reports["stopped_after"] = name
                reports["stop_reason"] = "hard_gate_failed"
                print(json.dumps(reports, indent=2, allow_nan=False))
                return
        reports["downstream_pending"] = True
        reports["next_command"] = (
            "run_generalized_ellipse_full_loop_v11_4.py --stage bridge"
        )
        print(json.dumps(reports, indent=2, allow_nan=False))
        return
    report = STAGE_RUNNERS[str(args.stage)](**common)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
